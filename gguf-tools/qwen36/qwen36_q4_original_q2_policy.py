#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from qwen36_routed_tuple_probe import build_report, classify_layer


ROUTED_SUFFIXES = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
)

Q2_SAFE_TYPES = {"q2_k", "iq2_xxs"}
FORBIDDEN_TYPES = {"iq2_s", "iq3_s"}
HYBRID_FULL_LAYERS = set(range(40))
DEFAULT_Q4_LATE_LAYERS = set(range(34, 40))


def tensor_name(layer: int, suffix: str) -> str:
    return f"blk.{layer}.{suffix}"


def parse_layer_set(text: str) -> set[int]:
    layers: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if hi < lo:
                raise ValueError(f"bad layer range: {part}")
            layers.update(range(lo, hi + 1))
        else:
            layers.add(int(part))
    bad = layers - HYBRID_FULL_LAYERS
    if bad:
        raise ValueError(f"layers out of range 0..39: {sorted(bad)}")
    return layers


def load_source_report(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return build_report(str(path))


def type_for(row: dict, suffix: str) -> str:
    if suffix == "ffn_gate_exps.weight":
        return row["routed_gate"]
    if suffix == "ffn_up_exps.weight":
        return row["routed_up"]
    if suffix == "ffn_down_exps.weight":
        return row["routed_down"]
    raise KeyError(suffix)


def build_policy(source: dict, q4_type: str) -> dict:
    layers = source.get("layers", [])
    if not layers:
        raise RuntimeError("source report has no routed layer rows")

    policy = {}
    forbidden = []
    q2_preserved = 0
    q4_assigned = 0

    rows = []
    for row in layers:
        layer = int(row["layer"])
        out_row = {"layer": layer}
        for suffix in ROUTED_SUFFIXES:
            src_type = type_for(row, suffix)
            if src_type in FORBIDDEN_TYPES:
                forbidden.append({
                    "layer": layer,
                    "suffix": suffix,
                    "source_type": src_type,
                })
            target_type = src_type if src_type in Q2_SAFE_TYPES else q4_type
            policy[tensor_name(layer, suffix)] = target_type
            out_row[suffix] = {
                "source_type": src_type,
                "target_type": target_type,
            }
            if target_type in Q2_SAFE_TYPES:
                q2_preserved += 1
            elif target_type == q4_type:
                q4_assigned += 1
        gate = policy[tensor_name(layer, "ffn_gate_exps.weight")]
        up = policy[tensor_name(layer, "ffn_up_exps.weight")]
        down = policy[tensor_name(layer, "ffn_down_exps.weight")]
        runtime = classify_layer(gate, up, down)
        out_row["runtime_tuple"] = {
            "gate": gate,
            "up": up,
            "down": down,
            "default_path": runtime["default_path"],
        }
        rows.append(out_row)

    if forbidden:
        raise RuntimeError(
            "source policy contains DS4Style-v0-only routed types; refusing to derive "
            f"Q4+original-Q2 policy: {forbidden[:8]}"
        )

    return {
        "source_artifact": source.get("artifact"),
        "q2_safe_types": sorted(Q2_SAFE_TYPES),
        "q4_type": q4_type,
        "policy": policy,
        "layers": rows,
        "summary": {
            "routed_tensors": len(policy),
            "q2_preserved": q2_preserved,
            "q4_assigned": q4_assigned,
        },
    }


def build_from_study_default(q4_type: str, late_q4_layers: set[int]) -> dict:
    policy = {}
    rows = []
    q2_preserved = 0
    q4_assigned = 0

    for layer in sorted(HYBRID_FULL_LAYERS):
        late_q4 = layer in late_q4_layers
        gate = q4_type if late_q4 else "iq2_xxs"
        up = q4_type if late_q4 else "iq2_xxs"
        down = q4_type if late_q4 else "q2_k"
        row = {"layer": layer}
        for suffix, target_type in (
            ("ffn_gate_exps.weight", gate),
            ("ffn_up_exps.weight", up),
            ("ffn_down_exps.weight", down),
        ):
            policy[tensor_name(layer, suffix)] = target_type
            row[suffix] = {
                "source_type": "study_default",
                "target_type": target_type,
            }
            if target_type in Q2_SAFE_TYPES:
                q2_preserved += 1
            else:
                q4_assigned += 1
        runtime = classify_layer(gate, up, down)
        row["runtime_tuple"] = {
            "gate": gate,
            "up": up,
            "down": down,
            "default_path": runtime["default_path"],
        }
        rows.append(row)

    return {
        "source_artifact": "study_default",
        "source_note": (
            "Recovered from qwen36_first_policy_table.md: routed gate/up low-bit, "
            "routed down low-bit, late routed down protected. Late layers are made "
            "q4/q4/q4 to stay on the native Qwen routed-MoE tuple set."
        ),
        "late_q4_layers": sorted(late_q4_layers),
        "q2_safe_types": sorted(Q2_SAFE_TYPES),
        "q4_type": q4_type,
        "policy": policy,
        "layers": rows,
        "summary": {
            "routed_tensors": len(policy),
            "q2_preserved": q2_preserved,
            "q4_assigned": q4_assigned,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Derive a Q4 + original-Q2-safe routed policy from a validated Q8+Q2 source report or GGUF."
    )
    ap.add_argument("--source", type=Path, help="Validated Q8+Q2 GGUF or qwen36_routed_tuple_probe JSON")
    ap.add_argument(
        "--from-study-default",
        action="store_true",
        help="Recover the documented Q4+Q2 policy from qwen36_first_policy_table.md",
    )
    ap.add_argument(
        "--late-q4-layers",
        default="34-39",
        help="Layer set promoted to q4/q4/q4 in --from-study-default mode",
    )
    ap.add_argument("--out", required=True, type=Path, help="Output routed policy JSON")
    ap.add_argument("--q4-type", default="q4_k", choices=["q4_k"])
    args = ap.parse_args()

    if args.from_study_default:
        late_q4_layers = parse_layer_set(args.late_q4_layers)
        result = build_from_study_default(args.q4_type, late_q4_layers)
        source_label = "study_default"
    else:
        if args.source is None:
            ap.error("--source is required unless --from-study-default is used")
        source = load_source_report(args.source)
        result = build_policy(source, args.q4_type)
        source_label = str(args.source)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    summary = result["summary"]
    print(f"[q4-original-q2-policy] source={source_label}")
    print(f"[q4-original-q2-policy] out={args.out}")
    print(
        "[q4-original-q2-policy] "
        f"routed_tensors={summary['routed_tensors']} "
        f"q2_preserved={summary['q2_preserved']} "
        f"q4_assigned={summary['q4_assigned']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
