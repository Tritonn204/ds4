#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


FULL_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
ATTN_LABELS = [
    "ATTN_IN",
    "QG",
    "KK",
    "VV",
    "Q_CUR",
    "K_CUR",
    "V_CUR",
    "ATTN_OUT",
    "PROJ_OUT",
    "RESIDUAL",
]

DBG_RE = re.compile(
    r"=== DBG_FULL\[(?P<layer>\d+)\] step=(?P<step>\d+) (?P<label>[A-Z_]+) "
    r"rmse=(?P<rmse>[0-9.]+) max_diff=(?P<max_diff>[0-9.]+) max_idx=(?P<max_idx>\d+) "
    r"mean_abs=(?P<mean_abs>[0-9.]+)"
)


def build_fixture_list(shared_dir: Path, full_layers: list[int]) -> list[Path]:
    full_set = set(full_layers)
    fixtures: list[Path] = []
    for blk in range(40):
        if blk in full_set:
            continue
        fixtures.append(shared_dir / f"blk{blk}.live.bin")
    return fixtures


def shell_join(items: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in items)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Focused decode-time full-attention stage study for owned-session Qwen3.6"
    )
    ap.add_argument("--hf", default="/mnt/e/tensors/Qwen3.6-35B-A3B")
    ap.add_argument(
        "--gguf",
        default="/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-Q8_0.gguf",
    )
    ap.add_argument("--owned-session-worker-bin", default="/mnt/f/git/ds4/qwen36-unified-owned-worker-rocm")
    ap.add_argument("--c-bin", default="/mnt/f/git/ds4/qwen36-c-prefix-q8-chain-live")
    ap.add_argument("--c-bin-worker-bin", default="/mnt/f/git/ds4/qwen36-live-contract-worker")
    ap.add_argument("--full-layer-worker-bin", default="/mnt/f/git/ds4/qwen36-gpu-full-layer-worker")
    ap.add_argument("--shared-fixture-dir", default="/mnt/f/git/ds4/.cache/qwen36_live_contracts/Qwen3.6-35B-A3B-Q8_0/shared")
    ap.add_argument("--prompt-file", default="/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_oracle_prompts/code_review.txt")
    ap.add_argument("--splice-layer", type=int, default=39)
    ap.add_argument("--full-debug-layer", default="39",
                    help="Full layer to debug, or 'all' to capture every full layer")
    ap.add_argument("--step-start", type=int, default=1)
    ap.add_argument("--step-end", type=int, default=32)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--json-out", default="/tmp/qwen36_full_attn_decode_study.json")
    ap.add_argument("--stderr-out", default="/tmp/qwen36_full_attn_decode_study.txt")
    ap.add_argument("--tail-source", choices=["auto", "hf", "gguf", "compare"], default="auto")
    ap.add_argument("--first-dirty-threshold", type=float, default=0.05,
                    help="Report the first full layer whose worst stage RMSE exceeds this threshold")
    ap.add_argument("--owned-session-env", action="append", default=[],
                    help="Extra owned-session env KEY=VALUE forwarded to behavior oracle")
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()

    shared_dir = Path(args.shared_fixture_dir)
    fixtures = build_fixture_list(shared_dir, FULL_LAYERS)
    missing = [str(p) for p in fixtures if not p.is_file()]
    if missing:
        print("missing fixtures:", file=sys.stderr)
        for path in missing:
            print(path, file=sys.stderr)
        return 2

    cmd = [
        "python3",
        "-u",
        "/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_behavior_oracle.py",
        "--mode",
        "patched-session",
        "--hf",
        args.hf,
        "--gguf",
        args.gguf,
        "--owned-session-worker-bin",
        args.owned_session_worker_bin,
        "--owned-session-unified-full-gpu",
        "--owned-session-unified-hybrid-gpu-cycles",
        "10",
        "--c-bin",
        args.c_bin,
        "--c-bin-worker-bin",
        args.c_bin_worker_bin,
        "--full-layer-worker-bin",
        args.full_layer_worker_bin,
    ]
    for layer in FULL_LAYERS:
        cmd.extend(["--full-layer", str(layer)])
    for fixture in fixtures:
        cmd.extend(["--fixture", str(fixture)])
    cmd.extend(
        [
            "--owned-session-env",
            "QWEN36_UNIFIED_VERBOSE=0",
            "--owned-session-env",
            f"QWEN36_DBG_FULL_LAYER={args.full_debug_layer}",
            "--owned-session-env",
            "QWEN36_DBG_FULL_BREADCRUMB=1",
            "--owned-session-env",
            f"QWEN36_DBG_FULL_STEP_START={args.step_start}",
            "--owned-session-env",
            f"QWEN36_DBG_FULL_STEP_END={args.step_end}",
            "--prompt-file",
            args.prompt_file,
            "--splice-layer",
            str(args.splice_layer),
            "--n-predict",
            str(args.n_predict),
            "--top-k",
            str(args.top_k),
            "--tail-source",
            args.tail_source,
            "--json-out",
            args.json_out,
        ]
    )
    for item in args.owned_session_env:
        cmd.extend(["--owned-session-env", item])

    print(f"[full-attn-study] cmd={shell_join(cmd)}")
    print(f"[full-attn-study] stderr_out={args.stderr_out}")
    if args.print_only:
        return 0

    env = os.environ.copy()
    env["QWEN36_UNIFIED_VERBOSE"] = "0"
    with open(args.stderr_out, "w", encoding="utf-8") as ef:
        rc = subprocess.run(cmd, env=env, stderr=ef, check=False).returncode
    if rc != 0:
        print(f"[full-attn-study] oracle failed rc={rc}")
        return rc

    rows: dict[int, dict[int, list[dict]]] = {}
    with open(args.stderr_out, "r", encoding="utf-8") as ef:
        for line in ef:
            m = DBG_RE.search(line)
            if not m:
                continue
            layer = int(m.group("layer"))
            label = m.group("label")
            if label not in ATTN_LABELS:
                continue
            step = int(m.group("step"))
            rows.setdefault(layer, {}).setdefault(step, []).append(
                {
                    "label": label,
                    "rmse": float(m.group("rmse")),
                    "max_diff": float(m.group("max_diff")),
                    "max_idx": int(m.group("max_idx")),
                    "mean_abs": float(m.group("mean_abs")),
                }
            )

    if not rows:
        print("[full-attn-study] no attention-stage DBG_FULL rows captured")
        return 1

    overall = None
    layer_summaries: list[dict] = []
    for layer in sorted(rows):
        print(f"[full-attn-study] layer={layer} worst attention stage per step")
        layer_overall = None
        for step in sorted(rows[layer]):
            worst = max(rows[layer][step], key=lambda item: item["rmse"])
            print(
                f"  step={step} worst={worst['label']} rmse={worst['rmse']:.8f} "
                f"max_diff={worst['max_diff']:.8f} max_idx={worst['max_idx']}"
            )
            if layer_overall is None or worst["rmse"] > layer_overall["rmse"]:
                layer_overall = {"layer": layer, "step": step, **worst}
            if overall is None or worst["rmse"] > overall["rmse"]:
                overall = {"layer": layer, "step": step, **worst}
        assert layer_overall is not None
        layer_summaries.append(layer_overall)
        print(
            f"[full-attn-study] layer={layer} overall worst step={layer_overall['step']} "
            f"stage={layer_overall['label']} rmse={layer_overall['rmse']:.8f} "
            f"max_diff={layer_overall['max_diff']:.8f} max_idx={layer_overall['max_idx']}"
        )

    assert overall is not None
    print(
        f"[full-attn-study] overall worst layer={overall['layer']} step={overall['step']} "
        f"stage={overall['label']} rmse={overall['rmse']:.8f} "
        f"max_diff={overall['max_diff']:.8f} max_idx={overall['max_idx']}"
    )
    print("[full-attn-study] local_full_layer_math_clean_only=1")
    print("[full-attn-study] does_not_validate_end_to_end_decode=1")
    first_dirty = None
    for item in sorted(layer_summaries, key=lambda x: x["layer"]):
        if item["rmse"] >= args.first_dirty_threshold:
            first_dirty = item
            break
    if first_dirty is None:
        print(
            f"[full-attn-study] no layer exceeded threshold rmse={args.first_dirty_threshold:.8f}"
        )
    else:
        print(
            f"[full-attn-study] first dirty layer={first_dirty['layer']} step={first_dirty['step']} "
            f"stage={first_dirty['label']} rmse={first_dirty['rmse']:.8f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
