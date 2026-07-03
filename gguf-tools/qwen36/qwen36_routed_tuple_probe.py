#!/usr/bin/env python3

import argparse
import json
from collections import Counter

from qwen36_inspect import parse_gguf


NATIVE_ROUTED_TUPLES = {
    ("q4_k", "q4_k"),
    ("iq2_xxs", "q2_k"),
    ("q2_k", "q2_k"),
}

GPU_DECODE_TYPES = {
    "q8_0",
    "q4_k",
    "q5_k",
    "q6_k",
}


def layer_kind(layer_idx: int) -> str:
    return "full_attention" if (layer_idx % 4) == 3 else "hybrid_ssm"


def build_tensor_map(tensors):
    return {t["name"]: t for t in tensors}


def tensor_type_name(tensor_map, name):
    t = tensor_map.get(name)
    return None if t is None else t["type_name"]


def classify_layer(gate, up, down):
    q8_expert_gpu = gate == "q8_0" and up == "q8_0" and down == "q8_0"
    routed_moe = gate == up and (gate, down) in NATIVE_ROUTED_TUPLES
    decoded_expert_gpu_capable = (
        gate in GPU_DECODE_TYPES and
        up in GPU_DECODE_TYPES and
        down in GPU_DECODE_TYPES
    )
    decoded_expert_gpu = decoded_expert_gpu_capable and not routed_moe and not q8_expert_gpu
    cpu_fallback = not routed_moe and not q8_expert_gpu and not decoded_expert_gpu
    default_path = (
        "routed_moe" if routed_moe else
        "q8_expert_gpu" if q8_expert_gpu else
        "cpu_fallback"
    )
    opt_in_path = (
        default_path if default_path != "cpu_fallback" else
        "decoded_expert_gpu" if decoded_expert_gpu else
        "cpu_fallback"
    )
    return {
        "q8_expert_gpu": q8_expert_gpu,
        "routed_moe": routed_moe,
        "decoded_expert_gpu": decoded_expert_gpu,
        "cpu_fallback": cpu_fallback,
        "default_path": default_path,
        "opt_in_path": opt_in_path,
    }


def guess_contract(root_token, root_output, routed_gate_types, routed_down_types):
    gate_set = set(routed_gate_types)
    down_set = set(routed_down_types)
    if gate_set <= {"iq2_xxs", "q2_k"} and down_set <= {"q2_k", "iq2_s", "iq3_s"}:
        if root_token == "q4_k" and root_output == "q4_k":
            return "ds4style_v0_like"
        return "q2_heavy_mixed"
    if {"q5_k", "q6_k"} & (gate_set | down_set):
        return "q4xl_family_bridge"
    if gate_set <= {"q4_k"} and down_set <= {"q4_k"}:
        return "uniform_q4_routed"
    return "unknown_mixed"


def build_report(model_path):
    parsed = parse_gguf(model_path)
    tensor_map = build_tensor_map(parsed["tensors"])
    block_count = int(parsed["metadata"].get("qwen35moe.block_count", 40))

    layers = []
    gate_types = []
    down_types = []
    path_counts = Counter()

    for i in range(block_count):
        prefix = f"blk.{i}."
        gate = tensor_type_name(tensor_map, prefix + "ffn_gate_exps.weight")
        up = tensor_type_name(tensor_map, prefix + "ffn_up_exps.weight")
        down = tensor_type_name(tensor_map, prefix + "ffn_down_exps.weight")
        shared_gate = tensor_type_name(tensor_map, prefix + "ffn_gate_shexp.weight")
        shared_up = tensor_type_name(tensor_map, prefix + "ffn_up_shexp.weight")
        shared_down = tensor_type_name(tensor_map, prefix + "ffn_down_shexp.weight")
        if not gate or not up or not down:
            continue
        gate_types.append(gate)
        down_types.append(down)
        path = classify_layer(gate, up, down)
        path_counts[path["default_path"]] += 1
        layers.append({
            "layer": i,
            "kind": layer_kind(i),
            "routed_gate": gate,
            "routed_up": up,
            "routed_down": down,
            "shared_gate": shared_gate,
            "shared_up": shared_up,
            "shared_down": shared_down,
            "runtime_default_path": path["default_path"],
            "runtime_opt_in_path": path["opt_in_path"],
            "runtime_flags": path,
        })

    root_token = tensor_type_name(tensor_map, "token_embd.weight")
    root_output = tensor_type_name(tensor_map, "output.weight")
    root_norm = tensor_type_name(tensor_map, "output_norm.weight")

    return {
        "artifact": model_path,
        "contract_guess": guess_contract(root_token, root_output, gate_types, down_types),
        "root_tensors": {
            "token_embd.weight": root_token,
            "output.weight": root_output,
            "output_norm.weight": root_norm,
        },
        "routed_gate_histogram": dict(sorted(Counter(gate_types).items())),
        "routed_down_histogram": dict(sorted(Counter(down_types).items())),
        "runtime_path_histogram": dict(sorted(path_counts.items())),
        "native_routed_tuples": sorted([list(x) for x in NATIVE_ROUTED_TUPLES]),
        "layers": layers,
    }


def print_human(report):
    print(f"artifact: {report['artifact']}")
    print(f"contract_guess: {report['contract_guess']}")
    print("root_tensors:")
    for k, v in report["root_tensors"].items():
        print(f"  {k}: {v}")
    print("routed_gate_histogram:")
    for k, v in report["routed_gate_histogram"].items():
        print(f"  {k}: {v}")
    print("routed_down_histogram:")
    for k, v in report["routed_down_histogram"].items():
        print(f"  {k}: {v}")
    print("runtime_path_histogram:")
    for k, v in report["runtime_path_histogram"].items():
        print(f"  {k}: {v}")
    print("layers:")
    for row in report["layers"]:
        print(
            f"  blk.{row['layer']:02d} {row['kind']} "
            f"gate={row['routed_gate']} up={row['routed_up']} down={row['routed_down']} "
            f"default={row['runtime_default_path']} opt_in={row['runtime_opt_in_path']}"
        )


def main():
    ap = argparse.ArgumentParser(description="Probe routed expert tuples and runtime path classification for a Qwen3.6 GGUF.")
    ap.add_argument("model", help="Path to GGUF")
    ap.add_argument("--json", action="store_true", help="Emit JSON")
    args = ap.parse_args()

    report = build_report(args.model)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
