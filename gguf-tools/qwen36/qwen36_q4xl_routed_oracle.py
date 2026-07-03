#!/usr/bin/env python3
import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


DEFAULT_Q8 = "/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-Q8_0.gguf"
DEFAULT_Q4 = "/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
DEFAULT_SUFFIXES = [
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(math.sqrt(float(np.mean(d * d))))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    an = float(np.linalg.norm(av))
    bn = float(np.linalg.norm(bv))
    if an == 0.0 or bn == 0.0:
        return float("nan")
    return float(np.dot(av, bv) / (an * bn))


def tensor_map(path: str) -> dict[str, object]:
    r = GGUFReader(path)
    return {t.name: t for t in r.tensors}


def expert_slice(arr: np.ndarray, suffix: str, expert: int) -> np.ndarray:
    if suffix in {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}:
        return np.asarray(arr[:, :, expert], dtype=np.float32)
    raise KeyError(suffix)


def compare_pair(arr_a: np.ndarray, arr_b: np.ndarray) -> dict:
    return {
        "rmse": rmse(arr_a, arr_b),
        "mae": mae(arr_a, arr_b),
        "max_abs": max_abs(arr_a, arr_b),
        "cosine": cosine(arr_a, arr_b),
        "shape": list(arr_a.shape),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Direct Q8_0 vs Q4_K_XL routed-expert oracle without fixture export")
    ap.add_argument("--q8-gguf", default=DEFAULT_Q8)
    ap.add_argument("--q4-gguf", default=DEFAULT_Q4)
    ap.add_argument("--layer", type=int, action="append", required=True, help="Layer index, may repeat")
    ap.add_argument("--expert", type=int, action="append", required=True, help="Expert index, may repeat")
    ap.add_argument("--suffix", action="append", default=[], help="Tensor suffix, default is gate/up/down experts")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    suffixes = args.suffix or list(DEFAULT_SUFFIXES)
    layers = sorted(set(args.layer))
    experts = sorted(set(args.expert))

    q8 = tensor_map(args.q8_gguf)
    q4 = tensor_map(args.q4_gguf)

    report = {
        "q8_gguf": args.q8_gguf,
        "q4_gguf": args.q4_gguf,
        "layers": {},
    }

    global_worst = None

    for layer in layers:
        layer_key = str(layer)
        report["layers"][layer_key] = {}
        for suffix in suffixes:
            name = f"blk.{layer}.{suffix}"
            t8 = q8[name]
            t4 = q4[name]
            print(f"[routed-oracle] dequant layer={layer} tensor={suffix} q8={t8.tensor_type.name} q4={t4.tensor_type.name}")
            arr8 = dequantize(t8.data, t8.tensor_type).astype(np.float32, copy=False)
            arr4 = dequantize(t4.data, t4.tensor_type).astype(np.float32, copy=False)

            tensor_entry = {
                "q8_type": t8.tensor_type.name,
                "q4_type": t4.tensor_type.name,
                "full_tensor": compare_pair(arr8, arr4),
                "experts": {},
            }
            for expert in experts:
                e8 = expert_slice(arr8, suffix, expert)
                e4 = expert_slice(arr4, suffix, expert)
                cmp = compare_pair(e8, e4)
                tensor_entry["experts"][str(expert)] = cmp
                if global_worst is None or cmp["rmse"] > global_worst["rmse"]:
                    global_worst = {
                        "layer": layer,
                        "suffix": suffix,
                        "expert": expert,
                        "rmse": cmp["rmse"],
                    }
            report["layers"][layer_key][suffix] = tensor_entry
            del arr8
            del arr4
            gc.collect()

    report["global_worst"] = global_worst

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if global_worst:
        print(
            "[routed-oracle] worst "
            f"layer={global_worst['layer']} suffix={global_worst['suffix']} "
            f"expert={global_worst['expert']} rmse={global_worst['rmse']:.8f}"
        )
    if args.json_out:
        print(f"[routed-oracle] json_out={args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
