#!/usr/bin/env python3
import argparse
import json
import math
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_Q8 = "/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-Q8_0.gguf"
DEFAULT_Q4 = "/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
DEFAULT_EXPORTER = "/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_live_contract_export.py"
MAGIC = b"Q36LCF01"
HYBRID_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38]


def slugify_label(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch.lower())
        elif ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "variant"


@dataclass
class Fixture:
    layer: int
    hidden: int
    num_v_heads: int
    num_k_heads: int
    head_k_dim: int
    head_v_dim: int
    key_dim: int
    value_dim: int
    topk: int
    inter: int
    tensors: dict[str, np.ndarray]


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


def export_fixture(exporter: str, gguf: str, layer: int, out_path: Path) -> None:
    cmd = [
        "python3",
        exporter,
        "--gguf",
        gguf,
        "--layer",
        str(layer),
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def read_f32(fp, count: int) -> np.ndarray:
    raw = fp.read(count * 4)
    if len(raw) != count * 4:
        raise EOFError("short fixture payload")
    return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)


def load_fixture(path: Path) -> Fixture:
    with path.open("rb") as fp:
        magic = fp.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic")
        hdr = struct.unpack("<IIIIIIIIII", fp.read(40))
        layer, hidden, num_v_heads, num_k_heads, head_k_dim, head_v_dim, key_dim, value_dim, topk, inter = hdr
        tensors = {
            "attn_norm_w": read_f32(fp, hidden).reshape(hidden),
            "post_attn_norm_w": read_f32(fp, hidden).reshape(hidden),
            "w_qkv": read_f32(fp, (key_dim * 2 + value_dim) * hidden).reshape(key_dim * 2 + value_dim, hidden),
            "w_z": read_f32(fp, value_dim * hidden).reshape(value_dim, hidden),
            "w_a": read_f32(fp, num_v_heads * hidden).reshape(num_v_heads, hidden),
            "w_b": read_f32(fp, num_v_heads * hidden).reshape(num_v_heads, hidden),
            "conv_w": read_f32(fp, (key_dim * 2 + value_dim) * 4).reshape(key_dim * 2 + value_dim, 4),
            "A_log": read_f32(fp, num_v_heads).reshape(num_v_heads),
            "dt_bias": read_f32(fp, num_v_heads).reshape(num_v_heads),
            "ssm_norm_w": read_f32(fp, head_v_dim).reshape(head_v_dim),
            "w_out": read_f32(fp, hidden * value_dim).reshape(hidden, value_dim),
            "router_w": read_f32(fp, 256 * hidden).reshape(256, hidden),
            "gate_shexp": read_f32(fp, inter * hidden).reshape(inter, hidden),
            "up_shexp": read_f32(fp, inter * hidden).reshape(inter, hidden),
            "down_shexp": read_f32(fp, hidden * inter).reshape(hidden, inter),
            "gate_inp_shexp": read_f32(fp, hidden).reshape(hidden),
        }
    return Fixture(layer, hidden, num_v_heads, num_k_heads, head_k_dim, head_v_dim, key_dim, value_dim, topk, inter, tensors)


def compare_fixture_pair(q8: Fixture, q4: Fixture) -> dict:
    if (
        q8.layer != q4.layer
        or q8.hidden != q4.hidden
        or q8.key_dim != q4.key_dim
        or q8.value_dim != q4.value_dim
        or q8.inter != q4.inter
    ):
        raise ValueError(f"fixture metadata mismatch for layer {q8.layer}")
    out = {"layer": q8.layer, "tensors": {}}
    worst_name = None
    worst_rmse = -1.0
    for name, arr8 in q8.tensors.items():
        arr4 = q4.tensors[name]
        item = {
            "rmse": rmse(arr8, arr4),
            "mae": mae(arr8, arr4),
            "max_abs": max_abs(arr8, arr4),
            "cosine": cosine(arr8, arr4),
            "shape": list(arr8.shape),
        }
        out["tensors"][name] = item
        if item["rmse"] > worst_rmse:
            worst_rmse = item["rmse"]
            worst_name = name
    out["worst_tensor"] = worst_name
    out["worst_rmse"] = worst_rmse
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare cached hybrid live fixtures between two Qwen3.6 GGUFs")
    ap.add_argument("--q8-gguf", default=DEFAULT_Q8)
    ap.add_argument("--q4-gguf", default=DEFAULT_Q4)
    ap.add_argument("--label-a", default="q8")
    ap.add_argument("--label-b", default="q4xl")
    ap.add_argument("--exporter", default=DEFAULT_EXPORTER)
    ap.add_argument("--out-dir", default="/tmp/qwen36_q4xl_fixture_study")
    ap.add_argument("--json-out", default="/tmp/qwen36_q4xl_fixture_study.json")
    ap.add_argument("--export-missing", action="store_true",
                    help="Export only the fixture files that are missing from --out-dir")
    ap.add_argument("--rebuild", action="store_true",
                    help="Re-export every fixture file even if a cached copy already exists")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    q8_dir = out_dir / f"{slugify_label(args.label_a)}_shared"
    q4_dir = out_dir / f"{slugify_label(args.label_b)}_shared"
    q8_dir.mkdir(parents=True, exist_ok=True)
    q4_dir.mkdir(parents=True, exist_ok=True)

    for layer in HYBRID_LAYERS:
        q8_path = q8_dir / f"blk{layer}.live.bin"
        q4_path = q4_dir / f"blk{layer}.live.bin"
        if args.rebuild or (args.export_missing and not q8_path.exists()):
            export_fixture(args.exporter, args.q8_gguf, layer, q8_path)
        if args.rebuild or (args.export_missing and not q4_path.exists()):
            export_fixture(args.exporter, args.q4_gguf, layer, q4_path)
        if not q8_path.exists() or not q4_path.exists():
            missing = []
            if not q8_path.exists():
                missing.append(str(q8_path))
            if not q4_path.exists():
                missing.append(str(q4_path))
            raise RuntimeError(
                "missing cached fixtures; rerun with --export-missing or --rebuild. "
                f"missing={missing}"
            )

    layer_reports = []
    global_worst = None
    for layer in HYBRID_LAYERS:
        rep = compare_fixture_pair(
            load_fixture(q8_dir / f"blk{layer}.live.bin"),
            load_fixture(q4_dir / f"blk{layer}.live.bin"),
        )
        layer_reports.append(rep)
        if global_worst is None or rep["worst_rmse"] > global_worst["rmse"]:
            global_worst = {"layer": layer, "tensor": rep["worst_tensor"], "rmse": rep["worst_rmse"]}

    tensor_agg: dict[str, list[float]] = {}
    for rep in layer_reports:
        for name, item in rep["tensors"].items():
            tensor_agg.setdefault(name, []).append(item["rmse"])

    tensor_summary = {}
    for name, values in tensor_agg.items():
        tensor_summary[name] = {
            "mean_rmse": float(np.mean(values)),
            "max_rmse": float(np.max(values)),
            "min_rmse": float(np.min(values)),
        }

    result = {
        "q8_gguf": args.q8_gguf,
        "q4_gguf": args.q4_gguf,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "out_dir": str(out_dir),
        "layers": layer_reports,
        "tensor_summary": tensor_summary,
        "global_worst": global_worst,
    }

    Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.rebuild:
        print("[q4xl-fixture-study] rebuilt hybrid live fixtures")
    elif args.export_missing:
        print("[q4xl-fixture-study] completed with export-missing")
    else:
        print("[q4xl-fixture-study] compared cached hybrid live fixtures")
    print(f"[q4xl-fixture-study] {args.label_a}_dir={q8_dir}")
    print(f"[q4xl-fixture-study] {args.label_b}_dir={q4_dir}")
    if global_worst:
        print(
            "[q4xl-fixture-study] worst "
            f"layer={global_worst['layer']} tensor={global_worst['tensor']} rmse={global_worst['rmse']:.8f}"
        )
    print("[q4xl-fixture-study] tensor mean/max rmse")
    for name, item in sorted(tensor_summary.items(), key=lambda kv: kv[1]["max_rmse"], reverse=True):
        print(
            f"  {name}: mean_rmse={item['mean_rmse']:.8f} "
            f"max_rmse={item['max_rmse']:.8f}"
        )
    print(f"[q4xl-fixture-study] json_out={args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
