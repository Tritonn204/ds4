#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


MAGIC = b"Q36LINF1"


def tmap(reader: GGUFReader):
    return {t.name: t for t in reader.tensors}


def load_f32(tensors, name: str) -> np.ndarray:
    t = tensors[name]
    return dequantize(t.data, t.tensor_type).astype(np.float32, copy=False)


def as_weight(arr: np.ndarray, out_rows: int, in_cols: int) -> np.ndarray:
    if arr.shape == (out_rows, in_cols):
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.shape == (in_cols, out_rows):
        return np.ascontiguousarray(arr.T, dtype=np.float32)
    raise ValueError(f"unexpected weight shape {arr.shape}, wanted {(out_rows, in_cols)} or {(in_cols, out_rows)}")


def reorder_qwen_qkv_v_rows(weight: np.ndarray, key_dim: int, value_dim: int, num_v_heads: int, head_v_dim: int) -> np.ndarray:
    qk_rows = key_dim * 2
    v = np.asarray(weight[qk_rows:qk_rows + value_dim], dtype=np.float32).reshape(num_v_heads, head_v_dim, weight.shape[1])
    gg_to_hf_perm = np.array(list(range(0, num_v_heads, 2)) + list(range(1, num_v_heads, 2)), dtype=np.int64)
    hf_to_gg_perm = np.empty_like(gg_to_hf_perm)
    hf_to_gg_perm[gg_to_hf_perm] = np.arange(num_v_heads, dtype=np.int64)
    v = v[hf_to_gg_perm]
    out = weight.copy()
    out[qk_rows:qk_rows + value_dim] = v.reshape(value_dim, weight.shape[1])
    return np.ascontiguousarray(out, dtype=np.float32)


def reorder_qwen_v_head_rows(weight: np.ndarray, num_v_heads: int, head_v_dim: int) -> np.ndarray:
    w = np.asarray(weight, dtype=np.float32).reshape(num_v_heads, head_v_dim, weight.shape[1])
    gg_to_hf_perm = np.array(list(range(0, num_v_heads, 2)) + list(range(1, num_v_heads, 2)), dtype=np.int64)
    hf_to_gg_perm = np.empty_like(gg_to_hf_perm)
    hf_to_gg_perm[gg_to_hf_perm] = np.arange(num_v_heads, dtype=np.int64)
    w = w[hf_to_gg_perm]
    return np.ascontiguousarray(w.reshape(num_v_heads * head_v_dim, weight.shape[1]), dtype=np.float32)


def reorder_qwen_v_head_cols(weight: np.ndarray, num_v_heads: int, head_v_dim: int) -> np.ndarray:
    w = np.asarray(weight, dtype=np.float32).reshape(weight.shape[0], num_v_heads, head_v_dim)
    gg_to_hf_perm = np.array(list(range(0, num_v_heads, 2)) + list(range(1, num_v_heads, 2)), dtype=np.int64)
    hf_to_gg_perm = np.empty_like(gg_to_hf_perm)
    hf_to_gg_perm[gg_to_hf_perm] = np.arange(num_v_heads, dtype=np.int64)
    w = w[:, hf_to_gg_perm, :]
    return np.ascontiguousarray(w.reshape(weight.shape[0], num_v_heads * head_v_dim), dtype=np.float32)


def reorder_qwen_head_rows(weight: np.ndarray, num_v_heads: int) -> np.ndarray:
    w = np.asarray(weight, dtype=np.float32).reshape(num_v_heads, weight.shape[1])
    gg_to_hf_perm = np.array(list(range(0, num_v_heads, 2)) + list(range(1, num_v_heads, 2)), dtype=np.int64)
    hf_to_gg_perm = np.empty_like(gg_to_hf_perm)
    hf_to_gg_perm[gg_to_hf_perm] = np.arange(num_v_heads, dtype=np.int64)
    w = w[hf_to_gg_perm]
    return np.ascontiguousarray(w, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a linear-attention projection fixture for C replay")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--trace", required=True, help="linear attention trace prefix or .npz")
    ap.add_argument("--layer", required=True, type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    trace_path = Path(args.trace)
    if trace_path.suffix != ".npz":
        trace_path = trace_path.with_suffix(".npz")
    meta_path = trace_path.with_suffix(".json")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bundle = np.load(trace_path)
    layer = args.layer

    input_ln = np.asarray(bundle["input_ln"], dtype=np.float32)
    qkv_ref = np.asarray(bundle["in_proj_qkv"], dtype=np.float32)
    z_ref = np.asarray(bundle["in_proj_z"], dtype=np.float32)
    a_ref = np.asarray(bundle["in_proj_a"], dtype=np.float32)
    b_ref = np.asarray(bundle["in_proj_b"], dtype=np.float32)
    out_in = np.asarray(bundle["out_proj_in"], dtype=np.float32)
    out_ref = np.asarray(bundle["out_proj_out"], dtype=np.float32)
    mixer_out = np.asarray(bundle["mixer_out"], dtype=np.float32)
    layer_out = np.asarray(bundle["layer_output"], dtype=np.float32)

    reader = GGUFReader(args.gguf)
    tensors = tmap(reader)
    w_qkv = as_weight(load_f32(tensors, f"blk.{layer}.attn_qkv.weight"), 8192, 2048)
    w_qkv = reorder_qwen_qkv_v_rows(w_qkv, 2048, 4096, 32, 128)
    w_z = as_weight(load_f32(tensors, f"blk.{layer}.attn_gate.weight"), 4096, 2048)
    w_z = reorder_qwen_v_head_rows(w_z, 32, 128)
    w_a = as_weight(load_f32(tensors, f"blk.{layer}.ssm_alpha.weight"), 32, 2048)
    w_a = reorder_qwen_head_rows(w_a, 32)
    w_b = as_weight(load_f32(tensors, f"blk.{layer}.ssm_beta.weight"), 32, 2048)
    w_b = reorder_qwen_head_rows(w_b, 32)
    w_out = as_weight(load_f32(tensors, f"blk.{layer}.ssm_out.weight"), 2048, 4096)
    w_out = reorder_qwen_v_head_cols(w_out, 32, 128)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        fp.write(MAGIC)
        fp.write(struct.pack("<IIIIIIII", layer, 2048, 8192, 4096, 32, 32, 4096, 2048))
        for arr in (input_ln, qkv_ref, z_ref, a_ref, b_ref, out_in, out_ref, mixer_out, layer_out):
            fp.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes(order="C"))
        for arr in (w_qkv, w_z, w_a, w_b, w_out):
            fp.write(arr.tobytes(order="C"))

    sidecar = {
        "trace": str(trace_path),
        "prompt": meta.get("prompt"),
        "layer": layer,
        "captured": meta.get("captured"),
        "out": str(out_path),
        "size_bytes": out_path.stat().st_size,
    }
    out_path.with_suffix(out_path.suffix + ".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote: {out_path}")
    print(f"size_bytes: {out_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
