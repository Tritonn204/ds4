#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


MAGIC = b"Q36LWF01"


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


def reorder_qwen_head_vector(vec: np.ndarray, num_v_heads: int) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(num_v_heads)
    gg_to_hf_perm = np.array(list(range(0, num_v_heads, 2)) + list(range(1, num_v_heads, 2)), dtype=np.int64)
    hf_to_gg_perm = np.empty_like(gg_to_hf_perm)
    hf_to_gg_perm[gg_to_hf_perm] = np.arange(num_v_heads, dtype=np.int64)
    return np.ascontiguousarray(v[hf_to_gg_perm], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export full weight-driven linear-attention layer fixture for narrow C replay")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--attn-trace", required=True, help="linear attention trace prefix or .npz")
    ap.add_argument("--core-trace", required=True, help="linear core trace prefix or .npz")
    ap.add_argument("--layer", required=True, type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    attn_path = Path(args.attn_trace)
    if attn_path.suffix != ".npz":
        attn_path = attn_path.with_suffix(".npz")
    core_path = Path(args.core_trace)
    if core_path.suffix != ".npz":
        core_path = core_path.with_suffix(".npz")

    attn_meta = json.loads(attn_path.with_suffix(".json").read_text(encoding="utf-8"))
    attn = np.load(attn_path)
    core = np.load(core_path)
    layer = args.layer

    layer_input_seq = np.asarray(attn["layer_input_seq"], dtype=np.float32)
    input_ln_seq = np.asarray(attn["input_ln_seq"], dtype=np.float32)
    qkv_seq = np.asarray(attn["in_proj_qkv_seq"], dtype=np.float32)
    z_seq = np.asarray(core["in_proj_z_seq"], dtype=np.float32)
    a_seq = np.asarray(core["in_proj_a_seq"], dtype=np.float32)
    b_seq = np.asarray(core["in_proj_b_seq"], dtype=np.float32)
    conv_raw = np.asarray(core["conv1d_raw"], dtype=np.float32)
    q_ref = np.asarray(core["query"], dtype=np.float32)
    k_ref = np.asarray(core["key"], dtype=np.float32)
    v_ref = np.asarray(core["value"], dtype=np.float32)
    beta_ref = np.asarray(core["beta"], dtype=np.float32)
    g_ref = np.asarray(core["g"], dtype=np.float32)
    core_ref = np.asarray(core["core_attn_out_pre_norm"], dtype=np.float32)
    out_in_seq = np.asarray(core["out_proj_in_seq"], dtype=np.float32)
    out_proj_out_seq = np.asarray(core["out_proj_out_seq"], dtype=np.float32)
    mixer_out = np.asarray(core["mixer_out"], dtype=np.float32)
    residual_after_mixer = layer_input_seq[-1] + mixer_out

    seq_len = int(layer_input_seq.shape[0])
    hidden = int(layer_input_seq.shape[1])

    reader = GGUFReader(args.gguf)
    tensors = tmap(reader)
    attn_norm_w = load_f32(tensors, f"blk.{layer}.attn_norm.weight")
    w_qkv = as_weight(load_f32(tensors, f"blk.{layer}.attn_qkv.weight"), 8192, 2048)
    w_qkv = reorder_qwen_qkv_v_rows(w_qkv, 2048, 4096, 32, 128)
    w_z = as_weight(load_f32(tensors, f"blk.{layer}.attn_gate.weight"), 4096, 2048)
    w_z = reorder_qwen_v_head_rows(w_z, 32, 128)
    w_a = as_weight(load_f32(tensors, f"blk.{layer}.ssm_alpha.weight"), 32, 2048)
    w_a = reorder_qwen_head_rows(w_a, 32)
    w_b = as_weight(load_f32(tensors, f"blk.{layer}.ssm_beta.weight"), 32, 2048)
    w_b = reorder_qwen_head_rows(w_b, 32)
    conv_w = load_f32(tensors, f"blk.{layer}.ssm_conv1d.weight")
    if conv_w.shape == (4, 8192):
        conv_w = conv_w.T.copy()
    elif conv_w.shape != (8192, 4):
        raise ValueError(f"unexpected conv weight shape {conv_w.shape}")
    conv_w = reorder_qwen_qkv_v_rows(conv_w, 2048, 4096, 32, 128)
    conv_w = np.ascontiguousarray(conv_w, dtype=np.float32)
    A_log = reorder_qwen_head_vector(load_f32(tensors, f"blk.{layer}.ssm_a"), 32)
    dt_bias = reorder_qwen_head_vector(load_f32(tensors, f"blk.{layer}.ssm_dt.bias"), 32)
    ssm_norm_w = load_f32(tensors, f"blk.{layer}.ssm_norm.weight")
    w_out = as_weight(load_f32(tensors, f"blk.{layer}.ssm_out.weight"), 2048, 4096)
    w_out = reorder_qwen_v_head_cols(w_out, 32, 128)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        fp.write(MAGIC)
        fp.write(struct.pack("<IIIIIIIII", layer, seq_len, hidden, 32, 16, 128, 128, 2048, 4096))
        for arr in (
            layer_input_seq, input_ln_seq, qkv_seq, z_seq, a_seq, b_seq, conv_raw,
            q_ref, k_ref, v_ref, beta_ref, g_ref, core_ref, out_in_seq,
            out_proj_out_seq, mixer_out, residual_after_mixer,
            attn_norm_w, w_qkv, w_z, w_a, w_b, conv_w, A_log, dt_bias, ssm_norm_w, w_out,
        ):
            fp.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes(order="C"))

    sidecar = {
        "attn_trace": str(attn_path),
        "core_trace": str(core_path),
        "prompt": attn_meta.get("prompt"),
        "layer": layer,
        "seq_len": seq_len,
        "out": str(out_path),
        "size_bytes": out_path.stat().st_size,
    }
    out_path.with_suffix(out_path.suffix + ".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote: {out_path}")
    print(f"size_bytes: {out_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
