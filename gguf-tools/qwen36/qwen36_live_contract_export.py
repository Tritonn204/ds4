#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


MAGIC = b"Q36LCF01"
ROUTER_COUNT = 256


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


def build_perms(num_v_heads: int):
    gg_to_hf = np.array(list(range(0, num_v_heads, 2)) + list(range(1, num_v_heads, 2)), dtype=np.int64)
    hf_to_gg = np.empty_like(gg_to_hf)
    hf_to_gg[gg_to_hf] = np.arange(num_v_heads, dtype=np.int64)
    return gg_to_hf, hf_to_gg


def reorder_qwen_qkv_v_rows(weight: np.ndarray, key_dim: int, value_dim: int, num_v_heads: int, head_v_dim: int) -> np.ndarray:
    _, hf_to_gg = build_perms(num_v_heads)
    qk_rows = key_dim * 2
    v = np.asarray(weight[qk_rows:qk_rows + value_dim], dtype=np.float32).reshape(num_v_heads, head_v_dim, weight.shape[1])
    v = v[hf_to_gg]
    out = weight.copy()
    out[qk_rows:qk_rows + value_dim] = v.reshape(value_dim, weight.shape[1])
    return np.ascontiguousarray(out, dtype=np.float32)


def reorder_qwen_v_head_rows(weight: np.ndarray, num_v_heads: int, head_v_dim: int) -> np.ndarray:
    _, hf_to_gg = build_perms(num_v_heads)
    w = np.asarray(weight, dtype=np.float32).reshape(num_v_heads, head_v_dim, weight.shape[1])
    w = w[hf_to_gg]
    return np.ascontiguousarray(w.reshape(num_v_heads * head_v_dim, weight.shape[1]), dtype=np.float32)


def reorder_qwen_v_head_cols(weight: np.ndarray, num_v_heads: int, head_v_dim: int) -> np.ndarray:
    _, hf_to_gg = build_perms(num_v_heads)
    w = np.asarray(weight, dtype=np.float32).reshape(weight.shape[0], num_v_heads, head_v_dim)
    w = w[:, hf_to_gg, :]
    return np.ascontiguousarray(w.reshape(weight.shape[0], num_v_heads * head_v_dim), dtype=np.float32)


def reorder_qwen_head_rows(weight: np.ndarray, num_v_heads: int) -> np.ndarray:
    _, hf_to_gg = build_perms(num_v_heads)
    w = np.asarray(weight, dtype=np.float32).reshape(num_v_heads, weight.shape[1])
    w = w[hf_to_gg]
    return np.ascontiguousarray(w, dtype=np.float32)


def reorder_qwen_head_vector(vec: np.ndarray, num_v_heads: int) -> np.ndarray:
    _, hf_to_gg = build_perms(num_v_heads)
    v = np.asarray(vec, dtype=np.float32).reshape(num_v_heads)
    return np.ascontiguousarray(v[hf_to_gg], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export prompt-agnostic live contract fixture for one Qwen3.6 hybrid layer")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layer", required=True, type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    layer = args.layer
    hidden = 2048
    num_v_heads = 32
    num_k_heads = 16
    head_k_dim = 128
    head_v_dim = 128
    key_dim = 2048
    value_dim = 4096
    topk = 8
    inter = 512

    reader = GGUFReader(args.gguf)
    tensors = tmap(reader)

    attn_norm_w = np.ascontiguousarray(load_f32(tensors, f"blk.{layer}.attn_norm.weight"), dtype=np.float32)
    post_attn_norm_w = np.ascontiguousarray(load_f32(tensors, f"blk.{layer}.post_attention_norm.weight"), dtype=np.float32)
    w_qkv = as_weight(load_f32(tensors, f"blk.{layer}.attn_qkv.weight"), 8192, 2048)
    w_qkv = reorder_qwen_qkv_v_rows(w_qkv, key_dim, value_dim, num_v_heads, head_v_dim)
    w_z = as_weight(load_f32(tensors, f"blk.{layer}.attn_gate.weight"), 4096, 2048)
    w_z = reorder_qwen_v_head_rows(w_z, num_v_heads, head_v_dim)
    w_a = as_weight(load_f32(tensors, f"blk.{layer}.ssm_alpha.weight"), num_v_heads, hidden)
    w_a = reorder_qwen_head_rows(w_a, num_v_heads)
    w_b = as_weight(load_f32(tensors, f"blk.{layer}.ssm_beta.weight"), num_v_heads, hidden)
    w_b = reorder_qwen_head_rows(w_b, num_v_heads)

    conv_w = load_f32(tensors, f"blk.{layer}.ssm_conv1d.weight")
    if conv_w.shape == (4, 8192):
        conv_w = conv_w.T.copy()
    elif conv_w.shape != (8192, 4):
        raise ValueError(f"unexpected conv weight shape {conv_w.shape}")
    conv_w = reorder_qwen_qkv_v_rows(np.ascontiguousarray(conv_w, dtype=np.float32), key_dim, value_dim, num_v_heads, head_v_dim)

    A_log = reorder_qwen_head_vector(load_f32(tensors, f"blk.{layer}.ssm_a"), num_v_heads)
    dt_bias = reorder_qwen_head_vector(load_f32(tensors, f"blk.{layer}.ssm_dt.bias"), num_v_heads)
    ssm_norm_w = np.ascontiguousarray(load_f32(tensors, f"blk.{layer}.ssm_norm.weight"), dtype=np.float32)
    w_out = as_weight(load_f32(tensors, f"blk.{layer}.ssm_out.weight"), hidden, value_dim)
    w_out = reorder_qwen_v_head_cols(w_out, num_v_heads, head_v_dim)

    router_w = load_f32(tensors, f"blk.{layer}.ffn_gate_inp.weight")
    if router_w.shape == (hidden, ROUTER_COUNT):
        router_w = router_w.T.copy()
    elif router_w.shape != (ROUTER_COUNT, hidden):
        raise ValueError(f"unexpected router weight shape {router_w.shape}")
    router_w = np.ascontiguousarray(router_w, dtype=np.float32)

    gate_shexp = as_weight(load_f32(tensors, f"blk.{layer}.ffn_gate_shexp.weight"), inter, hidden)
    up_shexp = as_weight(load_f32(tensors, f"blk.{layer}.ffn_up_shexp.weight"), inter, hidden)
    down_shexp = as_weight(load_f32(tensors, f"blk.{layer}.ffn_down_shexp.weight"), hidden, inter)
    gate_inp_shexp = np.ascontiguousarray(load_f32(tensors, f"blk.{layer}.ffn_gate_inp_shexp.weight").reshape(hidden), dtype=np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        fp.write(MAGIC)
        fp.write(struct.pack("<IIIIIIIIII", layer, hidden, num_v_heads, num_k_heads, head_k_dim, head_v_dim, key_dim, value_dim, topk, inter))
        for arr in (
            attn_norm_w,
            post_attn_norm_w,
            w_qkv,
            w_z,
            w_a,
            w_b,
            conv_w,
            A_log,
            dt_bias,
            ssm_norm_w,
            w_out,
            router_w,
            gate_shexp,
            up_shexp,
            down_shexp,
            gate_inp_shexp,
        ):
            fp.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes(order="C"))

    sidecar = {
        "gguf": str(args.gguf),
        "layer": layer,
        "hidden": hidden,
        "num_v_heads": num_v_heads,
        "num_k_heads": num_k_heads,
        "head_k_dim": head_k_dim,
        "head_v_dim": head_v_dim,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "topk": topk,
        "inter": inter,
        "out": str(out_path),
        "size_bytes": out_path.stat().st_size,
    }
    out_path.with_suffix(out_path.suffix + ".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote: {out_path}")
    print(f"size_bytes: {out_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
