#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


MAGIC = b"Q36DWF02"


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
    # GGUF stores V heads as even heads first, then odd heads. Reorder to HF runtime head order.
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
    ap = argparse.ArgumentParser(description="Export a full weight-driven Qwen3.6 decoder-layer fixture for narrow C replay")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--attn-trace", required=True, help="linear attention trace prefix or .npz")
    ap.add_argument("--core-trace", required=True, help="linear core trace prefix or .npz")
    ap.add_argument("--layer-trace", required=True, help="layer trace prefix or .npz")
    ap.add_argument("--layer", required=True, type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    attn_path = Path(args.attn_trace)
    if attn_path.suffix != ".npz":
        attn_path = attn_path.with_suffix(".npz")
    core_path = Path(args.core_trace)
    if core_path.suffix != ".npz":
        core_path = core_path.with_suffix(".npz")
    layer_path = Path(args.layer_trace)
    if layer_path.suffix != ".npz":
        layer_path = layer_path.with_suffix(".npz")

    attn_meta = json.loads(attn_path.with_suffix(".json").read_text(encoding="utf-8"))
    attn = np.load(attn_path)
    core = np.load(core_path)
    layer_bundle = np.load(layer_path)
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

    layer_input_last = np.asarray(layer_bundle[f"blk_{layer}.layer_input"], dtype=np.float32)
    residual_after_mixer = np.asarray(layer_bundle[f"blk_{layer}.residual_after_mixer"], dtype=np.float32)
    post_attn_ln = np.asarray(layer_bundle[f"blk_{layer}.post_attn_ln"], dtype=np.float32)
    mlp_out = np.asarray(layer_bundle[f"blk_{layer}.mlp_out"], dtype=np.float32)
    layer_output = np.asarray(layer_bundle[f"blk_{layer}.layer_output"], dtype=np.float32)
    layer_input_layer_seq = np.asarray(layer_bundle[f"blk_{layer}.layer_input_seq"], dtype=np.float32)
    residual_after_mixer_seq = np.asarray(layer_bundle[f"blk_{layer}.residual_after_mixer_seq"], dtype=np.float32)
    post_attn_ln_seq = np.asarray(layer_bundle[f"blk_{layer}.post_attn_ln_seq"], dtype=np.float32)
    mlp_out_seq = np.asarray(layer_bundle[f"blk_{layer}.mlp_out_seq"], dtype=np.float32)
    layer_output_seq = np.asarray(layer_bundle[f"blk_{layer}.layer_output_seq"], dtype=np.float32)
    seq_len = int(layer_input_seq.shape[0])
    hidden = int(layer_input_seq.shape[1])
    router_logits = np.asarray(layer_bundle[f"blk_{layer}.router_logits"], dtype=np.float32)
    router_indices = np.asarray(layer_bundle[f"blk_{layer}.router_indices"], dtype=np.uint32)
    router_scores = np.asarray(layer_bundle[f"blk_{layer}.router_scores"], dtype=np.float32)
    shared_gate_pre = np.asarray(layer_bundle[f"blk_{layer}.shared_gate_pre_sigmoid"], dtype=np.float32).reshape(-1)
    router_logits_seq = np.asarray(layer_bundle[f"blk_{layer}.router_logits_seq"], dtype=np.float32)
    router_indices_seq = np.asarray(layer_bundle[f"blk_{layer}.router_indices_seq"], dtype=np.uint32)
    router_scores_seq = np.asarray(layer_bundle[f"blk_{layer}.router_scores_seq"], dtype=np.float32)
    shared_gate_pre_seq = np.asarray(layer_bundle[f"blk_{layer}.shared_gate_pre_sigmoid_seq"], dtype=np.float32).reshape(seq_len, 1)

    topk = int(router_indices.shape[0])
    inter = 512

    reader = GGUFReader(args.gguf)
    tensors = tmap(reader)

    attn_norm_w = load_f32(tensors, f"blk.{layer}.attn_norm.weight")
    post_attn_norm_w = load_f32(tensors, f"blk.{layer}.post_attention_norm.weight")
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

    router_w = load_f32(tensors, f"blk.{layer}.ffn_gate_inp.weight")
    if router_w.shape == (2048, 256):
        router_w = router_w.T.copy()
    elif router_w.shape != (256, 2048):
        raise ValueError(f"unexpected router weight shape {router_w.shape}")

    gate_exps = load_f32(tensors, f"blk.{layer}.ffn_gate_exps.weight")
    up_exps = load_f32(tensors, f"blk.{layer}.ffn_up_exps.weight")
    down_exps = load_f32(tensors, f"blk.{layer}.ffn_down_exps.weight")
    if gate_exps.shape == (2048, 512, 256):
        gate_exps = np.transpose(gate_exps, (2, 1, 0)).copy()
    if up_exps.shape == (2048, 512, 256):
        up_exps = np.transpose(up_exps, (2, 1, 0)).copy()
    if down_exps.shape == (512, 2048, 256):
        down_exps = np.transpose(down_exps, (2, 1, 0)).copy()
    flat_router_indices = router_indices_seq.reshape(-1)
    unique_experts, inverse = np.unique(flat_router_indices, return_inverse=True)
    router_union_pos_seq = inverse.reshape(router_indices_seq.shape).astype(np.uint32, copy=False)
    gate_sel = np.ascontiguousarray(gate_exps[unique_experts], dtype=np.float32)
    up_sel = np.ascontiguousarray(up_exps[unique_experts], dtype=np.float32)
    down_sel = np.ascontiguousarray(down_exps[unique_experts], dtype=np.float32)

    gate_shexp = np.ascontiguousarray(load_f32(tensors, f"blk.{layer}.ffn_gate_shexp.weight"), dtype=np.float32)
    up_shexp = np.ascontiguousarray(load_f32(tensors, f"blk.{layer}.ffn_up_shexp.weight"), dtype=np.float32)
    down_shexp = np.ascontiguousarray(load_f32(tensors, f"blk.{layer}.ffn_down_shexp.weight"), dtype=np.float32)
    gate_inp_shexp = np.ascontiguousarray(load_f32(tensors, f"blk.{layer}.ffn_gate_inp_shexp.weight").reshape(-1), dtype=np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        fp.write(MAGIC)
        fp.write(struct.pack("<IIIIIIIIIII", layer, seq_len, hidden, 32, 16, 128, 128, 2048, 4096, topk, int(unique_experts.shape[0])))
        for arr in (
            layer_input_seq, input_ln_seq, qkv_seq, z_seq, a_seq, b_seq, conv_raw,
            q_ref, k_ref, v_ref, beta_ref, g_ref, core_ref, out_in_seq, out_proj_out_seq,
            mixer_out, layer_input_last, residual_after_mixer, post_attn_ln, mlp_out, layer_output,
            router_logits, router_indices.astype(np.float32), router_scores, shared_gate_pre,
            layer_input_layer_seq, residual_after_mixer_seq, post_attn_ln_seq, mlp_out_seq, layer_output_seq,
            router_logits_seq, router_indices_seq.astype(np.float32), router_scores_seq, shared_gate_pre_seq,
            attn_norm_w, post_attn_norm_w, w_qkv, w_z, w_a, w_b, conv_w, A_log, dt_bias, ssm_norm_w, w_out,
            np.ascontiguousarray(router_w, dtype=np.float32),
            unique_experts.astype(np.float32), router_union_pos_seq.astype(np.float32),
            gate_sel, up_sel, down_sel, gate_shexp, up_shexp, down_shexp, gate_inp_shexp,
        ):
            fp.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes(order="C"))

    sidecar = {
        "attn_trace": str(attn_path),
        "core_trace": str(core_path),
        "layer_trace": str(layer_path),
        "prompt": attn_meta.get("prompt"),
        "layer": layer,
        "seq_len": seq_len,
        "topk": topk,
        "out": str(out_path),
        "size_bytes": out_path.stat().st_size,
    }
    out_path.with_suffix(out_path.suffix + ".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote: {out_path}")
    print(f"size_bytes: {out_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
