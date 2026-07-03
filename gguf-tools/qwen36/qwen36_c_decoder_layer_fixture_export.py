#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


MAGIC = b"Q36DLF01"


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
    raise ValueError(f"unexpected weight shape {arr.shape}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export full decoder-layer fixture for narrow C replay")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--linear-trace", required=True, help="linear core trace prefix or .npz")
    ap.add_argument("--layer-trace", required=True, help="layer trace prefix or .npz")
    ap.add_argument("--layer", required=True, type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    linear_path = Path(args.linear_trace)
    if linear_path.suffix != ".npz":
        linear_path = linear_path.with_suffix(".npz")
    layer_path = Path(args.layer_trace)
    if layer_path.suffix != ".npz":
        layer_path = layer_path.with_suffix(".npz")

    linear_bundle = np.load(linear_path)
    layer_bundle = np.load(layer_path)
    layer = args.layer

    layer_input = np.asarray(linear_bundle["layer_input"], dtype=np.float32)
    q = np.asarray(linear_bundle["query"], dtype=np.float32)
    k = np.asarray(linear_bundle["key"], dtype=np.float32)
    v = np.asarray(linear_bundle["value"], dtype=np.float32)
    beta = np.asarray(linear_bundle["beta"], dtype=np.float32)
    g = np.asarray(linear_bundle["g"], dtype=np.float32)
    z_last = np.asarray(linear_bundle["in_proj_z_seq"], dtype=np.float32)[-1].reshape(32, 128)
    mixer_out = np.asarray(linear_bundle["mixer_out"], dtype=np.float32)
    residual_after_mixer = layer_input + mixer_out
    post_attn_ln = np.asarray(layer_bundle[f"blk_{layer}.post_attn_ln"], dtype=np.float32)
    mlp_out = np.asarray(layer_bundle[f"blk_{layer}.mlp_out"], dtype=np.float32)
    layer_output = np.asarray(layer_bundle[f"blk_{layer}.layer_output"], dtype=np.float32)
    router_indices = np.asarray(layer_bundle[f"blk_{layer}.router_indices"], dtype=np.uint32)
    router_scores = np.asarray(layer_bundle[f"blk_{layer}.router_scores"], dtype=np.float32)
    seq_len = int(q.shape[0])

    reader = GGUFReader(args.gguf)
    tensors = tmap(reader)
    norm_w = load_f32(tensors, f"blk.{layer}.ssm_norm.weight")
    out_w = as_weight(load_f32(tensors, f"blk.{layer}.ssm_out.weight"), 2048, 4096)

    gate_exps = load_f32(tensors, f"blk.{layer}.ffn_gate_exps.weight")
    up_exps = load_f32(tensors, f"blk.{layer}.ffn_up_exps.weight")
    down_exps = load_f32(tensors, f"blk.{layer}.ffn_down_exps.weight")
    gate_shexp = load_f32(tensors, f"blk.{layer}.ffn_gate_shexp.weight")
    up_shexp = load_f32(tensors, f"blk.{layer}.ffn_up_shexp.weight")
    down_shexp = load_f32(tensors, f"blk.{layer}.ffn_down_shexp.weight")
    gate_inp_shexp = load_f32(tensors, f"blk.{layer}.ffn_gate_inp_shexp.weight")

    if gate_exps.shape == (2048, 512, 256):
        gate_exps = np.transpose(gate_exps, (2, 1, 0)).copy()
    if up_exps.shape == (2048, 512, 256):
        up_exps = np.transpose(up_exps, (2, 1, 0)).copy()
    if down_exps.shape == (512, 2048, 256):
        down_exps = np.transpose(down_exps, (2, 1, 0)).copy()

    gate_sel = np.ascontiguousarray(gate_exps[router_indices], dtype=np.float32)
    up_sel = np.ascontiguousarray(up_exps[router_indices], dtype=np.float32)
    down_sel = np.ascontiguousarray(down_exps[router_indices], dtype=np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        fp.write(MAGIC)
        fp.write(struct.pack("<IIIIIIIII", layer, seq_len, 32, 16, 128, 128, 8, 512, 2048))
        for arr in (
            layer_input, q, k, v, beta, g, z_last, mixer_out, residual_after_mixer,
            post_attn_ln, mlp_out, layer_output, router_indices.astype(np.float32),
            router_scores, gate_sel, up_sel, down_sel,
            np.ascontiguousarray(gate_shexp, dtype=np.float32),
            np.ascontiguousarray(up_shexp, dtype=np.float32),
            np.ascontiguousarray(down_shexp, dtype=np.float32),
            np.ascontiguousarray(gate_inp_shexp.reshape(-1), dtype=np.float32),
            np.ascontiguousarray(norm_w, dtype=np.float32),
            np.ascontiguousarray(out_w, dtype=np.float32),
        ):
            fp.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes(order="C"))

    sidecar = {
        "linear_trace": str(linear_path),
        "layer_trace": str(layer_path),
        "layer": layer,
        "out": str(out_path),
        "size_bytes": out_path.stat().st_size,
    }
    out_path.with_suffix(out_path.suffix + ".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote: {out_path}")
    print(f"size_bytes: {out_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
