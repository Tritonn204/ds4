#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


MAGIC = b"Q36LLSF1"


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
    ap = argparse.ArgumentParser(description="Export a stubbed full linear-attention layer fixture for C replay")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--trace", required=True, help="linear core trace prefix or .npz")
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

    layer_input = np.asarray(bundle["layer_input"], dtype=np.float32)
    core_last = np.asarray(bundle["core_attn_out_pre_norm"], dtype=np.float32)[-1]
    z_last = np.asarray(bundle["in_proj_z_seq"], dtype=np.float32)[-1].reshape(32, 128)
    mixer_out = np.asarray(bundle["mixer_out"], dtype=np.float32)
    residual_after_mixer = layer_input + mixer_out
    post_attn_ln = np.asarray(bundle["post_attn_ln"], dtype=np.float32)

    reader = GGUFReader(args.gguf)
    tensors = tmap(reader)
    norm_w = load_f32(tensors, f"blk.{layer}.ssm_norm.weight")
    out_w = as_weight(load_f32(tensors, f"blk.{layer}.ssm_out.weight"), 2048, 4096)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        fp.write(MAGIC)
        fp.write(struct.pack("<IIII", layer, 32, 128, 2048))
        for arr in (layer_input, core_last, z_last, mixer_out, residual_after_mixer, post_attn_ln, norm_w, out_w):
            fp.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes(order="C"))

    sidecar = {
        "trace": str(trace_path),
        "prompt": meta.get("prompt"),
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
