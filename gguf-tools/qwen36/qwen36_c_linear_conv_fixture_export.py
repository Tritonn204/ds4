#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


MAGIC = b"Q36LCVF1"


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


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a full-sequence linear-attention conv fixture for C replay")
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

    input_ln_seq = np.asarray(bundle["input_ln_seq"], dtype=np.float32)
    qkv_seq = np.asarray(bundle["in_proj_qkv_seq"], dtype=np.float32)
    conv_raw = np.asarray(bundle["conv1d_raw"], dtype=np.float32)
    seq_len = int(input_ln_seq.shape[0])
    conv_post = silu(conv_raw[:, :seq_len].T).astype(np.float32, copy=False)

    reader = GGUFReader(args.gguf)
    tensors = tmap(reader)
    w_qkv = as_weight(load_f32(tensors, f"blk.{layer}.attn_qkv.weight"), 8192, 2048)
    conv_w = load_f32(tensors, f"blk.{layer}.ssm_conv1d.weight")
    if conv_w.shape == (4, 8192):
        conv_w = conv_w.T.copy()
    elif conv_w.shape != (8192, 4):
        raise ValueError(f"unexpected conv weight shape {conv_w.shape}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        fp.write(MAGIC)
        fp.write(struct.pack("<IIII", layer, seq_len, 2048, 8192))
        for arr in (input_ln_seq, qkv_seq, conv_post):
            fp.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes(order="C"))
        fp.write(np.ascontiguousarray(conv_w, dtype=np.float32).tobytes(order="C"))
        fp.write(np.ascontiguousarray(w_qkv, dtype=np.float32).tobytes(order="C"))

    sidecar = {
        "trace": str(trace_path),
        "prompt": meta.get("prompt"),
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
