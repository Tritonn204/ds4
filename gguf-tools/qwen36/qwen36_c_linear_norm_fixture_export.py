#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


MAGIC = b"Q36LNRF1"


def tmap(reader: GGUFReader):
    return {t.name: t for t in reader.tensors}


def load_f32(tensors, name: str) -> np.ndarray:
    t = tensors[name]
    return dequantize(t.data, t.tensor_type).astype(np.float32, copy=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export gated RMS norm fixture for Qwen3.6 linear-attention C replay")
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

    core = np.asarray(bundle["core_attn_out_pre_norm"], dtype=np.float32)
    z_seq = np.asarray(bundle["in_proj_z_seq"], dtype=np.float32)
    out_in = np.asarray(bundle["out_proj_in_seq"], dtype=np.float32)
    seq_len = int(core.shape[0])

    reader = GGUFReader(args.gguf)
    tensors = tmap(reader)
    norm_w = load_f32(tensors, f"blk.{layer}.ssm_norm.weight")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        fp.write(MAGIC)
        fp.write(struct.pack("<IIII", layer, seq_len, 32, 128))
        for arr in (core, z_seq, out_in, norm_w):
            fp.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes(order="C"))

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
