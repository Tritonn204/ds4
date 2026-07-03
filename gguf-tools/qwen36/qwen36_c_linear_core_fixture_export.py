#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


MAGIC = b"Q36LCRF2"


def tmap(reader: GGUFReader):
    return {t.name: t for t in reader.tensors}


def load_f32(tensors, name: str) -> np.ndarray:
    t = tensors[name]
    return dequantize(t.data, t.tensor_type).astype(np.float32, copy=False)


def reorder_qwen_head_vector(vec: np.ndarray, num_v_heads: int) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(num_v_heads)
    gg_to_hf_perm = np.array(list(range(0, num_v_heads, 2)) + list(range(1, num_v_heads, 2)), dtype=np.int64)
    hf_to_gg_perm = np.empty_like(gg_to_hf_perm)
    hf_to_gg_perm[gg_to_hf_perm] = np.arange(num_v_heads, dtype=np.int64)
    return np.ascontiguousarray(v[hf_to_gg_perm], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export DeltaNet core-boundary fixture for C replay")
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

    conv_raw = np.asarray(bundle["conv1d_raw"], dtype=np.float32)
    a_seq = np.asarray(bundle["in_proj_a_seq"], dtype=np.float32)
    b_seq = np.asarray(bundle["in_proj_b_seq"], dtype=np.float32)
    q_ref = np.asarray(bundle["query"], dtype=np.float32)
    k_ref = np.asarray(bundle["key"], dtype=np.float32)
    v_ref = np.asarray(bundle["value"], dtype=np.float32)
    beta_ref = np.asarray(bundle["beta"], dtype=np.float32)
    g_ref = np.asarray(bundle["g"], dtype=np.float32)
    core_ref = np.asarray(bundle["core_attn_out_pre_norm"], dtype=np.float32)

    seq_len = int(q_ref.shape[0])
    reader = GGUFReader(args.gguf)
    tensors = tmap(reader)
    A_log = reorder_qwen_head_vector(load_f32(tensors, f"blk.{layer}.ssm_a"), 32)
    dt_bias = reorder_qwen_head_vector(load_f32(tensors, f"blk.{layer}.ssm_dt.bias"), 32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        fp.write(MAGIC)
        fp.write(struct.pack("<IIIIIIII", layer, seq_len, 32, 16, 128, 128, 2048, 4096))
        for arr in (conv_raw, a_seq, b_seq, A_log, dt_bias, q_ref, k_ref, v_ref, beta_ref, g_ref, core_ref):
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
