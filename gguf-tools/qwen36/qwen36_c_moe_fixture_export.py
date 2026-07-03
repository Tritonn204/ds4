#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


MAGIC = b"Q36MOEF2"
HIDDEN = 2048
INTERMEDIATE = 512


def tmap(reader: GGUFReader):
    return {t.name: t for t in reader.tensors}


def load_f32(tensors, name: str) -> np.ndarray:
    t = tensors[name]
    return dequantize(t.data, t.tensor_type).astype(np.float32, copy=False)


def load_layer_weights(tensors, layer: int):
    gate_exps = load_f32(tensors, f"blk.{layer}.ffn_gate_exps.weight")
    up_exps = load_f32(tensors, f"blk.{layer}.ffn_up_exps.weight")
    down_exps = load_f32(tensors, f"blk.{layer}.ffn_down_exps.weight")
    gate_shexp = load_f32(tensors, f"blk.{layer}.ffn_gate_shexp.weight")
    up_shexp = load_f32(tensors, f"blk.{layer}.ffn_up_shexp.weight")
    down_shexp = load_f32(tensors, f"blk.{layer}.ffn_down_shexp.weight")

    if gate_exps.shape == (2048, 512, 256):
        gate_exps = np.transpose(gate_exps, (2, 1, 0)).copy()
    if up_exps.shape == (2048, 512, 256):
        up_exps = np.transpose(up_exps, (2, 1, 0)).copy()
    if down_exps.shape == (512, 2048, 256):
        down_exps = np.transpose(down_exps, (2, 1, 0)).copy()

    return {
        "gate_exps": gate_exps,
        "up_exps": up_exps,
        "down_exps": down_exps,
        "gate_shexp": gate_shexp.astype(np.float32, copy=False),
        "up_shexp": up_shexp.astype(np.float32, copy=False),
        "down_shexp": down_shexp.astype(np.float32, copy=False),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a single-layer Qwen3.6 MoE fixture for C replay")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--trace", required=True, help="Trace prefix or .npz path")
    ap.add_argument("--layer", required=True, type=int)
    ap.add_argument("--out", required=True, help="Output fixture path")
    args = ap.parse_args()

    trace_path = Path(args.trace)
    if trace_path.suffix != ".npz":
        trace_path = trace_path.with_suffix(".npz")
    meta_path = trace_path.with_suffix(".json")

    bundle = np.load(trace_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    layer = args.layer
    reader = GGUFReader(args.gguf)
    tensors = tmap(reader)
    weights = load_layer_weights(tensors, layer)

    layer_input = np.asarray(bundle[f"blk_{layer}.layer_input"], dtype=np.float32)
    mixer_out = np.asarray(bundle[f"blk_{layer}.mixer_out"], dtype=np.float32)
    post_attn_ln = np.asarray(bundle[f"blk_{layer}.post_attn_ln"], dtype=np.float32)
    residual = np.asarray(bundle[f"blk_{layer}.residual_after_mixer"], dtype=np.float32)
    hf_mlp = np.asarray(bundle[f"blk_{layer}.mlp_out"], dtype=np.float32)
    hf_layer = np.asarray(bundle[f"blk_{layer}.layer_output"], dtype=np.float32)
    router_indices = np.asarray(bundle[f"blk_{layer}.router_indices"], dtype=np.uint32)
    router_scores = np.asarray(bundle[f"blk_{layer}.router_scores"], dtype=np.float32)
    shared_gate_pre = float(np.asarray(bundle[f"blk_{layer}.shared_gate_pre_sigmoid"], dtype=np.float32).reshape(-1)[0])

    topk = int(router_indices.shape[0])
    gate_sel = np.ascontiguousarray(weights["gate_exps"][router_indices], dtype=np.float32)
    up_sel = np.ascontiguousarray(weights["up_exps"][router_indices], dtype=np.float32)
    down_sel = np.ascontiguousarray(weights["down_exps"][router_indices], dtype=np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        fp.write(MAGIC)
        fp.write(struct.pack("<IIII", layer, HIDDEN, INTERMEDIATE, topk))
        fp.write(layer_input.tobytes(order="C"))
        fp.write(mixer_out.tobytes(order="C"))
        fp.write(post_attn_ln.tobytes(order="C"))
        fp.write(residual.tobytes(order="C"))
        fp.write(hf_mlp.tobytes(order="C"))
        fp.write(hf_layer.tobytes(order="C"))
        fp.write(router_indices.astype(np.uint32, copy=False).tobytes(order="C"))
        fp.write(router_scores.tobytes(order="C"))
        fp.write(struct.pack("<f", shared_gate_pre))
        fp.write(gate_sel.tobytes(order="C"))
        fp.write(up_sel.tobytes(order="C"))
        fp.write(down_sel.tobytes(order="C"))
        fp.write(np.ascontiguousarray(weights["gate_shexp"], dtype=np.float32).tobytes(order="C"))
        fp.write(np.ascontiguousarray(weights["up_shexp"], dtype=np.float32).tobytes(order="C"))
        fp.write(np.ascontiguousarray(weights["down_shexp"], dtype=np.float32).tobytes(order="C"))

    sidecar = {
        "trace": str(trace_path),
        "prompt": meta.get("prompt"),
        "layer": layer,
        "topk": topk,
        "stage_contract": [
            "layer_input",
            "mixer_out",
            "post_attn_ln",
            "residual_after_mixer",
            "mlp_out",
            "layer_output",
        ],
        "router_indices": [int(x) for x in router_indices.tolist()],
        "router_scores": [float(x) for x in router_scores.tolist()],
        "out": str(out_path),
        "size_bytes": out_path.stat().st_size,
    }
    out_path.with_suffix(out_path.suffix + ".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote: {out_path}")
    print(f"size_bytes: {out_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
