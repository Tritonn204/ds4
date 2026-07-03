#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


def parse_layers(text: str | None, fallback: list[int]) -> list[int]:
    if not text:
        return fallback
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            for i in range(int(a), int(b) + 1):
                out.add(i)
        else:
            out.add(int(part))
    return sorted(out)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1).astype(np.float64, copy=False)
    bv = b.reshape(-1).astype(np.float64, copy=False)
    an = float(np.linalg.norm(av))
    bn = float(np.linalg.norm(bv))
    if an == 0.0 or bn == 0.0:
        return float("nan")
    return float(np.dot(av, bv) / (an * bn))


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


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
    gate_inp = load_f32(tensors, f"blk.{layer}.ffn_gate_inp.weight")
    gate_inp_shexp = load_f32(tensors, f"blk.{layer}.ffn_gate_inp_shexp.weight")

    if gate_exps.shape == (2048, 512, 256):
        gate_exps = np.transpose(gate_exps, (2, 1, 0)).copy()
    if up_exps.shape == (2048, 512, 256):
        up_exps = np.transpose(up_exps, (2, 1, 0)).copy()
    if down_exps.shape == (512, 2048, 256):
        down_exps = np.transpose(down_exps, (2, 1, 0)).copy()
    if gate_inp.shape == (2048, 256):
        gate_inp = gate_inp.T.copy()
    if gate_inp_shexp.shape == (1, 2048):
        gate_inp_shexp = gate_inp_shexp.reshape(2048).copy()

    return {
        "gate_exps": gate_exps,
        "up_exps": up_exps,
        "down_exps": down_exps,
        "gate_shexp": gate_shexp,
        "up_shexp": up_shexp,
        "down_shexp": down_shexp,
        "gate_inp": gate_inp,
        "gate_inp_shexp": gate_inp_shexp,
    }


def topk_from_logits(logits: np.ndarray, k: int):
    idx = np.argpartition(-logits, k - 1)[:k]
    idx = idx[np.argsort(-logits[idx])]
    vals = logits[idx]
    probs = np.exp(vals - np.max(vals))
    probs = probs / probs.sum()
    return idx.astype(np.int64), vals.astype(np.float32), probs.astype(np.float32)


def same_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def replay_layer(hidden: np.ndarray, weights: dict[str, np.ndarray], top_k: int):
    router_logits = hidden @ weights["gate_inp"].T
    idx, _, scores = topk_from_logits(router_logits, top_k)

    routed = np.zeros(hidden.shape[0], dtype=np.float32)
    for pos, eidx in enumerate(idx.tolist()):
        gate = hidden @ weights["gate_exps"][eidx].T
        up = hidden @ weights["up_exps"][eidx].T
        act = silu(gate) * up
        down = act @ weights["down_exps"][eidx].T
        routed += down * float(scores[pos])

    shared_gate = hidden @ weights["gate_shexp"].T
    shared_up = hidden @ weights["up_shexp"].T
    shared_act = silu(shared_gate) * shared_up
    shared_down = shared_act @ weights["down_shexp"].T
    shared_scale_pre = float(hidden @ weights["gate_inp_shexp"])
    shared_scale = 1.0 / (1.0 + np.exp(-shared_scale_pre))
    shared = shared_down * shared_scale

    return {
        "router_logits": router_logits.astype(np.float32, copy=False),
        "router_indices": idx,
        "router_scores": scores,
        "shared_gate_pre_sigmoid": shared_scale_pre,
        "routed_out": routed.astype(np.float32, copy=False),
        "shared_out": shared.astype(np.float32, copy=False),
        "mlp_out": (routed + shared).astype(np.float32, copy=False),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay Qwen3.6 sparse MoE from an exported layer trace")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--trace", required=True, help="Trace prefix or .npz path")
    ap.add_argument("--layers", help="Comma/range list; defaults to all layers in trace")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    trace_path = Path(args.trace)
    if trace_path.suffix != ".npz":
        trace_path = trace_path.with_suffix(".npz")
    meta_path = trace_path.with_suffix(".json")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fallback_layers = [int(x["layer"]) for x in meta["layers"]]
    layers = parse_layers(args.layers, fallback_layers)
    bundle = np.load(trace_path)
    reader = GGUFReader(args.gguf)
    tensors = tmap(reader)

    result = {
        "trace": str(trace_path),
        "gguf": args.gguf,
        "layers": [],
    }

    for layer in layers:
        top_k = int(bundle[f"blk_{layer}.router_indices"].shape[0])
        weights = load_layer_weights(tensors, layer)
        hidden = bundle[f"blk_{layer}.post_attn_ln"].astype(np.float32, copy=False)
        replay = replay_layer(hidden, weights, top_k)

        hf_router_idx = bundle[f"blk_{layer}.router_indices"].astype(np.int64).tolist()
        hf_router_scores = bundle[f"blk_{layer}.router_scores"].astype(np.float32, copy=False)
        hf_router_logits = bundle[f"blk_{layer}.router_logits"].astype(np.float32, copy=False)
        hf_shared_pre = float(bundle[f"blk_{layer}.shared_gate_pre_sigmoid"].reshape(-1)[0])
        hf_mlp = bundle[f"blk_{layer}.mlp_out"].astype(np.float32, copy=False)
        hf_resid = bundle[f"blk_{layer}.residual_after_mixer"].astype(np.float32, copy=False)
        hf_layer_out = bundle[f"blk_{layer}.layer_output"].astype(np.float32, copy=False)

        gg_layer_out = hf_resid + replay["mlp_out"]
        mlp_diff = replay["mlp_out"] - hf_mlp
        layer_diff = gg_layer_out - hf_layer_out

        rec = {
            "layer": layer,
            "same_top1": int(hf_router_idx[0] == int(replay["router_indices"][0])),
            "same_topk_set": sorted(hf_router_idx) == sorted(int(x) for x in replay["router_indices"].tolist()),
            "same_topk_prefix": same_prefix(hf_router_idx, [int(x) for x in replay["router_indices"].tolist()]),
            "router_logits_rmse": float(np.sqrt(np.mean((replay["router_logits"] - hf_router_logits) ** 2))),
            "router_logits_cosine": cosine(replay["router_logits"], hf_router_logits),
            "router_scores_rmse": float(np.sqrt(np.mean((replay["router_scores"] - hf_router_scores) ** 2))),
            "shared_gate_pre_abs_delta": float(abs(replay["shared_gate_pre_sigmoid"] - hf_shared_pre)),
            "mlp_rmse": float(np.sqrt(np.mean(mlp_diff * mlp_diff))),
            "mlp_mae": float(np.mean(np.abs(mlp_diff))),
            "mlp_cosine": cosine(replay["mlp_out"], hf_mlp),
            "layer_out_rmse": float(np.sqrt(np.mean(layer_diff * layer_diff))),
            "layer_out_mae": float(np.mean(np.abs(layer_diff))),
            "layer_out_cosine": cosine(gg_layer_out, hf_layer_out),
        }
        result["layers"].append(rec)
        print(
            f"blk.{layer}: top1={rec['same_top1']} topk_set={rec['same_topk_set']} prefix={rec['same_topk_prefix']} "
            f"mlp_rmse={rec['mlp_rmse']:.8f} mlp_cos={rec['mlp_cosine']:.8f} "
            f"layer_rmse={rec['layer_out_rmse']:.8f} layer_cos={rec['layer_out_cosine']:.8f}"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
