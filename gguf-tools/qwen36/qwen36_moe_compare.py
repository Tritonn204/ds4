#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_layers(text: str | None) -> list[int]:
    if not text:
        return [0, 3, 34]
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


def read_prompt(args) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


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


def tensor_map(reader: GGUFReader):
    return {t.name: t for t in reader.tensors}


def load_tensor_f32(tmap, name: str) -> np.ndarray:
    t = tmap[name]
    return dequantize(t.data, t.tensor_type).astype(np.float32, copy=False)


def load_layer_weights(tmap, layer: int):
    gate_exps = load_tensor_f32(tmap, f"blk.{layer}.ffn_gate_exps.weight")
    up_exps = load_tensor_f32(tmap, f"blk.{layer}.ffn_up_exps.weight")
    down_exps = load_tensor_f32(tmap, f"blk.{layer}.ffn_down_exps.weight")
    gate_shexp = load_tensor_f32(tmap, f"blk.{layer}.ffn_gate_shexp.weight")
    up_shexp = load_tensor_f32(tmap, f"blk.{layer}.ffn_up_shexp.weight")
    down_shexp = load_tensor_f32(tmap, f"blk.{layer}.ffn_down_shexp.weight")
    gate_inp_shexp = load_tensor_f32(tmap, f"blk.{layer}.ffn_gate_inp_shexp.weight")

    if gate_exps.shape == (2048, 512, 256):
        gate_exps = np.transpose(gate_exps, (2, 1, 0)).copy()
    elif gate_exps.shape != (256, 512, 2048):
        raise ValueError(f"unexpected gate_exps shape {gate_exps.shape}")
    else:
        gate_exps = gate_exps.copy()

    if up_exps.shape == (2048, 512, 256):
        up_exps = np.transpose(up_exps, (2, 1, 0)).copy()
    elif up_exps.shape != (256, 512, 2048):
        raise ValueError(f"unexpected up_exps shape {up_exps.shape}")
    else:
        up_exps = up_exps.copy()

    if down_exps.shape == (512, 2048, 256):
        down_exps = np.transpose(down_exps, (2, 1, 0)).copy()
    elif down_exps.shape != (256, 2048, 512):
        raise ValueError(f"unexpected down_exps shape {down_exps.shape}")
    else:
        down_exps = down_exps.copy()

    return {
        "gate_exps": gate_exps,
        "up_exps": up_exps,
        "down_exps": down_exps,
        "gate_shexp": gate_shexp.copy(),
        "up_shexp": up_shexp.copy(),
        "down_shexp": down_shexp.copy(),
        "gate_inp_shexp": gate_inp_shexp.reshape(-1).copy(),
    }


def moe_forward(hidden: np.ndarray, expert_idx: np.ndarray, expert_w: np.ndarray, weights: dict[str, np.ndarray]):
    gate_exps = weights["gate_exps"]
    up_exps = weights["up_exps"]
    down_exps = weights["down_exps"]

    routed = np.zeros(hidden.shape[0], dtype=np.float32)
    per_expert = []
    for pos, eidx in enumerate(expert_idx.tolist()):
        gate = hidden @ gate_exps[eidx].T
        up = hidden @ up_exps[eidx].T
        act = silu(gate) * up
        down = act @ down_exps[eidx].T
        contrib = down * float(expert_w[pos])
        routed += contrib
        per_expert.append(
            {
                "expert": int(eidx),
                "weight": float(expert_w[pos]),
                "gate_norm": float(np.linalg.norm(gate)),
                "up_norm": float(np.linalg.norm(up)),
                "down_norm": float(np.linalg.norm(down)),
                "contrib_norm": float(np.linalg.norm(contrib)),
            }
        )

    shared_gate = hidden @ weights["gate_shexp"].T
    shared_up = hidden @ weights["up_shexp"].T
    shared_act = silu(shared_gate) * shared_up
    shared_down = shared_act @ weights["down_shexp"].T
    shared_scale = 1.0 / (1.0 + np.exp(-(hidden @ weights["gate_inp_shexp"])))
    shared_out = shared_down * shared_scale

    total = routed + shared_out
    return {
        "total": total.astype(np.float32, copy=False),
        "routed": routed.astype(np.float32, copy=False),
        "shared": shared_out.astype(np.float32, copy=False),
        "shared_scale": float(shared_scale),
        "per_expert": per_expert,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare HF Qwen3.6 MoE outputs against exported GGUF tensors")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--layers", help="Comma/range list, default: 0,3,34")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    prompt = read_prompt(args)
    layers = parse_layers(args.layers)

    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    reader = GGUFReader(args.gguf)
    tmap = tensor_map(reader)
    captured: dict[int, dict] = {}
    handles = []

    def make_mlp_pre(layer_idx: int):
        def hook(_mod, inp):
            captured.setdefault(layer_idx, {})
            captured[layer_idx]["hidden"] = inp[0][0, -1].detach().float().cpu().numpy()
        return hook

    def make_mlp_post(layer_idx: int):
        def hook(_mod, _inp, out):
            captured.setdefault(layer_idx, {})
            captured[layer_idx]["mlp_out"] = out[0, -1].detach().float().cpu().numpy()
        return hook

    def make_router_hook(layer_idx: int):
        def hook(_mod, _inp, out):
            _, scores, indices = out
            captured.setdefault(layer_idx, {})
            captured[layer_idx]["scores"] = scores[-1].detach().float().cpu().numpy()
            captured[layer_idx]["indices"] = indices[-1].detach().cpu().numpy()
        return hook

    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        handles.append(layer.mlp.register_forward_pre_hook(make_mlp_pre(layer_idx)))
        handles.append(layer.mlp.register_forward_hook(make_mlp_post(layer_idx)))
        handles.append(layer.mlp.gate.register_forward_hook(make_router_hook(layer_idx)))

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        _ = model(**inputs)

    for h in handles:
        h.remove()

    result = {
        "prompt": prompt,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "layers": [],
    }
    print(f"prompt_tokens: {result['prompt_tokens']}")

    for layer_idx in layers:
        weights = load_layer_weights(tmap, layer_idx)
        item = captured[layer_idx]
        hidden = item["hidden"].astype(np.float32, copy=False)
        hf_out = item["mlp_out"].astype(np.float32, copy=False)
        expert_idx = item["indices"].astype(np.int64, copy=False)
        expert_w = item["scores"].astype(np.float32, copy=False)
        gg = moe_forward(hidden, expert_idx, expert_w, weights)

        diff = gg["total"] - hf_out
        rec = {
            "layer": layer_idx,
            "expert_indices": [int(x) for x in expert_idx.tolist()],
            "expert_weights": [float(x) for x in expert_w.tolist()],
            "output_rmse": float(np.sqrt(np.mean(diff * diff))),
            "output_mae": float(np.mean(np.abs(diff))),
            "output_max_abs": float(np.max(np.abs(diff))),
            "output_cosine": cosine(gg["total"], hf_out),
            "hf_norm": float(np.linalg.norm(hf_out)),
            "gguf_norm": float(np.linalg.norm(gg["total"])),
            "routed_norm": float(np.linalg.norm(gg["routed"])),
            "shared_norm": float(np.linalg.norm(gg["shared"])),
            "shared_scale": gg["shared_scale"],
            "per_expert": gg["per_expert"],
        }
        result["layers"].append(rec)
        print(
            f"blk.{layer_idx}: rmse={rec['output_rmse']:.8f} mae={rec['output_mae']:.8f} "
            f"max_abs={rec['output_max_abs']:.8f} cosine={rec['output_cosine']:.8f} "
            f"norms hf={rec['hf_norm']:.6f} gguf={rec['gguf_norm']:.6f} "
            f"routed={rec['routed_norm']:.6f} shared={rec['shared_norm']:.6f}"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
