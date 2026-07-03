#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
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


def topk_from_logits(logits: np.ndarray, k: int):
    idx = np.argpartition(-logits, k - 1)[:k]
    idx = idx[np.argsort(-logits[idx])]
    vals = logits[idx]
    probs = np.exp(vals - np.max(vals))
    probs = probs / probs.sum()
    return idx.astype(int), vals.astype(np.float32), probs.astype(np.float32)


def same_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def get_tensor_map(reader: GGUFReader):
    return {t.name: t for t in reader.tensors}


def load_router_weights(tmap, layer: int):
    gate = tmap[f"blk.{layer}.ffn_gate_inp.weight"].data
    shared = tmap[f"blk.{layer}.ffn_gate_inp_shexp.weight"].data
    gate = np.asarray(gate, dtype=np.float32)
    shared = np.asarray(shared, dtype=np.float32)
    # GGUF reader reports logical shapes; recover storage layout used by torch linear weights.
    if gate.shape == (2048, 256):
        gate = gate.T.copy()
    elif gate.shape != (256, 2048):
        raise ValueError(f"unexpected gate shape for layer {layer}: {gate.shape}")
    if shared.shape == (2048,):
        shared = shared.copy()
    elif shared.shape == (1, 2048):
        shared = shared.reshape(2048).copy()
    else:
        raise ValueError(f"unexpected shared gate shape for layer {layer}: {shared.shape}")
    return gate, shared


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare HF Qwen3.6 router outputs against exported GGUF router tensors")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--layers", help="Comma/range list, default: 0,3,34")
    ap.add_argument("--top-k", type=int, default=8)
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
    tmap = get_tensor_map(reader)

    captured: dict[int, dict] = {}
    handles = []

    def make_router_hook(layer_idx: int):
        def hook(_mod, inp, out):
            hidden = inp[0][-1].detach().float().cpu().numpy()
            router_logits, router_scores, router_indices = out
            captured[layer_idx] = {
                "hidden": hidden,
                "hf_logits": router_logits[-1].detach().float().cpu().numpy(),
                "hf_scores": router_scores[-1].detach().float().cpu().numpy(),
                "hf_indices": router_indices[-1].detach().cpu().numpy(),
            }
        return hook

    def make_shared_hook(layer_idx: int):
        def hook(_mod, _inp, out):
            captured.setdefault(layer_idx, {})
            captured[layer_idx]["hf_shared_gate"] = torch.sigmoid(out[-1].detach().float().cpu()).item()
        return hook

    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        handles.append(layer.mlp.gate.register_forward_hook(make_router_hook(layer_idx)))
        handles.append(layer.mlp.shared_expert_gate.register_forward_hook(make_shared_hook(layer_idx)))

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
        item = captured[layer_idx]
        gate_w, shared_w = load_router_weights(tmap, layer_idx)
        hidden = item["hidden"].astype(np.float32, copy=False)
        gguf_logits = hidden @ gate_w.T
        gguf_shared_gate = 1.0 / (1.0 + np.exp(-(hidden @ shared_w)))
        gguf_idx, gguf_top_logits, gguf_scores = topk_from_logits(gguf_logits, args.top_k)

        hf_idx = item["hf_indices"].astype(int).tolist()
        hf_scores = item["hf_scores"].astype(np.float32)
        hf_logits = item["hf_logits"].astype(np.float32)
        hf_top_logits = hf_logits[item["hf_indices"]]

        rec = {
            "layer": layer_idx,
            "same_top1": int(hf_idx[0] == int(gguf_idx[0])),
            "same_topk_set": sorted(hf_idx) == sorted(int(x) for x in gguf_idx.tolist()),
            "same_topk_prefix": same_prefix(hf_idx, [int(x) for x in gguf_idx.tolist()]),
            "hf_indices": hf_idx,
            "gguf_indices": [int(x) for x in gguf_idx.tolist()],
            "hf_scores": [float(x) for x in hf_scores.tolist()],
            "gguf_scores": [float(x) for x in gguf_scores.tolist()],
            "hf_top_logits": [float(x) for x in hf_top_logits.tolist()],
            "gguf_top_logits": [float(x) for x in gguf_top_logits.tolist()],
            "router_logits_rmse": float(np.sqrt(np.mean((gguf_logits - hf_logits) ** 2))),
            "router_logits_cosine": cosine(gguf_logits, hf_logits),
            "shared_gate_hf": float(item["hf_shared_gate"]),
            "shared_gate_gguf": float(gguf_shared_gate),
            "shared_gate_abs_delta": float(abs(gguf_shared_gate - item["hf_shared_gate"])),
        }
        result["layers"].append(rec)
        print(
            f"blk.{layer_idx}: top1={rec['same_top1']} topk_set={rec['same_topk_set']} "
            f"prefix={rec['same_topk_prefix']} logits_rmse={rec['router_logits_rmse']:.8f} "
            f"logits_cos={rec['router_logits_cosine']:.8f} shared_delta={rec['shared_gate_abs_delta']:.8f}"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
