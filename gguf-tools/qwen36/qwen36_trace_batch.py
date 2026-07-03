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
        return [0, 3, 19, 34]
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


def same_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def topk_from_logits(logits: np.ndarray, k: int):
    idx = np.argpartition(-logits, k - 1)[:k]
    idx = idx[np.argsort(-logits[idx])]
    vals = logits[idx]
    probs = np.exp(vals - np.max(vals))
    probs = probs / probs.sum()
    return idx.astype(np.int64), vals.astype(np.float32), probs.astype(np.float32)


def tensor_map(reader: GGUFReader):
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


def replay_moe(hidden: np.ndarray, weights: dict[str, np.ndarray], top_k: int):
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
        "mlp_out": (routed + shared).astype(np.float32, copy=False),
    }


def prompt_specs(args) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    if args.prompt_file:
        for item in args.prompt_file:
            path = Path(item)
            specs.append((path.stem, path.read_text(encoding="utf-8")))
    elif args.prompt_dir:
        for path in sorted(Path(args.prompt_dir).glob("*.txt")):
            specs.append((path.stem, path.read_text(encoding="utf-8")))
    else:
        specs.append(("inline", args.prompt))
    return specs


def arr(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch layer-trace and MoE replay validator for Qwen3.6 narrow runtime work")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layers", help="Comma/range list, default: 0,3,19,34")
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file", action="append")
    ap.add_argument("--prompt-dir")
    ap.add_argument("--limit-prompts", type=int)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    layers = parse_layers(args.layers)
    prompts = prompt_specs(args)
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]

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
    tensors = tensor_map(reader)
    layer_weights = {layer: load_layer_weights(tensors, layer) for layer in layers}

    handles = []
    captured: dict[int, dict] = {}

    def cap(layer_idx: int, key: str, value):
        captured.setdefault(layer_idx, {})
        captured[layer_idx][key] = value

    def layer_pre(layer_idx: int):
        def hook(_mod, inp):
            cap(layer_idx, "layer_input", arr(inp[0][0, -1]))
        return hook

    def layer_post(layer_idx: int):
        def hook(_mod, _inp, out):
            cap(layer_idx, "layer_output", arr(out[0, -1]))
        return hook

    def mixer_post(layer_idx: int):
        def hook(_mod, _inp, out):
            value = out[0] if isinstance(out, tuple) else out
            cap(layer_idx, "mixer_out", arr(value[0, -1]))
        return hook

    def post_attn_ln_post(layer_idx: int):
        def hook(_mod, _inp, out):
            cap(layer_idx, "post_attn_ln", arr(out[0, -1]))
        return hook

    def mlp_post(layer_idx: int):
        def hook(_mod, _inp, out):
            value = out[0] if isinstance(out, tuple) else out
            cap(layer_idx, "mlp_out", arr(value[0, -1]))
        return hook

    def router_post(layer_idx: int):
        def hook(_mod, _inp, out):
            logits, scores, indices = out
            cap(layer_idx, "router_logits", arr(logits[-1]))
            cap(layer_idx, "router_scores", arr(scores[-1]))
            cap(layer_idx, "router_indices", indices[-1].detach().cpu().numpy())
        return hook

    def shared_gate_post(layer_idx: int):
        def hook(_mod, _inp, out):
            cap(layer_idx, "shared_gate_pre_sigmoid", arr(out[-1]))
        return hook

    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        handles.append(layer.register_forward_pre_hook(layer_pre(layer_idx)))
        handles.append(layer.register_forward_hook(layer_post(layer_idx)))
        if hasattr(layer, "linear_attn"):
            handles.append(layer.linear_attn.register_forward_hook(mixer_post(layer_idx)))
        else:
            handles.append(layer.self_attn.register_forward_hook(mixer_post(layer_idx)))
        handles.append(layer.post_attention_layernorm.register_forward_hook(post_attn_ln_post(layer_idx)))
        handles.append(layer.mlp.register_forward_hook(mlp_post(layer_idx)))
        handles.append(layer.mlp.gate.register_forward_hook(router_post(layer_idx)))
        handles.append(layer.mlp.shared_expert_gate.register_forward_hook(shared_gate_post(layer_idx)))

    result = {
        "layers": layers,
        "prompts": [],
    }

    for prompt_name, prompt_text in prompts:
        captured.clear()
        inputs = tokenizer(prompt_text, return_tensors="pt")
        with torch.no_grad():
            _ = model(**inputs)

        prompt_rec = {
            "name": prompt_name,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "layers": [],
        }
        print(f"prompt: {prompt_name} tokens={prompt_rec['prompt_tokens']}")

        for layer_idx in layers:
            item = captured[layer_idx]
            residual_after_mixer = item["layer_input"] + item["mixer_out"]
            replay = replay_moe(item["post_attn_ln"].astype(np.float32, copy=False), layer_weights[layer_idx], int(item["router_indices"].shape[0]))
            gg_layer_out = residual_after_mixer + replay["mlp_out"]

            hf_router_idx = item["router_indices"].astype(np.int64).tolist()
            hf_router_scores = item["router_scores"].astype(np.float32, copy=False)
            hf_router_logits = item["router_logits"].astype(np.float32, copy=False)
            hf_shared_pre = float(item["shared_gate_pre_sigmoid"].reshape(-1)[0])
            hf_mlp = item["mlp_out"].astype(np.float32, copy=False)
            hf_layer_out = item["layer_output"].astype(np.float32, copy=False)

            mlp_diff = replay["mlp_out"] - hf_mlp
            layer_diff = gg_layer_out - hf_layer_out
            rec = {
                "layer": layer_idx,
                "same_top1": int(hf_router_idx[0] == int(replay["router_indices"][0])),
                "same_topk_set": sorted(hf_router_idx) == sorted(int(x) for x in replay["router_indices"].tolist()),
                "same_topk_prefix": same_prefix(hf_router_idx, [int(x) for x in replay["router_indices"].tolist()]),
                "router_logits_rmse": float(np.sqrt(np.mean((replay["router_logits"] - hf_router_logits) ** 2))),
                "router_logits_cosine": cosine(replay["router_logits"], hf_router_logits),
                "router_scores_rmse": float(np.sqrt(np.mean((replay["router_scores"] - hf_router_scores) ** 2))),
                "shared_gate_pre_abs_delta": float(abs(replay["shared_gate_pre_sigmoid"] - hf_shared_pre)),
                "mlp_rmse": float(np.sqrt(np.mean(mlp_diff * mlp_diff))),
                "mlp_cosine": cosine(replay["mlp_out"], hf_mlp),
                "layer_out_rmse": float(np.sqrt(np.mean(layer_diff * layer_diff))),
                "layer_out_cosine": cosine(gg_layer_out, hf_layer_out),
            }
            prompt_rec["layers"].append(rec)
            print(
                f"  blk.{layer_idx}: top1={rec['same_top1']} topk_set={rec['same_topk_set']} "
                f"prefix={rec['same_topk_prefix']} mlp_rmse={rec['mlp_rmse']:.6f} "
                f"mlp_cos={rec['mlp_cosine']:.6f} layer_cos={rec['layer_out_cosine']:.6f}"
            )

        result["prompts"].append(prompt_rec)

    for h in handles:
        h.remove()

    all_layers = [layer for prompt in result["prompts"] for layer in prompt["layers"]]
    summary = {
        "cases": len(all_layers),
        "prompts": len(result["prompts"]),
        "all_top1": all(x["same_top1"] == 1 for x in all_layers),
        "all_topk_sets": all(x["same_topk_set"] for x in all_layers),
        "min_prefix": min(x["same_topk_prefix"] for x in all_layers),
        "avg_mlp_rmse": float(np.mean([x["mlp_rmse"] for x in all_layers])),
        "max_mlp_rmse": float(np.max([x["mlp_rmse"] for x in all_layers])),
        "avg_mlp_cosine": float(np.mean([x["mlp_cosine"] for x in all_layers])),
        "min_mlp_cosine": float(np.min([x["mlp_cosine"] for x in all_layers])),
        "avg_layer_out_cosine": float(np.mean([x["layer_out_cosine"] for x in all_layers])),
        "min_layer_out_cosine": float(np.min([x["layer_out_cosine"] for x in all_layers])),
    }
    result["summary"] = summary
    print("summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
