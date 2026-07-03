#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_layers(text: str | None) -> list[int]:
    if not text:
        return [0]
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


def arr(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Export per-layer Qwen3.6 activation traces for narrow runtime validation")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--layers", help="Comma/range list, default: 0")
    ap.add_argument("--out", required=True, help="Output prefix; writes .json and .npz")
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

    captured: dict[int, dict] = {}
    handles = []

    def cap(layer_idx: int, key: str, value):
        captured.setdefault(layer_idx, {})
        captured[layer_idx][key] = value

    def layer_pre(layer_idx: int):
        def hook(_mod, inp):
            cap(layer_idx, "layer_input", arr(inp[0][0, -1]))
            cap(layer_idx, "layer_input_seq", arr(inp[0][0]))
        return hook

    def layer_post(layer_idx: int):
        def hook(_mod, _inp, out):
            cap(layer_idx, "layer_output", arr(out[0, -1]))
            cap(layer_idx, "layer_output_seq", arr(out[0]))
        return hook

    def input_ln_post(layer_idx: int):
        def hook(_mod, _inp, out):
            cap(layer_idx, "input_ln", arr(out[0, -1]))
        return hook

    def mixer_post(layer_idx: int):
        def hook(_mod, _inp, out):
            value = out[0] if isinstance(out, tuple) else out
            cap(layer_idx, "mixer_out", arr(value[0, -1]))
            cap(layer_idx, "mixer_out_seq", arr(value[0]))
        return hook

    def post_attn_ln_post(layer_idx: int):
        def hook(_mod, _inp, out):
            cap(layer_idx, "post_attn_ln", arr(out[0, -1]))
            cap(layer_idx, "post_attn_ln_seq", arr(out[0]))
        return hook

    def mlp_post(layer_idx: int):
        def hook(_mod, _inp, out):
            value = out[0] if isinstance(out, tuple) else out
            cap(layer_idx, "mlp_out", arr(value[0, -1]))
            cap(layer_idx, "mlp_out_seq", arr(value[0]))
        return hook

    def router_post(layer_idx: int):
        def hook(_mod, _inp, out):
            logits, scores, indices = out
            cap(layer_idx, "router_logits", arr(logits[-1]))
            cap(layer_idx, "router_scores", arr(scores[-1]))
            cap(layer_idx, "router_indices", indices[-1].detach().cpu().numpy())
            cap(layer_idx, "router_logits_seq", arr(logits))
            cap(layer_idx, "router_scores_seq", arr(scores))
            cap(layer_idx, "router_indices_seq", indices.detach().cpu().numpy())
        return hook

    def shared_gate_post(layer_idx: int):
        def hook(_mod, _inp, out):
            cap(layer_idx, "shared_gate_pre_sigmoid", arr(out[-1]))
            cap(layer_idx, "shared_gate_pre_sigmoid_seq", arr(out))
        return hook

    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        handles.append(layer.register_forward_pre_hook(layer_pre(layer_idx)))
        handles.append(layer.register_forward_hook(layer_post(layer_idx)))
        handles.append(layer.input_layernorm.register_forward_hook(input_ln_post(layer_idx)))
        if hasattr(layer, "linear_attn"):
            handles.append(layer.linear_attn.register_forward_hook(mixer_post(layer_idx)))
        else:
            handles.append(layer.self_attn.register_forward_hook(mixer_post(layer_idx)))
        handles.append(layer.post_attention_layernorm.register_forward_hook(post_attn_ln_post(layer_idx)))
        handles.append(layer.mlp.register_forward_hook(mlp_post(layer_idx)))
        handles.append(layer.mlp.gate.register_forward_hook(router_post(layer_idx)))
        handles.append(layer.mlp.shared_expert_gate.register_forward_hook(shared_gate_post(layer_idx)))

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    for h in handles:
        h.remove()

    out_prefix = Path(args.out)
    npz_map: dict[str, np.ndarray] = {}
    meta = {
        "prompt": prompt,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "layers": [],
        "final_logits_top8": [],
    }
    logits = outputs.logits[0, -1].detach().float().cpu()
    vals, idx = torch.topk(logits, 8)
    for i in range(idx.numel()):
        meta["final_logits_top8"].append({"token_id": int(idx[i]), "logit": float(vals[i])})

    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        item = captured[layer_idx]
        residual_after_mixer = item["layer_input"] + item["mixer_out"]
        residual_after_mixer_seq = item["layer_input_seq"] + item["mixer_out_seq"]
        mlp_residual_out = residual_after_mixer + item["mlp_out"]
        mlp_residual_out_seq = residual_after_mixer_seq + item["mlp_out_seq"]
        item["residual_after_mixer"] = residual_after_mixer.astype(np.float32, copy=False)
        item["residual_after_mixer_seq"] = residual_after_mixer_seq.astype(np.float32, copy=False)
        item["reconstructed_layer_output"] = mlp_residual_out.astype(np.float32, copy=False)
        item["reconstructed_layer_output_seq"] = mlp_residual_out_seq.astype(np.float32, copy=False)

        rec = {
            "layer": layer_idx,
            "layer_type": layer.layer_type,
            "hidden_size": int(item["layer_input"].shape[0]),
            "expert_topk": int(item["router_indices"].shape[0]),
        }
        meta["layers"].append(rec)
        for key, value in item.items():
            npz_map[f"blk_{layer_idx}.{key}"] = np.asarray(value)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_prefix.with_suffix(".npz"), **npz_map)
    out_prefix.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"prompt_tokens: {meta['prompt_tokens']}")
    print(f"layers: {','.join(str(x) for x in layers)}")
    print(f"wrote: {out_prefix.with_suffix('.npz')}")
    print(f"wrote: {out_prefix.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
