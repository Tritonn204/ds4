#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_prompt(args) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def arr(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a detailed linear-attention trace for one Qwen3.6 layer")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--out", required=True, help="Output prefix; writes .json and .npz")
    args = ap.parse_args()

    prompt = read_prompt(args)
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    layer = model.model.layers[args.layer]
    if not hasattr(layer, "linear_attn"):
        raise SystemExit(f"layer {args.layer} is not linear_attention")
    la = layer.linear_attn

    captured: dict[str, np.ndarray] = {}
    handles = []

    def cap(name: str, value):
        captured[name] = value

    handles.append(layer.register_forward_pre_hook(lambda _m, inp: cap("layer_input", arr(inp[0][0, -1]))))
    handles.append(layer.register_forward_pre_hook(lambda _m, inp: cap("layer_input_seq", arr(inp[0][0]))))
    handles.append(layer.input_layernorm.register_forward_hook(lambda _m, _i, out: cap("input_ln", arr(out[0, -1]))))
    handles.append(layer.input_layernorm.register_forward_hook(lambda _m, _i, out: cap("input_ln_seq", arr(out[0]))))
    handles.append(la.in_proj_qkv.register_forward_hook(lambda _m, _i, out: cap("in_proj_qkv", arr(out[0, -1]))))
    handles.append(la.in_proj_qkv.register_forward_hook(lambda _m, _i, out: cap("in_proj_qkv_seq", arr(out[0]))))
    handles.append(la.in_proj_z.register_forward_hook(lambda _m, _i, out: cap("in_proj_z", arr(out[0, -1]))))
    handles.append(la.in_proj_b.register_forward_hook(lambda _m, _i, out: cap("in_proj_b", arr(out[0, -1]))))
    handles.append(la.in_proj_a.register_forward_hook(lambda _m, _i, out: cap("in_proj_a", arr(out[0, -1]))))
    handles.append(la.conv1d.register_forward_hook(lambda _m, _i, out: cap("conv1d_raw", arr(out[0]))))
    handles.append(la.out_proj.register_forward_pre_hook(lambda _m, inp: cap("out_proj_in", arr(inp[0][0, -1]))))
    handles.append(la.out_proj.register_forward_hook(lambda _m, _i, out: cap("out_proj_out", arr(out[0, -1]))))
    handles.append(la.register_forward_hook(lambda _m, _i, out: cap("mixer_out", arr(out[0, -1]))))
    handles.append(layer.register_forward_hook(lambda _m, _i, out: cap("layer_output", arr(out[0, -1]))))

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    for h in handles:
        h.remove()

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "prompt": prompt,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "layer": args.layer,
        "layer_type": layer.layer_type,
        "config": {
            "hidden_size": int(la.hidden_size),
            "num_v_heads": int(la.num_v_heads),
            "num_k_heads": int(la.num_k_heads),
            "head_k_dim": int(la.head_k_dim),
            "head_v_dim": int(la.head_v_dim),
            "key_dim": int(la.key_dim),
            "value_dim": int(la.value_dim),
            "conv_kernel_size": int(la.conv_kernel_size),
        },
        "captured": sorted(captured.keys()),
        "final_logits_top8": [],
    }

    logits = outputs.logits[0, -1].detach().float().cpu()
    vals, idx = torch.topk(logits, 8)
    for i in range(idx.numel()):
        meta["final_logits_top8"].append({"token_id": int(idx[i]), "logit": float(vals[i])})

    np.savez(out_prefix.with_suffix(".npz"), **{k: np.asarray(v) for k, v in captured.items()})
    out_prefix.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"prompt_tokens: {meta['prompt_tokens']}")
    print(f"layer: {args.layer}")
    print(f"captured: {','.join(meta['captured'])}")
    print(f"wrote: {out_prefix.with_suffix('.npz')}")
    print(f"wrote: {out_prefix.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
