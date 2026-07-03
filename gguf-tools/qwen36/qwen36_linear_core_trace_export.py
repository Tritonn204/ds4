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
    ap = argparse.ArgumentParser(description="Export DeltaNet core input traces for one Qwen3.6 linear-attention layer")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--layer", required=True, type=int)
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--out", required=True)
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

    orig_chunk = la.chunk_gated_delta_rule

    def wrapped_chunk(query, key, value, g=None, beta=None, initial_state=None, output_final_state=None, use_qk_l2norm_in_kernel=None, cu_seqlens=None):
        cap("query", arr(query[0]))
        cap("key", arr(key[0]))
        cap("value", arr(value[0]))
        cap("g", arr(g[0]))
        cap("beta", arr(beta[0]))
        out, state = orig_chunk(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )
        cap("core_attn_out_pre_norm", arr(out[0]))
        return out, state

    la.chunk_gated_delta_rule = wrapped_chunk
    handles.append(layer.register_forward_pre_hook(lambda _m, inp: cap("layer_input", arr(inp[0][0, -1]))))
    handles.append(la.in_proj_qkv.register_forward_hook(lambda _m, _i, out: cap("in_proj_qkv_seq", arr(out[0]))))
    handles.append(la.in_proj_z.register_forward_hook(lambda _m, _i, out: cap("in_proj_z_seq", arr(out[0]))))
    handles.append(la.in_proj_a.register_forward_hook(lambda _m, _i, out: cap("in_proj_a_seq", arr(out[0]))))
    handles.append(la.in_proj_b.register_forward_hook(lambda _m, _i, out: cap("in_proj_b_seq", arr(out[0]))))
    handles.append(la.conv1d.register_forward_hook(lambda _m, _i, out: cap("conv1d_raw", arr(out[0]))))
    handles.append(la.out_proj.register_forward_pre_hook(lambda _m, inp: cap("out_proj_in_seq", arr(inp[0][0]))))
    handles.append(la.out_proj.register_forward_hook(lambda _m, _i, out: cap("out_proj_out_seq", arr(out[0]))))
    handles.append(la.register_forward_hook(lambda _m, _i, out: cap("mixer_out", arr(out[0, -1]))))
    handles.append(layer.post_attention_layernorm.register_forward_hook(lambda _m, _i, out: cap("post_attn_ln", arr(out[0, -1]))))
    handles.append(layer.register_forward_hook(lambda _m, _i, out: cap("layer_output", arr(out[0, -1]))))

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        _ = model(**inputs)

    la.chunk_gated_delta_rule = orig_chunk
    for h in handles:
        h.remove()

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
    }

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
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
