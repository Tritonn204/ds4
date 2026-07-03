#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


def read_prompt(args) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def arr(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a detailed full-attention trace for one Qwen3.6 layer")
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
    if not hasattr(layer, "self_attn"):
        raise SystemExit(f"layer {args.layer} does not expose self_attn")
    attn = layer.self_attn

    captured: dict[str, np.ndarray] = {}
    handles = []
    orig_forward = attn.forward

    def cap(name: str, value):
        captured[name] = value

    def layer_pre(_mod, inp):
        cap("layer_input", arr(inp[0][0, -1]))
        cap("layer_input_seq", arr(inp[0][0]))

    def input_ln_post(_mod, _inp, out):
        cap("input_ln", arr(out[0, -1]))
        cap("input_ln_seq", arr(out[0]))

    def post_attn_ln_post(_mod, _inp, out):
        cap("post_attn_ln", arr(out[0, -1]))
        cap("post_attn_ln_seq", arr(out[0]))

    def mlp_post(_mod, _inp, out):
        value = out[0] if isinstance(out, tuple) else out
        cap("mlp_out", arr(value[0, -1]))
        cap("mlp_out_seq", arr(value[0]))

    def layer_post(_mod, _inp, out):
        cap("layer_output", arr(out[0, -1]))
        cap("layer_output_seq", arr(out[0]))

    def router_post(_mod, _inp, out):
        logits, scores, indices = out
        cap("router_logits", arr(logits[-1]))
        cap("router_scores", arr(scores[-1]))
        cap("router_indices", indices[-1].detach().cpu().numpy())
        cap("router_logits_seq", arr(logits))
        cap("router_scores_seq", arr(scores))
        cap("router_indices_seq", indices.detach().cpu().numpy())

    def shared_gate_post(_mod, _inp, out):
        cap("shared_gate_pre_sigmoid", arr(out[-1]))
        cap("shared_gate_pre_sigmoid_seq", arr(out))

    def traced_forward(
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        q_proj_full = attn.q_proj(hidden_states)
        cap("q_proj_full", arr(q_proj_full[0]))
        q_proj_view = q_proj_full.view(*input_shape, -1, attn.head_dim * 2)
        query_states_raw, gate = torch.chunk(q_proj_view, 2, dim=-1)
        gate_flat = gate.reshape(*input_shape, -1)
        cap("query_raw", arr(query_states_raw[0]))
        cap("gate_raw", arr(gate[0]))
        cap("gate_flat", arr(gate_flat[0]))

        query_states_norm = attn.q_norm(query_states_raw.view(hidden_shape))
        key_proj_full = attn.k_proj(hidden_states)
        value_proj_full = attn.v_proj(hidden_states)
        key_states_norm = attn.k_norm(key_proj_full.view(hidden_shape))
        value_states_view = value_proj_full.view(hidden_shape)

        cap("q_norm", arr(query_states_norm[0]))
        cap("k_proj_raw", arr(key_proj_full[0]))
        cap("k_norm", arr(key_states_norm[0]))
        cap("v_proj_raw", arr(value_proj_full[0]))
        cap("v_view", arr(value_states_view[0]))

        query_states = query_states_norm.transpose(1, 2)
        key_states = key_states_norm.transpose(1, 2)
        value_states = value_states_view.transpose(1, 2)
        cap("q_transposed", arr(query_states[0]))
        cap("k_transposed", arr(key_states[0]))
        cap("v_transposed", arr(value_states[0]))

        cos, sin = position_embeddings
        cap("rope_cos", arr(cos[0]))
        cap("rope_sin", arr(sin[0]))
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        cap("q_rope", arr(query_states[0]))
        cap("k_rope", arr(key_states[0]))

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, attn.layer_idx)
            cap("k_cache_updated", arr(key_states[0]))
            cap("v_cache_updated", arr(value_states[0]))

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not attn.training else attn.attention_dropout,
            scaling=attn.scaling,
            **kwargs,
        )
        cap("attn_output_heads", arr(attn_output[0]))
        if attn_weights is not None:
            cap("attn_weights", arr(attn_weights[0]))

        attn_output_flat = attn_output.reshape(*input_shape, -1).contiguous()
        cap("attn_output_flat", arr(attn_output_flat[0]))
        attn_output_gated = attn_output_flat * torch.sigmoid(gate_flat)
        cap("attn_output_gated", arr(attn_output_gated[0]))

        o_proj_out = attn.o_proj(attn_output_gated)
        cap("o_proj_out", arr(o_proj_out[0]))
        return o_proj_out, attn_weights

    handles.append(layer.register_forward_pre_hook(layer_pre))
    handles.append(layer.input_layernorm.register_forward_hook(input_ln_post))
    handles.append(layer.post_attention_layernorm.register_forward_hook(post_attn_ln_post))
    handles.append(layer.mlp.register_forward_hook(mlp_post))
    handles.append(layer.register_forward_hook(layer_post))
    handles.append(layer.mlp.gate.register_forward_hook(router_post))
    handles.append(layer.mlp.shared_expert_gate.register_forward_hook(shared_gate_post))

    attn.forward = traced_forward
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
    finally:
        attn.forward = orig_forward
        for h in handles:
            h.remove()

    residual_after_mixer = captured["layer_input"] + captured["o_proj_out"][-1]
    residual_after_mixer_seq = captured["layer_input_seq"] + captured["o_proj_out"]
    captured["residual_after_mixer"] = residual_after_mixer.astype(np.float32, copy=False)
    captured["residual_after_mixer_seq"] = residual_after_mixer_seq.astype(np.float32, copy=False)
    captured["reconstructed_layer_output"] = (residual_after_mixer + captured["mlp_out"]).astype(np.float32, copy=False)
    captured["reconstructed_layer_output_seq"] = (residual_after_mixer_seq + captured["mlp_out_seq"]).astype(np.float32, copy=False)

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "prompt": prompt,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "layer": args.layer,
        "layer_type": layer.layer_type,
        "config": {
            "head_dim": int(attn.head_dim),
            "num_key_value_groups": int(attn.num_key_value_groups),
            "scaling": float(attn.scaling),
            "attention_dropout": float(attn.attention_dropout),
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
