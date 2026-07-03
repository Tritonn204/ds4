#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import create_causal_mask


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


def topk(logits: np.ndarray, k: int):
    idx = np.argpartition(-logits, k - 1)[:k]
    idx = idx[np.argsort(-logits[idx])]
    return [{"id": int(i), "logit": float(logits[i])} for i in idx]


def token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run only the HF tail of Qwen3.6 from a C-produced sequence hidden state")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--seq-file", required=True, help="float32 file of shape [seq_len, hidden]")
    ap.add_argument("--start-layer", type=int, required=True, help="first HF layer to execute in the tail")
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--json-out")
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
    print("model_loaded: true", flush=True)

    inputs = tokenizer(prompt, return_tensors="pt")
    print(f"prompt_tokens: {int(inputs['input_ids'].shape[1])}", flush=True)
    with torch.no_grad():
        base = model(**inputs)
    base_logits = base.logits[0, -1].detach().float().cpu().numpy()
    print("baseline_done: true", flush=True)

    text_model = model.model
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    seq_len = int(input_ids.shape[1])
    hidden = int(text_model.config.hidden_size)

    seq_hidden = np.fromfile(args.seq_file, dtype=np.float32)
    if seq_hidden.size != seq_len * hidden:
        raise SystemExit(f"bad seq-file size: got {seq_hidden.size}, expected {seq_len * hidden}")
    hidden_states = torch.from_numpy(seq_hidden.reshape(1, seq_len, hidden)).to(next(model.parameters()).device)
    hidden_states = hidden_states.to(dtype=next(model.parameters()).dtype)

    position_ids = torch.arange(seq_len, device=hidden_states.device).view(1, 1, -1).expand(4, 1, -1)
    text_position_ids = position_ids[0]
    position_ids = position_ids[1:]

    inputs_embeds = text_model.embed_tokens(input_ids)
    causal_mask = create_causal_mask(
        config=text_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    linear_attn_mask = text_model._update_linear_attn_mask(attention_mask, None)
    position_embeddings = text_model.rotary_emb(hidden_states, position_ids)
    print(f"tail_start_layer: {args.start_layer}", flush=True)

    with torch.no_grad():
        for i in range(args.start_layer, text_model.config.num_hidden_layers):
            decoder_layer = text_model.layers[i]
            layer_mask = linear_attn_mask if text_model.config.layer_types[i] == "linear_attention" else causal_mask
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )
            if i == args.start_layer or i == text_model.config.num_hidden_layers - 1 or ((i - args.start_layer + 1) % 8 == 0):
                print(f"tail_layer_done: {i}", flush=True)
        hidden_states = text_model.norm(hidden_states)
        patched_logits_t = model.lm_head(hidden_states)
    print("tail_done: true", flush=True)
    patched_logits = patched_logits_t[0, -1].detach().float().cpu().numpy()

    base_argmax = int(np.argmax(base_logits))
    patched_argmax = int(np.argmax(patched_logits))
    result = {
        "prompt": prompt,
        "prompt_tokens": seq_len,
        "start_layer": args.start_layer,
        "seq_file": args.seq_file,
        "logits_cosine": cosine(base_logits, patched_logits),
        "logits_rmse": float(np.sqrt(np.mean((base_logits - patched_logits) ** 2))),
        "argmax_equal": base_argmax == patched_argmax,
        "base_argmax": {
            "id": base_argmax,
            "text": token_text(tokenizer, base_argmax),
            "logit": float(base_logits[base_argmax]),
        },
        "patched_argmax": {
            "id": patched_argmax,
            "text": token_text(tokenizer, patched_argmax),
            "logit": float(patched_logits[patched_argmax]),
        },
        "base_topk": topk(base_logits, args.top_k),
        "patched_topk": topk(patched_logits, args.top_k),
    }

    print(f"prompt_tokens: {result['prompt_tokens']}")
    print(f"start_layer: {result['start_layer']}")
    print(f"logits_cosine: {result['logits_cosine']:.8f}")
    print(f"logits_rmse: {result['logits_rmse']:.8f}")
    print(f"argmax_equal: {result['argmax_equal']}")
    print(
        f"base_argmax: id={result['base_argmax']['id']} "
        f"text={json.dumps(result['base_argmax']['text'])} "
        f"logit={result['base_argmax']['logit']:.6f}"
    )
    print(
        f"patched_argmax: id={result['patched_argmax']['id']} "
        f"text={json.dumps(result['patched_argmax']['text'])} "
        f"logit={result['patched_argmax']['logit']:.6f}"
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
