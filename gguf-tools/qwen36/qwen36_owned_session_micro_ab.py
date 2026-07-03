#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import create_causal_mask

from qwen36_behavior_oracle import (
    OwnedSession,
    _load_tail_weights,
    _load_tail_weights_from_gguf,
    _load_tokenizer_and_prompt,
    read_prompt,
)


def topk_from_logits(logits: np.ndarray, k: int) -> list[dict]:
    idx = np.argpartition(-logits, k - 1)[:k]
    idx = idx[np.argsort(-logits[idx])]
    return [{"id": int(i), "logit": float(logits[i])} for i in idx]


def lightweight_tail(owned_seq: np.ndarray, norm_w: np.ndarray, lm_head_w: np.ndarray, tokenizer, top_k: int) -> dict:
    last_row = owned_seq[-1].astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(last_row.astype(np.float64) ** 2) + 1e-6)
    normed = (last_row / rms) * norm_w
    logits = normed @ lm_head_w.T
    next_id = int(np.argmax(logits))
    return {
        "next_id": next_id,
        "next_text": tokenizer.decode([next_id], clean_up_tokenization_spaces=False),
        "topk": topk_from_logits(logits, top_k),
    }


def compare_arrays(label: str, a: np.ndarray, b: np.ndarray) -> dict:
    a64 = a.astype(np.float64, copy=False).ravel()
    b64 = b.astype(np.float64, copy=False).ravel()
    abs_diff = np.abs(a64 - b64)
    max_idx = int(np.argmax(abs_diff))
    denom = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    cosine = float(np.dot(a64, b64) / denom) if denom > 0.0 else 1.0
    out = {
        "label": label,
        "shape": list(a.shape),
        "max_abs": float(abs_diff[max_idx]),
        "max_idx": max_idx,
        "mean_abs": float(abs_diff.mean()),
        "cosine": cosine,
        "sum_a": float(a64.sum()),
        "sum_b": float(b64.sum()),
    }
    print(
        f"{label}: shape={a.shape} max_abs={out['max_abs']:.8f} "
        f"mean_abs={out['mean_abs']:.8f} cosine={out['cosine']:.10f}"
    )
    return out


def selected_tail_view(scored: dict) -> dict:
    picked = scored[scored["selected"]]
    return {
        "selected": scored["selected"],
        "next_id": picked["next_id"],
        "next_text": picked["next_text"],
        "topk": picked["topk"],
    }


def last_row_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr[-1:, :]
    raise RuntimeError(f"unexpected array rank for last_row_2d: {arr.ndim}")


def clone_args(args, worker_bin: str, full_mode: str | None = None):
    cloned = copy.copy(args)
    cloned.owned_session_worker_bin = worker_bin
    if full_mode is not None:
        cloned.owned_session_unified_full_cpu = full_mode == "cpu"
        cloned.owned_session_unified_full_gpu = full_mode == "gpu"
    return cloned


def open_session(args, worker_bin: str, hidden: int, full_mode: str | None = None, label: str | None = None):
    cloned = clone_args(args, worker_bin, full_mode=full_mode)
    cloned.owned_session_label = label
    return OwnedSession(cloned, hidden)


def run_session_step(session: OwnedSession, token_ids: list[int], step_idx: int):
    owned_seq, meta = session.run_step(token_ids, step_idx)
    return np.asarray(owned_seq, dtype=np.float32), meta


def dump_cycle_compares(session_a: OwnedSession, session_b: OwnedSession, score_tail_fn) -> list[dict]:
    out: list[dict] = []
    n_cycles = min(session_a.cycle_count, session_b.cycle_count)
    for cycle_idx in range(n_cycles):
        row: dict = {"cycle": cycle_idx}
        try:
            post_a = last_row_2d(session_a.dump_cycle_last(cycle_idx))
            post_b = last_row_2d(session_b.dump_cycle_last(cycle_idx))
            row["post"] = compare_arrays(f"cycle{cycle_idx}/post/a_vs_b", post_a, post_b)
            post_a_tail = score_tail_fn(post_a)
            post_b_tail = score_tail_fn(post_b)
            row["post_worker_a_tail"] = selected_tail_view(post_a_tail)
            row["post_worker_b_tail"] = selected_tail_view(post_b_tail)
        except Exception as exc:
            row["post_error"] = str(exc)
            out.append(row)
            break
        try:
            pre_a = last_row_2d(session_a.dump_cycle_pre_last(cycle_idx))
            pre_b = last_row_2d(session_b.dump_cycle_pre_last(cycle_idx))
            row["pre"] = compare_arrays(f"cycle{cycle_idx}/pre/a_vs_b", pre_a, pre_b)
            pre_a_tail = score_tail_fn(pre_a)
            pre_b_tail = score_tail_fn(pre_b)
            row["pre_worker_a_tail"] = selected_tail_view(pre_a_tail)
            row["pre_worker_b_tail"] = selected_tail_view(pre_b_tail)
        except Exception as exc:
            row["pre_error"] = str(exc)
        out.append(row)
    return out


def run_hf_full_hidden(model, token_ids: list[int]) -> np.ndarray:
    text_model = model.model
    device = text_model.norm.weight.device
    hidden_dtype = text_model.norm.weight.dtype
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        hidden_states = model.get_input_embeddings()(input_ids).to(dtype=hidden_dtype)
        batch, seq_len, _ = hidden_states.shape
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).view(1, 1, -1).expand(4, batch, -1)
        text_position_ids = position_ids[0]
        rope_position_ids = position_ids[1:]
        causal_mask = create_causal_mask(
            config=text_model.config,
            inputs_embeds=hidden_states,
            attention_mask=None,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        linear_attn_mask = text_model._update_linear_attn_mask(None, None)
        position_embeddings = text_model.rotary_emb(hidden_states, rope_position_ids)
        for layer_idx in range(len(text_model.layers)):
            decoder_layer = text_model.layers[layer_idx]
            layer_mask = linear_attn_mask if text_model.config.layer_types[layer_idx] == "linear_attention" else causal_mask
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
    return hidden_states[0].detach().float().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--json-out")
    ap.add_argument("--owned-session-worker-bin-a", required=True)
    ap.add_argument("--owned-session-worker-bin-b", required=True)
    ap.add_argument("--owned-session-env", action="append", default=[])
    ap.add_argument("--owned-session-unified-full-cpu", action="store_true")
    ap.add_argument("--owned-session-unified-full-gpu", action="store_true")
    ap.add_argument("--owned-session-unified-hybrid-gpu-cycles", type=int, default=0)
    ap.add_argument("--prefix-seq-worker-bin")
    ap.add_argument("--prefix-seq-fixture")
    ap.add_argument("--c-bin")
    ap.add_argument("--c-bin-worker-bin")
    ap.add_argument("--full-layer-worker-bin")
    ap.add_argument("--fixture", action="append", default=[])
    ap.add_argument("--full-layer", action="append", type=int, default=[])
    ap.add_argument("--splice-layer", type=int, default=39)
    ap.add_argument("--same-process-tail", action="store_true")
    ap.add_argument("--forced-step-token", type=int)
    ap.add_argument("--compare-hf-reference", action="store_true")
    ap.add_argument("--tail-source", choices=["hf", "gguf", "compare"], default="hf")
    ap.add_argument("--worker-a-full-mode", choices=["cpu", "gpu"])
    ap.add_argument("--worker-b-full-mode", choices=["cpu", "gpu"])
    ap.add_argument("--n-steps", type=int, default=1,
                    help="Forced-path decode steps after prefill; step 1 uses --forced-step-token if given")
    ap.add_argument("--dump-cycle-boundaries", action="store_true")
    ap.add_argument("--prefill-only", action="store_true")
    args = ap.parse_args()

    prompt = read_prompt(args)
    tokenizer, prompt_token_ids = _load_tokenizer_and_prompt(args.hf, prompt)
    tail_weights = {"hf": _load_tail_weights(args.hf)}
    if args.tail_source in ("gguf", "compare"):
        tail_weights["gguf"] = _load_tail_weights_from_gguf(args.gguf)
    norm_w = tail_weights["hf"][0].numpy()
    lm_head_w = tail_weights["hf"][1].numpy()
    hidden = int(norm_w.shape[0])
    hf_model = None
    if args.compare_hf_reference:
        print(f"[micro-ab] loading HF reference model from {args.hf}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.hf,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        hf_model.eval()
        torch.set_num_threads(min(32, torch.get_num_threads()))

    print(f"[micro-ab] prompt_tokens={len(prompt_token_ids)} hidden={hidden}")

    def score_tail(owned_seq: np.ndarray) -> dict:
        if args.tail_source == "hf":
            tw = tail_weights["hf"]
            return {"selected": "hf", "hf": lightweight_tail(owned_seq, tw[0].numpy(), tw[1].numpy(), tokenizer, args.top_k)}
        if args.tail_source == "gguf":
            tw = tail_weights["gguf"]
            return {"selected": "gguf", "gguf": lightweight_tail(owned_seq, tw[0].numpy(), tw[1].numpy(), tokenizer, args.top_k)}
        hf_tw = tail_weights["hf"]
        gguf_tw = tail_weights["gguf"]
        return {
            "selected": "gguf",
            "hf": lightweight_tail(owned_seq, hf_tw[0].numpy(), hf_tw[1].numpy(), tokenizer, args.top_k),
            "gguf": lightweight_tail(owned_seq, gguf_tw[0].numpy(), gguf_tw[1].numpy(), tokenizer, args.top_k),
        }

    session_a = open_session(args, args.owned_session_worker_bin_a, hidden, full_mode=args.worker_a_full_mode, label="owned-session-a")
    session_b = open_session(args, args.owned_session_worker_bin_b, hidden, full_mode=args.worker_b_full_mode, label="owned-session-b")
    try:
        a_prefill, a_prefill_meta = run_session_step(session_a, prompt_token_ids, 0)
        b_prefill, b_prefill_meta = run_session_step(session_b, prompt_token_ids, 0)
        prefill_cmp = compare_arrays("prefill/a_vs_b", a_prefill, b_prefill)
        a_prefill_tail = score_tail(a_prefill)
        b_prefill_tail = score_tail(b_prefill)
        hf_prefill_cmp = None
        hf_prefill_tail = None
        if hf_model is not None:
            print("[micro-ab] starting HF prefill reference")
            hf_prefill = run_hf_full_hidden(hf_model, prompt_token_ids)
            print("[micro-ab] finished HF prefill reference")
            hf_prefill_last = last_row_2d(hf_prefill)
            hf_prefill_cmp = compare_arrays("prefill/hf_vs_a", hf_prefill_last, a_prefill)
            hf_prefill_tail = score_tail(hf_prefill_last)
            shown = hf_prefill_tail[hf_prefill_tail["selected"]]
            print(f"prefill/hf_tail next={shown['next_id']} {json.dumps(shown['next_text'])}")
            print(f"prefill/hf_tail topk={shown['topk']}")

        out = {
            "prompt": prompt,
            "prompt_tokens": len(prompt_token_ids),
            "worker_a": args.owned_session_worker_bin_a,
            "worker_b": args.owned_session_worker_bin_b,
            "worker_a_full_mode": args.worker_a_full_mode,
            "worker_b_full_mode": args.worker_b_full_mode,
            "prefill": {
                "compare": prefill_cmp,
                "worker_a_meta": a_prefill_meta,
                "worker_b_meta": b_prefill_meta,
                "worker_a_tail": {
                    "selected": a_prefill_tail["selected"],
                    "scores": {k: v for k, v in a_prefill_tail.items() if k != "selected"},
                },
                "worker_b_tail": {
                    "selected": b_prefill_tail["selected"],
                    "scores": {k: v for k, v in b_prefill_tail.items() if k != "selected"},
                },
                "hf_compare_vs_a": hf_prefill_cmp,
                "hf_tail": None if hf_prefill_tail is None else {"selected": hf_prefill_tail["selected"], "scores": {k: v for k, v in hf_prefill_tail.items() if k != "selected"}},
            },
            "steps": [],
        }
        out["prefill"]["worker_a_debug"] = [line for line in session_a.stderr_lines() if "DBG_PREFILL_CYCLE" in line]
        out["prefill"]["worker_b_debug"] = [line for line in session_b.stderr_lines() if "DBG_PREFILL_CYCLE" in line]
        if args.dump_cycle_boundaries:
            out["prefill"]["cycle_boundaries"] = dump_cycle_compares(session_a, session_b, score_tail)
        if args.prefill_only:
            if args.json_out:
                Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
                print(f"json_out: {args.json_out}")
            else:
                print(json.dumps(out, indent=2))
            return 0

        token_ids = list(prompt_token_ids)
        for step_idx in range(1, args.n_steps + 1):
            if step_idx == 1 and args.forced_step_token is not None:
                forced_step_token = args.forced_step_token
            else:
                forced_step_token = a_prefill_tail[a_prefill_tail["selected"]]["next_id"] if step_idx == 1 else out["steps"][-1]["worker_a_tail"]["next_id"]
            print(
                f"[micro-ab] forced_step_token step={step_idx} token={forced_step_token} "
                f"text={json.dumps(tokenizer.decode([forced_step_token], clean_up_tokenization_spaces=False))}"
            )
            token_ids.append(forced_step_token)
            a_step, a_step_meta = run_session_step(session_a, token_ids, step_idx)
            b_step, b_step_meta = run_session_step(session_b, token_ids, step_idx)
            step_cmp = compare_arrays(f"step{step_idx}/a_vs_b", a_step, b_step)
            a_step_tail = score_tail(a_step)
            b_step_tail = score_tail(b_step)
            hf_step_cmp = None
            hf_step_tail = None
            if hf_model is not None:
                print(f"[micro-ab] starting HF step{step_idx} reference")
                hf_step = run_hf_full_hidden(hf_model, token_ids)
                print(f"[micro-ab] finished HF step{step_idx} reference")
                hf_step_last = last_row_2d(hf_step)
                hf_step_cmp = compare_arrays(f"step{step_idx}/hf_vs_a", hf_step_last, a_step)
                hf_step_tail = score_tail(hf_step_last)
                shown = hf_step_tail[hf_step_tail["selected"]]
                print(f"step{step_idx}/hf_tail next={shown['next_id']} {json.dumps(shown['next_text'])}")
                print(f"step{step_idx}/hf_tail topk={shown['topk']}")
            step_out = {
                "step": step_idx,
                "forced_token": {
                    "id": forced_step_token,
                    "text": tokenizer.decode([forced_step_token], clean_up_tokenization_spaces=False),
                },
                "compare": step_cmp,
                "worker_a_meta": a_step_meta,
                "worker_b_meta": b_step_meta,
                "worker_a_tail": selected_tail_view(a_step_tail),
                "worker_b_tail": selected_tail_view(b_step_tail),
                "worker_a_tail_scores": {k: v for k, v in a_step_tail.items() if k != "selected"},
                "worker_b_tail_scores": {k: v for k, v in b_step_tail.items() if k != "selected"},
                "hf_compare_vs_a": hf_step_cmp,
                "hf_tail": None if hf_step_tail is None else {"selected": hf_step_tail["selected"], "scores": {k: v for k, v in hf_step_tail.items() if k != "selected"}},
            }
            if args.dump_cycle_boundaries:
                step_out["cycle_boundaries"] = dump_cycle_compares(session_a, session_b)
            out["steps"].append(step_out)
    finally:
        session_a.close()
        session_b.close()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")
    else:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
