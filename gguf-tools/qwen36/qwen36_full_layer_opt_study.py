#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from qwen36_behavior_oracle import OwnedSession
from qwen36_hybrid_prefix_tail_greedy import read_prompt, run_hf_patched_compare
from qwen36_owned_session_micro_ab import lightweight_tail
from qwen36_power2_owned_session_validator import row_rmse_summary, topk_overlap, run_hf_baseline_capture
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np


def clone_args(args, extra_env: dict[str, str]):
    class Obj:
        pass
    out = Obj()
    for k, v in vars(args).items():
        setattr(out, k, v)
    defaults = {
        "prefix_seq_worker_bin": None,
        "prefix_seq_fixture": None,
        "owned_session_unified_full_cpu": False,
        "same_process_tail": False,
        "fixture": [],
        "full_layer": [],
        "c_bin": None,
        "c_bin_worker_bin": None,
        "full_layer_worker_bin": None,
    }
    for k, v in defaults.items():
        if not hasattr(out, k):
            setattr(out, k, v)
    base = list(getattr(args, "owned_session_env", []) or [])
    merged = {item.split("=", 1)[0]: item.split("=", 1)[1] for item in base}
    merged.update(extra_env)
    out.owned_session_env = [f"{k}={v}" for k, v in merged.items()]
    return out


def run_case(case_name: str, args, model, tokenizer, token_ids):
    hidden = int(model.config.hidden_size)
    session = OwnedSession(args, hidden)
    try:
        owned_seq, meta = session.run_step(token_ids, 0)
    finally:
        session.close()
    hf_base = run_hf_baseline_capture(
        model=model,
        token_ids=token_ids,
        splice_layer=args.splice_layer,
        capture_layers=[args.splice_layer],
        tokenizer=tokenizer,
        top_k=args.top_k,
        layer_progress=False,
    )
    patched = run_hf_patched_compare(
        model=model,
        token_ids=token_ids,
        owned_seq=owned_seq,
        splice_layer=args.splice_layer,
        tokenizer=tokenizer,
        top_k=args.top_k,
        layer_progress=args.hf_patched_layer_progress,
        setup_progress=args.hf_patched_setup_progress,
    )
    norm_w = model.model.norm.weight.detach().float().cpu().numpy()
    lm_head_w = model.lm_head.weight.detach().float().cpu().numpy()
    owned_tail = lightweight_tail(np.asarray(owned_seq, dtype=np.float32), norm_w, lm_head_w, tokenizer, args.top_k)
    owned_arr = np.asarray(owned_seq, dtype=np.float32)
    hf_splice = np.asarray(hf_base["splice_seq"], dtype=np.float32)
    if owned_arr.ndim == 2 and hf_splice.ndim == 2 and owned_arr.shape[0] == 1 and hf_splice.shape[0] > 1:
        hf_splice = hf_splice[-1:, :]
    splice = row_rmse_summary(owned_arr, hf_splice)
    return {
        "case": case_name,
        "owned_prefix_ms": meta["owned_prefix_ms"],
        "worker_meta": meta,
        "hf_baseline_ms": hf_base["hf_baseline_ms"],
        "hf_patched_ms": patched["hf_patched_ms"],
        "argmax_equal": bool(owned_tail["next_id"] == patched["next_id"]),
        "topk_overlap": topk_overlap(owned_tail["topk"], patched["patched_topk"]),
        "splice_seq_rmse": splice["seq_rmse"],
        "worst_row": splice["worst_row"],
        "owned_next_id": owned_tail["next_id"],
        "hf_next_id": patched["next_id"],
        "owned_next_text": owned_tail["next_text"],
        "hf_next_text": patched["next_text"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Study full-layer optimizations independently")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prompt")
    ap.add_argument("--prompt-file")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--owned-session-worker-bin", required=True)
    ap.add_argument("--owned-session-env", action="append", default=[])
    ap.add_argument("--owned-session-unified-full-gpu", action="store_true")
    ap.add_argument("--owned-session-unified-hybrid-gpu-cycles", type=int, default=0)
    ap.add_argument("--c-bin")
    ap.add_argument("--c-bin-worker-bin")
    ap.add_argument("--full-layer-worker-bin")
    ap.add_argument("--fixture", action="append", default=[])
    ap.add_argument("--full-layer", action="append", type=int, default=[])
    ap.add_argument("--splice-layer", type=int, default=39)
    ap.add_argument("--hf-patched-layer-progress", action="store_true", default=True)
    ap.add_argument("--no-hf-patched-layer-progress", dest="hf_patched_layer_progress", action="store_false")
    ap.add_argument("--hf-patched-setup-progress", action="store_true")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    prompt = read_prompt(args)
    print(f"[opt-study] loading tokenizer from {args.hf}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    print(f"[opt-study] loading model from {args.hf}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    torch.set_num_threads(min(32, os.cpu_count() or 16))
    token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()

    fixture_count = len(args.fixture or [])
    full_layers = list(args.full_layer or [])
    if fixture_count % 3 != 0:
        raise RuntimeError(f"expected fixture count to be divisible by 3, got {fixture_count}")
    available_cycles = fixture_count // 3
    if available_cycles == 0:
        raise RuntimeError("need at least one owned cycle worth of fixtures")
    if len(full_layers) < available_cycles:
        raise RuntimeError(
            f"need at least {available_cycles} full-layer ids for {fixture_count} fixtures, got {len(full_layers)}"
        )
    if available_cycles < len(full_layers):
        kept_full_layers = full_layers[:available_cycles]
        print(
            f"[opt-study] trimming study scope to provided fixtures: "
            f"cycles={available_cycles} full_layers={kept_full_layers}"
        )
        args.full_layer = kept_full_layers
    effective_splice = min(args.splice_layer, args.full_layer[-1])
    if effective_splice != args.splice_layer:
        print(
            f"[opt-study] lowering splice_layer from {args.splice_layer} to {effective_splice} "
            f"to match provided fixtures"
        )
        args.splice_layer = effective_splice

    cases = [
        ("baseline", {}),
        ("batched_only", {"QWEN36_FULL_GPU_FFN_BATCHED_EXPERTS": "1"}),
        ("scratch_only", {"QWEN36_FULL_GPU_PERSISTENT_SCRATCH": "1"}),
        ("both", {
            "QWEN36_FULL_GPU_FFN_BATCHED_EXPERTS": "1",
            "QWEN36_FULL_GPU_PERSISTENT_SCRATCH": "1",
        }),
    ]
    results = []
    for case_name, extra_env in cases:
        print(f"[opt-study] case={case_name} env={extra_env}")
        case_args = clone_args(args, extra_env)
        results.append(run_case(case_name, case_args, model, tokenizer, token_ids))

    out = {
        "prompt_tokens": len(token_ids),
        "splice_layer": args.splice_layer,
        "results": results,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")
    else:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
