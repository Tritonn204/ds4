#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FIXTURE_MAGIC = b"Q36DWF02"
FIXTURE_MAGIC_LEN = 8


def load_hybrid_helpers():
    helper_path = Path(__file__).with_name("qwen36_hybrid_prefix_tail_greedy.py")
    spec = importlib.util.spec_from_file_location("qwen36_hybrid_prefix_tail_greedy", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load helper module from {helper_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


def read_prompt(args) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def read_fixture_meta(path: str) -> dict:
    with open(path, "rb") as fp:
        magic = fp.read(FIXTURE_MAGIC_LEN)
        if magic != FIXTURE_MAGIC:
            raise RuntimeError(f"unexpected fixture magic in {path}: {magic!r}")
        vals = struct.unpack("<IIIIIIIIIII", fp.read(11 * 4))
    return {
        "layer": int(vals[0]),
        "seq_len": int(vals[1]),
        "hidden": int(vals[2]),
        "topk": int(vals[9]),
        "union_experts": int(vals[10]),
        "path": path,
    }


def first_mismatch(a: list[int], b: list[int]) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def main() -> int:
    helper = load_hybrid_helpers()

    ap = argparse.ArgumentParser(
        description="Fast behavioral A/B for Qwen3.6 long decode: HF baseline vs owned-prefix patched tail"
    )
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--fixture", action="append", default=[], help="Owned layer fixtures in order")
    ap.add_argument("--splice-layer", type=int, default=None, help="HF layer output index to replace")
    ap.add_argument("--c-bin", default="./qwen36-c-prefix-q8-chain")
    ap.add_argument("--c-bin-prefix-flag", action="store_true")
    ap.add_argument("--prefix-seq-bin")
    ap.add_argument("--prefix-seq-fixture")
    ap.add_argument("--prefix-seq-dynamic", action="store_true")
    ap.add_argument("--full-layer-bin")
    ap.add_argument("--full-layer", action="append", type=int, default=None)
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    fixtures = list(args.fixture)
    if not fixtures:
        ap.error("at least one --fixture is required")
    full_layers = args.full_layer if args.full_layer is not None else [3]
    n_cycles = len(full_layers)
    expected_fixtures = n_cycles * 3
    if len(fixtures) != expected_fixtures:
        ap.error(f"expected {expected_fixtures} fixtures ({n_cycles} cycles x 3), got {len(fixtures)}")

    if args.splice_layer is not None:
        splice_layer = args.splice_layer
    elif args.full_layer_bin:
        splice_layer = full_layers[-1]
    else:
        splice_layer = len(fixtures) - 1

    fixture_meta = [read_fixture_meta(path) for path in fixtures]
    fixture_seq_lens = sorted({item["seq_len"] for item in fixture_meta})
    max_fixture_seq_len = max(fixture_seq_lens)

    prompt = read_prompt(args)
    print(f"[long-ab] prompt_chars={len(prompt)}")
    print(f"[long-ab] fixture_count={len(fixtures)} cycles={n_cycles} splice_layer={splice_layer}")
    print(f"[long-ab] full_layers={full_layers}")
    print(f"[long-ab] fixture_seq_lens={fixture_seq_lens}")
    print(f"[long-ab] loading tokenizer from {args.hf}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    print(f"[long-ab] loading model from {args.hf}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    torch.set_num_threads(min(32, os.cpu_count() or 16))
    hidden = int(model.config.hidden_size)
    prompt_token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    print(f"[long-ab] model loaded hidden={hidden} prompt_tokens={len(prompt_token_ids)}")

    baseline_token_ids = list(prompt_token_ids)
    baseline_generated = []
    baseline_steps = []
    hf_past = None
    hf_cached_len = 0
    print(f"[long-ab] running baseline decode n_predict={args.n_predict}")
    for step_idx in range(args.n_predict):
        step, hf_past, hf_cached_len = helper.run_hf_baseline_step(
            model=model,
            token_ids=baseline_token_ids,
            hf_past=hf_past,
            hf_cached_len=hf_cached_len,
            tokenizer=tokenizer,
            top_k=args.top_k,
        )
        next_id = step["hf_next_id"]
        baseline_token_ids.append(next_id)
        baseline_generated.append(next_id)
        baseline_steps.append({
            "step": step_idx,
            "seq_len": len(baseline_token_ids) - 1,
            "next_id": next_id,
            "next_text": step["hf_next_text"],
            "topk": step["hf_topk"],
            "baseline_ms": step["hf_baseline_ms"],
            "exceeds_fixture_seq_len": (len(baseline_token_ids) - 1) > max_fixture_seq_len,
        })
        print(f"[long-ab] baseline step={step_idx} next_id={next_id} next_text={json.dumps(step['hf_next_text'])}")

    patched_token_ids = list(prompt_token_ids)
    patched_generated = []
    patched_steps = []
    print(f"[long-ab] running patched decode n_predict={args.n_predict}")
    with tempfile.TemporaryDirectory(prefix="q36_long_ab_") as td:
        td_path = Path(td)
        for step_idx in range(args.n_predict):
            owned_seq, owned_meta = helper.run_owned_prefix_cycles(
                td_path=td_path,
                token_ids=patched_token_ids,
                hidden=hidden,
                gguf=args.gguf,
                fixtures=fixtures,
                full_layers=full_layers,
                c_bin=args.c_bin,
                c_bin_prefix_flag=args.c_bin_prefix_flag,
                full_layer_bin=args.full_layer_bin,
                prefix_seq_bin=args.prefix_seq_bin,
                prefix_seq_fixture=args.prefix_seq_fixture,
                prefix_seq_dynamic=args.prefix_seq_dynamic,
                step_idx=step_idx,
            )
            patched = helper.run_hf_patched_compare(
                model=model,
                token_ids=patched_token_ids,
                owned_seq=owned_seq,
                splice_layer=splice_layer,
                tokenizer=tokenizer,
                top_k=args.top_k,
            )
            next_id = patched["next_id"]
            patched_token_ids.append(next_id)
            patched_generated.append(next_id)
            patched_steps.append({
                "step": step_idx,
                "seq_len": len(patched_token_ids) - 1,
                "next_id": next_id,
                "next_text": patched["next_text"],
                "topk": patched["patched_topk"],
                "owned_prefix_ms": owned_meta["owned_prefix_ms"],
                "patched_ms": patched["hf_patched_ms"],
                "exceeds_fixture_seq_len": (len(patched_token_ids) - 1) > max_fixture_seq_len,
            })
            print(
                f"[long-ab] patched step={step_idx} next_id={next_id} "
                f"next_text={json.dumps(patched['next_text'])} owned_prefix_ms={owned_meta['owned_prefix_ms']:.2f}"
            )

    mismatch = first_mismatch(baseline_generated, patched_generated)
    compared = min(len(baseline_generated), len(patched_generated))
    matching = 0
    for i in range(compared):
        if baseline_generated[i] == patched_generated[i]:
            matching += 1
    token_agreement_rate = float(matching) / float(max(compared, 1))

    per_step = []
    for i in range(compared):
        a = baseline_steps[i]
        b = patched_steps[i]
        per_step.append({
            "step": i,
            "same_token": a["next_id"] == b["next_id"],
            "baseline_id": a["next_id"],
            "baseline_text": a["next_text"],
            "patched_id": b["next_id"],
            "patched_text": b["next_text"],
            "baseline_ms": a["baseline_ms"],
            "owned_prefix_ms": b["owned_prefix_ms"],
            "patched_ms": b["patched_ms"],
            "exceeds_fixture_seq_len": bool(a["exceeds_fixture_seq_len"] or b["exceeds_fixture_seq_len"]),
        })

    result = {
        "prompt": prompt,
        "prompt_tokens": len(prompt_token_ids),
        "fixture_seq_lens": fixture_seq_lens,
        "fixture_seq_len": max_fixture_seq_len,
        "post_trace_behavioral_only": True,
        "n_predict": args.n_predict,
        "full_layers": full_layers,
        "splice_layer": splice_layer,
        "baseline_generated_ids": baseline_generated,
        "patched_generated_ids": patched_generated,
        "baseline_generated_text": tokenizer.decode(baseline_generated, clean_up_tokenization_spaces=False),
        "patched_generated_text": tokenizer.decode(patched_generated, clean_up_tokenization_spaces=False),
        "first_mismatch_step": mismatch,
        "matching_steps": matching,
        "compared_steps": compared,
        "token_agreement_rate": token_agreement_rate,
        "baseline_steps": baseline_steps,
        "patched_steps": patched_steps,
        "per_step": per_step,
        "fixtures": fixture_meta,
    }

    print(f"first_mismatch_step: {result['first_mismatch_step']}")
    print(f"token_agreement_rate: {result['token_agreement_rate']:.6f}")
    print(f"baseline_generated_text: {json.dumps(result['baseline_generated_text'])}")
    print(f"patched_generated_text: {json.dumps(result['patched_generated_text'])}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
