#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
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


def vec_rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def vec_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    an = float(np.linalg.norm(aa))
    bn = float(np.linalg.norm(bb))
    if an == 0.0 or bn == 0.0:
        return float("nan")
    return float(np.dot(aa, bb) / (an * bn))


def topk(logits: np.ndarray, k: int):
    idx = np.argpartition(-logits, k - 1)[:k]
    idx = idx[np.argsort(-logits[idx])]
    return [{"id": int(i), "logit": float(logits[i])} for i in idx]


def topk_overlap(a_topk: list[dict], b_topk: list[dict]) -> int:
    return len({item["id"] for item in a_topk} & {item["id"] for item in b_topk})


def token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def read_fixture_seq_len(path: str) -> dict:
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


def run_hf_baseline_capture(*, model, token_ids: list[int], splice_layer: int):
    inputs = {"input_ids": torch.tensor([token_ids], dtype=torch.long)}
    captured = {}

    def capture_hook(_mod, _inp, out):
        value = out[0] if isinstance(out, tuple) else out
        captured["splice_seq"] = value.detach().float().cpu().numpy()[0]

    handle = model.model.layers[splice_layer].register_forward_hook(capture_hook)
    try:
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(**inputs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        handle.remove()

    if "splice_seq" not in captured:
        raise RuntimeError(f"failed to capture splice hidden for layer {splice_layer}")

    logits = out.logits[0, -1].detach().float().cpu().numpy()
    return {
        "splice_seq": captured["splice_seq"],
        "logits": logits,
        "elapsed_ms": elapsed_ms,
    }


def run_hf_patched_logits(*, model, token_ids: list[int], owned_seq: np.ndarray, splice_layer: int):
    inputs = {"input_ids": torch.tensor([token_ids], dtype=torch.long)}
    repl = torch.from_numpy(owned_seq).unsqueeze(0)

    def splice_patch(_mod, _inp, out):
        value = out[0] if isinstance(out, tuple) else out
        patched = value.clone()
        patched[:, :, :] = repl.to(device=patched.device, dtype=patched.dtype)
        if isinstance(out, tuple):
            return (patched,) + out[1:]
        return patched

    handle = model.model.layers[splice_layer].register_forward_hook(splice_patch)
    try:
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(**inputs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        handle.remove()
    return {
        "logits": out.logits[0, -1].detach().float().cpu().numpy(),
        "elapsed_ms": elapsed_ms,
    }


def parse_dynamic_lengths(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if line.startswith("dynamic_seq_len:"):
            parts = line.replace(":", "").split()
            # dynamic_seq_len 22 fixture_seq_len 21
            if len(parts) >= 4:
                out.append({
                    "dynamic_seq_len": int(parts[1]),
                    "fixture_seq_len": int(parts[3]),
                })
    return out


def main() -> int:
    helper = load_hybrid_helpers()

    ap = argparse.ArgumentParser(
        description="Forced-token shared-path validator for Qwen3.6 owned-prefix fidelity"
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
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--fail-on-fixture-overrun", action="store_true")
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

    fixture_meta = [read_fixture_seq_len(path) for path in fixtures]
    fixture_seq_lens = sorted({item["seq_len"] for item in fixture_meta})
    min_fixture_seq_len = min(fixture_seq_lens)
    max_fixture_seq_len = max(fixture_seq_lens)

    prompt = read_prompt(args)
    print(f"[shared-path] prompt_chars={len(prompt)}")
    print(f"[shared-path] fixture_count={len(fixtures)} cycles={n_cycles} splice_layer={splice_layer}")
    print(f"[shared-path] full_layers={full_layers}")
    print(f"[shared-path] fixture_seq_lens={fixture_seq_lens}")
    print(f"[shared-path] loading tokenizer from {args.hf}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    print(f"[shared-path] loading model from {args.hf}")
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
    token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    prompt_token_count = len(token_ids)
    print(f"[shared-path] model loaded hidden={hidden} prompt_tokens={prompt_token_count}")

    results = {
        "prompt": prompt,
        "prompt_tokens": prompt_token_count,
        "steps": [],
        "fixtures": fixture_meta,
        "full_layers": full_layers,
        "splice_layer": splice_layer,
        "min_fixture_seq_len": min_fixture_seq_len,
        "max_fixture_seq_len": max_fixture_seq_len,
        "shared_path_mode": "hf_forced_tokens",
    }

    with tempfile.TemporaryDirectory(prefix="q36_shared_path_") as td:
        td_path = Path(td)
        for step_idx in range(args.steps):
            seq_len = len(token_ids)
            print(f"[shared-path] step={step_idx} seq_len={seq_len}")
            if args.fail_on_fixture_overrun and seq_len > max_fixture_seq_len:
                raise RuntimeError(
                    f"shared prefix length {seq_len} exceeds max fixture_seq_len {max_fixture_seq_len}"
                )

            owned_seq, owned_meta = helper.run_owned_prefix_cycles(
                td_path=td_path,
                token_ids=token_ids,
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
            dynamic_len_flags = parse_dynamic_lengths(owned_meta["cycle_reports"])
            print(f"[shared-path] owned_prefix_ms={owned_meta['owned_prefix_ms']:.2f}")
            print("[shared-path] running hf baseline capture")
            baseline = run_hf_baseline_capture(model=model, token_ids=token_ids, splice_layer=splice_layer)
            print(f"[shared-path] hf_baseline_ms={baseline['elapsed_ms']:.2f}")
            print("[shared-path] running hf patched forward")
            patched = run_hf_patched_logits(
                model=model,
                token_ids=token_ids,
                owned_seq=owned_seq,
                splice_layer=splice_layer,
            )
            print(f"[shared-path] hf_patched_ms={patched['elapsed_ms']:.2f}")

            baseline_logits = baseline["logits"]
            patched_logits = patched["logits"]
            baseline_topk = topk(baseline_logits, args.top_k)
            patched_topk = topk(patched_logits, args.top_k)
            baseline_next_id = int(np.argmax(baseline_logits))
            patched_next_id = int(np.argmax(patched_logits))
            overlap = topk_overlap(baseline_topk, patched_topk)

            step = {
                "step": step_idx,
                "seq_len": seq_len,
                "fixture_seq_len_min": min_fixture_seq_len,
                "fixture_seq_len_max": max_fixture_seq_len,
                "exceeds_fixture_seq_len": seq_len > max_fixture_seq_len,
                "dynamic_len_flags": dynamic_len_flags,
                "owned_prefix_ms": owned_meta["owned_prefix_ms"],
                "hf_baseline_ms": baseline["elapsed_ms"],
                "hf_patched_ms": patched["elapsed_ms"],
                "owned_seq_cosine": vec_cosine(owned_seq, baseline["splice_seq"]),
                "owned_seq_rmse": vec_rmse(owned_seq, baseline["splice_seq"]),
                "owned_last_cosine": vec_cosine(owned_seq[-1], baseline["splice_seq"][-1]),
                "owned_last_rmse": vec_rmse(owned_seq[-1], baseline["splice_seq"][-1]),
                "logits_cosine": vec_cosine(patched_logits, baseline_logits),
                "logits_rmse": vec_rmse(patched_logits, baseline_logits),
                "argmax_equal": baseline_next_id == patched_next_id,
                "hf_next_id": baseline_next_id,
                "hf_next_text": token_text(tokenizer, baseline_next_id),
                "patched_next_id": patched_next_id,
                "patched_next_text": token_text(tokenizer, patched_next_id),
                "topk_overlap": overlap,
                "topk_overlap_rate": float(overlap) / float(max(args.top_k, 1)),
                "baseline_topk": baseline_topk,
                "patched_topk": patched_topk,
                "cycle_timings": owned_meta["cycle_timings"],
            }
            if owned_meta["prefix_seq_ms"] is not None:
                step["prefix_seq_ms"] = owned_meta["prefix_seq_ms"]

            results["steps"].append(step)

            print(f"[shared-path] owned_seq_cosine={step['owned_seq_cosine']:.8f}")
            print(f"[shared-path] logits_cosine={step['logits_cosine']:.8f}")
            print(f"[shared-path] argmax_equal={step['argmax_equal']}")
            print(
                f"[shared-path] hf_next=id={baseline_next_id} text={json.dumps(step['hf_next_text'])} "
                f"patched_next=id={patched_next_id} text={json.dumps(step['patched_next_text'])}"
            )
            if dynamic_len_flags:
                print(f"[shared-path] dynamic_len_flags={json.dumps(dynamic_len_flags)}")

            token_ids.append(baseline_next_id)

    results["final_shared_prefix_ids"] = token_ids
    results["forced_generated_ids"] = token_ids[prompt_token_count:]
    results["forced_generated_text"] = tokenizer.decode(results["forced_generated_ids"], clean_up_tokenization_spaces=False)
    results["all_argmax_equal"] = all(step["argmax_equal"] for step in results["steps"])
    results["min_owned_seq_cosine"] = min(step["owned_seq_cosine"] for step in results["steps"]) if results["steps"] else float("nan")
    results["min_logits_cosine"] = min(step["logits_cosine"] for step in results["steps"]) if results["steps"] else float("nan")
    results["any_fixture_overrun"] = any(step["exceeds_fixture_seq_len"] for step in results["steps"])

    print(f"all_argmax_equal: {results['all_argmax_equal']}")
    print(f"min_owned_seq_cosine: {results['min_owned_seq_cosine']:.8f}")
    print(f"min_logits_cosine: {results['min_logits_cosine']:.8f}")
    print(f"any_fixture_overrun: {results['any_fixture_overrun']}")
    print(f"forced_generated_text: {json.dumps(results['forced_generated_text'])}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
