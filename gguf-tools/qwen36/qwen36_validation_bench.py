#!/usr/bin/env python3
import argparse
import contextlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen36_hybrid_prefix_tail_greedy import (
    read_prompt,
    read_seq_hidden,
    run_hf_baseline_step,
    run_hf_patched_compare,
    run_owned_prefix_cycles,
)


def time_block(fn, repeat: int = 1):
    vals = []
    outs = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        vals.append((time.perf_counter() - t0) * 1000.0)
        outs.append(out)
    return outs, vals


def stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "mean_ms": float(np.mean(arr)),
    }


def log_phase(msg: str) -> None:
    print(f"[bench] {msg}", flush=True)


@contextlib.contextmanager
def hf_layer_progress(model, enabled: bool, label: str):
    if not enabled:
        yield
        return

    layer_times_ms: dict[int, float] = {}
    handles = []

    def make_pre(i: int):
        def _pre(_mod, _inp):
            layer_times_ms[i] = time.perf_counter()
        return _pre

    def make_post(i: int):
        def _post(_mod, _inp, _out):
            t0 = layer_times_ms.get(i)
            if t0 is None:
                return
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[bench] {label} layer={i} ms={dt_ms:.2f}", flush=True)
        return _post

    try:
        for i, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_pre_hook(make_pre(i)))
            handles.append(layer.register_forward_hook(make_post(i)))
        yield
    finally:
        for h in handles:
            h.remove()


def main() -> int:
    ap = argparse.ArgumentParser(description="Microbench strict-validation cost centers for Qwen3.6 hybrid ownership")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--hf-only", action="store_true",
                    help="Benchmark only the HF oracle path; skip owned-prefix and patched-forward phases")
    ap.add_argument("--fixture", action="append", default=[])
    ap.add_argument("--full-layer", action="append", type=int, default=None)
    ap.add_argument("--full-layer-bin")
    ap.add_argument("--c-bin")
    ap.add_argument("--c-bin-prefix-flag", action="store_true")
    ap.add_argument("--prefix-seq-bin")
    ap.add_argument("--prefix-seq-fixture")
    ap.add_argument("--prefix-seq-dynamic", action="store_true")
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--interop-threads", type=int, default=1)
    ap.add_argument("--hf-layer-progress", action="store_true",
                    help="Print per-layer timings during HF baseline/patched forwards")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--decode-steps", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    if not args.hf_only and not args.c_bin:
        ap.error("--c-bin is required unless --hf-only is set")
    if not args.hf_only and not args.fixture:
        ap.error("at least one --fixture is required")

    full_layers = args.full_layer if args.full_layer is not None else [3]
    prompt = read_prompt(args)
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    log_phase(f"loading model from {args.hf}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    threads = args.threads or min(32, os.cpu_count() or 16)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(args.interop_threads)
    except RuntimeError:
        log_phase("interop threads already initialized; leaving existing setting in place")
    hidden = int(model.config.hidden_size)
    token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    log_phase(
        f"model loaded hidden={hidden} prompt_tokens={len(token_ids)} "
        f"repeat={args.repeat} decode_steps={args.decode_steps} threads={threads}"
    )
    log_phase(f"fixture_count={len(args.fixture)} full_layers={full_layers}")

    results = {
        "prompt": prompt,
        "prompt_tokens": len(token_ids),
        "threads": threads,
        "repeat": args.repeat,
        "decode_steps": args.decode_steps,
        "full_layers": full_layers,
        "hf_only": bool(args.hf_only),
    }

    def do_prefill():
        with hf_layer_progress(model, args.hf_layer_progress, "hf_prefill"), torch.inference_mode():
            return model(input_ids=torch.tensor([token_ids], dtype=torch.long), use_cache=True)

    prefill_vals = []
    log_phase(f"starting hf baseline prefill x{args.repeat}")
    for i in range(args.repeat):
        log_phase(f"hf baseline prefill {i + 1}/{args.repeat}")
        t0 = time.perf_counter()
        do_prefill()
        ms = (time.perf_counter() - t0) * 1000.0
        prefill_vals.append(ms)
        log_phase(f"hf baseline prefill {i + 1}/{args.repeat} done ms={ms:.2f}")
    results["hf_baseline_prefill"] = stats(prefill_vals)

    def do_decode_series():
        hf_past = None
        hf_cached_len = 0
        cur_tokens = list(token_ids)
        series = []
        for _ in range(args.decode_steps):
            with hf_layer_progress(model, args.hf_layer_progress, "hf_decode"):
                step, hf_past, hf_cached_len = run_hf_baseline_step(
                    model=model,
                    token_ids=cur_tokens,
                    hf_past=hf_past,
                    hf_cached_len=hf_cached_len,
                    tokenizer=tokenizer,
                    top_k=args.top_k,
                )
            series.append(step["hf_baseline_ms"])
            cur_tokens.append(step["hf_next_id"])
        return series

    decode_series_runs = []
    log_phase(f"starting hf cached decode series x{args.repeat}")
    for i in range(args.repeat):
        log_phase(f"hf cached decode series {i + 1}/{args.repeat}")
        run = do_decode_series()
        decode_series_runs.append(run)
        log_phase(
            f"hf cached decode series {i + 1}/{args.repeat} done "
            f"step_ms={[round(x, 2) for x in run]}"
        )
    series = [ms for run in decode_series_runs for ms in run]
    results["hf_baseline_decode_step"] = stats(series)

    if args.hf_only:
        log_phase("hf-only benchmark complete")
        print(json.dumps(results, indent=2))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"json_out: {args.json_out}")
        return 0

    with tempfile.TemporaryDirectory(prefix="q36_val_bench_") as td:
        td_path = Path(td)
        log_phase("starting owned prefix microbench")
        owned_seq, owned_meta = run_owned_prefix_cycles(
            td_path=td_path,
            token_ids=token_ids,
            hidden=hidden,
            gguf=args.gguf,
            fixtures=args.fixture,
            full_layers=full_layers,
            c_bin=args.c_bin,
            c_bin_prefix_flag=args.c_bin_prefix_flag,
            full_layer_bin=args.full_layer_bin,
            prefix_seq_bin=args.prefix_seq_bin,
            prefix_seq_fixture=args.prefix_seq_fixture,
            prefix_seq_dynamic=args.prefix_seq_dynamic,
            step_idx=0,
        )
        log_phase(
            f"owned prefix done total_ms={owned_meta['owned_prefix_ms']:.2f} "
            f"cycle_ms={[round(x['total_ms'], 2) for x in owned_meta['cycle_timings']]}"
        )
        results["owned_prefix"] = {
            "owned_prefix_ms": float(owned_meta["owned_prefix_ms"]),
            "cycle_timings": owned_meta["cycle_timings"],
            "prefix_seq_ms": owned_meta["prefix_seq_ms"],
        }

        def do_patched():
            with hf_layer_progress(model, args.hf_layer_progress, "hf_patched"):
                return run_hf_patched_compare(
                    model=model,
                    token_ids=token_ids,
                    owned_seq=owned_seq,
                    splice_layer=full_layers[-1] if args.full_layer_bin else (len(args.fixture) - 1),
                    tokenizer=tokenizer,
                    top_k=args.top_k,
                )

        patched_runs = []
        log_phase(f"starting hf patched compare x{args.repeat}")
        for i in range(args.repeat):
            log_phase(f"hf patched compare {i + 1}/{args.repeat}")
            out = do_patched()
            patched_runs.append(out)
            log_phase(
                f"hf patched compare {i + 1}/{args.repeat} done "
                f"ms={out['hf_patched_ms']:.2f} next_id={out['next_id']}"
            )
        patched_ms = [item["hf_patched_ms"] for item in patched_runs]
        results["hf_patched_forward"] = stats(patched_ms)

    log_phase("benchmark complete")
    print(json.dumps(results, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
