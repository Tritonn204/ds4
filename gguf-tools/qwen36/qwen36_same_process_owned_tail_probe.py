#!/usr/bin/env python3
import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen36_hybrid_prefix_tail_greedy import (
    PrefixSeqWorker,
    HybridChainWorker,
    FullLayerWorker,
    split_cycle_fixtures,
    run_owned_prefix_cycles,
    run_hf_patched_compare,
    run_hf_baseline_step,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prefix-seq-worker-bin")
    ap.add_argument("--prefix-seq-fixture")
    ap.add_argument("--c-bin", required=True)
    ap.add_argument("--c-bin-worker-bin")
    ap.add_argument("--full-layer-worker-bin")
    ap.add_argument("--fixture", action="append", default=[])
    ap.add_argument("--full-layer", action="append", type=int, default=[])
    prompt_group = ap.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file")
    ap.add_argument("--splice-layer", type=int, default=15)
    ap.add_argument("--n-predict", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--compare-hf-token-only", action="store_true")
    ap.add_argument("--hf-baseline-layer-progress", action="store_true")
    ap.add_argument("--hf-patched-layer-progress", action="store_true")
    ap.add_argument("--close-workers-before-hf", action="store_true")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    prompt = args.prompt if args.prompt is not None else Path(args.prompt_file).read_text(encoding="utf-8")
    print(f"[sameproc] loading tokenizer from {args.hf}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    print(f"[sameproc] loading model from {args.hf}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    torch.set_num_threads(min(32, torch.get_num_threads() or 16))
    hidden = int(model.config.hidden_size)
    token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    initial_len = len(token_ids)
    print(f"[sameproc] model loaded hidden={hidden} prompt_tokens={len(token_ids)}")
    hf_past = None
    hf_cached_len = 0

    prefix_seq_worker = None
    hybrid_chain_workers = None
    full_layer_workers = None

    def close_prefix_worker() -> None:
        nonlocal prefix_seq_worker
        if prefix_seq_worker is not None:
            prefix_seq_worker.close()
            prefix_seq_worker = None

    def close_hybrid_workers() -> None:
        nonlocal hybrid_chain_workers
        if hybrid_chain_workers is not None:
            for worker in hybrid_chain_workers:
                worker.close()
            hybrid_chain_workers = None

    def close_full_workers() -> None:
        nonlocal full_layer_workers
        if full_layer_workers is not None:
            for worker in full_layer_workers:
                worker.close()
            full_layer_workers = None

    cycle_fixture_chunks = None
    if args.c_bin_worker_bin:
        cycle_fixture_chunks = split_cycle_fixtures(args.fixture, len(args.full_layer), args.prefix_seq_worker_bin is not None)

    with tempfile.TemporaryDirectory(prefix="q36_sameproc_") as td:
        td_path = Path(td)
        try:
            if args.prefix_seq_worker_bin:
                print("[sameproc] spawning prefix worker")
                prefix_seq_worker = PrefixSeqWorker(args.prefix_seq_worker_bin, args.gguf, args.prefix_seq_fixture)
            if args.c_bin_worker_bin:
                print("[sameproc] spawning hybrid workers")
                hybrid_chain_workers = []
                assert cycle_fixture_chunks is not None
                for idx, chunk in enumerate(cycle_fixture_chunks):
                    print(f"[sameproc] spawning hybrid worker cycle={idx}")
                    hybrid_chain_workers.append(HybridChainWorker(args.c_bin_worker_bin, args.gguf, chunk))
            if args.full_layer_worker_bin:
                print("[sameproc] spawning full-layer workers")
                full_layer_workers = []
                for layer_idx in args.full_layer:
                    print(f"[sameproc] spawning full-layer worker layer={layer_idx}")
                    full_layer_workers.append(FullLayerWorker(args.full_layer_worker_bin, args.gguf, layer_idx))

            steps = []
            for step_idx in range(args.n_predict):
                print(f"[sameproc] step={step_idx} seq_len={len(token_ids)}")
                print("[sameproc] running owned prefix")
                owned_seq, owned_meta = run_owned_prefix_cycles(
                    td_path=td_path,
                    token_ids=token_ids,
                    hidden=hidden,
                    gguf=args.gguf,
                    fixtures=args.fixture,
                    full_layers=args.full_layer,
                    c_bin=args.c_bin,
                    c_bin_prefix_flag=False,
                    full_layer_bin=None,
                    prefix_seq_bin=None,
                    prefix_seq_fixture=args.prefix_seq_fixture,
                    prefix_seq_dynamic=False,
                    prefix_seq_worker=prefix_seq_worker,
                    hybrid_chain_workers=hybrid_chain_workers,
                    full_layer_workers=full_layer_workers,
                    step_idx=step_idx,
                )
                print(f"[sameproc] owned_prefix_ms={owned_meta['owned_prefix_ms']:.2f}")

                if args.close_workers_before_hf:
                    print("[sameproc] closing workers before hf tail")
                    close_full_workers()
                    close_hybrid_workers()
                    close_prefix_worker()

                print("[sameproc] running hf patched tail")
                patched = run_hf_patched_compare(
                    model=model,
                    token_ids=token_ids,
                    owned_seq=owned_seq,
                    splice_layer=args.splice_layer,
                    tokenizer=tokenizer,
                    top_k=args.top_k,
                    layer_progress=args.hf_patched_layer_progress,
                )
                hf_base = None
                if args.compare_hf_token_only:
                    print("[sameproc] running hf baseline token")
                    hf_base, hf_past, hf_cached_len = run_hf_baseline_step(
                        model=model,
                        token_ids=token_ids,
                        hf_past=hf_past,
                        hf_cached_len=hf_cached_len,
                        tokenizer=tokenizer,
                        top_k=1,
                        layer_progress=args.hf_baseline_layer_progress,
                    )
                    print(
                        f"[sameproc] hf_next_id={hf_base['hf_next_id']} "
                        f"patched_next_id={patched['next_id']} equal={hf_base['hf_next_id'] == patched['next_id']}"
                    )

                print(f"[sameproc] hf_patched_ms={patched['hf_patched_ms']:.2f}")
                print(f"[sameproc] next_id={patched['next_id']} next_text={json.dumps(patched['next_text'])}")
                token_ids.append(patched["next_id"])
                decoded = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
                print(f"[sameproc] decoded_so_far={json.dumps(decoded)}")
                step = {
                    "step": step_idx,
                    "seq_len": len(token_ids) - 1,
                    "owned_prefix_ms": owned_meta["owned_prefix_ms"],
                    "patched": patched,
                }
                if hf_base is not None:
                    step["hf_next_id"] = hf_base["hf_next_id"]
                    step["hf_next_text"] = hf_base["hf_next_text"]
                    step["hf_baseline_ms"] = hf_base["hf_baseline_ms"]
                    step["argmax_equal"] = hf_base["hf_next_id"] == patched["next_id"]
                steps.append(step)

            out = {
                "prompt_tokens": initial_len,
                "generated_tokens": args.n_predict,
                "close_workers_before_hf": args.close_workers_before_hf,
                "compare_hf_token_only": args.compare_hf_token_only,
                "steps": steps,
                "generated_token_ids": token_ids[initial_len:],
                "generated_text": tokenizer.decode(token_ids[initial_len:], clean_up_tokenization_spaces=False),
                "final_text": tokenizer.decode(token_ids, clean_up_tokenization_spaces=False),
            }
            if args.json_out:
                Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
                print(f"json_out: {args.json_out}")
        finally:
            close_full_workers()
            close_hybrid_workers()
            close_prefix_worker()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
