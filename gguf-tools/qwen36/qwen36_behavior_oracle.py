#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen36_hybrid_prefix_tail_greedy import (
    PrefixSeqWorker,
    HybridChainWorker,
    FullLayerWorker,
    OwnedSessionWorker,
    write_owned_session_config,
    hf_layer_progress,
    read_prompt,
    run_hf_baseline_step,
    run_hf_patched_compare,
    run_owned_prefix_cycles,
    split_cycle_fixtures,
)


class OwnedSession:
    def __init__(self, args, hidden: int):
        self.args = args
        self.hidden = hidden
        self.use_native_owned_session = bool(args.owned_session_worker_bin)
        self.native_last_row_only = bool(args.owned_session_worker_bin and args.splice_layer >= 39 and len(args.full_layer) > 0)
        self.owned_session_worker = None
        self.owned_session_cfg_path = None
        self.prefix_seq_worker = None
        self.hybrid_chain_workers = None
        self.full_layer_workers = None
        self.td = tempfile.TemporaryDirectory(prefix="q36_owned_session_")
        self.td_path = Path(self.td.name)
        self.cycle_fixture_chunks = None
        self.owned_session_label = getattr(args, "owned_session_label", None)
        self.fixtures = list(args.fixture)
        self.owned_session_env: dict[str, str] = {}
        for item in getattr(args, "owned_session_env", []) or []:
            key, sep, value = item.partition("=")
            if not sep or not key:
                raise RuntimeError(f"bad --owned-session-env entry: {item!r}")
            self.owned_session_env[key] = value
        if getattr(args, "owned_session_unified_full_cpu", False):
            self.owned_session_env["QWEN36_UNIFIED_PREFILL_FULL_CPU"] = "1"
        if getattr(args, "owned_session_unified_full_gpu", False):
            self.owned_session_env["QWEN36_UNIFIED_FULL_GPU"] = "1"
        if getattr(args, "owned_session_unified_hybrid_gpu_cycles", 0):
            self.owned_session_env["QWEN36_UNIFIED_HYBRID_GPU_CYCLES"] = str(args.owned_session_unified_hybrid_gpu_cycles)
        effective_prefix_worker_bin = args.prefix_seq_worker_bin
        cycles_per_hybrid = 3
        n_cycles = len(args.full_layer)
        expected_fixtures = n_cycles * cycles_per_hybrid
        if self.use_native_owned_session and args.prefix_seq_fixture:
            self.fixtures = [args.prefix_seq_fixture, *self.fixtures]
            effective_prefix_worker_bin = None
        expected_worker_fixtures = expected_fixtures - 1 if effective_prefix_worker_bin is not None else expected_fixtures
        if effective_prefix_worker_bin is not None:
            if len(self.fixtures) == expected_fixtures:
                self.fixtures = self.fixtures[1:]
            elif len(self.fixtures) != expected_worker_fixtures:
                raise RuntimeError(
                    f"expected {expected_worker_fixtures} fixtures without blk.0 worker fixture "
                    f"or {expected_fixtures} with it included, got {len(self.fixtures)}"
                )
        elif len(self.fixtures) != expected_fixtures:
            raise RuntimeError(
                f"expected {expected_fixtures} fixtures ({n_cycles} cycles x {cycles_per_hybrid}), got {len(self.fixtures)}"
            )
        if args.c_bin_worker_bin:
            self.cycle_fixture_chunks = split_cycle_fixtures(
                self.fixtures,
                len(args.full_layer),
                effective_prefix_worker_bin is not None,
            )
        self.cycle_count = len(args.full_layer)

    def _ensure_workers(self) -> None:
        if self.args.owned_session_worker_bin:
            if self.owned_session_worker is None:
                print("[oracle] spawning owned-session worker")
                self.owned_session_cfg_path = self.td_path / "owned_session.cfg"
                write_owned_session_config(
                    self.owned_session_cfg_path,
                    fixtures=self.fixtures,
                    full_layers=self.args.full_layer,
                    prefix_seq_worker_bin=None,
                    prefix_seq_fixture=None,
                    c_bin_worker_bin=self.args.c_bin_worker_bin,
                    full_layer_worker_bin=self.args.full_layer_worker_bin,
                )
                self.owned_session_worker = OwnedSessionWorker(
                    self.args.owned_session_worker_bin,
                    self.args.gguf,
                    self.owned_session_cfg_path,
                    env=self.owned_session_env or None,
                    label=self.owned_session_label,
                )
            return
        if self.prefix_seq_worker is None and self.args.prefix_seq_worker_bin:
            print("[oracle] spawning prefix worker")
            self.prefix_seq_worker = PrefixSeqWorker(
                self.args.prefix_seq_worker_bin,
                self.args.gguf,
                self.args.prefix_seq_fixture,
            )
        if self.hybrid_chain_workers is None and self.args.c_bin_worker_bin:
            print("[oracle] spawning hybrid workers")
            self.hybrid_chain_workers = []
            assert self.cycle_fixture_chunks is not None
            for idx, chunk in enumerate(self.cycle_fixture_chunks):
                print(f"[oracle] spawning hybrid worker cycle={idx}")
                self.hybrid_chain_workers.append(
                    HybridChainWorker(self.args.c_bin_worker_bin, self.args.gguf, chunk)
                )
        if self.full_layer_workers is None and self.args.full_layer_worker_bin:
            print("[oracle] spawning full-layer workers")
            self.full_layer_workers = []
            for layer_idx in self.args.full_layer:
                print(f"[oracle] spawning full-layer worker layer={layer_idx}")
                self.full_layer_workers.append(
                    FullLayerWorker(self.args.full_layer_worker_bin, self.args.gguf, layer_idx)
                )

    def run_step(self, token_ids: list[int], step_idx: int) -> tuple[torch.Tensor, dict]:
        self._ensure_workers()
        if self.args.owned_session_worker_bin:
            if self.owned_session_worker is None:
                raise RuntimeError("owned-session worker was requested but is not available")
            dump_mode = "last" if self.native_last_row_only else "hidden"
            owned_seq, meta = self.owned_session_worker.run_for_token_ids(token_ids, self.hidden, dump_mode=dump_mode)
            return owned_seq, {"owned_prefix_ms": meta["worker_ms"], "owned_session_worker_meta": meta}
        owned_seq, owned_meta = run_owned_prefix_cycles(
            td_path=self.td_path,
            token_ids=token_ids,
            hidden=self.hidden,
            gguf=self.args.gguf,
            fixtures=self.fixtures,
            full_layers=self.args.full_layer,
            c_bin=self.args.c_bin,
            c_bin_prefix_flag=False,
            full_layer_bin=None,
            prefix_seq_bin=None,
            prefix_seq_fixture=self.args.prefix_seq_fixture,
            prefix_seq_dynamic=False,
            prefix_seq_worker=self.prefix_seq_worker,
            hybrid_chain_workers=self.hybrid_chain_workers,
            full_layer_workers=self.full_layer_workers,
            step_idx=step_idx,
        )
        return owned_seq, owned_meta

    def dump_cycle_hidden(self, cycle_idx: int) -> np.ndarray:
        self._ensure_workers()
        if self.owned_session_worker is None:
            raise RuntimeError("cycle dumps require owned-session worker mode")
        return self.owned_session_worker.dump_cycle_hidden(cycle_idx)

    def dump_cycle_pre_hidden(self, cycle_idx: int) -> np.ndarray:
        self._ensure_workers()
        if self.owned_session_worker is None:
            raise RuntimeError("cycle dumps require owned-session worker mode")
        return self.owned_session_worker.dump_cycle_pre_hidden(cycle_idx)

    def dump_cycle_last(self, cycle_idx: int) -> np.ndarray:
        self._ensure_workers()
        if self.owned_session_worker is None:
            raise RuntimeError("cycle dumps require owned-session worker mode")
        return self.owned_session_worker.dump_cycle_last(cycle_idx)

    def dump_cycle_pre_last(self, cycle_idx: int) -> np.ndarray:
        self._ensure_workers()
        if self.owned_session_worker is None:
            raise RuntimeError("cycle dumps require owned-session worker mode")
        return self.owned_session_worker.dump_cycle_pre_last(cycle_idx)

    def stderr_lines(self) -> list[str]:
        self._ensure_workers()
        if self.owned_session_worker is None:
            return []
        return self.owned_session_worker.stderr_lines()

    def close(self) -> None:
        if self.owned_session_worker is not None:
            self.owned_session_worker.close()
            self.owned_session_worker = None
        if self.full_layer_workers is not None:
            for worker in self.full_layer_workers:
                worker.close()
            self.full_layer_workers = None
        if self.hybrid_chain_workers is not None:
            for worker in self.hybrid_chain_workers:
                worker.close()
            self.hybrid_chain_workers = None
        if self.prefix_seq_worker is not None:
            self.prefix_seq_worker.close()
            self.prefix_seq_worker = None
        self.td.cleanup()


def _common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--json-out")


def _patched_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--hf-baseline-layer-progress", dest="hf_baseline_layer_progress", action="store_true", default=True)
    ap.add_argument("--no-hf-baseline-layer-progress", dest="hf_baseline_layer_progress", action="store_false")
    ap.add_argument("--prefix-seq-worker-bin")
    ap.add_argument("--prefix-seq-fixture")
    ap.add_argument("--c-bin")
    ap.add_argument("--c-bin-worker-bin")
    ap.add_argument("--owned-session-worker-bin")
    ap.add_argument("--full-layer-worker-bin")
    ap.add_argument("--owned-session-env", action="append", default=[])
    ap.add_argument("--owned-session-unified-full-cpu", action="store_true")
    ap.add_argument("--owned-session-unified-full-gpu", action="store_true")
    ap.add_argument("--owned-session-unified-hybrid-gpu-cycles", type=int, default=0)
    ap.add_argument("--fixture", action="append", default=[])
    ap.add_argument("--full-layer", action="append", type=int, default=[])
    ap.add_argument("--splice-layer", type=int, default=15)
    ap.add_argument("--hf-patched-layer-progress", dest="hf_patched_layer_progress", action="store_true", default=True)
    ap.add_argument("--no-hf-patched-layer-progress", dest="hf_patched_layer_progress", action="store_false")
    ap.add_argument("--same-process-tail", action="store_true",
                    help="Run HF tail in the same Python process after owned-session prefill; default uses a fresh subprocess")
    ap.add_argument(
        "--tail-source",
        choices=["auto", "hf", "gguf", "compare"],
        default="auto",
        help="Tail scorer source for all-GPU patched-session decode",
    )


def _load_model_and_prompt(hf: str, prompt: str):
    print(f"[oracle] loading tokenizer from {hf}")
    tokenizer = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    print(f"[oracle] loading model from {hf}")
    model = AutoModelForCausalLM.from_pretrained(
        hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    torch.set_num_threads(min(32, os.cpu_count() or 16))
    token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    print(f"[oracle] model loaded prompt_tokens={len(token_ids)} hidden={int(model.config.hidden_size)}")
    return tokenizer, model, token_ids


def _load_tokenizer_and_prompt(hf: str, prompt: str):
    print(f"[oracle] loading tokenizer from {hf}")
    tokenizer = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    print(f"[oracle] tokenizer loaded prompt_tokens={len(token_ids)}")
    return tokenizer, token_ids


def run_fresh_process_tail(args, owned_seq_path: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="q36_behavior_tail_") as td:
        td_path = Path(td)
        prompt_path = td_path / "prompt.txt"
        prompt = read_prompt(args)
        prompt_path.write_text(prompt, encoding="utf-8")
        json_path = td_path / "tail.json"
        cmd = [
            sys.executable,
            "/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_worker_residence_probe.py",
            "--hf", args.hf,
            "--gguf", args.gguf,
            "--owned-seq-f32", str(owned_seq_path),
            "--prompt-file", str(prompt_path),
            "--splice-layer", str(args.splice_layer),
            "--top-k", str(args.top_k),
            "--json-out", str(json_path),
        ]
        if args.hf_patched_layer_progress:
            cmd.append("--hf-tail-layer-progress")
        print("[oracle] running fresh-process hf tail")
        subprocess.run(cmd, check=True)
        out = json.loads(json_path.read_text(encoding="utf-8"))
        return out["result"]


def run_baseline_mode(args) -> int:
    prompt = read_prompt(args)
    tokenizer, model, token_ids = _load_model_and_prompt(args.hf, prompt)
    hf_past = None
    hf_cached_len = 0
    initial_len = len(token_ids)
    steps = []
    for step_idx in range(args.n_predict):
        print(f"[oracle] baseline step={step_idx} seq_len={len(token_ids)}")
        with hf_layer_progress(model, args.hf_baseline_layer_progress, "oracle_baseline"):
            step, hf_past, hf_cached_len = run_hf_baseline_step(
                model=model,
                token_ids=token_ids,
                hf_past=hf_past,
                hf_cached_len=hf_cached_len,
                tokenizer=tokenizer,
                top_k=args.top_k,
                layer_progress=False,
            )
        token_ids.append(step["hf_next_id"])
        decoded = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
        print(f"[oracle] baseline next_id={step['hf_next_id']} next_text={json.dumps(step['hf_next_text'])}")
        steps.append(
            {
                "step": step_idx,
                "seq_len": len(token_ids) - 1,
                "next_id": step["hf_next_id"],
                "next_text": step["hf_next_text"],
                "hf_baseline_ms": step["hf_baseline_ms"],
            }
        )
        print(f"[oracle] baseline decoded_so_far={json.dumps(decoded)}")

    out = {
        "mode": "baseline",
        "prompt": prompt,
        "prompt_tokens": initial_len,
        "generated_tokens": args.n_predict,
        "steps": steps,
        "generated_token_ids": token_ids[initial_len:],
        "generated_text": tokenizer.decode(token_ids[initial_len:], clean_up_tokenization_spaces=False),
        "full_text": tokenizer.decode(token_ids, clean_up_tokenization_spaces=False),
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")
    else:
        print(json.dumps(out, indent=2))
    return 0


def run_patched_mode(args) -> int:
    prompt = read_prompt(args)
    tokenizer, model, token_ids = _load_model_and_prompt(args.hf, prompt)
    hidden = int(model.config.hidden_size)
    initial_len = len(token_ids)
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

    with tempfile.TemporaryDirectory(prefix="q36_behavior_oracle_") as td:
        td_path = Path(td)
        try:
            if args.prefix_seq_worker_bin:
                print("[oracle] spawning prefix worker")
                prefix_seq_worker = PrefixSeqWorker(args.prefix_seq_worker_bin, args.gguf, args.prefix_seq_fixture)
            if args.c_bin_worker_bin:
                print("[oracle] spawning hybrid workers")
                hybrid_chain_workers = []
                assert cycle_fixture_chunks is not None
                for idx, chunk in enumerate(cycle_fixture_chunks):
                    print(f"[oracle] spawning hybrid worker cycle={idx}")
                    hybrid_chain_workers.append(HybridChainWorker(args.c_bin_worker_bin, args.gguf, chunk))
            if args.full_layer_worker_bin:
                print("[oracle] spawning full-layer workers")
                full_layer_workers = []
                for layer_idx in args.full_layer:
                    print(f"[oracle] spawning full-layer worker layer={layer_idx}")
                    full_layer_workers.append(FullLayerWorker(args.full_layer_worker_bin, args.gguf, layer_idx))

            steps = []
            for step_idx in range(args.n_predict):
                print(f"[oracle] patched step={step_idx} seq_len={len(token_ids)}")
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
                print(f"[oracle] patched owned_prefix_ms={owned_meta['owned_prefix_ms']:.2f}")
                patched = run_hf_patched_compare(
                    model=model,
                    token_ids=token_ids,
                    owned_seq=owned_seq,
                    splice_layer=args.splice_layer,
                    tokenizer=tokenizer,
                    top_k=args.top_k,
                    layer_progress=args.hf_patched_layer_progress,
                )
                token_ids.append(patched["next_id"])
                decoded = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
                print(f"[oracle] patched next_id={patched['next_id']} next_text={json.dumps(patched['next_text'])}")
                steps.append(
                    {
                        "step": step_idx,
                        "seq_len": len(token_ids) - 1,
                        "next_id": patched["next_id"],
                        "next_text": patched["next_text"],
                        "owned_prefix_ms": owned_meta["owned_prefix_ms"],
                        "hf_patched_ms": patched["hf_patched_ms"],
                    }
                )
                print(f"[oracle] patched decoded_so_far={json.dumps(decoded)}")

            out = {
                "mode": "patched",
                "prompt": prompt,
                "prompt_tokens": initial_len,
                "generated_tokens": args.n_predict,
                "steps": steps,
                "generated_token_ids": token_ids[initial_len:],
                "generated_text": tokenizer.decode(token_ids[initial_len:], clean_up_tokenization_spaces=False),
                "full_text": tokenizer.decode(token_ids, clean_up_tokenization_spaces=False),
            }
            if args.json_out:
                Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
                print(f"json_out: {args.json_out}")
            else:
                print(json.dumps(out, indent=2))
        finally:
            close_full_workers()
            close_hybrid_workers()
            close_prefix_worker()
    return 0


def _load_tail_weights(hf_path: str):
    """Load final norm and lm_head weights (~300MB) from safetensors directly."""
    hf = Path(hf_path)
    index_file = hf / "model.safetensors.index.json"

    if index_file.exists():
        index = json.loads(index_file.read_text())
        weight_map = index.get("weight_map", {})
        norm_shard = hf / weight_map.get("model.language_model.norm.weight", "model.safetensors")
        lm_head_shard = hf / weight_map.get("lm_head.weight", "model.safetensors")
        with safe_open(norm_shard, framework="pt", device="cpu") as f:
            norm_w = f.get_tensor("model.language_model.norm.weight")
        with safe_open(lm_head_shard, framework="pt", device="cpu") as f:
            lm_head_w = f.get_tensor("lm_head.weight")
        return norm_w.float(), lm_head_w.float()

    single = hf / "model.safetensors"
    if single.exists():
        with safe_open(single, framework="pt", device="cpu") as f:
            return f.get_tensor("model.language_model.norm.weight").float(), f.get_tensor("lm_head.weight").float()

    # nocommit: fallback loads everything if safetensors not found
    model = AutoModelForCausalLM.from_pretrained(str(hf), device_map="cpu", low_cpu_mem_usage=True, torch_dtype="auto")
    norm_w = model.model.norm.weight.detach().cpu()
    lm_head_w = model.lm_head.weight.detach().cpu()
    del model
    return norm_w, lm_head_w


def _gguf_tensor_map(reader: GGUFReader) -> dict[str, object]:
    return {t.name: t for t in reader.tensors}


def _gguf_load_f32(tensors: dict[str, object], name: str) -> np.ndarray:
    t = tensors[name]
    return dequantize(t.data, t.tensor_type).astype(np.float32, copy=False)


def _as_weight(arr: np.ndarray, out_rows: int, in_cols: int) -> np.ndarray:
    if arr.shape == (out_rows, in_cols):
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.shape == (in_cols, out_rows):
        return np.ascontiguousarray(arr.T, dtype=np.float32)
    raise ValueError(f"unexpected weight shape {arr.shape} for expected {(out_rows, in_cols)}")


def _load_tail_weights_from_gguf(gguf_path: str):
    reader = GGUFReader(gguf_path)
    tensors = _gguf_tensor_map(reader)
    norm_w = torch.from_numpy(_gguf_load_f32(tensors, "output_norm.weight"))
    lm_head_w = torch.from_numpy(_as_weight(_gguf_load_f32(tensors, "output.weight"), 248320, 2048))
    return norm_w, lm_head_w


def _run_lightweight_tail(owned_seq: np.ndarray, norm_w: torch.Tensor, lm_head_w: torch.Tensor, tokenizer, top_k: int) -> dict:
    """Run just RMS norm + lm_head on the last row of owned_seq."""
    t0 = time.perf_counter()
    last_row = owned_seq[-1]  # [hidden]
    rms = np.sqrt(np.mean(last_row.astype(np.float64) ** 2) + 1e-6)
    normed = (last_row / rms) * norm_w.numpy()
    logits = normed @ lm_head_w.numpy().T
    next_id = int(np.argmax(logits))
    k = max(1, min(int(top_k), int(logits.shape[0])))
    top_idx = np.argpartition(-logits, k - 1)[:k]
    top_idx = top_idx[np.argsort(-logits[top_idx])]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "next_id": next_id,
        "next_text": tokenizer.decode([next_id], clean_up_tokenization_spaces=False),
        "hf_patched_ms": elapsed_ms,
        "topk": [{"id": int(i), "text": tokenizer.decode([int(i)], clean_up_tokenization_spaces=False), "logit": float(logits[i])} for i in top_idx],
    }


def _resolve_tail_source(args) -> str:
    if args.tail_source != "auto":
        return args.tail_source
    gguf_name = Path(args.gguf).name.upper()
    return "gguf" if "Q4" in gguf_name or "Q5" in gguf_name or "Q6" in gguf_name else "hf"


def run_patched_session_mode(args) -> int:
    prompt = read_prompt(args)
    all_layers_on_gpu = args.splice_layer >= 39 and len(args.full_layer) > 0
    tail_weights = None
    tail_source = None
    if all_layers_on_gpu:
        tail_source = _resolve_tail_source(args)
        if tail_source == "hf":
            print("[oracle] all layers on GPU, loading lightweight tail weights from safetensors")
            tail_weights = {"hf": _load_tail_weights(args.hf)}
        elif tail_source == "gguf":
            print("[oracle] all layers on GPU, loading lightweight tail weights from gguf")
            tail_weights = {"gguf": _load_tail_weights_from_gguf(args.gguf)}
        else:
            print("[oracle] all layers on GPU, loading lightweight tail weights from safetensors and gguf")
            tail_weights = {
                "hf": _load_tail_weights(args.hf),
                "gguf": _load_tail_weights_from_gguf(args.gguf),
            }
    elif args.same_process_tail:
        tokenizer, model, token_ids = _load_model_and_prompt(args.hf, prompt)
        hidden = int(model.config.hidden_size)
    else:
        tokenizer, token_ids = _load_tokenizer_and_prompt(args.hf, prompt)
        model = None
        hidden = 2048

    if tail_weights is not None:
        tokenizer, token_ids = _load_tokenizer_and_prompt(args.hf, prompt)
        hidden = 2048

    initial_len = len(token_ids)
    session = OwnedSession(args, hidden)

    try:
        steps = []
        with tempfile.TemporaryDirectory(prefix="q36_behavior_session_") as td:
            td_path = Path(td)
            owned_seq_path = td_path / "owned_seq.f32"
            for step_idx in range(args.n_predict):
                print(f"[oracle] patched-session step={step_idx} seq_len={len(token_ids)}")
                owned_seq, owned_meta = session.run_step(token_ids, step_idx)
                owned_seq.astype("float32").tofile(owned_seq_path)
                if tail_weights is not None:
                    if tail_source == "hf":
                        norm_w, lm_head_w = tail_weights["hf"]
                        patched = _run_lightweight_tail(owned_seq, norm_w, lm_head_w, tokenizer, args.top_k)
                    elif tail_source == "gguf":
                        norm_w, lm_head_w = tail_weights["gguf"]
                        patched = _run_lightweight_tail(owned_seq, norm_w, lm_head_w, tokenizer, args.top_k)
                    else:
                        hf_norm_w, hf_lm_head_w = tail_weights["hf"]
                        gguf_norm_w, gguf_lm_head_w = tail_weights["gguf"]
                        hf_tail = _run_lightweight_tail(owned_seq, hf_norm_w, hf_lm_head_w, tokenizer, args.top_k)
                        gguf_tail = _run_lightweight_tail(owned_seq, gguf_norm_w, gguf_lm_head_w, tokenizer, args.top_k)
                        patched = gguf_tail
                        patched["tail_compare"] = {
                            "hf_next_id": hf_tail["next_id"],
                            "hf_next_text": hf_tail["next_text"],
                            "gguf_next_id": gguf_tail["next_id"],
                            "gguf_next_text": gguf_tail["next_text"],
                            "same_argmax": hf_tail["next_id"] == gguf_tail["next_id"],
                            "hf_topk": hf_tail["topk"],
                            "gguf_topk": gguf_tail["topk"],
                        }
                elif args.same_process_tail:
                    patched = run_hf_patched_compare(
                        model=model,
                        token_ids=token_ids,
                        owned_seq=owned_seq,
                        splice_layer=args.splice_layer,
                        tokenizer=tokenizer,
                        top_k=args.top_k,
                        layer_progress=args.hf_patched_layer_progress,
                    )
                else:
                    patched = run_fresh_process_tail(args, owned_seq_path)
                token_ids.append(patched["next_id"])
                decoded = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
                step = {
                    "step": step_idx,
                    "seq_len": len(token_ids) - 1,
                    "next_id": patched["next_id"],
                    "next_text": patched["next_text"],
                    "hf_patched_ms": patched["hf_patched_ms"],
                }
                if "tail_compare" in patched:
                    step["tail_compare"] = patched["tail_compare"]
                if step_idx == 0:
                    step["owned_prefill_ms"] = owned_meta["owned_prefix_ms"]
                    print(f"[oracle] patched-session owned_prefill_ms={owned_meta['owned_prefix_ms']:.2f}")
                else:
                    step["owned_step_ms"] = owned_meta["owned_prefix_ms"]
                    print(f"[oracle] patched-session owned_step_ms={owned_meta['owned_prefix_ms']:.2f}")
                steps.append(step)
                print(f"[oracle] patched-session next_id={patched['next_id']} next_text={json.dumps(patched['next_text'])}")
                print(f"[oracle] patched-session decoded_so_far={json.dumps(decoded)}")

        out = {
            "mode": "patched-session",
            "prompt": prompt,
            "prompt_tokens": initial_len,
            "generated_tokens": args.n_predict,
            "steps": steps,
            "generated_token_ids": token_ids[initial_len:],
            "generated_text": tokenizer.decode(token_ids[initial_len:], clean_up_tokenization_spaces=False),
            "full_text": tokenizer.decode(token_ids, clean_up_tokenization_spaces=False),
        }
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"json_out: {args.json_out}")
        else:
            print(json.dumps(out, indent=2))
    finally:
        session.close()
    return 0


def build_self_cmd(mode: str, args, json_out: str) -> list[str]:
    cmd = [sys.executable, __file__, "--mode", mode, "--hf", args.hf, "--gguf", args.gguf, "--n-predict", str(args.n_predict), "--top-k", str(args.top_k), "--json-out", json_out]
    if args.prompt_file:
        cmd.extend(["--prompt-file", args.prompt_file])
    else:
        cmd.extend(["--prompt", args.prompt])
    if mode == "patched":
        cmd.extend(["--splice-layer", str(args.splice_layer)])
        if args.prefix_seq_worker_bin:
            cmd.extend(["--prefix-seq-worker-bin", args.prefix_seq_worker_bin])
        if args.prefix_seq_fixture:
            cmd.extend(["--prefix-seq-fixture", args.prefix_seq_fixture])
        if args.c_bin:
            cmd.extend(["--c-bin", args.c_bin])
        if args.c_bin_worker_bin:
            cmd.extend(["--c-bin-worker-bin", args.c_bin_worker_bin])
        if args.owned_session_worker_bin:
            cmd.extend(["--owned-session-worker-bin", args.owned_session_worker_bin])
        if args.full_layer_worker_bin:
            cmd.extend(["--full-layer-worker-bin", args.full_layer_worker_bin])
        for layer in args.full_layer:
            cmd.extend(["--full-layer", str(layer)])
        for fixture in args.fixture:
            cmd.extend(["--fixture", fixture])
        if args.hf_patched_layer_progress:
            cmd.append("--hf-patched-layer-progress")
    return cmd


def run_orchestrate_mode(args) -> int:
    with tempfile.TemporaryDirectory(prefix="q36_behavior_orchestrate_") as td:
        td_path = Path(td)
        baseline_json = str(td_path / "baseline.json")
        patched_json = str(td_path / "patched.json")
        print("[oracle] running isolated baseline path")
        subprocess.run(build_self_cmd("baseline", args, baseline_json), check=True)
        print("[oracle] running isolated patched path")
        subprocess.run(build_self_cmd("patched", args, patched_json), check=True)
        baseline = json.loads(Path(baseline_json).read_text(encoding="utf-8"))
        patched = json.loads(Path(patched_json).read_text(encoding="utf-8"))
        base_ids = baseline["generated_token_ids"]
        patched_ids = patched["generated_token_ids"]
        compare_len = min(len(base_ids), len(patched_ids))
        first_mismatch = None
        for i in range(compare_len):
            if base_ids[i] != patched_ids[i]:
                first_mismatch = i
                break
        out = {
            "baseline_json": baseline_json,
            "patched_json": patched_json,
            "baseline_generated_text": baseline["generated_text"],
            "patched_generated_text": patched["generated_text"],
            "all_equal": first_mismatch is None and len(base_ids) == len(patched_ids),
            "first_mismatch_step": first_mismatch,
            "compared_steps": compare_len,
        }
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"json_out: {args.json_out}")
        else:
            print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Isolated baseline/patched behavior oracle for Qwen3.6 narrow-runtime decode")
    ap.add_argument("--mode", choices=["orchestrate", "baseline", "patched", "patched-session"], default="orchestrate")
    _common_args(ap)
    _patched_args(ap)
    args = ap.parse_args()

    if args.mode == "baseline":
        return run_baseline_mode(args)
    if args.mode == "patched":
        return run_patched_mode(args)
    if args.mode == "patched-session":
        return run_patched_session_mode(args)
    if not args.c_bin or not args.prefix_seq_worker_bin or not args.prefix_seq_fixture or not args.c_bin_worker_bin or not args.full_layer_worker_bin:
        ap.error("orchestrate mode requires the patched-path worker/bin arguments")
    if not args.fixture or not args.full_layer:
        ap.error("orchestrate mode requires --fixture and --full-layer values")
    return run_orchestrate_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
