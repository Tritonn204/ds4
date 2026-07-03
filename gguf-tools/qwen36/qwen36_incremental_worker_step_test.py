#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def vec_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    an = float(np.linalg.norm(aa))
    bn = float(np.linalg.norm(bb))
    if an == 0.0 or bn == 0.0:
        return float("nan")
    return float(np.dot(aa, bb) / (an * bn))


def vec_rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def read_line(proc: subprocess.Popen) -> str:
    line = proc.stdout.readline()
    if not line:
        stderr = proc.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"worker exited unexpectedly; stderr={stderr}")
    return line.decode("utf-8", errors="replace").strip()


def dump_hidden(proc: subprocess.Popen, seq_len: int, hidden: int) -> np.ndarray:
    proc.stdin.write(b"DUMP_HIDDEN\n")
    proc.stdin.flush()
    hdr = read_line(proc)
    if not hdr.startswith("HIDDEN "):
        raise RuntimeError(f"expected HIDDEN header, got: {hdr}")
    parts = hdr.split()
    n_floats = int(parts[1])
    n_bytes = int(parts[2])
    raw = proc.stdout.read(n_bytes)
    if len(raw) != n_bytes:
        raise RuntimeError(f"short hidden read: got {len(raw)} expected {n_bytes}")
    arr = np.frombuffer(raw, dtype=np.float32, count=n_floats)
    expect = seq_len * hidden
    if arr.size != expect:
        raise RuntimeError(f"hidden size mismatch: got {arr.size}, expected {expect}")
    return arr.reshape(seq_len, hidden)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate qwen36-incremental-worker PREFILL+STEP against HF blk.0")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--worker-bin", default=str(Path(__file__).resolve().parent.parent / "qwen36-incremental-worker"))
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    if not Path(args.worker_bin).exists():
        print(f"missing worker bin: {args.worker_bin}", file=sys.stderr)
        return 1

    print(f"[worker-step] loading tokenizer from {args.hf}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    prompt_token_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"][0].tolist()
    print(f"[worker-step] prompt_tokens={len(prompt_token_ids)}")

    print(f"[worker-step] loading model from {args.hf}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    torch.set_num_threads(args.threads)
    hidden = int(model.config.hidden_size)
    print(f"[worker-step] hidden={hidden} threads={args.threads}")

    prompt_hidden = {}

    def hook(_mod, _inp, out):
        value = out[0] if isinstance(out, tuple) else out
        prompt_hidden["seq"] = value.detach().float().cpu().numpy()[0]

    handle = model.model.layers[0].register_forward_hook(hook)
    print("[worker-step] running HF prompt forward")
    t0 = time.perf_counter()
    with torch.inference_mode():
        prompt_out = model(input_ids=torch.tensor([prompt_token_ids], dtype=torch.long))
    prompt_ms = (time.perf_counter() - t0) * 1000.0
    handle.remove()
    next_id = int(torch.argmax(prompt_out.logits[0, -1]).item())
    next_text = tokenizer.decode([next_id], clean_up_tokenization_spaces=False)
    print(f"[worker-step] hf_prompt_ms={prompt_ms:.2f} next_id={next_id} next_text={next_text!r}")

    full_token_ids = prompt_token_ids + [next_id]
    step_hidden = {}

    def hook_step(_mod, _inp, out):
        value = out[0] if isinstance(out, tuple) else out
        step_hidden["seq"] = value.detach().float().cpu().numpy()[0]

    handle = model.model.layers[0].register_forward_hook(hook_step)
    print("[worker-step] running HF prompt+step forward")
    t0 = time.perf_counter()
    with torch.inference_mode():
        model(input_ids=torch.tensor([full_token_ids], dtype=torch.long))
    full_ms = (time.perf_counter() - t0) * 1000.0
    handle.remove()
    print(f"[worker-step] hf_full_ms={full_ms:.2f}")

    hf_prompt_seq = prompt_hidden["seq"]
    hf_full_seq = step_hidden["seq"]

    proc = subprocess.Popen(
        [args.worker_bin, args.gguf],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(Path(args.worker_bin).resolve().parent),
    )
    try:
        ready = read_line(proc)
        if not ready.startswith("READY"):
            raise RuntimeError(f"expected READY, got: {ready}")

        prefill_cmd = "PREFILL " + " ".join(str(int(t)) for t in prompt_token_ids) + "\n"
        print("[worker-step] sending PREFILL")
        t0 = time.perf_counter()
        proc.stdin.write(prefill_cmd.encode("utf-8"))
        proc.stdin.flush()
        pref = read_line(proc)
        prefill_ms = (time.perf_counter() - t0) * 1000.0
        if not pref.startswith("PREFILL_OK "):
            raise RuntimeError(f"expected PREFILL_OK, got: {pref}")
        print(f"[worker-step] worker_prefill_ms={prefill_ms:.2f}")

        worker_prompt_seq = dump_hidden(proc, len(prompt_token_ids), hidden)

        print("[worker-step] sending STEP")
        t0 = time.perf_counter()
        proc.stdin.write(f"STEP {next_id}\n".encode("utf-8"))
        proc.stdin.flush()
        step = read_line(proc)
        step_ms = (time.perf_counter() - t0) * 1000.0
        if not step.startswith("STEP_OK "):
            raise RuntimeError(f"expected STEP_OK, got: {step}")
        print(f"[worker-step] worker_step_ms={step_ms:.2f}")

        worker_full_seq = dump_hidden(proc, len(full_token_ids), hidden)

        proc.stdin.write(b"QUIT\n")
        proc.stdin.flush()
        _ = read_line(proc)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    prompt_cos = vec_cosine(worker_prompt_seq, hf_prompt_seq)
    prompt_rmse = vec_rmse(worker_prompt_seq, hf_prompt_seq)
    full_cos = vec_cosine(worker_full_seq, hf_full_seq)
    full_rmse = vec_rmse(worker_full_seq, hf_full_seq)
    step_cos = vec_cosine(worker_full_seq[-1], hf_full_seq[-1])
    step_rmse = vec_rmse(worker_full_seq[-1], hf_full_seq[-1])

    print(f"prompt_seq_cosine: {prompt_cos:.8f}")
    print(f"prompt_seq_rmse: {prompt_rmse:.8f}")
    print(f"full_seq_cosine: {full_cos:.8f}")
    print(f"full_seq_rmse: {full_rmse:.8f}")
    print(f"step_last_cosine: {step_cos:.8f}")
    print(f"step_last_rmse: {step_rmse:.8f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
