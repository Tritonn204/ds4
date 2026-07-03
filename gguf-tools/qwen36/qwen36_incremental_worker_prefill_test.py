#!/usr/bin/env python3
"""Test qwen36-incremental-worker blk.0 prefill against HF reference.

Loads the full HF model with low_cpu_mem_usage=True, runs a short prompt,
captures blk.0 output via forward hook, compares against the C worker.
"""
import os
import struct
import subprocess
import sys
import time
import numpy as np
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/mnt/e/tensors/Qwen3.6-35B-A3B/"
GGUF_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/"
    "snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/"
    "Qwen3.6-35B-A3B-Q8_0.gguf"
)
WORKER_BIN = os.path.join(os.path.dirname(__file__), "..", "qwen36-incremental-worker")
TIMEOUT = 180
PROMPT = "Hello world how are you today"


def run_worker(gguf_path, worker_bin, token_ids):
    proc = subprocess.Popen(
        [worker_bin, gguf_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(os.path.abspath(worker_bin)),
    )

    def read_line():
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("worker exited unexpectedly")
        return line.decode().strip()

    try:
        ready = read_line()
        if not ready.startswith("READY"):
            raise RuntimeError(f"Expected READY, got: {ready}")

        tokens_str = " ".join(str(int(t)) for t in token_ids)
        proc.stdin.write(f"PREFILL {tokens_str}\n".encode())
        proc.stdin.flush()

        pref_line = read_line()
        if not pref_line.startswith("PREFILL_OK"):
            raise RuntimeError(f"Expected PREFILL_OK, got: {pref_line}")
        parts = pref_line.split()
        seq_len = int(parts[1])
        hidden = int(parts[2])

        proc.stdin.write(b"DUMP_HIDDEN\n")
        proc.stdin.flush()

        hidden_line = read_line()
        if not hidden_line.startswith("HIDDEN"):
            raise RuntimeError(f"Expected HIDDEN, got: {hidden_line}")
        parts = hidden_line.split()
        n_floats = int(parts[1])
        n_bytes = int(parts[2])

        raw = proc.stdout.read(n_bytes)
        if len(raw) != n_bytes:
            raise RuntimeError(f"Short read: {len(raw)} != {n_bytes}")

        proc.stdin.write(b"QUIT\n")
        proc.stdin.flush()
        proc.wait(timeout=5)

        data = np.frombuffer(raw, dtype=np.float32, count=n_floats)
        return data.reshape(seq_len, hidden)

    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def main():
    if not os.path.exists(GGUF_PATH):
        print(f"FAIL: GGUF model not found at {GGUF_PATH}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(WORKER_BIN):
        print(f"FAIL: Worker binary not found at {WORKER_BIN}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"FAIL: HF model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer from {MODEL_PATH}...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"Tokenizer loaded in {time.perf_counter() - t0:.1f}s")

    token_ids = tokenizer.encode(PROMPT)
    print(f"Prompt: {PROMPT!r} -> {len(token_ids)} tokens: {token_ids}")

    print(f"Loading HF model from {MODEL_PATH}...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    torch.set_num_threads(min(32, os.cpu_count() or 16))
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    hidden = int(model.config.hidden_size)
    seq_len = len(token_ids)
    print(f"Hidden size: {hidden}, Seq len: {seq_len}")

    layer0_hidden = None

    def hook(_mod, _inp, out):
        nonlocal layer0_hidden
        if isinstance(out, tuple):
            layer0_hidden = out[0]
        else:
            layer0_hidden = out

    handle = model.model.layers[0].register_forward_hook(hook)

    print("Running HF forward...")
    t0 = time.perf_counter()
    input_ids = torch.tensor([token_ids])
    with torch.no_grad():
        model(input_ids=input_ids)
    print(f"HF forward done in {time.perf_counter() - t0:.1f}s")

    handle.remove()

    if layer0_hidden is None:
        print("FAIL: Hook did not capture layer 0 output", file=sys.stderr)
        sys.exit(1)

    hf_output = layer0_hidden[0].detach().float().cpu().numpy()
    print(f"HF blk.0 output shape: {hf_output.shape}")

    print(f"Running C worker: {WORKER_BIN} {GGUF_PATH}")
    t0 = time.perf_counter()
    worker_hidden = run_worker(GGUF_PATH, WORKER_BIN, token_ids)
    print(f"Worker done in {time.perf_counter() - t0:.1f}s")
    print(f"Worker output shape: {worker_hidden.shape}")

    if hf_output.shape != worker_hidden.shape:
        print(f"FAIL: shape mismatch HF={hf_output.shape} worker={worker_hidden.shape}")
        sys.exit(1)

    all_cosines = []
    for t in range(seq_len):
        a = hf_output[t].astype(np.float64)
        b = worker_hidden[t].astype(np.float64)
        dot = np.dot(a, b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        cos = float(dot / (na * nb + 1e-12))
        mae = float(np.mean(np.abs(a - b)))
        all_cosines.append(cos)
        status = "OK" if cos > 0.99 else ("?" if cos > 0.95 else "!!")
        print(f"  pos[{t:2d}]: cosine={cos:.8f} mae={mae:.6e} {status}")

    mean_cos = float(np.mean(all_cosines))
    min_cos = float(np.min(all_cosines))
    print(f"\nMean cosine: {mean_cos:.6f}")
    print(f"Min cosine:  {min_cos:.6f}")

    if mean_cos > 0.99:
        print("PASS: blk.0 C prefill matches HF reference")
        return 0
    else:
        print("FAIL: blk.0 output does not match HF reference")
        return 1


if __name__ == "__main__":
    sys.exit(main())
