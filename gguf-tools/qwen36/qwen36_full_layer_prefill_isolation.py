#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_HF = "/mnt/e/tensors/Qwen3.6-35B-A3B"
DEFAULT_GGUF = (
    "/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/"
    "snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/"
    "Qwen3.6-35B-A3B-Q8_0.gguf"
)
DEFAULT_WORKER = "/mnt/f/git/ds4/qwen36-gpu-full-layer-worker"
DEFAULT_LAYERS = "3,7,11,15"
HIDDEN = 2048


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


def row_rmse_summary(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        raise RuntimeError(f"shape mismatch: {a.shape} vs {b.shape}")
    rows = []
    for row_idx in range(a.shape[0]):
        rows.append({"row": row_idx, "rmse": vec_rmse(a[row_idx], b[row_idx])})
    worst = max(rows, key=lambda item: item["rmse"])
    return {
        "seq_rmse": vec_rmse(a, b),
        "seq_cosine": vec_cosine(a, b),
        "last_row_rmse": rows[-1]["rmse"],
        "worst_row": worst,
        "row_rmses": rows,
    }


def read_exact(stream, n_bytes: int) -> bytes:
    chunks = []
    remaining = n_bytes
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise RuntimeError(f"unexpected EOF; wanted {remaining} more bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def run_worker_prefill(worker_bin: str, gguf: str, layer: int, input_seq: np.ndarray) -> np.ndarray:
    seq_len, hidden = input_seq.shape
    if hidden != HIDDEN:
        raise RuntimeError(f"unexpected hidden size: {hidden}")
    proc = subprocess.Popen(
        [worker_bin, gguf, "--layer", str(layer)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    try:
        ready = proc.stdout.readline().decode("utf-8", errors="replace").strip()
        if ready != "READY":
            err = proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"worker failed to start for layer {layer}: {ready} stderr={err}")
        proc.stdin.write(f"PREFILL_SEQ_BIN {seq_len} {hidden}\n".encode("utf-8"))
        proc.stdin.write(np.asarray(input_seq, dtype=np.float32).tobytes(order="C"))
        proc.stdin.flush()
        ack = proc.stdout.readline().decode("utf-8", errors="replace").strip()
        if not ack.startswith("PREFILL_OK "):
            err = proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"worker prefill failed for layer {layer}: {ack} stderr={err}")
        proc.stdin.write(b"DUMP_HIDDEN\n")
        proc.stdin.flush()
        hdr = proc.stdout.readline().decode("utf-8", errors="replace").strip()
        if not hdr.startswith("HIDDEN "):
            err = proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"worker dump failed for layer {layer}: {hdr} stderr={err}")
        _tag, n_floats_s, n_bytes_s = hdr.split()
        n_floats = int(n_floats_s)
        n_bytes = int(n_bytes_s)
        raw = read_exact(proc.stdout, n_bytes)
        arr = np.frombuffer(raw, dtype=np.float32, count=n_floats)
        expected = seq_len * hidden
        if arr.size != expected:
            raise RuntimeError(f"unexpected hidden dump size for layer {layer}: got {arr.size}, expected {expected}")
        proc.stdin.write(b"QUIT\n")
        proc.stdin.flush()
        _ = proc.stdout.readline()
        proc.wait()
        return arr.reshape(seq_len, hidden)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def capture_full_layer_io(model, token_ids: list[int], layers: list[int]) -> dict[int, dict[str, np.ndarray]]:
    wanted = set(layers)
    captured: dict[int, dict[str, np.ndarray]] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_mod, inp, out):
            captured[layer_idx] = {
                "input": inp[0].detach().float().cpu().numpy()[0],
                "output": (out[0] if isinstance(out, tuple) else out).detach().float().cpu().numpy()[0],
            }
        return hook

    for layer_idx in wanted:
        handles.append(model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
    try:
        with torch.inference_mode():
            _ = model(input_ids=torch.tensor([token_ids], dtype=torch.long))
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(wanted - set(captured))
    if missing:
        raise RuntimeError(f"failed to capture layers: {missing}")
    return captured


def main() -> int:
    ap = argparse.ArgumentParser(description="Isolate Qwen3.6 GPU full-layer prefill against HF layer IO.")
    ap.add_argument("--hf", default=DEFAULT_HF)
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--worker-bin", default=DEFAULT_WORKER)
    ap.add_argument("--layers", default=DEFAULT_LAYERS)
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    layers = [int(part.strip()) for part in args.layers.split(",") if part.strip()]
    prompt = read_prompt(args)

    print(f"[full-prefill] loading tokenizer from {args.hf}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    print(f"[full-prefill] loading model from {args.hf}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    print(f"[full-prefill] prompt_tokens={len(token_ids)} layers={layers}")

    captured = capture_full_layer_io(model, token_ids, layers)

    results = []
    for layer in layers:
        print(f"[full-prefill] layer={layer} capture_ready", flush=True)
        hf_input = captured[layer]["input"]
        hf_output = captured[layer]["output"]
        print(f"[full-prefill] layer={layer} worker_prefill_start", flush=True)
        worker_output = run_worker_prefill(args.worker_bin, args.gguf, layer, hf_input)
        summary = row_rmse_summary(worker_output, hf_output)
        result = {
            "layer": layer,
            "seq_rmse": summary["seq_rmse"],
            "seq_cosine": summary["seq_cosine"],
            "last_row_rmse": summary["last_row_rmse"],
            "worst_row": summary["worst_row"],
            "row_rmses": summary["row_rmses"],
        }
        results.append(result)
        print(
            f"[full-prefill] layer={layer} seq_rmse={result['seq_rmse']:.8f} "
            f"last_row_rmse={result['last_row_rmse']:.8f} "
            f"worst_row={result['worst_row']['row']}:{result['worst_row']['rmse']:.8f}"
        )

    out = {
        "prompt": prompt,
        "prompt_tokens": len(token_ids),
        "layers": layers,
        "results": results,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[full-prefill] json_out={args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
