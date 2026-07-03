#!/usr/bin/env python3
import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


PREFIX_MAGIC = b"Q36PFX01"


def read_prompt(args) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def write_prefix_fixture(path: Path, token_ids: list[int], hidden: int, input_seq: np.ndarray) -> None:
    tok = np.asarray(token_ids, dtype=np.uint32)
    seq = np.asarray(input_seq, dtype=np.float32)
    if seq.shape != (len(token_ids), hidden):
        raise RuntimeError(f"unexpected prefix input_seq shape: got {seq.shape}, expected {(len(token_ids), hidden)}")
    with path.open("wb") as fp:
        fp.write(PREFIX_MAGIC)
        fp.write(struct.pack("<II", len(token_ids), hidden))
        fp.write(tok.tobytes(order="C"))
        fp.write(seq.tobytes(order="C"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one GPU hybrid layer on exact HF input and splice the result back into the HF tail")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--gpu-bin", default="/mnt/f/git/ds4/qwen36-gpu-hybrid-layer-q8-dynamic")
    ap.add_argument("--splice-bin", default="/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_splice_hidden_check.py")
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--layer-trace", required=True, help="Layer trace prefix or .npz from qwen36_layer_trace_export.py")
    ap.add_argument("--layer", required=True, type=int)
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    prompt = read_prompt(args)
    trace_path = Path(args.layer_trace)
    if trace_path.suffix != ".npz":
        trace_path = trace_path.with_suffix(".npz")
    bundle = np.load(trace_path)
    key = f"blk_{args.layer}.layer_input_seq"
    if key not in bundle:
        raise RuntimeError(f"missing {key} in {trace_path}")
    input_seq = np.asarray(bundle[key], dtype=np.float32)
    seq_len, hidden = input_seq.shape

    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    if len(token_ids) != seq_len:
        raise RuntimeError(f"prompt token count mismatch: tokenizer={len(token_ids)} trace={seq_len}")

    with tempfile.TemporaryDirectory(prefix="q36_gpu_exact_") as td:
        td_path = Path(td)
        prefix_path = td_path / "prefix.bin"
        owned_seq_path = td_path / "owned_seq.f32"
        write_prefix_fixture(prefix_path, token_ids, hidden, input_seq)

        gpu_cmd = [
            args.gpu_bin,
            args.gguf,
            args.fixture,
            "--prefix", str(prefix_path),
            "--use-prefix-input-seq",
            "--dump-seq", str(owned_seq_path),
        ]
        print(f"[gpu-exact] running {' '.join(gpu_cmd)}")
        gpu_proc = subprocess.run(gpu_cmd, capture_output=True, text=True)
        if gpu_proc.returncode != 0:
            raise RuntimeError(f"gpu layer run failed:\nSTDOUT:\n{gpu_proc.stdout}\nSTDERR:\n{gpu_proc.stderr}")

        splice_cmd = [
            "python3",
            args.splice_bin,
            "--hf", args.hf,
            "--hidden-f32", str(owned_seq_path),
            "--splice-layer", str(args.layer),
            "--mode", "seq",
            "--top-k", str(args.top_k),
        ]
        if args.prompt_file:
            splice_cmd.extend(["--prompt-file", args.prompt_file])
        else:
            splice_cmd.extend(["--prompt", prompt])
        print(f"[gpu-exact] running {' '.join(splice_cmd)}")
        splice_proc = subprocess.run(splice_cmd, capture_output=True, text=True)
        if splice_proc.returncode != 0:
            raise RuntimeError(f"splice check failed:\nSTDOUT:\n{splice_proc.stdout}\nSTDERR:\n{splice_proc.stderr}")

        out = {
            "prompt": prompt,
            "layer": args.layer,
            "fixture": args.fixture,
            "layer_trace": str(trace_path),
            "gpu_stdout": gpu_proc.stdout,
            "splice_stdout": splice_proc.stdout,
        }
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"json_out: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
