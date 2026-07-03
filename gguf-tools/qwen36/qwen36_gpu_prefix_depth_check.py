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


def write_prefix_fixture(path: Path, token_ids: list[int], hidden: int, input_seq: np.ndarray | None = None) -> None:
    tok = np.asarray(token_ids, dtype=np.uint32)
    if input_seq is None:
        seq = np.zeros((len(token_ids), hidden), dtype=np.float32)
    else:
        seq = np.asarray(input_seq, dtype=np.float32)
        if seq.shape != (len(token_ids), hidden):
            raise RuntimeError(f"unexpected prefix input_seq shape: got {seq.shape}, expected {(len(token_ids), hidden)}")
    with path.open("wb") as fp:
        fp.write(PREFIX_MAGIC)
        fp.write(struct.pack("<II", len(token_ids), hidden))
        fp.write(tok.tobytes(order="C"))
        fp.write(seq.tobytes(order="C"))


def parse_depths(text: str | None, n_fixtures: int) -> list[int]:
    if not text:
        return list(range(1, n_fixtures + 1))
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        depth = int(part)
        if depth < 1 or depth > n_fixtures:
            raise RuntimeError(f"depth {depth} out of range 1..{n_fixtures}")
        out.append(depth)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep GPU-owned Qwen hybrid prefix depths and splice each result into the HF tail")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--gpu-prefix-bin", default="/mnt/f/git/ds4/qwen36-gpu-prefix-q8-chain-dynamic")
    ap.add_argument("--splice-bin", default="/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_splice_hidden_check.py")
    ap.add_argument("--fixture", action="append", required=True, help="Hybrid layer fixtures in order, e.g. blk0 blk1 blk2")
    ap.add_argument("--depths", help="Comma list of prefix depths to test; default: 1..N")
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    prompt = read_prompt(args)
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    hidden = 2048
    depths = parse_depths(args.depths, len(args.fixture))

    results = []
    with tempfile.TemporaryDirectory(prefix="q36_gpu_depth_") as td:
        td_path = Path(td)
        prefix_path = td_path / "prefix.bin"
        write_prefix_fixture(prefix_path, token_ids, hidden)

        for depth in depths:
            owned_seq_path = td_path / f"owned_depth{depth}.f32"
            c_cmd = [
                args.gpu_prefix_bin,
                args.gguf,
                *args.fixture[:depth],
                "--prefix", str(prefix_path),
                "--dump-seq", str(owned_seq_path),
            ]
            print(f"[gpu-depth] depth={depth} running {' '.join(c_cmd)}")
            c_proc = subprocess.run(c_cmd, capture_output=True, text=True)
            if c_proc.returncode != 0:
                result = {
                    "depth": depth,
                    "fixtures": args.fixture[:depth],
                    "ok": False,
                    "stage": "gpu_prefix",
                    "returncode": c_proc.returncode,
                    "stdout": c_proc.stdout,
                    "stderr": c_proc.stderr,
                }
                results.append(result)
                print(f"[gpu-depth] depth={depth} gpu_prefix_failed returncode={c_proc.returncode}")
                continue

            splice_layer = depth - 1
            splice_cmd = [
                "python3",
                args.splice_bin,
                "--hf", args.hf,
                "--hidden-f32", str(owned_seq_path),
                "--splice-layer", str(splice_layer),
                "--mode", "seq",
                "--top-k", str(args.top_k),
            ]
            if args.prompt_file:
                splice_cmd.extend(["--prompt-file", args.prompt_file])
            else:
                splice_cmd.extend(["--prompt", prompt])
            print(f"[gpu-depth] depth={depth} running {' '.join(splice_cmd)}")
            splice_proc = subprocess.run(splice_cmd, capture_output=True, text=True)
            if splice_proc.returncode != 0:
                result = {
                    "depth": depth,
                    "fixtures": args.fixture[:depth],
                    "ok": False,
                    "stage": "splice",
                    "returncode": splice_proc.returncode,
                    "gpu_stdout": c_proc.stdout,
                    "gpu_stderr": c_proc.stderr,
                    "splice_stdout": splice_proc.stdout,
                    "splice_stderr": splice_proc.stderr,
                }
                results.append(result)
                print(f"[gpu-depth] depth={depth} splice_failed returncode={splice_proc.returncode}")
                continue

            parsed = {
                "depth": depth,
                "fixtures": args.fixture[:depth],
                "ok": True,
                "gpu_stdout": c_proc.stdout,
                "splice_stdout": splice_proc.stdout,
            }
            for line in splice_proc.stdout.splitlines():
                if line.startswith("logits_cosine:"):
                    parsed["logits_cosine"] = float(line.split(":", 1)[1].strip())
                elif line.startswith("logits_rmse:"):
                    parsed["logits_rmse"] = float(line.split(":", 1)[1].strip())
                elif line.startswith("argmax_equal:"):
                    parsed["argmax_equal"] = line.split(":", 1)[1].strip() == "True"
                elif line.startswith("base_argmax:"):
                    parsed["base_argmax_line"] = line
                elif line.startswith("patched_argmax:"):
                    parsed["patched_argmax_line"] = line
            results.append(parsed)
            print(
                f"[gpu-depth] depth={depth} argmax_equal={parsed.get('argmax_equal')} "
                f"logits_cosine={parsed.get('logits_cosine')}"
            )

    out = {
        "prompt": prompt,
        "prompt_tokens": len(token_ids),
        "results": results,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
