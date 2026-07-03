#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_HF = "/mnt/e/tensors/Qwen3.6-35B-A3B"
DEFAULT_GGUF = (
    "/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/"
    "snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/"
    "Qwen3.6-35B-A3B-Q8_0.gguf"
)
DEFAULT_FIXTURE_DIR = Path(
    "/mnt/f/git/ds4/.cache/qwen36_live_contracts/Qwen3.6-35B-A3B-Q8_0/shared"
)


def read_prompt(args) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def run_and_stream(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, text=True)
    return proc.returncode


def maybe_existing(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    return str(p) if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description="One-shot blk0 prefill validator entrypoint")
    ap.add_argument("--hf", default=DEFAULT_HF)
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--fixture-dir", default=str(DEFAULT_FIXTURE_DIR))
    ap.add_argument("--owned-power2-validator", default="/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_power2_owned_session_validator.py")
    ap.add_argument("--full-layer-isolation", default="/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_full_layer_prefill_isolation.py")
    ap.add_argument("--rows-validator", default="/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_blk0_prefill_rows_validator.py")
    ap.add_argument("--owned-session-worker-bin", default="/mnt/f/git/ds4/qwen36-unified-owned-worker-rocm")
    ap.add_argument("--full-layer-worker-bin", default="/mnt/f/git/ds4/qwen36-gpu-full-layer-worker")
    ap.add_argument("--hybrid-worker-bin", default="/mnt/f/git/ds4/qwen36-unified-owned-worker-rocm")
    ap.add_argument("--full-layers", default="3,7,11,15")
    ap.add_argument("--depths", default="8,16,32")
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--prefill-log-a")
    ap.add_argument("--prefill-log-b")
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    prompt = read_prompt(args)
    fixture_dir = Path(args.fixture_dir)
    if not fixture_dir.exists():
        raise RuntimeError(f"missing fixture dir: {fixture_dir}")

    with tempfile.TemporaryDirectory(prefix="q36_blk0_one_shot_") as td:
        td_path = Path(td)
        full_layer_json = td_path / "full_layer_isolation.json"
        power2_json = td_path / "power2.json"
        full_layer_cmd = [
            "python3",
            args.full_layer_isolation,
            "--hf", args.hf,
            "--gguf", args.gguf,
            "--worker-bin", args.full_layer_worker_bin,
            "--layers", args.full_layers,
            "--json-out", str(full_layer_json),
        ]
        if args.prompt_file:
            full_layer_cmd.extend(["--prompt-file", args.prompt_file])
        else:
            full_layer_cmd.extend(["--prompt", prompt])

        print("[one-shot] full-layer prefill isolation")
        rc = run_and_stream(full_layer_cmd)
        if rc != 0:
            print(f"[one-shot] full-layer prefill isolation failed rc={rc}")
            return rc

        cmd = [
            "python3",
            args.owned_power2_validator,
            "--hf", args.hf,
            "--gguf", args.gguf,
            "--owned-session-worker-bin", args.owned_session_worker_bin,
            "--full-layer-worker-bin", args.full_layer_worker_bin,
            "--hybrid-worker-bin", args.hybrid_worker_bin,
            "--fixture-dir", str(fixture_dir),
            "--depths", args.depths,
            "--top-k", str(args.top_k),
            "--json-out", str(power2_json),
        ]
        if args.prompt_file:
            cmd.extend(["--prompt-file", args.prompt_file])
        else:
            cmd.extend(["--prompt", prompt])

        print("[one-shot] power-of-2 owned-layer sweep")
        rc = run_and_stream(cmd)
        if rc != 0:
            print(f"[one-shot] power-of-2 owned-layer sweep failed rc={rc}")
            return rc

        rows_json = None
        log_a = maybe_existing(args.prefill_log_a)
        log_b = maybe_existing(args.prefill_log_b)
        if log_a:
            rows_json = td_path / "rows.json"
            rows_cmd = [
                "python3",
                args.rows_validator,
                "--log-a", log_a,
                "--label-a", args.label_a,
                "--json-out", str(rows_json),
            ]
            if log_b:
                rows_cmd.extend(["--log-b", log_b, "--label-b", args.label_b])
            print("[one-shot] prefill row-state summary")
            rc = run_and_stream(rows_cmd)
            if rc != 0:
                return rc

        if args.json_out:
            out = {
                "hf": args.hf,
                "gguf": args.gguf,
                "fixture_dir": str(fixture_dir),
                "prompt": prompt,
                "full_layer_isolation_json": json.loads(full_layer_json.read_text(encoding="utf-8")),
                "power2_json": json.loads(power2_json.read_text(encoding="utf-8")),
                "rows_json": None if rows_json is None else json.loads(rows_json.read_text(encoding="utf-8")),
            }
            Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"[one-shot] json_out={args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
