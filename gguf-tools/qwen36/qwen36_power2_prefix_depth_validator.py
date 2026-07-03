#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def read_prompt(args) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def power2_depths(n: int) -> list[int]:
    out = []
    d = 1
    while d <= n:
        out.append(d)
        d *= 2
    if out[-1] != n:
        out.append(n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Run power-of-2 prefix-depth validation using the existing GPU prefix-depth checker")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--gpu-prefix-bin", default="/mnt/f/git/ds4/qwen36-gpu-prefix-q8-chain-dynamic")
    ap.add_argument("--splice-bin", default="/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_splice_hidden_check.py")
    ap.add_argument("--checker", default="/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_gpu_prefix_depth_check.py")
    ap.add_argument("--fixture", action="append", required=True)
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    depths = power2_depths(len(args.fixture))
    prompt = read_prompt(args)

    with tempfile.TemporaryDirectory(prefix="q36_power2_depth_") as td:
        json_path = Path(td) / "depth.json"
        cmd = [
            "python3",
            args.checker,
            "--hf", args.hf,
            "--gguf", args.gguf,
            "--gpu-prefix-bin", args.gpu_prefix_bin,
            "--splice-bin", args.splice_bin,
            "--depths", ",".join(str(d) for d in depths),
            "--top-k", str(args.top_k),
            "--json-out", str(json_path),
        ]
        for fixture in args.fixture:
            cmd.extend(["--fixture", fixture])
        if args.prompt_file:
            cmd.extend(["--prompt-file", args.prompt_file])
        else:
            cmd.extend(["--prompt", prompt])

        print(f"[power2-depth] running {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            return proc.returncode

        data = json.loads(json_path.read_text(encoding="utf-8"))

    okay = [item for item in data["results"] if item.get("ok")]
    okay.sort(key=lambda item: item.get("depth", 0))
    failed = [item for item in data["results"] if not item.get("ok")]
    print("[power2-depth] summary")
    for item in okay:
        print(
            f"  depth={item['depth']} argmax_equal={item.get('argmax_equal')} "
            f"logits_cosine={item.get('logits_cosine')} logits_rmse={item.get('logits_rmse')}"
        )
    if failed:
        print("[power2-depth] failures")
        for item in failed:
            print(
                f"  depth={item.get('depth')} stage={item.get('stage')} "
                f"returncode={item.get('returncode')}"
            )
            stdout = item.get("gpu_stdout") or item.get("stdout") or ""
            stderr = item.get("gpu_stderr") or item.get("stderr") or ""
            if stdout:
                first = str(stdout).strip().splitlines()[0]
                print(f"    stdout: {first}")
            if stderr:
                first = str(stderr).strip().splitlines()[0]
                print(f"    stderr: {first}")

    best = None
    if okay:
        best = max(
            okay,
            key=lambda item: (
                bool(item.get("argmax_equal")),
                float(item.get("logits_cosine", float("-inf"))),
            ),
        )
        print(
            f"[power2-depth] best depth={best['depth']} "
            f"argmax_equal={best.get('argmax_equal')} "
            f"logits_cosine={best.get('logits_cosine')} "
            f"logits_rmse={best.get('logits_rmse')}"
        )

    out = {
        "prompt": prompt,
        "depths": depths,
        "results": data["results"],
        "best": best,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[power2-depth] json_out={args.json_out}")
    return 0 if okay else 1


if __name__ == "__main__":
    raise SystemExit(main())
