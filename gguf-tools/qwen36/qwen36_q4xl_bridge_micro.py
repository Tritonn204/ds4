#!/usr/bin/env python3
import argparse
import shlex
import subprocess
from pathlib import Path


FULL_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]


def default_hf_path() -> str:
    return "/mnt/e/tensors/Qwen3.6-35B-A3B"


def default_q4xl_gguf() -> str:
    return (
        "/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/"
        "snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/"
        "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
    )


def default_prompt_file() -> str:
    return "/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_oracle_prompts/code_review.txt"


def build_fixture_list(shared_dir: Path) -> list[str]:
    full_set = set(FULL_LAYERS)
    fixtures: list[str] = []
    for blk in range(40):
        if blk in full_set:
            continue
        fixtures.append(str(shared_dir / f"blk{blk}.live.bin"))
    return fixtures


def shell_join(items: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in items)


def main() -> int:
    ap = argparse.ArgumentParser(description="One-step Q4XL owned-session bridge micro")
    ap.add_argument("--mode", choices=["micro", "decode", "cpu-gpu-oracle"], default="micro")
    ap.add_argument("--hf", default=default_hf_path())
    ap.add_argument("--gguf", default=default_q4xl_gguf())
    ap.add_argument("--prompt-file", default=default_prompt_file())
    ap.add_argument("--fixture-dir", default="/mnt/e/tensors/qwen36_q4xl_fixture_study/q4xl_shared")
    ap.add_argument("--worker-a", required=True)
    ap.add_argument("--worker-b")
    ap.add_argument("--full-layer-worker-bin", default="/mnt/f/git/ds4/qwen36-gpu-full-layer-worker")
    ap.add_argument("--c-bin", default="/mnt/f/git/ds4/qwen36-c-prefix-q8-chain-live")
    ap.add_argument("--c-bin-worker-bin", default="/mnt/f/git/ds4/qwen36-live-contract-worker")
    ap.add_argument("--hybrid-gpu-cycles", type=int, default=1)
    ap.add_argument("--full-mode", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--splice-layer", type=int, default=39)
    ap.add_argument("--forced-step-token", type=int)
    ap.add_argument("--compare-hf-reference", action="store_true")
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--n-steps", type=int, default=1)
    ap.add_argument("--dump-cycle-boundaries", action="store_true")
    ap.add_argument("--prefill-only", action="store_true")
    ap.add_argument("--owned-session-env", action="append", default=[])
    ap.add_argument("--tail-source", choices=["auto", "hf", "gguf", "compare"], default="gguf")
    ap.add_argument("--json-out", default="/tmp/qwen36_q4xl_bridge_micro.json")
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()

    fixture_dir = Path(args.fixture_dir)
    fixtures = build_fixture_list(fixture_dir)
    missing = [path for path in fixtures if not Path(path).is_file()]
    if missing:
        for path in missing:
            print(f"missing fixture: {path}")
        return 2

    if args.mode == "decode":
        cmd = [
            "python3",
            "-u",
            "/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_behavior_oracle.py",
            "--mode",
            "patched-session",
            "--hf",
            args.hf,
            "--gguf",
            args.gguf,
            "--owned-session-worker-bin",
            args.worker_a,
            "--owned-session-unified-full-gpu" if args.full_mode == "gpu" else "--owned-session-unified-full-cpu",
            "--owned-session-unified-hybrid-gpu-cycles",
            str(args.hybrid_gpu_cycles),
            "--c-bin",
            args.c_bin,
            "--c-bin-worker-bin",
            args.c_bin_worker_bin,
            "--full-layer-worker-bin",
            args.full_layer_worker_bin,
            "--prompt-file",
            args.prompt_file,
            "--splice-layer",
            str(args.splice_layer),
            "--n-predict",
            str(args.n_predict),
            "--top-k",
            str(args.top_k),
            "--tail-source",
            args.tail_source,
            "--json-out",
            args.json_out,
        ]
        for layer in FULL_LAYERS:
            cmd.extend(["--full-layer", str(layer)])
        for fixture in fixtures:
            cmd.extend(["--fixture", fixture])
        for item in args.owned_session_env:
            cmd.extend(["--owned-session-env", item])
        print(f"[q4xl-bridge-decode] cmd={shell_join(cmd)}")
        if args.print_only:
            return 0
        return subprocess.run(cmd, check=False).returncode

    if args.mode == "cpu-gpu-oracle":
        cmd = [
            "python3",
            "/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_owned_session_micro_ab.py",
            "--hf",
            args.hf,
            "--gguf",
            args.gguf,
            "--prompt-file",
            args.prompt_file,
            "--top-k",
            str(args.top_k),
            "--owned-session-worker-bin-a",
            args.worker_a,
            "--owned-session-worker-bin-b",
            args.worker_a,
            "--worker-a-full-mode",
            "cpu",
            "--worker-b-full-mode",
            "gpu",
            "--owned-session-unified-full-gpu",
            "--owned-session-unified-hybrid-gpu-cycles",
            str(args.hybrid_gpu_cycles),
            "--c-bin",
            args.c_bin,
            "--c-bin-worker-bin",
            args.c_bin_worker_bin,
            "--full-layer-worker-bin",
            args.full_layer_worker_bin,
            "--splice-layer",
            str(args.splice_layer),
            "--n-steps",
            str(args.n_steps),
            "--json-out",
            args.json_out,
        ]
        for layer in FULL_LAYERS:
            cmd.extend(["--full-layer", str(layer)])
        for fixture in fixtures:
            cmd.extend(["--fixture", fixture])
        if args.forced_step_token is not None:
            cmd.extend(["--forced-step-token", str(args.forced_step_token)])
        if args.compare_hf_reference:
            cmd.append("--compare-hf-reference")
        if args.dump_cycle_boundaries:
            cmd.append("--dump-cycle-boundaries")
        if args.prefill_only:
            cmd.append("--prefill-only")
        cmd.extend(["--tail-source", args.tail_source])
        for item in args.owned_session_env:
            cmd.extend(["--owned-session-env", item])
        print(f"[q4xl-cpu-gpu-oracle] cmd={shell_join(cmd)}")
        if args.print_only:
            return 0
        return subprocess.run(cmd, check=False).returncode

    if not args.worker_b:
        print("--worker-b is required in micro mode")
        return 2

    cmd = [
        "python3",
        "/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_owned_session_micro_ab.py",
        "--hf",
        args.hf,
        "--gguf",
        args.gguf,
        "--prompt-file",
        args.prompt_file,
        "--top-k",
        str(args.top_k),
        "--owned-session-worker-bin-a",
        args.worker_a,
        "--owned-session-worker-bin-b",
        args.worker_b,
        "--worker-a-full-mode",
        args.full_mode,
        "--worker-b-full-mode",
        args.full_mode,
        "--owned-session-unified-full-gpu" if args.full_mode == "gpu" else "--owned-session-unified-full-cpu",
        "--owned-session-unified-hybrid-gpu-cycles",
        str(args.hybrid_gpu_cycles),
        "--c-bin",
        args.c_bin,
        "--c-bin-worker-bin",
        args.c_bin_worker_bin,
        "--full-layer-worker-bin",
        args.full_layer_worker_bin,
        "--splice-layer",
        str(args.splice_layer),
        "--n-steps",
        str(args.n_steps),
        "--json-out",
        args.json_out,
    ]
    for layer in FULL_LAYERS:
        cmd.extend(["--full-layer", str(layer)])
    for fixture in fixtures:
        cmd.extend(["--fixture", fixture])
    if args.forced_step_token is not None:
        cmd.extend(["--forced-step-token", str(args.forced_step_token)])
    if args.compare_hf_reference:
        cmd.append("--compare-hf-reference")
    if args.dump_cycle_boundaries:
        cmd.append("--dump-cycle-boundaries")
    if args.prefill_only:
        cmd.append("--prefill-only")
    cmd.extend(["--tail-source", args.tail_source])
    for item in args.owned_session_env:
        cmd.extend(["--owned-session-env", item])

    print(f"[q4xl-bridge-micro] cmd={shell_join(cmd)}")
    if args.print_only:
        return 0
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
