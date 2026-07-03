#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


FULL_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]


def default_shared_fixture_dir() -> Path:
    return Path("/mnt/f/git/ds4/.cache/qwen36_live_contracts/Qwen3.6-35B-A3B-Q8_0/shared")


def default_prompt_file() -> Path:
    return Path("/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_oracle_prompts/code_review.txt")


def default_hf_path() -> Path:
    return Path("/mnt/e/tensors/Qwen3.6-35B-A3B")


def default_gguf_path() -> Path:
    return Path(
        "/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/"
        "snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-Q8_0.gguf"
    )


def build_fixture_list(shared_dir: Path, full_layers: list[int]) -> list[Path]:
    fixtures = []
    full_set = set(full_layers)
    for blk in range(40):
        if blk in full_set:
            continue
        fixture = shared_dir / f"blk{blk}.live.bin"
        fixtures.append(fixture)
    return fixtures


def shell_join(items: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in items)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stable wrapper for Qwen3.6 owned-session full-layer diagnostics."
    )
    ap.add_argument(
        "--probe",
        choices=["baseline", "boundary", "breadcrumb", "full-debug", "handoff-debug"],
        default="baseline",
        help="Diagnostic mode to run.",
    )
    ap.add_argument("--hf", default=str(default_hf_path()))
    ap.add_argument("--gguf", default=str(default_gguf_path()))
    ap.add_argument("--shared-fixture-dir", default=str(default_shared_fixture_dir()))
    ap.add_argument("--owned-session-worker-bin", default="/mnt/f/git/ds4/qwen36-unified-owned-worker-rocm")
    ap.add_argument("--c-bin", default="/mnt/f/git/ds4/qwen36-c-prefix-q8-chain-live")
    ap.add_argument("--c-bin-worker-bin", default="/mnt/f/git/ds4/qwen36-live-contract-worker")
    ap.add_argument("--full-layer-worker-bin", default="/mnt/f/git/ds4/qwen36-gpu-full-layer-worker")
    ap.add_argument("--prompt-file", default=str(default_prompt_file()))
    ap.add_argument("--json-out", default="/tmp/qwen36_full_layer_probe.json")
    ap.add_argument("--splice-layer", type=int, default=39)
    ap.add_argument("--n-predict", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--hybrid-gpu-cycles", type=int, default=1)
    ap.add_argument("--full-debug-layer", type=int, default=3)
    ap.add_argument("--step-start", type=int, default=0)
    ap.add_argument("--step-end", type=int, default=0)
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()

    shared_dir = Path(args.shared_fixture_dir)
    fixtures = build_fixture_list(shared_dir, FULL_LAYERS)
    missing = [str(p) for p in fixtures if not p.is_file()]
    if missing:
        print("missing fixtures:", file=sys.stderr)
        for path in missing:
            print(path, file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["QWEN36_UNIFIED_VERBOSE"] = "0"
    owned_session_env: dict[str, str] = {
        "QWEN36_UNIFIED_VERBOSE": "0",
    }

    if args.probe == "boundary":
        env["QWEN36_DBG_BOUNDARY"] = "1"
        env["QWEN36_DBG_HYBRID_STEP_START"] = str(args.step_start)
        env["QWEN36_DBG_HYBRID_STEP_END"] = str(args.step_end)
        if "QWEN36_DBG_HYBRID_LAYER" in os.environ:
            env["QWEN36_DBG_HYBRID_LAYER"] = os.environ["QWEN36_DBG_HYBRID_LAYER"]
        owned_session_env["QWEN36_DBG_BOUNDARY"] = "1"
        owned_session_env["QWEN36_DBG_HYBRID_STEP_START"] = str(args.step_start)
        owned_session_env["QWEN36_DBG_HYBRID_STEP_END"] = str(args.step_end)
        if "QWEN36_DBG_HYBRID_LAYER" in os.environ:
            owned_session_env["QWEN36_DBG_HYBRID_LAYER"] = os.environ["QWEN36_DBG_HYBRID_LAYER"]
    elif args.probe == "breadcrumb":
        env["QWEN36_DBG_FULL_BREADCRUMB"] = "1"
        env["QWEN36_DBG_FULL_STEP_START"] = str(args.step_start)
        env["QWEN36_DBG_FULL_STEP_END"] = str(args.step_end)
        owned_session_env["QWEN36_DBG_FULL_BREADCRUMB"] = "1"
        owned_session_env["QWEN36_DBG_FULL_STEP_START"] = str(args.step_start)
        owned_session_env["QWEN36_DBG_FULL_STEP_END"] = str(args.step_end)
    elif args.probe == "full-debug":
        env["QWEN36_DBG_FULL_LAYER"] = str(args.full_debug_layer)
        env["QWEN36_DBG_FULL_BREADCRUMB"] = "1"
        env["QWEN36_DBG_FULL_STEP_START"] = str(args.step_start)
        env["QWEN36_DBG_FULL_STEP_END"] = str(args.step_end)
        owned_session_env["QWEN36_DBG_FULL_LAYER"] = str(args.full_debug_layer)
        owned_session_env["QWEN36_DBG_FULL_BREADCRUMB"] = "1"
        owned_session_env["QWEN36_DBG_FULL_STEP_START"] = str(args.step_start)
        owned_session_env["QWEN36_DBG_FULL_STEP_END"] = str(args.step_end)
    elif args.probe == "handoff-debug":
        env["QWEN36_DBG_BOUNDARY"] = "1"
        env["QWEN36_DBG_HYBRID_STEP_START"] = str(args.step_start)
        env["QWEN36_DBG_HYBRID_STEP_END"] = str(args.step_end)
        if "QWEN36_DBG_HYBRID_LAYER" in os.environ:
            env["QWEN36_DBG_HYBRID_LAYER"] = os.environ["QWEN36_DBG_HYBRID_LAYER"]
        env["QWEN36_DBG_FULL_LAYER"] = str(args.full_debug_layer)
        env["QWEN36_DBG_FULL_BREADCRUMB"] = "1"
        env["QWEN36_DBG_FULL_STEP_START"] = str(args.step_start)
        env["QWEN36_DBG_FULL_STEP_END"] = str(args.step_end)
        owned_session_env["QWEN36_DBG_BOUNDARY"] = "1"
        owned_session_env["QWEN36_DBG_HYBRID_STEP_START"] = str(args.step_start)
        owned_session_env["QWEN36_DBG_HYBRID_STEP_END"] = str(args.step_end)
        if "QWEN36_DBG_HYBRID_LAYER" in os.environ:
            owned_session_env["QWEN36_DBG_HYBRID_LAYER"] = os.environ["QWEN36_DBG_HYBRID_LAYER"]
        owned_session_env["QWEN36_DBG_FULL_LAYER"] = str(args.full_debug_layer)
        owned_session_env["QWEN36_DBG_FULL_BREADCRUMB"] = "1"
        owned_session_env["QWEN36_DBG_FULL_STEP_START"] = str(args.step_start)
        owned_session_env["QWEN36_DBG_FULL_STEP_END"] = str(args.step_end)

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
        args.owned_session_worker_bin,
        "--owned-session-unified-full-gpu",
        "--owned-session-unified-hybrid-gpu-cycles",
        str(args.hybrid_gpu_cycles),
        "--c-bin",
        args.c_bin,
        "--c-bin-worker-bin",
        args.c_bin_worker_bin,
        "--full-layer-worker-bin",
        args.full_layer_worker_bin,
    ]

    for layer in FULL_LAYERS:
        cmd.extend(["--full-layer", str(layer)])
    for fixture in fixtures:
        cmd.extend(["--fixture", str(fixture)])
    for key, value in sorted(owned_session_env.items()):
        cmd.extend(["--owned-session-env", f"{key}={value}"])

    cmd.extend(
        [
            "--prompt-file",
            args.prompt_file,
            "--splice-layer",
            str(args.splice_layer),
            "--n-predict",
            str(args.n_predict),
            "--top-k",
            str(args.top_k),
            "--json-out",
            args.json_out,
        ]
    )

    print(f"[probe-driver] mode={args.probe}")
    for key in sorted(
        k
        for k in env
        if k.startswith("QWEN36_DBG_") or k == "QWEN36_UNIFIED_VERBOSE"
    ):
        print(f"[probe-driver] env {key}={env[key]}")
    print(f"[probe-driver] cmd={shell_join(cmd)}")

    if args.print_only:
        return 0

    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
