#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


ROUTED_ORACLE = "/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_q4xl_routed_oracle.py"
FIXTURE_ORACLE = "/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_q4xl_fixture_study.py"
DEFAULT_OUT_DIR = "/mnt/e/tensors/qwen36_q4xl_fixture_study"


def run_cmd(cmd: list[str]) -> None:
    print(f"[q4xl-oracle] cmd={' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def routed_cmd(args, json_out: str | None) -> list[str]:
    cmd = [
        sys.executable,
        ROUTED_ORACLE,
        "--q8-gguf", args.q8_gguf,
        "--q4-gguf", args.q4_gguf,
    ]
    for layer in args.layer:
        cmd += ["--layer", str(layer)]
    for expert in args.expert:
        cmd += ["--expert", str(expert)]
    for suffix in args.suffix:
        cmd += ["--suffix", suffix]
    if json_out:
        cmd += ["--json-out", json_out]
    return cmd


def fixture_cmd(args, json_out: str | None) -> list[str]:
    cmd = [
        sys.executable,
        FIXTURE_ORACLE,
        "--q8-gguf", args.q8_gguf,
        "--q4-gguf", args.q4_gguf,
        "--out-dir", args.fixture_out_dir,
    ]
    if args.export_missing:
        cmd.append("--export-missing")
    if args.rebuild:
        cmd.append("--rebuild")
    if json_out:
        cmd += ["--json-out", json_out]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description="One-command Q4_K_XL vs Q8_0 oracle runner")
    ap.add_argument("--mode", choices=["routed", "fixtures", "both"], default="both")
    ap.add_argument("--q8-gguf",
                    default="/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-Q8_0.gguf")
    ap.add_argument("--q4-gguf",
                    default="/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
    ap.add_argument("--layer", action="append", type=int, default=[1, 38, 39],
                    help="Layer index for routed-oracle. May repeat.")
    ap.add_argument("--expert", action="append", type=int, default=[0, 7, 42, 255],
                    help="Expert index for routed-oracle. May repeat.")
    ap.add_argument("--suffix", action="append", default=[],
                    help="Optional tensor suffix override for routed-oracle")
    ap.add_argument("--fixture-out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--export-missing", action="store_true",
                    help="For fixture mode, export only missing cache files")
    ap.add_argument("--rebuild", action="store_true",
                    help="For fixture mode, rebuild all cached fixtures")
    ap.add_argument("--json-out",
                    help="If set, writes a small index JSON and per-check JSON siblings next to it")
    args = ap.parse_args()

    outputs = {"mode": args.mode}
    base = Path(args.json_out) if args.json_out else None

    if args.mode in {"routed", "both"}:
        routed_json = None
        if base is not None:
            routed_json = str(base.with_name(base.stem + "_routed.json"))
        run_cmd(routed_cmd(args, routed_json))
        outputs["routed_json"] = routed_json

    if args.mode in {"fixtures", "both"}:
        fixtures_json = None
        if base is not None:
            fixtures_json = str(base.with_name(base.stem + "_fixtures.json"))
        run_cmd(fixture_cmd(args, fixtures_json))
        outputs["fixtures_json"] = fixtures_json

    if base is not None:
        base.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
        print(f"[q4xl-oracle] json_out={base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
