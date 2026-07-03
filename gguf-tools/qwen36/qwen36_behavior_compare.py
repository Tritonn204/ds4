#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare isolated baseline/patched decode JSON outputs")
    ap.add_argument("--baseline-json", required=True)
    ap.add_argument("--patched-json", required=True)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    baseline = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
    patched = json.loads(Path(args.patched_json).read_text(encoding="utf-8"))
    base_ids = baseline["generated_token_ids"]
    patched_ids = patched["generated_token_ids"]
    compare_len = min(len(base_ids), len(patched_ids))
    first_mismatch = None
    for i in range(compare_len):
        if base_ids[i] != patched_ids[i]:
            first_mismatch = i
            break
    out = {
        "baseline_json": args.baseline_json,
        "patched_json": args.patched_json,
        "baseline_generated_text": baseline["generated_text"],
        "patched_generated_text": patched["generated_text"],
        "all_equal": first_mismatch is None and len(base_ids) == len(patched_ids),
        "first_mismatch_step": first_mismatch,
        "compared_steps": compare_len,
        "baseline_generated_tokens": len(base_ids),
        "patched_generated_tokens": len(patched_ids),
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")
    else:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
