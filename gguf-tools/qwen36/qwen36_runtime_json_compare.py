#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="Compare two qwen36_behavior_oracle JSON runs, typically Q8 runtime vs Q4XL runtime."
    )
    ap.add_argument("--json-a", required=True, help="Reference behavior-oracle JSON")
    ap.add_argument("--json-b", required=True, help="Comparison behavior-oracle JSON")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--json-out")
    return ap.parse_args()


def load_run(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "generated_token_ids" not in data or "steps" not in data:
        raise RuntimeError(f"{path} is not a qwen36_behavior_oracle JSON")
    return data


def levenshtein(a: list[int], b: list[int]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, xa in enumerate(a, start=1):
        curr = [i]
        for j, xb in enumerate(b, start=1):
            cost = 0 if xa == xb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def mean_or_none(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def timing_summary(run: dict) -> dict:
    steps = run.get("steps", [])
    prefill_ms = None
    decode_ms = []
    for idx, step in enumerate(steps):
        if idx == 0 and "owned_prefill_ms" in step:
            prefill_ms = float(step["owned_prefill_ms"])
        if "owned_step_ms" in step:
            decode_ms.append(float(step["owned_step_ms"]))
    if not decode_ms:
        return {
            "prefill_ms": prefill_ms,
            "decode_avg_ms": None,
            "decode_count": 0,
            "decode_stage_avgs_ms": [],
        }
    n = len(decode_ms)
    stage_avgs = []
    for start in range(0, 3):
        lo = (n * start) // 3
        hi = (n * (start + 1)) // 3
        chunk = decode_ms[lo:hi]
        if chunk:
            stage_avgs.append(mean_or_none(chunk))
    return {
        "prefill_ms": prefill_ms,
        "decode_avg_ms": mean_or_none(decode_ms),
        "decode_count": len(decode_ms),
        "decode_stage_avgs_ms": stage_avgs,
    }


def compare_runs(a: dict, b: dict) -> dict:
    ids_a = a["generated_token_ids"]
    ids_b = b["generated_token_ids"]
    common = 0
    for xa, xb in zip(ids_a, ids_b):
        if xa != xb:
            break
        common += 1
    matching = sum(1 for xa, xb in zip(ids_a, ids_b) if xa == xb)
    denom = max(len(ids_a), len(ids_b), 1)
    first_mismatch = None if ids_a == ids_b else common
    step_deltas = []
    for idx, (sa, sb) in enumerate(zip(a["steps"], b["steps"])):
        if sa["next_id"] == sb["next_id"] and idx >= 8:
            continue
        step_deltas.append(
            {
                "step": idx,
                "same": sa["next_id"] == sb["next_id"],
                "id_a": sa["next_id"],
                "id_b": sb["next_id"],
                "text_a": sa["next_text"],
                "text_b": sb["next_text"],
            }
        )
    return {
        "prompt_tokens_a": a.get("prompt_tokens"),
        "prompt_tokens_b": b.get("prompt_tokens"),
        "generated_count_a": len(ids_a),
        "generated_count_b": len(ids_b),
        "same_token_prefix": common,
        "first_token_mismatch_step": first_mismatch,
        "matching_steps": matching,
        "token_agreement_rate": matching / denom,
        "levenshtein_distance": levenshtein(ids_a, ids_b),
        "timing_a": timing_summary(a),
        "timing_b": timing_summary(b),
        "sample_step_deltas": step_deltas[:16],
        "generated_text_a": a.get("generated_text", ""),
        "generated_text_b": b.get("generated_text", ""),
    }


def fmt_ms(value):
    return "n/a" if value is None else f"{value:.2f}"


def print_report(result: dict, label_a: str, label_b: str) -> None:
    print("tokens:")
    print(f"  prompt_tokens: {label_a}={result['prompt_tokens_a']} {label_b}={result['prompt_tokens_b']}")
    print(f"  generated: {label_a}={result['generated_count_a']} {label_b}={result['generated_count_b']}")
    print(f"  same_token_prefix: {result['same_token_prefix']}")
    print(f"  first_token_mismatch_step: {result['first_token_mismatch_step']}")
    print(f"  token_agreement_rate: {result['token_agreement_rate']:.3f}")
    print(f"  levenshtein_distance: {result['levenshtein_distance']}")
    print("timing:")
    print(
        f"  prefill_ms: {label_a}={fmt_ms(result['timing_a']['prefill_ms'])} "
        f"{label_b}={fmt_ms(result['timing_b']['prefill_ms'])}"
    )
    print(
        f"  decode_avg_ms: {label_a}={fmt_ms(result['timing_a']['decode_avg_ms'])} "
        f"{label_b}={fmt_ms(result['timing_b']['decode_avg_ms'])}"
    )
    if result["timing_a"]["decode_stage_avgs_ms"] or result["timing_b"]["decode_stage_avgs_ms"]:
        print("  decode_stage_avgs_ms:")
        for idx in range(max(len(result["timing_a"]["decode_stage_avgs_ms"]), len(result["timing_b"]["decode_stage_avgs_ms"]))):
            a_val = result["timing_a"]["decode_stage_avgs_ms"][idx] if idx < len(result["timing_a"]["decode_stage_avgs_ms"]) else None
            b_val = result["timing_b"]["decode_stage_avgs_ms"][idx] if idx < len(result["timing_b"]["decode_stage_avgs_ms"]) else None
            print(f"    stage{idx + 1}: {label_a}={fmt_ms(a_val)} {label_b}={fmt_ms(b_val)}")
    print("sample_step_deltas:")
    if not result["sample_step_deltas"]:
        print("  none")
    else:
        for item in result["sample_step_deltas"]:
            print(
                f"  step {item['step']}: same={item['same']} "
                f"{label_a}={item['id_a']}:{json.dumps(item['text_a'])} "
                f"{label_b}={item['id_b']}:{json.dumps(item['text_b'])}"
            )


def main() -> int:
    args = parse_args()
    run_a = load_run(args.json_a)
    run_b = load_run(args.json_b)
    result = compare_runs(run_a, run_b)
    print_report(result, args.label_a, args.label_b)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
