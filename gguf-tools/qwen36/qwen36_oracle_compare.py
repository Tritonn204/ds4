#!/usr/bin/env python3

import argparse
import json
import math
import os
import subprocess
import sys


DEFAULT_Q8 = "/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-Q8_0.gguf"
DEFAULT_Q4XL = "/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
DEFAULT_ORACLE = "/mnt/f/git/ds4/qwen36-llama-oracle"
DEFAULT_LLAMA_LIBDIR = "/mnt/f/git/llama.cpp/build/bin"
DEFAULT_PROMPT = (
    "Write a compact explanation of why role-specific quantization can preserve "
    "quality better than uniform low-bit quantization in a MoE model."
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Compare two Qwen3.6 GGUFs with a llama.cpp-backed token/logit oracle."
    )
    ap.add_argument("--oracle-bin", default=DEFAULT_ORACLE, help="Path to qwen36-llama-oracle")
    ap.add_argument("--llama-libdir", default=DEFAULT_LLAMA_LIBDIR, help="Directory containing libllama.so")
    ap.add_argument("--model-a", default=DEFAULT_Q8, help="Reference model path")
    ap.add_argument("--model-b", default=DEFAULT_Q4XL, help="Comparison model path")
    ap.add_argument("--label-a", default="Q8_0", help="Reference model label")
    ap.add_argument("--label-b", default="Q4_K_XL", help="Comparison model label")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="Raw prompt text")
    ap.add_argument("--prompt-file", help="Read prompt text from file")
    ap.add_argument("--ctx-size", type=int, default=4096, help="Context size")
    ap.add_argument("--n-predict", type=int, default=24, help="Greedy decode length")
    ap.add_argument("--top-k", type=int, default=10, help="Top logits to keep per step")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads")
    ap.add_argument("--json-out", help="Optional path to write the full comparison JSON")
    return ap.parse_args()


def read_prompt(args):
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as fp:
            return fp.read()
    return args.prompt


def bytes_to_text(items):
    buf = bytearray()
    for item in items:
        buf.extend(item.get("bytes", []))
    return buf.decode("utf-8", errors="replace")


def run_oracle(args, model_path, prompt, force_token_ids=None):
    env = os.environ.copy()
    libpath = args.llama_libdir
    old = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = libpath if not old else f"{libpath}:{old}"
    cmd = [
        args.oracle_bin,
        "--model", model_path,
        "--prompt", prompt,
        "--n-predict", str(args.n_predict),
        "--top-k", str(args.top_k),
        "--ctx-size", str(args.ctx_size),
        "--threads", str(args.threads),
    ]
    if force_token_ids:
        cmd += ["--force-tokens-csv", ",".join(str(x) for x in force_token_ids)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"oracle failed for {model_path}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        stdout = proc.stdout[:4000]
        stderr = proc.stderr[:4000]
        raise RuntimeError(
            f"oracle returned non-JSON for {model_path}\n"
            f"stdout_prefix:\n{stdout}\n"
            f"stderr_prefix:\n{stderr}"
        ) from exc


def compare_token_ids(ids_a, ids_b):
    common = 0
    for ta, tb in zip(ids_a, ids_b):
        if ta != tb:
            break
        common += 1
    return {
        "equal": ids_a == ids_b,
        "common_prefix": common,
        "count_a": len(ids_a),
        "count_b": len(ids_b),
        "first_mismatch_index": None if ids_a == ids_b else common,
    }


def levenshtein(a, b):
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


def summarize_run(data):
    prompt_tokens = data.get("prompt_tokens", [])
    generated_tokens = data.get("generated_tokens", [])
    steps = data.get("steps", [])
    return {
        "prompt_token_ids": [tok["id"] for tok in prompt_tokens],
        "prompt_text": bytes_to_text(prompt_tokens),
        "generated_token_ids": [tok["id"] for tok in generated_tokens],
        "generated_text": bytes_to_text(generated_tokens),
        "steps": steps,
        "raw": data,
    }


def compare_steps(a_steps, b_steps):
    a_ids = [s.get("argmax", {}).get("id") for s in a_steps]
    b_ids = [s.get("argmax", {}).get("id") for s in b_steps]
    out = {
        "same_token_prefix": 0,
        "first_token_mismatch_step": None,
        "step_count_a": len(a_steps),
        "step_count_b": len(b_steps),
        "matching_steps": 0,
        "token_agreement_rate": 0.0,
        "levenshtein_distance": levenshtein(a_ids, b_ids),
        "chosen_token_deltas": [],
    }
    for i, (sa, sb) in enumerate(zip(a_steps, b_steps)):
        a_arg = sa.get("argmax", {})
        b_arg = sb.get("argmax", {})
        same = a_arg.get("id") == b_arg.get("id")
        if same and out["first_token_mismatch_step"] is None:
            out["same_token_prefix"] += 1
        elif not same and out["first_token_mismatch_step"] is None:
            out["first_token_mismatch_step"] = i
        if same:
            out["matching_steps"] += 1
        out["chosen_token_deltas"].append({
            "step": i,
            "same_token": same,
            "id_a": a_arg.get("id"),
            "id_b": b_arg.get("id"),
            "bytes_a": a_arg.get("bytes", []),
            "bytes_b": b_arg.get("bytes", []),
            "logit_a": a_arg.get("logit"),
            "logit_b": b_arg.get("logit"),
            "logprob_a": a_arg.get("logprob"),
            "logprob_b": b_arg.get("logprob"),
        })
    denom = max(len(a_steps), len(b_steps), 1)
    out["token_agreement_rate"] = out["matching_steps"] / denom
    return out


def score_reference_path(reference_steps, scored_steps):
    pairs = []
    total_nll = 0.0
    same_argmax = 0
    count = min(len(reference_steps), len(scored_steps))
    for i in range(count):
        ref_arg = reference_steps[i].get("argmax", {})
        score_arg = scored_steps[i].get("argmax", {})
        score_chosen = scored_steps[i].get("chosen", {})
        lp = score_chosen.get("logprob")
        if lp is not None:
            total_nll += -lp
        same = ref_arg.get("id") == score_arg.get("id")
        if same:
            same_argmax += 1
        pairs.append({
            "step": i,
            "ref_id": ref_arg.get("id"),
            "ref_bytes": ref_arg.get("bytes", []),
            "scored_logprob": lp,
            "scored_argmax_id": score_arg.get("id"),
            "scored_argmax_bytes": score_arg.get("bytes", []),
            "same_argmax": same,
        })
    avg_nll = total_nll / count if count else None
    return {
        "step_count": count,
        "same_argmax_steps": same_argmax,
        "same_argmax_rate": (same_argmax / count) if count else None,
        "avg_nll": avg_nll,
        "perplexity_like": math.exp(avg_nll) if avg_nll is not None else None,
        "steps": pairs,
    }


def printable_bytes(bs):
    return bytes(bs).decode("utf-8", errors="replace")


def print_report(result, label_a, label_b):
    tok = result["tokenization"]
    dec = result["completion"]
    print("tokenization:")
    print(f"  {label_a}: {tok['count_a']} tokens")
    print(f"  {label_b}: {tok['count_b']} tokens")
    print(f"  equal: {tok['equal']}")
    print(f"  common_prefix: {tok['common_prefix']}")
    if tok["first_mismatch_index"] is not None:
        print(f"  first_mismatch_index: {tok['first_mismatch_index']}")
    print("completion:")
    print(f"  {label_a}: {len(result['a']['generated_token_ids'])} generated tokens")
    print(f"  {label_b}: {len(result['b']['generated_token_ids'])} generated tokens")
    print(f"  same_token_prefix: {dec['same_token_prefix']}")
    print(f"  first_token_mismatch_step: {dec['first_token_mismatch_step']}")
    print(f"  token_agreement_rate: {dec['token_agreement_rate']:.3f}")
    print(f"  levenshtein_distance: {dec['levenshtein_distance']}")
    print(f"  {label_a}_text: {json.dumps(result['a']['generated_text'])}")
    print(f"  {label_b}_text: {json.dumps(result['b']['generated_text'])}")
    for item in dec["chosen_token_deltas"][:8]:
        print(
            f"  step {item['step']}: same={item['same_token']} "
            f"{label_a}={item['id_a']}:{json.dumps(printable_bytes(item['bytes_a']))} "
            f"{label_b}={item['id_b']}:{json.dumps(printable_bytes(item['bytes_b']))}"
        )
    ref = result["reference_path_score"]
    print("reference_path_score:")
    print(f"  scored_model: {label_b} on {label_a} greedy path")
    print(f"  steps: {ref['step_count']}")
    if ref["avg_nll"] is not None:
        print(f"  avg_nll: {ref['avg_nll']:.6f}")
        print(f"  perplexity_like: {ref['perplexity_like']:.6f}")
        print(f"  same_argmax_rate: {ref['same_argmax_rate']:.3f}")


def main():
    args = parse_args()
    prompt = read_prompt(args)

    for path in (args.oracle_bin, args.model_a, args.model_b):
        if not os.path.exists(path):
            raise SystemExit(f"missing path: {path}")

    run_a = summarize_run(run_oracle(args, args.model_a, prompt))
    run_b = summarize_run(run_oracle(args, args.model_b, prompt))
    run_b_on_a_path = summarize_run(run_oracle(args, args.model_b, prompt, run_a["generated_token_ids"]))

    result = {
        "prompt": prompt,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "a": run_a,
        "b": run_b,
        "b_on_a_path": run_b_on_a_path,
        "tokenization": compare_token_ids(run_a["prompt_token_ids"], run_b["prompt_token_ids"]),
        "completion": compare_steps(run_a["steps"], run_b["steps"]),
        "reference_path_score": score_reference_path(run_a["steps"], run_b_on_a_path["steps"]),
    }

    print_report(result, args.label_a, args.label_b)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=2, ensure_ascii=False)
            fp.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
