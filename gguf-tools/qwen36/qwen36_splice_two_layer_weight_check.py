#!/usr/bin/env python3
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_prompt(args) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1).astype(np.float64, copy=False)
    bv = b.reshape(-1).astype(np.float64, copy=False)
    an = float(np.linalg.norm(av))
    bn = float(np.linalg.norm(bv))
    if an == 0.0 or bn == 0.0:
        return float("nan")
    return float(np.dot(av, bv) / (an * bn))


def topk(logits: np.ndarray, k: int):
    idx = np.argpartition(-logits, k - 1)[:k]
    idx = idx[np.argsort(-logits[idx])]
    return [{"id": int(i), "logit": float(logits[i])} for i in idx]


def token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Splice a C-computed Qwen3.6 decoder-prefix hidden state into HF and compare final logits")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--c-bin", default="./qwen36-c-decoder-chain-weight")
    ap.add_argument("--fixture", action="append", default=[], help="decoder fixture, one per owned layer in order")
    ap.add_argument("--layer0", help="back-compat alias for first fixture")
    ap.add_argument("--layer1", help="back-compat alias for second fixture")
    ap.add_argument("--splice-layer", type=int, help="HF layer index whose output should be replaced")
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    fixtures = list(args.fixture)
    if args.layer0:
        fixtures.append(args.layer0)
    if args.layer1:
        fixtures.append(args.layer1)
    if not fixtures:
        ap.error("at least one fixture is required")
    splice_layer = args.splice_layer if args.splice_layer is not None else (len(fixtures) - 1)

    prompt = read_prompt(args)
    print(f"[splice] prompt_chars={len(prompt)}")
    print(f"[splice] fixture_count={len(fixtures)} splice_layer={splice_layer}")
    for i, fixture in enumerate(fixtures):
        print(f"[splice] fixture[{i}]={fixture}")
    print(f"[splice] loading tokenizer from {args.hf}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    print(f"[splice] loading model from {args.hf}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("[splice] model loaded")

    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
        dump_path = tmp.name
    try:
        c_cmd = [args.c_bin, *fixtures, "--dump-last", dump_path]
        print(f"[splice] running c-chain: {' '.join(c_cmd)}")
        proc = subprocess.run(
            c_cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        c_report = proc.stdout
        c_hidden = np.fromfile(dump_path, dtype=np.float32)
        print(f"[splice] c-chain complete hidden_f32={c_hidden.shape[0]}")

        print("[splice] tokenizing prompt")
        inputs = tokenizer(prompt, return_tensors="pt")
        print(f"[splice] prompt_tokens={int(inputs['input_ids'].shape[1])}")
        print("[splice] running hf baseline forward")
        with torch.no_grad():
            base = model(**inputs)
        base_logits = base.logits[0, -1].detach().float().cpu().numpy()
        print("[splice] baseline forward complete")

        repl = torch.from_numpy(c_hidden).to(base.logits.device)

        def splice_patch(_mod, _inp, out):
            value = out[0] if isinstance(out, tuple) else out
            patched = value.clone()
            patched[0, -1, :] = repl.to(patched.dtype)
            if isinstance(out, tuple):
                return (patched,) + out[1:]
            return patched

        handle = model.model.layers[splice_layer].register_forward_hook(splice_patch)
        try:
            print("[splice] running hf patched forward")
            with torch.no_grad():
                patched = model(**inputs)
        finally:
            handle.remove()
        print("[splice] patched forward complete")

        patched_logits = patched.logits[0, -1].detach().float().cpu().numpy()
        base_argmax = int(np.argmax(base_logits))
        patched_argmax = int(np.argmax(patched_logits))

        result = {
            "prompt": prompt,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "fixtures": fixtures,
            "splice_layer": splice_layer,
            "c_report": c_report,
            "logits_cosine": cosine(base_logits, patched_logits),
            "logits_rmse": float(np.sqrt(np.mean((base_logits - patched_logits) ** 2))),
            "argmax_equal": base_argmax == patched_argmax,
            "base_argmax": {
                "id": base_argmax,
                "text": token_text(tokenizer, base_argmax),
                "logit": float(base_logits[base_argmax]),
            },
            "patched_argmax": {
                "id": patched_argmax,
                "text": token_text(tokenizer, patched_argmax),
                "logit": float(patched_logits[patched_argmax]),
            },
            "base_topk": topk(base_logits, args.top_k),
            "patched_topk": topk(patched_logits, args.top_k),
        }

        print(f"prompt_tokens: {result['prompt_tokens']}")
        print(f"logits_cosine: {result['logits_cosine']:.8f}")
        print(f"logits_rmse: {result['logits_rmse']:.8f}")
        print(f"argmax_equal: {result['argmax_equal']}")
        print(
            f"base_argmax: id={result['base_argmax']['id']} "
            f"text={json.dumps(result['base_argmax']['text'])} "
            f"logit={result['base_argmax']['logit']:.6f}"
        )
        print(
            f"patched_argmax: id={result['patched_argmax']['id']} "
            f"text={json.dumps(result['patched_argmax']['text'])} "
            f"logit={result['patched_argmax']['logit']:.6f}"
        )
        print("base_topk:")
        for item in result["base_topk"]:
            print(f"  id={item['id']} text={json.dumps(token_text(tokenizer, item['id']))} logit={item['logit']:.6f}")
        print("patched_topk:")
        for item in result["patched_topk"]:
            print(f"  id={item['id']} text={json.dumps(token_text(tokenizer, item['id']))} logit={item['logit']:.6f}")

        if args.json_out:
            print(f"[splice] writing json report to {args.json_out}")
            Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"json_out: {args.json_out}")
        return 0
    finally:
        Path(dump_path).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
