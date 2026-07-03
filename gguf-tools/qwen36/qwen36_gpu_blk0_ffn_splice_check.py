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
    ap = argparse.ArgumentParser(description="Splice a GPU-owned Qwen blk.0 FFN closure into the HF tail")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--fixture", required=True, help="blk.0 decoder-layer weight fixture")
    ap.add_argument("--c-bin", default="./qwen36-gpu-blk0-ffn-q8-oracle")
    ap.add_argument("--prefix-fixture")
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    prompt = read_prompt(args)
    print(f"[gpu-blk0] prompt_chars={len(prompt)}")
    print(f"[gpu-blk0] loading tokenizer from {args.hf}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    print(f"[gpu-blk0] loading model from {args.hf}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("[gpu-blk0] model loaded")

    inputs = tokenizer(prompt, return_tensors="pt")
    seq_len = int(inputs["input_ids"].shape[1])
    hidden = int(model.config.hidden_size)

    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
        seq_path = tmp.name
    try:
        c_cmd = [args.c_bin, args.gguf, args.fixture]
        if args.prefix_fixture:
            c_cmd.extend(["--prefix", args.prefix_fixture])
        c_cmd.extend(["--dump-seq", seq_path])
        print(f"[gpu-blk0] running oracle: {' '.join(c_cmd)}")
        proc = subprocess.run(c_cmd, capture_output=True, text=True, check=True)
        c_report = proc.stdout
        owned_seq = np.fromfile(seq_path, dtype=np.float32)
        if owned_seq.size != seq_len * hidden:
            raise RuntimeError(f"unexpected seq dump size: got {owned_seq.size}, expected {seq_len * hidden}")
        owned_seq = owned_seq.reshape(seq_len, hidden)

        print("[gpu-blk0] running hf baseline")
        with torch.no_grad():
            base = model(**inputs)
        base_logits = base.logits[0, -1].detach().float().cpu().numpy()

        repl = torch.from_numpy(owned_seq).unsqueeze(0)

        def splice_patch(_mod, _inp, out):
            value = out[0] if isinstance(out, tuple) else out
            patched = value.clone()
            patched[:, :, :] = repl.to(device=patched.device, dtype=patched.dtype)
            if isinstance(out, tuple):
                return (patched,) + out[1:]
            return patched

        handle = model.model.layers[0].register_forward_hook(splice_patch)
        try:
            print("[gpu-blk0] running hf patched tail from layer 1")
            with torch.no_grad():
                patched = model(**inputs)
        finally:
            handle.remove()

        patched_logits = patched.logits[0, -1].detach().float().cpu().numpy()
        base_argmax = int(np.argmax(base_logits))
        patched_argmax = int(np.argmax(patched_logits))
        result = {
            "prompt": prompt,
            "prompt_tokens": seq_len,
            "fixture": args.fixture,
            "prefix_fixture": args.prefix_fixture,
            "c_report": c_report,
            "logits_cosine": cosine(base_logits, patched_logits),
            "logits_rmse": float(np.sqrt(np.mean((base_logits - patched_logits) ** 2))),
            "argmax_equal": base_argmax == patched_argmax,
            "base_argmax": {"id": base_argmax, "text": token_text(tokenizer, base_argmax), "logit": float(base_logits[base_argmax])},
            "patched_argmax": {"id": patched_argmax, "text": token_text(tokenizer, patched_argmax), "logit": float(patched_logits[patched_argmax])},
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
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"json_out: {args.json_out}")
        return 0
    finally:
        Path(seq_path).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
