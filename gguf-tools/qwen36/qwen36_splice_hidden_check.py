#!/usr/bin/env python3
import argparse
import json
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


def logit_margin(logits: np.ndarray) -> float:
    idx = np.argpartition(-logits, 1)[:2]
    top2 = np.sort(logits[idx])[::-1]
    if top2.shape[0] < 2:
        return float("nan")
    return float(top2[0] - top2[1])


def main() -> int:
    ap = argparse.ArgumentParser(description="Splice an arbitrary Qwen hidden dump into the HF tail and compare logits")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--hidden-f32", required=True, help="Raw float32 dump, either [hidden] or [seq, hidden]")
    ap.add_argument("--splice-layer", type=int, required=True)
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--mode", choices=["auto", "last", "seq"], default="auto")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    prompt = read_prompt(args)
    print(f"[splice-hidden] prompt_chars={len(prompt)}")
    print(f"[splice-hidden] splice_layer={args.splice_layer} mode={args.mode}")
    print(f"[splice-hidden] loading tokenizer from {args.hf}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    print(f"[splice-hidden] loading model from {args.hf}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    hidden = int(model.config.hidden_size)
    print(f"[splice-hidden] model loaded hidden={hidden}")

    inputs = tokenizer(prompt, return_tensors="pt")
    seq_len = int(inputs["input_ids"].shape[1])
    raw = np.fromfile(args.hidden_f32, dtype=np.float32)
    if args.mode == "last":
        expect = hidden
        if raw.size != expect:
            raise RuntimeError(f"unexpected last-hidden size: got {raw.size}, expected {expect}")
        splice_mode = "last"
        repl = raw.reshape(hidden)
    elif args.mode == "seq":
        expect = seq_len * hidden
        if raw.size != expect:
            raise RuntimeError(f"unexpected seq-hidden size: got {raw.size}, expected {expect}")
        splice_mode = "seq"
        repl = raw.reshape(seq_len, hidden)
    else:
        if raw.size == hidden:
            splice_mode = "last"
            repl = raw.reshape(hidden)
        elif raw.size == seq_len * hidden:
            splice_mode = "seq"
            repl = raw.reshape(seq_len, hidden)
        else:
            raise RuntimeError(
                f"auto mode could not infer hidden shape: size={raw.size}, "
                f"expected last={hidden} or seq={seq_len * hidden}"
            )
    print(f"[splice-hidden] hidden_mode={splice_mode}")

    print("[splice-hidden] running hf baseline forward")
    with torch.no_grad():
        base = model(**inputs)
    base_logits = base.logits[0, -1].detach().float().cpu().numpy()
    print("[splice-hidden] baseline forward complete")

    if splice_mode == "last":
        repl_t = torch.from_numpy(repl)

        def splice_patch(_mod, _inp, out):
            value = out[0] if isinstance(out, tuple) else out
            patched = value.clone()
            patched[0, -1, :] = repl_t.to(device=patched.device, dtype=patched.dtype)
            if isinstance(out, tuple):
                return (patched,) + out[1:]
            return patched
    else:
        repl_t = torch.from_numpy(repl).unsqueeze(0)

        def splice_patch(_mod, _inp, out):
            value = out[0] if isinstance(out, tuple) else out
            patched = value.clone()
            patched[:, :, :] = repl_t.to(device=patched.device, dtype=patched.dtype)
            if isinstance(out, tuple):
                return (patched,) + out[1:]
            return patched

    handle = model.model.layers[args.splice_layer].register_forward_hook(splice_patch)
    try:
        print("[splice-hidden] running hf patched forward")
        with torch.no_grad():
            patched = model(**inputs)
    finally:
        handle.remove()
    print("[splice-hidden] patched forward complete")

    patched_logits = patched.logits[0, -1].detach().float().cpu().numpy()
    base_argmax = int(np.argmax(base_logits))
    patched_argmax = int(np.argmax(patched_logits))
    result = {
        "prompt": prompt,
        "prompt_tokens": seq_len,
        "hidden_f32": args.hidden_f32,
        "splice_layer": args.splice_layer,
        "hidden_mode": splice_mode,
        "logits_cosine": cosine(base_logits, patched_logits),
        "logits_rmse": float(np.sqrt(np.mean((base_logits - patched_logits) ** 2))),
        "argmax_equal": base_argmax == patched_argmax,
        "base_argmax": {"id": base_argmax, "text": token_text(tokenizer, base_argmax), "logit": float(base_logits[base_argmax])},
        "patched_argmax": {"id": patched_argmax, "text": token_text(tokenizer, patched_argmax), "logit": float(patched_logits[patched_argmax])},
        "base_margin_top2": logit_margin(base_logits),
        "patched_margin_top2": logit_margin(patched_logits),
        "base_topk": topk(base_logits, args.top_k),
        "patched_topk": topk(patched_logits, args.top_k),
    }

    print(f"prompt_tokens: {result['prompt_tokens']}")
    print(f"logits_cosine: {result['logits_cosine']:.8f}")
    print(f"logits_rmse: {result['logits_rmse']:.8f}")
    print(f"argmax_equal: {result['argmax_equal']}")
    print(f"base_margin_top2: {result['base_margin_top2']:.8f}")
    print(f"patched_margin_top2: {result['patched_margin_top2']:.8f}")
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


if __name__ == "__main__":
    raise SystemExit(main())
