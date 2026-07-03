#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MAGIC = b"Q36RHF01"


def read_prompt(args) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def main() -> int:
    ap = argparse.ArgumentParser(description="Light HF root probe for Qwen3.6 final hidden and top logits")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompt = read_prompt(args)
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    captured = {}

    def norm_pre(_mod, inp):
        captured["hidden_pre"] = inp[0][0, -1].detach().float().cpu().numpy()

    def norm_post(_mod, _inp, out):
        captured["hidden_post"] = out[0, -1].detach().float().cpu().numpy()

    h0 = model.model.norm.register_forward_pre_hook(norm_pre)
    h1 = model.model.norm.register_forward_hook(norm_post)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    h0.remove()
    h1.remove()

    logits = outputs.logits[0, -1].detach().float().cpu()
    vals, idx = torch.topk(logits, args.top_k)
    top_ids = idx.detach().cpu().numpy().astype(np.uint32)
    top_logits = vals.detach().cpu().numpy().astype(np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        fp.write(MAGIC)
        fp.write(struct.pack("<III", 2048, int(args.top_k), int(inputs["input_ids"].shape[1])))
        fp.write(np.ascontiguousarray(captured["hidden_pre"], dtype=np.float32).tobytes(order="C"))
        fp.write(np.ascontiguousarray(captured["hidden_post"], dtype=np.float32).tobytes(order="C"))
        fp.write(top_ids.astype(np.uint32, copy=False).tobytes(order="C"))
        fp.write(top_logits.tobytes(order="C"))

    sidecar = {
        "prompt": prompt,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "top_k": int(args.top_k),
        "top_ids": [int(x) for x in top_ids.tolist()],
        "top_logits": [float(x) for x in top_logits.tolist()],
        "out": str(out_path),
        "size_bytes": out_path.stat().st_size,
    }
    out_path.with_suffix(out_path.suffix + ".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"prompt_tokens: {sidecar['prompt_tokens']}")
    print(f"top_k: {sidecar['top_k']}")
    print(f"wrote: {out_path}")
    print(f"size_bytes: {out_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
