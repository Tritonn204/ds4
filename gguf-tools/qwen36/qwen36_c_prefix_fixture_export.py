#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MAGIC = b"Q36PFX01"


def read_prompt(args) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def main() -> int:
    ap = argparse.ArgumentParser(description="Export prompt token IDs and HF embedding-sequence reference for narrow Qwen3.6 prefix replay")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
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

    captured: dict[str, np.ndarray] = {}

    def cap_layer0_input(_mod, inp):
        captured["layer0_input_seq"] = inp[0][0].detach().float().cpu().numpy()

    handle = model.model.layers[0].register_forward_pre_hook(cap_layer0_input)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    token_ids = inputs["input_ids"][0].detach().cpu().numpy().astype(np.uint32)
    input_seq = np.asarray(captured["layer0_input_seq"], dtype=np.float32)
    seq_len = int(token_ids.shape[0])
    hidden = int(input_seq.shape[1])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        fp.write(MAGIC)
        fp.write(struct.pack("<II", seq_len, hidden))
        fp.write(token_ids.tobytes(order="C"))
        fp.write(np.ascontiguousarray(input_seq, dtype=np.float32).tobytes(order="C"))

    sidecar = {
        "prompt": prompt,
        "prompt_tokens": seq_len,
        "hidden": hidden,
        "token_ids": [int(x) for x in token_ids.tolist()],
        "out": str(out_path),
        "size_bytes": out_path.stat().st_size,
    }
    out_path.with_suffix(out_path.suffix + ".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"prompt_tokens: {seq_len}")
    print(f"hidden: {hidden}")
    print(f"wrote: {out_path}")
    print(f"size_bytes: {out_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
