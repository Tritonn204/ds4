#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from transformers import AutoModelForCausalLM, AutoTokenizer


MAGIC = b"Q36ROOT1"


def tmap(reader: GGUFReader):
    return {t.name: t for t in reader.tensors}


def load_f32(tensors, name: str) -> np.ndarray:
    t = tensors[name]
    return dequantize(t.data, t.tensor_type).astype(np.float32, copy=False)


def read_prompt(args) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def as_weight(arr: np.ndarray, out_rows: int, in_cols: int) -> np.ndarray:
    if arr.shape == (out_rows, in_cols):
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.shape == (in_cols, out_rows):
        return np.ascontiguousarray(arr.T, dtype=np.float32)
    raise ValueError(f"unexpected weight shape {arr.shape}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export root-surface fixture for narrow Qwen3.6 C replay")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
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
    handles = []

    def norm_pre(_mod, inp):
        captured["final_hidden_pre_norm"] = inp[0][0, -1].detach().float().cpu().numpy()

    def norm_post(_mod, _inp, out):
        captured["final_hidden_post_norm"] = out[0, -1].detach().float().cpu().numpy()

    handles.append(model.model.norm.register_forward_pre_hook(norm_pre))
    handles.append(model.model.norm.register_forward_hook(norm_post))

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    for h in handles:
        h.remove()

    logits = outputs.logits[0, -1].detach().float().cpu()
    vals, idx = torch.topk(logits, args.top_k)
    top_ids = idx.detach().cpu().numpy().astype(np.uint32)
    top_logits = vals.detach().cpu().numpy().astype(np.float32)

    reader = GGUFReader(args.gguf)
    tensors = tmap(reader)
    output_norm_w = load_f32(tensors, "output_norm.weight")
    output_w = load_f32(tensors, "output.weight")
    output_w = as_weight(output_w, 248320, 2048)
    sel_rows = np.ascontiguousarray(output_w[top_ids], dtype=np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        fp.write(MAGIC)
        fp.write(struct.pack("<III", 2048, int(args.top_k), int(inputs["input_ids"].shape[1])))
        for arr in (
            np.asarray(captured["final_hidden_pre_norm"], dtype=np.float32),
            np.asarray(captured["final_hidden_post_norm"], dtype=np.float32),
            np.asarray(output_norm_w, dtype=np.float32),
            top_ids.astype(np.float32),
            top_logits,
            sel_rows,
        ):
            fp.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes(order="C"))

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
