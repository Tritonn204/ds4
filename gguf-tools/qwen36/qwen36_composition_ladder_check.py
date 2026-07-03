#!/usr/bin/env python3
import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PREFIX_MAGIC = b"Q36PFX01"


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


def classify_stage(argmax_equal: bool, cosine_value: float, margin_delta: float) -> str:
    if argmax_equal and cosine_value >= 0.999:
        return "PASS_BEHAVIORAL"
    if argmax_equal and cosine_value >= 0.995:
        return "PASS_NEAR"
    if not argmax_equal and margin_delta < 0.25:
        return "FAIL_TIE_SENSITIVE"
    return "FAIL_COMPOSITION"


def write_prefix_fixture(path: Path, token_ids: list[int], hidden: int, input_seq: np.ndarray | None = None) -> None:
    tok = np.asarray(token_ids, dtype=np.uint32)
    if input_seq is None:
        seq = np.zeros((len(token_ids), hidden), dtype=np.float32)
    else:
        seq = np.asarray(input_seq, dtype=np.float32)
        if seq.shape != (len(token_ids), hidden):
            raise RuntimeError(f"unexpected prefix input_seq shape: got {seq.shape}, expected {(len(token_ids), hidden)}")
    with path.open("wb") as fp:
        fp.write(PREFIX_MAGIC)
        fp.write(struct.pack("<II", len(token_ids), hidden))
        fp.write(tok.tobytes(order="C"))
        fp.write(seq.tobytes(order="C"))


def read_seq_hidden(path: Path, seq_len: int, hidden: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    expect = seq_len * hidden
    if arr.size != expect:
        raise RuntimeError(f"unexpected hidden dump size: got {arr.size}, expected {expect}")
    return arr.reshape(seq_len, hidden)


def splice_logits(model, inputs, splice_layer: int, repl_seq: np.ndarray) -> np.ndarray:
    repl_t = torch.from_numpy(repl_seq).unsqueeze(0)

    def splice_patch(_mod, _inp, out):
        value = out[0] if isinstance(out, tuple) else out
        patched = value.clone()
        patched[:, :, :] = repl_t.to(device=patched.device, dtype=patched.dtype)
        if isinstance(out, tuple):
            return (patched,) + out[1:]
        return patched

    handle = model.model.layers[splice_layer].register_forward_hook(splice_patch)
    try:
        with torch.no_grad():
            patched = model(**inputs)
    finally:
        handle.remove()
    return patched.logits[0, -1].detach().float().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the owned-prefix composition ladder for Qwen3.6 depths 0..3")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--blk0-bin", default="./qwen36-gpu-blk0-dynamic-q8-oracle")
    ap.add_argument("--blk0-fixture", required=True)
    ap.add_argument("--prefix-chain-bin", default="./qwen36-c-prefix-q8-chain-dynamic")
    ap.add_argument("--prefix-chain-prefix-flag", action="store_true",
                    help="Call --prefix-chain-bin as MODEL.gguf FIXTURE... --prefix PREFIX.bin instead of MODEL.gguf PREFIX.bin FIXTURE...")
    ap.add_argument("--blk1-fixture", required=True)
    ap.add_argument("--blk2-fixture", required=True)
    ap.add_argument("--blk3-bin", default="./qwen36-gpu-full-layer-q8-dynamic")
    ap.add_argument("--blk3-enable", action="store_true")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    prompt = read_prompt(args)
    print(f"[ladder] prompt_chars={len(prompt)}")
    print(f"[ladder] loading tokenizer from {args.hf}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    print(f"[ladder] loading model from {args.hf}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    hidden = int(model.config.hidden_size)
    print(f"[ladder] model loaded hidden={hidden}")

    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs["input_ids"][0].tolist()
    seq_len = len(token_ids)
    print(f"[ladder] prompt_tokens={seq_len}")

    print("[ladder] running hf baseline forward")
    with torch.no_grad():
        base = model(**inputs)
    base_logits = base.logits[0, -1].detach().float().cpu().numpy()
    base_argmax = int(np.argmax(base_logits))
    base_margin = logit_margin(base_logits)

    stages = []
    with tempfile.TemporaryDirectory(prefix="q36_ladder_") as td:
        td_path = Path(td)
        prefix_path = td_path / "prefix.bin"
        blk0_seq_path = td_path / "blk0_seq.f32"
        owned_seq_path = td_path / "owned_seq.f32"
        blk3_seq_path = td_path / "blk3_seq.f32"

        write_prefix_fixture(prefix_path, token_ids, hidden)

        blk0_cmd = [args.blk0_bin, args.gguf, args.blk0_fixture, "--prefix", str(prefix_path), "--dump-seq", str(blk0_seq_path)]
        print(f"[ladder] stage=blk0 cmd={' '.join(blk0_cmd)}")
        blk0_proc = subprocess.run(blk0_cmd, capture_output=True, text=True, check=True)
        blk0_seq = read_seq_hidden(blk0_seq_path, seq_len, hidden)
        stages.append({
            "name": "blk0",
            "splice_layer": 0,
            "owned_seq_path": str(blk0_seq_path),
            "owned_seq": blk0_seq,
            "report": blk0_proc.stdout,
        })

        write_prefix_fixture(prefix_path, token_ids, hidden, input_seq=blk0_seq)
        depth_specs = [
            ("blk01", [args.blk1_fixture], 1),
            ("blk012", [args.blk1_fixture, args.blk2_fixture], 2),
        ]
        for name, fixtures, splice_layer in depth_specs:
            if args.prefix_chain_prefix_flag:
                c_cmd = [args.prefix_chain_bin, args.gguf, *fixtures, "--prefix", str(prefix_path)]
            else:
                c_cmd = [args.prefix_chain_bin, args.gguf, str(prefix_path), *fixtures]
            c_cmd.extend(["--use-prefix-input-seq", "--dump-seq", str(owned_seq_path)])
            print(f"[ladder] stage={name} cmd={' '.join(c_cmd)}")
            proc = subprocess.run(c_cmd, capture_output=True, text=True, check=True)
            owned_seq = read_seq_hidden(owned_seq_path, seq_len, hidden).copy()
            stages.append({
                "name": name,
                "splice_layer": splice_layer,
                "owned_seq_path": str(owned_seq_path),
                "owned_seq": owned_seq,
                "report": proc.stdout,
            })

        if args.blk3_enable:
            blk012 = stages[-1]["owned_seq"]
            blk012_path = td_path / "blk012_input.f32"
            blk012.astype(np.float32, copy=False).tofile(blk012_path)
            full_cmd = [args.blk3_bin, args.gguf, str(blk012_path), str(blk3_seq_path), "--layer", "3"]
            print(f"[ladder] stage=blk0123 cmd={' '.join(full_cmd)}")
            proc = subprocess.run(full_cmd, capture_output=True, text=True, check=True)
            blk3_seq = read_seq_hidden(blk3_seq_path, seq_len, hidden)
            stages.append({
                "name": "blk0123",
                "splice_layer": 3,
                "owned_seq_path": str(blk3_seq_path),
                "owned_seq": blk3_seq,
                "report": proc.stdout,
            })

        results = []
        for stage in stages:
            print(f"[ladder] splicing stage={stage['name']} at layer={stage['splice_layer']}")
            patched_logits = splice_logits(model, inputs, stage["splice_layer"], stage["owned_seq"])
            patched_argmax = int(np.argmax(patched_logits))
            patched_margin = logit_margin(patched_logits)
            margin_delta = abs(base_margin - patched_margin)
            result = {
                "name": stage["name"],
                "splice_layer": stage["splice_layer"],
                "owned_seq_path": stage["owned_seq_path"],
                "argmax_equal": base_argmax == patched_argmax,
                "classification": classify_stage(base_argmax == patched_argmax, cosine(base_logits, patched_logits), margin_delta),
                "logits_cosine": cosine(base_logits, patched_logits),
                "logits_rmse": float(np.sqrt(np.mean((base_logits - patched_logits) ** 2))),
                "base_argmax": {"id": base_argmax, "text": token_text(tokenizer, base_argmax), "logit": float(base_logits[base_argmax])},
                "patched_argmax": {"id": patched_argmax, "text": token_text(tokenizer, patched_argmax), "logit": float(patched_logits[patched_argmax])},
                "base_margin_top2": base_margin,
                "patched_margin_top2": patched_margin,
                "margin_delta_top2": margin_delta,
                "base_topk": topk(base_logits, args.top_k),
                "patched_topk": topk(patched_logits, args.top_k),
                "report": stage["report"],
            }
            results.append(result)
            print(
                f"[ladder] {stage['name']} cosine={result['logits_cosine']:.8f} "
                f"rmse={result['logits_rmse']:.8f} argmax_equal={result['argmax_equal']} "
                f"classification={result['classification']}"
            )

    out = {
        "prompt": prompt,
        "prompt_tokens": seq_len,
        "base_argmax": {"id": base_argmax, "text": token_text(tokenizer, base_argmax), "logit": float(base_logits[base_argmax])},
        "base_margin_top2": base_margin,
        "results": results,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
