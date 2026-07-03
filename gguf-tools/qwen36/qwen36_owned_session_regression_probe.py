#!/usr/bin/env python3
import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen36_behavior_oracle import OwnedSession
from qwen36_hybrid_prefix_tail_greedy import read_prompt, run_hf_patched_compare


def main() -> int:
    ap = argparse.ArgumentParser(description="A/B probe for owned-session -> HF tail regression")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prefix-seq-worker-bin")
    ap.add_argument("--prefix-seq-fixture")
    ap.add_argument("--c-bin")
    ap.add_argument("--c-bin-worker-bin")
    ap.add_argument("--owned-session-worker-bin", required=True)
    ap.add_argument("--full-layer-worker-bin")
    ap.add_argument("--fixture", action="append", default=[])
    ap.add_argument("--full-layer", action="append", type=int, default=[])
    ap.add_argument("--splice-layer", type=int, required=True)
    ap.add_argument("--prompt")
    ap.add_argument("--prompt-file")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--hf-patched-layer-progress", dest="hf_patched_layer_progress", action="store_true", default=True)
    ap.add_argument("--no-hf-patched-layer-progress", dest="hf_patched_layer_progress", action="store_false")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    prompt = read_prompt(args)
    print(f"[probe] loading tokenizer from {args.hf}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    print(f"[probe] loading model from {args.hf}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    torch.set_num_threads(min(32, torch.get_num_threads() or 16))
    token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    hidden = int(model.config.hidden_size)
    print(f"[probe] model loaded prompt_tokens={len(token_ids)} hidden={hidden}")

    session = OwnedSession(args, hidden)
    with tempfile.TemporaryDirectory(prefix="q36_owned_regress_") as td:
        td_path = Path(td)
        owned_seq_path = td_path / "owned_seq.f32"
        try:
            print("[probe] running owned-session prefill")
            owned_seq, owned_meta = session.run_step(token_ids, 0)
            owned_seq.astype("float32").tofile(owned_seq_path)
            print(f"[probe] owned_prefix_ms={owned_meta['owned_prefix_ms']:.2f}")

            print("[probe] running same-process hf tail")
            t0 = time.perf_counter()
            sameproc = run_hf_patched_compare(
                model=model,
                token_ids=token_ids,
                owned_seq=owned_seq,
                splice_layer=args.splice_layer,
                tokenizer=tokenizer,
                top_k=args.top_k,
                layer_progress=args.hf_patched_layer_progress,
            )
            sameproc_wall_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[probe] sameproc_hf_patched_ms={sameproc['hf_patched_ms']:.2f}")

            print("[probe] running fresh-process hf tail")
            fresh_json = td_path / "fresh_tail.json"
            cmd = [
                "python3",
                "/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_worker_residence_probe.py",
                "--hf", args.hf,
                "--gguf", args.gguf,
                "--owned-seq-f32", str(owned_seq_path),
                "--prompt-file", args.prompt_file if args.prompt_file else str(td_path / "prompt.txt"),
                "--splice-layer", str(args.splice_layer),
                "--top-k", str(args.top_k),
                "--json-out", str(fresh_json),
            ]
            if args.hf_patched_layer_progress:
                cmd.append("--hf-tail-layer-progress")
            if not args.prompt_file:
                (td_path / "prompt.txt").write_text(prompt, encoding="utf-8")
            print(f"[probe] fresh_tail_cmd={' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            fresh = json.loads(fresh_json.read_text(encoding="utf-8"))

            result = {
                "prompt_tokens": len(token_ids),
                "owned_prefix_ms": owned_meta["owned_prefix_ms"],
                "same_process": {
                    "wall_ms": sameproc_wall_ms,
                    **sameproc,
                },
                "fresh_process": fresh,
                "regression_ratio": (
                    float(sameproc["hf_patched_ms"]) / float(fresh["hf_patched_ms"])
                    if fresh.get("hf_patched_ms")
                    else None
                ),
            }
            if args.json_out:
                Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(f"json_out: {args.json_out}")
            else:
                print(json.dumps(result, indent=2))
        finally:
            session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
