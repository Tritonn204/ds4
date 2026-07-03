#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_prompt(args) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def parse_layers(text: str | None) -> set[int] | None:
    if not text:
        return None
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a)
            end = int(b)
            for i in range(start, end + 1):
                out.add(i)
        else:
            out.add(int(part))
    return out


def top_logits(logits: torch.Tensor, k: int):
    vals, idx = torch.topk(logits, k=k, dim=-1)
    return [
        {"id": int(i), "logit": float(v)}
        for v, i in zip(vals.tolist(), idx.tolist())
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="HF Qwen3.6 router/top-k micro oracle")
    ap.add_argument("--hf", required=True, help="HF model directory")
    ap.add_argument("--prompt", default="Hello world", help="Prompt text")
    ap.add_argument("--prompt-file", help="Read prompt text from file")
    ap.add_argument("--layers", help="Comma/range list, e.g. 0,3,34-39")
    ap.add_argument("--top-k", type=int, default=8, help="Final-logit top-k")
    ap.add_argument("--json-out", help="Optional JSON output path")
    args = ap.parse_args()

    prompt = read_prompt(args)
    selected_layers = parse_layers(args.layers)

    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    router_data: dict[int, dict] = {}
    handles = []

    def make_router_hook(layer_idx: int):
        def hook(_mod, _inp, out):
            router_logits, router_scores, router_indices = out
            last = router_logits[-1].detach().float().cpu()
            last_scores = router_scores[-1].detach().float().cpu()
            last_indices = router_indices[-1].detach().cpu()
            router_data[layer_idx] = {
                "top_indices": [int(x) for x in last_indices.tolist()],
                "top_scores": [float(x) for x in last_scores.tolist()],
                "top_logits": [float(last[i]) for i in last_indices.tolist()],
            }
        return hook

    def make_shared_gate_hook(layer_idx: int):
        def hook(_mod, _inp, out):
            value = torch.sigmoid(out[-1].detach().float().cpu()).reshape(-1)
            router_data.setdefault(layer_idx, {})
            router_data[layer_idx]["shared_gate"] = float(value[-1])
        return hook

    for i, layer in enumerate(model.model.layers):
        if selected_layers is not None and i not in selected_layers:
            continue
        handles.append(layer.mlp.gate.register_forward_hook(make_router_hook(i)))
        handles.append(layer.mlp.shared_expert_gate.register_forward_hook(make_shared_gate_hook(i)))

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    for h in handles:
        h.remove()

    logits = outputs.logits[0, -1].detach().float().cpu()
    result = {
        "prompt": prompt,
        "prompt_token_count": int(inputs["input_ids"].shape[1]),
        "selected_layers": sorted(router_data.keys()),
        "final_top_logits": top_logits(logits, args.top_k),
        "layers": [
            {"layer": i, **router_data[i]}
            for i in sorted(router_data.keys())
        ],
    }

    print(f"prompt_tokens: {result['prompt_token_count']}")
    print("final_top_logits:")
    for item in result["final_top_logits"]:
        print(f"  id={item['id']} logit={item['logit']:.6f}")
    print("router_layers:")
    for item in result["layers"]:
        print(
            f"  blk.{item['layer']} experts={item['top_indices']} "
            f"scores={[round(x, 6) for x in item['top_scores']]} "
            f"shared_gate={item.get('shared_gate', None):.6f}"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
