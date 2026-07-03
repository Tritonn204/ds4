#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen36_behavior_oracle import OwnedSessionWorker
from qwen36_hybrid_prefix_tail_greedy import hf_layer_progress, read_prompt, run_hf_patched_compare


DEFAULT_FIXTURE_DIR = Path(
    "/mnt/f/git/ds4/.cache/qwen36_live_contracts/Qwen3.6-35B-A3B-Q8_0/shared"
)
FULL_LAYER_STRIDE = 4
FULL_LAYER_OFFSET = 3


def token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def topk_overlap(a_topk: list[dict], b_topk: list[dict]) -> int:
    return len({item["id"] for item in a_topk} & {item["id"] for item in b_topk})


def topk(logits: np.ndarray, k: int) -> list[dict]:
    idx = np.argpartition(-logits, k - 1)[:k]
    idx = idx[np.argsort(-logits[idx])]
    return [{"id": int(i), "logit": float(logits[i])} for i in idx]


def vec_rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def vec_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    an = float(np.linalg.norm(aa))
    bn = float(np.linalg.norm(bb))
    if an == 0.0 or bn == 0.0:
        return float("nan")
    return float(np.dot(aa, bb) / (an * bn))


def row_rmse_summary(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        raise RuntimeError(f"owned vs hf splice shape mismatch: {a.shape} vs {b.shape}")
    row_rmses = []
    for row_idx in range(a.shape[0]):
        rmse = vec_rmse(a[row_idx], b[row_idx])
        row_rmses.append({"row": row_idx, "rmse": rmse})
    worst = max(row_rmses, key=lambda item: item["rmse"])
    return {
        "seq_rmse": vec_rmse(a, b),
        "seq_cosine": vec_cosine(a, b),
        "last_row_rmse": row_rmses[-1]["rmse"],
        "worst_row": worst,
        "row_rmses": row_rmses,
    }


def fixture_path(layer: int, fixture_dir: Path) -> str:
    return str(fixture_dir / f"blk{layer}.live.bin")


def is_full_layer(layer: int) -> bool:
    return layer % FULL_LAYER_STRIDE == FULL_LAYER_OFFSET


def build_layout(
    owned_layers: int,
    fixture_dir: Path,
) -> tuple[list[int], list[list[str]], list[str], list[int]]:
    if owned_layers <= 0 or owned_layers % FULL_LAYER_STRIDE != 0:
        raise RuntimeError(
            f"owned_layers must be a positive multiple of {FULL_LAYER_STRIDE}, got {owned_layers}"
        )
    full_layers = [layer for layer in range(owned_layers) if is_full_layer(layer)]
    cycle_chunks: list[list[str]] = []
    flat_fixtures: list[str] = []
    hybrid_layers: list[int] = []
    for full_layer in full_layers:
        cycle_layers = list(range(full_layer - FULL_LAYER_OFFSET, full_layer))
        chunk = [fixture_path(layer, fixture_dir) for layer in cycle_layers]
        for layer, path in zip(cycle_layers, chunk):
            if not Path(path).exists():
                raise RuntimeError(f"missing hybrid fixture for owned depth {owned_layers}: {path}")
            hybrid_layers.append(layer)
        cycle_chunks.append(chunk)
        flat_fixtures.extend(chunk)
    return full_layers, cycle_chunks, flat_fixtures, hybrid_layers


def write_native_config(
    path: Path,
    *,
    full_layers: list[int],
    cycle_chunks: list[list[str]],
    hybrid_worker_bin: str,
    full_worker_bin: str,
) -> None:
    lines = [
        f"hybrid_worker_bin {hybrid_worker_bin}",
        f"full_worker_bin {full_worker_bin}",
    ]
    for full_layer, chunk in zip(full_layers, cycle_chunks):
        lines.append(f"cycle {full_layer} {','.join(chunk)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_hf_baseline_capture(
    *,
    model,
    token_ids: list[int],
    splice_layer: int,
    capture_layers: list[int],
    tokenizer,
    top_k: int,
    layer_progress: bool,
) -> dict:
    inputs = {"input_ids": torch.tensor([token_ids], dtype=torch.long)}
    captured: dict[int, np.ndarray] = {}
    handles = []

    def make_capture_hook(layer_idx: int):
        def capture_hook(_mod, _inp, out):
            value = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = value.detach().float().cpu().numpy()[0]
        return capture_hook

    for layer_idx in sorted(set(capture_layers)):
        handles.append(model.model.layers[layer_idx].register_forward_hook(make_capture_hook(layer_idx)))
    try:
        t0 = time.perf_counter()
        with hf_layer_progress(model, layer_progress, "hf_baseline"), torch.inference_mode():
            out = model(**inputs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        for handle in handles:
            handle.remove()

    missing = [layer_idx for layer_idx in sorted(set(capture_layers)) if layer_idx not in captured]
    if missing:
        raise RuntimeError(f"failed to capture layers: {missing}")

    base_logits = out.logits[0, -1].detach().float().cpu().numpy()
    hf_argmax = int(np.argmax(base_logits))
    return {
        "splice_seq": captured[splice_layer],
        "captured_layers": captured,
        "hf_next_id": hf_argmax,
        "hf_next_text": token_text(tokenizer, hf_argmax),
        "hf_topk": topk(base_logits, top_k),
        "hf_baseline_ms": elapsed_ms,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sweep live owned-session GPU ownership depths using repo-truth 3 hybrid + 1 full topology"
    )
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--owned-session-worker-bin", default="/mnt/f/git/ds4/qwen36-unified-owned-worker-rocm")
    ap.add_argument("--full-layer-worker-bin", default="/mnt/f/git/ds4/qwen36-gpu-full-layer-worker")
    ap.add_argument("--hybrid-worker-bin", default="/mnt/f/git/ds4/qwen36-unified-owned-worker-rocm")
    ap.add_argument("--fixture-dir", default=str(DEFAULT_FIXTURE_DIR))
    ap.add_argument("--depths", default="8,16,32")
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--hf-layer-progress", action="store_true")
    ap.add_argument("--hf-patched-setup-progress", action="store_true")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    prompt = read_prompt(args)
    depths = [int(part.strip()) for part in args.depths.split(",") if part.strip()]
    fixture_dir = Path(args.fixture_dir)

    print(f"[owned-power2] loading tokenizer from {args.hf}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    print(f"[owned-power2] loading model from {args.hf}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    torch.set_num_threads(min(32, os.cpu_count() or 16))
    token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    hidden = int(model.config.hidden_size)
    print(f"[owned-power2] model loaded prompt_tokens={len(token_ids)} hidden={hidden}")

    results = []
    with tempfile.TemporaryDirectory(prefix="q36_owned_power2_") as td:
        td_path = Path(td)
        for owned_layers in depths:
            full_layers, cycle_chunks, flat_fixtures, hybrid_layers = build_layout(owned_layers, fixture_dir)
            splice_layer = owned_layers - 1
            pre_full_layers = [layer - 1 for layer in full_layers]
            cfg_path = td_path / f"owned_{owned_layers}.cfg"
            write_native_config(
                cfg_path,
                full_layers=full_layers,
                cycle_chunks=cycle_chunks,
                hybrid_worker_bin=args.hybrid_worker_bin,
                full_worker_bin=args.full_layer_worker_bin,
            )
            env = {
                "QWEN36_UNIFIED_FULL_GPU": "1",
                "QWEN36_UNIFIED_HYBRID_GPU_CYCLES": str(len(full_layers)),
            }
            print(
                f"[owned-power2] depth={owned_layers} splice_layer={splice_layer} "
                f"fixtures={len(flat_fixtures)} hybrid_layers={hybrid_layers} full_layers={full_layers}"
            )
            worker = OwnedSessionWorker(args.owned_session_worker_bin, args.gguf, cfg_path, env=env)
            try:
                owned_seq, meta = worker.run_for_token_ids(token_ids, hidden, dump_mode="hidden")
                cycle_pre_owned_seqs = [worker.dump_cycle_pre_hidden(cycle_idx) for cycle_idx in range(len(full_layers))]
                cycle_owned_seqs = [worker.dump_cycle_hidden(cycle_idx) for cycle_idx in range(len(full_layers))]
            finally:
                worker.close()

            hf_base = run_hf_baseline_capture(
                model=model,
                token_ids=token_ids,
                splice_layer=splice_layer,
                capture_layers=full_layers + pre_full_layers,
                tokenizer=tokenizer,
                top_k=args.top_k,
                layer_progress=args.hf_layer_progress,
            )
            patched = run_hf_patched_compare(
                model=model,
                token_ids=token_ids,
                owned_seq=owned_seq,
                splice_layer=splice_layer,
                tokenizer=tokenizer,
                top_k=args.top_k,
                layer_progress=args.hf_layer_progress,
                setup_progress=args.hf_patched_setup_progress,
            )
            splice_summary = row_rmse_summary(owned_seq, hf_base["splice_seq"])
            cycle_summaries = []
            for cycle_idx, full_layer in enumerate(full_layers):
                pre_full_layer = full_layer - 1
                pre_cycle_summary = row_rmse_summary(cycle_pre_owned_seqs[cycle_idx], hf_base["captured_layers"][pre_full_layer])
                cycle_summary = row_rmse_summary(cycle_owned_seqs[cycle_idx], hf_base["captured_layers"][full_layer])
                cycle_summaries.append(
                    {
                        "cycle_idx": cycle_idx,
                        "pre_full_layer": pre_full_layer,
                        "full_layer": full_layer,
                        "pre_seq_rmse": pre_cycle_summary["seq_rmse"],
                        "pre_seq_cosine": pre_cycle_summary["seq_cosine"],
                        "pre_last_row_rmse": pre_cycle_summary["last_row_rmse"],
                        "pre_worst_row": pre_cycle_summary["worst_row"],
                        "seq_rmse": cycle_summary["seq_rmse"],
                        "seq_cosine": cycle_summary["seq_cosine"],
                        "last_row_rmse": cycle_summary["last_row_rmse"],
                        "worst_row": cycle_summary["worst_row"],
                    }
                )
            overlap = topk_overlap(hf_base["hf_topk"], patched["patched_topk"])
            result = {
                "owned_layers": owned_layers,
                "splice_layer": splice_layer,
                "full_layers": full_layers,
                "hybrid_layers": hybrid_layers,
                "fixture_count": len(flat_fixtures),
                "fixtures": flat_fixtures,
                "owned_prefix_ms": meta["worker_ms"],
                "hf_baseline_ms": hf_base["hf_baseline_ms"],
                "hf_patched_ms": patched["hf_patched_ms"],
                "splice_seq_rmse": splice_summary["seq_rmse"],
                "splice_seq_cosine": splice_summary["seq_cosine"],
                "splice_last_row_rmse": splice_summary["last_row_rmse"],
                "splice_worst_row": splice_summary["worst_row"],
                "splice_row_rmses": splice_summary["row_rmses"],
                "cycle_summaries": cycle_summaries,
                "argmax_equal": hf_base["hf_next_id"] == patched["next_id"],
                "baseline_next_id": hf_base["hf_next_id"],
                "baseline_next_text": hf_base["hf_next_text"],
                "patched_next_id": patched["next_id"],
                "patched_next_text": patched["next_text"],
                "topk_overlap": overlap,
                "topk_overlap_rate": float(overlap) / float(max(args.top_k, 1)),
                "hf_topk": hf_base["hf_topk"],
                "patched_topk": patched["patched_topk"],
            }
            results.append(result)
            print(
                f"[owned-power2] depth={owned_layers} argmax_equal={result['argmax_equal']} "
                f"topk_overlap={result['topk_overlap']}/{args.top_k} "
                f"owned_prefix_ms={result['owned_prefix_ms']:.2f} "
                f"splice_seq_rmse={result['splice_seq_rmse']:.8f} "
                f"worst_row={result['splice_worst_row']['row']}:{result['splice_worst_row']['rmse']:.8f}"
            )
            for cycle_summary in cycle_summaries:
                print(
                    f"[owned-power2] depth={owned_layers} cycle={cycle_summary['cycle_idx']} "
                    f"pre_full_layer={cycle_summary['pre_full_layer']} "
                    f"pre_seq_rmse={cycle_summary['pre_seq_rmse']:.8f} "
                    f"pre_worst_row={cycle_summary['pre_worst_row']['row']}:{cycle_summary['pre_worst_row']['rmse']:.8f} "
                    f"full_layer={cycle_summary['full_layer']} "
                    f"seq_rmse={cycle_summary['seq_rmse']:.8f} "
                    f"worst_row={cycle_summary['worst_row']['row']}:{cycle_summary['worst_row']['rmse']:.8f}"
                )

    best = None
    if results:
        best = max(
            results,
            key=lambda item: (
                bool(item["argmax_equal"]),
                float(item["topk_overlap_rate"]),
            ),
        )
        print(
            f"[owned-power2] best depth={best['owned_layers']} "
            f"argmax_equal={best['argmax_equal']} "
            f"topk_overlap={best['topk_overlap']}/{args.top_k}"
        )

    out = {
        "prompt": prompt,
        "prompt_tokens": len(token_ids),
        "results": results,
        "best": best,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[owned-power2] json_out={args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
