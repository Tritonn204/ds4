#!/usr/bin/env python3
import argparse
import re
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

from qwen36_hybrid_prefix_tail_greedy import OwnedSessionWorker, read_prompt
from qwen36_power2_owned_session_validator import DEFAULT_FIXTURE_DIR, build_layout, write_native_config


DBG_RE = re.compile(
    r"=== DBG_FULL\[(?P<layer>\d+)\] step=(?P<step>\d+) (?P<label>[A-Z_]+) "
    r"rmse=(?P<rmse>[0-9.]+) max_diff=(?P<max_diff>[0-9.]+) max_idx=(?P<max_idx>\d+) "
    r"mean_abs=(?P<mean_abs>[0-9.]+)"
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run blk.3 full-layer prefill stage diffs row-by-row")
    ap.add_argument("--hf", default="/mnt/e/tensors/Qwen3.6-35B-A3B")
    ap.add_argument(
        "--gguf",
        default="/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-Q8_0.gguf",
    )
    ap.add_argument("--owned-session-worker-bin", default="/mnt/f/git/ds4/qwen36-unified-owned-worker-rocm")
    ap.add_argument("--full-layer-worker-bin", default="/mnt/f/git/ds4/qwen36-gpu-full-layer-worker")
    ap.add_argument("--hybrid-worker-bin", default="/mnt/f/git/ds4/qwen36-unified-owned-worker-rocm")
    ap.add_argument("--fixture-dir", default=str(DEFAULT_FIXTURE_DIR))
    ap.add_argument("--owned-layers", type=int, default=8)
    ap.add_argument("--full-layer", type=int, default=3)
    ap.add_argument("--row-start", type=int, default=0)
    ap.add_argument("--row-end", type=int, default=20)
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    args = ap.parse_args()

    prompt = read_prompt(args)
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    hidden = 2048

    fixture_dir = Path(args.fixture_dir)
    full_layers, cycle_chunks, _flat_fixtures, hybrid_layers = build_layout(args.owned_layers, fixture_dir)
    if args.full_layer not in full_layers:
        raise RuntimeError(f"full layer {args.full_layer} not in owned layout {full_layers}")

    with tempfile.TemporaryDirectory(prefix="q36_blk3_full_dbg_") as td:
        cfg_path = Path(td) / "owned.cfg"
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
            "QWEN36_DBG_FULL_PREFILL": "1",
            "QWEN36_DBG_FULL_LAYER": str(args.full_layer),
            "QWEN36_DBG_FULL_STEP_START": str(args.row_start),
            "QWEN36_DBG_FULL_STEP_END": str(args.row_end),
        }
        print(
            f"[blk3-full-debug] owned_layers={args.owned_layers} full_layer={args.full_layer} "
            f"hybrid_layers={hybrid_layers} full_layers={full_layers}"
        )
        worker = OwnedSessionWorker(args.owned_session_worker_bin, args.gguf, cfg_path, env=env)
        try:
            worker.run_for_token_ids(token_ids, hidden, dump_mode="hidden")
            stderr_lines = worker.stderr_lines()
        finally:
            worker.close()

    rows: dict[int, list[dict]] = {}
    for line in stderr_lines:
        m = DBG_RE.search(line)
        if not m:
            continue
        if int(m.group("layer")) != args.full_layer:
            continue
        step = int(m.group("step"))
        rows.setdefault(step, []).append(
            {
                "label": m.group("label"),
                "rmse": float(m.group("rmse")),
                "max_diff": float(m.group("max_diff")),
                "max_idx": int(m.group("max_idx")),
                "mean_abs": float(m.group("mean_abs")),
            }
        )

    if not rows:
        print("[blk3-full-debug] no DBG_FULL rows captured")
        return 1

    print("[blk3-full-debug] worst stage per row")
    overall = None
    for step in sorted(rows):
        worst = max(rows[step], key=lambda item: item["rmse"])
        print(
            f"  row={step} worst={worst['label']} rmse={worst['rmse']:.8f} "
            f"max_diff={worst['max_diff']:.8f} max_idx={worst['max_idx']}"
        )
        if overall is None or worst["rmse"] > overall["rmse"]:
            overall = {"step": step, **worst}

    assert overall is not None
    print(
        f"[blk3-full-debug] overall worst row={overall['step']} stage={overall['label']} "
        f"rmse={overall['rmse']:.8f} max_diff={overall['max_diff']:.8f} max_idx={overall['max_idx']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
