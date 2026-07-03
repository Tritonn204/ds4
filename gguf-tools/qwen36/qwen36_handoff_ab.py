#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


BOUNDARY_RE = re.compile(
    r"DBG_BOUNDARY step=(?P<step>\d+) (?P<label>[A-Z0-9_]+) "
    r"rmse=(?P<rmse>[-+0-9.eE]+) max_diff=(?P<max_diff>[-+0-9.eE]+) "
    r"max_idx=(?P<max_idx>\d+) mean_abs=(?P<mean_abs>[-+0-9.eE]+)"
)
FULL_RE = re.compile(
    r"DBG_FULL\[(?P<layer>\d+)\] step=(?P<step>\d+) (?P<label>[A-Z0-9_]+) "
    r"rmse=(?P<rmse>[-+0-9.eE]+) max_diff=(?P<max_diff>[-+0-9.eE]+) "
    r"max_idx=(?P<max_idx>\d+) mean_abs=(?P<mean_abs>[-+0-9.eE]+)"
)
BOUNDARY_SIG_RE = re.compile(
    r"DBG_BOUNDARY_SIG step=(?P<step>\d+) (?P<label>[A-Z0-9_]+) "
    r"mean=(?P<mean>[-+0-9.eE]+) rms=(?P<rms>[-+0-9.eE]+) max_abs=(?P<max_abs>[-+0-9.eE]+) "
    r"head=(?P<head>.*)"
)
TOPK_RE = re.compile(r"DBG_FULL\[(?P<layer>\d+)\] step=(?P<step>\d+) TOPK cpu=(?P<cpu>.+) gpu=(?P<gpu>.+)")
SIG_RE = re.compile(
    r"DBG_FULL_SIG\[(?P<layer>\d+)\] step=(?P<step>\d+) (?P<label>[A-Z0-9_]+) "
    r"mean=(?P<mean>[-+0-9.eE]+) rms=(?P<rms>[-+0-9.eE]+) max_abs=(?P<max_abs>[-+0-9.eE]+) "
    r"head=(?P<head>.*)"
)
DECODED_RE = re.compile(r'decoded_so_far="(?P<text>.*)"')


def parse_topk_list(text: str) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for item in text.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        idx_s, score_s = item.split(":", 1)
        try:
            out.append((int(idx_s), float(score_s)))
        except ValueError:
            continue
    return out


def parse_float_list(text: str) -> list[float]:
    out: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(float(item))
        except ValueError:
            continue
    return out


def run_probe(args, cycles: int, output_prefix: Path) -> dict:
    json_out = output_prefix.with_suffix(".json")
    log_out = output_prefix.with_suffix(".log")
    cmd = [
        sys.executable,
        "/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_full_layer_probe.py",
        "--probe",
        "handoff-debug",
        "--hf",
        args.hf,
        "--gguf",
        args.gguf,
        "--owned-session-worker-bin",
        args.owned_session_worker_bin,
        "--c-bin",
        args.c_bin,
        "--c-bin-worker-bin",
        args.c_bin_worker_bin,
        "--full-layer-worker-bin",
        args.full_layer_worker_bin,
        "--prompt-file",
        args.prompt_file,
        "--json-out",
        str(json_out),
        "--splice-layer",
        str(args.splice_layer),
        "--n-predict",
        str(args.n_predict),
        "--top-k",
        str(args.top_k),
        "--hybrid-gpu-cycles",
        str(cycles),
        "--full-debug-layer",
        str(args.full_debug_layer),
        "--step-start",
        str(args.step_start),
        "--step-end",
        str(args.step_end),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_out.write_text(proc.stdout, encoding="utf-8")
    return {
        "returncode": proc.returncode,
        "log_path": str(log_out),
        "json_path": str(json_out),
        "text": proc.stdout,
    }


def parse_metrics(text: str) -> dict:
    out: dict[str, dict] = {
        "boundary": {},
        "boundary_sig": {},
        "full": {},
        "topk": {},
        "sig": {},
        "decoded_last": None,
    }
    for line in text.splitlines():
        m = BOUNDARY_RE.search(line)
        if m:
            step = int(m.group("step"))
            label = m.group("label")
            out["boundary"].setdefault(step, {})[label] = {
                "rmse": float(m.group("rmse")),
                "max_diff": float(m.group("max_diff")),
                "max_idx": int(m.group("max_idx")),
                "mean_abs": float(m.group("mean_abs")),
            }
            continue
        m = FULL_RE.search(line)
        if m:
            layer = int(m.group("layer"))
            step = int(m.group("step"))
            label = m.group("label")
            out["full"].setdefault(layer, {}).setdefault(step, {})[label] = {
                "rmse": float(m.group("rmse")),
                "max_diff": float(m.group("max_diff")),
                "max_idx": int(m.group("max_idx")),
                "mean_abs": float(m.group("mean_abs")),
            }
            continue
        m = BOUNDARY_SIG_RE.search(line)
        if m:
            step = int(m.group("step"))
            label = m.group("label")
            out["boundary_sig"].setdefault(step, {})[label] = {
                "mean": float(m.group("mean")),
                "rms": float(m.group("rms")),
                "max_abs": float(m.group("max_abs")),
                "head": parse_float_list(m.group("head").strip()),
            }
            continue
        m = TOPK_RE.search(line)
        if m:
            layer = int(m.group("layer"))
            step = int(m.group("step"))
            out["topk"].setdefault(layer, {})[step] = {
                "cpu": m.group("cpu").strip(),
                "gpu": m.group("gpu").strip(),
            }
            continue
        m = SIG_RE.search(line)
        if m:
            layer = int(m.group("layer"))
            step = int(m.group("step"))
            label = m.group("label")
            out["sig"].setdefault(layer, {}).setdefault(step, {})[label] = {
                "mean": float(m.group("mean")),
                "rms": float(m.group("rms")),
                "max_abs": float(m.group("max_abs")),
                "head": parse_float_list(m.group("head").strip()),
            }
            continue
        m = DECODED_RE.search(line)
        if m:
            out["decoded_last"] = m.group("text")
    return out


def summarize_pair(parsed0: dict, parsed1: dict, full_layer: int) -> str:
    lines: list[str] = []
    lines.append(f"[handoff-ab] full_layer={full_layer}")
    lines.append("[handoff-ab] boundary compare: rmse(c0) -> rmse(c1)")
    all_b_steps = sorted(set(parsed0["boundary"]) | set(parsed1["boundary"]))
    for step in all_b_steps:
        labels = sorted(set(parsed0["boundary"].get(step, {})) | set(parsed1["boundary"].get(step, {})))
        for label in labels:
            a = parsed0["boundary"].get(step, {}).get(label)
            b = parsed1["boundary"].get(step, {}).get(label)
            if not a or not b:
                continue
            lines.append(
                f"step={step} boundary/{label} rmse0={a['rmse']:.8f} rmse1={b['rmse']:.8f}"
            )
    lines.append("[handoff-ab] boundary signature compare: c0 -> c1")
    bs0 = parsed0["boundary_sig"]
    bs1 = parsed1["boundary_sig"]
    bs_rows: list[tuple[float, int, str, float, float]] = []
    for step in sorted(set(bs0) | set(bs1)):
        a_step = bs0.get(step, {})
        b_step = bs1.get(step, {})
        for label in sorted(set(a_step) | set(b_step)):
            a = a_step.get(label)
            b = b_step.get(label)
            if not a or not b:
                continue
            head_n = min(len(a["head"]), len(b["head"]))
            if head_n:
                head_rmse = (
                    sum((a["head"][i] - b["head"][i]) ** 2 for i in range(head_n)) / head_n
                ) ** 0.5
            else:
                head_rmse = 0.0
            bs_rows.append((head_rmse, step, label, a["rms"], b["rms"]))
            lines.append(
                f"step={step} boundary_sig/{label} head_rmse={head_rmse:.8f} rms0={a['rms']:.8f} rms1={b['rms']:.8f}"
            )
    bs_rows.sort(reverse=True)
    lines.append("[handoff-ab] largest boundary signature deltas")
    for head_rmse, step, label, rms0, rms1 in bs_rows[:20]:
        lines.append(
            f"step={step} boundary_sig/{label} head_rmse={head_rmse:.8f} rms0={rms0:.8f} rms1={rms1:.8f}"
        )
    lines.append(f"[handoff-ab] full layer {full_layer} compare: rmse(c0) -> rmse(c1)")
    full0 = parsed0["full"].get(full_layer, {})
    full1 = parsed1["full"].get(full_layer, {})
    all_f_steps = sorted(set(full0) | set(full1))
    for step in all_f_steps:
        labels = sorted(set(full0.get(step, {})) | set(full1.get(step, {})))
        for label in labels:
            a = full0.get(step, {}).get(label)
            b = full1.get(step, {}).get(label)
            if not a or not b:
                continue
            lines.append(
                f"step={step} full/{label} rmse0={a['rmse']:.8f} rmse1={b['rmse']:.8f}"
            )
    delta_rows: list[tuple[float, int, str, float, float]] = []
    for step in all_f_steps:
        labels = sorted(set(full0.get(step, {})) | set(full1.get(step, {})))
        for label in labels:
            a = full0.get(step, {}).get(label)
            b = full1.get(step, {}).get(label)
            if not a or not b:
                continue
            delta_rows.append((abs(b["rmse"] - a["rmse"]), step, label, a["rmse"], b["rmse"]))
    delta_rows.sort(reverse=True)
    lines.append(f"[handoff-ab] largest c1-vs-c0 rmse deltas in full layer {full_layer}")
    for delta, step, label, rmse0, rmse1 in delta_rows[:20]:
        lines.append(
            f"step={step} full/{label} delta={delta:.8f} rmse0={rmse0:.8f} rmse1={rmse1:.8f}"
        )
    topk0 = parsed0["topk"].get(full_layer, {})
    topk1 = parsed1["topk"].get(full_layer, {})
    lines.append(f"[handoff-ab] topk cpu(c0)-vs-cpu(c1) compare for full layer {full_layer}")
    for step in sorted(set(topk0) | set(topk1)):
        a = topk0.get(step)
        b = topk1.get(step)
        if not a or not b:
            continue
        a_list = parse_topk_list(a["cpu"])
        b_list = parse_topk_list(b["cpu"])
        a_ids = [idx for idx, _ in a_list]
        b_ids = [idx for idx, _ in b_list]
        overlap = len(set(a_ids) & set(b_ids))
        lines.append(
            f"step={step} topk_overlap={overlap}/8 c0={a['cpu']} c1={b['cpu']}"
        )
    sig0 = parsed0["sig"].get(full_layer, {})
    sig1 = parsed1["sig"].get(full_layer, {})
    lines.append(f"[handoff-ab] signature compare for full layer {full_layer}")
    sig_rows: list[tuple[float, int, str, float, float]] = []
    for step in sorted(set(sig0) | set(sig1)):
        a_step = sig0.get(step, {})
        b_step = sig1.get(step, {})
        for label in sorted(set(a_step) | set(b_step)):
            a = a_step.get(label)
            b = b_step.get(label)
            if not a or not b:
                continue
            head_n = min(len(a["head"]), len(b["head"]))
            if head_n:
                head_rmse = (
                    sum((a["head"][i] - b["head"][i]) ** 2 for i in range(head_n)) / head_n
                ) ** 0.5
            else:
                head_rmse = 0.0
            sig_rows.append((head_rmse, step, label, a["rms"], b["rms"]))
    sig_rows.sort(reverse=True)
    for head_rmse, step, label, rms0, rms1 in sig_rows[:20]:
        lines.append(
            f"step={step} sig/{label} head_rmse={head_rmse:.8f} rms0={rms0:.8f} rms1={rms1:.8f}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run cycles=0 vs cycles=1 handoff tensor debug and summarize.")
    ap.add_argument("--hf", default="/mnt/e/tensors/Qwen3.6-35B-A3B")
    ap.add_argument("--gguf", default=(
        "/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/"
        "snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-Q8_0.gguf"
    ))
    ap.add_argument("--owned-session-worker-bin", default="/mnt/f/git/ds4/qwen36-unified-owned-worker-rocm")
    ap.add_argument("--c-bin", default="/mnt/f/git/ds4/qwen36-c-prefix-q8-chain-live")
    ap.add_argument("--c-bin-worker-bin", default="/mnt/f/git/ds4/qwen36-live-contract-worker")
    ap.add_argument("--full-layer-worker-bin", default="/mnt/f/git/ds4/qwen36-gpu-full-layer-worker")
    ap.add_argument("--prompt-file", default="/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_oracle_prompts/code_review.txt")
    ap.add_argument("--output-prefix", default="/tmp/qwen36_handoff_ab")
    ap.add_argument("--splice-layer", type=int, default=39)
    ap.add_argument("--n-predict", type=int, default=96)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--full-debug-layer", type=int, default=3)
    ap.add_argument("--step-start", type=int, default=78)
    ap.add_argument("--step-end", type=int, default=86)
    ap.add_argument("--cycles-a", type=int, default=0)
    ap.add_argument("--cycles-b", type=int, default=1)
    args = ap.parse_args()

    prefix = Path(args.output_prefix)
    run_a = run_probe(args, args.cycles_a, Path(f"{prefix}_c{args.cycles_a}"))
    run_b = run_probe(args, args.cycles_b, Path(f"{prefix}_c{args.cycles_b}"))

    parsed_a = parse_metrics(run_a["text"])
    parsed_b = parse_metrics(run_b["text"])
    summary_text = summarize_pair(parsed_a, parsed_b, args.full_debug_layer)
    summary_path = Path(f"{prefix}_summary.txt")
    summary_json_path = Path(f"{prefix}_summary.json")
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    summary_json_path.write_text(
        json.dumps(
            {
                "cycles_a": args.cycles_a,
                "cycles_b": args.cycles_b,
                "run_a": {k: v for k, v in run_a.items() if k != "text"},
                "run_b": {k: v for k, v in run_b.items() if k != "text"},
                "parsed_a": parsed_a,
                "parsed_b": parsed_b,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(summary_text)
    print(f"[handoff-ab] summary_txt={summary_path}")
    print(f"[handoff-ab] summary_json={summary_json_path}")
    print(f"[handoff-ab] log_a={run_a['log_path']}")
    print(f"[handoff-ab] log_b={run_b['log_path']}")
    return 0 if run_a["returncode"] == 0 and run_b["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
