#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path


PREFILL_RE = re.compile(
    r"DBG_HYBRID_PREFILL\[(?P<layer>\d+)\] row=(?P<row>\d+) (?P<state>GPU_STATE|CPU_STATE)"
)
SIG_RE = re.compile(
    r"DBG_HYBRID_SIG\[(?P<layer>\d+)\] step=(?P<step>\d+) "
    r"(?P<label>(GPU|CPU)_PREFILL_CONV[0-2]) "
    r"mean=(?P<mean>[-+0-9.eE]+) "
    r"rms=(?P<rms>[-+0-9.eE]+) "
    r"max_abs=(?P<max_abs>[-+0-9.eE]+) "
    r"head=(?P<head>.*)"
)


def parse_head(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part]


def head_rmse(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise RuntimeError(f"head length mismatch: {len(a)} vs {len(b)}")
    if not a:
        return 0.0
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)) / len(a))


def parse_log(path: Path, layer: int) -> dict[int, dict[str, dict[str, float | list[float]]]]:
    rows: dict[int, dict[str, dict[str, float | list[float]]]] = {}
    current_row = None
    current_state = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        prefill_match = PREFILL_RE.search(line)
        if prefill_match:
            if int(prefill_match.group("layer")) != layer:
                current_row = None
                current_state = None
                continue
            current_row = int(prefill_match.group("row"))
            current_state = "GPU" if prefill_match.group("state") == "GPU_STATE" else "CPU"
            rows.setdefault(current_row, {})
            continue

        sig_match = SIG_RE.search(line)
        if not sig_match or current_row is None or current_state is None:
            continue
        if int(sig_match.group("layer")) != layer:
            continue
        label = sig_match.group("label")
        label_state = "GPU" if label.startswith("GPU_") else "CPU"
        if label_state != current_state:
            continue
        conv = label.split("_")[-1]
        rows[current_row][f"{current_state}_{conv}"] = {
            "mean": float(sig_match.group("mean")),
            "rms": float(sig_match.group("rms")),
            "max_abs": float(sig_match.group("max_abs")),
            "head": parse_head(sig_match.group("head")),
        }
    return rows


def summarize_single(rows: dict[int, dict[str, dict[str, float | list[float]]]]) -> dict:
    out_rows = []
    for row in sorted(rows):
        best = None
        for conv_idx in range(3):
            gpu = rows[row].get(f"GPU_CONV{conv_idx}")
            cpu = rows[row].get(f"CPU_CONV{conv_idx}")
            if not gpu or not cpu:
                continue
            rmse = head_rmse(gpu["head"], cpu["head"])  # type: ignore[arg-type]
            rms_delta = abs(float(gpu["rms"]) - float(cpu["rms"]))
            item = {
                "conv": conv_idx,
                "head_rmse": rmse,
                "rms_delta": rms_delta,
                "gpu_rms": float(gpu["rms"]),
                "cpu_rms": float(cpu["rms"]),
            }
            if best is None or item["head_rmse"] > best["head_rmse"]:
                best = item
        if best is not None:
            out_rows.append({"row": row, **best})
    return {
        "rows": out_rows,
        "worst_row": None if not out_rows else max(out_rows, key=lambda item: item["head_rmse"]),
    }


def summarize_cross(
    a_rows: dict[int, dict[str, dict[str, float | list[float]]]],
    b_rows: dict[int, dict[str, dict[str, float | list[float]]]],
) -> dict:
    out_rows = []
    shared_rows = sorted(set(a_rows) & set(b_rows))
    for row in shared_rows:
        row_item = {"row": row}
        for state in ("GPU", "CPU"):
            best = None
            for conv_idx in range(3):
                a_sig = a_rows[row].get(f"{state}_CONV{conv_idx}")
                b_sig = b_rows[row].get(f"{state}_CONV{conv_idx}")
                if not a_sig or not b_sig:
                    continue
                item = {
                    "conv": conv_idx,
                    "head_rmse": head_rmse(a_sig["head"], b_sig["head"]),  # type: ignore[arg-type]
                    "rms_delta": abs(float(a_sig["rms"]) - float(b_sig["rms"])),
                    "a_rms": float(a_sig["rms"]),
                    "b_rms": float(b_sig["rms"]),
                }
                if best is None or item["head_rmse"] > best["head_rmse"]:
                    best = item
            row_item[state.lower()] = best
        out_rows.append(row_item)
    return {
        "rows": out_rows,
        "worst_gpu_row": None if not out_rows else max(
            [item for item in out_rows if item.get("gpu") is not None],
            key=lambda item: item["gpu"]["head_rmse"],
            default=None,
        ),
        "worst_cpu_row": None if not out_rows else max(
            [item for item in out_rows if item.get("cpu") is not None],
            key=lambda item: item["cpu"]["head_rmse"],
            default=None,
        ),
    }


def print_single(label: str, summary: dict) -> None:
    print(f"[prefill-rows] {label}")
    worst = summary["worst_row"]
    if worst is None:
        print("  no rows parsed")
        return
    print(
        f"  worst gpu-vs-cpu row={worst['row']} conv={worst['conv']} "
        f"head_rmse={worst['head_rmse']:.8f} rms_delta={worst['rms_delta']:.8f}"
    )
    for item in summary["rows"]:
        print(
            f"  row={item['row']} conv={item['conv']} "
            f"head_rmse={item['head_rmse']:.8f} rms_delta={item['rms_delta']:.8f}"
        )


def print_cross(summary: dict, label_a: str, label_b: str) -> None:
    print(f"[prefill-rows] {label_a}-vs-{label_b}")
    worst_gpu = summary["worst_gpu_row"]
    worst_cpu = summary["worst_cpu_row"]
    if worst_gpu is not None:
        gpu = worst_gpu["gpu"]
        print(
            f"  worst gpu row={worst_gpu['row']} conv={gpu['conv']} "
            f"head_rmse={gpu['head_rmse']:.8f} rms_delta={gpu['rms_delta']:.8f}"
        )
    if worst_cpu is not None:
        cpu = worst_cpu["cpu"]
        print(
            f"  worst cpu row={worst_cpu['row']} conv={cpu['conv']} "
            f"head_rmse={cpu['head_rmse']:.8f} rms_delta={cpu['rms_delta']:.8f}"
        )
    for item in summary["rows"]:
        parts = [f"  row={item['row']}"]
        if item.get("gpu") is not None:
            gpu = item["gpu"]
            parts.append(
                f"gpu(conv={gpu['conv']} head_rmse={gpu['head_rmse']:.8f} rms_delta={gpu['rms_delta']:.8f})"
            )
        if item.get("cpu") is not None:
            cpu = item["cpu"]
            parts.append(
                f"cpu(conv={cpu['conv']} head_rmse={cpu['head_rmse']:.8f} rms_delta={cpu['rms_delta']:.8f})"
            )
        print(" ".join(parts))


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize blk0 prefill row-state drift from DBG_HYBRID_PREFILL logs")
    ap.add_argument("--log-a", required=True)
    ap.add_argument("--log-b")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    log_a = Path(args.log_a)
    rows_a = parse_log(log_a, args.layer)
    single_a = summarize_single(rows_a)
    print_single(args.label_a, single_a)

    out = {
        "layer": args.layer,
        "log_a": str(log_a),
        "label_a": args.label_a,
        "single_a": single_a,
    }

    if args.log_b:
        log_b = Path(args.log_b)
        rows_b = parse_log(log_b, args.layer)
        single_b = summarize_single(rows_b)
        cross = summarize_cross(rows_a, rows_b)
        print_single(args.label_b, single_b)
        print_cross(cross, args.label_a, args.label_b)
        out.update(
            {
                "log_b": str(log_b),
                "label_b": args.label_b,
                "single_b": single_b,
                "cross": cross,
            }
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[prefill-rows] json_out={args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
