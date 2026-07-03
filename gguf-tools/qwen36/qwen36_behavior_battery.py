#!/usr/bin/env python3
import argparse
import json
import signal
import subprocess
import tempfile
from pathlib import Path


PROMPT_SPECS = [
    ("hello_world", 1),
    ("quant_report", 32),
    ("code_review", 48),
    ("patch_plan", 64),
    ("router_dispatch", 24),
    ("hybrid_bridge", 24),
    ("late_layers", 24),
    ("final_selection", 24),
    ("logit_commit", 24),
]


def first_false_step(steps):
    for step in steps:
        if step.get("argmax_equal") is False:
            return int(step["step"])
    return None


def run_live(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            tail.append(line)
            if len(tail) > 400:
                tail = tail[-400:]
        returncode = proc.wait()
        return returncode, "".join(tail)
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        try:
            returncode = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            returncode = proc.wait()
        raise KeyboardInterrupt() from None


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a serial behavioral parity battery for the Qwen3.6 owned-prefix bridge")
    ap.add_argument("--runner", default="/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_hybrid_prefix_tail_greedy.py")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--c-bin", required=True)
    ap.add_argument("--c-bin-prefix-flag", action="store_true")
    ap.add_argument("--prefix-seq-bin")
    ap.add_argument("--prefix-seq-fixture")
    ap.add_argument("--prefix-seq-dynamic", action="store_true")
    ap.add_argument("--full-layer-bin")
    ap.add_argument("--full-layer", action="append", type=int, default=None)
    ap.add_argument("--fixture", action="append", default=[])
    ap.add_argument("--prompt-dir", default="/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_oracle_prompts")
    ap.add_argument("--only")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--compare-hf-every", type=int, default=1)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    wanted = None
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
    specs = [(name, n) for name, n in PROMPT_SPECS if wanted is None or name in wanted]
    if not specs:
        raise SystemExit("no prompts selected")
    if not args.fixture:
        raise SystemExit("at least one --fixture is required")

    results = []
    with tempfile.TemporaryDirectory(prefix="q36_behavior_battery_") as td:
        td_path = Path(td)
        for name, n_predict in specs:
            prompt_path = Path(args.prompt_dir) / f"{name}.txt"
            if not prompt_path.exists():
                raise SystemExit(f"missing prompt file: {prompt_path}")
            run_json = td_path / f"{name}.json"
            cmd = [
                "python3",
                args.runner,
                "--hf", args.hf,
                "--gguf", args.gguf,
                "--c-bin", args.c_bin,
                "--prompt-file", str(prompt_path),
                "--n-predict", str(n_predict),
                "--top-k", str(args.top_k),
                "--compare-hf",
                "--compare-hf-every", str(args.compare_hf_every),
                "--json-out", str(run_json),
            ]
            if args.c_bin_prefix_flag:
                cmd.append("--c-bin-prefix-flag")
            if args.prefix_seq_bin:
                cmd.extend(["--prefix-seq-bin", args.prefix_seq_bin])
            if args.prefix_seq_fixture:
                cmd.extend(["--prefix-seq-fixture", args.prefix_seq_fixture])
            if args.prefix_seq_dynamic:
                cmd.append("--prefix-seq-dynamic")
            if args.full_layer_bin:
                cmd.extend(["--full-layer-bin", args.full_layer_bin])
                full_layers = args.full_layer if args.full_layer is not None else [3]
                for layer in full_layers:
                    cmd.extend(["--full-layer", str(layer)])
            for fixture in args.fixture:
                cmd.extend(["--fixture", fixture])

            print(f"[battery] prompt={name} n_predict={n_predict}")
            print(f"[battery] cmd={' '.join(cmd)}")
            returncode, stdout_tail = run_live(cmd)
            if returncode != 0:
                rec = {
                    "name": name,
                    "n_predict": n_predict,
                    "ok": False,
                    "returncode": returncode,
                    "stdout_tail": stdout_tail[-4000:],
                }
                results.append(rec)
                print(f"[battery] prompt={name} failed returncode={returncode}")
                continue

            data = json.loads(run_json.read_text(encoding="utf-8"))
            steps = data.get("steps", [])
            compared = [s for s in steps if "argmax_equal" in s]
            all_equal = bool(compared) and all(s.get("argmax_equal") is True for s in compared)
            mismatch = first_false_step(compared)
            rec = {
                "name": name,
                "n_predict": n_predict,
                "ok": True,
                "compared_steps": len(compared),
                "all_equal": all_equal,
                "first_mismatch_step": mismatch,
                "generated_text": data.get("generated_text", ""),
                "stdout_tail": stdout_tail[-4000:],
            }
            results.append(rec)
            print(
                f"[battery] prompt={name} all_equal={all_equal} "
                f"compared_steps={rec['compared_steps']} first_mismatch_step={mismatch}"
            )

    summary = {
        "prompt_count": len(results),
        "pass_count": sum(1 for r in results if r.get("all_equal") is True),
        "fail_count": sum(1 for r in results if r.get("all_equal") is not True),
        "results": results,
    }
    print(f"[battery] pass_count={summary['pass_count']} fail_count={summary['fail_count']}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")
    return 0 if summary["fail_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
