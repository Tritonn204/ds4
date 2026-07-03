#!/usr/bin/env python3
import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


ROOT_MAGIC = b"Q36ROOT1"
ROOT_PROBE_MAGIC = b"Q36RHF01"
ROOT_MAGIC_LEN = 8


def read_prompt(args) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def patch_root_hidden_pre(src_path: str, dst_path: str, hidden_pre: np.ndarray) -> None:
    data = bytearray(Path(src_path).read_bytes())
    magic = bytes(data[:ROOT_MAGIC_LEN])
    if magic not in (ROOT_MAGIC, ROOT_PROBE_MAGIC):
        raise ValueError(f"unexpected root fixture magic in {src_path}")
    hidden, topk, prompt_tokens = struct.unpack_from("<III", data, ROOT_MAGIC_LEN)
    if hidden_pre.shape != (hidden,):
        raise ValueError(f"hidden_pre shape {hidden_pre.shape} does not match fixture hidden={hidden}")
    offset = ROOT_MAGIC_LEN + 12
    data[offset:offset + hidden * 4] = np.ascontiguousarray(hidden_pre, dtype=np.float32).tobytes(order="C")
    Path(dst_path).write_bytes(data)


def parse_c_root_output(stdout: str):
    result = {"lines": stdout.splitlines(), "entries": []}
    for line in result["lines"]:
        if line.startswith("selected_logits_cosine:"):
            result["selected_logits_cosine"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("selected_logits_rmse:"):
            result["selected_logits_rmse"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("final_norm_cosine:"):
            result["final_norm_cosine"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("final_norm_rmse:"):
            result["final_norm_rmse"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("logit["):
            parts = line.strip().split()
            idx = int(parts[0][6:-2])
            token_id = int(parts[1].split("=")[1])
            ref = float(parts[2].split("=")[1])
            got = float(parts[3].split("=")[1])
            result["entries"].append({"rank": idx, "token_id": token_id, "ref": ref, "got": got})
    result["entries_sorted_by_got"] = sorted(result["entries"], key=lambda x: x["got"], reverse=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Check selected root logits for a C-owned Qwen3.6 prefix (token_embd + owned layers)")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prefix-fixture", required=True)
    ap.add_argument("--c-prefix-bin", default="./qwen36-c-prefix-q8-chain")
    ap.add_argument("--c-root-bin", default="./qwen36-c-root-q8")
    ap.add_argument("--root-exporter", default="/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_root_hf_probe.py")
    ap.add_argument("--fixture", action="append", default=[])
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    if not args.fixture:
        ap.error("at least one --fixture is required")

    prompt = read_prompt(args)
    print(f"[prefix-root] prompt_chars={len(prompt)}")
    print(f"[prefix-root] fixture_count={len(args.fixture)}")
    for i, fixture in enumerate(args.fixture):
        print(f"[prefix-root] fixture[{i}]={fixture}")

    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)

    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp_hidden, \
         tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_root, \
         tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_root_patched:
        hidden_path = tmp_hidden.name
        root_path = tmp_root.name
        root_patched_path = tmp_root_patched.name

    try:
        prefix_cmd = [args.c_prefix_bin, args.gguf, args.prefix_fixture, *args.fixture, "--dump-last", hidden_path]
        print(f"[prefix-root] running c-prefix: {' '.join(prefix_cmd)}")
        prefix_proc = subprocess.run(prefix_cmd, capture_output=True, text=True, check=True)
        hidden = np.fromfile(hidden_path, dtype=np.float32)
        print(f"[prefix-root] c-prefix complete hidden_f32={hidden.shape[0]}")

        export_cmd = [
            "python3", args.root_exporter,
            "--hf", args.hf,
            "--top-k", str(args.top_k),
            "--out", root_path,
        ]
        if args.prompt_file:
            export_cmd.extend(["--prompt-file", args.prompt_file])
        else:
            export_cmd.extend(["--prompt", args.prompt])
        print(f"[prefix-root] exporting root fixture: {' '.join(export_cmd)}")
        export_proc = subprocess.run(export_cmd, capture_output=True, text=True, check=True)

        patch_root_hidden_pre(root_path, root_patched_path, hidden)
        print(f"[prefix-root] patched root fixture hidden_pre -> {root_patched_path}")

        root_cmd = [args.c_root_bin, args.gguf, root_patched_path]
        print(f"[prefix-root] running c-root: {' '.join(root_cmd)}")
        root_proc = subprocess.run(root_cmd, capture_output=True, text=True, check=True)
        parsed = parse_c_root_output(root_proc.stdout)
        best = parsed["entries_sorted_by_got"][0]
        best_text = tokenizer.decode([best["token_id"]], clean_up_tokenization_spaces=False)

        result = {
            "prompt": prompt,
            "prefix_report": prefix_proc.stdout,
            "root_export_report": export_proc.stdout,
            "root_report": root_proc.stdout,
            "selected_logits_cosine": parsed.get("selected_logits_cosine"),
            "selected_logits_rmse": parsed.get("selected_logits_rmse"),
            "final_norm_cosine": parsed.get("final_norm_cosine"),
            "final_norm_rmse": parsed.get("final_norm_rmse"),
            "best_selected_token": {
                "id": best["token_id"],
                "text": best_text,
                "got": best["got"],
                "ref": best["ref"],
            },
            "entries_by_got": parsed["entries_sorted_by_got"],
        }

        print(f"selected_logits_cosine: {result['selected_logits_cosine']:.8f}")
        print(f"selected_logits_rmse: {result['selected_logits_rmse']:.8f}")
        print(f"best_selected_token: id={best['token_id']} text={json.dumps(best_text)} got={best['got']:.6f} ref={best['ref']:.6f}")

        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"json_out: {args.json_out}")
        return 0
    finally:
        Path(hidden_path).unlink(missing_ok=True)
        Path(root_path).unlink(missing_ok=True)
        Path(root_patched_path).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
