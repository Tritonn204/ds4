#!/usr/bin/env python3
import argparse
import contextlib
import json
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import create_causal_mask


PREFIX_MAGIC = b"Q36PFX01"


def read_seq_hidden(path: Path, hidden: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size == 0 or arr.size % hidden != 0:
        raise RuntimeError(f"hidden dump size {arr.size} not divisible by hidden={hidden}")
    seq_len = arr.size // hidden
    return arr.reshape(seq_len, hidden)


@contextlib.contextmanager
def hf_layer_progress(model, enabled: bool, label: str):
    if not enabled:
        yield
        return
    layer_times_ms: dict[int, float] = {}
    handles = []

    def make_pre(i: int):
        def _pre(_mod, _inp):
            layer_times_ms[i] = time.perf_counter()
        return _pre

    def make_post(i: int):
        def _post(_mod, _inp, _out):
            t0 = layer_times_ms.get(i)
            if t0 is None:
                return
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[residence] {label} layer={i} ms={dt_ms:.2f}", flush=True)
        return _post

    try:
        for i, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_pre_hook(make_pre(i)))
            handles.append(layer.register_forward_hook(make_post(i)))
        yield
    finally:
        for h in handles:
            h.remove()


def topk(logits: np.ndarray, k: int) -> list[dict]:
    idx = np.argpartition(logits, -k)[-k:]
    idx = idx[np.argsort(logits[idx])[::-1]]
    return [{"id": int(i), "logit": float(logits[i])} for i in idx]


class WorkerProc:
    def __init__(self, argv: list[str]):
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(Path(argv[0]).resolve().parent),
        )
        ready = self._read_line()
        if not ready.startswith("READY"):
            raise RuntimeError(f"worker failed to start: {ready}")

    def _read_line(self) -> str:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            assert self.proc.stderr is not None
            err = self.proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"worker exited unexpectedly; stderr={err}")
        return line.decode("utf-8", errors="replace").strip()

    def _write(self, data: bytes) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def prefill_prefix_bin(self, token_ids: list[int], hidden: int) -> None:
        tok = np.asarray(token_ids, dtype=np.uint32)
        seq = np.zeros((len(token_ids), hidden), dtype=np.float32)
        self._write(f"PREFILL_PREFIX_BIN {len(token_ids)} {hidden}\n".encode("utf-8"))
        self._write(tok.tobytes(order="C"))
        self._write(seq.tobytes(order="C"))
        line = self._read_line()
        if not line.startswith("PREFILL_OK "):
            raise RuntimeError(f"prefix prefill failed: {line}")

    def prefill_seq_bin(self, seq_len: int, hidden: int) -> None:
        seq = np.zeros((seq_len, hidden), dtype=np.float32)
        self._write(f"PREFILL_SEQ_BIN {seq_len} {hidden}\n".encode("utf-8"))
        self._write(seq.tobytes(order="C"))
        line = self._read_line()
        if not line.startswith("PREFILL_OK "):
            raise RuntimeError(f"seq prefill failed: {line}")

    def reset(self) -> None:
        self._write(b"RESET\n")
        line = self._read_line()
        if line != "OK":
            raise RuntimeError(f"reset failed: {line}")

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            self._write(b"QUIT\n")
            _ = self._read_line()
        except Exception:
            pass
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()


def split_cycle_fixtures(fixtures: list[str], n_cycles: int, has_prefix_worker: bool) -> list[list[str]]:
    out: list[list[str]] = []
    pos = 0
    for cycle_idx in range(n_cycles):
        want = 2 if has_prefix_worker and cycle_idx == 0 else 3
        chunk = fixtures[pos:pos + want]
        if len(chunk) != want:
            raise RuntimeError(f"fixture split failed for cycle {cycle_idx}: expected {want}, got {len(chunk)}")
        out.append(chunk)
        pos += want
    if pos != len(fixtures):
        raise RuntimeError(f"fixture split left trailing fixtures: used {pos}, total {len(fixtures)}")
    return out


def run_hf_patched_tail(*, model, tokenizer, token_ids: list[int], owned_seq: np.ndarray, splice_layer: int, top_k: int, layer_progress: bool) -> dict:
    text_model = model.model
    device = text_model.norm.weight.device
    hidden_dtype = text_model.norm.weight.dtype
    hidden_states = torch.from_numpy(np.array(owned_seq, dtype=np.float32, copy=True)).unsqueeze(0).to(device=device, dtype=hidden_dtype)
    batch, seq_len, _ = hidden_states.shape
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).view(1, 1, -1).expand(4, batch, -1)
    text_position_ids = position_ids[0]
    rope_position_ids = position_ids[1:]
    causal_mask = create_causal_mask(
        config=text_model.config,
        inputs_embeds=hidden_states,
        attention_mask=None,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    linear_attn_mask = text_model._update_linear_attn_mask(None, None)
    position_embeddings = text_model.rotary_emb(hidden_states, rope_position_ids)

    print("[residence] running hf patched tail")
    t0 = time.perf_counter()
    with hf_layer_progress(model, layer_progress, "hf_tail"), torch.inference_mode():
        for layer_idx in range(splice_layer + 1, len(text_model.layers)):
            decoder_layer = text_model.layers[layer_idx]
            layer_mask = linear_attn_mask if text_model.config.layer_types[layer_idx] == "linear_attention" else causal_mask
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        hidden_states = text_model.norm(hidden_states)
        logits = model.lm_head(hidden_states[:, -1:, :])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logits_np = logits[0, -1].detach().float().cpu().numpy()
    next_id = int(np.argmax(logits_np))
    return {
        "hf_patched_ms": elapsed_ms,
        "next_id": next_id,
        "next_text": tokenizer.decode([next_id], clean_up_tokenization_spaces=False),
        "topk": topk(logits_np, top_k),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--owned-seq-f32", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--splice-layer", type=int, required=True)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--hf-tail-layer-progress", action="store_true")
    ap.add_argument("--prefix-seq-worker-bin")
    ap.add_argument("--prefix-seq-fixture")
    ap.add_argument("--c-bin-worker-bin")
    ap.add_argument("--full-layer-worker-bin")
    ap.add_argument("--fixture", action="append", default=[])
    ap.add_argument("--full-layer", action="append", type=int, default=[])
    ap.add_argument("--spawn-prefix", action="store_true")
    ap.add_argument("--spawn-hybrid", action="store_true")
    ap.add_argument("--spawn-full", action="store_true")
    ap.add_argument("--prefill-workers", action="store_true")
    ap.add_argument("--reset-workers", action="store_true")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    print(f"[residence] loading tokenizer from {args.hf}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    print(f"[residence] loading model from {args.hf}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    hidden = int(model.config.hidden_size)
    print(f"[residence] model loaded hidden={hidden} prompt_tokens={len(token_ids)}")
    owned_seq = read_seq_hidden(Path(args.owned_seq_f32), hidden)

    prefix_worker = None
    hybrid_workers: list[WorkerProc] = []
    full_workers: list[WorkerProc] = []
    state = {
        "spawn_prefix": args.spawn_prefix,
        "spawn_hybrid": args.spawn_hybrid,
        "spawn_full": args.spawn_full,
        "prefill_workers": args.prefill_workers,
        "reset_workers": args.reset_workers,
    }

    try:
        if args.spawn_prefix:
            if not args.prefix_seq_worker_bin or not args.prefix_seq_fixture:
                raise RuntimeError("spawn_prefix requires --prefix-seq-worker-bin and --prefix-seq-fixture")
            print("[residence] spawning prefix worker")
            prefix_worker = WorkerProc([args.prefix_seq_worker_bin, args.gguf, args.prefix_seq_fixture])
            if args.prefill_workers:
                print("[residence] prefix prefill")
                prefix_worker.prefill_prefix_bin(token_ids, hidden)
            if args.reset_workers:
                print("[residence] prefix reset")
                prefix_worker.reset()

        if args.spawn_hybrid:
            if not args.c_bin_worker_bin:
                raise RuntimeError("spawn_hybrid requires --c-bin-worker-bin")
            n_cycles = len(args.full_layer) if args.full_layer else 1
            chunks = split_cycle_fixtures(args.fixture, n_cycles, args.spawn_prefix)
            for idx, chunk in enumerate(chunks):
                print(f"[residence] spawning hybrid worker cycle={idx}")
                worker = WorkerProc([args.c_bin_worker_bin, args.gguf, *chunk])
                hybrid_workers.append(worker)
                if args.prefill_workers:
                    print(f"[residence] hybrid prefill cycle={idx}")
                    worker.prefill_seq_bin(len(token_ids), hidden)
                if args.reset_workers:
                    print(f"[residence] hybrid reset cycle={idx}")
                    worker.reset()

        if args.spawn_full:
            if not args.full_layer_worker_bin or not args.full_layer:
                raise RuntimeError("spawn_full requires --full-layer-worker-bin and at least one --full-layer")
            for layer_idx in args.full_layer:
                print(f"[residence] spawning full worker layer={layer_idx}")
                worker = WorkerProc([args.full_layer_worker_bin, args.gguf, str(layer_idx)])
                full_workers.append(worker)
                if args.prefill_workers:
                    print(f"[residence] full prefill layer={layer_idx}")
                    worker.prefill_seq_bin(len(token_ids), hidden)
                if args.reset_workers:
                    print(f"[residence] full reset layer={layer_idx}")
                    worker.reset()

        result = run_hf_patched_tail(
            model=model,
            tokenizer=tokenizer,
            token_ids=token_ids,
            owned_seq=owned_seq,
            splice_layer=args.splice_layer,
            top_k=args.top_k,
            layer_progress=args.hf_tail_layer_progress,
        )
        out = {
            "prompt_tokens": len(token_ids),
            "hidden": hidden,
            "worker_state": state,
            "result": result,
        }
        print(f"[residence] hf_patched_ms={result['hf_patched_ms']:.2f}")
        print(f"[residence] next_id={result['next_id']} next_text={json.dumps(result['next_text'])}")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"json_out: {args.json_out}")
    finally:
        if prefix_worker is not None:
            prefix_worker.close()
        for worker in hybrid_workers:
            worker.close()
        for worker in full_workers:
            worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
