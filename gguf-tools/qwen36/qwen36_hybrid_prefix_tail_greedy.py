#!/usr/bin/env python3
import argparse
import contextlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import create_causal_mask


PREFIX_MAGIC = b"Q36PFX01"


def read_prompt(args) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


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


def write_row_hidden(path: Path, row: np.ndarray, hidden: int) -> None:
    arr = np.asarray(row, dtype=np.float32)
    if arr.shape != (hidden,):
        raise RuntimeError(f"unexpected row shape: got {arr.shape}, expected {(hidden,)}")
    arr.tofile(path)


def _read_exact_pipe(pipe, n_bytes: int) -> bytes:
    buf = bytearray()
    while len(buf) < n_bytes:
        chunk = pipe.read(n_bytes - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    if len(buf) != n_bytes:
        raise RuntimeError(f"short binary read: got {len(buf)} expected {n_bytes}")
    return bytes(buf)


def _read_fixture_layer(path: str) -> int:
    with open(path, "rb") as fp:
        magic = fp.read(8)
        if len(magic) != 8:
            raise RuntimeError(f"short fixture header: {path}")
        layer_raw = fp.read(4)
        if len(layer_raw) != 4:
            raise RuntimeError(f"short fixture layer header: {path}")
        return int(np.frombuffer(layer_raw, dtype=np.uint32, count=1)[0])


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
            print(f"[hybrid] {label} layer={i} ms={dt_ms:.2f}", flush=True)
        return _post

    try:
        for i, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_pre_hook(make_pre(i)))
            handles.append(layer.register_forward_hook(make_post(i)))
        yield
    finally:
        for h in handles:
            h.remove()


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
            print(f"[hybrid] {label} layer={i} ms={dt_ms:.2f}", flush=True)
        return _post

    try:
        for i, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_pre_hook(make_pre(i)))
            handles.append(layer.register_forward_hook(make_post(i)))
        yield
    finally:
        for h in handles:
            h.remove()


class PrefixSeqWorker:
    def __init__(self, worker_bin: str, gguf: str, fixture: str):
        self.proc = subprocess.Popen(
            [worker_bin, gguf, fixture],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(Path(worker_bin).resolve().parent),
        )
        ready = self._read_line()
        if not ready.startswith("READY"):
            raise RuntimeError(f"prefix worker failed to start: {ready}")
        self.prefilled = False
        self.seq_len = 0
        self.hidden = 0

    def _read_line(self) -> str:
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"prefix worker exited unexpectedly; stderr={err}")
        return line.decode("utf-8", errors="replace").strip()

    def _dump_hidden(self) -> np.ndarray:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(b"DUMP_HIDDEN\n")
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            err = "\n".join(self._stderr_lines)
            raise RuntimeError(f"owned session worker pipe broke during DUMP_HIDDEN; stderr={err}") from exc
        hdr = self._read_line()
        if not hdr.startswith("HIDDEN "):
            raise RuntimeError(f"prefix worker expected HIDDEN, got: {hdr}")
        parts = hdr.split()
        n_floats = int(parts[1])
        n_bytes = int(parts[2])
        raw = self.proc.stdout.read(n_bytes)
        if len(raw) != n_bytes:
            raise RuntimeError(f"prefix worker short hidden read: got {len(raw)} expected {n_bytes}")
        arr = np.frombuffer(raw, dtype=np.float32, count=n_floats)
        expect = self.seq_len * self.hidden
        if arr.size != expect:
            raise RuntimeError(f"prefix worker hidden size mismatch: got {arr.size}, expected {expect}")
        return arr.reshape(self.seq_len, self.hidden)

    def _dump_last(self) -> np.ndarray:
        assert self.proc.stdin is not None
        self.proc.stdin.write(b"DUMP_LAST\n")
        self.proc.stdin.flush()
        hdr = self._read_line()
        if not hdr.startswith("LAST "):
            raise RuntimeError(f"prefix worker expected LAST, got: {hdr}")
        parts = hdr.split()
        hidden = int(parts[1])
        n_bytes = int(parts[2])
        raw = _read_exact_pipe(self.proc.stdout, n_bytes)
        arr = np.frombuffer(raw, dtype=np.float32, count=hidden)
        if arr.size != self.hidden:
            raise RuntimeError(f"prefix worker last size mismatch: got {arr.size}, expected {self.hidden}")
        return arr

    def run_for_token_ids(self, token_ids: list[int], hidden: int, prefix_path: Path, dump_mode: str = "hidden") -> tuple[np.ndarray, dict]:
        assert self.proc.stdin is not None
        t0 = time.perf_counter()
        if not self.prefilled:
            self.hidden = hidden
            self.seq_len = len(token_ids)
            tok = np.asarray(token_ids, dtype=np.uint32)
            seq = np.zeros((len(token_ids), hidden), dtype=np.float32)
            cmd = f"PREFILL_PREFIX_BIN {len(token_ids)} {hidden}\n"
            self.proc.stdin.write(cmd.encode("utf-8"))
            self.proc.stdin.write(tok.tobytes(order="C"))
            self.proc.stdin.write(seq.tobytes(order="C"))
            self.proc.stdin.flush()
            line = self._read_line()
            if not line.startswith("PREFILL_OK "):
                raise RuntimeError(f"prefix worker prefill failed: {line}")
            parts = line.split()
            self.seq_len = int(parts[1])
            self.hidden = int(parts[2])
            self.prefilled = True
        else:
            if len(token_ids) != self.seq_len + 1:
                raise RuntimeError(
                    f"prefix worker seq mismatch: worker has {self.seq_len}, requested {len(token_ids)}"
                )
            cmd = f"STEP {token_ids[-1]}\n"
            self.proc.stdin.write(cmd.encode("utf-8"))
            self.proc.stdin.flush()
            line = self._read_line()
            if not line.startswith("STEP_OK "):
                raise RuntimeError(f"prefix worker step failed: {line}")
            parts = line.split()
            self.seq_len = int(parts[1])
            self.hidden = int(parts[2])
        if dump_mode == "hidden":
            seq = self._dump_hidden()
        elif dump_mode == "last":
            seq = self._dump_last()
        else:
            raise RuntimeError(f"bad dump_mode: {dump_mode}")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return seq, {
            "worker_ms": elapsed_ms,
            "worker_seq_len": self.seq_len,
        }

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(b"QUIT\n")
            self.proc.stdin.flush()
            _ = self._read_line()
        except Exception:
            pass
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()


class HybridChainWorker:
    def __init__(self, worker_bin: str, gguf: str, fixtures: list[str]):
        self.proc = subprocess.Popen(
            [worker_bin, gguf, *fixtures],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(Path(worker_bin).resolve().parent),
        )
        ready = self._read_line()
        if not ready.startswith("READY"):
            raise RuntimeError(f"hybrid chain worker failed to start: {ready}")
        self.first_layer = _read_fixture_layer(fixtures[0]) if fixtures else -1
        self.prefilled = False
        self.seq_len = 0
        self.hidden = 0

    def _read_line(self) -> str:
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"hybrid chain worker exited unexpectedly; stderr={err}")
        return line.decode("utf-8", errors="replace").strip()

    def _dump_hidden(self) -> np.ndarray:
        assert self.proc.stdin is not None
        self.proc.stdin.write(b"DUMP_HIDDEN\n")
        self.proc.stdin.flush()
        hdr = self._read_line()
        if not hdr.startswith("HIDDEN "):
            raise RuntimeError(f"hybrid chain worker expected HIDDEN, got: {hdr}")
        parts = hdr.split()
        n_floats = int(parts[1])
        n_bytes = int(parts[2])
        raw = _read_exact_pipe(self.proc.stdout, n_bytes)
        arr = np.frombuffer(raw, dtype=np.float32, count=n_floats)
        expect = self.seq_len * self.hidden
        if arr.size != expect:
            raise RuntimeError(f"hybrid chain hidden size mismatch: got {arr.size}, expected {expect}")
        return arr.reshape(self.seq_len, self.hidden)

    def _dump_last(self) -> np.ndarray:
        assert self.proc.stdin is not None
        self.proc.stdin.write(b"DUMP_LAST\n")
        self.proc.stdin.flush()
        hdr = self._read_line()
        if not hdr.startswith("LAST "):
            raise RuntimeError(f"hybrid chain worker expected LAST, got: {hdr}")
        parts = hdr.split()
        hidden = int(parts[1])
        n_bytes = int(parts[2])
        raw = _read_exact_pipe(self.proc.stdout, n_bytes)
        arr = np.frombuffer(raw, dtype=np.float32, count=hidden)
        if arr.size != self.hidden:
            raise RuntimeError(f"hybrid chain last size mismatch: got {arr.size}, expected {self.hidden}")
        return arr

    def run_for_seq(self, token_ids: list[int], hidden: int, input_seq: np.ndarray | None, dump_mode: str = "hidden") -> tuple[np.ndarray, dict]:
        assert self.proc.stdin is not None
        t0 = time.perf_counter()
        if not self.prefilled:
            self.hidden = hidden
            self.seq_len = len(token_ids)
            if input_seq is None:
                if self.first_layer != 0:
                    raise RuntimeError("hybrid chain prefill requires input_seq unless first fixture is blk.0")
                seq = np.zeros((len(token_ids), hidden), dtype=np.float32)
                token_arr = np.asarray(token_ids, dtype=np.uint32)
                cmd = f"PREFILL_PREFIX_BIN {len(token_ids)} {hidden}\n"
            else:
                seq = np.asarray(input_seq, dtype=np.float32)
                if seq.shape != (len(token_ids), hidden):
                    raise RuntimeError(f"hybrid chain prefill shape mismatch: got {seq.shape}, expected {(len(token_ids), hidden)}")
                token_arr = None
                cmd = f"PREFILL_SEQ_BIN {len(token_ids)} {hidden}\n"
        else:
            if len(token_ids) != self.seq_len + 1:
                raise RuntimeError(f"hybrid chain worker seq mismatch: worker has {self.seq_len}, requested {len(token_ids)}")
            if input_seq is None:
                if self.first_layer != 0:
                    raise RuntimeError("hybrid chain step requires input row unless first fixture is blk.0")
                row = None
                cmd = f"STEP {token_ids[-1]}\n"
            else:
                row = np.asarray(input_seq, dtype=np.float32)
                if row.shape != (hidden,):
                    raise RuntimeError(f"hybrid chain step row shape mismatch: got {row.shape}, expected {(hidden,)}")
                cmd = f"STEP_ROW_BIN {hidden}\n"
        self.proc.stdin.write(cmd.encode("utf-8"))
        if not self.prefilled:
            if token_arr is not None:
                self.proc.stdin.write(token_arr.tobytes(order="C"))
            self.proc.stdin.write(seq.tobytes(order="C"))
        elif row is not None:
            self.proc.stdin.write(row.tobytes(order="C"))
        self.proc.stdin.flush()
        line = self._read_line()
        if not line.startswith("STEP_OK ") and not line.startswith("PREFILL_OK "):
            raise RuntimeError(f"hybrid chain worker failed: {line}")
        parts = line.split()
        self.seq_len = int(parts[1])
        self.hidden = int(parts[2])
        self.prefilled = True
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if dump_mode == "hidden":
            out = self._dump_hidden()
        elif dump_mode == "last":
            out = self._dump_last()
        else:
            raise RuntimeError(f"bad dump_mode: {dump_mode}")
        return (
            out,
            {"worker_ms": elapsed_ms, "worker_seq_len": self.seq_len},
        )

    def dump_hidden(self) -> np.ndarray:
        return self._dump_hidden()

    def dump_last(self) -> np.ndarray:
        return self._dump_last()

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(b"QUIT\n")
            self.proc.stdin.flush()
            _ = self._read_line()
        except Exception:
            pass
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()


class FullLayerWorker:
    def __init__(self, worker_bin: str, gguf: str, layer: int):
        self.proc = subprocess.Popen(
            [worker_bin, gguf, "--layer", str(layer)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(Path(worker_bin).resolve().parent),
        )
        ready = self._read_line()
        if not ready.startswith("READY"):
            raise RuntimeError(f"full-layer worker failed to start: {ready}")
        self.prefilled = False
        self.seq_len = 0
        self.hidden = 0
        self.layer = layer

    def _read_line(self) -> str:
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"full-layer worker exited unexpectedly; stderr={err}")
        return line.decode("utf-8", errors="replace").strip()

    def _dump_hidden(self) -> np.ndarray:
        assert self.proc.stdin is not None
        self.proc.stdin.write(b"DUMP_HIDDEN\n")
        self.proc.stdin.flush()
        hdr = self._read_line()
        if not hdr.startswith("HIDDEN "):
            raise RuntimeError(f"full-layer worker expected HIDDEN, got: {hdr}")
        parts = hdr.split()
        n_floats = int(parts[1])
        n_bytes = int(parts[2])
        raw = _read_exact_pipe(self.proc.stdout, n_bytes)
        arr = np.frombuffer(raw, dtype=np.float32, count=n_floats)
        expect = self.seq_len * self.hidden
        if arr.size != expect:
            raise RuntimeError(f"full-layer worker hidden size mismatch: got {arr.size}, expected {expect}")
        return arr.reshape(self.seq_len, self.hidden)

    def _dump_last(self) -> np.ndarray:
        assert self.proc.stdin is not None
        self.proc.stdin.write(b"DUMP_LAST\n")
        self.proc.stdin.flush()
        hdr = self._read_line()
        if not hdr.startswith("LAST "):
            raise RuntimeError(f"full-layer worker expected LAST, got: {hdr}")
        parts = hdr.split()
        hidden = int(parts[1])
        n_bytes = int(parts[2])
        raw = _read_exact_pipe(self.proc.stdout, n_bytes)
        arr = np.frombuffer(raw, dtype=np.float32, count=hidden)
        if arr.size != self.hidden:
            raise RuntimeError(f"full-layer worker last size mismatch: got {arr.size}, expected {self.hidden}")
        return arr

    def run_for_seq(self, input_data: np.ndarray, token_ids: list[int], hidden: int, dump_mode: str = "hidden") -> tuple[np.ndarray, dict]:
        assert self.proc.stdin is not None
        t0 = time.perf_counter()
        if not self.prefilled:
            self.hidden = hidden
            self.seq_len = len(token_ids)
            seq = np.asarray(input_data, dtype=np.float32)
            if seq.shape != (len(token_ids), hidden):
                raise RuntimeError(f"full-layer prefill shape mismatch: got {seq.shape}, expected {(len(token_ids), hidden)}")
            cmd = f"PREFILL_SEQ_BIN {len(token_ids)} {hidden}\n"
            self.proc.stdin.write(cmd.encode("utf-8"))
            self.proc.stdin.write(seq.tobytes(order="C"))
            self.proc.stdin.flush()
            line = self._read_line()
            if not line.startswith("PREFILL_OK "):
                raise RuntimeError(f"full-layer worker prefill failed: {line}")
            parts = line.split()
            self.seq_len = int(parts[1])
            self.hidden = int(parts[2])
            self.prefilled = True
        else:
            if len(token_ids) != self.seq_len + 1:
                raise RuntimeError(
                    f"full-layer worker seq mismatch: worker has {self.seq_len}, requested {len(token_ids)}"
                )
            row = np.asarray(input_data, dtype=np.float32)
            if row.shape != (hidden,):
                raise RuntimeError(f"full-layer step row shape mismatch: got {row.shape}, expected {(hidden,)}")
            cmd = f"STEP_ROW_BIN {hidden}\n"
            self.proc.stdin.write(cmd.encode("utf-8"))
            self.proc.stdin.write(row.tobytes(order="C"))
            self.proc.stdin.flush()
            line = self._read_line()
            if not line.startswith("STEP_OK "):
                raise RuntimeError(f"full-layer worker step failed: {line}")
            parts = line.split()
            self.seq_len = int(parts[1])
            self.hidden = int(parts[2])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if dump_mode == "hidden":
            out = self._dump_hidden()
        elif dump_mode == "last":
            out = self._dump_last()
        else:
            raise RuntimeError(f"bad dump_mode: {dump_mode}")
        return (
            out,
            {
            "worker_ms": elapsed_ms,
            "worker_seq_len": self.seq_len,
            "layer": self.layer,
            },
        )

    def dump_hidden(self) -> np.ndarray:
        return self._dump_hidden()

    def dump_last(self) -> np.ndarray:
        return self._dump_last()

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(b"QUIT\n")
            self.proc.stdin.flush()
            _ = self._read_line()
        except Exception:
            pass
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()


def write_owned_session_config(
    path: Path,
    *,
    fixtures: list[str],
    full_layers: list[int],
    prefix_seq_worker_bin: str | None,
    prefix_seq_fixture: str | None,
    c_bin_worker_bin: str,
    full_layer_worker_bin: str,
) -> None:
    chunks = split_cycle_fixtures(fixtures, len(full_layers), prefix_seq_worker_bin is not None)
    lines = []
    if prefix_seq_worker_bin:
        if not prefix_seq_fixture:
            raise RuntimeError("owned-session config requires --prefix-seq-fixture with --prefix-seq-worker-bin")
        lines.append(f"prefix_worker_bin {prefix_seq_worker_bin}")
        lines.append(f"prefix_fixture {prefix_seq_fixture}")
    lines.append(f"hybrid_worker_bin {c_bin_worker_bin}")
    lines.append(f"full_worker_bin {full_layer_worker_bin}")
    for layer_idx, chunk in zip(full_layers, chunks):
        lines.append(f"cycle {layer_idx} {','.join(chunk)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class OwnedSessionWorker:
    def __init__(self, worker_bin: str, gguf: str, config_path: Path, env: dict[str, str] | None = None, label: str | None = None):
        self._stderr_lines: list[str] = []
        self.label = label or "owned-session"
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)
        self.proc = subprocess.Popen(
            [worker_bin, gguf, str(config_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(Path(worker_bin).resolve().parent),
            env=proc_env,
        )
        self._stderr_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_thread.start()
        ready = self._read_line()
        if not ready.startswith("READY"):
            raise RuntimeError(f"owned session worker failed to start: {ready}")
        self.prefilled = False
        self.seq_len = 0
        self.hidden = 0

    def _pump_stderr(self) -> None:
        if self.proc.stderr is None:
            return
        for raw in self.proc.stderr:
            line = raw.decode("utf-8", errors="replace").rstrip()
            self._stderr_lines.append(line)
            if len(self._stderr_lines) > 4000:
                self._stderr_lines = self._stderr_lines[-4000:]
            if line:
                print(f"[{self.label}] {line}", file=sys.stderr, flush=True)

    def _read_line(self) -> str:
        line = self.proc.stdout.readline()
        if not line:
            err = "\n".join(self._stderr_lines)
            raise RuntimeError(f"owned session worker exited unexpectedly; stderr={err}")
        return line.decode("utf-8", errors="replace").strip()

    def _dump_hidden(self) -> np.ndarray:
        assert self.proc.stdin is not None
        self.proc.stdin.write(b"DUMP_HIDDEN\n")
        self.proc.stdin.flush()
        hdr = self._read_line()
        if not hdr.startswith("HIDDEN "):
            raise RuntimeError(f"owned session worker expected HIDDEN, got: {hdr}")
        parts = hdr.split()
        n_floats = int(parts[1])
        n_bytes = int(parts[2])
        raw = _read_exact_pipe(self.proc.stdout, n_bytes)
        arr = np.frombuffer(raw, dtype=np.float32, count=n_floats)
        expect = self.seq_len * self.hidden
        if arr.size != expect:
            raise RuntimeError(f"owned session hidden size mismatch: got {arr.size}, expected {expect}")
        return arr.reshape(self.seq_len, self.hidden)

    def _dump_cycle_hidden(self, cycle_idx: int) -> np.ndarray:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(f"DUMP_CYCLE_HIDDEN {cycle_idx}\n".encode("utf-8"))
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            err = "\n".join(self._stderr_lines)
            raise RuntimeError(f"owned session worker pipe broke during DUMP_CYCLE_HIDDEN {cycle_idx}; stderr={err}") from exc
        hdr = self._read_line()
        if not hdr.startswith("CYCLE_HIDDEN "):
            raise RuntimeError(f"owned session worker expected CYCLE_HIDDEN, got: {hdr}")
        parts = hdr.split()
        got_cycle_idx = int(parts[1])
        n_floats = int(parts[2])
        n_bytes = int(parts[3])
        if got_cycle_idx != cycle_idx:
            raise RuntimeError(f"owned session cycle mismatch: got {got_cycle_idx}, expected {cycle_idx}")
        raw = _read_exact_pipe(self.proc.stdout, n_bytes)
        arr = np.frombuffer(raw, dtype=np.float32, count=n_floats)
        expect = self.seq_len * self.hidden
        if arr.size != expect:
            raise RuntimeError(f"owned session cycle hidden size mismatch: got {arr.size}, expected {expect}")
        return arr.reshape(self.seq_len, self.hidden)

    def _dump_cycle_last(self, cycle_idx: int) -> np.ndarray:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(f"DUMP_CYCLE_LAST {cycle_idx}\n".encode("utf-8"))
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            err = "\n".join(self._stderr_lines)
            raise RuntimeError(f"owned session worker pipe broke during DUMP_CYCLE_LAST {cycle_idx}; stderr={err}") from exc
        hdr = self._read_line()
        if not hdr.startswith("CYCLE_LAST "):
            raise RuntimeError(f"owned session worker expected CYCLE_LAST, got: {hdr}")
        parts = hdr.split()
        got_cycle_idx = int(parts[1])
        got_hidden = int(parts[2])
        n_bytes = int(parts[3])
        if got_cycle_idx != cycle_idx:
            raise RuntimeError(f"owned session cycle last mismatch: got {got_cycle_idx}, expected {cycle_idx}")
        raw = _read_exact_pipe(self.proc.stdout, n_bytes)
        arr = np.frombuffer(raw, dtype=np.float32, count=got_hidden)
        if arr.size != self.hidden:
            raise RuntimeError(f"owned session cycle last size mismatch: got {arr.size}, expected {self.hidden}")
        return arr.reshape(1, self.hidden)

    def _dump_cycle_pre_hidden(self, cycle_idx: int) -> np.ndarray:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(f"DUMP_CYCLE_PRE_HIDDEN {cycle_idx}\n".encode("utf-8"))
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            err = "\n".join(self._stderr_lines)
            raise RuntimeError(f"owned session worker pipe broke during DUMP_CYCLE_PRE_HIDDEN {cycle_idx}; stderr={err}") from exc
        hdr = self._read_line()
        if not hdr.startswith("CYCLE_PRE_HIDDEN "):
            raise RuntimeError(f"owned session worker expected CYCLE_PRE_HIDDEN, got: {hdr}")
        parts = hdr.split()
        got_cycle_idx = int(parts[1])
        n_floats = int(parts[2])
        n_bytes = int(parts[3])
        if got_cycle_idx != cycle_idx:
            raise RuntimeError(f"owned session pre cycle mismatch: got {got_cycle_idx}, expected {cycle_idx}")
        raw = _read_exact_pipe(self.proc.stdout, n_bytes)
        arr = np.frombuffer(raw, dtype=np.float32, count=n_floats)
        expect = self.seq_len * self.hidden
        if arr.size != expect:
            raise RuntimeError(f"owned session pre cycle hidden size mismatch: got {arr.size}, expected {expect}")
        return arr.reshape(self.seq_len, self.hidden)

    def _dump_cycle_pre_last(self, cycle_idx: int) -> np.ndarray:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(f"DUMP_CYCLE_PRE_LAST {cycle_idx}\n".encode("utf-8"))
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            err = "\n".join(self._stderr_lines)
            raise RuntimeError(f"owned session worker pipe broke during DUMP_CYCLE_PRE_LAST {cycle_idx}; stderr={err}") from exc
        hdr = self._read_line()
        if not hdr.startswith("CYCLE_PRE_LAST "):
            raise RuntimeError(f"owned session worker expected CYCLE_PRE_LAST, got: {hdr}")
        parts = hdr.split()
        got_cycle_idx = int(parts[1])
        got_hidden = int(parts[2])
        n_bytes = int(parts[3])
        if got_cycle_idx != cycle_idx:
            raise RuntimeError(f"owned session pre cycle last mismatch: got {got_cycle_idx}, expected {cycle_idx}")
        raw = _read_exact_pipe(self.proc.stdout, n_bytes)
        arr = np.frombuffer(raw, dtype=np.float32, count=got_hidden)
        if arr.size != self.hidden:
            raise RuntimeError(f"owned session pre cycle last size mismatch: got {arr.size}, expected {self.hidden}")
        return arr.reshape(1, self.hidden)

    def _dump_last(self) -> np.ndarray:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(b"DUMP_LAST\n")
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            err = "\n".join(self._stderr_lines)
            raise RuntimeError(f"owned session worker pipe broke during DUMP_LAST; stderr={err}") from exc
        hdr = self._read_line()
        if not hdr.startswith("LAST "):
            raise RuntimeError(f"owned session worker expected LAST, got: {hdr}")
        parts = hdr.split()
        got_hidden = int(parts[1])
        n_bytes = int(parts[2])
        raw = _read_exact_pipe(self.proc.stdout, n_bytes)
        arr = np.frombuffer(raw, dtype=np.float32, count=got_hidden)
        if arr.size != self.hidden:
            raise RuntimeError(f"owned session last size mismatch: got {arr.size}, expected {self.hidden}")
        return arr.reshape(1, self.hidden)

    def run_for_token_ids(self, token_ids: list[int], hidden: int, dump_mode: str = "hidden") -> tuple[np.ndarray, dict]:
        assert self.proc.stdin is not None
        t0 = time.perf_counter()
        try:
            if not self.prefilled:
                print(f"[owned-session] prefill start seq_len={len(token_ids)} hidden={hidden}", flush=True)
                self.hidden = hidden
                self.seq_len = len(token_ids)
                tok = np.asarray(token_ids, dtype=np.uint32)
                seq = np.zeros((len(token_ids), hidden), dtype=np.float32)
                cmd = f"PREFILL_PREFIX_BIN {len(token_ids)} {hidden}\n"
                print("[owned-session] sending prefill command", flush=True)
                self.proc.stdin.write(cmd.encode("utf-8"))
                print("[owned-session] sending token ids", flush=True)
                self.proc.stdin.write(tok.tobytes(order="C"))
                print("[owned-session] sending zero input seq", flush=True)
                self.proc.stdin.write(seq.tobytes(order="C"))
            else:
                if len(token_ids) != self.seq_len + 1:
                    raise RuntimeError(
                        f"owned session seq mismatch: worker has {self.seq_len}, requested {len(token_ids)}"
                    )
                print(f"[owned-session] step start next_token={token_ids[-1]} new_seq_len={len(token_ids)}", flush=True)
                cmd = f"STEP {token_ids[-1]}\n"
                self.proc.stdin.write(cmd.encode("utf-8"))
            print("[owned-session] flushing command payload", flush=True)
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            err = "\n".join(self._stderr_lines)
            raise RuntimeError(f"owned session worker pipe broke; stderr={err}") from exc
        print("[owned-session] waiting for worker ack", flush=True)
        line = self._read_line()
        if not line.startswith("PREFILL_OK ") and not line.startswith("STEP_OK "):
            raise RuntimeError(f"owned session worker failed: {line}")
        parts = line.split()
        self.seq_len = int(parts[1])
        self.hidden = int(parts[2])
        self.prefilled = True
        print(f"[owned-session] worker ack seq_len={self.seq_len}, dumping {dump_mode}", flush=True)
        if dump_mode == "hidden":
            out = self._dump_hidden()
        elif dump_mode == "last":
            out = self._dump_last()
        else:
            raise RuntimeError(f"bad dump_mode: {dump_mode}")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[owned-session] done worker_ms={elapsed_ms:.2f}", flush=True)
        return out, {"worker_ms": elapsed_ms, "worker_seq_len": self.seq_len}

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(b"QUIT\n")
            self.proc.stdin.flush()
            _ = self._read_line()
        except Exception:
            pass
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()

    def dump_cycle_hidden(self, cycle_idx: int) -> np.ndarray:
        return self._dump_cycle_hidden(cycle_idx)

    def dump_cycle_pre_hidden(self, cycle_idx: int) -> np.ndarray:
        return self._dump_cycle_pre_hidden(cycle_idx)

    def dump_cycle_last(self, cycle_idx: int) -> np.ndarray:
        return self._dump_cycle_last(cycle_idx)

    def dump_cycle_pre_last(self, cycle_idx: int) -> np.ndarray:
        return self._dump_cycle_pre_last(cycle_idx)

    def stderr_lines(self) -> list[str]:
        return list(self._stderr_lines)


def split_cycle_fixtures(fixtures: list[str], n_cycles: int, has_prefix_seq_worker: bool) -> list[list[str]]:
    out: list[list[str]] = []
    pos = 0
    for cycle_idx in range(n_cycles):
        want = 2 if has_prefix_seq_worker and cycle_idx == 0 else 3
        chunk = fixtures[pos:pos + want]
        if len(chunk) != want:
            raise RuntimeError(f"fixture split failed for cycle {cycle_idx}: expected {want}, got {len(chunk)}")
        out.append(chunk)
        pos += want
    if pos != len(fixtures):
        raise RuntimeError(f"fixture split left trailing fixtures: used {pos}, total {len(fixtures)}")
    return out


def topk(logits: np.ndarray, k: int):
    idx = np.argpartition(-logits, k - 1)[:k]
    idx = idx[np.argsort(-logits[idx])]
    return [{"id": int(i), "logit": float(logits[i])} for i in idx]


def run_owned_prefix_cycles(
    *,
    td_path: Path,
    token_ids: list[int],
    hidden: int,
    gguf: str,
    fixtures: list[str],
    full_layers: list[int],
    c_bin: str,
    c_bin_prefix_flag: bool,
    full_layer_bin: str | None,
    prefix_seq_bin: str | None,
    prefix_seq_fixture: str | None,
    prefix_seq_dynamic: bool,
    prefix_seq_worker: PrefixSeqWorker | None,
    hybrid_chain_workers: list[HybridChainWorker] | None,
    full_layer_workers: list[FullLayerWorker] | None,
    step_idx: int,
) -> tuple[np.ndarray, dict]:
    prefix_path = td_path / "prefix.bin"
    prefix_seq_path = td_path / "prefix_seq.f32"
    use_prefix_seq = bool(prefix_seq_bin and (prefix_seq_dynamic or step_idx == 0))
    has_prefix_input_seq = use_prefix_seq or (prefix_seq_worker is not None)
    prefix_proc = None
    prefix_elapsed_ms = None
    worker_meta = None
    hybrid_worker_meta = None
    full_layer_worker_meta = None
    initial_prefix_seq = None
    cycle_input_seq: np.ndarray | None = None
    cycle_input_row: np.ndarray | None = None
    if prefix_seq_worker is not None:
        if not prefix_seq_fixture:
            raise RuntimeError("--prefix-seq-worker requires --prefix-seq-fixture")
        prefix_out, worker_meta = prefix_seq_worker.run_for_token_ids(
            token_ids,
            hidden,
            prefix_path,
            dump_mode="hidden" if step_idx == 0 else "last",
        )
        if step_idx == 0:
            initial_prefix_seq = prefix_out
        else:
            cycle_input_row = prefix_out
        prefix_elapsed_ms = worker_meta["worker_ms"]
    elif use_prefix_seq:
        if not prefix_seq_fixture:
            raise RuntimeError("--prefix-seq-bin requires --prefix-seq-fixture")
        write_prefix_fixture(prefix_path, token_ids, hidden)
        prefix_cmd = [prefix_seq_bin, gguf, prefix_seq_fixture, "--prefix", str(prefix_path), "--dump-seq", str(prefix_seq_path)]
        print(f"[hybrid] running prefix-seq {' '.join(prefix_cmd)}")
        t0 = time.perf_counter()
        prefix_proc = subprocess.run(prefix_cmd, capture_output=True, text=True, check=True)
        prefix_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        prefix_seq = read_seq_hidden(prefix_seq_path, len(token_ids), hidden)
        initial_prefix_seq = prefix_seq
        write_prefix_fixture(prefix_path, token_ids, hidden, input_seq=prefix_seq)

    c_fixtures = list(fixtures)
    if prefix_seq_bin and not use_prefix_seq and step_idx > 0:
        c_fixtures = [prefix_seq_fixture, *c_fixtures]
    seq_source = None
    cycle_reports: list[str] = []
    cycle_timings: list[dict] = []
    n_cycles = len(full_layers)
    if n_cycles == 0:
        cycle_fixture_chunks = [list(fixtures)]
        n_cycles = 1
    else:
        cycle_fixture_chunks = split_cycle_fixtures(fixtures, n_cycles, prefix_seq_worker is not None)

    for cycle_idx in range(n_cycles):
        cy_fixtures = cycle_fixture_chunks[cycle_idx]
        cy_full_layer = full_layers[cycle_idx] if cycle_idx < len(full_layers) else None

        cy_prefix_path = td_path / f"cycle_{cycle_idx}_prefix.bin"
        if cycle_idx == 0:
            if has_prefix_input_seq:
                if step_idx == 0:
                    cycle_input_seq = initial_prefix_seq
                    cycle_input_row = None if initial_prefix_seq is None else initial_prefix_seq[-1]
            else:
                write_prefix_fixture(cy_prefix_path, token_ids, hidden)

        cy_seq_path = td_path / f"cycle_{cycle_idx}_seq.f32"
        cycle_full_worker = None if full_layer_workers is None else full_layer_workers[cycle_idx]
        cycle_worker = None if hybrid_chain_workers is None else hybrid_chain_workers[cycle_idx]
        worker_step_mode = step_idx > 0 and cycle_worker is not None
        if cycle_idx > 0:
            if cycle_worker is not None:
                if worker_step_mode:
                    if cycle_input_row is None:
                        raise RuntimeError(f"cycle {cycle_idx}: no input row from previous cycle")
                else:
                    if cycle_input_seq is None:
                        raise RuntimeError(f"cycle {cycle_idx}: no input seq from previous cycle")
                    write_prefix_fixture(cy_prefix_path, token_ids, hidden, input_seq=cycle_input_seq)
            else:
                if cycle_input_seq is None:
                    raise RuntimeError(f"cycle {cycle_idx}: no input seq for stateless hybrid subprocess")
                write_prefix_fixture(cy_prefix_path, token_ids, hidden, input_seq=cycle_input_seq)

        if cycle_worker is not None:
            print(f"[hybrid] cycle {cycle_idx}: hybrid-worker {c_bin} {gguf} {' '.join(cy_fixtures)}")
            if not cycle_worker.prefilled:
                if cycle_input_seq is None and cycle_worker.first_layer != 0:
                    raise RuntimeError(f"cycle {cycle_idx}: no prefill seq for hybrid worker")
                dump_mode = "hidden" if (cycle_idx == n_cycles - 1 and full_layer_bin is None and cycle_full_worker is None) else "hidden"
                seq, cycle_hybrid_meta = cycle_worker.run_for_seq(token_ids, hidden, cycle_input_seq, dump_mode=dump_mode)
            else:
                if cycle_input_row is None and cycle_worker.first_layer != 0:
                    raise RuntimeError(f"cycle {cycle_idx}: no step row for hybrid worker")
                seq, cycle_hybrid_meta = cycle_worker.run_for_seq(token_ids, hidden, cycle_input_row, dump_mode="last")
            prefix_ms = cycle_hybrid_meta["worker_ms"]
            cy_proc = None
        else:
            if c_bin_prefix_flag:
                cy_cmd = [c_bin, gguf, *cy_fixtures, "--prefix", str(cy_prefix_path)]
            else:
                cy_cmd = [c_bin, gguf, str(cy_prefix_path), *cy_fixtures]
            if cycle_idx > 0 or has_prefix_input_seq:
                cy_cmd.append("--use-prefix-input-seq")
            cy_cmd.extend(["--dump-seq", str(cy_seq_path)])
            print(f"[hybrid] cycle {cycle_idx}: c-prefix {' '.join(cy_cmd)}")
            t0 = time.perf_counter()
            cy_proc = subprocess.run(cy_cmd, capture_output=True, text=True, check=True)
            prefix_ms = (time.perf_counter() - t0) * 1000.0

        next_cycle_worker = None if hybrid_chain_workers is None or cycle_idx + 1 >= n_cycles else hybrid_chain_workers[cycle_idx + 1]
        next_cycle_needs_full_seq = cycle_idx < n_cycles - 1 and next_cycle_worker is None

        full_ms = None
        if cycle_full_worker is not None:
            print(f"[hybrid] cycle {cycle_idx}: full-layer-worker layer={cycle_full_worker.layer}")
            if cycle_worker is not None:
                full_input = seq
            else:
                full_input = read_seq_hidden(cy_seq_path, len(token_ids), hidden)
            if not cycle_full_worker.prefilled:
                seq, cycle_full_meta = cycle_full_worker.run_for_seq(full_input, token_ids, hidden, dump_mode="hidden")
            else:
                row = np.asarray(full_input, dtype=np.float32)
                if row.ndim == 2:
                    row = row[-1]
                dump_mode = "hidden" if (cycle_idx == n_cycles - 1 or next_cycle_needs_full_seq) else "last"
                seq, cycle_full_meta = cycle_full_worker.run_for_seq(row, token_ids, hidden, dump_mode=dump_mode)
            full_ms = cycle_full_meta["worker_ms"]
            c_stdout = cy_proc.stdout if cy_proc is not None else "[persistent hybrid worker]"
            cycle_reports.append(f"[cycle {cycle_idx} hybrid+full_worker]\nc: {c_stdout}\nfull_worker: {json.dumps(cycle_full_meta)}")
            if full_layer_worker_meta is None:
                full_layer_worker_meta = {}
            full_layer_worker_meta[f"cycle_{cycle_idx}"] = cycle_full_meta
        elif full_layer_bin:
            cy_full_path = td_path / f"cycle_{cycle_idx}_full.f32"
            full_cmd = [full_layer_bin, gguf, str(cy_seq_path), str(cy_full_path), "--layer", str(cy_full_layer)]
            print(f"[hybrid] cycle {cycle_idx}: full-layer {' '.join(full_cmd)}")
            t0 = time.perf_counter()
            cy_full_proc = subprocess.run(full_cmd, capture_output=True, text=True, check=True)
            full_ms = (time.perf_counter() - t0) * 1000.0
            c_stdout = cy_proc.stdout if cy_proc is not None else "[persistent hybrid worker]"
            cycle_reports.append(f"[cycle {cycle_idx} hybrid+c_full]\nc: {c_stdout}\nfull: {cy_full_proc.stdout}")
            cy_output_path = cy_full_path
        else:
            c_stdout = cy_proc.stdout if cy_proc is not None else "[persistent hybrid worker]"
            cycle_reports.append(f"[cycle {cycle_idx} hybrid only]\nc: {c_stdout}")
            cy_output_path = cy_seq_path

        cycle_timings.append({
            "cycle": cycle_idx,
            "prefix_ms": prefix_ms,
            "full_ms": full_ms,
            "total_ms": prefix_ms + (full_ms or 0.0),
            "full_layer": cy_full_layer if cy_full_layer is not None and full_layer_bin else None,
            "hybrid_worker_ms": None if cycle_worker is None else prefix_ms,
        })
        if cycle_worker is not None:
            if hybrid_worker_meta is None:
                hybrid_worker_meta = {}
            hybrid_worker_meta[f"cycle_{cycle_idx}"] = cycle_hybrid_meta

        if cycle_idx < n_cycles - 1:
            if cycle_full_worker is not None:
                next_cycle_worker = None if hybrid_chain_workers is None or cycle_idx + 1 >= n_cycles else hybrid_chain_workers[cycle_idx + 1]
                next_cycle_needs_full_seq = next_cycle_worker is None
                if not cycle_full_worker.prefilled or step_idx == 0 or next_cycle_needs_full_seq:
                    cycle_input_seq = seq
                    cycle_input_row = seq[-1]
                else:
                    cycle_input_row = seq
                    cycle_input_seq = None
            elif cycle_worker is not None:
                if not cycle_worker.prefilled or step_idx == 0:
                    cycle_input_seq = seq
                    cycle_input_row = seq[-1]
                else:
                    cycle_input_row = seq
                    cycle_input_seq = None
            else:
                cycle_input_seq = read_seq_hidden(cy_output_path, len(token_ids), hidden)
                cycle_input_row = cycle_input_seq[-1]
        else:
            if cycle_full_worker is not None or cycle_worker is not None:
                owned_seq = seq if isinstance(seq, np.ndarray) and seq.ndim == 2 else None
                if owned_seq is None:
                    if cycle_full_worker is not None:
                        owned_seq = cycle_full_worker.dump_hidden()
                    elif cycle_worker is not None:
                        owned_seq = cycle_worker.dump_hidden()
                seq_source = owned_seq
            else:
                seq_source = cy_output_path

    if seq_source is None:
        raise RuntimeError("owned prefix produced no final sequence")

    if isinstance(seq_source, np.ndarray):
        owned_seq = seq_source
    else:
        owned_seq = read_seq_hidden(seq_source, len(token_ids), hidden)
    meta = {
        "use_prefix_seq": use_prefix_seq,
        "use_prefix_seq_worker": prefix_seq_worker is not None,
        "prefix_seq_report": prefix_proc.stdout if prefix_proc is not None else None,
        "prefix_seq_ms": prefix_elapsed_ms,
        "prefix_seq_worker_meta": worker_meta,
        "hybrid_chain_worker_meta": hybrid_worker_meta,
        "full_layer_worker_meta": full_layer_worker_meta,
        "cycle_reports": "\n".join(cycle_reports),
        "cycle_timings": cycle_timings,
        "owned_prefix_ms": sum(item["total_ms"] for item in cycle_timings) + (prefix_elapsed_ms or 0.0),
    }
    return owned_seq, meta


def run_hf_baseline_step(
    *,
    model,
    token_ids: list[int],
    hf_past,
    hf_cached_len: int,
    tokenizer,
    top_k: int,
    layer_progress: bool = False,
) -> tuple[dict, object, int]:
    if hf_past is None:
        hf_input_ids = torch.tensor([token_ids], dtype=torch.long)
        hf_cached_len = len(token_ids)
    else:
        delta = token_ids[hf_cached_len:]
        if not delta:
            raise RuntimeError("hf baseline cache expected new tokens but found none")
        hf_input_ids = torch.tensor([delta], dtype=torch.long)
        hf_cached_len = len(token_ids)
    t0 = time.perf_counter()
    with hf_layer_progress(model, layer_progress, "hf_baseline"), torch.inference_mode():
        base = model(
            input_ids=hf_input_ids,
            past_key_values=hf_past,
            use_cache=True,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    base_logits = base.logits[0, -1].detach().float().cpu().numpy()
    hf_argmax = int(np.argmax(base_logits))
    return {
        "hf_next_id": hf_argmax,
        "hf_next_text": token_text(tokenizer, hf_argmax),
        "hf_topk": topk(base_logits, top_k),
        "hf_baseline_ms": elapsed_ms,
    }, base.past_key_values, hf_cached_len


def run_hf_patched_compare(
    *,
    model,
    token_ids: list[int],
    owned_seq: np.ndarray,
    splice_layer: int,
    tokenizer,
    top_k: int,
    layer_progress: bool = False,
    setup_progress: bool = False,
) -> dict:
    text_model = model.model
    device = text_model.norm.weight.device
    hidden_dtype = text_model.norm.weight.dtype
    t_setup = time.perf_counter()
    hidden_states = torch.from_numpy(np.array(owned_seq, dtype=np.float32, copy=True)).unsqueeze(0).to(device=device, dtype=hidden_dtype)
    if setup_progress:
        print(f"[hybrid] hf_tail setup tensor_ms={(time.perf_counter() - t_setup) * 1000.0:.2f}")
    batch, seq_len, _ = hidden_states.shape

    t_setup = time.perf_counter()
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).view(1, 1, -1).expand(4, batch, -1)
    text_position_ids = position_ids[0]
    rope_position_ids = position_ids[1:]
    attention_mask = None
    past_key_values = None
    if setup_progress:
        print(f"[hybrid] hf_tail setup pos_ids_ms={(time.perf_counter() - t_setup) * 1000.0:.2f}")

    t_setup = time.perf_counter()
    causal_mask = create_causal_mask(
        config=text_model.config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )
    if setup_progress:
        print(f"[hybrid] hf_tail setup causal_mask_ms={(time.perf_counter() - t_setup) * 1000.0:.2f}")

    t_setup = time.perf_counter()
    linear_attn_mask = text_model._update_linear_attn_mask(attention_mask, past_key_values)
    if setup_progress:
        print(f"[hybrid] hf_tail setup linear_mask_ms={(time.perf_counter() - t_setup) * 1000.0:.2f}")

    t_setup = time.perf_counter()
    position_embeddings = text_model.rotary_emb(hidden_states, rope_position_ids)
    if setup_progress:
        print(f"[hybrid] hf_tail setup rotary_ms={(time.perf_counter() - t_setup) * 1000.0:.2f}")

    print("[hybrid] running hf patched forward")
    t0 = time.perf_counter()
    with torch.inference_mode():
        for layer_idx in range(splice_layer + 1, len(text_model.layers)):
            decoder_layer = text_model.layers[layer_idx]
            layer_mask = linear_attn_mask if text_model.config.layer_types[layer_idx] == "linear_attention" else causal_mask
            lt0 = time.perf_counter()
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=False,
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            if layer_progress:
                print(f"[hybrid] hf_tail layer={layer_idx} ms={(time.perf_counter() - lt0) * 1000.0:.2f}")
        hidden_states = text_model.norm(hidden_states)
        logits = model.lm_head(hidden_states[:, -1:, :])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    patched_logits = logits[0, -1].detach().float().cpu().numpy()
    next_id = int(np.argmax(patched_logits))
    return {
        "next_id": next_id,
        "next_text": token_text(tokenizer, next_id),
        "patched_topk": topk(patched_logits, top_k),
        "hf_patched_ms": elapsed_ms,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Hybrid Qwen3.6 greedy decode: C owns token_embd+blk0..N, HF owns the tail")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--fixture", action="append", default=[], help="Owned layer fixtures in order")
    ap.add_argument("--splice-layer", type=int, default=None, help="HF layer output index to replace; default=len(fixtures)-1")
    ap.add_argument("--c-bin", default="./qwen36-c-prefix-q8-chain")
    ap.add_argument("--c-bin-prefix-flag", action="store_true",
                    help="Call --c-bin as MODEL.gguf FIXTURE... --prefix PREFIX.bin instead of MODEL.gguf PREFIX.bin FIXTURE...")
    ap.add_argument("--prefix-seq-bin", help="Optional first-stage owned runner that writes a prefix hidden sequence, e.g. GPU blk.0 oracle")
    ap.add_argument("--prefix-seq-fixture", help="Fixture for --prefix-seq-bin")
    ap.add_argument("--prefix-seq-dynamic", action="store_true",
                    help="Run --prefix-seq-bin on every decode step, not just step 0")
    ap.add_argument("--prefix-seq-worker-bin",
                    help="Persistent fixture-backed worker for the prefix-seq stage, e.g. qwen36-fixture-blk0-worker")
    ap.add_argument("--c-bin-worker-bin",
                    help="Persistent fixture-backed worker for the hybrid fixture chain, e.g. qwen36-fixture-blk0-worker; "
                         "do not pass the replay binary here")
    ap.add_argument("--owned-session-worker-bin",
                    help="Native owned-session coordinator that keeps prefix/hybrid/full workers behind one process")
    ap.add_argument("--full-layer-bin", help="Optional second-stage C/GPU binary for one full-attention layer")
    ap.add_argument("--full-layer-worker-bin", help="Persistent worker for full-attention layers")
    ap.add_argument("--full-layer", action="append", type=int, default=None,
                    help="Full-attention layer index (repeat for multi-cycle: e.g. --full-layer 3 --full-layer 7); "
                    "default: [3] for one cycle of 3 hybrid fixtures")
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--prompt-file")
    ap.add_argument("--n-predict", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--compare-hf", action="store_true")
    ap.add_argument("--compare-hf-every", type=int, default=1,
                    help="When --compare-hf is set, run the HF baseline only every N steps")
    ap.add_argument("--hf-baseline-layer-progress", dest="hf_baseline_layer_progress", action="store_true", default=True)
    ap.add_argument("--no-hf-baseline-layer-progress", dest="hf_baseline_layer_progress", action="store_false")
    ap.add_argument("--hf-patched-layer-progress", dest="hf_patched_layer_progress", action="store_true", default=True)
    ap.add_argument("--no-hf-patched-layer-progress", dest="hf_patched_layer_progress", action="store_false")
    ap.add_argument("--hf-patched-setup-progress", action="store_true")
    ap.add_argument("--worker-init-progress", action="store_true")
    ap.add_argument("--skip-hf-patched", action="store_true")
    ap.add_argument("--skip-owned-prefix", action="store_true")
    ap.add_argument("--owned-seq-f32", help="Precomputed owned hidden sequence dump for the current prompt/token length")
    ap.add_argument("--dump-owned-seq", help="Write final owned hidden sequence to this .f32 path")
    ap.add_argument("--close-full-layer-workers-before-hf", action="store_true")
    ap.add_argument("--close-all-workers-before-hf", action="store_true")
    ap.add_argument("--disable-full-layer-workers", action="store_true")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    fixtures = list(args.fixture)
    if not fixtures and not args.skip_owned_prefix:
        ap.error("at least one --fixture is required")
    if args.skip_owned_prefix and not args.owned_seq_f32:
        ap.error("--skip-owned-prefix requires --owned-seq-f32")
    full_layers: list[int] = args.full_layer if args.full_layer is not None else [3]

    CYCLES_PER_HYBRID = 3  # 3 hybrid SSM per full-attention layer in the schedule
    n_cycles = len(full_layers)
    expected_fixtures = n_cycles * CYCLES_PER_HYBRID
    expected_worker_fixtures = expected_fixtures - 1 if args.prefix_seq_worker_bin else expected_fixtures
    if not args.skip_owned_prefix:
        if args.prefix_seq_worker_bin:
            if len(fixtures) == expected_fixtures:
                fixtures = fixtures[1:]
            elif len(fixtures) != expected_worker_fixtures:
                ap.error(
                    f"expected {expected_worker_fixtures} fixtures without blk.0 worker fixture "
                    f"or {expected_fixtures} with it included, got {len(fixtures)}"
                )
        elif len(fixtures) != expected_fixtures:
            ap.error(f"expected {expected_fixtures} fixtures ({n_cycles} cycles x {CYCLES_PER_HYBRID}), got {len(fixtures)}")
    if args.splice_layer is not None:
        splice_layer = args.splice_layer
    elif args.full_layer_bin or args.full_layer_worker_bin:
        splice_layer = full_layers[-1]
    else:
        splice_layer = len(fixtures) - 1

    prompt = read_prompt(args)
    print(f"[hybrid] prompt_chars={len(prompt)}")
    print(f"[hybrid] fixture_count={len(fixtures)} cycles={n_cycles} splice_layer={splice_layer}")
    print(f"[hybrid] full_layers={full_layers}")
    print(f"[hybrid] c_bin={args.c_bin}")
    print(f"[hybrid] c_bin_prefix_flag={args.c_bin_prefix_flag}")
    if args.prefix_seq_bin:
        print(f"[hybrid] prefix_seq_bin={args.prefix_seq_bin} prefix_seq_fixture={args.prefix_seq_fixture}")
        print(f"[hybrid] prefix_seq_dynamic={args.prefix_seq_dynamic}")
    if args.prefix_seq_worker_bin:
        print(f"[hybrid] prefix_seq_worker_bin={args.prefix_seq_worker_bin} prefix_seq_fixture={args.prefix_seq_fixture}")
    if args.c_bin_worker_bin:
        print(f"[hybrid] c_bin_worker_bin={args.c_bin_worker_bin}")
    if args.owned_session_worker_bin:
        print(f"[hybrid] owned_session_worker_bin={args.owned_session_worker_bin}")
    if args.full_layer_bin:
        print(f"[hybrid] full_layer_bin={args.full_layer_bin} full_layers={full_layers}")
    if args.full_layer_worker_bin:
        print(f"[hybrid] full_layer_worker_bin={args.full_layer_worker_bin} full_layers={full_layers}")
    if args.disable_full_layer_workers:
        print("[hybrid] disable_full_layer_workers=True")
    if args.skip_owned_prefix:
        print(f"[hybrid] skip_owned_prefix=True owned_seq_f32={args.owned_seq_f32}")
    if args.compare_hf:
        print(f"[hybrid] compare_hf_every={args.compare_hf_every}")
    print(f"[hybrid] loading tokenizer from {args.hf}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    print(f"[hybrid] loading model from {args.hf}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    torch.set_num_threads(min(32, os.cpu_count() or 16))
    hidden = int(model.config.hidden_size)
    print(f"[hybrid] model loaded hidden={hidden}")

    token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    steps = []
    hf_past = None
    hf_cached_len = 0
    prefix_seq_worker = None
    hybrid_chain_workers = None
    full_layer_workers = None
    owned_session_worker = None
    owned_session_cfg_path = None

    def load_owned_seq_file(path: str, seq_len: int, hidden_dim: int) -> np.ndarray:
        return read_seq_hidden(Path(path), seq_len, hidden_dim)

    def close_prefix_worker() -> None:
        nonlocal prefix_seq_worker
        if prefix_seq_worker is not None:
            prefix_seq_worker.close()
            prefix_seq_worker = None

    def close_hybrid_workers() -> None:
        nonlocal hybrid_chain_workers
        if hybrid_chain_workers is not None:
            for worker in hybrid_chain_workers:
                worker.close()
            hybrid_chain_workers = None

    def close_full_workers() -> None:
        nonlocal full_layer_workers
        if full_layer_workers is not None:
            for worker in full_layer_workers:
                worker.close()
            full_layer_workers = None

    def close_owned_session_worker() -> None:
        nonlocal owned_session_worker
        if owned_session_worker is not None:
            owned_session_worker.close()
            owned_session_worker = None

    cycle_fixture_chunks = None
    if args.c_bin_worker_bin:
        cycle_fixture_chunks = split_cycle_fixtures(fixtures, n_cycles, args.prefix_seq_worker_bin is not None)

    def ensure_workers_initialized() -> None:
        nonlocal prefix_seq_worker
        nonlocal hybrid_chain_workers
        nonlocal full_layer_workers
        nonlocal owned_session_worker
        nonlocal owned_session_cfg_path
        if owned_session_worker is None and args.owned_session_worker_bin:
            if not args.c_bin_worker_bin:
                ap.error("--owned-session-worker-bin requires --c-bin-worker-bin")
            if not args.full_layer_worker_bin:
                ap.error("--owned-session-worker-bin requires --full-layer-worker-bin")
            owned_session_cfg_path = td_path / "owned_session.cfg"
            write_owned_session_config(
                owned_session_cfg_path,
                fixtures=fixtures,
                full_layers=full_layers,
                prefix_seq_worker_bin=args.prefix_seq_worker_bin,
                prefix_seq_fixture=args.prefix_seq_fixture,
                c_bin_worker_bin=args.c_bin_worker_bin,
                full_layer_worker_bin=args.full_layer_worker_bin,
            )
            if args.worker_init_progress:
                print("[hybrid] init owned-session worker start")
                t0 = time.perf_counter()
            owned_session_worker = OwnedSessionWorker(args.owned_session_worker_bin, args.gguf, owned_session_cfg_path)
            if args.worker_init_progress:
                print(f"[hybrid] init owned-session worker ms={(time.perf_counter() - t0) * 1000.0:.2f}")
            return
        if prefix_seq_worker is None and args.prefix_seq_worker_bin:
            if not args.prefix_seq_fixture:
                ap.error("--prefix-seq-worker-bin requires --prefix-seq-fixture")
            if args.worker_init_progress:
                print("[hybrid] init prefix worker start")
                t0 = time.perf_counter()
            prefix_seq_worker = PrefixSeqWorker(args.prefix_seq_worker_bin, args.gguf, args.prefix_seq_fixture)
            if args.worker_init_progress:
                print(f"[hybrid] init prefix worker ms={(time.perf_counter() - t0) * 1000.0:.2f}")
        if hybrid_chain_workers is None and args.c_bin_worker_bin:
            hybrid_chain_workers = []
            assert cycle_fixture_chunks is not None
            for idx, chunk in enumerate(cycle_fixture_chunks):
                if args.worker_init_progress:
                    print(f"[hybrid] init hybrid worker cycle={idx} start")
                    t0 = time.perf_counter()
                hybrid_chain_workers.append(HybridChainWorker(args.c_bin_worker_bin, args.gguf, chunk))
                if args.worker_init_progress:
                    print(f"[hybrid] init hybrid worker cycle={idx} ms={(time.perf_counter() - t0) * 1000.0:.2f}")
        if full_layer_workers is None and args.full_layer_worker_bin and not args.disable_full_layer_workers:
            full_layer_workers = []
            for layer_idx in full_layers:
                if args.worker_init_progress:
                    print(f"[hybrid] init full-layer worker layer={layer_idx} start")
                    t0 = time.perf_counter()
                full_layer_workers.append(FullLayerWorker(args.full_layer_worker_bin, args.gguf, layer_idx))
                if args.worker_init_progress:
                    print(f"[hybrid] init full-layer worker layer={layer_idx} ms={(time.perf_counter() - t0) * 1000.0:.2f}")

    with tempfile.TemporaryDirectory(prefix="q36_hybrid_") as td:
        td_path = Path(td)
        try:
            for step_idx in range(args.n_predict):
                pct = 100.0 * float(step_idx) / float(max(args.n_predict, 1))
                print(f"[hybrid] progress={step_idx}/{args.n_predict} ({pct:.1f}%)")
                print(f"[hybrid] step={step_idx} seq_len={len(token_ids)}")
                hf_argmax = None
                hf_text = None
                hf_top = None
                hf_baseline_ms = None
                do_compare_hf = bool(args.compare_hf and args.compare_hf_every > 0 and (step_idx % args.compare_hf_every) == 0)

                if args.skip_owned_prefix:
                    owned_seq = load_owned_seq_file(args.owned_seq_f32, len(token_ids), hidden)
                    owned_meta = {
                        "owned_prefix_ms": 0.0,
                        "cycle_reports": "",
                        "cycle_timings": [],
                        "prefix_seq_report": None,
                        "prefix_seq_ms": None,
                    }
                else:
                    ensure_workers_initialized()
                    if owned_session_worker is not None:
                        owned_seq, owned_worker_meta = owned_session_worker.run_for_token_ids(token_ids, hidden)
                        owned_meta = {
                            "owned_prefix_ms": owned_worker_meta["worker_ms"],
                            "cycle_reports": "[native owned session worker]",
                            "cycle_timings": [],
                            "prefix_seq_report": None,
                            "prefix_seq_ms": None,
                            "prefix_seq_worker_meta": None,
                            "hybrid_chain_worker_meta": None,
                            "full_layer_worker_meta": None,
                            "owned_session_worker_meta": owned_worker_meta,
                        }
                    else:
                        owned_seq, owned_meta = run_owned_prefix_cycles(
                            td_path=td_path,
                            token_ids=token_ids,
                            hidden=hidden,
                            gguf=args.gguf,
                            fixtures=fixtures,
                            full_layers=full_layers,
                            c_bin=args.c_bin,
                            c_bin_prefix_flag=args.c_bin_prefix_flag,
                            full_layer_bin=args.full_layer_bin,
                            prefix_seq_bin=args.prefix_seq_bin,
                            prefix_seq_fixture=args.prefix_seq_fixture,
                            prefix_seq_dynamic=args.prefix_seq_dynamic,
                            prefix_seq_worker=prefix_seq_worker,
                            hybrid_chain_workers=hybrid_chain_workers,
                            full_layer_workers=full_layer_workers,
                            step_idx=step_idx,
                        )
                    print(f"[hybrid] owned_prefix_ms={owned_meta['owned_prefix_ms']:.2f}")
                    if owned_meta.get("owned_session_worker_meta") is not None:
                        print(f"[hybrid] owned_session_worker_ms={owned_meta['owned_session_worker_meta']['worker_ms']:.2f}")
                    if owned_meta.get("prefix_seq_worker_meta") is not None:
                        print(f"[hybrid] prefix_seq_worker_ms={owned_meta['prefix_seq_worker_meta']['worker_ms']:.2f}")
                    if owned_meta.get("hybrid_chain_worker_meta") is not None:
                        worker_meta = owned_meta["hybrid_chain_worker_meta"]
                        worker_items = sorted(worker_meta.items())
                        worker_total = sum(item["worker_ms"] for _, item in worker_items)
                        print(f"[hybrid] hybrid_chain_worker_ms_total={worker_total:.2f}")
                        for cycle_name, item in worker_items:
                            print(f"[hybrid] {cycle_name}_hybrid_worker_ms={item['worker_ms']:.2f}")
                    if owned_meta.get("full_layer_worker_meta") is not None:
                        full_worker_meta = owned_meta["full_layer_worker_meta"]
                        full_worker_items = sorted(full_worker_meta.items())
                        full_worker_total = sum(item["worker_ms"] for _, item in full_worker_items)
                        print(f"[hybrid] full_layer_worker_ms_total={full_worker_total:.2f}")
                        for cycle_name, item in full_worker_items:
                            print(f"[hybrid] {cycle_name}_full_layer_worker_ms={item['worker_ms']:.2f}")
                    if args.dump_owned_seq and step_idx == 0:
                        owned_seq.astype(np.float32).tofile(args.dump_owned_seq)
                        print(f"[hybrid] dumped_owned_seq={args.dump_owned_seq}")

                if args.close_full_layer_workers_before_hf:
                    print("[hybrid] closing full-layer workers before hf")
                    close_full_workers()
                if args.close_all_workers_before_hf:
                    print("[hybrid] closing all workers before hf")
                    close_full_workers()
                    close_hybrid_workers()
                    close_prefix_worker()
                    close_owned_session_worker()

                if args.skip_hf_patched:
                    patched_step = {
                        "next_id": -1,
                        "next_text": "",
                        "patched_topk": [],
                        "hf_patched_ms": 0.0,
                    }
                else:
                    patched_step = run_hf_patched_compare(
                        model=model,
                        token_ids=token_ids,
                        owned_seq=owned_seq,
                        splice_layer=splice_layer,
                        tokenizer=tokenizer,
                        top_k=args.top_k,
                        layer_progress=args.hf_patched_layer_progress,
                        setup_progress=args.hf_patched_setup_progress,
                    )

                if do_compare_hf:
                    print("[hybrid] running hf baseline forward")
                    hf_base, hf_past, hf_cached_len = run_hf_baseline_step(
                        model=model,
                        token_ids=token_ids,
                        hf_past=hf_past,
                        hf_cached_len=hf_cached_len,
                        tokenizer=tokenizer,
                        top_k=args.top_k,
                        layer_progress=args.hf_baseline_layer_progress,
                    )
                    hf_argmax = hf_base["hf_next_id"]
                    hf_text = hf_base["hf_next_text"]
                    hf_top = hf_base["hf_topk"]
                    hf_baseline_ms = hf_base["hf_baseline_ms"]
                    print(f"[hybrid] hf_baseline_ms={hf_baseline_ms:.2f}")
                next_id = patched_step["next_id"]
                next_text = patched_step["next_text"]
                if not args.skip_hf_patched:
                    token_ids.append(next_id)
                    decoded = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
                else:
                    decoded = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)

                step = {
                    "step": step_idx,
                    "seq_len": len(token_ids) - 1,
                    "next_id": next_id,
                    "next_text": next_text,
                    "patched_topk": patched_step["patched_topk"],
                    "n_cycles": n_cycles,
                    "cycle_reports": owned_meta["cycle_reports"],
                    "cycle_timings": owned_meta["cycle_timings"],
                    "owned_prefix_ms": owned_meta["owned_prefix_ms"],
                    "hf_patched_ms": patched_step["hf_patched_ms"],
                }
                if owned_meta["prefix_seq_report"] is not None:
                    step["prefix_seq_report"] = owned_meta["prefix_seq_report"]
                    step["prefix_seq_ms"] = owned_meta["prefix_seq_ms"]
                if owned_meta.get("prefix_seq_worker_meta") is not None:
                    step["prefix_seq_worker_meta"] = owned_meta["prefix_seq_worker_meta"]
                if owned_meta.get("hybrid_chain_worker_meta") is not None:
                    step["hybrid_chain_worker_meta"] = owned_meta["hybrid_chain_worker_meta"]
                if owned_meta.get("full_layer_worker_meta") is not None:
                    step["full_layer_worker_meta"] = owned_meta["full_layer_worker_meta"]
                if do_compare_hf:
                    step["hf_next_id"] = hf_argmax
                    step["hf_next_text"] = hf_text
                    step["hf_topk"] = hf_top
                    step["argmax_equal"] = hf_argmax == next_id
                    step["hf_baseline_ms"] = hf_baseline_ms
                else:
                    step["hf_compare_skipped"] = bool(args.compare_hf)
                steps.append(step)

                print(f"[hybrid] next_id={next_id} next_text={json.dumps(next_text)}")
                if do_compare_hf:
                    print(f"[hybrid] hf_next_id={hf_argmax} hf_next_text={json.dumps(hf_text)} equal={hf_argmax == next_id}")
                elif args.compare_hf:
                    print("[hybrid] hf baseline skipped for this step")
                print(f"[hybrid] hf_patched_ms={patched_step['hf_patched_ms']:.2f}")
                print(f"[hybrid] decoded_so_far={json.dumps(decoded)}")
                if args.skip_hf_patched:
                    print("[hybrid] skip_hf_patched stopping after owned prefix")
                    break
            print(f"[hybrid] progress={args.n_predict}/{args.n_predict} (100.0%)")
        finally:
            close_prefix_worker()
            close_hybrid_workers()
            close_full_workers()
            close_owned_session_worker()

    generated_ids = token_ids[len(tokenizer(prompt, return_tensors='pt')["input_ids"][0].tolist()):]
    result = {
        "prompt": prompt,
        "fixtures": fixtures,
        "full_layers": full_layers,
        "n_cycles": n_cycles,
        "splice_layer": splice_layer,
        "generated_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_ids, clean_up_tokenization_spaces=False),
        "full_text": tokenizer.decode(token_ids, clean_up_tokenization_spaces=False),
        "steps": steps,
    }
    print(f"generated_text: {json.dumps(result['generated_text'])}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"json_out: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
