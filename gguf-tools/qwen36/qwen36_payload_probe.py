#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
from safetensors import safe_open

from gguf import GGUFReader
from gguf.quants import dequantize
from qwen36_v0_source_verify import all_specs


def load_weight_map(hf_dir: Path) -> dict[str, str]:
    import json

    return json.loads((hf_dir / "model.safetensors.index.json").read_text())["weight_map"]


def open_handle_cache():
    cache: dict[Path, object] = {}

    def get_handle(path: Path):
        handle = cache.get(path)
        if handle is None:
            handle = safe_open(str(path), framework="pt", device="cpu")
            cache[path] = handle
        return handle

    return get_handle


def find_spec(name: str):
    for spec in all_specs():
        if spec.gguf_name == name:
            return spec
    raise KeyError(name)


def load_source_storage(spec, hf_dir: Path, weight_map: dict[str, str], get_handle) -> np.ndarray:
    shard = hf_dir / weight_map[spec.hf_name]
    handle = get_handle(shard)
    arr = handle.get_tensor(spec.hf_name).detach()
    if hasattr(arr, "float"):
        arr = arr.float()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    arr = np.asarray(arr, dtype=np.float32)

    if spec.mode == "direct":
        out = arr
    elif spec.mode == "slice_gate_half":
        out = arr[:, : arr.shape[1] // 2, :]
    elif spec.mode == "slice_up_half":
        out = arr[:, arr.shape[1] // 2 :, :]
    elif spec.mode == "squeeze0":
        out = np.squeeze(arr, axis=0)
    elif spec.mode == "squeeze1_reverse":
        out = np.squeeze(arr, axis=1)
    else:
        raise ValueError(spec.mode)
    return np.ascontiguousarray(out, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1).astype(np.float64, copy=False)
    bv = b.reshape(-1).astype(np.float64, copy=False)
    an = float(np.linalg.norm(av))
    bn = float(np.linalg.norm(bv))
    if an == 0.0 or bn == 0.0:
        return float("nan")
    return float(np.dot(av, bv) / (an * bn))


def summarize(name: str, src: np.ndarray, got: np.ndarray, qtype: str) -> None:
    diff = got - src
    mse = float(np.mean(diff * diff))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    print(f"tensor: {name}")
    print(f"  gguf_type: {qtype}")
    print(f"  shape: {tuple(int(x) for x in src.shape)}")
    print(f"  src_minmax: {float(src.min()):.6f} {float(src.max()):.6f}")
    print(f"  got_minmax: {float(got.min()):.6f} {float(got.max()):.6f}")
    print(f"  rmse: {rmse:.8f}")
    print(f"  mae: {mae:.8f}")
    print(f"  max_abs: {max_abs:.8f}")
    print(f"  cosine: {cosine(src, got):.8f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare selected GGUF tensors against HF source after dequantization")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tensor", action="append", required=True, help="Exact GGUF tensor name, may be repeated")
    args = ap.parse_args()

    hf_dir = Path(args.hf)
    reader = GGUFReader(args.gguf)
    weight_map = load_weight_map(hf_dir)
    get_handle = open_handle_cache()
    tensor_map = {t.name: t for t in reader.tensors}

    for name in args.tensor:
        spec = find_spec(name)
        src = load_source_storage(spec, hf_dir, weight_map, get_handle)
        gt = tensor_map[name]
        got = dequantize(gt.data, gt.tensor_type).astype(np.float32, copy=False)
        if got.shape != src.shape:
            print(f"tensor: {name}")
            print(f"  error: shape mismatch source={src.shape} dequant={got.shape}")
            return 1
        summarize(name, src, got, gt.tensor_type.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
