#!/usr/bin/env python3
import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterable
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]

from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter
from qwen36_v0_source_verify import TensorSpec, all_specs


@dataclass(frozen=True)
class TargetType:
    name: str
    type_id: int
    requires_imatrix: bool


TYPE_F32 = TargetType("f32", 0, False)
TYPE_Q4_K = TargetType("q4_k", 12, False)
TYPE_Q5_K = TargetType("q5_k", 13, False)
TYPE_Q6_K = TargetType("q6_k", 14, False)
TYPE_Q8_0 = TargetType("q8_0", 8, False)
TYPE_IQ2_XXS = TargetType("iq2_xxs", 16, True)
TYPE_IQ3_S = TargetType("iq3_s", 21, False)
TYPE_IQ2_S = TargetType("iq2_s", 22, False)

PROFILE_V0 = "v0"
PROFILE_Q4XL_DOWN_Q4 = "q4xl-down-q4"
PROFILE_Q4_ORIGINAL_Q2 = "q4-original-q2"


_routed_policy: dict[str, TargetType] = {}


def target_type_for_tensor_v0(name: str) -> TargetType:
    if name == "token_embd.weight":
        return TYPE_Q4_K
    if name == "output_norm.weight":
        return TYPE_F32
    if name == "output.weight":
        return TYPE_Q4_K

    parts = name.split(".")
    if len(parts) < 3 or parts[0] != "blk":
        raise KeyError(name)
    layer = int(parts[1])
    suffix = ".".join(parts[2:])

    if suffix in {
        "attn_norm.weight",
        "post_attention_norm.weight",
        "ffn_gate_inp.weight",
        "ffn_gate_inp_shexp.weight",
        "ssm_a",
        "ssm_alpha.weight",
        "ssm_beta.weight",
        "ssm_conv1d.weight",
        "ssm_dt.bias",
        "ssm_norm.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
    }:
        return TYPE_F32
    if suffix in {"ffn_gate_exps.weight", "ffn_up_exps.weight"}:
        return TYPE_IQ2_XXS
    if suffix == "ffn_down_exps.weight":
        return TYPE_IQ3_S if layer >= 34 else TYPE_IQ2_S
    if suffix in {
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "attn_gate.weight",
        "attn_qkv.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
    }:
        return TYPE_Q5_K
    if suffix in {"ffn_down_shexp.weight", "ssm_out.weight"}:
        return TYPE_Q6_K
    raise KeyError(name)


def target_type_for_tensor_q4xl_down_q4(name: str) -> TargetType:
    if name == "token_embd.weight":
        return TYPE_Q8_0
    if name == "output_norm.weight":
        return TYPE_F32
    if name == "output.weight":
        return TYPE_Q8_0

    parts = name.split(".")
    if len(parts) < 3 or parts[0] != "blk":
        raise KeyError(name)
    layer = int(parts[1])
    suffix = ".".join(parts[2:])

    if suffix in {
        "attn_norm.weight",
        "post_attention_norm.weight",
        "ffn_gate_inp.weight",
        "ffn_gate_inp_shexp.weight",
        "ssm_a",
        "ssm_alpha.weight",
        "ssm_beta.weight",
        "ssm_conv1d.weight",
        "ssm_dt.bias",
        "ssm_norm.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
    }:
        return TYPE_F32
    if suffix in {"ffn_gate_exps.weight", "ffn_up_exps.weight"}:
        return TYPE_Q5_K if layer == 1 else TYPE_Q4_K
    if suffix == "ffn_down_exps.weight":
        return TYPE_Q4_K
    if suffix in {
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_down_shexp.weight",
        "attn_gate.weight",
        "attn_qkv.weight",
        "ssm_out.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
    }:
        return TYPE_Q8_0
    raise KeyError(name)


def target_type_for_tensor_q4_original_q2(name: str) -> TargetType:
    if name in _routed_policy:
        return _routed_policy[name]
    return target_type_for_tensor_q4xl_down_q4(name)


def target_type_for_tensor(profile: str, name: str) -> TargetType:
    if profile == PROFILE_V0:
        return target_type_for_tensor_v0(name)
    if profile == PROFILE_Q4XL_DOWN_Q4:
        return target_type_for_tensor_q4xl_down_q4(name)
    if profile == PROFILE_Q4_ORIGINAL_Q2:
        return target_type_for_tensor_q4_original_q2(name)
    raise KeyError(profile)


def should_rewrite_from_hf(profile: str, name: str) -> bool:
    if profile == PROFILE_V0:
        return True
    if profile == PROFILE_Q4XL_DOWN_Q4:
        return name.endswith("ffn_down_exps.weight")
    if profile == PROFILE_Q4_ORIGINAL_Q2:
        return name in _routed_policy
    raise KeyError(profile)


def load_routed_policy(path: str | None) -> dict[str, TargetType]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_policy = data.get("policy", data)
    out: dict[str, TargetType] = {}
    type_map = {
        "q2_k": TargetType("q2_k", 10, False),
        "q4_k": TYPE_Q4_K,
        "iq2_xxs": TYPE_IQ2_XXS,
    }
    forbidden = {"iq2_s", "iq3_s"}
    for name, type_name in raw_policy.items():
        if type_name in forbidden:
            raise ValueError(f"{name}: forbidden routed type for q4-original-q2 lane: {type_name}")
        qtype = type_map.get(type_name)
        if qtype is None:
            raise ValueError(f"{name}: unsupported routed policy type {type_name}")
        out[name] = qtype
    return out


def normalize_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    return value


def copy_template_kvs(reader: GGUFReader, writer: GGUFWriter) -> None:
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        if key == "general.architecture":
            continue
        value = normalize_value(field.contents())
        if key == "general.alignment":
            writer.add_custom_alignment(int(value))
            continue
        main_type = field.types[0]
        if main_type == GGUFValueType.ARRAY:
            writer.add_key_value(key, value, main_type, field.types[1])
        else:
            writer.add_key_value(key, value, main_type)


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


def load_source_tensor(spec: TensorSpec, hf_dir: Path, weight_map: dict[str, str], get_handle) -> np.ndarray:
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

    if tuple(reversed(out.shape)) != spec.gguf_shape:
        raise ValueError(
            f"{spec.gguf_name}: storage shape {tuple(out.shape)} does not match GGUF raw_shape {spec.gguf_shape}"
        )
    return np.ascontiguousarray(out, dtype=np.float32)


class Ds4QuantLib:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))
        self.lib.ds4q_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self.lib.ds4q_row_size.restype = ctypes.c_size_t
        self.lib.ds4q_requires_imatrix.argtypes = [ctypes.c_int]
        self.lib.ds4q_requires_imatrix.restype = ctypes.c_bool
        self.lib.ds4q_quantize_init.argtypes = [ctypes.c_int]
        self.lib.ds4q_quantize_init.restype = None
        self.lib.ds4q_quantize_chunk.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.ds4q_quantize_chunk.restype = ctypes.c_size_t

    def row_size(self, qtype: TargetType, ncols: int) -> int:
        return int(self.lib.ds4q_row_size(qtype.type_id, int(ncols)))

    def _quantize_single(self, qtype: TargetType, storage: np.ndarray, allow_synthetic: bool) -> tuple[np.ndarray, bool]:
        if storage.dtype != np.float32 or not storage.flags.c_contiguous:
            raise ValueError("quantize expects contiguous float32 storage")
        ncols = int(storage.shape[-1]) if storage.ndim > 1 else int(storage.shape[0])
        if ncols <= 0:
            raise ValueError("bad ncols")
        nrows = int(storage.size // ncols)
        row_size = self.row_size(qtype, ncols)
        out = np.empty(nrows * row_size, dtype=np.uint8)
        src = storage.reshape(-1)
        imatrix_ptr = None
        used_synthetic = False
        synthetic = None
        if qtype.requires_imatrix:
            if not allow_synthetic:
                raise ValueError(f"{qtype.name} requires imatrix and synthetic fallback is disabled")
            synthetic = np.sum(storage.reshape(nrows, ncols) ** 2, axis=0, dtype=np.float32)
            imatrix_ptr = synthetic.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            used_synthetic = True
        self.lib.ds4q_quantize_init(qtype.type_id)
        written = int(
            self.lib.ds4q_quantize_chunk(
                qtype.type_id,
                src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_void_p(out.ctypes.data),
                0,
                nrows,
                ncols,
                imatrix_ptr,
            )
        )
        if written != out.nbytes:
            raise ValueError(f"{qtype.name}: wrote {written} bytes, expected {out.nbytes}")
        return out, used_synthetic

    def quantize(self, qtype: TargetType, storage: np.ndarray, allow_synthetic: bool, threads: int) -> tuple[np.ndarray, bool]:
        if threads <= 1 or storage.ndim < 2:
            return self._quantize_single(qtype, storage, allow_synthetic)

        axis0 = int(storage.shape[0])
        if axis0 < 2:
            return self._quantize_single(qtype, storage, allow_synthetic)

        worker_count = min(max(1, threads), axis0)
        if worker_count == 1:
            return self._quantize_single(qtype, storage, allow_synthetic)

        # Split across the leading dimension:
        # - 2D tensors: row chunks
        # - 3D routed expert tensors: expert chunks
        bounds = []
        start = 0
        for i in range(worker_count):
            end = (axis0 * (i + 1)) // worker_count
            if end > start:
                bounds.append((start, end))
            start = end
        if len(bounds) <= 1:
            return self._quantize_single(qtype, storage, allow_synthetic)

        def work(bound: tuple[int, int]) -> tuple[np.ndarray, bool]:
            lo, hi = bound
            sub = np.ascontiguousarray(storage[lo:hi], dtype=np.float32)
            return self._quantize_single(qtype, sub, allow_synthetic)

        parts: list[np.ndarray] = []
        used_synthetic = False
        with ThreadPoolExecutor(max_workers=len(bounds)) as pool:
            for part, used in pool.map(work, bounds):
                parts.append(part)
                used_synthetic = used_synthetic or used
        return np.concatenate(parts), used_synthetic


def should_keep(name: str, only: set[str], limit: int | None, kept: int) -> bool:
    if only:
        return name in only
    if limit is None:
        return True
    return kept < limit


def selected_template_tensors(reader: GGUFReader, only: set[str], limit: int | None):
    kept = 0
    for tensor in reader.tensors:
        if should_keep(tensor.name, only, limit, kept):
            kept += 1
            yield tensor


def template_storage_shape(tensor) -> tuple[int, ...]:
    # GGUFReader exposes logical GGUF-order dims. GGUFWriter.add_tensor expects
    # storage-order raw_shape and writes its reverse into tensor metadata.
    return tuple(int(x) for x in reversed(tensor.shape))


def main() -> int:
    ap = argparse.ArgumentParser(description="Experimental Qwen3.6-35B-A3B exporter with selectable quantization profile")
    ap.add_argument("--hf", required=True, help="HF safetensors directory")
    ap.add_argument("--template", required=True, help="Template Q8_0 GGUF")
    ap.add_argument(
        "--lib",
        default=str(REPO_ROOT / "gguf-tools" / "libds4q.so"),
        help="Local shared quantization library",
    )
    ap.add_argument(
        "--profile",
        choices=[PROFILE_V0, PROFILE_Q4XL_DOWN_Q4, PROFILE_Q4_ORIGINAL_Q2],
        default=PROFILE_V0,
        help="Target quantization profile",
    )
    ap.add_argument(
        "--routed-policy-json",
        help="Required for q4-original-q2: routed tensor -> target type table from qwen36_q4_original_q2_policy.py",
    )
    ap.add_argument("--out", help="Output GGUF path for full export")
    ap.add_argument("--dry-run", action="store_true", help="Process tensors and report stats without writing a GGUF")
    ap.add_argument("--limit", type=int, help="Debug: process only the first N tensors in template order")
    ap.add_argument("--only-tensor", action="append", default=[], help="Debug: process only this exact tensor name, may be repeated")
    ap.add_argument("--no-synthetic-imatrix", action="store_true", help="Fail instead of using weight-energy fallback for iq2_xxs")
    ap.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1), help="Quantization worker count")
    ap.add_argument("--allow-partial-write", action="store_true", help="Allow writing a debug GGUF with only a subset of tensors")
    args = ap.parse_args()

    if not args.dry_run and not args.out:
        ap.error("either --dry-run or --out is required")
    global _routed_policy
    _routed_policy = load_routed_policy(args.routed_policy_json)
    if args.profile == PROFILE_Q4_ORIGINAL_Q2 and not _routed_policy:
        ap.error("--profile q4-original-q2 requires --routed-policy-json")
    if args.out and (args.limit is not None or args.only_tensor) and not args.allow_partial_write:
        ap.error("refusing to write a partial GGUF; use --dry-run with --limit/--only-tensor")

    hf_dir = Path(args.hf)
    template_path = Path(args.template)
    spec_map = {spec.gguf_name: spec for spec in all_specs()}
    reader = GGUFReader(template_path)
    quant = Ds4QuantLib(Path(args.lib))
    weight_map = load_weight_map(hf_dir)
    get_handle = open_handle_cache()
    only = set(args.only_tensor)

    writer = None
    if args.out:
        arch = str(reader.get_field("general.architecture").contents())
        writer = GGUFWriter(args.out, arch=arch, use_temp_file=True)
        copy_template_kvs(reader, writer)

    total = 0
    total_bytes = 0
    synthetic_count = 0
    synthetic_names: list[str] = []

    print(f"threads: {args.threads}")

    for tensor in selected_template_tensors(reader, only, args.limit):
        qtype = target_type_for_tensor(args.profile, tensor.name)
        total += 1
        rewrite = should_rewrite_from_hf(args.profile, tensor.name)

        if not rewrite:
            payload = np.ascontiguousarray(tensor.data)
            total_bytes += int(payload.nbytes)
            raw_shape = template_storage_shape(tensor)
            if writer is not None:
                if tensor.tensor_type == GGMLQuantizationType.F32:
                    writer.add_tensor(tensor.name, payload, raw_shape=raw_shape)
                else:
                    writer.add_tensor(
                        tensor.name,
                        payload.view(np.int8),
                        raw_shape=raw_shape,
                        raw_dtype=tensor.tensor_type,
                    )
            print(
                f"tensor: {tensor.name} "
                f"profile={args.profile} "
                f"type=template:{tensor.tensor_type.name.lower()} raw_shape={tuple(int(x) for x in tensor.shape)} "
                f"storage_shape={raw_shape} "
                f"bytes={payload.nbytes}"
            )
        else:
            spec = spec_map.get(tensor.name)
            if spec is None:
                raise KeyError(f"missing source spec for {tensor.name}")
            storage = load_source_tensor(spec, hf_dir, weight_map, get_handle)

            if qtype == TYPE_F32:
                payload = storage
                total_bytes += int(payload.nbytes)
                if writer is not None:
                    writer.add_tensor(tensor.name, payload, raw_shape=storage.shape)
            else:
                payload, used_synth = quant.quantize(
                    qtype,
                    storage,
                    allow_synthetic=not args.no_synthetic_imatrix,
                    threads=args.threads,
                )
                total_bytes += int(payload.nbytes)
                if used_synth:
                    synthetic_count += 1
                    synthetic_names.append(tensor.name)
                if writer is not None:
                    # gguf-py rewrites uint8 tensor shapes as "byte-row shapes" for
                    # quantized tensors.  We already know the exact logical GGUF
                    # shape from the narrow contract, so pass the raw bytes through
                    # an int8 view to preserve payload bytes while keeping the
                    # tensor-info shape under our control.
                    writer.add_tensor(
                        tensor.name,
                        payload.view(np.int8),
                        raw_shape=storage.shape,
                        raw_dtype=GGMLQuantizationType(qtype.type_id),
                    )

            print(
                f"tensor: {tensor.name} "
                f"profile={args.profile} "
                f"type={qtype.name} raw_shape={spec.gguf_shape} "
                f"storage_shape={tuple(int(x) for x in storage.shape)} "
                f"bytes={payload.nbytes}"
            )

    print(f"processed_tensors: {total}")
    print(f"processed_bytes: {total_bytes}")
    print(f"synthetic_imatrix_tensors: {synthetic_count}")
    if synthetic_names:
        for name in synthetic_names[:20]:
            print(f"synthetic_imatrix: {name}")
        if len(synthetic_names) > 20:
            print(f"synthetic_imatrix_more: {len(synthetic_names) - 20}")

    if writer is not None:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        print(f"wrote: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
