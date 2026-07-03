#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from qwen36_v0_source_verify import TensorSpec, all_specs, transform_shape

try:
    from gguf import GGUFReader
except ImportError:
    GGUFReader = None


TYPE_INFO = {
    "f32": {"requires_imatrix": False},
    "q4_k": {"requires_imatrix": False},
    "q5_k": {"requires_imatrix": False},
    "q6_k": {"requires_imatrix": False},
    "iq2_xxs": {"requires_imatrix": True},
    "iq2_s": {"requires_imatrix": False},
    "iq3_s": {"requires_imatrix": False},
}


def target_type_for_tensor(name: str) -> str:
    if name == "token_embd.weight":
        return "q4_k"
    if name == "output_norm.weight":
        return "f32"
    if name == "output.weight":
        return "q4_k"

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
        return "f32"
    if suffix in {"ffn_gate_exps.weight", "ffn_up_exps.weight"}:
        return "iq2_xxs"
    if suffix == "ffn_down_exps.weight":
        return "iq3_s" if layer >= 34 else "iq2_s"
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
        return "q5_k"
    if suffix in {"ffn_down_shexp.weight", "ssm_out.weight"}:
        return "q6_k"
    raise KeyError(name)


def apply_transform(spec: TensorSpec, arr: np.ndarray) -> np.ndarray:
    if spec.mode == "direct":
        return np.transpose(arr, tuple(reversed(range(arr.ndim)))).copy()
    if spec.mode == "slice_gate_half":
        return np.transpose(arr[:, : arr.shape[1] // 2, :], (2, 1, 0)).copy()
    if spec.mode == "slice_up_half":
        return np.transpose(arr[:, arr.shape[1] // 2 :, :], (2, 1, 0)).copy()
    if spec.mode == "squeeze0":
        return np.squeeze(arr, axis=0).copy()
    if spec.mode == "squeeze1_reverse":
        return np.transpose(np.squeeze(arr, axis=1), (1, 0)).copy()
    raise ValueError(spec.mode)


def load_weight_map(hf_dir: Path) -> dict[str, str]:
    index_path = hf_dir / "model.safetensors.index.json"
    return json.loads(index_path.read_text())["weight_map"]


def open_handle_cache():
    cache: dict[Path, object] = {}

    def get_handle(path: Path):
        handle = cache.get(path)
        if handle is None:
            handle = safe_open(str(path), framework="pt", device="cpu")
            cache[path] = handle
        return handle

    return get_handle


def stats_line(arr: np.ndarray) -> str:
    flat = arr.astype(np.float32, copy=False).reshape(-1)
    return (
        f"min={float(flat.min()):.6f} "
        f"max={float(flat.max()):.6f} "
        f"mean={float(flat.mean()):.6f} "
        f"std={float(flat.std()):.6f}"
    )


def find_spec(name: str) -> TensorSpec:
    for spec in all_specs():
        if spec.gguf_name == name:
            return spec
    raise KeyError(name)


def print_imatrix_list() -> int:
    need = []
    for spec in all_specs():
        qtype = target_type_for_tensor(spec.gguf_name)
        if TYPE_INFO[qtype]["requires_imatrix"]:
            need.append((spec.gguf_name, qtype, spec.gguf_shape))
    print(f"imatrix_targets: {len(need)}")
    for name, qtype, shape in need[:20]:
        print(f"  {name} type={qtype} gguf_shape={shape}")
    if len(need) > 20:
        print(f"  ... {len(need) - 20} more")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe one DS4Style-v0 Qwen3.6 tensor from HF source")
    ap.add_argument("--hf", required=True, help="HF model directory")
    ap.add_argument("--tensor", help="Exact GGUF tensor name to materialize")
    ap.add_argument("--template", help="Optional GGUF template for structural comparison")
    ap.add_argument("--dump-npy", help="Optional output path for transformed tensor")
    ap.add_argument("--meta-only", action="store_true", help="Do not load tensor payload; print contract/source metadata only")
    ap.add_argument("--list-imatrix", action="store_true", help="List tensors whose target quant type needs an importance matrix")
    args = ap.parse_args()

    if args.list_imatrix:
        return print_imatrix_list()
    if not args.tensor:
        ap.error("--tensor is required unless --list-imatrix is used")

    hf_dir = Path(args.hf)
    spec = find_spec(args.tensor)
    qtype = target_type_for_tensor(spec.gguf_name)
    weight_map = load_weight_map(hf_dir)
    shard_name = weight_map[spec.hf_name]
    shard_path = hf_dir / shard_name
    get_handle = open_handle_cache()
    print(f"gguf_tensor: {spec.gguf_name}")
    print(f"target_quant: {qtype}")
    print(f"requires_imatrix: {str(TYPE_INFO[qtype]['requires_imatrix']).lower()}")
    print(f"hf_tensor: {spec.hf_name}")
    print(f"hf_shard: {shard_name}")
    print(f"mode: {spec.mode}")
    print(f"hf_shape_expected: {spec.hf_expected_shape}")
    print(f"gguf_shape_expected: {spec.gguf_shape}")

    if not args.meta_only:
        handle = get_handle(shard_path)
        src = handle.get_tensor(spec.hf_name)
        src_arr = src.detach()
        if hasattr(src_arr, "float"):
            src_arr = src_arr.float()
        if hasattr(src_arr, "cpu"):
            src_arr = src_arr.cpu()
        if hasattr(src_arr, "numpy"):
            src_arr = src_arr.numpy()
        src_arr = np.asarray(src_arr, dtype=np.float32)
        hf_shape = tuple(int(x) for x in src_arr.shape)
        gguf_shape = transform_shape(spec, hf_shape)
        out = apply_transform(spec, src_arr)

        print(f"hf_shape: {hf_shape}")
        print(f"gguf_shape_from_rule: {gguf_shape}")
        print(f"materialized_shape: {tuple(int(x) for x in out.shape)}")
        print(f"materialized_dtype: {out.dtype}")
        print(f"materialized_nbytes: {out.nbytes}")
        print(f"materialized_stats: {stats_line(out)}")

        if tuple(int(x) for x in out.shape) != spec.gguf_shape:
            print("error: transformed tensor shape mismatch")
            return 1

    if args.template:
        if GGUFReader is None:
            print("template_compare: unavailable (gguf python package not importable)")
        else:
            reader = GGUFReader(args.template)
            found = None
            for tensor in reader.tensors:
                if tensor.name == spec.gguf_name:
                    found = tensor
                    break
            if found is None:
                print("template_compare: missing tensor")
                return 1
            print(f"template_shape: {tuple(int(x) for x in found.shape)}")
            print(f"template_type: {found.tensor_type.name}")
            print(f"template_nbytes: {found.n_bytes}")

    if args.dump_npy:
        if args.meta_only:
            print("error: --dump-npy requires full materialization")
            return 1
        out_path = Path(args.dump_npy)
        np.save(out_path, out)
        print(f"dump_npy: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
