#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open

from qwen36_hf_contract_probe import HYBRID_LAYERS, FULL_LAYERS, all_mappings


@dataclass(frozen=True)
class TensorSpec:
    gguf_name: str
    gguf_shape: tuple[int, ...]
    hf_name: str
    mode: str
    hf_expected_shape: tuple[int, ...]


def gguf_root_specs() -> list[TensorSpec]:
    return [
        TensorSpec("token_embd.weight", (2048, 248320), "model.language_model.embed_tokens.weight", "direct", (248320, 2048)),
        TensorSpec("output_norm.weight", (2048,), "model.language_model.norm.weight", "direct", (2048,)),
        TensorSpec("output.weight", (2048, 248320), "lm_head.weight", "direct", (248320, 2048)),
    ]


def layer_specs(layer: int) -> list[TensorSpec]:
    base = f"model.language_model.layers.{layer}"
    specs = [
        TensorSpec(f"blk.{layer}.attn_norm.weight", (2048,), f"{base}.input_layernorm.weight", "direct", (2048,)),
        TensorSpec(f"blk.{layer}.post_attention_norm.weight", (2048,), f"{base}.post_attention_layernorm.weight", "direct", (2048,)),
        TensorSpec(f"blk.{layer}.ffn_gate_inp.weight", (2048, 256), f"{base}.mlp.gate.weight", "direct", (256, 2048)),
        TensorSpec(f"blk.{layer}.ffn_gate_inp_shexp.weight", (2048,), f"{base}.mlp.shared_expert_gate.weight", "squeeze0", (1, 2048)),
        TensorSpec(f"blk.{layer}.ffn_gate_exps.weight", (2048, 512, 256), f"{base}.mlp.experts.gate_up_proj", "slice_gate_half", (256, 1024, 2048)),
        TensorSpec(f"blk.{layer}.ffn_up_exps.weight", (2048, 512, 256), f"{base}.mlp.experts.gate_up_proj", "slice_up_half", (256, 1024, 2048)),
        TensorSpec(f"blk.{layer}.ffn_down_exps.weight", (512, 2048, 256), f"{base}.mlp.experts.down_proj", "direct", (256, 2048, 512)),
        TensorSpec(f"blk.{layer}.ffn_gate_shexp.weight", (2048, 512), f"{base}.mlp.shared_expert.gate_proj.weight", "direct", (512, 2048)),
        TensorSpec(f"blk.{layer}.ffn_up_shexp.weight", (2048, 512), f"{base}.mlp.shared_expert.up_proj.weight", "direct", (512, 2048)),
        TensorSpec(f"blk.{layer}.ffn_down_shexp.weight", (512, 2048), f"{base}.mlp.shared_expert.down_proj.weight", "direct", (2048, 512)),
    ]
    if layer in HYBRID_LAYERS:
        specs.extend([
            TensorSpec(f"blk.{layer}.attn_gate.weight", (2048, 4096), f"{base}.linear_attn.in_proj_z.weight", "direct", (4096, 2048)),
            TensorSpec(f"blk.{layer}.attn_qkv.weight", (2048, 8192), f"{base}.linear_attn.in_proj_qkv.weight", "direct", (8192, 2048)),
            TensorSpec(f"blk.{layer}.ssm_a", (32,), f"{base}.linear_attn.A_log", "direct", (32,)),
            TensorSpec(f"blk.{layer}.ssm_alpha.weight", (2048, 32), f"{base}.linear_attn.in_proj_a.weight", "direct", (32, 2048)),
            TensorSpec(f"blk.{layer}.ssm_beta.weight", (2048, 32), f"{base}.linear_attn.in_proj_b.weight", "direct", (32, 2048)),
            TensorSpec(f"blk.{layer}.ssm_conv1d.weight", (4, 8192), f"{base}.linear_attn.conv1d.weight", "squeeze1_reverse", (8192, 1, 4)),
            TensorSpec(f"blk.{layer}.ssm_dt.bias", (32,), f"{base}.linear_attn.dt_bias", "direct", (32,)),
            TensorSpec(f"blk.{layer}.ssm_norm.weight", (128,), f"{base}.linear_attn.norm.weight", "direct", (128,)),
            TensorSpec(f"blk.{layer}.ssm_out.weight", (4096, 2048), f"{base}.linear_attn.out_proj.weight", "direct", (2048, 4096)),
        ])
    elif layer in FULL_LAYERS:
        specs.extend([
            TensorSpec(f"blk.{layer}.attn_q.weight", (2048, 8192), f"{base}.self_attn.q_proj.weight", "direct", (8192, 2048)),
            TensorSpec(f"blk.{layer}.attn_q_norm.weight", (256,), f"{base}.self_attn.q_norm.weight", "direct", (256,)),
            TensorSpec(f"blk.{layer}.attn_k.weight", (2048, 512), f"{base}.self_attn.k_proj.weight", "direct", (512, 2048)),
            TensorSpec(f"blk.{layer}.attn_k_norm.weight", (256,), f"{base}.self_attn.k_norm.weight", "direct", (256,)),
            TensorSpec(f"blk.{layer}.attn_v.weight", (2048, 512), f"{base}.self_attn.v_proj.weight", "direct", (512, 2048)),
            TensorSpec(f"blk.{layer}.attn_output.weight", (4096, 2048), f"{base}.self_attn.o_proj.weight", "direct", (2048, 4096)),
        ])
    else:
        raise ValueError(layer)
    return specs


def all_specs() -> list[TensorSpec]:
    specs = gguf_root_specs()
    for i in range(40):
        specs.extend(layer_specs(i))
    return specs


def reverse_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(reversed(shape))


def split_gate_up_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    experts, width, hidden = shape
    assert width % 2 == 0
    return (experts, width // 2, hidden)


def transform_shape(spec: TensorSpec, shape: tuple[int, ...]) -> tuple[int, ...]:
    if spec.mode == "direct":
        return reverse_shape(shape)
    if spec.mode in {"slice_gate_half", "slice_up_half"}:
        return reverse_shape(split_gate_up_shape(shape))
    if spec.mode == "squeeze0":
        assert len(shape) == 2 and shape[0] == 1
        return (shape[1],)
    if spec.mode == "squeeze1_reverse":
        assert len(shape) == 3 and shape[1] == 1
        return (shape[2], shape[0])
    raise ValueError(spec.mode)


def open_cache():
    cache: dict[Path, object] = {}

    def get_handle(path: Path):
        h = cache.get(path)
        if h is None:
            h = safe_open(str(path), framework="pt", device="cpu")
            cache[path] = h
        return h

    return cache, get_handle


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify Qwen3.6-35B-A3B HF source shapes for DS4Style-v0")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--show-fused", action="store_true")
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args()

    hf_dir = Path(args.hf)
    index = Path(hf_dir / "model.safetensors.index.json")
    import json
    weight_map = json.loads(index.read_text())["weight_map"]

    specs = all_specs()
    if args.layer is not None:
        prefix = f"blk.{args.layer}."
        specs = [s for s in specs if s.gguf_name.startswith(prefix)]

    cache, get_handle = open_cache()
    direct_ok = 0
    fused_ok = 0
    bad = []

    for spec in specs:
        shard_name = weight_map.get(spec.hf_name)
        if shard_name is None:
            bad.append((spec.gguf_name, "missing_hf", spec.hf_name))
            continue
        handle = get_handle(hf_dir / shard_name)
        t = handle.get_tensor(spec.hf_name)
        shape = tuple(int(x) for x in t.shape)
        if shape != spec.hf_expected_shape:
            bad.append((spec.gguf_name, "hf_shape", f"{shape} != {spec.hf_expected_shape}"))
            continue
        got = transform_shape(spec, shape)
        if got != spec.gguf_shape:
            bad.append((spec.gguf_name, "shape_transform", f"{got} != {spec.gguf_shape}"))
            continue
        if spec.mode == "direct" or spec.mode == "squeeze0" or spec.mode == "squeeze1_reverse":
            direct_ok += 1
        else:
            fused_ok += 1

    print(f"hf_dir: {hf_dir}")
    print(f"verified_targets: {len(specs)}")
    print(f"direct_ok: {direct_ok}")
    print(f"fused_ok: {fused_ok}")
    print(f"errors: {len(bad)}")

    if args.show_fused:
        print("special_transform_examples:")
        for spec in specs:
            if spec.mode == "squeeze0":
                print(f"  {spec.gguf_name} <- {spec.hf_name} hf={spec.hf_expected_shape} transform=squeeze0 gguf={spec.gguf_shape}")
                break
        for spec in specs:
            if spec.mode == "slice_gate_half":
                print(f"  {spec.gguf_name} <- {spec.hf_name} hf={spec.hf_expected_shape} transform=slice_gate_half gguf={spec.gguf_shape}")
                break
        for spec in specs:
            if spec.mode == "slice_up_half":
                print(f"  {spec.gguf_name} <- {spec.hf_name} hf={spec.hf_expected_shape} transform=slice_up_half gguf={spec.gguf_shape}")
                break
        for spec in specs:
            if spec.mode == "squeeze1_reverse":
                print(f"  {spec.gguf_name} <- {spec.hf_name} hf={spec.hf_expected_shape} transform=squeeze1_reverse gguf={spec.gguf_shape}")
                break

    if bad:
        print("first_errors:")
        for item in bad[:20]:
            print(" ", item[0], item[1], item[2])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
