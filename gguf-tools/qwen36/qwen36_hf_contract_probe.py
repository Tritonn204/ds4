#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path


HYBRID_LAYERS = {
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
}
FULL_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39}


@dataclass(frozen=True)
class Mapping:
    gguf: str
    hf: str
    mode: str


def root_mappings() -> list[Mapping]:
    return [
        Mapping("token_embd.weight", "model.language_model.embed_tokens.weight", "direct"),
        Mapping("output_norm.weight", "model.language_model.norm.weight", "direct"),
        Mapping("output.weight", "lm_head.weight", "direct"),
    ]


def layer_mappings(layer: int) -> list[Mapping]:
    base = f"model.language_model.layers.{layer}"
    out = [
        Mapping(f"blk.{layer}.attn_norm.weight", f"{base}.input_layernorm.weight", "direct"),
        Mapping(f"blk.{layer}.post_attention_norm.weight", f"{base}.post_attention_layernorm.weight", "direct"),
        Mapping(f"blk.{layer}.ffn_gate_inp.weight", f"{base}.mlp.gate.weight", "direct"),
        Mapping(f"blk.{layer}.ffn_gate_inp_shexp.weight", f"{base}.mlp.shared_expert_gate.weight", "direct"),
        Mapping(f"blk.{layer}.ffn_gate_exps.weight", f"{base}.mlp.experts.gate_up_proj", "slice_gate_half"),
        Mapping(f"blk.{layer}.ffn_up_exps.weight", f"{base}.mlp.experts.gate_up_proj", "slice_up_half"),
        Mapping(f"blk.{layer}.ffn_down_exps.weight", f"{base}.mlp.experts.down_proj", "direct"),
        Mapping(f"blk.{layer}.ffn_gate_shexp.weight", f"{base}.mlp.shared_expert.gate_proj.weight", "direct"),
        Mapping(f"blk.{layer}.ffn_up_shexp.weight", f"{base}.mlp.shared_expert.up_proj.weight", "direct"),
        Mapping(f"blk.{layer}.ffn_down_shexp.weight", f"{base}.mlp.shared_expert.down_proj.weight", "direct"),
    ]
    if layer in HYBRID_LAYERS:
        out.extend([
            Mapping(f"blk.{layer}.attn_gate.weight", f"{base}.linear_attn.in_proj_z.weight", "direct"),
            Mapping(f"blk.{layer}.attn_qkv.weight", f"{base}.linear_attn.in_proj_qkv.weight", "direct"),
            Mapping(f"blk.{layer}.ssm_a", f"{base}.linear_attn.A_log", "direct"),
            Mapping(f"blk.{layer}.ssm_alpha.weight", f"{base}.linear_attn.in_proj_a.weight", "direct"),
            Mapping(f"blk.{layer}.ssm_beta.weight", f"{base}.linear_attn.in_proj_b.weight", "direct"),
            Mapping(f"blk.{layer}.ssm_conv1d.weight", f"{base}.linear_attn.conv1d.weight", "direct"),
            Mapping(f"blk.{layer}.ssm_dt.bias", f"{base}.linear_attn.dt_bias", "direct"),
            Mapping(f"blk.{layer}.ssm_norm.weight", f"{base}.linear_attn.norm.weight", "direct"),
            Mapping(f"blk.{layer}.ssm_out.weight", f"{base}.linear_attn.out_proj.weight", "direct"),
        ])
    elif layer in FULL_LAYERS:
        out.extend([
            Mapping(f"blk.{layer}.attn_q.weight", f"{base}.self_attn.q_proj.weight", "direct"),
            Mapping(f"blk.{layer}.attn_q_norm.weight", f"{base}.self_attn.q_norm.weight", "direct"),
            Mapping(f"blk.{layer}.attn_k.weight", f"{base}.self_attn.k_proj.weight", "direct"),
            Mapping(f"blk.{layer}.attn_k_norm.weight", f"{base}.self_attn.k_norm.weight", "direct"),
            Mapping(f"blk.{layer}.attn_v.weight", f"{base}.self_attn.v_proj.weight", "direct"),
            Mapping(f"blk.{layer}.attn_output.weight", f"{base}.self_attn.o_proj.weight", "direct"),
        ])
    else:
        raise ValueError(f"unexpected layer {layer}")
    return out


def all_mappings() -> list[Mapping]:
    out = root_mappings()
    for layer in range(40):
        out.extend(layer_mappings(layer))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe exact HF source contract for Qwen3.6-35B-A3B DS4Style-v0")
    ap.add_argument("--hf", required=True, help="HF model directory with model.safetensors.index.json")
    ap.add_argument("--show-missing", action="store_true")
    ap.add_argument("--show-slices", action="store_true")
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args()

    hf_dir = Path(args.hf)
    index_path = hf_dir / "model.safetensors.index.json"
    data = json.loads(index_path.read_text())
    weight_map: dict[str, str] = data["weight_map"]

    mappings = all_mappings()
    if args.layer is not None:
        prefix = f"blk.{args.layer}."
        mappings = [m for m in mappings if m.gguf.startswith(prefix)]

    missing: list[Mapping] = []
    fused: list[Mapping] = []
    direct = 0

    print(f"hf_dir: {hf_dir}")
    print(f"weight_map_entries: {len(weight_map)}")
    print(f"probe_targets: {len(mappings)}")

    for m in mappings:
        shard = weight_map.get(m.hf)
        if shard is None:
            missing.append(m)
            continue
        if m.mode == "direct":
            direct += 1
        else:
            fused.append(m)

    print(f"direct_mappings_present: {direct}")
    print(f"fused_mappings_present: {len(fused)}")
    print(f"missing_mappings: {len(missing)}")

    if args.show_slices:
        print("fused_cases:")
        for m in fused:
            print(f"  {m.gguf} <- {m.hf} ({m.mode}) [{weight_map[m.hf]}]")

    if args.show_missing and missing:
        print("missing_cases:")
        for m in missing:
            print(f"  {m.gguf} <- {m.hf} ({m.mode})")

    print("sample_root:")
    for m in root_mappings():
        shard = weight_map.get(m.hf, "<missing>")
        print(f"  {m.gguf} <- {m.hf} [{shard}]")

    print("sample_fused:")
    for m in [
        Mapping("blk.0.ffn_gate_exps.weight", "model.language_model.layers.0.mlp.experts.gate_up_proj", "slice_gate_half"),
        Mapping("blk.0.ffn_up_exps.weight", "model.language_model.layers.0.mlp.experts.gate_up_proj", "slice_up_half"),
    ]:
        shard = weight_map.get(m.hf, "<missing>")
        print(f"  {m.gguf} <- {m.hf} ({m.mode}) [{shard}]")

    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
