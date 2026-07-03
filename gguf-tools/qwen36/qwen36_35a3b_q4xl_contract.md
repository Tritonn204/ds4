# Qwen3.6-35B-A3B `UD_Q4_K_XL` Contract

This is the first narrow mixed-quant contract worth treating as runtime-facing rather than just descriptive.

## Fixed architecture

- `general.architecture = qwen35moe`
- `general.name = Qwen3.6-35B-A3B`
- 40 blocks
- 30 hybrid SSM layers
- 10 full-attention layers at `blk.3,7,11,...,39`
- 256 routed experts, top-8 active

## Root tensor policy

- `token_embd.weight`: `q8_0`
- `output.weight`: `q8_0`
- `output_norm.weight`: `f32`

## Load-bearing tensors kept high

- Full-attention tensors stay `q8_0`
- Hybrid attention tensors stay `q8_0`
- Shared expert tensors stay `q8_0`
- `ffn_gate_inp.weight` and `ffn_gate_inp_shexp.weight` stay `f32`
- `ssm_alpha.weight` and `ssm_beta.weight` are promoted to `f32`

## Routed expert policy

- `ffn_gate_exps.weight`
  - `q5_k` on `blk.1`
  - `q4_k` on every other block
- `ffn_up_exps.weight`
  - `q5_k` on `blk.1`
  - `q4_k` on every other block
- `ffn_down_exps.weight`
  - `q6_k` on `blk.1,34,38,39`
  - `q5_k` on every other block

This is the first concrete hint that a ds4-style Qwen runtime should treat routed expert tensors as the primary compression surface, but not uniformly.
