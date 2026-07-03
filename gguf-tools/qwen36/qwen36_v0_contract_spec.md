# Qwen3.6-35B-A3B V0 Contract Spec

Date: `2026-06-19`

## Purpose

This document freezes the first exact custom mixed-quant target for a narrow
`Qwen3.6-35B-A3B` runtime.

It is the bridge between:

- policy discussion
- divergence probing
- future quantization/export work
- future exact GGUF validation

This is not yet an implemented file.

It is the contract we should build next.

## Contract identity

Proposed contract id:

- `Qwen3.6-35B-A3B-DS4Style-v0`

Proposed short id:

- `qwen36_35a3b_v0`

Design intent:

- aggressive asymmetric MoE quantization
- protect the load-bearing resident path
- crush routed experts hard
- promote only late routed `down` tensors
- keep future SSD streaming scoped to routed experts only

## Narrow target

- exact architecture: `qwen35moe`
- exact family: `Qwen3.6-35B-A3B`
- text only
- batch size `1`
- short context first
- greedy decode first
- CPU correctness first
- no vision
- no MTP
- no generic GGUF support

## Metadata contract

The `v0` contract should preserve the currently observed structural metadata:

- `general.architecture = qwen35moe`
- `general.name = Qwen3.6-35B-A3B`
- `qwen35moe.block_count = 40`
- `qwen35moe.embedding_length = 2048`
- `qwen35moe.context_length = 262144`
- `qwen35moe.attention.head_count = 16`
- `qwen35moe.attention.head_count_kv = 2`
- `qwen35moe.attention.key_length = 256`
- `qwen35moe.attention.value_length = 256`
- `qwen35moe.expert_count = 256`
- `qwen35moe.expert_used_count = 8`
- `qwen35moe.expert_feed_forward_length = 512`
- `qwen35moe.expert_shared_feed_forward_length = 512`
- `qwen35moe.full_attention_interval = 4`
- `qwen35moe.ssm.inner_size = 4096`
- `qwen35moe.ssm.state_size = 128`
- `qwen35moe.ssm.group_count = 16`
- `qwen35moe.ssm.conv_kernel = 4`
- `qwen35moe.ssm.time_step_rank = 32`
- `qwen35moe.rope.freq_base = 10000000.0`

The block schedule must remain:

- hybrid SSM blocks: `0,1,2,4,5,6,...,36,37,38`
- full-attention blocks: `3,7,11,15,19,23,27,31,35,39`

## Allowed tensor types

The first `v0` contract should allow only:

- `f32`
- `q4_k`
- `q5_k`
- `q6_k`
- `iq2_xxs`
- `iq2_s`
- `iq3_s`

Reason:

- this matches the current aggressive direction
- this is close to the best existing low-bit analog we already inspected
- it keeps the validator simple

## Root tensor mapping

| Tensor | Type |
|---|---|
| `token_embd.weight` | `q4_k` |
| `output_norm.weight` | `f32` |
| `output.weight` | `q4_k` |

## Per-layer always-present tensor mapping

These exist in all `40` blocks.

| Tensor suffix | Type |
|---|---|
| `attn_norm.weight` | `f32` |
| `post_attention_norm.weight` | `f32` |
| `ffn_gate_inp.weight` | `f32` |
| `ffn_gate_inp_shexp.weight` | `f32` |
| `ffn_gate_exps.weight` | `iq2_xxs` |
| `ffn_up_exps.weight` | `iq2_xxs` |
| `ffn_down_exps.weight` | see late-tail rule |
| `ffn_gate_shexp.weight` | `q5_k` |
| `ffn_up_shexp.weight` | `q5_k` |
| `ffn_down_shexp.weight` | `q6_k` |

## Hybrid SSM block mapping

These exist in the `30` hybrid blocks.

| Tensor suffix | Type |
|---|---|
| `attn_gate.weight` | `q5_k` |
| `attn_qkv.weight` | `q5_k` |
| `ssm_a` | `f32` |
| `ssm_alpha.weight` | `f32` |
| `ssm_beta.weight` | `f32` |
| `ssm_conv1d.weight` | `f32` |
| `ssm_dt.bias` | `f32` |
| `ssm_norm.weight` | `f32` |
| `ssm_out.weight` | `q6_k` |

## Full-attention block mapping

These exist in the `10` full-attention blocks.

| Tensor suffix | Type |
|---|---|
| `attn_q.weight` | `q5_k` |
| `attn_q_norm.weight` | `f32` |
| `attn_k.weight` | `q5_k` |
| `attn_k_norm.weight` | `f32` |
| `attn_v.weight` | `q5_k` |
| `attn_output.weight` | `q5_k` |

## Late-tail routed expert rule

Baseline rule:

- every block:
  - `ffn_gate_exps.weight = iq2_xxs`
  - `ffn_up_exps.weight = iq2_xxs`

Main-body routed down rule:

- blocks `0-33`:
  - `ffn_down_exps.weight = iq2_s`

Late-tail routed down rule:

- blocks `34-39`:
  - `ffn_down_exps.weight = iq3_s`

This is intentionally more explicit than the currently downloaded aggressive
export, which only promotes a subset of late `down` tensors.

The reason to over-protect here is simple:

- prompt-only probes strongly support late-tail sensitivity
- they do not yet cleanly justify promoting late routed `gate/up`
- so `v0` should spend its extra precision budget on late routed `down`

## Resident vs streamable contract

### Resident

These are the intended always-resident tensors in a future SSD-streaming design:

- root tensors
- router inputs
- shared expert tensors
- all attention tensors
- all hybrid bridge tensors
- all norms
- all small SSM/state/control tensors

### Streamable

These are the intended streamable tensors:

- `ffn_gate_exps.weight`
- `ffn_up_exps.weight`
- `ffn_down_exps.weight`

Streaming unit:

- complete expert payload by block and expert id

Not a goal:

- arbitrary tensor paging
- streaming non-routed tensors
- mixing data from multiple GGUFs at runtime

## Validation expectations

The future exact validator for `v0` should prove:

- metadata matches this exact architecture contract
- every required tensor exists
- every tensor type matches exactly
- every tensor dimension matches exactly
- every block is exactly one of:
  - hybrid SSM
  - full attention
- the full-attention cadence is exact
- late routed `down` promotion is exact in blocks `34-39`
- no unsupported tensor types appear

It should fail if:

- another export uses different low-bit families
- late promotion lands on different blocks
- gate/up are promoted unexpectedly
- hybrid/full block layout changes

## Naming for future code artifacts

Recommended future code artifact names:

- `qwen36_35a3b_v0.h`
- `qwen36_35a3b_v0.c`
- `qwen36_35a3b_v0_check.c`

Optional dump/helper:

- `qwen36_35a3b_v0_dump.c`

## What this spec deliberately does not decide

- how the custom quantizer computes these tensors
- whether quantization starts from BF16/F16/safetensors or another source
- whether `iq2_s` or `q2_k` would be marginally better for routed `down`
- whether later `v1` should promote late routed `gate/up`
- backend-specific kernel implementation

## Current recommendation

Treat this as the frozen first export target.

If we keep moving in the `ds4` direction, the next practical step is not more
probing. It is either:

1. write the exact `v0` validator and binder stub first, or
2. write the custom export/rewriter plan that would produce this file

My recommendation is:

- write the validator/binder stub first

Reason:

- it forces the contract to stay precise
- it gives the future export path a hard target
- it keeps the project narrow and honest
