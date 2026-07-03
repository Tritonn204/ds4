# Qwen3.6-35B-A3B Q8_0 Contract

Artifact inspected:

- GGUF: `Qwen3.6-35B-A3B-Q8_0.gguf`
- Path: `/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-Q8_0.gguf`
- GGUF version: `3`
- File size: `36903140320`

## Narrow target

- Exact target: `Qwen3.6-35B-A3B`
- Export target: `Q8_0` or fail
- Text only
- Batch size `1`
- Short context first
- Greedy decode only
- CPU only
- No server
- No vision
- No MTP

## Architecture contract

- GGUF architecture key: `qwen35moe`
- Layer count: `40`
- Embedding length: `2048`
- Context length: `262144`
- Full-attention interval: `4`
- Attention heads: `16`
- KV heads: `2`
- Key length: `256`
- Value length: `256`
- Expert count: `256`
- Active experts per token: `8`
- Expert FFN size: `512`
- Shared expert FFN size: `512`

This model is not pure Transformer MoE. The tensor set shows a hybrid:

- `30` layers with fused attention/SSM motifs
- `10` layers with explicit `attn_q`, `attn_k`, `attn_v`, `attn_output`
- MoE FFN in every layer

## Quant/export contract

Observed tensor types only:

- `f32`: `301`
- `q8_0`: `432`

No `f16`, `bf16`, `q4_k`, `q5_k`, `q6_k`, `iq*` tensors are present in this export.

## Per-layer repeated motifs

Present in all `40` layers:

- `attn_norm.weight`
- `post_attention_norm.weight`
- `ffn_gate_inp.weight`
- `ffn_gate_inp_shexp.weight`
- `ffn_gate_exps.weight`
- `ffn_up_exps.weight`
- `ffn_down_exps.weight`
- `ffn_gate_shexp.weight`
- `ffn_up_shexp.weight`
- `ffn_down_shexp.weight`

Present in `30` layers:

- `attn_gate.weight`
- `attn_qkv.weight`
- `ssm_a`
- `ssm_alpha.weight`
- `ssm_beta.weight`
- `ssm_conv1d.weight`
- `ssm_dt.bias`
- `ssm_norm.weight`
- `ssm_out.weight`

Present in `10` layers:

- `attn_q.weight`
- `attn_q_norm.weight`
- `attn_k.weight`
- `attn_k_norm.weight`
- `attn_v.weight`
- `attn_output.weight`

## Attention contract

Hybrid attention schedule is strongly suggested by:

- fused attention/SSM blocks:
  - `attn_gate.weight [2048, 4096] q8_0`
  - `attn_qkv.weight [2048, 8192] q8_0`
  - `ssm_out.weight [4096, 2048] q8_0`
- explicit full-attention blocks every 4th layer:
  - `attn_q.weight [2048, 8192] q8_0`
  - `attn_k.weight [2048, 512] q8_0`
  - `attn_v.weight [2048, 512] q8_0`
  - `attn_output.weight [4096, 2048] q8_0`

Working inference:

- Most layers are hybrid SSM-style mixer layers
- Every `4`th layer is a full attention layer

## MoE contract

Every layer contains routed and shared expert tensors:

- routed gate input:
  - `ffn_gate_inp.weight [2048, 256] f32`
- routed experts:
  - `ffn_gate_exps.weight [2048, 512, 256] q8_0`
  - `ffn_up_exps.weight [2048, 512, 256] q8_0`
  - `ffn_down_exps.weight [512, 2048, 256] q8_0`
- shared expert:
  - `ffn_gate_inp_shexp.weight [2048] f32`
  - `ffn_gate_shexp.weight [2048, 512] q8_0`
  - `ffn_up_shexp.weight [2048, 512] q8_0`
  - `ffn_down_shexp.weight [512, 2048] q8_0`

Likely layer FFN schedule:

1. normalize post-attention state
2. compute router logits with `ffn_gate_inp.weight`
3. select top-`8` of `256`
4. run selected routed experts
5. run shared expert
6. merge routed + shared outputs

## SSM contract

SSM-related tensors appear in `30` layers:

- `ssm_a [32] f32`
- `ssm_alpha.weight [2048, 32] q8_0`
- `ssm_beta.weight [2048, 32] q8_0`
- `ssm_conv1d.weight [4, 8192] f32`
- `ssm_dt.bias [32] f32`
- `ssm_norm.weight [128] f32`
- `ssm_out.weight [4096, 2048] q8_0`

Metadata:

- `qwen35moe.ssm.conv_kernel = 4`
- `qwen35moe.ssm.group_count = 16`
- `qwen35moe.ssm.inner_size = 4096`
- `qwen35moe.ssm.state_size = 128`
- `qwen35moe.ssm.time_step_rank = 32`

## Explicit non-goals for first runtime

- No vision tensors found in target text model path
- No MTP tensors found in this export
- No DeltaNet tensors detected by current naming heuristics
- No attempt to support `27B`
- No attempt to support other quant formats yet

## First implementation order

1. exact GGUF validation for this export
2. exact tensor binding for the `40`-layer template
3. tokenizer/oracle harness against llama.cpp or Transformers
4. single-token forward skeleton
5. hybrid attention/SSM layer schedule
6. MoE router top-`8`
7. selected expert execution
8. logits head
9. greedy decode
