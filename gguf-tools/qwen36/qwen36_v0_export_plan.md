# Qwen3.6-35B-A3B V0 Export Plan

Date: `2026-06-19`

## Purpose

This document defines the smallest honest path from the frozen
`DS4Style_v0` contract to a real GGUF file.

It answers:

- what kind of tool we should build
- what the input should be
- what the output should be
- which parts can be copied from the DS4 quant tooling pattern
- which parts need to be Qwen-specific

## Decision

The next export tool should be a narrow one-off quantizer/rewriter:

- input:
  - Qwen3.6-35B-A3B high-quality source weights
  - one compatible template GGUF
  - optional importance data later
- output:
  - one exact `Qwen3.6-35B-A3B-DS4Style-v0.gguf`

It should not be:

- a generic GGUF tool
- a runtime mixer of multiple GGUFs
- a long-lived general Qwen quantizer

## Why this is the right shape

The DS4 tooling already establishes the useful pattern:

- template GGUF supplies metadata, tokenizer, tensor order, and shapes
- source weights regenerate tensor payloads
- quant policy is tensor-family specific
- quality is checked with target-token NLL, not vibes

Relevant references in this repo:

- [gguf-tools/README.md](/mnt/f/git/ds4/gguf-tools/README.md)
- [gguf-tools/deepseek4-quantize.c](/mnt/f/git/ds4/gguf-tools/deepseek4-quantize.c)
- [gguf-tools/mixed/README.md](/mnt/f/git/ds4/gguf-tools/mixed/README.md)
- [gguf-tools/quality-testing/README.md](/mnt/f/git/ds4/gguf-tools/quality-testing/README.md)

## Recommended tool

Proposed tool name:

- `gguf-tools/qwen36-v0-quantize.c`

Possible helper later:

- `gguf-tools/qwen36-v0-splice.py`

But the main path should be the quantizer, not the splicer.

## Why not start with splicing

The DeepSeek mixed-layer splicer is useful for experiments, but it only copies
already-quantized tensors between compatible GGUFs.

For Qwen `v0`, that is not enough:

- we need a new late-tail rule on blocks `34-39`
- we may need a tensor-type mix not available in any single current export
- we eventually want our own exact file, not an inferred blend

So splicing is optional for experiments, but not the real destination.

## Inputs

## 1. Source weights

Preferred source:

- original or near-original Qwen3.6-35B-A3B safetensors in BF16/F16 class

Reason:

- building `v0` from an already-quantized GGUF would stack quantization error
- the DS4 pattern starts from stronger source weights, then quantizes once

Fallback:

- if only GGUF sources are available initially, use them for structural study
  only, not the final export

## 2. Template GGUF

We need one compatible template GGUF to provide:

- metadata
- tokenizer
- tensor ordering
- logical tensor names and shapes

Best initial candidate:

- the current `Q8_0` GGUF

Reason:

- it is structurally simple
- it already matches the exact target architecture
- it avoids inheriting someone else’s aggressive low-bit choices into the
  template itself

## 3. Importance data

Initial `v0` cannot honestly skip importance data for every target family.

Reason:

- the frozen `v0` policy assigns `iq2_xxs` to:
  - `ffn_gate_exps.weight`
  - `ffn_up_exps.weight`
- that is `80` tensors total across `40` layers
- the local quant backend marks `iq2_xxs` as `requires_imatrix`

Practical implication:

- a real first exporter must either:
  - provide a Qwen-specific imatrix/calibration path, or
  - add a narrow synthetic fallback and label the result experimental

Current narrow study result:

- [qwen36_v0_tensor_probe.py](/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_v0_tensor_probe.py)
  now materializes exact HF tensors into GGUF logical layout and reports
  target quant requirements
- current explicit blocker list:
  - `80` routed expert gate/up tensors need `iq2_xxs` importance data
- this makes the remaining exporter gap concrete instead of inferred

Current experimental implementation:

- [qwen36_v0_export_experimental.py](/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_v0_export_experimental.py)
  now drives:
  - HF safetensors source loading
  - template KV reuse from GGUF
  - local C quantization through [gguf-tools/libds4q.so](/mnt/f/git/ds4/gguf-tools/libds4q.so)
  - synthetic `iq2_xxs` fallback using per-column weight energy
- live smoke coverage now includes:
  - `f32`
  - `q4_k`
  - `q5_k`
  - `q6_k`
  - `iq2_xxs`

What this unlocks immediately:

- we can now do an honest full-file experimental export without inventing a
  second quant backend
- the next question is no longer "can we bridge Qwen HF weights into the DS4
  quant stack?"
- the next question is "how bad is the synthetic-importance full export versus
  Q8_0 under oracle divergence?"

## Concrete `v0` mapping to implement

From the frozen contract:

- root:
  - `token_embd.weight -> q4_k`
  - `output.weight -> q4_k`
  - `output_norm.weight -> f32`
- all blocks:
  - `attn_norm.weight -> f32`
  - `post_attention_norm.weight -> f32`
  - `ffn_gate_inp.weight -> f32`
  - `ffn_gate_inp_shexp.weight -> f32`
  - `ffn_gate_exps.weight -> iq2_xxs`
  - `ffn_up_exps.weight -> iq2_xxs`
  - `ffn_gate_shexp.weight -> q5_k`
  - `ffn_up_shexp.weight -> q5_k`
  - `ffn_down_shexp.weight -> q6_k`
- hybrid blocks:
  - `attn_gate.weight -> q5_k`
  - `attn_qkv.weight -> q5_k`
  - `ssm_a -> f32`
  - `ssm_alpha.weight -> f32`
  - `ssm_beta.weight -> f32`
  - `ssm_conv1d.weight -> f32`
  - `ssm_dt.bias -> f32`
  - `ssm_norm.weight -> f32`
  - `ssm_out.weight -> q6_k`
- full-attention blocks:
  - `attn_q.weight -> q5_k`
  - `attn_q_norm.weight -> f32`
  - `attn_k.weight -> q5_k`
  - `attn_k_norm.weight -> f32`
  - `attn_v.weight -> q5_k`
  - `attn_output.weight -> q5_k`
- routed down exceptions:
  - blocks `0-33`: `ffn_down_exps.weight -> iq2_s`
  - blocks `34-39`: `ffn_down_exps.weight -> iq3_s`

## Smallest implementation path

## Phase 1: write-only contract target

Goal:

- create the narrow quantizer skeleton with no claims of quality yet

Outputs:

- `gguf-tools/qwen36-v0-quantize.c`
- optional local `README` note for its exact usage

Capabilities:

- open source weights
- open template GGUF
- copy metadata/tokenizer/tensor order from template
- map tensor names to `v0` target types
- write a new GGUF file

Not required yet:

- fast performance
- multi-model support
- calibration data ingestion

## Phase 2: minimal quant families

Implement only the families `v0` needs:

- pass-through `f32`
- `q4_k`
- `q5_k`
- `q6_k`
- `iq2_xxs`
- `iq2_s`
- `iq3_s`

Important constraint:

- if the local `quants.[ch]` code does not already implement one of these
  output families, we either:
  - add only that family, narrowly, or
  - adjust the first export plan to a nearby supported family

Right now the DS4 local quant code covers:

- `q8_0`
- `q4_k`
- `q2_k`
- `q5_k`
- `q6_k`
- `iq2_xxs`
- `iq2_s`
- `iq3_s`

Audit result:

- `q5_k` and `q6_k` are now emitted locally
- `iq2_s` is now emitted locally
- `iq3_s` is now emitted locally
- see [qwen36_v0_quant_audit.md](/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_v0_quant_audit.md)

So `v0` is now directly exportable at the quant-family level.

The remaining work is not backend format support.
The remaining work is the narrow exporter itself.

## Phase 3: narrow name-family dispatcher

The quantizer should have an exact Qwen-specific name dispatcher:

- exact root tensor names
- exact block suffix families
- exact late-tail override on blocks `34-39`

This should be table-driven, not inferred from generic heuristics.

Example shape:

- `tensor_target_type(name, layer_index, block_kind) -> gguf_type`

## Phase 4: validator loop

After the first output file is written:

1. run `qwen36-35a3b-v0-check`
2. fix structural mismatches until it passes
3. only then run oracle divergence checks

## Phase 5: quality loop

Use the current oracle harness first:

- compare `v0` against `Q8_0`
- same prompt pack
- same reference-path scoring

If the file is promising, only then add a stronger comparison loop analogous to
the DS4 official-continuation scorer.

## Smallest useful CLI

First tool CLI should be narrow:

```sh
gguf-tools/qwen36-v0-quantize \
  --hf /path/to/Qwen3.6-35B-A3B \
  --template /path/to/Qwen3.6-35B-A3B-Q8_0.gguf \
  --out /path/to/Qwen3.6-35B-A3B-DS4Style-v0.gguf
```

Useful early options:

- `--dry-run`
- `--compare-tensor NAME`
- `--overwrite`
- `--threads N`

Nice-to-have later:

- `--imatrix FILE`

## Fastest honest first milestone

The first milestone is not "good quality."

It is:

- a structurally valid `v0` file
- that passes `qwen36-35a3b-v0-check`

That proves:

- source reading works
- template reuse works
- name-family mapping works
- type-selection logic works
- GGUF writing works

## Current scaffold result

The first exporter scaffold now exists:

- [gguf-tools/qwen36-v0-quantize.c](/mnt/f/git/ds4/gguf-tools/qwen36-v0-quantize.c)

Current scope:

- requires the exact local `Q8_0` template GGUF
- computes the exact `DS4Style-v0` tensor rewrite plan
- reports resident vs streamable planned bytes
- does not yet write a new GGUF from source weights

First dry-run result on the local `Q8_0` template:

- template tensor bytes:
  - `36,892,150,272`
- planned `v0` tensor bytes:
  - `10,833,676,800`
- planned resident bytes:
  - `1,681,705,472`
- planned streamable bytes:
  - `9,151,971,328`
- planned streamable fraction:
  - `0.844771`

Interpretation:

- the contract now has a concrete "resident core + routed bulk" split
- most of the planned footprint sits exactly where a later SSD-streaming
  design would want it:
  - routed expert tensors

## Main technical unknowns

## 1. Quant-family availability

This is now resolved for `v0`.

The local DS4 quant code now has reusable output implementations for:

- `q5_k`
- `q6_k`
- `iq2_s`
- `iq3_s`

## 2. Qwen source tensor layout

This is now substantially resolved.

The local source-contract probe:

- [qwen36_hf_contract_probe.py](/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_hf_contract_probe.py)

shows on the downloaded `35B-A3B` source tree:

- `733` probe targets
- `653` direct source mappings
- `80` fused source mappings
- `0` missing mappings

The only special source case identified so far is:

- routed `ffn_gate_exps.weight` and `ffn_up_exps.weight`
  - both come from fused HF `mlp.experts.gate_up_proj`
  - the writer must slice that fused source tensor into gate and up halves

That source contract is now also shape-verified against the downloaded local
`35B-A3B` safetensors:

- [qwen36_v0_source_verify.py](/mnt/f/git/ds4/gguf-tools/qwen36/qwen36_v0_source_verify.py)

Current result:

- `733` verified targets
- `653` direct transforms verified
- `80` fused routed gate/up transforms verified
- `0` errors

Verified nontrivial source transforms:

- `mlp.shared_expert_gate.weight`
  - squeeze leading singleton `(1, 2048) -> (2048,)`
- `mlp.experts.gate_up_proj`
  - split fused width `1024 -> 512 + 512`
- `linear_attn.conv1d.weight`
  - squeeze singleton middle channel and reverse `(8192, 1, 4) -> (4, 8192)`

## 3. Template compatibility

We assume the current `Q8_0` GGUF is a good template source.
That still needs a quick sanity check once export code begins.

## Recommended next code step

Build the full narrow quantizer next.

Recommended order:

1. keep `v0` frozen
2. scaffold `qwen36-v0-quantize.c`
3. reuse the current local quant backend as-is
4. use `qwen36-35a3b-v0-check` as the first structural gate

That is the smallest next step that keeps the project narrow and technically
honest.
