# Qwen3.6-35B-A3B First Mixed-Quant Policy Table

Date: `2026-06-19`

## Purpose

This is the first explicit tensor-role policy table for a future custom
`Qwen3.6-35B-A3B` mixed-quant GGUF.

It is not yet a quantizer spec.

It is the current best design target derived from:

- the inspected `Q8_0`, `Q4_K_XL`, and `UD_IQ2_XXS` exports
- the narrow GGUF validators
- the short divergence oracle probes
- the `ds4` design pattern

## Policy summary

The current recommended policy is:

- protect the always-touched path
- crush the routed expert bulk aggressively
- allow late-layer expert promotions
- design future SSD streaming around routed experts only

## Quant classes

The table uses these coarse classes:

- `high`
  - keep at `f32` where already tiny/state-like
  - otherwise target roughly `q5_k` / `q6_k` or better
- `medium`
  - target roughly `q4_k`
- `low`
  - target roughly `iq2_xxs` / `iq2_s` / `q2_k`-class
- `low+`
  - low-bit baseline with late-layer promotion one notch upward

These are policy classes, not final file-format commitments.

## First policy table

| Role family | Tensor names | Current policy class | Resident or streamable | Why |
|---|---|---|---|---|
| Root embeddings | `token_embd.weight` | `medium` | resident | touched by every token; root damage propagates globally |
| Output head | `output.weight` | `medium` | resident | directly shapes logits; late-path damage is expensive |
| Attention norms | `attn_norm.weight`, `post_attention_norm.weight` | `high` | resident | tiny and load-bearing; no point crushing |
| Router scalar inputs | `ffn_gate_inp.weight`, `ffn_gate_inp_shexp.weight` | `high` | resident | router quality is structural; every token depends on it |
| Shared expert gate/up | `ffn_gate_shexp.weight`, `ffn_up_shexp.weight` | `high` | resident | shared path runs every token; ds4-style load-bearing |
| Shared expert down | `ffn_down_shexp.weight` | `high` | resident | same reason as shared gate/up; output merge path is sensitive |
| Full attention projections | `attn_q.weight`, `attn_k.weight`, `attn_v.weight`, `attn_output.weight` | `high` | resident | classic load-bearing path; every full-attention layer depends on them |
| Full attention norms | `attn_q_norm.weight`, `attn_k_norm.weight` | `high` | resident | tiny and structurally sensitive |
| Hybrid bridge entry | `attn_gate.weight`, `attn_qkv.weight` | `high` | resident | hybrid bridge probe was the most fragile prompt class |
| Hybrid bridge exit | `ssm_out.weight` | `high` | resident | bridges mixer result back into residual path |
| Small SSM state tensors | `ssm_a`, `ssm_dt.bias`, `ssm_norm.weight`, `ssm_conv1d.weight` | `high` | resident | tiny state/control tensors; keep protected |
| SSM low-rank projections | `ssm_alpha.weight`, `ssm_beta.weight` | `high` | resident | hybrid mixer internals; likely too load-bearing to crush early |
| Routed expert gate | `ffn_gate_exps.weight` | `low` | streamable | bulk MoE mass; only selected experts are touched |
| Routed expert up | `ffn_up_exps.weight` | `low` | streamable | same as routed gate |
| Routed expert down, main body | `ffn_down_exps.weight` in most blocks | `low` | streamable | main MoE bulk; primary candidate for aggressive savings |
| Routed expert down, late tail | `ffn_down_exps.weight` in final blocks | `low+` | streamable | current evidence supports late-layer protection |

## First concrete mapping

If we had to choose a first custom contract today, before more tooling exists,
the first serious attempt should look roughly like this:

- `f32`
  - tiny state/control tensors only
- `q4_k`
  - `token_embd.weight`
  - `output.weight`
- `q5_k` / `q6_k`
  - router inputs
  - shared expert tensors
  - full-attention tensors
  - hybrid bridge tensors
  - `ssm_alpha.weight`
  - `ssm_beta.weight`
  - `ssm_out.weight`
- `iq2_xxs`
  - routed `ffn_gate_exps.weight`
  - routed `ffn_up_exps.weight`
- `iq2_s` or `q2_k`-class
  - routed `ffn_down_exps.weight` in most blocks
- one-notch promoted low-bit
  - routed `ffn_down_exps.weight` in the final few blocks

This is intentionally very close to the `ds4` philosophy:

- load-bearing resident path protected
- routed expert bulk crushed
- late routed tail optionally rescued

## What the current evidence supports most strongly

### Strong support

- protect the hybrid bridge path
- protect the shared expert path
- protect router inputs
- keep routed experts as the main low-bit target
- treat late routed expert down tensors as promotion candidates

Additional evidence:

- `late_layers.txt`
  - `Q4_K_XL` stayed locked to `Q8_0`
  - `IQ2_XXS` drifted immediately after the first token
- `logit_commit.txt`
  - `Q4_K_XL` stayed locked to `Q8_0`
  - `IQ2_XXS` drifted after `3` tokens

This is the strongest current support for late-tail protection.

### Moderate support

- embeddings/output at `q4_k` class instead of something higher
- `ssm_alpha` / `ssm_beta` staying in the protected path

### Weak or unresolved

- whether routed expert gate/up and down should use the same low-bit family
- how many late blocks deserve promotion
- whether late promotion should affect only `ffn_down_exps.weight` or also
  routed gate/up tensors

Prompt-only probing limit:

- `final_selection.txt` drifted at both `Q4_K_XL` and `IQ2_XXS`
- that means it is useful as a "late-stage phrasing sensitivity" probe
- but it is not a clean separator for deciding whether only `ffn_down_exps`
  needs protection

## First block-level exception rule

Current recommended first exception rule:

- baseline:
  - all routed expert tensors low-bit
- exception:
  - promote `ffn_down_exps.weight` in blocks `34-39`

This is not final because the current downloaded `UD_IQ2_XXS` export only
promotes a subset of those late blocks. But as a first custom policy it is a
reasonable over-protective starting point.

Current confidence by late-tail probe:

- strongest:
  - promote late `ffn_down_exps.weight`
- plausible but not yet proven:
  - promote late routed `ffn_gate_exps.weight`
  - promote late routed `ffn_up_exps.weight`

So the present recommendation is still:

- protect late routed `down` first
- do not automatically promote late routed `gate/up` until a custom export or
  tensor-ablation path gives cleaner evidence

## What would make us change this table

Raise protection if:

- more probes show early drift on prompts that isolate a role family
- `Q4_K_XL` stays stable while `IQ2_XXS` breaks hard in the same area

Lower protection if:

- a role family appears stable under more aggressive probes
- or size pressure makes it impossible to hit the fit target otherwise

## How this becomes a real contract

The next stages would be:

1. freeze this table as `v0`
2. map each tensor name family to a concrete output quant type
3. define exact late-block exception rules
4. build one custom mixed-quant GGUF
5. write an exact validator for that file
6. compare it against `Q8_0` with the same divergence harness
7. only then consider SSD streaming over routed experts

## Current recommendation

Use this table as the first custom-contract draft.

If we want to stay faithful to the `ds4` idea, the next important decision is
not runtime architecture. It is whether we are ready to turn this policy into a
real export plan.
