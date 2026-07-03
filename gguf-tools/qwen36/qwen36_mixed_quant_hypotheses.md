# Qwen3.6-35B-A3B Mixed-Quant Hypotheses

Date: `2026-06-19`

## Purpose

This note defines the first candidate mixed-quant contracts for a narrow
`Qwen3.6-35B-A3B` runtime patterned after `ds4`.

It is intentionally aggressive.

The target is not:

- a generic GGUF runtime
- a generic quantizer
- a generic Qwen runtime

The target is:

- one exact model family
- one exact custom GGUF contract
- one validation path
- one future SSD-streaming path over the routed-expert mass

## ds4 pattern to copy

The core `ds4` idea is not "use low bits everywhere."

The core idea is:

- identify what every token always touches
- keep that path relatively protected
- crush the routed expert bulk very hard
- measure drift against a stronger oracle
- package the result as one purpose-built GGUF
- if needed, stream only the routed experts from SSD

The useful language for this is:

- `load-bearing`: damage here propagates everywhere
- `bulk-routed`: huge parameter mass, but only a few experts are active per token
- `late-sensitive`: layers near the output where extra protection may buy back quality
- `resident`: weights that stay in RAM in future SSD-streaming mode
- `streamable`: weights that can live on disk and be cached on demand

## What we know about Qwen3.6-35B-A3B already

From the current inspector and validators:

- architecture key: `qwen35moe`
- `40` blocks total
- `30` hybrid SSM blocks
- `10` full-attention blocks
- full-attention every `4`th block
- `256` experts per block
- top-`8` routed experts active per token
- shared expert path in every block

From the existing exports:

- `Q8_0` is the structural/behavior oracle
- `Q4_K_XL` is currently a high-fidelity control
- `UD_IQ2_XXS` is already an aggressive low-bit export with this observed shape:
  - root embeddings/output: `q4_k`
  - routed expert gate/up: `iq2_xxs`
  - routed expert down: mostly `iq2_s`, with `iq3_s` late-layer promotions
  - shared experts: `q5_k` / `q6_k`
  - attention tensors: `q5_k`
  - small SSM/state tensors: `f32`
  - SSM output: `q6_k`

From the oracle harness:

- `Q4_K_XL` still tracks `Q8_0` exactly on the current short probes
- `UD_IQ2_XXS` is not uniformly bad
- `UD_IQ2_XXS` already shows the key ds4-like property:
  - trivial continuations can stay locked
  - implementation-shaped continuations can drift early

That does not prove the existing Unsloth `IQ2_XXS` policy is the right one.
It does suggest that a `ds4`-style asymmetric policy is plausible for this
model family.

Additional prompt-class evidence from the role-sensitive probes:

- `router_dispatch.txt`, `n_predict = 8`
  - `Q8_0` vs `IQ2_XXS`: identical `8/8`
  - current reading:
    - simple routed-expert prose is not enough by itself to force visible drift
- `late_layers.txt`, `n_predict = 8`
  - `Q8_0` vs `Q4_K_XL`: identical `8/8`
  - `Q8_0` vs `IQ2_XXS`: only `1/8` tokens matched
  - current reading:
    - this is strong directional evidence that aggressive low-bit expert tails
      need protection or promotion
- `hybrid_bridge.txt`, `n_predict = 8`
  - `Q8_0` vs `Q4_K_XL`: `7/8` tokens matched
  - `Q8_0` vs `IQ2_XXS`: `0/8` tokens matched
  - current reading:
    - the load-bearing bridge path is highly sensitive
    - this prompt is not a perfect clean separator because even `Q4_K_XL`
      shifts slightly, but `IQ2_XXS` is much worse

## First vocabulary for Qwen quant roles

### Tier A: load-bearing resident path

These are the first candidates to keep relatively protected and permanently
resident in future SSD-streaming mode:

- `token_embd.weight`
- `output.weight`
- all router inputs:
  - `ffn_gate_inp.weight`
  - `ffn_gate_inp_shexp.weight`
- all shared expert tensors:
  - `ffn_gate_shexp.weight`
  - `ffn_up_shexp.weight`
  - `ffn_down_shexp.weight`
- all full-attention tensors:
  - `attn_q.weight`
  - `attn_k.weight`
  - `attn_v.weight`
  - `attn_output.weight`
  - `attn_q_norm.weight`
  - `attn_k_norm.weight`
- all hybrid mixer entry/exit tensors:
  - `attn_gate.weight`
  - `attn_qkv.weight`
  - `ssm_out.weight`
- all norms and small state tensors:
  - `attn_norm.weight`
  - `post_attention_norm.weight`
  - `ssm_a`
  - `ssm_dt.bias`
  - `ssm_norm.weight`
  - `ssm_conv1d.weight`

Reason:

- every token touches these
- they bridge between routed-expert activations
- damage here has the best chance to compound
- these are the natural always-resident tensors in a future SSD-streamed design

### Tier B: bulk-routed streamable path

These are the first candidates for very aggressive compression and future SSD
streaming:

- `ffn_gate_exps.weight`
- `ffn_up_exps.weight`
- `ffn_down_exps.weight`

Reason:

- they dominate MoE parameter mass
- each token only touches `8 / 256` experts per layer
- MoE redundancy is the only plausible path to ds4-style aggression

### Tier C: late-sensitive routed tail

These are not a separate tensor kind. They are a policy exception on top of Tier
B:

- routed experts in the final few blocks

Reason:

- the current `UD_IQ2_XXS` export already promotes a few late `ffn_down_exps`
  tensors from `iq2_s` to `iq3_s`
- that is weak evidence that late-layer protection may matter
- ds4 also ships mixed profiles where some later expert layers are less crushed

## Candidate contracts

## H0: control contract

Purpose:

- high-fidelity control
- likely not the end-state memory target

Policy:

- use something in the current `Q4_K_XL` class
- no SSD streaming initially

Expected value:

- behavior floor
- runtime bring-up target
- comparison oracle for more aggressive contracts

## H1: ds4-style aggressive asymmetric contract

Purpose:

- first serious target
- closest to the `ds4` philosophy

Policy:

- Tier A stays at roughly `q4_k` / `q5_k` / `q6_k` class
- Tier B goes to roughly `iq2_xxs` / `iq2_s` class
- no all-model uniform low-bit policy
- no attempt to stream Tier A
- future SSD streaming only for Tier B

Sketch:

- root:
  - embeddings/output: `q4_k` or better
- router/shared:
  - router inputs and shared experts: `q5_k`/`q6_k` class
- attention/hybrid bridge:
  - attention tensors and `attn_qkv`/`attn_gate`/`ssm_out`: `q5_k` or better
- bulk routed experts:
  - gate/up: `iq2_xxs`
  - down: `iq2_s` or `q2_k`-class equivalent if tooling favors it

Why this is the main candidate:

- it matches the ds4 asymmetry, not just the bit count
- it aligns with the current observed `UD_IQ2_XXS` export shape
- it gives a clear future memory hierarchy:
  - resident load-bearing path
  - streamable routed-expert bulk

## H2: aggressive plus late-layer protection

Purpose:

- same basic aggression as `H1`
- buy back quality near the output if needed

Policy:

- same as `H1` for most layers
- promote routed expert down tensors in the last `N` blocks
- maybe also promote routed gate/up in the final blocks if the evidence is strong

First concrete version:

- blocks `34-39`:
  - `ffn_down_exps.weight` promoted one notch above baseline low-bit

Why this is plausible:

- the downloaded `UD_IQ2_XXS` export already does something close to this
- the late routed path is cheap to protect relative to the whole model

## H3: too-aggressive uniform low-bit

Purpose:

- reject this unless evidence surprises us

Policy:

- push attention/shared/router/root down near the routed-expert class too

Why this is likely wrong:

- it discards the main ds4 insight
- the current evidence already says code-like continuations are sensitive
- if `Q4_K_XL` tracks and `IQ2_XXS` drifts, the next move is selective rescue,
  not flattening everything downward

## What "validation" means before a custom GGUF exists

Before building a real custom quant export, we can only validate the contract
hypothesis, not the final contract itself.

What the current harness can prove:

- where greedy drift begins
- whether drift is prompt-class dependent
- whether higher-bit controls stay stable
- whether late layers appear special enough to justify protection

What it cannot yet prove:

- the exact final bit allocation
- whether our future custom mix will match the currently downloaded mixes
- exact memory savings
- exact SSD-streaming behavior

## Evidence needed to keep or kill each hypothesis

`H1` survives if:

- `Q4_K_XL` remains close to `Q8_0` on code/report/tool-like prompts
- low-bit divergence looks concentrated in routed-expert-heavy behavior rather
  than everywhere equally
- protecting the load-bearing path remains consistent with current exporters

`H2` survives if:

- drift concentrates late in the continuation
- or late-block exceptions correlate with better reference-path scores
- or the downloaded `IQ2` export's late promotions look directionally right

Current status:

- `H2` gained support from the `late_layers` probe
- `H2` gained further support from `logit_commit.txt`:
  - `Q4_K_XL` stayed locked to `Q8_0`
  - `IQ2_XXS` diverged after `3` tokens
- `final_selection.txt` is mixed evidence:
  - `Q4_K_XL` also drifted there, though less severely than `IQ2_XXS`
  - so it supports "late-stage phrasing is sensitive" more than a clean
    "only ultra-low-bit tail causes this" claim

`H3` survives only if:

- unexpectedly low drift appears even when the protected path is also crushed

That is currently not the leading expectation.

## How SSD streaming fits the plan

The intended `ds4`-style memory hierarchy for Qwen should be:

- RAM-resident:
  - root tensors
  - router path
  - shared expert path
  - attention path
  - hybrid SSM bridge path
  - scratch
  - KV/cache state
- SSD-streamable:
  - routed expert tensors only

That means the future streamed unit is not:

- arbitrary tensors
- mixed pages from multiple GGUFs
- generic offload

It is:

- complete routed experts from one exact mixed-quant GGUF
- cached by expert ID
- warmed by a hot-expert list if locality is strong enough

## Immediate next step

The next useful artifact is not the final quantizer.

It is a prompt pack and scoring pass aimed at this question:

- does Qwen low-bit drift look more like:
  - routed-expert damage
  - late-layer damage
  - globally load-bearing damage

That should drive the first custom contract attempt.

## Current recommended direction

Proceed with `H1` as the main design target:

- aggressive asymmetric contract
- protect the load-bearing resident path
- crush routed experts hard
- allow late-layer expert promotions if evidence keeps pointing there
- design future SSD streaming around routed experts only

This is the closest Qwen analogue to the `ds4` philosophy that still respects
what the current Qwen evidence says.
