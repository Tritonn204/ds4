# Qwen3.6-35B-A3B V0 Quant Audit

Date: `2026-06-19`

## Purpose

This note answers one narrow question:

- can the current local DS4 quant tooling emit the exact quant families needed
  by `Qwen3.6-35B-A3B-DS4Style-v0`?

## Files audited

- [gguf-tools/quants.h](/mnt/f/git/ds4/gguf-tools/quants.h)
- [gguf-tools/quants.c](/mnt/f/git/ds4/gguf-tools/quants.c)
- [gguf-tools/README.md](/mnt/f/git/ds4/gguf-tools/README.md)
- [gguf-tools/deepseek4-quantize.c](/mnt/f/git/ds4/gguf-tools/deepseek4-quantize.c)

## Current answer

Yes.

The local quant code can now emit the full `v0` type set directly.

## What `v0` needs

The frozen `v0` contract requires:

- `f32`
- `q4_k`
- `q5_k`
- `q6_k`
- `iq2_xxs`
- `iq2_s`
- `iq3_s`
- `iq3_s`

## What the current quant code can emit now

After the current backend extensions:

- `q8_0`
- `q2_k`
- `q4_k`
- `q5_k`
- `q6_k`
- `iq2_xxs`
- `iq2_s`

Evidence:

- `ds4q_type_traits[...]` now marks these with `can_quantize = true`
- `ds4q_quantize_chunk()` now dispatches concrete implementations for:
  - `DS4Q_TYPE_Q8_0`
  - `DS4Q_TYPE_Q2_K`
  - `DS4Q_TYPE_Q4_K`
  - `DS4Q_TYPE_Q5_K`
  - `DS4Q_TYPE_Q6_K`
  - `DS4Q_TYPE_IQ2_XXS`
  - `DS4Q_TYPE_IQ2_S`
  - `DS4Q_TYPE_IQ3_S`

Build check:

- `make -C gguf-tools` succeeds after the `q5_k` / `q6_k` / `iq2_s` / `iq3_s`
  ports

## Important distinction

The enum and trait table include names and row sizes for many more GGUF types,
including:

- `q5_k`
- `q6_k`
- `iq2_s`
- `iq3_s`

Those ids are now both metadata-compatible and emission-capable for the full
`v0` set.

## Consequence

`Qwen3.6-35B-A3B-DS4Style-v0` is now structurally specified, validator-ready,
and backend-exportable at the quant-family level.

The blocker is no longer quant-family availability.

The next blocker is exporter plumbing:

- source tensor loading
- template GGUF reuse
- exact name-to-type dispatch
- GGUF output writing

## What this means for next steps

There is now one honest path:

1. keep `v0` frozen
2. scaffold `gguf-tools/qwen36-v0-quantize.c`
3. produce a structurally valid `v0` GGUF
4. gate it with `qwen36-35a3b-v0-check`

## Smallest next coding task

Start the narrow Qwen exporter now.

Why the next order changes now:

- `q5_k` / `q6_k` cover the protected resident path
- `iq2_s` covers the main routed-down path
- `iq3_s` covers the late-tail exception

So the backend gap is closed and the next work is straightforward
table-driven plumbing.
