# Inferred `ds4` Runtime Schema

This is a code-derived baseline schema for `ds4`, intended as a comparison target for narrow runtimes for other models. It is inferred from `ds4.c` and `ds4.h` only; no DeepSeek GGUF is required locally.

## What This Is

The JSON artifact next to this file captures the runtime contract that `ds4` expects from a compatible GGUF:

- exact metadata keys and scalar/array invariants
- exact tensor naming and per-layer binding patterns
- execution-level motifs such as routed MoE, compressed attention, and hyper-connection state
- the quant families that the runtime actually implements

This is not a universal DeepSeek schema. It is the schema implied by `ds4`'s own validation, tensor binding, and forward pass code.

## Where It Comes From

The main code anchors are:

- `ds4.c:1529`
  - GGUF parsing and tensor directory loading
- `ds4.c:3888`
  - `config_validate_model()`: exact metadata contract and variant selection
- `ds4.c:4070`
  - `weights_bind()` and `weights_validate_layout()`: exact tensor binding contract
- `ds4.c:7045`
  - attention helpers and grouped output path
- `ds4.c:7307`
  - router selection and routed/shared expert MoE execution
- `ds4.c:8484`
  - KV/cache and compressor frontier layout
- `ds4.c:9915`
  - layer-major CPU prefill schedule
- `ds4.c:25546`
  - engine open path, validation, binding, inspect-only mode
- `ds4.c:26698`
  - session sync and the prefill/decode split

## How To Use This Baseline

Use the schema as a neutral checklist when inspecting another GGUF:

- `shape_contract`
  - Which scalar shape fields are hard requirements?
- `metadata_contract`
  - Which arrays or booleans carry semantic meaning?
- `layer_template`
  - What per-layer tensor groups exist, and under what conditions?
- `routing_contract`
  - Is the model dense, routed-MoE, mixed shared+routed, or something else?
- `state_cache_contract`
  - What persistent state must survive from prefill into decode?
- `quant_contract`
  - Which tensor families require specialized kernels?

For Qwen inspection, the goal is not to force Qwen into `ds4` names. The goal is to compare Qwen's evidence against the same categories:

- metadata
- tensor inventory
- layer template
- attention evidence
- routing / expert evidence
- state/cache evidence
- extra heads
- quant surface

## Important Limits

- `ds4` is deliberately narrow and model-specific.
- Some schema fields are runtime semantics, not exporter semantics.
- The code reveals enough to form a useful baseline, but not a universal architecture language.

That is acceptable. The point of this artifact is to give us a precise baseline vocabulary before inspecting `Qwen3.6-27B` and `Qwen3.6-35B-A3B`.
