# Qwen36 Unified Owned Worker Plan

## Objective

Replace the current multi-process owned-session layout:

- one prefix worker
- N hybrid workers
- N full-attention workers
- Python-side orchestration and handoff

with one unified owned runtime that:

- performs one prompt prefill
- keeps all decode state in one process
- advances decode without inter-worker pipe handoffs
- owns one GPU residency policy
- can absorb SSD/model-range streaming cleanly

## Why This Direction

The study now shows a hard scalability limit in the current design:

- `36-owned` with 9 persistent full-attention workers is healthy
- `40-owned` with 10 persistent full-attention workers hits a VRAM/residency cliff
- the cliff is dominated by the full-layer Q/K/V projection batch
- the problem is not hybrid CPU math or Python dump overhead

This means the worker farm architecture has reached its useful limit on RX 7900 XTX. The next step is not "slightly better orchestration"; it is eliminating the per-layer worker residency model.

## Key Observation

DS4 already has the right architectural shape:

- `ds4_engine`
- `ds4_session`
- one mutable inference timeline
- one prefill path
- one decode path
- one global graph/runtime allocation policy
- optional SSD streaming and expert-cache controls

Relevant surfaces:

- [ds4.h](/mnt/f/git/ds4/ds4.h:13)
- [ds4_gpu.h](/mnt/f/git/ds4/ds4_gpu.h:18)
- [ds4.c](/mnt/f/git/ds4/ds4.c:26049)
- [ds4.c](/mnt/f/git/ds4/ds4.c:26844)
- [ds4.c](/mnt/f/git/ds4/ds4.c:27130)

So the clean design is to converge the Qwen-owned runtime toward the `ds4_session` model, not to keep scaling the custom worker protocol.

## Current Split Responsibilities

### Prefix / hybrid workers

Currently own:

- token-embedding / blk.0 prefix logic
- hybrid DeltaNet recurrent state
- conv-ring state
- routed-expert CPU cache

### Full-attention workers

Currently own:

- per-layer KV history
- per-layer output sequence
- per-process ROCm context
- per-process model-range cache

### Owned-session coordinator

Currently owns:

- high-level cycle order
- inter-process row passing
- prompt token list
- concatenated owned hidden sequence

## Unified Worker Target Shape

One `qwen36-unified-owned-worker` should own:

- prompt tokens
- final owned hidden state / last row
- hybrid recurrent state for all owned hybrid layers
- KV state for all owned full-attention layers
- one ROCm context
- one global model-range cache
- one global expert streaming/cache policy
- one decode loop that walks layers in order

In other words:

- no per-cycle subprocesses
- no row serialization between owned stages
- no duplicate GPU context startup
- no one-process-per-full-layer residency cliff

## One Prefill Requirement

The unified worker should make prefill a first-class boundary:

1. tokenize / receive token ids
2. prefill all owned layers exactly once
3. persist all hybrid/full-attention state
4. on decode, consume only the next token and advance in place

This matches the existing DS4 session contract much more closely than the current worker farm.

## Recommended Runtime Strategy

### Hybrid layers

Keep the proven narrow math, but make it in-process:

- reuse `qwen36_live_contract_worker.c` logic as library-style code
- remove stdin/stdout worker protocol from the hot path
- preserve the same state structs:
  - DeltaNet state
  - conv-ring state
  - expert cache

### Full-attention layers

Do not keep one persistent ROCm process per layer.

Instead:

- keep one ROCm context
- keep one shared model map / range cache
- run a rolling full-attention executor across owned full layers
- maintain per-layer KV/output state in one process

The key idea is:

- state is per layer
- execution machinery should not be per process

### Tail handling

Support two modes:

- `splice-layer 39` style final norm + `lm_head` in-process
- optional handoff to a larger DS4/HF tail if needed during transition

## SSD Streaming Fit

SSD streaming belongs naturally in the unified runtime, because it needs:

- one residency policy
- one global budget
- one routing-aware hot cache
- one layer scheduler

Relevant existing DS4 hooks:

- `ds4_gpu_set_ssd_streaming(...)`
- `ds4_gpu_stream_expert_cache_*`
- session/graph setup paths in [ds4.c](/mnt/f/git/ds4/ds4.c:25552) and [ds4.c](/mnt/f/git/ds4/ds4.c:25725)
- layer-major/decode streaming paths in [ds4.c](/mnt/f/git/ds4/ds4.c:26459) and [ds4.c](/mnt/f/git/ds4/ds4.c:26592)

This is much harder to do correctly across many independent workers, and much easier in one runtime.

## Recommended Implementation Phases

### Phase 1: Unify orchestration, keep existing math

Build one owned worker process that:

- embeds the current hybrid/full layer code directly
- removes worker-to-worker pipes
- still uses the current Qwen-local kernels and fixtures

Goal:

- preserve current behavior
- eliminate inter-process handoff and duplicate ROCm contexts

Current status:

- protocol scaffold is live
- Microcheck 0 passed:
  - config parse
  - `READY`
  - `INFO`
  - `RESET`
  - `QUIT`
- Microcheck 1 control plane now passes:
  - one process
  - no child workers
  - `PREFILL_PREFIX_BIN`, `STEP`, `DUMP_HIDDEN`, and `DUMP_LAST` round-trip correctly
- math is still placeholder at this point, by design
- next useful milestone is the first in-process hybrid compute slice, not more protocol work

Latest checkpoint:

- first in-process hybrid slice is now live in the unified worker
- current unified mode is explicitly `hybrid_only`
- it flattens the configured live fixtures and executes them in one process against one GGUF binding
- the first validation target was blk.0..2

Validation result:

- compared unified worker vs `qwen36-live-contract-worker`
- same one-token prefill
- same one-token step
- same `DUMP_LAST`
- numerical agreement was effectively exact:
  - `rmse ~= 4.9e-9`
  - `max_abs ~= 5.96e-8`
  - `cos ~= 1.0`

So Phase 1 has crossed from protocol scaffolding into real compute migration.

Extended result:

- repeated the compare across the full flattened 30-fixture owned hybrid chain
- unified hybrid math remained tightly aligned end-to-end

Latest semantic checkpoint:

- the unified worker now also carries full-layer state across:
  - one full-attention boundary
  - then multiple boundaries
  - then the entire 10-cycle owned schedule in CPU-semantic mode

Observed schedule-level compare:

- two-cycle checkpoint:
  - `rmse ~= 2.57e-7`
  - `max_abs ~= 7.03e-6`
  - `cos ~= 1.0`
- full 10-cycle checkpoint:
  - `rmse ~= 4.06e-5`
  - `max_abs ~= 8.00e-4`
  - `cos ~= 0.99999988`

Interpretation:

- unified state ownership is now strong enough that continuing to spend time on pipe-level orchestration would be low value
- the next implementation value is to replace the worker farm backend with a single execution/residency policy
- that is exactly the shape needed before SSD streaming can be integrated coherently

## Pivot Status

The first real GPU pivot is now in tree.

Added:

- a separate ROCm-capable unified target:
  - `qwen36-unified-owned-worker-rocm`
- explicit oracle flag:
  - `--owned-session-unified-full-gpu`
- unified worker mode:
  - `QWEN36_UNIFIED_FULL_GPU=1`

Meaning:

- the unified worker can now execute owned full-attention layers in-process under one ROCm context instead of one worker per full layer

Latest external calibration:

- `36-owned` unified ROCm run succeeded end-to-end through owned layer `35`
- reported GPU footprint stayed around `8 GiB`
- owned prefill dropped to about `26.6s`
- owned decode steps were about `1.08s` then `0.84s`

Why this matters:

- the old worker-farm memory cliff was encountered when widening from `36-owned` to `40-owned`
- under the unified ROCm path, the first external sign is that the cliff mechanism is no longer present at `36-owned`
- this strongly suggests the duplicated per-process ROCm residency model was the dominant problem, not the abstract layer count itself

Next calibration target:

- rerun the former failure shape as a unified run:
  - add owned full layer `39`
  - use `--splice-layer 39`
  - keep unified GPU mode enabled
- if that succeeds within the same rough footprint class, the study can treat the old 10th-full-layer cliff as architecturally resolved

- we are no longer only proving semantics with CPU full-layer emulation
- the unified runtime can now be built with embedded in-process ROCm full-layer code
- the old full-layer subprocess path is no longer the only way to exercise GPU full layers

Current limit:

- this environment can compile the ROCm-capable target but is not the right place to make performance or residency claims
- the next external calibration should be a 36-owned run with the unified ROCm worker before widening further
- still matched to numerical noise:
  - `rmse ~= 6.5e-8`
  - `max_abs ~= 7.15e-7`
  - `cos ~= 0.99999994`

This matters because it means the unified worker is already faithful across the complete hybrid side of the current owned-session design. The next migration risk is concentrated in the full-attention/GPU slice rather than in hybrid unification.

Next checkpoint reached:

- first full-attention boundary is now unified as a CPU-semantic calibration slice
- this is currently enabled with:
  - `QWEN36_UNIFIED_PREFILL_FULL_CPU=1`
- validated scope:
  - blk.0..2 hybrid prefill
  - blk.3 full-layer prefill
  - blk.0..2 hybrid step
  - blk.3 full-layer incremental step

Validation quality:

- full prefill compare landed at effectively exact agreement
- full incremental step compare also landed at effectively exact agreement

Interpretation:

- unified state carrying is now proven across the first full-attention boundary
- the next big problem is no longer “can one process preserve semantics?”
- it is “which backend owns the full-attention layers without reintroducing the residency cliff?”

### Phase 2: Share GPU residency globally

Move full-attention execution onto:

- one ROCm context
- one shared range cache
- one per-session KV state table

Goal:

- prove that 40-owned no longer falls off the 10-worker cliff

### Phase 3: Integrate SSD/model-range streaming

Wire the unified Qwen owned runtime into DS4-style streaming controls:

- shared model-range cache budget
- expert streaming cache budget
- preload / hot-route seeding
- cold-start vs warm-start policy

Goal:

- make full ownership practical even when full static residency is impossible

## Minimal First Build

The smallest useful next prototype is:

- one new worker binary
- one process
- one ROCm init
- one prefill
- hybrid + full-attention execution in-process
- same prompt/decode protocol to Python as today:
  - `PREFILL_PREFIX_BIN`
  - `STEP`
  - `DUMP_LAST`
  - optional `DUMP_HIDDEN`

That lets the harness stay almost unchanged while the hot path becomes unified.

## Non-Goals

Do not optimize these first:

- Python oracle UX
- HF tail convenience
- worker farm logging
- squeezing more life out of the 10-process full-worker design

Those are now secondary to fixing the residency architecture.

## Decision

Proceed by converging toward a unified `ds4_session`-like Qwen owned runtime:

- one process
- one prefill
- one decode timeline
- one GPU residency policy
- optional SSD streaming integrated at runtime level

That is the correct successor to the current study results.
