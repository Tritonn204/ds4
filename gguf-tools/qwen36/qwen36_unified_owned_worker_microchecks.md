# Qwen36 Unified Owned Worker Microchecks

## Goal

Keep behavior and ownership calibration while collapsing the worker farm into one process.

## Microcheck 0: Process Envelope

Binary:

- `qwen36-unified-owned-worker`

Checks:

- parses the same config shape as `qwen36-owned-session-worker`
- emits `READY`
- supports:
  - `INFO`
  - `RESET`
  - `QUIT`

Why it matters:

- proves we can preserve the oracle-facing session envelope while changing the implementation under it

## Microcheck 1: Unified Control Plane, No Child Workers

First real behavioral slice:

- same protocol as today
- no child worker spawning
- internal state structs for:
  - token ids
  - owned hidden buffer
  - hybrid layer runtime placeholders
  - full-attention layer runtime placeholders

Why it matters:

- separates protocol migration from math migration

Status:

- complete for the protocol envelope
- `qwen36-unified-owned-worker` now supports:
  - `PREFILL_PREFIX_BIN`
  - `STEP`
  - `DUMP_HIDDEN`
  - `DUMP_LAST`
  - `INFO`
  - `RESET`
  - `QUIT`
- current behavior is intentionally placeholder:
  - prefill stores the provided input sequence as owned state
  - step appends a copy of the previous last row

Suggested smoke validation:

- build:
  - `make qwen36-unified-owned-worker`
- run a tiny binary-protocol probe that:
  - sends `PREFILL_PREFIX_BIN`
  - checks `DUMP_LAST`
  - sends one `STEP`
  - checks `DUMP_HIDDEN`

Exit criteria:

- prove the Python/oracle-facing session contract can survive after the worker farm is removed
- only then move math across the boundary

## Microcheck 2: In-Process Hybrid Slice

First compute slice to unify:

- blk.0 prefix + first hybrid cycle in-process
- no pipe handoff
- compare last-row output against current worker path

Suggested validation:

- one prompt
- prefill only
- then one decode step
- compare `DUMP_LAST`

Status:

- complete for the first hybrid cycle
- `qwen36-unified-owned-worker` now loads real GGUF + live fixtures and runs:
  - blk.0
  - blk.1
  - blk.2
  in-process with no child workers
- validated against `qwen36-live-contract-worker` on:
  - `PREFILL_PREFIX_BIN` with one token
  - one `STEP`
  - `DUMP_LAST`

Observed compare:

- `rmse ~= 4.9e-9`
- `max_abs ~= 5.96e-8`
- `cos ~= 1.0`

Meaning:

- the first unified compute slice is numerically faithful
- the control-plane migration did not distort the blk.0..2 hybrid math

Extended calibration:

- the same unified worker was then compared against `qwen36-live-contract-worker`
  across the full flattened 30-fixture hybrid chain used by the owned-session path
- same one-token prefill
- same one-token step
- same `DUMP_LAST`

Observed compare:

- `rmse ~= 6.5e-8`
- `max_abs ~= 7.15e-7`
- `cos ~= 0.99999994`

Meaning:

- the unified worker reproduces the whole owned hybrid chain in one process
- we can now move to the first unified full-layer slice with the hybrid side already locked down

## Microcheck 3: In-Process Full-Layer Slice

Next compute slice:

- run one owned full-attention layer in the same process after unified hybrid output
- still same public protocol

Suggested validation:

- compare against current `qwen36-gpu-full-layer-worker` output for:
  - prefill hidden
  - decode last row

Status:

- complete for a CPU-semantic calibration slice on the first full-attention boundary
- current mechanism is opt-in:
  - `QWEN36_UNIFIED_PREFILL_FULL_CPU=1`
- validated path:
  - unified blk.0..2 hybrid prefill
  - unified full layer `blk.3` prefill
  - unified blk.0..2 hybrid step
  - unified full layer `blk.3` incremental step

Prefill validation:

- compared unified worker vs:
  - `qwen36-live-contract-worker` for blk.0..2
  - then `qwen36-c-full-layer-q8-dynamic --layer 3`
- one-token prefill result:
  - `rmse ~= 8.1e-8`
  - `max_abs ~= 1.43e-6`
  - `cos ~= 1.0`

Step validation:

- compared unified worker one-token step vs full recompute oracle:
  - hybrid owned seq recomputed through `qwen36-live-contract-worker`
  - full layer recomputed through `qwen36-c-full-layer-q8-dynamic --layer 3`
- final last-row result:
  - `rmse ~= 1.12e-8`
  - `max_abs ~= 7.45e-8`
  - `cos ~= 1.0`

Meaning:

- the unified worker now holds correct incremental state across the first full-attention layer
- the remaining open migration risk is backend/residency, not layer semantics

## Microcheck 4: First Unified Mini-Schedule

Assemble:

- one prefix/hybrid segment
- one full-attention segment

Suggested validation:

- compare against today’s owned-session path on:
  - prompt prefill
  - 1-token decode
  - 2-token decode

Status:

- complete for the first multi-cycle composed schedule
- validated two-cycle unified path:
  - hybrid `blk.0..2`
  - full layer `blk.3`
  - hybrid `blk.4..6`
  - full layer `blk.7`

Observed compare:

- `rmse ~= 2.57e-7`
- `max_abs ~= 7.03e-6`
- `cos ~= 1.0`

Meaning:

- the unified worker preserves incremental state correctly across more than one full-attention boundary
- the state-machine merge is no longer only a single-cycle proof

Extended schedule calibration:

- the same unified CPU-semantic path was then compared across the full 10-cycle owned schedule:
  - hybrid fixtures `blk.0..2, 4..6, ..., 36..38`
  - full layers `blk.3, 7, 11, 15, 19, 23, 27, 31, 35, 39`
- one-token prefill plus one-token step
- compared final `DUMP_LAST` against the composed baseline

Observed compare:

- `rmse ~= 4.06e-5`
- `max_abs ~= 8.00e-4`
- `cos ~= 0.99999988`

Meaning:

- even across the entire 40-layer owned schedule, the one-process semantic path remains tightly aligned
- this is strong enough to treat unified sequencing/state ownership as proven for the current runtime study
- remaining risk now shifts to GPU residency/backend policy rather than correctness of flattened control flow

Oracle-boundary check:

- the unified worker also now runs as a drop-in `--owned-session-worker-bin` backend for `qwen36_behavior_oracle.py`
- explicit study flag:
  - `--owned-session-unified-full-cpu`
- observed drop-in behavior on the 10-cycle schedule:
  - one prefill in one process
  - one decode step in the same process
  - no child-worker handoff
  - `owned_step_ms ~= 2231`

Meaning:

- the public owned-session envelope is now good enough to drive the unified runtime directly
- the next step can focus on backend execution policy rather than more protocol surgery

## Microcheck 5: Full 36-Owned Before 40-Owned

Before retrying the memory-cliff case:

- first target `36-owned`
- prove unified worker preserves the healthy 9-full-layer regime

Only after that:

- attempt `40-owned`
- inspect whether one-process residency avoids the 10-worker cliff
