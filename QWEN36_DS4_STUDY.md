# Qwen3.6 DS4-Style Study

## Current Artifacts

- HF source: `/mnt/e/tensors/Qwen3.6-35B-A3B`
- Experimental export artifact: `/mnt/e/tensors/Qwen3.6-35B-A3B-DS4Style-v0-experimental.gguf`
- Contract checker: `qwen36-35a3b-v0-check`
- Current runtime/oracle Q8 GGUF:
  - `/home/tritonn/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-Q8_0.gguf`

## June 30, 2026 Prefill Checkpoint

The prefill investigation is now split into two distinct fault surfaces and they should not be mixed:

- standalone full-attention GPU worker math
- integrated owned-session prefill drift across owned depth

What changed:

- `qwen36_gpu_full_layer_worker.c` was still using the legacy `1 + w` RMSNorm contract for:
  - `attn_norm`
  - `post_attn_norm`
- the integrated paths and the current CPU reference already used the raw-weight contract instead
- after restoring the standalone worker to raw-weight RMSNorm, the dedicated full-layer prefill isolation harness improved materially

Current standalone full-layer prefill isolation result:

- harness:
  - `misc/qwen36_full_layer_prefill_isolation.py`
- worker:
  - `qwen36-gpu-full-layer-worker`
- prompt:
  - `misc/qwen36_oracle_prompts/code_review.txt`
- observed `seq_rmse` after the fix:
  - `blk.3 ~= 0.0061`
  - `blk.7 ~= 0.0089`
  - `blk.11 ~= 0.0102`
  - `blk.15 ~= 0.0090`

Interpretation:

- the standalone full-layer worker is no longer catastrophically wrong in prefill isolation
- the old isolation result was overstating the problem because the worker itself had a normalization bug
- remaining standalone error is still non-zero, but it is now in the same regime as a narrow numeric audit rather than an obviously broken contract

Current integrated owned-session prefill result:

- harness:
  - `misc/qwen36_power2_owned_session_validator.py`
- owned-session configuration:
  - repo-truth ownership pattern
  - for every 4 layers: 3 hybrid, 1 full
- measured splice drift on the same prompt:
  - `depth=4 splice_seq_rmse ~= 0.00321`
  - `depth=8 splice_seq_rmse ~= 0.00641`
  - `depth=16 splice_seq_rmse ~= 0.01297`
  - `depth=32 splice_seq_rmse ~= 0.02237`

Cycle-by-cycle localization from the updated owned-session validator:

- `depth=8`
  - `POST_L2 ~= 0.00064`
  - `POST_FULL3 ~= 0.00321`
  - `POST_L6 ~= 0.00399`
  - `POST_FULL7 ~= 0.00641`
- `depth=16`
  - `POST_L2 ~= 0.00064`
  - `POST_FULL3 ~= 0.00321`
  - `POST_L6 ~= 0.00399`
  - `POST_FULL7 ~= 0.00641`
  - `POST_L10 ~= 0.00792`
  - `POST_FULL11 ~= 0.01092`
  - `POST_L14 ~= 0.01127`
  - `POST_FULL15 ~= 0.01297`
- `depth=32`
  - `POST_L2 ~= 0.00064`
  - `POST_FULL3 ~= 0.00321`
  - `POST_L6 ~= 0.00399`
  - `POST_FULL7 ~= 0.00641`
  - `POST_L10 ~= 0.00792`
  - `POST_FULL11 ~= 0.01092`
  - `POST_L14 ~= 0.01127`
  - `POST_FULL15 ~= 0.01297`
  - `POST_L18 ~= 0.01359`
  - `POST_FULL19 ~= 0.01552`
  - `POST_L22 ~= 0.01592`
  - `POST_FULL23 ~= 0.01791`
  - `POST_L26 ~= 0.01856`
  - `POST_FULL27 ~= 0.01944`
  - `POST_L30 ~= 0.02027`
  - `POST_FULL31 ~= 0.02237`

Interpretation:

- integrated owned-session prefill still drifts monotonically with owned depth
- the first owned full-attention boundary at `blk.3` is already wrong
- hybrid-only pre-full drift does accumulate across cycles
- but each full-attention boundary also adds an extra increment on top of that pre-full drift
- the first full boundary is still the first major jump:
  - `POST_L2 ~= 0.00064`
  - `POST_FULL3 ~= 0.00321`
- later full-boundary jumps are smaller than that first jump, but remain real
- that drift compounds even after the standalone full-layer worker normalization fix
- therefore the remaining primary bug is in the integrated prefill boundary/state path, not in the HF tail splice and not in ownership topology selection

Important later update from the owned-depth validator:

- harness:
  - `misc/qwen36_power2_owned_session_validator.py`
- this validator now uses repo-truth ownership:
  - for every 4 layers: 3 hybrid, 1 full
- with the standalone full-layer raw-weight RMSNorm correction in place, behavior at the splice boundary is materially healthier than the older study state implied:
  - `depth=8`
    - `argmax_equal=True`
    - `topk_overlap=7/8`
    - `splice_seq_rmse ~= 0.00641`
  - `depth=16`
    - `argmax_equal=True`
    - `topk_overlap=5/8`
    - `splice_seq_rmse ~= 0.01297`
  - `depth=32`
    - `argmax_equal=True`
    - `topk_overlap=5/8`
    - `splice_seq_rmse ~= 0.02237`
- this does not mean the integrated drift problem is solved
- it does mean the behavior surface is now much better aligned with CPU/HF than the earlier catastrophic-looking prefill reports suggested
- practical interpretation:
  - the remaining discrepancy now looks like bounded contract drift in the DS4-style mixed GPU/CPU runtime, not proof that the owned-session design is fundamentally wrong
- the validator also makes depth-8 the current best owned splice point among the audited power-of-2 depths
- this is a strong sign that the full-layer raw-weight normalization fix corrected a real load-bearing semantic bug rather than merely moving error around

Runtime scheduling optimization now adopted on both paths:

- the routed-expert FFN path originally did one GPU command buffer and one device sync per expert
- that was already collapsed on hybrid layers into:
  - one `begin_commands()`
  - all expert gate/up, activation, and down projections enqueued back-to-back
  - one `end_commands()`
  - batched readback and CPU weighting afterward
- the same transformation is now applied to the standalone/integrated full-layer FFN path in `qwen36_unified_full_gpu.inc`
- implementation detail:
  - `expert_down` is now sized for the whole expert union
  - each expert writes into a `ds4_gpu_tensor_view(...)` slot
  - the weighted CPU accumulation still happens after readback, so semantics are preserved
- this is a scheduling/throughput fix, not a semantic contract fix
- it should reduce avoidable GPU synchronization overhead without changing the mathematical target
- because the hybrid version of this idea already produced real gains, the full-layer version is now part of repo truth and should be kept when auditing remaining drift

Immediate audit target after this localization:

- focus on the first owned cycle first:
  - hybrid `blk.0..2` prefill output into full `blk.3`
  - full `blk.3` prefill state/output handoff back into the next hybrid cycle
- then, if needed, audit the smaller recurring per-full-layer increment that remains at later ownership points

Immediate action harness:

- dedicated first-full-boundary prefill substage audit:
  - `misc/qwen36_blk3_prefill_full_debug.py`
  - enables GPU-vs-CPU stage diffs for the first owned full layer during prefill
  - emits per-row `DBG_FULL[3]` diffs for:
    - `ATTN_IN`
    - `QG`
    - `KK`
    - `VV`
    - `Q_CUR`
    - `K_CUR`
    - `V_CUR`
    - `ATTN_OUT`
    - `PROJ_OUT`
    - `RESIDUAL`
    - `POST_LN`
    - `ROUTER`
    - `SHARED_OUT`
    - `ROUTED_OUT`
    - `FINAL_OUT`
  - prints a row-wise “worst stage” summary so the next patch target is chosen from an immediate concrete failure rather than a splice aggregate

Groundbreaking contract correction from the `blk.3` harness:

- the dedicated `blk.3` prefill full-stage harness proved that the dominant first-full-boundary error was in `Q_CUR/K_CUR`, not in the MoE tail
- this matters because `1 + w` had been helpful in other normalization sites, so it was plausible that full-layer `attn_q_norm/attn_k_norm` still wanted that contract
- the harness showed the opposite for this path:
  - full-layer `q_norm/k_norm` should use raw weights, not `1 + w`
- patched files:
  - `qwen36_unified_owned_worker.c`
  - `qwen36_unified_full_gpu.inc`
  - `qwen36_gpu_full_layer_worker.c`

Measured effect from the focused `blk.3` prefill harness (`misc/qwen36_blk3_prefill_full_debug.py`):

- before the patch:
  - `row=0 Q_CUR ~= 0.0321`
  - `row=0 K_CUR ~= 0.0315`
  - `row=4 K_CUR ~= 0.0341`
  - `row=7 K_CUR ~= 0.0248`
- after the patch:
  - `row=0 Q_CUR ~= 0.0183`
  - `row=0 K_CUR ~= 0.0180`
  - `row=4 K_CUR ~= 0.0194`
  - `row=7 K_CUR ~= 0.0141`

Interpretation:

- this is not a tiny cosmetic change; it cuts the dominant early full-layer drift by roughly forty percent or better on the audited rows
- therefore the old assumption “`1 + w` is probably still right here because it helped elsewhere” is no longer defensible for full-layer `attn_q_norm/attn_k_norm`
- the remaining prefill drift is now a post-correction problem, not evidence that the correction was wrong
- after this patch, some rows still have `K_CUR` as the worst stage, but the gap is much smaller and later `ROUTER`/FFN stages are now competitive on some rows

Interpretation rule:

- if the first bad row already spikes at an early attention stage, audit full-layer projection/rope/attention math
- if attention stages are clean but `ROUTER` or later MoE stages spike first, audit the full-layer FFN/router contract
- if `FINAL_OUT` jumps while earlier stages stay small, audit residual/handoff semantics rather than core math

Operational rule for the next study phase:

- run standalone full-layer prefill isolation first
- if isolation is bad, audit full-layer worker math
- if isolation is sane but owned-depth splice drift still grows, audit integrated owned-session prefill boundary/state semantics

The one-shot entrypoint now reflects that ordering:

- `misc/qwen36_blk0_prefill_one_shot.py`
  - phase 1: standalone full-layer prefill isolation
  - phase 2: power-of-2 owned-layer splice validation
  - phase 3: optional GPU-vs-CPU row-state summary from prefill logs

## Transitional Runtime Contract

The current working runtime path is transitional, not final-form DS4-native execution.

Active provenance sources:

- HF safetensors / HF model directory:
  - tokenizer
  - lightweight tail weights under `--splice-layer 39`
  - prompt-derived prefix/static capture workflows
- HuggingFace-cache Q8 GGUF:
  - active executable tensor contract for the current unified runtime
  - currently bound through `qwen36_35a3b_q8_bind(...)`
- DS4-side exporter/runtime work:
  - live contract fixture export
  - prefix/static fixture export
  - unified owned-worker architecture
  - ROCm shortcut/runtime ownership work

Interpretation:

- The DS4 contract and DS4-style runtime ideas are already being exercised.
- The active executable model contract is still the Q8 runtime GGUF.
- Direct execution of the experimental DS4 GGUF is not yet the expected behavior of the current runtime surface.

## Fixture Relationships

Two artifact types exist and they are not interchangeable.

- Prefix/static fixture:
  - produced by `misc/qwen36_c_prefix_fixture_export.py`
  - prompt-specific
  - contains prompt token ids plus captured HF layer-0 input sequence
  - format magic: `Q36PFX01`
- Live contract fixture:
  - produced by `misc/qwen36_live_contract_export.py`
  - prompt-agnostic
  - contains per-layer hybrid live weights/contract state
  - format magic: `Q36LCF01`

Path semantics:

- Older composed worker path:
  - can use a prompt-specific prefix fixture to seed cycle `0`
  - then carries live owned sequence / final-row data forward cycle by cycle
- Native owned-session path:
  - writes only cycle fixture lists into the owned-session config
  - `qwen36-unified-owned-worker-rocm` loads every listed cycle artifact through `fixture_load(...)`
  - therefore requires live fixtures for all owned hybrid layers, including `blk0`

This was the key provenance gap in the earlier study text:

- a prompt-specific `blk0` prefix fixture and a live `blk0` fixture occupy similar conceptual positions in the decode graph
- but they are different artifacts and cannot be substituted silently

## Runtime Provenance Matrix

The orchestration layer supports two materially different fixture-consumption modes.

- Composed Python worker path:
  - orchestrated through `run_owned_prefix_cycles(...)` in `misc/qwen36_hybrid_prefix_tail_greedy.py`
  - can use a prompt-specific `blk0.prefix.bin`
  - cycle `0` may own only two live fixtures when a separate prefix worker seeds the layer-0 input sequence
  - this is the path where “prefix artifact + later live fixtures” is structurally valid
- Native owned-session / unified worker path:
  - orchestrated through `OwnedSession._ensure_workers()` in `misc/qwen36_behavior_oracle.py`
  - config emitted by `write_owned_session_config(...)`
  - config contains only `cycle <full_layer> <fixture0,fixture1,...>` lines
  - the unified C worker then calls `fixture_load(...)` on every listed path
  - therefore every listed owned hybrid artifact must be `Q36LCF01`, including `blk0.live.bin`

Practical consequence:

- `blk0.prefix.bin` is valid for composed-prefix flows
- `blk0.live.bin` is required for native owned-session / unified ROCm flows
- if native owned-session is pointed at a prefix fixture, startup can fail immediately
- if runtime GGUF and live-fixture family are mixed, startup can succeed while decode semantics collapse

Recommended durable layout:

- `.cache/qwen36_live_contracts/Qwen3.6-35B-A3B-Q8_0/shared/blk0.live.bin`
- `.cache/qwen36_live_contracts/Qwen3.6-35B-A3B-Q8_0/shared/blk1.live.bin`
- ...
- `.cache/qwen36_live_contracts/Qwen3.6-35B-A3B-Q8_0/prompts/code_review/blk0.prefix.bin`

## Restoration Recipe

To restore a known-good transitional runtime setup, regenerate artifacts in two families and do not cross them.

Prompt-specific prefix artifact:

- source:
  - HF model directory `/mnt/e/tensors/Qwen3.6-35B-A3B`
- exporter:
  - `misc/qwen36_c_prefix_fixture_export.py`
- output family:
  - `.cache/qwen36_live_contracts/Qwen3.6-35B-A3B-Q8_0/prompts/<prompt_name>/blk0.prefix.bin`
- purpose:
  - only for composed prefix-seeding flows

Prompt-agnostic live fixtures:

- source:
  - active executable runtime GGUF
  - currently the HuggingFace-cache `Qwen3.6-35B-A3B-Q8_0.gguf`
- exporter:
  - `misc/qwen36_live_contract_export.py`
- output family:
  - `.cache/qwen36_live_contracts/Qwen3.6-35B-A3B-Q8_0/shared/blk*.live.bin`
- purpose:
  - required for native owned-session / unified ROCm runs

Known-good transitional rule:

- HF directory supplies tokenizer, prompt capture, and lightweight splice tail
- HuggingFace-cache Q8 GGUF supplies the executable runtime contract
- live fixtures must be regenerated from that same Q8 GGUF
- DS4 experimental GGUF remains a separate study/export family until the runtime binder is widened

What went wrong during the broken durable-cache phase:

- a `blk0` artifact from one family was treated as if it were from another
- later layers were mixed across executable-contract families
- the system therefore moved from:
  - “startup-valid and semantically coherent”
  - to
  - “startup-valid but semantically broken”

That failure mode is now part of the proof, not just an accident:

- artifact provenance is a first-class runtime input
- correctness depends on family consistency, not only file presence

Operational rule for the current native unified ROCm runtime:

- HF source from `/mnt/e/tensors/Qwen3.6-35B-A3B`
- runtime GGUF from the HuggingFace-cache `Q8_0.gguf`
- live fixtures regenerated from that same Q8 GGUF
- experimental DS4-style GGUF fixtures kept as a separate family until the binder is widened beyond `qwen36_35a3b_q8_bind(...)`

## What Is Proven

High-level:

- DS4-style runtime philosophy already has strong support:
  - one session timeline
  - one process
  - one GPU residency/cache policy
  - shortcut surfaces that reduce redundant work without obviously destroying semantics
- DS4-style exported contract artifacts are semantically meaningful enough to drive real runtime shortcuts while remaining compatible with the baseline executable contract during the transitional stage

Low-level:

- unified hybrid execution can reproduce prior worker math in one process
- unified full-layer ownership can cross the first full-attention boundary faithfully
- full 40-layer owned sequencing can remain numerically tight under composed calibration
- native oracle integration works end-to-end
- hybrid GPU-projection shortcuts produce real decode wins
- coherent decode in the current runtime requires provenance consistency across:
  - executable GGUF
  - live fixture family
  - orchestration mode

Recent operational evidence:

- clean Q8-aligned durable fixture run:
  - `/tmp/qwen36_unified_rocm_40layer_q8clean.json`
  - coherent 32-token decode
  - generated text continued the expected MoE/router discussion
  - prefill reached `~= 26.3s`
  - decode mean reached `~= 505 ms/token`
  - decode median reached `~= 458 ms/token`
  - late decode tokens reached `~= 266 .. 294 ms`
- mixed-family durable fixture run:
  - `/tmp/qwen36_unified_rocm_40layer_fullcycles.json`
  - gibberish from step `0`
  - first generated token already diverged to `_FINE`

Interpretation:

- this confirms the provenance rule operationally, not just structurally
- mixing DS4-derived live fixtures with the Q8 executable runtime surface can preserve startup while still destroying decode semantics
- on the clean Q8-aligned path, the unified ROCm runtime is now delivering sustained decode wins, not isolated fast tokens

## What Is Not Yet Proven

- Direct end-to-end execution of the experimental DS4 GGUF in the current runtime
- A binder/loader surface that can treat the experimental DS4 GGUF as the active executable contract instead of the Q8 GGUF

This is not a failure of the DS4 contract. It is a transitional-stage fact about the current runtime surface.

## Current Velocity

The project is no longer in a “prove one layer in isolation” phase.

It has already crossed into:

- one-process runtime unification
- full owned 40-layer decode runs
- real external ROCm speed improvements
- artifact-discipline lessons that now make rebuilds/restores reproducible

So the present advancement rate is best described as:

- semantic/runtime architecture is advancing quickly
- executable-contract migration is one major stage behind it
- decode performance is already in a real optimization regime rather than a rescue/debug regime

## Unified Worker Checkpoint

We now have the first real unified owned-worker compute slice, not just a protocol scaffold.

Binary:

- `qwen36-unified-owned-worker`

Current mode:

- `hybrid_only`
- one process
- one GGUF binding
- flattened live-fixture chain
- no child-worker handoff

First validated slice:

- `blk.0`
- `blk.1`
- `blk.2`

Validation:

- compared against `qwen36-live-contract-worker`
- same one-token `PREFILL_PREFIX_BIN`
- same one-token `STEP`
- compared `DUMP_LAST`

Result:

- `rmse ~= 4.9e-9`
- `max_abs ~= 5.96e-8`
- `cos ~= 1.0`

Interpretation:

- the first unified compute migration is numerically faithful
- the worker-farm removal itself is not changing the blk.0..2 hybrid math
- we can now extend the unified path upward from a real, calibrated base rather than a protocol-only scaffold

Extended unified hybrid result:

## Current Handoff Status

As of June 29, 2026, the `hybrid_gpu_cycles=0` path is again the stable baseline.

What changed in the study:

- the earlier full-layer debug harness was overstating `POST_LN` / `ROUTER` drift because the CPU reference path still used the legacy `1 + w` RMSNorm contract for:
  - `attn_norm`
  - `post_attn_norm`
- after restoring the raw-weight RMSNorm contract in the full-layer CPU/reference paths, the full-layer GPU-vs-CPU probe for `blk.3` became numerically sane again
  - typical `POST_LN`, `ROUTER`, `SHARED_OUT`, `ROUTED_OUT`, and `FINAL_OUT` probe RMSEs dropped from obviously-bad large values to small calibration-noise values

What is now actually proven:

- the probe itself is no longer the main source of noise
- `hybrid_gpu_cycles=1` still diverges semantically from `hybrid_gpu_cycles=0`
- that divergence is visible as a direct `cycles=0` vs `cycles=1` difference inside full layer `blk.3`

Current strongest direct `cycles=0` vs `cycles=1` signals for the step window around the repetition onset:

- `Q_CUR`
- `K_CUR`
- `ROUTER`
- then downstream:
  - `SHARED_OUT`
  - `ROUTED_OUT`
  - `FINAL_OUT`

Interpretation:

- this is no longer best described as a pure FFN/router-only bug
- the router amplifies the failure, but upstream full-layer attention state already differs between `cycles=0` and `cycles=1`
- the most likely remaining fault surface is now:
  - full-layer entry / cache-state / position-sensitive handoff into `blk.3`
  - followed by router/top-k amplification

Important practical read:

- per-run GPU-vs-CPU agreement can now look good for both runs individually
- but `cycles=0` and `cycles=1` still choose materially different router top-k sets at the same decode steps
- therefore the remaining bug is a real runtime semantic drift, not just a broken comparison harness

- the unified worker was also compared against `qwen36-live-contract-worker`
  across the full flattened 30-fixture owned hybrid chain
- same one-token prefill
- same one-token decode step
- compared `DUMP_LAST`

Result:

- `rmse ~= 6.5e-8`
- `max_abs ~= 7.15e-7`
- `cos ~= 0.99999994`

Interpretation:

- the entire current owned hybrid side can already be reproduced in one process
- hybrid unification is not the open risk anymore
- the next architectural risk moves to the first unified full-attention/GPU slice

## First Unified Full-Attention Boundary

We now also have the first unified full-attention boundary calibrated in-process.

Current mechanism:

- `qwen36-unified-owned-worker`
- opt-in mode:
  - `QWEN36_UNIFIED_PREFILL_FULL_CPU=1`

Validated scope:

- hybrid `blk.0..2` prefill in-process
- full layer `blk.3` prefill in-process
- hybrid `blk.0..2` incremental step in-process
- full layer `blk.3` incremental step in-process

Prefill compare:

- unified worker vs:
  - `qwen36-live-contract-worker` for blk.0..2
  - then `qwen36-c-full-layer-q8-dynamic --layer 3`
- result:
  - `rmse ~= 8.1e-8`
  - `max_abs ~= 1.43e-6`
  - `cos ~= 1.0`

Incremental step compare:

- unified worker one-token step vs full recompute oracle
- result:
  - `rmse ~= 1.12e-8`
  - `max_abs ~= 7.45e-8`
  - `cos ~= 1.0`

Interpretation:

- one-process semantics are now calibrated across the first full-attention boundary
- the remaining hard problem is backend/residency ownership, not semantic correctness of the unified state machine

## Unified Multi-Cycle Schedule Calibration

The unified worker now has multi-cycle evidence, not just a single boundary proof.

First multi-cycle checkpoint:

- unified schedule:
  - hybrid `blk.0..2`
  - full layer `blk.3`
  - hybrid `blk.4..6`
  - full layer `blk.7`
- one-token prefill plus one-token step
- compared against the composed baseline path

Result:

- `rmse ~= 2.57e-7`
- `max_abs ~= 7.03e-6`
- `cos ~= 1.0`

Full 10-cycle checkpoint:

- unified schedule:
  - all 30 hybrid fixtures in-process
  - full layers `3, 7, 11, 15, 19, 23, 27, 31, 35, 39`
- one-token prefill plus one-token step
- compared final `DUMP_LAST` against the composed baseline

Result:

- `rmse ~= 4.06e-5`
- `max_abs ~= 8.00e-4`
- `cos ~= 0.99999988`

Interpretation:

- the flattened one-process owned schedule remains numerically tight even across the full 40-layer path
- unified sequencing, recurrent state carry, KV carry, and final-row ownership are now sufficiently validated for study purposes
- this moves the main open question away from semantic correctness and toward backend residency policy:
  - one ROCm context
  - one allocation policy
  - one streaming/cache budget
  instead of the current worker-farm residency cliff

Oracle-boundary checkpoint:

- `qwen36_behavior_oracle.py` can now drive `qwen36-unified-owned-worker` directly as `--owned-session-worker-bin`
- explicit flag:
  - `--owned-session-unified-full-cpu`
- observed 10-cycle drop-in run:
  - prefill stayed inside the unified worker
  - decode stayed inside the unified worker
  - `owned_step_ms ~= 2231`
  - generated text remained coherent for the short smoke run:
    - `"Runtime is"`

Interpretation:

- the unified runtime is no longer only a private harness artifact
- it already fits the existing oracle/session interface closely enough to be exercised end-to-end

## Unified ROCm Pivot Checkpoint

The next architectural move is now started in code, not just planned.

New integration surface:

- `qwen36-unified-owned-worker-rocm`
- same unified owned-session envelope
- same one-process timeline
- explicit oracle flag:
  - `--owned-session-unified-full-gpu`

Current scope:

- hybrid fixture layers still run in-process on the existing CPU semantic path
- owned full-attention layers can now be selected through an in-process ROCm full-layer path in the unified worker
- ROCm full-layer support is compiled into the unified runtime behind:
  - `QWEN36_UNIFIED_FULL_GPU=1`

What is proven at this checkpoint:

- the ROCm-capable unified worker target builds successfully
- the oracle can now pass explicit unified GPU mode through its owned-session env plumbing
- this is the first build-time elimination step away from the per-layer full-worker farm

What is not yet proven here:

- no trustworthy device-runtime timing in this sandbox
- no claimed 40-owned residency result yet from the new in-process ROCm path

External ROCm checkpoint on RX 7900 XTX:

- `36-owned` unified run now completes through owned layer `35` in one process
- owned schedule:
  - hybrid fixtures `blk.0..2`, `blk.4..6`, ..., `blk.32..34`
  - in-process ROCm full layers `3, 7, 11, 15, 19, 23, 27, 31, 35`
- observed prefill:
  - unified owned prefill `~= 26.6s`
  - per owned full layer after the first cold load:
    - roughly `0.21s .. 0.28s`
- observed decode step:
  - unified owned decode step `~= 1.08s` at `seq_len=22`
  - next decode step `~= 0.84s` at `seq_len=23`
  - owned full GPU layers were typically in the low double-digit milliseconds each
- user-observed GPU footprint was only about `8 GiB`

## Hybrid GPU-Closure Checkpoint

The unified ROCm path has now crossed a more important threshold than simple projection offload.

Earlier decode shortcut state:

- hybrid GPU-projection cycles moved only selected hybrid projections onto GPU
- the hybrid FFN/router closure still ran on CPU
- this produced real wins, but decode still centered in the high hundreds of milliseconds per token

Current decode shortcut state:

- hybrid step scratch is persistent across tokens
- hybrid `gpu-proj` cycles now also run the hybrid FFN/router closure on GPU through GGUF-backed offsets:
  - router logits
  - shared expert gate/up/down
  - routed expert gate/up/down
- the DeltaNet / conv recurrent core still remains on CPU

Observed coherent 40-layer run:

- artifact:
  - `/tmp/qwen36_unified_rocm_40layer_q8clean.json`
- generated text:
  - coherent continuation of the MoE/router discussion
- timings:
  - `owned_prefill_ms ~= 26343`
  - `owned_step_ms` mean `~= 505`
  - `owned_step_ms` median `~= 458`
  - late owned tokens reached `~= 266 .. 294`

Interpretation:

- the main decode path is now materially GPU-owned beyond the full-attention layers alone
- the hybrid FFN/router closure was a real remaining bottleneck, and moving it to GPU produced another substantial sustained win
- the next likely decode bottleneck is the remaining hybrid recurrent core:
  - DeltaNet update
  - conv-ring closure
  - surrounding CPU-side recurrent math

Interpretation:

- this is the first real external evidence that the unified ROCm runtime removes the old worker-farm residency cliff
- the old 9-worker / 10-worker boundary problem was not an intrinsic cost of "owning more layers"; it was primarily a consequence of duplicating GPU process state and cache policy across many workers
- the single-process runtime now shows the expected DS4-style property:
  - per-layer state remains distinct
  - residency policy is shared
  - GPU cache growth is global rather than multiplied by worker count
- this makes the next experiment straightforward:
  - widen from `36-owned` to `40-owned`
  - include full layer `39`
  - use `--splice-layer 39` so the final norm + `lm_head` stay lightweight and in-process
  - verify that the former 10th-full-layer memory cliff is gone under one ROCm context

Interpretation:

- the work is now aligned with the DS4-style end state:
  - one process
  - one session timeline
  - one GPU context policy
- remaining work is runtime calibration and progressive replacement of the old worker-farm backend

External ROCm follow-up on RX 7900 XTX:

- `40-owned` unified run now completes through owned layer `39` in one process with:
  - `--owned-session-unified-full-gpu`
  - `--splice-layer 39`
- observed owned schedule:
  - all 30 hybrid fixtures in-process
  - in-process ROCm full layers `3, 7, 11, 15, 19, 23, 27, 31, 35, 39`
- observed baseline:
  - unified owned prefill `~= 27.0s`
  - unified owned decode step `~= 0.96s .. 1.04s/token`
  - full layers already cheap, typically single-digit to low-double-digit milliseconds each

Hybrid GPU-projection follow-up:

- new opt-in control:
  - `--owned-session-unified-hybrid-gpu-cycles N`
- current meaning:
  - convert the first `N` hybrid cycles from CPU-only step math to a mixed path where the hybrid projection slice runs on GPU
  - recurrent DeltaNet update, conv state handling, router closure, and shared-expert closure remain on CPU

Observed decode results at `seq_len=22`:

- `N = 1`
  - first converted hybrid layers logged as `[gpu-proj]`
  - owned decode step stayed around `~= 0.96s`
  - result proved the hook was live but not yet materially faster
- `N = 4`
  - owned decode step dropped to `~= 769.8 ms`
  - this is a real per-token win of roughly `190 ms`
- `N = 10`
  - all 30 hybrid fixture layers logged as `[gpu-proj]`
  - owned decode step was still `~= 765.4 ms`
  - this was near-parity with `N = 4` in the short early-decode sample, not another large drop in that particular measurement

Interpretation:

- the projection-only GPU cut is worth keeping
- it scales enough to remove a meaningful chunk of decode time
- short early-decode measurements did not show a large additional `4 -> 10` gain, but longer user runs reported token times as low as `~= 300 ms`, so the overall scaling behavior should not yet be described as a settled plateau
- current evidence is better described as a widespread real win whose magnitude remains prompt/token dependent
- this still strongly suggests the remaining decode bottleneck is not hybrid projection alone; it is the CPU middle/end of each hybrid layer:
  - conv/recurrent DeltaNet update
  - router and expert closure
  - shared-expert closure
  - CPU/GPU boundary glue and readback
- therefore the next optimization target should not be only "push this same projection trick to even more cycles"; it should be:
  - move more of hybrid step internals onto GPU, especially MoE/shared closure, or
  - pivot effort from hybrid polishing toward the fuller narrow GPU runtime end state

Late-window repetition diagnosis:

- user-visible degradation under `--owned-session-unified-hybrid-gpu-cycles > 0` was later pinned down more precisely:
  - weak semantic looping already appears with `hybrid_gpu_cycles = 1`
  - stronger/stamping repetition appears with larger owned GPU-hybrid coverage
  - onset is not immediate; it emerges later in decode rather than at the first generated tokens
- targeted late-window probes were then added around the onset region using:
  - `QWEN36_DBG_HYBRID_STEP_START`
  - `QWEN36_DBG_HYBRID_STEP_END`
- per-layer late-window compares for the GPU-owned hybrid layers showed:
  - `blk.0` hybrid GPU step remained tight versus CPU reference
  - `blk.1` hybrid GPU step remained tight versus CPU reference
  - `blk.2` hybrid GPU step remained tight versus CPU reference
  - typical `FINAL_OUT` error in those probes stayed in the low `1e-4 .. 1e-3` regime

Interpretation:

- the repetition regression is **not** explained by the math inside the first GPU-owned hybrid layers themselves
- the likely fault surface moved from "hybrid internals" to "cycle boundary or full-layer decode step semantics"

Cycle-boundary probe result:

- a direct CPU-reference boundary probe was added around cycle `0`:
  - `POST_L2`
  - `POST_FULL3`
  - `PRE_L4`
  - `POST_L4`
- the decisive result was:
  - `POST_L2` stays tight
    - typically `~= 2e-4 .. 1e-3` RMSE
  - the first major jump appears immediately at `POST_FULL3`
    - commonly `~= 0.01 .. 0.05` RMSE
    - with larger outliers in some late steps
  - `PRE_L4` is identical to `POST_FULL3`
    - therefore no extra corruption is introduced by the handoff buffer itself
  - `POST_L4` is only slightly different from `PRE_L4`
    - layer `4` propagates the error but does not originate it

Current conclusion:

- the first harmful decode-step divergence under the unified ROCm path is the **GPU full-layer step** at `blk.3`
- this is not a `full -> next hybrid` handoff corruption bug
- heavy full-layer debug then localized the first bad math inside `blk.3` more precisely:
  - `ATTN_IN` remains essentially exact against CPU reference
  - the first meaningful drift appears immediately in the raw projection outputs:
    - `QG`
    - `KK`
    - `VV`
  - downstream tensors such as `ATTN_OUT`, `PROJ_OUT`, and `FINAL_OUT` were still much tighter in early decode steps
- that moved the immediate mitigation target from "full-layer attention/state semantics in general" to the narrower surface:
  - full-attention `attn_q / attn_k / attn_v` projection generation inside the GPU full-layer path
- the first incremental mitigation now in tree is:
  - keep the unified full-layer GPU path for RMS norm, output projection, and FFN
  - but source full-layer `Q/K/V` projections from decoded CPU weights cached in `full_layer_small_cache`
  - this is intended to eliminate the first proven divergence site while preserving the rest of the narrowed GPU execution shape
- the next debug/fix focus should be:
  - whether this projection-only mitigation restores sane decode behavior for `hybrid_gpu_cycles >= 1`
  - if not, the next remaining full-layer surfaces are:
    - output projection
    - FFN / router path
    - full-layer state update semantics after those corrected projections
- to reduce shell/invocation mistakes during these probes, the canonical entrypoint is now:
  - `misc/qwen36_full_layer_probe.py`
  - it assembles the known-good patched-session oracle command, the ten full-layer placements, and the shared fixture set automatically
  - intended modes:
    - `--probe baseline`
    - `--probe boundary`
    - `--probe breadcrumb`
    - `--probe full-debug`

## Confirmed Contract Facts

- Target is `Qwen3.6-35B-A3B` only.
- Architecture observed through HF:
  - 40 layers
  - alternating schedule of 3 linear-attention layers, then 1 full-attention layer
  - 256 routed experts
  - top-8 experts per token
  - shared expert path per MoE block
- Exported `v0` policy:
  - load-bearing tensors retained at higher precision
  - routed experts pushed aggressively:
    - `ffn_gate_exps.weight` -> `IQ2_XXS`
    - `ffn_up_exps.weight` -> `IQ2_XXS`
    - `ffn_down_exps.weight` -> `IQ2_S` or `IQ2_XXS`

## What Earlier Llama.cpp Tests Proved

- The full `v0` GGUF behaves catastrophically in llama.cpp compared with Q8.
- That does **not** settle the DS4-style question.
- DS4 docs make clear that DwarfStar is not a generic GGUF runner; runtime + artifact are co-designed.
- Therefore llama.cpp remains a useful oracle for broad drift, but not a final validator of a narrow runtime artifact.

## Micro Validators Added

- `misc/qwen36_payload_probe.py`
  - dequantized selected GGUF tensors and compared them to HF source tensors
  - results showed sane numerical correlation rather than malformed payloads
- `misc/qwen36_router_probe.py`
  - native HF router oracle for selected prompts/layers
- `misc/qwen36_router_compare.py`
  - recomputes router logits from exported GGUF router tensors under HF hidden states
- `misc/qwen36_moe_compare.py`
  - replays the sparse MoE block using exported GGUF tensors under HF hidden states
- `misc/qwen36_layer_trace_export.py`
  - exports reusable per-layer activation fixtures from native HF Qwen
- `misc/qwen36_moe_replay_from_trace.py`
  - replays the MoE half of a traced layer offline from GGUF + saved fixtures
- `misc/qwen36_c_moe_fixture_export.py`
  - exports a single-layer staged fixture with selected experts expanded to `f32` for C-side replay
- `qwen36_c_moe_replay.c`
  - pure C replay of the post-mixer residual boundary, traced MoE branch, and final residual merge

## Router Findings

Prompt: `Hello world`

- `blk.0`: `top1=1`, `topk_set=True`, `prefix=8`, `logits_rmse=0.00889606`, `logits_cos=0.99999893`
- `blk.3`: `top1=1`, `topk_set=True`, `prefix=6`, `logits_rmse=0.00900498`, `logits_cos=0.99999890`
- `blk.34`: `top1=1`, `topk_set=True`, `prefix=5`, `logits_rmse=0.00921786`, `logits_cos=0.99999867`

Interpretation:

- Router behavior is extremely well preserved on sampled layers.
- The exported gate tensors are not the immediate cause of the full-model collapse.

## MoE Block Findings

Prompt: `Hello world`

- `blk.0`: `rmse=0.00306428`, `mae=0.00243222`, `max_abs=0.01214167`, `cosine=0.94642385`
- `blk.3`: `rmse=0.00456983`, `mae=0.00363975`, `max_abs=0.01707505`, `cosine=0.92051146`
- `blk.34`: `rmse=0.01186258`, `mae=0.00945264`, `max_abs=0.04355486`, `cosine=0.97934429`

Interpretation:

- The aggressive routed-expert quantization still reproduces the sampled HF MoE block outputs reasonably well.
- This supports the DS4-style hypothesis that routed experts are more crushable than load-bearing surfaces.
- The remaining failure is more likely in end-to-end runtime semantics than in router tensor export alone.

## Current Best Reading

The evidence so far supports a narrow-runtime path:

- load-bearing routing path is stable
- routed expert math is noisy but still directionally faithful on sampled blocks
- full-model failure in llama.cpp likely reflects runtime/semantic mismatch, not merely bad tensor bytes

## Runtime Direction

- We are no longer treating mixed-GGUF behavior inside llama.cpp as the final verdict.
- The active plan is a Qwen-native narrow runtime with:
  - exact target `Qwen3.6-35B-A3B`
  - text-only
  - batch size `1`
  - short context first
  - greedy decode first
  - CPU-first validation
- The current validator strategy is:
  - prove local layer math in C
  - chain multiple validated layers
  - splice the C-owned prefix back into HF and compare tail logits / greedy token
  - only then widen prefix length and consider more aggressive runtime ownership

## Offline Layer Trace Findings

Trace prompt: `Hello world`

Trace bundle:

- `/tmp/qwen36_trace_hello.npz`
- `/tmp/qwen36_trace_hello.json`

Offline replay against the trace:

- `blk.0`: `top1=1`, `topk_set=True`, `prefix=8`, `mlp_rmse=0.00306633`, `mlp_cos=0.94643281`, `layer_rmse=0.00306719`, `layer_cos=0.99243166`
- `blk.3`: `top1=1`, `topk_set=True`, `prefix=6`, `mlp_rmse=0.00454776`, `mlp_cos=0.92077417`, `layer_rmse=0.00454773`, `layer_cos=0.99520122`

Interpretation:

- We now have a DS4-style offline validator loop for the MoE half of a layer.
- The routed path remains stable under frozen fixtures.
- The residualized full layer output is even closer than the raw MoE branch, which suggests substantial quality headroom remains for runtime tuning before generation-level behavior collapses.

## C Replay Findings

We now have a narrow C validator for the MoE half of a traced layer. It does not depend on live HF inference during replay; the expensive path is moved into one-time fixture export.

Fixtures used:

- `/tmp/qwen36_blk0_moe_fixture.bin`
- `/tmp/qwen36_blk34_moe_fixture.bin`
- `/tmp/qwen36_blk39_moe_fixture.bin`

Results:

- `blk.0`
  - `mlp_rmse=0.00306396`
  - `mlp_cosine=0.94666774`
  - `layer_rmse=0.00306481`
  - `layer_cosine=0.99244641`
- `blk.34`
  - `mlp_rmse=0.01186659`
  - `mlp_cosine=0.97921546`
  - `layer_rmse=0.01187569`
  - `layer_cosine=0.99843615`
- `blk.39`
  - `mlp_rmse=0.02305724`
  - `mlp_cosine=0.99759201`
  - `layer_rmse=0.02306749`
  - `layer_cosine=0.99859360`

Interpretation:

- The narrow C math path reproduces the Python replay closely.
- Across early and late layers, the residualized layer output remains extremely close to HF on the sampled trace.
- This materially strengthens the case that the current bottleneck is full runtime semantics and scheduling, not the selective-quant MoE branch itself.

## Weight-Driven Full Decoder Findings

We then moved from isolated MoE replay to a full decoder-layer replay driven directly by exported layer weights.

Results:

- `blk.0`
  - `layer_output_cosine=0.98409028`
  - `layer_output_rmse=0.00340938`
  - `router_logits_cosine=0.99998500`
  - `router_scores_cosine=0.99906697`
- `blk.1`
  - `layer_output_cosine=0.98393710`
  - `layer_output_rmse=0.00394468`
  - `router_logits_cosine=0.99999163`
  - `router_scores_cosine=0.99983778`

Interpretation:

- The Qwen-native linear-attention plus MoE block can be reproduced in narrow C with high numerical fidelity.
- Router exact-index equality is not yet guaranteed, but router surfaces are close enough that we now treat output-token agreement as the real pass/fail criterion.

## Prefix Chain And Tail Splice Findings

Two weight-driven decoder layers were chained in C and then spliced back into the HF tail.

Results:

- C-owned chain:
  - `layer0_output_cosine=0.98409028`
  - `layer1_input_handoff_cosine=0.98409028`
  - `layer1_output_cosine=0.97163435`
- HF tail splice on `patch_plan`:
  - `logits_cosine=0.99953073`
  - `logits_rmse=0.07489589`
  - `argmax_equal=True`
  - baseline next token id `17`, text `"2"`
  - patched next token id `17`, text `"2"`

Interpretation:

- This is the strongest current evidence that the narrow runtime path is viable.
- Even with visible hidden-state drift after two owned layers, the HF tail still lands on the same next greedy token.
- The immediate goal is to extend prefix ownership beyond two layers, not to chase bit-identical activations.

## Dynamic Prefix Decode Findings

We replaced the prompt-bound expert-fixture replay with a dynamic GGUF-backed prefix runner:

- new binary: `qwen36-c-prefix-q8-chain-dynamic`
- C-owned path:
  - token embedding row fetch from Q8 GGUF
  - `blk.0..2` hybrid SSM / DeltaNet forward
  - dynamic router logits
  - dynamic router top-k
  - on-demand routed-expert Q8 decode from GGUF with cache
- HF-owned path:
  - splice at `blk.2`
  - tail forward and greedy next-token choice

Prompt:

- `misc/qwen36_oracle_prompts/patch_plan.txt`

Result:

- token 0: `2` vs HF `2`
- token 1: `.` vs HF `.`
- token 2: ` \n` vs HF ` \n`
- token 3: `3` vs HF `3`

Decoded continuation:

- hybrid: `"2. \n3"`
- HF: exact match for the first 4 greedy tokens

Interpretation:

- This is the first real multi-token Qwen-native DS4-style result.
- The earlier step-1 failure was caused by prompt-bound expert fixtures, not by a bad architectural contract.
- The owned prefix is now participating in true decode rather than trace replay theater.
- This is strong enough to justify shifting effort from heavy CPU validator work toward:
  - longer token-prefix match studies
  - owning `blk.3` next
  - GPU implementation with final-token validation as the primary oracle

## First Full-Attention Boundary Crossed

We then extended ownership across the first full-attention layer:

- `blk.0..2`: `qwen36-c-prefix-q8-chain-dynamic`
- `blk.3`: `qwen36-c-full-layer-q8-dynamic`
- HF splice point moved from layer `2` to layer `3`

Prompt:

- `misc/qwen36_oracle_prompts/patch_plan.txt`

Result:

- token 0: `2` vs HF `2`
- token 1: `.` vs HF `.`
- token 2: ` \n` vs HF ` \n`
- token 3: `3` vs HF `3`

Decoded continuation:

- hybrid (`blk.0..3` owned): `"2. \n3"`
- HF: exact match for the first 4 greedy tokens

Interpretation:

- The first full-attention seam is now crossed in a real hybrid decode path.
- The project is no longer bottlenecked on “can we survive the first attention layer?”
- `token_embd + blk.0..3` is strong enough that the next frontier is not more CPU validator depth.
- The correct next emphasis is:
  - GPU ownership of the same prefix path
  - token-level oracle checks
  - longer prompt / workload validation

## Sequence-Aware Chain Fix

The first sequence-aware decoder chain regressed badly, but that turned out to be a validator issue rather than evidence that the narrow contract had failed.

What was wrong:

- the chain needed full `seq_len x hidden` handoff between owned linear-attention layers
- the fixture needed per-token routed-expert bindings rather than only the last token's top-k experts
- the exact target model uses `Qwen3_5MoeRMSNorm`, whose block norms apply:
  - `output = normalized_hidden * (1 + weight)`

The last point was re-checked against the exact target implementation loaded by `trust_remote_code=True`:

- class: `transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeRMSNorm`
- forward:
  - normalize
  - multiply by `(1 + weight)`

That means the earlier `+1` norm contract was correct for `Qwen3.6-35B-A3B`.

## Sequence-Aware 3-Layer Chain

After fixing the sequence handoff path and restoring the exact RMSNorm contract, the 3-layer owned prefix became materially healthier.

Results:

- `blk.0`
  - `layer0_output_cosine=0.85298485`
  - `layer0_output_seq_cosine=0.80735630`
- `blk.1`
  - `layer1_output_cosine=0.75390723`
  - `layer1_output_seq_cosine=0.75713388`
- `blk.2`
  - `layer2_output_cosine=0.62624228`
  - `layer2_output_seq_cosine=0.71140070`

Interpretation:

- The earlier `blk.2` collapse was primarily a harness problem.
- Sequence-faithful handoff matters for Qwen's linear-attention schedule.
- Three owned layers are now drifting, but no longer catastrophically.
- This is back in the range where a tail-level greedy-token splice check is worth doing again.

## Reconciliation Note

The maintained `.99x` results remain important and should not be discarded just because the newer sequence-chain path reports lower cosines.

The strongest reconciliation clue is that the dedicated narrow validators still reproduce some of the original high-fidelity surfaces:

- `qwen36-c-linear-conv-replay`
  - `qkv_cosine=0.99991498`
  - `conv_post_cosine=0.99998799`
- `qwen36-c-linear-norm-replay`
  - `gated_norm_cosine=0.99999846`

Those are the same kinds of load-bearing linear-attention surfaces that the newer sequence-aware decoder-chain instrumentation currently reports much lower for `blk.0`, for example:

- `layer0_qkv_seq_cosine=0.47046128`

Interpretation:

- The old `.99x` checkpoints are still real.
- The current sequence-aware decoder-chain path is not yet fully trustworthy as a measurement harness.
- Until that contradiction is resolved, the lower sequence-chain cosines should be treated as a harness/debugging signal, not as a final statement about runtime viability.

## Root Surface Check

We also added a direct GGUF row-read path for root surfaces:

- `qwen36_gguf_read_tensor_bytes(...)`
- `qwen36_c_root_q8_selected_logits.c`
- `misc/qwen36_root_hf_probe.py`

Current Q8-oracle root check:

- `final_norm_cosine=0.99668541`
- `selected_logits_cosine=0.99968571`
- absolute logit scale still needs tightening

Interpretation:

- The root contract is close enough to keep moving.
- Remaining mismatch looks more like numerical scale drift than a broken tensor-binding contract.

## Next Immediate Steps

- Wire and use `qwen36-c-decoder-chain-weight` as the reusable prefix runner.
- Export `blk.2` fixtures serially only, never with concurrent HF export jobs.
- Extend the HF tail-splice test from two owned layers to three.
- If the next-token argmax still matches, keep widening the owned prefix before doing any new quant-policy redesign.

## Short Instruction Prompt Check

We also ran the same C replay path on a short instruction-style prompt from `misc/qwen36_oracle_prompts/patch_plan.txt`.

Trace:

- `/tmp/qwen36_trace_patch_plan.npz`
- `/tmp/qwen36_trace_patch_plan.json`

C replay fixtures:

- `/tmp/qwen36_patch_plan_blk0.bin`
- `/tmp/qwen36_patch_plan_blk34.bin`

Results:

- `blk.0`
  - `mlp_rmse=0.00324508`
  - `mlp_cosine=0.95605996`
  - `layer_rmse=0.00324547`
  - `layer_cosine=0.98511075`
- `blk.34`
  - `mlp_rmse=0.01777709`
  - `mlp_cosine=0.96867080`
  - `layer_rmse=0.01777382`
  - `layer_cosine=0.99531901`

Interpretation:

- The lean contract still looks healthy on a prompt that is more instruction-shaped than `Hello world`.
- Late-layer MoE drift grows, which is expected under aggressive expert quantization, but the residualized full-layer output still stays very close to the traced HF target.
- This is another point in favor of moving on to a full-layer narrow C shim rather than spending more time debating whether the selective MoE policy is fundamentally viable.

## Stage-Aware C Fixture

The C fixture format was then extended from a collapsed MoE fixture to a staged post-mixer fixture.

Current staged fields:

- `layer_input`
- `mixer_out`
- `post_attn_ln`
- `residual_after_mixer`
- `mlp_out`
- `layer_output`

This lets the C validator prove the layer plumbing explicitly instead of only consuming a pre-collapsed residual vector.

Example check:

- `/tmp/qwen36_patch_plan_blk0_v2.bin`
  - `residual_rmse=0.00000000`
  - `residual_cosine=1.00000000`
  - `mlp_rmse=0.00324508`
  - `mlp_cosine=0.95605996`
  - `layer_rmse=0.00324547`
  - `layer_cosine=0.98511075`

Interpretation:

- The post-mixer residual boundary is now explicit in the narrow C validator.
- The next missing semantic piece is no longer MoE residual wiring; it is the token-mixer implementation itself.

## Linear-Attention Trace Contract

We extracted the first dedicated `linear_attention` trace for `blk.0` on the short instruction-style prompt:

- trace:
  - `/tmp/qwen36_linear_attn_blk0_patch_plan.npz`
  - `/tmp/qwen36_linear_attn_blk0_patch_plan.json`

Prompt:

- `"Implementation plan:\n1. \n"`
- `prompt_tokens=7`

Captured stage surfaces:

- `layer_input` shape `2048`
- `input_ln` shape `2048`
- `in_proj_qkv` shape `8192`
- `in_proj_z` shape `4096`
- `in_proj_b` shape `32`
- `in_proj_a` shape `32`
- `out_proj_in` shape `4096`
- `out_proj_out` shape `2048`
- `mixer_out` shape `2048`
- `layer_output` shape `2048`

Layer config surfaced by HF:

- `hidden_size=2048`
- `num_v_heads=32`
- `num_k_heads=16`
- `head_k_dim=128`
- `head_v_dim=128`
- `key_dim=2048`
- `value_dim=4096`
- `conv_kernel_size=4`

Relevant weight shapes:

- `in_proj_qkv.weight`: `(8192, 2048)`
- `in_proj_z.weight`: `(4096, 2048)`
- `in_proj_b.weight`: `(32, 2048)`
- `in_proj_a.weight`: `(32, 2048)`
- `conv1d.weight`: `(8192, 1, 4)`
- `out_proj.weight`: `(2048, 4096)`
- `A_log`: `(32,)`
- `dt_bias`: `(32,)`

Interpretation:

- We now have a concrete, model-specific token-mixer surface for the linear-attention path.
- The immediate C target is not the whole DeltaNet rule at once; it is the projection and staging contract:
  - normalized layer input
  - four projections
  - gated mixer core feeding `out_proj`
  - `mixer_out`
- Once those boundaries are reproduced in C, the remaining work is the recurrent/chunked delta rule itself.

## Linear-Attention C Projection Replay

We added a narrow C validator for the linear-attention projection/staging boundary:

- `misc/qwen36_c_linear_fixture_export.py`
- `qwen36_c_linear_replay.c`

Fixture used:

- `/tmp/qwen36_linear_blk0_patch_plan.bin`

This validator currently checks:

- `in_proj_qkv(input_ln)`
- `in_proj_z(input_ln)`
- `in_proj_a(input_ln)`
- `in_proj_b(input_ln)`
- `out_proj(out_proj_in)`

Results for `blk.0` on the short instruction-style prompt:

- `qkv_rmse=0.02860457`, `qkv_cosine=0.99992437`
- `z_rmse=0.03159917`, `z_cosine=0.99992645`
- `a_rmse=0.00437301`, `a_cosine=0.99999912`
- `b_rmse=0.00302457`, `b_cosine=0.99999833`
- `out_proj_rmse=0.00052508`, `out_proj_cosine=0.99885428`

Interpretation:

- The load-bearing linear-attention projection surfaces are reproducing extremely closely under the current quant policy.
- The control projections (`a`, `b`) are especially tight.
- `qkv` and `z` show larger absolute error than the scalar paths, but remain extremely aligned directionally.
- This further supports the view that the remaining unknown is the DeltaNet core semantics, not obvious corruption in the surrounding projection weights.

## Linear-Attention Conv Replay

We then extended the linear-attention trace to include full-sequence projection and conv surfaces:

- `input_ln_seq`
- `in_proj_qkv_seq`
- `conv1d_raw`

New tools:

- `misc/qwen36_c_linear_conv_fixture_export.py`
- `qwen36_c_linear_conv_replay.c`

Fixture:

- `/tmp/qwen36_linear_conv_blk0_patch_plan.bin`

This validator checks the pre-core sequence path:

- full-sequence `in_proj_qkv(input_ln_seq)`
- full-sequence causal depthwise `conv1d + SiLU`

Results for `blk.0` on the short instruction-style prompt:

- `qkv_rmse=0.02849316`, `qkv_cosine=0.99991498`
- `conv_post_rmse=0.00312225`, `conv_post_cosine=0.99998799`

Interpretation:

- The sequence-aware pre-core linear-attention path is also reproducing extremely closely.
- This materially narrows the remaining semantic gap:
  - projections are validated
  - causal depthwise conv is validated
  - out projection is validated
- The remaining unknown for `linear_attention` is now concentrated in the gated DeltaNet recurrent/chunked rule and the gated RMS normalization feeding `out_proj_in`.

## DeltaNet Core-Boundary Replay

We then extracted the actual inputs consumed by the DeltaNet rule:

- `query`
- `key`
- `value`
- `beta`
- `g`
- and the traced `core_attn_out_pre_norm`

New tools:

- `misc/qwen36_linear_core_trace_export.py`
- `misc/qwen36_c_linear_core_fixture_export.py`
- `qwen36_c_linear_core_replay.c`

Fixture:

- `/tmp/qwen36_linear_core_blk0_patch_plan.bin`

This validator currently checks the full core boundary formation:

- split `conv1d + SiLU` output into `query`, `key`, `value`
- repeat-interleave `query` and `key` from `16` K-heads to `32` V-heads
- compute `beta = sigmoid(b)`
- compute `g = -exp(A_log) * softplus(a + dt_bias)`

Results for `blk.0` on the short instruction-style prompt:

- `query_rmse=0.00028200`, `query_cosine=0.99999937`
- `key_rmse=0.00017058`, `key_cosine=0.99999837`
- `value_rmse=0.00045719`, `value_cosine=0.99999986`
- `beta_rmse=0.00087667`, `beta_cosine=0.99999904`
- `g_rmse=0.00000076`, `g_cosine=1.00000000`

Interpretation:

- The entire pre-recurrence DeltaNet core boundary is now validated extremely tightly in the narrow C path.
- This removes nearly all ambiguity around tensor shaping and control-signal formation for `linear_attention`.
- The remaining semantic gap is now sharply localized to:
  - the recurrent/chunked gated delta rule itself
  - the gated RMS normalization that transforms `core_attn_out_pre_norm` into `out_proj_in`

## Gated RMS Norm Replay

We then validated the gated RMS normalization stage directly:

- `misc/qwen36_c_linear_norm_fixture_export.py`
- `qwen36_c_linear_norm_replay.c`

Fixture:

- `/tmp/qwen36_linear_norm_blk0_patch_plan.bin`

This validator checks the transformation from:

- `core_attn_out_pre_norm`
- `z`
- `ssm_norm.weight`

into:

- `out_proj_in`

using the exact HF formula:

- RMS norm on the `128`-wide head state
- elementwise multiply by `ssm_norm.weight`
- elementwise multiply by `silu(z)`

Results for `blk.0` on the short instruction-style prompt:

- `gated_norm_rmse=0.00005202`
- `gated_norm_cosine=0.99999846`

Interpretation:

- The gated RMS normalization stage is essentially exact in the narrow C path.
- This removes the last major ambiguity outside the actual DeltaNet recurrence.
- At this point the remaining `linear_attention` unknown is the recurrent/chunked gated delta rule itself.

## Stubbed Linear-Attention Layer Replay

We then completed a stubbed end-to-end `linear_attention` layer validator in C:

- `misc/qwen36_c_linear_layer_stub_fixture_export.py`
- `qwen36_c_linear_layer_stub_replay.c`

Fixture:

- `/tmp/qwen36_linear_layer_stub_blk0_patch_plan_v2.bin`

This validator injects the traced `core_attn_out_pre_norm` and validates everything around it:

- gated RMS norm
- `out_proj`
- `mixer_out`
- residual add to produce `residual_after_mixer`

Results for `blk.0` on the short instruction-style prompt:

- `mixer_rmse=0.00052311`
- `mixer_cosine=0.99885901`
- `residual_after_mixer_rmse=0.00052311`
- `residual_after_mixer_cosine=0.99940726`

Interpretation:

- We now have one traced `linear_attention` layer validated end-to-end in C except for the DeltaNet recurrence itself.
- The surrounding runtime contract is no longer speculative.
- This is the strongest signal so far that a DS4-style narrow Qwen runtime is technically tractable.

## CPU DeltaNet Recurrence Reference

We then replaced the core-boundary-only replay with a real CPU recurrence reference in C.

Updated tool:

- `qwen36_c_linear_core_replay.c`

Updated fixture:

- `/tmp/qwen36_linear_core_blk0_patch_plan_v2.bin`

This replay now:

- consumes traced `query`, `key`, `value`, `beta`, and `g`
- applies the recurrent gated delta update on CPU
- produces `core_attn_out_pre_norm`

Results for `blk.0` on the short instruction-style prompt:

- `query_rmse=0.00028200`, `query_cosine=0.99999937`
- `key_rmse=0.00017058`, `key_cosine=0.99999837`
- `value_rmse=0.00045719`, `value_cosine=0.99999986`
- `beta_rmse=0.00087667`, `beta_cosine=0.99999904`
- `g_rmse=0.00000076`, `g_cosine=1.00000000`
- `core_rmse=0.00006985`, `core_cosine=0.99999530`

Interpretation:

- The CPU recurrence reference already matches the traced DeltaNet core extremely closely on the sampled layer and prompt.
- This means the remaining work is no longer discovering the semantics; it is integrating the validated recurrence back into the full traced `linear_attention` layer validator.
- The narrow Qwen runtime path has now crossed from boundary validation into an actually working CPU reference implementation of the hard mixer core.

## Full Linear-Attention Layer Replay

We then integrated the validated CPU recurrence back into the traced `linear_attention` layer replay:

- `misc/qwen36_c_linear_layer_full_fixture_export.py`
- `qwen36_c_linear_layer_full_replay.c`

Fixture:

- `/tmp/qwen36_linear_layer_full_blk0_patch_plan.bin`

This validator now runs, in one narrow C path:

- traced `query/key/value/beta/g`
- CPU DeltaNet recurrence
- gated RMS norm
- `out_proj`
- `mixer_out`
- residual add to `residual_after_mixer`

Results for `blk.0` on the short instruction-style prompt:

- `mixer_rmse=0.00052311`
- `mixer_cosine=0.99885931`
- `residual_after_mixer_rmse=0.00052311`
- `residual_after_mixer_cosine=0.99940735`

Interpretation:

- We now have a completed traced `linear_attention` mixer path in narrow C on CPU.
- The remaining work for a full decoder layer is no longer on the mixer side; it is joining this completed mixer path with the MoE half and then validating the full decoder block boundary.

## Full Decoder-Layer Replay

We then joined the completed `linear_attention` mixer path with the staged MoE replay into one traced decoder-layer validator:

- `misc/qwen36_c_decoder_layer_fixture_export.py`
- `qwen36_c_decoder_layer_replay.c`

Fixture:

- `/tmp/qwen36_decoder_layer_blk0_patch_plan.bin`

This validator runs, in one narrow C path:

- CPU DeltaNet recurrence
- gated RMS norm
- `out_proj`
- residual add to `residual_after_mixer`
- routed experts
- shared expert
- final residual add to `layer_output`

Results for `blk.0` on the short instruction-style prompt:

- `mixer_rmse=0.00052311`
- `mixer_cosine=0.99885931`
- `residual_after_mixer_rmse=0.00052311`
- `residual_after_mixer_cosine=0.99940735`
- `mlp_rmse=0.00324614`
- `mlp_cosine=0.95611212`
- `layer_output_rmse=0.00327484`
- `layer_output_cosine=0.98485016`

Interpretation:

- We now have one traced Qwen decoder block replaying end to end in narrow CPU C.
- The remaining work has shifted from “can we model the block” to “how far can we scale this into multi-layer forward and short greedy continuation.”

## Next Steps

1. Decide whether the next milestone is a `full_attention` layer validator or a minimal multi-layer greedy decode path.
2. If greedy decode comes next, chain two or more validated layers with traced embeddings/logits for a very short continuation path.
3. Compare produced next-token choices against HF or llama.cpp oracle on short prompts.
4. Use those results to decide when backend work becomes justified.

## Two-Layer Decoder Chain Replay

We then took the next concrete step toward an actual stacked narrow runtime by chaining two traced decoder fixtures in one C binary:

- `qwen36_c_two_decoder_layers_replay.c`

Fixtures:

- `/tmp/qwen36_decoder_layer_blk0_patch_plan.bin`
- `/tmp/qwen36_decoder_layer_blk1_patch_plan.bin`

This replay:

- runs the full narrow CPU decoder implementation for `blk.0`
- feeds the produced `layer_output` directly into the narrow CPU replay for `blk.1`
- checks both the layer-to-layer handoff and the final `blk.1` output against traced references

Results on the short instruction-style prompt:

- `layer0_output_rmse=0.00327484`
- `layer0_output_cosine=0.98485016`
- `layer1_input_handoff_rmse=0.00327484`
- `layer1_input_handoff_cosine=0.98485016`
- `layer1_output_rmse=0.00519399`
- `layer1_output_cosine=0.97569324`

Interpretation:

- The narrow CPU path is no longer limited to isolated block replays; it now preserves a real decoder-layer handoff across consecutive layers.
- Error does accumulate, but it remains bounded enough after two layers to justify continuing toward a short stacked forward path.
- This is the first direct evidence that the Qwen-native narrow runtime contract is coherent beyond a single decoder block.

Immediate implication:

- The next useful milestone is not generic infrastructure. It is a minimal stacked forward path that adds embeddings and logits around a small prefix of validated layers so we can compare actual next-token choices and short English continuations against the HF oracle.

## Weight-Driven Hybrid Mixer

We then replaced the last major traced shortcut on the hybrid mixer side.

New artifacts:

- `misc/qwen36_c_linear_layer_weight_fixture_export.py`
- `qwen36_c_linear_layer_weight_replay.c`

This validator no longer consumes traced `query/key/value/beta/g` directly. Instead it computes, in one narrow C path:

- input RMSNorm from `layer_input_seq`
- `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta` projections
- sequence conv + SiLU
- `query/key/value` split
- `beta = sigmoid(b)`
- `g = -exp(A_log) * softplus(a + dt_bias)`
- gated DeltaNet recurrence
- gated RMS norm
- `out_proj`
- final mixer residual

Important Qwen-specific detail discovered during this step:

- the input RMSNorm uses `(1 + weight)` rather than plain `weight`
- the gated RMS norm on the DeltaNet output still uses plain `weight`

Without the `1 + weight` rule, the validator diverged immediately at `input_ln`. Once corrected, the full weight-driven path snapped back into tight agreement.

### `blk.0` results

Prompt: short instruction-style `patch_plan`

- `input_ln_rmse=0.00212731`, `input_ln_cosine=0.99999822`
- `qkv_rmse=0.02854908`, `qkv_cosine=0.99991460`
- `conv_post_rmse=0.00312441`, `conv_post_cosine=0.99998783`
- `query_rmse=0.00162056`, `query_cosine=0.99997253`
- `key_rmse=0.00146921`, `key_cosine=0.99988032`
- `value_rmse=0.00417756`, `value_cosine=0.99998876`
- `beta_rmse=0.00091337`, `beta_cosine=0.99999897`
- `g_rmse=0.00337579`, `g_cosine=0.99999999`
- `core_rmse=0.00015893`, `core_cosine=0.99997679`
- `out_in_rmse=0.00020593`, `out_in_cosine=0.99997449`
- `out_proj_rmse=0.00058402`, `out_proj_cosine=0.99913538`
- `mixer_rmse=0.00054499`, `mixer_cosine=0.99877075`
- `residual_after_mixer_rmse=0.00054499`, `residual_after_mixer_cosine=0.99936069`

### `blk.1` results

Prompt: short instruction-style `patch_plan`

- `input_ln_rmse=0.00155498`, `input_ln_cosine=0.99999877`
- `qkv_rmse=0.02739690`, `qkv_cosine=0.99978358`
- `conv_post_rmse=0.00238844`, `conv_post_cosine=0.99997849`
- `query_rmse=0.00226057`, `query_cosine=0.99990863`
- `key_rmse=0.00154048`, `key_cosine=0.99974005`
- `value_rmse=0.00283421`, `value_cosine=0.99998439`
- `beta_rmse=0.00106051`, `beta_cosine=0.99999885`
- `g_rmse=0.00283332`, `g_cosine=0.99999907`
- `core_rmse=0.00009234`, `core_cosine=0.99988307`
- `out_in_rmse=0.00028129`, `out_in_cosine=0.99905305`
- `out_proj_rmse=0.00027697`, `out_proj_cosine=0.99927468`
- `mixer_rmse=0.00027069`, `mixer_cosine=0.99910619`
- `residual_after_mixer_rmse=0.00027069`, `residual_after_mixer_cosine=0.99990261`

Interpretation:

- The hybrid `linear_attention` path is now truly weight-driven in narrow CPU C for at least two consecutive sampled layers.
- The remaining traced shortcut for a full decoder block is now on the MoE half, not the mixer half.
- This is a material shift in project state: the narrow runtime is no longer only replaying traced mixer internals, it is actually computing them from GGUF weights.

Immediate next step:

- Replace the traced MoE half in the decoder-layer validator with a weight-driven routed/shared expert path, then validate a full decoder block in which both the hybrid mixer and the MoE branch are computed directly from GGUF weights.

## Weight-Driven Decoder Block

We then completed the next major replacement step by removing the traced MoE shortcut from the decoder-layer validator.

New artifacts:

- `misc/qwen36_c_decoder_layer_weight_fixture_export.py`
- `qwen36_c_decoder_layer_weight_replay.c`

This validator computes, in one narrow CPU C path:

- input RMSNorm with Qwen’s `(1 + weight)` rule
- hybrid `linear_attention` projections
- conv + SiLU staging
- `query/key/value`, `beta`, `g`
- DeltaNet recurrence
- gated RMS norm
- `out_proj`
- residual add to `residual_after_mixer`
- post-attention RMSNorm with Qwen’s `(1 + weight)` rule
- router logits from `ffn_gate_inp.weight`
- top-k selection and top-k softmax scores
- routed experts
- shared expert
- shared expert gate from `ffn_gate_inp_shexp.weight`
- final residual add to `layer_output`

The fixture still includes only the selected routed expert tensors for the traced top-k experts. So this is not yet the final runtime artifact. But the block semantics are now computed directly from weights instead of consuming traced mixer internals or traced MoE outputs.

### `blk.0` results

Prompt: short instruction-style `patch_plan`

- `input_ln_cosine=0.99999822`
- `qkv_cosine=0.99991460`
- `conv_post_cosine=0.99998783`
- `core_cosine=0.99997679`
- `mixer_cosine=0.99877075`
- `residual_after_mixer_cosine=0.99936069`
- `post_attn_ln_cosine=0.99841774`
- `router_logits_cosine=0.99998500`
- `router_logits_rmse=0.03417409`
- `router_topk_exact=0`
- `router_scores_cosine=0.99906697`
- `router_scores_rmse=0.00605754`
- `shared_gate_pre_rmse=0.00255251`
- `mlp_cosine=0.95626730`
- `mlp_rmse=0.00337608`
- `layer_output_cosine=0.98409028`
- `layer_output_rmse=0.00340938`

### `blk.1` results

Prompt: short instruction-style `patch_plan`

- `input_ln_cosine=0.99999877`
- `qkv_cosine=0.99978358`
- `conv_post_cosine=0.99997849`
- `core_cosine=0.99988307`
- `mixer_cosine=0.99910619`
- `residual_after_mixer_cosine=0.99990261`
- `post_attn_ln_cosine=0.99975312`
- `router_logits_cosine=0.99999163`
- `router_logits_rmse=0.02661570`
- `router_topk_exact=0`
- `router_scores_cosine=0.99983778`
- `router_scores_rmse=0.00276717`
- `shared_gate_pre_rmse=0.00370216`
- `mlp_cosine=0.92737220`
- `mlp_rmse=0.00393013`
- `layer_output_cosine=0.98393710`
- `layer_output_rmse=0.00394468`

Interpretation:

- We now have a full decoder block whose hybrid mixer and MoE branch are both computed from GGUF weights in narrow CPU C.
- The remaining mismatch is no longer “can we compute the block.” It is mostly about:
  - exact routed-expert selection/order plumbing under near-tied router logits
  - replacing selected-expert fixture shortcuts with a runtime path that reads the needed expert tensors directly from the model surface
- The final block outputs remain very close on both sampled layers, which is the strongest signal so far that the project is ready to push into real stacked forward execution rather than staying in validator-only mode.

Immediate next step:

- Join two full weight-driven decoder blocks in one narrow C path, then begin adding the root surfaces (`token_embd`, final norm, output head) needed for a short greedy next-token comparison against the HF or llama.cpp oracle.

## Two-Block Weight-Driven Chain

We then chained the two full weight-driven decoder blocks directly in C:

- `qwen36_c_two_decoder_layers_weight_replay.c`

Results on the short instruction-style `patch_plan` prompt:

- `layer0_output_rmse=0.00340938`
- `layer0_output_cosine=0.98409028`
- `layer1_input_handoff_rmse=0.00340938`
- `layer1_input_handoff_cosine=0.98409028`
- `layer1_output_rmse=0.00577828`
- `layer1_output_cosine=0.97163435`

Interpretation:

- Error does continue to accumulate across the stacked weight-driven blocks, as expected.
- But the path remains coherent enough to justify checking the output surface now rather than demanding tighter internal equality first.

## HF Tail Splice Check

We then performed the first real output-surface sanity check using a splice method:

- `misc/qwen36_splice_two_layer_weight_check.py`

Method:

- run the narrow C two-block chain for `blk.0 -> blk.1`
- dump the computed `blk.1` last-token hidden state
- inject that hidden state back into HF at the output of layer `1`
- let HF run layers `2..39`, final norm, and logits
- compare the final logits and greedy argmax against the pure HF baseline

Results on the `patch_plan` prompt:

- `prompt_tokens=7`
- `logits_cosine=0.99953073`
- `logits_rmse=0.07489589`
- `argmax_equal=True`

Top token comparison:

- baseline argmax: token id `17`, text `"2"`, logit `14.4375`
- patched argmax: token id `17`, text `"2"`, logit `14.375`

Top-8 remained nearly identical, with only small reordering among nearby alternatives.

Interpretation:

- This is the first real evidence that the current weight-driven narrow runtime path is already functionally close enough to preserve the next-token decision on a nontrivial prompt.
- It validates the current project stance: exact internal equality is not required if the output surface remains stable.

Immediate next step:

- move from this splice check toward a true end-to-end narrow forward path by adding root surfaces (`token_embd`, final norm, output head`) and increasing the number of C-owned layers beyond `blk.0 -> blk.1`

## Contract Repair: Hybrid SSM Weight Order and Semantics

We revisited the weight-driven hybrid SSM path after a regression where the narrow boundary validators still looked excellent, but the stacked decoder-layer chain collapsed.

The important result is that the earlier near-perfect validators were correct. The failure came from an incomplete understanding of the GGUF contract for the Qwen3.6 linear-attention block, not from the model being fundamentally hard to replay.

### Confirmed GGUF-to-HF contract fixes

We verified the following against the HF runtime:

- `attn_norm.weight` and `post_attention_norm.weight` already include the `+1` fold used by the HF RMSNorm variant.
  - The narrow runtime must multiply by the GGUF norm tensor directly.
  - It must not apply an extra `1 + weight`.
- The hybrid linear-attention tensors use a nontrivial V-head packing:
  - `attn_qkv.weight` V rows
  - `attn_gate.weight`
  - `ssm_alpha.weight`
  - `ssm_beta.weight`
  - `ssm_dt.bias`
  - `ssm_out.weight` input columns
  - the V-channel portion of `ssm_conv1d.weight`
  all need the same even-heads-then-odd-heads to HF-order remap.
- `ssm_a` is not raw `A_log` in GGUF.
  - HF uses `g = -exp(A_log) * softplus(a + dt_bias)`.
  - GGUF stores the already expanded decay coefficient `-exp(A_log)`, still in the same packed head order.
  - After reordering, GGUF `ssm_a` matched `-exp(HF A_log)` with effectively exact agreement.

### Dedicated note: permutation and semantic-transform entropy

This deserves to be called out explicitly because it is one of the core reasons a narrow runtime rollout can fail even when all tensor names, shapes, dtypes, and high-level architecture guesses look correct.

The key lesson is that a model contract is not just:

- tensor presence
- tensor shape
- tensor dtype
- nominal architectural role

It also includes hidden layout and semantic choices such as:

- row order
- column order
- head packing order
- grouped-channel packing order
- whether a parameter is stored in raw, folded, or pre-expanded form
- whether runtime-side formulas assume the stored tensor is preprocessed already

In this Qwen3.6 case, the narrow path did not initially fail because of math bugs in the recurrent update. It failed because multiple tensors were individually “close enough to look plausible” while still violating the true runtime contract.

That is an important pattern:

- some contract mismatches do not explode immediately
- they can preserve local probes
- they can even preserve partial validators
- but they compound once the real layer schedule is exercised end to end

#### The concrete entropy classes we hit

This round surfaced three distinct classes of contract entropy.

1. Packed-order entropy

The tensor is numerically the same object as HF uses, but the storage order differs.

Observed examples:

- `attn_qkv.weight` V rows
- `attn_gate.weight`
- `ssm_alpha.weight`
- `ssm_beta.weight`
- `ssm_dt.bias`
- `ssm_out.weight` input columns
- V-channel slice of `ssm_conv1d.weight`

Shared pattern:

- GGUF stored V-head-facing surfaces in even-heads-first, then odd-heads
- HF runtime expected canonical head order

This is the narrow-runtime danger zone because:

- direct tensor inspection still looks “reasonable”
- local slices often have the right scales
- shapes are identical
- a naive validator can still pass if it never crosses the packed boundary

2. Folded-parameter entropy

The stored tensor is mathematically related to the HF parameter, but not identical.

Observed example:

- RMSNorm weights in GGUF were already folded so the runtime should multiply by the tensor directly

HF-side mental model:

- normalize then multiply by `(1 + weight)`

GGUF-side contract:

- normalize then multiply by stored tensor directly

This kind of mismatch is subtle because:

- values look similar
- nothing appears permuted
- the error is systematic rather than catastrophic

3. Semantic-transform entropy

The stored tensor is not the raw parameter at all. It already represents a runtime-side transformed quantity.

Observed example:

- `ssm_a` in GGUF is not `A_log`
- it is already `-exp(A_log)`

That is the most dangerous category because the runtime may then apply the transform a second time and produce nonsense that still looks finite and stable.

That is exactly what happened here:

- the early path interpreted GGUF `ssm_a` as raw `A_log`
- the replay applied `-exp(...)` again
- the resulting `g` term was wrong even though the surrounding probes looked strong

#### Why this matters for a general narrow-rollout process

This suggests a more coherent language for “architectural contract” than just “the C baseline says these tensors exist.”

A useful narrow-runtime contract vocabulary should include at least these fields for every bound tensor family:

- identity:
  - what conceptual role the tensor plays
- shape contract:
  - logical dimensions as consumed by the runtime
- storage contract:
  - row-major or column-major expectations
  - transposition expectations
- packing contract:
  - head order
  - expert order
  - grouped-channel order
  - fused-subtensor order
- semantic contract:
  - raw parameter
  - folded parameter
  - pre-expanded runtime coefficient
  - quant-domain artifact
- execution boundary:
  - which intermediate should be compared immediately after consuming this tensor

That is the beginning of a real vocabulary for rolling narrow pipelines on arbitrary models without forcing everything into DS4’s DeepSeek-specific terms.

#### Practical rollout method implied by this finding

For future narrow-model bring-up, the process should be:

1. bind and validate names/shapes/dtypes
2. identify execution boundaries that isolate each tensor family
3. compare GGUF and HF at those boundaries
4. test for storage transforms:
   - transpose
   - row reorder
   - column reorder
   - grouped reorder
5. test for semantic transforms:
   - folded bias or norm
   - exponentiated or logged coefficients
   - pre-normalized runtime constants
6. only then trust stacked-layer failure signals

In other words:

- stacked-layer divergence is often not “the math is wrong”
- it is frequently “one or two tensor families have the wrong storage or semantic interpretation”

That is exactly what happened here.

### What we patched

- Exporters now reorder all affected hybrid tensors into HF runtime order before writing weight-driven fixtures:
  - `misc/qwen36_c_decoder_layer_weight_fixture_export.py`
  - `misc/qwen36_c_linear_layer_weight_fixture_export.py`
  - `misc/qwen36_c_linear_core_fixture_export.py`
  - `misc/qwen36_c_linear_fixture_export.py`
- The C weight-driven replayers now consume `ssm_a` as the GGUF-native decay coefficient rather than re-exponentiating it:
  - `qwen36_c_linear_core_replay.c`
  - `qwen36_c_linear_layer_weight_replay.c`
  - `qwen36_c_decoder_layer_weight_replay.c`
  - `qwen36_c_two_decoder_layers_weight_replay.c`
  - `qwen36_c_decoder_chain_weight_replay.c`

### Revalidated narrow results

After the fixes:

- DeltaNet core boundary returned to the trusted regime:
  - `query_cosine=0.99999937`
  - `key_cosine=0.99999837`
  - `value_cosine=0.99999986`
  - `beta_cosine=0.99999904`
  - `g_cosine=1.00000000`
  - `core_cosine=0.99999530`
- Full weight-driven hybrid layer replay for `blk.0` also returned to the trusted regime:
  - `qkv_cosine=0.99999641`
  - `z_cosine=0.99999619`
  - `a_cosine=0.99999547`
  - `b_cosine=0.99999654`
  - `conv_post_cosine=0.99999753`
  - `value_cosine=0.99999744`
  - `core_cosine=0.99999322`
  - `out_proj_cosine=0.99992581`
  - `mixer_cosine=0.99989466`
  - `residual_after_mixer_cosine=0.99994525`

### Three-layer chain proof

We regenerated corrected decoder fixtures for `blk.0`, `blk.1`, and `blk.2` and reran the generic weight-driven chain:

- `layer0_output_cosine=0.99994644`
- `layer1_output_cosine=0.99993941`
- `layer2_output_cosine=0.99995261`
- full-sequence:
  - `layer0_output_seq_cosine=0.99995271`
  - `layer1_output_seq_cosine=0.99995966`
  - `layer2_output_seq_cosine=0.99996838`

Interpretation:

- The narrow Qwen-native hybrid SSM contract is now materially understood rather than hand-waved.
- The earlier low-cosine chain failure was caused by multiple stacked GGUF semantic mismatches:
  - packed V-head ordering
  - GGUF-folded RMSNorm weights
  - GGUF storing `ssm_a` as `-exp(A_log)`
- With those fixed, the narrow CPU C path is once again behaving like a legitimate runtime foundation, not just a set of disconnected validators.

Immediate next step:

- keep this contract and move outward:
  - either extend the chain deeper and re-run the HF splice/output check
  - or start the true narrow end-to-end text path using `token_embd`, the corrected stacked blocks, final norm, and `output.weight`

## Three-Layer Output-Surface Splice Check

After repairing the hybrid SSM contract, we repeated the HF splice check with a corrected three-layer narrow chain:

- `blk.0`
- `blk.1`
- `blk.2`

Using:

- narrow C chain: `qwen36-c-decoder-chain-weight`
- splice layer: `2`
- prompt: `misc/qwen36_oracle_prompts/patch_plan.txt`

Results:

- `prompt_tokens=7`
- `logits_cosine=0.99995496`
- `logits_rmse=0.02399506`
- `argmax_equal=True`

Top-token comparison:

- baseline argmax:
  - token id `17`
  - text `"2"`
  - logit `14.437500`
- patched argmax:
  - token id `17`
  - text `"2"`
  - logit `14.437500`

Top-8 behavior remained extremely tight. The only differences were minor reordering among near-tied alternatives:

- baseline and patched both kept:
  - `17 -> "2"`
  - `16 -> "1"`
  - `760 -> "The"`
  - `256 -> "  "`
  - `3886 -> "Create"`
  - `71093 -> "```"`
  - `12 -> "-"`
  - `64 -> "a"`

Interpretation:

- This is stronger than the earlier two-layer splice result.
- The corrected narrow three-layer prefix now preserves:
  - final hidden-state geometry
  - output-surface logits
  - greedy next-token choice
- At this point, the remaining gap is no longer “does the corrected narrow prefix behave like HF?”
- The remaining gap is simply ownership:
  - replacing the HF front and back edges with narrow runtime components

What this proves:

- the repaired contract is good enough to survive a real output-surface test
- the narrow C path is already functionally equivalent enough for first-token greedy decisions on this prompt
- the next meaningful milestone is not another internal validator, but a true narrow-owned first-token path:
  - token embedding lookup
  - corrected stacked decoder blocks
  - final norm
  - output head scoring

## C-Owned Front Edge: token_embd + blk.0..2

We then moved the ownership boundary forward.

Previously, the splice check still relied on HF-owned prompt embeddings and only replaced the hidden state after the narrow-owned decoder prefix.

The new step was:

- own prompt token embedding lookup directly from GGUF in C
- feed that live embedding sequence into the corrected narrow C chain for:
  - `blk.0`
  - `blk.1`
  - `blk.2`
- splice the resulting hidden state back into HF at layer `2`
- compare final logits again

### New artifacts

- `misc/qwen36_c_prefix_fixture_export.py`
  - exports prompt token IDs and HF `blk.0` input-sequence reference
- `qwen36_c_prefix_q8_chain_replay.c`
  - reads `token_embd.weight` from the Q8 GGUF
  - decodes prompt embedding rows in C
  - runs the corrected narrow `blk.0..2` chain
  - dumps the resulting hidden state for splice checks
- `misc/qwen36_splice_prefix_q8_check.py`
  - performs the output-surface splice check using the C-owned front edge

### Prefix ownership results

On the same `patch_plan` prompt:

- embedding sequence from live GGUF rows:
  - `embedding_rmse=0.00008180`
  - `embedding_cosine=0.99995997`
- three-layer prefix output:
  - `layer0_output_seq_cosine=0.99994324`
  - `layer1_output_seq_cosine=0.99995364`
  - `layer2_output_seq_cosine=0.99996593`

### Output-surface result with C-owned front edge

Using the C-owned prefix hidden state and splicing it back into HF after `blk.2`:

- `prompt_tokens=7`
- `logits_cosine=0.99997935`
- `logits_rmse=0.01619956`
- `argmax_equal=True`

Interpretation:

- this is stronger than the previous splice result
- we no longer depend on HF for prompt embeddings at the front edge
- the narrow runtime now owns:
  - token IDs to embeddings
  - the corrected `blk.0..2` execution
  - the hidden state handed to the rest of the model

What this means:

- the project has crossed from “faithful layer replay” into a real partial runtime
- the next step is no longer theoretical
- we can now build the first true narrow-owned greedy path on top of:
  - GGUF token embeddings
  - corrected narrow prefix blocks
  - final norm / output scoring

The remaining missing ownership is mainly:

- either own more layers
- or keep using HF as a temporary tail while we add a narrow root-scoring loop and then progressively extend the owned prefix depth

## Hardware Envelope Heuristic

This section is not a benchmark claim. It is a planning heuristic for what a DS4-style narrow runtime could plausibly make viable if it achieves both:

- strong size reduction versus naive `Q8`
- partial residency, where only the load-bearing path and a hot sparse cache must remain live in fast memory

The useful shift in mindset is:

- fit is no longer `whole_model_in_vram`
- fit becomes `resident_path + hot_sparse_cache + kv_and_scratch`

For a narrow sparse runtime, a rough envelope is:

- `effective_fast_mem ~= r * c * naive_q8_model_size + cache/scratch`

Where:

- `c` = final compressed size as a fraction of naive `Q8`
- `r` = resident fraction of the compressed model that must stay in fast memory

If we assume an aggressive but plausible DS4-style regime:

- `c ~= 0.30`
- `r ~= 0.25` to `0.45`

then fast-memory residency is only about:

- `0.075x` to `0.135x` of naive `Q8` model size

### Why 128 GB system RAM matters

On a discrete GPU system, large RAM changes the problem materially.

With `24 GB VRAM + 128 GB RAM`:

- VRAM can hold the protected resident path, scratch, and a hot expert cache
- system RAM can hold much larger cold sparse weights before SSD is needed
- SSD becomes a second-stage overflow rather than the first streaming tier

That is meaningfully better than `24 GB VRAM` by itself.

It does **not** make the system equivalent to Apple unified memory, because:

- host-to-GPU transfer still crosses PCIe
- SSD misses still cost more than on-package memory traffic

But it does move the practical target class upward by a lot.

### Small planning table

| Device tier | Fast memory | Slow memory tier | Narrow-runtime implication | Plausible sparse target class |
|---|---:|---:|---|---|
| `7900 XTX + 128 GB RAM` | `24 GB VRAM` | `128 GB RAM`, then SSD | best non-unified-memory setup in this project so far; RAM can act as a large expert backing store before SSD | `Qwen3.6-35B-A3B` is conservative; `80B-120B total` sparse MoE looks realistic; `150B-250B total` sparse MoE is an ambitious stretch if active path and streaming locality are favorable |
| `Mac mini 16 GB` | `16 GB unified` | SSD | much friendlier memory hierarchy than dGPU+PCIe, but much less total fast memory | smaller sparse models or very aggressive mixed quant; `Qwen3.6-35B-A3B` would require an unusually strong resident/non-resident split and very disciplined cache behavior |
| `Strong phone` | roughly `8-16 GB shared` | flash storage | storage fit is not the main issue; bandwidth, thermals, and sustained compute dominate | mobile-first tiny dense models or genuinely mobile-scale MoE only, not frontier coding MoE |

### Concrete implication for this project

For your machine specifically:

- `Qwen3.6-35B-A3B` is comfortably inside the class that a successful narrow runtime should target
- `GLM-4.5-Air` becomes a serious candidate rather than a fantasy target
- `DeepSeek-V3` class models still look too large for this hardware tier unless you accept a much steeper latency slope and a much more disk-heavy design

So the current Qwen work is not just a one-off model project.
It is also the right proving ground for a general desktop sparse-runtime strategy:

- protect the load-bearing path in VRAM
- use large host RAM as the primary cold sparse store
- use SSD as the final overflow tier

## ROCm Bring-Up Milestone

We crossed the first real GPU substrate milestone on the local `7900 XTX` machine.

### What succeeded

- The DS4 ROCm backend now compiles locally under WSL with:
  - vendored `rocWMMA` headers
  - `HIPBLAS_V2` enabled
  - CUDA-style warp-intrinsic compatibility shims
- The narrow Qwen scaffold binary links and runs:
  - `qwen36-gpu-oracle-scaffold`
- Runtime bring-up succeeded:
  - `ds4: ROCm backend initialized on AMD Radeon RX 7900 XTX (sm_110)`
  - `backend_gpu_init: ok`

### What this proves

- The current blocker is no longer ROCm packaging or link-time backend failure.
- The DS4 GPU substrate is now live enough on this machine to start the real Qwen-native GPU oracle path.
- The Qwen runtime contract also surfaced correctly through that backend:
  - `contract: Qwen3.6-35B-A3B Q8_0`
  - `block_count: 40`
  - `expert_topk: 8`
  - `full_attention_interval: 4`

### Immediate consequence

The next milestone is no longer generic backend repair.

It is:

- HIP-owned `token_embd + blk.0`
- read back the resulting hidden state
- splice into the HF tail
- compare next-token greedy parity

If that works, we can extend from:

- `blk.0`
- to `blk.0..2`
- to `blk.0..3`

without returning to generic CPU-only validation as the main proving method.

### First post-bring-up oracle artifact

We added the first practical GPU oracle artifact after ROCm bring-up:

- `qwen36-gpu-blk0-ffn-q8-oracle`

Scope:

- consumes an existing `blk.0` decoder-layer weight fixture
- optionally checks Q8 token embeddings against the existing prefix fixture
- executes the reusable `blk.0` FFN closure on the DS4 GPU substrate:
  - post-attention RMSNorm
  - router logits/top-k
  - shared expert gate/up/down
  - routed MoE gate/up/down
- combines the final per-token shared gate scalar on CPU for now
- dumps a full `blk.0` output sequence for HF-tail splicing

This is intentionally not the final custom-kernel answer.

The hybrid SSM mixer core is still CPU-owned in this first cut.

What this gives us anyway:

- a real GPU-owned validation slice around the load-bearing MoE/FFN closure
- a sequence output that can be spliced into the HF tail at `layer 1`
- a clean isolation boundary for the next custom HIP work:
  - the hybrid SSM core inside `blk.0`

## DeepSeek Fixed-6 Trap And Qwen Escape Path

The first live GPU oracle failure was not a generic ROCm failure.

It was a DS4 architectural contract mismatch:

- DS4 ROCm router/MoE batch helpers are still hard-shaped for DeepSeek's resident contract
- that path hardcodes `n_expert_used = 6`
- Qwen3.6-35B-A3B requires `topk = 8`

So the first GPU oracle died exactly where it should:

- ROCm init worked
- Q8 embedding probe worked
- router logits matmul worked
- DeepSeek-specific batch top-k rejected the Qwen contract

This is important because it clarifies the boundary.

We do **not** need more generic backend debugging first.

We need a Qwen-native orchestration layer over the DS4 substrate.

### Immediate design correction

Instead of cloning the full DS4 routed-MoE launch stack immediately, the first Qwen-local GPU oracle now takes the narrower path:

- keep DS4 public q8 matmul / norm / swiglu primitives
- stop calling DeepSeek-specific `router_select_batch` and `routed_moe_batch`
- do Qwen router selection on the host:
  - GPU router logits matmul
  - CPU top-k=8 softmax
- execute routed experts as Qwen-local host orchestration:
  - per-expert GPU gate/up q8 pair matmul
  - GPU swiglu
  - GPU down q8 matmul
  - CPU weighted accumulation across selected experts

This is still narrow.

It is also the right next proof:

- it removes the DS4 fixed-6 assumption
- it keeps most heavy FFN math on GPU
- it avoids inventing a generic meta-backend
- it validates the Qwen MoE contract before we spend time on a custom `topk=8` HIP batch kernel family

### Practical consequence

The next expected validation sequence is:

1. `qwen36-gpu-blk0-ffn-q8-oracle` succeeds end to end with the Qwen-local router/MoE orchestration.
2. `misc/qwen36_gpu_blk0_ffn_splice_check.py` shows the HF-tail logits remain close.
3. Then we decide whether to:
   - keep extending Qwen-local host orchestration for the early oracle path, or
   - replace the host-routed expert loop with true `qwen36_*` HIP top-k=8 batch kernels.

## First GPU Tail-Splice Result

We now have the first real HF-tail splice result from a GPU-owned Qwen block.

Prompt:

- `misc/qwen36_oracle_prompts/patch_plan.txt`

Oracle result for `blk.0` FFN closure:

- `router_topk_exact: true`
- `router_scores_rmse: 0.00066103`
- `blk0_ffn_seq_cosine: 0.99644166`
- `blk0_ffn_last_cosine: 0.99416924`

HF tail splice result:

- `prompt_tokens: 7`
- `logits_cosine: 0.99747349`
- `logits_rmse: 0.19523133`
- `argmax_equal: True`
- baseline next token: `2`
- patched next token: `2`

What this means:

- the Qwen-native GPU FFN closure is already functionally interchangeable enough for next-token continuation
- the DS4 fixed-6 MoE contract is no longer blocking a Qwen-native narrow path
- the current Qwen-local host orchestration over DS4 GPU primitives is viable as an oracle and bootstrap runtime path

What it does **not** mean yet:

- we do not yet own the hybrid SSM mixer on GPU
- we do not yet know long-prefix or multi-token drift from this single-block splice alone
- we do not yet know whether the current host-routed expert loop is fast enough for practical decode

But it is enough to justify the next rung:

- own `blk.0..2` on GPU in the same style
- then splice into HF at `blk.3`
- then compare greedy continuation over several tokens

## Bridged `blk.0 -> blk.1..2 -> blk.3` Harness

We now have the bridge needed to test a longer owned prefix without waiting for full GPU ownership of the hybrid SSM layers.

New handoff rule:

- `qwen36-gpu-blk0-ffn-q8-oracle` owns `blk.0` FFN closure and writes the full hidden sequence
- `qwen36-c-prefix-q8-chain-dynamic` can now optionally seed from `prefix.input_seq_ref` instead of re-decoding token embeddings
- that lets us hand the GPU-owned `blk.0` sequence directly into owned `blk.1..2`
- `qwen36-c-full-layer-q8-dynamic` can then own `blk.3`
- HF owns the remaining tail

This is not the final all-GPU narrow runtime.

It is the shortest path to the next practical question:

- does a longer owned prefix still preserve greedy continuation over several tokens?

## First Bridged Multi-Token Greedy Result

The bridged harness now passed a real multi-token greedy continuation check.

Path:

- step 0:
  - GPU-owned `blk.0` FFN closure
  - owned `blk.1..2`
  - owned `blk.3`
  - HF tail
- later steps:
  - fully dynamic owned `blk.0..2`
  - owned `blk.3`
  - HF tail

Prompt:

- `misc/qwen36_oracle_prompts/patch_plan.txt`

Result:

- generated text: `2. \n3`
- full text: `Implementation plan:\n1. \n2. \n3`
- greedy token match vs HF: `4 / 4`

Per-step:

- step 0: `2` == HF `2`
- step 1: `.` == HF `.`
- step 2: ` \n` == HF ` \n`
- step 3: `3` == HF `3`

What this proves:

- the step-0 GPU bridge is not just producing close internals; it preserves actual greedy continuation
- the owned prefix through `blk.3` is behaviorally stable across multiple decode steps on this prompt
- we now have a credible narrow-runtime validation ladder:
  - single-block splice
  - step-0 GPU bridge
  - bridged multi-token owned-prefix continuation

What remains open:

- longer prompts
- harder prompts
- throughput
- true dynamic GPU ownership of hybrid `blk.0`
- eventual replacement of the host-routed expert loop with a real `qwen36_*` HIP MoE path

## Dynamic `blk.0` GPU Cut

The next useful GPU step is now clear enough to name precisely.

Not the final custom-kernel target:

- do **not** wait for a full end-to-end HIP DeltaNet implementation before moving forward

Useful intermediate target:

- a true dynamic `blk.0` runner with this split:
  - GPU:
    - token embedding
    - input RMSNorm
    - q8 `qkv` projection
    - q8 `z` projection
    - f32 `a` / `b` projections
    - q8 `out_proj`
    - post-attention RMSNorm
    - router logits
    - shared expert
    - routed experts
  - CPU:
    - conv1d over projected `qkv`
    - Q/K/V expansion
    - `beta` / `g`
    - DeltaNet state scan
    - per-head RMS + `z` gate to `out_in`

Why this cut matters:

- it makes `blk.0` truly dynamic across decode steps
- it moves most heavy matmul work to GPU immediately
- it removes the current step-0-only bridge limitation
- it avoids blocking on the hardest custom HIP kernel family

Repository state:

- a compile-clean scaffold target now exists:
  - `qwen36-gpu-blk0-dynamic-q8-oracle`
- current status of that binary:
  - interface and contract stub only
  - dynamic execution path not implemented yet

So the next coding pass is specific:

1. implement GPU embeddings + input norm + `qkv/z/a/b` projection reads
2. run the CPU DeltaNet scan on those projected tensors
3. push `out_in` back to GPU for q8 `out_proj`
4. reuse the existing GPU FFN closure pattern on the resulting residual stream

## `attn_qkv` Audit and GPU Host-Bridge Fix

We added a dedicated narrow audit binary:

- `qwen36-q8-attn-qkv-audit`

Purpose:

- load `blk.0.attn_qkv.weight` directly from the Q8 GGUF
- dequantize the raw Q8 rows
- compare that raw matrix to the trusted CPU-side fixture
- apply the known Qwen hybrid V-row remap
- compare both matrix form and projected outputs against the HF-derived trace

Run:

```bash
./qwen36-q8-attn-qkv-audit MODEL.gguf /tmp/qwen36_decoder_layer_weight_blk0_patch_plan_v8.bin
```

Result on the current `blk.0` fixture:

- `matrix_raw_vs_fixture_cosine: 0.54207308`
- `matrix_reordered_vs_fixture_cosine: 1.00000000`
- `qkv_raw_vs_trace_cosine: 0.47251920`
- `qkv_reordered_vs_trace_cosine: 0.99999684`

Meaning:

- the GGUF tensor itself is now fully explained
- raw `attn_qkv.weight` is in GG-order
- after the already-known Qwen V-row remap, it matches the trusted CPU contract exactly
- the earlier GPU failure was not caused by an unknown extra tensor permutation in GGUF

This led directly to a second finding:

- the GPU dynamic oracle helper that split `attn_qkv` into Q/K/V sub-projections was writing results in the wrong host layout

Bug:

- the helper emitted:
  - all-token `Q`
  - then all-token `K`
  - then all-token `V`
- but every trusted trace and fixture in this project uses per-token packed layout:
  - token `t`: `[Q | K | V]`

Why this matters:

- it can make a correct projection look catastrophically wrong
- it falsely suggests lower-level GPU math is broken when the real bug is host packing

Fix:

- `run_gpu_q8_qkv_split_rowwise()` now gathers `Q`, `K`, and `V` into temporary buffers and repacks them per token before downstream comparison

Updated interpretation:

- the Qwen `attn_qkv` GGUF contract is confirmed
- the next GPU debug pass should start from the new host-fixed bridge
- only after re-running that path should we judge whether any remaining drift is in:
  - the GPU q8 matmul implementation
  - later hybrid-block math
  - or another host-side ordering mistake

## Dynamic `blk.0` GPU Bridge: Tight End-to-End Result

After fixing both host-side `attn_qkv` packing mistakes:

- split Q/K/V projection output is now repacked per token as `[Q | K | V]`
- the V-head reorder inside `qkv` now respects token-major row stride instead of treating V as one flat contiguous slab

the dynamic `blk.0` oracle moved from obviously wrong to numerically tight.

Observed metrics on the current `patch_plan` fixture:

- `qkv_reordered_cosine: 0.99998415`
- `qkv_gpu_vs_cpu_contract_cosine: 0.99998724`
- `q_proj_gpu_vs_cpu_contract_cosine: 0.99999577`
- `k_proj_gpu_vs_cpu_contract_cosine: 0.99998665`
- `v_proj_gpu_vs_cpu_contract_cosine: 0.99998522`
- `conv_post_cosine: 0.99999607`
- `query_cosine: 0.99999374`
- `key_cosine: 0.99997461`
- `value_cosine: 0.99999613`
- `core_cosine: 0.99999219`
- `out_in_cosine: 0.99999553`
- `out_proj_cosine: 0.99992180`
- `residual_after_mixer_seq_cosine: 0.99994427`
- `blk0_dynamic_seq_cosine: 0.99639881`
- `blk0_dynamic_last_cosine: 0.99409008`

Interpretation:

- the owned dynamic `blk.0` path is now in the same regime as the earlier tight CPU bridges
- this is no longer a “shape seems right” milestone
- it is strong enough to justify an immediate splice-oracle pass into the HF tail

One remaining imperfection:

- `router_topk_exact: false`
- `router_scores_rmse: 0.00142371`

Current reading of that mismatch:

- it is small compared to the earlier structural failures
- it likely reflects near-tie or ordering sensitivity rather than a broken router surface
- the right next validation is not more internal probing first
- the right next validation is logits / greedy-token behavior after splicing this GPU-owned `blk.0` output into the HF tail

## Dynamic `blk.0` GPU Splice Oracle

We ran the dynamic GPU `blk.0` bridge through the existing HF-tail splice harness:

- C-owned:
  - token embedding
  - hybrid `blk.0` path through post-MoE output
- HF-owned:
  - `blk.1..end`
  - final norm
  - logits

Result on `patch_plan.txt`:

- `prompt_tokens: 7`
- `logits_cosine: 0.99908634`
- `logits_rmse: 0.11171936`
- `argmax_equal: True`
- baseline argmax:
  - token id `17`
  - text `"2"`
  - logit `14.4375`
- patched argmax:
  - token id `17`
  - text `"2"`
  - logit `14.1875`

Interpretation:

- the dynamic GPU-owned `blk.0` path now preserves the actual next-token decision through the HF tail
- this is the first proper behavioral proof for the GPU dynamic bridge
- it upgrades the current status from:
  - “numerically tight internal bridge”
  to:
  - “behaviorally valid owned prefix stage”

Practical consequence:

- `blk.0` no longer needs to be treated as a fragile research stub
- the next shortest path is to combine:
  - GPU-owned `blk.0`
  - already-proven CPU-owned `blk.1..3`
  - existing greedy decode harness

That is the right bridge before attempting broader GPU ownership or longer prompt evaluations.

## Owned Prefix: GPU `blk.0` + CPU `blk.1..3`

We then promoted the splice-oracle result into a true owned-prefix greedy bridge:

- GPU-owned:
  - dynamic `blk.0`
- CPU-owned:
  - `blk.1`
  - `blk.2`
  - full `blk.3`
- HF-owned:
  - remaining tail
  - logits oracle

Prompt:

- `patch_plan.txt`

Observed result:

- generated text: `2. \n3`
- full text: `Implementation plan:\n1. \n2. \n3`
- step count: `4`
- greedy token match vs HF: `4 / 4`

Per-step:

- step `0`: `2` == HF `2`
- step `1`: `.` == HF `.`
- step `2`: ` \n` == HF ` \n`
- step `3`: `3` == HF `3`

Interpretation:

- the owned-prefix architecture now survives growing decode, not just a single splice
- GPU `blk.0` is promoted from an isolated validated bridge to a real prefix stage
- the narrow runtime now has a behaviorally valid heterogeneous prefix:
  - GPU early hybrid block
  - CPU-following owned blocks
  - HF oracle tail

Immediate consequence:

- the next useful work is longer prompt behavior, not more microscopic `blk.0` doubt
- router exactness remains a cleanup item, but it is not blocking owned-prefix continuation on the tested prompt

## GPU `blk.3` First Cut

We added a new HIP binary:

- `qwen36-gpu-full-layer-q8-dynamic`

Scope of this first cut:

- input: owned hidden-state sequence from `blk.0..2`
- GPU-owned:
  - `attn_norm`
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - post-attention MoE/shared-expert FFN matmuls
- CPU-owned:
  - Qwen full-attention host shaping
  - RoPE application
  - causal softmax attention loop
  - router top-k / weighted expert accumulation

This is intentionally not the final narrow runtime. It is the smallest honest attempt to remove the old CPU full-layer binary from the prefix harness without pretending the attention contract was fully solved.

### Smoke Test Result

We replaced CPU `blk.3` in the hybrid harness:

- GPU-owned:
  - `blk.0`
  - `blk.1`
  - `blk.2`
  - new GPU `blk.3`
- HF tail:
  - unchanged oracle

Observed on `patch_plan.txt`:

- first owned next-token: `"1"`
- HF baseline next-token: `"2"`
- result: failed on the first token

Interpretation:

- the new `blk.3` path is wired correctly enough to:
  - run on ROCm
  - hand off a full hidden-state sequence to the HF tail
- but it is not yet numerically close enough to preserve the greedy decision

### Direct CPU-vs-GPU `blk.3` Check

To isolate whether this was only a harness issue, we ran both full-layer binaries on the same owned `blk.0..2` sequence:

- CPU:
  - `qwen36-c-full-layer-q8-dynamic`
- GPU first cut:
  - `qwen36-gpu-full-layer-q8-dynamic`

Observed:

- full-sequence RMSE: `0.23291785`
- full-sequence cosine: `0.75318100`
- last-token RMSE: `0.14942040`
- last-token cosine: `0.72152792`

Interpretation:

- this is not just an HF splice artifact
- the new GPU `blk.3` implementation diverges materially from the existing CPU full-layer runner
- the next debugging pass should focus on the full-attention contract itself, especially:
  - Q/K/V projection interpretation
  - per-head norm semantics
  - RoPE application order
  - grouped-query attention head mapping
  - post-attention gate placement

Current conclusion:

- `GPU blk.0..2` remains the validated owned prefix
- `GPU blk.3` exists and runs, but is not yet validated

## `blk.3` Isolation Against Exact HF Input

The earlier `GPU blk.0..3` smoke failure was still ambiguous because `blk.3` was being fed an owned `blk.0..2` sequence. That means the first-token mismatch could have come from:

- a bad `blk.3` contract
- accumulated upstream drift from `blk.0..2`
- or both

So we isolated `blk.3` with the exact HF `layer_input_seq` from a new full-attention trace export.

### New Artifact

- `misc/qwen36_full_attn_trace_export.py`

It exports `blk.3` full-attention internals including:

- `q_proj_full`
- split `query_raw` and `gate_raw`
- `q_norm`
- `k_proj_raw`
- `k_norm`
- `v_proj_raw`
- transposed Q/K/V
- RoPE’d Q/K
- attention output before and after sigmoid gating
- `o_proj_out`
- post-attention residual
- downstream router/shared-expert/layer-output references

### Exact-HF-Input Comparison

Using `trace['layer_input_seq']` as the input to both full-layer runners:

- CPU full-layer:
  - `qwen36-c-full-layer-q8-dynamic`
- GPU first cut:
  - `qwen36-gpu-full-layer-q8-dynamic`

Observed:

- CPU vs HF:
  - RMSE: `0.03766897`
  - cosine: `0.76455170`
- GPU vs HF:
  - RMSE: `0.00365469`
  - cosine: `0.99417548`
- GPU vs CPU:
  - RMSE: `0.03785883`
  - cosine: `0.76274964`

Interpretation:

- the new GPU `blk.3` path is dramatically closer to HF than the existing CPU full-layer runner when tested on the exact same input
- this changes the diagnosis substantially:
  - `blk.3` is not a failed path
  - the existing CPU full-layer runner is actually the worse approximation on this isolated test

### Behavioral Oracle Check

We then spliced the GPU `blk.3` output, produced from the exact HF `blk.3` input, back into the HF tail:

- logits cosine: `0.99782462`
- logits RMSE: `0.17463283`
- argmax equal: `True`
- baseline argmax:
  - token id `17`
  - text `"2"`
- patched argmax:
  - token id `17`
  - text `"2"`

Interpretation:

- isolated `GPU blk.3` is now behaviorally valid at the next-token level
- the remaining `GPU blk.0..3` end-to-end mismatch should be treated primarily as an upstream composition problem:
  - `blk.0..2` drift
  - cross-layer accumulation
  - or interaction between owned early blocks and the exact `blk.3` contract

Updated conclusion:

- `GPU blk.3` is not the blocker
- the next useful work is to tighten `blk.1..2` composition and then re-run the owned-prefix bridge

## Root-Cause Isolation Workflow

At this point, broad end-to-end retries are no longer the right tool. We already have strong evidence for three distinct facts:

- CPU-owned `blk.0..2` can compose well enough to preserve greedy behavior on the small oracle prompt.
- `GPU blk.0` is behaviorally valid both as a local block and as the first stage of an owned prefix.
- exact-HF-input `GPU blk.3` is behaviorally valid and materially closer to HF than the older standalone CPU `blk.3` approximation.

That means the remaining question is narrower:

- where is the first harmful composition boundary?

To keep that question precise, we now split validation into three tool classes.

### 1. Exact-Input Isolation

Artifact:

- `misc/qwen36_splice_hidden_check.py`

Purpose:

- take any raw hidden dump, either `[hidden]` or `[seq, hidden]`
- splice it back into the HF tail at a chosen layer
- measure:
  - logits cosine
  - logits RMSE
  - argmax equality
  - top-2 logit margin drift

This is the correct tool when we want to answer:

- if this layer output were exact, would the tail still behave correctly?
- is the local block itself acceptable independent of upstream drift?

### 2. Composition Ladder

Artifact:

- `misc/qwen36_composition_ladder_check.py`

Purpose:

- run the owned prefix at increasing depths on one prompt:
  - `blk0`
  - `blk01`
  - `blk012`
  - optionally `blk0123`
- splice each owned sequence back into the HF tail
- classify each stage as:
  - `PASS_BEHAVIORAL`
  - `PASS_NEAR`
  - `FAIL_TIE_SENSITIVE`
  - `FAIL_COMPOSITION`

This is the correct tool when we want to answer:

- what is the first depth where behavior or margin meaningfully degrades?
- does the break happen before `blk.3`, at `blk.3`, or only after multiple approximations accumulate?

### 3. Behavioral Oracle

Existing artifact:

- `misc/qwen36_hybrid_prefix_tail_greedy.py`

Purpose:

- own a prefix with C and/or GPU blocks
- let HF own the tail
- compare actual greedy continuation tokens over multiple decode steps

This is the final behavioral gate for a proposed owned-prefix configuration. It should not be the first debugging tool anymore; it should only be used after exact-input and composition-ladder checks say the local surfaces are credible.

### Practical Rule Going Forward

When a new owned block fails, we should avoid jumping straight into long greedy runs.

Instead:

1. prove the block on exact HF input
2. splice that exact-output hidden state into the HF tail
3. run the composition ladder to find the first bad boundary
4. only then promote the owned prefix into multi-token greedy checks

This reduces repeated work and keeps the diagnosis tied to one specific failure surface at a time.

### First Composition Ladder Result

We ran the new ladder on `patch_plan.txt` with this owned path:

- `GPU blk.0`
- `CPU blk.1`
- `CPU blk.2`
- optional `GPU blk.3`

Observed:

- `blk0`
  - logits cosine: `0.99908634`
  - logits RMSE: `0.11171936`
  - argmax equal: `True`
- `blk01`
  - logits cosine: `0.99781991`
  - logits RMSE: `0.18405716`
  - argmax equal: `True`
- `blk012`
  - logits cosine: `0.99879205`
  - logits RMSE: `0.12505564`
  - argmax equal: `True`
- `blk0123`
  - logits cosine: `0.99463499`
  - logits RMSE: `0.28297278`
  - argmax equal: `True`

Top-2 margin comparison:

- HF baseline margin: `2.125`
- `blk0` patched margin: `1.6875`
- `blk01` patched margin: `1.6875`
- `blk012` patched margin: `1.8125`
- `blk0123` patched margin: `1.75`

Interpretation:

- the first noticeable numerical weakening shows up only after adding `blk.3`
- however, on this prompt it is still not a behavioral failure:
  - argmax remains exact at every depth
  - the top-2 margin remains comfortably positive
- therefore the current ladder result should be read as:
  - `blk0..2` composition is solid on this oracle prompt
  - adding `blk.3` introduces the largest drift seen so far
  - but `blk.3` is still within a plausibly usable range for next-token behavior on this prompt

This is an important distinction:

- `FAIL_COMPOSITION` in the ladder output is only a heuristic label from a cosine threshold
- it does **not** mean the owned `blk0123` path is behaviorally broken
- it means `blk0123` is the first depth that deserves stronger prompts and multi-token continuation checks

## GPU Hybrid Prefix Chain: Real First-Bad Boundary and Fix

We then isolated the earlier `GPU blk.0..2` hello-world failure more carefully instead of assuming a new block-level math bug.

Observed sequence:

- standalone `GPU blk.0` splice on `hello_world`
  - logits cosine: `0.99963910`
  - argmax equal: `True`
- raw chained `GPU blk.0..1`
  - first bad depth before the fix
- exact-input `GPU blk.1` using HF `blk.1.layer_input_seq`
  - logits cosine: `0.99984778`
  - logits RMSE: `0.05983782`
  - argmax equal: `True`
- standalone `GPU blk.1` using the saved `GPU blk.0` output as `--use-prefix-input-seq`
  - logits cosine: `0.99952507`
  - logits RMSE: `0.10250001`
  - argmax equal: `True`

Interpretation:

- `blk.1` local GPU math was not the problem
- saved `blk.0` output was already good enough as the next-layer input
- therefore the real bug had to be inside the multi-layer GPU prefix runner itself

Root cause:

- `qwen36_gpu_blk0_dynamic_q8_oracle.c` only allocated `state_seq` when `--use-prefix-input-seq` was enabled
- in a chained GPU prefix without that flag, later hybrid layers were not actually receiving the previous layer output
- the first attempted fix then over-corrected by seeding layer 0 from zeroed `state_seq` instead of token embeddings
- the final fix was:
  - always allocate mutable `state_seq` for chaining
  - seed it from `prefix.input_seq_ref` only when `--use-prefix-input-seq` is active
  - otherwise let the first layer consume embeddings and only use `state_seq` for later layers

Post-fix depth sweep on `hello_world`:

- `depth=1` (`GPU blk.0`)
  - logits cosine: `0.99963910`
  - argmax equal: `True`
- `depth=2` (`GPU blk.0..1`)
  - logits cosine: `0.99952507`
  - argmax equal: `True`
- `depth=3` (`GPU blk.0..2`)
  - logits cosine: `0.99819592`
  - argmax equal: `True`

What this proves:

- the earlier `GPU blk.0..2` failure was a prefix-chain ownership bug, not evidence that `blk.1` hybrid math had regressed
- the narrow GPU-owned hybrid prefix now composes correctly through `blk.2` on the fast oracle prompt
- the next useful validation is no longer another microscopic chain audit; it is a longer prompt / multi-token greedy check on the repaired owned GPU prefix

## Direct Hybrid GGUF Contract Check

To avoid blindly starting a fixture-free runtime, we added a direct GGUF-vs-fixture checker:

- binary: `qwen36-direct-hybrid-contract-check`
- input:
  - a `Qwen3.6-35B-A3B-Q8_0.gguf`
  - one traced hybrid-layer fixture such as `blk.0`
- goal:
  - compare the already-proven fixture weight surfaces against direct GGUF tensors
  - identify which tensors are already runtime-ready
  - identify which tensors still need Qwen-specific transform logic

Ran on:

- model: `Qwen3.6-35B-A3B-Q8_0.gguf`
- fixture: `blk.0` `code_review` trace

Direct matches already proven:

- `attn_norm`
  - RMSE: `0.0`
  - cosine: `1.0`
- `post_attn_norm`
  - RMSE: `0.0`
  - cosine: `1.0`
- `attn_gate`
  - raw: bad
  - reordered by head groups: exact
- `ssm_alpha`
  - raw: bad
  - reordered by head groups: exact
- `ssm_beta`
  - raw: bad
  - reordered by head groups: exact
- `ssm_norm`
  - RMSE: `0.0`
  - cosine: `1.0`
- `router_w`
  - RMSE: `0.0`
  - cosine: `1.0`
- shared-expert weights
  - `gate_shexp`, `up_shexp`, `down_shexp`, `gate_inp_shexp`
  - all exact

Direct contract surfaces now explained:

- `attn_qkv.weight`
  - raw direct rows do not match fixture contract
  - the known Qwen V-row remap restores exact agreement
- `ssm_conv1d.weight`
  - raw direct rows are only partially aligned
  - naive transpose is wrong
  - the same Qwen V-row remap restores exact agreement
- `ssm_a`
  - raw direct tensor does not match
  - simple head-scalar reorder restores exact agreement
  - note: this fixture surface is the GGUF-native decay coefficient despite older `A_log` naming
- `ssm_dt_bias`
  - raw direct tensor does not match
  - simple head-scalar reorder restores exact agreement
- `ssm_out.weight`
  - raw direct rows do not match fixture `w_out`
  - the known Qwen V-head column reorder restores exact agreement

Interpretation:

- the hybrid block is now structurally explained from direct GGUF:
  - norms
  - router
  - shared expert
  - all hybrid projection and bridge tensors after the known head/column reorder rules
- the remaining blocker is no longer hidden hybrid tensor layout
- the next blocker is engineering:
  - turning the proven direct-GGUF contract into a persistent incremental owned worker
  - preserving hybrid recurrent state, conv history, and full-attention cache across decode steps

This materially changes the next runtime step:

- a real fixture-free persistent worker should not be started from a naive direct-GGUF hybrid implementation
- after encoding these proven reorder rules, the remaining work is stateful decode ownership rather than contract discovery

## Reset Point: Last Proven Qwen Path

After the later direct-worker and DeepSeek detours, the correct reset point is still the fixture-driven narrow Qwen path, not the direct GGUF incremental worker.

Trust this stack:

- `qwen36_gpu_blk0_dynamic_q8_oracle`
  - fixed chained-input ownership bug
  - proven good enough for splice checks
- `qwen36_c_prefix_q8_chain_dynamic`
  - trusted dynamic hybrid replay for the owned linear-attention blocks
  - this is the proven CPU/GPU-owned prefix composition path, even though it still replays the full prefix
- `qwen36_gpu_full_layer_q8_dynamic`
  - trusted full-attention ownership block for the `blk.3 / 7 / 11 / 15` positions
- `misc/qwen36_hybrid_prefix_tail_greedy.py`
  - trusted harness for the known-good replay-based owned prefix
- `misc/qwen36_strict_first_step_validator.py`
  - trusted strict prompt-bound validator
- `misc/qwen36_shared_path_forced_validator.py`
  - trusted forced-token fidelity probe
- `misc/qwen36_long_decode_ab.py`
  - trusted behavioral long-decode A/B harness

Do **not** currently build on:

- `qwen36_incremental_worker.c`
- `qwen36-incremental-worker`

Reason:

- we added a real `STEP <token_id>` path so it now reuses conv history, DeltaNet state, and loaded experts instead of restarting from scratch
- however, the direct worker is still mathematically wrong even before cache reuse becomes relevant

Observed append test on `prompt="Hello world"`:

- HF prompt forward
  - next token: `!`
- worker prefill
  - completed successfully
- worker step
  - completed successfully
- fidelity vs HF `blk.0`
  - `prompt_seq_cosine: -0.02770423`
  - `full_seq_cosine: -0.01814704`
  - `step_last_cosine: 0.00863440`

Interpretation:

- the direct worker is not “almost correct”
- its internal contract is still wrong enough that the persistent cache protocol is irrelevant
- therefore it should be treated as a failed exploratory branch, not the basis for the next runtime layer

Practical consequence:

- the next serious persistent-runtime attempt should be built upward from the last proven Qwen replay stack above
- that means:
  - keep the proven fixture-driven dynamic layer math
  - keep the proven GPU full-attention block
  - add persistence and append/state reuse **on top of that**
  - do **not** restart from the direct GGUF worker until its math is independently repaired

## Persistent `blk.0` Worker Reset

We started the cache-reuse rebuild from the last proven fixture-driven path rather than from the failed direct-GGUF worker.

New artifact:

- `qwen36-fixture-blk0-worker`

Purpose:

- own `blk.0` persistently using the proven `Q36DWF02` fixture contract
- retain:
  - conv ring state
  - DeltaNet state
  - output sequence
  - loaded expert cache
- support:
  - `PREFILL_PREFIX <prefix.bin>`
  - `STEP <token_id>`
  - `DUMP_HIDDEN`

Harness integration:

- `misc/qwen36_hybrid_prefix_tail_greedy.py`
  - added `--prefix-seq-worker-bin`
  - added a persistent subprocess wrapper `PrefixSeqWorker`
  - fixed fixture counting so worker mode accepts `3N-1` hybrid fixtures
  - fixed handoff so downstream dynamic hybrid replay receives `--use-prefix-input-seq`

What we proved in sandbox:

- the worker starts correctly on the known-good `blk.0` code-review fixture
- the worker prefill path emits a valid hidden dump with:
  - `seq_len=21`
  - `hidden=2048`
  - `HIDDEN 43008 172032`
- the worker-backed `blk.0 -> blk.1..2` CPU-owned handoff is accepted by:
  - `qwen36-c-prefix-q8-chain-dynamic`
- measured on the code-review prompt:
  - `prefix_seq_worker_ms=1160.39`
  - `owned_prefix_ms=5581.45`

Interpretation:

- persistent `blk.0` ownership is now integrated into the known-good replay stack
- `blk.0` is no longer being recomputed by relaunching a one-shot oracle every token
- the remaining replay cost is still dominated by:
  - `blk.1..2` replay
  - `blk.3` full-attention replay
  - HF tail validation

Sandbox limitation:

- the next validation hop (`qwen36-gpu-full-layer-q8-dynamic` after the worker-backed `blk.1..2` handoff) cannot be validated inside this sandbox because ROCm devices are not visible here
- direct isolated repro showed:
  - `qwen36-c-prefix-q8-chain-dynamic` succeeds with worker-produced prefix input
  - `qwen36-gpu-full-layer-q8-dynamic` fails in sandbox with:
    - `ds4: ROCm set device failed: no ROCm-capable device is detected`

Current best reading:

- the correct next runtime direction is still:
  - fixture-backed persistent ownership first
  - then migrate more replay surfaces into stateful GPU-owned execution
- the direct-GGUF incremental worker remains a dead branch until its tensor loading contract is repaired

## Persistent `blk.1..2` Chain Worker

We extended the same fixture-backed persistence idea from `blk.0` into the hybrid replay chain.

Current worker capabilities:

- `qwen36-fixture-blk0-worker` now accepts one or more hybrid fixtures
- commands:
  - `PREFILL_PREFIX <prefix.bin>`
  - `STEP <token_id>` for chains whose first owned layer is `blk.0`
  - `STEP_PREFIX <prefix.bin>` for chains whose input is an upstream hidden-state row
  - `DUMP_HIDDEN`
- per-layer retained state:
  - DeltaNet state
  - conv ring
  - expert cache

Harness support:

- `misc/qwen36_hybrid_prefix_tail_greedy.py`
  - new `--c-bin-worker-bin`
  - current scope: one hybrid cycle only (`blk.1..2` in front of `blk.3`)

Validated in sandbox on the code-review prompt:

- configuration:
  - persistent `blk.0` worker
  - persistent `blk.1..2` worker
  - HF splice at `blk.2`
- result:
  - `prefix_seq_worker_ms=1373.53`
  - `hybrid_chain_worker_ms=2244.82`
  - `owned_prefix_ms=3618.35`
  - `hf_next="the"`
  - `patched_next="the"`
  - `argmax_equal=True`

Interpretation:

- `blk.0..2` can now be owned with persistent fixture-backed state instead of replaying those layers from scratch every token
- the remaining replay cost in this 4-layer slice is now concentrated in:
  - `blk.3` full-attention ownership
  - HF tail validation

Important comparison:

- earlier worker-backed `blk.0` plus replayed `blk.1..2` was about `15.4s` owned-prefix time on the same prompt
- persistent `blk.0..2` is now about `3.6s`

This is the first strong sign that the decode-speed problem was primarily cache/state ownership rather than raw math throughput.

## Persistent `blk.0..3` 4-Step Run

We then ran the real `blk.0..3` path with:

- persistent `blk.0` worker
- persistent `blk.1..2` worker
- replayed GPU `blk.3`
- HF tail validation every step

Prompt:

- `code_review.txt`

Observed behavior:

- all 4 checked decode steps matched the HF reference token
- first step timings:
  - `owned_prefix_ms=11151.09`
  - `prefix_seq_worker_ms=1101.53`
  - `hybrid_chain_worker_ms=2021.48`
  - `hf_baseline_ms=68719.36`
  - `hf_patched_ms=10298.51`

Interpretation:

- the persistent-owned path is now behaving correctly through `blk.3`
- the remaining owned-path waste is concentrated in `blk.3`, which is still replayed every token
- the dominant wall in validation is now the HF oracle path, not the owned prefix

Current best next step:

- extend persistence into the full-attention `blk.3 / 7 / 11 / 15` positions
- after that, widen from the 4-layer slice to the 16-layer owned prefix

## Persistent Full-Attention Worker Wiring

We confirmed that the persistent hybrid reuse was already working and that the remaining owned-path replay was coming from the full-attention positions:

- `blk.3`
- `blk.7`
- `blk.11`
- `blk.15`

Evidence from the 16-layer code-review run:

- first decode step:
  - `owned_prefix_ms=42723.45`
  - `prefix_seq_worker_ms=1118.81`
  - `hybrid_chain_worker_ms_total=9530.96`
  - `cycle_0_hybrid_worker_ms=2028.04`
  - `cycle_1_hybrid_worker_ms=2360.05`
  - `cycle_2_hybrid_worker_ms=2559.43`
  - `cycle_3_hybrid_worker_ms=2583.44`
- second decode step:
  - `owned_prefix_ms=34424.94`
  - `prefix_seq_worker_ms=38.59`
  - `hybrid_chain_worker_ms_total=595.66`
  - `cycle_0_hybrid_worker_ms=267.55`
  - `cycle_1_hybrid_worker_ms=94.95`
  - `cycle_2_hybrid_worker_ms=85.40`
  - `cycle_3_hybrid_worker_ms=147.76`

Interpretation:

- the `blk.0` prefix worker reuse is working
- the hybrid chain reuse across `blk.1..2`, `blk.4..6`, `blk.8..10`, `blk.12..14` is also working
- the remaining waste is the per-token replay of the full-attention ownership points

To address that, we added:

- `qwen36-gpu-full-layer-worker`
  - persistent worker around the standalone GPU full-layer path
  - protocol:
    - `PREFILL_SEQ <seq.f32>`
    - `STEP_ROW <row.f32>`
    - `DUMP_HIDDEN`
    - `DUMP_LAST`
    - `RESET`
    - `QUIT`
- harness support in `misc/qwen36_hybrid_prefix_tail_greedy.py`
  - new `--full-layer-worker-bin`
  - one persistent worker per full-attention cycle layer
  - current integration strategy:
    - prefill from the hybrid-owned sequence on step 0
    - step from only the new hybrid output row on later decode steps
    - still dump the full hidden sequence between cycles so the already-proven hybrid workers do not need a protocol change

This is the correct next optimization because it removes the last obvious per-token owned replay without reopening the already-validated hybrid cache path.

Important update from the later prefill audit:

- the standalone full-layer worker was not actually clean at this point in the timeline
- it still carried the legacy `1 + w` RMSNorm contract on:
  - `attn_norm`
  - `post_attn_norm`
- after restoring raw-weight RMSNorm there, standalone full-layer prefill isolation became much healthier
- so any older study text that treats the worker as already-proven full-layer math should be read as historical, not current truth

## 36-Owned vs 40-Owned Residency Cliff

We instrumented the native owned-session coordinator and the persistent full-attention worker directly:

- `qwen36-owned-session-worker`
  - per-cycle hybrid ms
  - per-cycle full-layer ms
  - per-step totals
- `qwen36-gpu-full-layer-worker`
  - startup VRAM report via `ds4_gpu_print_memory_report()`
  - per-step stage timings:
    - allocation
    - Q/K/V projection batch
    - Q/K/V readback
    - CPU attention
    - output projection
    - FFN shared stage
    - FFN expert loop
    - KV append
  - end-of-step VRAM report

### 36-owned reference

Configuration:

- 9 persistent full-attention workers:
  - `blk.3`
  - `blk.7`
  - `blk.11`
  - `blk.15`
  - `blk.19`
  - `blk.23`
  - `blk.27`
  - `blk.31`
  - `blk.35`
- HF tail still owns `blk.36..39`

Observed steady-state decode:

- `hybrid_ms ~= 730..800 ms` total per token
- `full_ms ~= 100..135 ms` total per token
- per full layer step:
  - generally `5..26 ms`

Interpretation:

- the persistent full-attention worker design is healthy at 9 resident workers
- hybrid ownership remains the dominant cost in this configuration

### 40-owned full-GPU attempt

Configuration:

- 10 persistent full-attention workers:
  - `blk.3`
  - `blk.7`
  - `blk.11`
  - `blk.15`
  - `blk.19`
  - `blk.23`
  - `blk.27`
  - `blk.31`
  - `blk.35`
  - `blk.39`
- lightweight tail only:
  - final RMS norm + `lm_head`

Observed steady-state decode:

- `hybrid_ms ~= 1.0..1.2 s` total per token
- `full_ms ~= 12..17 s` total per token
- per full layer step:
  - generally `1.0..1.5 s`
  - with severe outliers:
    - `blk.31 ~= 3.1 s`
    - `blk.35 ~= 4.1 s`

### Stage-level diagnosis

The 40-owned slowdown is overwhelmingly concentrated in the first GPU projection batch of the full-layer worker:

- `qkv_gpu_ms` dominates each slow full-layer step
- `attn_cpu_ms` is negligible
- `out_proj_ms` is negligible
- `ffn_shared_*` is negligible
- `ffn_expert_*` is small relative to the wall time

Representative 40-owned step data:

- `blk.3`
  - `qkv_gpu_ms=1023.94`
  - full step total `=1038.82`
- `blk.31`
  - `qkv_gpu_ms=3117.29`
  - full step total `=3130.64`
- `blk.35`
  - `qkv_gpu_ms=4099.84`
  - full step total `=4110.45`

Interpretation:

- the decode cliff is not caused by:
  - Python dump overhead
  - hybrid fixture math
  - CPU-side attention
  - FFN expert routing
- the decode cliff is caused by catastrophic slowdown in the full-layer Q/K/V projection stage under the 10-worker resident configuration

### VRAM evidence

Per-worker startup reports on RX 7900 XTX show steady VRAM growth as persistent full workers are added:

- early worker startup:
  - `used ~= 4.29 GiB`
- last worker startup:
  - `used ~= 6.22 GiB`

After decode begins, end-of-step memory reports converge near saturation:

- `used ~= 22.3..22.45 GiB`
- `free ~= 1.5..1.6 GiB`
- `total ~= 23.96 GiB`

Interpretation:

- the 10-worker full-ownership configuration pushes the card into a near-full VRAM regime
- once there, the Q/K/V projection path degrades by roughly two orders of magnitude versus the healthy 9-worker case
- this is a residency / memory-pressure cliff, not a generic compute regression

### Practical conclusion

The current persistent-per-layer full-attention worker architecture does **not** scale to full 40-layer ownership on RX 7900 XTX.

What works:

- 9 resident full-attention workers + HF tail

What fails:

- 10 resident full-attention workers + lightweight tail

Therefore the correct runtime direction is not "one persistent full worker per owned full-attention layer forever". The runtime needs a global residency strategy, for example:

- one unified owned worker process instead of many child workers
- a rolling or shared full-attention executor instead of one process per full layer
- explicit model-range / expert streaming
- SSD-backed warm caches coordinated at the runtime level rather than by independent worker processes

This result is strong enough to move study emphasis from "prove persistence works" to "design the proper unified runtime and streaming ownership model."
