# Qwen3.6 GPU Oracle Plan

This is the shortest path from the working CPU-owned prefix to a practical GPU oracle.

## Current Proven Boundary

- contract: `Qwen3.6-35B-A3B` only
- artifact: local `Q8_0` GGUF
- owned prefix:
  - `token_embd`
  - `blk.0`
  - `blk.1`
  - `blk.2`
  - `blk.3`
- validation:
  - exact greedy-token agreement with HF for the first `4` tokens on `patch_plan.txt`

## Goal

Build a narrow GPU path that can answer:

- does the owned prefix produce the same next token?
- how many greedy tokens match exactly on longer prompts?
- when it diverges, is the continuation still functionally equivalent?

The GPU oracle is not a generic runtime.

## Success Metric

Primary:

- greedy token agreement against HF or llama.cpp oracle

Secondary:

- selected-logit agreement at the output surface
- continuation-level agreement on short prompts

Not primary:

- internal tensor-by-tensor equality

## Oracle Shape

Stage 1:

- GPU owns `token_embd + blk.0..3`
- HF still owns layers `4..39` and final logits
- compare greedy tokens only

Stage 2:

- GPU owns `token_embd + blk.0..3`
- CPU narrow root scorer owns final norm/output rows for selected logits
- reduce HF involvement further

Stage 3:

- GPU owns a longer prefix
- HF becomes fallback oracle only

## Recommended GPU Backend Order

For this machine:

1. `HIP / ROCm`
2. CPU fallback

For Mac deployment later:

1. `Metal`

Reason:

- user hardware here favors AMD GPU work first
- Metal becomes the deployment path for the Mac Mini target
- algorithmic ownership should be proven before duplicating backend effort

## Kernel Scope

The first GPU pass should cover only hot owned-prefix math:

- token embedding row fetch
- hybrid SSM / DeltaNet layers
- first full-attention layer
- router logits and top-k
- routed expert FFNs
- shared expert FFN

Keep:

- batch size `1`
- short context first
- greedy decode only
- no vision
- no MTP
- no server

## Immediate Engineering Steps

1. Freeze the owned-prefix contract at `blk.0..3`.
2. Add a GPU prefix runner mirroring the current CPU chain.
3. Reuse current hybrid Python splice harness as the oracle shell.
4. Compare only:
   - next token
   - short generated continuation
5. Add a small prompt set:
   - `patch_plan`
   - a short coding prompt
   - a short prose prompt
6. Once stable, run longer prompts.
7. Then try SWE-style harnessing.

## First HIP Compile Findings

The first HIP/ROCm scaffold compile surfaced backend-level blockers before any Qwen-specific GPU runner code was attempted.

WSL path:

- `rocWMMA` static assert: `Unsupported architecture`
- multiple `hipblasGemmEx` / `hipblasGemmStridedBatchedEx` signature mismatches
- multiple missing warp intrinsics such as `__shfl_sync` / `__shfl_down_sync`

Windows ROCm path:

- compile is reaching the HIP backend
- current first blocker is missing `rocwmma/rocwmma-version.hpp`
- this is an include-path / ROCm-layout issue, not a link failure and not a Qwen-contract failure

Interpretation:

- the current DS4 ROCm backend does not compile cleanly in the WSL ROCm environment as-is
- Windows ROCm is the primary repair path for this machine
- this remains a backend compatibility issue, not a Qwen architectural-contract issue
- the next step is to repair the ROCm substrate first, then hang the Qwen GPU oracle on it

Practical consequence:

- the GPU-oracle phase now starts with ROCm backend repair on Windows using `gfx1100`
- after that, the narrow Qwen host-orchestration path can resume on top of a working HIP backend

## ROCm Bring-Up Status

This machine has now crossed the backend bring-up threshold.

Observed locally:

- vendored `rocWMMA` include path works
- WSL ROCm build now succeeds for `qwen36-gpu-oracle-scaffold`
- runtime init succeeds on `AMD Radeon RX 7900 XTX`
- scaffold reports:
  - `contract: Qwen3.6-35B-A3B Q8_0`
  - `backend_gpu_init: ok`
  - `block_count: 40`
  - `expert_topk: 8`
  - `full_attention_interval: 4`

Interpretation:

- generic ROCm substrate repair is no longer the main blocker
- the next blocker is the first Qwen-owned GPU execution slice
- the shortest meaningful slice is `token_embd + blk.0`

## Longer-Prompt Validation

Once the GPU oracle matches early tokens reliably:

- run longer coding prompts
- run structured instruction prompts
- run paragraph/report prompts
- then attempt SWE-bench-style task prompts

Do not jump to full benchmark suites before the GPU oracle can:

- generate visible text
- survive dozens of decode steps
- maintain a useful exact-match prefix

## What This Unlocks

If the GPU oracle works:

- we can validate longer prompts without the current CPU validator tax
- we can test size/faithfulness tradeoffs on real generations
- we can start deciding where SSD streaming matters most
- we can move toward practical SWE-style evaluation instead of only micro proofs

## Fixed-6 DS4 ROCm Boundary

The first real GPU-oracle failure clarified the DS4/Qwen split cleanly.

Observed:

- ROCm init succeeded
- Q8 embedding probe matched closely
- the failure hit at DS4 routed-MoE/router batch helpers

Root cause:

- DS4 ROCm still hardcodes the DeepSeek routed-expert contract at `n_expert_used = 6`
- Qwen3.6-35B-A3B requires `topk = 8`

Meaning:

- this is not "GPU bring-up still broken"
- this is "the DeepSeek narrow path is doing its job and rejecting a different MoE contract"

## Narrow Qwen GPU Oracle Direction

Immediate plan change:

- do not genericize DS4 ROCm router/MoE batch code yet
- do not clone the full DS4 routed-MoE launch stack yet
- instead, build the first Qwen GPU oracle as Qwen-local host orchestration over DS4 public GPU primitives

That means:

- GPU:
  - post-attention RMSNorm
  - router logits matmul
  - shared expert gate/up/down
  - routed expert gate/up/down per selected expert
- CPU:
  - top-k=8 router select + softmax
  - unique-expert union construction
  - weighted accumulation across the selected experts

This is intentionally asymmetric.

It is the smallest GPU-owning proof that:

- respects the Qwen contract
- bypasses the DeepSeek fixed-6 assumption
- keeps the expensive q8 expert math on HIP
- avoids premature generic kernel work
