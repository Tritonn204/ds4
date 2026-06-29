# DS4 Architecture Deep Study & Qwen3.6-35B-A3B Application Plan

## 1. DS4 Attention Kernel Specs

### Kernel Variants (from `ds4_rocm_attention.cuh` + `ds4_rocm_attention_launch.cuh`)

| Kernel | Use Case | Grid | Threads/Block | Shared Mem |
|--------|----------|------|---------------|------------|
| `attention_decode_mixed_one_fast_oldhip_kernel` | Oldhip decode (default for `!g_quality_mode`) | `(n_head, 1, 1)` — **1 block per head** | **256** | `scores[n_raw + n_comp]` (dynamic) |
| `attention_decode_mixed_kernel` | Quality-mode decode | `(1, n_head, 1)` — 1 block/head | **256** | ~1024 floats score buffer |
| `attention_decode_indexed_mixed_one_fast_oldhip_kernel` | Indexed sparse attention | `(n_head, 1, 1)` | **256** | `scores[]` + `comp_rows[]` |
| `attention_decode_mixed_heads8_online_kernel` | 8-heads-per-block online (no score buffer) | `(1, (n_head+7)/8, 1)` | 256 | None large |
| `attention_prefill_raw_kernel` | Raw prefill | `(n_tokens, n_head, 1)` — 1 block per (tok,head) | **128** | `scores[256]` + `partial[128]` |
| `attention_prefill_mixed_kernel` | Mixed prefill | `(n_tokens, n_head, 1)` | **256** | `scores[2048]` + `partial[256]` |

**Key finding**: DS4 decode attention uses **1 head per block, 256 threads**. Each block processes a circular ring buffer of raw KV rows + compressed KV rows.

### KV Cache Format (from `ds4_rocm_fp8_kv.cuh`)
- **DS4 KV cache = FP16** (not FP8 as the filename might suggest)
- The `store_raw_kv_batch_kernel` converts from float to **FP16** using `f32_to_f16_bits_hip_round()` before storing
- Cache is a **circular ring buffer** indexed as `(raw_start + r) % raw_cap`
- DS4 also has an FP8 quantization path for the *nope* (non-rotary) dimensions via `fp8_kv_quantize_kernel` — the first `head_dim - n_rot` elements are E4M3-quantized, rotary dims stay FP16
- The compressed KV is stored separately with FP8 quantization too

### KV Cache Read
- Raw cache: `float raw_kv[raw_cap * head_dim]` — a flat interleaved array
- Comp cache: `float comp_kv[n_comp * head_dim]`
- Reads use modulo indexing for the ring buffer; each head's block computes scores over all raw rows + compressed rows

---

## 2. DS4 WMMA Matmul Spec (from `ds4_rocm_q8.cuh`)

### Two WMMA kernels exist:

#### A. `matmul_q8_0_f32_batch_wmma_4w_kernel` (lines 683-782)
- **Tile**: 64 × 64 output, 32 K-tile, 4 waves × 16 output rows/wave
- **128 threads** per block, **4 warps**
- Uses `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (raw intrinsics)
- Shared memory: `_Float16 lds_x[64 × 32]` = 4096 bytes
- Weights dequantized on-the-fly from Q8_0 format
- Activations loaded from float buffer, converted to f16 in LDS

#### B. `matmul_q8_0_f32_batch_wmma_onthefly_kernel` (template, lines 785-855)
- **Tile**: configurable `BM=16, BN=16, BK=16`, `TILES_N=8` by default
- Uses **rocwmma** API (fragment, load_matrix_sync, mma_sync, store_matrix_sync)
- Shared memory: `(BM + TILES_N*BN)*BK*sizeof(half) + TILES_N*BM*BN*sizeof(float)`

### When WMMA is used (from `ds4_rocm_matmul.cuh:276-294`):
```c
if (n_tok > 1 && !g_quality_mode && (in_dim % 32) == 0 &&
    out_dim >= 1024 && n_tok >= 256) {
    // Use WMMA 4w kernel
}
```

### 1-token path (from `ds4_rocm_matmul.cuh:251-274`):
When `n_tok == 1`:
1. If `in_dim & 31 == 0` and `in_dim <= 8192`: Uses `matmul_q8_0_f32_sharedx_warp_rows_w32_kernel` — loads entire activation vector into shared memory (1 block, 1024 threads, `in_dim*4` bytes shared)
2. Otherwise: `matmul_q8_0_f32_warp8_kernel` — 1 block per 8 output rows, 256 threads, 1 warp per row, uses warp_sum

**There is NO WMMA path for n_tok=1**. The WMMA kernels have a hard floor at `n_tok >= 256`.

---

## 3. DS4 Sync/Fusion Strategy

### `ds4_gpu_begin_commands()` is a NO-OP on ROCm
From `ds4_rocm_runtime.cuh:4597-4602`:
```c
int ds4_gpu_begin_commands(void) { return 1; }         // NO-OP
int ds4_gpu_flush_commands(void) { return cuda_ok(cudaDeviceSynchronize(), "flush"); }
int ds4_gpu_end_commands(void) { return cuda_ok(cudaDeviceSynchronize(), "end commands"); }
```

So `begin_commands` + `end_commands` on ROCm is semantically just: launch everything, then sync. **No actual command batch fencing — it's just a synchronize point.**

### How DS4 uses it (in the GPU graph decode):
DS4's Metal/CUDA graph builds a monolithic capture that runs the entire layer (norm→Q→KV→RoPE→KV store→attention→inverse RoPE→grouped output→HC post→FFN norms→shared expert→routed MoE→shared down→HC expand) as a **single graph**. The sync points are implicit in the graph.

So for an un-batched non-graph path on ROCm, every `end_commands()` is a full `cudaDeviceSynchronize()`.

### What DS4 fuses:
1. **Q projection + KV projection** from single input norm (separate matmul calls but same batch)
2. **Shared gate/up + SwiGLU** into `shared_gate_up_swiglu_q8_0_rows_w32_kernel` (2 matmuls + swiglu in 1 kernel, with 32 rows per block × 32 threads per row = 1024 threads)
3. **Shared down + HC expand** into `shared_down_hc_expand_q8_0_tensor` (down projection + add routed + HC split/combine in 1 kernel)
4. Some Q8→F16 cached paths use cuBLAS HIPBLASLT gemm as acceleration

---

## 4. Qwen Current State (from `qwen36_unified_owned_worker.c`)

### Qwen's `run_hybrid_layer_step_gpuproj` sync points per layer:
This is the "GPU projections" variant of the hybrid layer. Counting `end_commands()`:
1. **Batch 1** (lines 1347-1388): norm + z + a + b + q + k + v matmuls → **1 sync**
2. CPU: reorder, conv, head splitting, deltanet step, norm + gate
3. **Batch 2** (lines 1469-1481): out_proj matmul → **1 sync**
4. CPU: residual, post_norm
5. **Batch 3** (lines 1501-1529): router + shared gate/up/swiglu + shared down → **1 sync**
6. CPU: topk, per-expert routing...
7. **Batch 4-N** (lines 1543-1569): One batch per selected expert (gate+up+swiglu+down) → **1 sync per expert**

**Total: 4 + n_experts sync points per hybrid layer** — very fragmented.

### Qwen's `run_full_layer_step_gpu` (pure attention layers):
1. **Batch 1**: norm + Q + K + V matmuls → **1 sync**
2. CPU: all of attention (norms, RoPE, softmax, PV reduction) — **attention entirely on CPU!**
3. **Batch 2**: o_proj matmul → **1 sync**
4. CPU/GPU FFN: varies — shared gate/up + down as batch → **1+ sync**

The pure attention layers are CPU-heavy and the hybrid layers have many small GPU batches.

---

## 5. Expert Dispatch Analysis

### Qwen's current approach:
- Read router logits back to CPU via `ds4_gpu_tensor_read`
- Run `topk_softmax256` on CPU
- For each selected expert: `begin_commands → pair matmul → swiglu → down matmul → end_commands → readback`
- **Each expert gets its own command batch and sync point**

### DS4's approach (from `ds4_rocm_moe_launch.cuh`):
- DS4 uses a batched MoE kernel: `ds4_gpu_routed_moe_one_tensor` (all selected experts in one call)
- Gate/up for all selected experts computed in a single WMMA-hotlist kernel
- Mid (SwiGLU) computed as a fused element-wise
- Down for all selected computed in another WMMA-hotlist kernel
- Expert weights loaded via a resident cache or stream loader that tracks which weights are already on GPU

---

## 6. GPU Residency in DS4

**Kept on GPU between layers:**
- Q8_0 weights (via Q8→F16 cache — a key performance feature)
- KV cache (raw + compressed)
- Internal state (compressor state KV, indexer state KV)

**Read back to CPU:**
- Router logits (for topk selection on CPU)
- Selected expert IDs (for orchestrating expert dispatch)

**Not read back:**
- Intermediate activations between kernels within a layer (all GPU-resident)
- Attention heads (stay on GPU through fused graph)

---

## 7. Concrete Qwen Plan

### Phase 1: Pure Attention Layer GPU Kernel (36 layers)

#### New GPU kernels needed:
1. **Fused attention decode kernel** — adapted from DS4's `attention_decode_mixed_one_fast_oldhip_kernel`
   - Parameters: `head_dim=256, n_head=16, n_kv_heads=2, block_size=256`
   - 1 block per head, 256 threads
   - Handles Q-gate split (first 256 dims = query, next 256 = gate)
   - Q-norm + K-norm fused into kernel (apply RMS norm per-head blocks from shared mem)
   - RoPE integrated (rotary dim=64)
   - KV cache format: FP16, flat `[pos * n_kv_heads * head_dim]`

2. **Single WMMA matmul for 1-token Q8_0** — adapted from `matmul_q8_0_f32_batch_wmma_4w`
   - New kernel: tile output rows across 4 waves, use `tok_tile=1` instead of 64
   - Each wave handles 16 output rows
   - Strip shared memory tile loading — just load 1 token × 32 K elements per iteration
   - Use `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` for the actual MAC
   - **For small dims (256, 512, 2048):** Use 1-2 blocks, 4 waves each, all output rows covered

3. **Fused RMS-norm weight + matmul** kernel for Q-projection and K-projection:
   - Combine the separate norm + matmul into one kernel
   - Similar to DS4's `rms_norm_weight_kernel` but feed directly into matmul

4. **QK-norm + K/V split kernel**:
   - The Qwen full layer projects Q_and_gate together (4096→2×2048), then splits
   - Fuse the per-head Q-norm and K-norm into a single kernel

#### Reuse from DS4:
- `rms_norm_plain_kernel` / `rms_norm_weight_kernel` — just parameterize with eps=1e-6
- `head_rms_norm_rope_tail_kernel` — for Q/K RoPE (adapt dims from 512→256, n_rot from 128→64)
- `matmul_q8_0_f32_warp8_kernel` — for 1-token matmuls on small dims
- `shared_gate_up_swiglu_q8_0_rows_w32_kernel` — need to adapt for `in_dim=2048, out_dim=512`

#### Sync target: **1 batch per layer**
```
begin → fused_norm_QKV → fused_attention_decode → fused_o_proj →
        shared_gate_up_swiglu → shared_down → end(sync once)
```

### Phase 2: Hybrid Layer Kernel (4 layers with deltanet)

#### New GPU kernels needed:
1. **Deltanet GPU kernel** — replaces the CPU loop over heads:
   ```c
   // Problem: Qwen's deltanet has head_k_dim=128, head_v_dim=64, num_v_heads=4
   // State size: 4 × 128 × 64 = 32768 floats (128 KB)
   // State stays GPU-resident
   // Kernel: 1 thread per head_v_dim element, head_v_dim threads per head
   // No syncthreads needed (each thread updates independent state rows)

   __global__ void deltanet_step_kernel(
       float *state,       // [num_v_heads * head_k_dim * head_v_dim]
       const float *k,     // [num_v_heads * head_k_dim]
       const float *q,     // [num_v_heads * head_k_dim]
       const float *v,     // [num_v_heads * head_v_dim]
       const float *g_exp, // [num_v_heads]
       const float *beta,  // [num_v_heads]
       float *out          // [num_v_heads * head_v_dim]
   );
   ```

2. **Fused SSM output norm + gating kernel**
3. **Fused conv3 kernel** — convolution over 3 prior steps on GPU
4. **Fused A_log/softplus + Dt_bias kernel**

#### Reuse from DS4:
- All the Q8_0 matmul variants for 1-token (Q-proj, K-proj, V-proj, Z, A, B, out_proj)
- Router matmul (already works via `ds4_gpu_matmul_q8_0_tensor`)
- Shared gate/up swiglu fusion

#### Sync target: **1 batch per hybrid layer**
```
begin → fused_norm → fused_QZVAB → fused_conv → fused_deltanet(4 heads) →
        fused_ssm_out → fused_out_proj → router → shared_gate_up_swiglu →
        shared_down → end(sync once)
```
**Exception**: experts routed per-batch (see Phase 3)

### Phase 3: Expert Dispatch Optimization

#### DS4-style batched MoE for Qwen:
- Keep expert weight Q8_0 tensors in a resident cache (LRU, holds ~50 experts)
- New kernel: `moe_gate_up_mid_q8_0_hotlist_kernel` — gate+up for all selected experts in one GPU call
- New kernel: `moe_down_q8_0_hotlist_kernel` — down for all selected in one call
- Do router + topk on GPU to avoid readback (TODO: topk kernel)

#### Short-term: Reduce expert syncs to 1 total
```
begin → gate1_up1 → gate2_up2 → ... → swiglu_all → down1 → down2 → ... → end
```
Queue all expert kernels in ONE batch, sync once at the end.

### Phase 4: GPU-resident KV Cache
- Allocate GPU KV cache as FP16 circular buffers
- 36 pure layers × 2 KV heads × 256 dims = 18,432 elements per position, ~72 KB per pos at FP16
- For 8192 context: ~590 MB total KV cache — fits in GPU memory

---

## 8. Tensor Allocation Plan

| Tensor | Location | Size (Qwen) | Notes |
|--------|----------|-------------|-------|
| Model weights (Q8_0) | GPU (munmap/mmap) | ~15 GB | Already mmaped, streamed on demand |
| Q8→F16 cache | GPU | Variable | L1 acceleration cache |
| Input/Output activations | GPU | 2×2048×4 B=16KB | Single token, tiny |
| Intermediate activations | GPU | ~50-100 KB | Z, A, B, Q, K, V, core, etc. |
| Deltanet state | GPU | 128 KB/head × 4 heads × 4 layers = 2 MB | Persistent across steps |
| Conv ring buffers | GPU | 3×4×(128+128+256)*4B = 24 KB/layer | Persistent |
| KV cache | GPU | ~590 MB | Persistent, FP16 |
| Router logits | **CPU** | 256×4 B=1 KB | Readback for topk |
| Expert weights | GPU stream cache | Variable | LRU resident |

**What MUST come to CPU:**
- Router logits (for CPU topk, unless GPU topk is implemented)
- Final output row (for sampler)

---

## 9. Risk Assessment & Validation Order

### Highest Risk → Validate First:
1. **WMMA 1-token kernel** — uncharted path. DS4 gates WMMA at 256 tokens. Need to design a new 1-token WMMA kernel with correct numerical results. **Validate with CPU reference first**.
2. **Deltanet GPU kernel numerical accuracy** — The SSM state update involves `exp(g)`, `beta`, and per-element sums that must match CPU bit-identical. **Test with single head against CPU before scaling**.
3. **Signal ordering for RoPE on GPU** — Qwen uses different head layouts than DS4 (interleaved vs sequential). Need correct reordering or fused norm_rope handling.

### Medium Risk:
4. **Fused norm+matmul kernel** — Straightforward adaptation of existing patterns, but must match Qwen's eps=1e-6 and weight ordering.
5. **Expert cache on GPU** — DS4 has a working pattern; mainly needs sizing for Qwen's 256 experts × different dimensions.
6. **KV cache synchronization** — When attention runs on GPU and updates KV cache on GPU, CPU must not race.

### Low Risk:
7. **RMS norm kernels** — Straight parameterization.
8. **SwiGLU kernels** — DS4's already work, just parameterize clamp=80.
9. **Q8_0 matmul kernels** — DS4 already has warp8 and shared-x variants for 1-token.

---

## 10. Estimated Files/Kernels to Build

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `qwen36_full_attention_gpu.cuh` | Fused Qwen pure-attention decode kernel | ~300 |
| `qwen36_full_ffn_gpu.cuh` | Fused Qwen FFN (gate/up/swiglu/down) | ~200 |
| `qwen36_deltanet_gpu.cuh` | Deltanet SSM GPU kernel | ~200 |
| `qwen36_wmma_q8_1tok.cuh` | 1-token WMMA matmul for Q8_0 | ~250 |
| `qwen36_norm_fused.cuh` | Fused norm+proj kernels | ~150 |
| `qwen36_kv_cache_gpu.cuh` | GPU KV cache store/load | ~80 |
| `qwen36_router_gpu.cuh` | Optional GPU router topk | ~100 |
| `qwen36_moe_gpu.cuh` | Batched expert dispatch | ~200 |
| `qwen36_gpu_runtime.c` | Orchestration, init, cleanup | ~500 |
| **Total** | | **~2000** |
