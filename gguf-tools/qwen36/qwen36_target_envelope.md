# Qwen Narrow Runtime Target Envelope

This file is a planning note, not a benchmark result.

It captures two things:

- which open/open-weight sparse models are most interesting for DS4-style narrow runtimes
- whether `Qwen3.6-35B-A3B` looks plausible on a `16 GB M4 Mac mini` if we achieve aggressive mixed quantization and sparse-weight streaming

## Hardware Envelope Heuristic

For narrow sparse runtimes, fit is better modeled as:

- `resident_path + hot_sparse_cache + kv/scratch`

instead of:

- `whole_model_in_fast_memory`

A rough planning equation is:

- `effective_fast_mem ~= r * c * naive_q8_model_size + cache/scratch`

Where:

- `c` = final compressed size as a fraction of naive `Q8`
- `r` = resident fraction of the compressed model that must stay in fast memory

Assumed aggressive DS4-style regime:

- `c ~= 0.30`
- `r ~= 0.25` to `0.45`

This implies fast-memory residency around:

- `0.075x` to `0.135x` of naive `Q8` size

## Device Planning Table

| Device tier | Fast memory | Backing tier | Narrow-runtime implication | Plausible sparse target class |
|---|---:|---:|---|---|
| `7900 XTX + 128 GB RAM` | `24 GB VRAM` | `128 GB RAM`, then SSD | strong desktop sparse-runtime target; RAM can act as cold sparse backing store before SSD | `Qwen3.6-35B-A3B` is conservative; `80B-120B total` sparse MoE looks realistic; `150B-250B total` sparse MoE is an ambitious stretch |
| `Mac mini M4 16 GB` | `16 GB unified` | SSD | smaller fast-memory pool, but much friendlier hierarchy than PCIe dGPU; resident/non-resident split matters more than raw total size | `Qwen3.6-35B-A3B` is challenging but not absurd if the resident path is very small and sparse streaming works well |
| `Strong phone` | roughly `8-16 GB shared` | flash storage | thermal and bandwidth limits dominate sooner than storage fit | mobile-first tiny dense models or mobile-scale MoE only |

## Open/Open-Weight Sparse Model Table

| Model | Total / Active | License | Why it matters for narrow runtime work | Source |
|---|---:|---|---|---|
| `Qwen3.6-35B-A3B` | `35B / 3B` | Apache-2.0 | current proving ground; tiny active path relative to total size | https://huggingface.co/Qwen/Qwen3.6-35B-A3B |
| `Qwen3-Next-80B-A3B-Instruct` | `80B / 3B` | Apache-2.0 | strongest same-family follow-on target; materially larger total size with the same tiny active path | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct |
| `GLM-4.5-Air` | `106B / 12B` | MIT | direct hit for the `80B-120B` band; stronger stretch than Qwen-Next because active params are larger | https://huggingface.co/zai-org/GLM-4.5-Air and https://arxiv.org/abs/2508.06471 |
| `Mixtral-8x22B-Instruct-v0.1` | about `141B total` | Apache-2.0 | older, well-understood sparse baseline near the lower edge of the stretch band | https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1 |
| `Qwen3-235B-A22B` | `235B / 22B` | Apache-2.0 | direct hit for the `150B-250B` stretch band; major candidate if Qwen narrow runtime generalizes | https://en.wikipedia.org/wiki/Qwen |

Larger but outside the current comfort band:

| Model | Total / Active | Why it is outside the current target band | Source |
|---|---:|---|---|
| `Tencent Hunyuan-Large` | `389B / 52B` | active path and total size are both much larger; likely a next-tier project | https://huggingface.co/tencent/Tencent-Hunyuan-Large |
| `DeepSeek-V3` | `671B / 37B` | much too large for the current desktop target class | https://huggingface.co/deepseek-ai/DeepSeek-V3 and https://arxiv.org/abs/2412.19437 |
| `Kimi K2` | `1T / 32B` | clearly outside current hardware goals | https://arxiv.org/abs/2507.20534 |

## Does Qwen3.6-35B-A3B Have a Shot on a 16 GB M4 Mac mini?

Short answer:

- yes, it has a real shot
- no, it is not guaranteed

Why it has a shot:

- the model is only `3B active` at inference
- the total size is modest relative to the frontier MoE class
- Apple unified memory is a much friendlier hierarchy for sparse streaming than a discrete GPU over PCIe
- SSD streaming on macOS is much more aligned with the DS4 memory-dial idea than typical dGPU systems

Why it is still risky:

- `16 GB` is a small fast-memory pool
- the resident path must stay extremely lean
- KV, scratch, tokenizer/runtime overhead, and expert cache compete for the same memory budget
- performance may be acceptable only if the hot-expert hit rate is high

Bottom line:

- `Qwen3.6-35B-A3B` on a `16 GB M4 Mac mini` is not obviously impossible
- it is exactly the kind of target where a DS4-style narrow runtime could convert a hard cliff into a slower but usable slope
- if this project succeeds technically, the `16 GB` Mac mini is a credible best-case deployment target for this model class
