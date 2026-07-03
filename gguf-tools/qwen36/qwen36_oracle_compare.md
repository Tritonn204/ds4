# Qwen3.6 Oracle Compare

This harness answers a narrow question:

- do two Qwen3.6 GGUF exports tokenize the same prompt the same way?
- when run through `llama.cpp`, do they keep the same greedy token path?
- if they diverge, at which generated step does the first divergence appear?
- how much probability does a lower-bit export assign to the `Q8_0` greedy path?

It is not a validator for our future runtime.

It is the first empirical bridge between:

- structural contract validation
- observed behavior under a trusted external implementation

Current scope:

- exact `Qwen3.6-35B-A3B` only
- raw prompt string first
- batch size `1`
- greedy decode only
- local `libllama` via `qwen36-llama-oracle` as the oracle
- compare `Q8_0` vs lower-bit exports such as `Q4_K_XL` and `IQ2_XXS`

Current metrics:

- prompt tokenization equality
- greedy continuation token agreement
- common greedy prefix length
- full-sequence token agreement rate
- token-level Levenshtein distance
- reference-path score:
  - run `Q8_0` greedily
  - force the lower-bit export across the exact same chosen token IDs
  - record the lower-bit model's logprob on each forced token
  - summarize average negative log likelihood and a perplexity-like score

This is closer to the kind of divergence measurement used in quantization
comparisons than a simple first-token smoke test.

First short-result baseline on `quant_report.txt`, `n_predict = 8`:

- `Q8_0` vs `Q4_K_XL`
  - tokenization equal
  - greedy continuation identical across all `8` steps
  - reference-path `avg_nll = 1.116229`
  - reference-path `perplexity_like = 3.053320`
- `Q8_0` vs `IQ2_XXS`
  - tokenization equal
  - greedy continuation matched `7/8` steps
  - first mismatch at step `5`: `retain` vs `use`
  - reference-path `avg_nll = 1.473487`
  - reference-path `perplexity_like = 4.364429`

Additional short-result probes:

- `code_review.txt`, `n_predict = 8`
  - `Q8_0` vs `Q4_K_XL`
    - tokenization equal
    - greedy continuation identical across all `8` steps
    - reference-path `avg_nll = 1.440111`
    - reference-path `perplexity_like = 4.221163`
  - `Q8_0` vs `IQ2_XXS`
    - tokenization equal
    - greedy continuation matched only `1/8` steps
    - first mismatch at step `1`
    - `Q8_0`: `"the token routing and dispatch logic. This"`
    - `IQ2_XXS`: `"the MoE routing logic. The Mo"`
    - reference-path `avg_nll = 1.740127`
    - reference-path `perplexity_like = 5.698066`
- `patch_plan.txt`, `n_predict = 8`
  - `Q8_0` vs `IQ2_XXS`
    - tokenization equal
    - greedy continuation identical across all `8` steps
    - reference-path `avg_nll = 0.343507`
    - reference-path `perplexity_like = 1.409883`

Interpretation so far:

- trivial continuation scaffolds remain stable even at `IQ2_XXS`
- implementation-shaped continuations can diverge immediately at `IQ2_XXS`
- `Q4_K_XL` remains a strong control and currently tracks the `Q8_0` greedy
  path exactly on these short probes

Useful command:

```sh
make -C /mnt/f/git/ds4 qwen36-oracle
python3 /mnt/f/git/ds4/gguf-tools/qwen36/qwen36_oracle_compare.py \
  --json-out /tmp/qwen36_oracle_compare.json
```
