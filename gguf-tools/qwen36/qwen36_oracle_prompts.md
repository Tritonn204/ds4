# Qwen3.6 Oracle Prompts

These prompts are not benchmarks.

They exist to probe whether lower-bit exports preserve behavior in the places we
actually care about for a narrow coding-oriented runtime.

Use them in this order:

1. tokenization sanity
2. short coding/report continuation
3. longer coding/report continuation
4. role-sensitive continuation probes
5. only then lower-bit stress such as `Q2`

## Prompt set

### `hello_world`

Purpose:

- cheapest tokenizer and first-token smoke test

Prompt:

```text
Hello world
```

Suggested compare:

- `--n-predict 1`

### `quant_report`

Purpose:

- prose/report continuation
- shaped as an unfinished note, not an instruction block

Prompt:

```text
Write one compact paragraph explaining why role-specific quantization in a MoE model can preserve quality better than uniform low-bit quantization. Keep it technical and concrete.
```

Suggested compare:

- `--n-predict 32`

### `code_review`

Purpose:

- coding-oriented explanation path
- shaped as an unfinished review sentence

Prompt:

```text
You are reviewing a narrow C inference runtime for a hybrid MoE model.

Write one paragraph identifying the highest-risk implementation area first, then give two concrete checks to validate it.
```

Suggested compare:

- `--n-predict 48`

### `patch_plan`

Purpose:

- planning/report style close to real coding-agent output
- shaped as an unfinished numbered plan

Prompt:

```text
We have validated the GGUF contract and tensor binding for a narrow Qwen3.6 runtime.

Write a short implementation note that names the next three runtime tasks in order and briefly justifies each one.
```

Suggested compare:

- `--n-predict 64`

### `router_dispatch`

Purpose:

- routed-expert sensitivity probe
- tries to keep the continuation near expert selection, dispatch, and top-k
  semantics

Prompt:

```text
Routing note:
The most failure-prone part of MoE inference is selecting and dispatching the top-k experts because
```

Suggested compare:

- `--n-predict 24`

### `hybrid_bridge`

Purpose:

- load-bearing path probe
- aims at the always-touched bridge between hybrid mixer state and residual flow

Prompt:

```text
Hybrid block note:
Quantization damage compounds fastest when the attention-SSM bridge path is unstable because
```

Suggested compare:

- `--n-predict 24`

### `late_layers`

Purpose:

- late-sensitive probe
- aimed at the policy question of whether final routed-expert blocks deserve
  extra protection

Prompt:

```text
Late-layer note:
Protecting the final few expert-down projections can matter disproportionately because
```

Suggested compare:

- `--n-predict 24`

### `final_selection`

Purpose:

- late-tail probe aimed at end-of-network choice refinement
- tries to expose cases where small late-path shifts change the selected token

Prompt:

```text
Final-selection note:
The last few expert projections matter most when the model is choosing between several plausible next tokens because
```

Suggested compare:

- `--n-predict 24`

### `logit_commit`

Purpose:

- late-tail/logit sensitivity probe
- aimed at the idea that small final-layer changes can change greedy output

Prompt:

```text
Logit-commit note:
Small activation errors near the end of the network can flip the chosen next token because
```

Suggested compare:

- `--n-predict 24`

## Recommended progression

- First prove `Q8_0` vs `Q4_K_XL` on `hello_world`
- Then prove `Q8_0` vs `Q4_K_XL` on `quant_report`
- Then prove `Q8_0` vs `Q4_K_XL` on `code_review`
- Then use the role-sensitive prompts:
  - `router_dispatch`
  - `hybrid_bridge`
  - `late_layers`
  - `final_selection`
  - `logit_commit`
- After that, add `Q2` and compare all three against the same prompt set

## What to look for

- exact prompt tokenization match
- first divergence step in greedy decode
- whether divergence is shallow logit drift or an immediate path split
- whether coding/report prompts diverge earlier than trivial prose prompts
- whether routed-expert prompts diverge differently from load-bearing prompts
- whether late-layer prompts justify targeted late-block protection
- whether end-of-network choice prompts strengthen the late-tail hypothesis
