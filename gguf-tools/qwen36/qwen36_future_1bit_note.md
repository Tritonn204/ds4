# Qwen3.6 Future 1-Bit Note

Date: `2026-06-21`

## Question

If selective `Q2` can work because the runtime protects load-bearing tensors and only crushes the redundant MoE mass, could a stricter selective `1-bit` or `1.58-bit` contract also work?

## Short answer

Yes, in principle.

But it would likely be a narrower and harsher contract than the current `Q2`
one:

- more tensor families would need to stay above the ultra-low-bit floor
- only the least load-bearing routed-expert subset would be a plausible target
- validation would need to be redone from scratch
- a new storage format and kernels would be required anyway

This is the same kind of architectural question as the current `Q2` work, not a
generic “BitNet magic transfers automatically” question.

## Why this is structurally coherent

The working hypothesis behind the current Qwen DS4-style plan is:

- keep the always-touched path higher precision
- crush the mostly-routed MoE mass aggressively
- rely on MoE redundancy plus high-precision re-anchoring between low-bit
  regions

That logic does not stop at `Q2`.

In principle, it can be pushed further to `1-bit` or `1.58-bit` on a stricter
sub-bucket of tensors if the architecture tolerates it.

The real question is not:

- “can a `1-bit` file exist?”

It is:

- “which tensors can survive that floor while preserving behavioral parity?”

## Why this is a future experiment, not the first target

The current sequence should remain:

1. prove the selective `Q2` contract
2. identify which tensors inside the low-bit bucket are safest
3. probe whether a smaller subset can collapse further to `1-bit` / `1.58-bit`

So `1-bit` belongs after the first selective-`Q2` contract is behaviorally
stable.

## Why the current Qwen export already shrank so much

The important distinction is:

- selective by **role**
- not selective by **parameter count**

The current `v0` policy keeps the load-bearing path at higher precision, but it
pushes the routed-expert tensors aggressively:

- `ffn_gate_exps.weight` -> `IQ2_XXS`
- `ffn_up_exps.weight` -> `IQ2_XXS`
- `ffn_down_exps.weight` -> `IQ2_S` or `IQ2_XXS`

Those tensors are exactly where most of the model’s parameter mass lives.

So even though the policy is “selective,” it is selective over a very large
fraction of the total bytes.

That is why the total size reduction can be dramatic:

- the protected path is small in parameter count but high in importance
- the crushable path is huge in parameter count but sparse in per-token use

This is the same DS4-style asymmetry:

- load-bearing path: small, resident, protected
- routed-expert mass: huge, compress hard, potentially stream

## Practical implication

A future `1-bit` experiment would not try to replace the whole current low-bit
bucket.

It would ask a narrower question:

- inside the current MoE expert bucket, is there a further sub-bucket that can
  be pushed below `Q2` without changing greedy behavior on the validation
  prompts?

That is the right future framing.
