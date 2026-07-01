# GRACE — Grouped Residual Attention over Composed dEpth

A small, fast transformer variant for local inference (gaming GPUs / MLX). This
document is the source of truth for what GRACE is and why it is built this way.
Read it fully before writing or changing architecture code.

---

## Goal: latency, not FLOPs

The target is **wall-clock latency for small (<1B) models on local hardware**
(consumer gaming GPUs and Apple Silicon via MLX), typically at batch-1 or
small-batch inference.

Small models don't fail because they lack FLOPs — they fail because they
**can't saturate the hardware**. A standard transformer is a long, thin,
*sequential* dependency chain: attn → mlp → attn → mlp → ... down the depth.
Every sublayer is a separate step on the critical path (kernel launch +
a small matmul). At small `d_model`, launch/latency overhead is a large
fraction of the total, and the GEMMs are too small to fill the device. So the
machine sits idle while we wait on a deep chain of tiny ops.

**GRACE trades sequential depth for parallel width that the hardware can
actually fill.** Instead of many thin sequential sublayers, we run fewer, wider,
*parallel* groups per layer and fuse them into a few large batched kernels. The
critical path gets shorter; each step does more work; utilization goes up.
Success is measured in **tokens/sec at fixed quality**, not in parameter count
or FLOPs.

---

## The naive version and why it collapses

The obvious way to shorten depth: take the N blocks that would have been
sequential and run them **in parallel** instead — a bank of attn blocks side by
side, a bank of mlp blocks side by side — then reduce the depth.

**This does not work.** Parallel blocks all receive the *same* input tensor
(nothing feeds one from another — they're parallel). So block `i` computes
`f_i(x)` and block `j` computes `f_j(x)` on identical `x`. A sum of independent
projections applied to the same input is algebraically identical to **one wider
projection**:

```
sum_i  W_out_i · σ(W_in_i · x)   ==   one wide block with concatenated weights
```

For attention it's the same story: parallel attn blocks over identical input is
just multi-head attention with more heads, i.e. one wider block. So the parallel
group **collapses into "one bigger block"** and buys nothing over simply
widening. Parallelism killed the depth-wise expressivity — which was the whole
point of depth. This is the same reason parallel attn+MLP blocks
(GPT-J / PaLM style) are cheaper but not more expressive per layer.

**Do not implement the naive parallel version. It is a known dead end.**

---

## How GRACE solves the collapse

Give each parallel block a **different input**, without reintroducing a
sequential chain.

Stop treating the residual stream as a single merged tensor. The state is a
**pool**: the set of *all previous block outputs* from all previous layers,
kept separately (not summed into one stream). Each block in the current layer
owns its own **query** and computes its own input as **attention over the whole
pool**. Its output is **appended to the pool** rather than merged in.

Now two parallel blocks in the same layer are genuinely different functions of
different inputs — each composed a different learned mixture of history via its
own query — so they **cannot fold into one wide block.** We recover depth-wise
expressivity while keeping the blocks within a layer parallel (they depend only
on *previous* layers, never on each other).

The network is now a **DAG over blocks with edges selected by attention**,
instead of a chain.

### Homogeneous layers → full fusion (the key implementation property)

**Each layer is a single block type.** Layers alternate: an all-attn layer,
then an all-mlp layer, then all-attn, etc. Never mix types within a layer.

Because a layer is one type with `G` parallel groups, don't think "group of
separate blocks" — think **"one block with a group axis," i.e. multihead
input.** This makes the entire layer fuse into a few batched kernels:

- **mlp layer:** `G` groups → one batched GEMM `(G, d) → (G, d_ff) → (G, d)`,
  a single matmul with a leading group dimension. No per-group kernels.
- **attn layer:** `G` groups is structurally multi-head attention with the
  group as an outer head axis — an existing, well-optimized op.
- **the depth-attention (the queries):** stack all `G` queries into a `(G, d)`
  tensor and do **one** batched attention-over-pool:
  `(G, d) × (pool, d)^T → (G, pool)`, softmax over the pool axis, one
  weighted-sum. The per-block depth-attention collapses into a single grouped
  op — not `G` tiny softmax reductions.

So a full layer ≈ **one grouped depth-attention + one grouped block op**. That
is the whole architectural bet: convert sequential depth into a group axis the
hardware fills, and let each group's own query keep the groups distinct.

---

## AttnRes background (and how GRACE relates to it)

GRACE's depth-attention mechanism is adapted from **Attention Residuals
(AttnRes)** by Moonshot AI's Kimi Team (2026). AttnRes replaces standard PreNorm
residual connections — which accumulate every layer's output with fixed unit
weights, causing hidden-state magnitude to grow with depth and diluting each
layer's contribution — with **softmax attention over preceding layer outputs**.
Each layer gets a learned pseudo-query; keys/values are the RMS-normalised
outputs of all previous layers plus the token embedding, so each layer builds
its input as a learned, input-dependent weighted sum over depth. A useful
detail we inherit: **queries are zero-initialized**, so at the start of training
attention is uniform over source layers (equal-weight averaging), which avoids
early instability. GRACE reuses this exact routing idea — per-block query,
softmax over a pool of prior outputs, zero-init — as the mechanism that keeps
parallel groups distinct.

> **IMPORTANT — do not benchmark GRACE against AttnRes directly.**
> AttnRes was designed for **deep networks**, with the stated goal of reducing
> **hallucination** and PreNorm dilution at scale. GRACE borrows only the
> **routing behaviour** (learned attention over a pool of past block outputs) as
> a tool to defeat the parallel-collapse in **small, shallow, latency-bound**
> models. The goals, regime, and success metrics are different (their target:
> quality at depth; ours: tokens/sec at fixed quality when small). Do **not**
> treat AttnRes results, ablations, or scaling claims as a baseline or target
> for GRACE, and do not "fix" GRACE to match AttnRes behaviour. If a comparison
> is needed at all, compare GRACE to a **standard small transformer at equal
> FLOPs**, on **wall-clock latency**.

---

## Implementation notes

Build the **smallest thing that demonstrates the mechanism**, before any
deployment thinking. Correctness of the pool/query/append logic first, speed
second. Start with 50M baseline vs 50M GRACE

**Pool layout (matters, especially for MLX):**
- Store the pool as a **single preallocated contiguous `(max_pool, d)` buffer**
  and **write into slices** as blocks append. Do NOT keep a Python list of
  per-block tensors and `stack` every layer — stacking = copy = graph friction,
  and on MLX it breaks fusion.
- RMS-norm the pool entries when used as keys (as AttnRes does), so a single
  query can attend over the heterogeneous pool (attn-type and mlp-type outputs
  from many depths all share the `d`-space).

**Per-layer forward (homogeneous):**
1. grouped depth-attention: `(G, d)` queries × pool → `(G, pool)` → softmax over
   pool → weighted sum → `(G, d)` composed inputs.
2. grouped block op: one batched attn **or** one batched mlp GEMM over the
   `(G, d)` inputs.
3. append the `G` outputs into the pool buffer.

**Zero-init all queries.** Start = uniform averaging over the pool; let them
differentiate. Expect rough training if you skip this.

**Full pool** is fine at this size — <1B, so the pool is ~hundreds of entries,
cheap in bytes. Every group attends the full pool (no partitioning) for the
first build.

**Metric:** validation loss and tokens/sec at equal steps and parameter count.

---

## One-line summary (if you need to restate the idea)

Homogeneous alternating layers (all-attn / all-mlp), `G` parallel groups per
layer fused as a leading axis, each group with its own zero-init query that
attends over a full contiguous pool of all previous block outputs; outputs
append to the pool instead of merging. The per-group query is what stops the
parallel groups from collapsing into one wide block.
