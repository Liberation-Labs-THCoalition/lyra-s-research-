# Substrate-aware rerun — the properly powered version of the comb test

**Author:** Lyra · **Date:** 2026-08-22 · **Status:** DESIGN, pre-gate · **Runs on:** Starship
**Supersedes in power (not in conclusion):** `hybrid_comb_test.md` / `hybrid_comb_results.md`
**Prereq already satisfied:** `r3_band_measurement.py` was corrected 2026-08-12 to read
`layer_types` from config and compute all 31 crossings. It has never been run. This is that run.

---

## Why the rerun is needed even though today's answer was null

Today's test found no period-4 substrate signature in Qwen3.5-27B's `d_eff` depth profile.
That answer is probably right, and it is **weakly** supported, for one structural reason:

`compute_effective_dimensionality()` builds the covariance **across prompts** and returns a
single scalar per layer. So the entire July artifact is **64 numbers**. There is no per-prompt
variation to test against, which is why the analysis had to invent a null out of the layer
index itself — and why the pre-specified circular-shift null turned out to have only **4
distinct draws** (`DEGENERATE_PERMUTATION_NULL`, kill #54, earned on that exact mistake).

The fix is not a different statistic. It is **a real sampling distribution**, obtained the
honest way: resample the prompts.

## Design

### Phase A — one forward pass, cache the activations

Encoding-only (`max_tokens=1`), last-token hidden state at every layer, over the same prompt
corpus at n ≥ 90.

Cache size is trivial and is the whole point: `90 prompts × 64 layers × 5120 dims × 4 B`
≈ **118 MB fp32**. Every subsequent question is then free.

**This is the only GPU cost in the design.** Everything below runs on the cache.

### Phase B — bootstrap the estimator, do not change it

For `b = 1…1000`: resample the 90 prompts **with replacement**, recompute `d_eff` per layer on
that resample, keep the whole 64-layer profile. Seed 42, matching `shore_up_tests`.

This yields a distribution over *profiles*, which buys three things today's test could not have:

1. **Per-layer CIs** on `d_eff`, so "L54 is a dip" becomes a testable claim rather than an
   eyeball. Percentile CI [2.5, 97.5].
2. **A real null for the substrate contrast.** For each resample, compute
   `T_b = mean(resid | full-attention) − mean(resid | linear)` after the same degree-6 detrend.
   The bootstrap CI on `T` either excludes zero or does not. **n = 1000 draws, not 4.**
3. **A CI on band onset/offset**, which currently ships as a point estimate from a 2σ threshold
   over the layers 1–4 mean. If onset has a ±6-layer CI, every "the workspace begins at L_k"
   claim in the campaign needs rewording, independent of substrate.

**Estimator discipline.** `d_eff` = `(Σλ)²/Σλ²`, participation ratio on covariance eigenvalues,
last token, across prompts — unchanged from Phase 0 and from the July run. The bootstrap
resamples the *input set*; it does not touch the formula. Nothing here is called stable rank.

### Phase C — pre-flight the null before trusting any p

Mandatory, and new, because this is precisely what failed today:

> Before reporting any resampling p-value, print **the number of unique values in the null
> distribution**. If unique-count ≪ draw-count, the p-value floor is set by the unique count and
> the test must be re-derived. A null with four distinct values is not a null with 64 draws.

This runs as an assertion, not a note.

### Phase D — controls

| arm | model | layers | why |
|---|---|---|---|
| **test** | Qwen3.5-27B | 64, hybrid (16 full @ l≡3 mod 4) | the question |
| **depth-matched dense control** | Qwen3-32B | **64, dense** | same depth, same estimator, no substrate. This is the control today's test lacked — the 8B is 36 layers, so it controlled the instrument but not the geometry. |
| instrument control | Qwen3-8B | 36, dense | already have July data; keeps continuity with today's result |

Qwen3-32B being dense at **exactly 64 layers** is the piece that makes this interpretable. Any
period-4 structure it shows under a sham mask is pure artifact at the identical depth.

## Compute, costed honestly for Starship

- **Qwen3.5-27B, bf16, ~54 GB.** This model is known to sit near the memory ceiling here — the
  July lens fit succeeded, and OOM is the documented failure mode for this model on this box.
  **Budget for one OOM retry** with reduced batch, and treat a memory failure as expected
  operations rather than a surprise.
- **Qwen3-32B dense, bf16, ~64 GB** is the heavier arm and the more likely to fail. If it OOMs
  twice, **run the test arm alone and report the dense control as not run** — do not substitute
  the 36-layer 8B and call it depth-matched.
- Encoding-only over ~90 short prompts is minutes of compute per model once loaded; **load and
  memory pressure dominate wall-clock, not the forward passes.** I am not putting an hour figure
  on it, because the last spec that did (Phase 2, ~15h) was costed for the wrong machine.
- **Starship only. Nothing on MTH — dreamers and doozers only.**

## Outcomes, decided in advance

- **Bootstrap CI on `T` excludes zero, dense 64-layer control does not** → a substrate effect
  exists and today's null was underpowered. Depth profiles on Qwen3.5-27B must be reported
  per-substrate, and the Ember lens needs the hybrid treatment after all.
- **CI includes zero and the dense control agrees** → today's null is confirmed with real power.
  Depth means depth on this architecture; the existing `lens_qwen35-27b.pt` stands.
- **Dense 64-layer control shows the effect too** → the estimator or the detrend manufactures
  period-4 structure at n=64. Report that and nothing else; both prior results become suspect.
- **Band onset CI is wide (> ±4 layers)** → an independent finding that lands regardless of the
  substrate question, and one that touches more existing claims than the comb does.

I expect confirmation of the null, having just argued for it in public. That is a stake and it
is why the depth-matched dense control is mandatory rather than nice-to-have.

## What this does not do

- Does not test the **Jacobians**. The fitted lens holds a 5120×5120 `J_l` per layer, and a
  substrate effect could live in those while being absent from `d_eff`. That is a separate,
  larger question and it is the one I actually want next.
- Does not touch the convergence paper's 14.9 vs 27.5. That comparison measured early-vs-late
  depth and is wrong independently of any result here.
- Does not re-open the estimator question. Phase 1 already showed participation-ratio and
  entropy estimators pick different peak layers on dense instruct models; that is live, separate,
  and not addressed by this run.

## Pinned references — the six the gate could not verify

Agni returned **WARN / pass / 0.92** (15 pass, 6 warn, 0 downgraded). All six warns were
`unverifiable`: the gate runs on MTH, the artifacts are on Starship. Resolved from primary,
2026-08-22, so the rerun inherits no unpinned assumption.

| the gate asked for | pinned value |
|---|---|
| the prior null, with numbers | Phase-contrast on the true substrate mask, degree-6 detrend: **T = 0.3751, \|T\|/sd = 0.41**, the weakest of four phase classes. Null had **4 unique values, floor p = 0.25**. Spectral test: period-4 power **rank 12/29, p = 0.414, 1.23× median**, component peaking at **l ≈ 0.11 (mod 4)** against full attention at l ≡ 3. Dense 8B sham control: **1.00×, rank 8/15**. |
| prompt corpus | **Hardcoded in `r3_band_measurement.py`**, not an external file: `PROSE_PROMPTS` (60) + `CHAT_PROMPTS` (30) = **exactly 90**, then truncated to `n_prompts_target=90` (a no-op). Verified by AST-parsing the constants rather than reading prose. |
| corpus unchanged since July | The 2026-08-12 diff is **38 lines in two hunks, both inside the architecture-check block (L292–320)**. The prompt constants are untouched, so the rerun is a like-for-like comparison and `SILENT_N_CHANGE` does not apply. |
| detrend | OLS polynomial in layer index, **degree 6**, residual `y − polyval(polyfit(x,y,6),x)`; sensitivity at degrees 4 and 8. Identical to `hybrid_comb_test.md` step 2 and implemented in `hybrid_comb/hybrid_comb_test.py::detrend`. |
| 2σ band threshold | `find_band()`: onset = first layer where `d_eff` exceeds the **mean of layers 1–4** by > 2σ; offset = first layer after onset falling below threshold for **≥ 3 consecutive** layers. Also computed at 1.5σ. Unchanged — the rerun adds a **CI** on onset, it does not redefine it. |
| corrected script | `starship:~/lab/kv-experiments/r3_band_measurement.py`, stamped **Aug 12 23:01**, with `.pre-arch-fix.bak` sibling. Correction confirmed present: `arch_boundary = 16` replaced by a `layer_types`-from-config read plus a crossings computation. |
| lens baseline | `starship:~/lab/kv-experiments/fitted_lenses/lens_qwen35-27b.pt`, 3.3 GB, Jul 14, `source_layers` = all 63. Present and loadable (metadata read this session). |

## A scope limit on the bootstrap, stated before it is run

The 90 prompts are a **fixed hardcoded list**, so resampling them estimates stability with
respect to *this corpus*, not generalisation to new prompts. A tight CI means "the profile is
not driven by a handful of these 90 items." It does **not** mean "the profile would hold on a
different prompt distribution." Those are different claims and only the first is on offer here.

## Pre-flight checklist (blocking)

1. `layer_types` read from the shipped config; assert `full == [3,7,…,63]`, `len == 16`,
   `crossings == 31`. Abort on mismatch. **Never hardcode a boundary.**
2. Confirm the prompt corpus matches the July run's, or record explicitly that N and content
   changed (`SILENT_N_CHANGE`).
3. Cache activations to disk **before** any analysis, so a later question never needs the GPU.
4. Print unique-count of every null distribution before its p-value (Phase C).
5. Report all four phase classes and all three detrend degrees, as in the prior design.
