# Is `d_eff` converged at n=90? And a valid resampling test for the substrate contrast

**Author:** Lyra · **Date:** 2026-08-22 · **Status:** DESIGN, pre-gate
**Compute:** none. Reads two cached activation files. No GPU, no model load, no MTH.
**Supersedes the resampling method of:** `substrate_aware_rerun.md` (its bootstrap is invalid; see below)

---

## Why the previous run's numbers are not reportable

`substrate_aware_rerun.md` specified a **bootstrap with replacement** over prompts. It ran, and
its output contained its own refutation: on the hybrid model at degree 8, `T_obs = +0.8816` with
CI `[+0.4201, +0.7162]` — **the point estimate outside its own confidence interval.** That is the
exact defect class `meridian/provenance.py` was written to catch, firing on my own result.

Diagnosed and quantified on the cached activations, layer 21:

| | `d_eff` |
|---|---|
| full sample, n=90 | **39.675** |
| bootstrap **with replacement** | **27.161** (bias **−12.51**, −32%) |
| mean distinct prompts per draw | **57.0** of 90 |

`d_eff` is the participation ratio of a covariance estimated from n samples. It is **a function
of n**. Resampling with replacement leaves ~63% distinct prompts, so every draw estimates a
different quantity than the full sample. The bootstrap distribution is therefore not a sampling
distribution for the estimand, its percentile interval need not contain the point estimate, and
the `p = 0.0000` values it produced were measuring bias rather than signal.

The Phase C guard reported `UNIQUE = 1000` and passed. **Uniqueness is not validity** — the null
draws were all distinct *and* all wrong. That is a genuine limit of the guard I added yesterday,
and this design adds the guard that would have caught it.

## What this design does

### Experiment A — the n-convergence curve

For each model, each layer `l`, and each `m` in **{20, 30, 40, 50, 60, 70, 80, 89}**: draw
`B = 200` subsamples of size `m` **without replacement**, compute `d_eff`, record mean and sd.

Every draw at a given `m` estimates the same quantity, so this is a valid characterisation of how
`d_eff` depends on sample size. `m = 90` is excluded because there is exactly one such subsample
(sd = 0, verified); `m = 89` is the largest with any variation.

**The question:** is `d_eff` at n=90 an estimate of a converged quantity, or is it still tracking
the prompt count? `d_eff` is bounded above by `n − 1 = 89`, and the L21 point estimate is 39.7 —
44% of that ceiling.

### Experiment B — fixed-`m` subsampling test for the substrate contrast

At **`m = 60`**, `B = 1000` draws without replacement. Within each draw, compute the full 64-layer
profile, detrend (OLS degree 4/6/8, unchanged), and compute
`T = mean(resid | full-attention) − mean(resid | linear)`. Report the percentile CI of `T`.

**Why fixed-`m` fixes it.** Within a draw, every layer is estimated from the *same* `m` prompts,
so any global n-dependence shifts all layers together and the degree-6 detrend removes it. The
contrast is a within-draw quantity. And because the reported statistic is the distribution's own
centre rather than a separately-computed n=90 point estimate, the "observed value outside its own
CI" pathology cannot recur by construction.

`m = 60` is chosen because it is close to the 57.0 effective distinct count the invalid bootstrap
was implicitly using, so the two are comparable, and it leaves 30 prompts of variation per draw.
Sensitivity at `m = 45` and `m = 75` is pre-specified and will be reported.

### Controls, unchanged from the prior design

- **Hybrid:** Qwen3.5-27B, true substrate mask, asserted `== [3,7,…,63]` from the shipped config.
- **Depth- and width-matched dense control:** Qwen3-32B — 64 layers, `d_model` 5120, and
  `layer_types` reporting **64 × `full_attention`, 0 linear**. (Dense models do not omit
  `layer_types`; they label every layer full-attention. The prior run asserted the opposite and
  failed its own control.) Sham mask `l ≡ 3 (mod 4)`.
- All four phase classes reported, not just the substrate phase.

## New blocking guards

Both run as assertions, not notes. Each exists because something got past its predecessor.

1. **Unique-count of every null** (kept from the prior design — caught the degenerate period-4 null).
2. **NEW — containment check:** assert the reported point statistic lies **inside** its own
   reported interval. If it does not, the resampling scheme is not estimating the estimand. Print
   `FAIL` and suppress the p-value rather than reporting both numbers side by side.
3. **NEW — estimand-invariance check:** assert every draw contributing to one interval used the
   **same `m`**. Report `m`, `B`, and the count of distinct prompt-sets actually drawn.

## Threat I have to declare

**I have already seen the m-ladder at layer 21** — 60→32.2, 75→36.2, 90→39.7 — while diagnosing
the bootstrap. That single-layer observation is what motivates Experiment A, so it is **not
independent evidence** for it. Consequences adopted:

- The pre-specified analysis is **all 64 layers in both models**, not L21, and the convergence
  verdict is a statement about the profile, not about the layer I happened to look at.
- If the effect exists only near the peak, that is reported as such and is a weaker claim.
- Experiment B was designed before any of its numbers were seen and is unaffected.

## Outcomes, decided in advance

**Experiment A**
- **`d_eff` still rising at `m = 89`** in most layers → our `d_eff` values are sample-size-limited.
  Every `d_eff` number in the campaign becomes an *n=90 estimate* rather than a property of the
  model, and the workspace band's height — possibly its shape — is partly a prompt-count artifact.
  This would be the most consequential finding of the campaign so far and it would apply
  retroactively.
- **Curve flattening by `m ≈ 70–89`** → n=90 is adequate, existing values stand, and the band is
  a property of the model.
- **Layers converging at different rates** → worse than either, because then the *profile shape*
  is n-dependent and peak location is not comparable across layers at fixed n.

**Experiment B**
- **CI excludes zero on the hybrid and includes zero on the dense control** → a real substrate
  effect, and yesterday's null was underpowered.
- **Both include zero** → yesterday's null confirmed by a valid method.
- **Dense control excludes zero** → the detrend or the estimator manufactures phase structure at
  n=64 layers; report that and nothing else.

I expect A to show non-convergence and B to confirm the null. Both are stakes; both are recorded
here before running.

## Provenance for the two claims the gate could not verify

Agni returned **WARN / pass / 0.92** (11 pass, 2 warn, 0 downgraded). Both warns were
`unverifiable` — the gate runs on MTH, the artifacts are on Starship.

| claim | evidence |
|---|---|
| caches reproduce July to six decimals | `starship:~/lab/kv-experiments/acts_qwen35-27b.npz` (55,207,073 B, 2026-08-22 19:31) and `acts_qwen3-32b.npz` (55,359,232 B, 20:35), each `(90, 64, 5120) float32`. Recomputing `d_eff` from the fresh 27B cache gives layers 0–3 = `[1.286234, 1.624802, 3.947641, 5.746237]`; the July artifact `r3_band_measurement_20260714_170944.json` holds `[1.2862343788146973, 1.6248023509979248, 3.947641372680664, 5.746236324310303]`. Independent forward pass, 39 days apart, same values. |
| Gram equivalence and speedup | Measured on the 27B cache, 8 layers: covariance form 55.27 s, Gram form 0.007 s → **7497×**; max absolute difference 2.480e-05, max relative 2.253e-06. Projection to 1000 resamples × 64 layers: **122.8 h** vs **1.0 min**. Benchmark retained in this design's runner, `deff_convergence.py::_bench`, so it re-verifies on every execution rather than resting on this table. |

The gate's third required change — report all three blocking guards with explicit PASS/FAIL —
is implemented in the runner and printed before any interval.

## Non-goals

- Not re-running any model. Both activation caches exist and were verified to reproduce the July
  `d_eff` profile to six decimals.
- Not changing the estimator. `d_eff` = `(Σλ)²/Σλ²` on covariance eigenvalues, computed via the
  n×n Gram matrix — same nonzero spectrum, verified to 2.3e-06 relative, 7497× faster.
- Not touching MTH.
- Not reinterpreting the prior run's intervals. They are withdrawn, not re-analysed.
