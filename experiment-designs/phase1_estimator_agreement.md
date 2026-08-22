# Phase 1: Estimator Agreement on a Dense Model

**Status:** DESIGN — pending Agni gate
**Campaign:** MoE Geometry (Phase 1)
**Question:** Do our spectral estimators agree about workspace depth when computed on *identical* matrices from a *dense* model?

---

## Why this exists

On GLM-5.2 (MoE, MLA latent) three estimators gave three different depth
profiles: `d_eff` peaks at 33% (Qwen), stable rank at 10.4%, MP-edge count at
2.6%. I began constructing an MoE explanation **without first checking whether
the estimators disagree on a dense model too.**

If they agree on dense, the GLM disagreement is architecture-specific and is a
finding. If they disagree on dense as well, the problem is estimator choice, not
architecture — and the convergence paper's "workspace peaks at 31–33%" becomes
estimator-dependent, which materially weakens a claim currently in review.

This is the control I skipped. It gates Phase 4 and it is a live risk to a paper.

## Design

**Model:** Qwen3-8B — 36 layers, all full-attention, dense FFN. No MoE routing,
no hybrid substrate, no MLA compression. The clean case.
(Qwen3.5-27B is the other dense-FFN option but needs ~54GB; Starship has ~24GB
free with Nemotron up. 8B in bf16 fits and answers the question.)

**Corpus:** the identical 90 prompts used by `r3_band_measurement.py`
(60 prose + 30 chat), so the recomputed `d_eff` must reproduce the published
profile exactly. That reproduction is itself a check on the pipeline.

**Extraction:** `out.hidden_states[l+1][0, -1, :]` for every layer
`l in 0..35`, one vector per prompt. Identical to the `d_eff` extraction path.
**Save the raw H matrices** (36 × 90 × 4096) so every estimator is computed on
byte-identical inputs and the run is never repeated for a new estimator.

**Estimators**, all applied to the same mean-centred `H_l` (90 × 4096):

| estimator | formula | note |
|---|---|---|
| `d_eff` (participation ratio) | $(\sum\lambda)^2 / \sum\lambda^2$ on covariance eigenvalues | the published one |
| stable rank | $\sum s^2 / s_\max^2$ | what I used on GLM |
| spectral entropy | $\exp(-\sum p\log p)$, $p_i = s_i^2/\sum s^2$ | |
| MP-edge count | #$\{s^2 > \sigma^2(1+\sqrt\gamma)^2\}$, $\sigma^2 = \text{median}(s^2)$ | **CC's estimator, not my MAD version** |
| middle-band count | between MP edge and Gavish–Donoho threshold | CC's Exp-51 statistic |

**Also uncentred** for `d_eff` and stable rank, since centring is a free choice
that could itself drive disagreement.

## Pre-registered primary endpoint

**Do the estimators agree on peak location?**

Agreement is defined *before* looking: two estimators agree if their argmax
layers fall within **±2 layers** (±5.6% relative depth on a 36-layer model,
comfortably inside the paper's own "<5% relative depth" cross-scale tolerance).

- All five agree → GLM disagreement is architecture-specific. Phase 4 proceeds.
- Any estimator disagrees on dense → the problem is estimator choice. The
  convergence paper's depth claim needs an explicit estimator qualification, and
  the GLM result cannot be read as an architecture finding.

## Secondary

- Spearman rank correlation between full 36-layer profiles, all pairs.
- Peak-location bootstrap: resample the 90 prompts with replacement (2000×),
  recompute each profile, report a CI on each argmax. **This is the error bar the
  convergence paper does not have** and it applies directly to the claim in review.
- Centred vs uncentred comparison.

## Controls

1. **Random-matrix null.** Same shapes (90 × 4096), i.i.d. Gaussian. Every
   estimator must return a flat profile. Any estimator showing structure on
   noise is measuring its own shape, not the model's.
2. **Reproduction check.** Recomputed `d_eff` must match
   `r3_band_measurement_20260714_171708.json` to rounding. If it does not, the
   extraction differs from the published one and nothing downstream is
   comparable.
3. **Shuffle control.** Shuffle prompt labels within layer; profiles should be
   unchanged (the estimators are label-free), confirming no leakage in the
   harness.

## Known problem: aspect ratio (flagging before the gate finds it)

`H_l` is 90 × 4096, so $\gamma = 4096/90 \approx 45.5$ — **worse than the GLM
case (16.5) that already broke my MP estimator.** The covariance has rank ≤ 89,
so at most 89 non-zero eigenvalues out of 4096.

Consequences:
- MP-edge and middle-band counts may be unreliable here for the same reason they
  were on GLM. If they fail, that is informative — it says those estimators are
  not usable at our aspect ratios, which is a Phase-4 constraint — but it must
  not be read as "the estimators disagree about the model."
- `d_eff` and stable rank are unaffected by rank deficiency in the same way; both
  are ratios over the non-zero spectrum.
- CC's stacking fix does not apply: `d_eff` already uses all 90 prompts. The only
  way to improve γ is more prompts.

**Pre-registered handling:** report MP-edge and middle-band results separately
from `d_eff`/stable-rank/entropy, and treat estimator disagreement involving
only the MP-family as an *aspect-ratio* finding, not an architecture finding.

## Falsification

- Estimators agree on dense AND on random-matrix null they are flat →
  GLM disagreement is real and architecture-related.
- Estimators disagree on dense → the depth claim is estimator-dependent;
  the convergence paper needs qualification. This is the outcome I do not want,
  which is the reason to pre-register it.

## Seeds & environment

`torch.manual_seed(42)`, `np.random.seed(42)`, bootstrap `RandomState(42)`.
Starship, `~/miniforge/envs/oracle/bin/python3`, bf16, MPS.
Model: `Qwen/Qwen3-8B` (base — matching the R3 provenance, not the distill).

## Cost

One extraction (~10 min on 8B), then all analysis is offline on saved H.
No further GPU for any additional estimator.
