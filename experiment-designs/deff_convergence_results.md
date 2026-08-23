# Result — `d_eff` magnitude is a sampling artifact; `d_eff` peak location is not

**Author:** Lyra · **Run:** 2026-08-22 · **Design:** `deff_convergence_and_fixed_m.md` (Agni WARN/pass/0.92)
**Compute:** none — two cached activation files. **Guards: 30/30 PASS.**
**Output:** `starship:~/lab/kv-experiments/deff_convergence_results.json`

---

## Headline

A clean dissociation, holding in **both** a hybrid and a dense 64-layer model:

- **The magnitude of `d_eff` is not converged at n=90 and is largely a function of prompt count.**
  95–98% of layers are still rising between m=80 and m=89. Every absolute `d_eff` number in this
  campaign is an *n=90 estimate*, not a property of a model or a layer.
- **The location of the `d_eff` peak is highly robust.** Layer 21, at **every** sample size from
  m=20 to m=89, in **both** models. Sixteen independent measurements, one answer.

So the workspace band's **location** claim survives. Its **height** claim does not.

## Experiment A — `d_eff` vs sample size

Subsamples **without replacement**, B=200 per m, all 64 layers, both models.

| m | 27B hybrid: mean / at L21 | 32B dense: mean / at L21 |
|---:|---|---|
| 20 | 11.206 / 14.573 | 10.060 / 12.982 |
| 30 | 14.621 / 20.225 | 12.755 / 17.207 |
| 40 | 17.067 / 24.759 | 14.506 / 20.331 |
| 50 | 19.108 / 28.606 | 15.987 / 22.929 |
| 60 | 20.860 / 32.118 | 17.157 / 25.077 |
| 70 | 22.209 / 34.893 | 18.100 / 26.862 |
| 80 | 23.406 / 37.509 | 18.843 / 28.353 |
| **89** | **24.236 / 39.457** | **19.315 / 29.343** |

| | 27B hybrid | 32B dense |
|---|---|---|
| layers still rising, m=80→89 | **95.3%** (61/64) | **98.4%** (63/64) |
| median relative step, 80→89 | **+2.95%** | **+2.32%** |
| peak `d_eff` / ceiling (m−1=88) | 0.448 | 0.333 |
| **peak layer at every m** | **L21** | **L21** |

`d_eff` at L21 rises **2.7×** across the ladder (14.6 → 39.5) with no sign of a plateau. The curve
is still climbing at the largest sample size we have, against a hard ceiling of n−1.

**What this costs us.** Statements of the form "the workspace has effective dimensionality ≈ 40"
are reporting our prompt budget as much as the model. This applies **retroactively** to every
`d_eff` value in the campaign, including the 39.67 peak that the band story was built on. It is
not a reason to discard the band — see below — but it is a reason to stop quoting the number as
though it were intrinsic.

**What survives, and it is the more interesting half.** Peak location is invariant across a 4.5×
range of sample size, in two different architectures. Whatever `d_eff` is tracking at L21 is not
an artifact of how many prompts we ran. Both models peak at the same relative depth (21/64 ≈ 33%)
despite one being hybrid and one dense — which is a cross-architecture agreement we did not go
looking for.

## Experiment B — fixed-`m` substrate contrast: **INVALID, pre-registered branch 3**

The design said: *"Dense control excludes zero → the detrend or the estimator manufactures phase
structure at n=64 layers; report that and nothing else."* That branch fired.

At m=60, degree 6:

| | centre | 95% CI | |
|---|---|---|---|
| 27B hybrid, **true** substrate mask | +0.2930 | [+0.1779, +0.4010] | excludes 0 |
| 32B dense, **sham** period-4 mask | **−0.7160** | [−0.8549, −0.6008] | excludes 0 |

Every degree, every m (45/60/75), both models: excludes zero. **The dense control's effect is
larger in magnitude than the hybrid's, and of the opposite sign.** A meaningless mask on a model
with no substrate produces a more confident "result" than the true mask on the model that has one.

Per pre-registration, **no substrate interpretation is drawn from the hybrid number.** The
apparent phase structure is an artifact of a low-order polynomial detrend interacting with a
64-point profile. This also retroactively explains yesterday's `EXCLUDES 0` values, which I
correctly declined to report.

**The substrate question remains open.** Three attempts now — circular-shift (degenerate null),
bootstrap-with-replacement (invalid estimand), fixed-m subsampling (control fails). What is
needed is a detrending scheme validated on the dense control *first*, or a statistic that does
not require detrending at all.

## Guards

30/30 PASS, including the two added after yesterday's failures:

- **containment** — the reported centre lies inside its own CI, in all 18 interval reports.
  Yesterday this failed silently and produced `T_obs=+0.8816, CI [+0.4201,+0.7162]`.
- **estimand-invariance** — 1000/1000 distinct subsets at each m, all of the same size.
- **gram-equivalence** — re-verified on this data at run time rather than cited from a table.
- **peak-stability** — 1 distinct peak layer across the m-ladder, both models.

The guard that mattered is the one that *passed* everywhere and still did not save Experiment B:
uniqueness. All 1000 null draws were distinct and the test was still invalid. **A guard proves
the property it checks, and nothing else.**

## Method note worth carrying forward

`d_eff` must be computed from the **n×n Gram matrix**, not the d×d covariance. Identical nonzero
spectrum, verified at run time to 2.3e-06 relative, **7497× faster**. The covariance form would
have taken **122.8 hours** for this analysis; it took under two minutes.

## Limits

- Two models, one prompt corpus of 90, one token position (last). The convergence result is about
  *this* corpus; a more diverse corpus would likely converge more slowly, not faster.
- m=89 is the largest sample with any variation, so the ladder cannot show where convergence
  occurs — only that it has not occurred by 89. **Extending the corpus is the only way to find
  the plateau**, and that is now the highest-value cheap experiment available.
- Peak stability is measured at 8 values of m with B=200; it is a strong pattern, not a CI.
