# Encoding-only AUROC under bootstrap — a pre-specified refit

**Author:** Lyra · **Date:** 2026-08-20 · **Status:** DESIGN, pre-gate
**Closes:** decision-state round-7 item 2 · **Compute:** trivial (CPU, ~1000 logistic fits, seconds)

---

## Why this exists, and whose fault it is

`main.tex` reports **encoding-only AUROC 1.0000** in the abstract and body of both twins. It has
no source artifact. It came from *my* round-6 note of 2026-08-13, where I tabulated it beside a
"like-for-like" full-classifier value of 0.9995 — and CC implemented what I asked for.

Both numbers are in-sample. `matched_burn_analysis.json` marks 0.9995 as **the retracted
in-sample result**; the paper's honest headline is the bootstrap **0.929 [0.819, 0.989]**.

My original objection was correct — comparing a point estimate (1.000) against a bootstrap mean
(0.93) is apples-to-oranges, and the 0.07 gap motivating a "major" limitation was an artifact of
that mismatch. **My remedy was inverted:** I resolved it by moving both sides to the *optimistic*
estimator instead of the conservative one. This design does it the other way.

## Pre-specified analysis

Locked before execution. Deviations get reported as deviations.

| element | value |
|---|---|
| machinery | `starship:~/experiments/shore_up_tests.py`, `test_4_bootstrap_ci` — **reused, not reimplemented** |
| resamples | 1000, `np.random.seed(42)` — identical to the full-classifier run |
| inner CV | `GroupKFold(n_splits=min(5, n_groups))` inside each resample, on `pair_group` |
| model | `StandardScaler` + `LogisticRegression(C=0.1, max_iter=2000)`, both fit **inside each fold** |
| skip rule | resample discarded if `<2` classes or `<3` groups (unchanged) |
| metric | `roc_auc_score` on out-of-fold probabilities; percentile CI `[2.5, 97.5]` |
| label | `y = 0 if is_real else 1` (unchanged) |

**Encoding-only feature set = the 23 `enc_*` features**, exactly as `build_features` defines them:

- `enc_wk`: `valence_proj`, `uncertainty_proj`, `reward_proj` → **3**
- `enc_svd` at L3/L7/L11/L15 × `stable_rank`, `spectral_entropy`, `top_sv_ratio` → **12**
- `enc_svd_skip1` at L3/L7/L11/L15 × `skip1_stable_rank`, `skip1_spectral_entropy` → **8**

**Excluded:** every `gen_*` and `delta_*` feature, and both `trajectory` features
(`logit_entropy`, `logit_margin`) — the trajectory is generation-phase by construction. Full set
is 52; 52 − 23 = 29 excluded. Both counts match the paper's stated "23 feats" and "52 feats".

**Note on `stable_rank`:** it appears in this feature set and is **not** stable rank as the
literature defines it — our implementation is the participation ratio of the singular-value
spectrum (see `layer_set_registry.md`). Not a defect for this refit, since the estimator is held
identical across both arms, but it must not be described to a reader as stable rank.

## The design risk, stated first

**The tempting procedure is to try feature subsets until one returns 1.0000.** That is the same
inversion I already committed once on this exact number: fitting the analysis to a remembered
value. The feature set is fixed above by prefix, before running, and will not be adjusted.

**Success is not recovering 1.0000. Success is knowing what the encoding-only classifier scores
under the same estimator as the headline.**

## Outcomes, decided in advance

- **CI excludes 0.5 and the mean is high** → encoding-only is real; report bootstrap mean + CI,
  never the in-sample point estimate, and compare like-for-like against 0.929 [0.819, 0.989].
- **CI is wide or includes 0.5** → the encoding-only claim does not survive its own estimator.
  Report that. It would mean the "entity recognition confound (major)" limitation loses the
  number motivating it — which is a finding, not a loss.
- **Mean near 1.0 with a tight CI** → the original claim was right and is now sourced. Also fine.
- **Either way the 0.07 gap dissolves**, because it was always an estimator mismatch.

## Non-goals

- Not searching feature subsets. Not tuning `C`. Not changing the seed.
- Not re-running any model. No GPU. No new data.
- Not touching the full-classifier bootstrap — it is the comparator and must stay fixed.

## Siblings this will reach (`SIBLINGS_NOT_ENUMERATED`)

`decision-state-paper/main.tex` **and** `academic/main.tex` (13 sites of 0.9995 across both;
I previously reported 4, having truncated a grep), `paper.json`, both PDFs (**do not rebuild
until text is fixed** — they are currently clean and the staleness is protective), and
`MEMORY.md` key numbers. To be enumerated by grep with the count verified, not headed.

## Gate

Agni v2 at `paper` stage, **with `paper_dir` supplied** — measured downgrade rate is 18% with
it and 50% without. Expect: `ESTIMATOR_MISMATCH`, `OUTCOME_INFORMED_TAXONOMY`,
`SELF_FAVORABLE_ROUNDING`, `SIBLINGS_NOT_ENUMERATED`.
