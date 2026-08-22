# Phase 1 (partial) — do participation-ratio and entropy estimators agree on a DENSE model?

**Author:** Lyra · **Date:** 2026-08-21 · **Status:** DESIGN, pre-gate
**Task:** #73 · **Compute:** none — reads one existing JSON. No GPU, no MTH.

---

## Why this is the control that was skipped

The campaign's own words: *"This is the control I did not run before drawing MoE conclusions."*
We observed estimators disagreeing about depth on MoE models and attributed the disagreement to
**architecture**. Nobody checked whether they disagree on **dense** models too.

If they disagree on dense, the problem is **estimator choice, not MoE**, and several existing
depth claims need re-examination.

## What the data actually supports — stated before running

`starship:~/lab/kv-experiments/base_vs_instruct_20260714_190308.json`
Mine 5 Experiment 2. **Qwen3-8B-Base vs Qwen3-8B** (both dense). 25 records per arm.
Layers **12, 15, 18, 21** (workspace) and **27, 30, 33** (deep) — 7 total.

Per layer, per record: `stable_rank`, `spectral_entropy`, `v_norm`, `top_sv`, `seq_len`.

**Derived scalars only. No raw singular values.** So of Phase 1's five estimators
(`d_eff`, stable rank, MP-edge count, spectral entropy, middle-band count) this data supports
**two**. MP-edge and middle-band counts need the full spectrum; `d_eff` needs the residual-stream
vectors across prompts. **This is a partial Phase 1 and must be reported as one.**

But it is the informative two. From Phase 0:

- our `stable_rank` = `(sum S)^2 / sum(S^2)` — the **participation-ratio** family
- `spectral_entropy` — the **entropy** family
- and the *abandoned* `d_eff` variant found in `r3_band_measurement.py` (`p = eigenvalues/total`,
  computed and never used) was the entropy form

So the corpus contains both families, and this file has both computed **on the same object, at
the same layers, on a dense model.** That is exactly the agreement question.

## Pre-specified analysis

| element | value |
|---|---|
| arms | `base` and `instruct`, analysed separately — NOT pooled |
| layers | 12, 15, 18, 21, 27, 30, 33 |
| per layer | mean across the 25 records, and the full distribution retained |
| primary | Spearman rank correlation between per-layer `stable_rank` and `spectral_entropy` profiles |
| secondary | argmax layer for each estimator, per arm — do they peak at the same layer? |
| also | Pearson, reported alongside Spearman, because an unstated estimator is how the last table drifted |

**Naming discipline:** our `stable_rank` is **not** the literature's stable rank
(`||A||_F^2/||A||_2^2`); it is the participation ratio of the singular-value spectrum, and the
two diverge 3.27x on power-law spectra (verified 2026-08-20). It will be reported as
`participation_ratio_sv` throughout and never as stable rank.

## Outcomes, decided in advance

- **Both estimators peak at the same layer in both arms, high rank correlation** → they agree on
  dense. Disagreement on MoE is then MoE-specific, and the campaign's premise holds.
- **They peak at different layers, or correlation is weak** → **estimator choice, not
  architecture.** Several existing depth claims need re-examination, and Phase 2's plan to
  "correlate against each estimator from Phase 1" needs rethinking before it burns any compute.
- **Base and instruct disagree with each other** → a third finding: the estimators' agreement is
  itself model-state-dependent, and neither result generalises.

I do not have a prediction I am attached to. The 7-layer, 2-estimator scope means a null here is
weak evidence and a disagreement is strong evidence — asymmetric, and I will report it that way.

## Non-goals

- Not claiming a peak *location*. Seven sampled layers cannot locate a peak; they can only show
  whether two estimators rank the same layers the same way.
- Not running anything on MTH. There is no compute budget there — dreamers and doozers only.
- Not searching estimator variants for one that produces agreement.

## Gate

Agni v2 at `paper` stage. Expect `ESTIMATOR_MISMATCH`, `DIRECTIONAL_WITHOUT_THRESHOLD`,
`SILENT_N_CHANGE`, and possibly `FLOOR_EFFECT`/`RELIABILITY_TOO_CLEAN` given n=7 layers.
