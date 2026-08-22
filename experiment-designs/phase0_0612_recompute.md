# Phase 0 — Is the published 0.612 reproducible? A pre-specified recompute

**Author:** Lyra · **Date:** 2026-08-20 · **Status:** DESIGN, pre-gate
**Blocks:** closing Phase 0 (#72) · **Compute:** trivial (17.5 MB JSON, CPU, seconds)

---

## The claim under test

Published in the Phase 2 attention probe: **band AUROC 0.612, p=0.046**, described as the
FWL-surviving sparsity result for "the workspace band."

Two independent problems are already established (registry, 2026-08-14):

1. **The label is wrong.** L35–47 is the **ignition zone**, not the workspace band. The
   workspace peak on 27B is L19–23. The finding is real; its name is not.
2. **The number does not reproduce.** Four FWL specs were tried. None returns the published
   pair:

| FWL spec | band AUROC | p |
|---|---|---|
| none | 0.6423 | 0.0026 |
| prompt_len only | 0.5079 | 0.8678 |
| gen_len only | 0.7704 | <0.0001 |
| prompt + gen | 0.6103 | 0.0196 |
| mean-of-per-layer | 0.588 | — |

Closest is 0.6103 (Δ=0.0017 on AUROC) but **p differs by 2×** (0.0196 vs 0.046). An AUROC
match with a doubled p-value is not a match; it means a different n, a different test, or a
different pooling.

## The design risk, stated first because it is the whole problem

**The tempting procedure is to keep trying specs until one yields 0.612.** That is p-hacking
inverted — fitting the analysis to a remembered number instead of fitting the number to the
data. It would "succeed" eventually and the result would be worthless.

**So this design pre-specifies one primary spec on principled grounds, computes it once, and
reports it regardless of whether it matches the published value.** Reproducing 0.612 is not the
success condition. Establishing what the correct number *is* under a defensible, stated spec is
the success condition.

## Pre-specified primary analysis

Locked before execution. Any deviation is reported as a deviation.

| element | value | why this and not the alternative |
|---|---|---|
| data | `starship:~/oracle-experiments/results/a1_benchmark/attention_phase2.json` (17.5 MB, n=200: 50 grounded / 150 confab) | the shipped artifact; the question is about *it* |
| feature | `mean_sparsity` | the published feature |
| band layers | **L35, 39, 43, 47** | the actual ignition zone; named correctly this time |
| control layers | **L7, L55, L63** | L15 **excluded** — see below |
| FWL regressors | **prompt_len AND gen_len** | both are length confounds; residualizing one and not the other is an arbitrary choice, and the table above shows it swings AUROC from 0.51 to 0.77 |
| FWL fitting | within-fold if any CV is used; here no CV, so fit on all data and state it | avoids the global-scaler leakage class |
| test | Mann-Whitney U, tie-corrected, two-sided | published test |
| direction | AUROC = confab vs grounded | published direction |

**Why L15 is excluded — CORRECTED after gate review, see GATE RESPONSE below.**

L15 is a **band layer by an independent, derived, published definition**, not by its AUROC.
`ghost-dimensions/main.tex:106-112` derives the workspace band as low-rank INTERSECT gate-valid
= **L11-13, L15-17, L24-L48** (31 layers, arithmetic checks). L15 is explicitly in that set,
and the derivation is reproducible from the paper and predates every number in this document.

Listing L15 as an "off-band control" was therefore a **flat labeling error**, correctable
without reference to any outcome.

The AUROC agreement (L15 0.6109 vs band pooled 0.6103, three decimals) is **corroboration of
the band definition by data that never referenced it** — evidence FOR the ghost-dimensions
derivation, not the criterion for exclusion. The three genuine controls are correctly off-band
under the same definition: L7 sits below the band's L11 start, L55 and L63 above its L48 end.

Note also that the ignition zone (L35-47) is a **subset** of the band (L24-48), not a disjoint
region. The probe layers L35/39/43/47 are both.

## Both pooling estimators reported, always

Pooled-across-layers gives 0.6103; mean-of-per-layer-AUROCs gives 0.588. **Neither is
privileged and I am not going to pick the one nearer 0.612.** Both get reported at every
site, with the estimator named inline.

This is the circumplex lesson applied: an unstated estimator is how a published table drifts.
Two numbers with their formulas beat one number with a footnote.

## Outcomes, decided in advance

- **Primary spec reproduces the published pair (both AUROC and p)** → the published number
  stands; only the *label* needs correcting from "workspace band" to "ignition zone."
- **Primary spec gives a different value** → the published (0.612, p=0.046) pair is **not
  recoverable from the shipped artifact**, and the paper needs a correction carrying the new
  number, both estimators, and the exact spec. This is the outcome the evidence currently
  favours and I expect it.
- **Either way**, the finding's *substance* is unaffected: band-versus-control separation on
  sparsity is real and gets stronger under the clean contrast set. This is a correction of a
  number and a name, not a retraction of a result.

## Explicit non-goals

- Not searching for a spec that recovers 0.612.
- Not re-running the probe. No new data. No GPU.
- Not touching the ghost-dimensions band definition, which this analysis independently
  corroborates and must not be allowed to circularly re-derive.

## Sibling artifacts this will need to reach

Per `SIBLINGS_NOT_ENUMERATED`, named in advance rather than discovered afterwards:
the Phase 2 attention probe write-up, `layer_set_registry.md`, `moe_geometry_campaign.md`
findings log, MEMORY.md key numbers (`Attention sparsity FWL 0.612` currently sits in the
UNRESOLVED list), and any paper citing the 0.612. **To be enumerated concretely by grep before
any edit, and listed in the commit message including the ones deliberately not touched.**

## Gate

Submit to Agni v2 (`meridian/`, now carrying the 50-kill list) at `paper` stage before
execution. Relevant kills to expect it to fire on: `ESTIMATOR_MISMATCH`,
`LABEL_WITHOUT_ITS_STATISTIC`, `SILENT_N_CHANGE`, `OUTCOME_INFORMED_TAXONOMY`,
`SIBLINGS_NOT_ENUMERATED`.


---

## GATE RESPONSE — Agni v2, 2026-08-20

Submitted at `paper` stage. **Verdict: WARN**, confidence 0.94, reviewer
`claude-haiku-4-5-20251001`. Kills cited (5): `ESTIMATOR_MISMATCH`,
`LABEL_WITHOUT_ITS_STATISTIC`, `OUTCOME_INFORMED_TAXONOMY`, `SIBLINGS_NOT_ENUMERATED`,
`SILENT_N_CHANGE` — exactly the five this document predicted in its Gate section.

**The finding that changed the design:**

> "The design excludes L15 from controls based on post-hoc observation that it matches the
> band, but this creates a subtle circularity risk."

**Correct, and accepted.** The original text justified the exclusion by L15's AUROC matching
the band's — i.e. selecting the control set using the outcome variable the analysis then
reports on. That is `OUTCOME_INFORMED_TAXONOMY` and the gate named it.

**The conclusion survives; the reasoning did not.** Checking the primary
(`ghost-dimensions/main.tex:106-112`) shows L15 is in the band by a derived definition that
predates and is independent of any AUROC. The exclusion is justified; my stated justification
was the wrong one. Section corrected above.

This is the first design gated by Agni v2 since it was wired to `kills.py` on 2026-08-19 —
previously it carried no kill list at all, and could not reach any backend. The catch is
attributable to a kill that was in the file and invisible to the reviewer four days ago.