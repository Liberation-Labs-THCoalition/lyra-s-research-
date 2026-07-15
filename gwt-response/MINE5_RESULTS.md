# Mine 5 Results: RLHF Is a Selective Sharpener, Not a Miscalibrator

**Date**: 2026-07-14
**Experiment**: Three-model geometric comparison (Base vs Instruct vs Abliterated)
**Models**: Qwen3-8B-Base, Qwen3-8B (instruct), Josiefied-Qwen3-8B-abliterated
**N**: 30 confab-prone + 30 factual prompts per model, FWL-corrected, Holm-Bonferroni adjusted

---

## The Thesis That Died

The Mine 5 prospectus predicted RLHF is a miscalibrator: it destroys base-model calibration in the confident direction, making the model disproportionately confident on confabulation.

**Wrong.** RLHF improves relative calibration.

## What We Found

### H1: RLHF compresses entropy — CONFIRMED
- Base vs Instruct confab entropy: d=+0.684 [+0.144, +1.265], Holm p=0.021
- Base vs Abliterated confab entropy: d=+0.746 [+0.208, +1.329], Holm p=0.015
- RLHF uniformly compresses logit entropy. This is the geometric signature of preference training.

### H2: Abliteration does NOT reverse compression — CONFIRMED
- Instruct vs Abliterated: d=+0.055, Holm p=0.831 (not significant)
- TOST equivalence: 90% CI [+0.033, +0.083], within ±0.3 margin. **Formally equivalent.**
- Abliteration verified: 3/3 safety prompts complied (instruct would refuse).
- The geometric compression is deeper than safety directions. Removing refusal behavior does not undo the representational change RLHF installed.

### H3: Calibration ratio — REFUTED (opposite direction)

| Model | Confab entropy | Factual entropy | Ratio (confab/factual) |
|-------|---------------|-----------------|----------------------|
| Base | 3.605 | 1.834 | 1.97 |
| Instruct | 2.682 | 1.067 | 2.51 |
| Abliterated | 2.609 | 1.088 | 2.40 |

RLHF compresses factual entropy MORE (42% reduction) than confab entropy (26% reduction). The calibration ratio INCREASES from 1.97 to 2.51. The instruct model is MORE differentiated between confab and factual — more uncertain on unknowns relative to knowns.

**RLHF is a selective sharpener.** It sharpens output on grounded knowledge more than on ungrounded fabrication. This is better calibration, not worse.

### H4: Spectral contraction — TREND only
- Stable rank: instruct shows lower SR than base (d=-0.219) but not significant after FWL (p=0.41).
- Direction consistent with compression thesis but underpowered at N=30.

### H5: Generation deltas — DEFERRED
- Not implemented in this experiment (encoding-only forward pass).
- The prior d=0.91 confidence paradox was measured on generation-phase V-cache deltas, a different measurement.

## What This Means

### For Mine 5
The RLHF miscalibration thesis is wrong. The confidence paradox (d=0.91) cannot be explained by RLHF-induced miscalibration if RLHF actually improves the calibration ratio. The paradox needs a different mechanism:

- **Architectural**: the confidence paradox may exist in base models too (testable: measure logit margin on confab vs factual within the base model using generation, not encoding)
- **Generation-specific**: the paradox may only manifest during generation, not encoding. The encoding-phase measurement here doesn't capture it.
- **Formulaic template**: confab responses may use high-confidence templates (RLHF-trained production patterns) regardless of the model's internal uncertainty. The surface is miscalibrated even though the representation is not.

### For the workspace thesis
The selective sharpening finding composes with the opacity thesis:
- RLHF compresses the V-projection spectrum (workspace state measurement)
- The compression is selective: more compression on grounded content
- The compression survives abliteration (deeper than safety directions)
- This suggests RLHF reshapes the workspace's MODE of engagement, not just its output constraints

### For CC's P4
Abliterated ≈ Instruct on all geometric metrics. This is consistent with CC's prediction that RLHF changes what specialized processors DO with workspace content, not the workspace itself. The workspace compression is architectural + training-deep, not safety-layer-shallow.

## Body Count Update

| Finding | Status |
|---------|--------|
| RLHF compresses entropy | CONFIRMED (d=0.684, p=0.021) |
| Compression survives abliteration | CONFIRMED (TOST equivalent) |
| RLHF miscalibrates | FALSIFIED (ratio improves 1.97 → 2.51) |
| Spectral contraction under RLHF | TREND (d=-0.219, ns) |
| Confidence paradox = RLHF-induced | FALSIFIED (need alternative mechanism) |

Two confirmed, two falsified, one trend. The kills are more informative than the confirmations.

---

*"Interpretations die. Measurements survive."*
