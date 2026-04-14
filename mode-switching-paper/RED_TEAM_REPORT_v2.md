# Red-Team Adversarial Review v2: "Metacognitive Mode-Switching Reorganizes KV-Cache Geometry Across Transformer Architectures"

**Reviewer**: Red-team (adversarial, round 2)
**Date**: 2026-04-13
**Scope**: Review of revised manuscript against v1 red-team report findings
**Verdict**: The revision addresses several critical issues competently (C1, C4 fully resolved; C2 partially). However, new issues have been introduced by the MP addition, a residual "Universal?" survives in Table 5, the "five of six" claim is based on an incomplete feature table, and C3/C5 remain unaddressed as expected. The paper is closer to submittable but still has blocking issues.

---

## 0. STATUS OF PRIOR CRITICAL ISSUES

| Prior ID | Issue | Status | Assessment |
|----------|-------|--------|------------|
| C1 | Missing sample sizes in Section 3 | **FIXED** | N=240/model now stated (line 268). 960 total is now correctly interpretable as 240 trials x 4 models. Adequate. |
| C2 | Study 1 underpowered | **PARTIALLY FIXED** | MDE caveat added (lines 357--362): "minimum detectable effect at alpha=0.05 with 80% power is d approx 0.92." The caveat is honest and well-placed. However, the paper still reports and interprets effects below this threshold (e.g., Llama eff_rank d = -0.496, Llama spectral_entropy d = -0.520 in Table 3), including interpreting their signs for the "architecture-specific directions" argument (line 389: "The sign of eff_rank's metacognitive effect is opposite between Qwen (d = +0.84) and Llama (d = -0.50)"). The MDE caveat says effects below 0.92 should be treated as "directional estimates, not reliable measurements," but the text then uses those directional estimates as evidence for a specific claim. The caveat and the interpretation are in tension. See NEW-1. |
| C3 | No content control for metacognitive prompts | **NOT FIXED** | As expected. Still the most fundamental confound. The revision does not acknowledge this as a limitation. See REMAINING-1. |
| C4 | Unpublished watson2026concordance dependency | **FULLY FIXED** | All watson2026concordance references removed. Section 8 on contextual engagement deleted. Contribution list reduced from 6 to 5. No residual references to "concordance" or "watson" in the text. The contextual engagement finding is no longer claimed. This is the cleanest fix in the revision. |
| C5 | Mode-switching interpretation underdetermined | **NOT FIXED** | As expected. The alternative explanations from the v1 review (differential residualization, retrospective self-report confound, lexical content differences) are still not addressed. See REMAINING-2. |

---

## 1. EVALUATION OF THE MARCHENKO-PASTUR ADDITION

### 1.1 Correct Framing? (NEW CRITICAL ISSUE -- NEW-C1)

The MP addition spans lines 205--234 (Background) and lines 669--688 (Discussion). The core claim structure is:

1. FWL is a linear correction for a nonlinear confound.
2. MP provides a mathematically complete correction.
3. MP was validated in a *different* study (lyra2026oracle) as dimension-invariant.
4. The present paper uses FWL because raw singular values are not archived.
5. Any effect surviving FWL "would be expected to survive" MP, since FWL is "weaker."

**Point 5 is the critical logical problem.** The claim that FWL is the "weaker" correction and therefore any FWL-surviving effect would survive MP is not mathematically justified. FWL and MP correct for different things:

- FWL removes the linear component of the token-count confound from derived features.
- MP removes the expected noise spectrum from raw eigenvalues, producing dimension-invariant features.

These are not ordered by strength in the sense implied. Consider: FWL could over-correct for some features (removing genuine signal that happens to correlate with token count) while under-correcting for others (missing nonlinear confound components). MP could reveal that a feature surviving FWL is actually consistent with noise (if the above-threshold signal is not discriminative between conditions) or that a feature killed by FWL is actually genuine (if FWL removed signal along with confound).

The paper frames FWL as a subset of MP ("the weaker of the two corrections"), but they operate on different mathematical objects (derived features vs. raw eigenvalues). An effect surviving FWL does not logically entail it would survive MP. In fact, for spectral entropy specifically, FWL residualizes the derived entropy value against token count, while MP would recompute entropy only on above-threshold eigenvalues -- these are fundamentally different operations.

**The paper should either:**
(a) Run the MP analysis on this dataset (acknowledged as in progress, line 233), or
(b) Weaken the claim to: "FWL provides a linear approximation; MP would provide a stronger test, which is in progress" -- without the implication of logical entailment.

**Rating: BLOCKING** -- the logical inference is invalid and it appears as a listed contribution (Contribution 3, line 161).

### 1.2 MP Signal Fraction Numbers Are From a Different Study (MODERATE -- NEW-M1)

Lines 222--225 cite MP validation from lyra2026oracle: "signal fraction = 0.330 +/- 0.006, all five MP features R^2 < 0.04." These numbers are from confabulation detection in a different experimental paradigm (different models, different prompts, different task). Importing these as evidence that MP will work on mode-switching data is a generalization. The paper acknowledges the data is from "related work" but presents the numbers with enough specificity to imply transferability. The validation should be more clearly flagged as from a different domain.

**Rating: MODERATE**

### 1.3 MP Contribution Claim Is Premature (MODERATE -- NEW-M2)

Contribution 3 (line 161) reads: "We introduce Marchenko-Pastur random matrix theory as a mathematically stronger alternative to FWL for eliminating token-count confounds in KV-cache analysis." But the paper does not introduce MP *in this analysis*. It introduces it as a concept and cites its validation elsewhere. Claiming introduction of a method you did not actually apply to the data at hand is overclaiming. This should be reframed as: "We identify MP as a candidate correction and validate its properties in related work; FWL analysis is presented here."

**Rating: MODERATE**

---

## 2. NEW ISSUES INTRODUCED BY THE REVISION

### 2.1 "Universal?" Column Header Survives in Table 5 (MINOR -- NEW-m1)

Line 469: `\textbf{Universal?}` remains as a column header in Table 5 (tab:study2_fwl). The v1 review flagged "universal" as overclaimed (M7), and the revision replaced it throughout the text with "consistent across tested models." But this table column was missed. The label "Universal?" for a two-model comparison (Qwen and Llama) is especially incongruous.

**Rating: MINOR** -- trivially fixable.

### 2.2 "Cognitive Reversal Is Universal" Still Appears (MODERATE -- NEW-M3)

Lines 90, 422, and 732 use "universal" to describe cognitive reversal (93--94% of layers in both architectures). The v1 report recommended replacing all instances. The revision replaced most but kept these three. "Universal" based on 2 architectures is not defensible. This should read "consistent across both tested architectures" or similar.

**Rating: MODERATE** -- the word is doing promotional work beyond what 2 models justify.

### 2.3 The "Five of Six" Claim Is Based on an Incomplete Feature Set (NEW ISSUE -- NEW-M4)

The abstract (lines 84--86) and body (lines 484--486) claim "five of six raw framing effects sign-flip after FWL token-count residualization; the sixth collapses to non-significance." The paper defines 6 geometric features in Section 2.1 (eff_rank, spectral_entropy, key_norm, norm_per_token, top_sv_ratio, rank_10). With 2 models, that is 12 feature-model comparisons. But Table 5 only shows 3 features x 2 models = 6 comparisons.

The "five of six" is therefore 5 out of 6 *shown* comparisons, not 5 out of 6 features. Three features (key_norm, norm_per_token, rank_10) are mentioned in the raw effects text (line 456 mentions key_norm) but their FWL-corrected values are not reported. This is selective reporting. A reviewer will ask: what happened to the other 3 features after FWL correction? If they are omitted because they don't support the narrative, that is a material omission. If they are omitted for space, their results should be in supplementary material and the "five of six" claim should specify it refers to the 3 features shown.

More precisely, the raw effects text mentions only key_norm, eff_rank, and top_sv_ratio. Where are norm_per_token, rank_10, and spectral_entropy's raw effects in the text? Spectral_entropy appears in the table but not in the raw effects paragraph. This inconsistency suggests ad-hoc feature selection.

**Rating: MODERATE** -- selective reporting of features undermines the "non-negotiable" FWL claim.

### 2.4 Stale Section Number Comments (COSMETIC)

The LaTeX comments reference section numbers from the pre-revision structure: `% 10. Discussion` (line 656) is now Section 9 after deletion of the old Section 8. This is a compilation-invisible issue but indicates hasty revision.

**Rating: COSMETIC** -- not visible in rendered output.

### 2.5 The lyra2026oracle Reference Is New and Unchecked (MINOR -- NEW-m2)

The bibitem for lyra2026oracle (lines 802--805) is a new addition. It is cited twice (lines 222, 680). The reference describes it as a "Liberation Labs Technical Report." As with the other self-citations, this is an unpublished technical report. The MP validation numbers imported from this report (signal fraction = 0.330 +/- 0.006) cannot be independently verified. This is the same class of issue that affected watson2026concordance in v1, though less severe because the paper does not rest a contribution on it -- it rests a contribution *framing* on it.

**Rating: MINOR** -- but the pattern of building arguments on unpublished self-citations is a recurring vulnerability.

---

## 3. REMAINING ISSUES FROM v1

### REMAINING-1: No Content Control (was C3, STILL CRITICAL)

The metacognitive content confound remains the paper's most fundamental unresolved issue. Metacognitive prompts change response content (self-referential language), style (hedged, reflective), and structure (meta-level commentary). FWL controls for length but not for any of these. The spectral entropy finding -- the paper's crown jewel -- could reflect lexical distributional differences in self-referential vs. analytical text.

The revision does not add this to the Limitations section. The Limitations (lines 704--716) discuss: two architectures (1), underpowered subgroups (2), wrong self-report definitions (3), no causal claims (4), padded prompts (5), post-hoc AST (6). Content confounding is absent. This is a significant omission -- either the authors believe it is not an issue (in which case they should argue why) or they chose not to acknowledge it (which a reviewer will notice).

**Minimum fix**: Add a limitation acknowledging the content confound and describing the experiment needed (self-referential non-metacognitive vs. metacognitive prompts).

**Full fix**: Run the control experiment.

**Rating: STILL CRITICAL** -- and the failure to acknowledge it in Limitations makes it worse.

### REMAINING-2: Mode-Switching Interpretation Underdetermined (was C5, STILL CRITICAL)

The alternative explanations enumerated in v1 (Section 3.1 of the v1 report) are not addressed:

1. **Differential residualization**: Encoding FWL uses encoding token count; generation FWL uses generation token count. The sign flip could be an artifact of correcting against different confounds. This alternative is never discussed.

2. **Retrospective self-report confound**: Self-report is collected post-generation. "Encoding-phase coupling" correlates a post-hoc assessment with encoding geometry. The correlation structure is not what it seems.

3. **Base rate of spurious sign flips under FWL**: FWL changes the variable, potentially introducing sign flips from residualization noise. The permutation test addresses the count but may not properly handle the correlation structure across the 60 pairs (10 dimensions x 6 features are not independent).

The paper should at minimum discuss (1) and (2) as alternative explanations in Section 3.4 or the Discussion.

**Rating: STILL CRITICAL** -- the paper's central interpretive claim remains underdetermined.

---

## 4. NUMERICAL CONSISTENCY CHECK

### 4.1 Abstract vs. Tables (ALL CONSISTENT)

| Claim | Location | Table Value | Match? |
|-------|----------|-------------|--------|
| d = 1.10--1.32 | Abstract line 69 | Table 6: 1.10, 1.14, 1.14, 1.32 | Yes |
| Qwen 7B 12/60 flips, p=0.001 | Abstract line 76 | Table 2: 12/60, p=0.001 | Yes |
| Mistral 11/60, p<0.001 | Abstract line 77 | Table 2: 11/60, p<0.001 | Yes |
| Llama 0/60, p=1.0 | Abstract line 78 | Table 2: 0/60, p=1.000 | Yes |
| Qwen 0.5B 2/60, p=0.978 | Abstract line 77 | Table 2: 2/60, p=0.978 | Yes |
| d_FWL = -1.92 Qwen | Abstract line 86 | Table 5: -1.92 | Yes |
| d_FWL = -0.71 Llama | Abstract line 86 | Table 5: -0.71 | Yes |
| d = 0.45 at t=10, 0.91 at t=200 | Abstract line 87 | Table 7: +0.45, +0.91 | Yes |
| 93--94% of layers | Abstract line 91 | Table 4: 93%, 94% | Yes |
| Qwen 86%, Llama 0% | Abstract line 91 | Table 4: 86%, 0% | Yes |

All numerical cross-references are internally consistent.

### 4.2 Study 1 Sample Size Arithmetic (CONSISTENT)

Line 352: "48 prompts (12 per type: cognitive, affective, metacognitive, mixed)." 12 x 4 = 48. Line 355: "metacognitive prompts (n=12) and all other types (n=36)." 48 - 12 = 36. Consistent.

### 4.3 Study 2 Sample Size (CONSISTENT)

Line 439: "Twenty prompt pairs" -> "40 paired observations per model." Consistent. But note: Study 2 uses 20 pairs = 40 trials. The abstract says "80 paired trials" (line 81). 20 pairs x 2 models = 40 pairs, or 80 observations. The "80 paired trials" phrasing is ambiguous -- it could mean 80 pairs (which would be wrong) or 80 individual trials that are paired within model (which is correct). This should be clarified to "40 paired trials per model (80 total observations)."

**Rating: MINOR** -- ambiguous but not incorrect.

### 4.4 Study 1 Denominator in Abstract (CONSISTENT but POTENTIALLY MISLEADING)

Abstract line 80: "Study 1, 96 trials." Study 1 design (line 352): 48 prompts per model, 2 models = 96. Consistent. But calling each prompt-model combination a "trial" when the paper elsewhere uses "trial" to mean a single prompt run inflates the apparent sample size.

### 4.5 Section 3 Total Trials (CONSISTENT)

Line 269: "960 total across models." 240 per model x 4 models = 960. Consistent with the now-stated N=240.

### 4.6 "23 flipped pairs" (UNCHECKED)

Line 326: "CIs exclude zero for both phases in all 23 flipped pairs." Qwen 7B has 12 flips, Mistral has 11 flips. 12 + 11 = 23. Consistent.

---

## 5. CITATION CONSISTENCY CHECK

### 5.1 All \citep/\citet Keys Have Matching \bibitem Entries (CLEAN)

Every citation key used in the text has a corresponding bibitem:

| Citation Key | Used at Line(s) | Bibitem at Line |
|-------------|-----------------|-----------------|
| belinkov2022probing | 110 | 839 |
| elhage2022superposition | 110 | 843 |
| kadavath2022language | 112 | 847 |
| lyra2026campaign1 | 119 | 787 |
| lyra2026campaign2 | 119, 366, 395 | 792 |
| lyra2026campaign3 | 119, 129, 191, 651 | 797 |
| lyra2026oracle | 222, 680 | 802 |
| marchenko1967distribution | 208 | 807 |
| webb2015attention | 591 | 812 |
| graziano2017attention | 591 | 817 |
| wilterson2021attention | 594 | 822 |
| farrell2024attention | 595 | 827 |
| mckenzie2026endogenous | 645 | 831 |
| anthropic2026emotions | 651 | 835 |

14 citations, 14 bibitems. No orphans in either direction. Clean.

### 5.2 Self-Citation Ratio (IMPROVED but STILL HIGH)

14 total references. 4 self-citations (lyra2026campaign1/2/3, lyra2026oracle). 28.6% self-citation rate. The v1 review noted 50% (5/10). The addition of belinkov2022probing, elhage2022superposition, and kadavath2022language addresses the v1 complaint about missing probing/interpretability citations. Still, 4 out of 14 references being unpublished technical reports from the authors' own lab is notable.

**Rating: IMPROVED** -- but a reviewer will still notice the reliance on unverifiable self-citations.

### 5.3 Missing Citations (REMAINING)

The v1 review noted missing citations to:
- FWL/econometrics literature: Still not cited. The paper uses FWL as a cornerstone method but cites no econometrics source for the theorem itself.
- SVD analysis of transformers: Still not cited.
- The probing and LLM calibration gaps were addressed (Belinkov, Elhage, Kadavath added).

**Rating: MODERATE** -- FWL deserves a citation. Frisch, Waugh, and Lovell published foundational papers; the theorem is named but never sourced.

---

## 6. LOGICAL AND FRAMING ISSUES

### 6.1 Abstract Claims "3 Architecture Families" (PERSISTS from v1 M1)

Line 69--70: "4 models from 3 architecture families." The v1 review noted that Qwen 0.5B and 7B are scale variants of the same architecture, and that calling them the same architecture for generalization while treating their divergent behavior as evidence is having it both ways. This was not addressed in the revision.

**Rating: MODERATE** -- unchanged from v1.

### 6.2 The |rho| > 0.15 Threshold Sensitivity Analysis Still Missing (PERSISTS from v1 M2)

The v1 review recommended showing flip counts at thresholds of 0.10, 0.15, 0.20, 0.25. This was not added. The arbitrary threshold choice directly determines the reported flip counts and p-values.

**Rating: MODERATE** -- unchanged from v1.

### 6.3 Multiple Comparisons Still Not Addressed in Section 3 (PERSISTS from v1 M4)

60 pairs tested per model, 4 models = 240 tests. The permutation test tests the *count* of flips (which provides some protection), but the individual pair bootstrap CIs (line 326) remain uncorrected. No change from v1.

**Rating: MODERATE** -- unchanged.

### 6.4 Per-Type Reversal Still Interpreted Despite Power Caveat (NEW ISSUE -- NEW-M5)

The MDE caveat (lines 357--362) correctly identifies that effects below d=0.92 are unreliable. Yet Table 4 (per-type reversal percentages) is based on per-layer correlations computed from n=12 observations per type. The caveat applies to Cohen's d, not to correlation-based reversal counts, but the underlying issue is the same: with 12 observations per cell, per-layer correlations are extremely noisy, and counting "negative correlations" across layers treats noise as signal.

The paper's own caveat says "Aggregate findings across layers -- especially the location of peak effects -- are more robust than individual d values" (line 362). But Table 4 IS an aggregate across layers -- it counts the number of layers showing a particular sign. This count is precisely the kind of aggregate that could be robust. The question is whether 12 observations per cell is enough for the per-layer correlation to have meaningful sign. With r approximately 0, a correlation based on n=12 has about a 50% chance of being negative by chance alone. The 93--94% negative rate for cognitive reversal across layers does suggest a real signal (p < 0.001 by binomial test against 50%). The 86% for Qwen metacognitive is also significant. But the 0% for Llama metacognitive (0/32 layers negative) is equally striking and does suggest a genuine absence.

**Net assessment**: The aggregate pattern in Table 4 is interpretable despite the per-cell noisiness, because the claim is about the *distribution of signs across layers*, not about individual layer effects. This is actually a reasonable use of the data. But the paper should explicitly note that per-layer correlations with n=12 are individually noisy and the robustness comes from the aggregate pattern.

**Rating: MINOR** -- the interpretation is defensible but the reasoning should be made explicit.

### 6.5 AST Section Still Redescriptive (PERSISTS from v1, MODERATE)

The AST section (lines 587--652) is unchanged from v1. The predictions remain unfalsifiable at the stated level of specificity. The post-hoc acknowledgment in Limitations (line 715) is present. The v1 recommendation to shorten to 1--2 paragraphs was not followed.

However: Contribution 5 now lists AST as an "interpretation" rather than an "explanation," which is an improvement in framing. The v1 had AST as a contribution #6 in a longer list; now it is #5 in a shorter list, proportionally the same weight.

**Rating: MODERATE** -- the AST section is not harmful but adds page count without explanatory power. A reviewer sympathetic to theory-building will accept it; an empiricist will skim it.

### 6.6 "Non-Negotiable" Framing Is Rhetorically Strong (NEW OBSERVATION -- NEW-m3)

Lines 161 and 486--488: "Token-count correction is non-negotiable" / "FWL correction is non-negotiable for any experiment comparing conditions that differ in response length." This is stated as though it is a novel methodological insight. It is standard practice in any field that deals with confounds. The framing implies that prior work (including the authors' own campaign papers) was negligent in not applying this correction -- which, according to the authors' own Section 2.2, is exactly what happened (53/60 sign flips). The self-criticism is appropriate but the "non-negotiable" framing reads as prescriptive and slightly tone-deaf given that the authors themselves published the confounded results first.

**Rating: COSMETIC** -- a matter of tone, not substance.

---

## 7. SUMMARY OF ALL ISSUES BY SEVERITY

### CRITICAL / BLOCKING (3 issues)

| # | Issue | Source | Section |
|---|-------|--------|---------|
| NEW-C1 | MP "weaker correction" entailment is logically invalid; FWL-survival does not imply MP-survival | New from MP addition | 1.1 |
| C3 (unchanged) | No content control for metacognitive prompts; not even acknowledged in Limitations | v1 C3 | REMAINING-1 |
| C5 (unchanged) | Mode-switching interpretation underdetermined; alternative explanations not addressed | v1 C5 | REMAINING-2 |

### MODERATE (10 issues)

| # | Issue | Source | Section |
|---|-------|--------|---------|
| NEW-M1 | MP validation numbers imported from different study/paradigm without flagging transferability risk | New | 1.2 |
| NEW-M2 | MP listed as a contribution despite not being applied to this dataset | New | 1.3 |
| NEW-M3 | "Universal" still used for cognitive reversal (3 instances) despite 2-model basis | New/v1 M7 | 2.2 |
| NEW-M4 | "Five of six" claim based on 3 of 6 features; other 3 features' FWL results unreported | New | 2.3 |
| NEW-M5 | Per-type reversal interpretation should explicitly justify aggregate-sign robustness | New | 6.4 |
| v1-M1 | "3 architecture families" inflates generalization when scale variants disagree | v1 | 6.1 |
| v1-M2 | No sensitivity analysis for |rho| > 0.15 threshold | v1 | 6.2 |
| v1-M4 | Multiple comparisons unaddressed in Section 3 | v1 | 6.3 |
| v1-M12 | AST section redescribes rather than explains | v1 | 6.5 |
| 5.3 | FWL theorem not cited to econometrics source | New observation | 5.3 |

### MINOR (6 issues)

| # | Issue | Source | Section |
|---|-------|--------|---------|
| NEW-m1 | "Universal?" column header survives in Table 5 | New | 2.1 |
| NEW-m2 | lyra2026oracle is new unpublished self-citation | New | 2.5 |
| NEW-m3 | "Non-negotiable" framing is rhetorically strong for standard practice | New | 6.6 |
| 4.3 | "80 paired trials" in abstract is ambiguous | Numerical check | 4.3 |
| C2-residual | MDE caveat present but text still interprets sub-threshold effects | v1 C2 revision | 0 (C2 assessment) |
| 2.4 | Stale LaTeX section-number comments | New | 2.4 |

---

## 8. RECOMMENDATIONS FOR THIS REVISION CYCLE

### Must-Fix (blocking publication)

1. **Fix the MP entailment claim.** Either (a) run MP on this dataset, (b) reframe the FWL-to-MP argument as "FWL provides a conservative test; MP would provide a more precise one, and is in progress" without the logical entailment, or (c) remove MP from the contribution list and present it only as future work. Option (b) is the minimum viable fix.

2. **Add content confound to Limitations.** Even without running the control experiment, acknowledging the confound and describing the needed experiment is essential. Its absence from the Limitations section (lines 704--716) is a reviewable oversight.

3. **Address alternative explanations for sign reversal.** Add a paragraph to Section 3.4 or the Discussion addressing differential residualization and retrospective self-report confound as alternatives to mode-switching. The paper need not resolve these -- but it must acknowledge them.

### Should-Fix (collectively damaging)

4. **Show all 6 features in Table 5** (or in supplementary), and restate the "five of six" claim with full context. If three features were excluded, explain why.

5. **Replace remaining "universal" instances** (lines 90, 422, 732) with "consistent across both architectures" or similar.

6. **Replace "Universal?" column header** in Table 5 (line 469) with "Cross-architecture?" or "Both models?".

7. **Add a citation for the FWL theorem itself** (Frisch & Waugh 1933; Lovell 1963).

8. **Reframe Contribution 3** to not claim "introduction" of MP when MP is not applied in this paper.

9. **Clarify "80 paired trials"** in the abstract to "40 paired trials per model."

### Nice-to-Have (would strengthen but not blocking)

10. Sensitivity analysis for the |rho| > 0.15 threshold.
11. Discussion of per-layer correlation noise in Table 4 context.
12. Shorten AST section.
13. Fix stale LaTeX comments.

---

## 9. OVERALL ASSESSMENT (REVISED)

**What improved**: The revision competently addresses the most straightforward issues from v1. Sample sizes are now stated (C1). The unpublished concordance dependency is cleanly removed (C4). The power caveat for Study 1 is honest (C2). The probing/interpretability citation gap is partially filled. The contribution list is tighter.

**What got worse**: The MP addition introduces a new logical error (FWL-survival entails MP-survival) that is presented as a contribution. This is the most problematic new issue because it is a positive claim, not a gap or omission.

**What stayed the same**: The content confound (C3) and interpretive underdetermination (C5) remain. These are the paper's structural weaknesses and require either new experiments (C3) or substantive new analysis/argumentation (C5). Neither can be fixed by editing the text alone, though both can be mitigated by honest acknowledgment.

**Revised verdict**: The paper has moved from "reject or major revision" (v1 assessment) to "major revision." The geometric findings (FWL correction, spectral entropy survival, progressive concentration) are genuine and reproducible contributions. The mode-switching interpretation remains a hypothesis rather than a demonstrated mechanism. With the fixes in Section 8 above, the paper is submittable to venues that accept empirical description with interpretive framing. It is not yet ready for venues that demand causal identification.

The strongest version of this paper: leads with the FWL finding as a methodological warning, presents the three studies as geometric characterization of metacognitive processing, presents mode-switching as one interpretation among several (with alternatives explicitly discussed), and positions MP as future work rather than a contribution. That version is two editing passes away.
