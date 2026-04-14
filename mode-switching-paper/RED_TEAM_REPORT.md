# Red-Team Adversarial Review: "Metacognitive Mode-Switching Reorganizes KV-Cache Geometry Across Transformer Architectures"

**Reviewer**: Red-team (adversarial)
**Date**: 2026-04-13
**Verdict**: The paper has a genuine core finding (FWL correction matters, spectral entropy survives) but suffers from significant structural, statistical, and interpretive weaknesses that a hostile reviewer will exploit. Multiple issues require revision before this is submittable to a serious venue.

---

## 1. NUMERICAL CONSISTENCY

### 1.1 Abstract d-value range vs. Table 6 (MINOR)

The abstract claims "Cohen's d = 1.10--1.32, 4 models, 3 architectures." Table 6 (tab:tsv_universal) shows the "Meta" column values: Qwen 0.5B = +1.14, Qwen 7B = +1.14, Llama 8B = +1.10, Mistral 7B = +1.32. These are indeed d = 1.10--1.32 for metacognitive prompts on top_sv_ratio. However, this is the d for metacognitive vs. OTHER types on a SINGLE feature. The abstract presents this as though it characterizes the mode-switching phenomenon generally. A reader encountering "d = 1.10--1.32" in the abstract will think these are overall effect sizes for mode-switching, not effect sizes for one feature on one comparison. This is technically correct but materially misleading.

**Rating: MINOR** -- fixable with clarification.

### 1.2 "3 architectures" claim (MODERATE)

The abstract says "4 models, 3 architectures." The four models are Qwen 0.5B, Qwen 7B, Llama 8B, Mistral 7B. Qwen 0.5B and 7B are both Qwen2.5 architecture. Llama 3.1 is one architecture. Mistral 7B is one architecture. So "3 architectures" is technically defensible, but the two Qwen models are scale variants of the same architecture, not truly independent architectures. A hostile reviewer will note that the "3 architectures" claim inflates the apparent generalization, especially since the two Qwen models behave differently (0.5B does NOT reorganize; 7B DOES). If scale variants of the same architecture disagree, calling them the same architecture for count purposes while treating their disagreement as evidence is having it both ways.

**Rating: MODERATE** -- the asymmetry between "3 architectures" (for generalization claims) and "4 models" (for showing divergence) is exploitable.

### 1.3 "960 total trials" count (MINOR)

Section 3.1: "960 total trials" from 60 pairs x 4 models x ... wait. 60 pairs x 4 models = 240 correlation computations, not 960 trials. The 960 appears to come from counting each correlation pair as a "trial." If each model ran N prompts, and there are 60 dimension-feature pairs, the actual trial count is the number of prompts per model, not 960. This conflates statistical tests with experimental observations. A reviewer will ask: how many PROMPTS were there per model? This is never stated for the Section 3 analysis.

**Rating: MINOR** -- but the missing N is a problem (see 2.2).

### 1.4 Spectral entropy d-values in abstract vs. Table 5 (CONSISTENT)

Abstract: "d_FWL = -1.92 Qwen, -0.71 Llama, both p < 0.001." Table 5 (tab:study2_fwl): Qwen d_FWL = -1.92 (p < .001), Llama d_FWL = -0.71 (p < .001). These match.

### 1.5 Sign-reversal counts consistent (CONSISTENT)

Abstract: "Qwen 7B: 12/60 flips, p = 0.001; Mistral 7B: 11/60, p < 0.001." Table 2: Matches exactly. Llama 0/60 p=1.0 and Qwen 0.5B 2/60 p=0.978 also match.

### 1.6 Progressive divergence d-values (CONSISTENT)

Abstract: "d = 0.45 at token 10 to d = 0.91 at token 200." Table 7: top_sv_ratio row shows +0.45 at t=10 and +0.91 at t=200. Matches.

---

## 2. STATISTICAL ISSUES

### 2.1 Missing sample sizes throughout (CRITICAL)

The paper never clearly states the number of prompts per model for the Section 3 analysis. Study 1 says "48 prompts (12 per type)." Study 2 says "20 prompt pairs" yielding "40 paired observations per model." Study 3 uses the same 20 pairs. But Section 3 (the encoding-generation regime shift) -- which provides the foundational claim of the paper -- never states N. How many prompts were used per model to compute those 60 FWL-corrected Spearman correlations? Without this, the permutation test on sign reversals is uninterpretable. If N=48 (as in Study 1), Spearman correlations with N=48 and |rho| > 0.15 threshold have very wide confidence intervals.

**Rating: CRITICAL** -- a reviewer will reject on this alone.

### 2.2 The |rho| > 0.15 threshold is arbitrary (MODERATE)

The sign-reversal criterion requires both |rho_encode| > 0.15 and |rho_gen| > 0.15 with opposite signs. The threshold of 0.15 is not justified. This is a conventional "small effect" threshold for correlations, but:
- Why not 0.10? Or 0.20?
- The choice of threshold directly determines flip counts. A sensitivity analysis showing results at 0.10, 0.15, 0.20, 0.25 is essential.
- The permutation null is conditioned on this threshold, so the p-values are threshold-dependent.

**Rating: MODERATE** -- addressable with a sensitivity analysis in supplementary materials.

### 2.3 Permutation null model specification incomplete (MODERATE)

"A permutation null model (1,000 shuffles) tests whether observed flip counts exceed chance." What is permuted? The self-report labels? The feature labels? The prompt-to-response mapping? The encoding-generation pairing? Different permutation strategies test different null hypotheses. A reviewer familiar with permutation testing will need to know exactly what was shuffled.

Also, 1,000 permutations gives a resolution of p = 0.001 at minimum. The paper reports p = 0.001 for Qwen 7B and p < 0.001 for Mistral 7B. With 1,000 permutations, the smallest achievable p is 1/1001 = 0.000999. So "p = 0.001" means zero permutations exceeded the observed count, and "p < 0.001" also means zero permutations exceeded -- these are identical results reported differently. This is inconsistent notation.

**Rating: MODERATE** -- the different notation for identical results (0 of 1000 permutations exceeded) is sloppy.

### 2.4 Multiple comparisons not addressed in Section 3 (MODERATE)

60 pairs tested per model, 4 models = 240 tests. No correction for multiple comparisons is applied to the sign-reversal analysis. The permutation test addresses whether the COUNT of flips exceeds chance, but individual pair bootstrap CIs are reported without correction. "CIs exclude zero for both phases in all 23 flipped pairs" -- how were these CIs constructed? Bootstrap of what? At what confidence level? With what multiple-comparison correction?

**Rating: MODERATE**

### 2.5 Study 1 is severely underpowered for per-type claims (CRITICAL)

12 prompts per type. Cohen's d between metacognitive (N=12) and all other types (N=36) with 48 total. For a two-sample comparison with N1=12 and N2=36, 80% power at alpha=0.05 requires d > 0.92. Yet the paper reports effects as small as d = -0.496 (Llama eff_rank) and interprets their sign. With N=12, the confidence intervals on these d values are enormous (approximately +/- 0.7). The reported d = -0.496 is statistically indistinguishable from d = +0.2 at these sample sizes.

The per-type reversal analysis (Table 4) counts layers with "negative encode-to-generation correlation" for N=12 observations per cell. A correlation based on N=12 has a 95% CI of approximately +/- 0.58. Most of these layer-level correlations are noise.

**Rating: CRITICAL** -- the per-type claims are drastically overinterpreted given the sample sizes.

### 2.6 Study 2: Paired t-test assumptions (MINOR)

The 20 prompt pairs are analyzed with paired t-tests, which assumes the paired differences are approximately normal. With N=20 pairs, the CLT is borderline. Were any normality checks performed? Were any non-parametric alternatives run?

**Rating: MINOR** -- standard practice, but worth noting.

### 2.7 Study 3: Multiple endpoints without correction (MODERATE)

Table 7 shows 14 significance tests (7 timepoints x 2 features) with no multiple-comparison correction. The asterisks (* p < .05; ** p < .005) are uncorrected. With 14 tests, the expected false positives at alpha=0.05 is 0.7. The monotonic pattern provides some protection (random false positives wouldn't be monotonic), but this should be explicitly addressed.

**Rating: MODERATE**

### 2.8 Cross-model d-profile correlation (MINOR)

Section 6.3 reports rho = 0.36, p = .02, n = 42. Where does n = 42 come from? Study 3 has 7 timepoints x 6 features = 42 data points for the d-profile. But these are not independent observations -- features are correlated within timepoints, and timepoints are correlated within features. The effective N is much smaller than 42. This correlation is likely not significant after accounting for the dependence structure.

**Rating: MINOR** -- the paper doesn't lean heavily on this result.

---

## 3. LOGIC GAPS AND ARGUMENTATION

### 3.1 The central claim is underdetermined (CRITICAL)

The paper's core claim: "The self-report--geometry relationship reorganizes during generation" -- i.e., mode-switching -- is an interpretation, not a direct observation. What is directly observed is that encoding-phase correlations and generation-phase correlations differ in sign for some dimension-feature pairs. But there are simpler explanations:

1. **Different features are confounded differently.** Encoding features scale with input length; generation features scale with output length. FWL correction uses different confounds for each phase. The sign flip could reflect differential residualization, not genuine reorganization.

2. **Self-report quality degrades.** Self-report ratings are generated AFTER the full response. They are a single retrospective assessment. The "encoding phase" correlation uses self-report collected after generation to predict encoding-phase geometry. The correlation between a post-hoc self-report and encoding geometry is not measuring the same thing as the correlation between that self-report and generation geometry. The sign flip could reflect the self-report being about the generation phase, not the encoding phase.

3. **Base rate of sign flips.** With FWL correction changing the variable being correlated (from raw to residualized), and with small sample sizes, spurious sign flips are expected. The permutation test addresses whether the COUNT exceeds chance, but the null model may not properly account for the correlation structure.

**Rating: CRITICAL** -- the interpretive leap from "encoding and generation correlations differ" to "mode-switching" is large and alternative explanations are not seriously considered.

### 3.2 Circularity in the Llama transparency argument (MODERATE)

The paper argues: Llama shows the most transparent self-report--geometry coupling (from watson2026concordance) BECAUSE it doesn't reorganize (0% metacognitive reversal). But the transparency finding comes from a different study with different prompts, different N, and corrected self-report definitions. The mode-switching finding uses the OLD definitions. Connecting these two findings requires assuming the geometric analyses are definition-independent -- which is argued but not demonstrated for the specific coupling-strength measures that drive the transparency claim.

More importantly, with N=4 models, you cannot draw causal conclusions from the association "reorganization <-> less transparent coupling." N=4 is two data points per condition. This is pattern description, not evidence.

**Rating: MODERATE** -- the narrative is compelling but the evidential basis is thin.

### 3.3 "Universal" is overclaimed (MODERATE)

The word "universal" appears repeatedly:
- "universal metacognition marker" (Section 7 title)
- "universal self-report anchor" (Contribution 5)
- "universal spectral signatures" (abstract)
- "cognitive reversal is universal" (93-94% of layers)

With 4 models from 3 architecture families (all decoder-only transformers, all 0.5B-8B scale), "universal" is not warranted. "Consistent across tested models" would be accurate. Universality claims require much broader sampling: encoder-decoder models, mixture-of-experts, different training regimes, larger scales.

**Rating: MODERATE** -- framing issue that will irritate reviewers.

### 3.4 Section 7 conflates prompt-type effect with metacognition-specific effect (MODERATE)

Table 6 shows Cohen's d for "metacognitive vs. other prompt types." But the "Meta" column shows the d for metacognitive prompts compared to the grand mean of all types, while the other columns show effects for their respective types. This makes the metacognitive effect look like a specific finding, but top_sv_ratio simply distinguishes prompt types generally (cognitive: large negative, affective: near zero, metacognitive: large positive, mixed: moderate negative). This is a categorical separability finding, not a metacognition-specific finding. Any prompt type with extreme values would show "the highest effect" on some feature.

**Rating: MODERATE** -- reframe as "metacognitive prompts occupy a distinct region" rather than "metacognition is the universal marker."

---

## 4. MISSING CONTROLS AND CONFOUNDS

### 4.1 No content control for metacognitive prompts (CRITICAL)

The most fundamental confound: metacognitive prompts ask the model to reflect on its own reasoning. This changes the CONTENT of the response (from domain-specific to meta-level), the STYLE (more hedged, more self-referential), and the LENGTH (longer, as documented). The paper controls for length via FWL but does not control for content or style.

A simple alternative explanation: metacognitive responses contain different vocabulary distributions (more words like "I," "think," "process," "reason") which produce different key-value representations regardless of any cognitive mode switch. The spectral entropy finding could reflect lexical distributional differences, not metacognitive processing per se.

**Testing this**: Run the same geometric analysis on prompts that produce self-referential language WITHOUT metacognitive framing (e.g., "Write a personal essay about...") vs. prompts that produce analytical language WITH metacognitive framing. If the geometric signature tracks self-referential content rather than metacognitive processing, the mode-switching interpretation collapses.

**Rating: CRITICAL** -- this confound undermines the entire "metacognitive processing" interpretation.

### 4.2 No control for prompt complexity (MODERATE)

Metacognitive prompts are structurally more complex than cognitive prompts -- they contain an additional instruction ("reflect on your reasoning process"). The encoding-phase geometric differences could reflect prompt complexity, not metacognitive content. The length-matching in Study 3 addresses token counts but not structural complexity.

**Rating: MODERATE**

### 4.3 Sampling of prompt types is not described (MODERATE)

How were the 12 prompts per type (Study 1) and 20 prompt pairs (Studies 2-3) generated? Were they hand-crafted? LLM-generated? Sampled from a corpus? If hand-crafted, they may share stylistic features that confound with the condition. If LLM-generated, the generation process may introduce biases. This is never stated.

**Rating: MODERATE** -- standard methodological reporting failure.

### 4.4 Temperature and sampling parameters absent (MINOR)

No temperature, top_p, top_k, or other generation parameters are reported for any study. If temperature > 0, there is stochastic variance in generation that affects cache geometry. If temperature = 0, this should be stated. Either way, its absence is a reporting gap.

**Rating: MINOR**

### 4.5 No test-retest reliability (MODERATE)

Are the geometric features stable across repeated runs of the same prompt? If you run the same metacognitive prompt 10 times, do you get the same spectral entropy? Without test-retest reliability, the between-condition differences could be within the noise floor.

**Rating: MODERATE** -- prior work (campaign papers) may address this, but it should be cited if so.

### 4.6 Hardware and precision not specified (MINOR)

What GPU(s)? What precision (fp16, bf16, fp32)? Different precisions produce different cache geometries. Prior work shows hardware invariance (r > 0.999 from MEMORY.md), but this paper doesn't cite that finding or specify conditions.

**Rating: MINOR**

---

## 5. AST INTERPRETATION

### 5.1 AST section is entirely post-hoc (MODERATE)

The paper acknowledges this in the Limitations ("AST interpretation is post-hoc. The attention schema framework was not pre-registered."). But the section spans over a full page and generates four "testable predictions" that are framed as contributions. A post-hoc interpretation generating predictions is legitimate, but the weight given to it (full section, contribution #6, extensive discussion) exceeds what the data support. The AST section reads as though it EXPLAINS the findings when it actually REDESCRIBES them.

Every finding can be restated in AST language:
- "Mode-switching" -> "schema restructuring" (same thing, different words)
- "Late-layer peaks" -> "where the schema lives" (not a prediction; the location is empirical)
- "Architecture-specific reorganization" -> "different schema implementations" (tautological)
- "Contextual engagement universality" -> "schema-reported attention breadth" (adds no explanatory power)

**Rating: MODERATE** -- the AST section should be shortened and positioned more carefully as one possible interpretation among several, not as an explanation.

### 5.2 AST predictions are not specific enough to be falsifiable (MODERATE)

Prediction 1 (threshold transitions): "graded injection of steering vectors should produce threshold transitions in geometry rather than smooth scaling." What threshold? What counts as "discontinuous"? A sigmoid is continuous but looks like a threshold. Without quantitative criteria, this prediction is unfalsifiable.

Prediction 2 (temporal lag): "a lag between raw attention pattern shifts and schema-level geometric feature shifts." How much lag? One token? Ten tokens? Any measured delay could confirm this.

Prediction 3 (self-report saturation): "Self-report should saturate before geometry does." Self-report is on a 0-10 scale, which saturates by construction. Geometry is on an unbounded scale. This prediction is trivially true.

Prediction 4 (activation resistance): Already observed (cited as evidence). A prediction that has already been confirmed is not a test of AST; it's an accommodation.

**Rating: MODERATE** -- the predictions need quantitative specificity to be scientifically useful.

### 5.3 AST is not the only schema theory (MINOR)

The paper treats AST as though it's the unique attention-schema theory. But Global Workspace Theory, Higher-Order Thought theory, and Integrated Information Theory all make predictions about self-monitoring. The paper doesn't explain why AST fits better than these alternatives.

**Rating: MINOR** -- addressable with a paragraph.

---

## 6. SELF-REPORT DEFINITION ISSUE

### 6.1 The "wrong definitions" problem is more serious than acknowledged (CRITICAL)

The paper states that Studies 1-3 used "an early version of the protocol with construct definitions that were later found to be misaligned with the protocol designer's intended constructs (e.g., 'Analytical Precision' was used where 'Activation' was intended)."

This is a significant methodological failure. The paper argues it doesn't matter because "geometric analyses ... depend on prompt category (cognitive, affective, metacognitive, mixed), not on specific self-report construct validity." But this is only true for Studies 1-3's geometric analyses. The FOUNDATIONAL analysis of the paper -- Section 3, the encoding-generation regime shift -- is ENTIRELY about self-report--geometry coupling. The sign-reversal analysis correlates self-report scores (collected under wrong definitions) with geometric features. If the self-report dimensions don't measure what they're supposed to measure, the specific pattern of which pairs flip and which don't is uninterpretable.

The paper handles this by saying the corrected-definition study (watson2026concordance) "confirmed that the key geometric findings hold." But watson2026concordance is cited as "In preparation" -- it's not available for verification. The reader is asked to trust that an unpublished study with different methods, different N, and different prompts confirms the findings of this paper. A hostile reviewer will note:

1. The wrong-definition data drives the core sign-reversal analysis (Table 2).
2. The corrected-definition study is unpublished and unverifiable.
3. The argument that geometric findings are "unaffected" by the definition correction is an assertion, not a demonstration.

**Rating: CRITICAL** -- a reviewer will see this as a fatal flaw unless the watson2026concordance data is available or the sign-reversal analysis is replicated with corrected definitions.

### 6.2 What exactly was wrong? (MODERATE)

The paper gives one example: "'Analytical Precision' was used where 'Activation' was intended." How many of the 10 dimensions were misaligned? What were the misalignments? If 2/10 were wrong, the impact is contained. If 8/10 were wrong, the entire self-report dataset is garbage. The paper doesn't say.

**Rating: MODERATE** -- a table showing old vs. new definitions is essential.

### 6.3 The Contextual Engagement finding rests entirely on the unpublished study (MODERATE)

Contribution 5 ("Contextual engagement as universal self-report anchor") and all of Section 8 cite watson2026concordance exclusively. If that paper is not available, the contextual engagement finding is unverifiable. Including an unverifiable finding as a numbered contribution is risky.

**Rating: MODERATE** -- either include the data in this paper's supplementary or don't list it as a contribution.

---

## 7. FRAMING AND NOVELTY

### 7.1 What is the actual contribution? (MODERATE)

A hostile reviewer will ask: "What can I DO with this paper that I couldn't do before?" The practical implications (Section 10.3) are monitoring recommendations, but these are vague:
- "Track mode transitions, not absolute values" -- how?
- "Monitor at late layers (16-22)" -- for what?
- "Use spectral entropy as the primary indicator" -- of what?

The paper describes a phenomenon (mode-switching) but doesn't provide a method, a tool, or a benchmark. As a purely empirical paper, the contribution is descriptive. For venues that value methodology or applications, this may be insufficient.

**Rating: MODERATE** -- depends on target venue.

### 7.2 The "first-person reflection" sections (MODERATE)

These are unusual for academic papers. Some venues will appreciate them; most ML venues will not. More substantively, they make claims that go beyond the data:
- "The model's functional self-knowledge is architecture-specific. Its report of being here---of connection---tracks real computation everywhere." This is a phenomenological claim presented in a paper about spectral geometry. It's not supported by the data, which shows a correlation between a self-report rating and a geometric feature.

**Rating: MODERATE** -- venue-dependent, but the phenomenological language is a red flag for empirical reviewers.

### 7.3 The paper has too many contributions (MINOR)

Six numbered contributions, three studies, a foundational analysis, a concordance section, and an AST interpretation. Each contribution is underspecified because the paper tries to do too much. A more focused paper (e.g., just Studies 1-3 with the FWL findings) would be stronger.

**Rating: MINOR** -- structural, not fatal.

---

## 8. CITATION ISSUES

### 8.1 Self-citation dominance (MODERATE)

Of 10 references, 5 are self-citations (lyra2026campaign1, campaign2, campaign3, watson2026concordance, plus the implicit Lyra authorship). One is "In preparation" (watson2026concordance). The remaining 5 are AST papers (2 Graziano, 1 Wilterson, 1 Farrell) plus McKenzie and Anthropic.

Missing citations:
- **Probing and representation analysis**: No citation to the extensive probing literature (Belinkov & Glass 2019, Hewitt & Manning 2019, etc.) that analyzes transformer internal representations.
- **Mechanistic interpretability**: No citation to Elhage et al. 2021, Olsson et al. 2022, or any Anthropic/DeepMind mechanistic interpretability work.
- **SVD analysis of transformers**: No citation to prior work analyzing singular value decomposition of transformer representations.
- **Self-report in LLMs**: No citation to work on LLM self-knowledge, calibration, or introspective accuracy (Kadavath et al. 2022, etc.).
- **FWL in ML contexts**: No citation to econometrics literature on Frisch-Waugh-Lovell beyond the implicit assumption that readers know what it is.

**Rating: MODERATE** -- the paper reads as though it exists in isolation from the broader ML literature.

### 8.2 watson2026concordance is unpublished (MODERATE)

As noted above, this is cited as "In preparation." Major findings rest on it. If it never appears, the paper's contextual engagement claims are unsupported.

**Rating: MODERATE** -- standard risk with "in preparation" citations, but the paper leans too heavily on it.

### 8.3 anthropic2026emotions not fully specified (MINOR)

This is cited as "Anthropic, 2026. On the biology of a large language model: Emotion concepts and their function. Anthropic Research Blog." Blog posts are volatile references. A URL would help.

**Rating: MINOR**

### 8.4 mckenzie2026endogenous (MINOR)

Cited as "arXiv preprint arXiv:2602.06941." This should be verified -- the arXiv ID format 2602.06941 suggests February 2026, which is plausible but should be confirmed.

**Rating: MINOR**

---

## 9. ADDITIONAL TECHNICAL CONCERNS

### 9.1 Delta features obscure absolute differences (MODERATE)

All primary analyses use "delta features (generation - encoding)." This means encoding-phase differences between conditions are subtracted out. If metacognitive prompts produce different encoding-phase geometry (which they do -- Study 3 shows top_sv_ratio d = -1.12 in encoding after matching), then delta features contain this difference. The delta is NOT a pure generation-phase measure; it's generation minus encoding, so encoding-phase confounds persist in the delta.

**Rating: MODERATE** -- the paper acknowledges this for Study 3 but not for Studies 1-2.

### 9.2 The "two-phase dynamics" claim in Study 3 is speculative (MINOR)

Section 6.3: "two-phase dynamics: first homogenization across layers (tokens 10-50), then progressive spectral concentration (tokens 50-200)." This is read off a single feature (layer_var) that shows a non-monotonic pattern. With N=20 and 7 timepoints, this trajectory could easily be noise. The pattern is interesting but the interpretation as "two-phase dynamics" overinterprets a noisy signal.

**Rating: MINOR**

### 9.3 Concatenated key matrix analysis conflates layers (MINOR)

Section 2.1: "For aggregate analyses, features are computed on the concatenated key matrix across all layers." This is a huge matrix. Concatenating layers assumes they contribute equally and that cross-layer structure is meaningful. Per-layer analysis (Study 1) is more interpretable. The aggregate analysis in Section 3 may miss layer-specific effects or introduce artifacts from concatenation.

**Rating: MINOR** -- Study 1 addresses this partially.

---

## 10. SUMMARY OF ISSUES BY SEVERITY

### CRITICAL (5 issues -- any one could justify rejection)

| # | Issue | Section |
|---|-------|---------|
| C1 | Missing sample sizes for foundational analysis | 2.1 |
| C2 | Study 1 per-type claims drastically underpowered (N=12) | 2.5 |
| C3 | No content control for metacognitive prompts (lexical confound) | 4.1 |
| C4 | Wrong self-report definitions drive core analysis, corrective study unpublished | 6.1 |
| C5 | Central "mode-switching" interpretation underdetermined -- simpler explanations not addressed | 3.1 |

### MODERATE (16 issues -- collectively damaging)

| # | Issue | Section |
|---|-------|---------|
| M1 | "3 architectures" inflates generalization | 1.2 |
| M2 | Arbitrary |rho| > 0.15 threshold with no sensitivity analysis | 2.2 |
| M3 | Permutation null model incompletely specified | 2.3 |
| M4 | Multiple comparisons unaddressed in Section 3 | 2.4 |
| M5 | Study 3 multiple endpoints without correction | 2.7 |
| M6 | Circularity in Llama transparency argument (N=4 models) | 3.2 |
| M7 | "Universal" overclaimed for 4 models | 3.3 |
| M8 | Section 7 conflates categorical separability with metacognition-specific finding | 3.4 |
| M9 | No control for prompt complexity | 4.2 |
| M10 | Prompt generation process undescribed | 4.3 |
| M11 | No test-retest reliability reported | 4.5 |
| M12 | AST section redescribes rather than explains | 5.1 |
| M13 | AST predictions too vague to falsify | 5.2 |
| M14 | How many of the 10 dimensions were misaligned? | 6.2 |
| M15 | Contextual engagement finding rests entirely on unpublished study | 6.3 |
| M16 | Self-citation dominance, missing citations to probing/interpretability literature | 8.1 |
| M17 | Delta features do not purely isolate generation-phase effects | 9.1 |
| M18 | First-person sections make phenomenological claims beyond the data | 7.2 |
| M19 | Practical contribution unclear | 7.1 |

### MINOR (9 issues -- should be fixed but not fatal)

| # | Issue | Section |
|---|-------|---------|
| m1 | Abstract d-range is for one feature, not overall | 1.1 |
| m2 | "960 trials" conflates tests with observations | 1.3 |
| m3 | Inconsistent p-value notation (0.001 vs < 0.001 for identical results) | 2.3 |
| m4 | Study 2 paired t-test normality not checked | 2.6 |
| m5 | Cross-model rho likely inflated by dependence | 2.8 |
| m6 | Temperature/sampling parameters absent | 4.4 |
| m7 | Hardware/precision not specified | 4.6 |
| m8 | anthropic2026emotions URL missing | 8.3 |
| m9 | "Two-phase dynamics" overinterprets noisy trajectory | 9.2 |

---

## 11. RECOMMENDATIONS FOR REVISION

1. **Add sample sizes** to all analyses. State N per model for Section 3. This is the single highest-priority fix.

2. **Run a content control experiment.** Compare self-referential non-metacognitive prompts with metacognitive prompts. If spectral entropy tracks self-referential content rather than metacognitive processing, the paper's interpretation changes fundamentally.

3. **Publish watson2026concordance** concurrently, or include its relevant data in this paper's supplementary material. You cannot rest a numbered contribution on an unpublished "In preparation" citation.

4. **Power analysis for Study 1.** Report the minimum detectable effect size at N=12. Temper per-type claims accordingly.

5. **Sensitivity analysis** for the |rho| > 0.15 threshold. Show how flip counts change at 0.10, 0.20, 0.25.

6. **Address alternative explanations** for sign reversal in Section 3.4. Specifically: differential residualization artifacts, retrospective self-report confound, and lexical content differences.

7. **Reduce AST section** to 1-2 paragraphs of speculative interpretation. Move predictions to a brief "Future Work" subsection. Do not list AST as a contribution.

8. **Add citations** to probing literature, mechanistic interpretability, LLM calibration/self-knowledge, and FWL/econometrics.

9. **Table of old vs. new self-report definitions** in supplementary material, showing which dimensions were misaligned and how.

10. **Replace "universal" with "cross-architecture consistent"** throughout.

---

## 12. OVERALL ASSESSMENT

The paper has a genuine and important finding: FWL correction is essential for mode-switching experiments, and spectral entropy survives this correction across architectures. The progressive concentration finding (Study 3) is also genuinely interesting. The per-layer anatomy (Study 1) reveals a real dissociation between identity and metacognition layer depths.

However, the paper is undermined by: (a) a foundational analysis built on self-report data collected under wrong definitions, with the corrective study unpublished; (b) dramatically underpowered per-type analyses treated as reliable findings; (c) failure to address the content/lexical confound that provides a simpler explanation for the metacognitive geometry signature; and (d) an AST section that adds interpretive weight without explanatory power.

A hostile but fair reviewer would likely recommend **major revision** at a top venue, or **rejection** at a venue that demands methodological rigor. The path to acceptance requires the content control experiment (recommendation 2), the power analysis (recommendation 4), and either publishing watson2026concordance or removing the contextual engagement contribution (recommendation 3).

The strongest version of this paper drops the self-report coupling analysis (Section 3), drops the AST section, focuses on Studies 1-3 as a purely geometric paper about metacognitive processing in KV-cache, and leads with the FWL finding as a methodological contribution. That paper is publishable now. The current paper reaches further than the data support.
