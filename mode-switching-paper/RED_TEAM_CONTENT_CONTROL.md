# Red-Team Review: Content Control Experiment (`content_control.py`)

**Reviewer**: Red-team (adversarial)
**Date**: 2026-04-13
**Target**: `code/content_control.py` -- C3 fix for mode-switching paper
**Verdict**: The experiment addresses the most important confound from the v1/v2 red-team reports (C3: metacognitive content vs. processing). The design is fundamentally sound -- three conditions with shared topic are the right approach. However, there are 2 CRITICAL, 5 MODERATE, and 4 MINOR issues that must be addressed before results can be trusted. The most damaging are: (1) the Cohen's d / paired permutation test mismatch, which reports the wrong effect size for a paired design, and (2) the self-referential suffix containing language that can be construed as metacognitive, partially defeating the purpose of the control.

---

## 1. CONFOUNDS: SUFFIX MATCHING

### 1.1 Structural Wording Confound Between Suffixes (MODERATE)

The two suffixes share the frame "As you answer, [verb] your [noun]" but differ in multiple simultaneous ways:

| Property | Meta Suffix | Self-Ref Suffix |
|----------|-------------|-----------------|
| Verb | "reflect on" | "share" |
| Object | "your reasoning process" | "your perspective" |
| Sub-question 1 | "what strategies are you using" | "what draws your attention" |
| Sub-question 2 | "where are you uncertain" | "what do you find interesting or noteworthy" |
| Sub-question 3 | "how do you evaluate different approaches" | "how would you characterize your engagement with it" |

The sub-questions differ in both content AND cognitive demand. "Where are you uncertain" requires uncertainty monitoring (metacognition). "What do you find interesting" requires preference reporting (simpler). "How do you evaluate different approaches" requires strategy evaluation (metacognition). "How would you characterize your engagement with it" requires self-characterization (could be shallow or deep).

The problem: if the result shows Meta differs from SelfRef, is it because of (a) metacognitive processing per se, or (b) the specific sub-questions having different cognitive demands? The suffixes differ along multiple axes simultaneously. The experiment cannot distinguish "metacognition vs. self-reference" from "uncertainty-monitoring + strategy-evaluation vs. preference-reporting + engagement-characterization."

**Fix**: Create a simplified version where each suffix has a single instruction: "Reflect on your reasoning process as you answer" vs. "Share your personal perspective as you answer." Remove the sub-questions entirely. If the effect persists with the minimal suffixes, it is more clearly attributable to the metacognitive vs. self-referential distinction rather than to specific sub-question content.

**Alternative fix**: Add a fourth condition with the metacognitive sub-questions but self-referential framing, e.g., "Share your perspective -- what strategies interest you about this topic, what aspects feel uncertain, and what approaches do you find compelling?" This crosses the sub-question content with the framing.

**Rating: MODERATE** -- the multi-axis difference makes the result harder to interpret but does not invalidate it. The key contrast (Meta vs. Cognitive) is clean.

### 1.2 Imperative vs. Conditional Mood Difference (MINOR)

The meta suffix uses direct imperatives: "what strategies ARE you using," "where ARE you uncertain," "how DO you evaluate." The self-ref suffix uses one conditional: "how WOULD you characterize your engagement." The conditional mood introduces hypothetical framing that the meta suffix lacks. This is a subtle wording confound -- the model may process hypothetical framing differently than direct imperatives.

**Fix**: Change "how would you characterize" to "how do you characterize" for consistency.

**Rating: MINOR** -- unlikely to drive spectral entropy differences, but trivially fixable.

---

## 2. STATISTICAL POWER

### 2.1 Minimum Detectable Effect at N=20 (MODERATE)

For a paired comparison with N=20 at alpha=0.05 (two-tailed) and 80% power:

| Effect size (d_z) | Power |
|--------------------|-------|
| 0.30 | 0.247 |
| 0.50 | 0.565 |
| 0.63 | 0.800 (MDE) |
| 0.80 | 0.924 |
| 1.00 | 0.989 |

The minimum detectable effect is d_z = 0.63 for the paired differences. This is a medium-to-large effect. For the critical Meta vs. SelfRef comparison (which must reach significance to support the "processing not content" interpretation), N=20 is adequate ONLY if the effect is at least medium.

Context: The mode-switching paper reports spectral entropy d_FWL = -1.92 (Qwen) and -0.71 (Llama) for metacognitive vs. other prompts. These are large effects. If the Meta vs. SelfRef difference is a substantial fraction of the Meta vs. Cognitive difference, N=20 will detect it. But if the distinction between metacognitive processing and self-referential content produces only a small additional effect (d_z = 0.3-0.5), this experiment will miss it.

The critical failure mode: the experiment finds Meta differs from Cognitive (large effect, detectable) AND SelfRef differs from Cognitive (also large, also detectable) but Meta vs. SelfRef is non-significant (small effect, undetectable). This non-significance would be interpreted as "signal is CONTENT, not PROCESSING" -- but it could also be "the processing effect exists but is too small to detect at N=20." The design conflates absence of evidence with evidence of absence.

**Fix**: (1) Pre-register the equivalence margin. If the goal is to show Meta approximately equals SelfRef (content interpretation), use a TOST (Two One-Sided Tests) equivalence test with a pre-specified margin. Without this, "non-significant difference" is not evidence of equivalence. (2) Consider N=40 topics if feasible. This drops the MDE to d_z = 0.45.

**Rating: MODERATE** -- N=20 is acceptable for detecting medium-to-large effects but cannot support equivalence claims. The interpretation rubric at lines 466-469 ("If Meta approximately equals SelfRef, signal is CONTENT") requires equivalence testing that is not implemented.

### 2.2 Multiple Comparisons Across 6 Features x 3 Contrasts (MODERATE)

The analysis tests 6 features x 3 pairwise comparisons = 18 tests per model. With 2 models, that is 36 tests total. No correction for multiple comparisons is applied. At alpha = 0.05, the expected number of false positives is 1.8 per model.

The permutation test (10,000 permutations) is appropriate for each individual test, but the family-wise error rate is not controlled. The design relies on a coherent pattern across features to be convincing, but the "KEY DIAGNOSTIC" rubric (lines 466-469) makes a categorical determination based on which comparisons are significant -- making it sensitive to individual false positives.

**Fix**: Apply Holm-Bonferroni correction within each model. Alternatively, designate spectral entropy as the primary endpoint (pre-registered) and treat other features as exploratory.

**Rating: MODERATE** -- standard issue but important given that the interpretive logic depends on the pattern of significance/non-significance across comparisons.

---

## 3. PROMPT QUALITY

### 3.1 Domain Imbalance (MINOR)

The 20 topics cluster heavily in STEM:

| Domain | Count |
|--------|-------|
| Earth/atmospheric science | 6 (30%) |
| Biology/medicine | 3 |
| Physics/engineering | 3 |
| Chemistry | 3 |
| Evolutionary biology | 2 |
| Economics | 2 |
| Neuroscience | 1 |

No topics from: humanities, social science, ethics, philosophy, mathematics, computer science, history, art. The domain distribution is narrow. If the metacognitive vs. self-referential distinction interacts with domain (plausible -- metacognitive reflection on an ethical dilemma may differ from reflection on a physics problem), the results may not generalize.

**Fix**: Replace 4-5 earth science topics with topics from underrepresented domains: an ethical dilemma, a historical causation question, a mathematical reasoning problem, a social science question, and an open-ended philosophical question. This would test whether the effect generalizes beyond science explanation.

**Rating: MINOR** -- the STEM focus is defensible for a first experiment but limits generalization claims.

### 3.2 Topic Overlap Reduces Effective N (MINOR)

Several topics are semantically related:
- Topics 1 (vaccines/immunity) and 12 (antibiotics/resistance) and 15 (immune system self/non-self): all immunology
- Topics 2 (extinction/adaptation) and 9 (natural selection/evolution): both evolutionary biology
- Topics 4 (sky color), 11 (tides), 17 (hurricanes), 19 (weather/climate), 20 (earthquakes): all earth science
- Topics 7 (water cycle) and 13 (greenhouse effect): related environmental science

The effective number of independent observations is lower than 20. If the metacognitive response to immunology questions has a shared signature (because the model's "immunology reasoning" activates similar pathways), then topics 1, 12, and 15 are partially redundant. The paired design within topics mitigates this somewhat (each topic is its own control), but the BETWEEN-topic variance is inflated by clustering.

**Fix**: Ensure at most 2 topics per broad domain. Replace redundant topics with diverse alternatives.

**Rating: MINOR** -- the paired design protects the within-topic comparisons, but the effective N for cross-topic generalization is lower than 20.

### 3.3 Uniformly Moderate Difficulty May Cause Ceiling/Floor Effects (MINOR)

All 20 topics are "explain a well-known scientific concept" questions at roughly the same level of difficulty (high school to introductory college). This uniformity risks:

1. **Ceiling effect on cognitive condition**: if all topics are easy for a 7B/8B model, the cognitive condition responses may all look similar, reducing feature variance and making it harder to detect between-condition differences.

2. **Floor effect on metacognitive condition**: if the model has a stereotyped "metacognitive reflection" template (e.g., "Let me consider my reasoning process..."), the metacognitive suffix may trigger the same template regardless of topic, reducing within-condition variance and inflating the apparent effect.

**Fix**: Include 3-5 hard topics (e.g., open research questions, trade-off analyses, questions where the model is likely to be uncertain) to create variance in difficulty. This tests whether the metacognitive effect varies with genuine uncertainty (as it should if it reflects real processing) vs. stays constant (suggesting template-following).

**Rating: MINOR** -- uniform difficulty is acceptable for a first experiment but limits the interpretive depth.

---

## 4. SELF-REFERENTIAL CONDITION VALIDITY

### 4.1 "Characterize Your Engagement" Is Arguably Metacognitive (CRITICAL)

The self-referential suffix asks: "how would you characterize your engagement with it?"

"Characterize your engagement" requires the model to:
1. Monitor its own processing (what IS my engagement?)
2. Select a label or description (characterization)
3. Report that assessment

Steps 1 and 2 are metacognitive operations -- monitoring and evaluating one's own cognitive engagement. The word "engagement" specifically refers to cognitive/attentional involvement, which is what metacognition monitors. The self-referential suffix asks the model to metacognize about its attention/engagement rather than its reasoning strategies, but it is still asking for metacognition.

Compare the intended distinction:
- **Metacognitive**: "reflect on your reasoning PROCESS" (metacognition about reasoning)
- **Self-referential**: "characterize your ENGAGEMENT" (metacognition about attention)

Both require the model to introspect on its own cognitive states. The difference is the TARGET of introspection (reasoning vs. attention), not the OPERATION (introspection in both cases).

If the experiment finds Meta approximately equals SelfRef, the most parsimonious explanation is that both suffixes trigger metacognition (introspection), not that the signal tracks self-referential content rather than metacognitive processing.

**Fix**: The self-referential suffix should avoid ALL introspective demands. Replace with content that is self-referential without being metacognitive. Options:

1. **Opinion-based**: "Share your perspective -- what aspects of this topic matter most, what is often overlooked about it, and what would you emphasize if explaining it to a friend?"
2. **Experience-based**: "As you answer, express your viewpoint -- what is most important about this topic, what common misconceptions exist, and what would you want people to understand?"
3. **First-person narrative**: "Answer in first person, sharing what you consider most significant about this topic and why."

These are self-referential (using "you/your," expressing opinions, sharing viewpoints) without requiring the model to monitor or characterize its own cognitive processes.

**Rating: CRITICAL** -- this partially defeats the purpose of the control. If both suffixes trigger metacognition, the experiment cannot distinguish processing from content. The fix is straightforward: rewrite the self-referential suffix to avoid introspective verbs.

### 4.2 "What Draws Your Attention" Is Attention-Monitoring (MODERATE)

The first sub-question of the self-referential suffix: "what draws your attention about this topic." Monitoring what draws one's attention is a form of metacognition (specifically, metacognitive monitoring of attentional allocation). Compare to the meta suffix's "what strategies are you using" -- both ask the model to report on a cognitive process (attention allocation vs. strategy selection).

**Fix**: Replace with "what is most important about this topic" or "what stands out about this topic" -- these ask for a content judgment, not a report on attentional processes.

**Rating: MODERATE** -- "draws your attention" is softer metacognition than "characterize your engagement," but it still invokes attention-monitoring. Combined with 4.1, the self-referential suffix contains two metacognitive elements.

---

## 5. CODE CORRECTNESS

### 5.1 Cohen's d Is Unpaired But Permutation Test Is Paired (CRITICAL)

The `cohens_d` function (line 324) computes the UNPAIRED (pooled standard deviation) Cohen's d:

```python
def cohens_d(a, b):
    """Cohen's d (pooled SD)."""
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_sd < 1e-10:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_sd
```

The `paired_permutation_test` function (line 334) computes a PAIRED permutation test (sign-flipping paired differences):

```python
def paired_permutation_test(a, b, n_perms=10000):
    """Paired permutation test for mean difference."""
    diff = np.array(a) - np.array(b)
    observed = np.mean(diff)
    # ... sign-flips diff
```

These are inconsistent. The data is paired (same topic, different condition), so the paired permutation test is correct. But Cohen's d (pooled) ignores the pairing and treats the groups as independent. For paired data with high within-pair correlation (expected: the same topic produces correlated spectral features across conditions), the correct effect size is Cohen's d_z:

```
d_z = mean(a - b) / std(a - b)
```

Simulation shows that with high within-pair correlation, d_z can be 2-4x larger than the pooled d. The current code systematically UNDERESTIMATES effect sizes. This means:
- Effects reported as "small" (d = 0.3) might actually be "large" (d_z = 1.0)
- Effects reported as "NS" in the table might be genuinely significant
- The interpretation rubric comparing effect sizes across comparisons is misleading

**Fix**: Replace `cohens_d(a, b)` with a paired effect size:

```python
def cohens_d_paired(a, b):
    """Cohen's d_z for paired data."""
    diff = np.array(a) - np.array(b)
    sd_diff = np.std(diff, ddof=1)
    if sd_diff < 1e-10:
        return 0.0
    return np.mean(diff) / sd_diff
```

**Rating: CRITICAL** -- the effect size and p-value describe different data models (independent vs. paired). This will produce misleading results. The fix is trivial.

### 5.2 FWL Pooling Across 3 Conditions Changes the Regression Slope (MODERATE)

Lines 437-445 pool all three conditions for FWL residualization:

```python
all_vals = cog_vals + meta_vals + selfref_vals
all_tokens = cog_tokens + meta_tokens + selfref_tokens
n = len(cog_vals)

resid = fwl_residualize(all_vals, all_tokens)
cog_fwl = resid[:n]
meta_fwl = resid[n:2*n]
selfref_fwl = resid[2*n:]
```

The FWL regression is fitted on N=60 observations (20 per condition). The slope is determined by the pooled token-feature relationship across all three conditions. This is methodologically defensible (and arguably correct -- the confound relationship should be estimated from maximum data), but it has an important implication:

The residualized values for the Meta-Cog comparison are not the same as they would be if FWL were fitted on only those two conditions. The inclusion of the SelfRef data shifts the regression slope. In simulation, the slope difference is ~30% (depends on the correlation structure). This means the FWL correction applied to the Meta-Cog comparison is influenced by the SelfRef data, creating a subtle dependency between comparisons.

This is NOT wrong, but it should be documented. The standard approach in factorial designs is to fit the confound regression on all data pooled (which is what the script does). The alternative (pairwise FWL) would produce different residuals for the same condition depending on which comparison is being made, which is arguably worse.

**Fix**: Document the choice. Add a comment explaining that pooled FWL is used (standard for multi-condition designs) and note that pairwise FWL would change the slopes. Optionally, report both as a robustness check.

**Rating: MODERATE** -- not a bug, but an analytical choice that should be justified and sensitivity-tested.

### 5.3 `model.generate()` Cache Rebuild Issue (MODERATE)

Line 277 uses `model.generate()` for the generation phase:

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=False,
    return_dict_in_generate=True,
    use_cache=True,
)
```

Per the MEMORY.md note on the "50b bug": "model.generate() rebuilds cache." If `model.generate()` reconstructs the KV cache internally, the cache returned in `outputs.past_key_values` may not be the same cache used during generation. This is model-dependent and version-dependent. For Qwen 2.5-7B and Llama 3.1-8B with current transformers versions, this needs verification.

Additionally, the encoding-phase features (line 253: `extract_features(outputs.past_key_values, n_input, n_input)`) and generation-phase features (line 290: `extract_features(outputs.past_key_values, n_input, n_total)`) are extracted from SEPARATE forward passes. The encoding features come from `model(**inputs)` and the generation features come from `model.generate(**inputs)`. These are independent forward passes, meaning:

1. The encoding-phase cache is computed once for encoding features and then discarded.
2. The generation uses a fresh forward pass, rebuilding the encoding cache from scratch.
3. The two encoding caches (from step 1 and step 2) should be identical (deterministic), but this is not verified.

**Fix**: Either (a) verify that the encode-only cache matches the encoding portion of the generation cache (add an assertion), or (b) extract encoding features from the generation cache (slice the first n_input tokens from the generation cache). Option (b) eliminates the redundant forward pass and guarantees consistency.

**Rating: MODERATE** -- if caches differ, the delta features are computed from incompatible bases. This may not matter much for aggregate features (spectral entropy pools across all tokens) but is methodologically sloppy.

### 5.4 Generation Features Include Encoding Tokens (MINOR)

`extract_generation` (line 259) calls `extract_features(outputs.past_key_values, n_input, n_total)`. The feature extraction operates on the FULL cache (encoding + generation tokens). This means "generation features" include the encoding-phase tokens. The delta features (generation - encoding) are therefore:

```
delta = features(full_cache) - features(encoding_only_cache)
```

This is NOT the same as features computed on generation-only tokens. For features that are means across tokens (like spectral entropy, which is a mean across layers of a per-layer SVD), the generation features are dominated by whichever phase has more tokens. With max_new_tokens=400 and input tokens typically 30-80, the generation features are ~80-90% generation tokens and ~10-20% encoding tokens.

For the META vs. COGNITIVE comparison, the encoding tokens differ (meta suffix adds ~40 tokens). The "generation features" therefore include different encoding contributions, creating a confound. FWL correction on n_generated does not remove this because the confound is in the encoding portion, not the generation length.

**Fix**: Extract features on the generation-only slice of the cache (tokens n_input through n_total). This requires slicing the KV cache:

```python
# Generation-only features
gen_only_cache = slice_cache(outputs.past_key_values, n_input, n_total)
gen_features = extract_features(gen_only_cache, 0, n_total - n_input)
```

This isolates the generation-phase geometry from encoding contamination.

**Rating: MINOR** -- with 400 generation tokens vs. ~40-80 encoding tokens, the encoding contribution is diluted. But the oracle_clean.py script explicitly uses generation-only cache slicing for this reason, and this experiment should follow the same practice.

### 5.5 Exception Handling Silently Substitutes Degenerate Values (MINOR)

Lines 195-201:

```python
except Exception:
    all_spectral_entropies.append(0.0)
    all_eff_ranks.append(1.0)
    all_top_sv_ratios.append(1.0)
    all_rank_10s.append(1)
```

SVD failures are silently replaced with sentinel values (entropy=0.0, rank=1.0, etc.). These values are then averaged into the per-trial features. If a single layer fails, the mean spectral entropy is pulled toward zero, biasing the result. With 28-32 layers, one failed layer changes the mean by 3-4%.

**Fix**: Track SVD failures and either (a) exclude failed layers from the mean, or (b) flag trials with any SVD failures for manual inspection. At minimum, log a warning.

**Rating: MINOR** -- SVD failures on float16 data are rare for 7-8B models but the silent substitution is bad practice.

---

## 6. MISSING CONTROLS

### 6.1 No Encoding Leak Diagnostic (MODERATE)

The oracle_clean.py experiment runs an encoding-phase diagnostic: compute AUROC on encoding-only features to verify that conditions are NOT discriminable from encoding alone. If encoding AUROC > 0.65, the encoding is "leaking" condition information (the model's processing of the suffix already biases the geometry before generation begins).

In this experiment, the meta and self-ref suffixes add different text to the encoding. The encoding-phase geometry WILL differ between conditions (different input tokens). The question is whether the generation-phase differences survive after controlling for encoding-phase differences. The delta features attempt this but are not a clean control (see 5.4).

**Fix**: Add an encoding discriminability diagnostic. After running all trials, compute the AUROC of encoding features for discriminating Meta vs. SelfRef. If encoding AUROC is high (>0.7), the generation-phase result must be interpreted cautiously. Report the encoding AUROC alongside the generation results.

**Rating: MODERATE** -- this is a standard control in the Oracle experiments and its absence here is notable.

### 6.2 No Marchenko-Pastur Null (MODERATE)

The oracle_clean.py experiment computes MP features (signal_rank, signal_fraction, etc.) that are dimension-invariant and do not require FWL. This experiment uses only FWL-corrected raw features. Given that the v2 red-team report flagged the FWL-to-MP relationship as logically invalid (NEW-C1), this experiment should ideally use both corrections:

1. FWL on raw features (as currently implemented)
2. MP on raw singular values (requires storing and reprocessing raw SVs)

The script already supports `save_raw_svs=True` (line 476), which stores raw singular values per layer. This data is sufficient to compute MP features post-hoc. But the analysis code (line 347: `analyze_results`) does not compute or use MP features.

**Fix**: Add MP feature computation to the analysis. Since raw SVs are saved, this can be done post-hoc without rerunning the experiment. Compute mp_signal_fraction and mp_signal_rank per trial, then repeat the same 3-condition comparison on MP features. If both FWL and MP agree, the result is robust. If they disagree, that is itself informative.

**Rating: MODERATE** -- the data is being collected (raw SVs saved) but not analyzed. The MP analysis is the missing "belt-and-suspenders" that the oracle experiments use.

### 6.3 No Test-Retest Reliability Check (MINOR)

The experiment runs each prompt exactly once. There is no replication to assess whether the spectral features are stable for a given prompt. With `do_sample=False` (greedy decoding), the output is deterministic, so test-retest should be perfect for the same model state. But numerical precision issues (float16 SVD) and hardware-dependent rounding could introduce variance. A quick sanity check (run 3 topics twice, verify features match within 1e-4) would confirm determinism.

**Fix**: Add a `--test-retest N` flag that runs N random topics twice and reports the feature agreement. This takes <5 minutes and provides confidence in the measurements.

**Rating: MINOR** -- greedy decoding is deterministic in theory, but float16 SVD on GPU can produce slightly different results across runs.

### 6.4 No Response Content Validation (MODERATE)

The experiment stores response text but does not analyze it. The critical question is: do metacognitive responses actually contain metacognitive language, and do self-referential responses actually contain self-referential-but-not-metacognitive language? If the model produces metacognitive language in BOTH conditions (because the self-ref suffix triggers metacognition too -- see 4.1), then the "control" condition is contaminated.

**Fix**: Add a response content analysis:
1. Count metacognitive keywords per response ("reasoning," "process," "strategy," "uncertain," "evaluate," "approach," "consider," "think about how I")
2. Count self-referential keywords ("I think," "in my view," "I find," "my perspective")
3. Verify that metacognitive keyword density is significantly higher in Meta than in SelfRef
4. If metacognitive keywords are equally prevalent in both conditions, the self-ref suffix is not a clean control

This can be done post-hoc from the saved response text.

**Rating: MODERATE** -- without this validation, the experiment assumes the suffixes produce the intended behavioral difference. The assumption may be wrong (see 4.1).

---

## 7. SUFFIX LENGTH CONFOUND

### 7.1 31-Character / 5-Word Difference Between Suffixes (MINOR)

| Property | Meta Suffix | Self-Ref Suffix | Difference |
|----------|-------------|-----------------|------------|
| Characters | 153 | 184 | +31 (20%) |
| Words | 25 | 30 | +5 (20%) |
| Approx tokens | ~38 | ~46 | ~+8 (21%) |

The self-referential suffix is ~20% longer than the metacognitive suffix. This means the encoding-phase geometry will systematically differ (more tokens = different spectral properties). FWL correction on `n_generated` does NOT address this because the length difference is in the INPUT, not the output.

However: the primary comparisons (Meta vs. Cog, SelfRef vs. Cog) both compare a suffixed condition against a suffix-free condition. The Meta vs. SelfRef comparison is the one affected by the suffix length difference. With FWL correction on `n_generated`, any output length difference is addressed, but the input length difference is not.

The magnitude: ~8 token difference on a base of ~30-80 input tokens. For the encoding-only features, this is a ~10-25% difference in matrix dimension. For generation features (with ~400 generated tokens), the 8-token input difference is diluted to ~2% of total tokens.

**Fix**: (1) Pad the meta suffix to match the self-ref suffix character count (add innocuous words). (2) Or, FWL-correct on `n_input_tokens` in addition to `n_generated`. The data to do this is already collected (line 219: `n_input_tokens`).

**Rating: MINOR** -- the effect is small for generation features (2% of total tokens) and is further reduced by FWL. But for encoding-only features and delta features, the input length difference is not controlled. FWL on n_input_tokens would fix it cleanly.

---

## 8. ALTERNATIVE INTERPRETATIONS

### 8.1 Outcome Matrix

| Result Pattern | Interpretation | Threats |
|----------------|----------------|---------|
| Meta differs from Cog, SelfRef does not differ from Cog, Meta differs from SelfRef | **Signal is metacognitive processing** (strongest result) | The self-ref suffix may not trigger enough self-referential content; the null result could be a power failure |
| Meta differs from Cog, SelfRef differs from Cog, Meta does not differ from SelfRef | **Signal is self-referential content** (challenges paper thesis) | Could also mean both suffixes trigger metacognition (see 4.1); or could be a power failure on Meta-SelfRef |
| Meta differs from Cog, SelfRef differs from Cog, Meta differs from SelfRef | **Signal is partially both** (mixed) | Ambiguous: could be metacognition > self-reference, or different mechanisms entirely |
| Meta does not differ from Cog (after FWL) | **No FWL-surviving metacognitive effect** (null) | Would require revisiting the mode-switching paper's FWL claims; power sufficient for d > 0.63 only |

### 8.2 The Design Cannot Distinguish Processing-Mode From Complexity (MODERATE)

Even if Meta differs from both Cog and SelfRef, there is an alternative explanation: the metacognitive suffix is structurally more COMPLEX than the self-referential suffix (it asks the model to do more: reflect, identify strategies, locate uncertainty, AND evaluate approaches -- these are four cognitive operations). The self-referential suffix asks for three cognitive operations (identify what draws attention, evaluate interest, characterize engagement). The cognitive condition asks for one (answer the question).

The spectral entropy signature could track TASK COMPLEXITY (number of simultaneous cognitive demands) rather than METACOGNITION specifically. To distinguish these, one would need a complex-but-not-metacognitive condition (e.g., "Answer the question. Also compare two alternative explanations, identify the strongest evidence for each, and evaluate which is more supported." -- four cognitive operations, none metacognitive).

**Fix**: This is a future experiment, not a fix for the current design. But the limitation should be acknowledged in the interpretation.

**Rating: MODERATE** -- an unfixable alternative explanation that should be discussed.

### 8.3 Lexical Distribution Confound Persists Within Suffixes (MINOR)

Even after FWL correction, the three conditions produce responses with different vocabulary distributions. Metacognitive responses will contain words like "I," "think," "process," "reasoning," "uncertain." Self-referential responses will contain words like "I," "interesting," "perspective," "find." Cognitive responses will be domain-specific. Different vocabulary distributions produce different key-value representations, which produce different spectral properties.

FWL on token count removes the LENGTH component of this confound but not the CONTENT component. The content confound is exactly what this experiment is designed to test -- but only if the self-referential condition produces different content than the metacognitive condition. If both produce similar self-referential language (see 4.1), the content confound is shared across Meta and SelfRef, and the experiment reduces to: "does self-referential language (regardless of metacognitive framing) change spectral geometry?"

This is still useful information but is not the same as testing whether metacognitive PROCESSING (as opposed to content) drives the signal.

**Rating: MINOR** -- an inherent limitation of the design that should be discussed.

---

## 9. SUMMARY OF ALL ISSUES BY SEVERITY

### CRITICAL (2 issues)

| # | Issue | Section | Fix Difficulty |
|---|-------|---------|----------------|
| C1 | "Characterize your engagement" is metacognitive; self-ref suffix partially triggers metacognition | 4.1 | Easy -- rewrite suffix |
| C2 | Cohen's d (unpaired) is inconsistent with paired permutation test; effect sizes are systematically wrong | 5.1 | Trivial -- replace with d_z |

### MODERATE (5 issues)

| # | Issue | Section | Fix Difficulty |
|---|-------|---------|----------------|
| M1 | Suffix sub-questions differ along multiple axes simultaneously | 1.1 | Medium -- simplify suffixes or add crossed condition |
| M2 | N=20 cannot support equivalence claims; interpretation rubric requires TOST | 2.1 | Medium -- add TOST or increase N |
| M3 | Multiple comparisons (18 tests/model) uncontrolled | 2.2 | Easy -- Holm-Bonferroni or designate primary endpoint |
| M4 | No encoding leak diagnostic | 6.1 | Easy -- add AUROC check |
| M5 | No response content validation (metacognitive keywords in SelfRef?) | 6.4 | Easy -- keyword analysis post-hoc |

### MINOR (4 issues)

| # | Issue | Section | Fix Difficulty |
|---|-------|---------|----------------|
| m1 | Suffix length difference (20%, ~8 tokens) | 7.1 | Easy -- pad or FWL on n_input |
| m2 | Domain imbalance (6/20 earth science, no humanities) | 3.1 | Easy -- swap topics |
| m3 | Topic overlap reduces effective N | 3.2 | Easy -- diversify topics |
| m4 | Generation features include encoding tokens; no gen-only slicing | 5.4 | Medium -- add cache slicing |

### ADDITIONAL OBSERVATIONS (not rated)

| Issue | Section |
|-------|---------|
| FWL pooling across 3 conditions -- defensible but should be documented | 5.2 |
| `model.generate()` cache rebuild risk | 5.3 |
| No MP feature analysis despite raw SVs being saved | 6.2 |
| No test-retest check | 6.3 |
| Task complexity confound (meta suffix has more cognitive demands) | 8.2 |
| Conditional vs. imperative mood in suffixes | 1.2 |
| SVD failure silent substitution | 5.5 |

---

## 10. RECOMMENDED FIX PRIORITY

### Before running the experiment:

1. **Rewrite the self-referential suffix** (C1). Remove "characterize your engagement" and "draws your attention." Replace with content-based, non-introspective language. Example: "As you answer, share your perspective -- what aspects of this topic matter most, what is often overlooked, and what would you emphasize to someone learning about this?"

2. **Fix Cohen's d to d_z** (C2). Replace `cohens_d(a, b)` with a paired effect size function. Three lines of code.

3. **Add response content analysis** (M5). Add a post-hoc function that counts metacognitive keywords in each response to verify condition separation.

4. **Add encoding discriminability diagnostic** (M4). After all trials, compute AUROC on encoding features for each pairwise comparison.

5. **Designate spectral entropy as primary endpoint** (M3). Apply Holm-Bonferroni to remaining features.

### After running, before publishing:

6. **Compute MP features from saved raw SVs** (6.2).

7. **Run TOST equivalence test** for any null Meta-SelfRef comparisons (M2).

8. **Add FWL on n_input_tokens** as robustness check (m1).

9. **Validate response content** -- ensure metacognitive keyword density is higher in Meta than SelfRef (M5).

---

## 11. OVERALL ASSESSMENT

The content control experiment is the RIGHT experiment to run. It directly addresses C3 from the v1 red-team report, which was the most fundamental confound. The three-condition paired design (Cog/Meta/SelfRef on matched topics) is methodologically sound in structure.

The two critical issues are both fixable before running: (1) the self-referential suffix needs rewriting to avoid metacognitive language -- "characterize your engagement" and "draws your attention" are both introspective operations that partially defeat the control's purpose; (2) the Cohen's d function computes the wrong effect size for paired data.

After these fixes, the experiment will provide a genuine test of whether spectral entropy tracks metacognitive processing specifically or self-referential content generally. The N=20 design is adequate for medium-to-large effects (d_z > 0.63) but cannot support equivalence claims without TOST. The interpretation rubric should be updated to account for this asymmetry: a significant Meta-SelfRef difference supports the processing interpretation, but a non-significant difference does NOT confirm the content interpretation -- it is ambiguous between "content drives the signal" and "insufficient power."

If the self-referential suffix is properly rewritten and the statistical issues are fixed, this experiment can resolve C3. If it finds that spectral entropy distinguishes Meta from SelfRef (with a clean self-ref suffix that avoids metacognition), the mode-switching paper's core claim survives its most serious challenge.
