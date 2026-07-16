# AGNI REVIEW V2: Position-Specific Cache Tracing (Post-Fix)

**Reviewer**: Project Agni (hostile methodological review)
**Date**: 2026-07-16
**Script**: `position_specific_cache_tracing.py` (revised)
**Pre-registration**: `position_specific_prereg.json` (NOT revised -- see P1)
**Prior review**: `POSITION_SPECIFIC_AGNI.md`
**Review scope**: Verify K1 and K2 fixes, check for remaining or new issues

---

## VERDICT: CONDITIONAL PASS

The two critical kills from V1 (K1: wrong hidden state, K2: content-density confound) are both genuinely fixed. No new kills. One serious issue carried over from V1 (S1: no statistical test) must be addressed before the results can be interpreted. Several moderate issues should be documented or fixed but are not blocking.

**Run after fixing S1. Budget 1-2 hours.**

---

## K1 FIX VERIFICATION: CORRECT

Lines 252-264:

```python
H_input = out.hidden_states[l][0, :, :].float().cpu()
ln = model.model.layers[l].input_layernorm
H = ln(H_input.to(model.device)).float().cpu()
W_K = model.model.layers[l].self_attn.k_proj.weight.float().cpu()
W_V = model.model.layers[l].self_attn.v_proj.weight.float().cpu()
```

Analysis:

1. **`hidden_states[l]`**: In HuggingFace transformers, `hidden_states[0]` = embedding output, `hidden_states[l]` = output of layer l-1 = INPUT to layer l. Using `hidden_states[l]` as the input to layer l is **correct**. The prior version used `hidden_states[l+1]` (output of layer l), which included the attention and MLP residuals -- the very computation being studied.

2. **`input_layernorm` applied**: The actual cache computes `K = RoPE(W_K @ LN1(h))` and `V = W_V @ LN1(h)`. Applying `model.model.layers[l].input_layernorm` before W_K/W_V projection matches this. **Correct.**

3. **RoPE omitted for K**: The docstring (lines 26-31) explains that within-position comparisons are RoPE-invariant. This is correct: RoPE applies the same position-dependent rotation to ALL content at a given position, so comparing two directions at the same position cancels RoPE. The experiment computes R-squared per-position (directions and target are both pre-RoPE at the same position), then compares R-squared VALUES across positions. Since R-squared is invariant under orthogonal rotation (RoPE is orthogonal per-head), the pre-RoPE simplification is **valid and well-documented**.

**K1 verdict: FIXED. This now measures what is actually in the cache.**

---

## K2 FIX VERIFICATION: WELL-DESIGNED

The fix adds two components:

### A. Concept-specific directions (lines 133-144, 214-222)

For each factual probe, the ANSWER TOKEN's transported J-lens direction is computed:

```python
concept_token_id = tok.encode(concept_text, add_special_tokens=False)[0]
concept_transported = W[concept_token_id] @ J_l  # (d_model,)
```

This gives a d_model-dimensional direction specific to "Italy", "Jupiter", etc. At concept-carrying positions, this direction should align more than at non-carrying positions -- a test that content-density alone cannot explain, because the direction is ANSWER-specific, not content-generic.

### B. Cross-concept control (lines 274-296)

For each factual probe, a DIFFERENT probe's concept direction is tested at the same positions. The pairing is a circular shift through the 7 factual probes:

```
Italy -> Jupiter, Jupiter -> Shakespeare, Shakespeare -> H2O,
H2O -> Einstein, Einstein -> gold, gold -> heart, heart -> Italy
```

Logic: if own-concept R-squared > cross-concept R-squared at concept-carrying positions, content-density is ruled out. Both directions are equally "concept-like" (both are transported vocabulary directions of similar structure), but only the matching one should preferentially align with the relevant positions.

**K2 verdict: FIXED. The cross-concept control is the right design. See S-NEW and M-NEW below for refinements.**

---

## CARRIED-OVER ISSUES FROM V1

### S1: No statistical test (STILL SERIOUS -- MUST FIX)

**Lines**: 508-528 (generic verdict), 539-551 (concept-specific verdict)

The generic verdict still uses hardcoded thresholds (0.5x for "strong", 0.2x for "weak"). The concept-specific verdict uses a raw comparison (`own > cross`). Neither arm has a significance test.

With 7 factual probes, ~2-3 concept positions per probe vs ~10-15 non-concept positions, the variance across probes can easily dominate the signal. A single outlier probe can flip the verdict.

**Why this blocks**: Without a p-value, you cannot distinguish a real position-specific signal from noise. The prior last-token result was null -- if this result is marginally positive, you need statistical rigor to claim it differs from the prior null.

**Fix** (30 minutes): Add a within-probe permutation test. For each probe x layer, permute concept/non-concept labels across positions, recompute the mean difference, repeat 10000 times. The p-value is the fraction of permuted differences exceeding the observed difference. Report per-probe p-values and a combined p-value (Fisher's method or Stouffer's). For the concept-specific arm, the test is: permute which positions get the "own concept" R-squared vs "cross concept" R-squared measurement.

Alternatively, a Mann-Whitney U per probe x layer is simpler and adequate. Report the per-cell p-values and Bonferroni-correct for n_layers x n_probes comparisons.

---

### M1: Position confound unaddressed (STILL MODERATE)

Concept regions cluster at positions ~3-8 in ~12-15 token prompts. Non-concept positions include position 0 (BOS/attention-sink effects) and the final position (generation position, where residual stream is dominated by next-token prediction). These structural position effects can create spurious lift differences unrelated to concept content.

The prereg prescribes this mitigation (T2) but the code does not implement it.

**Fix**: When computing the concept-vs-non-concept contrast, restrict non-concept positions to those within +/-2 of any concept position. This removes the BOS and final-position outliers. Alternatively, include absolute position as a covariate (but with so few data points per probe, a covariate model is fragile -- the position-matching approach is simpler and more robust).

---

### S2: Fabricated/factual separation (PARTIALLY FIXED)

The concept-specific summary (lines 481-505) correctly filters for factual probes only. Good.

The generic summary (lines 457-472) and generic verdict (lines 508-528) still aggregate ALL probes including fabricated. This means:
- The generic verdict includes fabricated probes in the signal estimate
- H4 (fabricated should show no signal) is never explicitly tested in the generic arm

**Fix**: Split the generic summary into factual-only and fabricated-only blocks. Explicitly evaluate H4 by reporting fabricated probe lift separately and comparing to factual.

---

### M3: Generic summary reports only lift, not absolute R-squared (STILL MODERATE)

The generic summary reports lift ratios (concept_mean_lift_k, etc.) but not absolute R-squared. With 10 directions in 4096 dimensions, expected random R-squared is ~10/4096 = 0.0024. A "3x lift" headline could mask an absolute R-squared of 0.007 -- statistically above random but practically negligible (0.7% variance explained).

**Fix**: Report absolute R-squared alongside lift in the summary. Flag results where absolute R-squared < 0.01 as "above random but practically negligible."

---

### N2: R-squared clipped at zero (STILL MINOR)

Line 159: `return max(..., 0.0)` discards negative R-squared. Negative R-squared is informative: it means the directions ANTI-predict the target. If concept positions show positive R-squared and non-concept positions show negative R-squared, clipping hides half the contrast.

**Fix**: Return unclipped R-squared. Let downstream analysis handle negatives appropriately.

---

## NEW ISSUES

### S-NEW: Residual stream positive control uses wrong hidden state

**Severity**: SERIOUS (conceptual error in control, though conservative direction)
**Lines**: 301, 312, 345, 350

The variable `h_pos = H[pos]` is the POST-layernorm state (`LN1(hidden_states[l])`). This is correctly used for K/V projection (lines 302-303). But it is ALSO used for the "residual stream positive control" (line 312: `r2_resid = compute_r2(dirs, h_pos)`), and for the concept-specific residual R-squared (lines 345, 350).

The J-lens is fitted to predict output logits from the RESIDUAL STREAM, which is `hidden_states[l]` (pre-layernorm), not `LN1(hidden_states[l])` (post-layernorm). The positive control should test whether J-lens directions align with the residual stream -- and it should use the actual residual stream state.

The pre-layernorm state IS available as `H_input` (line 257) but is never used for any R-squared computation.

**Impact**: RMSNorm rescales the vector and applies learned weights. It does not destroy high-dimensional structure, so a strongly positive control will likely still pass. But marginal signals could be affected, and conceptually the measurement is wrong: you are comparing J-lens directions (designed for the residual stream) against a post-layernorm state (which is not the residual stream).

**Conservative direction**: If the control PASSES despite the mismatch, it is actually a STRONGER positive signal (works even on the wrong representation). If it FAILS, the failure is ambiguous -- it could be because concepts are not position-localized OR because the layernorm mismatch obscures them.

**Fix** (5 minutes): Add `h_pos_raw = H_input[pos]` and use it for all residual stream R-squared computations:

```python
h_pos_raw = H_input[pos]   # pre-layernorm: actual residual stream
h_pos = H[pos]              # post-layernorm: for K/V projection
k_pos = W_K @ h_pos
v_pos = W_V @ h_pos
r2_resid = compute_r2(dirs, h_pos_raw)
```

---

### M-NEW-1: Single-token assumption in concept direction

**Severity**: MODERATE
**Line**: 141

```python
concept_token_id = tok.encode(concept_text, add_special_tokens=False)[0]
```

Only the FIRST token of the concept text is used. For single-token concepts ("Italy", "Jupiter", "gold", "heart"), this is fine. For multi-token concepts, it discards all but the first subword.

Likely tokenizations for Qwen3-8B:
- "Italy" -- likely single token
- "Jupiter" -- likely single token
- "Shakespeare" -- likely single token (common proper noun)
- "H2O" -- could be "H" + "2" + "O" (3 tokens), in which case the direction is for "H" alone
- "Einstein" -- likely single token
- "gold" -- single token
- "heart" -- single token

If "H2O" tokenizes to multiple tokens, the concept-specific direction for that probe is the direction for "H", not for "H2O". This weakens the concept-specific test for that one probe.

**Fix**: Add an assertion that checks the tokenization length. If multi-token, either (a) average the transported directions across all concept tokens, or (b) use the final subword token (which typically carries the most concept-specific information in autoregressive models). Print a warning for any multi-token concept.

---

### M-NEW-2: Single cross-concept control per probe

**Severity**: MODERATE
**Lines**: 277-280

Each factual probe gets exactly one cross-concept control (circular shift). With 7 probes, there are 7 own-vs-cross comparisons but each uses a fixed pairing. If any pair happens to be semantically related (e.g., "gold" and "heart" could both relate to physical/body concepts), the cross R-squared for that pair could be inflated, weakening the control.

**Fix**: Average over ALL other factual concepts as the cross-concept control. For probe i, compute cross R-squared as the mean R-squared using each of the other 6 concept directions. This gives a more robust baseline with lower variance. Implementation cost: one additional loop and ~6x more R-squared computations per position (negligible given the overall runtime).

---

### M-NEW-3: Generic direction projection recomputed inside position loop

**Severity**: MODERATE (performance, not correctness)
**Lines**: 306-309

```python
for pos in range(seq_len):
    dirs_k = (W_K @ dirs.T).T       # same for every pos
    dirs_v = (W_V @ dirs.T).T       # same for every pos
```

These projections do not depend on `pos`. They are recomputed seq_len times per layer per probe (approximately 10 probes x 4 layers x 15 positions = 600 redundant matrix multiplications). Move them outside the position loop.

---

### P1: Pre-registration not updated to match fixes

**Severity**: DOCUMENTATION (not methodological)
**Lines**: prereg.json line 43

The prereg states:
```json
"L4": "hidden_states[l+1] for layer l output"
```

This describes the OLD (incorrect) behavior. The script now correctly uses `hidden_states[l]`. The prereg was not updated.

Additionally, the prereg hypotheses (H1-H4) cover only the generic direction analysis. The concept-specific analysis (K2 fix) introduces new implicit hypotheses that are not pre-registered:

- H5 (implicit): Own-concept R-squared > cross-concept R-squared at concept-carrying positions in K-space
- H6 (implicit): Own-concept R-squared > cross-concept R-squared at concept-carrying positions in V-space

These should be added to the prereg before running.

**Fix**: Update the prereg to reflect the K1 fix (L4 → "hidden_states[l] for layer l input + input_layernorm") and add H5/H6 with falsification thresholds.

---

### N-NEW-1: Own-vs-cross direction similarity not reported

**Severity**: MINOR (interpretability)

If the J-lens has poor frequency resolution across vocabulary items, the own-concept and cross-concept directions could be nearly parallel (high cosine similarity). In that case, own approximately equals cross for all probes, and the cross-concept control cannot discriminate -- but also cannot false-positive. Reporting the cosine similarity between own and cross directions per probe would help interpret the cross-concept comparison.

**Fix**: After computing own and cross concept directions, report `cos(own, cross)` per probe per layer. If consistently > 0.9, note that the J-lens directions for these vocabulary items are poorly separated and the cross-concept control has low discriminative power.

---

### N-NEW-2: Concept-specific analysis has no negative control arm

**Severity**: MINOR (design limitation)

The concept-specific analysis only applies to factual probes (correctly, since fabricated probes have `concept=None` and no answer token). This means the H4 negative control (fabricated entities should show no signal) only applies to the generic direction analysis, not to the concept-specific analysis.

This is inherent to the design (fabricated concepts have no ground-truth answer token to compute a concept-specific direction from) and not fixable without a different control strategy. Document it.

---

## ANSWERS TO SPECIFIC QUESTIONS

### Q1: Does the K1 fix correctly capture what is actually in the KV cache?

**Yes.** `hidden_states[l]` is the input to layer l (the residual stream at that point). Applying `input_layernorm` gives `LN1(h_l)`, which is exactly the argument to W_K and W_V in the Qwen3 decoder:

```
K_cached = RoPE(W_K @ LN1(h_l))
V_cached = W_V @ LN1(h_l)
```

The script computes `W_K @ LN1(h_l)` (pre-RoPE K) and `W_V @ LN1(h_l)` (exact V). Pre-RoPE is a valid simplification for this experiment because: (a) RoPE is an orthogonal rotation per head, (b) R-squared is invariant under orthogonal rotation, and (c) both the target and the reference directions are in the same pre-RoPE frame.

The one caveat: the "residual stream positive control" incorrectly uses the post-layernorm state instead of the pre-layernorm residual stream (see S-NEW above). This affects the CONTROL, not the primary K/V measurement.

### Q2: Does the K2 fix adequately address the content-density confound?

**Yes, for the concept-specific analysis arm.** The cross-concept control is the right design: if `R-squared(own_concept, concept_positions) > R-squared(cross_concept, concept_positions)`, content-density cannot explain the difference because both directions are equally "concept-like" but only the matching one should preferentially align.

**No, for the generic direction analysis arm.** The generic analysis (top-10 transported directions, same for all probes) still compares content words vs function words. This arm remains confounded by content density and should be labeled as exploratory/preliminary, not confirmatory.

The experiment would benefit from positioning the concept-specific analysis as PRIMARY and the generic analysis as SECONDARY/EXPLORATORY, rather than the current structure which presents generic first and concept-specific second.

### Q3: Are there remaining confounds the prior review missed?

**One new finding**: the residual stream positive control uses the post-layernorm state (S-NEW). This was not identified in V1 because V1 used `hidden_states[l+1]` for everything (which was wrong for a different reason). The fix for K1 introduced a correct distinction between pre-layernorm and post-layernorm states but only applied it to K/V, not to the residual stream control.

Carried-over confounds from V1 that were not addressed:
- M1 (position confound): concept positions cluster mid-sequence; non-concept positions include BOS and final position
- S2 (generic summary mixes factual and fabricated): partially fixed for concept-specific arm only

### Q4: Is the cross-concept control well-designed?

**The principle is sound. The implementation is adequate but could be stronger.**

Sound: using one concept's direction at another concept's positions directly tests specificity. Content-density cannot produce own > cross because both directions have the same structural relationship to the vocabulary.

Adequate: each probe gets exactly one cross-concept (circular shift), giving 7 pairwise comparisons. This is sufficient for a pilot.

Could be stronger: averaging over ALL 6 other concepts as the cross-concept baseline (instead of just one) would reduce variance and eliminate sensitivity to accidental semantic similarity between paired concepts (see M-NEW-2).

---

## KNOWN KILLS CHECKLIST (UPDATED)

| Kill | Status | Notes |
|------|--------|-------|
| C1: model.generate() rebuilds cache | CLEAR | Forward pass only |
| C2: Directions on test data | CLEAR | Directions from pre-fitted lens |
| C3: Different code paths | CLEAR | Same code for concept and non-concept |
| L4: hidden_states off-by-one | **FIXED** | Now uses hidden_states[l] (input to layer l) + input_layernorm |
| S4: No FWL on token count | CLEAR | Within-prompt comparison |
| M2: No seed | CLEAR | Seeds set (lines 50-51) |
| K1 (V1): Wrong hidden state | **FIXED** | hidden_states[l] + input_layernorm. RoPE simplification valid. |
| K2 (V1): Content-density confound | **FIXED** | Concept-specific directions + cross-concept control |
| S1 (V1): No statistical test | **NOT FIXED** | Still hardcoded thresholds. Must add permutation or rank test. |
| S2 (V1): Fabricated not separated | **PARTIALLY FIXED** | Fixed for concept-specific arm. Generic arm still mixes. |
| M1 (V1): Position confound | **NOT FIXED** | Still compares all non-concept positions. |
| M3 (V1): Lift without absolute R-squared | **NOT FIXED** | Generic summary still lift-only. |
| N2 (V1): R-squared clipped | **NOT FIXED** | Still clips at zero. |
| S-NEW: Residual control wrong state | **NEW** | Uses post-layernorm instead of pre-layernorm. |

---

## MINIMUM FIXES BEFORE RUNNING

1. **S1: Add a statistical test.** Mann-Whitney U per probe x layer, or within-probe permutation test. Report p-values. Remove hardcoded verdict thresholds. (30 min)
2. **S-NEW: Fix residual stream positive control.** Use `H_input[pos]` for `r2_resid`, `r2_concept_resid`, `r2_cross_resid`. (5 min)
3. **P1: Update prereg.** Fix L4 entry. Add H5/H6 for concept-specific analysis. (10 min)

## SHOULD-FIX BEFORE INTERPRETING RESULTS

4. **M-NEW-1: Assert single-token concept.** Print warning or handle multi-token. (10 min)
5. **M-NEW-2: Average all cross-concepts.** Use mean over 6 other concepts instead of single circular-shift partner. (20 min)
6. **S2: Separate generic summary.** Split factual vs fabricated in generic arm. Explicitly evaluate H4. (15 min)
7. **M1: Restrict position comparison.** Exclude BOS and final position from non-concept baseline, or restrict to +/-2 of concept positions. (15 min)
8. **M-NEW-3: Move dirs_k/dirs_v outside position loop.** Performance only. (5 min)

Total estimated fix time: S1 + S-NEW + P1 = 45 minutes mandatory. Items 4-8 = 65 minutes recommended.

---

*Project Agni. The K1 and K2 fixes are genuine and correct. The experiment now measures what it claims to measure. The concept-specific cross-concept control is well-designed. The remaining issues are tractable. Fix S1 (stats) and S-NEW (residual control state), then run.*
