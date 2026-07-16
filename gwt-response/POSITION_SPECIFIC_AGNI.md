# AGNI REVIEW: Position-Specific Cache Tracing

**Reviewer**: Project Agni (hostile methodological review)
**Date**: 2026-07-15
**Script**: `position_specific_cache_tracing.py`
**Pre-registration**: `position_specific_prereg.json`

---

## VERDICT: FAIL — DO NOT RUN

Two critical kills, two serious issues, and a conceptual weakness that limits interpretability even if the technical errors are fixed. One of the kills is a KNOWN issue from your own prior synthesis that was documented and not addressed.

---

## CRITICAL ISSUES (kills)

### K1: Wrong hidden state for K/V projection (L4 variant — three-layer error)

**Severity**: CRITICAL / KILL
**Lines**: 205, 208-209, 213-216

The experiment title is "Cache Tracing" but the code does not compute what is in the cache.

From DECOMP_AGNI_OPUS.md lines 60-68, the Qwen3 decoder layer computes:

```
residual = hidden_states                              # h_L = hidden_states[l]
hidden_states = input_layernorm(hidden_states)        # LN1(h_L)
K = W_K @ LN1(h_L)  [+ RoPE]                        # <-- actual cache
V = W_V @ LN1(h_L)                                   # <-- actual cache
attn_output = self_attn(LN1(h_L), ...)               # post-W_O
hidden_states = residual + attn_output                # h_L + attn
hidden_states = hidden_states + MLP(LN2(h_L + attn)) # h_L + attn + MLP = hidden_states[l+1]
```

The script computes:

```python
H = out.hidden_states[l + 1]  # h_L + attn + MLP  (OUTPUT of layer l)
k_pos = W_K @ H[pos]          # W_K @ (h_L + attn + MLP)
v_pos = W_V @ H[pos]          # W_V @ (h_L + attn + MLP)
```

The actual K/V cache at layer l stores:

```
K_cached = RoPE(W_K @ LN1(hidden_states[l]))
V_cached = W_V @ LN1(hidden_states[l])
```

Three errors stacked:
1. **Wrong hidden state**: `hidden_states[l+1]` (output of layer l) instead of `hidden_states[l]` (input to layer l). These differ by the full attn + MLP residual — the very computation being studied.
2. **Missing input_layernorm**: The actual K/V are computed from `LN1(h)`, not raw `h`. RMSNorm rescales and can change relative magnitudes.
3. **Missing RoPE for K**: RoPE rotates each head's K vector by a position-dependent angle. Since this experiment is SPECIFICALLY about position effects, ignoring position-dependent rotations is especially problematic.

For V-space, errors (1) and (2) apply. For K-space, all three apply.

The quantity `W_K @ hidden_states[l+1]` does not exist anywhere in the model's computation. You are measuring alignment with a synthetic object, not with the cache.

**Your own synthesis documents this kill.** COALITION_GWT_SYNTHESIS.md line 117:

> **R2**: Mathematically wrong -- cached keys go through RMSNorm + RoPE. Projecting post-RoPE keys onto unrotated basis injects position artifacts. Fix: pre-RoPE extraction.

FABLE_AST_GWT_CACHE_SYNTHESIS.md line 58:

> (ii) apply pre-W_V RMSNorm

cache_tracing.py line 180:

> V-cache entry: W_V @ h (simplified -- ignoring RMSNorm for now)

The fix was prescribed. It was not implemented. This experiment repeats a documented error.

**Fix**: Replace lines 205-216 with:

```python
h_input = out.hidden_states[l][0, :, :].float().cpu()  # INPUT to layer l
# Apply input_layernorm
ln = model.model.layers[l].input_layernorm
h_normed = ln(h_input.to(model.device)).float().cpu()

for pos in range(seq_len):
    h_pos = h_normed[pos]
    k_pos = W_K @ h_pos   # pre-RoPE K (acceptable simplification if documented)
    v_pos = W_V @ h_pos   # actual V-cache value
```

For K, either (a) apply RoPE per-position and project J-lens directions through the same rotation, or (b) work pre-RoPE and document the simplification. Option (b) is defensible since RoPE is position-dependent and applies uniformly to all content, so within-position comparisons are RoPE-invariant. But this reasoning must be stated.

---

### K2: Content-density confound (concept = content words, non-concept = function words)

**Severity**: CRITICAL / KILL
**Lines**: 38-89, 186-189

The concept_region tokens are semantically rich content words: "boot", "largest planet", "Danish prince", "general relativity", "atomic number 79", "pumps blood". The non-concept positions contain function words: "the", "is", "on", "of", "in", "our", "with".

ANY content-sensitive feature set will show higher alignment at content positions than at function-word positions. This is not evidence of concept-specific caching. It is evidence that content words carry more information than function words — a trivial finding.

The J-lens directions are the SAME for all probes (lines 173-176): the top-10 globally-dominant transported vocabulary directions. These are not concept-specific. The experiment tests whether generic high-information directions align more with high-information positions. The answer is trivially yes.

This confound cannot be resolved by the fabricated-entity control alone. Fabricated entity names ("Thornberry-Nakamura") are also semantically rich tokens. If they show signal too, the prereg interprets this as "entity-recognition, not concept-caching" — but the actual explanation is simpler: content words vs. function words, with no concept or entity recognition involved.

**Fix**: Two options, either alone is sufficient:

(a) **Within-content matching**: For each concept position, select a non-concept position matched on token type (content word at a similar absolute position). Compare concept content-words to non-concept content-words, not concept content-words to function words.

(b) **Concept-specific directions**: Instead of the same 10 generic directions for all probes, use concept-SPECIFIC directions — e.g., the transported vector for the ANSWER token (the "Italy" row of `W_U @ J_l` for probe 0). Then test whether Italy-specific directions align more with "boot" positions than with "largest planet" positions. This is a cross-concept specificity test that content density alone cannot explain.

---

## SERIOUS ISSUES

### S1: No statistical test

**Severity**: SERIOUS
**Lines**: 300-334

The verdict uses hardcoded thresholds (0.5x for "strong", 0.2x for "weak") applied to mean lift differences. There is no significance test, no confidence interval, no effect size estimate with uncertainty.

With 7 factual probes and ~2-3 concept positions per probe vs ~10-15 non-concept positions, the variance across probes could easily dominate the signal. A permutation test (shuffle concept/non-concept labels within each probe, recompute lift difference, 10000 permutations) would establish whether the observed difference exceeds chance.

**Fix**: Add a permutation test. For each probe x layer, permute the position labels (concept vs non-concept) and recompute the mean lift difference. The p-value is the fraction of permuted differences exceeding the observed difference. Report the p-value alongside the lift difference.

---

### S2: Fabricated control not separated in summary statistics

**Severity**: SERIOUS
**Lines**: 299-315

The summary loop at lines 299-315 aggregates ALL results, including fabricated-entity probes. H4 states fabricated probes should show no position-specific signal, but the analysis never separates them. The fabricated probes dilute the factual signal in the summary AND the fabricated control is never explicitly evaluated.

**Fix**: Separate the summary into three blocks:
1. Factual probes only (test H1/H2/H3)
2. Fabricated probes only (test H4)
3. Factual vs fabricated comparison (direct contrast)

---

## MODERATE ISSUES

### M1: Position confound (early vs late) unaddressed in code

**Severity**: MODERATE
**Lines**: 186-189

The prereg acknowledges T2 ("position effects independent of concepts") and even prescribes a mitigation: "Compare concept positions to non-concept positions at SIMILAR sequence positions." The code does not implement this mitigation. Concept tokens tend to cluster in the mid-sequence (positions ~4-8 in ~12-15 token prompts), while non-concept positions include position 0 (beginning) and the final token (end). Positional encoding effects, attention sink effects at position 0, and recency effects at the final position could all create spurious differences.

**Fix**: Either (a) restrict non-concept comparison to positions within +/-2 of the concept positions, or (b) include position index as a covariate in the statistical test.

---

### M2: Token-matching for concept_region is fragile under BPE context sensitivity

**Severity**: MODERATE
**Lines**: 131-149

`find_concept_positions` tokenizes `concept_region` independently (`tok.encode(concept_region)`) and searches for the resulting token sequence in the full prompt's tokenization. BPE tokenizers are context-sensitive: the same substring can tokenize differently depending on surrounding characters (e.g., leading space). If exact matching fails, the fallback (lines 146-148) uses word-level substring matching:

```python
if any(word.lower() in text.lower() for word in concept_region.split()):
```

This matches ANY token whose decoded text contains any word from the region. For "largest planet", it matches any token containing "largest" OR any token containing "planet" — which could match tokens at unintended positions if these words appear elsewhere, and it also could include partial-word matches (e.g., a token like "plants" would match "plan").

**Fix**: (a) Print the identified positions and matched tokens for EVERY probe in the output (already done at line 195 — good). (b) Add an assertion or warning if no positions are found. (c) Consider adding leading-space variants to the exact-match attempt: `tok.encode(" " + concept_region)`.

---

### M3: Lift metric amplifies noise at low baselines

**Severity**: MODERATE
**Lines**: 257-259

The lift metric divides actual R^2 by mean random R^2. With 10 directions in 4096 dimensions, expected random R^2 is approximately k/d = 10/4096 = 0.0024. If actual R^2 is 0.005, lift is ~2.1x. This sounds meaningful but the absolute R^2 is 0.5% — effectively zero explanatory power.

The summary reports only lifts, not absolute R^2 values. A "3x lift" headline could mask an absolute R^2 of 0.007.

**Fix**: Report absolute R^2 alongside lift in the summary. Add an interpretability threshold: if absolute R^2 < 0.01 (1%), flag the result as "statistically above random but practically negligible."

---

## MINOR ISSUES

### N1: 20 random draws — adequate for means, not for distributional claims

**Severity**: MINOR
**Lines**: 36, 234-246

20 random draws is sufficient for estimating the mean random R^2 (the distribution is tight because k/d is small). It is NOT sufficient for percentile-based thresholds (e.g., "above the 95th percentile of random"). The current code only uses the mean, so 20 is adequate. But if the analysis is later extended to use percentiles, increase to 100+.

---

### N2: R^2 clipped at zero hides anti-correlation

**Severity**: MINOR
**Line**: 127

`return max(float((1 - ss_res / ss_tot).item()), 0.0)` clips negative R^2 to zero. Negative R^2 means the J-lens directions ANTI-correlate with the target — the model is doing the OPPOSITE of what the directions predict. This is informative: if concept positions show positive R^2 and non-concept positions show negative R^2, the contrast would be stronger (and more meaningful) than if non-concept positions show zero. Clipping discards this information.

**Fix**: Return the raw R^2 (allow negative). Let the lift computation handle it appropriately (e.g., report signed R^2 separately from unsigned lift).

---

### N3: Same directions for all probes limits interpretability

**Severity**: MINOR (design choice, not a bug)
**Lines**: 173-176

The top-10 transported directions are the same for every probe. They capture the globally most-amplified vocabulary items, not concept-specific directions. Even with all technical fixes applied, the experiment measures "does the generic workspace signal concentrate at content-carrying positions?" — not "does the SPECIFIC CONCEPT appear at its source position."

This limits the theoretical contribution. A positive result shows positional concentration of generic J-lens content. A negative result is ambiguous: concept-specific directions might still localize, but the generic ones don't.

**Upgrade path**: After fixing K1 and K2, consider adding a concept-specific arm using the answer token's transported vector as the direction set. This directly tests the paper's thesis.

---

## ANSWERS TO SPECIFIC QUESTIONS

### Q1: Is the position-identification method (token matching for concept_region) sound?

**Partially.** The exact-match approach is fragile under BPE context sensitivity but will work for most of these probes because the concept regions are common phrases that tokenize consistently. The fallback is too loose (word-level substring matching matches ANY occurrence of any word). The critical flaw is not in identification but in what is identified: these positions are content words, and any content-sensitive metric will favor them (see K2). The position identification is technically adequate but the interpretation is confounded.

### Q2: Is comparing concept-position K/V to non-concept-position K/V a valid test?

**No, as designed.** Two fatal problems:
1. The K/V values computed are not what the cache stores (K1).
2. The comparison is content-words vs function-words, not concept-specific vs non-concept-specific (K2). Even with perfect K/V computation, a positive result is uninterpretable because of the content-density confound.

### Q3: Are there confounds in position (early vs late) that could produce false signal?

**Yes.** Concept regions cluster mid-sequence. Non-concept positions include position 0 (attention sink effects, BOS token) and the final position (where the model is actively generating). Positional encoding gradients, attention patterns, and RoPE angles all vary with position. The prereg names this threat (T2) but the script does not implement the prescribed mitigation.

### Q4: Is the fabricated-entity control well-designed?

**The design is good in principle but the implementation fails to evaluate it.** H4 (fabricated should show no signal) is a meaningful control. But:
- The summary mixes fabricated and factual results without separation (S2).
- Fabricated names ARE semantically rich content words, so the content-density confound (K2) applies equally — the control cannot distinguish "entity recognition" from "content density."
- A truly clean negative control would use a fabricated prompt with NO proper nouns at the concept_region positions — e.g., "The process that occurs when water reaches 100 degrees Celsius is" with concept_region = "water reaches" (a real concept, "boiling," that uses common words rather than named entities).

### Q5: Any issues with computing R^2 per-position with only 20 random draws?

**No.** 20 draws is adequate for mean-baseline estimation. With 10 directions in 4096 dimensions, the random R^2 distribution is extremely tight (coefficient of variation < 10%). The mean from 20 draws is a reliable baseline. This is the least of the experiment's problems.

---

## KNOWN KILLS CHECKLIST

| Kill | Status | Notes |
|------|--------|-------|
| C1: model.generate() rebuilds cache | CLEAR | Forward pass only, no generation |
| C2: Directions on test data | CLEAR | Directions from pre-fitted lens |
| C3: Different code paths | CLEAR | Same code for concept and non-concept |
| L1: Hooks alter behavior | CLEAR | No hooks — uses output_hidden_states |
| L3: No null control | CLEAR | Random baseline serves as null |
| L4: hidden_states off-by-one | **FAIL** | Uses hidden_states[l+1] for K/V but cache stores from hidden_states[l]. Also missing LN1 and RoPE. See K1. |
| S4: No FWL on token count | CLEAR | Within-prompt comparison, no cross-prompt confound |
| M2: No seed setting | CLEAR | Seeds set (lines 31-32) |
| N_eff=1: Greedy decoding | CLEAR | No generation |
| Hook ordering | CLEAR | No hooks |
| **NEW: Content-density confound** | **FAIL** | Concept positions are content words, non-concept are function words. See K2. |
| **NEW: Missing RMSNorm** | **FAIL** | Documented in COALITION_GWT_SYNTHESIS.md, prescribed fix not implemented. |
| **NEW: Missing RoPE for K** | **FAIL** | Same. Especially problematic for a position-specific experiment. |

---

## MINIMUM FIXES REQUIRED BEFORE RUNNING

1. **Use `hidden_states[l]`** (input to layer l) instead of `hidden_states[l+1]` (output).
2. **Apply `input_layernorm`** before W_K and W_V projection.
3. **Either apply RoPE to K** and rotate J-lens directions through the same rotation per-position, **or** document pre-RoPE as a deliberate simplification with stated reasoning.
4. **Address content-density confound**: either match concept positions to non-concept content-words, or add concept-specific directions, or both.
5. **Separate fabricated from factual** in the summary analysis.
6. **Add a statistical test** (permutation test or Mann-Whitney U).
7. **Report absolute R^2** alongside lift.

Items 1-3 are mechanical fixes (30 minutes). Item 4 requires design rethinking (2-4 hours). Items 5-7 are straightforward additions (1 hour).

---

*Project Agni. Review is hostile but the experiment is worth doing once the kills are fixed. The position-specificity question is genuine and important. The prior last-token nulls need explanation. But this script, as written, cannot provide that explanation because it doesn't measure what's actually in the cache.*
