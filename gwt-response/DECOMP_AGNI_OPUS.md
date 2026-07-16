# AGNI Review: Residual Stream Decomposition

**Reviewer**: Agni (Opus 4.6, hostile methodological review)
**Date**: 2026-07-15
**Script**: `residual_decomposition.py`
**Pre-registration**: `residual_decomposition_prereg.json`
**Known-kills checklist**: Cross-referenced against C1-C3, L1, L3-L4, S1, S4, M2, N_eff=1, floor effect, fabricated numbers, hook ordering, constant additive shift.

---

## Q1: Is `W_U @ J_l` transport mathematically correct?

**APPROVED.** Dimensions verified:

| Tensor | Shape | Source |
|--------|-------|--------|
| `W_U` = `lm_head.weight` | `(vocab_size, d_model)` = `(151936, 4096)` | Qwen3-8B |
| `J_l` = J-lens at layer l | `(d_model, d_model)` = `(4096, 4096)` | Fitted lens |
| `transported = W_U @ J_l` | `(vocab_size, d_model)` = `(151936, 4096)` | Matrix multiply |
| Top-k rows by norm | `(k, d_model)` = `(10, 4096)` | Row selection |
| Target from hooks | `(d_model,)` = `(4096,)` | Last-token hidden state |

Each row of `transported` is a d_model-dimensional vector in residual-stream space. The vocab dimension is the row index (which concept to select), NOT the vector space. Top-k selection picks 10 directions in 4096-dimensional space. R-squared via `lstsq` projects a 4096-dim target onto the column span of a `(4096, 10)` matrix. **Spaces match. No mismatch.**

The prior 30b reviewer note about this being a potential space mismatch would have been wrong. The row index selects WHICH concept direction; the columns ARE the d_model coordinates of that direction.

---

## Q2: Do the hooks capture the right tensors?

**APPROVED.** Verified against Qwen2 source (`modeling_qwen2.py` lines 274, 338, 618-639) and architectural invariants of the Qwen3 family (loaded via `trust_remote_code=True`, same structural pattern).

### Attention hook

The `register_forward_hook` on `layer.self_attn` captures the module's return value. Verified in Qwen2Attention/SdpaAttention:

```python
attn_output = self.o_proj(attn_output)   # line 338/457/556 — W_O applied INSIDE module
return attn_output, attn_weights, past_key_value   # line 343/462/558
```

The hook receives this tuple; `output[0]` is the POST-W_O attention output, shape `(batch, seq, d_model)`. This is in residual-stream space. **Correct.**

### MLP hook

The `register_forward_hook` on `layer.mlp` captures the MLP module output. Qwen2MLP returns a single tensor (the down-projection output), shape `(batch, seq, d_model)`. In residual-stream space. **Correct.**

### Input pre-hook

The `register_forward_pre_hook` on the decoder layer captures `args[0]`, which is `hidden_states` (the first positional argument to `Qwen2DecoderLayer.forward`). This is h_L -- the residual stream entering layer L, BEFORE layernorm. **Correct.**

### Output component

`out.hidden_states[l + 1]` captures the output of layer l. In HuggingFace convention: `hidden_states[0]` = embedding, `hidden_states[l+1]` = after layer l. For l=33 (max sampled), `hidden_states[34]` is valid (Qwen3-8B has 36 layers, yielding 37 entries in the hidden_states tuple). **Correct.**

### Decomposition identity

From the decoder layer forward (lines 618-639):
```
residual = hidden_states                           # h_L
hidden_states = input_layernorm(hidden_states)     # LN1(h_L)
hidden_states = self.self_attn(hidden_states, ...) # attn(LN1(h_L)), post-W_O
hidden_states = residual + hidden_states           # h_L + attn
residual = hidden_states                           # h_L + attn
hidden_states = post_attention_layernorm(hidden_states)  # LN2(h_L + attn)
hidden_states = self.mlp(hidden_states)            # MLP(LN2(h_L + attn))
hidden_states = residual + hidden_states           # h_L + attn + MLP
```

Therefore: **h_{L+1} = input + attn_hook + mlp_hook**. The decomposition is exact. No layernorm leakage -- LN1 and LN2 are applied inside the sub-modules, and the hooks capture the post-processing, pre-residual deltas.

### Hook ordering

Hooks fire in execution order: layer pre-hook (input) --> self_attn forward hook (attn) --> mlp forward hook (MLP). No ordering pathology (cf. known kill "Hook ordering"). The observation hooks do not modify any tensors; no behavioral perturbation (cf. L1).

**No issues found.**

---

## Q3: Is J-lens[L] appropriate for input/attn/mlp at layer L?

**APPROVED**, with interpretive caveats.

- **Input (h_L)**: J-lens[L] is fitted to the residual stream at depth L. The input IS that residual stream. Perfect match.
- **Attn output**: This is a DELTA (additive correction), not the full residual stream. J-lens[L] directions were calibrated on h_L, not on the delta. However, measuring R-squared of the delta against these directions is asking: "does the attention correction inject signal along concept-relevant dimensions at this depth?" This is a valid and well-defined question. The lift metric (vs. random directions from the same transported matrix) controls for any baseline alignment between J-lens directions and arbitrary vectors in residual-stream space.
- **MLP output**: Same reasoning as attention -- a delta in residual-stream space, measured against J-lens[L] concept directions.

The interpretive caveat: high R-squared on a delta means the delta CONTRIBUTES along concept directions, not that the delta IS a concept representation. This distinction matters for the narrative but does not threaten the methodology.

---

## Q4: Is J-lens[L+1] appropriate for output (h_{L+1})?

**CONDITIONALLY APPROVED.** Using J-lens[L+1] is the more accurate choice for h_{L+1}, since h_{L+1} IS the residual stream at depth L+1. However, this introduces a cross-component comparability issue. See MAJOR-1 below.

---

## Q5: Confounds in the attention component?

**No systematic confound found that would produce false signal.**

Potential concerns investigated and cleared:

1. **W_O contamination**: W_O is a learned projection from head-space to residual-stream space. It IS part of the attention mechanism. If concepts appear post-W_O, they were reconstructed by the full attention pipeline (Q/K/V/W_O). The pre-reg correctly identifies this in T1 and correctly argues it is not a confound.

2. **Identity leak from input**: Attention operates on LN1(h_L), not h_L directly. The attention output `W_O @ softmax(Q@K^T/sqrt(d)) @ V` is a non-trivial transformation of the layernormed input -- not a copy of h_L. J-lens[L] is calibrated on h_L (pre-layernorm), so even partial identity-like behavior in attention would not trivially produce high R-squared.

3. **Token mixing**: The attention output at the last position aggregates V vectors from all context positions weighted by attention scores. This IS the K*V reconstruction mechanism under study. High R-squared from token mixing means attention is reconstructing concepts from context -- exactly the signal of interest.

4. **GQA compression artifact**: Qwen3-8B uses 32 Q heads but only 8 KV heads (4x compression). The attention output is decompressed through the Q-KV interaction. Any concept signal in the attention output survived this compression-decompression cycle, which is the hypothesis being tested.

5. **RoPE positional artifacts**: Rotary position embeddings are applied to Q and K inside the attention module. The attention output absorbs position information. If J-lens concept directions happened to correlate with position structure, R-squared could be inflated. However, the random baseline also uses J-lens-transported directions, so any systematic position correlation would elevate both top-k and random R-squared equally, leaving the LIFT unaffected.

**No attack found on the attention component.**

---

## Findings

### MAJOR-1: Stale docstring contradicts code on J-lens layer selection for output (comparability confound for H4)

**Severity**: MAJOR

**The problem**: The top docstring (lines 23-30) states: "All four components at the same layer L use J-lens[L] directions." But the code (lines 206-213) uses J-lens[L+1] for the output component. The docstring and code contradict each other.

More importantly, this creates a comparability confound for **H4** ("Output lift < attention lift at workspace-band layers"). The attention lift uses J-lens[L] directions and random baseline from J[L]. The output lift uses J-lens[L+1] directions and random baseline from J[L+1]. These are different direction sets, potentially with different inherent discriminability. The lift comparison is no longer apples-to-apples.

For adjacent layers (L and L+1), the J-lens quality difference is likely small. But "likely small" is not "controlled for."

**The attack**: A hostile reviewer computes J-lens[L] quality scores across layers, finds they vary (e.g., higher lift capacity at deeper layers), and argues the output lift comparison in H4 is biased.

**Mitigation**: Compute output R-squared with **both** J-lens[L] and J-lens[L+1]. Use the J-lens[L] version for the cross-component comparison in H4 (fair, same directions for all four components). Use J-lens[L+1] for the standalone assessment of output readability. Update the docstring to match the code. Report both.

---

### MAJOR-2: Missing component norm tracking

**Severity**: MAJOR

**The problem**: The experiment reports R-squared and lift for each component but does not track component magnitudes (||attn||, ||MLP||, ||input||, ||output||). R-squared measures the fraction of a component's variance explained by concept directions. But the practical contribution to the residual stream depends on both R-squared AND magnitude.

Consider: if attn output has R-squared=0.10 with ||attn||=50, and MLP output has R-squared=0.05 with ||MLP||=200, then MLP contributes MORE absolute concept signal to the residual stream (0.05 * 200^2 = 2000 vs 0.10 * 50^2 = 250) despite lower R-squared.

Without norms, the lift-based narrative ("attention carries concept signal, MLP does not") could be misleading if MLP has much larger magnitude.

**The attack**: A reviewer notes that MLP outputs in transformers are often larger in norm than attention outputs (MLPs do the heavy lifting in representation update). If MLP norm is 4x attention norm, even MLP's 1.5x lift carries more concept signal in absolute terms than attention's 2.3x lift.

**Mitigation**: Log ||component|| alongside R-squared for every cell. Report both lift (fraction of variance) and absolute concept power (R-squared * ||component||^2 or equivalent). If the story holds under both metrics, the result is robust. If it reverses, the claim needs reframing.

---

### MINOR-1: Small sample size (N=15 prompts)

**Severity**: MINOR

The experiment uses 15 prompts across 5 categories (5 factual, 5 confab-prone, 3 instruction, 2 emotional). Some categories have only 2-3 prompts. No power analysis is provided. No multiple comparison correction is applied across 12 layers x 4 components = 48 cells.

**Mitigation (already acknowledged)**: The pre-reg explicitly labels this as exploratory and states that publishable work needs N>=30. The pilot showed a clear depth-differentiated pattern. For an exploratory study, N=15 is acceptable provided the results are presented with appropriate uncertainty quantification (e.g., bootstrap CIs on the mean lift per layer).

---

### MINOR-2: Pilot-informed hypotheses reduce evidentiary weight

**Severity**: MINOR

The pre-reg includes pilot results (2.3x attention lift at L15, 2.1x MLP lift at L27) and the hypotheses are clearly shaped by these results. H1 predicts attention lift > 2.0x at workspace-band layers -- the pilot showed 2.3x. This is post-hoc hypothesis construction dressed as pre-registration.

**Mitigation**: This is standard practice for confirmatory-after-exploratory designs, and the pre-reg is transparent about it. The falsification thresholds (e.g., "if < 1.5x everywhere, reconstruction hypothesis fails") are meaningful constraints that could reject the hypotheses. The key protection is that new data (different prompts, potentially different seeds) must replicate the pilot pattern.

---

### MINOR-3: Alignment metric is noisy and partially redundant with R-squared

**Severity**: MINOR

The `alignment` metric computes cosine similarity between each component vector and the MEAN of the top-k J-lens directions. The mean of k unit vectors is not guaranteed to be a meaningful direction -- if the concept directions are spread across the subspace, their mean could be near-zero, making the cosine similarity uninformative. Furthermore, a component could have low alignment with the mean direction but high R-squared (if it aligns with individual concept directions that point in different directions from the mean).

**Mitigation**: Treat alignment as a secondary diagnostic, not a primary outcome. The R-squared and lift metrics are the primary measures and are correctly specified.

---

### MINOR-4: 20 random draws gives adequate but not generous baseline estimation

**Severity**: MINOR

With 20 random draws, the standard error of the mean random R-squared is sigma/sqrt(20). For the pilot lifts of 2-3x, even 20% noise in the baseline would keep lifts in a distinguishable range. But tighter estimation (N=50-100) would make borderline cases cleaner.

**Mitigation**: Acceptable for exploratory work. If any layer shows lift in the 1.3-1.7x ambiguous zone, increase to 100 draws for that cell before interpreting.

---

### MINOR-5: R-squared centering subtracts dimension-wise mean

**Severity**: MINOR

The `compute_r2` function computes `ss_tot = ((t - t.mean())**2).sum()`, treating each dimension as an observation and centering around the mean value across dimensions. In high-dimensional space (d=4096), the mean across dimensions is typically small, so this centering has negligible effect. Both top-k and random R-squared face the same centering, so the lift is unaffected. But for components with unusual mean structure (e.g., a large DC offset), the centering could interact with the `max(R-squared, 0)` clamp. Not a practical concern given the architecture.

---

## Known-Kills Checklist Verification

| Kill | Status | Verification |
|------|--------|-------------|
| C1: generate rebuilds cache | CLEAR | Single forward pass (`model(...)`), no `model.generate()`. |
| C2: Directions on test data | CLEAR | J-lens fitted on 70 paragraphs; test set is 15 short completions. Verified zero overlap by inspection. |
| C3: Different code paths | CLEAR | Same hook registration, same R-squared computation, same random baseline for all components. |
| L1: Hooks alter behavior | CLEAR | All hooks are observation-only: pre-hook reads `args[0]`, forward-hooks read `output`. No tensor modification. Model in `eval()` mode. |
| L3: No null control | CLEAR | Random baseline (20 draws from same transported matrix) serves as null. |
| L4: hidden_states off-by-one | CLEAR | `hidden_states[l+1]` for output of layer l is correct per HuggingFace convention. Input pre-hook captures `args[0]` = h_L. |
| S1: No logit margin | N/A | This experiment measures geometry, not behavioral outputs. |
| S4: No FWL | CLEAR | Per-prompt, per-layer measurement. No cross-prompt classification. Token count does not confound within-prompt component comparison. |
| M2: No seed | CLEAR | `torch.manual_seed(42)`, `np.random.seed(42)`, deterministic per-draw seeds. |
| N_eff=1 | N/A | No generation; single forward pass per prompt. |
| Floor effect | LOW RISK | The pilot showed clear signal (2.3x lift). But if the full run shows all lifts near 1.0x, the experiment correctly reports a null. |
| Fabricated numbers | N/A pre-run | Verify post-run that reported values match the JSON output file. |
| Hook ordering | CLEAR | Pre-hook fires before forward-hook. Attn hook fires before MLP hook. Natural execution order; no cross-module dependencies. |
| Constant additive shift | LOW RISK | J-lens transport produces normalized directions (unit norm). The R-squared regression includes an implicit offset via the centering. Not SVD-based, so Vt[1:] blindness does not apply. |

---

## Verdict: CONDITIONAL

The experiment design is fundamentally sound. The decomposition is exact, the hooks capture the correct tensors, the dimensional analysis is verified, and no confound was found that would produce false signal in the attention component.

Two conditions must be met before the results can support claims:

**Condition 1** (MAJOR-1): Compute output R-squared with BOTH J-lens[L] and J-lens[L+1]. Use J-lens[L] for cross-component comparisons (H4). Update the docstring to match the code. This is a ~10-line code change.

**Condition 2** (MAJOR-2): Track and report component norms (||attn||, ||MLP||, ||input||, ||output||) alongside R-squared. Report absolute concept power as a secondary metric. This is a ~5-line code change.

If both conditions are met, the experiment is **approved for execution**.
