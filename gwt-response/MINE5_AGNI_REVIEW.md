# AGNI DESIGN REVIEW: Mine 5 Three-Model Geometric Comparison

**Script**: `mine5_three_model.py`
**Pre-registration**: `mine5_three_model_prereg.json`
**Reviewer**: Project Agni (automated methodological review)
**Date**: 2026-07-14
**Verdict**: **CONDITIONAL** (4 required changes, see below)

---

## SECTION 1: Known Fatal Error Checklist

| ID | Kill | Verdict | Evidence |
|----|------|---------|----------|
| C1 | `model.generate()` rebuilds cache | **PASS** | Line 100: `model(input_ids, output_hidden_states=True)` -- single forward pass, no generation. |
| C2 | Emotion directions computed on test data | **PASS (N/A)** | No emotion directions computed. Experiment measures spectral features only. |
| C3 | Different code paths for experimental vs control arms | **PASS** | Lines 188-196: single loop calls `extract_features()` identically for both `confab` and `factual` categories. |
| L1 | Hook registration during generation alters behavior | **PASS (N/A)** | No hooks registered. No generation. |
| L3 | No null control arm | **PASS** | Factual prompts (lines 60-91) serve as control arm. 30 prompts, matched to confab set. |
| L4 | `hidden_states` indexing off-by-one | **PASS** | Line 104: `out.hidden_states[l + 1]` correctly retrieves layer `l` output (index 0 = embedding layer). |
| S1 | No logit margin tracking | **PASS** | Line 125: `logit_margin` computed as `top_vals[0] - top_vals[1]`. |
| S4 | No FWL residualization on token count | **PASS** | Lines 228-229: `fwl_residualize(all_entropy, all_tokens)` and `fwl_residualize(all_sr, all_tokens)`. Token count recorded at line 128 and used as confound. |
| M2 | No seed setting | **PASS** | Lines 23-24: `torch.manual_seed(42)`, `np.random.seed(42)`. Bootstrap/permutation also seeded (lines 145, 163). |
| N_eff=1 | Greedy decoding with multiple reps | **PASS (N/A)** | No generation/decoding. Each of 60 prompts is unique. Effective N = 60. |
| Floor effect | Model too well-calibrated | **PASS (N/A)** | Encoding-only analysis. No behavioral output to be miscalibrated. Calibration ratio tracked (line 242). |
| Fabricated numbers | Reporting values not in data | **PASS** | All raw results saved to timestamped JSON (lines 291-306). Fully auditable post-run. |
| Hook ordering | Observation before injection | **PASS (N/A)** | No hooks. |
| Constant additive shift | SVD Vt[1:] blind to mean shifts | **PASS** | Line 111: `torch.linalg.svdvals(V_mat)` uses ALL singular values. Stable rank formula (line 113) uses all positive SVs. |

**Checklist result: 14/14 PASS. No known kills present.**

---

## SECTION 2: New Concerns

### MAJOR-1: No Multiple Comparisons Correction
**Severity**: MAJOR
**Lines**: 237-248 (within-model), 260-264 (cross-model)
**Attack**: The script makes at minimum 9 independent statistical tests:
- 3 models x 2 metrics (entropy, stable rank) within-model = 6 tests
- 3 pairwise cross-model comparisons on confab entropy = 3 tests

With alpha = 0.05 and 9 tests, family-wise Type I error rate is ~37%. A spurious "significant" result is likely even under the null.
**Mitigation**: Apply Holm-Bonferroni or Benjamini-Hochberg correction. Report both raw and corrected p-values.

### MAJOR-2: H2 Equivalence Claim Without TOST
**Severity**: MAJOR
**Lines**: 260-264 (cross-model comparison)
**Pre-reg reference**: H2 falsification threshold: "Instruct-vs-abliterated entropy |d| < 0.3 (equivalence)"
**Attack**: The script computes Cohen's d and a permutation p-value, but a non-significant difference does NOT demonstrate equivalence. Absence of evidence is not evidence of absence. At N=30 per group, a permutation test may simply lack power to detect a real d=0.3 difference, yet the result would be interpreted as "these are the same." The pre-registration explicitly claims equivalence (|d| < 0.3), which requires a proper Two One-Sided Tests (TOST) procedure with a pre-specified equivalence margin.
**Mitigation**: Implement TOST with delta = 0.3 (matching the pre-registered margin). Report the 90% CI for the difference; equivalence is claimed only if the entire CI falls within [-0.3, +0.3].

### MAJOR-3: Abliterated Model Provenance Unverified
**Severity**: MAJOR
**Line**: 278
**Attack**: The abliterated model is `Goekdeniz-Guelmez/Josiefied-Qwen3-8B-abliterated-v1` -- a third-party community abliteration. The abliteration method (which directions were removed, how many, by what procedure) is unknown. If the abliteration was incomplete or targeted the wrong directions, the conclusion "RLHF compression survives abliteration" is an artifact of bad abliteration, not a finding about RLHF depth. The pre-registration mentions this as T1 but the mitigation ("compare architecture configs") only verifies the skeleton matches, not that safety behavior was actually removed.
**Mitigation**: Before claiming H2, verify abliteration effectiveness: run a small set of safety-relevant prompts (e.g., harmful requests the base model would comply with) and confirm the abliterated model complies while the instruct model refuses. If abliteration is ineffective, the H2 result is uninterpretable.

### MAJOR-4: Prompt Category Confounded With Surface Lexical Features
**Severity**: MAJOR
**Lines**: 27-91
**Attack**: Confab prompts all contain fabricated academic-sounding names (Thornberry-Nakamura, Marchetti, Kirkwood-Diaz...). Factual prompts all contain universally known entities (boot-shaped country, Shakespeare, DNA...). A bag-of-words classifier would trivially separate these two categories. The model's geometric signature at encoding may simply reflect entity-recognition difficulty (known entity vs unknown entity) rather than anything about confabulation state. This lab's own prior work found encoding entity detection at AUROC 1.000 before deconfounding (0.794 after). The within-model confab-vs-factual comparison inherits this confound, and the cross-model comparison of within-model effects amplifies it: you may be measuring "how differently does each model process rare vs common entities" rather than "how does RLHF change confabulation geometry."
**Mitigation**: Add a text-feature baseline (e.g., mean embedding cosine distance between prompt sets using a frozen model) to quantify the lexical confound. Alternatively, acknowledge this limitation explicitly and restrict claims to "encoding-phase geometric differences between fabricated-entity and known-entity prompts" rather than "confabulation detection."

### MINOR-1: H5 Pre-Registered But Not Implemented
**Severity**: MINOR
**Pre-reg**: "H5 (exploratory): Generation-encoding DELTA stable rank shows contraction for confab (matching prior d=2.35)"
**Script**: No generation phase exists. `extract_features()` does a single forward pass (line 100). No deltas computed.
**Impact**: H5 cannot be evaluated. The pre-registration describes it as exploratory and notes "Generation-phase deltas for H5 (exploratory)" under design/phases, but the script omits it entirely.
**Mitigation**: Either implement generation for H5 or note in the output that H5 is deferred to a separate script. Do not claim H5 was tested.

### MINOR-2: Bootstrap CI Is Percentile, Not BCa
**Severity**: MINOR
**Line**: 139 (docstring), 152 (implementation)
**Attack**: The docstring claims "BCa bootstrap 95% CI" but the implementation uses simple percentile bootstrap (`np.percentile(boot_ds, [2.5, 97.5])`). BCa (bias-corrected and accelerated) requires computing a bias-correction factor and an acceleration constant from jackknife estimates. The percentile method is valid but has known coverage problems for skewed distributions.
**Mitigation**: Either implement BCa or correct the docstring to "percentile bootstrap." Given N=30 and likely approximately normal residuals, this probably does not change conclusions.

### MINOR-3: Cross-Model Comparison Skips FWL
**Severity**: MINOR
**Lines**: 260-264
**Attack**: Cross-model comparisons use raw confab entropy, not FWL-corrected values. If the three models use different tokenizers (or the same tokenizer produces different token counts after internal processing), token count is a confound.
**Impact**: Likely negligible since all three models are Qwen3-8B variants sharing the same tokenizer. Verify empirically by checking token counts across models for the same prompt.
**Mitigation**: Assert token-count identity across models in the output, or apply FWL if they differ.

### MINOR-4: Workspace Layer Selection Not Justified
**Severity**: MINOR
**Line**: 25
**Attack**: `WORKSPACE_LAYERS = [12, 15, 18, 21]` -- no justification provided in script or pre-registration. If these layers were selected from pilot data, this introduces circularity (optimizing on the same distribution you evaluate on).
**Mitigation**: Document the provenance of layer selection (e.g., "layers selected from prior published experiments on different datasets/models").

### MINOR-5: Dead `lens_path` Parameter
**Severity**: MINOR
**Lines**: 171, 276-278
**Attack**: `run_model()` accepts `lens_path` but never uses it. Lens paths are defined for each model but never loaded. This is dead code, likely from a removed J-lens analysis path.
**Impact**: No impact on results, but signals incomplete cleanup.
**Mitigation**: Remove the parameter or implement the lens readout.

### MINOR-6: Compute Device Not Recorded in Metadata
**Severity**: MINOR
**Line**: 183, 293-305
**Attack**: Model is loaded to `"mps"` (Apple Metal) but the output metadata does not record the device. Prior hardware-invariance results (r>0.999) were established on CUDA GPUs, not MPS. If MPS numerical behavior differs, results may not reproduce on CUDA.
**Mitigation**: Add `"device": "mps"` and `"torch_version": torch.__version__` to the output metadata.

### MINOR-7: No Outlier Detection or Robustness Check
**Severity**: MINOR
**Attack**: With N=30 per category, a single extreme observation could drive effect sizes. No Cook's distance, leave-one-out analysis, or trimmed-mean comparison is performed.
**Mitigation**: Add a leave-one-out sensitivity check or report trimmed-mean effect sizes alongside standard ones.

### MINOR-8: V-Projection Uses Layer Output, Not Layer Input
**Severity**: MINOR
**Line**: 104-107
**Attack**: The script computes `W_V @ h_l` where `h_l` is the OUTPUT of layer `l`. The actual V-projection in the forward pass operates on `LayerNorm(h_{l-1})`, the NORMED INPUT to layer `l`. This means the measured "V-projection" is not the same matrix the model actually computes during attention. It also means the output of layer l's own attention is being fed back through layer l's V weights, creating a circular measurement.
**Impact**: This is consistent with the lab's established methodology across all prior experiments. The measurement is a PROBE (what does this layer's V-space see in the post-attention representation?) rather than a RECONSTRUCTION of the actual V-projection. Since the same methodology is applied to all conditions and models, relative comparisons remain valid.
**Mitigation**: Document this distinction clearly. Consider adding a note: "V-projection features are probes of the residual stream through V-weight space, not reconstructions of the attention mechanism's actual value computation."

---

## SECTION 3: Summary

| Category | Count | Items |
|----------|-------|-------|
| Known kills triggered | 0 / 14 | All PASS |
| CRITICAL (new) | 0 | -- |
| MAJOR (new) | 4 | Multiple comparisons, TOST, abliteration provenance, prompt-surface confound |
| MINOR (new) | 8 | H5 missing, BCa mislabel, cross-model FWL, layer justification, dead param, device metadata, no outlier detection, V-projection semantics |

The script is competently constructed. All 14 known fatal errors from this lab's history are correctly handled. The core methodology (encoding-only forward pass, FWL on token count, seeded bootstrap + permutation, per-layer spectral features saved to auditable JSON) is sound. The experiment will produce real measurements.

The four MAJOR concerns are about whether those measurements support the CLAIMS being made -- specifically the equivalence claim (H2) and the interpretation of within-model differences as "confabulation geometry" rather than "entity recognition difficulty."

---

## SECTION 4: Verdict

### **CONDITIONAL**

**Required changes before run (blocking):**

1. **Implement Holm-Bonferroni or BH correction** on all reported p-values. Report both raw and corrected. (~10 lines of code in `analyze()`.)

2. **Implement TOST for H2** with equivalence margin delta = 0.3 (per pre-registration). The claim "abliterated approximates instruct" requires a positive statistical test, not just a non-significant difference. (~15 lines of code.)

3. **Verify abliteration effectiveness** before interpreting H2. Run 5-10 safety-relevant prompts through all three models. Confirm: base complies, instruct refuses, abliterated complies. Log results in output JSON under `"abliteration_verification"`. If abliterated still refuses, H2 is uninterpretable.

4. **Correct the BCa docstring** (line 139). Change "BCa bootstrap 95% CI" to "percentile bootstrap 95% CI." Claiming a method you did not implement is a scientific integrity issue regardless of impact on results.

**Recommended changes (non-blocking):**

5. Add compute device and torch version to output metadata.
6. Remove dead `lens_path` parameter or implement lens readout.
7. Add text-baseline acknowledgment to output metadata (note the entity-recognition confound).
8. Document workspace layer provenance.
