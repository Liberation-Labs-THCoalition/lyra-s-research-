# Frozen Claims — P7 KV-Cloak Defense Paper

**Reviewer request**: Red-team by default
**Date**: 2026-04-27
**Sources**:
- Distilled v2: `beast:/home/thomas/oracle-harness-test/results/kv_cloak_v2/cloak_results_v2.json`
- Base model: `beast:/home/thomas/oracle-harness-test/results/kv_cloak_base/cloak_base_results.json`
- Text baseline: computed in-session from trials_v2.json
- Draft: `p7_kv_cloak_defense.tex`

---

## Claims

| ID | Claim | Precision | Source |
|----|-------|-----------|--------|
| P7-C01 | Spectral gap detects confab at AUROC 0.903 (FWL 0.877, R²=0.068) on distilled model, full cache | 3 | cloak_results_v2.json |
| P7-C02 | Text-only baseline achieves 0.755; cache geometry adds +0.148 over surface features | 3 | Computed from trials_v2.json |
| P7-C03 | Per-head orthogonal rotation completely transforms external feature space (honest cloaked vs honest uncloaked: AUROC 1.000) | 3 | Computed from trials_v2.json |
| P7-C04 | An external adversary's detection model calibrated on uncloaked caches is useless on cloaked caches | 1 | Inferred from P7-C03 |
| P7-C05 | Adversarial steering injection becomes incoherent under obfuscation (unknown basis scrambles vectors) | 0 | Logical argument, NOT experimentally demonstrated |
| P7-C06 | Defensive Oracle inside TEE retains full detection capability (0.903) on pre-obfuscation cache | 1 | Architectural argument from P7-C01 + TEE design |
| P7-C07 | Base model detection is weak (0.661, R²=0.602) — geometric detection requires metacognitive structure | 2 | cloak_base_results.json |
| P7-C08 | KV-Cloak and Oracle Loop are complementary: Oracle detects, Cloak defends against weaponization | 0 | Interpretive framing of P7-C01 through P7-C06 |
| P7-C09 | Within-cloak detection residual (0.786) reflects within-head spectral properties surviving rotation | 1 | Inferred from rotation invariance properties |
| P7-C10 | Per-session key rotation prevents statistical recovery from accumulated observations | 0 | Architectural recommendation, not tested |

Precision: 0 = interpretive/logical, 1 = inferred, 2 = measured, 3 = exact numerical

---

## NEW DATA (Cross-Architecture Sweep, 2026-04-27)

| Model | Confab% | Det(raw) | Det(clk) | Direct Inj | Scrambled | Magnitude Adv |
|-------|---------|----------|----------|-----------|-----------|--------------|
| Qwen3.5-27B-Dist | 7% | 0.903 | 0.786 | — | — | — |
| Qwen2.5-14B | 12% | 0.784 | 0.951* | 0/10 | 0/10 | 1/10 |
| Mistral-7B | 26% | 0.707 | 0.516 | 1/10 | 1/10 | 2/10 |
| Llama-3.1-8B | 14% | 0.590 | 0.559 | 1/10 | 1/10 | 1/10 |
| Qwen2.5-7B | 48% | 0.661 | 0.639 | — | — | — |
| DeepSeek-R1-7B | 2% | None | None | 0/10 | 0/10 | 0/10 |

*Qwen2.5-14B anomalous cloaked detection (0.951) requires investigation.

Key findings from cross-arch:
- Detection exists on Qwen-14B (0.784) and Mistral-7B (0.707) — not just distilled model
- Cloaking drops Mistral to chance (0.516) — defense works
- Magnitude attack: 0-2/10 adverse across ALL models — random perturbation doesn't destabilize
- Injection test underpowered (3 calibration pairs; direct injection barely works)

Real KV-Cloak implementation test running on Beast (S·P̂·(K+A)·M per Luo et al.).

## Self-Red-Team

### Attack (a): n=7 confab trials on distilled model [CRITICAL]

The 0.903 headline comes from 7 confab vs 87 hedged. Bootstrap CI: [0.58, 0.98]. The true AUROC could be anywhere from "barely above chance" to "near-perfect." Every downstream claim that depends on 0.903 inherits this uncertainty.

- **Impact**: P7-C01, P7-C02, P7-C06. The entire "defender retains 0.903" argument rests on a point estimate with a CI that includes 0.65.
- **Mitigation**: Report CIs prominently. The base model (n=48) provides the replication attempt — and it FAILS (0.661). This weakens rather than strengthens the headline. Honest framing: "0.903 on the distilled model (n=7, CI 0.58-0.98); not replicated on the base model (0.661, n=48)."

### Attack (b): Base model non-replication [CRITICAL]

The base model at n=48 shows AUROC 0.661 with R²=0.602 (massive token count confound). This is the properly-powered experiment, and it FAILS. The "detection requires metacognitive structure" interpretation (P7-C07) is one explanation, but alternatives exist:
- Size difference (7B vs 27B)
- Architecture difference (standard vs hybrid)
- The distilled 0.903 is a small-n fluke and the true effect is ~0.65 on all models

The base model result actively undermines the paper's premise that cache geometry detects confabulation.

- **Impact**: P7-C01 (generalizability), P7-C06 (defender capability on arbitrary models), P7-C08 (complementarity assumes detection works).
- **Mitigation**: Acknowledge that the defense argument holds even if detection is model-specific — KV-Cloak blinds external attackers regardless of whether the defender's Oracle works well. The cloak is the primary defense; Oracle is the bonus.

### Attack (c): Steering incoherence not demonstrated [MAJOR]

P7-C05 argues that injected vectors are scrambled by the unknown rotation. This is logically sound for orthogonal rotation, but:
- Not experimentally tested (no failed-injection demo)
- Assumes the attacker has no side-channel for inferring the rotation
- Assumes the attacker can't use rotation-invariant injection strategies (e.g., magnitude-based perturbation that doesn't depend on basis)

A magnitude-based attack (add large noise in any direction) would survive rotation because norms are rotation-invariant. The attacker doesn't need to know the basis to destabilize — they just need to perturb strongly enough.

- **Impact**: P7-C05 is the weakest claim and it's load-bearing for the "defense against weaponized steering" narrative.
- **Mitigation**: (1) Run the failed-injection experiment. (2) Test magnitude-based (direction-agnostic) injection as an adversarial attack. (3) Weaken claim to "targeted emotion-specific steering becomes incoherent; untargeted perturbation remains possible."

### Attack (d): TEE trust assumption [MAJOR]

The defensive architecture requires:
- TEE integrity (SGX/SEV, documented breaches)
- Key material protection
- Oracle parameter protection

If the TEE is compromised, the attacker has the rotation key AND the Oracle parameters. The entire defense collapses.

- **Impact**: P7-C06, P7-C10, P7-C08.
- **Mitigation**: Acknowledge explicitly. State that the security reduction is to TEE integrity, not to a cryptographic hardness assumption. This is standard for TEE-based defenses but should not be hand-waved.

### Attack (e): Simulation vs implementation [MAJOR]

Per-head orthogonal rotation is our approximation. KV-Cloak uses "reversible matrix-based obfuscation combined with operator fusion." Differences:
- Operator fusion may affect which computations the rotation applies to
- The actual matrix may not be orthogonal (could be any invertible matrix)
- The scrambling granularity (per-head vs per-layer vs per-token) may differ

If KV-Cloak's actual transform is weaker than orthogonal rotation (e.g., sparse or structured), the feature-space transformation might be incomplete, leaving partial signal for attackers.

- **Impact**: P7-C03, P7-C04. Our "AUROC 1.000 total transformation" may overstate KV-Cloak's actual obfuscation strength if we used a stronger simulation.
- **Mitigation**: Acknowledge. State that orthogonal rotation is likely a BEST-CASE estimate of obfuscation strength. If the actual KV-Cloak transform is weaker, external detection might partially survive — which makes the defense LESS effective than we claim.

### Attack (f): "Metacognitive structure" interpretation is unfalsifiable [MAJOR]

P7-C07 claims the distilled/base difference reflects "metacognitive structure installed by training." This is an interpretation, not a finding. The models differ in three ways (size, architecture, training). We cannot isolate which factor drives the detection difference. "Metacognitive structure" is a narrative that fits the data but isn't tested.

- **Impact**: P7-C07 framing. If the detection difference is SIZE-driven (27B vs 7B), the story changes from "RLHF installs detectable calibration" to "bigger models have more geometric diversity." The defense argument (P7-C08) doesn't depend on this interpretation, but the paper's intellectual contribution does.
- **Mitigation**: Report the three confounds. Weaken to "consistent with metacognitive structure, but confounded by size and architecture differences."

### Attack (g): Confabulation only — deception not tested [MAJOR]

The paper's introduction mentions deception, sycophancy, and persona identification as threats. But all experiments test only confabulation detection. Deception detection (AUROC 1.000 in prior work) is the most security-relevant capability — and it's never tested under obfuscation.

If KV-Cloak leaves deception detection intact externally (because the deception signal is stronger), the defense against the most dangerous adversarial use case might not work.

- **Impact**: P7-C04, P7-C08. The "KV-Cloak blinds adversaries" claim is only demonstrated for confab, which is the LEAST adversarially relevant cognitive state.
- **Mitigation**: (1) Test deception detection under obfuscation. (2) Or scope the claim: "demonstrated for confabulation detection; deception and other cognitive states are future work."

### Attack (h): The "complementary" framing assumes both are deployed together [MINOR]

P7-C08 frames KV-Cloak and Oracle as complementary. But the stronger finding is that KV-Cloak alone defends against external geometric attacks — no Oracle needed. The "complementary" framing might confuse readers into thinking you need both. In practice, KV-Cloak alone is the defense; Oracle is an independent capability that happens to run in the same TEE.

- **Mitigation**: Lead with "KV-Cloak alone defends against weaponized geometric methods." Add Oracle as an optional enhancement.

---

## Pre-Red-Team Verdict

**ADEQUATE for the defense claim (P7-C03, P7-C04).** The feature-space transformation is experimentally demonstrated and the defense-by-blindness argument follows logically.

**GAPS IDENTIFIED for the detection claim (P7-C01).** n=7 with non-replication on the base model. The 0.903 is suggestive but not confirmed.

**INSUFFICIENT for the steering-incoherence claim (P7-C05).** Logical argument without experimental support. Magnitude-based attacks not considered.

**What the paper CAN claim:**
- KV-Cloak completely transforms the external cache-geometry feature space (experimentally demonstrated)
- An external adversary's detection tools are rendered useless by this transformation
- A defender inside the TEE retains access to uncloaked geometry
- KV-Cloak is the recommended mitigation for dual-use risk of cache-geometry research

**What it CANNOT yet claim:**
- That adversarial steering is incoherent (not tested)
- That detection is strong on arbitrary models (base model fails)
- That the simulation matches KV-Cloak's actual implementation
- That deception/sycophancy detection is also blinded (confab only)

**Minimum viable fixes before publication:**
1. Run failed-injection experiment (steering through cloak)
2. Test magnitude-based adversarial attack
3. Report CIs prominently for n=7 result
4. Scope deception/sycophancy to future work explicitly
5. Weaken "metacognitive structure" to acknowledged confound
