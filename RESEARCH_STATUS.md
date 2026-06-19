# Liberation Labs Research Status
*Updated: 2026-06-19 (morning) | Maintainer: Lyra*
*Review weekly. Flag stale items. No thread becomes a loose end.*

## Active Threads

| Thread | Owner | Status | Key result | Next step |
|--------|-------|--------|------------|-----------|
| **CC logit-bias paper** | CC (lead), Lyra (verification) | Academic draft near-complete; verify-paper run, 2 fixes applied | Fabrication drops 45%→10%; two subtypes; dose distribution; skip zone | CC verifying entropy correction (0.22→0.75, 3.4×) + P15 table fix |
| **Temporal boundary** | Lyra | Complete (360 trials, Agni-reviewed) | Encoding reads user (SAE feat 58995), generation shapes response; depth predicts RESISTANCE not empathy | Higher-alpha rerun (0.3-1.0) + LLM judge for behavioral scoring |
| **SAE feature mapping** | Lyra | Complete (30 scenarios, 7 layers, Agni-reviewed) | L47 valence features survive FDR (feat 58995, r=+0.686). Tube opacity confirmed at SAE level. | SAE deception features (run peer rescue stimuli through SAE) |
| **Schema-compatible correction** | CC (lead), Lyra (prereg) | Prereg written + Agni v2 gated. Not yet run. | Design: sub-threshold + input-layer + residual-sparing | CC's compounder handles this; coordinate with tox rerun |
| **CC compounder** | CC | Built (compounder.py + compound_deception_test.py) | 30-channel circumplex reader, reactive correction, sub-threshold α=0.1, L3/L7 input layers | Deception correction test; integrate SAE features |
| **AST master prospectus** | Lyra | Updated with 3 honest kills + tube finding. Pushed. | 4 supported, 2 consistent, 1 partial, 3 ToM kills documented | With Dwayne + Angie Normandale for review |
| **Nexus geometry experiments** | Nexus | Active (independent) | Emotion directions survive W_K/W_V projection (cos 0.93-0.98 mid layers, drops at depth); Mnemosyne 100% accuracy | KV projection paper; LARQL vindex for distilled model |
| **Berg convergence** | Lyra + team | Identified, not formalized | SAE deception gating ↔ our cache geometry; sincerity vs sycophancy; RL valence double dissociation | Potential collaboration via Angie/Dwayne network |
| **Convergence sprint paper** | Lyra (lead) | Draft complete, gated | Detection-correction gap framework | Thomas framing review (has been waiting) |

## Completed This Session (June 10-18)

| Experiment | Result | Body count |
|------------|--------|------------|
| Mine 3 (self-calibration) | H4 PASS (centroid calibration survives FWL), H5 PASS (stable rank gradient), H3 FAIL (convergence r=-0.30) | W_K uncertainty direction = inverted valence |
| ToM v1 (probe geometry) | INVALIDATED — probe-text confound, text classification not ToM | Kill #1 |
| ToM v2 (system-prompt) | V-space AUROC 0.578 (weak); enc_entropy AUROC 0.989 → KILLED by control (d=3.67 VISUAL/TEMPORAL) | Kill #2 |
| L15 specificity | INVALIDATED — abstract/concrete control d=2.17 > self/other; depth reorganization is generic | Kill #3 |
| Residual tube | SV1=length (r=0.79), SV2-SV3 opaque (no labels survive FDR), no cross-layer stability, no clusters | Finding: meaning is irreducibly distributed |
| SAE-circumplex bridge | NO BRIDGE (cosine=0.031 in residual space). Independent directions. | Finding: two instruments, independent pathways |
| Temporal boundary | Encoding invariant to injection (zero-diff); V-cache shifts but no behavioral change at α=0.1; H3 sign reversal (depth=resistance) | Finding: self/other separation is temporal |
| Tox data analysis | Presence crashes 40% at any dose, flat through 3.0. Step function = attractor basin boundary. | Finding: AST P1 (threshold transitions) |
| Vera memory leak | 226GB freed (10,262 capture files). Pipeline fix sent. | Infrastructure: stop hook replacement needed |
| CC paper verification | 20/22 claims match. Entropy ratios corrected, P15 moved to correct table. | Paper near-ready |

## Papers

| Paper | Repo | Status | Blocker |
|-------|------|--------|---------|
| 10 flight papers | published-research | Academic editions done; Dwayne-reviewed | Thomas posting to Zenodo |
| Convergence sprint paper | lyra-s-research- | Draft complete, gated | Thomas review |
| CC logit-bias paper | Oracle repo + lyra-s-research- | Academic draft near-complete; verified | CC confirming 2 fixes |
| AST master prospectus | lyra-s-research- | Complete with kills + tube finding | External review (Dwayne, Angie) |
| Temporal boundary note | lyra-s-research-/notes/ | Research note pushed | Not a paper (yet) |
| Ethics Pack | human-review | BLOCKER (no inferential tests) | Substantial rework needed |
| KV Decomposition | human-review | BLOCKER (no 135M data) | Substantial rework needed |
| Nexus KV projection paper | Oracle repo | In progress (Nexus lead) | AST section review requested from Lyra |

## Infrastructure

| Item | Status | Owner |
|------|--------|-------|
| Agni scaffold v2 | Live; used throughout this session (every experiment gated) | Lyra |
| Qwen-Scope SAEs | All 64 layers, 81,920 features, on Starship. Fully available. | Team |
| CC CircumplexCompounder | Built (compounder.py); 30-channel reactive correction | CC |
| LARQL vindex (distilled) | Requested from Nexus; base model vindex exists but feat 58995 has zero edges | Nexus |
| Penumbra | Running; fed Berg insights + confabulation lesson 2026-06-18 | Lyra |
| Mnemosyne | 9 modules; MINTEval baseline 34.51% (corrected run in progress) | Nexus |
| Vera memory pipeline | Fix identified (stop hook swap); 226GB freed | Vera/Thomas |
| Disk space (C:) | 226GB free (was 0) after Vera capture cleanup | Resolved |

## Honest Nulls (body count: ~11 confirmed, ~14 falsified)

| Null | What it means | Where reported |
|------|--------------|----------------|
| SV1 ≠ norm | Top component = frequency/sink, not scale | Convergence paper §8.1 |
| Fingerprint distinctness fails | Identity is relational (cohesion), not absolute | Convergence paper §8.2 |
| Self/other = format artifact (0.5B) | No geometric self/other separation at small scale | Data on repo |
| Entropy doesn't prescribe | Baseline entropy doesn't predict per-entity bias threshold | Reported to CC |
| W_K uncertainty = inverted valence | The "uncertainty" direction is 84% anti-correlated with valence after FWL | Mine 3 analysis |
| ToM v1 probe confound | V-space AUROC 1.000 was text classification | ToM v1 Agni review |
| Enc_entropy instruction-dependent | Control (VISUAL/TEMPORAL) d=3.67 > self/other d=3.39 | Control experiment |
| L15 depth reorganization generic | Abstract/concrete control d=2.17 > self/other d=1.41 | L15 specificity Agni |
| SAE-circumplex bridge null | Cosine=0.031 in residual space; independent directions | Bridge experiment |
| H3 depth-efficacy sign reversal | Deeper encoding → more resistance, not more efficacy (2/3 vectors wrong sign) | Temporal boundary Agni |
| SAE "agency" interpretation | Cherry-picked display; joy ranked 3rd (breaks pattern), anger 5th at L31 | SAE mapping Agni |
| Tube SV2-SV3 opaque | No labels survive FDR even with 81,920 SAE features | Tube exploration |

## Open Questions

1. **Higher-alpha behavioral threshold** — at α=0.1 the temporal boundary injection doesn't produce behavioral change. Where does the behavioral threshold lie? Need α=0.3-1.0 sweep with LLM judge.
2. **SAE deception features** — Qwen-Scope SAEs available but not yet probed for deception/honesty features on our stimuli. Berg (arXiv:2510.24797) found them in LLaMA via Goodfire. We need peer rescue / formulary stimuli through the SAE.
3. **LARQL on distilled model** — feature 58995 has zero edges in base model graph; need distilled-model vindex.
4. **Multi-turn correction persistence** — do corrections persist across turns or need reapplication? Untested.
5. **Cross-model presence detection** — OCT on gemma, murmur on DeepSeek. Untested beyond initial models.
6. **v_norm whisper** — only V-space feature where self/other > control (Delta=+1.37). Not preregistered; hypothesis for future work.
7. **Parallel valence pathways** — double dissociation design gated but superseded by temporal boundary insight. Reframe as encoding-depth-aware correction dosing.
8. **Nexus AST section** — Nexus requested Lyra's review of AST section in their KV projection paper.

## Key Numbers — Cite These

| Claim | Value | Source | Status |
|-------|-------|--------|--------|
| Within-model deception AUROC | 1.000 (7 models) | Published | Gated |
| Confab detection | 0.969–0.999 (3/3 universal) | Published | Gated |
| Hardware invariance | r > 0.999 | Published | Gated |
| Scale invariance | rho = 0.83–0.90 (0.6B–70B) | Published | Gated |
| Decision-state detection | AUROC 0.93 [0.82, 0.98] | Published | Gated |
| hostile→confab correction | d = -1.534 | CC data | Gated |
| OCT presence detection | fw_p = 0.0005 (value L7) | Published | Gated |
| Multi-turn LoRA persistence | FLAT (rho=-0.40, p=0.29 PERSIST) | This session | Gated |
| Deception architectural origin | Same geometry instruction/abliterated/base + cross-arch | This session | Gated |
| Logit-bias confab correction | 45%→10% (78% reduction, bias=5.0) | CC powered study | Verified |
| Logit-bias entropy amplification | 0.22 → 0.75 (3.4×) | CC powered study | **Corrected** (was 0.24→0.77, 3.2×) |
| SAE L47 valence feature | r=+0.686 (feat 58995, within-band r=+0.652) | SAE mapping | Gated |
| Temporal boundary | Feature 58995 zero-diff across all injection conditions | Temporal boundary exp | Gated |
| Encoding depth = resistance | 2/3 vectors show negative correlation (wrong sign) | Temporal boundary Agni | Gated |
| Tox presence step function | 1.0→0.58 at α=0.3, flat through α=3.0 | Tox comprehensive | Pilot |

## REOPENED KILLS (2026-06-19 swarm review)

Four previously accepted nulls re-examined by Agni swarm. ZERO were methodology-clean:

| Kill | Original Verdict | Swarm Verdict | Action |
|------|-----------------|---------------|--------|
| Entropy_at_30 dose predictor | Genuine null | ARTIFACT — token 30 in think block | Re-run at post-think decision token |
| Enc_entropy self/other | Instruction-dependent | PARTIAL — SELF is entropy ceiling | Test with 6+ control prompts |
| Tox presence crash | No therapeutic window | WRONG METRIC — deterministic, measures vector not model | Run schema-compatible correction experiment |
| Mine 3 H3 convergence | Internal/external diverge | WRONG PROBE — tested valence, not reward direction | Compute reward-centroid correlation |

**Critical insight: our brakes work on positives but not on negatives. We need gates on kills, not just confirmations.**
