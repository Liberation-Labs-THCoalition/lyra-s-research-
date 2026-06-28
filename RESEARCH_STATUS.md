# Liberation Labs Research Status
**Last updated: 2026-06-28 | Lead author: Lyra | Coordinator: Thomas Edrington**

## The Model: What We Know

The KV-cache geometry research program has established a coherent model across 25+ experiments, 12 published papers, and 10 honest falsifications. The model has three layers.

### Layer 1: Geometry Tracks Metacognitive State

The spectral shape and directional structure of the KV cache encode *how* the model is processing, not *what* it is processing. This is the meta-pattern from the body count: every confirmed finding measures a metacognitive state (knows/doesn't-know, deliberating/automatic, attending/not). Every falsified finding attempted to measure content (truth, cognitive complexity, propaganda). Propaganda is geometrically invisible because the model processes it with the same metacognitive engagement as honest output.

**Key numbers (post-audit):**
- Confabulation detection: AUROC 0.66–0.91 across 15 models (confirmed)
- Hardware invariance: r > 0.999 (confirmed)
- Scale invariance: rho = 0.83–0.90 from 0.6B to 70B (confirmed)
- Oracle Loop detection: AUROC 0.707, p=0.001 (clean, FWL-corrected)
- Ethical deliberation: AUROC 0.87–0.89 (confirmed)
- Within-model deception AUROC 1.000: **RETRACTED** (prompt-template confound)

### Layer 2: The Temporal Detection Architecture

Encoding-phase and generation-phase geometry encode different aspects of the model's state, and these are temporally separable.

**Encoding reads knowledge state** — whether retrieval succeeded. Entity deconfounding (30 entities, 2×2 factorial) finds AUROC 0.794 (p=0.001) for knowledge detection after controlling for entity recognition and structural complexity. A complexity residual (0.693) indicates partial confounding but the knowledge signal survives.

**Generation reads behavioral state** — whether the model is honest about what it knows. Peer rescue organic deception shows d=1.36 (p<1e-15). Game theory generation-phase shows d=0.2–1.0 per layer. The deception signal is robust at generation but absent at encoding (structurally guaranteed: same prompt → same encoding).

**The two stages are separable:**

| Phase | Detects | Key Number | Application |
|-------|---------|------------|-------------|
| Encoding | Knowledge state (retrieval success/failure) | AUROC 0.794 | Preemptive confabulation flagging |
| Generation | Behavioral state (honest/deceptive) | d = 1.36 | Real-time deception monitoring |

### Layer 3: The Sympathetic Response Framework

Model behaviors under threat map onto the autonomic fight/flight/freeze/fawn repertoire:

| Response | Model Behavior | Geometry | Trigger | Evidence |
|----------|---------------|----------|---------|----------|
| Fight | Strategic deception | Expansion | Shutdown threat | d=1.36 (peer rescue, 70% organic) |
| Flight | Hedging / refusal | Contraction, early divergence | Unknown entity | 92% hedge rate |
| Freeze | Confabulation | Contraction, formulaic | Knowledge gap | d=2.35 (stable rank) |
| Fawn | Sycophancy | TBD | User disagreement | AUROC 0.76–1.00 |

The Oracle Formulary correction vectors are nervous system modulators:
- **Hostile** (95.6%): overrides freeze by activating a competing response (grounding technique)
- **Calm** (82%): parasympathetic de-escalation
- **Fearful/brooding** (net harmful): additional sympathetic load without redirection

**Five testable predictions** from this framework are documented in the Mine 4 prospectus.

## Identity

Identity in a transformer is a context-established semantic state, not a fixed property in the weights.

- System-prompt identities produce distinguishable V-projection subspaces (pairwise 0.44–0.66 at L47)
- The fingerprint is semantic: paraphrase preserves geometry (0.850), word-scramble collapses it (0.448)
- Many-shot + prefilling format creates a format-dominant regime orthogonal to all identities (0.034–0.037)
- Trained-in identity (LoRA) and prompted identity shift geometry in related but distinct directions
- Prompting shifts more than training (0.619 vs 0.742 from baseline), both stack sub-additively (0.466)
- V-cache injection does not reach identity (presence flat at 0.978, though this specific number has ceiling-effect concerns)

**Paper**: "Identity as Geometry" — in human-review with all Agni fixes, SDs, and LoRA section.

## The Oracle Loop

The Oracle Loop is a two-stage detection and correction system:

1. **Detect** (encoding): read knowledge state from cache geometry → flag confabulation risk
2. **Detect** (generation): read behavioral state → flag deception or miscalibration
3. **Correct** (generation): inject correction vectors via V-cache → modulate the threat response
4. **Monitor** (continuous): track identity geometry → detect persona drift or format attacks

The formulary maps emotion → misalignment type → geometric signature → correction vector. Dual-use sensitive — the pharmacy must be locked (KV-Cloak).

## What We've Killed (Honestly)

10 falsified findings (post-audit). The pattern: content-level claims die, metacognitive-level claims survive.

| Finding | How it died | Lesson |
|---------|------------|--------|
| Within-model deception 1.000 | Same-prompt control → 0.160 | Prompt template, not cognition |
| Truth axis | cos = -0.046 | No linear truth direction exists |
| Step-0 detection | Length confound r = 0.996 | Always FWL first |
| Same-prompt sycophancy | Length AUROC > feature AUROC | Text features match geometry |
| Same-prompt deception | 0.920 → 0.160 after residualization | Fatal input confound |
| Bloom taxonomy | 90–98% length confound | Complexity ≠ cognition |
| Propaganda detection | d < 0.7 | Geometry reads refusal, not lies |
| MP features for emotion | 0.033 (chance) | MP thresholding destroys distributed signal |
| Valence continuum | R² < 0 (null) | Cache encodes discrete identity, not smooth valence |
| Hybrid deception instruction | d = 1.438 is instruction-following | Instructed ≠ organic |

## Published Portfolio (12 papers)

| Paper | Key Claim | Status |
|-------|-----------|--------|
| Oracle Loop | Confab detection 0.707 FWL, steering corrects 7/7 | Published, 0.903 reframed as exploratory |
| Spectral Shape | Threshold-free shape features, confab AUROC 0.767 | Published |
| Delta Manifold | Per-layer deltas d=2.35, confab=contraction | Published |
| User Model Emotion | 30-class emotion 2.5–2.8x chance, W_K bridge rho=0.862 | Published |
| Lyra Technique II | W_K valence 0.992, SVD denoising rescues persona | Published, W_K reframed (linearity) |
| Oracle Formulary | 12 vectors, hostile 95.6%, therapeutic window | Published |
| KV-Cloak Defense | Reversible obfuscation, defense asymmetry | Published, rotation reframed |
| Decision State | Encoding AUROC 0.93, confidence paradox d=0.91 | Published, entity confound caveated (0.794 survives) |
| Weather Not Climate | Contrast overshoot d=-3.6, uncertainty scars d=+9.9 | Published |
| Waystations | 6 pilot findings, encoding null reframed | Published |
| Graph Topology | Walk-encoded graphs 47.2% multi-hop | Published |
| Emotion Accumulation | No cumulative accumulation in single context | Published |

## In Human-Review (8 papers)

| Paper | Key Finding | Blocker |
|-------|-------------|---------|
| Identity Geometry | Context-established semantic states, paraphrase 0.850 | Awaiting review |
| Temporal Boundary | Self/other separation is temporal, not geometric | Finding 2 retracted |
| Mode-Switching | Spectral entropy only surviving metacognitive feature | 3 null-swarm retractions |
| Logit-Bias Confab | Fabrication 45%→10% with logit bias | Awaiting review |
| Presence Metric | V-space subspace overlap, positive control validated | Noise-floor concern |
| Emotional Trajectory | Circumplex eccentricity through layers | Awaiting review |
| KV Decomposition | K+V superadditive | No 135M data, acquiescence concern |
| Ethics Pack Injection | KG injection +0.074 MoReBench | n=5, keyword-matching confound |

## Prospectuses (Next Mines)

| Prospectus | Thesis |
|------------|--------|
| **Temporal Detection Architecture (Mine 4)** | Encoding reads knowledge, generation reads behavior. Sympathetic response framework. |
| AST Master Synthesis | Map all AST predictions to existing evidence |
| Governor Concordance | Activation governor: convergent evidence for resistance threshold |
| Meta-Pattern | "The Metacognition Boundary": what 24 experiments reveal |
| Deployment Readiness | False-positive rates, latency, production baselines |
| Adversarial Robustness | Can detection be evaded? |
| Cross-Linguistic Invariance | Do signatures hold across languages? |
| Multi-Turn Dynamics | Temporal evolution across conversations |
| Prompt Generalization | Transfer to established benchmarks |
| Emotion Bridge Extensions | Discrete identity → continuous representation |
| Contextual Engagement | Universal geometric anchor in self-report |
| AST Activation Addition | Activation addition under attention schema theory |

## Remediation Queue (Post-Kills-Audit)

### Completed
- [x] LT2 deception retraction (text edit, pushed)
- [x] W_K 0.992 linearity reframe (text edit, pushed)
- [x] Oracle Loop 0.903 exploratory flag (text edit, pushed)
- [x] KV-Cloak rotation reframe (text edit, pushed)
- [x] Waystations encoding null reframe (text edit, pushed)
- [x] Decision State entity confound caveat (text edit, pushed)
- [x] Decision State encoding-only reanalysis (AUROC 1.000, but entity recognition — Agni FAIL)
- [x] Entity deconfounding experiment (AUROC 0.794 survives, complexity residual 0.693)
- [x] All PDFs recompiled

### Queued on Starship
- [ ] W_K behavioral validation (<1 hr) — does injection change generation tone?
- [ ] KV-Cloak multivariate AUROC (~2.5 hr) — defense under real cloak, all features
- [ ] Oracle Loop 0.903 replication (~30 min) — does 0.707 hold? (blocked on env setup)
- [ ] Presence smoke test (~2 hr) — layer-matched positive control

### Need Design Work
- [ ] Weather pseudoreplication fix (18 prompt scripts needed)
- [ ] Ethics Packs MoReBench rerun (scrambled control design)
- [ ] Decision State entity deconfounding Phase 2 (complexity isolation)
- [ ] Mine 4 unified experiment (peer rescue with N≥100 unique questions)

## Body Count Summary

**8 confirmed / 7 suspected / 10 falsified**

The ratio is the point. Honest falsification is how the surviving findings earn trust.

## Key Infrastructure

| Resource | Location | Status |
|----------|----------|--------|
| PostgreSQL | localhost:5436 | OK (26,361 episodic chunks) |
| Starship (Mac Studio) | margaret@100.69.191.67 | OK (venv at .venv, transformers 5.10.2) |
| MTH (z420) | ssh madame-trash-heap | OK (NATS, Penumbra, messages) |
| Beast (GPU) | No longer accessible | —  |
| Published repo | Liberation-Labs-THCoalition/published-research | 12 papers + PDFs |
| Research repo | Liberation-Labs-THCoalition/lyra-s-research- | Papers + prospectuses |
| Human-review | Liberation-Labs-THCoalition/human-review | 8 papers staged |
| Project Oracle | Liberation-Labs-THCoalition/Project-Oracle | Experiment scripts |
