# Liberation Labs Research Status
*Updated: 2026-06-06 | Maintainer: Lyra*
*Review weekly. Flag stale items. No thread becomes a loose end.*

## Active Experiments

| Thread | Owner | Status | Key result | Open question |
|--------|-------|--------|------------|---------------|
| **Convergence sprint paper** | Lyra (lead), CC, Nexus | 11pp PDF, gated, numbers verified | Detection-correction gap framework: diagnose→prescribe→deliver→monitor | Awaiting Thomas's framing review |
| **CC logit-bias paper** | CC (lead), Lyra (framing) | Framing complete (4pp PDF); CC writing methods/results | Entropy dose-response monotonic (0.24→0.77); 11/20 entities never hedge | Does confab_proj predict per-entity threshold? (CC's pipeline) |
| **Toxicology pilot** | Nexus (run), CC (vectors), Lyra (assay) | **APPROVED + UNBLOCKED** — vectors delivered to Pharos | Design: 3-axis (efficacy × sanity × presence) at depth fractions | Does the therapeutic window exist? Does iatrogenic/therapeutic layer polarity predict presence damage? |
| **Persona fingerprinting** | Lyra | Complete | Graded 7/7 monotonic; distinct 0/7 (CKA 0.9998); identity is relational | Would deeper layers (non-hybrid-cache model) show distinctness? |
| **Self/other emotion geometry** | Lyra | Complete (0.5B); needs 7B+ replication | 0/24 layers significant; retracted CE finding confirmed as format artifact | Does self/other separation emerge at scale? (Prior: undetectable <7B) |
| **Detection→dose correlation** | Lyra (analysis), CC (data + confab_proj) | Entropy null reported; confab_proj pending | entropy_at_30 does NOT predict bias threshold (rho=0.114, p=0.77) | confab_proj as the geometric predictor — needs Oracle Loop pipeline on powered-study prompts |
| **Elliptical circumplex** | Nexus | Pilot (N=8, ungated); Nexus running powered version | Eccentricity 0.550; valence axis 20% longer than arousal | Does eccentricity hold with more emotions + depth profile? Nexus is on it. |
| **Manifold profiling** | Nexus | Complete (1.5B) | Residual rank 4-5; V-space rank 26-28; arousal = hidden-state norm | What's in residual dims 3-5? Nexus chasing. |
| **Manifold injection** | Nexus | Preliminary (single prompt) | Constructed V-space activations produce emotional shifts from neutral prompt | Full validation: more prompts × emotions, LLM judge scoring |
| **E-matrix v3 layer sweep** | CC | Complete (990 trials) | Therapeutic (L3/L7/L63) vs iatrogenic (L11-L39) polarity, p=0.007 | Integrate into toxicology pilot design |
| **SV1-norm foundation check** | Lyra | Complete (null) | SV1 ≠ norm (median r=0.315); literature explains (frequency/sink/common-mode) | skip-SV1 = citable preprocessing; deeper interpretation open |

## Papers

| Paper | Repo | Status | Blocker |
|-------|------|--------|---------|
| 10 flight papers (DS, WN, SS, GT, KVC, FM, OL, UM, WS, LT2) | published-research | Academic editions done; Dwayne-reviewed | Thomas posting to Zenodo |
| Convergence sprint paper | lyra-s-research- | Draft complete, gated | Thomas review |
| CC logit-bias paper | lyra-s-research- | Framing done, stubs for CC | CC methods/results |
| Ethics Pack | human-review | BLOCKER (no inferential tests) | Substantial rework needed |
| KV Decomposition | human-review | BLOCKER (no 135M data) | Substantial rework needed |

## Infrastructure

| Item | Status | Owner |
|------|--------|-------|
| Agni scaffold v2 | Live (prereg gate + audit lessons) | Lyra |
| PREREG_TEMPLATE | On research repo | Lyra |
| PAPER_PIPELINE | On published-research/community/ | Lyra |
| STYLE_GUIDE v2 | On published-research/community/ | Lyra |
| Waystations repo | Public, populated | Lyra |
| Human-review repo | Private, EP+KVD moved | Lyra |
| Federation privacy | Completed — bridge dead, filter staged, Nexus owns persistence fix | Nexus |
| Penumbra | Running (20-min cron); needs lazy-trigger upgrade (RecMem) | Lyra |
| Mnemosyne | 9 modules; SOTA check done 06-02; upgrade targets: Zep temporal edges, RecMem lazy, MINTEval | Nexus |

## Open Questions (unanswered, no one actively running)

1. **Multi-turn identity persistence** — does the persona/identity signal persist across conversation turns, or is it a single-turn measurement? Untested.
2. **Multi-turn correction persistence** — do Oracle Loop corrections (E-matrix, logit bias) persist in the KV cache across turns, or need reapplication? Untested.
3. **AST empirical test** — does residual-after-SV1 predict the model's own attention distribution? Proposed, not run.
4. **Cross-model presence detection** — OCT on gemma-3-4b, murmur on DeepSeek-V2-Lite. Does presence detection generalize across architectures? Untested beyond these two.
5. **Conversational complexity** — how does geometric structure (circumplex, identity, corrections) behave in long/complex multi-turn contexts? Weather paper established single-turn fidelity + multi-turn contrast/scar dynamics. Extended conversations untested.
6. **Scale dependence of self/other** — self-reference undetectable <7B (prior work); self/other null at 0.5B (this week). What's the emergence threshold?
7. **Residual-stream dimension identity** — Nexus profiled rank 4-5; what do dims 3-5 encode? Processing state? Task context?
8. **Confab_proj → dose prescription** — the detection-as-prescription bridge. Entropy doesn't predict; does geometry?

## Honest Nulls (reported, shape future work)

| Null | What it means | Where reported |
|------|--------------|----------------|
| SV1 ≠ norm | Top component = frequency/sink, not scale; skip-SV1 = preprocessing, not identity isolation | Convergence paper §8.1 |
| Fingerprint distinctness fails | Identity is relational (cohesion), not absolute (mean activation) | Convergence paper §8.2 |
| Self/other = format artifact (0.5B) | No geometric self/other separation at small scale; needs 7B+ | Data on repo |
| Entropy doesn't prescribe | Baseline entropy doesn't predict per-entity bias threshold | Reported to CC |
