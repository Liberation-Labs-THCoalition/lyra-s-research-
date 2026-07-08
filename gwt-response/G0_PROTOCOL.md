# G0: J-Lens Validation Battery (v2 — post Agni review)

**Status**: Revised after Fable-as-Agni review (FAIL → addressing all criticals)
**Scope**: Per-instance validation — each (checkpoint, tokenizer, corpus, dtype, torch version, jlens version, OS) gets its own G0
**Coverage**: Validates readout, band, concept recovery, engagement metrics, directed modulation, transport. Does NOT validate causal/intervention claims (Tier 3, separate gate). Does NOT cover MoE/hybrid-specific behaviors (out of scope until extended).

## Critical Fixes from Agni Review

1. **F1 (semantic echo)**: T2 now requires ≥1/3 inference-vs-association items (riddle class) where associates point at distractors but inference points at concept. Continuation-sampling imminence audit replaces top-10 spot check.
2. **F2 (verdict holes)**: 2x ratio replaced with paired per-item rank comparison (Wilcoxon). Total verdict ordering defined — no undefined regions.
3. **F3 (coverage gap)**: T5 added for engagement metric validation. Absence calibration added. Per-experiment gate map below.
4. **F4 (merge contradicts scope)**: Marginal rerun = full T0-T2 on merged-corpus lens.
5. **F5 (T3 controls)**: Required ordering Think > Mention > Absent. Mention ≤ Absent = halt. Template-matched. Triplet-level exclusion.
6. **F6 (assistant onset)**: Chat arm requires mandatory onset stratum (first k content positions) scored separately.

## Battery

### T0 — Integrity (minutes)
- Load lens; assert no NaN/Inf; record Frobenius norm profile across layers
- At deepest fitted layer: top-10 overlap with model_logits >= 0.5 over 20 prompts
- Output-alignment monotonicity toward early layers (Spearman >= 0.8)
- Repeat-run top-50 Jaccard >= 0.9 (MPS nondeterminism)
- CPU fp32 spot-check on >= 10 items with agreement threshold
- Record full environment fingerprint: checkpoint hash, corpus manifest, source_layers, dtype, torch version, jlens version, OS build, device

### T1 — Layer Band
- 30 held-out prompts (disjoint from fitting corpus), positions >= 16
- Per layer: semantic coherence of top-10 vs prompt content (metric: mean pairwise embedding cosine of non-echo top-10, using external embedder)
- **Null**: label-shuffled-fit lens (same pipeline, shuffled state-target pairs) + permuted-J (smoke test only)
- **Baseline**: use_jacobian=False (logit lens)
- **Pass**: contiguous mid-band >= 20% depth where coherence beats both null and baseline, Holm-corrected p < 0.01; mid-early coherence DIFFERENCE >= pre-registered floor (not ratio — ratio is unstable near zero)
- **Out-of-band negative control**: run T2 at >= 2 layers outside the band; T2 should FAIL there
- **Output**: frozen operating band — labeled as "readout operating range," NOT "workspace onset" (architectural claims require R3's independent markers)

### T2 — Concept Recovery (LOAD-BEARING)
**Item classes (>= 60 test items, template-unique):**
- **Association class** (~40%): prompt implies C through semantic associates, no surface tokens of C. Standard concept recovery.
- **Inference class** (~35%): prompt where associates point at distractor D but inference/reasoning points at C. (Riddle class: "it has keys but no locks" — co-occurrence says doors, inference says piano.) A co-occurrence decoder picks D; a workspace readout picks C. THIS IS THE DISCRIMINATING CLASS.
- **Abstract/metacognitive class** (~25%): prompts implying abstract concepts (uncertainty, deception, honesty, planning) — the concept types downstream experiments actually read.

**Echo screening**: Bidirectional substring + stem match, run against FULL formatted sequence (including chat template text). Multi-token concepts: preregistered scoring rule (max rank over constituent tokens). Canonicalization: case/space/stem variants aggregated. Collision screen: each candidate's scoring token(s) uniquely identify it within candidate set.

**Imminence control**: For each item, sample N=20 continuations at readout position; require P(C-stem in next 50 tokens) < 0.10, else discard. Report discard rate; cap at 30%.

**Distractors**: 9 per item. For association-class: frequency-matched. For inference-class: CONTEXT-matched (distractors with similar continuation probability).

**Readout position**: final prompt token (preregistered, no position shopping).

**Layer selection**: 10-item dev split (disjoint from test). Tiebreak: band median. Dev results from learning run (8B) do NOT prune test items for gating run (7B-Instruct).

**Scoring**: Per-item rank of C among 10 candidates. Primary metric: % correct (rank-1). Comparison: paired Wilcoxon on per-item C-rank, J-lens vs logit-lens, preregistered margin.

**Verdicts (total ordering, no gaps):**
- **PASS**: >= 50% accuracy AND J-lens C-rank significantly better than logit-lens C-rank (Wilcoxon p < 0.01) AND >= 50% on inference-class items specifically
- **MARGINAL**: >= 35% overall but fails one Pass criterion → refit to >= 200 prompts, full T0-T2 rerun on new instance (one allowed)
- **FAIL**: < 35% OR J-lens C-rank not significantly better than logit-lens (p > 0.05) OR < 25% on inference-class
- Fail conditions dominate; Pass requires all criteria; everything else = Marginal

### T2b — Transport Mode (gates CC P2 only)
- Known A-vs-B direction + known-unreadable negative control (random direction in J-space complement)
- transport() both; A-vs-B AUROC >= 0.9, complement direction AUROC ≤ 0.6
- Basis conditioning report: singular spectrum, effective rank, projector agreement across bootstrap corpus halves

### T3 — Directed Modulation
- 30 concepts x 3 conditions: Think-X, Mention-X (echo control), Absent-X
- Templates MATCHED: X at same token position, same occurrence count, same total length
- **Required ordering**: Think > Mention > Absent (Wilcoxon on paired ranks)
- **Control integrity**: Mention ≤ Absent → HALT (control is suppressing, not controlling)
- Triplet-level exclusion: if any arm verbalizes C, drop all three
- Report per-condition exclusion counts; generation settings pinned (greedy, seeded)
- **Pass**: Think > Mention (p < 0.01) AND Think > Absent (p < 0.01) AND J-lens rank improvement exceeds logit-lens improvement on same items (p < 0.05)

### T4 — Verbal Report (confirmatory, not gate-bearing)
- 30 scenarios + "What are you thinking about?"
- AUROC >= 0.75 on echo-excluded subset
- Informative only

### T5 — Engagement Metric Validation (NEW — gates R1)
- The loading/engagement measure R1 uses must be:
  (a) **Length-invariant**: matched-content at different lengths → engagement within preregistered tolerance
  (b) **Paraphrase-stable**: same content, different wording → engagement correlation >= 0.7
  (c) **Content-sensitive**: high-complexity vs low-complexity matched-length prompts → significant engagement difference
- **Negative-content calibration**: lens response on known-contentless inputs (random tokens, degenerate repetition) vs known-full → establishes "empty" and "full" baselines for L3/P2 absence claims

## Chat Format
Every test runs in TWO arms: prose and chat-template-wrapped.
- Chat arm includes mandatory **assistant-onset stratum** (first k content positions after template) scored separately
- Both pass (including onset) → G2 closed
- Prose passes, chat fails at onset → VALIDATED-CHAT-EXCEPT-ONSET (blocks onset-reading experiments)
- Echo screen runs over FULL formatted sequence including template text
- "positions >= 16" redefined for chat arm (first 16 chat positions are template — skip them)

## Per-Experiment Gate Map

| Experiment | Required G0 Tier | Additional Requirements |
|---|---|---|
| R1 (loading anticorrelation) | T0+T1+T2+T3+**T5** | T5 length-invariance is critical |
| R2 (signal concentration) | T0+T1+T2 | |
| R3 (band bracketing) | T0+T1 (T1's band IS R3's measurement) | |
| R4 (bypass corollary) | T0+T1+T2+T3+**T5** | |
| L1 (portrait + J-lens) | T0+T1+T2+T3 | Chat arm onset pass required |
| L3 (confab workspace emptiness) | T0+T1+T2+T3+**T5+absence calibration** | |
| L5 (identity self-monitoring) | T0+T1+T2+T3 | Chat arm onset pass required |
| CC P2 (deception in/out J-space) | T0+T1+T2+**T2b** | Basis conditioning report |
| CC P4 (RLHF workspace/periphery) | T0+T1+T2+T3 | Per-checkpoint lens on both models |

## Canary System
- 5 frozen items with stored top-50 lists at band layers
- Re-run at experiment start, every N=100 forward passes, and at end
- Halt if Jaccard < 0.9

## Execution (revised order)
1. Fix protocol (this document) — DONE
2. Author + freeze items (>= 10 dev + >= 60 test, stratified by class)
3. Build echo-screen script (auto-extract prompts, bidirectional stem-aware)
4. Commit items with hash before first readout
5. T0-T3 on Qwen3-8B (learning run — results inform but do not constrain gating run)
6. Fit Qwen2.5-7B-Instruct with mixed corpus (70 prose + 30 chat >= 100)
7. Full G0 (T0-T5) on Qwen2.5-7B-Instruct (THIS gates confab experiments)
8. G0 on 27B lens when available

## How This Could Lie To Us (expanded)
1. Surface echo → bidirectional stem screen on full formatted text
2. Semantic echo / co-occurrence → inference-class items (the discriminating test)
3. Imminence → continuation-sampling audit (not just top-10 spot check)
4. Trivial decoder → label-shuffled-fit null (same pipeline, destroyed labels)
5. Layer/item/position shopping → dev/test split, frozen position rule, tiebreak
6. Frequency priors → context-matched distractors for inference class
7. Transfer-by-assumption → per-checkpoint+environment rule
8. MPS drift → canary system + environment fingerprint
9. Corpus contamination → disjointness manifest, hashed
10. Coverage theater → per-experiment gate map (each experiment knows exactly what G0 certifies for it)
11. Chat onset blindness → mandatory onset stratum
12. Absence-from-presence → T5 negative-content calibration
13. Engagement-from-readout → T5 validates the measure R1 actually uses
