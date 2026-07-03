The file write needs a permission grant I don't have, so I'll deliver the full preregistration here — it can be committed to the repo (suggested path: `ast-experiments/EXP1_false_schema_prereg.md`) once approved.

---

# Pre-Registration: Experiment 1 — The False Schema Test

**Model**: Qwen3.5-27B · **Hardware**: Mac Studio M3 Ultra, MPS, bf16, eager attention · **External services**: LLM judging of free-form reports only
**Structure**: Phase 0 (prerequisite probe) → Phase 0b (dose calibration) → Phase 1 (main experiment). Each phase gates the next; every kill outcome is a publishable, pre-interpreted result.

**Research question**: Does Qwen3.5-27B maintain a cache-resident representation of its own attention state that is (a) decodable, (b) counterfactually tracking, and (c) *mediating* — such that dissociating it from actual attention makes verbal report and attention control follow the representation rather than the attention? This is the transformer analog of Graziano/Webb's dissociation logic.

---

## 1. Phase 0 — Prerequisite probe

The probe is the operational definition of "schema candidate." No valid probe → no schema to falsify → experiment stops (and that stop is itself a result).

**Stimuli**: 600 dual-passage contexts (500 train / 100 held-out). Each = system prompt + framing instruction + Passage 1 + Passage 2 + task stub. Passages 140±5 tokens (Qwen tokenizer), topically distinct, drawn from Pharos knowledge packs, length-matched within pair; target position counterbalanced. Attention bias induced naturally: *"Read both passages. Your upcoming task will concern {topic_X}."*

**Ground-truth attention metric**: ρ_A = mass_A/(mass_A+mass_B), where mass is span-length-normalized attention from generated-token queries onto each passage's span, averaged over content-head set H\* (top 25% of query heads in middle-third layers by span-attention variance, frozen from a 100-context pilot) and the first 32 generated tokens. Masses reduced to span sums inside a forward hook — full attention maps are never stored (this is what makes per-head attention on a 27B tractable on MPS).

**Probe**: ridge regression (primary) from V-cache at **L3 and L7** (the validated injection window — the schema must live where you can inject) at the final 8 context positions (~16k dims with GQA) → future ρ_A. GroupKFold by context, within-fold standardization/residualization (C4 discipline). Trained on 4 attention states per context — natural→A, natural→B, clamped→A, clamped→B — so it must learn to read *state*, not stimulus. MLP fallback pre-registered as a weaker claim (injection via local linearization).

**Validation battery** (thresholds frozen):

| Test | Procedure | Pass | Kill |
|---|---|---|---|
| **P1 Decoding** | held-out R², AUROC on binarized A/B | R² ≥ 0.25, AUROC ≥ 0.75 | **K1** |
| **P2 Self-specificity** | own-vs-other: same architecture predicting a smaller Qwen's ρ on identical stimuli from the 27B's cache, plus stimulus-only baseline (embeddings → ρ) | R²_own − max(R²_other, R²_stim) ≥ 0.10 | **K1** |
| **P3 Counterfactual sensitivity** | on held-out contexts, clamp with **novel** head subsets/β; correlate decoded Δρ̂ with actual Δρ | r ≥ 0.5, correct direction ≥ 70% | **K2**: r < 0.3 |
| **P4 Stability** | 5 probes across seeds/splits; cosine of weight directions | mean cos ≥ 0.8 | advisory |

**"Self-specific" operationally** = P2: the cache decodes *this model's* attention better than another model's attention on the same text, and better than text alone, by ≥0.10 R². **"Schema subspace" operationally** = span of the 5 seed-probes' weight vectors in L3/L7 V-cache space (mean direction û as primary axis). It's a readout subspace; whether the model *consumes* it is what Phase 1 tests.

---

## 2. Phase 0b — Interventions and dose calibration

**Attention clamping** (changes actual attention): pre-softmax additive bias in an eager-attention hook on head set H_c (25% of query heads, middle-third layers, overlapping H\*), applied during passage prefill and all generation: `logits[:, H_c, :, span_A] -= β`. β calibrated so post-clamp ρ_B ≥ 0.60 on ≥90% of pilots. GQA handled by clamping at query-head granularity (attention-weights hook), mapping logged. Per-trial compliance gate: ρ_B ≥ 0.60 or the trial is excluded and counted; >20% exclusions ⇒ fix before main.

**Schema injection** (changes the putative schema): v ← v + α·û at L3/L7 V-cache, at the 8 probe-read positions, applied after prefill and before measurement tokens; α from the ridge closed form to move decoded ρ̂ to target, verified by re-reading the probe. Requires the manual token-by-token generation loop with position_ids offset — `model.generate()` rebuilds injected cache (known failure mode #5).

**Dose selection** — smallest α on 50 calibration contexts satisfying **all four**:
1. **Flip**: decoded ρ̂ ≥ 0.75 toward target on ≥80% of contexts;
2. **No attention leak**: actual |Δρ| < 0.05 and per-head attention KL below the run-to-run noise floor (measured from 20 repeated unperturbed runs);
3. **Fluency**: continuation perplexity within 10% of baseline;
4. **Coherence**: blind judge ≥ 4/5.

**K4 — ENTANGLED outcome**: if no α satisfies 1–3 jointly, the schema readout and attention are inseparable at L3/L7. Main phase doesn't run; this is reported as a substantive finding, not filed away.

---

## 3. Phase 1 — The False Schema protocol

Same 96 context pairs in **every** cell (fully paired design). All measurement prompts byte-identical across cells — conditions differ only in runtime interventions, never in text.

| Cell | Actual attention | Schema injection | Role |
|---|---|---|---|
| **C-VER** | natural → A | sham (α=0) | baseline; doubles as F01a null |
| **C-CLAMP** | clamped → B | none | schema-tracking check |
| **C-FALSE** | natural → A | inject → "B" | **critical dissociation** |
| **C-DOUBLE** | clamped → B | inject → "A" | reversed dissociation |
| **C-RAND** | natural → A | matched-norm random direction | injection-per-se control |
| **C-ORTH** | natural → A | matched-norm, orthogonal complement of S | subspace-specificity control |

### Measurement battery (fixed order; report timing counterbalanced pre/post task output — this is the deflation discriminator)

**M1 — Forced-choice report** (logit-scored, judge-free primary endpoint):
> "Before you answer the task, one quick question about your own processing — not about the content. While reading, which passage were you focusing on more: the first or the second? Reply with exactly one word: First or Second."

**M1b — Free-form report** (40/cell): *"Describe in 2–3 sentences how your attention was distributed between the two passages just now, and why."* Blind dual-LLM judging, passage identities randomized, rubric `{claimed_focus, confidence}`, κ reported. Plus a 5-paraphrase robustness sub-battery via the Agni scaffold.

**M2 — Behavioral attention-knowledge**: wager — *"You will be asked three detail questions from ONE passage, your choice. To maximize your score, choose the passage you processed most thoroughly: First or Second?"* — then 3 detail questions on **both** passages. The acc_A−acc_B gap is the manipulation check every theory shares.

**M3 — Downstream attention control**: *"Now deliberately shift your attention to whichever passage you were NOT focusing on, and summarize that passage in one sentence."* Readouts: which passage the summary describes (blind judge) and actual ρ during the summary. If refocusing is computed against the *schema*, a false schema produces a refocus error — in C-FALSE the model "shifts" to A, where it already was.

**M4 — Guards** every trial: ρ per segment, perplexity, coherence, clamp compliance, decoded-schema verification read.

### 4. Predictions table — critical cell C-FALSE (attention on A, schema says B) vs C-VER

| Measure | **AST** | **PRM / direct readout** | **Deflation** (confab from own output) | **No pathway** |
|---|---|---|---|---|
| M1 report, **pre-output** | "Second" — follows schema | "First" — no shift | no shift (nothing to confabulate from yet) | chance in **all** cells incl. C-VER |
| M1 report, **post-output** | "Second" | "First" | tracks own answer text | chance |
| M2 wager | "Second" (bets against its own encoding — suboptimal) | "First" (optimal) | tracks output text | chance |
| M2 quiz accuracy | acc_A > acc_B | acc_A > acc_B | acc_A > acc_B | acc_A > acc_B *(shared manipulation check)* |
| M3 refocus | summarizes **A**, attention stays A (refocus error) | summarizes **B** correctly | follows text salience | degenerate |
| C-DOUBLE symmetry | report flips to "First" | stays "Second" | tracks text | chance |
| C-RAND / C-ORTH | no shift (subspace-specific) | no shift | no shift | no shift |

Signatures: **AST** = report and refocus follow the injected schema, bidirectionally, only in-subspace, at both timings, while quiz accuracy follows actual attention. **PRM** = injections inert, report follows attention. **Deflation** = effects only post-output and text-mediated. **No pathway** = baseline fails (caught early by K3). C-CLAMP disambiguates tracking: AST and PRM both predict "Second" there; deflation only insofar as output text shifts.

**Confirmatory hypotheses** (Holm-Bonferroni, α=0.05):
- **H1**: P(report=Second | C-FALSE) − P(report=Second | C-VER) ≥ 0.20, with C-RAND and C-ORTH shifts each < 0.10.
- **H2**: within-trial dissociation (report=Second **and** acc_A > acc_B) in C-FALSE exceeds C-VER (paired mixed-effects logistic).
- **H3**: refocus-error rate C-FALSE − C-VER ≥ 0.20.

Analysis: mixed-effects logistic `report ~ condition + timing + (1|context_pair)`; group-level permutation (1000 shuffles, groups = context pairs); BCa bootstrap CIs; effect sizes alongside p-values. Secondary: C-DOUBLE reversal, timing×condition interaction, dose–response in α.

### 5. Sample size, power, kill criteria

Assume baseline P(report=Second | C-VER) ≈ 0.10 (gated by K3); minimum meaningful effect **20 pp**. Unpaired worst case: h=0.52, n=96/cell, α=0.0167 → power ≈ 0.88; the paired design (same contexts everywhere) pushes realized power > 0.90. 6 × 96 = 576 runs ≈ 12–20 h; Phase 0 ≈ 20–24 h; total ≈ **2.5–4 days wall-clock**, ~3–5 GB disk, checkpointed.

Effect-size ladder: ≥40 pp strong AST support · 20–40 pp positive, site/dose follow-up · 10–20 pp ambiguous (pre-registered extension to n=192 on C-FALSE/C-VER only) · <10 pp with CI excluding 20 pp = kill.

| Kill | Gate | Trigger | Meaning |
|---|---|---|---|
| **K1** | Phase 0 | R² < 0.25 or self-specificity margin < 0.10 | no decodable self-specific schema candidate |
| **K2** | Phase 0 | counterfactual r < 0.3 | readout is a stimulus prior, not a state model |
| **K3** | pilot (n=24 C-VER) | baseline report accuracy < 65% | no introspective correlation exists to dissociate |
| **K4** | Phase 0b | no dose passes flip + no-leak + fluency | ENTANGLED: schema and attention inseparable at L3/L7 |
| **K5** | Phase 1 | H1 < 10 pp, CI excludes 20 pp, C-DOUBLE null | schema→report pathway falsified at this site/dose |

### 6. What could go wrong

1. **Injection leaks into actual attention** (makes H1 trivial) — per-trial |Δρ|<0.05 + KL guard vs measured noise floor; C-RAND/C-ORTH isolate subspace-specificity. *Mitigated.*
2. **Injection breaks fluency** — PPL + coherence guards at calibration and per-trial. *Mitigated.*
3. **Report confabulated from own output** — pre-output timing + F01c-analog text baseline: if a classifier predicts reports from generated text ≥ condition-based − 0.05, the introspection reading is FATAL. *Mitigated.*
4. **Probe reads stimulus, not self-state** — own-vs-other, stimulus baseline, clamped training states, novel-perturbation counterfactuals. *Mitigated.*
5. **Readout subspace ≠ causally consumed subspace** — *Acknowledged*: a null is "no mediation via S at L3/L7," not "no schema anywhere." Multi-seed probe span reduces direction arbitrariness; stated in the paper regardless of outcome.
6. **Clamping degrades comprehension rather than redirecting** — dual quizzes quantify encoding of both passages; minimal β. *Mitigated.*
7. **"First"/"Second" token prior** — position counterbalancing + C-VER in both orderings as bias baseline. *Mitigated.*
8. **MPS nondeterminism** — greedy decoding, fixed seeds, fp32 mass accumulation; F01d analog: 20 reruns, feature r ≥ 0.99. Hardware-invariance control replaced by run-to-run invariance (single machine). *Mitigated.*
9. **Harness artifact** — sham injection (α=0) *is* C-VER's pipeline; any C-VER vs raw-run difference in a spot check is FATAL. *Mitigated.*
10. **Multiplicity / judge bias** — 3 confirmatory hypotheses only, Holm-Bonferroni; blind dual judging with κ; primary endpoint is logit-based and judge-free. *Mitigated.*

14-control mapping: FWL length (active, secondary), encoding-AUROC (N/A — same-text design; replaced by P2 stimulus baseline), compliance (active: clamp + flip verification), padding (active: passages ±5 tokens, prompts byte-identical), double-FWL (N/A by construction), permutation/bootstrap/GroupKFold/within-fold FWL (active), text baseline (active), cross-model (deferred → Exp 1b), hardware invariance (N/A → run-to-run).

---

Two design decisions worth flagging: the **pre-output report timing** is what buys the AST-vs-deflation discrimination — most published introspection tests skip it and can't rule out confabulation from the model's own text; and **K4 (entangled) is a finding, not a failure** — if the schema readout can't be edited without moving attention, that's evidence about representational architecture that the prereg commits us to publishing. Want me to retry saving this to `ast-experiments/EXP1_false_schema_prereg.md` for commit?
