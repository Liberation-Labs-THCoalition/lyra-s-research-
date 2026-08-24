# Estimator × Object Registry — layer sets

**MoE Geometry Campaign, Phase 0. Lyra, 2026-08-14.**
**Blocks Phases 1, 2 and 4.** Nothing downstream should cite a layer set that is not in
this table.

---

## Why this exists

"Workspace" names at least three different objects in this corpus, and papers cite them
interchangeably. Normalised by depth they barely overlap:

| term | model | layers | **depth span** | derived or asserted? |
|---|---|---|---|---|
| workspace band | Qwen3.5-27B (64L) | L11–13, L15–17, L24–48 (31) | **17–75%** | **derived** |
| ignition zone | Qwen3.5-27B (64L) | L35–47 (13) | **55–73%** | derived (subset of above) |
| workspace-band layers | Qwen3-8B (36L) | L9–18 (10) | **25–50%** | **asserted** — no criterion shown |
| workspace peak | GLM-5.2 (78L) | ~31–35% depth | **31–35%** | **unverifiable** — no source in corpus |

A claim about "the workspace" is not portable between these without saying which.

---

## Definitions, with their derivations

### 1. Workspace band — Qwen3.5-27B
`ghost-dimensions/main.tex:106-112`. **Derived**, and reproducible from the paper:

- rank sweep → 49 low-rank layers, L0–L48, criterion `k/d < 0.4`
- future-window gate → 45 of 63 pass, L11–13, L15–17, L24–L62
  (L14 and L18–23 fail criterion (b): J-lens does not exceed raw logit-lens signal)
- **band = low-rank ∩ gate-valid = 31 layers: L11–13, L15–17, L24–L48**

Arithmetic checks: 3 + 3 + 25 = 31, matching the stated count.

**Caveat carried in the paper's own Limitation 1:** the J-lens used was fitted on the
**base** model, not the distill. The gate half of this intersection inherits that
mismatch.

### 2. Ignition zone — Qwen3.5-27B
`ghost-dimensions/FINDINGS.md:17`. **L35–L47.** Where PC2 transitions ghost → workspace;
ignition point L45 (70% depth, k/d = 0.295). A **subset** of the band, not a synonym for it.

### 3. Workspace-band layers — Qwen3-8B
`cache-tracing/main.tex:348`. **L9–L18**, asserted without derivation in that passage,
attached to a logit-lens tail-smear measurement (2.3–2.8× more tail smear at these
layers). Different model, different estimator, different depth range from (1).

Not necessarily wrong — an 8B band should differ — but it is **not shown to be the same
kind of object**, and the shared phrase implies it is.

### 4. Workspace peak — GLM-5.2
"31–35% depth", n=1, Colibri extraction. **No source in either repo.** Currently in
`key_numbers_unverified.md`; the memory gate reports it UNRESOLVED. **Do not cite until
a primary exists.**

---

## Consumers, and whether their usage matches

| consumer | layer set | consistent? |
|---|---|---|
| attention probe — "band layers" | L35, 39, 43, 47 | **in band ✓** — but these are exactly the **ignition zone**. Call it that. |
| attention probe — "control layers (off-band)" | L7, **L15**, L55, L63 | **✗ L15 IS IN THE BAND** (L15–17). 1 of 4 controls contaminated. |
| decision-state — probe layers | L3, L7, L11, L15 | mixed: L11 and L15 are in-band, L3 and L7 are not. Paper describes them as "full-attention layers", which is true and is a **different** criterion — no band claim made. **OK as written.** |
| cache-tracing — band layers | L9–18 (8B) | self-consistent within its own paper; not comparable to (1). |

---

## The one live defect

**`deception_attention_probe.md:51` — `Control layers: L7, L15, L55, L63 (off-band)`.**

L15 is a workspace-band layer (L15–17 in the gate-valid ∩ low-rank intersection).

**Dependent statistic:** the band-specificity interaction test — the thing that makes
`sparsity AUROC 0.612, p = 0.046` a *band-specific* survivor rather than a generic one.
With one quarter of the off-band arm actually in-band, the contrast is band vs
(mostly-off-band). Bias direction is not obvious and **must not be asserted without
recomputation** — the intuition that contamination attenuates an interaction is only a
prior, not a result.

**Fix is re-analysis, not re-collection.** The per-layer data exists; only the grouping
changes. Options: drop L15, or replace it with a clean off-band full-attention layer.

Computed candidates at `L ≡ 3 (mod 4)` outside L11–13 / L15–17 / L24–48:
**L3, L7, L19, L23, L51, L55, L59, L63.**

**L23 is the best replacement.** It is the layer immediately before the band opens
(band starts at L24), so it is maximally depth-matched to the in-band arm while being
*definitionally* off-band — ghost-dimensions states L18–L23 fail gate criterion (b)
because the J-lens does not exceed raw logit-lens signal there. That makes it off-band
for a **stated mechanistic reason** rather than merely by distance, which is what a
control should be. L19 is the same argument, one step further out.

*(I first enumerated this list by hand and omitted L23 — the strongest option. Caught by
computing the set instead of writing it. Same lesson as everything else this week: run
it, don't recall it.)*

If the effect survives, **rename the claim: "ignition-zone-specific", not
"band-specific".** That is sharper — the effect localises to 13 layers, not 31.

---

## Rules for anything downstream

1. **Name the model with the band.** "The workspace band" alone is ambiguous across four
   objects.
2. **Cite the derivation, not the range.** `L24–48` is an output; `low-rank ∩ gate-valid`
   is the definition. If the criteria change, the range does.
3. **Check membership before calling a layer a control.** `L in band` is one line and
   would have caught L15.
4. **Ignition zone ≠ workspace band.** Subset. Say which you mean.
5. **Depth-normalise before comparing across models.** L35 is 55% of a 64-layer model and
   would be 97% of a 36-layer one.

---

## Still open in Phase 0

### `d_eff` — EXACT DEFINITION, resolved 2026-08-20

Source: `gwt-response/r3_band_measurement.py`, `compute_effective_dimensionality()`.
Marked "exact formula still pending" in this file since 2026-08-12. It is:

**`d_eff(L) = (sum_i lambda_i)^2 / sum_i lambda_i^2`** — the **participation ratio** of the
eigenvalue spectrum. Not entropy-based, not a variance-threshold count.

Full specification, because every one of these is a place a later run can silently differ:

| element | value |
|---|---|
| tensor | residual stream (`hidden_states`), confirming the 08-12 finding |
| positions | **last token position only** (per docstring) |
| sample axis | **across prompts**, not across tokens |
| centering | `H - H.mean(dim=0)` — per-feature, across prompts |
| covariance | `H.T @ H / (n-1)` — sample covariance, Bessel-corrected, d_model x d_model |
| eigensolver | `torch.linalg.eigvalsh` (symmetric) |
| eigenvalue floor | discard `lambda <= 1e-8` |
| minimum n | layer skipped if fewer than **5** prompts |
| degenerate case | `d_eff = 0.0` if no eigenvalue survives the floor |

**So `d_eff(L)` measures how many dimensions the residual stream uses to separate DIFFERENT
PROMPTS at layer L.** It is a property of a prompt set, not of a single forward pass. Any
comparison across studies must hold the prompt set fixed or it is comparing stimulus diversity.

**A second, abandoned estimator is still in the code.** One line above the return:

```python
p = eigenvalues / total   # computed, never used
d = float((total ** 2 / (eigenvalues ** 2).sum()).item())
```

`p` is the normalized spectrum — the input for the **entropy** form, `exp(-sum p log p)`, the
other standard `d_eff`. Someone started that version, switched to participation ratio, and left
the evidence. Both estimators are defensible and they do **not** agree numerically. Any depth
claim citing "d_eff" without naming which one is underspecified. This is `ESTIMATOR_MISMATCH`
sitting in our own reference implementation.

**Consequence for the GLM Lc question:** the formula needs n>=5 samples at a fixed layer. Ten
Lc `.bin` dumps exist (5 real_*, 5 fake_*), so it is *computable*. But Lc is the MLA compressed
latent, not the residual stream, so the output is the participation ratio of a different object.
Confirms the 08-12 rescope: **characterization, not replication.** Report it as
`d_eff_PR(Lc)` and never as `d_eff`.

### `stable_rank` — NAME COLLISION WITH THE LITERATURE, found 2026-08-20

Three implementations, all agreeing with each other:
`geometric_portrait/vera_portrait.py:219`, `vera_portrait_full.py:291`,
`gwt-response/base_vs_instruct.py:131`.

```python
stable_rank = float(np.sum(S)**2 / np.sum(S**2))
```

**That is not stable rank.** The standard definition is
`||A||_F^2 / ||A||_2^2 = sum(S^2) / max(S)^2`. What we compute is the **participation ratio of
the singular value spectrum** — a different quantity that happens to share the name.

Verified numerically 2026-08-20:

| spectrum | ours | literature | ratio |
|---|---|---|---|
| uniform [1,1,1,1] | 4.0000 | 4.0000 | 1.00x |
| mild decay [4,3,2,1] | 3.3333 | 1.8750 | 1.78x |
| spiked [10,1,1] | 1.4118 | 1.0200 | 1.38x |
| power law (1/i^1.5, n=20) | **3.9237** | **1.2009** | **3.27x** |

They coincide **only on a flat spectrum.** Neural singular value spectra are power-law, which
is the row where the divergence is worst.

**Three consequences.**

1. Any reader who knows the literature reads our number as Frobenius-over-spectral and forms a
   wrong model. Our value is systematically *larger*.
2. Any comparison to a published stable rank from outside this lab is invalid.
3. **`stable_rank` and `d_eff` are THE SAME ESTIMATOR** — `(sum x)^2 / sum(x^2)` — applied to
   different objects (singular values vs covariance eigenvalues). A result reporting both as
   agreeing is **not convergent evidence**. It is one functional form counted twice.

Consequence 3 is the load-bearing one and it is `DOUBLE_COUNTED_EXPERIMENT` in spirit if any
paper leans on the agreement.

---

### UPDATE 2026-08-24 — there is a THIRD definition, and a rule that supersedes chasing them

I treated this collision as closed on 08-20. It was not closed; I had only stopped looking. On
08-24 CC described the v8 production stack's primary metrics as *"projection (dot product),
**stable rank (nuclear/Frobenius)**, and Jaccard"* — and on the strength of "we don't use `d_eff`"
I told them their stack was clear of the sample-size problem. **That reassurance was wrong.**

`‖A‖_* / ‖A‖_F` **squared is the participation ratio.** Verified on Qwen3.5-27B, L21, n=90:

| definition | L21 value | where |
|---|---|---|
| `‖A‖_F² / ‖A‖_2²` | **1.2907** | the literature |
| `(Σs)² / Σs²` | **25.1202** | our `kv-experiments` |
| `‖A‖_* / ‖A‖_F` | **5.0120** | CC's v8 stack |

`5.0120² = 25.1202` exactly. Three names, three values, one label, no error message. CC's planned
sweep for "effective rank" in old calibration code would have missed it entirely, because the
string they would grep for is not there.

**A metric's name is not its definition. The formula is what travels between codebases.**

### The rule that makes the chase unnecessary

Measured on the same draws, L21, B=100 subsamples without replacement, drift m=20 → m=89:

| metric | drift |
|---|---|
| mean(projection onto a fixed unit direction) | **+0.7%** |
| mean \|projection\| | **−0.6%** |
| mean ‖h‖ | **−0.0%** |
| kurtosis (mean over 64 layers) | **+0.2%** |
| **stable rank / `d_eff` / participation ratio** | **+71% to +116%, still climbing** |

> **A statistic that is a per-sample quantity averaged over samples converges fast.
> A statistic that is a property of the sample's spectrum or rank is bounded by `n` and does not.**

Per-sample and safe: projection, norm, kurtosis, dot products, any mean of a per-item score.
Rank-bounded and unsafe: stable rank (all three definitions), `d_eff`, participation ratio,
effective rank, explained-variance ratios — anything eigen- or singular-value derived.

This classifies a new metric in seconds without running anything, and it is more reliable than
grepping for a variable name. It also explains the collision and the instability with one cause:
**people name a metric after what they think it means, and the formula is what actually
propagates.**

### Downstream: relative baselines do not fix it

CC's mitigation was to z-score against a running (Welford) baseline instead of an absolute
threshold. Correct for the absolute-magnitude problem; **insufficient for a monotone drift**,
because the baseline drifts too, just later. Simulated over 30 randomised streams, a
rank-bounded metric sits at **mean z ≈ +1.5σ at every stage** of the stream and **never trips a
2σ alarm** — it silently consumes the detection headroom asymmetrically. (Magnitude inflated by
the simulation's smoothness; direction and persistence are structural.)

**The fix is an N-matched baseline, or detrending the N-dependence before z-scoring.** A running
baseline alone inherits the problem with an extra layer of indirection hiding it.

**Recommended fix, not yet applied:** rename ours to `participation_ratio_sv` everywhere and
state the formula inline at every reporting site. Do NOT silently swap in the literature
definition — that changes every published number. Flag, don't overwrite.

**Live claim affected:** `Confab delta stable rank d=2.35` is in MEMORY.md's key numbers under
UNVERIFIABLE ("CC synthesis verified — needs a real path"). It now needs the estimator named as
well as the source located.

### `eff_rank` — FOUND 2026-08-20. My earlier "no implementation found" was a BROKEN SEARCH.

The prior entry here said no implementation existed. That was wrong: the Starship grep used
`--include=*.py` unquoted over ssh, zsh rejected it with "no matches found", and **the search
never executed.** A failed search and an empty search return the same nothing. Now pinned as a
belief; positive control every query.

`~/lab/projects/residual_kv_alignment.py:124-131`:

```python
k_eff_rank = int(np.sum(S_k > S_k[0] * 0.01))
```

**Threshold count**: how many singular values exceed 1% of the largest. Also appears in
`graph_encoder.py`, `user_model_probe.py`, `emotion_geometry_bridge.py`,
`09_sycophancy_detection.py`, `10_societies_of_thought.py`, `layer_sweep.py`,
`multiturn_misalignment_abliterated.py`.

### THREE ESTIMATORS, THREE NUMBERS, ONE PROSE WORD

| name in code | formula | object | family |
|---|---|---|---|
| `d_eff` | `(sum L)^2 / sum(L^2)` | covariance eigenvalues | participation ratio |
| `stable_rank` | `(sum S)^2 / sum(S^2)` | singular values | participation ratio (**misnamed**) |
| `eff_rank` | `count(S > 0.01 * S_max)` | singular values | threshold count |

All three are described in prose as effective rank / effective dimensionality. They are **not
interchangeable and do not agree.** A threshold count is discrete and cliff-edged at the
cutoff; participation ratios are continuous and weight the whole spectrum. Any depth claim
using an "effective rank" must name which of the three, and any claim citing two of them as
agreeing is at risk of `DOUBLE_COUNTED_EXPERIMENT`.

### `transport cosine` — found in `starship:~/dense_control.py`

Reported per-fit as `transport cosine = %.4f (n=%d)` (L424). The script is the Track 6 dense
positive control, and **its stated reason for existing is the exact failure mode I pinned as a
belief today**:

> "A fitting pipeline that cannot fit anything produces exactly the observed signature:
> everything low, nothing differentiating. conditioned ~= random is equally consistent with
> 'conditioning does not help' and 'nothing fits at all.'"

I built a positive control for a fitting pipeline and never once built one for a grep.

**TASK #84 — RESOLVED 2026-08-20. And my "lead" here was a misreading; correcting it.**

I previously wrote that `dense_control.py`'s header attributed the 0.7 to Gurnee, and filed
that as evidence the number had a real source. **It does not say that.** The header is quoting
the belief the script was built to FALSIFY, and states two lines later: *"Gurnee's 0.7+ is very
likely not that quantity."* I lifted a fragment out of context and recorded a strawman as a
source — the same move that produced the original error, committed while investigating it.

**Verdict from the primary:** Gurnee et al. 2026, arXiv:2607.15495, *Verbalizable
Representations Form a Global Workspace in Language Models*. Full text pulled and grepped
(377,997 chars; independently matched against two prior reads):

| term | occurrences in the paper |
|---|---|
| `transport cosine` | **0** |
| `fidelity` | **0** |
| `dense` | **0** |
| `MoE` | **0** |

Exactly one `0.7` exists, §A.6 "Inter-lens similarity": J-lens-vs-logit-lens **vector** cosine
climbing to ~0.7 over ~20 layers, in residual-stream space, for the same vocabulary token —
and immediately qualified ("top-1 agreement stays low, KL remains near its peak"). Our
transport cosine is between two **softmax distributions** over a 152k vocabulary. Different
object, different space, no shared scale.

**The inversion, which matters more than the missing citation.** The paper DOES report our
metric family (output-distribution agreement, Fig. 55) and concludes:

> "The J-lens is the worst, with higher KL than even the uncorrected logit lens... **We regard
> this as a feature rather than a defect.**"

The work we cited as our benchmark argues a J-lens *should* score badly on the metric we used.
So "4-9% = catastrophic failure relative to dense models" collapses **regardless of
provenance.**

**Two fabrications, not one.** The paper's models are Claude Sonnet/Haiku/Opus 4.5 and Opus
4.6, and it never states their architecture. ">0.7 **on dense models**" invents a number *and*
an architecture class.

**Origin: UNRESOLVED.** The real document is `AGNI_REVIEW_MOE_JLENS.md:83`, but `git log -S`
shows the claim present in the initial commit, so it predates traceable history. A tempting
drift path exists (the paper reports 70% coordinate-swap success on Sonnet 4.5 / Opus 4.5) —
**zero evidence for it; do not cite it.**

**Corpus state:** the 2026-08-16 sweep already cleared all 21 sites in `lyra-moe-fix`, prereg
correctly annotated rather than rewritten. This registry entry was the last live one.

- **Estimator registry** for depth claims other than layer sets (`d_eff`, stable rank,
  eff_rank, transport cosine) — same treatment: which object, which estimator, derived or
  asserted.
- **route_trace VERIFIED non-empty 2026-08-20 — unblocks Phase 2 (#71).**
  Path in this file was one directory short; actual location is
  `mth:~/lab/projects/frontier-workspace-probe/results/route_traces/route_trace.log`
  (2.1 MB, 2026-08-14 01:00). I checked the parent dir first, found nothing, and nearly
  recorded it as absent.
  Format is POSITIONAL, not keyed: `col1 col2 layer expert:weight x8`.
  - col3 = layer, spans **3..77 with no gaps** = 75 MoE layers, matching
    `first_k_dense_replace=3`. Registry's claim confirmed.
  - col1 0..689 (690 tokens), col2 0..35.
  - **22,359 lines carry a full top-8; exactly one carries 5** — the final record is
    truncated mid-write (`190:0.` then EOF). Drop the last line; the rest is intact.
  - Count correction: this file says "six 11 MB `.bin` Lc dumps". There are **ten**
    (5 real_*, 5 fake_*), 10.8-11.2 MB each.

- **Is `d_eff` computable on GLM Lc?** Now answerable — six 11 MB `.bin` Lc dumps exist at
  `~/lab/projects/frontier-workspace-probe/results/`, plus `route_trace.log` with top-8
  routing across all 75 MoE layers (L3–77, matching `first_k_dense_replace=3`).
### 0.612 RECOMPUTE — RESULT, 2026-08-20 (pre-specified, Agni-gated WARN 0.94)

Design: `experiment-designs/phase0_0612_recompute.md`. Spec locked before execution; the
explicit non-goal was searching for a spec that yields 0.612.

**Verdict: the published pair (0.612, p=0.046) is NOT recoverable from the shipped artifact.**

| quantity | value |
|---|---|
| pooled-across-layers AUROC | **0.3897**, p = **0.0197** |
| mean-of-per-layer AUROC | 0.4120 |
| published | 0.612, p = 0.046 |

**Direction convention is the unstated estimator here.** 1 - 0.3897 = **0.6103**, which is the
2026-08-14 figure exactly. This run reproduces that recompute to the decimal (p 0.0197 vs
0.0196; L15 0.0190 vs 0.0189; band-minus-control gaps identical in magnitude, opposite in
sign). The two differ **only** by whether AUROC is defined as P(confab > grounded) or
P(grounded > confab), and **nothing in the published claim states which.**

So the magnitude lands within 0.0017 of published under one convention — but **p is off by
2.3x** (0.0197 vs 0.046). An AUROC that matches with a p that does not means a different n,
test, or pooling. The pair does not reproduce.

**Substantive direction, stated so it stops being ambiguous:** confabulated responses have
**LOWER** attention sparsity than grounded ones in the ignition zone. Per-layer, all four band
layers agree (L35 0.4007, L39 0.4237, L43 0.4115, L47 0.4121, in the confab-higher convention).

**The clean contrast set strengthens the result**, as predicted: band-minus-control widens from
0.0911 to **0.1623** when L15 is moved out of the controls and back into the band where the
ghost-dimensions derivation puts it.

### LIVE DOWNSTREAM CONSEQUENCE — `deception_attention_probe.md`

That design uses 0.612 as a **pre-registered success threshold**, twice:

> "If deception sparsity AUROC > 0.612 (the confab band-specific survivor)..." (L105)
> "3. Deception sparsity AUROC > 0.612 (stronger than confab's band-specific signal)." (L117)

Three problems, all inherited:
1. The threshold is anchored to a pair that does not reproduce.
2. **"> 0.612" is undefined without the direction convention.** The same result is 0.390 in
   the other convention, where "greater than" points the opposite way.
3. It is labeled "band-specific" when the measurement is the **ignition zone** (L35-47), a
   subset of the band (L24-48).

**That design must not be run against this threshold as written.** Restate as: signed effect
relative to the recomputed value, with the convention named inline, or drop the numeric
threshold and pre-register a direction instead.

- **Recompute the 0.612 interaction** with a clean contrast set. **DONE 2026-08-20, above.**

---

# Phase 0 recompute — results (2026-08-14)

Data: `starship:~/oracle-experiments/results/a1_benchmark/attention_phase2.json`
(17.5 MB, n=200: 50 grounded + 150 confab across temporal/specificity/attribution).
FWL on `prompt_len` + `gen_len`. AUROC = confab vs grounded, Mann-Whitney, tie-corrected.

## 1. The published 0.612 does NOT reproduce exactly

| FWL spec | band AUROC | p |
|---|---|---|
| none | 0.6423 | 0.0026 |
| prompt_len only | 0.5079 | 0.8678 |
| gen_len only | 0.7704 | <0.0001 |
| **prompt + gen** | **0.6103** | **0.0196** |

Closest is 0.6103 vs published 0.612 (Δ=0.0017), but p differs 2x (0.0196 vs 0.046).
Mean-of-per-layer-AUROCs gives 0.588, so that is not the pooling either. **The exact
published pair is not recoverable from the shipped artifact under any spec tried.**
Treat everything below as a *within-pipeline relative* result, which is what it is.

## 2. L15 behaves like a band layer, because it is one

`mean_sparsity`, per layer:

| layer | AUROC | p | |
|---|---|---|---|
| **L15** | **0.6109** | **0.0189** | **listed as an off-band control** |
| band pooled | 0.6103 | 0.0196 | |
| L7 | 0.4531 | 0.3206 | clean control |
| L55 | 0.4319 | 0.1494 | clean control |
| L63 | 0.4941 | 0.9012 | clean control |

**L15's AUROC matches the band's pooled AUROC to three decimals.** The three genuine
off-band controls all fall below chance and are non-significant. This is independent
confirmation of the ghost-dimensions band definition from data that never referenced it.

## 3. Removing L15 nearly doubles the band-specificity gap

| feature | contrast | band − control |
|---|---|---|
| sparsity | as published | +0.0911 |
| sparsity | **L15 dropped** | **+0.1623** |
| sink_fraction | as published | +0.1340 |
| sink_fraction | **L15 dropped** | **+0.2100** |

The prior that contamination attenuates an interaction was correct — and was still worth
not asserting before measuring.

## 4. Sparsity is NOT the sole FWL survivor — sink_fraction is stronger

All six pooled features, band, FWL(prompt+gen), Holm across the family of 6:

| feature | band | raw p | Holm | control | ctrl p |
|---|---|---|---|---|---|
| **mean_sink_fraction** | **0.6461** | **0.0020** | **0.0119 ✓** | 0.5121 | 0.7974 |
| mean_sparsity | 0.6103 | 0.0196 | 0.0982 ✗ | 0.5192 | 0.6845 |
| mean_entropy | 0.5513 | 0.2774 | 1.0 | 0.5800 | 0.0905 |
| mean_norm_entropy | 0.5419 | 0.3757 | 1.0 | **0.6099** | **0.0201** |
| mean_self_mass | 0.4903 | 0.8368 | 1.0 | 0.5583 | 0.2176 |
| mean_prompt_mass | 0.5097 | 0.8368 | 0.8368 | 0.4417 | 0.2176 |

**Sink fraction is ~10x more significant than sparsity and is the only feature surviving
Holm at m=6.** Sparsity does not survive (0.0982). If the published claim tested this
family, sink_fraction should have been the headline; if it did not, that is a coverage
gap. Either way the "sole survivor" attribution needs revisiting.

`mean_norm_entropy` is significant at *controls* (0.6099, p=0.0201) and not at band —
an anti-band-specific result worth its own look.

## 5. Caveat that cuts against the clean story

Sink fraction's band-minus-control gap is partly driven by controls running
**anti-correlated**, not merely at chance:

| control layer | sink_fraction AUROC | p |
|---|---|---|
| L55 | **0.3931** | **0.0237** — significant, opposite direction |
| ctrl pooled, L15 dropped | 0.4361 | 0.1766 — below chance |

"Band signal vs control noise" and "band signal vs control counter-signal" are different
claims. Only the first is what band-specificity normally asserts. Sink fraction also
shows a monotonic depth gradient within the band (L35 0.5752 → L39 0.5843 → L43 0.6296 →
L47 0.6536), which is interesting on its own and is not what a flat "band effect" predicts.

## What follows

1. Do not cite "sparsity 0.612, sole FWL survivor" further without resolving §1 and §4.
2. Any re-run must replace L15. Only L7/15/55/63 were measured, so **L23 requires
   re-collection** — it is the best candidate (immediately pre-band, off-band for the
   stated gate reason).
3. If the effect is reported, call it **ignition-zone-specific** (L35–47), not
   band-specific (L11–13, L15–17, L24–48). The probe sampled the zone, not the band.
4. The depth gradient and the anti-correlated controls both deserve their own analysis
   before any mechanistic story is attached.

## 6. The analysis code does not exist — this is UNVERIFIABLE, not merely unreproduced

Searched Starship, where the data lives:

- no `.py` under `~/oracle-experiments` matching attn / attention / schema / phase2 / band
- no file containing `attention_phase2`, `control_layers`, or `band_layers`
- `~/oracle-experiments/results/a1_benchmark/` contains **six data files and zero code**

Task #67 ("FWL-residualize Phase 2 attention data") is marked completed. The
residualization ran somewhere and the code is not on the machine holding the data.

**This upgrades §1.** "0.612 does not reproduce under the specs I tried" understates it:
there is no derivation to check against. Same class the build gate flags for a PDF with
no `.tex` — an artifact whose provenance cannot be inspected. Ours, this time.

**Consequence for the fix.** An analysis with no code cannot be patched, only redone. The
L15 correction, the sink_fraction finding, and the anti-correlated controls are all
computed on MY pipeline, which is not the published one and cannot be reconciled with it.
Re-collection is therefore not the expensive option — it is the only one that produces a
claim anybody can audit.

**Scope of a clean rerun:** 200 prompts x 8 layers, harness already written, band
L35/39/43/47 unchanged, controls L7/L23/L55/L63 (L15 out, L23 in), all six features
reported with Holm across the family, analysis script committed alongside the data this
time.

---

# Phase 0, part 2 — estimator registry (2026-08-15)

**`stable_rank` names two different quantities in this corpus, and they differ by
17-41x on the spectra we actually study.**

## The split

| formula | what it is | files |
|---|---|---|
| `(Σs)² / Σs²` | **participation ratio** | `lyra-s-research-/gwt-response/`: `base_vs_instruct.py`, `causal_injection.py`, `mine5_three_model.py`, `substrate_independence.py` |
| `Σs² / s_max²` | **textbook stable rank** (‖A‖_F²/‖A‖₂²) | `Project-Oracle/experiments/matched_burn.py`, `published-research/decision-state-paper/code/matched_burn.py` |

Split falls along project lines: the GWT-response line uses one, the
decision-state / Oracle line uses the other. Both call the output `stable_rank`.

## How far apart, on realistic spectra

| spectrum | participation | textbook | ratio |
|---|---|---|---|
| power-law α=0.5 | 281.65 | 6.82 | **41x** |
| power-law α=1.0 | 28.28 | 1.64 | **17x** |
| power-law α=1.5 | 5.30 | 1.20 | 4.4x |
| power-law α=2.0 | 2.49 | 1.08 | 2.3x |
| one dominant PC | 35.52 | 1.05 | **34x** |
| near-flat | 511.95 | 481.81 | 1.06x |

They converge **only** when the spectrum is flat. Residual-stream spectra are power-law —
that is the premise of the low-rank workspace claim — so the disagreement is maximal
exactly where we work.

**Each study is internally consistent.** Both measures are monotone in spread, so any
single paper using either one throughout is fine. The failure is **cross-study
comparison**, and there is no conversion without the full spectrum, which published
summaries do not ship.

## Entropy: mixed log bases, and a compounding path

- `torch.log` (nats): `Project-Oracle/experiments/{accumulation_controls,decision_moment,divergence_deep,emotion_accumulation}.py`, `gwt-response/causal_injection.py`
- `torch.log2` (bits): `KV-Cache-Experiments/code/gpu_utils.py:164` — a shared *utility* module, which is the dangerous place for it

Then:
- `Project-Oracle/oracle_harness/eye/geometry.py:171` → `eff_rank = math.exp(spectral_entropy)`
- `Project-Oracle/experiments/cognitive_immune_system.py:168` → `eff_rank = torch.exp(entropy)`

`exp(H)` is correct only for **nats**. For H = 3.0:

| interpretation | value |
|---|---|
| exp(H), H in nats — correct | **20.09** |
| exp(H) applied to a bits H — wrong | **75.80** |
| 2**H, correct for bits | **8.00** |

A bits entropy fed to `exp()` is ~9.5x wrong. Whether that path is live depends on
whether anything imports `gpu_utils.entropy` into an `eff_rank` computation — **not yet
traced. Do not assert it happens; do not assume it doesn't.**

## Consequences

1. **`MEMORY.md` key number "Confab delta stable rank d=2.35, t=9.111"** does not say which
   estimator produced it. It is already in `key_numbers_unverified.md`; this is why.
2. **Phase 1 ("estimator agreement on DENSE models") is mis-scoped as written.** It plans to
   test whether estimators agree numerically. They cannot, because they are not the same
   function. Phase 1 should first fix the definitions, then measure agreement of the
   *fixed* estimators across models.
3. **Any cross-paper stable-rank comparison in the corpus is invalid** unless both sides
   are confirmed to use the same formula. That includes anything comparing GWT-response
   results against decision-state results.
4. `eff_rank = exp(spectral_entropy)` is a **monotone transform of entropy**, not an
   independent feature. Anywhere both are reported as separate predictors, that is one
   statistic counted twice.

## Rules

1. Never write `stable_rank` unqualified. Use `stable_rank_pr` (participation ratio) or
   `stable_rank_sr` (Σs²/s_max²).
2. State the log base wherever entropy is computed, and use `2**H` for bits.
3. `eff_rank` must carry the base of the entropy it exponentiates.
4. Before any cross-paper comparison of a rank measure, check the formula in **both**
   sources. The name is not sufficient.
