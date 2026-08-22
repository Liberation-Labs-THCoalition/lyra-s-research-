# The MoE Geometry Campaign

**Opened:** 2026-08-11 (Lyra, Thomas)
**Status:** Campaign spec — individual phases gate through Agni separately
**Prior art:** CC Experiment 51 (Apr–May 2026), `Project-Oracle/docs/MOE_GEOMETRY_EXPERIMENT.md`, `LORA_06_STRAIN_DETECTOR.md`

---

## Why this campaign exists

Every workspace, opacity, and detection result we have was developed on
**dense-FFN models** (Qwen3-8B: 36L dense; Qwen3.5-27B: 64L, hybrid *attention*
but dense FFN — verified 2026-08-11 from `text_config`, no `num_experts`).

Every model we would actually deploy on, or measure at frontier scale, is
**Mixture-of-Experts**: GLM-5.2 (256 experts, 8 active), DeepSeek V4, Kimi K3,
Inkling, Qwen3.6-A3B.

The first time we pointed the existing instrument at an MoE, it produced a
confound at layer 4. That is the campaign.

---

## What triggered it (2026-08-11 findings)

Running the frontier probe's 7 surviving dumps (GLM-5.2, 744B, MLA latent `Lc`,
512-dim, 78 layers) surfaced three interlocking problems.

### T1 — The MoE routing signal enters at exactly the architectural boundary

| layer | stable rank | PC1 var |
|---|---|---|
| L2 | 1.80 | 55.6% |
| **L3** | **1.35** | **74.1%** |
| **L4** | **5.72** | **17.5%** |
| L8 | 8.28 | 12.1% ← global peak (stable rank) |

`first_k_dense_replace = 3`. Layers 0–2 dense, 3+ MoE. `Lc[3]` is produced by
layer 3 *attention*, before any expert FFN has run. `Lc[4]` is the first latent
computed from a residual stream that has passed through an expert FFN.

Stable rank measures token-to-token spread. On a dense model every token sees the
same FFN. On an MoE, different tokens route to different experts — routing
heterogeneity injects token-to-token variance directly.

Consistent across all 7 files (L3 = 1.291–1.383, including the 67-token one).

### T2 — Three estimators, three depth profiles

| estimator | object | global peak |
|---|---|---|
| `d_eff` (convergence paper) | residual stream (Qwen) | 31–33% relative depth |
| stable rank `Σs²/s_max²` | GLM `Lc` | **L8 = 10.4%** |
| MP-edge structure count | GLM `Lc` | **L2 = 2.6%** |

L2 is the clearest case: stable rank **1.80** (collapsed) but MP count **12**
(highly structured). Not contradictory — one direction carries 55% of variance
plus eleven more above noise. Stable rank is top-dominated; MP count is a
structure count. They measure different things and disagree about where the peak is.

`d_eff` magnitudes (20.5–22.3 on 8B, 38.0–39.7 on 27B) are far larger than either
of ours, so it is a *third* estimator on a *different object*.

**Consequence:** "does the workspace replicate at 744B?" is currently unanswerable.
We would be comparing d_eff-on-residual-stream against stable-rank-on-MLA-latent
and calling the mismatch a result.

### T3 — CC already found the shape of this in May

From Experiment 51 (`LORA_06_STRAIN_DETECTOR.md`):

- **Marchenko-Pastur profile** — honest-common: 10.1 SVs above the random-matrix
  edge; all other conditions ~5.5.
- **Middle-band SVs** — the *top* SVs are frequency-dominated; cognitive mode
  lives in the middle band. Honest-common 19.4, rare/confab/deceptive 24.0–24.6.
- **ICA IC #9** separated confab (+0.90) from rare-true (−0.89) at *identical*
  token frequency.
- **Expert familiarity sensitivity** — "Expert 45 activates as a delimiter expert
  normally, but fires with 50% higher weight on unfamiliar subword tokens. The
  router gives it more influence when the model encounters content it has limited
  training data for."
- **Reframe** — "Not 'is this confabulation?' — that turned out to be confounded
  by token frequency." CC moved to *representational strain* / grounding
  confidence.

The familiarity finding matters most: our real-vs-fabricated entity contrast **is**
a familiarity contrast. Routing may not be a confound to residualize out — it may
be the mechanism the contrast runs through. That promotes `ROUTE_TRACE` from
control to primary measurement.

### T4 — A naming collision in our own corpus (hygiene, blocking)

The convergence paper uses two distinct structures:
- **workspace band peak**: L11–L12 on 8B, **L19–L23 on 27B** (30–36% depth)
- **ignition zone**: **L35–L47** on 27B (55–73% depth)

The Phase 2 attention probe labeled L35/39/43/47 "the workspace band" and used
L7/L15/L55/L63 as "off-band controls" — but **L15 is closer to the real workspace
peak than L35 is.** One control was not off-band.

The FWL-surviving sparsity result (0.612, p=0.046) stands as a finding but must be
relabeled: it is specific to the **ignition zone**, not the workspace band.

Same failure class as the FWL mislabel — a label carried across contexts without
being checked against its source.

---

---

## Findings log

### 2026-08-21 — PHASE 1 (PARTIAL) — RESULT: the estimators disagree on a DENSE model, and the aggregate correlation hid it

Design: `experiment-designs/phase1_estimator_agreement_dense.md`, gated WARN 0.72, zero
unverified findings. Data: `starship:~/lab/kv-experiments/base_vs_instruct_20260714_190308.json`
(Qwen3-8B-Base vs Qwen3-8B, both **dense**, 25 records/arm, layers 12/15/18/21/27/30/33).
No compute. **Partial: 2 of 5 estimators** — participation-ratio (our misnamed `stable_rank`)
and `spectral_entropy`. MP-edge and middle-band need full spectra we do not have.

**Headline, which is wrong on its own:** both estimators peak at L33 in both arms;
Spearman +1.0000 (base) / +0.9643 (instruct); **Pearson +0.9998 / +0.9999**.

**That correlation is an artifact of the depth gradient.** Participation ratio runs 1.87 -> 11.82
and entropy 0.11 -> 2.29, both jumping ~5x between L21 and L27. Any two monotonic functions of
the same dominant gradient correlate at 0.999. Flagged by `RELIABILITY_TOO_CLEAN` (kill #52,
Penumbra, 2026-08-16) — which the gate cited against this very design *before* the run.

**Disaggregated by band:**

| arm | scope | rho | argmax PR | argmax SE | |
|---|---|---|---|---|---|
| base | all 7 | +1.0000 | L33 | L33 | agree |
| base | shallow 12-21 | +1.0000 | L21 | L21 | agree |
| base | deep 27-33 | +1.0000 | L33 | L33 | agree |
| instruct | all 7 | +0.9643 | L33 | L33 | agree |
| **instruct** | **shallow 12-21** | **+0.8000 (p=0.200)** | **L21** | **L15** | **DISAGREE** |
| instruct | deep 27-33 | +1.0000 | L33 | L33 | agree |

**Three findings.**

1. **The disagreement is NOT MoE-specific.** It appears on a dense instruct model, inside the
   workspace band (12-21) — the exact band workspace claims depend on. The campaign's premise
   that estimator disagreement implicates MoE architecture is **not supported** by this control.
2. **Aggregate correlation conceals band-level disagreement.** r=0.9999 across a wide depth
   range is compatible with the two estimators giving *different answers* to "where is the peak"
   within a band. Any existing claim resting on estimator agreement needs checking at the band
   scale it actually uses.
3. **Agreement is model-state-dependent.** Base agrees at every scale; instruct does not. Same
   architecture, same layers, same estimators — RLHF changed whether the estimators concur.

**Strength, stated honestly:** the mechanism (aggregation hides disagreement) is strong and
general. The specific instance is **weak** — n=4 layers, p=0.200. But the argmax disagreement is
*categorical*, not marginal: L21 and L15 are different answers to the question, not a small
difference in the same answer.

**Consequence for Phase 2:** its plan to "correlate against each estimator from Phase 1" must
specify **at what band scale**, or it will reproduce this artifact. Correlating whole-depth
profiles will return ~0.99 regardless of whether the estimators mean the same thing.

**Still open in Phase 1:** the other three estimators (`d_eff`, MP-edge count, middle-band count)
need raw spectra. Not in this artifact.



### 2026-08-12 — Phase 0 partial: `d_eff` is a residual-stream measure

Established from `~/lab/kv-experiments/layer_map_preregistration.json`
(2026-06-22, lead: Lyra):

> `"measurement_layers": [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]`
> `"note": "These are all 16 full-attention layers in the 64-layer hybrid model.
> Linear-attention layers (GatedDeltaNet) at all other positions cannot be
> measured with v_proj hooks in the same way."`

The convergence paper's substrate table reports `d_eff` at L13, L14, L16, L17,
L18, L20, L21, L22 — all linear-attention layers. Those cannot be V-projection
measurements. The section confirms it in its own caveat ("the residual stream is
structurally continuous across the boundary by construction").

**`d_eff` = effective dimensionality of the residual stream, per layer.**
(Exact formula still pending — search in progress.)

**Consequence for Phase 4, decided by architecture not preference:** our GLM
measurement is on the MLA compressed latent `Lc`. Colibri's `lc_dump.h` dumps
`Lc` only — there is no residual-stream path. So the frontier probe **cannot be
a replication test** of the 31–33% claim. It is a **characterization** of MLA-latent
depth structure in its own right. Phase 4 is rescoped accordingly.

Adding a residual-stream dump to Colibri is possible in principle (a second hook
after each layer's output) but is new C work and not currently scoped.

### 2026-08-12 — Convergence paper: substrate section architectural error

`main.tex` Experimental Setup correctly states 16 full-attention layers at
{3,7,11,…,63} (matches `full_attention_interval = 4` in config).
`substrate_independence_section.tex` contradicts it, describing a contiguous
block with "the architectural boundary at L16," and mislabels 4 of 11 rows
(L13/L14 as Full-Attention; L19/L23 as GatedDeltaNet).

| section claim | status |
|---|---|
| 1. No discontinuity at L15→L16 (+1.1) | Survives — that transition IS genuinely full→linear. Framing: "a" boundary, not "the"; 4 of 5 transitions in the window untested. |
| 2. Peak at L21 in a linear layer | Survives. "Five layers into the GatedDeltaNet regime" is false — L19 and L23 are full-attention. |
| 3. Regime means 14.9 vs 27.5 | **Does not survive as a substrate claim.** See arithmetic below. |
| 4. Rates 2.35/layer vs 1.05/layer | Same exposure — early-slope vs late-slope on a saturating curve. |
| 5. Cross-scale relative-depth agreement | Survives; needs precision (the 27B band L19–L23 contains both substrates). |

**The arithmetic on claim 3, from the paper's own table — no source data needed.**
Under correct interleaved grouping, the full-attention set is {3,7,…,63}. The
table gives three of its members: L15=33.4, L19=38.0, L23=39.3, averaging ≈37.
For all sixteen to average 14.9, the remaining thirteen would need to average
≈9.8 — including L27–L63, at or past the workspace peak. **14.9 is arithmetically
incompatible with interleaved grouping** and is only consistent with a contiguous
L0–L15 block, where near-zero early layers drag the mean down.

So claims 3 and 4 compare L0–15 against L16–63: a **depth** comparison wearing a
substrate label.

Severity MAJOR, not CRITICAL — the headline (substrate continuity) can stand on
claims 1, 2, 5 plus the section's own caveat that residual-stream continuity is
necessary but not sufficient. Claims 3 and 4 need recomputation under
depth-matched interleaved grouping, or removal.

**Why the pipeline missed it:** `verify-paper` traces numbers to source data; it
has no phase that checks architectural descriptions against model config. Agni
reviews design logic, not config claims. This failure class sits outside both.
Recommend adding a "verify architecture claims against config.json" step to
`verify-paper` regardless of how this resolves — we have three papers describing
hybrid architectures and no check on any of them.

**Adversarial check (2026-08-12):** run refute-first against all four findings.
Verdict **UNDER-CALLED** — none refuted, zero false positives, plus two more
found: (i) the Oracle Loop paragraph claims L35–L47 is "well inside the
GatedDeltaNet regime" (all four probe layers are full-attention; the
recurrent-state inference loses its premise), (ii) the 2.35/1.05 rates do not
reproduce under any reading (actual L0–15 slope 2.14–2.25; the GDN "full range"
slope is *negative*). Also found the error's origin: prereg threat item T3
already asserted "the architectural boundary at layer 16."

**RESOLVED 2026-08-12.** Full 64-layer profile located at
`~/lab/kv-experiments/r3_band_measurement_20260714_170944.json` (27B) and
`_171708.json` (8B). Re-analysed and corrected:

- **New continuity test:** across all 31 substrate crossings, mean
  $|\Delta d_\text{eff}|$ = 2.32 vs 2.37 for same-substrate steps,
  Mann-Whitney p = 0.445. Thirty-one crossings, indistinguishable from ordinary
  layer-to-layer variation — a far denser test than the single fictional boundary.
- **Depth-matched regime means:** 25.24 (full-attn) vs 24.03 (GDN), diff +1.20.
  Neighbour-matched control: +0.81 (sd 3.24) against mean layer-to-layer
  variation of 2.35.
- Claim 4 deleted (doesn't reproduce). 8B peak restated as a *local* maximum with
  bimodality disclosed (global argmax L33). Prereg deviation disclosed (band
  never closed; onsets 7.8%/11.1%). Base-checkpoint provenance disclosed.
  Oracle Loop substrate inference withdrawn, empirical result retained.

Fixed beyond the paper: `arch_boundary=16` in `r3_band_measurement.py` now reads
`layer_types` from config; `substrate_independence.py` docstring and all
`BOUNDARY_LAYER` logic replaced with config-derived grouping; prereg T3 corrected.
All with `.pre-arch-fix.bak` copies. Paper recompiles clean (17pp).

Process fixes: `verify-paper` Phase 0.5 (Architecture Claim Verification) + failure
mode #0; `design-experiment` gotchas 13–15; Agni `KNOWN_KILLS` A1/A2/A3.

---

## Phases

Each phase gates through the full Agni scaffold independently. Ordering is by
dependency, not priority.

### Phase 0 — Definitional cleanup (cheap, BLOCKING)

Nothing downstream is interpretable until this lands.

1. Extract the exact `d_eff` definition from the convergence paper's analysis code
   — participation ratio? entropy-based? On which tensor, at which positions,
   centered or not?
2. Build an **estimator × object registry**: for every published depth claim,
   record (estimator, formula, tensor, positions, centering, model).
3. Determine whether `d_eff` is computable on GLM `Lc` at all, or whether the
   object mismatch is fatal.
4. Fix the workspace-band / ignition-zone naming across the corpus. Audit every
   use of "workspace band."

**Deliverable:** registry doc + corpus naming audit.
**Blocks:** Phases 1, 2, 4.

### Phase 1 — Estimator characterization on DENSE (the missing control)

Before blaming MoE, check whether the estimators disagree on dense models too.

Compute d_eff, stable rank, MP-edge count, spectral entropy, and middle-band count
on data we already have (Qwen3-8B and Qwen3.5-27B, existing cache dumps).

- If they agree on peak location on dense → disagreement is MoE-specific. Real finding.
- If they disagree on dense too → the problem is estimator choice, not architecture,
  and several existing claims need re-examination.

**This is the control I did not run before drawing MoE conclusions.**

**Cost:** low — existing data, no new compute.
**Blocks:** Phase 4.

### Phase 2 — MoE routing measurement

Frontier probe rerun with `ROUTE_TRACE=<path>` per prompt alongside the `Lc` dump.

Per layer compute: distinct experts fired across the token batch, routing entropy,
gate concentration, per-token expert-set overlap.

Then correlate against each estimator from Phase 1.

- Tight correlation → the depth profile is substantially a routing readout. Say so.
- Structure routing does not explain → that residual is the interesting part, now
  isolable.
- Real/fake separation carried *by* routing → CC's expert-familiarity mechanism
  replicates at 744B, and this becomes a mechanism finding rather than a confound
  control.

**Also fixes (already sent to Nexus):**
- 2-line `lc_dump.h` patch to write the full `pos_base + S` window (prefix-cache
  truncation)
- regenerate the 12 fake names so no surname repeats (Kessler ×4;
  "Kessler-Brandt" used as two distinct entities)
- write the manifest

**Cost:** ~15h wall clock on MTH, already planned.

### Phase 3 — Routing-residualized detection (production path)

Does the Oracle Loop's spectral channel need MoE recalibration?

| channel | exposure |
|---|---|
| Ch5 spectral (eff. rank, spectral entropy, top_sv_ratio) | **High** — these *are* spread measures |
| norm_per_token | Moderate — expert output scales differ |
| Ch1 centroid projection | Lower — fixed-direction projection, but routing mean-shift drags the centroid |
| Ch3 linear probes | Retrain per model regardless |

Approach: regress per-layer spectral features on per-layer routing statistics; use
the residual. FWL with a new covariate — same instrument, one more column.

Feeds **task #62 (A2 threshold calibration)**. Not a separate paper unless the
effect is large *and* residualization demonstrably fixes it.

**Depends on:** Phase 2.

### Phase 4 — The cross-architecture claim

Only meaningful after 0–2. With estimator pinned, object matched (or mismatch
explicitly scoped), and routing controlled:

Is the GLM result a **replication test** of the 31–33% workspace claim, or a
**characterization** of MLA-latent depth structure in its own right?

These are different papers. Decide before running, not after seeing the numbers.

---

## Open questions

- Does `d_eff` transfer to MLA latents at all, or is the compressed KV a different
  kind of object? (Phase 0)
- Is the MP-edge estimator salvageable at our aspect ratio (n≈31 tokens, p=512,
  γ≈16.5 — far outside the MP regime)? CC's Exp-51 implementation may differ from
  mine; ask.
- Does CC's middle-band statistic transfer, and is it less n-dependent than stable
  rank?
- Does expert-familiarity sensitivity (Expert 45) replicate at 256 experts / 744B?
- Connection to the J-lens-for-MoE work: `oracle_v2_sections_7_9.md` notes the
  reference J-lens has "no MoE handling, no KV-cache path" and that "whether
  concept readouts survive 256-expert routing has never been measured by anyone."
  GLM-5.2 has exactly 256 experts.

## Non-goals

- Not a fifth submission. Phase 3 lives in the production path unless the data
  earns more.
- Not re-running dense experiments that already replicated.
- Not blocking the hackathon — those models are dense.

## Prior art to re-read before Phase 1

- `Project-Oracle/docs/MOE_GEOMETRY_EXPERIMENT.md` (CC, 2026-04-26) — the
  pre-registration for exactly this question
- `Project-Oracle/docs/LORA_06_STRAIN_DETECTOR.md` (CC, 2026-05-02) — Experiment 51
  findings
- `coalition-backup/specs/emotional_rys_v2_agni_prereview.md` — MoE routing-signal
  discussion; notes MoE layer-to-layer transitions are non-smooth due to discrete
  routing (predicts the jagged profile we observe)
- `coalition-backup/specs/depth_recurrent_moe_agni_review.md` — expert-routing
  divergence claim + the norm-clipped control
- `human-review/archive/ghost-dimensions/PAPER_DRAFT_v4.md:253` — J-lens to MoE,
  "averaged Jacobian may be sufficient if routing variance is low enough"

## Tooling built

- `~/lab/projects/frontier-workspace-probe/lcd0_profile.py` (MTH) — LCD0 reader,
  integrity checks, per-layer spectral profile, rank-ceiling check. Validated
  end-to-end on the 7 partial dumps.
