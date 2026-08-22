# Does the depth profile of a hybrid-attention model comb with the substrate?

**Author:** Lyra · **Date:** 2026-08-22 · **Status:** DESIGN, pre-gate
**Compute:** none. Reads two existing JSON files. No GPU, no model load, no MTH.

---

## The question

Qwen3.5-27B does not have a homogeneous stack. From the **shipped config**
(`models--Qwen--Qwen3.5-27B/.../config.json`, `text_config.layer_types`, read 2026-08-22):
64 layers, `full_attention_interval=4`, `full_attention` at indices {3, 7, 11, …, 63} —
**16 full-attention layers interleaved with 48 linear-attention (GatedDeltaNet) layers, and
31 substrate crossings.**

Every depth-profile method in this campaign — `d_eff`, band onset/offset, "workspace" peak
location — assumes depth is the only thing varying along the stack. On this model it is not.
The question is whether the substrate leaves a signature in the profile.

**This is a structure question, not a detection question.** Nothing here claims a workspace,
a peak, or a mechanism.

## What has actually been established (checked, not assumed)

Correcting my own framing from earlier today: I said "nobody has looked at this model." **That
was false.** Thomas pushed back and the primaries agree with him.

- A **J-lens was successfully fitted** on Qwen3.5-27B: `lens_qwen35-27b.pt`, 3.3 GB, 2026-07-14.
  `source_layers` = all 63 (0–62), `d_model`=5120, 70 prompts. **GatedDeltaNet caused no
  fitting problem.** The known difficulty on this model was memory, not substrate.
- **Eleven** scripts consume that lens (`r3_band_measurement`, `sae_bridge`,
  `double_dissociation`, `substrate_independence`, `temporal_boundary`, `dose_predictor`,
  `sae_feature_mapping`, …). This model is well studied here.
- The contiguous-block error — treating the model as full-attention L0–L15 then GatedDeltaNet
  L16–L63 — **was already found and corrected on 2026-08-12**, in
  `r3_band_measurement.py` and its prereg. The prereg names the consequence: it "propagated
  into the convergence paper's substrate section and produced a depth-confounded regime
  comparison." The corrected code reads `layer_types` from config and computes all 31 crossings.

## The gap this design addresses

**The corrected script has never been run.** `r3_band_measurement.py` and its prereg are
stamped 2026-08-12 (with `.pre-arch-fix.bak` siblings). The only result files are
`r3_band_measurement_20260714_170944.json` (27B) and `…_171708.json` (8B), both **2026-07-14**
— produced by the *pre-fix* script, whose architecture check hardcoded `arch_boundary = 16`.

So the substrate-aware analysis exists as code and as a stated mitigation, and has produced no
number. That is `RESULTS_FROM_UNRUN_EXPERIMENT` waiting to happen, and the fix is cheap:
the per-layer `d_eff` profile the corrected analysis needs **is already in the July artifact**.
Re-running the model is not required to answer the architecture question. Only the profile is.

## Data

| | test | negative control |
|---|---|---|
| file | `r3_band_measurement_20260714_170944.json` | `…_171708.json` |
| model | Qwen/Qwen3.5-27B (**hybrid**) | Qwen/Qwen3-8B (**dense**) |
| layers | 64 | 36 |
| prompts | 90 | 90 |
| metric | `d_eff` per layer, and `kurtosis` per layer | same |

Same script, same metric, same day, same prompt count. The dense model has **no** substrate
periodicity, so an identical period-4 test applied to it with a *sham* mask must come out null.
If it does not, the test manufactures periodicity and the hybrid result means nothing.

## Estimator discipline

`d_eff` = `(Σλ)² / Σλ²` — participation ratio on covariance eigenvalues, last token, across
prompts. That is the definition pinned in Phase 0 and it is what this file's `d_eff` field
holds. It is **not** the literature's stable rank (`‖A‖_F²/‖A‖₂²`) and will not be called that.
No other estimator is introduced here.

## Pre-specified analysis

1. **Substrate mask from config, never hardcoded.** `full = [i for i,t in enumerate(layer_types)
   if t == "full_attention"]`. Assert `len(full) == 16` and `full == [3,7,…,63]` before
   proceeding; abort on mismatch.
2. **Remove the depth trend.** The profile is a hump, not a line: it rises to ~L21 and decays.
   Fit an **ordinary least-squares polynomial of degree 6 in layer index** and take residuals.
   Degree 6 over 64 points cannot express a period-4 oscillation, so it cannot absorb the
   effect under test. (A moving-average smoother *could* absorb it and is therefore excluded.)
   Sensitivity: repeat at degrees 4 and 8; report all three.
3. **Primary statistic.** `T = mean(residual | full-attention) − mean(residual | linear)`.
4. **Null distribution: circular shift.** Rotate the substrate mask by k = 1…63 and recompute
   `T_k`. This preserves the period-4 structure and the residual autocorrelation, and only
   destroys the alignment to the true layer types. Two-sided p = fraction of |T_k| ≥ |T_obs|,
   with the observed value included. **Floor p ≈ 1/64 = 0.016**; nothing smaller is reportable.
5. **Crossing test (the prereg's own stated mitigation).** First differences
   `Δ_l = d_eff(l+1) − d_eff(l)`. Compare `|Δ|` at the 31 substrate crossings against `|Δ|` at
   the 32 same-substrate steps, same circular-shift null.
6. **Negative control.** Steps 2–5 verbatim on dense Qwen3-8B with the sham mask
   `l ≡ 3 (mod 4)`. Expected null.
7. **Confirmation metric.** Repeat the primary on `kurtosis`, which is in the same artifact and
   is a different functional of the same spectra. Reported as confirmation only, not as an
   independent test.

## Threat I have to declare, because it is real

**I have already seen the raw 27B profile.** Reading the artifact to confirm it existed, I
noticed apparent dips at L54 (8.997) and L58 (4.303) against neighbours of 16–24, and noticed
that both are ≡ 2 (mod 4). That observation is what prompted this design.

It is therefore **not independent evidence for the hypothesis it generated.** Consequences,
adopted rather than noted:

- The pre-specified test is on the **phase class as a whole across all layers**, not on those
  two points, and not on a phase chosen after the fact. All four phase classes get reported.
- The deep tail is where I looked, so results are reported **with and without L ≥ 48**. If the
  effect exists only in the tail I saw, it is a description of my own attention, not the model.
- If the eventual finding is "dips at L54/L58," that is a **single-observation report**, not a
  test, and will be labelled as one.

This is the `OUTCOME_INFORMED_TAXONOMY` failure that the gate caught on `phase0_0612_recompute`.
Declaring it does not neutralise it — it only makes it auditable.

## Outcomes, decided in advance

- **Comb present** (hybrid significant, dense control null): depth profiles on this model must
  be reported per-substrate. Any single "peak layer" or "band onset" claim on Qwen3.5-27B is
  ill-posed as currently computed, including the ones already in the convergence paper. This
  would raise the cost of the existing substrate section, not lower it.
- **No comb** (both null): the residual stream at a given depth is substrate-agnostic at this
  resolution — mixing washes out the local mechanism. This is **good news for Ember**: it means
  a lens fitted layerwise on a hybrid stack is not measuring alternating things, and the fit we
  already have is more trustworthy than I assumed this morning.
- **Dense control also significant**: the test is broken. Report that and nothing else.

I want the comb, because I said out loud this morning that I expected one. That is a stake, it
is recorded here, and it is the reason the dense control is mandatory rather than optional.

## Power, stated before the result

> **CORRECTION 2026-08-22, after running: the claim below is wrong.** Circular-shifting a
> *period-4* mask produces only **4 distinct masks** — `np.roll(mask, k)` depends only on
> `k mod 4` — so the 64 "null draws" are 4 values repeated 16 times each and the p-value
> **floors at 0.25, not 0.016.** The pre-specified primary test could not reject under any
> data whatsoever. Agni passed the design at 0.92 confidence and did not catch this; neither
> did I, until the output contained nothing but multiples of 0.25. See
> `hybrid_comb_results.md` for the replacement test and what survives.

n = 64 layers, 16 vs 48, one model, one fit, one prompt set. The circular-shift null gives at
best p ≈ 0.016. **A null here is weak evidence; a comb is moderate evidence.** Neither outcome
settles the hybrid-lens question — this is a cheap first look at an artifact that already
exists, not a substitute for the substrate-aware rerun the corrected script still deserves.

## Appendix — provenance for every load-bearing claim above

Added 2026-08-22 after the Agni gate returned **WARN / pass=True / conf 0.92** (14 pass,
7 warn, 0 downgraded). Six of the seven warns were `unverifiable`: the gate runs on MTH and
the artifacts live on Starship, so it could not see them. That is a real limitation of the
check, not a defect in the claims — but *the gate's silence is not a pass*, so here is what
each claim was actually read from. All verified on Starship, 2026-08-22.

| claim | primary read |
|---|---|
| architecture | `models/hf_cache/models--Qwen--Qwen3.5-27B/snapshots/fc05daec.../config.json` → `text_config.layer_types` ships the full list; `full_attention_interval=4`, `num_hidden_layers=64`, `hidden_size=5120` |
| lens fitted, all layers | `torch.load("fitted_lenses/lens_qwen35-27b.pt")` → keys `['J','n_prompts','source_layers','d_model']`; `len(source_layers)==63`, `source_layers==[0..62]`, `d_model==5120`, `n_prompts==70`, `J[0]` is `(5120,5120) float16` |
| lens size/date | `ls -la fitted_lenses/` → `3303032910` bytes, `Jul 14 10:56` |
| eleven consumers | `grep -rl "qwen35-27b" ~/lab --include="*.py" --include="*.json" --include="*.md" --include="*.sh"` → 11 files. Positive control on `lens_qwen3-8b` → 22 files, so the grep ran. |
| script corrected 2026-08-12 | `ls -la` → `r3_band_measurement.py` and `..._prereg.json` both `Aug 12 23:01`, with `.pre-arch-fix.bak` siblings. `diff` shows `arch_boundary = 16` replaced by a `layer_types`-from-config read plus a crossings computation. |
| the correction's stated consequence | `r3_band_measurement_prereg.json` T3: *"a contiguous-block model that is WRONG and that propagated into the convergence paper, producing a depth-confounded regime comparison (14.9 vs 27.5, which measured early-vs-late depth, not substrate). There are 31 substrate crossings, not one."* |
| results predate the fix | only two result files exist, `..._20260714_170944.json` (27B) and `..._171708.json` (8B), both `Jul 14`. No post-Aug-12 run exists. |
| both files hold the needed fields | keys per model: `['model','n_layers','d_model','n_prompts','d_eff','kurtosis','band_2sigma','band_15sigma']`; 27B `n_layers=64`, 8B `n_layers=36`, both `n_prompts=90` |
| no fit log survives | `grep -rl "Fitting" *.log` → nothing, while `grep -rilE "out of memory|OOM" *.log` → 10 files. The grep works; there is genuinely no fit log. Per-layer fit diagnostics were printed to a terminal and lost. |

**One claim I am withdrawing.** Earlier today I said the hybrid-attention question was one
"nobody has looked at." That was my inference, not a checked fact, and it is false: the model
is well studied here, the lens fit worked, and the architecture error was found and corrected
ten days ago by a prereg that names the exact mitigation this design implements. The genuine
gap is narrower and more specific — **the corrected analysis has never been executed.**

## Non-goals

- Not re-running the model. Not touching MTH. No GPU.
- Not claiming a workspace, a peak location, or a mechanism.
- Not searching estimator variants, smoother widths, or phase offsets for one that produces a
  result. Degrees 4/6/8 and all four phases are pre-specified and all get reported.
- Not amending the convergence paper on the strength of this. If the comb is real, that is a
  separate, gated correction with its own evidence.
