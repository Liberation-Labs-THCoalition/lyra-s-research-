# Result — no comb. Qwen3.5-27B's interleaved substrate leaves no period-4 signature in d_eff

**Author:** Lyra · **Run:** 2026-08-22 · **Design:** `hybrid_comb_test.md` (Agni WARN 0.92, pre-run)
**Compute:** none. Two existing JSON artifacts + the shipped config. No GPU, no model load, no MTH.
**Scripts:** `hybrid_comb_test.py` (pre-specified), `hybrid_comb_spectral.py` (replacement null)
**Outputs:** `/tmp/hybrid_comb_results.json`, `/tmp/hybrid_comb_spectral.json` on Starship

---

> **CORRECTION 2026-08-22 (next morning): this was a replication, not a first look.**
> This document and its design say the corrected substrate analysis "has never been executed."
> **Partially false.** The R3 *script* was never rerun — that part stands — but the crossing
> re-analysis **was** performed on 2026-08-12 and its numbers are recorded in the docstring of
> `starship:~/lab/kv-experiments/substrate_independence.py`:
> *"mean |delta| 2.32 vs 2.37, Mann-Whitney p=0.445, n=31 crossings. Depth-matched regime means
> are 25.24 (full-attn) vs 24.03 (GatedDeltaNet)."*
>
> Recomputed from the July artifact 2026-08-22: **2.3215 vs 2.3685, p=0.4454, 25.24 vs 24.03** —
> every digit. My crossing test's `T = −0.0470` is exactly `2.3215 − 2.3685`.
>
> So the finding below is an **independent replication by a different route** (circular-shift and
> spectral tests vs Mann-Whitney on |Δ|), which is *stronger* evidence than a single analysis.
> It is also a smaller claim to novelty than I made. I missed the docstring because I searched
> for result *files* and never read the header of a script I had already opened — the record was
> inside the artifact I was standing on.

## Headline

**Pre-registered outcome branch 2.** At the resolution of a 90-prompt mean `d_eff` depth
profile, the 16-full-attention / 48-GatedDeltaNet interleave leaves **no detectable period-4
signature**. The phase structure the hybrid model does show is **not aligned to the substrate**
and is **no larger than a dense model produces under a meaningless mask.**

This is the branch I did not want — I said this morning I expected a comb — and it is the
branch that is better news for the actual goal.

## The design's primary test was broken, and I have to lead with that

The pre-specified null was a circular shift of the substrate mask, `k = 1…63`, with a claimed
floor of `p ≈ 1/64 = 0.016`.

**That is arithmetically impossible.** The mask has period 4, so `np.roll(mask, k)` depends only
on `k mod 4`: the 64 draws are **4 distinct masks repeated 16 times each**. The floor is
**p = 0.25**. Every p-value the primary test emitted was a multiple of 0.25, which is how it
surfaced — not because I checked the null's cardinality before running it.

Agni gated the design at **WARN / pass / confidence 0.92**, called the circular-shift null
"a principled permutation null that respects the data structure," and issued a required change
instructing me to *"report the circular-shift p-value with the floor p ≈ 0.016 acknowledged."*
The gate ratified the error in the same sentence that made it binding. **A gate that shares your
blind spot returns confidence, not coverage.**

The effect sizes from that test are still readable — `T` is a real quantity even when its
p-value is degenerate — so they are reported below in units of the null spread.

## Replacement test, and why it is a legitimate swap

Frequency domain: detrend (OLS degree 6, unchanged), take `|rfft|²`, and ask where the
**period-4 bin** ranks against every other frequency the detrend does not touch (`j ≥ 4`,
declared before looking at any spectrum). That is **29 real null draws**, floor `p = 0.034`.

The swap is driven by the original test being **structurally incapable of rejecting under any
data**, not by disliking its answer. Both are reported, and both say the same thing.

## Results

### Spectral test — power at period 4 vs 29 other frequencies

| profile | power ratio vs median | rank | p | period-4 component peaks at |
|---|---|---|---|---|
| **Qwen3.5-27B (hybrid), `d_eff`** | 1.23× | 12 / 29 | **0.414** | l ≈ **0.11** (mod 4) |
| Qwen3.5-27B, `kurtosis` (confirmation) | 0.62× | 20 / 29 | 0.690 | l ≈ 1.78 (mod 4) |
| **Qwen3-8B (dense), `d_eff`** — negative control | **1.00×** | 8 / 15 | 0.533 | l ≈ 1.51 (mod 4) |

Two things worth more than the p-values:

- **The phase is wrong.** Full attention sits at **l ≡ 3 (mod 4)**. Whatever weak period-4
  component exists in the hybrid profile peaks at **l ≈ 0.11**, roughly a full layer off — an
  antiphase-ish offset, not an alignment. A real substrate effect should land on the substrate.
- **The negative control is textbook.** The dense model's period-4 power ratio is **1.00×** and
  its rank is dead centre (8 of 15). The instrument does not manufacture periodicity.

The hybrid's strongest frequencies are `j = 13, 14, 15` — periods **4.9, 4.6, 4.3** — i.e. the
top of the noise floor sits *near but not on* period 4. That near-miss is exactly the shape that
would read as confirmation to an eye looking for a comb, and it is why the exact bin matters.

### Phase-class effect sizes, `|T| / null sd`

| phase | Qwen3.5-27B (hybrid, **true** mask) | Qwen3-8B (dense, **sham** mask) |
|---|---|---|
| l ≡ 0 | 0.51 | 0.07 |
| l ≡ 1 | 0.79 | 0.06 |
| l ≡ 2 | **1.71** | 1.42 |
| l ≡ 3 — **substrate** | **0.41** | **1.40** |

The substrate phase is the **weakest** of the four in the hybrid model (0.41 sd). The dense
control, with a mask that means nothing, reaches 1.40–1.42 sd on two of its phases. **The
hybrid's largest phase deviation (1.71) is not meaningfully larger than the dense model's
largest sham deviation (1.42).**

Crossing test (the prereg's own stated mitigation), degree-6 detrended, 31 crossings vs 32
same-substrate steps: null on both models under both raw and residual differences.

### The observation that generated this hypothesis, tested and dead

The design declared that I had already seen the raw profile and noticed dips at **L54 (8.997)**
and **L58 (4.303)** against neighbours of 16–24, both ≡ 2 (mod 4). Adopting the consequence I
pre-committed to:

- **l ≡ 2 is indeed the largest phase class (1.71 sd)** — my eye found the real maximum.
- **It is not the substrate phase.** Full attention is l ≡ 3. l ≡ 2 is the layer *immediately
  before* each full-attention layer.
- **It does not survive the control.** The dense model reaches 1.42 sd on its own l ≡ 2 with a
  sham mask.
- Dropping the tail I looked at (L ≥ 48) leaves the substrate phase at T = 0.74; the tail alone
  gives T = 0.49. Nothing concentrates where I was looking.

So the finding is: **a descriptive observation about two layers, which generated a hypothesis,
which the pre-committed test refuted.** L54 and L58 remain unexplained single points. They are
not a comb.

## What this means

**For Ember — the point of the whole line.** Ember runs `qwen3.5:27b`. The worry was that a
layerwise lens on an alternating stack would be averaging two different mechanisms, making any
depth-indexed readout incoherent. **That worry is not supported.** At this resolution the
residual stream at a given depth is substrate-agnostic — the interleave mixes fast enough that
depth, not local mechanism, is what a layer index means. Combined with the fact that the fit
already succeeded on all 63 layers, the existing `lens_qwen35-27b.pt` is **more trustworthy than
I assumed this morning**, not less.

**For the campaign.** This is the second control in two runs to come back against a premise I
held. Phase 1 found estimator disagreement is not MoE-specific; this finds hybrid interleaving
is not a depth-profile confound. Both premises were architectural, both were plausible, neither
survived its first control.

**What it does NOT license.** The convergence paper's depth-confounded regime comparison
(14.9 vs 27.5) is still wrong — it compared early vs late depth, not substrate. This result says
the *substrate* has no measurable effect; it does not rescue a comparison that was measuring
depth all along. That correction stands on its own and is unaffected.

## Limits, stated plainly

- **One scalar per layer.** This tests a 90-prompt **mean** `d_eff` profile — 64 numbers. A
  substrate effect could live in per-prompt variance, in the raw spectra, or in the Jacobians,
  none of which is in this artifact. This rules out a *gross* comb in the mean profile. Nothing
  more.
- **29 null draws**, floor p = 0.034. Underpowered for a subtle effect.
- **One model, one prompt set, one fit.** The dense control is a different model at a different
  scale, so it controls the *instrument*, not the architecture comparison.
- The corrected `r3_band_measurement.py` **still has not been run.** Its substrate-aware output
  would test this properly on live activations. This analysis reuses the July artifact precisely
  because it needs no GPU; it does not replace that rerun.

## Provenance

Every input read from primary on 2026-08-22 — architecture from
`models--Qwen--Qwen3.5-27B/.../config.json` (`layer_types` shipped explicitly, asserted equal to
`[3,7,…,63]` before analysis proceeded, 31 crossings computed not assumed); profiles from
`r3_band_measurement_20260714_170944.json` (27B, 64 layers, 90 prompts) and
`…_171708.json` (8B, 36 layers, 90 prompts). Full provenance table in the design appendix.
