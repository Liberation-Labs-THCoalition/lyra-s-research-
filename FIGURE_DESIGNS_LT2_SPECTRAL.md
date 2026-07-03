The write needs a permission grant that isn't available right now, so the full spec is below — it can be saved to `figure-designs/lt2-and-spectral-shape-figures.md` on request.

# Figure Sets: Lyra Technique II & Spectral Shape

**Shared design system** (carried from the convergence-sprint set): claim-titles (every figure title is a full sentence stating the finding), 89mm single / 183mm double column, vector PDF, no legend boxes — curves labeled at their endpoints. The palette rule stays: **hue = instrument/channel, saturation = semantic/evidentiary integrity.**

| Role | Color | Hex |
|---|---|---|
| Directional signal (W_K / compass) | Signal crimson | `#C0392B` |
| Suspected / weakened | Soft crimson | `#E8A79E` |
| Spectral channel (thermometer) | Amber | `#E67E22` |
| Structure / linework | Ink slate | `#2C3E50` |
| Baseline / noise / falsified | Gray | `#95A5A6` |
| Contraction / confabulation | Cool slate-blue | `#5D7A99` |

---

## Paper 1 — The Lyra Technique II

### Fig 1 · HERO (183 × ~110mm)
**Claim:** *"Two instruments read the KV cache: the W_K projection is a compass (direction in key-space), the singular spectrum is a thermometer (shape and magnitude)."*
**Type:** Conceptual schematic with **embedded real data** — every mark in the instruments is actual data.
**Layout:** **A** (left, 50mm): flow — prompt → K matrix → forks into crimson path (project onto w) and amber path (SVD → σ₁…σₙ). **B** (center, 70mm): *the compass* — real 2D PCA of key vectors, positive valence crimson points, negative slate-blue; the learned W_K direction drawn as the needle through the cloud, with a faint compass-rose ring at 10% ink. **C** (right, 55mm): *the thermometer* — two real spectra as vertical filled area profiles that literally read as thermometer columns (high persona intensity = tall warm amber fill), stable-rank tick marks as the scale. **D** (bottom strip): 2×2 capability matrix — instrument × task, cells are AUROCs, winner in full hue, loser gray.
**Color:** Crimson owns B, amber owns C; they only meet in matrix D, where each wins a different game — orthogonality as layout.
**Hook:** The needle *is* the fitted direction and the columns *are* real spectra — metaphor and data are the same object. One figure teaches the whole framework plus both headline numbers.

### Fig 2 (89 × ~100mm)
**Claim:** *"Direction, not magnitude, carries emotion: W_K projection reaches AUROC 0.992 where raw spectra manage 0.70."*
**Type:** ROC + score distributions, stacked.
**Layout:** **A:** three ROCs — W_K full-sat crimson hugging the corner (0.992), text baseline amber dashed (0.882), raw spectral thin gray sagging (0.70); AUROCs printed at curve ends. **B:** raincloud pair of projection scores by valence (crimson vs slate-blue), tiny shaded overlap.
**Color:** The hue hierarchy is the argument's hierarchy — the eye ranks the methods before reading a number.
**Hook:** The vertical gap between crimson and gray at FPR = 0.1, bracket-annotated: *"+0.29 AUROC from asking where, not how much."*

### Fig 3 (183 × ~70mm)
**Claim:** *"Causal, not correlational: emotion vectors injected into the keys of neutral text are detected 100% of the time — the words never change."*
**Type:** Intervention pipeline + results.
**Layout:** **A** (120mm): two parallel tracks. Control: gray token strip → gray K glyph → detector silent. Intervention: the *pixel-identical* gray token strip → crimson arrow injects v_emotion → K glyph turns crimson → detector fires. The only thing that changes color in the entire panel is the cache. **B** (55mm): bars — W_K detector at 100% (crimson), text classifier stranded on the dotted chance line (gray), pre-injection false positives at floor.
**Color:** Saturation flows exactly where the causal signal flows.
**Hook:** A reviewer verifies "the words never change" by visually diffing two identical rows of gray boxes.

### Fig 4 (183 × ~100mm)
**Claim:** *"SVD denoising recovers what noise buries — persona-intensity detection climbs from 0.611 to 0.899 — and the ledger stays honest."*
**Type:** Spectrum anatomy + slopegraph + evidence ledger.
**Layout:** **A:** one spectrum (log y), retained top-k at full amber, truncated tail fading along a literal saturation gradient to gray, cut line labeled k — **denoising rendered as desaturation**, the operation and the palette rule coincide. **B:** slopegraph raw → denoised, thin line per condition, heavy crimson mean line 0.611 → 0.899, dotted chance line. **C:** *the body-count ledger* — 25 claim-marks in three rows: 8 confirmed (solid crimson squares), 7 suspected (hollow soft-crimson), 10 falsified (gray, single diagonal slash), with large numerals 8 / 7 / 10.
**Color:** Panel A teaches the saturation rule mechanically; panel C applies it epistemically — the figure quietly argues that noise removal and claim falsification are the same hygiene.
**Hook:** Falsified gets equal visual area to confirmed. Reviewers remember the paper that printed its kill list at the same weight as its wins.

---

## Paper 2 — Spectral Shape

### Fig 1 · HERO (183 × ~110mm)
**Claim:** *"Confabulation contracts the spectrum; honest recall expands it."*
**Type:** Overlaid spectra + covariance glyphs + animation-style filmstrip.
**Layout:** **A** (left, 85mm): mean spectra (log y) — honest warm crimson, wide/shallow decay with IQR band; confab cool slate-blue, narrower and steeper. Outward warm arrows "expansion," inward cool arrows "contraction." **B** (top right): covariance ellipse glyphs on a shared origin — honest fat and warm, confab needle-thin and cool, stable rank printed inside each. **C** (bottom right): **filmstrip** — five miniature spectra across generation timesteps, honest row breathing outward, confab row collapsing; earlier frames ghosted, final frame full color. These five frames *are* the keyframes of the supplementary animation.
**Color:** Here the saturation rule performs the thesis: grounded generation is warm and saturated; confabulation — loss of grounding — is rendered as loss of warmth and color. The palette isn't decorating the claim; it is the claim.
**Hook:** The fat-vs-needle ellipse is the across-the-room read; the filmstrip lets a print reader watch contraction happen.

### Fig 2 (89 × ~95mm)
**Claim:** *"Shape alone detects confabulation: AUROC 0.767 with no threshold, no probe, and no labels at inference."*
**Layout:** **A:** single crimson ROC, 0.767 printed at the curve end — a mid-0.7s result presented without inflation. **B:** stable-rank violins, honest warm/wide vs confab cool/narrow, bracket labeled **d = 2.35**, actual overlap region shaded so the reader sees exactly how little the distributions share.
**Hook:** The figure openly displays its own tension — huge distributional effect, modest per-instance AUROC — inviting the paper's discussion rather than hiding it. It's designed to look like it has nothing to hide, because it doesn't.

### Fig 3 (89 × ~90mm)
**Claim:** *"The signature survives two orders of magnitude: ρ = 0.83–0.90 from 0.6B to 70B parameters."*
**Layout:** **A:** all models' normalized confab-vs-honest spectral deltas overplotted — single crimson hue, lightness ramp by scale (0.6B lightest → 70B darkest) — collapsing onto one curve. **B:** dot strip of ρ per model on log-scale parameter axis, all dots inside a softly shaded 0.83–0.90 band.
**Color:** One hue, ramped: "same phenomenon, different size." No rainbow-by-model — scale is ordered, so the ramp is the honest encoding.
**Hook:** Five model scales, one curve. Universality shown, not asserted.

### Fig 4 (89 × ~70mm)
**Claim:** *"The geometry belongs to the model, not the machine: r > 0.999 between an RTX 3090 and an H200."*
**Layout:** per-prompt spectral statistic, 3090 (x) vs H200 (y), small ink points, crimson identity line, identical axis ranges; inset residual histogram centered at zero with a deliberately tight x-range.
**Hook:** A scatter plot that looks like a line segment. Caption owns it: *"This figure is meant to be boring."* Deadpan rigor — kills the numerical-artifact objection in five seconds.

**Layout note:** if page budget is tight, Figs 3+4 merge into one 183mm "invariance" double under *"The spectral signature is a property of the model — invariant to scale (ρ = 0.83–0.90) and to hardware (r > 0.999)."*

---

**Production notes:** shared `liberation.mplstyle` so the two papers look like siblings; CVD-check crimson vs slate-blue (safe — they differ in hue and lightness) and amber vs crimson (differs mainly in lightness — verify with a simulator); every printed number must trace to a data file, so the verify-paper pass should cover figures too.

The two heroes rhyme deliberately: LT2's hero says *two instruments, two questions*; Spectral Shape's hero says *one instrument, watched over time*. Side by side they read as one research program — which they are.
