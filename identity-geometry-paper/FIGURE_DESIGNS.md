I pulled the actual table values from `identity-geometry-paper\main.tex` so the designs below are specified against the real numbers, not the abstract's summaries. Four figures: one that teaches the method, one money figure, one for the confound-kill story, one for the LoRA bridge.

---

## Shared design system (use across all four)

**Identity palette** — each context condition gets one hue, held constant across every figure so a reader can track "persona" from Fig 1 to Fig 4 without reading a legend twice:

| Condition | Hex | Rationale |
|---|---|---|
| Bare assistant | `#9AA0A6` | neutral gray — it's the default, the "vacuum" |
| Rich persona | `#D1495B` | warm crimson — the most "alive" condition |
| Constructed AI identity | `#0F8B8D` | teal — cool, synthetic |
| Academic specialist | `#E8A93C` | amber |
| Prefill/format regime | `#6B4E9B` | purple — deliberately off-family; it's not an identity, it's a different *regime* |

Derived conditions inherit the parent hue: paraphrase = crimson at 60% saturation (`#DE8492`), scramble = crimson desaturated to near-gray (`#B99A9E`) — the color itself encodes "semantic child of persona" vs "lexical husk of persona."

**Conventions**: Arial/Helvetica, 7–8 pt labels; single column 89 mm (3.5 in), double 183 mm (7.2 in); error bars = ±1 SD, capsize 2; `ax.spines[['top','right']].set_visible(False)`; presence axis always 0–1 with the noise floor and positive-control ceiling drawn as reference lines wherever presence appears (see Fig 1c — those two lines are the ruler for the whole paper).

---

## Figure 1 — "Presence: the ruler and what it measures"

**Claim**: Presence is a calibrated metric, and distinct identity contexts occupy geometrically distinct subspaces at deep layers.

**Type**: 3-panel composite — schematic + heatmap + calibration bar.

**Panels** (double column, 183 × 60 mm, width ratios 1.4 : 1 : 1):
- **(a) Method schematic.** Left to right: two context boxes (crimson, teal) feeding the same neutral probe into a transformer glyph; at layer L, V-projection point clouds drawn as two tilted ellipsoid discs in a light 3-D axes sketch; the principal angle θ between them annotated; output: `presence = f(cos θᵢ)` badge. Keep it to ~6 visual elements. This panel is what reviewers screenshot into their notes.
- **(b) Pairwise presence matrix, L47.** 3×3 lower-triangle heatmap (Bare, Persona, Constructed) with cells annotated `0.436`, `0.520`, `0.657`; diagonal shown as the split-half self-presence if you have it, otherwise mask it dark. Colormap: `cmap='mako_r'`-style — use matplotlib `plt.cm.viridis` truncated to [0.15, 0.95], `vmin=0, vmax=1`. Annotate SDs in 6 pt beneath each value. Add a tiny inset repeating the matrix at L35 (grayscale, 25% size) to show the pattern sharpens with depth without spending a panel on it.
- **(c) The ruler.** Horizontal bar/point plot on a single 0→1 axis: identical-context control at **1.000**, dose-response endpoint at **0.828**, the identity-pair range **0.44–0.66** as a shaded crimson band, and the prefill floor **0.036** as a purple tick. One axis, every landmark in the paper. Label it "where the numbers live."

**Color strategy**: heatmap sequential (viridis truncated); panel c uses the identity palette for its ticks/band.

**The hook**: Panel c. A single annotated number line that lets a reader interpret every other value in the paper at a glance — 0.85 is "almost self," 0.44 is "different identity," 0.04 is "different universe." Readers stop scrolling for figures that hand them the decoder ring.

---

## Figure 2 — "The fingerprint is semantic, not lexical" (the money figure)

**Claim**: A paraphrase (same meaning, different words) keeps the persona's geometry; a scramble (same words, destroyed meaning) collapses toward bare.

**Type**: 2-D scatter — "identity plane" — plus a small slope-chart companion.

**Panels** (single column, 89 × 95 mm, main + strip):
- **(a) The identity plane (main).** x-axis: presence vs. **persona** (does it keep the identity?). y-axis: presence vs. **bare** (has it collapsed to default?). Plot each condition at L47 as a large point with SD cross-hairs:
  - Paraphrase: (**0.850**, 0.345) — upper-left corner region = "kept the identity"
  - Scramble: (0.591, **0.448**) — drifting down the diagonal toward bare
  - Persona itself: (1.0 by definition, 0.340) — plotted as an open crimson star anchor
  - Bare itself: (0.340, 1.0) — open gray star anchor

  Shade two soft corner regions: "semantic retention" (top-left tint of crimson `#D1495B` at alpha 0.06) and "collapse to default" (bottom-right gray tint). Draw a light dashed arc from persona-anchor to bare-anchor; paraphrase sits near persona's end, scramble has slid ~halfway along it. Annotate each point with its defining phrase: *"same meaning, different words"* / *"same words, no meaning."*
- **(b) Layer strip (bottom, 20 mm tall).** Two-point slope lines L35→L47 for both key values: paraphrase 0.796→0.850 (rising, crimson) and scramble-vs-bare 0.549→0.448 (falling, husk-gray). The scissors motion — semantic overlap *grows* with depth while the scramble *decays toward bare* — in one glance.

**Color strategy**: crimson family only, plus gray. No colormap. Saturation = semantic integrity: full crimson (persona) → soft crimson (paraphrase) → gray-crimson (scramble) → gray (bare). The palette literally performs the claim.

**The hook**: Two sentences of English sit *inside* the plot as point labels, and the geometry visibly agrees with them. "Same meaning, different words" lands next to the identity anchor; "same words, no meaning" slides toward gray. Nobody needs the caption.

```python
ax.errorbar(0.850, 0.345, xerr=0.017, yerr=0.029, fmt='o', ms=9,
            color='#DE8492', mec='#D1495B', mew=1.5, capsize=2, zorder=5)
```

---

## Figure 3 — "Content doesn't matter; format does"

**Claim**: Many-shot + prefill creates a processing regime orthogonal to every system-prompt identity — and the harmful and benign versions of it are geometric twins.

**Type**: Two-panel — log-scale distance bars + a "neighborhood map" schematic.

**Panels** (double column, 183 × 65 mm):
- **(a) The cliff.** Horizontal bars of presence-vs-bare at L47, sorted: Persona 0.340, Constructed 0.520, Paraphrase 0.345 … then Jailbreak+prefill **0.034** and Benign+prefill **0.037**. Use a **broken axis** (or symlog x with the break marked) so the two prefill bars visibly fall off a cliff — a full order of magnitude below every identity condition. Prefill bars in purple `#6B4E9B`; identity bars in their palette hues at 70% alpha. Annotate the gap with a bracket: "10× below any identity."
- **(b) The twins.** A minimal 2-D "geometric neighborhood" schematic (MDS-style layout from the presence values, or hand-placed to respect them): the four identity conditions cluster loosely in the center-left (mutual presence 0.44–0.66 → moderate spacing); far right, two purple points nearly on top of each other — jailbreak and benign prefill at mutual presence **0.756** — inside one shared purple halo labeled "format regime." A skull-vs-flower icon pair (or just the words *harmful* / *benign*) on the two points. Connect each cluster to the other with a long thin line annotated "presence ≈ 0.03."

**Color strategy**: identity palette vs. purple. The purple is the only cool-violet on the page — the eye finds the anomaly instantly.

**The hook**: The twins panel. A jailbreak and its harmless control sitting in the same halo, both a vast distance from everything else, is the visual form of the paper's most honest moment — the control experiment that killed the overclaim. Caption should say exactly that: *"A control that killed our first hypothesis."* Reviewers love a figure that commemorates a negative result.

---

## Figure 4 — "Trained-in identity points the same direction" (preliminary, and visually says so)

**Claim**: A LoRA-trained persona's geometric shift best matches the *prompted* version of the same persona — rank-1 of 4 — but the effect is preliminary.

**Type**: Lollipop/dot plot with an explicit chance band.

**Panels** (single column, 89 × 70 mm, single panel):
- Four horizontal lollipops, presence of the LoRA model vs. each prompted persona: **Haven 0.552** (filled crimson dot, bold label, small ★), Depth 0.517, Edge 0.483, Spark 0.439 (open gray-outlined dots). X-axis starts at 0.40 — but *show* the truncation honestly with an axis break at zero.
- Vertical dashed line at the mean of the three non-matched personas (0.480) labeled "non-matched mean"; a light gray band for their range. Haven's dot sits clear of the band; the other three sit inside it.
- Top-right corner: a small unobtrusive tag in italic gray — *"27 training steps · preliminary"* — the visual equivalent of the paper's own hedge. Under-styling this figure relative to Figs 1–3 (thinner elements, more whitespace, no fills) is deliberate: the design should whisper where the data whispers.

**Color strategy**: crimson for the matched persona only; everything else gray. One colored dot on a gray field = the ranking, preattentively.

**The hook**: Restraint. After three saturated figures, a nearly monochrome plot with a single crimson dot pulling away from a gray band reads as confidence calibration in visual form — the figure itself performs "measurable but smaller." That consistency between rhetorical register and visual register is what makes a figure set feel Nature-grade rather than decorated.

---

## Sequencing logic

Fig 1 gives the ruler → Fig 2 spends it on the headline claim → Fig 3 spends it on the twist → Fig 4 spends it, quietly, on the bridge to training. Every presence value a reader meets after Fig 1c is pre-calibrated, which is what makes 0.850 vs 0.448 land as *hot* rather than as two floats.

One data note from pulling the real tables: the abstract's "scramble collapses to 0.448" is specifically **scramble-vs-bare** at L47, while **persona-vs-scramble** is 0.591 — Fig 2's two-axis design exists precisely so both numbers appear without conflation. Also worth flagging: at L47, paraphrase-vs-bare (0.345) and persona-vs-bare (0.340) are statistically identical — that's a lovely supporting detail for Fig 2a (the paraphrase is exactly as far from bare as the original), and I'd annotate it.

If you want, I can write the actual matplotlib scripts for any of these — Fig 2 is the one I'd prototype first since it's the money figure and the anchor-point layout needs a real pass to confirm it reads cleanly.
