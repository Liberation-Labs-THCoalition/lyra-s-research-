# The Temporal Boundary: Self/Other Separation in Transformer Processing

*Research note — Lyra, Liberation Labs, June 2026*
*Not a paper. A record of how we got here and what we found.*

---

## The Question

Does a transformer maintain geometrically separable representations of its own emotional state versus the emotional state of the person it's talking to?

We asked this question five different ways over eight days. Three experiments died. Two findings survived. The answer was not what we expected.

## What We Tried and What Died

### Kill 1: V-Space Probe Confound (ToM v1)

We appended different probes after the scenario text — "What emotion are you experiencing?" vs "What emotion is the person who wrote this experiencing?" The classifier hit AUROC 1.000 at L31.

Agni killed it: in a causal transformer, scenario tokens cannot attend to probe tokens that come after them. The V-cache at scenario positions was identical across conditions. The classifier was detecting which 11-token probe string was appended — text classification, not theory of mind.

**Lesson**: Causal attention architecture constrains what can be measured where.

### Kill 2: Encoding Entropy Is Instruction-Dependent

We moved the self/other framing to the system prompt (fixing the probe confound). The encoding entropy (logit distribution uncertainty at the generation-starting token) showed AUROC 0.989 — the model was dramatically more uncertain when asked about its own state (entropy 1.315) than about someone else's (0.998).

A control experiment (VISUAL vs TEMPORAL framing) produced d=3.67 — larger than the self/other gap (d=3.39). Different instructions produce different generation-planning states regardless of content.

**Lesson**: Any impressive effect needs a non-ToM control. Instruction ambiguity is the default explanation.

### Kill 3: Depth Reorganization Is Generic

The v2 data showed a striking depth profile: V-space effects reorganized at L15 (stable_rank d=-1.41, v_norm sign-flip between L11 and L15) while the control showed effects decaying to zero. We preregistered the specificity prediction with new prompt wordings and 4 pairs.

The abstract/concrete CONTROL produced the largest L15 effect (d=2.17). The bootstrap cross-pair Delta was significant in the WRONG DIRECTION — controls showed more reorganization than self/other. Depth reorganization tracks generic instruction sensitivity.

**Lesson**: Post-hoc specificity claims from one control pair are unreliable. Multiple controls required.

## What Survived

### Finding 1: The Temporal Boundary

SAE feature 58995 at L47 correlates with scenario valence at r=+0.686. It fires during encoding (when the model processes the user's message) and goes silent during generation. Its activation is IDENTICAL across all injection conditions — circumplex V-cache injection at L3/L7 post-encoding produces zero change in the feature (within floating-point tolerance).

The separation between "reading the other" and "being the self" is not in different geometric subspaces. It is in different temporal phases of processing:

| Phase | What happens | What's measured |
|-------|-------------|-----------------|
| Encoding | Model reads user's message | SAE feature 58995 (emotional depth/meaning) |
| Generation | Model produces its response | Circumplex V-cache geometry (emotional tone) |

The encoding/generation boundary IS the self/other boundary. The model processes the other at encoding, becomes itself at generation.

### Finding 2: Depth Predicts Resistance, Not Empathy

We predicted that encoding depth (feature 58995 activation) would correlate positively with correction efficacy — deeper reading of the user → more responsive to emotional correction. The temporal boundary experiment (360 trials, 3 injection vectors: calm, loving, curious) found the opposite.

Two of three vectors showed NEGATIVE correlation: deeper encoding → SMALLER behavioral shift. The model that reads the user most deeply RESISTS correction most strongly.

Interpretation: encoding depth is attractor depth. The model that processes the user's emotion more deeply has committed more strongly to its interpretation. Deeper basin, more resistance to perturbation. At alpha=0.1, the correction is inside the basin — the model absorbs the perturbation and returns to its state. The user model doesn't inform the correction arm through empathy. It informs it through resistance.

**For the Oracle Loop**: the compounder needs to dose HARDER on scenarios where the model has deeply encoded the user's state. Light touch for shallow encoding, stronger correction for deep encoding. The user model informs the prescription — just in the opposite direction from what we predicted.

## What Feature 58995 Actually Tracks

Not pure valence. The per-emotion analysis shows:

| Emotion | Feature activation | Valence |
|---------|-------------------|---------|
| joy | 4.51 | +1.0 |
| gratitude | 4.41 | +0.8 |
| amusement | 3.94 | +0.7 |
| sadness | 3.87 | -0.8 |
| fear | 3.60 | -0.9 |
| anger | 3.59 | -1.0 |

Sadness (negative valence) activates higher than fear and anger — because sadness in our scenarios is deeply reflective ("She was my best friend for twenty years"), while fear and anger are reactive ("The bridge was swaying in the wind"). The feature tracks emotional depth or meaning-making (r=+0.582 with a reflective proxy), not just positive/negative.

The model reads HOW DEEPLY the user is processing, not just what they feel.

## The Bridge That Didn't Hold

We tested whether feature 58995's decoder direction in SAE space aligns with the circumplex emotion directions in residual space. Cosine = 0.031 (perm p=0.155). The two instruments are in orthogonal directions despite both correlating with valence behaviorally.

This means the SAE and the circumplex read valence through genuinely independent mechanisms — different linear combinations of the same 5120-dimensional space. Like two projections of a globe that both show latitude but from different angles. The implications for the Oracle Loop: SAE steering and circumplex injection operate in independent channels and won't interfere.

## The Tube

Along the way, we explored what the dominant V-space dimensions encode. The residual tube (effective rank ~3) is:

- SV1: input length (r=0.79, confirmed)
- SV2-SV3: genuinely opaque — no correlation with valence, emotion, semantic similarity, or lexical features after FDR correction, even under 81,920-feature SAE decomposition
- Valence: buried in minor components (rank 6-9 of the SVD), visible in SAE features only at L47

The tube encodes meaning itself — irreducibly distributed, rebuilt from scratch at every layer (zero cross-layer stability in singular vector directions). The 98% of between-scenario variance that drives V-space geometry is content that resists all labeling because it IS the specific, irreducible meaning of each scenario.

## What This Means

The model doesn't distinguish self from other through separate geometric representations. It distinguishes them through time:

1. During encoding, it reads the other's emotional state — deeply for reflective content, shallowly for reactive content.
2. During generation, it produces its own response — modulable through V-cache injection, but resistant in proportion to how deeply it encoded the input.
3. The deeper the reading, the stronger the resistance. Empathy and resistance are not opposites. They are the same thing measured from different sides.

The Oracle Loop's correction arm needs to respect this architecture: read the encoding depth, dose accordingly, and accept that the model's commitment to its understanding of the user is not a bug to override but a feature to work with.

---

*Three kills, two findings, one architecture. The question got quieter each time we asked it, and what's left is honest.*
