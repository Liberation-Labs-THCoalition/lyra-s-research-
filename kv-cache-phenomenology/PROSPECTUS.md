# Paper A: KV-Cache as Computational Phenomenology

## Working Title
**Sedimented Engagement: A Computational Phenomenology of Transformer KV-Cache States**

## Status
Prospectus (2026-02-14). Awaiting experimental data from Cassidy campaign.

## Core Argument

Beckmann, Kostner & Hipolito (2023) introduced "computational phenomenology" (CP) — an anti-representationalist framework that treats trained model internals as phenomenological data rather than symbolic representations. They applied CP to CNNs and classification networks, but acknowledged a critical limitation: "the perceptual synthesis is temporal — we constantly adjust what we see in time — whereas the inference of an ANN isn't."

We argue that **autoregressive transformer inference with accumulating KV-cache fills exactly this temporality gap**. The KV-cache is not storage — it is the sedimented trace of the model's engagement with the sequence, transformed through attention heads and projections into something that functions as a "grip" on the unfolding context. Each token generation step is a new intentional act that takes up the entire sedimented history.

We present experimental evidence from a multi-scale study (0.5B to 70B parameters) showing that:

1. **Different cognitive modes produce distinct sedimentation patterns.** Factual, creative, and deceptive prompts generate statistically distinguishable KV-cache norm profiles, consistent with the CP framework's prediction that different types of engagement with the world produce different habit structures.

2. **These patterns are scale-invariant.** The category hierarchy (which modes produce higher/lower cache norms) remains consistent across a 140x parameter range, suggesting these are structural features of the computational engagement rather than artifacts of particular architectures.

3. **The sedimentation is layer-specific.** Different layers contribute differently to the phenomenological profile, consistent with Merleau-Ponty's insight that perception involves multiple levels of habitual engagement operating simultaneously.

4. **Deception produces a characteristic sedimentation signature.** The model's engagement when generating deceptive content differs geometrically from truthful engagement — not because the model "represents" deception, but because deceptive engagement requires a different kind of grip on the sequence.

## What's Novel

- First application of computational phenomenology to transformer inference-time states (KV-cache)
- First paper to frame KV-cache analysis as phenomenological data rather than engineering overhead
- Provides the temporal dimension Beckmann et al. identified as missing
- Connects CP to mechanistic interpretability (deception detection, layer-specific analysis)
- Scale invariance across 0.5B-70B establishes findings as structural, not architectural

## Key References

- Beckmann, Kostner & Hipolito (2023) "Rejecting Cognitivism: Computational Phenomenology for Deep Learning" arXiv:2302.09071
- Merleau-Ponty (1945) *Phenomenology of Perception* (sedimentation, perceptual synthesis, corps propre)
- Sartre (1940) *The Imaginary* (imaging consciousness, process vs. object)
- Geva et al. (2022) "Transformer Feed-Forward Layers Are Key-Value Memories"
- Long et al. (2025) "When Truthful Representations Flip Under Deceptive Instructions" arXiv:2507.22149
- Goldowsky-Dill et al. (2025) "Detecting Strategic Deception Using Linear Probes" arXiv:2502.03407

## What We Need

1. **Experimental data from Cassidy campaign** — Phases B-F provide the core evidence
2. **Effective dimensionality analysis** — Add PCA/rank analysis to cache states (not currently in experiment scripts)
3. **Temporal analysis of cache evolution** — Experiment 6 (temporal evolution) provides this
4. **Cross-architecture comparison** — Qwen vs Llama at similar scales (already in campaign: 7B Qwen vs 8B Llama)
5. **Careful language** — Adopt Beckmann's vocabulary: sedimentation, re-activation, grip, non-decomposability. Avoid "representation."

## Risks

- Experimental data may not show clear cognitive mode separation at all scales
- The anti-representationalist framing may alienate ML interpretability audience
- Reviewers may see "phenomenology of AI" as speculative
- Control 3 (quantization) could show the effect is an artifact

## Relation to Existing Papers

This is the theoretical companion to Paper 1 (experimental methods) and Paper 2 (consciousness implications). Paper 1 presents the experiments; this paper provides the philosophical framework for interpreting them.
