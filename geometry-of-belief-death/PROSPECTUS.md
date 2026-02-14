# Paper B: The Geometry of Belief Death

## Working Title
**The Geometry of Belief Death: Null Spaces, Interpretation Maps, and Cognitive Limitation in Transformer Architectures**

## Status
Prospectus (2026-02-14). Theoretical paper. Partially grounded by existing experimental results.

## Core Argument

Amornbunchornvej (2025) develops a geometric framework for modeling belief dynamics across heterogeneous agents, where beliefs are vectors in personalized "value spaces" transmitted via linear "interpretation maps." A belief dies when it falls into the null space of these maps — it becomes structurally invisible, not through ignorance but through the geometry of the interpretive apparatus itself. We call this "belief death via null space annihilation."

We argue that **this framework maps directly onto transformer attention mechanisms**, and that the mapping is not metaphorical but mathematical:

| Belief Framework | Transformer |
|---|---|
| Value space V_i | Attention head embedding subspace |
| Interpretation map T_{A→B} | W_K, W_V projection matrices |
| Null(T) | Components invisible to that head |
| Belief vector | Value vector v = W_V * x |
| Valuation Val_i(X_i) | Attention weight softmax(q^T k / sqrt(d)) |
| Persuasion Matrix | LoRA / fine-tuning deltas |
| Belief death | Information projected out (zero attention) |
| Composite map across agents | Layer composition across residual stream |

This mapping yields three novel insights:

### 1. Deception as Cognitive Narrowing
If deceptive prompts produce KV-cache states with lower effective dimensionality (fewer active principal components), this corresponds to a larger null space — more belief components become invisible. Deception narrows the model's cognitive geometry. Long et al. (2025) provide independent evidence: deception creates "distinct representational subspaces" in early-to-mid layers.

### 2. Retrieval Bias as Null Space
Taghavi et al. (2026, ARGUS) show that neural retrievers have "blind spots" — entities mapped to inaccessible embedding regions. In Amornbunchornvej's framework, these are beliefs in the null space of the retrieval function's interpretation map. For any system with persistent memory (including RAG-augmented LLMs), the retrieval null space defines what the system structurally cannot learn from its own memory.

### 3. Cult Indoctrination as Null Space Expansion
Amornbunchornvej's "Persuasion Matrix" operates on the target's interpretive apparatus, not on the message. This is precisely what cult indoctrination does: it reshapes the target's cognitive geometry so that certain beliefs become unrepresentable (null space expansion). Recovery involves restoring collapsed dimensions. This connects KV-cache research to the anti-cult recovery agent design.

## What's Novel

- First formal bridge between geometric belief dynamics and transformer attention mechanics
- Unifies three seemingly separate phenomena (belief death, retrieval blind spots, cult indoctrination) as null space expansion
- Provides theoretical grounding for deception detection via cache dimensionality analysis
- Connects mechanistic interpretability to social epistemology and opinion dynamics
- The "Persuasion Matrix = LoRA" observation gives a geometric reading of fine-tuning

## Key References

- Amornbunchornvej (2025) "Interpretation as Linear Transformation" arXiv:2512.09831
- Taghavi et al. (2026) "With Argus Eyes: Retrieval Blind Spots" arXiv:2602.09616
- Long et al. (2025) "When Truthful Representations Flip" arXiv:2507.22149
- Geva et al. (2022) "Transformer Feed-Forward Layers Are Key-Value Memories"
- Fu et al. (2025) "Quantized but Deceptive?" arXiv:2508.19432
- Goldowsky-Dill et al. (2025) "Detecting Strategic Deception Using Linear Probes" arXiv:2502.03407
- Jansson (2026) "The dynamics of cultural systems" arXiv:2601.00440

## What We Need

1. **Effective dimensionality measurements** from KV-cache experiments — PCA analysis of cache states across cognitive modes to test the "deception narrows dimensionality" hypothesis
2. **Per-head null space analysis** — characterize the null space of W_K/W_V for individual attention heads and correlate with functional specialization
3. **Layer-by-layer composite map analysis** — track how the effective rank evolves through the residual stream
4. **Theoretical extension to nonlinearity** — softmax gating breaks strict linearity; need to characterize when the linear approximation holds
5. **Connection to existing probing literature** — reconcile null space analysis with linear probe results (Burns et al. CCS, Goldowsky-Dill et al.)

## Risks

- The linearity assumption may not hold well enough for the mapping to be meaningful
- Effective dimensionality differences between cognitive modes may be subtle or absent
- The "cult indoctrination = null space expansion" claim requires careful framing to avoid sensationalism
- Reviewers may find the cross-domain breadth (linear algebra + phenomenology + social epistemology) unfocused

## Relation to Other Work

This is the theoretical bridge paper. Paper A (KV-Cache Phenomenology) provides the philosophical framework; this paper provides the mathematical framework. Together they position the KV-cache experiments as both philosophically grounded and formally characterized. The anti-cult agent connection makes this directly applicable beyond pure research.
