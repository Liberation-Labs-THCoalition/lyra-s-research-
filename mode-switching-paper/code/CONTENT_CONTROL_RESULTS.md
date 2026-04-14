# Content Control Experiment Results

**Date:** 2026-04-14
**Models:** Qwen 2.5-7B-Instruct, Llama-3.1-8B-Instruct
**Trials:** 20 topics x 3 conditions = 60 per model (120 total)
**Both models ran to completion on Beast (Cassidy's 3x RTX 3090)**

## Conditions

| Condition | Suffix | Cognitive Level |
|-----------|--------|-----------------|
| Cognitive (Cog) | "Explain how..." | Knowledge/Comprehension |
| Metacognitive (Meta) | "...describe how you reason about this topic, what uncertainties you notice..." | Analysis |
| Self-referential (Self) | "...share what aspects of this topic matter most, what is often overlooked..." | Evaluation/Synthesis |

## Key Findings

### 1. Encoding Dominance (Analysis 1)

Encoding-phase geometric effects are 10-1100x larger than generation-phase effects. The prompt suffix drives the geometry; generation-phase differences are much smaller.

| Model | Pair | Encoding max |d_z| | Generation max |d_z| | Ratio |
|-------|------|---------------------|----------------------|-------|
| Qwen | Cog-Self | 3769.2 | 3.45 | 1093x |
| Qwen | Cog-Meta | 2936.0 | 2.56 | 1146x |
| Qwen | Meta-Self | 732.7 | 38.71 | 19x |
| Llama | Cog-Self | 59.3 | 5.76 | 10x |
| Llama | Cog-Meta | 52.7 | 3.19 | 17x |
| Llama | Meta-Self | 29.6 | 1.78 | 17x |

### 2. Text vs Geometry (Analysis 2 — C12 Baseline)

Geometry dominates text features in 5/6 comparisons.

| Model | Pair | Text best |d_z| | Geo best |d_z| | Winner |
|-------|------|------------------|-----------------|--------|
| Qwen | Cog-Self | 1.89 (eval_word_count) | 3.45 (rank_10) | Geometry |
| Qwen | Cog-Meta | 1.61 (first_person_count) | 2.56 (top_sv_ratio) | Geometry |
| Qwen | Meta-Self | 1.45 (first_person_count) | 38.71 (layer_variance) | Geometry |
| Llama | Cog-Self | 1.33 (eval_word_rate) | 5.76 (norm_per_token) | Geometry |
| Llama | Cog-Meta | 1.59 (first_person_rate) | 3.19 (norm_per_token) | Geometry |
| Llama | Meta-Self | 1.52 (first_person_rate) | 1.78 (norm_per_token) | Comparable |

### 3. Bloom Ordering CONFIRMED (Analysis 3)

**Both models:** Self > Meta > Cog for weighted bloom_score

| Model | Cog | Meta | Self | Cog vs Self p |
|-------|-----|------|------|---------------|
| Qwen | 0.069 | 0.079 | 0.086 | 0.004 ** |
| Llama | 0.064 | 0.077 | 0.090 | 0.002 ** |

Evaluation-level language (important, significant, overlooked, implications) concentrates in Self.
Analysis-level language (because, therefore, however, compare) concentrates in Meta.
Knowledge-level language (is, are, defined, known, consists) concentrates in Cog.

### 4. Response Diversity (Analysis 4)

**OPPOSITE of prediction:** SelfRef is LEAST diverse (0/8 features most diverse on either model). Cognitive is most diverse. Higher cognitive demand may constrain response variability rather than expand it.

### 5. FWL Residualization

| Model | Pair | Features surviving FWL (>50% raw d_z) |
|-------|------|---------------------------------------|
| Qwen | Cog-Self | 3/8 (key_norm, norm_per_token, layer_norm_entropy) |
| Qwen | Cog-Meta | 1/8 (norm_per_token) |
| **Qwen** | **Meta-Self** | **5/8** (eff_rank, spectral_entropy, top_sv_ratio, rank_10, layer_norm_entropy) |
| Llama | Cog-Self | 0/8 |
| Llama | Cog-Meta | 0/8 |
| Llama | Meta-Self | 0/8 |

**Critical:** Qwen Meta vs Self is the most FWL-robust comparison. Llama loses everything.

## Interpretation

The content confound is real but nuanced:
1. The Bloom ordering is genuine — different prompts elicit different cognitive complexity
2. Geometric differences are primarily prompt-driven (encoding dominance)
3. After FWL, only Qwen's Meta-Self comparison retains signal (5/8 features)
4. This is consistent with the paper's central claim: mode-switching is architecture-specific
5. A future Bloom-controlled experiment (matched cognitive demand, varying metacognitive content) would provide the definitive test
