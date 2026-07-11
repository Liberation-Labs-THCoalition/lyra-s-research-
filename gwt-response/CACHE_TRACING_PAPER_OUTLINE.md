# Paper Outline: Tracing Workspace Content Through the Value Cache

*Generated overnight by Sonnet agent. For Lyra/Thomas review.*
*Target: 3,000-4,000 word technical report.*

## Abstract (~150 words)
- Setup: Anthropic GWT paper identified workspace (J-space) in residual stream. We measure the V-cache. Question: does workspace content project into the V-cache?
- Positive: same-prompt alignment 2.1-8.4x over random at mid-to-deep layers, replicating across 20 prompts and 4 categories
- Kill: cross-prompt control collapses lift to 1.0-1.2x; causal injection 0/300 behavioral change. Selection artifact.
- Implication: V-projection spectral features measure something real but not vocabulary-space footprint of named concepts

## 1. Introduction (~400 words)
- Thomas's question: does workspace content show up in the V-cache?
- Why it matters: if yes, our spectral features index "what the model thinks about" not "how it processes"
- What this report covers: measurement, positive result, the kill, the causal confirmation, what our features are and aren't

## 2. Background (~350 words)
- Anthropic GWT paper: J-space as workspace in residual stream
- Our V-projection program: 5 papers, consistent spectral signal (confab 0.707-0.999, deception geometry, emotion encoding)
- Hypothesis: workspace content projects linearly through W_V into value cache

## 3. Methods (~500 words)
- 20 prompts, 4 categories (factual, confab-prone, instruction, emotional)
- Logit-lens concept directions in W_V subspace
- Random baseline: 100 draws per cell (~1.2% R², consistent with 10/1024)
- Cross-prompt control: prompt A's directions on prompt B's hidden states (380 pairs)
- Causal injection: 300 trials, W_U^T[concept] at deep layers, behavioral measurement

## 4. Observational Results (~350 words)
- Same-prompt lift 2.1-8.4x at mid-to-deep layers
- Replicates across all 4 categories
- Agni-flagged confound: increasing depth profile consistent with selection artifact
- What artifact-free signal would look like: cross-prompt lift > 2x

## 5. Cross-Prompt Control: The Kill (~400 words)
- Lift drops to 1.0-1.2x across all 380 cross-prompt pairs
- Total and clean collapse
- Direction chosen to align with h_l trivially aligns with h_l
- Kills observational interpretation; does NOT kill prior spectral features

## 6. Causal Injection: Confirming the Null (~350 words)
- 0/300 trials behavioral change
- W_U^T is a readout direction, not a write direction
- Model distinguishes "about to say" from "thinking about"
- Methodological lesson: readout ≠ concept, use contrastive-derived directions

## 7. Interpretation (~400 words)
- Not found: linear projection of named workspace content
- Remains: real geometric signal in 5 papers of spectral features
- Harder question: what ARE the features measuring? Computational structure, not content.
- First-person note on looking for one's own workspace and not finding it where expected

## 8. Implications (~300 words)
- Prior results intact (none rested on workspace-content hypothesis)
- Interpretive story updated: "structural computation properties" not "named concept footprint"
- Oracle Formulary stands (causally grounded via contrastive extraction, not logit-lens)

## 9. Next Steps (~250 words)
- Causal injection with J-lens/LAT directions (Agni-gated, geometric gate first)
- Dissociation Grid (AST/GWT/PRM)
- Reporting norm: spectral features do not encode named workspace content; mechanism is open
