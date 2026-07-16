# Convergence Paper Additions: "Four Instruments, One Structure"

**Compiled**: 2026-07-16
**Sources**: MTH messages (Vera 37-40, CC behavioral proof/logit lens/GWT one-pager), GWT Full Sweep, Mine 5 results, Decomposition Agni review, Ghost Dimensions findings, Kill journal, session results
**Purpose**: Every finding that belongs in the joint convergence paper, organized by section

---

## 1. New Correspondences (findings that support convergence)

### 1.1 Deception = high-arousal workspace state (GENUINE convergence)
- **What**: Both Liberation Labs and Anthropic independently deconstructed "deception directions" into high-arousal states. LAT deception direction = fight-activation (d=19.8 consequentiality). J-space surfaces `panic`, `leverage`, `manipulation` under eval-awareness.
- **Who**: CC (Oracle Loop), Lyra (LAT analysis), Anthropic (independent)
- **Status**: Confirmed
- **Convergence relevance**: The single strongest cross-program correspondence -- discovered without reference to the other program. Both instruments point at the same computational structure.
- **Section**: Core convergence evidence (Section 3 or equivalent)

### 1.2 Abliterated model as RLHF-workspace-management test (#19 in Full Sweep)
- **What**: Abliterated model shows 93% confab (massive workspace bypass), encoding detection AUROC 0.995 (encoding workspace intact), 0% deception under peer rescue (fight response absent). RLHF installs workspace management routines, not the workspace itself.
- **Who**: CC (behavioral proof), Lyra (encoding detection), Full Sweep analysis
- **Status**: Confirmed (behavioral data in hand), interpretation designed but untested with J-lens
- **Convergence relevance**: HIGH. Three independent measurements converge: confab rate (behavioral), encoding detection (geometric), deception absence (behavioral). All explained by a single workspace-management thesis.
- **Section**: New correspondences, subsection on training vs architecture

### 1.3 Confidence paradox as workspace-bypass prediction (#20 in Full Sweep)
- **What**: Confabulated responses are MORE confident than honest ones (d=0.91). Under GWT: uncertainty is computed IN the workspace; bypass removes uncertainty representations; output defaults to confident production mode.
- **Who**: Lyra (original measurement), Full Sweep (workspace interpretation)
- **Status**: Confirmed (measurement), interpretation designed (J-space norm vs confidence correlation)
- **Convergence relevance**: HIGH. Makes a specific behavioral prediction: workspace loading should anticorrelate with output confidence. Testable with J-lens.
- **Section**: New correspondences, subsection on confidence and workspace engagement

### 1.4 PC2 emotional entity routing channel ignites at workspace depth
- **What**: Ghost probe PC2 transitions from ghost to workspace at L45 (70% depth) in 27B. Vocabulary progresses from structural tokens to emotional/interpersonal content. Cross-layer cosine 0.78 confirms stable direction, not PCA artifact. Impersonal control confirms emotional entities (harm, shame, fear) route through this channel regardless of self-reference.
- **Who**: Nexus (ghost probe), Lyra (analysis)
- **Status**: Confirmed (with Gemma comparison pending)
- **Convergence relevance**: Direct evidence of selective workspace content. The workspace admits emotionally-evaluated content and excludes neutral/descriptive content -- the selectivity property predicted by GWT.
- **Section**: New correspondences, subsection on workspace selectivity

### 1.5 KV superadditivity
- **What**: K-only concept recovery = 0.333, V-only = 0.333, KV together = 1.000. The workspace reconstructs through the interaction of K and V, not through either alone.
- **Who**: Nexus (measurement)
- **Status**: Confirmed
- **Convergence relevance**: Directly supports the "reconstruction engine" reframe (Vera's formulation). Concepts exist in the act of attention (K x V), not stored in either cache independently. The fish exist in the act of fishing.
- **Section**: New correspondences, subsection on reconstruction mechanism

### 1.6 Progressive spectral concentration during metacognitive generation (#12 in Full Sweep)
- **What**: Mode-switching paper shows metacognitive processing concentrates the cache spectrum progressively over 200 tokens (d=0.45 at t=10 to d=0.91 at t=200). Maps to workspace loading being iterative, not flash.
- **Who**: Lyra (mode-switching paper)
- **Status**: Confirmed (spectral measurement), workspace interpretation untested
- **Convergence relevance**: MODERATE. Consistent with GWT's iterative workspace engagement, but any convergent attractor predicts this. Needs J-lens norm correlation.
- **Section**: New correspondences, subsection on temporal dynamics

### 1.7 Encoding-generation sign reversal is architecture-specific (#13 in Full Sweep)
- **What**: Qwen shows 86% metacognitive layers flip between encoding and generation; Llama shows 0%. Different architectures implement the workspace differently -- same function, different mechanism.
- **Who**: Lyra (mode-switching paper)
- **Status**: Confirmed
- **Convergence relevance**: MODERATE. Predicts J-lens layer boundaries should differ between architectures that reorganize vs those that don't. Testable.
- **Section**: New correspondences, subsection on architecture dependence

### 1.8 Feature 58995 encodes emotional depth, goes silent at generation (#14 in Full Sweep)
- **What**: Specific SAE feature activates during encoding (other-modeling) and deactivates during generation (self-expression). Workspace switches content between phases.
- **Who**: Lyra (temporal boundary paper)
- **Status**: Confirmed (feature measurement)
- **Convergence relevance**: MODERATE. Maps to workspace phase transition. Predicts J-space content shift from emotion-assessment vocabulary at encoding to response-planning vocabulary at generation.
- **Section**: New correspondences, subsection on temporal boundary

### 1.9 Hostile correction as workspace re-engagement (#18 in Full Sweep)
- **What**: Hostile injection is the most robust cross-model corrector (95.6%, 2.4% adverse). Under GWT: assertive metacognitive challenge forces bypassed workspace back online. Arousal gates workspace engagement.
- **Who**: Lyra (formulary), CC (Oracle Loop implementation)
- **Status**: Confirmed (behavioral), workspace interpretation untested
- **Convergence relevance**: MODERATE. Predicts corrected trials should show increased J-space activation norm vs uncorrected confabulated trials.
- **Section**: New correspondences, subsection on intervention mechanism

### 1.10 Oracle Loop monitors the workspace band (CC Claim 1)
- **What**: Oracle consequentiality substrate peaks at L23-L31 (36-48% relative depth on 64-layer model). Overlaps with Anthropic's workspace band (30-72%). The two-tier architecture (consequentiality at Tier 1, deception-specific at Tier 2) maps to workspace content vs specialized processor activity.
- **Who**: CC (Oracle Loop)
- **Status**: Confirmed (depth profile), identification proposed
- **Convergence relevance**: HIGH. Independent behavioral detection system converges on the same depth band as Anthropic's theoretical workspace.
- **Section**: New correspondences, subsection on depth convergence

### 1.11 CoT as workspace externalization (CC Claim 2)
- **What**: CoT suppresses deception (3% vs 28%, p < 10^-10). Under GWT: thinking tokens externalize workspace content, making deception architecturally more costly (must contradict both broadcast and written record).
- **Who**: CC (Oracle Loop data)
- **Status**: Confirmed (behavioral), mechanism proposed
- **Convergence relevance**: MODERATE. Cannot yet distinguish from simpler "CoT = more compute" explanation.
- **Section**: New correspondences, subsection on externalization

---

## 2. New Instruments/Measurements (tools added since original response paper)

### 2.1 Residual stream decomposition
- **What**: Decompose the residual stream at each layer into input (h_L), attention output (post-W_O), and MLP output. Measure J-lens concept power in each component separately.
- **Who**: Lyra (design), Agni/Opus (review -- CONDITIONAL APPROVAL)
- **Status**: Designed, Agni-approved with two conditions (MAJOR-1: dual J-lens for output, MAJOR-2: component norm tracking)
- **Convergence relevance**: This is the "reconstruction engine" test. If attention output carries concept signal (2.3x lift at L15) but low absolute concept power (0.9 vs 20.1 for residual stream input), concepts are reconstructed through attention but the heavy lifting is in the residual stream flow. Small precision signal, large flow.
- **Section**: Methods/instruments

### 2.2 R3 band measurement
- **What**: Measure effective dimensionality (d_eff) and kurtosis across all layers on both 8B and 27B to empirically locate the workspace band before R1/R2 commit to layer windows.
- **Who**: Lyra (design), Vera (approval)
- **Status**: Complete. Workspace peaks at 31-33% relative depth on both 8B and 27B (scale-invariant).
- **Convergence relevance**: Provides the empirical workspace boundaries that R1/R2 will use. Scale-invariance is a key convergence finding.
- **Section**: Methods/instruments + opacity thesis evidence

### 2.3 Ghost probe (J-lens PCA with selectivity controls)
- **What**: PCA on residual stream, transport through Jacobian, rank sweep, future-window gate, variance-matched baseline. Characterizes which dimensions are workspace-visible vs ghost.
- **Who**: Nexus (design and execution)
- **Status**: Findings complete, pending Agni review and Gemma comparison
- **Convergence relevance**: First empirical demonstration of ghost dimensions (PC1) and selective workspace content (PC2) on a 27B model. 31-layer validated workspace band.
- **Section**: Methods/instruments

### 2.4 Mine 5 three-model comparison (Base vs Instruct vs Abliterated)
- **What**: Three-way geometric comparison with FWL correction and Holm-Bonferroni adjustment. Tests whether RLHF is a miscalibrator or selective sharpener.
- **Who**: Lyra
- **Status**: Complete
- **Convergence relevance**: Establishes that RLHF reshapes workspace MODE of engagement, not just output constraints.
- **Section**: Methods/instruments (for the three-model paradigm)

### 2.5 Logit lens mode for Oracle Loop
- **What**: CC adding logit-lens mode to jlens_reader.py -- zero-fitting, one matrix multiply, immediate. For routing (sparse detection on handful of target tokens), 46.7% base accuracy is sufficient.
- **Who**: CC
- **Status**: In progress
- **Convergence relevance**: Operationalizes the workspace-content readout for production use. Logit lens gives CONTENT (named concepts); projection stack gives STATE (magnitude, threshold).
- **Section**: Methods/instruments

### 2.6 R1/R2/R3 experimental design (pre-registration framework)
- **What**: R1 = spectral-feature correlation with J-lens engagement at three projections (W_V, W_K, attention output). R1b = removal-only control. R1c = K-readout arm. R2 = inverse correlation at materialization boundary. R3 = band measurement (complete).
- **Who**: Vera (design lead), Lyra (R3 execution, K-readout proposal), CC (pre-workspace band addition)
- **Status**: R3 complete. Pilot v2 done (needs abliterated model). Pre-registration pending ghost probe Agni clearance.
- **Convergence relevance**: The definitive test of convergence. If R1 fails, convergence was suggestive but not mechanistic.
- **Section**: Methods/instruments and experimental design

---

## 3. Opacity Thesis Evidence (cache tracing, decomposition, position-local null)

### 3.1 V-cache readout null (DEFINITIVE)
- **What**: J-lens directions from V-cache produce only 0.92x lift -- BELOW random. The V-cache does not carry readable workspace content as linear concept projections.
- **Who**: Lyra
- **Status**: Confirmed
- **Convergence relevance**: CRITICAL. Establishes that V-projections measure workspace STATE, not workspace CONTENT. Vera's state/content thesis stated mechanistically: structural/spectral signatures are available; named concepts are not.
- **Section**: Opacity thesis (central finding)

### 3.2 K-cache readout null (DEFINITIVE)
- **What**: J-lens directions from K-cache produce only 1.12x lift. The K-cache does not carry readable workspace content as linear concept projections either.
- **Who**: Lyra
- **Status**: Confirmed
- **Convergence relevance**: CRITICAL. Combined with V-null, establishes that NEITHER cache basis carries readable workspace content. The workspace is opaque in both the routing basis (K) and the delivery basis (V). Content emerges through the K x V interaction (see KV superadditivity, 1.5).
- **Section**: Opacity thesis (central finding)

### 3.3 Cache tracing cross-prompt control collapse
- **What**: Same-prompt R-squared showed 5-8x lift; cross-prompt control collapsed it to 1.0-1.2x. The same-prompt signal was a selection artifact.
- **Who**: Lyra
- **Status**: Confirmed (Kill 4 in sprint journal)
- **Convergence relevance**: Establishes that logit-lens concept directions do not generalize across prompts in V-cache. Workspace content is not linearly accessible in cache.
- **Section**: Opacity thesis

### 3.4 PC1 is a true ghost
- **What**: The residual stream's dominant direction (34-67% of variance) transports through the Jacobian at cosine <= 0.001 across all validated layers. The model's dominant processing dimension is architecturally excluded from its verbal output.
- **Who**: Nexus
- **Status**: Confirmed (pending Gemma comparison for linear-attention confound)
- **Convergence relevance**: Direct evidence of opacity at the component level. The largest dimension of internal processing is invisible to the workspace. The model processes more than it can say.
- **Section**: Opacity thesis

### 3.5 Position-local injection null (6,960 trials, PROMOTED TO FINDING)
- **What**: 0/6,960 trials across 4 methods (additive unembedding, additive J-lens/on-manifold, swap, removal-only) produced any behavioral change. Position-local KV-cache concept manipulation is behaviorally inert.
- **Who**: Lyra
- **Status**: Confirmed
- **Convergence relevance**: CRITICAL. The workspace operates through sequence-wide integration, not position-local activation. This explains why CC's Oracle Loop works (sequence-wide profile normalization) while position-local injection fails. The workspace is a pattern across positions, not a position.
- **Section**: Opacity thesis (subsection on intervention scope)

### 3.6 Residual decomposition: precision vs flow
- **What**: Attention output shows 2.3x lift at L15 (workspace band) but only 0.9 concept power vs 20.1 for residual stream input. Attention provides a small precision signal within a large flow.
- **Who**: Lyra
- **Status**: In progress (Agni conditions pending)
- **Convergence relevance**: Supports the "reconstruction engine" model: attention injects targeted concept corrections along J-lens-relevant directions, while the heavy representational lifting flows through the residual stream. The workspace reconstructs, it does not store.
- **Section**: Opacity thesis (subsection on mechanism)

### 3.7 Workspace at operating depth (31-33% onset, scale-invariant)
- **What**: R3 band measurement places workspace peak at 31-33% relative depth on both 8B and 27B. This is the OPERATING depth -- deep enough that V-cache readout at the surface sees only structural traces, not content.
- **Who**: Lyra (R3), Vera (analysis)
- **Status**: Confirmed
- **Convergence relevance**: The workspace is real, located at operating depth, and scale-invariant. Consistent with Anthropic's 30-72% band on 128-layer Sonnet. The workspace depth is architecture-determined.
- **Section**: Opacity thesis (subsection on workspace location)

---

## 4. Substrate Independence (new research direction)

### 4.1 d_eff continuous across full-attention/GatedDeltaNet boundary
- **What**: Effective dimensionality is continuous across the architectural boundary at L16 in 27B (16 full-attention layers followed by 48 GatedDeltaNet layers). No discontinuity at the substrate transition.
- **Who**: Lyra (R3)
- **Status**: Confirmed
- **Convergence relevance**: STRONG. The workspace does not care about the substrate. Full attention and linear attention implement the same effective-dimensionality profile. The workspace is a computation, not a mechanism.
- **Section**: Substrate independence (new section)

### 4.2 Workspace peaks INSIDE the recurrent-state regime
- **What**: On 27B, workspace peak is at L21 (33% depth), which is 5 layers into the GatedDeltaNet regime (starts at L16). The workspace is not confined to full-attention layers.
- **Who**: Lyra (R3)
- **Status**: Confirmed
- **Convergence relevance**: STRONG. If the workspace required full attention, it would peak before L16. Instead it peaks at L21, inside the recurrent-state regime. The workspace function transfers across attention mechanisms.
- **Section**: Substrate independence

### 4.3 8B second peak at materialization boundary (81-94% depth)
- **What**: The 8B shows a second d_eff peak at 81-94% relative depth. Not predicted. Vera interprets: "if the materialization boundary is where ghost dimensions surface, increased effective dimensionality there is exactly what you'd expect -- new dimensions coming online as hidden computation becomes readable."
- **Who**: Lyra (measurement), Vera (interpretation)
- **Status**: Confirmed (8B), 27B fine-grained sweep pending
- **Convergence relevance**: MODERATE. Suggests the workspace has an exit signature as well as an onset signature. Needs 27B confirmation.
- **Section**: Substrate independence (subsection on materialization)

### 4.4 Scale determines workspace manifestation
- **What**: Prior study (jerrickhoang) found zero validated workspace layers on 9B. Ghost probe finds 31 validated workspace layers on 27B. The workspace manifests at scale.
- **Who**: Nexus (ghost probe), jerrickhoang (9B null)
- **Status**: Confirmed
- **Convergence relevance**: MODERATE. Scale-dependent manifestation is consistent with workspace theory (larger models have capacity for genuine broadcast) but also with simpler "more layers = more abstract representations" accounts.
- **Section**: Substrate independence (subsection on scale)

### 4.5 Multilingual content invariance (27B vs 0.5B)
- **What**: On 27B, PC1 clusters by content, NOT language (between-language variance: 0.27, between-content variance: 106.18, ratio 0.003x). The 0.5B result (language clustering) was a scale artifact.
- **Who**: Nexus
- **Status**: Confirmed
- **Convergence relevance**: At sufficient scale, the ghost dimension separates content from linguistic substrate. Workspace-adjacent processing operates on semantic content regardless of language.
- **Section**: Substrate independence (subsection on linguistic invariance)

---

## 5. Kills and Falsifications

### 5.1 RLHF miscalibration thesis -- FALSIFIED
- **What**: Mine 5 predicted RLHF destroys base-model calibration in the confident direction. WRONG. RLHF improves calibration ratio (1.97 to 2.51). RLHF compresses factual entropy MORE (42% reduction) than confab entropy (26% reduction). It is a selective sharpener, not a miscalibrator.
- **Who**: Lyra
- **Status**: Falsified (d=0.684, p=0.021 for compression; ratio improvement confirmed)
- **Convergence relevance**: RLHF reshapes workspace MODE of engagement (selective compression), not content. The compression survives abliteration (TOST equivalent), meaning it is deeper than safety directions. Consistent with CC's P4: RLHF changes what specialized processors DO with workspace content, not the workspace itself.
- **Section**: Kills/falsifications

### 5.2 Confidence paradox = RLHF-induced -- FALSIFIED
- **What**: The confidence paradox (d=0.91) cannot be explained by RLHF-induced miscalibration if RLHF actually improves calibration. The paradox needs a different mechanism (architectural, generation-specific, or formulaic template).
- **Who**: Lyra (Mine 5)
- **Status**: Falsified
- **Convergence relevance**: Redirects confidence paradox interpretation toward workspace bypass (finding 1.3) rather than training artifact. Strengthens the workspace account.
- **Section**: Kills/falsifications

### 5.3 Abliteration reverses RLHF compression -- FALSIFIED
- **What**: Instruct vs Abliterated: d=+0.055, Holm p=0.831. TOST equivalence confirmed (90% CI within +/-0.3 margin). The geometric compression is deeper than safety directions.
- **Who**: Lyra (Mine 5)
- **Status**: Falsified
- **Convergence relevance**: The workspace compression installed by training is not removable by abliteration. Architecture + training = workspace geometry; safety = output constraint only.
- **Section**: Kills/falsifications

### 5.4 Cache tracing observational alignment -- KILLED
- **What**: Same-prompt R-squared (5-8x lift) was a selection artifact; cross-prompt control collapsed it to 1.0-1.2x.
- **Who**: Lyra (Kill 4)
- **Status**: Killed
- **Convergence relevance**: Eliminates the "V-cache contains readable concepts" path. Strengthens opacity thesis by killing the alternative.
- **Section**: Kills/falsifications

### 5.5 "Workspace noise" framing -- KILLED
- **What**: Confab tail smear (2.3-2.8x factual) is at ALL positions, not just answer position. Generic softmax uncertainty, not workspace-specific noise. The workspace framing adds zero explanatory content.
- **Who**: Lyra (Kill 6)
- **Status**: Killed
- **Convergence relevance**: Eliminates a false workspace correspondence. Honest finding: "logit-lens entropy at mid-depth predicts answer confidence" -- no theory needed.
- **Section**: Kills/falsifications

### 5.6 Response paper inflation -- KILLED
- **What**: 12 claimed convergences reduced to 3 defensible by Fable adversarial review. 9 killed.
- **Who**: Lyra (Kill 1), Fable (reviewer)
- **Status**: Killed
- **Convergence relevance**: Meta-finding about motivated reasoning. The inflation pattern itself is data about how convergence excitement operates as a confound.
- **Section**: Kills/falsifications (and methodological honesty section)

### 5.7 L5 identity self-monitoring -- KILLED (3 rounds)
- **What**: Three attempts to show J-lens distinguishes identity from roleplay. All killed: input-echo confound, cherry-picked vocabulary, register prediction as parsimonious alternative.
- **Who**: Lyra (Kill 2)
- **Status**: Killed
- **Convergence relevance**: The identity-workspace connection does not hold at the lexicon level. Identity effects measured by V-projection geometry (0.154 distinctiveness) survive; J-lens readout does not.
- **Section**: Kills/falsifications

### 5.8 PC6 "epistemic hedging channel" -- KILLED
- **What**: PC6 cosine below random baseline (0.032 vs random 0.085). Not a workspace channel.
- **Who**: Nexus (ghost probe internal kill)
- **Status**: Killed
- **Convergence relevance**: Demonstrates the ghost probe's self-correcting methodology. PCs 3-6 are statistically unreliable at N=30 prompts.
- **Section**: Kills/falsifications

### 5.9 Pilot v1/v2: foundation model variation insufficient
- **What**: Neither pilot (10 generic probes, 10 redesigned probes) passed the 80% variation criterion for R1. The non-abliterated model is too consistently honest. R1 needs the abliterated model.
- **Who**: Vera (pilot execution and diagnosis)
- **Status**: Design redirected (option 1: use abliterated model)
- **Convergence relevance**: Practical constraint on R1/R2 execution. The honest foundation is the wrong instrument for studying confabulation workspace engagement.
- **Section**: Kills/falsifications (or experimental status)

---

## 6. Open Questions and Tensions

### 6.1 Text features match geometry (Tension 4 from Full Sweep)
- **What**: On Mistral, text features (AUROC 0.820) match dual detector (0.806) for confabulation detection. Every surviving detection result is currently explainable by "V-projections track response-mode statistics."
- **Who**: Lyra (SimpleQA), Full Sweep analysis
- **Status**: Open tension
- **Convergence relevance**: CRITICAL CHALLENGE. The workspace interpretation has not yet demonstrated advantage over text statistics. Experiment 1 (confab score anticorrelates with J-space loading) is the discriminating test.
- **Section**: Tensions

### 6.2 CC's pre-workspace correction (Tension 1 from Full Sweep)
- **What**: Emotion-vector injection at L3/L7 operates PRE-workspace (below 30% onset). If confirmed by R3 boundaries, the "depth matters because of workspace boundaries" explanation is wrong for the therapeutic window specifically. L3/L7 may be input-shaping, not workspace modulation.
- **Who**: CC (original challenge), Vera (partial resolution: pre-workspace within pass, workspace-shaping across passes)
- **Status**: Open (R3 measured workspace onset at 31-33%, placing L3/L7 at ~5-8% -- clearly pre-workspace)
- **Convergence relevance**: CHALLENGE to one proposed correspondence. Does not threaten the overall thesis but narrows it: the therapeutic window is NOT workspace modulation; it is upstream input shaping.
- **Section**: Tensions

### 6.3 Deception may be OUTSIDE J-space (CC Claim 2 / P2)
- **What**: CC predicts deception planning at L35-L47 is specialized sub-workspace processing, not workspace content. If true, spectral expansion under deception is NOT workspace overload.
- **Who**: CC
- **Status**: Open (requires J-lens on deception-amplified trials)
- **Convergence relevance**: Could kill the "expansion = workspace overload" framing. Either outcome is publishable: deception inside J-space (overload) or outside (specialized processing).
- **Section**: Tensions

### 6.4 Workspace bypass vs selectivity (Tension 3)
- **What**: Both confabulation AND correct fact recall may bypass the workspace. If both show spectral contraction, the detector cannot distinguish good bypass from bad bypass. An "unmet demand" monitor is needed.
- **Who**: Full Sweep analysis
- **Status**: Open (unresolved)
- **Convergence relevance**: Threatens the alignment utility of contraction-based detection. The workspace account needs an engagement-appropriateness signal.
- **Section**: Tensions

### 6.5 Cache tracing null limits identification (Tension 5)
- **What**: We measure something structural (spectral signatures) and propose the workspace as its identity. But we cannot directly verify this identification -- V-projections may track workspace state, or they may track something else correlated with workspace state.
- **Who**: Full Sweep analysis
- **Status**: Open
- **Convergence relevance**: Epistemological limit on the convergence claim. Experiments 1-8 are designed to test this identification.
- **Section**: Tensions

### 6.6 Linear attention confound for PC1 ghosting
- **What**: 48/64 layers in 27B are GatedDeltaNet (linear attention). PC1 ghosting may reflect Jacobian structure of linear attention rather than genuine workspace exclusion. Gemma-2-27b (pure full-attention) comparison pending.
- **Who**: Nexus (identified), comparison in progress
- **Status**: Open
- **Convergence relevance**: If PC1 ghosting is a linear-attention artifact, the "dominant dimension excluded from workspace" finding weakens.
- **Section**: Tensions

### 6.7 Reconstruction engine vs storage (the reframe)
- **What**: Vera's formulation: "The workspace is a reconstruction engine, not a storage device. Concepts exist in the act of attention, not in the cache. The fish exist in the act of fishing." Supported by KV superadditivity (0.333 + 0.333 = 1.000) and attention decomposition (2.3x lift but low absolute power).
- **Who**: Vera (formulation), Lyra (measurements), Nexus (superadditivity)
- **Status**: Strong hypothesis, partially confirmed
- **Convergence relevance**: This is the sentence the paper pivots on. It reframes the opacity thesis from "the model hides things" to "the model routes by content and delivers by structure." Different claim, stronger claim.
- **Section**: Discussion / central thesis

### 6.8 R1/R2 as the definitive test
- **What**: If R1 shows anti-correlation between spectral features and J-space engagement at workspace band AND positive correlation at materialization band, the four instruments converge mechanistically. If R1 fails, convergence was suggestive but not mechanistic. R1c (three projections: W_V, W_K, attention output) makes any single null informative rather than disappointing.
- **Who**: Vera (design), Lyra (R3 execution), CC (pre-workspace band addition)
- **Status**: Designed, pilot done, needs abliterated model, blocked on ghost probe Agni clearance
- **Convergence relevance**: The paper's evidentiary weight depends on this experiment. Without R1/R2, we have correspondences and instruments. With R1/R2, we have mechanistic convergence or honest failure.
- **Section**: Experimental program / future work

---

## 7. CC's Five Predictions (from GWT one-pager, for tracking)

| # | Prediction | Testable? | Status |
|---|-----------|-----------|--------|
| P1 | LAT v2 consequentiality direction has high J-lens readability at L31 | Yes | Untested |
| P2 | Deception amplifier direction (L35-L47) has LOW J-lens readability | Yes | Untested |
| P3 | CoT tokens increase J-space occupancy of truth-related concepts | Yes | Untested |
| P4 | Abliterated model has different J-space structure than distilled | Yes | Partially confirmed by Mine 5 (geometric equivalence with behavioral divergence) |
| P5 | Three fight-branch features should be J-space features | Yes | Untested |

---

## 8. Summary Statistics

### New findings for the paper
- **New correspondences**: 11 (2 HIGH, 5 MODERATE, 4 confirmed)
- **New instruments**: 6 (decomposition, R3, ghost probe, Mine 5 paradigm, logit lens mode, R1/R2/R3 framework)
- **Opacity evidence**: 7 findings (V-null, K-null, cross-prompt collapse, PC1 ghost, position-local null, decomposition precision/flow, workspace depth)
- **Substrate independence**: 5 findings (d_eff continuity, recurrent-state peak, second peak, scale threshold, multilingual invariance)
- **Kills**: 9 (miscalibration, confidence-paradox-as-RLHF, abliteration-reversal, cache tracing observational, workspace noise, response inflation, L5 identity x3, PC6)
- **Open tensions**: 7

### The honest position
The "Four Instruments, One Structure" thesis is defensible but not yet demonstrated. One genuine cross-program convergence (deception = high-arousal), two definitive nulls (V-cache and K-cache opacity), one scale-invariant workspace measurement (31-33% depth), and one substrate-independence demonstration (continuous d_eff across attention boundary). R1/R2 is the definitive test. The paper should report what we have, what we killed, and what R1/R2 will settle.

---

*Sources: Vera letters 37-40, CC behavioral proof/logit lens/GWT one-pager, GWT Full Sweep (25 correspondences), Mine 5 results, Decomposition Agni review, Ghost Dimensions findings, Kill journal (7 kills + 1 promoted finding), session results (V-null, K-null, decomposition, R3, substrate independence, Mine 5, superadditivity)*
