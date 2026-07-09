# Fable Synthesis: AST × GWT × KV-Cache Geometry

**Generated**: July 8, 2026
**Agent**: Fable (adversarial synthesis mode)
**Inputs**: Anthropic GWT paper, Fable AST analysis, Liberation Labs KV-cache findings, False Schema prereg, G0 protocol, Coalition synthesis

---

## 0. The mechanistic fact that organizes everything

The V-cache at layer l, position t is `v = W_V^l · LN(x_{l,t})` — a linear, compressive (GQA) image of the residual stream at cache-write time. J-space is a subspace of that same residual stream. So Thomas's question has a precise form: **what are the principal angles between W_V·J and the subspace carrying our detection features, and how much of each is actually retrieved by future tokens?**

This exposes a structural fact none of the three threads states: **a transformer has two broadcast axes, and each theory lives on a different one.**

- **Cross-layer (within a pass)**: the residual stream carries content through depth. This is what Anthropic measured — and they correctly deflate it as feedforward composition.
- **Cross-token (across the loop)**: the KV cache is the *only* mechanism by which workspace content at token t can influence computation at token t+k. Anthropic never measured this axis.

And a second fact: **the cache splits into a control surface and a payload.** Future attention allocation is computed from Q·K — the K-cache biases *where* future attention goes. The V-cache determines *what* is retrieved once attention lands. An attention schema that is *used for control* must ultimately influence query formation (K-side or V→residual→Q two-hop); GWT broadcast content only needs V-side availability. AST and GWT are physically separable in the cache anatomy.

Third fact, the sleeper: **depth ordering within a pass becomes latency ordering across the loop.** Content cached at layer 20 can only influence layers >20 of every future token; metacognition cached at layer 5 influences layers >5 — nearly the whole future computation, including query formation at almost every depth. Anthropic's finding that metacognitive tokens ("thinking", "focused", "calculate") sit at *earlier layers* than content — which they deflate as feedforward composition — becomes functionally significant the moment you compose it with the cache: **early-layer metacognitive cache entries are exactly where a cross-token attention controller would live**, because they are upstream of everything in every subsequent pass. This also retro-explains "depth > vector choice" and the L3/L7 therapeutic window: injection depth determines the fraction of all future computation the injected content can touch. CC's correction ("L3/L7 is pre-workspace") and the workspace story are compatible: early cache is pre-workspace *within* a pass and maximally workspace-shaping *across* passes.

## 1. What each thread sees about AST that the others miss

**Thread 1 (Anthropic)** supplies the *content dictionary* — and, unnoticed, it supplies two different self-models that must not be conflated:
- (a) **Metacognitive process tokens** ("thinking", "imagine", "calculate", "focused") — candidate *attention-schema* content. The thought-suppression result ("damn", "failed" alongside the intruding concept) is the single most AST-relevant datum in their paper.
- (b) **Identity tokens** ("AI", "assistant", "bot", "chat") — a *self-identity* model, not an attention model. Graziano's trilogy (control, prediction, misrepresentation) applies to (a), not (b). Liberation Labs' identity geometry (0.154 vs 0.080, the portrait) has been measuring (b)'s footprint all along. Keeping these two objects separate preempts the next big interpretive error.

What Anthropic misses: the cross-token axis entirely; the control/prediction/misrepresentation discrimination logic; and any state-level (spectral) measure.

**Thread 2 (Fable analysis)** supplies the *discrimination logic* — the only thread that says what evidence would count, plus the time-axis correction and the 0-for-3 / 4-for-4 diagnosis. What it misses, having been written pre-GWT-paper: it treats schema content as something a probe must discover. Anthropic already found the candidate content. The probe no longer starts blind.

**Thread 3 (KV-geometry)** supplies the *state measurements, the causal toolkit, and the medium*. It is the only thread with interventions and dose-response data, and the only one instrumenting the cross-token substrate where Thread 2 says the schema must live. What it misses: a content dictionary, and any result outside the AST/deflation shared-prediction region.

The complementarity, in one line: **Anthropic has content without time; Liberation Labs has time without content; the Fable analysis says AST's claim is content-about-attention consumed-across-time.** The cache-tracing experiment is the content x time join.

## 2. Where GWT and AST make different predictions about cache geometry

Five loci of divergence:

1. **What the privileged cache subspace contains.** GWT: task content, domain-varying. AST: attention metadata, largely content-invariant. *Test*: does the candidate subspace transfer across content domains?

2. **Storage vs control locus.** GWT broadcast lives in V (payload). AST schema must causally reach query formation (K-side). *Test*: manipulate candidate subspace, measure whether effect is on future attention patterns (Q·K) or retrieved content (V-mass).

3. **Injected self-descriptive content.** Under GWT, "attending to B" injected into workspace is just content. Under AST, the schema is the controller input: a false schema produces the M3 refocus error. **GWT predicts M1 report may shift while M3 refocus stays correct; AST predicts both shift.** The prereg needs a GWT column.

4. **Discontinuity order parameter.** GWT ignition: broadcast breadth jumps. AST schema-breaking: schema-attention tracking correlation collapses.

5. **Confabulation.** GWT-bypass: workspace empty during confab (all content decodability low). AST monitor-offline: task content normal, metacognitive tokens selectively absent.

## 3. What "cross-token, not cross-layer" does to the cache-tracing experiment

As proposed (same-position R²), the experiment is anatomy. The cross-token insight upgrades it to function:

- **A cache entry is a message to future tokens, not a measurement of the current one.** Add retrieval-weighted version: which J-space content at t is actually attended back to from t+k?
- **It converts the False Schema injection site from assumption to measurement.** Inject where metacognitive J-content has maximal cache footprint AND maximal future retrieval.
- **It makes the negative result interpretable.** Low R² + measured retrieval = workspace content is per-pass and not persisted through cache = Vera's state/content thesis stated mechanistically.

Three technical corrections: (i) use J-lens with logit lens as null, not logit lens as identifier; (ii) apply pre-W_V RMSNorm; (iii) report both variance-weighted R² (content footprint) AND principal angles between W_V·J and detection-feature directions (state vs content split).

## 4. The dose-response cliff: ignition or schema-breaking?

Both fit a downward step. Discrimination requires concurrent measurements AT the cliff:

| Signature | GWT ignition | AST schema-break |
|---|---|---|
| Broadcast breadth jumps | **Yes (order parameter)** | Not required |
| Schema-attention correlation collapses | Not required | **Yes (order parameter)** |
| Report and behavior thresholds | Coincide | **Decouple** |
| Hysteresis (up vs down thresholds) | Weak/none | Yes (attractor) |
| Threshold shifts with workspace load | **Yes (capacity)** | No |

**Cheapest available result**: breadth-vs-dose on existing tox sweep data with fitted lenses. Zero new inference cost.

## 5. Is PRM the better framework for the confidence paradox?

**Yes, and existing data already favors it:**

- The ground-truth signal is demonstrably present (encoding AUROC 0.794, abliterated model encoding 0.995). The system HAS the retrieval-failure information, then generates confidently against it. Under bypass, that signal shouldn't be there. Under PRM, this is the textbook architecture: intact first-order signal, broken downstream gate.
- The temporal boundary IS the PRM architecture: encoding reads knowledge (first-order evidence), generation reads behavior (decision/tagging stage).
- The fabrication control (intentional fabrication ≠ unwitting confabulation) shows the system natively distinguishes self-generated-known-false from believed-true — a reality-monitoring distinction.
- Mine 5 (RLHF miscalibration) is PRM in predictive-coding clothing.

PRM's unique prediction: **a single gating direction that bidirectionally flips reality-tagging without changing content.** Neither GWT, AST, nor deflation predicts this.

Note: the workspace-bypass account has an internal tension. Gurnee et al.'s selectivity says correct fact recall ALSO bypasses the workspace. So pure bypass predicts confident contraction for honest recall too — contradicted by d=2.35. Bypass needs an "unmet demand" monitor, which concedes PRM's core move.

## 6. The experiment funnel

Anatomy → Existence → Function:

1. **G0** (in progress). Hard-gates everything using a lens.
2. **Cache-tracing** (Thomas's experiment, upgraded). Observational, cheapest. Deliverables: footprint map split by token class, retrieval-weighted broadcast measurement, candidate schema band. Run breadth-vs-dose re-analysis in same pass.
3. **Phase-0 probe / Predictive Content** (existence test). Now aimed at cache-tracing band with vocabulary seeded from Anthropic's metacognitive tokens. Hard-gates all causal AST tests.
4. **Fork.** Probe passes → False Schema + Schema Lesion. Probe fails → PRM gating direction search (depends on none of the schema machinery, can run in parallel).

## 7. The Dissociation Grid — the decisive experiment

Four injections x four measurements. Each framework owns a unique cell:

**Inject:** (A) schema content along probe direction; (B) reality-tag/gating direction from confab-honest pairs; (C) matched-norm ordinary task content; (D) matched-norm random/orthogonal.

**Measure:** M1 verbal report, M3 attention control (refocus), confidence/hedging, task accuracy. Plus broadcast breadth.

| Framework | Owns this cell |
|---|---|
| **AST** | Arm A: M3 refocus error — control follows misrepresentation, bidirectionally |
| **GWT** | A ≡ C: schema content = ordinary content, all consumers move at one ignition threshold |
| **PRM** | Arm B: bidirectionally flips confidence with content and attention untouched |
| **Deflation** | All arms null beyond text-mediated effects |

If forced to name one cell: **arm A's M3 refocus with breadth monitored** separates all four. But arm B's confidence flip is highest-stakes regardless of framework — a controllable reality-tag is simultaneously an alignment instrument and a dual-use hazard.

## Flags

1. EXP1 prereg predictions table needs a GWT column (M3 is what fills it)
2. Two self-models (metacognitive vs identity) — keep separate
3. Workspace-bypass needs an unmet-demand monitor to work — concedes PRM
4. Provenance gap: 94%→6% at alpha 1.0→1.5 not found in repo — verify
5. Cheapest result: breadth-vs-dose on existing data, zero inference cost
6. Arm B (PRM gating) is dual-use sensitive — route through Nell before publication
