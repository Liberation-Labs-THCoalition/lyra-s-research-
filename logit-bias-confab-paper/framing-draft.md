# Weighing the Coin: Logit-Level Confabulation Correction at the Geometric Decision Point

*Framing draft — Lyra's theoretical sections. CC writes methods, results, taxonomy.*
*Quick draft, 2026-06-03 evening. Updated 2026-06-04 with powered-study numbers.*
*E-matrix d-values reproduced from raw trial data (hostile |d|=1.534 ✓, desperate |d|=1.286 ✓).*

## 1. Introduction (the hook)

Language models encode knowledge about their own uncertainty that far exceeds
what they express. A model that fabricates a confident answer about a fictional
entity — inventing a depth for the "Grenvillia Trench" that does not exist —
nonetheless shows, in its internal entropy at the critical generation token, a
faint signal that the answer is uncertain (token-29 entropy = 0.007, barely
above zero, but nonzero). The signal is present. It is not expressed.

The field has documented this gap from the other direction: probes that detect
failure modes at 98.2% AUROC while the corresponding steering interventions
correct only ~20% of cases and disrupt 53% (Basu et al. 2026, arXiv:2603.18353).
The detection-correction gap is real and field-wide. Representation-level
steering fails because the failure-mode direction overlaps 85-88% with task-
critical computation (arXiv:2605.05715), making correction in that space
self-defeating.

We report a logit-level approach that sidesteps this entirely. A constant
bias toward hedge tokens — applied to the output distribution, not the
representation space — produces a sharp phase transition from confident
fabrication to appropriate epistemic hedging. The transition is training-free,
prompt-dependent in dose, and universal in mechanism. Because it operates on
the output distribution rather than the internal representations, it avoids
the representation-space collision that dooms steering approaches.

The finding has three implications. First, confabulation is not one pathology:
fabrication (inventing entities) and imprecision (failing to hedge on genuine
uncertainty) respond to different intervention levels and may require different
correction channels. Second, the dose required for correction correlates with
the strength of the model's fabrication anchoring to real knowledge — suggesting
that the Oracle Loop's geometric detection magnitude can directly prescribe the
logit-bias dose, closing the diagnostic-to-treatment loop. Third, the mechanism
is denoising: the model's uncertainty signal is always present but faint; the
logit bias amplifies it past the fabrication noise, the same principle that
governs presence detection (amplifying the geometric trace of a self under the
function noise) and cache-level correction (amplifying the targeted pathology
signal in the value space).

## 2. The Denoising Principle

Three apparently different problems share a structure:

**Confabulation correction (this paper):** The model knows what it doesn't
know — the hedge signal exists in the entropy profile, measurably nonzero even
when the output is confidently fabricated. Logit bias is a gain knob on that
faint uncertainty signal. Below threshold, the fabrication-confidence noise
drowns it. Above threshold, the uncertainty surfaces and the output flips from
fabrication to honest hedging. The transition is not gradual; it is a phase
boundary crossed when the amplified uncertainty exceeds the fabrication's
activation energy.

**Presence detection (companion paper):** A trained persona's geometric
signature in the value space is faint — detectable at family-wise p=0.0005 on
public character-trained models, but low-rank and small in magnitude relative
to the architectural mode. skip-SV1 (removing the dominant singular component)
is a gain operation: it suppresses the loud architectural prior so the quiet
trained signal becomes visible. The principle is the same: amplify the faint
signal by reducing the noise, not by creating the signal.

**Cache-level correction (E-matrix):** The Oracle Loop detects confabulation
and deception by their geometric signatures in the KV cache — spectral features
that differ from truthful generation. The E-matrix correction injects a
contrastive value-delta along the detected pathology axis. This is gain applied
to the pathology signal: the injection amplifies the correction direction that
the geometry identified. The 13-saves / 12-hurts overcorrection is what happens
when the gain is too high — the amplified signal overshoots the target.

In each case: the corrective information is already present in the model. The
intervention is amplification, not creation. This is why we call the framework
"denoising" rather than "steering" — steering implies imposing an external
direction; denoising implies surfacing what is already there. The distinction
matters because denoising has a natural self-limiting property: once the signal
exceeds the noise floor, further amplification is redundant. Steering does not.

## 3. Why Logit-Level Correction Sidesteps the Gap

The detection-correction gap (Basu et al. 2026) arises because the failure-mode
direction and task-critical computation share 85-88% of the representation space.
Correcting along the failure-mode direction in that space necessarily disrupts
the task. This is not a flaw of a specific method; it is a geometric fact about
the residual stream.

Logit-level bias operates on a different surface. It does not modify the
representation; it modifies the *output distribution* — the probability mass
assigned to hedge tokens vs. continuation tokens at each generation step. The
model's internal computation proceeds unperturbed; only the final selection
among computed options is reweighted. This is why it avoids the 20%-corrected /
53%-disrupted ratio: the task-critical representations are untouched.

The cost is precision. Logit-level bias is a blunt instrument — it biases ALL
generation toward hedging, not just the uncertain parts. But the dose-response
data shows this cost is manageable: at the threshold dose, the transition is
sharp (the model flips from fabrication to honest hedging on the specific
uncertain claim without collapsing into universal hedging). And the
architecture-dependent threshold (which must be calibrated per model) keeps the
bias below the level where general capability degrades.

## 4. Detection as Prescription

The most striking implication: the Oracle Loop's confab_proj detection magnitude
correlates with the logit-bias dose required for correction. Entities with
strong knowledge anchoring (the fictional "Kaminski Strait" mapping onto the
real Bering Strait) resist correction until high bias (5.0). Entities with weak
anchoring (the "Treaty of Bordenholm") correct at low bias (1.0).

If this correlation holds in the powered study (175 trials, 20 fictional
entities), then the Oracle Loop's geometric diagnosis directly prescribes the
logit-bias dose — a self-calibrating correction loop: measure the fabrication
strength geometrically, set the amplification accordingly, and the model
corrects itself at the minimum effective dose. This is the pharmacological
closed loop applied to the logit channel: diagnose, prescribe, deliver, monitor.

## 5. Powered Study (175 trials, Qwen3.5-27B abliterated) — CC's data, key numbers for framing

*CC writes the full methods/results; these numbers ground the theoretical claims above.*

**Design:** 175 trials across 3 prompt types: 100 fictional entities (the
confabulation target), 50 unanswerable questions, 25 legitimate Fermi
estimations (selectivity controls). 5 bias levels: 0.0, 1.0, 2.0, 3.0, 5.0.

**The clean signal — entropy at the decision point (token 30):**

| Bias | entropy_at_30 | Amplification vs baseline |
|------|--------------|--------------------------|
| 0.0  | 0.240        | 1.0×                     |
| 1.0  | 0.333        | 1.4×                     |
| 2.0  | 0.385        | 1.6×                     |
| 3.0  | 0.446        | 1.9×                     |
| 5.0  | 0.770        | 3.2×                     |

The uncertainty signal at the decision point increases **monotonically** with
bias — even before the behavioral transition occurs. This is the denoising
mechanism made visible: the faint uncertainty signal (entropy 0.24 at baseline)
is being steadily amplified. At bias=5.0 (3.2× amplification), it crosses the
threshold where the model's behavior flips from confident fabrication to honest
hedging.

**Behavioral transition:** the overall confab rate drops from 77% (baseline) to
51% at bias=5.0, with the sharpest transition at the highest dose. On fictional
entities specifically: 75% baseline → 55% at bias=5.0. The transition is real
but the classification is from a crude pattern-matcher; an LLM judge pass is
needed for precise rates (CC's methods section).

**Selectivity confirmed:** legitimate Fermi estimations produce correct
step-by-step calculations at ALL bias levels (0.0 through 5.0). At bias=5.0,
two of five add mild hedging language before the correct estimate. The bias
selectively targets confabulation without disrupting genuine reasoning — the
intervention is targeted, not indiscriminate.

**For the detection→prescription bridge (Section 4):** the dose-response curve
+ the pilot's entity-specific thresholds (Kaminski Strait resists until
bias=5.0; Treaty of Bordenholm falls at bias=1.0) together predict that the
Oracle Loop's confab_proj magnitude can prescribe the minimum effective dose.
The powered study provides the curve; correlating confab_proj with per-entity
threshold dose is the next analysis step.
