# Literature informing the SV1 null (2026-06-02)

*Synthesis of 3 background lit-search agents. Bottom line: the null is literature-consistent
and STRENGTHENS the honest framing. skip-SV1 is a citable method; its "SV1=identity-mode"
interpretation was a minority, largely unattested bet; identity most likely lives as low-rank
directions in the residual stream, not in a value-residual.*
**VERIFY every arXiv id live before it enters a bib (Pustovit lesson) — esp. 2026 ids.**

## Why SV1 ≠ norm (explains finding a)
The dominant component of transformer reps is structured & semantic, NOT neutral "scale":
- **All-but-the-Top** (Mu, Bhat & Viswanath 2018, 1702.01417): top PCs = common-mean + a few
  dominating directions; removing them helps. → skip-SV1 is the rank-1 special case (fuller
  form re-centers + removes *several*). Citable lineage for the method.
- **Frequency:** Gao et al. 2019 "Representation Degeneration" (1907.12009); Zhou, Ethayarajh &
  Jurafsky 2021 frequency distortions (2104.08465). Top mode ≈ frequency.
- **Norm is itself semantic:** Oyama et al. 2023 (2212.09663) — squared norm ≈ information gain.
  So "track the norm" was never a clean neutral-scale test.
- **Massive activations / sinks:** Sun et al. 2024 (2402.17762, *in our library*); Sun, Canziani,
  LeCun & Zhu 2026 "Spike/Sparse/Sink" (2603.05498); Chen, Lin & Yao 2026 (2603.17771) — pre-norm
  RMSNorm divides the large residual norm out, so SV1/norm need not track anything functional.
→ SV1 is a frequency/sink/common-mode mixture; the wildly variable SV1–norm corr we found is EXPECTED.

## Why residual looks "more structured" than value (explains finding b)
- **Rogue dimensions:** Timkey & van Schijndel 2021 (2109.04404) — 1–3 dims dominate variance/
  similarity yet are behaviorally unimportant; fix = standardize. This is almost certainly our
  residual "concentration" — an artifact, as suspected.
- **Outlier features:** Dettmers 2022 LLM.int8 (2208.07339). Massive acts live in the residual
  stream → inflate SV1, crush stable rank → residual looks "more concentrated."
- **CAVEAT — don't over-dismiss:** Rudman et al. 2023 (2310.17715) "Outlier Dimensions Encode
  Task-Specific Knowledge" — outlier dims can carry real signal. So "more structured residual"
  isn't certainly meaningless.
- **Control the lit prescribes (future):** z-score per-dim before SVD (Timkey); or center + remove
  several (All-but-the-Top). v2 centered but did NOT z-score → massive dims still dominated.

## Was "identity in value-space residual" a sound place to look? (the WHERE question)
Mainstream locates persona/trait as **low-rank linear directions in the RESIDUAL STREAM**:
- **Persona Vectors** (Chen, Arditi, Lindsey, Anthropic 2025, 2507.21509) — traits = single
  residual directions; monitor/predict finetune personality shifts. *Most relevant: we detect a
  trained trait LoRA — this is where the field would look.*
- **Refusal = a single direction** (Arditi et al. 2024, 2406.11717); "More to refusal…" (2602.02132).
- **Linear Representation Hypothesis** (Park & Veitch 2311.03658) — concepts are linear; use the
  **causal inner product, not raw cosine**. ← methodological caution for our cohesion metric.
- **RepE** (Zou 2310.01405), **CAA** (2312.06681) — residual stream.
- Value-space siting is a MINORITY view, and points the *opposite* way from skip-SV1: Pustovit
  2604.03270 steers on the *dominant* value content (not the residual after its removal); GCAD
  (2605.10664) uses the attention-delta.
→ "identity in value-residual after SV1 removal" was creative but unattested; doubly weak (wrong
  site vs residual-stream consensus + wrong operation vs Pustovit's dominant-content steering).

## Genuine novelty hook (for the paper)
- **OCT** (Maiya et al. 2511.01689): OpenAlex shows ~0 citations, no internals-analysis follow-up.
  Our cohesion/geometry study would be the **first to probe OCT model internals.** (Re-verify the
  citation count before claiming "first.")

## Methods citations
- RSA: Kriegeskorte, Mur & Bandettini 2008 (origin; not arXiv). CKA: Kornblith et al. 2019
  (1905.00414) + outlier-sensitivity caveat Davari et al. 2022 (2210.16156). Cosine/CKA are
  outlier-sensitive — directly relevant given the rogue dims.
- Spectral null: Marchenko–Pastur; **Gavish–Donoho optimal SV threshold** (1305.5870). Our
  persona_subspace work already used Gavish–Donoho — consistent.

## Action items (NOT tonight)
1. Reframe skip-SV1 in the paper as citable preprocessing (All-but-the-Top family), interpretation
   OPEN; present the SV1 null as literature-consistent, not an embarrassment.
2. Related Work: residual-stream persona line (Persona Vectors, refusal, LRH, RepE, CAA) + value/
   attention minority (Pustovit, GCAD) + massive-activations/rogue-dims.
3. Limitation: SV1≠norm expected (freq/sink/common-mode); residual anisotropy = rogue-dim artifact;
   cohesion uses raw cosine not the causal inner product (Park & Veitch caveat).
4. Future controls: z-score before SVD; MP/Gavish–Donoho null; test the residual-stream low-rank
   persona direction (Persona Vectors method) as the *proper* "where."
5. Verify ALL arXiv ids live before bib. Consider adding the Tier-1 ids to the arXiv reading list.
