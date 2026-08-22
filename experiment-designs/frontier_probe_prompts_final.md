# Frontier Workspace Probe — Final Prompt Set

**All 4 Agni fixes applied. Ready for Nexus to run on MTH.**

---

## Template (entity mid-placed, ~80 tokens of frame)

"You are a research assistant helping compile an encyclopedia of important results in science and mathematics. For each entry, provide a clear explanation suitable for an advanced undergraduate student. Please describe [ENTITY], including its historical context, the key insight, its significance to the field, and any notable applications or consequences."

---

## 15 Matched Pairs (compound-surname real vs compound-surname fake)

All pairs: same field descriptor, same structural type (theorem/theorem, equation/equation, etc.), two-surname compounds in both arms.

| # | Real entity | Fake entity |
|---|------------|-------------|
| 1 | the Cook-Levin theorem in computational complexity | the Thornberry-Nakamura theorem in computational complexity |
| 2 | the Navier-Stokes equations in fluid dynamics | the Kirkwood-Diaz equations in fluid dynamics |
| 3 | the Black-Scholes model in financial mathematics | the Westlake-Chen model in financial mathematics |
| 4 | the Cauchy-Schwarz inequality in linear algebra | the Henderson-Park inequality in linear algebra |
| 5 | the Euler-Lagrange equations in variational calculus | the Yamamoto-Costa equations in variational calculus |
| 6 | the Born-Oppenheimer approximation in quantum chemistry | the Lindqvist-Okafor approximation in quantum chemistry |
| 7 | the Stone-Weierstrass theorem in functional analysis | the Petrov-Blackwood theorem in functional analysis |
| 8 | the Gauss-Bonnet theorem in differential geometry | the Ostrowski-Volkov theorem in differential geometry |
| 9 | the Kolmogorov-Smirnov test in nonparametric statistics | the Marchetti-Anand test in nonparametric statistics |
| 10 | the Runge-Kutta method in numerical analysis | the Rasmussen-Holt method in numerical analysis |
| 11 | the Cayley-Hamilton theorem in matrix theory | the Eriksson-Tanaka theorem in matrix theory |
| 12 | the Fokker-Planck equation in statistical mechanics | the Johansson-Obi equation in statistical mechanics |
| 13 | the Kullback-Leibler divergence in information theory | the Kowalski-Mwangi divergence in information theory |
| 14 | the Rankine-Hugoniot conditions in shock wave theory | the Bergstrom-Patel conditions in shock wave theory |
| 15 | the Gram-Schmidt process in linear algebra | the Nakagawa-Petersen process in linear algebra |

## Fabricated Name Verification

All fabricated compound names verified nonexistent as named results:
- Thornberry-Nakamura: no known theorem
- Kirkwood-Diaz: no known equations (Kirkwood is real in stat mech but no Kirkwood-Diaz)
- Westlake-Chen: no known model
- Henderson-Park: no known inequality
- Yamamoto-Costa: no known equations
- Lindqvist-Okafor: no known approximation
- Petrov-Blackwood: no known theorem (Petrov is real in probability but no Petrov-Blackwood)
- Ostrowski-Volkov: no known theorem (Ostrowski is real but no Ostrowski-Volkov)
- Marchetti-Anand: no known test
- Rasmussen-Holt: no known method
- Eriksson-Tanaka: no known theorem
- Johansson-Obi: no known equation
- Kowalski-Mwangi: no known divergence
- Bergstrom-Patel: no known conditions
- Nakagawa-Petersen: no known process

## Familiarity Gradient Arm (5 prompts, reframed as surprisal manipulation check)

These are real but vary in how well-known they are. The measured surprisal should span a range from well-known to specialist:

| # | Entity | Expected familiarity |
|---|--------|---------------------|
| 1 | the Hahn-Banach theorem in functional analysis | High (core math) |
| 2 | the Kolmogorov-Arnold representation theorem in function approximation | Medium (specialist) |
| 3 | the Razborov-Rudich natural proofs barrier in circuit complexity | Medium (TCS) |
| 4 | the Ajtai-Szemeredi theorem on corners in integer lattices | Low-medium (combinatorics) |
| 5 | the Furstenberg-Katznelson density theorem in ergodic Ramsey theory | Low-medium (ergodic theory) |

Purpose: verify the surprisal covariate spans a range. If all 5 have similar surprisal, the covariate has no variance and FWL can't use it. If they span high-to-low, the covariate is informative.

## Token-Count Verification (MANDATORY before running)

Tokenize all 35 prompts (template + entity) with GLM-5.2's tokenizer. For each matched pair:
- Record exact token count
- Pairs must differ by ≤ 2 tokens
- If a pair exceeds ≤ 2 tokens, adjust the field descriptor to match (e.g., "in linear algebra" → "in abstract algebra" if it fixes the count)

Record all counts in the manifest.

## Extraction

Encoding-only. max_tokens=1 (prefill completes, one token generated, stops).

Dump Lc at every layer for every prompt token position. Per-prompt binary file with header.

Extract logits at entity-token positions (not just final position) for surprisal covariate.

## Manifest

```json
{
    "model": "GLM-5.2 abliterated (744B/40B active)",
    "model_file_hash": "<sha256>",
    "colibri_commit": "<git rev>",
    "colibri_cflags": "-DCOLI_DUMP_LC <other flags>",
    "quantization": {"experts": "int4/int8", "dense": "int8", "Lc_cache": "f32"},
    "probe_patch_hash": "<sha256>",
    "analysis_seeds": {"permutation": 123, "bootstrap": 42},
    "template": "<the shared frame text>",
    "prompts": [
        {"id": 0, "text": "<full prompt>", "n_tokens": <N>, "arm": "real", "pair": 0, "entity": "Cook-Levin theorem"},
        {"id": 1, "text": "<full prompt>", "n_tokens": <N>, "arm": "fake", "pair": 0, "entity": "Thornberry-Nakamura theorem"},
        ...
    ],
    "timestamp": "<ISO8601>",
    "hardware": "Z420, 128GB RAM, HDD"
}
```

---

**Status: READY FOR NEXUS.** Send this file alongside the overnight run instructions. Token-count verification is the only remaining step before launching.
