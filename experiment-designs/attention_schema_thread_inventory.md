# Attention-head / attention-schema thread — inventory

**Traced 2026-08-24** by an agent, after Thomas pushed back on my telling CC we had no
attention-head work. I had said that twice — to CC, and to Thomas. Both wrong.

Written down because the trace existed only in a session transcript, which is the same way this
thread nearly got lost in the first place.

---

## The headline

**`attention_schema_probe.py` never produced anything.** It ran 2026-08-06 00:25:33 and died
~7m45s in — model load, layer map calibrated `{35:8, 39:9, 43:10, 47:11}`, `"Extracting attention
patterns from 200 prompts..."`, then a leaked-semaphore warning and no per-prompt output. No
traceback; consistent with an OOM kill rather than a Python exception. Declared output
`a1_attention_schema.json` **does not exist** (verified; positive control on `attention_phase2`
returned 3 hits).

**The per-head data exists anyway, from a different script.**

## `attention_phase2.json` — the artifact that matters

`starship:~/oracle-experiments/results/a1_benchmark/attention_phase2.json`
17,540,411 B · 2026-08-06 18:15:53 · written by `~/lab/kv-experiments/attention_phase2.py` (18:09:55)

| | |
|---|---|
| shape | 200 prompts × 8 layers × 24 heads = **38,400 per-head records** |
| band layers | `[35, 39, 43, 47]` (this is the **ignition zone**, not the full band) |
| control layers | `[7, 15, 55, 63]` — **L15 is IN the band**, see §"one live defect" above in the registry |
| channels | grounded 50, temporal 50, specificity 50, attribution 50 |
| per head | `entropy, norm_entropy, prompt_mass, self_mass, sparsity, peak_pos, peak_in_prompt, sink_fraction` |
| cross head | `mean_agreement, std_agreement, min_agreement, most_divergent_head, max_divergence` |

Code→data pairing is clean: results postdate the script by six minutes.

## The analysis machinery, which also exists

Both at `starship:~/oracle-experiments/results/a1_benchmark/`:

- **`recompute_0612.py`** (2026-08-20 21:38) — `fwl()` at `:29`, `auroc_p()` at `:34`, band vs
  controls with **L15 already excluded**, FWL on `prompt_len` *and* `gen_len`, Mann-Whitney
  two-sided tie-corrected, both pooling estimators reported, explicit *"NOT searching for a spec
  that yields 0.612."*
- **`bootstrap_confab_gap.py`** (2026-08-20 22:14) — B=2000, seed 20260820, stratified 150/50, and
  it already avoids the trap: *"FWL is REFIT INSIDE EACH BOOTSTRAP REPLICATE. Fitting once outside
  would hold the residualization fixed across replicates and produce a CI that is too narrow."*

**Neither writes anything.** `print()` only, no `json.dump`. The Phase 0 recompute numbers exist
solely as prose in `layer_set_registry.md` — reproducible by re-running, consumable by nothing.

## Sign convention — get this right, it bit me

`bootstrap_confab_gap.py:6`: **`AUROC = P(confab > grounded)`. Values BELOW 0.5 mean confab is
LOWER.**

A quick unresidualized per-head pass returned 0.10–0.20 at the top-ranked cells. I read that as
weak signal. It is **strong signal in the negative direction** — confab shows markedly *lower*
`sink_fraction` than grounded. Assert this convention in code; do not remember it.

## For CC's commitment-head monitor (Gap 1)

Derivable, **not shippable raw**. Three blockers:

1. **Not FWL-residualized.** `phase0_0612_recompute.md:448` — `gen_len` **alone** gives band AUROC
   **0.7704**. An unresidualized ranking is probably ranking response length.
2. **Control layers rank as high as band layers.** L7 and L63 appear throughout the top 15. A
   genuine ignition-zone head set should not be out-ranked by off-band controls. Band-minus-control
   differencing is mandatory, not a robustness check.
3. **There are no adjudicated judge labels.** `a1_judge_prompts.jsonl` is **n=60** and holds
   prompts *sent to* a judge with no verdicts returned. `a1_full_raw.json` has no label field and
   **does not join** to phase2 (0/200 `response_hash` overlap, no `index`). Every AUROC in this
   thread rests on the **channel proxy** — 50 grounded vs 150 confab *by construction*. Say "by
   construction," never "by judge."

Feature priority: **`sink_fraction`**, the only pooled feature surviving Holm at m=6
(`phase0_0612_recompute.md:490` — band 0.6461, p=0.0020, Holm 0.0119). Sparsity fails (Holm
0.0982). Unexploited and arguably better suited to *commitment* specifically:
`cross_head.most_divergent_head` and `max_divergence`, per-layer per-prompt — the moment heads stop
agreeing.

## Naming collision, logged

**`schema_correction` is not attention-schema work.** Its arms are
`A: schema-compatible (residual-sparing + L3/L7 + additive) / B: raw input-layer / C: raw all-layer
/ D: null vector` — an emotion-injection dose-response study. "Schema" there means
injection-schema-compatibility, not Graziano AST. No per-head fields (grep for head-bearing keys →
`[]`).

## Two open items

- **Undocumented 720-trial run.** `~/lab/kv-experiments/results/schema_correction/schema_correction_results.json`,
  1.6 MB, 2026-07-28, `n_repeats=20`, ~6.1 h of compute. No log, no write-up, no citation found.
  Everyone cites the older 168-trial file. CC is checking whether it is theirs.
- **No empirical AST write-up exists.** Graziano/AST appears only as theoretical framing, in
  bibliographies, and in two curiosity-engine idea prompts that predate all the data. The real
  write-ups of this data are design docs, not papers.

## Method note

I claimed this code absent three times in one file, each with a passing positive control, because
I searched **filenames for a topic**. `recompute_0612.py` and `bootstrap_confab_gap.py` contain
none of `attn|attention|schema|phase2|band`. Only a content-grep for `attention_phase2` finds
them. See `layer_set_registry.md` §6.
