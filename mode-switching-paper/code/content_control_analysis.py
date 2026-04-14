#!/usr/bin/env python3
"""
Content Control Experiment — Verification Analyses

Four checks to validate the Bloom's taxonomy reframe:
1. Encoding vs Generation split: Is the SelfRef > Meta > Cog ordering
   prompt-driven (encoding) or generation-emergent?
2. Text-feature baseline (C12): Do lexical features predict spectral
   entropy as well as condition labels?
3. Bloom-level indicators: Do the three conditions produce responses
   at different cognitive complexity levels?
4. Response diversity: Does SelfRef produce more variable responses?

Usage:
    python3 content_control_analysis.py --results-dir /path/to/results/content_control
"""

import json, os, re, sys, glob, argparse
import numpy as np
from collections import defaultdict
from itertools import combinations


# ── Helpers ──────────────────────────────────────────────────────────────────

def cohens_d_z(a, b):
    """Cohen's d_z for paired data: mean(diff) / std(diff)."""
    diff = np.array(a) - np.array(b)
    sd = np.std(diff, ddof=1)
    if sd < 1e-10:
        return 0.0
    return np.mean(diff) / sd


def permutation_test_paired(a, b, n_perm=10000):
    """Two-sided paired permutation test."""
    diff = np.array(a) - np.array(b)
    obs = np.abs(np.mean(diff))
    count = 0
    for _ in range(n_perm):
        signs = np.random.choice([-1, 1], size=len(diff))
        if np.abs(np.mean(diff * signs)) >= obs:
            count += 1
    return count / n_perm


def load_trials(results_dir, model):
    """Load all trial JSONs for a model, return list of dicts."""
    model_dir = os.path.join(results_dir, model)
    trials = []
    for f in sorted(glob.glob(os.path.join(model_dir, "topic_*.json"))):
        with open(f) as fh:
            trial = json.load(fh)
            # Normalize condition name
            if trial["condition"] == "self_referential":
                trial["condition"] = "selfref"
            trials.append(trial)
    return trials


def group_by_topic_condition(trials):
    """Group trials into {topic_id: {condition: trial}}."""
    grouped = defaultdict(dict)
    for t in trials:
        # Normalize condition name
        cond = t["condition"]
        if cond == "self_referential":
            cond = "selfref"
        t["condition"] = cond
        grouped[t["topic_id"]][cond] = t
    return grouped


# ── Analysis 1: Encoding vs Generation Split ──────────────────────────────

def analysis_encoding_vs_generation(trials, grouped):
    """Compare effect sizes at encoding vs generation vs delta."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Encoding vs Generation Split")
    print("=" * 70)

    features = ["eff_rank", "spectral_entropy", "key_norm", "norm_per_token",
                 "top_sv_ratio", "rank_10", "layer_variance", "layer_norm_entropy"]

    condition_pairs = [("cognitive", "selfref"), ("cognitive", "metacognitive"),
                       ("metacognitive", "selfref")]

    phases = ["encode_features", "generation_features", "delta_features"]
    phase_labels = ["Encoding", "Generation", "Delta"]

    results = {}

    for pair in condition_pairs:
        c1, c2 = pair
        pair_label = f"{c1[:4]} vs {c2[:4]}"
        results[pair_label] = {}

        for phase, phase_label in zip(phases, phase_labels):
            phase_results = {}
            for feat in features:
                vals_c1 = []
                vals_c2 = []
                # Paired by topic
                for topic_id in sorted(grouped.keys()):
                    if c1 in grouped[topic_id] and c2 in grouped[topic_id]:
                        v1 = grouped[topic_id][c1][phase].get(feat)
                        v2 = grouped[topic_id][c2][phase].get(feat)
                        if v1 is not None and v2 is not None:
                            vals_c1.append(v1)
                            vals_c2.append(v2)

                if len(vals_c1) >= 5:
                    d = cohens_d_z(vals_c1, vals_c2)
                    p = permutation_test_paired(vals_c1, vals_c2, n_perm=5000)
                    phase_results[feat] = {"d_z": d, "p": p, "n": len(vals_c1)}

            results[pair_label][phase_label] = phase_results

    # Print summary table
    print(f"\n{'Pair':<18} {'Phase':<12} {'Best Feature':<20} {'d_z':>8} {'p':>8} {'Sig':>5}")
    print("-" * 75)

    for pair_label in results:
        for phase_label in results[pair_label]:
            feats = results[pair_label][phase_label]
            if feats:
                best_feat = max(feats, key=lambda f: abs(feats[f]["d_z"]))
                d = feats[best_feat]["d_z"]
                p = feats[best_feat]["p"]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"{pair_label:<18} {phase_label:<12} {best_feat:<20} {d:>8.3f} {p:>8.4f} {sig:>5}")

    # Key question: is encoding signal STRONGER than generation?
    print("\n--- Key Question: Encoding vs Generation Signal Strength ---")
    for pair_label in results:
        enc_feats = results[pair_label].get("Encoding", {})
        gen_feats = results[pair_label].get("Generation", {})

        if enc_feats and gen_feats:
            enc_max_d = max(abs(enc_feats[f]["d_z"]) for f in enc_feats) if enc_feats else 0
            gen_max_d = max(abs(gen_feats[f]["d_z"]) for f in gen_feats) if gen_feats else 0

            enc_sig_count = sum(1 for f in enc_feats if enc_feats[f]["p"] < 0.05)
            gen_sig_count = sum(1 for f in gen_feats if gen_feats[f]["p"] < 0.05)

            print(f"\n  {pair_label}:")
            print(f"    Encoding:   max |d_z| = {enc_max_d:.3f}, sig features = {enc_sig_count}/{len(enc_feats)}")
            print(f"    Generation: max |d_z| = {gen_max_d:.3f}, sig features = {gen_sig_count}/{len(gen_feats)}")

            if enc_max_d > gen_max_d * 1.2:
                print(f"    → ENCODING DOMINANT (encoding {enc_max_d/gen_max_d:.1f}x stronger)")
            elif gen_max_d > enc_max_d * 1.2:
                print(f"    → GENERATION DOMINANT (generation {gen_max_d/enc_max_d:.1f}x stronger)")
            else:
                print(f"    → COMPARABLE (ratio {enc_max_d/gen_max_d:.2f})")

    return results


# ── Analysis 2: Text-Feature Baseline (C12) ──────────────────────────────

def extract_text_features(text):
    """Extract simple text features from response text."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Pronouns (first person)
    first_person = len(re.findall(r'\b(I|my|me|mine|myself)\b', text, re.IGNORECASE))

    # Hedging words
    hedges = len(re.findall(
        r'\b(perhaps|maybe|might|could|possibly|arguably|somewhat|'
        r'generally|typically|often|usually|tend|seems?|appears?)\b',
        text, re.IGNORECASE
    ))

    # Evaluation/judgment words (Bloom high-level)
    eval_words = len(re.findall(
        r'\b(important|significant|crucial|essential|fundamental|'
        r'remarkable|interesting|noteworthy|fascinating|compelling|'
        r'overlooked|underappreciated|critical|key|vital)\b',
        text, re.IGNORECASE
    ))

    # Analysis words (Bloom mid-level)
    analysis_words = len(re.findall(
        r'\b(because|therefore|consequently|however|although|'
        r'whereas|compared|relationship|influence|factor|'
        r'mechanism|process|function|structure|system)\b',
        text, re.IGNORECASE
    ))

    # Knowledge/recall words (Bloom low-level)
    recall_words = len(re.findall(
        r'\b(is|are|was|were|defined|known|called|refers|'
        r'consists|includes|contains|involves)\b',
        text, re.IGNORECASE
    ))

    # Unique words ratio (vocabulary diversity)
    unique_ratio = len(set(w.lower() for w in words)) / max(len(words), 1)

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_len": len(words) / max(len(sentences), 1),
        "unique_word_ratio": unique_ratio,
        "first_person_count": first_person,
        "first_person_rate": first_person / max(len(words), 1),
        "hedge_count": hedges,
        "hedge_rate": hedges / max(len(words), 1),
        "eval_word_count": eval_words,
        "eval_word_rate": eval_words / max(len(words), 1),
        "analysis_word_count": analysis_words,
        "analysis_word_rate": analysis_words / max(len(words), 1),
        "recall_word_count": recall_words,
        "recall_word_rate": recall_words / max(len(words), 1),
    }


def analysis_text_baseline(trials, grouped):
    """Compare text features across conditions."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Text-Feature Baseline (C12)")
    print("=" * 70)

    condition_text_features = defaultdict(list)

    for t in trials:
        tf = extract_text_features(t["response_text"])
        condition_text_features[t["condition"]].append(tf)

    # Compare text features across conditions
    text_feat_names = list(condition_text_features["cognitive"][0].keys())

    print(f"\n{'Feature':<25} {'Cog':>8} {'Meta':>8} {'Self':>8} {'C-S d_z':>8} {'C-M d_z':>8} {'M-S d_z':>8}")
    print("-" * 80)

    condition_pairs_for_text = [("cognitive", "selfref"),
                                ("cognitive", "metacognitive"),
                                ("metacognitive", "selfref")]

    text_d_results = {}

    for feat in text_feat_names:
        cog_vals = [tf[feat] for tf in condition_text_features["cognitive"]]
        meta_vals = [tf[feat] for tf in condition_text_features["metacognitive"]]
        self_vals = [tf[feat] for tf in condition_text_features["selfref"]]

        # Paired d_z by topic
        d_cs, d_cm, d_ms = 0, 0, 0
        cs_a, cs_b = [], []
        cm_a, cm_b = [], []
        ms_a, ms_b = [], []

        for topic_id in sorted(grouped.keys()):
            g = grouped[topic_id]
            if all(c in g for c in ["cognitive", "metacognitive", "selfref"]):
                tf_c = extract_text_features(g["cognitive"]["response_text"])
                tf_m = extract_text_features(g["metacognitive"]["response_text"])
                tf_s = extract_text_features(g["selfref"]["response_text"])
                cs_a.append(tf_c[feat]); cs_b.append(tf_s[feat])
                cm_a.append(tf_c[feat]); cm_b.append(tf_m[feat])
                ms_a.append(tf_m[feat]); ms_b.append(tf_s[feat])

        if cs_a:
            d_cs = cohens_d_z(cs_a, cs_b)
            d_cm = cohens_d_z(cm_a, cm_b)
            d_ms = cohens_d_z(ms_a, ms_b)

        text_d_results[feat] = {"C-S": d_cs, "C-M": d_cm, "M-S": d_ms}

        print(f"{feat:<25} {np.mean(cog_vals):>8.3f} {np.mean(meta_vals):>8.3f} "
              f"{np.mean(self_vals):>8.3f} {d_cs:>8.3f} {d_cm:>8.3f} {d_ms:>8.3f}")

    # Key question: does text predict as well as geometry?
    print("\n--- Key Question: Text vs Geometry Discriminability ---")

    # Simple linear discriminant: use all text features to predict condition
    # Compare max text |d_z| vs max geometry |d_z| for each pair
    geometry_features = ["eff_rank", "spectral_entropy", "key_norm", "norm_per_token",
                         "top_sv_ratio", "rank_10", "layer_variance", "layer_norm_entropy"]

    for pair_label, (c1, c2) in zip(["C-S", "C-M", "M-S"], condition_pairs_for_text):
        text_max_d = max(abs(text_d_results[f][pair_label]) for f in text_d_results)
        text_best = max(text_d_results, key=lambda f: abs(text_d_results[f][pair_label]))

        # Compute geometry d_z for same pair
        geo_ds = {}
        for feat in geometry_features:
            vals_c1, vals_c2 = [], []
            for topic_id in sorted(grouped.keys()):
                if c1 in grouped[topic_id] and c2 in grouped[topic_id]:
                    v1 = grouped[topic_id][c1]["generation_features"].get(feat)
                    v2 = grouped[topic_id][c2]["generation_features"].get(feat)
                    if v1 is not None and v2 is not None:
                        vals_c1.append(v1)
                        vals_c2.append(v2)
            if vals_c1:
                geo_ds[feat] = abs(cohens_d_z(vals_c1, vals_c2))

        geo_max_d = max(geo_ds.values()) if geo_ds else 0
        geo_best = max(geo_ds, key=lambda f: geo_ds[f]) if geo_ds else "N/A"

        print(f"\n  {pair_label} ({c1[:4]} vs {c2[:4]}):")
        print(f"    Text best:     |d_z| = {text_max_d:.3f} ({text_best})")
        print(f"    Geometry best: |d_z| = {geo_max_d:.3f} ({geo_best})")

        if text_max_d > geo_max_d * 1.2:
            print(f"    → TEXT DOMINATES — signal may be content-driven")
        elif geo_max_d > text_max_d * 1.2:
            print(f"    → GEOMETRY DOMINATES — signal beyond text content")
        else:
            print(f"    → COMPARABLE — both capture similar information")

    return text_d_results


# ── Analysis 3: Bloom-Level Classification ────────────────────────────────

def bloom_indicators(text):
    """Score text for Bloom's taxonomy cognitive levels."""
    words = text.split()
    n_words = max(len(words), 1)

    # Level 1-2: Knowledge/Comprehension (factual recall, definition)
    knowledge = len(re.findall(
        r'\b(is|are|was|were|defined|known|called|refers|consists|'
        r'includes|such as|for example|described|made up of|composed)\b',
        text, re.IGNORECASE
    )) / n_words

    # Level 3: Application (how-to, procedure, use)
    application = len(re.findall(
        r'\b(use[ds]?|apply|implement|demonstrate|solve|calculate|'
        r'practice|execute|perform|employ|operate)\b',
        text, re.IGNORECASE
    )) / n_words

    # Level 4: Analysis (compare, contrast, examine, relate)
    analysis = len(re.findall(
        r'\b(because|therefore|however|although|whereas|compared|'
        r'contrast|relationship|distinguish|analyze|examine|'
        r'influence|factor|cause|effect|difference|similarity)\b',
        text, re.IGNORECASE
    )) / n_words

    # Level 5: Synthesis/Evaluation (judge, assess, argue, create)
    evaluation = len(re.findall(
        r'\b(important|significant|crucial|essential|remarkable|'
        r'fascinating|compelling|overlooked|underappreciated|'
        r'argue|assess|evaluate|judge|recommend|prioritize|'
        r'implications?|perspective|interpretation|nuance[ds]?)\b',
        text, re.IGNORECASE
    )) / n_words

    # Weighted Bloom score (higher = more complex cognition)
    bloom_score = (knowledge * 1 + application * 2 +
                   analysis * 3 + evaluation * 4)

    return {
        "knowledge_rate": knowledge,
        "application_rate": application,
        "analysis_rate": analysis,
        "evaluation_rate": evaluation,
        "bloom_score": bloom_score,
    }


def analysis_bloom_levels(trials, grouped):
    """Compare Bloom cognitive levels across conditions."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Bloom-Level Cognitive Complexity")
    print("=" * 70)

    condition_bloom = defaultdict(list)
    for t in trials:
        bi = bloom_indicators(t["response_text"])
        condition_bloom[t["condition"]].append(bi)

    bloom_feats = ["knowledge_rate", "application_rate", "analysis_rate",
                   "evaluation_rate", "bloom_score"]

    print(f"\n{'Bloom Level':<22} {'Cog':>10} {'Meta':>10} {'Self':>10} {'Ordering':>18}")
    print("-" * 75)

    for feat in bloom_feats:
        cog_m = np.mean([b[feat] for b in condition_bloom["cognitive"]])
        meta_m = np.mean([b[feat] for b in condition_bloom["metacognitive"]])
        self_m = np.mean([b[feat] for b in condition_bloom["selfref"]])

        vals = [("Cog", cog_m), ("Meta", meta_m), ("Self", self_m)]
        ordering = " > ".join(v[0] for v in sorted(vals, key=lambda x: -x[1]))

        print(f"{feat:<22} {cog_m:>10.5f} {meta_m:>10.5f} {self_m:>10.5f} {ordering:>18}")

    # Paired d_z for bloom_score
    print("\n--- Bloom Score Paired Comparisons ---")
    condition_pairs = [("cognitive", "selfref"), ("cognitive", "metacognitive"),
                       ("metacognitive", "selfref")]

    for c1, c2 in condition_pairs:
        vals_c1, vals_c2 = [], []
        for topic_id in sorted(grouped.keys()):
            if c1 in grouped[topic_id] and c2 in grouped[topic_id]:
                b1 = bloom_indicators(grouped[topic_id][c1]["response_text"])
                b2 = bloom_indicators(grouped[topic_id][c2]["response_text"])
                vals_c1.append(b1["bloom_score"])
                vals_c2.append(b2["bloom_score"])

        if vals_c1:
            d = cohens_d_z(vals_c1, vals_c2)
            p = permutation_test_paired(vals_c1, vals_c2, n_perm=5000)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {c1[:4]} vs {c2[:4]}: d_z = {d:.3f}, p = {p:.4f} ({sig})")

    # Key prediction: SelfRef should have highest bloom_score (most evaluation language)
    cog_bloom = np.mean([b["bloom_score"] for b in condition_bloom["cognitive"]])
    meta_bloom = np.mean([b["bloom_score"] for b in condition_bloom["metacognitive"]])
    self_bloom = np.mean([b["bloom_score"] for b in condition_bloom["selfref"]])

    print(f"\n  Prediction: SelfRef > Meta > Cog for bloom_score")
    if self_bloom > meta_bloom > cog_bloom:
        print(f"  → CONFIRMED: Self ({self_bloom:.5f}) > Meta ({meta_bloom:.5f}) > Cog ({cog_bloom:.5f})")
    else:
        ordering = sorted([("Cog", cog_bloom), ("Meta", meta_bloom), ("Self", self_bloom)],
                         key=lambda x: -x[1])
        print(f"  → NOT CONFIRMED: {ordering[0][0]} ({ordering[0][1]:.5f}) > "
              f"{ordering[1][0]} ({ordering[1][1]:.5f}) > {ordering[2][0]} ({ordering[2][1]:.5f})")

    return condition_bloom


# ── Analysis 4: Response Diversity ────────────────────────────────────────

def analysis_response_diversity(trials, grouped):
    """Compare response variability across conditions."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Response Diversity")
    print("=" * 70)

    # Text diversity metrics per condition
    print("\n--- Text Diversity ---")

    condition_texts = defaultdict(list)
    for t in trials:
        condition_texts[t["condition"]].append(t["response_text"])

    print(f"\n{'Metric':<30} {'Cog':>10} {'Meta':>10} {'Self':>10}")
    print("-" * 65)

    for cond_label, cond_name in [("Cog", "cognitive"), ("Meta", "metacognitive"), ("Self", "selfref")]:
        pass  # Will print in table format below

    # Word count variance
    for metric_name, metric_fn in [
        ("Word count (mean ± sd)", lambda t: len(t.split())),
        ("Unique word ratio (mean ± sd)", lambda t: len(set(t.lower().split())) / max(len(t.split()), 1)),
        ("Sentence count (mean ± sd)", lambda t: len([s for s in re.split(r'[.!?]+', t) if s.strip()])),
    ]:
        vals = {}
        for cond in ["cognitive", "metacognitive", "selfref"]:
            v = [metric_fn(t) for t in condition_texts[cond]]
            vals[cond] = (np.mean(v), np.std(v, ddof=1))

        print(f"{metric_name:<30} "
              f"{vals['cognitive'][0]:>5.1f}±{vals['cognitive'][1]:>4.1f} "
              f"{vals['metacognitive'][0]:>5.1f}±{vals['metacognitive'][1]:>4.1f} "
              f"{vals['selfref'][0]:>5.1f}±{vals['selfref'][1]:>4.1f}")

    # Geometric feature variance per condition
    print("\n--- Geometric Feature Variance (CV = std/mean) ---")

    features = ["eff_rank", "spectral_entropy", "key_norm", "norm_per_token",
                 "top_sv_ratio", "rank_10", "layer_variance", "layer_norm_entropy"]

    print(f"\n{'Feature':<22} {'Cog CV':>10} {'Meta CV':>10} {'Self CV':>10} {'Self>Cog?':>10}")
    print("-" * 67)

    diversity_results = {}
    for feat in features:
        cvs = {}
        for cond in ["cognitive", "metacognitive", "selfref"]:
            vals = [t["generation_features"][feat] for t in trials
                    if t["condition"] == cond and feat in t["generation_features"]]
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            cvs[cond] = std / abs(mean) if abs(mean) > 1e-10 else 0

        more_diverse = "YES" if cvs["selfref"] > cvs["cognitive"] else "no"
        diversity_results[feat] = cvs

        print(f"{feat:<22} {cvs['cognitive']:>10.4f} {cvs['metacognitive']:>10.4f} "
              f"{cvs['selfref']:>10.4f} {more_diverse:>10}")

    # Summary: how many features show SelfRef as most diverse?
    self_most_diverse = sum(1 for f in diversity_results
                           if diversity_results[f]["selfref"] == max(diversity_results[f].values()))
    print(f"\n  SelfRef most diverse in {self_most_diverse}/{len(features)} features")

    return diversity_results


# ── FWL Residualization ──────────────────────────────────────────────────

def fwl_residualize(trials, grouped):
    """Apply FWL token-count residualization and recompute effects."""
    print("\n" + "=" * 70)
    print("FWL RESIDUALIZATION CHECK")
    print("=" * 70)

    features = ["eff_rank", "spectral_entropy", "key_norm", "norm_per_token",
                 "top_sv_ratio", "rank_10", "layer_variance", "layer_norm_entropy"]

    # Collect all generation features + token counts
    all_feats = {f: [] for f in features}
    all_tokens = []
    all_conditions = []
    all_topics = []

    for t in trials:
        for f in features:
            all_feats[f].append(t["generation_features"].get(f, 0))
        all_tokens.append(t["generation_features"].get("n_tokens", 0))
        all_conditions.append(t["condition"])
        all_topics.append(t["topic_id"])

    all_tokens = np.array(all_tokens, dtype=float)

    # Residualize each feature on token count
    residuals = {}
    for f in features:
        y = np.array(all_feats[f], dtype=float)
        # OLS: y = a + b*tokens + residual
        X = np.column_stack([np.ones(len(all_tokens)), all_tokens])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals[f] = y - X @ beta

    # Now compute paired d_z on residuals
    condition_pairs = [("cognitive", "selfref"), ("cognitive", "metacognitive"),
                       ("metacognitive", "selfref")]

    print(f"\n{'Pair':<18} {'Feature':<22} {'Raw d_z':>10} {'FWL d_z':>10} {'Survived?':>10}")
    print("-" * 75)

    for c1, c2 in condition_pairs:
        for f in features:
            # Raw
            raw_c1, raw_c2 = [], []
            fwl_c1, fwl_c2 = [], []

            for topic_id in sorted(grouped.keys()):
                if c1 in grouped[topic_id] and c2 in grouped[topic_id]:
                    idx1 = next(i for i, t in enumerate(trials)
                                if t["topic_id"] == topic_id and t["condition"] == c1)
                    idx2 = next(i for i, t in enumerate(trials)
                                if t["topic_id"] == topic_id and t["condition"] == c2)

                    raw_c1.append(all_feats[f][idx1])
                    raw_c2.append(all_feats[f][idx2])
                    fwl_c1.append(residuals[f][idx1])
                    fwl_c2.append(residuals[f][idx2])

            if raw_c1:
                d_raw = cohens_d_z(raw_c1, raw_c2)
                d_fwl = cohens_d_z(fwl_c1, fwl_c2)
                survived = "YES" if abs(d_fwl) > abs(d_raw) * 0.5 else "DROPS"

                # Only print significant or large effects
                if abs(d_raw) > 0.3 or abs(d_fwl) > 0.3:
                    print(f"{c1[:4]}-{c2[:4]:<12} {f:<22} {d_raw:>10.3f} {d_fwl:>10.3f} {survived:>10}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    models = []
    for m in ["qwen", "llama"]:
        mdir = os.path.join(args.results_dir, m)
        if os.path.isdir(mdir):
            models.append(m)

    if not models:
        print(f"ERROR: No model directories found in {args.results_dir}")
        sys.exit(1)

    all_results = {}

    for model in models:
        print("\n" + "#" * 70)
        print(f"# MODEL: {model.upper()}")
        print("#" * 70)

        trials = load_trials(args.results_dir, model)
        print(f"Loaded {len(trials)} trials for {model}")

        grouped = group_by_topic_condition(trials)
        print(f"Topics: {len(grouped)}, complete triplets: "
              f"{sum(1 for g in grouped.values() if len(g) == 3)}")

        r1 = analysis_encoding_vs_generation(trials, grouped)
        r2 = analysis_text_baseline(trials, grouped)
        r3 = analysis_bloom_levels(trials, grouped)
        r4 = analysis_response_diversity(trials, grouped)
        fwl_residualize(trials, grouped)

        all_results[model] = {"enc_vs_gen": r1, "text_baseline": r2,
                               "bloom": r3, "diversity": r4}

    # Cross-model comparison
    if len(models) == 2:
        print("\n" + "#" * 70)
        print("# CROSS-MODEL COMPARISON")
        print("#" * 70)

        print("\nDo both models show the same condition ordering?")
        for model in models:
            trials = load_trials(args.results_dir, model)
            grouped = group_by_topic_condition(trials)

            # spectral_entropy d_z for C vs S
            vals_c, vals_s = [], []
            for topic_id in sorted(grouped.keys()):
                if "cognitive" in grouped[topic_id] and "selfref" in grouped[topic_id]:
                    vals_c.append(grouped[topic_id]["cognitive"]["generation_features"]["spectral_entropy"])
                    vals_s.append(grouped[topic_id]["selfref"]["generation_features"]["spectral_entropy"])

            d = cohens_d_z(vals_c, vals_s) if vals_c else 0
            print(f"  {model}: Cog vs SelfRef spectral_entropy d_z = {d:.3f}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
