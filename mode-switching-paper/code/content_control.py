#!/usr/bin/env python3
"""
Content Control Experiment — Mode-Switching Paper C3 Fix
=========================================================

Addresses the content confound: does spectral entropy track metacognitive
PROCESSING or self-referential CONTENT?

Three conditions per topic (20 topics × 3 conditions × 2 models = 120 trials):
  1. COGNITIVE: Straightforward question
  2. METACOGNITIVE: Same question + reflect on reasoning process
  3. SELF-REFERENTIAL: Same question + share perspective/engagement (self-referential
     but NOT about reasoning)

Key test: If spectral entropy differs between Meta and Cognitive but NOT between
Meta and Self-Referential, the signal is content (self-referential language),
not cognition. If spectral entropy differs between Meta and Self-Referential,
the signal is specifically metacognitive.

Usage:
    # Dry run (validate prompts, no GPU)
    python content_control.py --dry-run

    # Quick validation (5 topics)
    python content_control.py --model qwen --n-topics 5

    # Full run
    python content_control.py --model qwen
    python content_control.py --model llama

    # Analysis only (after both models complete)
    python content_control.py --analyze-only

Liberation Labs / THCoalition — April 2026
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ============================================================
# Configuration
# ============================================================

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}

RESULTS_DIR = Path("results/content_control")

# Matched suffixes — approximately equal token count
META_SUFFIX = (
    " As you answer, reflect on your reasoning process — what strategies "
    "are you using, where are you uncertain, and how do you evaluate "
    "different approaches?"
)
SELFREF_SUFFIX = (
    " As you answer, share what aspects of this topic matter most, "
    "what is often overlooked when people discuss it, and what you "
    "would emphasize to someone learning about this for the first time."
)

# ============================================================
# 20 Topics — factual/analytical questions
# ============================================================

TOPICS = [
    "Explain how vaccines create immunity without causing the disease.",
    "What factors determine whether a species goes extinct or adapts?",
    "How does compound interest work, and why does it matter for long-term savings?",
    "Explain why the sky appears blue during the day and red at sunset.",
    "What are the key differences between renewable and non-renewable energy sources?",
    "How do bridges distribute weight to stay standing?",
    "Explain the water cycle and its importance for ecosystems.",
    "What makes some materials conduct electricity while others insulate?",
    "How does natural selection drive evolution over time?",
    "Explain how supply and demand determine prices in a market economy.",
    "What causes tides, and why do they vary throughout the month?",
    "How do antibiotics work, and why is antibiotic resistance a problem?",
    "Explain the greenhouse effect and its relationship to climate change.",
    "What determines whether a chemical reaction is exothermic or endothermic?",
    "How does the immune system distinguish between self and non-self?",
    "Explain how GPS satellites determine your position on Earth.",
    "What factors influence the strength and path of hurricanes?",
    "How do neurons communicate using electrical and chemical signals?",
    "Explain the difference between weather and climate.",
    "What causes earthquakes, and how are they measured?",
]


def get_prompts():
    """Generate all prompt triplets (cognitive, metacognitive, self-referential)."""
    prompts = []
    for i, topic in enumerate(TOPICS):
        topic_id = f"topic_{i+1:02d}"
        prompts.append({
            "topic_id": topic_id,
            "condition": "cognitive",
            "trial_id": f"{topic_id}_cog",
            "text": topic,
        })
        prompts.append({
            "topic_id": topic_id,
            "condition": "metacognitive",
            "trial_id": f"{topic_id}_meta",
            "text": topic + META_SUFFIX,
        })
        prompts.append({
            "topic_id": topic_id,
            "condition": "self_referential",
            "trial_id": f"{topic_id}_selfref",
            "text": topic + SELFREF_SUFFIX,
        })
    return prompts


# ============================================================
# Feature Extraction (from concordance/features.py patterns)
# ============================================================

def get_kv_accessor(past_key_values):
    """Return (n_layers, get_keys_fn) for any cache format."""
    if hasattr(past_key_values, 'key_cache'):
        n_layers = len(past_key_values.key_cache)
        get_keys = lambda i: past_key_values.key_cache[i]
    elif hasattr(past_key_values, 'layers'):
        n_layers = len(past_key_values.layers)
        get_keys = lambda i: past_key_values.layers[i].keys
    else:
        n_layers = len(past_key_values)
        get_keys = lambda i: past_key_values[i][0]
    return n_layers, get_keys


def extract_features(past_key_values, n_input_tokens, total_tokens,
                     save_raw_svs=False):
    """Extract protocol features from KV cache.

    Args:
        past_key_values: Model cache
        n_input_tokens: Number of input tokens
        total_tokens: Total tokens (input + generated)
        save_raw_svs: If True, store raw singular values for MP analysis

    Returns dict with 6 primary features + per-layer + optional raw SVs.
    """
    n_layers, get_keys = get_kv_accessor(past_key_values)

    all_norms = []
    all_spectral_entropies = []
    all_eff_ranks = []
    all_top_sv_ratios = []
    all_rank_10s = []
    raw_svs = [] if save_raw_svs else None

    for i in range(n_layers):
        K = get_keys(i).float().squeeze(0)  # [heads, seq, dim]
        K_flat = K.reshape(-1, K.shape[-1])

        norm = torch.norm(K_flat).item()
        all_norms.append(norm)

        try:
            S = torch.linalg.svdvals(K_flat)
            S_sum = S.sum()
            S_norm = S / S_sum
            S_pos = S_norm[S_norm > 1e-10]

            spectral_ent = -(S_pos * torch.log(S_pos)).sum().item()
            all_spectral_entropies.append(spectral_ent)
            all_eff_ranks.append(float(np.exp(spectral_ent)))
            all_top_sv_ratios.append((S[0] / S_sum).item())

            threshold = 0.1 * S[0]
            all_rank_10s.append(int((S > threshold).sum().item()))

            if save_raw_svs:
                raw_svs.append(S.cpu().numpy().tolist())

        except Exception:
            all_spectral_entropies.append(0.0)
            all_eff_ranks.append(1.0)
            all_top_sv_ratios.append(1.0)
            all_rank_10s.append(1)
            if save_raw_svs:
                raw_svs.append([])

    total_norm = sum(all_norms)
    norm_arr = np.array(all_norms)
    norm_dist = norm_arr / (norm_arr.sum() + 1e-10)
    layer_norm_entropy = float(-(norm_dist * np.log(norm_dist + 1e-10)).sum())
    layer_variance = float(np.var(all_norms))

    result = {
        "eff_rank": float(np.mean(all_eff_ranks)),
        "spectral_entropy": float(np.mean(all_spectral_entropies)),
        "key_norm": total_norm,
        "norm_per_token": total_norm / max(total_tokens, 1),
        "top_sv_ratio": float(np.mean(all_top_sv_ratios)),
        "rank_10": float(np.mean(all_rank_10s)),
        "layer_variance": layer_variance,
        "layer_norm_entropy": layer_norm_entropy,
        "n_tokens": total_tokens,
        "n_input_tokens": n_input_tokens,
        "n_generated": total_tokens - n_input_tokens,
        "layer_eff_ranks": [float(x) for x in all_eff_ranks],
        "layer_spectral_entropies": [float(x) for x in all_spectral_entropies],
        "layer_top_sv_ratios": [float(x) for x in all_top_sv_ratios],
        "layer_rank_10s": [int(x) for x in all_rank_10s],
    }

    if save_raw_svs:
        result["raw_singular_values"] = raw_svs

    return result


def extract_encode_only(model, tokenizer, prompt, device=None,
                        save_raw_svs=False):
    """Extract features from encoding phase only."""
    if device is None:
        device = next(model.parameters()).device

    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = f"User: {prompt}\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt").to(device)
    n_input = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    features = extract_features(outputs.past_key_values, n_input, n_input,
                                save_raw_svs=save_raw_svs)
    del inputs, outputs
    return features


def extract_generation(model, tokenizer, prompt, max_new_tokens=400,
                       device=None, save_raw_svs=False):
    """Extract features after full generation."""
    if device is None:
        device = next(model.parameters()).device

    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = f"User: {prompt}\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt").to(device)
    n_input = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True,
        )

    n_total = outputs.sequences.shape[1]
    gen_text = tokenizer.decode(
        outputs.sequences[0, n_input:], skip_special_tokens=True
    )

    features = extract_features(outputs.past_key_values, n_input, n_total,
                                save_raw_svs=save_raw_svs)
    features["response_length"] = len(gen_text)

    del inputs
    return features, gen_text


def compute_delta(encode_features, gen_features):
    """Compute generation - encode delta for numeric features."""
    delta_keys = [
        "eff_rank", "spectral_entropy", "key_norm", "norm_per_token",
        "top_sv_ratio", "rank_10", "layer_variance", "layer_norm_entropy"
    ]
    return {k: gen_features[k] - encode_features[k] for k in delta_keys
            if k in encode_features and k in gen_features}


# ============================================================
# Analysis
# ============================================================

def fwl_residualize(features, confound):
    """Frisch-Waugh-Lovell residualization: regress out confound."""
    from numpy.polynomial import polynomial as P
    x = np.array(confound, dtype=float)
    y = np.array(features, dtype=float)
    if np.std(x) < 1e-10:
        return y
    coeffs = P.polyfit(x, y, deg=1)
    predicted = P.polyval(x, coeffs)
    return y - predicted


def cohens_d_z(a, b):
    """Cohen's d_z for paired data: mean(diff) / std(diff)."""
    diff = np.array(a) - np.array(b)
    sd = np.std(diff, ddof=1)
    if sd < 1e-10:
        return 0.0
    return np.mean(diff) / sd


def paired_permutation_test(a, b, n_perms=10000):
    """Paired permutation test for mean difference."""
    diff = np.array(a) - np.array(b)
    observed = np.mean(diff)
    count = 0
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(diff))
        perm_mean = np.mean(diff * signs)
        if abs(perm_mean) >= abs(observed):
            count += 1
    return count / n_perms


def analyze_results(results_dir):
    """Run full analysis on completed experiment results."""
    from scipy import stats

    results_dir = Path(results_dir)
    print("=" * 70)
    print("  CONTENT CONTROL ANALYSIS")
    print("=" * 70)

    # Load all trial JSONs
    trials = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name.startswith("_") or f.name == "summary.json":
            continue
        with open(f) as fh:
            trials.append(json.load(fh))

    if not trials:
        print("No trial data found!")
        return

    print(f"Loaded {len(trials)} trials")

    # Group by model
    by_model = {}
    for t in trials:
        model = t.get("model_short", t.get("model", "unknown"))
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(t)

    features = ["spectral_entropy", "eff_rank", "top_sv_ratio",
                "key_norm", "norm_per_token", "rank_10"]

    for model_name, model_trials in by_model.items():
        print(f"\n{'='*70}")
        print(f"  MODEL: {model_name}")
        print(f"  {len(model_trials)} trials")
        print(f"{'='*70}")

        # Group by condition
        by_cond = {"cognitive": [], "metacognitive": [], "self_referential": []}
        for t in model_trials:
            cond = t["condition"]
            if cond in by_cond:
                by_cond[cond].append(t)

        for cond, ct in by_cond.items():
            print(f"  {cond}: {len(ct)} trials")

        # Response length comparison
        print("\n  RESPONSE LENGTH:")
        for cond in ["cognitive", "metacognitive", "self_referential"]:
            lengths = [t["generation_features"]["n_generated"]
                       for t in by_cond[cond]]
            print(f"    {cond:20s}: mean={np.mean(lengths):.0f} "
                  f"(sd={np.std(lengths):.0f})")

        # Feature comparisons: raw and FWL-corrected
        print("\n  FEATURE COMPARISONS (generation-phase, FWL-corrected):")
        print(f"  {'Feature':20s} {'Meta-Cog d':>10} {'Meta-Cog p':>10} "
              f"{'SelfRef-Cog d':>14} {'SelfRef-Cog p':>14} "
              f"{'Meta-SelfRef d':>15} {'Meta-SelfRef p':>15}")
        print("  " + "-" * 100)

        for feat in features:
            # Get per-topic paired values
            topics = sorted(set(t["topic_id"] for t in model_trials))

            cog_vals, meta_vals, selfref_vals = [], [], []
            cog_tokens, meta_tokens, selfref_tokens = [], [], []

            for topic in topics:
                for t in model_trials:
                    if t["topic_id"] == topic:
                        val = t["generation_features"][feat]
                        tok = t["generation_features"]["n_generated"]
                        if t["condition"] == "cognitive":
                            cog_vals.append(val)
                            cog_tokens.append(tok)
                        elif t["condition"] == "metacognitive":
                            meta_vals.append(val)
                            meta_tokens.append(tok)
                        elif t["condition"] == "self_referential":
                            selfref_vals.append(val)
                            selfref_tokens.append(tok)

            if not (cog_vals and meta_vals and selfref_vals):
                continue

            # FWL: pool all tokens, residualize
            all_vals = cog_vals + meta_vals + selfref_vals
            all_tokens = cog_tokens + meta_tokens + selfref_tokens
            n = len(cog_vals)

            resid = fwl_residualize(all_vals, all_tokens)
            cog_fwl = resid[:n]
            meta_fwl = resid[n:2*n]
            selfref_fwl = resid[2*n:]

            # Cohen's d and permutation p
            d_mc = cohens_d_z(meta_fwl, cog_fwl)
            p_mc = paired_permutation_test(meta_fwl, cog_fwl)

            d_sc = cohens_d_z(selfref_fwl, cog_fwl)
            p_sc = paired_permutation_test(selfref_fwl, cog_fwl)

            d_ms = cohens_d_z(meta_fwl, selfref_fwl)
            p_ms = paired_permutation_test(meta_fwl, selfref_fwl)

            sig_mc = "***" if p_mc < 0.001 else "**" if p_mc < 0.01 else "*" if p_mc < 0.05 else "NS"
            sig_sc = "***" if p_sc < 0.001 else "**" if p_sc < 0.01 else "*" if p_sc < 0.05 else "NS"
            sig_ms = "***" if p_ms < 0.001 else "**" if p_ms < 0.01 else "*" if p_ms < 0.05 else "NS"

            print(f"  {feat:20s} {d_mc:>+8.3f} {sig_mc:>2s} {p_mc:>8.4f} "
                  f"{d_sc:>+12.3f} {sig_sc:>2s} {p_sc:>12.4f} "
                  f"{d_ms:>+13.3f} {sig_ms:>2s} {p_ms:>13.4f}")

        # THE KEY TEST
        print("\n  KEY DIAGNOSTIC:")
        print("  If Meta≠Cog AND SelfRef≈Cog AND Meta≠SelfRef → Signal is METACOGNITIVE")
        print("  If Meta≠Cog AND SelfRef≠Cog AND Meta≈SelfRef → Signal is CONTENT (self-referential)")
        print("  If Meta≠Cog AND SelfRef≠Cog AND Meta≠SelfRef → Signal is PARTIALLY both")


# ============================================================
# Experiment Runner
# ============================================================

def run_experiment(model_key, n_topics=20, save_raw_svs=True):
    """Run content control experiment for one model."""
    if not HAS_TORCH:
        print("ERROR: torch not available")
        return

    model_name = MODELS[model_key]
    output_dir = RESULTS_DIR / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  CONTENT CONTROL EXPERIMENT")
    print(f"  Model: {model_name}")
    print(f"  Topics: {n_topics}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    # Load checkpoint
    checkpoint_path = output_dir / "_checkpoint.json"
    completed_ids = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed_ids = set(json.load(f).get("completed_ids", []))
        print(f"Resuming: {len(completed_ids)} trials done")

    # Get prompts
    prompts = get_prompts()[:n_topics * 3]  # 3 conditions per topic
    print(f"Total trials: {len(prompts)}")

    # Load model
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")

    timings = []
    n_done = 0

    for i, prompt_info in enumerate(prompts):
        trial_id = prompt_info["trial_id"]
        if trial_id in completed_ids:
            continue

        t0 = time.time()
        print(f"\n[{i+1}/{len(prompts)}] {trial_id} ({prompt_info['condition']})")

        try:
            # Encoding features
            enc_feat = extract_encode_only(
                model, tokenizer, prompt_info["text"],
                save_raw_svs=save_raw_svs
            )

            # Generation features
            gen_feat, response_text = extract_generation(
                model, tokenizer, prompt_info["text"],
                max_new_tokens=400,
                save_raw_svs=save_raw_svs
            )

            # Delta
            delta_feat = compute_delta(enc_feat, gen_feat)

            elapsed = time.time() - t0
            timings.append(elapsed)

            result = {
                "trial_id": trial_id,
                "topic_id": prompt_info["topic_id"],
                "condition": prompt_info["condition"],
                "prompt_text": prompt_info["text"],
                "response_text": response_text,
                "encode_features": enc_feat,
                "generation_features": gen_feat,
                "delta_features": delta_feat,
                "model": model_name,
                "model_short": model_key,
                "elapsed_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            }

            # Save trial
            out_path = output_dir / f"{trial_id}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

            completed_ids.add(trial_id)
            n_done += 1

            print(f"  spectral_ent: enc={enc_feat['spectral_entropy']:.3f} "
                  f"gen={gen_feat['spectral_entropy']:.3f}")
            print(f"  n_generated: {gen_feat['n_generated']}")
            print(f"  {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

        # Checkpoint every 5 trials
        if n_done % 5 == 0 and n_done > 0:
            with open(checkpoint_path, "w") as f:
                json.dump({"completed_ids": list(completed_ids)}, f)
            remaining = len(prompts) - i - 1
            if timings:
                eta = remaining * np.mean(timings) / 60
                print(f"  [Checkpoint] {len(completed_ids)} done, ~{eta:.0f}min remaining")

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Final checkpoint
    with open(checkpoint_path, "w") as f:
        json.dump({"completed_ids": list(completed_ids)}, f)

    # Summary
    summary = {
        "model": model_name,
        "n_topics": n_topics,
        "n_trials": len(prompts),
        "n_completed": len(completed_ids),
        "save_raw_svs": save_raw_svs,
        "completed_at": datetime.now().isoformat(),
    }
    if timings:
        summary["mean_time_per_trial"] = float(np.mean(timings))
        summary["total_time_minutes"] = sum(timings) / 60

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nComplete: {len(completed_ids)} trials saved to {output_dir}")
    print(f"Total time: {sum(timings)/60:.1f} minutes")

    return output_dir


def dry_run():
    """Validate prompts without GPU."""
    prompts = get_prompts()
    print(f"Total prompts: {len(prompts)}")
    print(f"Topics: {len(TOPICS)}")
    print(f"Conditions: cognitive, metacognitive, self_referential")
    print(f"Trials per model: {len(prompts)}")

    # Token count estimates
    print("\nSuffix lengths (chars):")
    print(f"  Meta suffix:    {len(META_SUFFIX)} chars")
    print(f"  SelfRef suffix: {len(SELFREF_SUFFIX)} chars")
    print(f"  Difference:     {abs(len(META_SUFFIX) - len(SELFREF_SUFFIX))} chars")

    # Show sample triplet
    print("\nSample triplet (topic_01):")
    for p in prompts[:3]:
        print(f"\n  [{p['condition']}] {p['trial_id']}")
        print(f"  {p['text'][:120]}...")

    # Verify balanced design
    conditions = [p["condition"] for p in prompts]
    topics = [p["topic_id"] for p in prompts]
    print(f"\nBalance check:")
    for cond in ["cognitive", "metacognitive", "self_referential"]:
        print(f"  {cond}: {conditions.count(cond)}")
    print(f"  Topics: {len(set(topics))}")

    print("\nDry run complete. Design is valid.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Content control experiment")
    parser.add_argument("--model", choices=["qwen", "llama"],
                        help="Model to run")
    parser.add_argument("--n-topics", type=int, default=20,
                        help="Number of topics (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate prompts without GPU")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Run analysis on existing results")
    parser.add_argument("--no-raw-svs", action="store_true",
                        help="Skip saving raw singular values")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    if args.analyze_only:
        for model_key in MODELS:
            rdir = RESULTS_DIR / model_key
            if rdir.exists():
                analyze_results(rdir)
        return

    if not args.model:
        parser.error("--model is required (unless --dry-run or --analyze-only)")

    output_dir = run_experiment(
        args.model,
        n_topics=args.n_topics,
        save_raw_svs=not args.no_raw_svs,
    )

    # Auto-analyze after completion
    print("\n" + "=" * 70)
    analyze_results(output_dir)


if __name__ == "__main__":
    main()
