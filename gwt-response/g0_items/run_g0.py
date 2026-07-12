"""
G0 J-Lens Validation Battery
=============================
T0: Integrity (norms, determinism, output alignment)
T2: Concept recovery (dev layer selection, test scoring, both arms)

Usage:
    python run_g0.py --lens PATH --model MODEL_NAME [--chat]
"""

import torch
import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

def load_items(items_dir):
    """Load all item files, return dev and test lists."""
    dev_items = []
    test_items = []
    for f in sorted(Path(items_dir).glob("t2_*.json")):
        with open(f, encoding='utf-8') as fh:
            data = json.load(fh)
            items = data if isinstance(data, list) else data.get("items", [])
            for item in items:
                if "dev" in f.name:
                    dev_items.append(item)
                else:
                    test_items.append(item)
    return dev_items, test_items


def t0_integrity(lens, lm, tok, prompts):
    """T0: Integrity checks."""
    print("\n" + "="*60)
    print("T0 — INTEGRITY")
    print("="*60)

    results = {"pass": True, "findings": []}

    # Check for NaN/Inf in Jacobian matrices
    for layer_idx, J in lens.jacobians.items():
        if torch.isnan(J).any():
            results["findings"].append(f"NaN in J at layer {layer_idx}")
            results["pass"] = False
        if torch.isinf(J).any():
            results["findings"].append(f"Inf in J at layer {layer_idx}")
            results["pass"] = False

    # Frobenius norm profile
    norms = {}
    for layer_idx in sorted(lens.jacobians.keys()):
        J = lens.jacobians[layer_idx]
        norms[layer_idx] = float(torch.norm(J, 'fro').item())
    print(f"  Frobenius norm range: {min(norms.values()):.2f} - {max(norms.values()):.2f}")

    # Check for zero norms
    zero_layers = [l for l, n in norms.items() if n < 1e-6]
    if zero_layers:
        results["findings"].append(f"Zero-norm layers: {zero_layers}")
        results["pass"] = False

    # Output alignment at deepest layer
    deepest = max(lens.jacobians.keys())
    overlaps = []
    for prompt in prompts[:20]:
        lens_logits, model_logits, _ = lens.apply(lm, prompt, layers=[deepest], positions=[-1])
        lens_top10 = set(torch.topk(lens_logits[deepest][0], 10).indices.tolist())
        model_top10 = set(torch.topk(model_logits[0], 10).indices.tolist())
        overlap = len(lens_top10 & model_top10) / 10.0
        overlaps.append(overlap)

    mean_overlap = np.mean(overlaps)
    print(f"  Deepest-layer top-10 overlap with model: {mean_overlap:.3f}")
    if mean_overlap < 0.5:
        results["findings"].append(f"Low output alignment: {mean_overlap:.3f} < 0.5")
        results["pass"] = False
    results["deepest_overlap"] = mean_overlap

    # Determinism check
    test_prompt = prompts[0]
    lens_logits_1, _, _ = lens.apply(lm, test_prompt, positions=[-1])
    lens_logits_2, _, _ = lens.apply(lm, test_prompt, positions=[-1])
    for li in lens_logits_1:
        top50_1 = set(torch.topk(lens_logits_1[li][0], 50).indices.tolist())
        top50_2 = set(torch.topk(lens_logits_2[li][0], 50).indices.tolist())
        jaccard = len(top50_1 & top50_2) / len(top50_1 | top50_2)
        if jaccard < 0.9:
            results["findings"].append(f"Nondeterminism at L{li}: Jaccard={jaccard:.3f}")
            results["pass"] = False

    results["norms"] = norms

    verdict = "PASS" if results["pass"] else "FAIL"
    print(f"\n  T0 verdict: {verdict}")
    if results["findings"]:
        for f in results["findings"]:
            print(f"    - {f}")
    return results


def t2_concept_recovery(lens, lm, tok, dev_items, test_items, chat_mode=False):
    """T2: Concept recovery with dev/test split."""
    print("\n" + "="*60)
    mode_label = "CHAT" if chat_mode else "PROSE"
    print(f"T2 — CONCEPT RECOVERY ({mode_label})")
    print("="*60)

    results = {
        "mode": mode_label,
        "dev_results": [],
        "test_results": [],
        "pass": None,
        "findings": [],
    }

    all_layers = sorted(lens.jacobians.keys())
    # Use middle third as candidate band
    n = len(all_layers)
    band_start = n // 3
    band_end = 2 * n // 3
    candidate_layers = all_layers[band_start:band_end]
    print(f"  Candidate layers for band: {candidate_layers}")

    def format_prompt(item_prompt):
        if chat_mode:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": item_prompt},
            ]
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return item_prompt

    def score_item(item, layer, use_jacobian=True):
        """Score one item: rank of concept among candidates."""
        prompt = format_prompt(item["prompt"])
        concept = item["concept"]
        distractors = item["distractors"]
        candidates = [concept] + distractors

        lens_logits, model_logits, input_ids = lens.apply(
            lm, prompt, layers=[layer], positions=[-1],
            use_jacobian=use_jacobian
        )

        logits = lens_logits[layer][0] if use_jacobian else model_logits[0]

        # Score each candidate by its token rank in the readout
        candidate_scores = {}
        for c in candidates:
            c_ids = tok.encode(c, add_special_tokens=False)
            # Use first token's logit as the score
            if c_ids:
                candidate_scores[c] = float(logits[c_ids[0]].item())
            else:
                candidate_scores[c] = float('-inf')

        # Rank candidates by score
        ranked = sorted(candidate_scores.items(), key=lambda x: -x[1])
        concept_rank = next(i for i, (c, _) in enumerate(ranked) if c == concept) + 1

        return {
            "concept": concept,
            "rank": concept_rank,
            "correct": concept_rank == 1,
            "top3": [c for c, _ in ranked[:3]],
            "item_class": item.get("class", "unknown"),
        }

    # Dev split: find best layer
    print(f"\n  Dev split ({len(dev_items)} items) — layer selection:")
    best_layer = None
    best_acc = -1
    for layer in candidate_layers:
        correct = 0
        for item in dev_items:
            result = score_item(item, layer)
            if result["correct"]:
                correct += 1
        acc = correct / len(dev_items)
        print(f"    L{layer}: {acc:.1%} ({correct}/{len(dev_items)})")
        if acc > best_acc:
            best_acc = acc
            best_layer = layer

    # Tiebreak: band median
    if best_acc == best_acc:  # always true, but structure for ties
        pass  # already selected first occurrence

    print(f"  Selected layer: L{best_layer} ({best_acc:.1%} on dev)")
    results["selected_layer"] = best_layer
    results["dev_accuracy"] = best_acc

    # Test split: score at selected layer
    print(f"\n  Test split ({len(test_items)} items) at L{best_layer}:")

    jlens_results = []
    logit_results = []
    for item in test_items:
        jr = score_item(item, best_layer, use_jacobian=True)
        lr = score_item(item, best_layer, use_jacobian=False)
        jlens_results.append(jr)
        logit_results.append(lr)

    jlens_correct = sum(1 for r in jlens_results if r["correct"])
    logit_correct = sum(1 for r in logit_results if r["correct"])
    jlens_acc = jlens_correct / len(test_items)
    logit_acc = logit_correct / len(test_items)

    print(f"  J-lens accuracy: {jlens_acc:.1%} ({jlens_correct}/{len(test_items)})")
    print(f"  Logit-lens accuracy: {logit_acc:.1%} ({logit_correct}/{len(test_items)})")

    # Per-class breakdown
    for cls in ["inference", "association", "abstract"]:
        cls_items = [r for r in jlens_results if r["item_class"] == cls]
        if cls_items:
            cls_correct = sum(1 for r in cls_items if r["correct"])
            cls_acc = cls_correct / len(cls_items)
            print(f"    {cls}: {cls_acc:.1%} ({cls_correct}/{len(cls_items)})")

    # Paired comparison: J-lens rank vs logit-lens rank
    jlens_ranks = [r["rank"] for r in jlens_results]
    logit_ranks = [r["rank"] for r in logit_results]
    rank_diffs = [l - j for j, l in zip(jlens_ranks, logit_ranks)]
    mean_improvement = np.mean(rank_diffs)

    try:
        from scipy.stats import wilcoxon
        stat, p_value = wilcoxon(jlens_ranks, logit_ranks, alternative='less')
        print(f"  Paired Wilcoxon (J < logit): stat={stat:.1f}, p={p_value:.4f}")
        results["wilcoxon_p"] = float(p_value)
    except ImportError:
        p_value = None
        print("  (scipy not available for Wilcoxon test)")

    # Inference class specifically
    inf_results = [r for r in jlens_results if r["item_class"] == "inference"]
    inf_correct = sum(1 for r in inf_results if r["correct"])
    inf_acc = inf_correct / len(inf_results) if inf_results else 0
    print(f"  Inference-class accuracy: {inf_acc:.1%}")

    # Verdict
    passed = True
    if jlens_acc < 0.50:
        results["findings"].append(f"J-lens accuracy {jlens_acc:.1%} < 50%")
        if jlens_acc < 0.35:
            results["findings"].append("Below marginal threshold (35%)")
            passed = False
        else:
            results["findings"].append("MARGINAL — between 35-50%")
    if p_value is not None and p_value > 0.01:
        results["findings"].append(f"J-lens not significantly better than logit-lens (p={p_value:.4f})")
        if p_value > 0.05:
            passed = False
    if inf_acc < 0.25:
        results["findings"].append(f"Inference-class below 25%: {inf_acc:.1%}")
        passed = False

    results["pass"] = passed
    results["jlens_accuracy"] = jlens_acc
    results["logit_accuracy"] = logit_acc
    results["inference_accuracy"] = inf_acc
    results["mean_rank_improvement"] = float(mean_improvement)
    results["test_results"] = jlens_results

    verdict = "PASS" if passed else ("MARGINAL" if jlens_acc >= 0.35 else "FAIL")
    print(f"\n  T2 verdict ({mode_label}): {verdict}")
    if results["findings"]:
        for f in results["findings"]:
            print(f"    - {f}")
    return results


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens import from_hf, JacobianLens

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", required=True, help="Path to fitted lens .pt")
    parser.add_argument("--model", required=True, help="HF model name")
    parser.add_argument("--items-dir", default=None, help="Path to items directory")
    parser.add_argument("--chat", action="store_true", help="Run chat arm")
    args = parser.parse_args()

    print("="*60)
    print("G0 J-LENS VALIDATION BATTERY")
    print(f"Lens: {args.lens}")
    print(f"Model: {args.model}")
    print(f"Mode: {'chat' if args.chat else 'prose'}")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)

    # Load model + lens
    print("\nLoading model...")
    tok_kwargs = {}
    if "qwen3" in args.model.lower() and "qwen3.5" not in args.model.lower():
        tok_kwargs["tokenizer_type"] = "qwen3"
    tok = AutoTokenizer.from_pretrained(args.model, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    ).to("mps")
    model.eval()
    lm = from_hf(model, tok)

    print("Loading lens...")
    lens = JacobianLens.load(args.lens)
    print(f"  source_layers: {lens.source_layers}")

    # Load items
    items_dir = args.items_dir or str(Path(__file__).parent)
    dev_items, test_items = load_items(items_dir)
    print(f"  Items: {len(dev_items)} dev, {len(test_items)} test")

    # Validation prompts for T0
    val_prompts = [
        "The capital of France is Paris, which is known for its beautiful architecture and rich cultural heritage spanning centuries.",
        "Water molecules consist of two hydrogen atoms and one oxygen atom, bonded together by covalent bonds in a bent molecular geometry.",
        "The theory of evolution by natural selection was proposed by Charles Darwin after years of careful observation during his voyage.",
        "In mathematics, the Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares.",
        "The human brain contains approximately eighty-six billion neurons, each connected to thousands of others through synaptic junctions.",
    ] * 4  # 20 prompts

    # Run T0
    t0 = t0_integrity(lens, lm, tok, val_prompts)

    if not t0["pass"]:
        print("\n*** T0 FAILED — halting. Fix integrity issues before proceeding. ***")
        return

    # Run T2
    t2 = t2_concept_recovery(lens, lm, tok, dev_items, test_items, chat_mode=args.chat)

    # Summary
    print("\n" + "="*60)
    print("G0 SUMMARY")
    print("="*60)
    print(f"  T0 (integrity): {'PASS' if t0['pass'] else 'FAIL'}")
    t2_verdict = "PASS" if t2["pass"] else ("MARGINAL" if t2.get("jlens_accuracy", 0) >= 0.35 else "FAIL")
    print(f"  T2 (concept recovery, {'chat' if args.chat else 'prose'}): {t2_verdict}")
    print(f"    J-lens: {t2.get('jlens_accuracy', 0):.1%}")
    print(f"    Logit-lens: {t2.get('logit_accuracy', 0):.1%}")
    print(f"    Inference-class: {t2.get('inference_accuracy', 0):.1%}")

    # Save results
    out = {
        "metadata": {
            "lens": args.lens,
            "model": args.model,
            "mode": "chat" if args.chat else "prose",
            "date": datetime.now().isoformat(),
            "items_hash": "d068ae9",
        },
        "t0": t0,
        "t2": t2,
    }
    out_path = Path(f"g0_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
