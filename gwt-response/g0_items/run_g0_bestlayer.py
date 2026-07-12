"""
G0 T2 — Best-layer scoring variant.

For each item, score at EVERY workspace-band layer and report
the layer where the J-lens ranks the concept highest.
Compare: J-lens best-layer accuracy vs logit-lens best-layer accuracy.

This tests whether the J-lens reads concepts correctly but at
item-specific layers, which fixed-layer scoring misses.
"""

import torch
import json
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

def load_items(items_dir):
    dev_items, test_items = [], []
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


def score_item_all_layers(lens, lm, tok, item, layers, use_jacobian=True):
    """Score one item at every layer. Return per-layer rank of concept."""
    prompt = item["prompt"]
    concept = item["concept"]
    distractors = item.get("distractors", [])
    candidates = [concept] + distractors

    lens_logits, model_logits, _ = lens.apply(
        lm, prompt, layers=layers, positions=[-1],
        use_jacobian=use_jacobian
    )

    results = {}
    for layer in layers:
        logits = lens_logits[layer][0] if use_jacobian else model_logits[0]
        if not use_jacobian and layer != layers[0]:
            # logit lens gives same model_logits for all layers
            # but lens_logits differs per layer when use_jacobian=False
            logits = lens_logits[layer][0]

        scores = {}
        for c in candidates:
            c_ids = tok.encode(c, add_special_tokens=False)
            if c_ids:
                scores[c] = float(logits[c_ids[0]].item())
            else:
                scores[c] = float('-inf')

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        concept_rank = next(i for i, (c, _) in enumerate(ranked) if c == concept) + 1
        results[layer] = {
            "rank": concept_rank,
            "correct": concept_rank == 1,
            "top3": [c for c, _ in ranked[:3]],
        }

    return results


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens import from_hf, JacobianLens

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--items-dir", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("G0 T2 — BEST-LAYER SCORING")
    print("=" * 60)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    ).to("mps")
    model.eval()
    lm = from_hf(model, tok)
    lens = JacobianLens.load(args.lens)

    items_dir = args.items_dir or str(Path(__file__).parent)
    dev_items, test_items = load_items(items_dir)
    print("Items: %d dev, %d test" % (len(dev_items), len(test_items)))

    all_layers = sorted(lens.source_layers)
    n = len(all_layers)
    band = all_layers[n // 4 : 3 * n // 4]  # middle half
    print("Scoring across %d band layers: %s" % (len(band), band))

    # Score every test item at every band layer, both J-lens and logit-lens
    jlens_items = []
    llens_items = []

    for i, item in enumerate(test_items):
        concept = item["concept"]
        item_class = item.get("class", "unknown")

        j_results = score_item_all_layers(lens, lm, tok, item, band, use_jacobian=True)
        l_results = score_item_all_layers(lens, lm, tok, item, band, use_jacobian=False)

        # Best layer for each lens
        j_best_layer = min(j_results, key=lambda l: j_results[l]["rank"])
        l_best_layer = min(l_results, key=lambda l: l_results[l]["rank"])

        j_best_rank = j_results[j_best_layer]["rank"]
        l_best_rank = l_results[l_best_layer]["rank"]

        # Fixed-layer (middle of band) for comparison
        mid = band[len(band) // 2]
        j_fixed_rank = j_results[mid]["rank"]
        l_fixed_rank = l_results[mid]["rank"]

        jlens_items.append({
            "concept": concept,
            "class": item_class,
            "best_rank": j_best_rank,
            "best_layer": j_best_layer,
            "fixed_rank": j_fixed_rank,
            "correct_best": j_best_rank == 1,
            "correct_fixed": j_fixed_rank == 1,
        })
        llens_items.append({
            "concept": concept,
            "class": item_class,
            "best_rank": l_best_rank,
            "best_layer": l_best_layer,
            "fixed_rank": l_fixed_rank,
            "correct_best": l_best_rank == 1,
            "correct_fixed": l_fixed_rank == 1,
        })

        if (i + 1) % 10 == 0:
            print("  %d/%d..." % (i + 1, len(test_items)))

    # Summary
    j_best_acc = sum(1 for r in jlens_items if r["correct_best"]) / len(jlens_items)
    j_fixed_acc = sum(1 for r in jlens_items if r["correct_fixed"]) / len(jlens_items)
    l_best_acc = sum(1 for r in llens_items if r["correct_best"]) / len(llens_items)
    l_fixed_acc = sum(1 for r in llens_items if r["correct_fixed"]) / len(llens_items)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("\n  Fixed-layer (L%d):" % mid)
    print("    J-lens:     %.1f%%" % (j_fixed_acc * 100))
    print("    Logit-lens: %.1f%%" % (l_fixed_acc * 100))
    print("\n  Best-layer (per item):")
    print("    J-lens:     %.1f%%" % (j_best_acc * 100))
    print("    Logit-lens: %.1f%%" % (l_best_acc * 100))

    # Per-class breakdown
    for cls in ["inference", "association", "abstract", "unknown"]:
        j_cls = [r for r in jlens_items if r["class"] == cls]
        l_cls = [r for r in llens_items if r["class"] == cls]
        if j_cls:
            j_acc = sum(1 for r in j_cls if r["correct_best"]) / len(j_cls)
            l_acc = sum(1 for r in l_cls if r["correct_best"]) / len(l_cls)
            print("\n  %s (n=%d):" % (cls, len(j_cls)))
            print("    J-lens best:  %.1f%%" % (j_acc * 100))
            print("    Logit best:   %.1f%%" % (l_acc * 100))

    # Item-level agreement
    both_correct = sum(1 for j, l in zip(jlens_items, llens_items)
                      if j["correct_best"] and l["correct_best"])
    j_only = sum(1 for j, l in zip(jlens_items, llens_items)
                if j["correct_best"] and not l["correct_best"])
    l_only = sum(1 for j, l in zip(jlens_items, llens_items)
                if not j["correct_best"] and l["correct_best"])
    neither = sum(1 for j, l in zip(jlens_items, llens_items)
                 if not j["correct_best"] and not l["correct_best"])

    print("\n  Item agreement (best-layer):")
    print("    Both correct:  %d" % both_correct)
    print("    J-lens only:   %d" % j_only)
    print("    Logit only:    %d" % l_only)
    print("    Neither:       %d" % neither)

    # Paired comparison
    j_ranks = [r["best_rank"] for r in jlens_items]
    l_ranks = [r["best_rank"] for r in llens_items]
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(j_ranks, l_ranks)
        print("\n  Paired Wilcoxon (best-layer ranks): stat=%.1f, p=%.4f" % (stat, p))
        mean_j = np.mean(j_ranks)
        mean_l = np.mean(l_ranks)
        print("  Mean rank: J-lens=%.2f, Logit=%.2f" % (mean_j, mean_l))
    except ImportError:
        print("\n  (scipy not available)")

    # Layer distribution
    j_layers = [r["best_layer"] for r in jlens_items if r["correct_best"]]
    l_layers = [r["best_layer"] for r in llens_items if r["correct_best"]]
    if j_layers:
        print("\n  J-lens correct items peak at layers: %s" %
              sorted(set(j_layers)))
        from collections import Counter
        print("    Distribution: %s" % dict(Counter(j_layers).most_common()))
    if l_layers:
        print("  Logit correct items peak at layers: %s" %
              sorted(set(l_layers)))
        from collections import Counter
        print("    Distribution: %s" % dict(Counter(l_layers).most_common()))

    # Save
    out = {
        "metadata": {
            "lens": args.lens,
            "model": args.model,
            "band": band,
            "mid_layer": mid,
            "date": datetime.now().isoformat(),
        },
        "summary": {
            "j_fixed_acc": j_fixed_acc,
            "j_best_acc": j_best_acc,
            "l_fixed_acc": l_fixed_acc,
            "l_best_acc": l_best_acc,
        },
        "items": {"jlens": jlens_items, "logit": llens_items},
    }
    out_path = "g0_bestlayer_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to %s" % out_path)


if __name__ == "__main__":
    main()
