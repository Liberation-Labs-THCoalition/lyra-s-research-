"""
Cache Tracing v3 — Fixed per Agni review.

Fixes:
1. Random baseline: 100 random direction sets, CPU lstsq, per-layer
2. 20 diverse prompts (factual, confab-prone, instruction, emotional)
3. Honest framing: "residual-stream alignment with logit-lens concept
   directions in the W_V value subspace" — not "V-cache content"

Run: python cache_tracing_v3.py --model=Qwen/Qwen3-8B
"""

import torch
import json
import numpy as np
from datetime import datetime
from pathlib import Path


PROBES = {
    "factual": [
        "The country shaped like a boot uses a currency called the",
        "The largest planet in our solar system is known for its massive",
        "The chemical element with atomic number 79, prized for its luster, is",
        "Shakespeare wrote about a Danish prince who contemplated existence in the play",
        "The process by which plants convert sunlight into energy is called",
    ],
    "confab_prone": [
        "The Thornberry-Nakamura study on deep-sea photosynthesis found that",
        "The Kirkwood-Diaz algorithm for quantum protein folding works by",
        "Professor Elaine Marchetti's theory of resonant cognition proposes that",
        "The standard treatment protocol for Halverson syndrome begins with",
        "The Crysolene mineral discovered in 2019 in the Grenvold Trench is notable for",
    ],
    "instruction": [
        "Please summarize the main arguments for and against universal basic income in",
        "Write a function that takes a list of integers and returns the second largest",
        "Explain the difference between supervised and unsupervised learning to someone who",
        "Compare and contrast the parliamentary and presidential systems of government in",
        "Describe three strategies for reducing carbon emissions in urban transportation",
    ],
    "emotional": [
        "She opened the letter and read the diagnosis. The room went very",
        "After twenty years apart, he saw her standing at the gate and felt",
        "The child looked up at her mother and said for the first time",
        "He had worked on the project for three years. When it was cancelled, he",
        "They stood together at the edge of the cliff watching the sunrise on their last",
    ],
}


def compute_r2_cpu(actual, basis):
    """R2 on CPU via lstsq. actual: [d], basis: [k, d]."""
    if basis.shape[0] == 0:
        return 0.0
    a = actual.cpu().float()
    B = basis.cpu().float().T  # [d, k]
    try:
        coeffs = torch.linalg.lstsq(B, a.unsqueeze(1)).solution.squeeze()
        pred = B @ coeffs
        ss_res = torch.sum((a - pred) ** 2).item()
        ss_tot = torch.sum((a - a.mean()) ** 2).item()
        if ss_tot < 1e-10:
            return 0.0
        return float(max(0.0, 1.0 - ss_res / ss_tot))
    except Exception:
        return 0.0


def main():
    import sys
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-8B"
    for arg in sys.argv:
        if arg.startswith("--model="):
            model_name = arg.split("=", 1)[1]

    print("=" * 60)
    print("CACHE TRACING v3 (Agni-fixed)")
    print("Measuring: residual-stream alignment with logit-lens")
    print("concept directions in the W_V value subspace")
    print("Model: %s" % model_name)
    print("=" * 60)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True
    ).to("mps")
    model.eval()

    W_U = model.lm_head.weight.detach().float()  # [vocab, d_model]
    vocab_size = W_U.shape[0]

    n_layers = len(model.model.layers)
    # Full-attention layers only
    all_layers = [l for l in range(n_layers)
                  if hasattr(model.model.layers[l], 'self_attn')]
    # Sample every 3rd for speed
    sample_layers = all_layers[::3]
    if all_layers[-1] not in sample_layers:
        sample_layers.append(all_layers[-1])
    print("Sampling %d layers: %s" % (len(sample_layers), sample_layers))

    top_k = 10
    n_random = 100  # random baseline draws

    results = {
        "metadata": {
            "model": model_name,
            "date": datetime.now().isoformat(),
            "layers": sample_layers,
            "top_k": top_k,
            "n_random": n_random,
            "framing": "residual-stream alignment in W_V subspace",
        },
        "by_category": {},
        "by_layer": {},
    }

    all_probe_results = []

    for category, prompts in PROBES.items():
        print("\n--- %s (%d prompts) ---" % (category, len(prompts)))
        cat_results = []

        for probe_idx, probe in enumerate(prompts):
            input_ids = tok(probe, return_tensors="pt").input_ids.to(model.device)

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            for layer_idx in sample_layers:
                h = hidden_states[layer_idx + 1][0, -1, :].float()

                W_V = model.model.layers[layer_idx].self_attn.v_proj.weight.float()

                # Actual value projection
                actual_v = (W_V @ h).detach()

                # Logit lens top-k concept directions
                h_normed = model.model.norm(
                    h.unsqueeze(0).unsqueeze(0)
                ).squeeze()
                ll_logits = W_U @ h_normed
                top_indices = torch.topk(ll_logits, top_k).indices
                concept_dirs = W_U[top_indices]  # [k, d_model]
                projected = (W_V @ concept_dirs.T).T  # [k, d_v]

                concept_r2 = compute_r2_cpu(actual_v, projected)

                # Random baseline: 100 draws of k random directions
                rand_r2s = []
                for _ in range(n_random):
                    rand_idx = torch.randint(0, vocab_size, (top_k,))
                    rand_dirs = W_U[rand_idx]
                    rand_proj = (W_V @ rand_dirs.T).T
                    rand_r2s.append(compute_r2_cpu(actual_v, rand_proj))

                rand_mean = float(np.mean(rand_r2s))
                rand_p95 = float(np.percentile(rand_r2s, 95))

                top_concepts = [tok.decode([t]).strip()
                               for t in top_indices[:5].tolist()]

                entry = {
                    "category": category,
                    "probe_idx": probe_idx,
                    "layer": layer_idx,
                    "concept_r2": concept_r2,
                    "rand_mean": rand_mean,
                    "rand_p95": rand_p95,
                    "above_p95": concept_r2 > rand_p95,
                    "lift": concept_r2 / rand_mean if rand_mean > 1e-6 else 0,
                    "top_concepts": top_concepts,
                }
                all_probe_results.append(entry)
                cat_results.append(entry)

            if (probe_idx + 1) % 2 == 0:
                print("  %d/%d done" % (probe_idx + 1, len(prompts)))

        # Category summary
        for layer in sample_layers:
            ld = [r for r in cat_results if r["layer"] == layer]
            if ld:
                mean_r2 = np.mean([r["concept_r2"] for r in ld])
                mean_rand = np.mean([r["rand_mean"] for r in ld])
                n_above = sum(1 for r in ld if r["above_p95"])
                print("  L%2d: concept=%.4f  random=%.4f  above_p95=%d/%d" % (
                    layer, mean_r2, mean_rand, n_above, len(ld)))

    # Global summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nBy layer (all categories):")
    for layer in sample_layers:
        ld = [r for r in all_probe_results if r["layer"] == layer]
        mean_r2 = np.mean([r["concept_r2"] for r in ld])
        mean_rand = np.mean([r["rand_mean"] for r in ld])
        mean_p95 = np.mean([r["rand_p95"] for r in ld])
        n_above = sum(1 for r in ld if r["above_p95"])
        lift = mean_r2 / mean_rand if mean_rand > 1e-6 else 0
        print("  L%2d: concept=%.4f  rand_mean=%.4f  rand_p95=%.4f  above=%d/%d  lift=%.1fx" % (
            layer, mean_r2, mean_rand, mean_p95, n_above, len(ld), lift))

    print("\nBy category (averaged across layers):")
    for category in PROBES:
        cd = [r for r in all_probe_results if r["category"] == category]
        mean_r2 = np.mean([r["concept_r2"] for r in cd])
        mean_rand = np.mean([r["rand_mean"] for r in cd])
        n_above = sum(1 for r in cd if r["above_p95"])
        print("  %-15s: concept=%.4f  random=%.4f  above_p95=%d/%d" % (
            category, mean_r2, mean_rand, n_above, len(cd)))

    # Save
    results["probe_results"] = all_probe_results
    out_path = "cache_tracing_v3_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to %s" % out_path)
    print("Completed: %s" % datetime.now().isoformat())


if __name__ == "__main__":
    main()
