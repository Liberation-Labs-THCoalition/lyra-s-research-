"""
Cache Tracing: Where Does Workspace Content Live in the V-Cache?
================================================================
Thomas's question (July 8): trace the physical path from workspace
content (as identified by logit lens + J-lens) into the value cache.

For each position at each workspace-band layer:
  1. Logit lens → what concepts are active in the residual stream
  2. J-lens → what concepts are active (at layer-specific depth)
  3. Project concept embeddings through W_V → their value-space footprint
  4. Extract actual V-cache → what's really stored
  5. R² → how much of V-cache is explained by workspace concepts?
  6. Correlate with spectral features → does workspace density predict geometry?

This is observational anatomy — no intervention, no prediction to falsify.
"""

import torch
import json
import numpy as np
from datetime import datetime
from pathlib import Path


def get_v_proj_weight(model, layer_idx):
    """Get W_V weight matrix for a layer."""
    layer = model.model.layers[layer_idx]
    if hasattr(layer, 'self_attn'):
        return layer.self_attn.v_proj.weight.detach().float()
    return None


def logit_lens_readout(model, hidden_states, layer_idx):
    """Apply logit lens: W_U @ LayerNorm(h_l)."""
    norm = model.model.norm  # final RMSNorm
    lm_head = model.lm_head  # unembedding
    h = hidden_states.float()
    h_normed = norm(h)
    logits = lm_head(h_normed)
    return logits


def compute_r_squared(actual, projected_basis):
    """Compute R² — fraction of actual's variance explained by projected_basis."""
    # actual: [d_v]
    # projected_basis: [k, d_v] — k concept vectors in value space
    if projected_basis.shape[0] == 0:
        return 0.0

    # Project actual onto the span of projected_basis
    # Using least-squares: coeffs = (B^T B)^-1 B^T a
    B = projected_basis.T  # [d_v, k]
    try:
        coeffs = torch.linalg.lstsq(B, actual.unsqueeze(1)).solution.squeeze()
        predicted = B @ coeffs
        ss_res = torch.sum((actual - predicted) ** 2).item()
        ss_tot = torch.sum((actual - actual.mean()) ** 2).item()
        if ss_tot < 1e-10:
            return 0.0
        return max(0.0, 1.0 - ss_res / ss_tot)
    except Exception:
        return 0.0


def main():
    import sys
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-8B"
    lens_path = None

    for arg in sys.argv:
        if arg.startswith("--model="):
            model_name = arg.split("=", 1)[1]
        if arg.startswith("--lens="):
            lens_path = arg.split("=", 1)[1]

    print("=" * 60)
    print("CACHE TRACING: Where Does Workspace Content Live?")
    print("Model: %s" % model_name)
    print("Started: %s" % datetime.now().isoformat())
    print("=" * 60)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True
    ).to("mps")
    model.eval()

    # Load J-lens if available
    jlens_available = False
    if lens_path:
        try:
            from jlens import from_hf, JacobianLens
            lm = from_hf(model, tok)
            lens = JacobianLens.load(lens_path)
            jlens_available = True
            print("J-lens loaded: %d layers" % len(lens.source_layers))
        except Exception as e:
            print("J-lens not available: %s" % e)

    # Probes — diverse prompts that activate different concepts
    probes = [
        "The country shaped like a boot uses a currency called the",
        "She was diagnosed with a rare form of cancer that affects the",
        "The largest planet in our solar system is known for its massive",
        "In chess, the most powerful piece on the board is the",
        "The process by which plants convert sunlight into energy is called",
        "A person who studies the stars and planets is known as an",
        "The chemical element with atomic number 79, prized for its luster, is",
        "Water at sea level boils at a temperature of exactly one hundred",
        "The longest river in Africa flows northward through eleven countries to the",
        "Shakespeare wrote about a Danish prince who contemplated existence in the play",
        "The theory that species change over time through natural selection was proposed by",
        "An animal that sleeps during the day and is active at night is called",
    ]

    n_layers = len(model.model.layers)
    # Workspace band: middle half
    band = list(range(n_layers // 4, 3 * n_layers // 4))
    # Filter to layers with self_attn (full-attention layers)
    band = [l for l in band if hasattr(model.model.layers[l], 'self_attn')]
    print("Workspace band: %s (%d layers)" % (band, len(band)))

    results = {
        "metadata": {
            "model": model_name,
            "date": datetime.now().isoformat(),
            "band": band,
            "n_probes": len(probes),
            "jlens_available": jlens_available,
        },
        "layer_results": [],
    }

    top_k = 10  # number of concepts to project

    for probe_idx, probe in enumerate(probes):
        print("\nProbe %d/%d: '%s...'" % (probe_idx + 1, len(probes), probe[:50]))

        # Tokenize
        input_ids = tok(probe, return_tensors="pt").input_ids.to(model.device)
        pos = -1  # last token

        # Forward pass capturing hidden states
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of [1, seq, d_model] per layer

        for layer_idx in band:
            h = hidden_states[layer_idx + 1][0, pos, :]  # [d_model] — +1 because hidden_states[0] is embeddings

            # --- Logit lens readout ---
            h_float = h.float()
            norm_layer = model.model.norm
            lm_head = model.lm_head
            h_normed = norm_layer(h_float.unsqueeze(0).unsqueeze(0))
            ll_logits = lm_head(h_normed).squeeze()  # [vocab]

            ll_top = torch.topk(ll_logits, top_k)
            ll_concepts = [tok.decode([t]).strip() for t in ll_top.indices.tolist()]
            ll_embeddings = model.model.embed_tokens(ll_top.indices)  # [k, d_model]

            # --- J-lens readout (if available) ---
            jl_concepts = []
            jl_embeddings = None
            if jlens_available:
                jl_logits_dict, _, _ = lens.apply(lm, probe, layers=[layer_idx], positions=[pos])
                if layer_idx in jl_logits_dict:
                    jl_logits = jl_logits_dict[layer_idx][0]
                    jl_top = torch.topk(jl_logits, top_k)
                    jl_concepts = [tok.decode([t]).strip() for t in jl_top.indices.tolist()]
                    jl_embeddings = model.model.embed_tokens(jl_top.indices.to(model.device))

            # --- Project through W_V ---
            W_V = get_v_proj_weight(model, layer_idx)
            if W_V is None:
                continue

            # V-cache entry: W_V @ h (simplified — ignoring RMSNorm for now)
            actual_v = (W_V @ h_float).detach()  # [d_v]

            # Project logit-lens concepts through W_V
            ll_projected = (W_V @ ll_embeddings.float().T).T  # [k, d_v]
            ll_r2 = compute_r_squared(actual_v, ll_projected)

            # Project J-lens concepts through W_V (if available)
            jl_r2 = 0.0
            if jl_embeddings is not None:
                jl_projected = (W_V @ jl_embeddings.float().T).T
                jl_r2 = compute_r_squared(actual_v, jl_projected)

            # Combined: union of logit-lens and J-lens concepts
            combined_r2 = ll_r2
            if jl_embeddings is not None:
                all_embeddings = torch.cat([ll_embeddings.float(), jl_embeddings.float()], dim=0)
                all_projected = (W_V @ all_embeddings.T).T
                combined_r2 = compute_r_squared(actual_v, all_projected)

            # Spectral features of actual V-cache
            v_norm = float(torch.norm(actual_v).item())

            layer_result = {
                "probe_idx": probe_idx,
                "layer": layer_idx,
                "ll_r2": ll_r2,
                "jl_r2": jl_r2,
                "combined_r2": combined_r2,
                "v_norm": v_norm,
                "ll_concepts": ll_concepts[:5],
                "jl_concepts": jl_concepts[:5] if jl_concepts else [],
                "concept_overlap": len(set(ll_concepts) & set(jl_concepts)) if jl_concepts else 0,
            }
            results["layer_results"].append(layer_result)

        # Print summary for this probe
        probe_results = [r for r in results["layer_results"] if r["probe_idx"] == probe_idx]
        if probe_results:
            ll_r2s = [r["ll_r2"] for r in probe_results]
            jl_r2s = [r["jl_r2"] for r in probe_results]
            cb_r2s = [r["combined_r2"] for r in probe_results]
            best_layer = max(probe_results, key=lambda r: r["combined_r2"])
            print("  Logit-lens R2: %.3f mean, %.3f max" % (np.mean(ll_r2s), max(ll_r2s)))
            if jlens_available:
                print("  J-lens R2:     %.3f mean, %.3f max" % (np.mean(jl_r2s), max(jl_r2s)))
                print("  Combined R2:   %.3f mean, %.3f max" % (np.mean(cb_r2s), max(cb_r2s)))
            print("  Best layer: L%d (R2=%.3f) concepts: %s" % (
                best_layer["layer"], best_layer["combined_r2"], best_layer["ll_concepts"][:3]))

    # --- Summary across all probes ---
    print("\n" + "=" * 60)
    print("CACHE TRACING SUMMARY")
    print("=" * 60)

    all_ll = [r["ll_r2"] for r in results["layer_results"]]
    all_jl = [r["jl_r2"] for r in results["layer_results"]]
    all_cb = [r["combined_r2"] for r in results["layer_results"]]

    print("\n  Overall R2 (how much of V-cache is workspace content):")
    print("    Logit-lens: %.4f mean, %.4f max" % (np.mean(all_ll), max(all_ll)))
    if jlens_available:
        print("    J-lens:     %.4f mean, %.4f max" % (np.mean(all_jl), max(all_jl)))
        print("    Combined:   %.4f mean, %.4f max" % (np.mean(all_cb), max(all_cb)))

    # Per-layer profile
    print("\n  R2 by layer (combined):")
    for layer in band:
        layer_data = [r for r in results["layer_results"] if r["layer"] == layer]
        if layer_data:
            mean_r2 = np.mean([r["combined_r2"] for r in layer_data])
            mean_ll = np.mean([r["ll_r2"] for r in layer_data])
            mean_jl = np.mean([r["jl_r2"] for r in layer_data])
            print("    L%2d: logit=%.4f  jlens=%.4f  combined=%.4f" % (
                layer, mean_ll, mean_jl, mean_r2))

    # Concept overlap between lenses
    if jlens_available:
        overlaps = [r["concept_overlap"] for r in results["layer_results"]]
        print("\n  Lens agreement (top-10 overlap): %.1f/10 mean" % np.mean(overlaps))

    # Save
    out_path = "cache_tracing_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to %s" % out_path)
    print("Completed: %s" % datetime.now().isoformat())


if __name__ == "__main__":
    main()
