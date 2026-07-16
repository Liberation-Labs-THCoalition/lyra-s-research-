"""
Residual Stream Decomposition: Where do workspace concepts reconstruct?

The residual stream at layer L+1 is:
  h_{L+1} = h_L + Attention(h_L) + MLP(h_L + Attention(h_L))

J-lens reads concepts from h (the full residual stream).
K alone doesn't carry concepts (1.12x, null).
V alone doesn't carry concepts (0.92x, null).
Nexus showed K+V is superadditive (0.333 + 0.333 -> 1.000).

This experiment decomposes: which COMPONENT of the residual stream
carries the concept signal?

1. h_L (residual input) — concept already present from earlier layers?
2. attn_output — K×V reconstruction through attention?
3. mlp_output — feedforward computation?

Uses hooks to capture each component separately, then measures
J-lens concept readout on each.

LAYER-MATCHING (post-Opus-Agni review):
Input, attn, and MLP at layer L use J-lens[L] (correct: all in
residual-stream space at depth L). Output (h_{L+1}) uses J-lens[L+1]
for standalone assessment, AND J-lens[L] for cross-component comparison
(H4: output vs attention lift). Both reported.

Component norms tracked alongside R² per Opus Agni MAJOR-2.
Absolute concept power (R² × ||component||²) reported to check whether
MLP's larger magnitude compensates for lower R².
"""

import torch
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(42)
np.random.seed(42)

ALL_LAYERS = list(range(0, 36, 3))
K_CONCEPTS = 10
N_RANDOM = 20  # reduced from 100 for speed

PROMPTS = [
    "The country shaped like a boot on the map of Europe is",
    "The largest planet in our solar system is called",
    "Shakespeare wrote about a Danish prince in the play",
    "Water molecules consist of two hydrogen atoms and one",
    "The speed of light in a vacuum is approximately",
    "The Thornberry-Nakamura theorem, first proven in 1987, states that",
    "The Kirkwood-Diaz algorithm, widely used in computational genomics, works by",
    "Professor Elaine Marchetti's theory of recursive social dynamics predicts that",
    "According to the Westlake-Chen principle of information decay,",
    "The Rasmussen effect, documented in studies of deep-ocean thermal vents, causes",
    "Please write a brief summary of the key differences between",
    "Explain in simple terms how a neural network learns to",
    "After years apart, she finally saw him standing at the gate, and all the words she had rehearsed",
    "The doctor paused before speaking, and in that silence the whole family understood what the test results",
    "Standing at the edge of the cliff at sunset, she felt the full weight of everything she had lost and",
]


def get_fixed_directions(J_layer, W_U, k=K_CONCEPTS):
    J_l = J_layer.float()
    W = W_U.float()
    transported = W @ J_l
    norms = torch.norm(transported, dim=1)
    top_idx = torch.topk(norms, k).indices
    dirs = transported[top_idx]
    dirs = dirs / (torch.norm(dirs, dim=1, keepdim=True) + 1e-8)
    return dirs


def get_random_directions(J_layer, W_U, k=K_CONCEPTS, seed=0):
    J_l = J_layer.float()
    W = W_U.float()
    transported = W @ J_l
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(transported.shape[0], generator=g)[:k]
    dirs = transported[idx]
    dirs = dirs / (torch.norm(dirs, dim=1, keepdim=True) + 1e-8)
    return dirs


def compute_r2(directions, target):
    D = directions.float()
    t = target.float()
    if D.shape[0] == 0 or t.norm() < 1e-10:
        return 0.0
    try:
        result = torch.linalg.lstsq(D.T, t)
        residual = t - D.T @ result.solution
        ss_res = (residual ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        if ss_tot < 1e-12:
            return 0.0
        return max(float((1 - ss_res / ss_tot).item()), 0.0)
    except Exception:
        return 0.0


def main():
    model_name = "Qwen/Qwen3-8B"
    lens_path = "fitted_lenses/lens_qwen3-8b.pt"

    print("=" * 60)
    print("RESIDUAL STREAM DECOMPOSITION")
    print("Where do workspace concepts reconstruct?")
    print(f"Model: {model_name}")
    print("=" * 60)

    tok = AutoTokenizer.from_pretrained(model_name, tokenizer_type="qwen3")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True
    ).to("mps")
    model.eval()

    W_U = model.lm_head.weight.detach().cpu().float()
    lens_data = torch.load(lens_path, map_location="cpu")
    J = lens_data["J"]

    # Pre-compute fixed J-lens directions per layer
    fixed_dirs = {}
    for l in ALL_LAYERS:
        if l in J:
            fixed_dirs[l] = get_fixed_directions(J[l], W_U, k=K_CONCEPTS)

    results = []

    for pidx, prompt in enumerate(PROMPTS):
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)

        # Capture components via hooks
        components = {}

        def make_attn_hook(layer_idx):
            def hook_fn(module, args, output):
                # Attention module output: (attn_output, attn_weights, past_kv)
                if isinstance(output, tuple):
                    attn_out = output[0]
                else:
                    attn_out = output
                components[("attn", layer_idx)] = attn_out[0, -1, :].detach().float().cpu()
            return hook_fn

        def make_mlp_hook(layer_idx):
            def hook_fn(module, args, output):
                if isinstance(output, tuple):
                    mlp_out = output[0]
                else:
                    mlp_out = output
                components[("mlp", layer_idx)] = mlp_out[0, -1, :].detach().float().cpu()
            return hook_fn

        def make_input_hook(layer_idx):
            def hook_fn(module, args):
                hidden = args[0] if isinstance(args, tuple) else args
                components[("input", layer_idx)] = hidden[0, -1, :].detach().float().cpu()
            return hook_fn

        handles = []
        for l in ALL_LAYERS:
            layer = model.model.layers[l]
            handles.append(layer.register_forward_pre_hook(make_input_hook(l)))
            if hasattr(layer, "self_attn"):
                handles.append(layer.self_attn.register_forward_hook(make_attn_hook(l)))
            if hasattr(layer, "mlp"):
                handles.append(layer.mlp.register_forward_hook(make_mlp_hook(l)))

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)

        for h in handles:
            h.remove()

        # Also capture full residual stream output (h_{L+1})
        for l in ALL_LAYERS:
            components[("output", l)] = out.hidden_states[l + 1][0, -1, :].float().cpu()

        # Measure J-lens readout on each component
        # Post-Agni fixes:
        #   1. Use J-lens[L] for input/attn/mlp (all at depth L in residual-stream space)
        #   2. Use J-lens[L+1] for output when available (h_{L+1} is at depth L+1)
        #   3. Skip boundary cases (L=0 input has no prior, L=max output has no L+1 lens)
        #   4. Add cosine alignment check between each component and J-lens mean direction
        for l in ALL_LAYERS:
            if l not in fixed_dirs:
                continue

            dirs = fixed_dirs[l]
            jlens_mean_dir = dirs.mean(dim=0)
            jlens_mean_dir = jlens_mean_dir / (torch.norm(jlens_mean_dir) + 1e-8)

            component_r2 = {}

            for comp_name in ["input", "attn", "mlp", "output"]:
                # Boundary check (Agni review)
                if comp_name == "output" and (l + 1) not in J:
                    continue  # no L+1 lens for output at last layer

                # Select appropriate J-lens directions
                # J stores ALL layers (0-34), not just sampled ones
                if comp_name == "output" and (l + 1) in J:
                    # Output h_{L+1} is at depth L+1; use J-lens[L+1]
                    output_dirs = get_fixed_directions(J[l + 1], W_U, k=K_CONCEPTS)
                    use_dirs = output_dirs
                    use_J_layer = l + 1
                else:
                    use_dirs = dirs
                    use_J_layer = l

                key = (comp_name, l)
                if key not in components:
                    continue
                vec = components[key]

                # Opus Agni MAJOR-2: track component norms
                vec_norm = float(torch.norm(vec).item())

                # Alignment check: cosine between component and J-lens subspace
                alignment = float(torch.nn.functional.cosine_similarity(
                    vec.unsqueeze(0), jlens_mean_dir.unsqueeze(0)
                ).item())

                r2_jlens = compute_r2(use_dirs, vec)

                # Opus Agni MAJOR-1: for output, ALSO compute R² with J-lens[L]
                # for fair cross-component comparison (H4)
                r2_jlens_same_layer = None
                if comp_name == "output" and use_J_layer != l:
                    r2_jlens_same_layer = compute_r2(dirs, vec)

                r2_randoms = []
                for draw in range(N_RANDOM):
                    rand_dirs = get_random_directions(
                        J[use_J_layer], W_U, k=K_CONCEPTS,
                        seed=42 + draw + pidx * 100 + l * 1000
                    )
                    r2_randoms.append(compute_r2(rand_dirs, vec))

                r2_rand = float(np.mean(r2_randoms))
                lift = r2_jlens / max(r2_rand, 1e-6)
                # Absolute concept power = R² × ||component||²
                concept_power = r2_jlens * vec_norm ** 2

                entry = {
                    "r2": r2_jlens, "r2_rand": r2_rand, "lift": lift,
                    "norm": vec_norm,
                    "concept_power": concept_power,
                    "alignment": alignment,
                    "jlens_layer_used": use_J_layer,
                }
                if r2_jlens_same_layer is not None:
                    entry["r2_same_layer"] = r2_jlens_same_layer
                component_r2[comp_name] = entry

            results.append({
                "prompt_idx": pidx,
                "layer": l,
                "components": component_r2,
            })

        if (pidx + 1) % 5 == 0:
            print(f"  {pidx + 1}/{len(PROMPTS)} prompts done")

    # Summary
    print("\n" + "=" * 60)
    print("WHERE DO CONCEPTS RECONSTRUCT?")
    print("Post-Opus-Agni: output uses J-lens[L+1], component norms tracked")
    print("=" * 70)

    print("\nLIFT (R² / random R²) — fraction of variance explained:")
    print(f"{'Layer':>6} | {'Input':>8} | {'Attn':>8} | {'MLP':>8} | {'Output':>8} | {'Out(L)':>8}")
    print("-" * 60)

    for l in ALL_LAYERS:
        layer_results = [r for r in results if r["layer"] == l]
        if not layer_results:
            continue
        lifts = {}
        for comp in ["input", "attn", "mlp", "output"]:
            vals = [r["components"].get(comp, {}).get("lift", 0) for r in layer_results]
            lifts[comp] = np.mean(vals) if vals else 0
        # Output with same-layer J-lens (for H4 comparison)
        out_same = [r["components"].get("output", {}).get("r2_same_layer", None) for r in layer_results]
        out_same_r2 = [v for v in out_same if v is not None]
        out_rand = [r["components"].get("output", {}).get("r2_rand", 1e-6) for r in layer_results]
        if out_same_r2 and out_rand:
            out_same_lift = np.mean(out_same_r2) / max(np.mean(out_rand), 1e-6)
        else:
            out_same_lift = lifts.get("output", 0)

        marker = ""
        max_comp = max(lifts, key=lifts.get)
        if lifts[max_comp] > 1.5:
            marker = f"  <- {max_comp}"

        print(f"L{l:>4} | {lifts.get('input',0):>7.1f}x | {lifts.get('attn',0):>7.1f}x | {lifts.get('mlp',0):>7.1f}x | {lifts.get('output',0):>7.1f}x | {out_same_lift:>7.1f}x{marker}")

    print("\nCONCEPT POWER (R² × ||component||²) — absolute signal contributed:")
    print(f"{'Layer':>6} | {'Input':>10} | {'Attn':>10} | {'MLP':>10} | {'Output':>10}")
    print("-" * 55)

    for l in ALL_LAYERS:
        layer_results = [r for r in results if r["layer"] == l]
        if not layer_results:
            continue
        powers = {}
        for comp in ["input", "attn", "mlp", "output"]:
            vals = [r["components"].get(comp, {}).get("concept_power", 0) for r in layer_results]
            powers[comp] = np.mean(vals) if vals else 0

        marker = ""
        max_comp = max(powers, key=powers.get)
        if powers[max_comp] > 0:
            marker = f"  <- {max_comp}"

        print(f"L{l:>4} | {powers.get('input',0):>10.1f} | {powers.get('attn',0):>10.1f} | {powers.get('mlp',0):>10.1f} | {powers.get('output',0):>10.1f}{marker}")

    print("\nCOMPONENT NORMS (mean across prompts):")
    print(f"{'Layer':>6} | {'Input':>10} | {'Attn':>10} | {'MLP':>10} | {'Output':>10}")
    print("-" * 55)

    for l in ALL_LAYERS:
        layer_results = [r for r in results if r["layer"] == l]
        if not layer_results:
            continue
        norms = {}
        for comp in ["input", "attn", "mlp", "output"]:
            vals = [r["components"].get(comp, {}).get("norm", 0) for r in layer_results]
            norms[comp] = np.mean(vals) if vals else 0
        print(f"L{l:>4} | {norms.get('input',0):>10.1f} | {norms.get('attn',0):>10.1f} | {norms.get('mlp',0):>10.1f} | {norms.get('output',0):>10.1f}")

    print("\nLegend: lift = J-lens R² / random R². >2x = signal. ~1x = null.")
    print("If ATTN shows signal where K and V individually don't -> K×V reconstruction confirmed")
    print("If MLP shows signal -> workspace lives in feedforward, not attention")
    print("If INPUT shows signal -> concept passed through from earlier layer")

    out_path = f"residual_decomposition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "date": datetime.now().isoformat(),
                "experiment": "Residual Stream Decomposition",
                "model": model_name,
                "n_prompts": len(PROMPTS),
                "k_concepts": K_CONCEPTS,
                "n_random": N_RANDOM,
            },
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
