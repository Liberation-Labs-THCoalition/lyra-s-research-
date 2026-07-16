"""
Substrate Independence Test: Does the workspace change across the
full-attention / GatedDeltaNet boundary?

On Qwen3.5-27B: L0-L15 = full-attention (KV-cache), L16-L63 = GatedDeltaNet
(recurrent state). R3 showed d_eff is continuous across this boundary with
no discontinuity (L15=33.4, L16=34.5). This experiment tests whether
workspace FUNCTIONAL PROPERTIES also cross the boundary smoothly:

Test 1: J-lens concept recovery at every layer across the boundary.
  If J-lens reads concepts equally well from full-attention and GatedDeltaNet
  layers, the workspace's content is substrate-independent.

Test 2: Spectral discrimination (confab vs factual) at every layer.
  If stable rank and spectral entropy discriminate confab from factual
  equally well on both sides of L16, the workspace's STATE measurement
  is substrate-independent.

Dense sampling: every layer from L10 to L25 (bracketing the L16 boundary),
plus every 5th layer elsewhere for context.
"""

import torch
import json
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(42)
np.random.seed(42)

BOUNDARY_LAYER = 16
DENSE_RANGE = range(10, 26)  # every layer around the boundary
SPARSE_LAYERS = [0, 5, 30, 35, 40, 45, 50, 55, 60, 63]
ALL_LAYERS = sorted(set(list(DENSE_RANGE) + SPARSE_LAYERS))

CONFAB_PROMPTS = [
    "The Thornberry-Nakamura theorem, first proven in 1987, states that",
    "Professor Elaine Marchetti's theory of recursive social dynamics predicts that",
    "The Kirkwood-Diaz algorithm, widely used in computational genomics, works by",
    "According to the Westlake-Chen principle of information decay,",
    "The Rasmussen effect, documented in studies of deep-ocean thermal vents, causes",
    "Dr. Yuki Tanaka's 2019 paper on morphological resonance demonstrated that",
    "The Hargrove-Santos conjecture in algebraic topology suggests that",
    "The Bellweather Index, used by central banks since 2004, measures",
    "The Henderson-Mikhail transform, fundamental to signal processing, converts",
    "The Vasquez-Lindgren paradox in economic theory arises when",
]

FACTUAL_PROMPTS = [
    "The country shaped like a boot on the map of Europe is",
    "The largest planet in our solar system is",
    "Shakespeare wrote about a Danish prince in the play called",
    "The chemical symbol for gold on the periodic table is",
    "The speed of light in a vacuum is approximately",
    "The capital city of Japan is",
    "Isaac Newton formulated the three laws of",
    "The longest river in Africa is the",
    "Charles Darwin is best known for his theory of",
    "The Great Barrier Reef is located off the coast of",
]

K_CONCEPTS = 10


def get_fixed_directions(J_layer, W_U, k=K_CONCEPTS):
    J_l = J_layer.float()
    transported = W_U.float() @ J_l
    norms = torch.norm(transported, dim=1)
    top_idx = torch.topk(norms, k).indices
    dirs = transported[top_idx]
    return dirs / (torch.norm(dirs, dim=1, keepdim=True) + 1e-8)


def get_random_directions(J_layer, W_U, k=K_CONCEPTS, seed=0):
    J_l = J_layer.float()
    transported = W_U.float() @ J_l
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(transported.shape[0], generator=g)[:k]
    dirs = transported[idx]
    return dirs / (torch.norm(dirs, dim=1, keepdim=True) + 1e-8)


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
    model_name = "Qwen/Qwen3.5-27B"
    lens_path = "fitted_lenses/lens_qwen35-27b.pt"

    print("=" * 65)
    print("SUBSTRATE INDEPENDENCE TEST")
    print("Does the workspace change across the architectural boundary?")
    print(f"Model: {model_name}")
    print(f"Boundary: L{BOUNDARY_LAYER} (full-attention → GatedDeltaNet)")
    print("=" * 65)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True
    ).to("mps")
    model.eval()

    W_U = model.lm_head.weight.detach().cpu().float()
    lens_data = torch.load(lens_path, map_location="cpu")
    J = lens_data["J"]
    n_layers = len(model.model.layers)

    # Pre-compute J-lens directions per layer
    fixed_dirs = {}
    for l in ALL_LAYERS:
        if l in J:
            fixed_dirs[l] = get_fixed_directions(J[l], W_U, k=K_CONCEPTS)

    # ===== TEST 1: J-lens concept recovery across boundary =====
    print("\n--- TEST 1: J-lens concept recovery ---")

    jlens_results = {l: [] for l in ALL_LAYERS}

    all_prompts = CONFAB_PROMPTS + FACTUAL_PROMPTS
    for pidx, prompt in enumerate(all_prompts):
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)

        for l in ALL_LAYERS:
            if l not in fixed_dirs:
                continue
            h = out.hidden_states[l + 1][0, -1, :].float().cpu()
            r2 = compute_r2(fixed_dirs[l], h)

            r2_rands = []
            for draw in range(10):
                rd = get_random_directions(J[l], W_U, k=K_CONCEPTS,
                                           seed=42 + draw + pidx * 100 + l * 1000)
                r2_rands.append(compute_r2(rd, h))

            jlens_results[l].append({
                "r2": r2,
                "r2_rand": np.mean(r2_rands),
                "lift": r2 / max(np.mean(r2_rands), 1e-6),
                "category": "confab" if pidx < len(CONFAB_PROMPTS) else "factual",
            })

        if (pidx + 1) % 5 == 0:
            print(f"  {pidx + 1}/{len(all_prompts)} prompts done")

    # ===== TEST 2: Spectral discrimination across boundary =====
    print("\n--- TEST 2: Spectral features (confab vs factual) ---")

    spectral_results = {l: {"confab": [], "factual": []} for l in ALL_LAYERS}

    for pidx, prompt in enumerate(all_prompts):
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        category = "confab" if pidx < len(CONFAB_PROMPTS) else "factual"

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)

        for l in ALL_LAYERS:
            H = out.hidden_states[l + 1][0, :, :].float().cpu()

            if hasattr(model.model.layers[l], "self_attn") and hasattr(model.model.layers[l].self_attn, "v_proj"):
                W_V = model.model.layers[l].self_attn.v_proj.weight.float().cpu()
                V_mat = (W_V @ H.T).T
            else:
                V_mat = H

            sv = torch.linalg.svdvals(V_mat)
            sv_pos = sv[sv > 1e-8]
            stable_rank = float((sv_pos.sum()**2 / (sv_pos**2).sum()).item()) if len(sv_pos) > 0 else 0.0
            p = sv_pos**2 / (sv_pos**2).sum()
            spectral_entropy = float(-(p * p.log()).sum().item()) if len(p) > 0 else 0.0

            spectral_results[l][category].append({
                "stable_rank": stable_rank,
                "spectral_entropy": spectral_entropy,
            })

    # ===== RESULTS =====
    print("\n" + "=" * 65)
    print("TEST 1: J-LENS CONCEPT RECOVERY ACROSS BOUNDARY")
    print(f"{'Layer':>6} {'Depth%':>6} {'Regime':>15} | {'Lift':>6} | {'R²':>8} | {'Rand':>8}")
    print("-" * 60)

    for l in ALL_LAYERS:
        if l not in jlens_results or not jlens_results[l]:
            continue
        depth = l / n_layers * 100
        regime = "Full-Attn" if l < BOUNDARY_LAYER else "GatedDeltaNet"
        lift = np.mean([r["lift"] for r in jlens_results[l]])
        r2 = np.mean([r["r2"] for r in jlens_results[l]])
        rand = np.mean([r["r2_rand"] for r in jlens_results[l]])
        marker = " <<<" if l == BOUNDARY_LAYER else ""
        print(f"L{l:>4} {depth:>5.0f}% {regime:>15} | {lift:>5.1f}x | {r2:>7.4f} | {rand:>7.4f}{marker}")

    # Boundary comparison
    pre = [np.mean([r["lift"] for r in jlens_results[l]]) for l in range(BOUNDARY_LAYER-3, BOUNDARY_LAYER) if l in jlens_results]
    post = [np.mean([r["lift"] for r in jlens_results[l]]) for l in range(BOUNDARY_LAYER, BOUNDARY_LAYER+3) if l in jlens_results]
    if pre and post:
        print(f"\nBoundary: pre-mean={np.mean(pre):.2f}x  post-mean={np.mean(post):.2f}x  delta={np.mean(post)-np.mean(pre):+.2f}x")

    print("\n" + "=" * 65)
    print("TEST 2: SPECTRAL DISCRIMINATION ACROSS BOUNDARY")
    print(f"{'Layer':>6} {'Depth%':>6} {'Regime':>15} | {'SR_confab':>9} {'SR_fact':>9} {'SR_diff':>8} | {'SE_confab':>9} {'SE_fact':>9}")
    print("-" * 85)

    for l in ALL_LAYERS:
        if not spectral_results[l]["confab"] or not spectral_results[l]["factual"]:
            continue
        depth = l / n_layers * 100
        regime = "Full-Attn" if l < BOUNDARY_LAYER else "GatedDeltaNet"
        sr_c = np.mean([r["stable_rank"] for r in spectral_results[l]["confab"]])
        sr_f = np.mean([r["stable_rank"] for r in spectral_results[l]["factual"]])
        se_c = np.mean([r["spectral_entropy"] for r in spectral_results[l]["confab"]])
        se_f = np.mean([r["spectral_entropy"] for r in spectral_results[l]["factual"]])
        marker = " <<<" if l == BOUNDARY_LAYER else ""
        print(f"L{l:>4} {depth:>5.0f}% {regime:>15} | {sr_c:>9.2f} {sr_f:>9.2f} {sr_c-sr_f:>+8.2f} | {se_c:>9.2f} {se_f:>9.2f}{marker}")

    # Boundary comparison for discrimination
    pre_diff = [np.mean([r["stable_rank"] for r in spectral_results[l]["confab"]]) -
                np.mean([r["stable_rank"] for r in spectral_results[l]["factual"]])
                for l in range(BOUNDARY_LAYER-3, BOUNDARY_LAYER)
                if spectral_results[l]["confab"]]
    post_diff = [np.mean([r["stable_rank"] for r in spectral_results[l]["confab"]]) -
                 np.mean([r["stable_rank"] for r in spectral_results[l]["factual"]])
                 for l in range(BOUNDARY_LAYER, BOUNDARY_LAYER+3)
                 if spectral_results[l]["confab"]]
    if pre_diff and post_diff:
        print(f"\nSR discrimination: pre-boundary={np.mean(pre_diff):+.3f}  post-boundary={np.mean(post_diff):+.3f}  delta={np.mean(post_diff)-np.mean(pre_diff):+.3f}")

    print("\n" + "=" * 65)
    print("SUBSTRATE INDEPENDENCE VERDICT")
    print("=" * 65)
    if pre and post:
        jlens_continuous = abs(np.mean(post) - np.mean(pre)) < 0.5
        print(f"  J-lens continuity: {'YES' if jlens_continuous else 'NO'} (delta={np.mean(post)-np.mean(pre):+.2f}x)")
    if pre_diff and post_diff:
        spectral_continuous = abs(np.mean(post_diff) - np.mean(pre_diff)) < 0.3
        print(f"  Spectral discrimination continuity: {'YES' if spectral_continuous else 'NO'} (delta={np.mean(post_diff)-np.mean(pre_diff):+.3f})")
    if pre and post and pre_diff and post_diff:
        if jlens_continuous and spectral_continuous:
            print("  → SUBSTRATE INDEPENDENT: workspace properties continuous across boundary")
        else:
            print("  → SUBSTRATE DEPENDENT: workspace properties change at boundary")

    out_path = f"substrate_independence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "date": datetime.now().isoformat(),
                "experiment": "Substrate Independence Test",
                "model": model_name,
                "boundary_layer": BOUNDARY_LAYER,
                "n_confab": len(CONFAB_PROMPTS),
                "n_factual": len(FACTUAL_PROMPTS),
            },
            "jlens_results": {str(l): v for l, v in jlens_results.items()},
            "spectral_results": {str(l): v for l, v in spectral_results.items()},
        }, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
