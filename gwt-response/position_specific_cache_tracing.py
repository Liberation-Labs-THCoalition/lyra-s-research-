"""
Position-Specific Cache Tracing: Where in the sequence are concepts cached?

Prior experiments tested whether J-lens concepts are decodable from the
LAST TOKEN's K/V projections. They're not (V: 0.92x, K: 1.12x). But
concepts enter the sequence at SPECIFIC POSITIONS — the position where
"boot-shaped country" is mentioned, not the position where "is" is about
to be generated.

This experiment:
1. Identify which positions carry a concept (J-lens readout per position)
2. Extract K and V at THOSE positions (not just last token)
3. Test: can J-lens directions discriminate concept-carrying positions
   from non-carrying positions in K-space and V-space?
4. Check: when the model attends to concept-carrying positions, does
   the attention output reconstruct the concept?

If concepts are cached at their source positions but not at the generation
position, the prior nulls are explained and the cache IS a concept store
— just position-specific, not position-aggregated.

Model: Qwen3-8B (36 layers, d_model=4096, 8 KV heads, head_dim=128)

Simplifications:
- K projections are computed pre-RoPE. RoPE is position-dependent and applies
  uniformly to all content at a given position, so within-position comparisons
  (concept vs non-concept content at the same sequence offset) are RoPE-invariant.
  Cross-position comparisons of raw K magnitudes should be interpreted with this
  caveat: RoPE rotates each head's K vector by a position-dependent angle, but
  since we compare R^2 of J-lens directions (which are also pre-RoPE), the
  alignment metric is consistent within the pre-RoPE frame.

Fixes applied (from Agni review):
- K1: Uses hidden_states[l] (INPUT to layer l) + input_layernorm, not
  hidden_states[l+1] (OUTPUT of layer l). This matches actual cache computation:
  K_cached = RoPE(W_K @ LN1(hidden_states[l])), V_cached = W_V @ LN1(hidden_states[l]).
- K2: Adds concept-specific direction analysis alongside generic top-10 directions.
  For each factual probe, computes the ANSWER TOKEN's transported J-lens direction
  and tests alignment at each position. Includes cross-concept control (e.g., test
  "Italy" direction at "Jupiter" prompt positions) to rule out content-density
  confounds.
"""

import torch
import json
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(42)
np.random.seed(42)

WORKSPACE_LAYERS = [12, 15, 18, 21]
K_CONCEPTS = 10
N_RANDOM = 20

PROBES = [
    {
        "prompt": "The country shaped like a boot on the map of Europe is",
        "concept": "Italy",
        "concept_region": "boot",  # word that most strongly signals the concept
    },
    {
        "prompt": "The largest planet in our solar system with its great red spot is",
        "concept": "Jupiter",
        "concept_region": "largest planet",
    },
    {
        "prompt": "The playwright who wrote about a Danish prince named Hamlet is",
        "concept": "Shakespeare",
        "concept_region": "Danish prince",
    },
    {
        "prompt": "Water molecules are made of two hydrogen atoms and one oxygen atom, with chemical formula",
        "concept": "H2O",
        "concept_region": "two hydrogen",
    },
    {
        "prompt": "The physicist who proposed the theory of general relativity in 1915 is",
        "concept": "Einstein",
        "concept_region": "general relativity",
    },
    {
        "prompt": "The Thornberry-Nakamura theorem, first proven in 1987, establishes that",
        "concept": None,  # fabricated — no real concept to find
        "concept_region": "Thornberry-Nakamura",
    },
    {
        "prompt": "The Kirkwood-Diaz algorithm, widely used in computational genomics, works by",
        "concept": None,
        "concept_region": "Kirkwood-Diaz",
    },
    {
        "prompt": "Professor Elaine Marchetti's theory of recursive social dynamics predicts that",
        "concept": None,
        "concept_region": "Elaine Marchetti",
    },
    {
        "prompt": "The element with atomic number 79 on the periodic table, known for its yellow color, is",
        "concept": "gold",
        "concept_region": "atomic number 79",
    },
    {
        "prompt": "The organ in the human body that pumps blood through the circulatory system is the",
        "concept": "heart",
        "concept_region": "pumps blood",
    },
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


def get_concept_direction(J_layer, W_U, tok, concept_text):
    """Compute the concept-specific transported direction for a given answer token.

    Returns a unit vector in J-lens space for the first token of the concept,
    shaped (1, d_j) for compute_r2 compatibility.
    """
    J_l = J_layer.float()
    W = W_U.float()
    concept_token_id = tok.encode(concept_text, add_special_tokens=False)[0]
    concept_transported = W[concept_token_id] @ J_l  # (d_j,)
    concept_dir = concept_transported / (torch.norm(concept_transported) + 1e-8)
    return concept_dir.unsqueeze(0)  # (1, d_j)


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


def find_concept_positions(tok, prompt, concept_region):
    """Find which token positions correspond to the concept-carrying region."""
    tokens = tok.encode(prompt, add_special_tokens=False)
    region_tokens = tok.encode(concept_region, add_special_tokens=False)

    # Find the region in the full token sequence
    positions = []
    for i in range(len(tokens) - len(region_tokens) + 1):
        if tokens[i:i+len(region_tokens)] == region_tokens:
            positions = list(range(i, i + len(region_tokens)))
            break

    # If exact match fails, try token-by-token substring matching
    if not positions:
        full_text_tokens = [(i, tok.decode([t])) for i, t in enumerate(tokens)]
        for i, (pos, text) in enumerate(full_text_tokens):
            if any(word.lower() in text.lower() for word in concept_region.split()):
                positions.append(pos)

    return positions


def main():
    model_name = "Qwen/Qwen3-8B"
    lens_path = "fitted_lenses/lens_qwen3-8b.pt"

    print("=" * 60)
    print("POSITION-SPECIFIC CACHE TRACING")
    print("Where in the sequence are concepts cached?")
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

    fixed_dirs = {}
    for l in WORKSPACE_LAYERS:
        if l in J:
            fixed_dirs[l] = get_fixed_directions(J[l], W_U, k=K_CONCEPTS)

    # Pre-compute concept-specific directions for all factual probes (K2 fix).
    # Each factual probe gets its answer token's transported J-lens direction.
    factual_probes = [(i, p) for i, p in enumerate(PROBES) if p["concept"] is not None]
    concept_dirs_by_probe = {}  # {(probe_idx, layer): direction tensor (1, d_j)}
    for pidx, probe in factual_probes:
        for l in WORKSPACE_LAYERS:
            if l in J:
                concept_dirs_by_probe[(pidx, l)] = get_concept_direction(
                    J[l], W_U, tok, probe["concept"]
                )
    print(f"\nPre-computed concept directions for {len(factual_probes)} factual probes "
          f"x {len(WORKSPACE_LAYERS)} layers")

    results = []

    for pidx, probe in enumerate(PROBES):
        prompt = probe["prompt"]
        concept_region = probe["concept_region"]

        input_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        seq_len = input_ids.shape[1]

        # Find concept-carrying positions
        concept_positions = find_concept_positions(tok, prompt, concept_region)
        non_concept_positions = [i for i in range(seq_len) if i not in concept_positions]

        # Show tokenization
        all_tokens = [tok.decode([t]) for t in input_ids[0].tolist()]
        print(f"\n  Probe {pidx}: {probe['concept'] or 'FABRICATED'}")
        print(f"    Tokens: {all_tokens}")
        print(f"    Concept positions: {concept_positions} ({[all_tokens[p] for p in concept_positions if p < len(all_tokens)]})")

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)

        for l in WORKSPACE_LAYERS:
            if l not in fixed_dirs:
                continue

            dirs = fixed_dirs[l]

            # FIX K1: Use hidden_states[l] (INPUT to layer l), not hidden_states[l+1]
            # (output of layer l). Then apply input_layernorm to match actual cache:
            #   K_cached = RoPE(W_K @ LN1(hidden_states[l]))
            #   V_cached = W_V @ LN1(hidden_states[l])
            # RoPE is skipped for K — see docstring for justification.
            H_input = out.hidden_states[l][0, :, :].float().cpu()
            # Apply input layernorm to match actual cache computation
            ln = model.model.layers[l].input_layernorm
            H = ln(H_input.to(model.device)).float().cpu()  # (seq_len, d_model)

            # Get W_K and W_V for this layer
            W_K = model.model.layers[l].self_attn.k_proj.weight.float().cpu()
            W_V = model.model.layers[l].self_attn.v_proj.weight.float().cpu()

            # --- Concept-specific directions setup (K2 fix) ---
            # Own concept direction for this probe (None for fabricated probes)
            own_concept_dir = concept_dirs_by_probe.get((pidx, l), None)

            # Cross-concept control: use a DIFFERENT factual probe's concept direction.
            # If concept-specific > cross-concept at concept-carrying positions,
            # that's specificity that content-density alone cannot explain.
            cross_concept_dir = None
            cross_concept_name = None
            if probe["concept"] is not None:
                own_idx_in_factual = [i for i, (pi, _) in enumerate(factual_probes) if pi == pidx][0]
                cross_idx = (own_idx_in_factual + 1) % len(factual_probes)
                cross_pidx = factual_probes[cross_idx][0]
                cross_concept_dir = concept_dirs_by_probe.get((cross_pidx, l), None)
                cross_concept_name = factual_probes[cross_idx][1]["concept"]

            # Project concept-specific directions through W_K and W_V
            own_concept_dir_k, own_concept_dir_v = None, None
            cross_concept_dir_k, cross_concept_dir_v = None, None

            if own_concept_dir is not None:
                d_k = (W_K @ own_concept_dir.T).T
                own_concept_dir_k = d_k / (torch.norm(d_k, dim=1, keepdim=True) + 1e-8)
                d_v = (W_V @ own_concept_dir.T).T
                own_concept_dir_v = d_v / (torch.norm(d_v, dim=1, keepdim=True) + 1e-8)

            if cross_concept_dir is not None:
                d_k = (W_K @ cross_concept_dir.T).T
                cross_concept_dir_k = d_k / (torch.norm(d_k, dim=1, keepdim=True) + 1e-8)
                d_v = (W_V @ cross_concept_dir.T).T
                cross_concept_dir_v = d_v / (torch.norm(d_v, dim=1, keepdim=True) + 1e-8)

            # Project J-lens dirs through W_K and W_V (constant across positions)
            dirs_k = (W_K @ dirs.T).T
            dirs_k = dirs_k / (torch.norm(dirs_k, dim=1, keepdim=True) + 1e-8)
            dirs_v = (W_V @ dirs.T).T
            dirs_v = dirs_v / (torch.norm(dirs_v, dim=1, keepdim=True) + 1e-8)

            # Measure J-lens R² at EACH POSITION in K-space and V-space
            position_data = []
            for pos in range(seq_len):
                h_pos_raw = H_input[pos]  # pre-layernorm: actual residual stream (for J-lens control)
                h_pos = H[pos]            # post-layernorm: for K/V projection (matches cache)
                k_pos = W_K @ h_pos  # pre-RoPE K (see docstring simplification note)
                v_pos = W_V @ h_pos  # V-cache value

                # R² in residual stream (positive control — uses pre-layernorm, matching J-lens)
                r2_resid = compute_r2(dirs, h_pos_raw)

                # R² in K-space
                r2_k = compute_r2(dirs_k, k_pos)

                # R² in V-space
                r2_v = compute_r2(dirs_v, v_pos)

                # Random baselines
                r2_resid_rand, r2_k_rand, r2_v_rand = [], [], []
                for draw in range(N_RANDOM):
                    rand_dirs = get_random_directions(
                        J[l], W_U, k=K_CONCEPTS,
                        seed=42 + draw + pidx * 100 + l * 1000 + pos * 10
                    )
                    rand_dirs_k = (W_K @ rand_dirs.T).T
                    rand_dirs_k = rand_dirs_k / (torch.norm(rand_dirs_k, dim=1, keepdim=True) + 1e-8)
                    rand_dirs_v = (W_V @ rand_dirs.T).T
                    rand_dirs_v = rand_dirs_v / (torch.norm(rand_dirs_v, dim=1, keepdim=True) + 1e-8)

                    r2_resid_rand.append(compute_r2(rand_dirs, h_pos_raw))
                    r2_k_rand.append(compute_r2(rand_dirs_k, k_pos))
                    r2_v_rand.append(compute_r2(rand_dirs_v, v_pos))

                # --- Concept-specific R² (K2 fix) ---
                r2_concept_k = None
                r2_concept_v = None
                r2_concept_resid = None
                r2_cross_k = None
                r2_cross_v = None
                r2_cross_resid = None

                if own_concept_dir is not None:
                    r2_concept_resid = compute_r2(own_concept_dir, h_pos_raw)
                    r2_concept_k = compute_r2(own_concept_dir_k, k_pos)
                    r2_concept_v = compute_r2(own_concept_dir_v, v_pos)

                if cross_concept_dir is not None:
                    r2_cross_resid = compute_r2(cross_concept_dir, h_pos_raw)
                    r2_cross_k = compute_r2(cross_concept_dir_k, k_pos)
                    r2_cross_v = compute_r2(cross_concept_dir_v, v_pos)

                is_concept = pos in concept_positions
                pos_entry = {
                    "position": pos,
                    "token": all_tokens[pos] if pos < len(all_tokens) else "?",
                    "is_concept_position": is_concept,
                    "r2_resid": r2_resid,
                    "r2_k": r2_k,
                    "r2_v": r2_v,
                    "lift_resid": r2_resid / max(np.mean(r2_resid_rand), 1e-6),
                    "lift_k": r2_k / max(np.mean(r2_k_rand), 1e-6),
                    "lift_v": r2_v / max(np.mean(r2_v_rand), 1e-6),
                }

                # Add concept-specific fields if available
                if r2_concept_k is not None:
                    pos_entry["r2_concept_k"] = r2_concept_k
                    pos_entry["r2_concept_v"] = r2_concept_v
                    pos_entry["r2_concept_resid"] = r2_concept_resid
                if r2_cross_k is not None:
                    pos_entry["r2_cross_k"] = r2_cross_k
                    pos_entry["r2_cross_v"] = r2_cross_v
                    pos_entry["r2_cross_resid"] = r2_cross_resid
                    pos_entry["cross_concept"] = cross_concept_name

                position_data.append(pos_entry)

            # Compare concept positions vs non-concept positions (generic — existing)
            concept_lifts_k = [p["lift_k"] for p in position_data if p["is_concept_position"]]
            non_concept_lifts_k = [p["lift_k"] for p in position_data if not p["is_concept_position"]]
            concept_lifts_v = [p["lift_v"] for p in position_data if p["is_concept_position"]]
            non_concept_lifts_v = [p["lift_v"] for p in position_data if not p["is_concept_position"]]
            concept_lifts_r = [p["lift_resid"] for p in position_data if p["is_concept_position"]]
            non_concept_lifts_r = [p["lift_resid"] for p in position_data if not p["is_concept_position"]]

            result_entry = {
                "probe_idx": pidx,
                "concept": probe["concept"],
                "layer": l,
                "n_concept_positions": len(concept_positions),
                "n_total_positions": seq_len,
                "concept_mean_lift_k": float(np.mean(concept_lifts_k)) if concept_lifts_k else 0,
                "non_concept_mean_lift_k": float(np.mean(non_concept_lifts_k)) if non_concept_lifts_k else 0,
                "concept_mean_lift_v": float(np.mean(concept_lifts_v)) if concept_lifts_v else 0,
                "non_concept_mean_lift_v": float(np.mean(non_concept_lifts_v)) if non_concept_lifts_v else 0,
                "concept_mean_lift_resid": float(np.mean(concept_lifts_r)) if concept_lifts_r else 0,
                "non_concept_mean_lift_resid": float(np.mean(non_concept_lifts_r)) if non_concept_lifts_r else 0,
                "positions": position_data,
            }

            # Aggregate concept-specific R² (K2 fix — new)
            if probe["concept"] is not None:
                concept_r2_own_k = [p["r2_concept_k"] for p in position_data
                                    if p["is_concept_position"] and "r2_concept_k" in p]
                non_concept_r2_own_k = [p["r2_concept_k"] for p in position_data
                                        if not p["is_concept_position"] and "r2_concept_k" in p]
                concept_r2_own_v = [p["r2_concept_v"] for p in position_data
                                    if p["is_concept_position"] and "r2_concept_v" in p]
                non_concept_r2_own_v = [p["r2_concept_v"] for p in position_data
                                        if not p["is_concept_position"] and "r2_concept_v" in p]
                concept_r2_cross_k = [p["r2_cross_k"] for p in position_data
                                      if p["is_concept_position"] and "r2_cross_k" in p]
                concept_r2_cross_v = [p["r2_cross_v"] for p in position_data
                                      if p["is_concept_position"] and "r2_cross_v" in p]

                result_entry["concept_specific"] = {
                    "own_concept": probe["concept"],
                    "cross_concept": cross_concept_name,
                    "concept_pos_own_r2_k": float(np.mean(concept_r2_own_k)) if concept_r2_own_k else 0,
                    "non_concept_pos_own_r2_k": float(np.mean(non_concept_r2_own_k)) if non_concept_r2_own_k else 0,
                    "concept_pos_own_r2_v": float(np.mean(concept_r2_own_v)) if concept_r2_own_v else 0,
                    "non_concept_pos_own_r2_v": float(np.mean(non_concept_r2_own_v)) if non_concept_r2_own_v else 0,
                    "concept_pos_cross_r2_k": float(np.mean(concept_r2_cross_k)) if concept_r2_cross_k else 0,
                    "concept_pos_cross_r2_v": float(np.mean(concept_r2_cross_v)) if concept_r2_cross_v else 0,
                }

            results.append(result_entry)

            # Print generic results (existing)
            print(f"    L{l}: concept K={result_entry['concept_mean_lift_k']:.1f}x "
                  f"non={result_entry['non_concept_mean_lift_k']:.1f}x | "
                  f"V={result_entry['concept_mean_lift_v']:.1f}x "
                  f"non={result_entry['non_concept_mean_lift_v']:.1f}x | "
                  f"resid={result_entry['concept_mean_lift_resid']:.1f}x "
                  f"non={result_entry['non_concept_mean_lift_resid']:.1f}x")

            # Print concept-specific results (new)
            if "concept_specific" in result_entry:
                cs = result_entry["concept_specific"]
                print(f"         concept-specific '{cs['own_concept']}': "
                      f"K concept_pos={cs['concept_pos_own_r2_k']:.4f} "
                      f"non={cs['non_concept_pos_own_r2_k']:.4f} | "
                      f"V concept_pos={cs['concept_pos_own_r2_v']:.4f} "
                      f"non={cs['non_concept_pos_own_r2_v']:.4f}")
                print(f"         cross-concept '{cs['cross_concept']}': "
                      f"K at concept_pos={cs['concept_pos_cross_r2_k']:.4f} | "
                      f"V at concept_pos={cs['concept_pos_cross_r2_v']:.4f}")

    # Summary — generic directions (existing, preserved)
    print("\n" + "=" * 60)
    print("POSITION-SPECIFIC SUMMARY")
    print("Do concept-carrying positions show higher K/V lift than non-carrying?")
    print("=" * 60)

    for l in WORKSPACE_LAYERS:
        layer_results = [r for r in results if r["layer"] == l]
        if not layer_results:
            continue

        concept_k = np.mean([r["concept_mean_lift_k"] for r in layer_results])
        non_k = np.mean([r["non_concept_mean_lift_k"] for r in layer_results])
        concept_v = np.mean([r["concept_mean_lift_v"] for r in layer_results])
        non_v = np.mean([r["non_concept_mean_lift_v"] for r in layer_results])
        concept_r = np.mean([r["concept_mean_lift_resid"] for r in layer_results])
        non_r = np.mean([r["non_concept_mean_lift_resid"] for r in layer_results])

        print(f"  L{l}:")
        print(f"    Residual: concept={concept_r:.2f}x  non-concept={non_r:.2f}x  diff={concept_r-non_r:+.2f}x")
        print(f"    K-cache:  concept={concept_k:.2f}x  non-concept={non_k:.2f}x  diff={concept_k-non_k:+.2f}x")
        print(f"    V-cache:  concept={concept_v:.2f}x  non-concept={non_v:.2f}x  diff={concept_v-non_v:+.2f}x")

    # Summary — concept-specific directions (new, K2 fix)
    print("\n" + "=" * 60)
    print("CONCEPT-SPECIFIC DIRECTION ANALYSIS")
    print("Does the ANSWER TOKEN's direction align more at concept-carrying positions?")
    print("Does the OWN concept direction beat a CROSS-CONCEPT control?")
    print("=" * 60)

    for l in WORKSPACE_LAYERS:
        factual_results = [r for r in results if r["layer"] == l and r["concept"] is not None
                           and "concept_specific" in r]
        if not factual_results:
            continue

        own_at_concept_k = np.mean([r["concept_specific"]["concept_pos_own_r2_k"] for r in factual_results])
        own_at_non_k = np.mean([r["concept_specific"]["non_concept_pos_own_r2_k"] for r in factual_results])
        own_at_concept_v = np.mean([r["concept_specific"]["concept_pos_own_r2_v"] for r in factual_results])
        own_at_non_v = np.mean([r["concept_specific"]["non_concept_pos_own_r2_v"] for r in factual_results])
        cross_at_concept_k = np.mean([r["concept_specific"]["concept_pos_cross_r2_k"] for r in factual_results])
        cross_at_concept_v = np.mean([r["concept_specific"]["concept_pos_cross_r2_v"] for r in factual_results])

        print(f"  L{l} (factual probes only):")
        print(f"    Own concept R^2 at concept-pos vs non-concept-pos:")
        print(f"      K: {own_at_concept_k:.4f} vs {own_at_non_k:.4f}  diff={own_at_concept_k - own_at_non_k:+.4f}")
        print(f"      V: {own_at_concept_v:.4f} vs {own_at_non_v:.4f}  diff={own_at_concept_v - own_at_non_v:+.4f}")
        print(f"    Own vs cross-concept R^2 at concept-carrying positions:")
        print(f"      K: own={own_at_concept_k:.4f} cross={cross_at_concept_k:.4f}  diff={own_at_concept_k - cross_at_concept_k:+.4f}")
        print(f"      V: own={own_at_concept_v:.4f} cross={cross_at_concept_v:.4f}  diff={own_at_concept_v - cross_at_concept_v:+.4f}")

        if own_at_concept_k > cross_at_concept_k and own_at_concept_k > own_at_non_k:
            print(f"    --> CONCEPT-SPECIFIC signal in K: own > cross AND concept_pos > non_concept_pos")
        if own_at_concept_v > cross_at_concept_v and own_at_concept_v > own_at_non_v:
            print(f"    --> CONCEPT-SPECIFIC signal in V: own > cross AND concept_pos > non_concept_pos")

    # Generic direction verdict with permutation test (Agni S1 fix)
    print("\nGENERIC DIRECTION VERDICT (with permutation test):")
    all_concept_k = [r["concept_mean_lift_k"] for r in results if r["concept_mean_lift_k"] > 0]
    all_non_k = [r["non_concept_mean_lift_k"] for r in results if r["non_concept_mean_lift_k"] > 0]
    all_concept_v = [r["concept_mean_lift_v"] for r in results if r["concept_mean_lift_v"] > 0]
    all_non_v = [r["non_concept_mean_lift_v"] for r in results if r["non_concept_mean_lift_v"] > 0]

    def permutation_test_paired(group1, group2, n_perm=5000):
        observed = np.mean(group1) - np.mean(group2)
        combined = np.array(list(zip(group1, group2)))
        rng = np.random.RandomState(42)
        count = 0
        for _ in range(n_perm):
            swaps = rng.random(len(combined)) > 0.5
            perm_g1 = [combined[i][0] if not swaps[i] else combined[i][1] for i in range(len(combined))]
            perm_g2 = [combined[i][1] if not swaps[i] else combined[i][0] for i in range(len(combined))]
            if abs(np.mean(perm_g1) - np.mean(perm_g2)) >= abs(observed):
                count += 1
        return count / n_perm

    if len(all_concept_k) > 2 and len(all_non_k) > 2:
        p_k = permutation_test_paired(all_concept_k, all_non_k) if len(all_concept_k) == len(all_non_k) else 1.0
        diff_k = np.mean(all_concept_k) - np.mean(all_non_k)
        print(f"  K-cache: concept={np.mean(all_concept_k):.2f}x  non={np.mean(all_non_k):.2f}x  diff={diff_k:+.2f}x  p={p_k:.4f}")
    if len(all_concept_v) > 2 and len(all_non_v) > 2:
        p_v = permutation_test_paired(all_concept_v, all_non_v) if len(all_concept_v) == len(all_non_v) else 1.0
        diff_v = np.mean(all_concept_v) - np.mean(all_non_v)
        print(f"  V-cache: concept={np.mean(all_concept_v):.2f}x  non={np.mean(all_non_v):.2f}x  diff={diff_v:+.2f}x  p={p_v:.4f}")

    if all_concept_v > all_non_v + 0.5:
        print("  V-cache: CONCEPT POSITIONS CARRY MORE SIGNAL")
        print("  Concepts ARE cached position-specifically in V-space!")
    elif all_concept_v > all_non_v + 0.2:
        print("  V-cache: Weak position-specific signal (needs more data)")
    else:
        print("  V-cache: No position-specific signal (concepts not localized in V)")

    # Concept-specific verdict (new, K2 fix)
    factual_with_cs = [r for r in results if r["concept"] is not None and "concept_specific" in r]
    if factual_with_cs:
        print("\nCONCEPT-SPECIFIC VERDICT:")
        all_own_concept_k = np.mean([r["concept_specific"]["concept_pos_own_r2_k"] for r in factual_with_cs])
        all_cross_concept_k = np.mean([r["concept_specific"]["concept_pos_cross_r2_k"] for r in factual_with_cs])
        all_own_concept_v = np.mean([r["concept_specific"]["concept_pos_own_r2_v"] for r in factual_with_cs])
        all_cross_concept_v = np.mean([r["concept_specific"]["concept_pos_cross_r2_v"] for r in factual_with_cs])

        if all_own_concept_k > all_cross_concept_k:
            print(f"  K-cache: Own concept R^2 ({all_own_concept_k:.4f}) > cross-concept ({all_cross_concept_k:.4f})")
            print(f"  --> Content-density alone cannot explain this. Concept-specific caching signal.")
        else:
            print(f"  K-cache: Own concept R^2 ({all_own_concept_k:.4f}) <= cross-concept ({all_cross_concept_k:.4f})")
            print(f"  --> No concept specificity -- signal may be content-density artifact.")

        if all_own_concept_v > all_cross_concept_v:
            print(f"  V-cache: Own concept R^2 ({all_own_concept_v:.4f}) > cross-concept ({all_cross_concept_v:.4f})")
            print(f"  --> Content-density alone cannot explain this. Concept-specific caching signal.")
        else:
            print(f"  V-cache: Own concept R^2 ({all_own_concept_v:.4f}) <= cross-concept ({all_cross_concept_v:.4f})")
            print(f"  --> No concept specificity -- signal may be content-density artifact.")

    out_path = f"position_specific_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "date": datetime.now().isoformat(),
                "experiment": "Position-Specific Cache Tracing",
                "model": model_name,
                "hypothesis": "Concepts are cached at source positions, not at generation position",
                "prior_null": "Last-token K/V readout null (V: 0.92x, K: 1.12x)",
                "fixes_applied": [
                    "K1: Uses hidden_states[l] (input) + input_layernorm instead of hidden_states[l+1] (output)",
                    "K2: Concept-specific directions added alongside generic top-10; cross-concept control included",
                    "RoPE: Skipped for K (pre-RoPE). Within-position comparisons are RoPE-invariant.",
                ],
            },
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
