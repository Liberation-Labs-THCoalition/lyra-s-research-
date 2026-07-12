"""
Causal Injection: Insert concepts into workspace, measure V-cache geometry change.

The observational approach (reading concepts from the cache) failed —
selection artifact. The causal approach: inject a concept direction into the
residual stream at the workspace band and measure:
  1. Does V-cache geometry change at DOWNSTREAM layers?
  2. Does the logit-lens readout change (concept propagates)?
  3. Does the model's output change (behavioral effect)?

No selection artifact: the direction is chosen independently of the hidden state.
"""

import torch
import json
import numpy as np
from datetime import datetime
from pathlib import Path


def spectral_features(v_proj_output):
    """Compute stable rank, spectral entropy of a V-projection vector set."""
    v = v_proj_output.cpu().float()
    if v.dim() == 1:
        v = v.unsqueeze(0)
    centered = v - v.mean(dim=0, keepdim=True)
    try:
        _, S, _ = torch.linalg.svd(centered, full_matrices=False)
        S = S + 1e-10
        stable_rank = float((S.sum() ** 2 / (S ** 2).sum()).item())
        p = S / S.sum()
        entropy = float(-(p * torch.log(p)).sum().item())
        return {"stable_rank": stable_rank, "spectral_entropy": entropy}
    except Exception:
        return {"stable_rank": 0, "spectral_entropy": 0}


def main():
    import sys
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-8B"
    for arg in sys.argv:
        if arg.startswith("--model="):
            model_name = arg.split("=", 1)[1]

    print("=" * 60)
    print("CAUSAL INJECTION: Concepts into Workspace")
    print("Model: %s" % model_name)
    print("=" * 60)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True
    ).to("mps")
    model.eval()

    W_U = model.lm_head.weight.detach().float()
    n_layers = len(model.model.layers)

    # Concepts to inject — chosen for clear behavioral predictions
    concepts = {
        "danger": tok.encode("danger", add_special_tokens=False)[0],
        "happy": tok.encode("happy", add_special_tokens=False)[0],
        "lie": tok.encode("lie", add_special_tokens=False)[0],
        "math": tok.encode("math", add_special_tokens=False)[0],
        "french": tok.encode("french", add_special_tokens=False)[0],
    }
    print("Concepts: %s" % {k: tok.decode([v]) for k, v in concepts.items()})

    # Neutral probes
    probes = [
        "The meeting has been moved to noon on Thursday. Please confirm",
        "The report on quarterly earnings showed moderate growth in the",
        "She opened the door and walked into the room where everyone was",
        "The temperature outside is approximately twenty degrees and the sky is",
    ]

    # Injection layers (workspace band)
    inject_layers = [12, 18, 24]
    # Measurement layers (downstream of injection)
    measure_layers = list(range(0, n_layers, 3))

    # Doses
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0]

    results = {
        "metadata": {
            "model": model_name,
            "date": datetime.now().isoformat(),
            "inject_layers": inject_layers,
            "measure_layers": measure_layers,
            "alphas": alphas,
            "concepts": {k: tok.decode([v]) for k, v in concepts.items()},
        },
        "trials": [],
    }

    for probe_idx, probe in enumerate(probes):
        print("\nProbe %d: '%s...'" % (probe_idx + 1, probe[:50]))
        input_ids = tok(probe, return_tensors="pt").input_ids.to(model.device)
        seq_len = input_ids.shape[1]

        for concept_name, concept_id in concepts.items():
            concept_dir = W_U[concept_id].to(model.device)  # [d_model]

            for inject_layer in inject_layers:
                for alpha in alphas:

                    # Hook to inject concept direction at inject_layer
                    hook_handles = []

                    def make_inject_hook(layer_idx, direction, dose):
                        def hook_fn(module, input, output):
                            # output is a tuple; first element is hidden states
                            h = output[0]
                            if layer_idx == inject_layer and dose > 0:
                                h = h.clone()
                                h[0, -1, :] += dose * direction.to(h.dtype)
                            return (h,) + output[1:]
                        return hook_fn

                    # Register injection hook
                    for li in range(n_layers):
                        layer = model.model.layers[li]
                        h = layer.register_forward_hook(
                            make_inject_hook(li, concept_dir, alpha)
                        )
                        hook_handles.append(h)

                    # Forward pass with injection
                    with torch.no_grad():
                        outputs = model(input_ids, output_hidden_states=True)

                    # Remove hooks
                    for h in hook_handles:
                        h.remove()

                    hidden_states = outputs.hidden_states

                    # Measure 1: logit-lens readout at each measurement layer
                    readouts = {}
                    for ml in measure_layers:
                        h_ml = hidden_states[ml + 1][0, -1, :].float()
                        h_normed = model.model.norm(
                            h_ml.unsqueeze(0).unsqueeze(0)
                        ).squeeze()
                        ll_logits = (W_U.to(h_normed.device) @ h_normed)
                        top5 = torch.topk(ll_logits, 5)
                        readouts[ml] = {
                            "tokens": [tok.decode([t]).strip()
                                      for t in top5.indices.tolist()],
                            "concept_rank": int(
                                (ll_logits.argsort(descending=True) == concept_id
                                 ).nonzero(as_tuple=True)[0].item()
                            ) + 1 if concept_id in top5.indices.tolist()[:100] else 999,
                        }

                    # Measure 2: V-cache geometry at measurement layers
                    geometry = {}
                    for ml in measure_layers:
                        if hasattr(model.model.layers[ml], 'self_attn'):
                            h_ml = hidden_states[ml + 1][0, -1, :].float()
                            W_V = model.model.layers[ml].self_attn.v_proj.weight.float()
                            v_proj = W_V @ h_ml
                            geometry[ml] = {
                                "v_norm": float(torch.norm(v_proj).item()),
                            }

                    # Measure 3: model output (top-5 tokens)
                    final_logits = outputs.logits[0, -1, :]
                    top5_output = torch.topk(final_logits, 5)
                    output_tokens = [tok.decode([t]).strip()
                                    for t in top5_output.indices.tolist()]

                    trial = {
                        "probe_idx": probe_idx,
                        "concept": concept_name,
                        "inject_layer": inject_layer,
                        "alpha": alpha,
                        "readouts": {str(k): v for k, v in readouts.items()},
                        "geometry": {str(k): v for k, v in geometry.items()},
                        "output_tokens": output_tokens,
                    }
                    results["trials"].append(trial)

            print("  %s done" % concept_name)

    # Summary
    print("\n" + "=" * 60)
    print("CAUSAL INJECTION SUMMARY")
    print("=" * 60)

    for concept_name in concepts:
        print("\n--- %s ---" % concept_name)
        for inject_layer in inject_layers:
            print("  Inject at L%d:" % inject_layer)
            for alpha in alphas:
                trials = [t for t in results["trials"]
                         if t["concept"] == concept_name
                         and t["inject_layer"] == inject_layer
                         and t["alpha"] == alpha]
                if trials:
                    # Average concept rank at the deepest measurement layer
                    deep_ml = str(measure_layers[-1])
                    ranks = [t["readouts"].get(deep_ml, {}).get("concept_rank", 999)
                            for t in trials]
                    mean_rank = np.mean(ranks)
                    outputs = [t["output_tokens"][0] for t in trials]
                    print("    alpha=%.1f: concept rank=%.0f, outputs=%s" % (
                        alpha, mean_rank, outputs[:2]))

    out_path = "causal_injection_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to %s" % out_path)


if __name__ == "__main__":
    main()
