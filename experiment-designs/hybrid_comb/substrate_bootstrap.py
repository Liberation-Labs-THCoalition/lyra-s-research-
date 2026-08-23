"""
Task #94 — substrate-aware rerun with a REAL sampling distribution.

Pre-registered: experiment-designs/substrate_aware_rerun.md (Agni WARN/pass/0.92).

Phase A: one forward pass per model, cache last-token hidden states per layer.
Phase B: 1000 prompt bootstrap resamples -> per-layer d_eff CIs, a 1000-draw CI on
         the full-vs-linear phase contrast, and a CI on band onset.
Phase C: BLOCKING -- print the UNIQUE-value count of every null before its p-value.
         (Earned 2026-08-22: a period-4 circular-shift null had 4 distinct draws,
         not 64, so its claimed floor p=0.016 was really 0.25.)
Phase D: depth-matched dense control, Qwen3-32B (dense, 64 layers).

d_eff and kurtosis are lifted verbatim from r3_band_measurement.py so the estimator
is unchanged. The bootstrap resamples the INPUT SET; it does not touch the formula.
"""
import json, os, sys, time, gc
import numpy as np
import torch

KV = os.environ.get("KV_DIR", os.path.expanduser("~/lab/kv-experiments"))
OUT = os.environ.get("OUT_DIR", KV)
sys.path.insert(0, KV)

N_BOOT = int(os.environ.get("N_BOOT", "1000"))
SEED = 42
DEGREES = (4, 6, 8)

MODELS = [
    ("qwen35-27b", "Qwen/Qwen3.5-27B", "hybrid"),
    ("qwen3-32b", "Qwen/Qwen3-32B", "dense"),
]


# ---------------------------------------------------------------- estimators
def d_eff_from(H):
    """H: (n_prompts, d_model) float tensor. SAME ESTIMATOR as r3_band_measurement.

    r3 eigendecomposes the d x d covariance (5120 x 5120). With n=90 prompts that
    matrix has rank <= n-1, so its nonzero spectrum is identical to that of the
    n x n Gram matrix -- and d_eff = (sum L)^2 / sum(L^2) depends only on the
    NONZERO eigenvalues (r3 filters ev > 1e-8 too). So this is the same estimator,
    not an approximation of it.

    Verified 2026-08-22 on the cached 27B activations: max relative difference
    2.3e-06 across layers, 7497x faster. The covariance form would have taken
    122.8 HOURS for 1000 resamples x 64 layers; this takes ~1 minute.
    """
    H = H - H.mean(dim=0, keepdim=True)
    gram = H @ H.T / (H.shape[0] - 1)          # n x n, same nonzero spectrum
    ev = torch.linalg.eigvalsh(gram)
    ev = ev[ev > 1e-8]
    if len(ev) == 0:
        return 0.0
    total = ev.sum()
    return float((total ** 2 / (ev ** 2).sum()).item())


def kurtosis_from(H):
    norms = torch.norm(H, dim=1, keepdim=True)
    Hn = H / (norms + 1e-8)
    vals = Hn.flatten()
    std = vals.std()
    if std < 1e-10:
        return 0.0
    z = (vals - vals.mean()) / std
    return float((z ** 4).mean().item()) - 3.0


def detrend(y, deg):
    x = np.arange(len(y), dtype=float)
    return y - np.polyval(np.polyfit(x, y, deg), x)


def find_onset(profile, sigma=2.0):
    """First layer exceeding the mean of layers 1-4 by > sigma SD. Same rule as R3."""
    early = profile[1:5]
    thr = early.mean() + sigma * early.std()
    hits = np.where(profile > thr)[0]
    return int(hits[0]) if len(hits) else -1


# ---------------------------------------------------------------- Phase C guard
def report_null(name, null_vals, obs):
    """BLOCKING: a null is only as strong as its number of DISTINCT values."""
    null_vals = np.asarray(null_vals, dtype=float)
    uniq = len(np.unique(np.round(null_vals, 10)))
    n = len(null_vals)
    floor = 1.0 / uniq if uniq else float("nan")
    p = float((np.abs(null_vals) >= abs(obs) - 1e-12).sum()) / n
    print(f"    [{name}] draws={n}  UNIQUE={uniq}  floor_p={floor:.4f}  "
          f"obs={obs:+.4f}  p={p:.4f}", end="")
    if uniq < 20:
        print("   <-- DEGENERATE NULL, p is not interpretable")
    elif abs(p - floor) < 1e-9:
        print("   <-- p PINNED AT FLOOR, underpowered")
    else:
        print()
    return dict(draws=n, unique=uniq, floor_p=floor, obs=float(obs), p=p)


# ---------------------------------------------------------------- Phase A
def capture(model_name, model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import r3_band_measurement as r3

    prompts = r3.PROSE_PROMPTS + r3.CHAT_PROMPTS
    prompts = prompts[:90]
    print(f"\n{'='*72}\nPHASE A — {model_name} ({model_id})\n{'='*72}")
    print(f"  prompts: {len(prompts)} (PROSE {len(r3.PROSE_PROMPTS)} + CHAT {len(r3.CHAT_PROMPTS)})")

    cache_path = os.path.join(OUT, f"acts_{model_name}.npz")
    if os.path.exists(cache_path):
        print(f"  cache exists -> {cache_path}, skipping forward pass")
        # layer_types is saved dtype=object, so allow_pickle is required. Without
        # it np.load raises and the model is silently reported NOT RUN.
        z = np.load(cache_path, allow_pickle=True)
        return z["acts"], [str(t) for t in z["layer_types"]], prompts

    tok_kwargs = {}
    if "qwen3" in model_id.lower() and "qwen3.5" not in model_id.lower():
        tok_kwargs["tokenizer_type"] = "qwen3"
    tok = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, trust_remote_code=True).to("mps")
    model.eval()
    n_layers = len(model.model.layers)
    d_model = model.config.hidden_size if hasattr(model.config, "hidden_size") \
        else model.config.text_config.hidden_size
    print(f"  loaded in {time.time()-t0:.0f}s  n_layers={n_layers} d_model={d_model}")

    cfg = model.config
    lt = getattr(cfg, "layer_types", None)
    if lt is None and hasattr(cfg, "text_config"):
        lt = getattr(cfg.text_config, "layer_types", None)
    lt = list(lt) if lt is not None else ["dense"] * n_layers
    n_full = sum(1 for t in lt if t == "full_attention")
    print(f"  layer_types from config: {n_full} full_attention / {n_layers - n_full} other")

    acts = np.zeros((len(prompts), n_layers, d_model), dtype=np.float32)
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        for l in range(n_layers):
            acts[i, l] = out.hidden_states[l + 1][0, -1, :].float().cpu().numpy()
        del out
        if (i + 1) % 15 == 0:
            print(f"    {i+1}/{len(prompts)}")

    np.savez_compressed(cache_path, acts=acts, layer_types=np.array(lt, dtype=object))
    print(f"  cached -> {cache_path}  ({acts.nbytes/1e6:.0f} MB in memory)")

    del model
    gc.collect()
    torch.mps.empty_cache()
    return acts, lt, prompts


# ---------------------------------------------------------------- Phase B
def bootstrap(model_name, acts, layer_types, kind):
    n_prompts, n_layers, _ = acts.shape
    full = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    linear = [i for i, t in enumerate(layer_types) if "linear" in t or "delta" in t.lower()]
    if kind == "hybrid":
        assert full == list(range(3, n_layers, 4)) and len(full) == 16, \
            f"substrate mask unexpected: {full}"
        assert len(linear) == n_layers - 16, f"expected 48 linear layers, got {len(linear)}"
        mask = np.zeros(n_layers, bool); mask[full] = True
        mask_label = "TRUE substrate mask"
    else:
        # A dense model is one with NO linear-attention layers. Note it does NOT
        # omit layer_types -- Qwen3-32B labels all 64 layers "full_attention".
        # The earlier `assert not full` was wrong and would have failed the control.
        assert not linear, f"dense control reports linear-attention layers: {linear}"
        mask = (np.arange(n_layers) % 4) == 3
        mask_label = "SHAM period-4 mask (dense control)"

    print(f"\n{'='*72}\nPHASE B — {model_name}: {N_BOOT} prompt bootstrap  [{mask_label}]\n{'='*72}")
    T = torch.from_numpy(acts)
    rng = np.random.default_rng(SEED)

    point = np.array([d_eff_from(T[:, l, :]) for l in range(n_layers)])
    point_kurt = np.array([kurtosis_from(T[:, l, :]) for l in range(n_layers)])
    print(f"  point-estimate d_eff: min={point.min():.2f} max={point.max():.2f} "
          f"argmax=L{int(point.argmax())}  onset(2sig)=L{find_onset(point)}")

    boot = np.zeros((N_BOOT, n_layers))
    onsets = np.zeros(N_BOOT, int)
    t0 = time.time()
    for b in range(N_BOOT):
        idx = rng.integers(0, n_prompts, n_prompts)
        Tb = T[idx]
        prof = np.array([d_eff_from(Tb[:, l, :]) for l in range(n_layers)])
        boot[b] = prof
        onsets[b] = find_onset(prof)
        if (b + 1) % 200 == 0:
            print(f"    {b+1}/{N_BOOT}  ({time.time()-t0:.0f}s)")

    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    res = {"point_d_eff": point.tolist(), "point_kurtosis": point_kurt.tolist(),
           "ci_lo": lo.tolist(), "ci_hi": hi.tolist(),
           "n_prompts": int(n_prompts), "n_layers": int(n_layers),
           "n_boot": N_BOOT, "mask_label": mask_label}

    # find_onset returns -1 when no layer clears threshold. Those are NOT layer
    # indices and must not enter a percentile. Report them as a failure rate.
    ok = onsets >= 0
    n_undef = int((~ok).sum())
    if ok.sum() >= 2:
        olo, ohi = int(np.percentile(onsets[ok], 2.5)), int(np.percentile(onsets[ok], 97.5))
    else:
        olo = ohi = -1
    print(f"\n  BAND ONSET (2sigma): point L{find_onset(point)}  "
          f"bootstrap CI [L{olo}, L{ohi}] over {int(ok.sum())}/{N_BOOT} defined resamples"
          f"  unique={len(np.unique(onsets[ok]))}")
    if n_undef:
        print(f"    !! onset UNDEFINED in {n_undef}/{N_BOOT} resamples "
              f"({100*n_undef/N_BOOT:.1f}%) — the band itself is not robust")
    res["onset_point"] = int(find_onset(point))
    res["onset_ci"] = [olo, ohi]
    res["onset_undefined_frac"] = n_undef / N_BOOT

    print(f"\n  PHASE CONTRAST  mean(resid|masked) - mean(resid|rest), bootstrap CI")
    res["contrast"] = {}
    for deg in DEGREES:
        Tb = np.array([detrend(boot[b], deg) for b in range(N_BOOT)])
        stats = Tb[:, mask].mean(axis=1) - Tb[:, ~mask].mean(axis=1)
        obs = float(detrend(point, deg)[mask].mean() - detrend(point, deg)[~mask].mean())
        clo, chi = np.percentile(stats, [2.5, 97.5])
        excl = "EXCLUDES 0" if (clo > 0 or chi < 0) else "includes 0"
        print(f"    deg {deg}:  T_obs={obs:+.4f}  CI [{clo:+.4f}, {chi:+.4f}]  {excl}"
              f"   unique={len(np.unique(np.round(stats,10)))}")
        res["contrast"][f"deg{deg}"] = dict(obs=obs, ci=[float(clo), float(chi)],
                                            excludes_zero=bool(clo > 0 or chi < 0),
                                            unique=int(len(np.unique(np.round(stats, 10)))))

    print(f"\n  PHASE C GUARD — all four phase classes, deg 6, bootstrap null")
    r6_point = detrend(point, 6)
    Tb6 = np.array([detrend(boot[b], 6) for b in range(N_BOOT)])
    res["phases"] = {}
    for ph in range(4):
        m = (np.arange(n_layers) % 4) == ph
        stats = Tb6[:, m].mean(axis=1) - Tb6[:, ~m].mean(axis=1)
        obs = float(r6_point[m].mean() - r6_point[~m].mean())
        tag = "  <-- substrate" if np.array_equal(m, mask) else ""
        res["phases"][f"l{ph}"] = report_null(f"l={ph} mod 4{tag}", stats - stats.mean(), obs)
    return res


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    all_res = {}
    for name, mid, kind in MODELS:
        try:
            acts, lt, _ = capture(name, mid)
            all_res[name] = bootstrap(name, acts, lt, kind)
            all_res[name]["status"] = "ok"
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"\n  !! {name} FAILED: {type(e).__name__}: {e}")
            print(f"  !! Reporting as NOT RUN. Do NOT substitute a different-depth model.")
            all_res[name] = {"status": "NOT RUN", "error": f"{type(e).__name__}: {e}"}
        gc.collect()
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    out = os.path.join(OUT, "substrate_bootstrap_results.json")
    json.dump(all_res, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
