"""
Hybrid comb test — does Qwen3.5-27B's interleaved substrate leave a period-4
signature in the per-layer d_eff profile?

Pre-registered: experiment-designs/hybrid_comb_test.md (Agni WARN 0.92, pre-run).
No GPU. No model load. Reads two existing JSON artifacts plus the shipped config.
"""
import json, os, sys
import numpy as np

# Host-specific roots come from the environment so this file carries no local paths.
KV = os.environ.get("KV_DIR", os.path.expanduser("~/lab/kv-experiments"))
HF = os.environ.get("HF_CACHE", os.path.expanduser("~/models/hf_cache"))
CFG = (f"{HF}/models--Qwen--Qwen3.5-27B/"
       "snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654/config.json")

F27 = f"{KV}/r3_band_measurement_20260714_170944.json"
F8B = f"{KV}/r3_band_measurement_20260714_171708.json"

DEGREES = (4, 6, 8)


def profile(path, model_key, field):
    d = json.load(open(path))["results"][model_key]
    prof = d[field]
    n = d["n_layers"]
    x = np.array([prof[str(i)] for i in range(n)], dtype=float)
    assert len(x) == n and np.isfinite(x).all(), f"bad profile {path}:{field}"
    return x, n, d


def detrend(y, deg):
    x = np.arange(len(y), dtype=float)
    c = np.polyfit(x, y, deg)
    return y - np.polyval(c, x)


def group_stat(resid, mask):
    """mean(resid | mask) - mean(resid | ~mask)"""
    return resid[mask].mean() - resid[~mask].mean()


def shift_null(resid, mask, stat_fn):
    """Circular-shift null. Returns (T_obs, p_two_sided, null_array).
    k=0 (observed) is included in the null set, so p floors at 1/n."""
    n = len(mask)
    Ts = np.array([stat_fn(resid, np.roll(mask, k)) for k in range(n)])
    T_obs = Ts[0]
    p = float((np.abs(Ts) >= abs(T_obs) - 1e-12).sum()) / n
    return T_obs, p, Ts


def crossings_from(mask):
    return np.array([i for i in range(len(mask) - 1) if mask[i] != mask[i + 1]])


def crossing_stat(absdelta, mask):
    """mean|delta| at substrate crossings - mean|delta| at same-substrate steps."""
    cr = crossings_from(mask)
    sel = np.zeros(len(absdelta), dtype=bool)
    sel[cr] = True
    if sel.all() or (~sel).all():
        return np.nan
    return absdelta[sel].mean() - absdelta[~sel].mean()


def crossing_null(absdelta, mask):
    n = len(mask)
    Ts = np.array([crossing_stat(absdelta, np.roll(mask, k)) for k in range(n)])
    T_obs = Ts[0]
    ok = np.isfinite(Ts)
    p = float((np.abs(Ts[ok]) >= abs(T_obs) - 1e-12).sum()) / ok.sum()
    return T_obs, p, Ts


def run(label, y, mask, n, tail_cut=None):
    out = {}
    print(f"\n{'='*72}\n{label}   (n_layers={n}, mask_true={int(mask.sum())})\n{'='*72}")

    # --- primary: phase-class group difference, three polynomial degrees ---
    print("\n[1] PRIMARY  mean(resid|full) - mean(resid|linear), circular-shift null")
    print(f"    {'deg':>4} {'T_obs':>10} {'p':>8}   {'null sd':>9}")
    for deg in DEGREES:
        r = detrend(y, deg)
        T, p, Ts = shift_null(r, mask, group_stat)
        out[f"deg{deg}"] = dict(T=float(T), p=float(p), null_sd=float(Ts.std()))
        print(f"    {deg:>4} {T:>10.4f} {p:>8.4f}   {Ts.std():>9.4f}")

    # --- all four phase classes, degree 6 ---
    print("\n[2] ALL FOUR PHASE CLASSES (deg 6) — pre-specified, no phase selection")
    r6 = detrend(y, 6)
    idx = np.arange(n)
    for ph in range(4):
        m = (idx % 4) == ph
        T, p, _ = shift_null(r6, m, group_stat)
        tag = "  <-- TRUE substrate phase" if np.array_equal(m, mask) else ""
        print(f"    l = {ph} (mod 4): n={int(m.sum()):>2}  T={T:>9.4f}  p={p:.4f}{tag}")

    # --- crossing test (the prereg's own stated mitigation) ---
    print("\n[3] CROSSING TEST  mean|delta| at crossings - at same-substrate steps")
    ad_raw = np.abs(np.diff(y))
    T, p, _ = crossing_null(ad_raw, mask)
    print(f"    raw d_eff  (pre-specified):  n_cross={len(crossings_from(mask))}  "
          f"T={T:>9.4f}  p={p:.4f}")
    out["cross_raw"] = dict(T=float(T), p=float(p))
    ad_res = np.abs(np.diff(r6))
    T, p, _ = crossing_null(ad_res, mask)
    print(f"    deg-6 residual (ADDED, not pre-specified): T={T:>9.4f}  p={p:.4f}")
    out["cross_resid"] = dict(T=float(T), p=float(p))

    # --- tail sensitivity: I looked at L>=48, so report without it ---
    if tail_cut is not None and tail_cut < n:
        print(f"\n[4] WITHOUT THE TAIL I LOOKED AT (drop L >= {tail_cut}) — deg 6")
        yt, mt = y[:tail_cut], mask[:tail_cut]
        rt = detrend(yt, 6)
        T, p, _ = shift_null(rt, mt, group_stat)
        print(f"    n={tail_cut}  mask_true={int(mt.sum())}  T={T:>9.4f}  p={p:.4f}")
        out["no_tail"] = dict(T=float(T), p=float(p), n=int(tail_cut))
        print(f"\n[5] TAIL ONLY (L >= {tail_cut}) — descriptive, NOT a pre-specified test")
        yb, mb = y[tail_cut:], mask[tail_cut:]
        rb = detrend(yb, 2)
        T, p, _ = shift_null(rb, mb, group_stat)
        print(f"    n={n-tail_cut} mask_true={int(mb.sum())}  T={T:>9.4f}  p={p:.4f}  (deg-2 detrend)")
        out["tail_only"] = dict(T=float(T), p=float(p), n=int(n - tail_cut))
    return out


def main():
    results = {}

    # ---------- architecture from the SHIPPED config, never hardcoded ----------
    lt = json.load(open(CFG))["text_config"]["layer_types"]
    n27 = len(lt)
    full = [i for i, t in enumerate(lt) if t == "full_attention"]
    expected = list(range(3, n27, 4))
    print(f"ARCHITECTURE (from shipped config)")
    print(f"  n_layers            = {n27}")
    print(f"  full_attention at   = {full}")
    print(f"  n full / n linear   = {len(full)} / {n27 - len(full)}")
    print(f"  substrate crossings = {len(crossings_from(np.isin(np.arange(n27), full)))}")
    assert n27 == 64, f"expected 64 layers, got {n27}"
    assert len(full) == 16, f"expected 16 full-attention layers, got {len(full)}"
    assert full == expected, f"expected {expected}, got {full}"
    print("  ASSERTIONS PASS -> proceeding\n")

    y27, n, meta27 = profile(F27, "qwen35-27b", "d_eff")
    assert n == n27, f"profile has {n} layers, config says {n27}"
    mask27 = np.isin(np.arange(n27), full)
    print(f"27B artifact: {meta27['model']}  n_prompts={meta27['n_prompts']}  "
          f"d_model={meta27['d_model']}")

    results["hybrid_d_eff"] = run(
        "TEST — Qwen3.5-27B (HYBRID), d_eff, TRUE substrate mask",
        y27, mask27, n27, tail_cut=48)

    # confirmation metric, same artifact, different functional
    k27, _, _ = profile(F27, "qwen35-27b", "kurtosis")
    results["hybrid_kurtosis"] = run(
        "CONFIRMATION ONLY — Qwen3.5-27B, kurtosis, TRUE substrate mask",
        k27, mask27, n27, tail_cut=48)

    # ---------- dense negative control, sham mask ----------
    y8, n8, meta8 = profile(F8B, "qwen3-8b", "d_eff")
    print(f"\n8B artifact: {meta8['model']}  n_prompts={meta8['n_prompts']}  "
          f"d_model={meta8['d_model']}  n_layers={n8}")
    sham = (np.arange(n8) % 4) == 3
    results["dense_control_d_eff"] = run(
        "NEGATIVE CONTROL — Qwen3-8B (DENSE), d_eff, SHAM period-4 mask",
        y8, sham, n8, tail_cut=None)

    json.dump(results, open("/tmp/hybrid_comb_results.json", "w"), indent=2)
    print("\nsaved -> /tmp/hybrid_comb_results.json")


if __name__ == "__main__":
    main()
