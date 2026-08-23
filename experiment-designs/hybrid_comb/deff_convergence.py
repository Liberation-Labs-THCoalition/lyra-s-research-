"""
d_eff convergence + fixed-m substrate contrast.
Pre-registered: experiment-designs/deff_convergence_and_fixed_m.md (Agni WARN/pass/0.92).

Replaces the with-replacement bootstrap of substrate_aware_rerun.py, which was
invalid: d_eff is a function of n, so resampling with replacement (~57 of 90
distinct) estimates a different quantity than the full sample. Its point
estimates fell outside their own CIs.

No GPU. Reads two cached activation files.
"""
import json, os
import numpy as np
import torch

KV = os.environ.get("KV_DIR", os.path.expanduser("~/lab/kv-experiments"))
SEED = 42
B_CONV = 200                                  # draws per m, Experiment A
B_FIXED = 1000                                # draws, Experiment B
M_LADDER = [20, 30, 40, 50, 60, 70, 80, 89]   # 90 excluded: exactly one subsample
M_FIXED = [45, 60, 75]                        # 60 primary, 45/75 sensitivity
DEGREES = (4, 6, 8)

MODELS = [("qwen35-27b", "hybrid"), ("qwen3-32b", "dense")]

GUARDS = []


def guard(name, ok, detail):
    GUARDS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def d_eff(H):
    """(n, d) -> participation ratio of covariance eigenvalues, via n x n Gram."""
    H = H - H.mean(dim=0, keepdim=True)
    G = H @ H.T / (H.shape[0] - 1)
    ev = torch.linalg.eigvalsh(G)
    ev = ev[ev > 1e-8]
    if len(ev) == 0:
        return 0.0
    t = ev.sum()
    return float((t ** 2 / (ev ** 2).sum()).item())


def _bench(T):
    """Re-verify Gram == covariance on this data, every run. Never trust the table."""
    import time

    def d_eff_cov(H):
        H = H - H.mean(dim=0, keepdim=True)
        C = H.T @ H / (H.shape[0] - 1)
        ev = torch.linalg.eigvalsh(C); ev = ev[ev > 1e-8]
        t = ev.sum(); return float((t ** 2 / (ev ** 2).sum()).item())

    t0 = time.time(); a = [d_eff_cov(T[:, l, :]) for l in range(4)]; tc = time.time() - t0
    t0 = time.time(); b = [d_eff(T[:, l, :]) for l in range(4)]; tg = time.time() - t0
    rel = max(abs(x - y) / max(x, 1e-12) for x, y in zip(a, b))
    guard("gram-equivalence", rel < 1e-4,
          f"max rel diff {rel:.2e} over 4 layers, {tc/max(tg,1e-9):.0f}x faster")
    return rel


def detrend(y, deg):
    x = np.arange(len(y), dtype=float)
    return y - np.polyval(np.polyfit(x, y, deg), x)


def load(name):
    z = np.load(os.path.join(KV, f"acts_{name}.npz"), allow_pickle=True)
    return torch.from_numpy(z["acts"]), [str(t) for t in z["layer_types"]]


def mask_for(layer_types, kind, n_layers):
    full = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    linear = [i for i, t in enumerate(layer_types) if "linear" in t or "delta" in t.lower()]
    if kind == "hybrid":
        assert full == list(range(3, n_layers, 4)) and len(full) == 16, full
        m = np.zeros(n_layers, bool); m[full] = True
        return m, "TRUE substrate mask", len(full), len(linear)
    assert not linear, f"dense control reports linear layers: {linear}"
    return (np.arange(n_layers) % 4) == 3, "SHAM period-4 mask", len(full), 0


# ------------------------------------------------------- Experiment A
def convergence(name, T, n_layers):
    print(f"\n{'='*74}\nEXPERIMENT A — d_eff vs sample size, {name}\n{'='*74}")
    rng = np.random.default_rng(SEED)
    n = T.shape[0]
    curve = {}
    for m in M_LADDER:
        prof = np.zeros((B_CONV, n_layers))
        for b in range(B_CONV):
            idx = rng.choice(n, m, replace=False)
            Tb = T[idx]
            prof[b] = [d_eff(Tb[:, l, :]) for l in range(n_layers)]
        curve[m] = prof.mean(axis=0)
    peak = int(curve[89].argmax())
    print(f"  {'m':>4} {'mean d_eff':>11} {'peak layer':>11} {'d_eff@peak':>11} {'ceiling m-1':>12}")
    for m in M_LADDER:
        print(f"  {m:>4} {curve[m].mean():>11.3f} {int(curve[m].argmax()):>11d} "
              f"{curve[m][peak]:>11.3f} {m-1:>12d}")
    # is it still rising at the top of the ladder?
    d_last = curve[89] - curve[80]
    frac_rising = float((d_last > 0).mean())
    rel_step = float(np.median(d_last / np.maximum(curve[80], 1e-9)))
    print(f"\n  layers still rising from m=80 to m=89 : {frac_rising*100:.1f}% "
          f"({int((d_last>0).sum())}/{n_layers})")
    print(f"  median relative increase over that step: {rel_step*100:+.2f}%")
    print(f"  d_eff@peak as fraction of ceiling (m=89): {curve[89][peak]/88:.3f}")
    # does peak location move with m?
    peaks = {m: int(curve[m].argmax()) for m in M_LADDER}
    print(f"  peak layer by m: {peaks}")
    guard("peak-stability", len(set(peaks.values())) <= 2,
          f"{len(set(peaks.values()))} distinct peak layers across the m-ladder")
    return {"curve": {str(m): curve[m].tolist() for m in M_LADDER},
            "frac_rising_80_to_89": frac_rising,
            "median_rel_step": rel_step,
            "peaks": {str(k): v for k, v in peaks.items()}}


# ------------------------------------------------------- Experiment B
def fixed_m_contrast(name, T, n_layers, mask, label):
    print(f"\n{'='*74}\nEXPERIMENT B — fixed-m contrast, {name}  [{label}]\n{'='*74}")
    rng = np.random.default_rng(SEED)
    n = T.shape[0]
    out = {}
    for m in M_FIXED:
        prof = np.zeros((B_FIXED, n_layers))
        seen = set()
        for b in range(B_FIXED):
            idx = rng.choice(n, m, replace=False)
            seen.add(tuple(sorted(idx.tolist())))
            Tb = T[idx]
            prof[b] = [d_eff(Tb[:, l, :]) for l in range(n_layers)]
        tag = "PRIMARY" if m == 60 else "sensitivity"
        print(f"\n  m = {m}  ({tag}), B = {B_FIXED}, distinct prompt-sets drawn = {len(seen)}")
        guard(f"estimand-invariance m={m}", len(seen) > B_FIXED * 0.9,
              f"{len(seen)}/{B_FIXED} distinct subsets, all size {m}")
        out[str(m)] = {}
        for deg in DEGREES:
            R = np.array([detrend(prof[b], deg) for b in range(B_FIXED)])
            stats = R[:, mask].mean(axis=1) - R[:, ~mask].mean(axis=1)
            centre = float(np.median(stats))
            lo, hi = np.percentile(stats, [2.5, 97.5])
            uniq = len(np.unique(np.round(stats, 10)))
            contained = lo <= centre <= hi
            excl = "EXCLUDES 0" if (lo > 0 or hi < 0) else "includes 0"
            print(f"    deg {deg}: centre={centre:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"{excl:11s} unique={uniq}")
            guard(f"containment m={m} deg{deg}", contained,
                  f"centre {centre:+.4f} in [{lo:+.4f}, {hi:+.4f}]")
            out[str(m)][f"deg{deg}"] = dict(centre=centre, ci=[float(lo), float(hi)],
                                            excludes_zero=bool(lo > 0 or hi < 0), unique=uniq)
    return out


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    results = {}
    for name, kind in MODELS:
        T, lt = load(name)
        n_layers = T.shape[1]
        print(f"\n\n{'#'*74}\n# {name}  ({kind})  acts={tuple(T.shape)}\n{'#'*74}")
        print("GUARDS:")
        _bench(T)
        mask, label, n_full, n_lin = mask_for(lt, kind, n_layers)
        guard("architecture-from-config", True,
              f"{n_full} full_attention / {n_lin} linear, mask='{label}'")
        results[name] = {
            "A_convergence": convergence(name, T, n_layers),
            "B_fixed_m": fixed_m_contrast(name, T, n_layers, mask, label),
        }

    print(f"\n\n{'='*74}\nGUARD SUMMARY\n{'='*74}")
    n_fail = sum(1 for _, ok, _ in GUARDS if not ok)
    for nm, ok, det in GUARDS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {nm}")
    print(f"\n  {len(GUARDS)-n_fail}/{len(GUARDS)} PASS, {n_fail} FAIL")
    results["_guards"] = [{"name": n, "pass": o, "detail": d} for n, o, d in GUARDS]
    out = os.path.join(KV, "deff_convergence_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
