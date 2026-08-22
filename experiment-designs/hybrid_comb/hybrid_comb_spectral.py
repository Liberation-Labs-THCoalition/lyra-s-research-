"""
Addendum to hybrid_comb_test.py.

WHY THIS EXISTS: the pre-specified circular-shift null is DEGENERATE. Rolling a
period-4 mask gives only 4 distinct masks (roll depends on k mod 4), so the 64
"draws" are 4 values x16 and the p-value floors at 0.25 — not the 0.016 the design
claimed. The design's power statement was arithmetically wrong.

This is replaced for a structural reason, not an outcome reason: the original test
could not reject under ANY data. Both tests are reported.

The replacement asks the period-4 question directly, in the frequency domain, where
the null is the distribution of power across the other frequencies of the same
detrended profile — a real null with ~29 draws.
"""
import json, os
import numpy as np

# Host-specific roots come from the environment so this file carries no local paths.
KV = os.environ.get("KV_DIR", os.path.expanduser("~/lab/kv-experiments"))
HF = os.environ.get("HF_CACHE", os.path.expanduser("~/models/hf_cache"))
CFG = (f"{HF}/models--Qwen--Qwen3.5-27B/"
       "snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654/config.json")
F27 = f"{KV}/r3_band_measurement_20260714_170944.json"
F8B = f"{KV}/r3_band_measurement_20260714_171708.json"

# Frequencies below this index are suppressed by the degree-6 detrend and are
# therefore excluded from the null. Declared before looking at any spectrum.
J_MIN = 4
DETREND_DEG = 6


def profile(path, key, field="d_eff"):
    d = json.load(open(path))["results"][key]
    n = d["n_layers"]
    return np.array([d[field][str(i)] for i in range(n)], float), n


def detrend(y, deg=DETREND_DEG):
    x = np.arange(len(y), dtype=float)
    return y - np.polyval(np.polyfit(x, y, deg), x)


def spectral_test(y, label, n_full=None):
    n = len(y)
    r = detrend(y)
    R = np.fft.rfft(r)
    P = np.abs(R) ** 2
    j4 = n // 4                      # index of the period-4 component
    assert abs(n / j4 - 4.0) < 1e-9, f"n={n} does not admit an exact period-4 bin"

    js = np.arange(J_MIN, len(P))
    assert j4 in js, f"period-4 bin {j4} excluded by J_MIN={J_MIN}"
    p = float((P[js] >= P[j4]).sum()) / len(js)
    rank = int((P[js] > P[j4]).sum()) + 1
    ratio = P[j4] / np.median(P[js])
    phase = np.angle(R[j4])
    # layer offset (mod 4) at which the period-4 component peaks
    peak_off = (-phase / (2 * np.pi) * 4) % 4

    print(f"\n--- {label}  (n={n}, period-4 bin j={j4}, null draws={len(js)}, "
          f"floor p={1/len(js):.4f})")
    print(f"    power at period 4      : {P[j4]:.4g}")
    print(f"    median power elsewhere : {np.median(P[js]):.4g}   ratio = {ratio:.2f}x")
    print(f"    rank among {len(js)} freqs   : {rank}   ->  p = {p:.4f}")
    print(f"    component peaks at l = {peak_off:.2f} (mod 4)"
          + (f"   [full attention sits at l = 3 (mod 4)]" if n_full else ""))
    top = js[np.argsort(P[js])[::-1][:4]]
    print(f"    strongest frequencies   : " +
          ", ".join(f"j={j} (period {n/j:.1f}, P={P[j]:.3g})" for j in top))
    return dict(n=n, j4=int(j4), p=p, rank=rank, ratio=float(ratio),
                peak_offset=float(peak_off), floor=1.0 / len(js))


def effect_sizes(y, mask, label):
    """Report |T| in units of the (degenerate) null sd — readable even though p is not."""
    r = detrend(y)
    n = len(mask)
    Ts = np.array([r[np.roll(mask, k)].mean() - r[~np.roll(mask, k)].mean()
                   for k in range(n)])
    sd = Ts.std()
    idx = np.arange(n)
    print(f"\n--- {label}: phase-class effect sizes, |T| / null sd (sd={sd:.3f})")
    for ph in range(4):
        m = (idx % 4) == ph
        T = r[m].mean() - r[~m].mean()
        tag = "  <-- substrate phase" if np.array_equal(m, mask) else ""
        print(f"    l = {ph} (mod 4):  T = {T:>8.4f}   |T|/sd = {abs(T)/sd:>5.2f}{tag}")


def main():
    lt = json.load(open(CFG))["text_config"]["layer_types"]
    full = [i for i, t in enumerate(lt) if t == "full_attention"]
    assert full == list(range(3, 64, 4)) and len(full) == 16
    print("substrate: full_attention at l = 3 (mod 4), 16 of 64 layers  [config-verified]")

    out = {}
    y27, n27 = profile(F27, "qwen35-27b")
    y8, n8 = profile(F8B, "qwen3-8b")

    print("\n" + "=" * 72)
    print("SPECTRAL TEST — is there period-4 power in the detrended depth profile?")
    print("=" * 72)
    out["hybrid_d_eff"] = spectral_test(y27, "TEST: Qwen3.5-27B (HYBRID) d_eff", n_full=16)
    k27, _ = profile(F27, "qwen35-27b", "kurtosis")
    out["hybrid_kurtosis"] = spectral_test(k27, "CONFIRMATION: Qwen3.5-27B kurtosis", n_full=16)
    out["dense_d_eff"] = spectral_test(y8, "NEGATIVE CONTROL: Qwen3-8B (DENSE) d_eff")

    print("\n" + "=" * 72)
    print("EFFECT SIZES from the pre-specified statistic (p was degenerate; T is not)")
    print("=" * 72)
    mask27 = np.isin(np.arange(n27), full)
    effect_sizes(y27, mask27, "Qwen3.5-27B (HYBRID) d_eff")
    effect_sizes(y8, (np.arange(n8) % 4) == 3, "Qwen3-8B (DENSE) d_eff, sham mask")

    json.dump(out, open("/tmp/hybrid_comb_spectral.json", "w"), indent=2)
    print("\nsaved -> /tmp/hybrid_comb_spectral.json")


if __name__ == "__main__":
    main()
