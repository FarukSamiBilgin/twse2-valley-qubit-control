"""Twist-disorder Monte Carlo -> optical channel map (gorev 5 / ortak modul).

Model: operating point theta0 = 2.0 deg (optical nu=1 Mott architecture).
Local twist theta(r) = theta0 + delta(r), delta = correlated Gaussian field
(correlation length xi ~ 150 nm, sigma_theta in {0.02, 0.036} deg,
experimental range from twist-angle-disorder literature).

Site optical transition shift: dE = S_tot * delta_theta.
 - Hole-side slope S_h = d(E_band-max)/d(theta) computed from the continuum
   model (exact, this file).  Electron-side moire + binding-energy variation
   are NOT in our hole-only model; total slope is bracketed as
   S_tot = f * S_h with f in {1.0, 1.5, 2.0} (assumption, flagged honest).

Outputs: out/tWSe2_kanal_haritasi.png (per-site channel map + histogram),
         out/mc_disorder.json (channel statistics + electrical-calibration
         comparison numbers).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from continuum import MoireContinuum, TWSe2Params

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
rng = np.random.default_rng(42)

THETA0 = 2.0
XI_NM = 150.0          # disorder correlation length
CHANNEL_UEV = 100.0    # frequency-channel bin width (2 x Gamma_h ~ 100 ueV)
NSITE = 48             # lattice of NSITE x NSITE moire sites


def band_max(theta):
    m = MoireContinuum(TWSe2Params(theta_deg=theta), NG=6)
    N = 15
    fr = np.arange(N) / N
    grid = (fr[:, None, None] * m.b1[None, None, :]
            + fr[None, :, None] * m.b2[None, None, :]).reshape(-1, 2)
    return float(m.bands(grid, nbands=1).max()), m.p.aM


# hole-side slope from the model (meV/deg)
dth = 0.05
Ep, aM = band_max(THETA0 + dth)
Em, _ = band_max(THETA0 - dth)
E0, _ = band_max(THETA0)
S_h = (Ep - Em) / (2 * dth)
print(f"theta0={THETA0} deg, aM={aM:.2f} nm: hole slope S_h = {S_h:.1f} meV/deg")


def correlated_field(n, dx_nm, xi_nm, sigma):
    """Gaussian random field with Gaussian correlation, std sigma."""
    k = np.fft.fftfreq(n, d=dx_nm) * 2 * np.pi
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    spec = np.exp(-K2 * xi_nm ** 2 / 4.0)
    noise = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    f = np.real(np.fft.ifft2(np.fft.fft2(noise) * np.sqrt(spec)))
    return sigma * f / f.std()


results = {"theta0_deg": THETA0, "aM_nm": aM, "S_hole_meV_per_deg": S_h,
           "xi_nm": XI_NM, "channel_bin_ueV": CHANNEL_UEV, "cases": []}

fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5),
                         gridspec_kw={"height_ratios": [2.2, 1]})
F_CENTRAL = 1.5
for col, sigma_th in enumerate([0.02, 0.036]):
    theta_field = correlated_field(NSITE, aM, XI_NM, sigma_th)
    for f_i, f in enumerate([1.0, 1.5, 2.0]):
        dE = f * S_h * theta_field * 1e3          # ueV
        sigE = float(dE.std())
        ch = np.round(dE / CHANNEL_UEV).astype(int)
        nch95 = int(np.ceil(4 * sigE / CHANNEL_UEV))  # +-2sigma coverage
        frac_center = float((np.abs(dE) < CHANNEL_UEV / 2).mean())
        results["cases"].append({
            "sigma_theta_deg": sigma_th, "f_electron": f,
            "sigma_E_ueV": sigE,
            "n_channels_95pct": max(nch95, 1),
            "frac_addressable_single_freq": frac_center,
        })
        if f == F_CENTRAL:
            ax = axes[0, col]
            im = ax.imshow(dE / 1e3, cmap="RdBu_r", origin="lower",
                           extent=[0, NSITE * aM, 0, NSITE * aM])
            ax.set_title(f"$\\sigma_\\theta={sigma_th}°$  "
                         f"($\\sigma_E$={sigE:.0f} µeV, f={f})")
            ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
            plt.colorbar(im, ax=ax, label="site detuning (meV)")
            axh = axes[1, col]
            axh.hist(dE.ravel() / 1e3, bins=41, color="C0", alpha=0.8)
            for n in range(-6, 7):
                axh.axvline(n * CHANNEL_UEV / 1e3, color="k", lw=0.4, alpha=0.4)
            axh.set_xlabel("site detuning (meV)")
            axh.set_ylabel("site sayisi")
            axh.set_title(f"kanal genisligi {CHANNEL_UEV:.0f} µeV; "
                          f"±2σ → {max(int(np.ceil(4*sigE/CHANNEL_UEV)),1)} kanal")

# --- panel 3: electrical vs optical calibration burden vs N ---
ax = axes[0, 2]
N = np.geomspace(1, 1e4, 60)
# electrical (Wurzburg tarzi): her site ~3 DC hat (plunger+2 tuner) + soguk
# kablolama; kalibrasyon olcumu ~ her site ayri
ax.plot(N, 3 * N, label="elektriksel: DC hat sayısı ≈ 3N", c="C3")
ax.plot(N, N, ls="--", c="C3", alpha=0.6, label="elektriksel: kalibrasyon ölçümü ∝ N")
# optical: lazer sayisi sabit (2 surus + 1 okuma); kanal sayisi sigma'dan
nch = [max(int(np.ceil(4 * F_CENTRAL * S_h * 1e3 * s / CHANNEL_UEV)), 1)
       for s in (0.02, 0.036)]
ax.axhline(3, c="C0", label="optik: lazer sayısı = 3 (N'den bağımsız)")
ax.fill_between(N, nch[0], nch[1], color="C0", alpha=0.25,
                label=f"optik: frekans kanalı {nch[0]}–{nch[1]}")
ax.plot(N, N, ls=":", c="C0", alpha=0.6, label="optik: spektral ön-tarama ∝ N (tek seferlik)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("qubit sayısı N"); ax.set_ylabel("kaynak sayısı")
ax.set_title("adresleme kaynak ölçeklenmesi")
ax.legend(fontsize=7, loc="upper left")
axes[1, 2].axis("off")
axes[1, 2].text(0, 0.9,
    ("Not: dE = f·S_h·δθ; S_h continuum modelden,\n"
     f"S_h = {S_h:.0f} meV/°; f = elektron+bağlanma katkısı\n"
     "(1–2 aralığı VARSAYIM — dürüst bilinmeyen).\n"
     f"ξ = {XI_NM:.0f} nm, kanal = {CHANNEL_UEV:.0f} µeV.\n"
     "Elektriksel kıyas: aynı σ_E her sitede kapı-voltaj\n"
     "trimlemesi gerektirir → hat+kalibrasyon ∝ N;\n"
     "optikte düzensizlik frekans kanallarına katlanır."),
    fontsize=8, va="top")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "tWSe2_kanal_haritasi.png"), dpi=160)

# electrical-side analytic comparison at 5.08 deg (Wurzburg dot):
# X = Dkappa^2 l^2/4 ~ 6.3; dlnDelta1S/dtheta = -X/theta
X = 6.3
for s in (0.02, 0.036):
    results.setdefault("wurzburg_dot_sensitivity", []).append({
        "sigma_theta_deg": s,
        "dDelta1S_over_Delta1S_pct": round(100 * X * s / 5.08, 2),
        "sigma_f_qubit_ueV_at_400ueV": round(400 * X * s / 5.08, 1),
    })
with open(os.path.join(OUT, "mc_disorder.json"), "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results["cases"], indent=1))
print("wrote kanal haritasi + mc_disorder.json")
