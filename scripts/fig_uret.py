"""Kol D preprint figure generator.

Reads data/*.json (copies + precomputed scans, see README.md) and writes
figs/fig{1..4}_*.pdf + .png.  Pure matplotlib; no solver dependency.

Usage:  python fig_uret.py
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
})


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def save(fig, stem):
    fig.savefig(os.path.join(FIGS, stem + ".pdf"))
    fig.savefig(os.path.join(FIGS, stem + ".png"), dpi=300)
    plt.close(fig)
    print("wrote", stem, "(.pdf/.png)")


MUB = 5.7883818060e-2  # meV/T

# ===================================================================== fig 1
scan = load("magic_angle_scan.json")
ladder = load("dot_ladder.json")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

th = np.array(scan["theta_deg"])
ax1.semilogy(th, scan["W1_meV"], "-", c="C0", label=r"$W_1$ (bandwidth)")
ax1.semilogy(th, scan["eps12_meV"], "-", c="C1",
             label=r"$\varepsilon_{12}$ (gap)")
ax1.axvline(scan["theta_m_deg"], color="r", ls="--", lw=0.9)
ax1.annotate(rf"$\theta_m={scan['theta_m_deg']:.2f}°$"
             "\n(ref. 1.43°)",
             (scan["theta_m_deg"], 2.0e-2), xytext=(1.55, 2.5e-2),
             arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
ax1.set_xlabel(r"twist angle $\theta$ (deg)")
ax1.set_ylabel("energy (meV)")
ax1.set_title("(a) anchor 1: magic-angle scan")
ax1.legend(loc="lower right")

Ws = [5, 7, 10]
ours = [ladder[f"W{w}"]["Delta1S_ueV"] for w in Ws]
# reference magnitude bands read from Saleem et al. (Fig. 3e / text):
ref_lo = [400, 10, 1]
ref_hi = [440, 100, 10]
for w, lo, hi in zip(Ws, ref_lo, ref_hi):
    ax2.fill_between([w - 0.35, w + 0.35], lo, hi, color="C1", alpha=0.3,
                     label="ref. range" if w == 5 else None)
ax2.semilogy(Ws, ours, "o-", c="C0", label="this work")
for w, v in zip(Ws, ours):
    ax2.annotate(f"{v:.3g}", (w, v), textcoords="offset points",
                 xytext=(6, 4), fontsize=8)
ax2.set_xticks(Ws)
ax2.set_xlabel(r"dot radius $W = R/a_M$")
ax2.set_ylabel(r"$\Delta_{1S}$ ($\mu$eV)")
ax2.set_title(r"(b) anchor 2: valley-splitting ladder, $\theta=5.08°$")
ax2.legend()

fig.tight_layout()
save(fig, "fig1_validation")

# ===================================================================== fig 2
lev = load("level_scheme.json")["angles"]
thL = [a["theta_deg"] for a in lev]
UoW = [a["U_over_W1"] for a in lev]
geff = [a["g_eff_kramers"] for a in lev]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.6), sharex=True)

ax1.semilogy(thL, UoW, "o-", c="C0", label="Wannier estimate (this work)")
ax1.semilogy([1.43], [70.0 / 0.1], "s", mfc="none", mec="C0", ms=7,
             label=r"published $U\approx70$ meV")
ax1.axhspan(1e-1, 10, color="0.85")
ax1.axvspan(3.3, 5.4, color="C1", alpha=0.10)
ax1.text(4.3, 2.2e3, "top band\nnot isolated\n($\\varepsilon_{12}<0$)",
         fontsize=7, ha="center", color="C1")
from matplotlib.patches import Patch  # noqa: E402
nomott = Patch(fc="0.85", label=r"$U/W_1<10$: no Mott (itinerant)")
ax1.annotate(r"optical: $\nu=1$ Mott sites", (1.43, UoW[0]),
             xytext=(2.1, 7e2), fontsize=8,
             arrowprops=dict(arrowstyle="->", lw=0.8))
ax1.annotate("electrical:\ngate-defined dot", (5.08, UoW[3]),
             xytext=(3.6, 40), fontsize=8, ha="center",
             arrowprops=dict(arrowstyle="->", lw=0.8))
ax1.set_ylabel(r"$U/W_1$ (top miniband)")
ax1.set_ylim(0.8, 2e4)
ax1.set_xlim(1.1, 5.4)
h, lab = ax1.get_legend_handles_labels()
ax1.legend(h + [nomott], lab + [nomott.get_label()], fontsize=6.5,
           loc="lower left")
ax1.set_title("(a) one material, two operating points")

ax2.plot(thL, geff, "s-", c="C2")
ax2.set_ylabel(r"$g_{\rm eff}$ (Kramers, model)")
ax2.set_xlabel(r"twist angle $\theta$ (deg)")
sec = ax2.secondary_yaxis(
    "right", functions=(lambda g: g * MUB, lambda e: e / MUB))
sec.set_ylabel(r"$\delta_q$ at $B=1$ T (meV)")
ax2.set_title("(b) small-angle site-qubit splitting")

fig.tight_layout()
save(fig, "fig2_two_regimes")

# ===================================================================== fig 3
rm = load("raman_map.json")
sens = load("raman_budget.json")["sensitivity"]
Deltas = np.array(rm["Delta_meV"])
Omegas = np.array(rm["Omega1_meV"])

fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2),
                         gridspec_kw={"width_ratios": [1, 1, 1.15],
                                      "wspace": 0.45})
for pl, ax, beta in zip("ab", axes[:2], ("0.3", "1.0")):
    m = rm["beta_maps"][beta]
    ERR = np.array(m["err"])
    TG = np.array(m["t_gate_ps"])
    pc = ax.pcolormesh(Omegas, Deltas, np.log10(np.clip(ERR, 1e-6, 1)),
                       cmap="viridis", shading="auto", vmin=-3, vmax=0)
    cs = ax.contour(Omegas, Deltas, TG / 1e3, levels=[0.05, 0.2, 1, 5, 20],
                    colors="w", linewidths=0.7)
    ax.clabel(cs, fmt="%g ns", fontsize=7)
    b = m["best"]
    ax.plot(b["Omega1_meV"], b["Delta_meV"], "r*", ms=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Omega_1$ (meV)")
    ax.set_title(rf"({pl}) $\beta={float(beta):g}$", fontsize=9)
    ax.text(0.03, 0.03,
            rf"min. err $={b['min_error']:.1e}$" "\n"
            rf"$t_\pi={b['t_gate_ps']/1e3:.2g}$ ns",
            transform=ax.transAxes, fontsize=7.5, color="w", va="bottom")
axes[0].set_ylabel(r"$\Delta$ (meV)")
cb = fig.colorbar(pc, ax=axes[:2], shrink=0.9, pad=0.025, aspect=28)
cb.set_label(r"$\log_{10}$(gate error)", fontsize=8)

ax = axes[2]
DMIN = 2.0   # optima with Delta below this are optical-pumping corners
betas = sorted({r["beta"] for r in sens if r["beta"] >= 0.03})
for G, c in zip((20.0, 50.0, 200.0), ("C0", "C1", "C3")):
    rows = sorted([r for r in sens if r["Gamma_ueV"] == G
                   and r["beta"] >= 0.03], key=lambda r: r["beta"])
    coh = [r for r in rows if r["Delta_meV"] >= DMIN]
    ax.loglog([r["beta"] for r in coh], [r["min_error"] for r in coh],
              "o-", c=c, label=rf"$\Gamma={G:.0f}\,\mu$eV")
    pump = [r for r in rows if r["Delta_meV"] < DMIN]
    if pump:
        ax.loglog([r["beta"] for r in pump], [r["min_error"] for r in pump],
                  "o", mfc="w", mec=c)
pw = load("raman_power.json")["capped"]
cap = sorted([r for r in pw if r["P_cap_mW"] == 1.0],
             key=lambda r: r["beta"])
ax.loglog([r["beta"] for r in cap], [r["error"] for r in cap], "s--",
          c="C1", mfc="w", label=r"$\Gamma=50$, $P\leq1$ mW, $d=25$ D")
for r in cap:
    ax.annotate(f"{r['t_gate_ps']/1e3:.1f} ns", (r["beta"], r["error"]),
                textcoords="offset points", xytext=(4, 3), fontsize=6.5,
                color="C1")
ax.set_xlabel(r"branching ratio $\beta=\Omega_2/\Omega_1$")
ax.set_ylabel(r"optimal $\pi$-gate error")
ax.set_xticks(betas, [f"{b:g}" for b in betas])
ax.legend(loc="upper right", fontsize=6.3)
ax.set_title(r"(c) the $\beta$ penalty ($T_2^*=100$ ns)", fontsize=8.5)

save(fig, "fig3_raman_budget")

# ===================================================================== fig 4
mc = load("mc_field.json")
mcs = load("mc_disorder.json")
dE = np.array(mc["dE_ueV"])
aM, n = mc["aM_nm"], mc["NSITE"]
CH = mc["channel_bin_ueV"]
sigE = mc["sigma_E_ueV"]
nch = int(np.ceil(4 * sigE / CH))

fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4))
axes = axes.ravel()

ax = axes[0]
im = ax.imshow(dE / 1e3, cmap="RdBu_r", origin="lower",
               extent=[0, n * aM, 0, n * aM])
ax.set_xlabel("x (nm)")
ax.set_ylabel("y (nm)")
ax.set_title(rf"(a) site detunings, spec target"
             rf" $\sigma_\theta={mc['sigma_theta_deg']}°$,"
             rf" $f={mc['f_electron']:g}$", fontsize=8.5)
fig.colorbar(im, ax=ax, label="detuning (meV)", shrink=0.85)

ax = axes[1]
ax.hist(dE.ravel() / 1e3, bins=41, color="C0", alpha=0.85)
for k in range(-12, 13):
    ax.axvline(k * CH / 1e3, color="k", lw=0.35, alpha=0.35)
ax.set_xlabel("site detuning (meV)")
ax.set_ylabel("site count")
ax.set_title(rf"(b) $\sigma_E={sigE:.0f}\,\mu$eV $\to$ {nch} channels"
             rf" ({CH:.0f} $\mu$eV bins, $\pm2\sigma$)", fontsize=8.5)

# ---- (c) both calibration currencies vs sigma_theta (axis framing)
ax = axes[2]
S_h = abs(mcs["S_hole_meV_per_deg"])
sig = np.geomspace(0.015, 1.5, 120)
ch_f1 = np.maximum(np.ceil(4 * 1.0 * S_h * 1e3 * sig / CH), 1)
ch_f2 = np.maximum(np.ceil(4 * 2.0 * S_h * 1e3 * sig / CH), 1)
ax.fill_between(sig, ch_f1, ch_f2, color="C0", alpha=0.3,
                label=r"optical channels ($f=1$–$2$)")
X_W = 6.3  # dlnDelta_1S/dlntheta at W=5 (electrical dot)
spread = 100 * X_W * sig / 5.08
ok = spread < 60
ax.plot(sig[ok], spread[ok], c="C3",
        label=r"electrical $\Delta_{1S}$ spread (%)")
ax.plot(sig[~ok], spread[~ok], c="C3", ls=":",
        label="(linear scaling breaks)")
for x, lab in ((0.036, "spec\ntarget"), (0.3, "metrology\nprecision"),
               (1.0, "measured\nvariation")):
    ax.axvline(x, color="k", lw=0.6, ls="--", alpha=0.6)
    ax.text(x * 1.07, 1.35, lab, fontsize=6.5, va="bottom")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"twist disorder $\sigma_\theta$ (deg)")
ax.set_ylabel("channels  /  spread (%)")
ax.set_title(r"(c) both currencies vs. $\sigma_\theta$", fontsize=8.5)
ax.legend(fontsize=6.5, loc="upper left")

ax = axes[3]
N = np.geomspace(1, 1e4, 60)
ax.plot(N, 3 * N, c="C3", label=r"electrical: dc lines $\approx 3N$")
ax.plot(N, N, ls="--", c="C3", alpha=0.7,
        label=r"electrical: calibration $\propto N$")
ax.axhline(3, c="C0", label="optical: lasers = 3 ($N$-independent)")
ch_lo = min(c["n_channels_95pct"] for c in mcs["cases"]
            if c["sigma_theta_deg"] == 0.036)
ch_hi = max(c["n_channels_95pct"] for c in mcs["cases"]
            if c["sigma_theta_deg"] == 0.036)
ax.fill_between(N, ch_lo, ch_hi, color="C0", alpha=0.25,
                label=f"optical: {ch_lo}–{ch_hi} freq. channels")
ax.plot(N, N, ls=":", c="C0", alpha=0.7,
        label=r"optical: one-time parallel pre-scan $\propto N$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("qubit number $N$")
ax.set_ylabel("resource count")
ax.set_title("(d) addressing-resource scaling (at spec target)",
             fontsize=8.5)
ax.legend(fontsize=6.5, loc="upper left")

fig.tight_layout()
save(fig, "fig4_disorder_scaling")

print("all figures written to", FIGS)
