"""Bulk-band validation of the Devakul-Fu continuum implementation.

Checks (all against Nat. Commun. 12, 6730):
  1. gamma-point 6-state analytic formula (SM Note 4) — convention check.
  2. theta = 5.08 deg: top valence bands along the k-path; band maxima at
     kappa_+/-; fitted band-edge hole mass ~ 0.50 m_e (Wurzburg Fig. 1c).
  3. magic-angle scan: bandwidth minimum at theta_m ~ 1.43 deg with
     epsilon_12 ~ 3.7 meV >> W1 ~ 0.1 meV.

Writes out/bulk_validation.json, out/bands_5p08.png, out/magic_angle.png.
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
os.makedirs(OUT, exist_ok=True)
res = {}

# ----------------------------------------------------------------- check 1
model = MoireContinuum(TWSe2Params(theta_deg=5.08), NG=6)
ana, num = model.gamma_sixstate_check()
# analytic uses a 6-state truncation; agreement should be ~few meV level
res["gamma_sixstate"] = {
    "analytic_meV": ana.tolist(),
    "numeric_top6_meV": num.tolist(),
    "max_abs_diff_meV": float(np.max(np.abs(ana - num))),
}
print("check1 gamma 6-state: analytic", np.round(ana, 2))
print("                     numeric ", np.round(num, 2))

# ----------------------------------------------------------------- check 2
kpts, dist, ticks, labels = model.kpath(60)
E = model.bands(kpts, nbands=6)
Etop = E[:, 0].max()

fig, ax = plt.subplots(figsize=(5, 4))
for b in range(E.shape[1]):
    ax.plot(dist, E[:, b] - Etop, "b-", lw=1)
ax.set_xticks(ticks, labels)
ax.set_ylabel("E - E_max (meV)")
ax.set_title(r"tWSe$_2$ continuum, $\theta=5.08°$ (Devakul-Fu params)")
ax.set_ylim(-120, 5)
for t in ticks:
    ax.axvline(t, color="k", lw=0.4, alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bands_5p08.png"), dpi=160)
plt.close(fig)

# band maximum location
imax = np.argmax(E[:, 0])
kmax = kpts[imax]
d_kp = np.linalg.norm(kmax - model.kappa_p)
d_km = np.linalg.norm(kmax - model.kappa_m)
mstar_fit = model.effective_mass(band=0, at="kappa_p")
res["theta_5p08"] = {
    "band_max_at_kappa": bool(min(d_kp, d_km) < 1e-6 * model.p.gm + 1e-9),
    "dist_max_to_kappa_over_g": float(min(d_kp, d_km) / model.p.gm),
    "fitted_hole_mass_me": float(mstar_fit),
    "wurzburg_reference_me": 0.50,
    "mass_dev_percent": float(100 * (mstar_fit - 0.50) / 0.50),
    "W1_bandwidth_meV": float(E[:, 0].max() - E[:, 0].min()),
}
print(f"check2 theta=5.08: band max dist to kappa = {min(d_kp,d_km)/model.p.gm:.4f} g")
print(f"       fitted hole mass = {mstar_fit:.3f} m_e (ref 0.50)")

# ----------------------------------------------------------------- check 3
thetas = np.arange(1.0, 2.3, 0.02)
W1s, e12s = [], []
for th in thetas:
    m = MoireContinuum(TWSe2Params(theta_deg=th), NG=6)
    ks, _, _, _ = m.kpath(24)
    # denser sampling: uniform BZ grid for bandwidth/gap
    N = 15
    fr = np.arange(N) / N
    grid = (fr[:, None, None] * m.b1[None, None, :]
            + fr[None, :, None] * m.b2[None, None, :]).reshape(-1, 2)
    Eg = m.bands(grid, nbands=2)
    W1s.append(Eg[:, 0].max() - Eg[:, 0].min())
    e12s.append(Eg[:, 0].min() - Eg[:, 1].max())
W1s, e12s = np.array(W1s), np.array(e12s)
i_m = int(np.argmin(W1s))
theta_m = float(thetas[i_m])

fig, ax = plt.subplots(figsize=(5, 3.6))
ax.plot(thetas, W1s, label="$W_1$ (bandwidth)")
ax.plot(thetas, e12s, label=r"$\varepsilon_{12}$ (indirect gap)")
ax.axvline(theta_m, color="r", ls="--", lw=0.8,
           label=f"$\\theta_m={theta_m:.2f}°$")
ax.set_xlabel(r"$\theta$ (deg)")
ax.set_ylabel("meV")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "magic_angle.png"), dpi=160)
plt.close(fig)

res["magic_angle"] = {
    "theta_m_deg": theta_m,
    "reference_deg": 1.43,
    "W1_at_magic_meV": float(W1s[i_m]),
    "eps12_at_magic_meV": float(e12s[i_m]),
    "reference_W1_meV": 0.1,
    "reference_eps12_meV": 3.7,
}
print(f"check3 magic angle = {theta_m:.2f} deg (ref 1.43), "
      f"W1={W1s[i_m]:.3f} meV (ref ~0.1), eps12={e12s[i_m]:.2f} meV (ref 3.7)")

with open(os.path.join(OUT, "bulk_validation.json"), "w") as f:
    json.dump(res, f, indent=2)
print("wrote", os.path.join(OUT, "bulk_validation.json"))
