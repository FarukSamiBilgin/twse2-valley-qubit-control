"""Spin-valley level scheme for tWSe2 (gorev 2).

Per-site qubit under the nu=1 Mott assumption:
  |0> = |K, up>  (moire orbital at kappa_+ sector),  |1> = |K', down>
Kramers pair at B=0; splitting under Bz:
  Delta_Z = g_eff mu_B B,
  g_eff = 2 (s + l_at + m_orb/mu_B)   with s=1 (spin), l_at=2 (W d+/-2),
  m_orb = moire-band orbital magnetic moment at the band edge (model-derived).

m_orb(k) = -(e/hbar) Im sum_{n != 0} <u0|dxH|un><un|dyH|u0> / (eps_n - eps_0)
(computed in meV/T via unit bookkeeping below, then expressed in mu_B).

Also: nu=1 Mott sanity numbers (bandwidth, gap, U estimate, well depth)
at the optical-architecture operating angle and at 5.08 deg.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from continuum import MoireContinuum, TWSe2Params, HBAR2_2ME

MU_B_meV_per_T = 5.7883818060e-2   # Bohr magneton, meV/T
E_CHARGE = 1.0                      # bookkeeping only

OUT = os.path.join(os.path.dirname(__file__), "..", "out")


def orbital_moment(model: MoireContinuum, k: np.ndarray, band: int = 0,
                   nsum: int = 40) -> float:
    """m_orb at k for `band` (0 = top valence), in units of mu_B.

    dH/dk is diagonal in the plane-wave/layer basis:
      dH/dkx|layer l, G> = -2*kin*(k+G-kappa_l)_x
    m_z = -(e/hbar) Im sum_n <0|dxH|n><n|dyH|0>/(e_n - e_0)
    Units: [dH/dk] = meV nm, energies meV -> m = e/hbar * meV nm^2
           1 meV nm^2 * e/hbar = e*meV*nm^2/hbar; mu_B = e hbar/(2 m_e)
           => m/mu_B = 2 m_e/hbar^2 * (meV nm^2 value) = (1/HBAR2_2ME)*value
    """
    H = model.hamiltonian(k)
    e, v = np.linalg.eigh(H)
    order = np.argsort(e)[::-1]
    e, v = e[order], v[:, order]
    p = model.p
    kg = k[None, :] + model.Gs
    dx = np.concatenate([-2 * p.kin * (kg[:, 0] - model.kappa_p[0]),
                         -2 * p.kin * (kg[:, 0] - model.kappa_m[0])])
    dy = np.concatenate([-2 * p.kin * (kg[:, 1] - model.kappa_p[1]),
                         -2 * p.kin * (kg[:, 1] - model.kappa_m[1])])
    v0 = v[:, band]
    tot = 0.0
    for n in range(min(nsum, len(e))):
        if n == band:
            continue
        de = e[n] - e[band]
        if abs(de) < 1e-9:
            continue
        a = np.vdot(v0, dx * v[:, n])      # <0|dxH|n>
        b = np.vdot(v[:, n], dy * v0)      # <n|dyH|0>
        tot += np.imag(a * b) / de
    m_meVnm2 = -tot                        # in meV nm^2 (times e/hbar)
    return m_meVnm2 / HBAR2_2ME            # in mu_B


def analyze_angle(theta_deg: float, NG: int = 6) -> dict:
    m = MoireContinuum(TWSe2Params(theta_deg=theta_deg), NG=NG)
    p = m.p
    # BZ grid quantities
    N = 18
    fr = np.arange(N) / N
    grid = (fr[:, None, None] * m.b1[None, None, :]
            + fr[None, :, None] * m.b2[None, None, :]).reshape(-1, 2)
    E = m.bands(grid, nbands=3)
    W1 = float(E[:, 0].max() - E[:, 0].min())
    e12 = float(E[:, 0].min() - E[:, 1].max())
    # moire potential landscape (layer-1 diagonal potential): depth scale
    xs = np.linspace(0, 1, 60, endpoint=False)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    a1 = p.aM * np.array([1.0, 0.0])
    a2 = p.aM * np.array([0.5, np.sqrt(3) / 2])
    rr = X[..., None] * a1 + Y[..., None] * a2
    gj = [m.b1, m.b2 - m.b1, -m.b2]      # g1, g3, g5
    d1 = sum(2 * p.V * np.cos(rr @ g + p.psi) for g in gj)
    d2 = sum(2 * p.V * np.cos(rr @ g - p.psi) for g in gj)
    depth = float(max(d1.max() - d1.min(), d2.max() - d2.min()))
    # orbital moment at band edge kappa_+ (top band); kappa_- is -m by TRS*
    morb = orbital_moment(m, m.kappa_p, band=0)
    # on-site U estimate: screened Coulomb over Wannier spread.
    # Wannier rms spread ~ harmonic length of the local well:
    # expand local well: E(r) ~ 0.5 k r^2 with k from depth/(aM/2)^2 scale —
    # use curvature of d1+d2 at its max (site) instead.
    dtot = d1 + d2
    imax = np.unravel_index(np.argmax(dtot), dtot.shape)
    h = p.aM / 60.0
    # numeric curvature (isotropic average) at the site along x
    def val(di, dj):
        return dtot[(imax[0] + di) % 60, (imax[1] + dj) % 60]
    k_curv = -(val(1, 0) + val(-1, 0) - 2 * val(0, 0)) / h ** 2  # meV/nm^2
    k_curv = abs(k_curv)
    mstar_edge = abs(m.effective_mass(0, "kappa_p"))
    lw = (HBAR2_2ME * 2 / mstar_edge / k_curv) ** 0.25 if k_curv > 0 else np.nan
    eps_r = 5.0                                  # hBN environment
    U = 1439.96 / (eps_r * max(lw, 1e-3))        # e^2/(4 pi eps0) = 1439.96 meV nm
    return {
        "theta_deg": theta_deg,
        "aM_nm": float(p.aM),
        "W1_meV": W1,
        "eps12_meV": e12,
        "moire_depth_meV": depth,
        "mstar_edge_me": float(mstar_edge),
        "m_orb_muB_top_band_at_kappa": float(morb),
        "g_eff_kramers": float(2 * (1 + 2 + morb)),
        "wannier_length_nm": float(lw),
        "U_estimate_meV_epsr5": float(U),
        "U_over_W1": float(U / W1) if W1 > 0 else None,
    }


if __name__ == "__main__":
    res = {"delta_SO_valence_meV_lit": [450, 470],
           "notes": {
               "qubit": "|0>=|K,up>, |1>=|K',down> Kramers pair; "
                        "nu=1 Mott: 1 hole / moire cell",
               "g_eff": "Delta_Z = g_eff mu_B Bz between the Kramers pair; "
                        "g_eff = 2(1 + 2 + m_orb/mu_B): spin + W d(+/-2) "
                        "atomic + moire-orbital (model)",
               "U": "e^2/(4 pi eps0 eps_r l_W), eps_r=5 (hBN); order-of-"
                    "magnitude only",
           },
           "angles": [analyze_angle(t) for t in (1.43, 2.0, 3.0, 5.08)]}
    path = os.path.join(OUT, "level_scheme.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    for a in res["angles"]:
        print(f"theta={a['theta_deg']:5.2f}  aM={a['aM_nm']:.2f}nm  "
              f"W1={a['W1_meV']:7.3f}  e12={a['eps12_meV']:7.3f}  "
              f"m*={a['mstar_edge_me']:.3f}  m_orb={a['m_orb_muB_top_band_at_kappa']:+.2f}muB  "
              f"g_eff={a['g_eff_kramers']:.2f}  lW={a['wannier_length_nm']:.2f}nm  "
              f"U~{a['U_estimate_meV_epsr5']:.0f}meV  U/W1={a['U_over_W1']:.1f}")
    print("wrote", path)
