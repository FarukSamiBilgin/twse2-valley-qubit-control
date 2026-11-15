"""Registry-DFT -> continuum (V, psi, w) fit.

Inputs (this directory): {mm,mx,xm}.bands.out  (QE, verbosity=high, K point)
                         {mm,mx,xm}.pot.dat    (pp.x 1D planar V_bare+V_H, Ry)

Model (first harmonic, vacuum-referenced K-point top valence doublet):
  MM:  E_pm = C + 6 V cos(psi) +/- 3 w
  MX/XM: E_1,2 = C + 6 V cos(psi -/+ 2pi/3)   (tunneling ~ 0)
Closed form:
  w          = split_MM / 6
  V cos(psi) = (mean_MM - mean_MX) / 9
  V |sin psi|= |diff_MX| / (6 sqrt(3))
psi branch fixed to (90, 180) deg (DF: 128).  All meV.
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RY = 13.605693122994
NELEC = 52          # noncolin: one state per electron


def k_eigenvalues(path):
    """Top-2 valence eigenvalues (eV) at the (single) k-point of a bands run."""
    txt = open(path, errors="ignore").read()
    blocks = re.findall(r"bands \(ev\):\s*\n\n((?:\s+[-\d.]+)+)", txt)
    if not blocks:
        raise RuntimeError(f"no bands block in {path}")
    ev = np.array([float(x) for x in blocks[-1].split()])
    ev.sort()
    return ev[NELEC - 2], ev[NELEC - 1]          # (lower, upper) of top doublet


def vacuum_level(path, c_ang=31.0):
    """Average of the two vacuum plateaus of the 1D potential (eV).
    pp.x iflag=1 output: col0 = position (alat units), col1 = V (Ry)."""
    d = np.loadtxt(path)
    x = d[:, 0] / d[:, 0].max()                  # 0..1 along c
    v = d[:, 1] * RY
    # windows symmetric about the slab centre (0.5c) so the average cancels
    # the linear tilt left by the *absent* dipole correction (polar slabs:
    # R-stack ferroelectric step ~60 mV -> residual alignment error ~2 meV)
    lo = v[(x > 0.03) & (x < 0.26)]              # below slab (slab ~0.33-0.67)
    hi = v[(x > 0.74) & (x < 0.97)]              # mirror window above slab
    return 0.5 * (lo.mean() + hi.mean()), lo.mean(), hi.mean()


def main():
    data = {}
    for st in ("mm", "mx", "xm"):
        e1, e2 = k_eigenvalues(os.path.join(HERE, f"{st}.bands.out"))
        vac, vlo, vhi = vacuum_level(os.path.join(HERE, f"{st}.pot.dat"))
        data[st] = {"E_top2_eV": [e1, e2], "vac_eV": vac,
                    "vac_lo_hi_eV": [vlo, vhi],
                    "E_ref_meV": [(e1 - vac) * 1e3, (e2 - vac) * 1e3]}
        print(f"{st}: E_top2 = {e1:.4f}, {e2:.4f} eV; vac = {vac:.4f} eV "
              f"(lo {vlo:.4f} / hi {vhi:.4f}) -> ref {data[st]['E_ref_meV']}")

    mm = data["mm"]["E_ref_meV"]; mx = data["mx"]["E_ref_meV"]
    xm = data["xm"]["E_ref_meV"]
    split_mm = abs(mm[1] - mm[0]); mean_mm = np.mean(mm)
    diff_mx = abs(mx[1] - mx[0]); mean_mx = np.mean(mx)

    w = split_mm / 6.0
    Vc = (mean_mm - mean_mx) / 9.0               # V cos(psi)
    Vs = diff_mx / (6.0 * np.sqrt(3.0))          # V |sin(psi)|
    V = float(np.hypot(Vc, Vs))
    psi = float(np.degrees(np.arctan2(Vs, Vc)))  # branch (90,180) if Vc<0

    ref = {"V": 9.0, "psi": 128.0, "w": 18.0}
    fit = {"V": V, "psi": psi, "w": w}
    dev = {k: 100.0 * (fit[k] - ref[k]) / ref[k] for k in ref}
    consist = {"xm_vs_mx_meV": [abs(sorted(xm)[i] - sorted(mx)[i]) for i in range(2)]}

    out = {"data": data, "fit_meV_deg": fit, "devakul_fu": ref,
           "deviation_pct": dev, "xm_mx_consistency": consist,
           "method": "PBE+SOC, rel-PAW, ecutwfc 45 Ry, k 6x6, fixed DF "
                     "geometry (7.15/6.60 A), NO dipole correction "
                     "(tefield+dipfield diverges SCF in noncolin QE runs); "
                     "vacuum = mean of two symmetric plateaus (tilt cancels "
                     "to 1st order; residual ~2 meV for polar stackings)"}
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"fit": fit, "dev_pct": dev, "consistency": consist},
                     indent=2))


if __name__ == "__main__":
    main()
