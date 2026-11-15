"""Focal laser power of the Raman working points, and power-capped optima.

Omega -> power for a Gaussian focus w0 = 1 um and transition dipole d:
    P = 1.924 mW * (Omega_1 / 0.1 meV)^2 * (5 D / d)^2 .
The dipole is not a free input: it is tied to the trion's radiative
lifetime (see dipol_tutarlilik.py).  Radiative lifetimes of 10 ps -- 1 ns
correspond to d ~ 25--350 D; the weakest consistent dipole, d = 25 D, is
used for the capped optima (most demanding case).
Reads ../data/raman_budget.json and ../data/raman_map.json (Gamma = 50 ueV
maps), writes ../data/raman_power.json.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
P_REF_MW_5D = 1.924         # focal power at Omega_1 = 0.1 meV for d = 5 D
D_RANGE = (25.0, 350.0)     # dipole range consistent with tau_rad 1 ns..10 ps
D_CAP = 25.0                # dipole used for the capped optima


def power_mW(O1, d):
    return P_REF_MW_5D * (O1 / 0.1) ** 2 * (5.0 / d) ** 2


bud = json.load(open(os.path.join(DATA, "raman_budget.json")))
mp = json.load(open(os.path.join(DATA, "raman_map.json")))
Om = np.array(mp["Omega1_meV"])

out = {"P_ref_mW_at_0p1meV_5D": P_REF_MW_5D, "d_range_D": D_RANGE,
       "d_cap_D": D_CAP, "optimum_power": [], "capped": []}
for r in bud["sensitivity"]:
    out["optimum_power"].append(dict(
        beta=r["beta"], Gamma_ueV=r["Gamma_ueV"],
        P_focal_mW_range=[power_mW(r["Omega1_meV"], D_RANGE[1]),
                          power_mW(r["Omega1_meV"], D_RANGE[0])]))
for beta, m in mp["beta_maps"].items():
    E = np.array(m["err"])
    T = np.array(m["t_gate_ps"])
    for cap in (0.1, 1.0):
        ok = power_mW(Om, D_CAP) <= cap
        Ec = E[:, ok]
        i, j = np.unravel_index(np.argmin(Ec), Ec.shape)
        row = dict(beta=float(beta), Gamma_ueV=50.0, d_D=D_CAP, P_cap_mW=cap,
                   error=float(Ec[i, j]), t_gate_ps=float(T[:, ok][i, j]),
                   Omega1_meV=float(Om[ok][j]))
        out["capped"].append(row)
        print(f"beta={beta} d={D_CAP:.0f} D, cap {cap} mW: err={row['error']:.2e} "
              f"t={row['t_gate_ps']/1e3:.2f} ns (O1={row['Omega1_meV']:.3f})")
for r in out["optimum_power"]:
    if r["Gamma_ueV"] == 50.0:
        lo, hi = r["P_focal_mW_range"]
        print(f"unconstrained optimum beta={r['beta']}: P_focal = "
              f"{lo:.3g}-{hi:.3g} mW (d = {D_RANGE[1]:.0f}-{D_RANGE[0]:.0f} D)")
json.dump(out, open(os.path.join(DATA, "raman_power.json"), "w"), indent=1)
