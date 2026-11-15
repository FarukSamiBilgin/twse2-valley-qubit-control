"""Dipole / lifetime / linewidth consistency of the Raman budget.

The budget's excited-state decay Gamma returns population to the two
Lambda legs, i.e. it is treated as radiative.  The power mapping, however,
used an unsourced 5 D transition dipole, whose radiative linewidth is four
orders of magnitude below Gamma.  This script

1. computes the radiative rate of a point dipole in a medium of index n,
     Gamma_rad = omega^3 d^2 n / (3 pi eps0 hbar c^3),  hbar omega = 1.70 eV,
   and inverts it: the dipole consistent with a given radiative lifetime;
2. checks with a 4-level Lindblad model (|0>,|1>,|e>, leak |L>) that the
   gate error at fixed total Gamma barely depends on the radiative
   fraction eta = Gamma_rad / Gamma (non-radiative decay leaves the Lambda
   system instead of returning to it);
3. rescales the focal power, P = 1.924 mW (Omega_1/0.1 meV)^2 (5 D/d)^2,
   and re-derives power-capped optima from the Gamma = 50 ueV error maps.

Writes ../data/dipol_tutarlilik.json.
"""
import json
import os

import numpy as np
import qutip as qt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
HBAR_MEVPS = 0.6582119569
hbar, eps0, c, e, Debye = 1.054571817e-34, 8.8541878128e-12, 2.99792458e8, \
    1.602176634e-19, 3.33564e-30
E_X = 1.70                          # eV
omega = E_X * e / hbar

out = {"E_X_eV": E_X}


def gamma_rad_ueV(d_D, n):
    d = d_D * Debye
    rate = omega ** 3 * d ** 2 * n / (3 * np.pi * eps0 * hbar * c ** 3)
    return hbar * rate / e * 1e6


def d_for_tau(tau_ps, n):
    g = HBAR_MEVPS / tau_ps * 1e3          # ueV
    return 5.0 * np.sqrt(g / gamma_rad_ueV(5.0, n))


# ---------------------------------------------------------------- 1
r = {}
for n in (1.0, 2.1):
    g5 = gamma_rad_ueV(5.0, n)
    r[f"n={n}"] = {"Gamma_rad_5D_ueV": g5,
                   "tau_rad_5D_ns": HBAR_MEVPS / (g5 * 1e-3) / 1e3,
                   "d_for_tau_rad_D": {f"{t}ps": d_for_tau(t, n)
                                       for t in (10, 30, 100, 300, 1000)},
                   "d_for_Gamma_rad_eq_Gamma_D": {
                       f"{G}ueV": 5.0 * np.sqrt(G / g5) for G in (20, 50, 200)}}
    print(f"n={n}: 5 D -> Gamma_rad={g5:.3f} ueV, tau_rad="
          f"{r[f'n={n}']['tau_rad_5D_ns']:.1f} ns; d(tau_rad=10ps..1ns) = "
          f"{d_for_tau(10, n):.0f}..{d_for_tau(1000, n):.0f} D; "
          f"d(Gamma_rad=50ueV) = {5*np.sqrt(50/g5):.0f} D")
out["radiative"] = r

# ---------------------------------------------------------------- 2
GPHI, T2Q = 0.050, 100.0e3


def err4(Delta, O1, beta, Gtot, eta):
    b = [qt.basis(4, i) for i in range(4)]
    k0, k1, ke, kL = b
    O2 = beta * O1
    H = (-Delta * ke * ke.dag() + O1 / 2 * (ke * k0.dag() + k0 * ke.dag())
         + O2 / 2 * (ke * k1.dag() + k1 * ke.dag()))
    H = H - (O1 ** 2 - O2 ** 2) / (4 * Delta) / 2 * (k0 * k0.dag() - k1 * k1.dag())
    Gr, Gnr = eta * Gtot, (1 - eta) * Gtot
    c_ops = [np.sqrt(Gr / 2 / HBAR_MEVPS) * k0 * ke.dag(),
             np.sqrt(Gr / 2 / HBAR_MEVPS) * k1 * ke.dag(),
             np.sqrt(GPHI / HBAR_MEVPS) * ke * ke.dag(),
             np.sqrt(0.5 / T2Q) * (k0 * k0.dag() - k1 * k1.dag())]
    if Gnr > 0:
        c_ops.append(np.sqrt(Gnr / HBAR_MEVPS) * kL * ke.dag())
    t_pi = np.pi * HBAR_MEVPS / (O1 * O2 / (2 * Delta))
    ts = np.linspace(0, 1.35 * t_pi, 240)
    res = qt.mesolve(qt.Qobj(H.full()) / HBAR_MEVPS, k0, ts, c_ops,
                     e_ops=[k1 * k1.dag()],
                     options={"nsteps": 400000, "atol": 1e-11, "rtol": 1e-9})
    p = np.array(res.expect[0])
    return float(1 - p.max())


chk = []
for beta, D, O1 in ((1.0, 25.0, 0.680), (0.3, 25.0, 1.200), (0.1, 25.0, 1.200)):
    for G in (0.050, 0.200):
        row = {"beta": beta, "Gamma_ueV": G * 1e3, "Delta": D, "Omega1": O1}
        for eta in (1.0, 0.3, 0.1, 0.01):
            row[f"eta={eta}"] = err4(D, O1, beta, G, eta)
        chk.append(row)
        print(f"beta={beta} Gamma={G*1e3:.0f}: " + ", ".join(
            f"eta={eta}: {row[f'eta={eta}']:.2e}" for eta in (1.0, 0.3, 0.1, 0.01)))
out["eta_insensitivity"] = chk

# ---------------------------------------------------------------- 3
mp = json.load(open(os.path.join(DATA, "raman_map.json")))
Om = np.array(mp["Omega1_meV"])
dlist = [5.0, 25.0, 35.0, 75.0, 110.0, 240.0, 350.0]
caps = []
for beta, m in mp["beta_maps"].items():
    E = np.array(m["err"]); T = np.array(m["t_gate_ps"])
    i, j = np.unravel_index(np.argmin(E), E.shape)
    for d in dlist:
        P = 1.924 * (Om / 0.1) ** 2 * (5.0 / d) ** 2      # mW
        row = {"beta": float(beta), "d_D": d,
               "P_at_unconstrained_opt_mW": float(P[j])}
        for cap in (0.1, 1.0, 20.0):
            ok = P <= cap
            if not ok.any():
                row[f"cap_{cap}mW"] = None
                continue
            Ec = E[:, ok]
            a, b2 = np.unravel_index(np.argmin(Ec), Ec.shape)
            row[f"cap_{cap}mW"] = {"error": float(Ec[a, b2]),
                                   "t_gate_ns": float(T[:, ok][a, b2] / 1e3)}
        caps.append(row)
        print(f"beta={beta} d={d:5.0f} D: P(opt)={P[j]:.3g} mW; " + "; ".join(
            f"cap {cap} mW: " + ("--" if row[f'cap_{cap}mW'] is None else
                                 f"{row[f'cap_{cap}mW']['error']:.1e}@"
                                 f"{row[f'cap_{cap}mW']['t_gate_ns']:.2f}ns")
            for cap in (0.1, 1.0, 20.0)))
out["power_vs_dipole"] = caps
json.dump(out, open(os.path.join(DATA, "dipol_tutarlilik.json"), "w"), indent=1)
print("wrote dipol_tutarlilik.json")
