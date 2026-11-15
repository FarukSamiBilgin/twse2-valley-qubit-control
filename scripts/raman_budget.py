"""Three-level Lambda-Raman pi-gate budget on the nu=1 site (corrected).

Same model as the original notebook (uc_seviye_raman_butcesi.ipynb), with
one correction: the differential ac-Stark shift is now compensated with
the right sign.  In the rotating frame H = -Delta|e><e| + ..., the lower
states are pushed UP by Omega_i^2/(4 Delta); the two-photon resonance is
restored by shifting |0> down and |1> up by (Omega_1^2-Omega_2^2)/(8 Delta).
(The notebook added the term with the opposite sign, doubling the
detuning; this left beta = 1 untouched but corrupted every beta < 1 point.)

Writes ../data/raman_budget.json (optimal working points) and
../data/raman_map.json (error / gate-time maps for figures).
"""
import json
import os
from multiprocessing import Pool

import numpy as np
import qutip as qt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")
HBAR = 0.6582119569          # meV ps
GPHI = 0.050                 # meV, pure excited-state dephasing
T2Q_PS = 100.0e3             # qubit-subspace T2* (ps)
T_SIM_MAX = 5.0e3            # integrate explicitly up to 5 ns; beyond,
                             # adiabatic-elimination estimate (~2x
                             # conservative, checked against explicit runs)

Deltas = np.geomspace(0.8, 25, 14)      # meV
Omegas = np.geomspace(0.03, 1.2, 14)    # meV (Omega_1)


def raman_error(args):
    Delta, O1, beta, Gamma = args
    k0, k1, ke = [qt.basis(3, i) for i in range(3)]
    O2 = beta * O1
    H = (-Delta * ke * ke.dag()
         + O1 / 2 * (ke * k0.dag() + k0 * ke.dag())
         + O2 / 2 * (ke * k1.dag() + k1 * ke.dag()))
    stark = (O1 ** 2 - O2 ** 2) / (4 * Delta)
    H = H - stark / 2 * (k0 * k0.dag() - k1 * k1.dag())
    c_ops = [np.sqrt(Gamma / 2 / HBAR) * (k0 * ke.dag()),
             np.sqrt(Gamma / 2 / HBAR) * (k1 * ke.dag()),
             np.sqrt(GPHI / HBAR) * (ke * ke.dag()),
             np.sqrt(0.5 / T2Q_PS) * (k0 * k0.dag() - k1 * k1.dag())]
    t_pi = np.pi * HBAR / (O1 * O2 / (2 * Delta))
    if t_pi > T_SIM_MAX:
        pe = (O1 ** 2 + O2 ** 2) / (8 * Delta ** 2)
        err = min(1.0, (Gamma + GPHI) / HBAR * pe * t_pi
                  + 0.5 * t_pi / T2Q_PS)
        return err, t_pi
    ts = np.linspace(0, 1.35 * t_pi, 240)
    res = qt.mesolve(qt.Qobj(H.full()) / HBAR, k0, ts, c_ops,
                     e_ops=[k1 * k1.dag()],
                     options={"nsteps": 400000, "atol": 1e-11, "rtol": 1e-9})
    p1 = np.array(res.expect[0])
    return float(1 - p1.max()), float(ts[int(p1.argmax())])


def sweep(beta, Gamma, pool):
    grid = [(D, O, beta, Gamma) for D in Deltas for O in Omegas]
    out = pool.map(raman_error, grid, chunksize=4)
    ERR = np.array([o[0] for o in out]).reshape(len(Deltas), len(Omegas))
    TG = np.array([o[1] for o in out]).reshape(len(Deltas), len(Omegas))
    i, j = np.unravel_index(np.argmin(ERR), ERR.shape)
    best = dict(beta=beta, Gamma_ueV=Gamma * 1e3, min_error=float(ERR[i, j]),
                t_gate_ps=float(TG[i, j]), Delta_meV=float(Deltas[i]),
                Omega1_meV=float(Omegas[j]))
    return ERR, TG, best


if __name__ == "__main__":
    rows, maps = [], {}
    with Pool(4) as pool:
        for beta in (1.0, 0.5, 0.3, 0.1, 0.03, 0.01):
            for G in (0.020, 0.050, 0.200, 1.000):
                ERR, TG, best = sweep(beta, G, pool)
                rows.append(best)
                print(f"beta={beta:5.2f} Gamma={G*1e3:6.0f} ueV: "
                      f"err={best['min_error']:.2e} "
                      f"t={best['t_gate_ps']/1e3:7.2f} ns "
                      f"(Delta={best['Delta_meV']:.1f}, "
                      f"O1={best['Omega1_meV']:.3f})", flush=True)
                if G == 0.050 and beta in (0.3, 1.0, 0.1):
                    maps[str(beta)] = {"err": ERR.tolist(),
                                       "t_gate_ps": TG.tolist(), "best": best}
    meta = {"model": "3-level Lambda, Lindblad; corrected ac-Stark sign",
            "gamma_phi_ueV": GPHI * 1e3, "T2q_ns": T2Q_PS / 1e3,
            "grid": {"Delta_meV": [0.8, 25.0, 14], "Omega1_meV": [0.03, 1.2, 14],
                     "spacing": "geometric"},
            "explicit_integration_up_to_ns": T_SIM_MAX / 1e3,
            "beyond": "adiabatic-elimination estimate, ~2x above explicit integration on spot checks (conservative)"}
    with open(os.path.join(OUT, "raman_budget.json"), "w") as f:
        json.dump({"meta": meta, "sensitivity": rows}, f, indent=1)
    with open(os.path.join(OUT, "raman_map.json"), "w") as f:
        json.dump({"meta": meta, "Delta_meV": Deltas.tolist(),
                   "Omega1_meV": Omegas.tolist(), "beta_maps": maps}, f,
                  indent=1)
    print("wrote raman_budget.json, raman_map.json")
