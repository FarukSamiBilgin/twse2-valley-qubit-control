"""Gate-defined Gaussian quantum dot on the tWSe2 moire lattice.

Reproduces the setup of Saleem, Potasz, Hankiewicz, arXiv:2608.05229:
confined hole states of

    H = H_K + V_QD(r) + (D/2) sigma_z ,   V_QD = V0 exp(-r^2/R^2),  R = W aM

expanded in the bulk moire-band Bloch basis on an N1 x N2 k-grid with NB
retained valence bands.  Matrix elements (their Eq. S4):

  <Phi_pk| V_QD |Phi_p'k'> = (1/A) sum_{alpha,G,G'} A*_{p,k+G} A_{p',k'+G'}
                             * Vt(k'-k+G'-G)
  Vt(Q) = pi R^2 V0 exp(-Q^2 R^2 / 4)

Because Vt decays on the scale 2/R << g, the k-space matrix is banded:
only folded momentum transfers |dk+dG| <= Qcut survive.  Confined dot
states are the LARGEST eigenvalues (V0 > 0 raises the local valence-band
maximum; holes bind to it).

Units: meV, nm.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from continuum import MoireContinuum, TWSe2Params


class GateDefinedDot:
    def __init__(self, params: TWSe2Params, W: float, V0: float = 100.0,
                 N1: int = 60, NG: int = 4, NB: int = 2,
                 tail_exponent: float = 30.0, verbose: bool = True):
        """tail_exponent: keep couplings with (Q R / 2)^2 <= tail_exponent
        (e^-30 ~ 1e-13 relative truncation of Vt)."""
        self.p = params
        self.model = MoireContinuum(params, NG=NG)
        self.W, self.V0, self.N1, self.NB = W, V0, N1, NB
        self.R = W * params.aM
        self.Qcut = 2.0 * np.sqrt(tail_exponent) / self.R
        self.verbose = verbose
        self._solve_bulk()
        self._assemble()

    # ------------------------------------------------------------------
    def _solve_bulk(self):
        m, N1, NB = self.model, self.N1, self.NB
        fr = np.arange(N1) / N1
        ii, jj = np.meshgrid(np.arange(N1), np.arange(N1), indexing="ij")
        self.kfrac = np.stack([ii.ravel(), jj.ravel()], axis=1)  # integer (i,j)
        kvecs = (self.kfrac[:, 0:1] / N1) * m.b1[None, :] \
              + (self.kfrac[:, 1:2] / N1) * m.b2[None, :]
        self.kvecs = kvecs
        Nk = N1 * N1

        eps = np.empty((Nk, NB))
        A = np.empty((Nk, m.dim, NB), dtype=complex)
        chunk = 256
        for s in range(0, Nk, chunk):
            ks = kvecs[s:s + chunk]
            Hs = np.empty((len(ks), m.dim, m.dim), dtype=complex)
            for t, k in enumerate(ks):
                Hs[t] = m.hamiltonian(k)
            e, v = np.linalg.eigh(Hs)
            eps[s:s + chunk] = e[:, ::-1][:, :NB]          # descending
            A[s:s + chunk] = v[:, :, ::-1][:, :, :NB]
        self.eps, self.A = eps, A
        if self.verbose:
            print(f"bulk basis: {Nk} k-points x {NB} bands "
                  f"(dim {Nk*NB}), band-1 top {eps[:,0].max():.2f} meV")

    # ------------------------------------------------------------------
    def _assemble(self):
        m, N1, NB = self.model, self.N1, self.NB
        Nk = N1 * N1
        dimH = Nk * NB
        A = self.A                       # (Nk, 2*nG, NB)
        nG = m.nG
        Acell = (np.sqrt(3.0) / 2.0) * self.p.aM ** 2
        pref = np.pi * self.R ** 2 * self.V0 / (Nk * Acell)

        # admissible (di, dj, dG) classes ------------------------------
        # signed k-index differences di, dj in (-N1, N1); folded transfer
        # Q = (di/N1) b1 + (dj/N1) b2 + m1 b1 + m2 b2, keep |Q| <= Qcut.
        classes = []
        mrange = range(-2, 3)
        for di in range(-N1 + 1, N1):
            for dj in range(-N1 + 1, N1):
                dk = (di / N1) * m.b1 + (dj / N1) * m.b2
                for m1 in mrange:
                    for m2 in mrange:
                        Q = dk + m1 * m.b1 + m2 * m.b2
                        q2 = Q @ Q
                        if q2 <= self.Qcut ** 2:
                            classes.append((di, dj, m1, m2,
                                            np.exp(-q2 * self.R ** 2 / 4.0)))
        if self.verbose:
            print(f"assembly: {len(classes)} (dk,dG) classes, "
                  f"Qcut={self.Qcut:.3f} 1/nm = {self.Qcut/m.p.gm:.3f} g")

        # precompute G-shift index maps per dG --------------------------
        shift_cache = {}
        for (_, _, m1, m2, _) in classes:
            if (m1, m2) not in shift_cache:
                dst, src = m._pairs((m1, m2))
                # both layers: block-diagonal shift in layer space
                src2 = np.concatenate([src, nG + src])
                dst2 = np.concatenate([dst, nG + dst])
                shift_cache[(m1, m2)] = (dst2, src2)

        idx_i = self.kfrac[:, 0]
        idx_j = self.kfrac[:, 1]
        kindex = idx_i * N1 + idx_j      # identity, but explicit

        rows, cols, vals = [], [], []
        for (di, dj, m1, m2, vfac) in classes:
            # pairs: source k at (i,j), target k' at (i+di, j+dj) — in-range only
            i0, i1 = max(0, -di), min(N1, N1 - di)
            j0, j1 = max(0, -dj), min(N1, N1 - dj)
            if i0 >= i1 or j0 >= j1:
                continue
            isrc = np.arange(i0, i1)
            jsrc = np.arange(j0, j1)
            II, JJ = np.meshgrid(isrc, jsrc, indexing="ij")
            ksrc = (II * N1 + JJ).ravel()
            ktgt = ((II + di) * N1 + (JJ + dj)).ravel()

            dst2, src2 = shift_cache[(m1, m2)]
            # S[pair, p', p] = sum_x conj(A[ktgt, dst2, p']) * A[ksrc, src2, p]
            At = A[ktgt][:, dst2, :]                 # (M, nx, NB)
            As = A[ksrc][:, src2, :]                 # (M, nx, NB)
            S = np.einsum("mxp,mxq->mpq", np.conj(At), As)  # (M, NB(tgt), NB(src))
            val = (pref * vfac) * S

            # rows: (p', ktgt), cols: (p, ksrc); state index = k*NB + p
            for pt in range(NB):
                for ps in range(NB):
                    rows.append((ktgt * NB + pt).astype(np.int32))
                    cols.append((ksrc * NB + ps).astype(np.int32))
                    vals.append(np.ascontiguousarray(val[:, pt, ps]))

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        Hc = sp.coo_matrix((vals, (rows, cols)), shape=(dimH, dimH))
        del rows, cols, vals
        Hc = Hc.tocsr()
        # both (dk,dG) directions are enumerated, so Hc is Hermitian by
        # construction (verified: herm err 0.0 on small grids).
        diag = self.eps.reshape(-1)      # (Nk*NB,) ordering k*NB+p
        Hc = Hc + sp.diags(diag)
        if self.verbose:
            print(f"H: dim {dimH}, nnz {Hc.nnz/1e6:.2f}M", flush=True)
        self.H = Hc
        self.A = None                    # free the Bloch coefficients

    # ------------------------------------------------------------------
    def solve(self, nstates: int = 24):
        """Largest eigenvalues = confined dot states (descending)."""
        e, v = spla.eigsh(self.H, k=nstates, which="LA", tol=0,
                          maxiter=5000)
        order = np.argsort(e)[::-1]
        self.E = e[order]
        self.Vec = v[:, order]
        return self.E

    # ------------------------------------------------------------------
    def valley_weights(self, state: int) -> tuple[float, float]:
        """Weight of eigenstate near kappa_+ vs kappa_- (folded distance)."""
        m = self.model
        C = self.Vec[:, state].reshape(-1, self.NB)     # (Nk, NB)
        wk = np.sum(np.abs(C) ** 2, axis=1)
        # folded distances to the kappa points
        d_p = np.full(len(self.kvecs), np.inf)
        d_m = np.full(len(self.kvecs), np.inf)
        for m1 in range(-1, 2):
            for m2 in range(-1, 2):
                shift = m1 * m.b1 + m2 * m.b2
                d_p = np.minimum(d_p, np.linalg.norm(
                    self.kvecs + shift - m.kappa_p, axis=1))
                d_m = np.minimum(d_m, np.linalg.norm(
                    self.kvecs + shift - m.kappa_m, axis=1))
        wp = float(wk[d_p < d_m].sum())
        wm = float(wk[d_m <= d_p].sum())
        return wp, wm

    def analyze(self, nstates: int = 24):
        E = self.solve(nstates)
        E0 = E - E[0]
        d1s = (E[0] - E[1]) * 1000.0          # ueV
        # omega = spacing from 1S doublet (mean) to the lowest 2P state,
        # i.e. the first state below the doublet (paper Fig. 2/3 definition)
        omega = float(np.mean(E[:2]) - E[2])
        gap23 = float(E[1] - E[2])
        out = {
            "W": self.W, "V0": self.V0, "N1": self.N1,
            "NG": self.model.NG, "NB": self.NB,
            "R_nm": float(self.R),
            "E_rel_meV": E0[:12].tolist(),
            "Delta1S_ueV": float(d1s),
            "omega_meV": omega,
            "gap_1S2P_raw_meV": gap23,
            "valley_weights_1S": [self.valley_weights(0),
                                  self.valley_weights(1)],
        }
        return out


def ho_omega(V0: float, mstar_me: float, R_nm: float) -> float:
    """Harmonic-oscillator prediction hbar*omega = sqrt(2 V0 hb^2/(m* R^2))."""
    from continuum import HBAR2_2ME
    return float(np.sqrt(2.0 * V0 * 2.0 * HBAR2_2ME / mstar_me) / R_nm)


if __name__ == "__main__":
    import json, sys, time
    W = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    N1 = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    NB = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    NG = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    t0 = time.time()
    dot = GateDefinedDot(TWSe2Params(theta_deg=5.08), W=W, N1=N1, NG=NG, NB=NB)
    out = dot.analyze()
    out["ho_omega_meV"] = ho_omega(100.0, 0.50, dot.R)
    out["runtime_s"] = round(time.time() - t0, 1)
    print(json.dumps(out, indent=2))
