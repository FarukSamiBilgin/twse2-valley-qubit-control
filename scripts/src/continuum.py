"""tWSe2 K-valley continuum model — Devakul-Fu parametrization.

Conventions follow Devakul, Crepel, Zhang, Fu, Nat. Commun. 12, 6730 (2021):

    Delta_{1,2}(r) = 2 V sum_{j=1,3,5} cos(g_j . r +/- psi)
    Delta_T(r)     = w (1 + e^{-i g2 . r} + e^{-i g3 . r})

    H_up(k) = [[ -hb^2 |k+G-kappa_+|^2 / 2m*  + Delta_1 ,  Delta_T   ],
               [  Delta_T^dag ,  -hb^2 |k+G-kappa_-|^2 / 2m* + Delta_2 ]]

g_j are (j-1)*60deg CCW rotations of g1 = (4*pi/(sqrt(3) aM)) x_hat.
kappa_+/- are the two inequivalent mBZ corners (layer 1 <-> kappa_+).
Spin-down (K') sector is the time-reversal conjugate; energies identical.

Optional extensions:
  - layer-odd "ferroelectric" first-harmonic term  +/- Delta_z(r) with
    Delta_z = 2 Vz sum_{j=1,3,5} cos(g_j . r + psi_z)   (R-stack moire
    polarization; off by default — DF fit already carries the MX/XM
    asymmetry through +/-psi).
  - uniform displacement field D/2 * sigma_z (layer basis), as in
    Saleem-Potasz-Hankiewicz arXiv:2608.05229.

Units: meV, nm.  All eigenvalues are *valence* energies (band max ~ 0-ish);
holes live at the top of the spectrum.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

HBAR2_2ME = 38.09982  # hbar^2/(2 m_e) in meV nm^2


@dataclass
class TWSe2Params:
    V: float = 9.0          # meV      (Devakul-Fu)
    psi_deg: float = 128.0  # deg      (Devakul-Fu)
    w: float = 18.0         # meV      (Devakul-Fu)
    m_star: float = 0.43    # m_e      (Devakul-Fu band-structure value)
    a0: float = 0.3317      # nm       monolayer lattice constant
    theta_deg: float = 5.08
    Vz: float = 0.0         # meV      layer-odd harmonic (moire ferroelectric)
    psi_z_deg: float = 90.0
    D: float = 0.0          # meV      displacement field (layer detuning)

    @property
    def theta(self) -> float:
        return np.deg2rad(self.theta_deg)

    @property
    def aM(self) -> float:
        return self.a0 / (2.0 * np.sin(self.theta / 2.0))

    @property
    def gm(self) -> float:
        """moire reciprocal-lattice constant |b1|"""
        return 4.0 * np.pi / (np.sqrt(3.0) * self.aM)

    @property
    def psi(self) -> float:
        return np.deg2rad(self.psi_deg)

    @property
    def psi_z(self) -> float:
        return np.deg2rad(self.psi_z_deg)

    @property
    def kin(self) -> float:
        """hbar^2/(2 m*) in meV nm^2"""
        return HBAR2_2ME / self.m_star


def _rot(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[c, -s], [s, c]])


class MoireContinuum:
    """Plane-wave solver for the two-layer K-valley continuum model."""

    def __init__(self, p: TWSe2Params, NG: int = 5):
        self.p = p
        self.NG = NG
        g = p.gm
        self.b1 = g * np.array([1.0, 0.0])
        self.b2 = g * np.array([0.5, np.sqrt(3.0) / 2.0])
        # inequivalent mBZ corners; layer 1 parabola sits at kappa_+.
        # The corner assignment is NOT free: with Delta_T = w(1 + e^{-i g2 r}
        # + e^{-i g3 r}) as written by Devakul-Fu, the 6-state gamma ring
        # (3 layer-1 + 3 layer-2 plane waves at |k| = g/sqrt(3)) must close
        # with 6 tunneling links (SM Note 4).  That forces layer 1 onto the
        # *lower* corner for our g_j orientation; the other choice is
        # equivalent to psi -> -psi (wrong MX/XM registry).
        self.kappa_p = g * np.array([0.5, -0.5 / np.sqrt(3.0)])
        self.kappa_m = g * np.array([0.5, 0.5 / np.sqrt(3.0)])

        # hexagonally truncated G lattice: max(|n1|,|n2|,|n1+n2|) <= NG
        ns = [(n1, n2)
              for n1 in range(-NG, NG + 1)
              for n2 in range(-NG, NG + 1)
              if max(abs(n1), abs(n2), abs(n1 + n2)) <= NG]
        ns.sort()
        self.gidx = {t: i for i, t in enumerate(ns)}
        self.gints = np.array(ns)                       # (nG, 2)
        self.Gs = self.gints @ np.vstack([self.b1, self.b2])  # (nG, 2)
        self.nG = len(ns)
        self.dim = 2 * self.nG   # layer-major blocks: [layer1 Gs, layer2 Gs]

        self._build_static()

    # ------------------------------------------------------------------
    def _pairs(self, shift: tuple[int, int]):
        """index pairs (i_target, i_source) such that G_target = G_source + shift"""
        src, dst = [], []
        m1, m2 = shift
        for (n1, n2), i in self.gidx.items():
            j = self.gidx.get((n1 + m1, n2 + m2))
            if j is not None:
                src.append(i)
                dst.append(j)
        return np.array(dst), np.array(src)

    def _build_static(self):
        """k-independent part of H (potential terms)."""
        p = self.p
        nG, dim = self.nG, self.dim
        H = np.zeros((dim, dim), dtype=complex)

        # intralayer moire potential: Delta_l = 2V sum_{j=1,3,5} cos(g_j.r + s_l psi)
        # e^{+i g.r} couples source G -> target G+g with the given amplitude.
        # g1=(1,0), g3=(-1,1), g5=(0,-1) in integer (b1,b2) coords.
        odd_shifts = [(1, 0), (-1, 1), (0, -1)]
        for l, sgn in ((0, +1.0), (1, -1.0)):          # layer 1: +psi, layer 2: -psi
            amp = p.V * np.exp(1j * sgn * p.psi)
            off = l * nG
            for sh in odd_shifts:
                d, s = self._pairs(sh)
                H[off + d, off + s] += amp
                H[off + s, off + d] += np.conj(amp)
            if p.Vz != 0.0:
                zsgn = +1.0 if l == 0 else -1.0
                zamp = zsgn * p.Vz * np.exp(1j * p.psi_z)
                for sh in odd_shifts:
                    d, s = self._pairs(sh)
                    H[off + d, off + s] += zamp
                    H[off + s, off + d] += np.conj(zamp)

        # interlayer tunneling Delta_T = w (1 + e^{-i g2.r} + e^{-i g3.r})
        # block (layer1, layer2): shifts 0, -g2=(0,-1), -g3=(1,-1)
        for sh in [(0, 0), (0, -1), (1, -1)]:
            d, s = self._pairs(sh)
            H[d, nG + s] += p.w
            H[nG + s, d] += p.w        # hermitian conjugate

        # displacement field D/2 sigma_z (layer basis)
        if p.D != 0.0:
            H[np.arange(nG), np.arange(nG)] += p.D / 2.0
            H[nG + np.arange(nG), nG + np.arange(nG)] += -p.D / 2.0

        self.Hstatic = H

    # ------------------------------------------------------------------
    def hamiltonian(self, k: np.ndarray) -> np.ndarray:
        p = self.p
        H = self.Hstatic.copy()
        kg = k[None, :] + self.Gs                       # (nG,2)
        e1 = -p.kin * np.sum((kg - self.kappa_p) ** 2, axis=1)
        e2 = -p.kin * np.sum((kg - self.kappa_m) ** 2, axis=1)
        i = np.arange(self.nG)
        H[i, i] += e1
        H[self.nG + i, self.nG + i] += e2
        return H

    def bands(self, kpts: np.ndarray, nbands: int | None = None,
              vectors: bool = False):
        """Eigen-decompose at each k (rows of kpts). Returns energies sorted
        DESCENDING (top valence band first)."""
        Es, Vs = [], []
        for k in np.atleast_2d(kpts):
            if vectors:
                e, v = np.linalg.eigh(self.hamiltonian(k))
                order = np.argsort(e)[::-1]
                Es.append(e[order][:nbands])
                Vs.append(v[:, order][:, :nbands])
            else:
                e = np.linalg.eigvalsh(self.hamiltonian(k))[::-1]
                Es.append(e[:nbands])
        E = np.array(Es)
        return (E, Vs) if vectors else E

    # ------------------------------------------------------------------
    def kpath(self, npts: int = 40):
        """gamma -> kappa_+ -> m -> kappa_- -> gamma path."""
        gpt = np.zeros(2)
        m = 0.5 * (self.kappa_p + self.kappa_m)   # edge midpoint (g/2, 0)
        nodes = [gpt, self.kappa_p, m, self.kappa_m, gpt]
        labels = ["$\\gamma$", "$\\kappa_+$", "$m$", "$\\kappa_-$", "$\\gamma$"]
        pts, dist, ticks = [], [0.0], [0.0]
        for a, b in zip(nodes[:-1], nodes[1:]):
            seg = np.linspace(0, 1, npts, endpoint=False)[:, None] * (b - a) + a
            pts.extend(seg)
            ticks.append(ticks[-1] + np.linalg.norm(b - a))
        pts.append(nodes[-1])
        pts = np.array(pts)
        d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        return pts, d, ticks, labels

    def effective_mass(self, band: int = 0, at: str = "kappa_p",
                       dq_frac: float = 0.01) -> float:
        """Fitted band-edge mass (in m_e) from a 5-point quadratic stencil,
        averaged over x and y directions. Positive = hole-like maximum."""
        k0 = {"kappa_p": self.kappa_p, "kappa_m": self.kappa_m,
              "gamma": np.zeros(2)}[at]
        dq = dq_frac * self.p.gm
        curvs = []
        for direction in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
            qs = np.array([-2, -1, 0, 1, 2]) * dq
            ks = k0[None, :] + qs[:, None] * direction[None, :]
            E = self.bands(ks, nbands=band + 1)[:, band]
            # second derivative via central 5-point
            d2 = (-E[0] + 16 * E[1] - 30 * E[2] + 16 * E[3] - E[4]) / (12 * dq ** 2)
            curvs.append(d2)
        d2 = float(np.mean(curvs))
        # E = -hb^2 q^2 / 2 m*  =>  d2 = -hb^2/m* ;  m*/m_e = -2*HBAR2_2ME/d2
        return -2.0 * HBAR2_2ME / d2

    # ------------------------------------------------------------------
    def gamma_sixstate_check(self):
        """Devakul-Fu SM Note 4: top-6 states at gamma should match
        e0 + E_n,  E_n = 2 w cos(pi n/3) + 2 V cos(2 pi n/3 - psi)."""
        p = self.p
        e0 = -p.kin * (p.gm / np.sqrt(3.0)) ** 2
        En = np.array([2 * p.w * np.cos(np.pi * n / 3)
                       + 2 * p.V * np.cos(2 * np.pi * n / 3 - p.psi)
                       for n in range(1, 7)])
        analytic = np.sort(e0 + En)[::-1]
        numeric = self.bands(np.zeros((1, 2)), nbands=6)[0]
        return analytic, numeric
