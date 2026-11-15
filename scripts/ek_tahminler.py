"""Two order-of-magnitude estimates quoted in the text.

1. In-plane footprint (FWHM) of a gate electrode of width w at height z
   above the moire plane: line-source kernel z/(x^2+z^2) (FWHM 2z)
   convolved with a box of width w.
2. Quasistatic dipolar field of the hBN encapsulant nuclei (10B, 11B, 14N)
   averaged over a Gaussian hole density exp(-r^2/l^2)/(pi l^2); nuclei as
   a continuum filling |z| > d on both sides; isotropic random moments.
   Var(B_z) = sides * n <mu_z^2> (mu0/2)^2 (1/2pi) Int q^2 e^{-q^2 l^2/2}
              e^{-2qd} dq ;  T2* = sqrt(2) hbar / (g mu_B B_rms).
"""
import numpy as np
from scipy import integrate

# ------------------------------------------------------------ gate footprint
def fwhm(z, w):
    x = np.linspace(-200, 200, 40001)
    k = z / (x ** 2 + z ** 2)
    prof = np.convolve(k, (np.abs(x) <= w / 2).astype(float), mode="same") \
        if w > 0 else k
    prof = prof / prof.max()
    above = x[prof >= 0.5]
    return above.max() - above.min()


print("gate footprint FWHM (nm):")
for w in (0.0, 10.0, 20.0):
    print(f"  w={w:4.0f} nm: z=3 -> {fwhm(3, w):5.1f}, z=10 -> {fwhm(10, w):5.1f},"
          f" z=20 -> {fwhm(20, w):5.1f}")

# ------------------------------------------------------------ hBN dipolar
mu0, muN, muB, hbar = 4e-7 * np.pi, 5.0507837e-27, 9.2740101e-24, 1.054571817e-34
m2 = (0.80 * 2.6886 ** 2 * 2.5 / 4.5          # 11B, I=3/2
      + 0.20 * 1.8006 ** 2 * 4.0 / 9.0        # 10B, I=3
      + 0.99636 * 0.40376 ** 2 * 2.0 / 3.0)   # 14N, I=1
m2 *= muN ** 2                                # <mu_z^2> per formula unit
n = 2.27e3 / (24.82e-3 / 6.02214e23)          # formula units / m^3


def B_rms(l_nm, d_nm, sides=2):
    l, d = l_nm * 1e-9, d_nm * 1e-9
    f = lambda q: q ** 2 * np.exp(-q ** 2 * l ** 2 / 2 - 2 * q * d)
    I, _ = integrate.quad(f, 0, 50 / d, limit=400)
    return np.sqrt(sides * n * m2 * (mu0 / 2) ** 2 * I / (2 * np.pi))


print("hBN nuclear dipolar ceiling on T2* (Kramers qubit):")
for l in (1.0, 1.6):
    for d in (0.6, 1.0):
        B = B_rms(l, d)
        T = [np.sqrt(2) * hbar / (g * muB * B) * 1e6 for g in (2.3, 4.2)]
        print(f"  l={l} nm d={d} nm: B_rms={B*1e6:5.1f} uT, "
              f"T2* = {T[1]:.2f}-{T[0]:.2f} us (g=4.2-2.3)")
