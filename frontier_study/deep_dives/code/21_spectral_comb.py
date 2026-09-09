"""
Deep dive 21 — The n-channel spectral comb (BL73), the harmonic-collision phase law,
and the price of budget neutrality (BL70).

Run:  python 21_spectral_comb.py            # all sections
      python 21_spectral_comb.py S3 S7      # selected sections

Sections
  S1  comb orthogonality under heterogeneous adstock; the trend coupling -6/(T^2-7)
  S2  off-grid Gram penalty (channel-separation rule)
  S3  the harmonic-collision phase law, VIF = 1/sin^2(p psi_j - psi_k)   [p = 2 and 3]
  S4  the same collision in the FULL structural model: phase as a design variable
  S5  packing capacity + the record-length coprimality law
  S6  budget-neutral rank theorem (the sum direction lies in the null space)
  S7  the budget-neutral escape law and its DC / Nyquist nulls
  S8  Lambda: the price of budget neutrality in dive-20 frontier units
  S9  attacks: on/off flighting harmonics, clipping, seasonal sidebands, alpha error,
      record truncation
  S10 comb optimality: simultaneous frontier attainment, and additivity
  S11 negative / power controls (seasonal bin, duplicate bin, synchronised flighting)
  S12 the harmonic readout: model-free c2, c3 and the 3w/2w shape test
  S13 readout power: se(c2hat) = 4 sigma sqrt(2/T)/B^2, SNR, and the modulus bias
  S14 rank of the synchronised design (full Hill vs calibrated shape)
  S15 full nonlinear Monte Carlo, n = 6, four designs at matched cost
  S16 the n-channel price of identification and channel-scale invariance

Model
  y_t = z_t'gamma + sum_j beta_j h_j(a_jt) + eps_t,  a_jt = x_jt + alpha_j a_{j,t-1},
  h(a) = a^S/(a^S + K^S);  z = intercept + centred trend + Fourier harmonics k<=3 at period 52.
  Channels are parametrised by (K, S, alpha, H*) with H* the saturation level at the myopic
  profit optimum; beta follows from the first-order condition m beta h'(abar) = 1 - alpha.
"""
import sys
import numpy as np
from scipy.optimize import brentq, least_squares, milp, LinearConstraint, Bounds
from scipy.signal import lfilter

TW = 364          # analysis window (7 years; 364/52 = 7, coprime to 6 — see S5)
SIGMA = 0.05      # residual sales sd
MARGIN = 2.0

# --------------------------------------------------------------------------- primitives

def hill(a, K, S):
    a = np.maximum(a, 1e-12)
    return a**S / (a**S + K**S)

def hill_d(a, K, S):
    a = np.maximum(a, 1e-12); u = a**S; v = K**S
    return S * u * v / (a * (u + v)**2)

def hill_d2(a, K, S, eps=1e-5):
    return (hill_d(a + eps, K, S) - hill_d(a - eps, K, S)) / (2 * eps)

def hill_d3(a, K, S, eps=1e-3):
    return (hill_d2(a + eps, K, S) - hill_d2(a - eps, K, S)) / (2 * eps)

def adstock(x, alpha, a0=None):
    zi = [alpha * ((x[0] / (1 - alpha)) if a0 is None else a0)]
    return lfilter([1.0], [1.0, -alpha], x, zi=zi)[0]

def adstock_periodic(x, alpha, burn=10):
    """Steady state: replay the periodic extension of x, keep the last window."""
    xx = np.concatenate([np.tile(x, burn), x])
    return adstock(xx, alpha, a0=0.0)[-len(x):]

def Hf(alpha, w):
    """Adstock frequency response 1/(1 - alpha e^{-i w})."""
    return 1.0 / (1 - alpha * np.exp(-1j * np.asarray(w, dtype=float)))

def opt_spend(beta, K, S, alpha, m):
    f = lambda a: m * beta * hill_d(a, K, S) - (1 - alpha)
    grid = np.linspace(K * 0.02, K * 60, 40000)
    idx = np.where(np.diff(np.sign(f(grid))) != 0)[0]
    if len(idx) == 0:
        raise RuntimeError("no interior operating point")
    i = idx[-1]                                   # upper crossing = concave branch
    abar = brentq(f, grid[i], grid[i + 1])
    return abar, abar * (1 - alpha)

def nuisance(T, n_harm=3, period=52.0, trend=True):
    t = np.arange(T); cols = [np.ones(T)]
    if trend:
        cols.append((t - t.mean()) / T)
    for k in range(1, n_harm + 1):
        cols.append(np.cos(2 * np.pi * k * t / period))
        cols.append(np.sin(2 * np.pi * k * t / period))
    return np.column_stack(cols)

def resid_maker(Z):
    Q, _ = np.linalg.qr(Z)
    return lambda v: v - Q @ (Q.T @ v)


class Channel:
    """Give either beta, or H* (saturation at the profit optimum) and beta is solved."""
    def __init__(self, beta, K, S, alpha, m=MARGIN, name="", Hstar=None):
        if Hstar is not None:
            abar = K * (Hstar / (1 - Hstar))**(1.0 / S)
            beta = (1 - alpha) / (m * hill_d(abar, K, S))
        self.beta, self.K, self.S, self.alpha, self.m = beta, K, S, alpha, m
        self.name = name
        self.abar, self.xbar = opt_spend(beta, K, S, alpha, m)
        self.c1 = beta * hill_d(self.abar, K, S)
        self.c2 = beta * hill_d2(self.abar, K, S)
        self.c3 = beta * hill_d3(self.abar, K, S)
        self.kappa = m * beta * abs(hill_d2(self.abar, K, S))
        self.tau = 1.0 / (1 - alpha)
        self.mROAS = m * self.c1 / (1 - alpha)


SPECS = [('A', 1.5, 2.0, 0.60, 0.68), ('B', 2.0, 1.6, 0.30, 0.55), ('C', 1.0, 2.4, 0.75, 0.75),
         ('D', 3.0, 1.3, 0.45, 0.60), ('E', 0.8, 3.0, 0.20, 0.70), ('F', 2.5, 2.0, 0.55, 0.50)]
MIS_COMB = [19, 23, 25, 29, 31, 37]

def channels(k=6):
    return [Channel(None, K, S, al, MARGIN, name=n, Hstar=Hs) for n, K, S, al, Hs in SPECS[:k]]


# --------------------------------------------------------------------------- MMM / FIM

class Design:
    def __init__(self, chs, mis, amps, phis, Tw=TW):
        self.chs, self.Tw = chs, Tw
        self.mis = np.asarray(mis, float); self.amps = np.asarray(amps, float)
        self.phis = np.asarray(phis, float)
    def spend(self):
        t = np.arange(self.Tw)
        return np.array([c.xbar * (1 + self.amps[j] * np.cos(2 * np.pi * self.mis[j] / self.Tw * t
                                                             + self.phis[j]))
                         for j, c in enumerate(self.chs)])

def theta_true(chs, gamma):
    th = list(gamma)
    for c in chs:
        th += [c.beta, c.K, c.S, c.alpha]
    return np.array(th, float)

def mean_fn(theta, chs, X, Z):
    p = Z.shape[1]; mu = Z @ theta[:p]
    for j, c in enumerate(chs):
        b, K, S, al = theta[p + 4 * j: p + 4 * j + 4]
        mu = mu + b * hill(adstock_periodic(X[j], al), K, S)
    return mu

def jac(theta, chs, X, Z, rel=1e-5):
    base = mean_fn(theta, chs, X, Z); J = np.empty((len(base), len(theta)))
    for i in range(len(theta)):
        h = rel * max(abs(theta[i]), 1e-3)
        tp = theta.copy(); tm = theta.copy(); tp[i] += h; tm[i] -= h
        J[:, i] = (mean_fn(tp, chs, X, Z) - mean_fn(tm, chs, X, Z)) / (2 * h)
    return J

def mroas_val(th, chs, p, j, xbar_obs):
    """mROAS_j = m beta_j h'(abar_j)/(1-alpha_j) with abar_j = OBSERVED mean spend/(1-alpha_j).
    The operating point follows the data; re-solving the FOC inside makes the gradient zero."""
    b, K, S, al = th[p + 4 * j: p + 4 * j + 4]
    return chs[j].m * b * hill_d(xbar_obs / (1 - al), K, S) / (1 - al)

def mroas_grad(theta, chs, Z, j, xbar_obs, rel=1e-5):
    p = Z.shape[1]; g = np.zeros(len(theta))
    for i in range(p + 4 * j, p + 4 * j + 4):
        h = rel * max(abs(theta[i]), 1e-3)
        tp = theta.copy(); tm = theta.copy(); tp[i] += h; tm[i] -= h
        g[i] = (mroas_val(tp, chs, p, j, xbar_obs) - mroas_val(tm, chs, p, j, xbar_obs)) / (2 * h)
    return g

def keepset(Z, n, free):
    p = Z.shape[1]; keep = list(range(p))
    if free == 'full':
        for j in range(n): keep += [p + 4 * j + k for k in range(4)]
    elif free == 'local':                       # beta, alpha free; K, S calibrated
        for j in range(n): keep += [p + 4 * j, p + 4 * j + 3]
    return keep

def se_mroas(chs, X, Z, gamma, sigma=SIGMA, free='full'):
    th = theta_true(chs, gamma); J = jac(th, chs, X, Z)
    keep = keepset(Z, len(chs), free); Jk = J[:, keep]
    C = np.linalg.pinv(Jk.T @ Jk / sigma**2)
    out = []
    for jj in range(len(chs)):
        g = mroas_grad(th, chs, Z, jj, X[jj].mean())[keep]
        out.append(float(np.sqrt(max(g @ C @ g, 0))))
    return np.array(out)


# --------------------------------------------------------------------------- sections

def S1():
    print("S1 — comb orthogonality under heterogeneous adstock")
    chs = channels(); amp = 0.10; t = np.arange(TW)
    def devs(kind):
        out = []
        for c, mi in zip(chs, MIS_COMB):
            w = 2 * np.pi * mi / TW; u = amp * c.xbar * np.cos(w * t)
            if kind == 'steady':   a = adstock_periodic(c.xbar + u, c.alpha)
            elif kind == 'switch': a = adstock(u, c.alpha, a0=0.0)     # baseline already in steady state
            out.append(a - a.mean())
        return out
    for kind in ('steady', 'switch'):
        for tr in (True, False):
            Z = nuisance(TW, trend=tr); M = resid_maker(Z)
            X = np.column_stack([M(v) for v in devs(kind)]); G = X.T @ X
            d = np.sqrt(np.diag(G)); R = G / np.outer(d, d)
            print("   %-7s trend=%-5s  max|off-diag corr| = %.3e   maxVIF = %.6f"
                  % (kind, tr, np.abs(R - np.eye(6)).max(), np.diag(np.linalg.inv(R)).max()))
    print("   trend-induced coupling of two pure on-grid cosines (should be -6/(T^2-7), bin-free):")
    Z = nuisance(TW, n_harm=0, trend=True); M = resid_maker(Z)
    for (m1, m2) in [(5, 7), (19, 23), (80, 83)]:
        a = M(np.cos(2 * np.pi * m1 * t / TW)); b = M(np.cos(2 * np.pi * m2 * t / TW))
        print("     bins (%3d,%3d): %.6e    -6/(T^2-7) = %.6e   -6/T^2 = %.6e"
              % (m1, m2, a @ b / np.sqrt((a @ a) * (b @ b)), -6 / (TW**2 - 7), -6 / TW**2))


def S2():
    print("S2 — off-grid Gram penalty between two channel tones")
    Z = nuisance(TW); M = resid_maker(Z); t = np.arange(TW)
    for d in [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        v = []
        for mi in (20.0, 20.0 + d):
            w = 2 * np.pi * mi / TW
            v.append(M(np.real(0.1 * Hf(0.6, w) * np.exp(1j * w * t))))
        r = v[0] @ v[1] / np.sqrt((v[0] @ v[0]) * (v[1] @ v[1]))
        print("   separation %.2f bins: corr = %+.4f   VIF = %.3f" % (d, r, 1 / (1 - r**2)))


def _volterra_vif(p, mA, alA, alB, phiB, amp=0.10):
    """VIF of the coefficient on channel B's tone when w_B = p w_A, local Volterra model."""
    t = np.arange(TW); Z = nuisance(TW); M = resid_maker(Z)
    wA = 2 * np.pi * mA / TW; wB = p * wA
    vA = amp * abs(Hf(alA, wA)) * np.cos(wA * t + np.angle(Hf(alA, wA)))
    vB = amp * abs(Hf(alB, wB)) * np.cos(wB * t + phiB + np.angle(Hf(alB, wB)))
    cols = [Z, vA[:, None], 0.5 * (vA**2)[:, None]]
    if p == 3: cols.append((vA**3 / 6)[:, None])
    cols += [vB[:, None], 0.5 * (vB**2)[:, None]]
    X = np.hstack(cols); C = np.linalg.inv(X.T @ X)
    r = M(vB); return C[-2, -2] * (r @ r)

def S3():
    print("S3 — harmonic-collision phase law:  VIF = 1/sin^2(p psi_A - psi_B)")
    alA, alB, mA = 0.60, 0.30, 19; wA = 2 * np.pi * mA / TW
    for p in (2, 3):
        psiA = np.angle(Hf(alA, wA)); off = np.angle(Hf(alB, p * wA))
        errs = []
        print("   p = %d" % p)
        for phiB in np.linspace(0, np.pi, 19):
            vif = _volterra_vif(p, mA, alA, alB, phiB)
            pred = 1 / np.sin(p * psiA - (phiB + off))**2
            errs.append(abs(vif / pred - 1))
            if int(round(phiB / (np.pi / 18))) % 3 == 0:
                print("     phi_B=%.3f  VIF=%10.4f  theory=%10.4f  ratio=%.5f" % (phiB, vif, pred, vif / pred))
        print("     max RELATIVE deviation over [0,pi]: %.2e" % max(errs))
    print("   (without the trend column the law is exact to ~1e-10; the residual is the trend)")


def S4():
    print("S4 — the same collision in the full structural model: phase as a design variable")
    chs = channels(2); Z = nuisance(TW); gamma = np.zeros(Z.shape[1]); gamma[0] = 1.0
    b = se_mroas(chs, Design(chs, [19, 23], [.1, .1], [0, 0]).spend(), Z, gamma)
    print("   clean baseline (bins 19,23): se(mROAS) = (%.4f, %.4f)" % tuple(b))
    f = lambda ph: se_mroas(chs, Design(chs, [19, 38], [.1, .1], [0, ph]).spend(), Z, gamma)[1] / b[1]
    for lbl, ph in [("naive phi_B = 0", 0.0), ("local-theory quadrature 0.9650", 0.9650),
                    ("full-model optimum 1.5267", 1.5267)]:
        print("     %-32s se_B ratio = %8.4f" % (lbl, f(ph)))
    for N in (49, 401, 4001):
        r = np.array([f(p) for p in np.linspace(0, np.pi, N)])
        print("     %5d-point grid: min %.4f  median %.3f  max %.1f" % (N, r.min(), np.median(r), r.max()))
    print("   the maximum is a POLE — its measured value is a grid artefact; the min and median converge.")
    print("   amplitude sweep (median se_B ratio over phase):")
    for amp in (0.05, 0.10, 0.20):
        bb = se_mroas(chs, Design(chs, [19, 23], [amp] * 2, [0, 0]).spend(), Z, gamma)
        r = [se_mroas(chs, Design(chs, [19, 38], [amp] * 2, [0, p]).spend(), Z, gamma)[1] / bb[1]
             for p in np.linspace(0, np.pi, 49)]
        print("     amp=%.2f  min %.3f  median %.2f" % (amp, min(r), np.median(r)))


def admissible(T, tau, P=3, q=2, min_cycles=4):
    k = T / 52.0
    seas = {int(round(p * k)) for p in range(1, P + 1) if abs(p * k - round(p * k)) < 1e-9}
    mmax = int(np.floor(T / (4 * tau)))
    return [m for m in range(min_cycles, mmax + 1)
            if all((r * m) not in seas for r in range(1, q + 1))], sorted(seas), mmax

def max_harmonic_free(A, ratios=(2, 3)):
    n = len(A); idx = {m: i for i, m in enumerate(A)}
    edges = [(idx[m], idx[r * m]) for m in A for r in ratios if r * m in idx]
    if not edges: return n, A
    Aub = np.zeros((len(edges), n))
    for e, (i, j) in enumerate(edges): Aub[e, i] = 1; Aub[e, j] = 1
    res = milp(c=-np.ones(n), constraints=LinearConstraint(Aub, -np.inf, 1),
               integrality=np.ones(n), bounds=Bounds(0, 1))
    return int(round(-res.fun)), [A[i] for i in range(n) if res.x[i] > 0.5]

def S5():
    print("S5 — packing capacity and the record-length coprimality law")
    print("   T     k    |A| q=1  q=2  q=3   (tau=2.5, P=3)")
    from math import gcd
    for k in range(2, 14):
        T = 52 * k
        a = [len(admissible(T, 2.5, 3, q)[0]) for q in (1, 2, 3)]
        print("   %4d  %2d      %4d %4d %4d   %s" % (T, k, a[0], a[1], a[2],
              "gcd(k,6)=1 -> protection free" if gcd(k, 6) == 1 else
              ("costs %d,%d bins" % (a[0] - a[1], a[0] - a[2]) if a[0] != a[2] or a[0] != a[1] else "")))
    print("   capacity at T=364, P=3, q=2:")
    print("   tau   m_max  |A|  max 2x/3x-free subset")
    for tau in (1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        A, seas, mmax = admissible(364, tau, 3, 2); nf, sel = max_harmonic_free(A)
        print("   %4.2f   %4d   %3d        %3d   (%.2f of |A|)" % (tau, mmax, len(A), nf, nf / len(A)))
    print("   longer records (tau=2.5):")
    for k in (5, 7, 9, 11, 13, 15):
        A, _, mmax = admissible(52 * k, 2.5, 3, 2); nf, _ = max_harmonic_free(A)
        print("   T=%4d (%2d yr)  m_max=%3d  |A|=%3d  harmonic-free=%3d" % (52 * k, k, mmax, len(A), nf))


def _neutral_gram(alphas, U, Z=None):
    Z = nuisance(TW) if Z is None else Z; M = resid_maker(Z)
    V = [adstock_periodic(U[j], alphas[j]) for j in range(len(alphas))]
    X = np.column_stack([M(v - v.mean()) for v in V])
    return X.T @ X

def S6():
    print("S6 — budget-neutral rank theorem: the sum direction is in the null space")
    t = np.arange(TW)
    tone = lambda mi: np.cos(2 * np.pi * mi * t / TW)
    pats = [np.array([1, -1, 0, 0.]), np.array([0, 0, 1, -1.]),
            np.array([1, 1, -1, -1.]), np.array([1, -1, 1, -1.])]
    for lbl, al in [("homogeneous (.6,.6,.6,.6)", [.6] * 4),
                    ("distinct (.6,.5,.4,.3)", [.6, .5, .4, .3])]:
        for ntone in (1, 2, 4):
            U = sum(0.08 * np.outer(p, tone(mi)) for mi, p in zip([19, 23, 25, 29][:ntone], pats[:ntone]))
            G = _neutral_gram(al, U); s = np.linalg.svd(G, compute_uv=False)
            r = int((s > s.max() * 1e-10).sum())
            _, _, vt = np.linalg.svd(G); leak = np.linalg.norm(vt[r:] @ np.ones(4)) if r < 4 else 0.0
            print("   %-26s %d tone(s): rank %d/4   ||P_null 1||/||1|| = %.3e"
                  % (lbl, ntone, r, leak / 2))
    print("   -> equal adstock: rank saturates at n-1 and 1 lies entirely in the null space,")
    print("      no matter how many budget-neutral tones are added.  Distinct adstock: 2 tones suffice.")


def S7():
    print("S7 — the budget-neutral escape law")
    Z = nuisance(TW); M = resid_maker(Z); t = np.arange(TW)
    def num(a1, a2, mi, A):
        w = 2 * np.pi * mi / TW; u = A * np.cos(w * t)
        v1 = adstock_periodic(u, a1); v2 = adstock_periodic(-u, a2)
        X = np.column_stack([M(v1 - v1.mean()), M(v2 - v2.mean())])
        return SIGMA * np.sqrt(np.ones(2) @ np.linalg.inv(X.T @ X) @ np.ones(2))
    def theory(a1, a2, mi, A):
        w = 2 * np.pi * mi / TW
        D = 1 + a1 * a2 - a1 * np.exp(1j * w) - a2 * np.exp(-1j * w)
        im = abs(a1 - a2) * abs(np.sin(w)) / abs(D)**2
        return SIGMA * np.sqrt(2 / TW) * abs(Hf(a1, w) + Hf(a2, w)) / (A * im)
    print("   a1   a2  bin     numeric      theory     ratio")
    for (a1, a2, mi) in [(.6, .3, 19), (.6, .3, 9), (.6, .3, 37), (.8, .2, 19),
                         (.7, .65, 25), (.9, .1, 13), (.5, .45, 31), (.6, .3, 70)]:
        n_, t_ = num(a1, a2, mi, 0.05), theory(a1, a2, mi, 0.05)
        print("   %.2f %.2f %4d  %10.5g  %10.5g   %.6f" % (a1, a2, mi, n_, t_, n_ / t_))
    print("   Im(conj(H1)H2) = (a1-a2) sin w / |1+a1a2-a1 e^{iw}-a2 e^{-iw}|^2  -> 0 at w=0 and w=pi:")
    for mi in (0.001, 1, 4, 19, 37, 120, 181.9):
        w = 2 * np.pi * mi / TW
        print("     bin %8.3f (period %8.1f wk): %.6f"
              % (mi, TW / max(mi, 1e-9), np.imag(np.conj(Hf(.6, w)) * Hf(.3, w))))
    print("   optimal budget-neutral probe period at matched cost:")
    kap = 0.3146
    for (a1, a2) in [(.6, .3), (.8, .2), (.7, .5), (.9, .6), (.5, .45), (.85, .15)]:
        bins = np.arange(2, 120, 0.25)
        se = [theory(a1, a2, m, np.sqrt(4 / (TW * kap * (abs(Hf(a1, 2 * np.pi * m / TW))**2
              + abs(Hf(a2, 2 * np.pi * m / TW))**2)))) for m in bins]
        i = int(np.argmin(se))
        print("     alphas (%.2f,%.2f): %.1f wk   se_min %.4f   (4 tau_max = %.0f wk)"
              % (a1, a2, TW / bins[i], se[i], 4 / (1 - max(a1, a2))))


def S8():
    print("S8 — Lambda, the price of budget neutrality (dive-20 frontier units)")
    def parts(a1, a2, w):
        H1, H2 = Hf(a1, w), Hf(a2, w)
        D = 1 + a1 * a2 - a1 * np.exp(1j * w) - a2 * np.exp(-1j * w)
        im = (a1 - a2) * np.sin(w) / abs(D)**2
        g = abs(H1)**2 + abs(H2)**2
        return abs(H1 + H2)**2 * g / im**2, abs(H1 - H2)**2 * g / im**2
    print("   (a1,a2)  period   Lambda_neutral   Lambda_total   ratio")
    for (a1, a2) in [(.6, .3), (.8, .2), (.7, .5), (.9, .6), (.5, .45), (.65, .6)]:
        for P in (13.0, 19.16, 40.44):
            ln, lt = parts(a1, a2, 2 * np.pi / P)
            print("   (%.2f,%.2f)  %5.1f  %13.4g  %13.4g   %8.1f" % (a1, a2, P, ln, lt, ln / lt))
    print("   reference: a single channel's own c1 sits at Lambda = |H(w)|^2 ~ 3.4")


def S9():
    print("S9 — attacks")
    t = np.arange(TW); m0 = 13
    hp = lambda s, ms: {m: 2 * abs(np.fft.rfft(s)[m]) / TW for m in ms}
    print("   (a) harmonic content of realisable probes, relative to the fundamental")
    for lbl, s in [("pure sine", np.cos(2 * np.pi * m0 * t / TW)),
                   ("on/off square (flighting)", np.sign(np.cos(2 * np.pi * m0 * t / TW)))]:
        h = hp(s, [m0, 2 * m0, 3 * m0, 5 * m0, 7 * m0])
        print("     %-26s 2w %.4f  3w %.4f  5w %.4f  7w %.4f"
              % (lbl, h[2*m0]/h[m0], h[3*m0]/h[m0], h[5*m0]/h[m0], h[7*m0]/h[m0]))
    for a in (1.0, 1.3, 2.0, 3.0):
        s = np.clip(1 + a * np.cos(2 * np.pi * m0 * t / TW), 0, None); s = s - s.mean()
        h = hp(s, [m0, 2 * m0, 3 * m0])
        print("     clipped sine amp=%.1f (clip %.0f%% of weeks)  2w %.4f  3w %.4f"
              % (a, 100 * np.mean(1 + a * np.cos(2 * np.pi * m0 * t / TW) < 0), h[2*m0]/h[m0], h[3*m0]/h[m0]))
    c = channels(1)[0]
    for amp in (0.05, 0.10, 0.20, 0.40):
        w = 2 * np.pi * m0 / TW
        a_ = adstock_periodic(c.xbar * (1 + amp * np.cos(w * t)), c.alpha)
        r = c.beta * hill(a_, c.K, c.S); h = hp(r - r.mean(), [m0, 2 * m0, 3 * m0])
        print("     saturation response amp=%3.0f%%          2w %.4f  3w %.4f"
              % (100 * amp, h[2*m0]/h[m0], h[3*m0]/h[m0]))
    print("   (b) cost of a square-wave odd-harmonic collision (se_B ratio over phase)")
    chs = channels(2); Z = nuisance(TW); gamma = np.zeros(Z.shape[1]); gamma[0] = 1.0
    sq = lambda mA, mB, ph: np.array([chs[0].xbar * (1 + .1 * np.sign(np.cos(2*np.pi*mA*t/TW))),
                                      chs[1].xbar * (1 + .1 * np.cos(2*np.pi*mB*t/TW + ph))])
    base = se_mroas(chs, sq(13, 17, 0), Z, gamma)
    for mB, lbl in [(39, "3x (square 3rd harmonic)"), (65, "5x"), (26, "2x (square has none)")]:
        r = [se_mroas(chs, sq(13, mB, p), Z, gamma)[1] / base[1] for p in np.linspace(0, np.pi, 25)]
        print("     mB=%3d %-26s min %.2f  median %.2f  max %.2f" % (mB, lbl, min(r), np.median(r), max(r)))
    print("   (c) record truncation (probes go off-grid)")
    chs6 = channels()
    for Tobs in (364, 360, 352, 338, 312, 273):
        dev = []
        for c, mi in zip(chs6, MIS_COMB):
            x = c.xbar * (1 + .1 * np.cos(2 * np.pi * mi / 364 * np.arange(364)))
            a = adstock_periodic(x, c.alpha)[:Tobs]; dev.append(a - a.mean())
        Mt = resid_maker(nuisance(Tobs))
        X = np.column_stack([Mt(v) for v in dev]); G = X.T @ X
        d = np.sqrt(np.diag(G)); R = G / np.outer(d, d)
        print("     T_obs=%3d (%3.0f%%)  max|off| %.4f  maxVIF %.4f"
              % (Tobs, 100 * Tobs / 364, np.abs(R - np.eye(6)).max(), np.diag(np.linalg.inv(R)).max()))
    print("   (d) alpha misspecified when the quadrature phase is designed")
    chs2 = channels(2); b0 = se_mroas(chs2, Design(chs2, [19, 23], [.1, .1], [0, 0]).spend(), Z, gamma)[1]
    for da in (0.0, 0.05, 0.10, 0.20, -0.10):
        ca = [Channel(None, 1.5, 2.0, .6 + da, MARGIN, Hstar=.68),
              Channel(None, 2.0, 1.6, .3 + da, MARGIN, Hstar=.55)]
        ph = np.linspace(0, np.pi, 49)
        p = ph[int(np.argmin([se_mroas(ca, Design(ca, [19, 38], [.1, .1], [0, q]).spend(),
                                       Z, np.zeros(Z.shape[1]))[1] for q in ph]))]
        r = se_mroas(chs2, Design(chs2, [19, 38], [.1, .1], [0, p]).spend(), Z, gamma)[1] / b0
        print("     alpha error %+0.2f -> designed phi %.3f, realised se ratio %.3f" % (da, p, r))


def S10():
    print("S10 — comb optimality: simultaneous frontier attainment and exact additivity")
    chs = channels(); Z = nuisance(TW); M = resid_maker(Z); t = np.arange(TW)
    V, C = [], []
    for j, c in enumerate(chs):
        w = 2 * np.pi * MIS_COMB[j] / TW
        a = adstock_periodic(c.xbar * (1 + .1 * np.cos(w * t)), c.alpha)
        V.append(M(a - a.mean())); C.append(0.5 * c.kappa * np.var(a) * TW)
    Gi = np.linalg.inv(np.column_stack(V).T @ np.column_stack(V))
    print("   ch   C_j     Var(c1hat)    C_j*Var / (kappa_j sigma^2/2)")
    for j, c in enumerate(chs):
        v = SIGMA**2 * Gi[j, j]
        print("   %s  %.4f  %.6e   %.6f" % (c.name, C[j], v, C[j] * v / (0.5 * c.kappa * SIGMA**2)))
    gamma = np.zeros(Z.shape[1]); gamma[0] = 1.0
    sj = se_mroas(chs, Design(chs, MIS_COMB, [.1] * 6, [0] * 6).spend(), Z, gamma)
    print("   additivity: se in the 6-channel comb vs the channel probed alone")
    for j, c in enumerate(chs):
        solo = se_mroas([c], Design([c], [MIS_COMB[j]], [.1], [0]).spend(), Z, gamma)[0]
        print("     %s  joint %.4f   solo %.4f   ratio %.6f" % (c.name, sj[j], solo, sj[j] / solo))


def S11():
    print("S11 — negative / power controls")
    chs = channels(); Z = nuisance(TW); gamma = np.zeros(Z.shape[1]); gamma[0] = 1.0
    for lbl, mis in [("compliant comb", MIS_COMB),
                     ("one channel on the 2nd annual harmonic (bin 14)", [19, 14, 25, 29, 31, 37]),
                     ("two channels sharing bin 25", [19, 23, 25, 25, 31, 37]),
                     ("all six on a common 13-wk flight (bin 28)", [28] * 6)]:
        se = se_mroas(chs, Design(chs, mis, [.1] * 6, [0] * 6).spend(), Z, gamma)
        print("   %-46s se = %s" % (lbl, np.array2string(se, precision=3, max_line_width=200)))
    print("   NOTE: the last two rows are pinv artefacts on a SINGULAR information matrix — see S14.")


def S12():
    print("S12 — the harmonic readout: model-free c2, c3 and the 3w/2w shape test")
    t = np.arange(TW); m0 = 19; w = 2 * np.pi * m0 / TW
    line = lambda y, m: 2 * np.vdot(np.exp(1j * 2 * np.pi * m * t / TW), y) / TW
    print("   ch  amp   B        c2 true   c2 readout   c3 true   c3 readout")
    for nm, K, S, al, Hs in [('A', 1.5, 2.0, .6, .68), ('C', 1.0, 2.4, .75, .75), ('E', .8, 3.0, .2, .70)]:
        c = Channel(None, K, S, al, MARGIN, name=nm, Hstar=Hs)
        for amp in (0.10, 0.20):
            B = amp * c.xbar * abs(Hf(c.alpha, w))
            y = c.beta * hill(adstock_periodic(c.xbar * (1 + amp * np.cos(w * t)), c.alpha), c.K, c.S)
            y = y - y.mean()
            print("   %s   %.2f  %.4f   %+.4f    %+.4f    %+.4f   %+.4f"
                  % (nm, amp, B, c.c2, -4 * abs(line(y, 2 * m0)) / B**2,
                     c.c3, 24 * abs(line(y, 3 * m0)) / B**3))
    print("   shape test: R = |line(3w)|/|line(2w)| = (|c3/c2|) B/6")
    for nm, K, S, al, Hs in [('A', 1.5, 2.0, .6, .68), ('C', 1.0, 2.4, .75, .75),
                             ('E', .8, 3.0, .2, .70), ('F', 2.5, 2.0, .55, .50)]:
        c = Channel(None, K, S, al, MARGIN, name=nm, Hstar=Hs)
        B = .1 * c.xbar * abs(Hf(c.alpha, w))
        y = c.beta * hill(adstock_periodic(c.xbar * (1 + .1 * np.cos(w * t)), c.alpha), c.K, c.S)
        y = y - y.mean(); R = abs(line(y, 3 * m0)) / abs(line(y, 2 * m0))
        print("     %s (K=%.1f,S=%.1f,H*=%.2f)  R=%.5f  implied |c3/c2| %.4f  true %.4f"
              % (nm, K, S, Hs, R, 6 * R / B, abs(c.c3 / c.c2)))
    print("     (F sits at the Hill inflection, c3 = 0 exactly — the shape test's own null control)")


def S13():
    print("S13 — readout power: se(c2hat) = 4 sigma sqrt(2/T)/B^2")
    c = channels(1)[0]; m0 = 19; w = 2 * np.pi * m0 / TW; t = np.arange(TW)
    Z = nuisance(TW); M = resid_maker(Z); psi = np.angle(Hf(c.alpha, w))
    print("   amp    B      SNR(2w)   se theory   se MC(20k)          modulus mean   projection mean")
    for amp in (0.10, 0.20, 0.30, 0.40):
        B = amp * c.xbar * abs(Hf(c.alpha, w))
        mu = c.beta * hill(adstock_periodic(c.xbar * (1 + amp * np.cos(w * t)), c.alpha), c.K, c.S)
        se_th = 4 * SIGMA * np.sqrt(2 / TW) / B**2
        snr = abs(c.c2) * B**2 / 4 / (SIGMA * np.sqrt(2 / TW))
        rng = np.random.default_rng(5); N = 20000
        Y = M((mu[None, :] + rng.standard_normal((N, TW)) * SIGMA).T).T
        z = 2 * (Y @ np.exp(-1j * 2 * np.pi * 2 * m0 * t / TW)) / TW
        proj = 4 * np.real(z * np.exp(-1j * 2 * psi)) / B**2
        print("   %.2f  %.4f  %7.2f   %.5f    %.5f+-%.5f   %+.4f       %+.4f"
              % (amp, B, snr, se_th, proj.std(), proj.std() / np.sqrt(2 * N),
                 (-4 * np.abs(z) / B**2).mean(), proj.mean()))
    Bs = np.sqrt(8 * SIGMA * np.sqrt(2 / TW) / abs(c.c2))
    print("   true c2 = %.4f;  B* for z=2 is %.4f = %.1f%% of mean spend"
          % (c.c2, Bs, 100 * Bs / (c.xbar * abs(Hf(c.alpha, w)))))


def S14():
    print("S14 — rank of the synchronised design (full Hill vs calibrated shape)")
    chs = channels(); Z = nuisance(TW); gamma = np.zeros(Z.shape[1]); gamma[0] = 1.0
    th = theta_true(chs, gamma)
    for lbl, mis in [("synchronised bin 28 (13 wk)", [28] * 6), ("comb", MIS_COMB)]:
        X = Design(chs, mis, [.1] * 6, [0] * 6).spend(); J = jac(th, chs, X, Z)
        for tag, free in [("full Hill (4/ch)", 'full'), ("calibrated shape (beta,alpha)", 'local')]:
            keep = keepset(Z, 6, free); Jk = J[:, keep]
            u, s, vt = np.linalg.svd(Jk); r = int((s > s[0] * 1e-10).sum())
            leaks = [np.linalg.norm(vt[r:] @ mroas_grad(th, chs, Z, j, X[j].mean())[keep])
                     / np.linalg.norm(mroas_grad(th, chs, Z, j, X[j].mean())[keep])
                     if r < len(keep) else 0.0 for j in range(6)]
            print("   %-28s %-30s rank %2d/%2d  smin/smax %.2e  max ||P_null g||/||g|| %.3f"
                  % (lbl, tag, r, len(keep), s[-1] / s[0], max(leaks)))
    print("   T/28 = 13, so every column of a bin-28 design is exactly 13-periodic.")


def S15(nrep=150):
    print("S15 — full nonlinear Monte Carlo, n = 6, calibrated-shape MMM (beta, alpha free)")
    chs = channels(); Z = nuisance(TW); nz = Z.shape[1]; t = np.arange(TW)
    gamma = np.zeros(nz); gamma[0] = 1.0
    true = np.array([c.mROAS for c in chs])
    spend = lambda mis, amps, phis: np.array(
        [c.xbar * (1 + amps[j] * np.cos(2 * np.pi * mis[j] / TW * t + phis[j])) for j, c in enumerate(chs)])
    cost = lambda X: sum(0.5 * c.kappa * np.var(adstock_periodic(X[j], c.alpha)) * TW
                         for j, c in enumerate(chs))
    def mu_of(X, p):
        mu = Z @ p[:nz]
        for j, c in enumerate(chs):
            mu = mu + np.exp(p[nz + 2*j]) * hill(adstock_periodic(X[j], 1/(1+np.exp(-p[nz+2*j+1]))),
                                                 c.K, c.S)
        return mu
    p0 = np.array(list(gamma) + [v for c in chs for v in (np.log(c.beta), np.log(c.alpha/(1-c.alpha)))])
    def mroas_of(p, X):
        return np.array([c.m * np.exp(p[nz+2*j]) *
                         hill_d(X[j].mean()/(1-1/(1+np.exp(-p[nz+2*j+1]))), c.K, c.S) /
                         (1 - 1/(1+np.exp(-p[nz+2*j+1]))) for j, c in enumerate(chs)])
    designs = {'comb (rule-compliant)': (MIS_COMB, [.1]*6, [0.]*6),
               'synchronised 13-wk flight': ([28]*6, [.1]*6, [0.]*6),
               'comb w/ 2 octave collisions': ([19, 38, 25, 50, 31, 37], [.1]*6, [0.]*6),
               'comb, random phases': (MIS_COMB, [.1]*6,
                                       list(np.random.default_rng(3).uniform(0, 2*np.pi, 6)))}
    cref = cost(spend(*designs['comb (rule-compliant)']))
    rng = np.random.default_rng(11); res = {}
    for k, (mis, amps, phis) in designs.items():
        s = np.sqrt(cref / cost(spend(mis, amps, phis)))
        X = spend(mis, [a * s for a in amps], phis); mu = mu_of(X, p0); est = []
        for _ in range(nrep):
            y = mu + SIGMA * rng.standard_normal(TW)
            r = least_squares(lambda p: mu_of(X, p) - y, p0, method='lm', xtol=1e-10, ftol=1e-10)
            est.append(mroas_of(r.x, X))
        est = np.array(est); rmse = np.sqrt(((est - true)**2).mean(0)); res[k] = rmse
        print("   %-30s per-channel RMSE %s  median %.4f"
              % (k, np.array2string(rmse, precision=3, max_line_width=200), np.median(rmse)))
    base = res['comb (rule-compliant)']
    print("   RMSE ratio to the compliant comb (MC se ~%.0f%%):" % (100 / np.sqrt(2 * nrep)))
    for k, v in res.items():
        print("     %-30s median %6.2fx  max %6.2fx" % (k, np.median(v / base), (v / base).max()))


def S16():
    print("S16 — the n-channel price of identification and channel-scale invariance")
    chs = channels(); Tr = 260; S_tot = sum(c.xbar for c in chs) * Tr
    for Psi, lbl in [(1.0, "fully calibrated"), (9.87, "full Hill free")]:
        L = MARGIN * SIGMA * np.sqrt(Psi * Tr)
        print("   Psi=%.2f (%s): L* per channel = %.4f, independent of channel size" % (Psi, lbl, L))
        for n in (1, 6, 12, 20):
            print("     %2d channels: total %8.3f = %6.3f%% of media spend over the horizon"
                  % (n, n * L, 100 * n * L / S_tot))
    Z = nuisance(TW); M = resid_maker(Z); t = np.arange(TW); w = 2 * np.pi * 19 / TW
    print("   channel-scale invariance (scale K and xbar by s):")
    for s in (0.25, 0.5, 1.0, 2.0, 4.0):
        c = Channel(None, 1.5 * s, 2.0, 0.60, MARGIN, Hstar=0.68)
        a = adstock_periodic(c.xbar * (1 + .1 * np.cos(w * t)), c.alpha); v = M(a - a.mean())
        C = 0.5 * c.kappa * np.var(a) * TW
        print("     s=%.2f  kappa %.5f  C %.4f  C*Var/(kappa sigma^2/2) = %.6f"
              % (s, c.kappa, C, C * SIGMA**2 / (v @ v) / (0.5 * c.kappa * SIGMA**2)))


ALL = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16]

if __name__ == "__main__":
    want = sys.argv[1:]
    for f in ALL:
        if want and f.__name__ not in want:
            continue
        print("=" * 100)
        f()
        print()
