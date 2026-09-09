"""
Dive 14 — EXTEND of dive 13: the experimental route to causal Shapley attribution.

Objects.  n channels; a "cell" exposes the channel set A (bitmask) to a random user slice and suppresses the rest;
its mean outcome is v(A) = v_pop(A) (dive 13).  Harsanyi dividends D(T): v(A) = Σ_{T⊆A} D(T).  Interaction cap k:
D(T)=0 for |T|>k (Grabisch k-additivity).  Causal Shapley: Sh_j = Σ_{T∋j} D(T)/|T|.
Two-layer design B(m_b, m_t) = {A : |A| ≤ m_b  or  |N∖A| ≤ m_t}  (m_b = "expose at most" depth, m_t = "hold out at
most" depth).  Dive 13's layered design L_m = B(m, m).

Theorem (this dive; proved in the writeup via the Stab(j)-symmetrised row space):
    Sh identified from B(m_b, m_t) under cap k   iff   m_b + m_t ≥ min(k, n−1).
Closed-form estimator (symmetrised): with σ_t := Σ_{T∋j,|T|=t} D(T) and Q(H) := Σ_{T⊇H} D(T)
    = Σ_{B⊆H} (−1)^{|H∖B|} v((N∖H)∪B),   R_h := Σ_{H∋j,|H|=h} Q(H) = Σ_t C(t−1,h−1) σ_t,
    Sh_j = Σ_{t≤m_b} σ_t/t + Σ_{h=1}^{r} λ_h [R_h − Σ_{t≤m_b} C(t−1,h−1) σ_t],  r = k−m_b,
    λ solving Σ_h λ_h C(t−1,h−1) = 1/t for t = m_b+1..k  (an r×r binomial-Vandermonde system).
Aliasing law: an order-(k+1) dividend T∋j leaks into Ŝh_j with coefficient p(k+1)−1/(k+1) where p is the
degree-(r−1) polynomial interpolating 1/t at t=m_b+1..k, i.e. leak = (−1)^{r+1} m_b! r!/(k+1)!, so
|leak| = 1/((k+1)·C(k, m_b)) — minimised by the balanced design m_b = ⌊k/2⌋; leak into channels j∉T is 0.
Over-identification: every exactly-capped B(m_b,m_t) (m_b+m_t=k≤n−2) carries exactly ONE restriction — the closed-form
credits add up to the global-holdout lift (E14); the BLUE exploits it (2.3× variance gain at k=2) but spreads
higher-order aliasing onto innocent channels (E15).  Lift price of precision: N·ℓ·trV → κ = min_C (Σ_{A≠N}
||C_A|| s_A √ℓ_A)² as the lift forgone ℓ → 0 (E13).

Run `python 14_attribution_impossibility_ext.py quick` (≈1 min: e1,e2,e3,e4,e10,e14,e15) or `e1` … `e15`
(e5/e13 ≈ 3–5 min each; e6 ≈ 2 min; e7 ≈ 2 min).
"""
import sys, itertools, time
from math import comb, factorial
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit
from scipy.optimize import minimize, brentq
from scipy.stats import chi2, norm

rng_global = np.random.default_rng(0)

# ----------------------------------------------------------------------------- set algebra
def popcount(m): return bin(m).count("1")
def members(m, n): return [j for j in range(n) if (m >> j) & 1]

def mobius(v, n):
    d = np.array(v, dtype=float).copy()
    for j in range(n):
        for m in range(1 << n):
            if (m >> j) & 1:
                d[m] -= d[m ^ (1 << j)]
    return d

def zeta(d, n):
    v = np.array(d, dtype=float).copy()
    for j in range(n):
        for m in range(1 << n):
            if (m >> j) & 1:
                v[m] += v[m ^ (1 << j)]
    return v

def shapley_from_v(v, n):
    d = mobius(v, n)
    out = np.zeros(n)
    for m in range(1, 1 << n):
        t = popcount(m)
        for j in members(m, n):
            out[j] += d[m] / t
    return out

def sh_weight_matrix(n, kcap=None):
    """W[j, T] = 1[j∈T]/|T| over dividend index T (all 2^n; columns |T|>kcap dropped if kcap given)."""
    cols = [T for T in range(1, 1 << n) if kcap is None or popcount(T) <= kcap]
    W = np.zeros((n, len(cols)))
    for c, T in enumerate(cols):
        for j in members(T, n):
            W[j, c] = 1.0 / popcount(T)
    return W, cols

def incidence(cells, cols):
    """X[A, T] = 1[T ⊆ A]."""
    return np.array([[1.0 if (T & ~A) == 0 else 0.0 for T in cols] for A in cells])

# ----------------------------------------------------------------------------- designs
def design_B(n, mb, mt):
    full = (1 << n) - 1
    return sorted({A for A in range(1 << n) if popcount(A) <= mb or popcount(full & ~A) <= mt})

def design_layered(n, m): return design_B(n, m, m)
def design_ghost(n): return design_B(n, 0, 1)          # global holdout + single holdouts + no-holdout
def design_full(n): return list(range(1 << n))

def in_rowspace(X, w, tol=1e-9):
    """Is w (over dividend columns) in the row space of X?  Return (bool, residual)."""
    c, res, rk, sv = np.linalg.lstsq(X.T, w, rcond=None)
    r = np.linalg.norm(X.T @ c - w)
    return r < tol, r

def identified(n, cells, k):
    W, cols = sh_weight_matrix(n, k)
    cols = [T for T in cols]  # include ∅ column for intercept
    X = incidence(cells, [0] + cols)
    Wfull = np.hstack([np.zeros((n, 1)), W])
    return all(in_rowspace(X, Wfull[j])[0] for j in range(n))

# ----------------------------------------------------------------------------- estimators (cell-weight vectors)
def blue_weights(n, cells, k, var=None):
    """Minimum-variance cell weights C (n × #cells) with Cᵀ-constraint X_kᵀ c_j = w_j (unbiased under cap k).
    var: per-cell variance of the cell mean (σ_A²/N_A); default equal."""
    W, cols = sh_weight_matrix(n, k)
    cols = [0] + cols
    X = incidence(cells, cols)
    Wf = np.hstack([np.zeros((n, 1)), W])
    S = np.ones(len(cells)) if var is None else np.asarray(var, float)
    Si = 1.0 / S
    M = X.T @ (Si[:, None] * X)
    Mp = np.linalg.pinv(M, rcond=1e-12)
    C = (Si[:, None] * X) @ Mp @ Wf.T            # (#cells × n)
    # check unbiasedness
    err = np.abs(X.T @ C - Wf.T).max()
    return C.T, err

def closed_form_weights(n, mb, mt, k):
    """Symmetrised closed-form estimator of Sh_j on B(mb, mt) under cap k (exactly-determined case k−mb ≤ mt).
    Returns C (n × #cells) aligned with design_B(n, mb, mt)."""
    cells = design_B(n, mb, mt)
    idx = {A: i for i, A in enumerate(cells)}
    full = (1 << n) - 1
    r = k - mb
    C = np.zeros((n, len(cells)))
    if r > mt:
        raise ValueError("not identified: k - mb > mt")
    # λ: Σ_{h=1..r} λ_h C(t−1,h−1) = 1/t  for t = mb+1..k   (use h = 1..r)
    if r > 0:
        Mat = np.array([[comb(t - 1, h - 1) for h in range(1, r + 1)] for t in range(mb + 1, k + 1)], float)
        lam = np.linalg.solve(Mat, np.array([1.0 / t for t in range(mb + 1, k + 1)]))
    else:
        lam = np.zeros(0)
    for j in range(n):
        # σ_t for t ≤ mb via bottom cells: D(T) = Σ_{B⊆T} (−1)^{|T−B|} v(B)
        coef_sigma = {}   # accumulate weight on σ_t as a function of cells
        for t in range(1, mb + 1):
            wt = 1.0 / t - sum(lam[h - 1] * comb(t - 1, h - 1) for h in range(1, r + 1))
            if wt == 0: continue
            for T in range(1 << n):
                if popcount(T) == t and (T >> j) & 1:
                    B = T
                    while True:
                        C[j, idx[B]] += wt * (-1) ** (t - popcount(B))
                        if B == 0: break
                        B = (B - 1) & T
        # R_h = Σ_{H∋j,|H|=h} Q(H), Q(H) = Σ_{B⊆H} (−1)^{|H−B|} v((N∖H)∪B)
        for h in range(1, r + 1):
            for H in range(1 << n):
                if popcount(H) == h and (H >> j) & 1:
                    B = H
                    while True:
                        A = (full & ~H) | B
                        C[j, idx[A]] += lam[h - 1] * (-1) ** (h - popcount(B))
                        if B == 0: break
                        B = (B - 1) & H
    return C, cells

def leak_matrix(n, cells, C, order):
    """Response of estimator C (n × cells) to a unit dividend on each T of size `order`, minus the true Shapley
    weight 1[j∈T]/|T|.  Returns dict T -> leak vector (n,)."""
    out = {}
    for T in range(1 << n):
        if popcount(T) == order:
            x = np.array([1.0 if (T & ~A) == 0 else 0.0 for A in cells])
            truth = np.array([1.0 / order if (T >> j) & 1 else 0.0 for j in range(n)])
            out[T] = C @ x - truth
    return out

# ----------------------------------------------------------------------------- random games
def random_game(n, k, rng, scale=None, decay=0.6):
    """Random dividends with |T| ≤ k, magnitude decaying with order."""
    d = np.zeros(1 << n)
    for T in range(1, 1 << n):
        t = popcount(T)
        if t <= k:
            s = (scale[t] if scale is not None else decay ** (t - 1))
            d[T] = rng.normal(0, s)
    return zeta(d, n), d

# ----------------------------------------------------------------------------- DGP for finite-user simulations
class DGP:
    """Dive-13 style: U~N(0,1); Z_j|U ~ Bern(σ(a_j+g_j U)); Y(z)|U ~ Bern(σ(β0+βU+θ·z+½ zᵀΘ2 z + θ3·z1z2z3))."""
    def __init__(self, a, g, beta0, beta, theta, theta2=None, theta3=0.0):
        self.a, self.g = np.asarray(a, float), np.asarray(g, float); self.n = len(a)
        self.beta0, self.beta, self.theta = beta0, beta, np.asarray(theta, float)
        self.theta2 = np.zeros((self.n, self.n)) if theta2 is None else np.asarray(theta2, float)
        self.theta3 = theta3
        x, w = hermegauss(120); self.u, self.w = x, w / w.sum()
        self.N = 1 << self.n
        self._cache()

    def lin(self, z, u):
        out = self.beta0 + self.beta * u + z @ self.theta + 0.5 * z @ self.theta2 @ z
        if self.n >= 3 and self.theta3:
            out = out + self.theta3 * z[0] * z[1] * z[2]
        return out

    def _cache(self):
        n = self.n
        E = expit(self.a[None, :] + self.g[None, :] * self.u[:, None])
        PZ = np.ones((len(self.u), self.N))
        for S in range(self.N):
            for j in range(n):
                PZ[:, S] *= E[:, j] if (S >> j) & 1 else (1 - E[:, j])
        self.PZu, self.E = PZ, E
        Pu = np.zeros((len(self.u), self.N))
        for A in range(self.N):
            z = np.array([(A >> j) & 1 for j in range(n)], float)
            Pu[:, A] = expit(self.lin(z, self.u))
        self.Pu = Pu
        # v_pop(A) = E[ p(Z ∧ 1_A, U) ]
        self.v_pop = np.array([self.w @ np.sum(PZ * Pu[:, [S & A for S in range(self.N)]], axis=1)
                               for A in range(self.N)])
        self.sh = shapley_from_v(self.v_pop, n)
        self.div = mobius(self.v_pop, n)

    def simulate_cell(self, A, N, rng):
        """Mean outcome of N users randomised to cell A (channels outside A suppressed)."""
        u = rng.standard_normal(N)
        E = expit(self.a[None, :] + self.g[None, :] * u[:, None])
        Z = (rng.random((N, self.n)) < E).astype(float)
        mask = np.array([(A >> j) & 1 for j in range(self.n)], float)
        z = Z * mask[None, :]
        lin = self.beta0 + self.beta * u + z @ self.theta + 0.5 * np.einsum("ij,jk,ik->i", z, self.theta2, z)
        if self.n >= 3 and self.theta3:
            lin = lin + self.theta3 * z[:, 0] * z[:, 1] * z[:, 2]
        y = rng.random(N) < expit(lin)
        return y.mean()

def benchmark3():
    return DGP(a=(-0.5, -1.0, -1.5), g=(0.4, 1.0, 1.6), beta0=-2.5, beta=1.0, theta=(0.5, 0.5, 0.0))

def dgp5(theta2_scale=0.0, theta3=0.0):
    n = 5
    a = np.array([-0.5, -0.8, -1.0, -1.2, -1.5]); g = np.array([0.4, 0.7, 1.0, 1.3, 1.6])
    theta = np.array([0.5, 0.4, 0.3, 0.2, 0.0])
    th2 = np.zeros((n, n)); th2[0, 1] = th2[1, 0] = 0.6 * theta2_scale; th2[1, 2] = th2[2, 1] = -0.4 * theta2_scale
    th2[0, 3] = th2[3, 0] = 0.3 * theta2_scale
    return DGP(a, g, -2.5, 1.0, theta, th2, theta3)

# ----------------------------------------------------------------------------- precision / lift / frontier
def cell_stats(v, cells):
    """Bernoulli variances p(1-p) and lift forgone per user (v(N) − v(A)) for each cell."""
    full = len(v) - 1
    p = np.array([v[A] for A in cells])
    return p * (1 - p), np.array([v[full] - v[A] for A in cells]), v[full] - v[0]

def frontier_fixed_design(n, cells, k, v, mus, criterion="trace"):
    """For fixed design & BLUE-at-each-allocation weights, trace the min Var vs lift-forgone frontier over
    allocation shares f (convex program): min_f tr Var_BLUE(f) + μ·lift(f).  Returns list of (lift_frac, var_trace, f)."""
    s2, lf, total = cell_stats(v, cells)
    m = len(cells)
    W, cols = sh_weight_matrix(n, k); cols = [0] + cols
    X = incidence(cells, cols); Wf = np.hstack([np.zeros((n, 1)), W])
    def objective(logf, mu):
        f = np.exp(logf); f = f / f.sum()
        M = X.T @ ((f / s2)[:, None] * X)
        Mp = np.linalg.pinv(M, rcond=1e-12)
        V = Wf @ Mp @ Wf.T
        val = np.trace(V) if criterion == "trace" else np.max(np.diag(V))
        return val + mu * (f @ lf) / total
    out = []
    x0 = np.zeros(m)
    for mu in mus:
        best = None
        for trial in range(2):
            xi = x0 if trial == 0 else x0 + rng_global.normal(0, 0.5, m)
            r = minimize(objective, xi, args=(mu,), method="L-BFGS-B", options={"maxiter": 2000})
            if best is None or r.fun < best.fun: best = r
        f = np.exp(best.x); f = f / f.sum(); x0 = best.x
        M = X.T @ ((f / s2)[:, None] * X); V = Wf @ np.linalg.pinv(M, rcond=1e-12) @ Wf.T
        out.append(((f @ lf) / total, np.trace(V), np.max(np.diag(V)), f))
    return out

def neyman_alloc(C, s2):
    """Allocation minimising Σ_A (Σ_j C_jA²) s2_A / f_A  → f_A ∝ sqrt(a_A)."""
    a = (C ** 2).sum(0) * s2
    f = np.sqrt(a); return f / f.sum()

# ----------------------------------------------------------------------------- lack-of-fit test of the cap
def cap_test(vhat, cells, n, k, Nc):
    """Weighted lack-of-fit χ² of the k-additive model to the cell means (binomial variances plugged in).
    Returns (stat, df, pvalue)."""
    W, cols = sh_weight_matrix(n, k); cols = [0] + cols
    X = incidence(cells, cols)
    s2 = np.maximum(vhat * (1 - vhat), 1e-9) / Nc
    Wt = 1 / s2
    M = X.T @ (Wt[:, None] * X)
    D = np.linalg.pinv(M, rcond=1e-12) @ (X.T @ (Wt * vhat))
    res = vhat - X @ D
    stat = float(res @ (Wt * res))
    df = len(cells) - np.linalg.matrix_rank(X)
    return stat, df, 1 - chi2.cdf(stat, df) if df > 0 else np.nan

# ============================================================================= experiments
def e1():
    """Theorem check: identification of Sh from B(mb, mt) under cap k, all n ≤ 8."""
    print("E1: identification of Sh from B(mb,mt) under cap k — theorem: iff mb+mt ≥ min(k, n−1)")
    bad = 0; tot = 0
    for n in range(3, 9):
        for k in range(1, n + 1):
            for mb in range(0, n):
                for mt in range(0, n):
                    if mb + mt > n + 1: continue
                    cells = design_B(n, mb, mt)
                    idn = identified(n, cells, k)
                    pred = (mb + mt) >= min(k, n - 1)
                    tot += 1
                    if idn != pred:
                        bad += 1; print(f"  MISMATCH n={n} k={k} mb={mb} mt={mt}: rank says {idn}, theorem {pred}")
    print(f"  {tot} (n,k,mb,mt) cases, {bad} mismatches")
    # explicit necessity witness at n=4, k=3, L1: game with all L1 cells zero and Sh ≠ 0
    n = 4; d = np.zeros(16)
    pairs = {(0, 1): 1, (0, 2): -1, (0, 3): 1, (1, 2): 0, (1, 3): -2, (2, 3): 1}
    for (i, j), val in pairs.items(): d[(1 << i) | (1 << j)] = val
    trip = {(0, 1, 2): 0, (0, 1, 3): 0, (0, 2, 3): -1, (1, 2, 3): 1}
    for (i, j, l), val in trip.items(): d[(1 << i) | (1 << j) | (1 << l)] = val
    v = zeta(d, n)
    cells = design_layered(4, 1)
    print("  witness n=4,k=3,L1: cell values", np.round([v[A] for A in cells], 12), " Sh =", np.round(shapley_from_v(v, 4), 6))

def e2():
    """Closed-form symmetrised estimator = Shapley exactly on random capped games; holdout-only formula."""
    print("E2: closed-form estimator exactness on random k-additive games")
    rng = np.random.default_rng(1)
    for n, mb, mt, k in [(3, 1, 1, 2), (4, 0, 2, 2), (4, 2, 0, 2), (5, 1, 2, 3), (5, 2, 1, 3), (6, 0, 3, 3),
                         (6, 2, 2, 4), (7, 1, 3, 4), (8, 3, 2, 5)]:
        C, cells = closed_form_weights(n, mb, mt, k)
        errs = []
        for _ in range(20):
            v, d = random_game(n, k, rng)
            est = C @ np.array([v[A] for A in cells])
            errs.append(np.abs(est - shapley_from_v(v, n)).max())
        print(f"  n={n} B({mb},{mt}) k={k}: cells={len(cells):3d}, max|err|={max(errs):.1e}, |weight| row-sum max={np.abs(C).sum(1).max():.2f}")
    # holdout-only k=2 formula spelled out at n=4
    n = 4; C, cells = closed_form_weights(n, 0, 2, 2)
    print("  holdout-only (B(0,2), k=2) weights for channel 0 by cell (exposed set):")
    for A, c in zip(cells, C[0]):
        if abs(c) > 1e-12: print(f"    hold out {sorted(set(range(n)) - set(members(A, n)))}: {c:+.3f}")

def e3():
    """Aliasing law: leak of order-(k+1) dividends, closed form vs computed; BLUE vs closed-form."""
    print("E3: aliasing coefficients of order k+1 dividends, closed form |leak| = 1/((k+1)·C(k,mb)), sign (−1)^(k−mb+1)")
    for n, k in [(4, 2), (5, 3), (6, 4), (7, 5)]:
        for mb in range(0, k + 1):
            mt = k - mb
            if mb + mt > n - 1: continue
            C, cells = closed_form_weights(n, mb, mt, k)
            L = leak_matrix(n, cells, C, k + 1)
            ins = []; outs = []
            for T, lv in L.items():
                for j in range(n):
                    (ins if (T >> j) & 1 else outs).append(lv[j])
            pred = (-1) ** (k - mb + 1) * factorial(mb) * factorial(k - mb) / factorial(k + 1)
            print(f"  n={n} k={k} B({mb},{mt}): leak(j∈T) = {np.mean(ins):+.5f} (sd {np.std(ins):.1e}), pred {pred:+.5f}; "
                  f"leak(j∉T) max|.|={np.max(np.abs(outs)) if outs else 0:.1e}; #cells={len(cells)}")
    # higher-order leak: order k+2
    n, k = 6, 3
    for mb in range(0, k + 1):
        C, cells = closed_form_weights(n, mb, k - mb, k)
        for order in [k + 1, k + 2, k + 3]:
            L = leak_matrix(n, cells, C, order)
            ins = [lv[j] for T, lv in L.items() for j in range(n) if (T >> j) & 1]
            # prediction: p(order) − 1/order with p interpolating 1/t at t=mb+1..k
            ts = np.arange(mb + 1, k + 1)
            pred = -1.0 / order if len(ts) == 0 else np.polyval(np.polyfit(ts, 1 / ts, len(ts) - 1), order) - 1 / order
            print(f"    n=6 k=3 B({mb},{3-mb}) order {order}: leak {np.mean(ins):+.5f}, interpolation-error pred {pred:+.5f}")

def e4():
    """BLUE on over-identified designs vs closed form: variance and aliasing."""
    print("E4: BLUE (min-variance unbiased under cap) vs closed form; over-identified designs")
    for n, mb, mt, k in [(4, 1, 1, 2), (4, 1, 2, 2), (4, 2, 2, 2), (5, 2, 2, 3), (5, 1, 3, 3), (6, 2, 2, 3)]:
        cells = design_B(n, mb, mt)
        Cb, err = blue_weights(n, cells, k)
        try:
            Cc, _ = closed_form_weights(n, mb, mt, k); vc = (Cc ** 2).sum(1)
        except ValueError:
            vc = np.full(n, np.nan)
        vb = (Cb ** 2).sum(1)
        Lb = leak_matrix(n, cells, Cb, k + 1)
        lk = np.array([lv[j] for T, lv in Lb.items() for j in range(n) if (T >> j) & 1])
        print(f"  n={n} B({mb},{mt}) k={k} cells={len(cells)}: Σc² BLUE={vb.mean():.3f} closed={np.nanmean(vc):.3f} "
              f"(unbias err {err:.0e}); BLUE leak(k+1) mean {lk.mean():+.4f} sd {lk.std():.4f}")

def e5():
    """Precision–lift frontier: structured designs (with Neyman allocation) vs the global convex optimum over all 2^n cells."""
    print("E5: precision (tr Var per user) vs lift-forgone frontier, n=4 and n=5, k=2 and k=3, dgp5 truth")
    for n, k in [(4, 2), (5, 2), (5, 3)]:
        dg = dgp5(theta2_scale=1.0) if n == 5 else DGP((-0.5, -0.8, -1.0, -1.5), (0.4, 0.7, 1.0, 1.6), -2.5, 1.0,
                                                       (0.5, 0.4, 0.3, 0.0), np.array([[0, .6, 0, 0], [.6, 0, -.4, 0], [0, -.4, 0, 0], [0, 0, 0, 0]]))
        v = dg.v_pop; full = (1 << n) - 1
        print(f"  n={n} k={k}: v(N)={v[full]:.4f} v(∅)={v[0]:.4f} lift={v[full]-v[0]:.4f}  Sh={np.round(dg.sh,4)}")
        designs = {}
        for mb in range(0, k + 1):
            mt = k - mb
            if mb + mt <= n - 1: designs[f"B({mb},{mt})"] = design_B(n, mb, mt)
        designs["L_k (over-id)"] = design_layered(n, k)
        designs["full 2^n"] = design_full(n)
        for name, cells in designs.items():
            s2, lf, total = cell_stats(v, cells)
            # (i) equal allocation, BLUE weights; (ii) Neyman allocation for the BLUE
            m = len(cells)
            f = np.ones(m) / m
            C, _ = blue_weights(n, cells, k, var=s2 / f)
            V_eq = ((C ** 2) * (s2 / f)).sum(1)
            fN = neyman_alloc(C, s2)
            C2, _ = blue_weights(n, cells, k, var=s2 / fN)
            V_N = ((C2 ** 2) * (s2 / fN)).sum(1)
            print(f"    {name:14s} cells={m:3d}  equal: tr V·N={V_eq.sum():.4f} lift%={100*(f@lf)/total:5.1f} | "
                  f"Neyman: tr V·N={V_N.sum():.4f} lift%={100*(fN@lf)/total:5.1f}")
        # frontier for full-factorial (global optimum) and best structured
        mus = [0, 1, 3, 10, 30, 100, 300, 1000]
        for name in ["full 2^n", "L_k (over-id)"] + [d for d in designs if d.startswith("B(")]:
            fr = frontier_fixed_design(n, designs[name], k, v, mus)
            print(f"    frontier {name}: " + "  ".join(f"({100*l:.1f}%, {tv:.4f})" for l, tv, mv, f in fr))

def e6():
    """Finite-user end-to-end on dive 13's n=3 benchmark and a 5-channel DGP: bias/sd/MSE by design."""
    print("E6: finite-N Monte Carlo (binomial cells) — n=3 benchmark and n=5 with pairwise synergy + 3-way term")
    rng = np.random.default_rng(7)
    for label, dg, N, reps in [("n=3 benchmark", benchmark3(), 2_000_000, 60), ("n=5 synergy", dgp5(1.0, 0.0), 4_000_000, 40),
                               ("n=5 synergy+3way(0.8)", dgp5(1.0, 0.8), 4_000_000, 40)]:
        n = dg.n; v = dg.v_pop
        print(f"  {label}: Sh={np.round(dg.sh, 5)}  dividends by order: " +
              ", ".join(f"|D{t}|max={max(abs(dg.div[T]) for T in range(1<<n) if popcount(T)==t):.4f}" for t in range(1, n + 1)))
        designs = {"ghost B(0,1)": design_B(n, 0, 1), "layered B(1,1) k2": design_B(n, 1, 1), "holdout-only B(0,2) k2": design_B(n, 0, 2)}
        if n >= 4: designs["B(1,2) k3"] = design_B(n, 1, 2)
        for name, cells in designs.items():
            k = 2 if "k2" in name else (3 if "k3" in name else 1)
            if not identified(n, cells, k):
                print(f"    {name:26s}: NOT identified under cap {k} (rank test) — skipped"); continue
            s2, lf, total = cell_stats(v, cells)
            C, _ = blue_weights(n, cells, k, var=s2)
            fN = neyman_alloc(C, s2); C, _ = blue_weights(n, cells, k, var=s2 / fN)
            Nc = np.maximum((fN * N).astype(int), 1000)
            ests = []
            for r in range(reps):
                vhat = np.array([dg.simulate_cell(A, Nc[i], rng) for i, A in enumerate(cells)])
                ests.append(C @ vhat)
            ests = np.array(ests); bias = ests.mean(0) - dg.sh; sd = ests.std(0)
            pred_sd = np.sqrt(((C ** 2) * (s2 / Nc)).sum(1))
            print(f"    {name:26s}: bias={np.round(bias, 5)} sd={np.round(sd, 5)} pred sd={np.round(pred_sd, 5)} "
                  f"lift%={100*(fN@lf)/total:.1f} rmse/|Sh|={np.sqrt((bias**2+sd**2)).sum()/np.abs(dg.sh).sum():.3f}")

def e7():
    """Cap lack-of-fit test: null rejection and power against a 3-way dividend (n=4, over-identified designs).
    For B(1,1) the df=1 statistic is the add-up test: do the ½(solo+removal) credits sum to the global-holdout lift?"""
    print("E7: lack-of-fit test of the pairwise cap (k=2) from over-identified cells, n=4, N=2M users per design, 60 reps")
    rng = np.random.default_rng(11)
    n = 4
    for th3 in [0.0, 0.4, 0.8, 1.2]:
        dg = DGP((-0.5, -0.8, -1.0, -1.5), (0.4, 0.7, 1.0, 1.6), -2.5, 1.0, (0.5, 0.4, 0.3, 0.0),
                 np.array([[0, .6, 0, 0], [.6, 0, -.4, 0], [0, -.4, 0, 0], [0, 0, 0, 0]]), theta3=th3)
        d3 = [dg.div[T] for T in range(16) if popcount(T) == 3]
        for name, cells in [("B(1,1)", design_B(n, 1, 1)), ("B(1,2)", design_B(n, 1, 2)), ("full", design_full(n))]:
            N = 2_000_000; Nc = np.full(len(cells), N // len(cells))
            rej = 0; stats = []
            for r in range(60):
                vhat = np.array([dg.simulate_cell(A, Nc[i], rng) for i, A in enumerate(cells)])
                s, df, p = cap_test(vhat, cells, n, 2, Nc); stats.append(s); rej += p < 0.05
            print(f"  θ3={th3}: D3 = {np.round(d3, 4)}  {name:7s} df={df} mean stat={np.mean(stats):.1f} (sd {np.std(stats):.1f}) reject@5%={rej}/60")

def e8():
    """Robustness sweep 1: conversion-rate scale and heterogeneity of cell variances; Neyman vs equal; seed sensitivity."""
    print("E8: robustness — outcome base rate sweep (β0) and seeds; BLUE+Neyman vs equal allocation, n=5 k=2")
    n = 5
    for beta0 in [-4.0, -2.5, -1.0]:
        dg = DGP([-0.5, -0.8, -1.0, -1.2, -1.5], [0.4, 0.7, 1.0, 1.3, 1.6], beta0, 1.0, [0.5, 0.4, 0.3, 0.2, 0.0])
        v = dg.v_pop
        for name, cells in [("B(1,1)", design_B(n, 1, 1)), ("B(0,2)", design_B(n, 0, 2)), ("B(2,0)", design_B(n, 2, 0))]:
            s2, lf, total = cell_stats(v, cells); m = len(cells)
            f = np.ones(m) / m; C, _ = blue_weights(n, cells, 2, var=s2 / f); Veq = ((C ** 2) * (s2 / f)).sum(1).sum()
            fN = neyman_alloc(C, s2); C2, _ = blue_weights(n, cells, 2, var=s2 / fN); VN = ((C2 ** 2) * (s2 / fN)).sum(1).sum()
            print(f"  β0={beta0} v(N)={v[-1]:.4f} lift={total:.4f} {name}: trV·N equal={Veq:.4f} Neyman={VN:.4f} (gain {100*(1-VN/Veq):.1f}%), "
                  f"lift% {100*(f@lf)/total:.1f}→{100*(fN@lf)/total:.1f}, relative sd of Sh at N=1e6: {np.sqrt(VN/1e6)/np.abs(dg.sh).sum():.3f}")
    # seed sensitivity of E6-type MC at n=3
    rng_seeds = [1, 2, 3, 4, 5]; dg = benchmark3(); cells = design_B(3, 0, 2); s2, lf, total = cell_stats(dg.v_pop, cells)
    C, _ = blue_weights(3, cells, 2, var=s2); Nc = np.full(len(cells), 300_000)
    for sd_ in rng_seeds:
        rng = np.random.default_rng(sd_); ests = np.array([C @ np.array([dg.simulate_cell(A, Nc[i], rng) for i, A in enumerate(cells)]) for _ in range(30)])
        print(f"  seed {sd_}: mean est {np.round(ests.mean(0), 5)} sd {np.round(ests.std(0), 5)} (truth {np.round(dg.sh, 5)})")

def monotone_game(n, rng, base=0.05, main=(0.01, 0.03), pair_sd=0.003):
    d = np.zeros(1 << n); d[0] = base
    for T in range(1, 1 << n):
        t = popcount(T)
        if t == 1: d[T] = rng.uniform(*main)
        elif t == 2: d[T] = rng.normal(0, pair_sd)
    return zeta(d, n), d

def e9():
    """Robustness sweep 2: scaling in n — cells, precision, lift for the k=2 designs, n=3..8 (monotone pairwise game)."""
    print("E9: scaling in n for k=2 designs (monotone pairwise game: base 5%, main effects U(1%,3%), pair sd 0.3%)")
    rng = np.random.default_rng(3)
    for n in range(3, 9):
        v, d = monotone_game(n, rng)
        row = []
        for name, cells in [("B(1,1)", design_B(n, 1, 1)), ("B(0,2)", design_B(n, 0, 2)), ("B(2,0)", design_B(n, 2, 0))]:
            s2, lf, total = cell_stats(v, cells)
            C, _ = blue_weights(n, cells, 2, var=s2); fN = neyman_alloc(C, s2); C, _ = blue_weights(n, cells, 2, var=s2 / fN)
            V = ((C ** 2) * (s2 / fN)).sum(1)
            kap = lift_price(n, cells, 2, v)
            row.append(f"{name}: cells {len(cells):3d} trV·N {V.sum():.3f} lift% {100*(fN@lf)/total:5.1f} κ {kap:.2f}")
        print(f"  n={n} lift={v[-1]-v[0]:.3f}: " + " | ".join(row))

def lift_price(n, cells, k, v, eps=1e-7):
    """κ = lim_{ℓ→0} N·ℓ·tr V = min over cap-k-unbiased C of (Σ_{A≠N} ||C_A||₂ s_A √ℓ_A)²  (group-L1 program;
    Cauchy–Schwarz gives the allocation g_A ∝ ||C_A|| s_A/√ℓ_A).  Solved by parametrising the unbiased family as
    C0 + Σ α_i Z_i (Z = null space of Xᵀ) and minimising a smoothed group norm."""
    from scipy.linalg import null_space
    s2, lf, total = cell_stats(v, cells); full = (1 << n) - 1
    lfrac = lf / total; idx_full = cells.index(full)
    W, cols = sh_weight_matrix(n, k); X = incidence(cells, [0] + cols); Wf = np.hstack([np.zeros((n, 1)), W])
    C0, _ = blue_weights(n, cells, k)                         # some unbiased C (n × cells)
    Z = null_space(X.T)                                        # cells × df
    wts = np.sqrt(s2) * np.sqrt(np.maximum(lfrac, 0)); wts[idx_full] = 0
    def obj(alpha):
        A = alpha.reshape(n, -1)
        C = C0 + A @ Z.T
        return (np.sqrt((C ** 2).sum(0) + eps) * wts).sum() ** 2
    best = obj(np.zeros(n * Z.shape[1]))
    r = minimize(obj, np.zeros(n * Z.shape[1]), method="BFGS", options={"gtol": 1e-10, "maxiter": 5000})
    r2 = minimize(obj, r.x, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 20000})
    return min(best, r.fun, r2.fun)

def e13():
    """The lift price of precision: κ = lim N·ℓ·trV — frontier limit vs the Cauchy–Schwarz closed form."""
    print("E13: κ law — frontier N·ℓ·trV at small ℓ vs κ = min_C (Σ_{A≠N} ||C_A|| s_A √ℓ_A)²")
    for n, k, dg in [(4, 2, DGP((-0.5, -0.8, -1.0, -1.5), (0.4, 0.7, 1.0, 1.6), -2.5, 1.0, (0.5, 0.4, 0.3, 0.0),
                                np.array([[0, .6, 0, 0], [.6, 0, -.4, 0], [0, -.4, 0, 0], [0, 0, 0, 0]]))),
                     (5, 2, dgp5(1.0)), (5, 3, dgp5(1.0)), (6, 2, DGP([-0.5, -0.7, -0.9, -1.1, -1.3, -1.5], [0.4, 0.6, 0.8, 1.0, 1.3, 1.6],
                                                                    -2.5, 1.0, [0.5, 0.4, 0.3, 0.25, 0.2, 0.0]))]:
        v = dg.v_pop
        designs = {f"B({mb},{k-mb})": design_B(n, mb, k - mb) for mb in range(0, k + 1) if k <= n - 1}
        designs["full"] = design_full(n)
        for name, cells in designs.items():
            kap = lift_price(n, cells, k, v)
            fr = frontier_fixed_design(n, cells, k, v, [300, 3000, 30000])
            prod = [l * tv for l, tv, mv, f in fr]
            print(f"  n={n} k={k} {name:7s}: κ closed-form {kap:.3f}; frontier N·ℓ·trV at ℓ={100*fr[0][0]:.1f}%/{100*fr[1][0]:.2f}%/{100*fr[2][0]:.3f}%: "
                  f"{prod[0]:.3f}/{prod[1]:.3f}/{prod[2]:.3f}")

def e14():
    """The one over-identifying restriction of B(mb,mt), mb+mt=k: it is the add-up (efficiency) discrepancy of the
    closed-form credits; the equal-split add-up projection is nearly the BLUE."""
    print("E14: add-up restriction = null vector; efficiency-projected closed form vs BLUE (equal cell variances)")
    from scipy.linalg import null_space
    for n, mb, mt, k in [(4, 1, 1, 2), (5, 1, 1, 2), (5, 0, 2, 2), (5, 2, 0, 2), (5, 1, 2, 3), (6, 2, 1, 3), (6, 0, 3, 3), (6, 2, 2, 4), (7, 1, 1, 2), (8, 1, 1, 2)]:
        cells = design_B(n, mb, mt); W, cols = sh_weight_matrix(n, k); X = incidence(cells, [0] + cols)
        nu = null_space(X.T); assert nu.shape[1] == 1; nu = nu[:, 0]
        Cc, _ = closed_form_weights(n, mb, mt, k); Cb, _ = blue_weights(n, cells, k)
        full = (1 << n) - 1; idx = {A: i for i, A in enumerate(cells)}
        e = np.zeros(len(cells)); e[idx[full]] = 1; e[idx[0]] = -1
        ceff = Cc.sum(0) - e
        cos = abs(ceff @ nu) / np.linalg.norm(ceff) / np.linalg.norm(nu)
        Cp = Cc - np.outer(np.ones(n) / n, ceff)
        print(f"  n={n} B({mb},{mt}) k={k}: df={len(cells)-np.linalg.matrix_rank(X)}, cos(add-up, ν)={cos:.6f}; Σc² closed {(Cc**2).sum(1).mean():.3f} "
              f"→ add-up projected {(Cp**2).sum(1).mean():.3f} vs BLUE {(Cb**2).sum(1).mean():.3f}")

def e10():
    """Negative/power control: a non-identified design (ghost menu B(0,1) at k=2; L1 at k=3) yields biased pseudo-estimates,
    and the identification test discriminates."""
    print("E10: negative control — non-identified designs give biased min-norm pseudo-estimates; rank test discriminates")
    rng = np.random.default_rng(5)
    for n, mb, mt, k in [(3, 0, 1, 2), (4, 0, 1, 2), (4, 1, 1, 3), (5, 1, 1, 3), (5, 1, 1, 2)]:
        cells = design_B(n, mb, mt)
        W, cols = sh_weight_matrix(n, k); cols = [0] + cols; X = incidence(cells, cols); Wf = np.hstack([np.zeros((n, 1)), W])
        res = [in_rowspace(X, Wf[j])[1] for j in range(n)]
        errs = []
        for _ in range(200):
            v, d = random_game(n, k, rng)
            D = np.linalg.pinv(X) @ np.array([v[A] for A in cells])
            errs.append(np.abs(Wf @ D - shapley_from_v(v, n)).max())
        print(f"  n={n} B({mb},{mt}) k={k}: identified={identified(n, cells, k)} row-space residuals {np.round(res, 3)}; "
              f"pseudo-estimate max|err| median {np.median(errs):.3f} (Sh scale {1.0:.1f})")

def e11():
    """MSE-optimal cap/design choice under a prior on higher-order dividends: variance vs aliasing² tradeoff."""
    print("E11: variance + aliasing² tradeoff across designs (monotone game), prior sd τ3 of order-3 dividends (τ4=τ3/3), N users")
    rng = np.random.default_rng(9)
    for n in [5, 7]:
        v, d = monotone_game(n, rng)
        designs = {"B(1,1) k2": (design_B(n, 1, 1), 2), "B(0,2) k2": (design_B(n, 0, 2), 2), "B(2,0) k2": (design_B(n, 2, 0), 2),
                   "B(1,2) k3": (design_B(n, 1, 2), 3), "B(2,1) k3": (design_B(n, 2, 1), 3), "B(0,3) k3": (design_B(n, 0, 3), 3),
                   "B(2,2) k4": (design_B(n, 2, 2), 4)}
        print(f"  n={n}: lift={v[-1]-v[0]:.3f}, Sh sum={shapley_from_v(v, n).sum():.3f}, mean |D1|={np.mean([abs(d[T]) for T in range(1<<n) if popcount(T)==1]):.4f}")
        pre = {}
        for name, (cells, k) in designs.items():
            s2, lf, total = cell_stats(v, cells)
            C, _ = blue_weights(n, cells, k, var=s2); fN = neyman_alloc(C, s2); C, _ = blue_weights(n, cells, k, var=s2 / fN)
            V = ((C ** 2) * (s2 / fN)).sum(1)
            L3 = leak_matrix(n, cells, C, 3); L4 = leak_matrix(n, cells, C, 4)
            b3 = sum((lv ** 2) for lv in L3.values()); b4 = sum((lv ** 2) for lv in L4.values())
            pre[name] = (V, b3, b4, 100 * (fN @ lf) / total, len(cells))
        for N in [2e5, 2e6, 2e7]:
            for tau3 in [0.0, 0.001, 0.003]:
                row = []
                for name, (V, b3, b4, lift, m) in pre.items():
                    mse = (V / N + b3 * tau3 ** 2 + b4 * (tau3 / 3) ** 2).sum()
                    row.append(f"{name}: {np.sqrt(mse):.5f}")
                print(f"    N={N:.0e} τ3={tau3}: rmse " + " | ".join(row))
        print("    cells / lift%: " + " | ".join(f"{name}: {m}/{lift:.0f}%" for name, (V, b3, b4, lift, m) in pre.items()))
        print("    trV·N: " + " | ".join(f"{name}: {V.sum():.3f}" for name, (V, b3, b4, lift, m) in pre.items()))

def e15():
    """Structured vs unstructured aliasing: closed form, add-up-projected, and BLUE on B(1,1) under genuine 3-way
    dividends (n=5 logistic DGP) — exact biases (no MC) and variances at N=4M."""
    print("E15: exact bias of closed-form / add-up-projected / BLUE (B(1,1), k=2) under 3-way dividends, n=5")
    n = 5
    for th3 in [0.0, 0.8, 1.5]:
        dg = dgp5(1.0, th3); v = dg.v_pop; cells = design_B(n, 1, 1); vv = np.array([v[A] for A in cells])
        s2, lf, total = cell_stats(v, cells); N = 4_000_000; f = np.ones(len(cells)) / len(cells)
        Cc, _ = closed_form_weights(n, 1, 1, 2); Cb, _ = blue_weights(n, cells, 2, var=s2 / f)
        full = (1 << n) - 1; idx = {A: i for i, A in enumerate(cells)}
        e = np.zeros(len(cells)); e[idx[full]] = 1; e[idx[0]] = -1
        Cp = Cc - np.outer(np.ones(n) / n, Cc.sum(0) - e)
        d3 = {T: dg.div[T] for T in range(1 << n) if popcount(T) == 3 and abs(dg.div[T]) > 1e-6}
        print(f"  θ3={th3}: Sh={np.round(dg.sh, 5)}; D3 nonzero: " + ", ".join(f"{members(T,n)}:{d:+.4f}" for T, d in d3.items()) +
              f"; add-up gap Σ_j Ŝh_j−lift = {Cc.sum(0)@vv - (v[full]-v[0]):+.5f} (pred D123/2 = {dg.div[7]/2:+.5f})")
        for name, C in [("closed ½(solo+removal)", Cc), ("add-up projected", Cp), ("BLUE", Cb)]:
            bias = C @ vv - dg.sh; sd = np.sqrt(((C ** 2) * (s2 / (f * N))).sum(1))
            print(f"    {name:24s}: bias={np.round(bias, 5)}  sd={np.round(sd, 5)}  |bias|/sd max={np.max(np.abs(bias)/sd):.2f}")

def e12():
    """Competing method: KernelSHAP / permutation-style random-coalition cells vs structured designs at matched N and cap."""
    print("E12: random-coalition (permutation-sampling) designs vs B(mb,mt): precision and lift at matched N, n=5, k=2 and uncapped")
    n = 5; rng = np.random.default_rng(21)
    dg = dgp5(1.0, 0.0); v = dg.v_pop; full = (1 << n) - 1
    # permutation sampling: cells are all prefixes of random orders => all coalitions appear; equivalent to full factorial
    # with weights ∝ (|A|!(n-|A|-1)!/n!) ... compare: (i) full factorial with Shapley-permutation allocation, (ii) B designs
    perm_share = np.zeros(1 << n)
    for A in range(1 << n):
        s = popcount(A); perm_share[A] = 1.0 / (n + 1) / comb(n, s)   # uniform over sizes, uniform within size (Owen/permutation)
    for label, cells, f, k in [("permutation cells (uncapped)", design_full(n), perm_share, n),
                               ("permutation cells, k=2 fit", design_full(n), perm_share, 2),
                               ("B(1,1) k=2 Neyman", design_B(n, 1, 1), None, 2),
                               ("B(0,2) k=2 Neyman", design_B(n, 0, 2), None, 2),
                               ("B(2,2) k=4 Neyman", design_B(n, 2, 2), None, 4),
                               ("full factorial k=n Neyman", design_full(n), None, n)]:
        s2, lf, total = cell_stats(v, cells)
        if f is None:
            C, _ = blue_weights(n, cells, k, var=s2); f = neyman_alloc(C, s2)
        else:
            f = np.array([f[A] for A in cells])
        C, _ = blue_weights(n, cells, k, var=s2 / f)
        V = ((C ** 2) * (s2 / f)).sum(1)
        # bias under the true (uncapped) game
        est = C @ np.array([v[A] for A in cells]); bias = est - dg.sh
        print(f"  {label:30s}: cells {len(cells):2d} trV·N={V.sum():.4f} lift%={100*(f@lf)/total:.1f} |bias|max={np.abs(bias).max():.5f}")

def quick():
    t0 = time.time()
    for f in [e1, e2, e3, e4, e10, e14, e15]:
        f(); print(f"  [{time.time()-t0:.0f}s]\n")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "quick"
    globals()[cmd]()
