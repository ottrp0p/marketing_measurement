"""
Dive 10 — Estimands under algorithmic delivery (B4 / M3).
Core library: DGP with divergent delivery, estimands, estimators, bounds.

Model
-----
User i has covariates X_i (5-dim). Potential outcomes:
    Y(0)      ~ Bernoulli(p0(X))            (no ad)
    Y(1, a)   ~ Bernoulli(p0(X) + delta_a(X))   a in {A, B}
Platform scores each creative for each user: m_a(X) = predicted P(Y=1 | creative a) + prediction noise.
Delivery in arm a exposes user i with probability
    pi_a(X) = sigmoid(b0 + b1 * standardized(m_a(X)) + b2 * comp(X)) * cap
where comp(X) is auction competition (confounder), unlogged. Arms: control (no ad), A, B randomized.

Estimands
---------
ITT_a           = E[pi_a(X) delta_a(X)]                    (effect of creative a under ITS OWN delivery)
ITTdiff         = ITT_A - ITT_B
tau(pi)         = E[pi(X) (delta_A(X) - delta_B(X))]        (creative effect under FIXED delivery pi)
Common choices: pi = pi_B ("A in B's audience"), pi = pi_A, pi = (pi_A+pi_B)/2.

Divergence-bias identity:  tau(pi_B) - ITTdiff = E[(pi_B - pi_A) delta_A].
"""
import numpy as np
from scipy.special import expit

# ---------------------------------------------------------------- DGP ----
def make_population(N, rng, het=1.0, base=0.02, lift=0.15, score_noise=0.3,
                    comp_conf=0.0, b1=1.5, cap=0.6, b0=-0.4, delta_corr=0.5,
                    exclusion=0.0, comp_x1=0.0, log_noise=0.0):
    """Return dict with X, p0, dA, dB, mA, mB, piA, piB, comp.
    het: heterogeneity scale of creative effects.
    delta_corr: correlation between dA and dB heterogeneity (1 -> parallel, 0 -> independent).
    comp_conf: strength with which competition (correlated with high-value users) drives delivery.
    exclusion: fraction of users with pi_A forced to 0 (support failure)."""
    X = rng.standard_normal((N, 5))
    # baseline conversion prob, heterogeneous
    p0 = base * np.exp(0.5 * X[:, 0] - 0.125)          # mean ~ base
    p0 = np.clip(p0, 1e-4, 0.3)
    # creative effects (absolute lift in conversion prob), heterogeneous, positive on average
    hA = X[:, 1]
    hB = delta_corr * X[:, 1] + np.sqrt(1 - delta_corr ** 2) * X[:, 2]
    dA = lift * p0 * np.exp(het * 0.8 * hA - 0.32 * het ** 2)
    dB = lift * p0 * np.exp(het * 0.8 * hB - 0.32 * het ** 2) * 0.9   # B is 10% weaker on average
    dA = np.minimum(dA, 1 - p0); dB = np.minimum(dB, 1 - p0)
    # platform prediction of P(Y=1|creative) -- noisy but informative
    mA = np.log(p0 + dA) + score_noise * rng.standard_normal(N)
    mB = np.log(p0 + dB) + score_noise * rng.standard_normal(N)
    # auction competition: high for high-baseline users (unlogged confounder)
    comp = 0.7 * X[:, 0] + 0.7 * X[:, 3] + comp_x1 * X[:, 1]
    def pol(m):
        z = (m - m.mean()) / m.std()
        return cap * expit(b0 + b1 * z - comp_conf * comp)
    piA, piB = pol(mA), pol(mB)
    if exclusion > 0:
        excl = rng.random(N) < exclusion * 2 * expit(-2 * X[:, 4])  # exclude low-x4 users from A
        piA = np.where(excl, 0.0, piA)
    # logged scores may be noisier/coarser than the scores that drove delivery
    mA_log = mA + log_noise * rng.standard_normal(N); mB_log = mB + log_noise * rng.standard_normal(N)
    return dict(X=X, p0=p0, dA=dA, dB=dB, mA=mA_log, mB=mB_log, mA_true=mA, mB_true=mB, piA=piA, piB=piB, comp=comp)


def true_estimands(P):
    dA, dB, piA, piB = P['dA'], P['dB'], P['piA'], P['piB']
    return dict(ITT_A=np.mean(piA * dA), ITT_B=np.mean(piB * dB),
                ITTdiff=np.mean(piA * dA - piB * dB),
                tau_piB=np.mean(piB * (dA - dB)), tau_piA=np.mean(piA * (dA - dB)),
                tau_avg=np.mean(0.5 * (piA + piB) * (dA - dB)),
                tau_ov=np.mean(2 * piA * piB / (piA + piB + 1e-12) * (dA - dB)),
                bias_identity=np.mean((piB - piA) * dA))


def run_experiment(P, rng, arms=('C', 'A', 'B'), shares=None):
    """Randomize users to arms, realize exposure and outcomes. Returns dict of arrays."""
    N = len(P['p0'])
    k = len(arms)
    Z = rng.integers(0, k, N) if shares is None else rng.choice(k, N, p=shares)
    pi = np.zeros(N)
    pi[Z == 1] = P['piA'][Z == 1]
    if k > 2:
        pi[Z == 2] = P['piB'][Z == 2]
    D = rng.random(N) < pi
    p = P['p0'].copy()
    p[(Z == 1) & D] += P['dA'][(Z == 1) & D]
    if k > 2:
        p[(Z == 2) & D] += P['dB'][(Z == 2) & D]
    Y = (rng.random(N) < p).astype(float)
    return dict(Z=Z, D=D, Y=Y)


# ---------------------------------------------------------- estimators ----
def bucketize(v, K):
    q = np.quantile(v, np.linspace(0, 1, K + 1)[1:-1])
    return np.searchsorted(q, v)


def cell_estimator(P, R, cells, target='piB', return_cells=False, cell_pol=None):
    """Logged-cell estimator (holdout design).
    cells: integer cell id per user (e.g. bucketized (mA, mB) or segment).
    Within cell c: ITT_a(c) = mean(Y|Z=a,c) - mean(Y|Z=C,c); r_a(c) = mean(D|Z=a,c);
    delta_a(c) = ITT_a(c)/r_a(c); tau = sum_c p_c pi(c) (delta_A(c) - delta_B(c)),
    where pi(c) is the target policy's cell reach (r_B(c) for target=piB) -- identified from logs."""
    Z, D, Y = R['Z'], R['D'], R['Y']
    K = cells.max() + 1
    out = 0.0; rows = []
    for c in range(K):
        m = cells == c
        pc = m.mean()
        mc, ma, mb = m & (Z == 0), m & (Z == 1), m & (Z == 2)
        y0 = Y[mc].mean()
        ittA, ittB = Y[ma].mean() - y0, Y[mb].mean() - y0
        rA, rB = D[ma].mean(), D[mb].mean()
        if cell_pol is not None:
            pic = cell_pol[m].mean()
        else:
            pic = {'piB': rB, 'piA': rA, 'avg': 0.5 * (rA + rB)}[target]
        dA = ittA / max(rA, 1e-9); dB = ittB / max(rB, 1e-9)
        out += pc * pic * (dA - dB)
        rows.append((pc, pic, rA, rB, ittA, ittB))
    return (out, rows) if return_cells else out


def naive_estimators(R):
    Z, D, Y = R['Z'], R['D'], R['Y']
    itt = Y[Z == 1].mean() - Y[Z == 2].mean()
    exposed = Y[(Z == 1) & D].mean() - Y[(Z == 2) & D].mean()      # per-exposed comparison
    # per-exposed times B-reach (scale to same units as tau_piB)
    rB = D[Z == 2].mean()
    return dict(ITTdiff=itt, exposed_diff=exposed, exposed_scaled=exposed * rB)


def poststrat_exposed(R, seg, target_seg_weights=None):
    """Reweight arm-A exposed users to arm-B exposed segment mix, compare to B exposed (in per-exposure units),
    then scale by B reach. The 'industry' correction."""
    Z, D, Y = R['Z'], R['D'], R['Y']
    K = seg.max() + 1
    wB = np.array([((Z == 2) & D & (seg == s)).sum() for s in range(K)], float)
    wB /= wB.sum()
    yA = np.array([Y[(Z == 1) & D & (seg == s)].mean() if ((Z == 1) & D & (seg == s)).any() else 0 for s in range(K)])
    yB = np.array([Y[(Z == 2) & D & (seg == s)].mean() if ((Z == 2) & D & (seg == s)).any() else 0 for s in range(K)])
    rB = D[Z == 2].mean()
    return float(np.sum(wB * (yA - yB))) * rB


def lambda_bounds(R, seg, Lam, return_parts=False):
    """No-cross-score bounds with a holdout: within segment s, pi_B/pi_A in [rho_bar/Lam, rho_bar*Lam],
    rho_bar = r_B(s)/r_A(s); delta_A >= 0.  tau_piB in [sum p_s rho_bar/Lam ITT_A(s) - ITT_B, sum p_s rho_bar Lam ITT_A(s) - ITT_B]."""
    Z, D, Y = R['Z'], R['D'], R['Y']
    K = seg.max() + 1
    lo = hi = 0.0
    ittB_tot = Y[Z == 2].mean() - Y[Z == 0].mean()
    for s in range(K):
        m = seg == s
        ps = m.mean()
        y0 = Y[m & (Z == 0)].mean()
        ittA = Y[m & (Z == 1)].mean() - y0
        rA, rB = D[m & (Z == 1)].mean(), D[m & (Z == 2)].mean()
        rho = rB / max(rA, 1e-9)
        ittA_pos = max(ittA, 0.0)
        lo += ps * rho / Lam * ittA_pos
        hi += ps * rho * Lam * ittA_pos
    return lo - ittB_tot, hi - ittB_tot


def true_lambda(P, seg):
    """Realized within-segment divergence: max over users of |log((piB/piA)/rho_bar(s))| and its 95th pct."""
    K = seg.max() + 1
    lr = np.zeros(len(seg))
    for s in range(K):
        m = seg == s
        rho_bar = P['piB'][m].mean() / P['piA'][m].mean()
        lr[m] = np.log((P['piB'][m] / np.maximum(P['piA'][m], 1e-12)) / rho_bar)
    return np.exp(np.quantile(np.abs(lr), 0.95)), np.exp(np.abs(lr).max())



# ------------------------------------------------ population limits ----
def cell_limit(P, cells, target='piB', pol=None):
    """Population (N->inf) limit of the cell estimator, using true conditional means within cells.
    Returns (limit, mass_unidentified) where mass_unidentified is the target-policy mass in cells
    where the source arm has zero reach."""
    K = cells.max() + 1
    out = 0.0; miss = 0.0
    piA, piB, dA, dB = P['piA'], P['piB'], P['dA'], P['dB']
    for c in range(K):
        m = cells == c
        if not m.any():
            continue
        pc = m.mean()
        rA, rB = piA[m].mean(), piB[m].mean()
        ittA, ittB = (piA[m] * dA[m]).mean(), (piB[m] * dB[m]).mean()
        pic = {'piB': rB, 'piA': rA, 'avg': 0.5 * (rA + rB), 'ov': 2 * rA * rB / (rA + rB + 1e-12)}[target] if pol is None else pol[m].mean()
        if rA <= 0 or rB <= 0:
            miss += pc * pic
            continue
        out += pc * pic * (ittA / rA - ittB / rB)
    return out, miss


def naive_limits(P):
    piA, piB, dA, dB, p0 = P['piA'], P['piB'], P['dA'], P['dB'], P['p0']
    yA = (piA * (p0 + dA)).mean() / piA.mean()
    yB = (piB * (p0 + dB)).mean() / piB.mean()
    return dict(ITTdiff=(piA * dA - piB * dB).mean(), exposed_scaled=(yA - yB) * piB.mean())


def poststrat_limit(P, seg):
    piA, piB, dA, dB, p0 = P['piA'], P['piB'], P['dA'], P['dB'], P['p0']
    K = seg.max() + 1
    wB = np.array([piB[seg == s].sum() for s in range(K)]); wB /= wB.sum()
    out = 0.0
    for s in range(K):
        m = seg == s
        yA = (piA[m] * (p0[m] + dA[m])).sum() / piA[m].sum()
        yB = (piB[m] * (p0[m] + dB[m])).sum() / piB[m].sum()
        out += wB[s] * (yA - yB)
    return out * piB.mean()


def lambda_bounds_limit(P, seg, Lam):
    piA, piB, dA, dB = P['piA'], P['piB'], P['dA'], P['dB']
    K = seg.max() + 1
    lo = hi = 0.0
    ittB = (piB * dB).mean()
    for s in range(K):
        m = seg == s
        ps = m.mean()
        ittA = (piA[m] * dA[m]).mean()
        rho = piB[m].mean() / piA[m].mean()
        lo += ps * rho / Lam * ittA; hi += ps * rho * Lam * ittA
    return lo - ittB, hi - ittB


# ------------------------------------------------ influence-function CI ----
def cell_estimator_se(P, R, cells, target='piB'):
    """Cell estimator with influence-function standard error (delta method per cell; strata random).
    Cells with zero reach in either arm are dropped (their target mass is returned as `miss`)."""
    Z, D, Y = R['Z'], R['D'], R['Y']
    N = len(Z)
    K = cells.max() + 1
    phi = np.zeros(N)
    gs = np.zeros(K); pcs = np.zeros(K); miss = 0.0
    for c in range(K):
        m = cells == c
        if not m.any():
            continue
        pc = m.mean(); pcs[c] = pc
        idx = {a: m & (Z == a) for a in (0, 1, 2)}
        n = {a: idx[a].sum() for a in (0, 1, 2)}
        if min(n.values()) < 2:
            continue
        y0 = Y[idx[0]].mean(); yA = Y[idx[1]].mean(); yB = Y[idx[2]].mean()
        rA = D[idx[1]].mean(); rB = D[idx[2]].mean()
        if rA <= 0 or rB <= 0:
            miss += pc * (rB if target == 'piB' else rA if target == 'piA' else 0.0); pcs[c] = 0; continue
        ittA, ittB = yA - y0, yB - y0
        if target == 'piB':
            g = rB * ittA / rA - ittB
            gy0 = -rB / rA + 1; gyA = rB / rA; gyB = -1.0
            grA = -rB * ittA / rA ** 2; grB = ittA / rA
        elif target == 'piA':
            g = ittA - rA * ittB / rB
            gy0 = -1 + rA / rB; gyA = 1.0; gyB = -rA / rB
            grA = -ittB / rB; grB = rA * ittB / rB ** 2
        else:  # 'ov': pi = 2 rA rB/(rA+rB); g = 2 rB ittA/(rA+rB) - 2 rA ittB/(rA+rB)
            S = rA + rB
            g = 2 * (rB * ittA - rA * ittB) / S
            gy0 = -2 * (rB - rA) / S; gyA = 2 * rB / S; gyB = -2 * rA / S
            grA = 2 * (-ittB) / S - 2 * (rB * ittA - rA * ittB) / S ** 2
            grB = 2 * ittA / S - 2 * (rB * ittA - rA * ittB) / S ** 2
        gs[c] = g
        phi[idx[0]] += pc * gy0 * (Y[idx[0]] - y0) * N / n[0]
        phi[idx[1]] += pc * (gyA * (Y[idx[1]] - yA) + grA * (D[idx[1]] - rA)) * N / n[1]
        phi[idx[2]] += pc * (gyB * (Y[idx[2]] - yB) + grB * (D[idx[2]] - rB)) * N / n[2]
    est = float(np.sum(pcs * gs))
    phi += gs[cells] - est
    se = phi.std() / np.sqrt(N)
    return est, se, miss


# ------------------------------------------------ no-holdout construction ----
def noholdout_limit(P, cells, target='piB'):
    """Population limit of the no-holdout cross-score estimator: within cell, delta_a(c) is estimated by the
    exposed-minus-unexposed contrast in arm a. Valid iff D _||_ (Y(0), delta) | cell in each arm (full balancing)."""
    piA, piB, dA, dB, p0 = P['piA'], P['piB'], P['dA'], P['dB'], P['p0']
    K = cells.max() + 1; out = 0.0
    for c in range(K):
        m = cells == c
        if not m.any(): continue
        pc = m.mean()
        rA, rB = piA[m].mean(), piB[m].mean()
        if rA <= 0 or rB <= 0: continue
        yA1 = (piA[m] * (p0[m] + dA[m])).mean() / rA; yA0 = ((1 - piA[m]) * p0[m]).mean() / (1 - rA)
        yB1 = (piB[m] * (p0[m] + dB[m])).mean() / rB; yB0 = ((1 - piB[m]) * p0[m]).mean() / (1 - rB)
        pic = {'piB': rB, 'piA': rA, 'ov': 2 * rA * rB / (rA + rB)}[target]
        out += pc * pic * ((yA1 - yA0) - (yB1 - yB0))
    return out


def noholdout_estimator(R, cells, target='piB'):
    Z, D, Y = R['Z'], R['D'], R['Y']
    K = cells.max() + 1; out = 0.0
    for c in range(K):
        m = cells == c
        a1, a0 = m & (Z == 1) & D, m & (Z == 1) & ~D
        b1, b0 = m & (Z == 2) & D, m & (Z == 2) & ~D
        if min(a1.sum(), a0.sum(), b1.sum(), b0.sum()) < 2: continue
        rA = a1.sum() / (a1.sum() + a0.sum()); rB = b1.sum() / (b1.sum() + b0.sum())
        pic = {'piB': rB, 'piA': rA, 'ov': 2 * rA * rB / (rA + rB)}[target]
        out += m.mean() * pic * ((Y[a1].mean() - Y[a0].mean()) - (Y[b1].mean() - Y[b0].mean()))
    return out


# ------------------------------------------------ fast vectorized estimator ----
def cell_fast(R, cells, target='piB', K=None):
    """Vectorized cell estimator + IF standard error. Returns (est, se, miss)."""
    Z, D, Y = R['Z'], R['D'].astype(float), R['Y']
    N = len(Z); K = cells.max() + 1 if K is None else K
    n0 = np.bincount(cells[Z == 0], minlength=K).astype(float)
    n1 = np.bincount(cells[Z == 1], minlength=K).astype(float)
    n2 = np.bincount(cells[Z == 2], minlength=K).astype(float)
    y0 = np.bincount(cells[Z == 0], Y[Z == 0], minlength=K) / np.maximum(n0, 1)
    yA = np.bincount(cells[Z == 1], Y[Z == 1], minlength=K) / np.maximum(n1, 1)
    yB = np.bincount(cells[Z == 2], Y[Z == 2], minlength=K) / np.maximum(n2, 1)
    rA = np.bincount(cells[Z == 1], D[Z == 1], minlength=K) / np.maximum(n1, 1)
    rB = np.bincount(cells[Z == 2], D[Z == 2], minlength=K) / np.maximum(n2, 1)
    pc = (n0 + n1 + n2) / N
    ok = (n0 > 1) & (n1 > 1) & (n2 > 1) & (rA > 0) & (rB > 0)
    ittA, ittB = yA - y0, yB - y0
    rAs, rBs = np.where(ok, rA, 1.0), np.where(ok, rB, 1.0)
    if target == 'piB':
        g = rBs * ittA / rAs - ittB
        gy0 = -rBs / rAs + 1; gyA = rBs / rAs; gyB = -np.ones(K)
        grA = -rBs * ittA / rAs ** 2; grB = ittA / rAs
        miss = np.sum(pc[~ok] * rB[~ok])
    elif target == 'piA':
        g = ittA - rAs * ittB / rBs
        gy0 = -1 + rAs / rBs; gyA = np.ones(K); gyB = -rAs / rBs
        grA = -ittB / rBs; grB = rAs * ittB / rBs ** 2
        miss = np.sum(pc[~ok] * rA[~ok])
    else:
        S = rAs + rBs
        g = 2 * (rBs * ittA - rAs * ittB) / S
        gy0 = -2 * (rBs - rAs) / S; gyA = 2 * rBs / S; gyB = -2 * rAs / S
        grA = -2 * ittB / S - 2 * (rBs * ittA - rAs * ittB) / S ** 2
        grB = 2 * ittA / S - 2 * (rBs * ittA - rAs * ittB) / S ** 2
        miss = 0.0
    g = np.where(ok, g, 0.0); pcu = np.where(ok, pc, 0.0)
    est = float(np.sum(pcu * g))
    c = cells
    phi = np.zeros(N)
    m0, m1, m2 = Z == 0, Z == 1, Z == 2
    phi[m0] = pcu[c[m0]] * gy0[c[m0]] * (Y[m0] - y0[c[m0]]) * N / n0[c[m0]]
    phi[m1] = pcu[c[m1]] * (gyA[c[m1]] * (Y[m1] - yA[c[m1]]) + grA[c[m1]] * (D[m1] - rA[c[m1]])) * N / n1[c[m1]]
    phi[m2] = pcu[c[m2]] * (gyB[c[m2]] * (Y[m2] - yB[c[m2]]) + grB[c[m2]] * (D[m2] - rB[c[m2]])) * N / n2[c[m2]]
    phi += g[c] - est
    return est, float(phi.std() / np.sqrt(N)), float(miss)


# ------------------------------------------------ Round 3: SIMEX for noisy logged scores ----
def simex(P, rng, sig_u, K=40, lams=(0, 0.5, 1.0, 1.5, 2.0), B=4, R=None, target='piB', fit='quad'):
    """SIMEX on the cell variable: add N(0, lam*sig_u^2) noise to the logged scores, recompute the
    estimator (population limit if R is None, else finite-sample), fit polynomial in lam, extrapolate to lam=-1."""
    vals = []
    for lam in lams:
        acc = []
        reps = 1 if lam == 0 else B
        for b in range(reps):
            mA = P['mA'] + np.sqrt(lam) * sig_u * rng.standard_normal(len(P['mA']))
            mB = P['mB'] + np.sqrt(lam) * sig_u * rng.standard_normal(len(P['mB']))
            c = bucketize(mA, K) * K + bucketize(mB, K)
            acc.append(cell_limit(P, c, target)[0] if R is None else cell_fast(R, c, target)[0])
        vals.append(np.mean(acc))
    lams = np.array(lams); vals = np.array(vals)
    deg = 2 if fit == 'quad' else 1
    coef = np.polyfit(lams, vals, deg)
    return float(np.polyval(coef, -1.0)), vals


def simex_rel(P, rng, sig_u, K=40, lams=(0, 0.5, 1.0, 2.0, 4.0), B=4, R=None, target='piB'):
    """Reliability-extrapolation SIMEX: the attenuation of a cell estimator built on a noisy conditioning
    variable scales (Gaussian heuristic) like the unreliability u(lam) = (1+lam) s_u^2 / (s_m^2 + (1+lam) s_u^2).
    Fit est(lam) = a + b u(lam) by least squares and return a (= value at u=0, i.e. lam=-1)."""
    s_m2 = max(np.var(P['mA']) - sig_u ** 2, 1e-9)   # variance of the true score (from logged variance)
    vals = []; us = []
    for lam in lams:
        acc = []
        for b in range(1 if lam == 0 else B):
            mA = P['mA'] + np.sqrt(lam) * sig_u * rng.standard_normal(len(P['mA']))
            mB = P['mB'] + np.sqrt(lam) * sig_u * rng.standard_normal(len(P['mB']))
            c = bucketize(mA, K) * K + bucketize(mB, K)
            acc.append(cell_limit(P, c, target)[0] if R is None else cell_fast(R, c, target)[0])
        vals.append(np.mean(acc)); us.append((1 + lam) * sig_u ** 2 / (s_m2 + (1 + lam) * sig_u ** 2))
    A = np.vstack([np.ones(len(us)), np.array(us)]).T
    coef, *_ = np.linalg.lstsq(A, np.array(vals), rcond=None)
    return float(coef[0]), np.array(vals), np.array(us)


# ------------------------------------------------ drivers ----
def quick(seed=1, N=600_000):
    """Reproduce the headline numbers in ~1 min: sign flip, cell estimator, negative control, finite-N."""
    from scipy.special import expit
    CTR = dict(base=0.0005, lift=40)
    rng = np.random.default_rng(seed)
    P = make_population(N, rng, **CTR); T = true_estimands(P)
    c = bucketize(P['mA'], 40) * 40 + bucketize(P['mB'], 40)
    print('CTR regime (x1e-4): ITTdiff %.3f  tau(piB) %.3f  tau(piA) %.3f  tau(ov) %.3f  identity resid %.1e' % (
        T['ITTdiff'] * 1e4, T['tau_piB'] * 1e4, T['tau_piA'] * 1e4, T['tau_ov'] * 1e4,
        (T['tau_piB'] - T['ITTdiff']) - T['bias_identity']))
    print('cross-score cell limit K=40: piB %.3f  ov %.3f | mA-only %.3f  mB-only %.3f' % (
        cell_limit(P, c, 'piB')[0] * 1e4, cell_limit(P, c, 'ov')[0] * 1e4,
        cell_limit(P, bucketize(P['mA'], 40), 'piB')[0] * 1e4, cell_limit(P, bucketize(P['mB'], 40), 'piB')[0] * 1e4))
    R = run_experiment(P, rng)
    c20 = bucketize(P['mA'], 20) * 20 + bucketize(P['mB'], 20)
    e, se, _ = cell_fast(R, c20, 'piB'); eo, seo, _ = cell_fast(R, c20, 'ov')
    print('finite-N (K=20): piB %.3f ± %.3f | ov %.3f ± %.3f | sample ITTdiff %.3f' % (e * 1e4, se * 1e4, eo * 1e4, seo * 1e4, naive_estimators(R)['ITTdiff'] * 1e4))
    # negative control
    Pn = make_population(N, rng, **CTR); Pn['dB'] = Pn['dA'].copy()
    mB = np.log(Pn['p0'] + Pn['dB']) + 0.8 * rng.standard_normal(N); z = (mB - mB.mean()) / mB.std()
    Pn['piB'] = 0.6 * expit(-0.4 + 1.5 * z); Pn['mB'] = mB; Tn = true_estimands(Pn)
    cn = bucketize(Pn['mA'], 40) * 40 + bucketize(Pn['mB'], 40)
    print('NEGATIVE CONTROL (identical creatives, B scored noisier): ITTdiff %.3f  cell %.3f  (truth 0)' % (Tn['ITTdiff'] * 1e4, cell_limit(Pn, cn, 'piB')[0] * 1e4))
    # conversion regime
    Pc = make_population(N, rng); Tc = true_estimands(Pc)
    print('CONVERSION regime: ITTdiff %.3f tau(piB) %.3f (ratio %.2f)' % (Tc['ITTdiff'] * 1e4, Tc['tau_piB'] * 1e4, Tc['ITTdiff'] / Tc['tau_piB']))


# ------------------------------------------------ experiment drivers (as run in the dive) ----
CTR = dict(base=0.0005, lift=40)

def _cells(P, K):
    return bucketize(P['mA'], K) * K + bucketize(P['mB'], K)

def e4_coarseness(N=1_000_000, seed=11):
    rng = np.random.default_rng(seed); P = make_population(N, rng, **CTR); T = true_estimands(P)
    for K in (2, 3, 5, 10, 20, 40, 80):
        c = _cells(P, K)
        print('K=%2d holdout piB %.3f piA %.3f ov %.3f | no-holdout piB %.3f | truth piB %.3f' % (K, cell_limit(P, c, 'piB')[0] * 1e4,
              cell_limit(P, c, 'piA')[0] * 1e4, cell_limit(P, c, 'ov')[0] * 1e4, noholdout_limit(P, c) * 1e4, T['tau_piB'] * 1e4))

def e5_confounder(N=1_000_000, seed=11):
    for reg, kw in (('CTR', CTR), ('CONV', {})):
        for cc, cx in ((0, 0), (1, 0), (1, 0.5), (1, 1.0), (2, 1.0)):
            rng = np.random.default_rng(seed); P = make_population(N, rng, comp_conf=cc, comp_x1=cx, **kw); T = true_estimands(P)
            c = _cells(P, 20)
            print('%s comp=%.0f x1=%.1f truth %.3f holdout %.3f no-holdout %.3f' % (reg, cc, cx, T['tau_piB'] * 1e4, cell_limit(P, c)[0] * 1e4, noholdout_limit(P, c) * 1e4))

def e6_lognoise(N=1_000_000, seed=5):
    for ln in (0, 0.1, 0.2, 0.3, 0.5, 1.0):
        rng = np.random.default_rng(seed); P = make_population(N, rng, log_noise=ln, **CTR); T = true_estimands(P)
        h = cell_limit(P, _cells(P, 40))[0]
        print('log_noise %.1f reliability %.2f truth %.3f est %.3f' % (ln, np.corrcoef(P['mA'], P['mA_true'])[0, 1] ** 2, T['tau_piB'] * 1e4, h * 1e4))

def e7_exclusion(N=600_000, seed=41):
    for ex in (0.3, 0.5):
        rng = np.random.default_rng(seed); P = make_population(N, rng, **CTR)
        excl = rng.random(N) < ex * 2 * expit(-2 * P['X'][:, 1]); P['piA'] = np.where(excl, 0.0, P['piA']); T = true_estimands(P)
        c = _cells(P, 40); c2 = c * 2 + excl.astype(int)
        tid = np.mean(np.where(~excl, P['piB'] * (P['dA'] - P['dB']), 0))
        print('excl %.1f truth %.3f identified-part %.3f | no flag %.3f | flag %.3f missed %.3f' % (ex, T['tau_piB'] * 1e4, tid * 1e4, cell_limit(P, c)[0] * 1e4, cell_limit(P, c2)[0] * 1e4, cell_limit(P, c2)[1]))

def e9_negative_control(N=1_000_000, seed=5):
    for sn_B, bias_B in ((0.3, 0), (0.8, 0), (0.3, 0.5), (0.8, 0.5)):
        rng = np.random.default_rng(seed); P = make_population(N, rng, **CTR); P['dB'] = P['dA'].copy()
        mB = np.log(P['p0'] + P['dB']) + sn_B * rng.standard_normal(N) + bias_B * P['X'][:, 3]
        z = (mB - mB.mean()) / mB.std(); P['piB'] = 0.6 * expit(-0.4 + 1.5 * z); P['mB'] = mB
        T = true_estimands(P); print('noiseB %.1f biasB %.1f ITTdiff %.3f cell %.3f' % (sn_B, bias_B, T['ITTdiff'] * 1e4, cell_limit(P, _cells(P, 40))[0] * 1e4))

def e10_crossover(reps=20, seed=21):
    for reg, kw in (('CTR', CTR), ('CONV', {})):
        for N in (300_000, 1_000_000, 3_000_000):
            rng = np.random.default_rng(seed); P = make_population(N, rng, **kw); T = true_estimands(P); K = 20 if N >= 1_000_000 else 10
            c = _cells(P, K); es = []; ov = []; it = []
            for r in range(reps):
                R = run_experiment(P, rng); es.append(cell_fast(R, c, 'piB')[0]); ov.append(cell_fast(R, c, 'ov')[0]); it.append(naive_estimators(R)['ITTdiff'])
            db = T['ITTdiff'] - T['tau_piB']
            print('%s N=%.1fM bias %.3f sd(piB) %.3f sd(ov) %.3f sd(ITT) %.3f bias/sd %.2f' % (reg, N / 1e6, db * 1e4, np.std(es) * 1e4, np.std(ov) * 1e4, np.std(it) * 1e4, db / np.std(es)))

def e11_lambda_bounds(N=1_000_000, seed=8):
    rng = np.random.default_rng(seed); P = make_population(N, rng, **CTR); T = true_estimands(P)
    seg = bucketize(P['X'][:, 0], 3) * 3 + bucketize(P['X'][:, 3], 3); l95, lmax = true_lambda(P, seg)
    print('Lambda95 %.2f max %.2f truth %.3f' % (l95, lmax, T['tau_piB'] * 1e4))
    for Lam in (1.0, 1.5, 2, 3, 5): lo, hi = lambda_bounds_limit(P, seg, Lam); print(' Lam %.1f [%.2f, %.2f]' % (Lam, lo * 1e4, hi * 1e4))

def e12_het_map(N=1_000_000, seed=9):
    for het in (0.25, 0.5, 1.0, 1.5):
        for dc in (0.0, 0.5, 0.9):
            rng = np.random.default_rng(seed); P = make_population(N, rng, het=het, delta_corr=dc, **CTR); T = true_estimands(P)
            print('het %.2f dc %.1f ITT %.2f piB %.2f piA %.2f ov %.2f est %.2f' % (het, dc, T['ITTdiff'] * 1e4, T['tau_piB'] * 1e4, T['tau_piA'] * 1e4, T['tau_ov'] * 1e4, cell_limit(P, _cells(P, 40))[0] * 1e4))

def e13_simex(N=400_000, seed=31):
    for ln in (0.2, 0.3, 0.5, 0.8):
        rng = np.random.default_rng(seed); P = make_population(N, rng, log_noise=ln, **CTR); T = true_estimands(P)
        naive = cell_limit(P, _cells(P, 40))[0]; sq, _ = simex(P, rng, ln, K=40, B=2); sr, _, _ = simex_rel(P, rng, ln, K=40, B=2)
        print('log_noise %.1f truth %.3f naive %.3f quad-SIMEX %.3f rel-SIMEX %.3f' % (ln, T['tau_piB'] * 1e4, naive * 1e4, sq * 1e4, sr * 1e4))

def e14_asymmetric_shock(N=600_000, seed=51):
    for cc in (0.0, 0.5, 1.0, 2.0):
        for cx in (0.0, 1.0):
            rng = np.random.default_rng(seed); P = make_population(N, rng, comp_x1=cx, **CTR)
            z = (P['mA_true'] - P['mA_true'].mean()) / P['mA_true'].std(); P['piA'] = 0.6 * expit(-0.4 + 1.5 * z - cc * P['comp'])
            T = true_estimands(P); c = _cells(P, 40)
            print('cc %.1f x1 %.1f piB truth %.3f est %.3f | ov truth %.3f est %.3f' % (cc, cx, T['tau_piB'] * 1e4, cell_limit(P, c)[0] * 1e4, T['tau_ov'] * 1e4, cell_limit(P, c, 'ov')[0] * 1e4))

def e15_K_selection(N=1_000_000, seed=52, reps=12):
    rng = np.random.default_rng(seed); P = make_population(N, rng, **CTR); T = true_estimands(P); Rs = [run_experiment(P, rng) for r in range(reps)]
    for K in (5, 10, 20, 30, 50):
        c = _cells(P, K); es = np.array([cell_fast(R, c, 'piB')[0] for R in Rs]); eo = np.array([cell_fast(R, c, 'ov')[0] for R in Rs])
        print('K %2d piB bias %.3f sd %.3f rmse %.3f | ov bias %.3f sd %.3f' % (K, (cell_limit(P, c)[0] - T['tau_piB']) * 1e4, es.std() * 1e4, np.sqrt(((es - T['tau_piB']) ** 2).mean()) * 1e4, (cell_limit(P, c, 'ov')[0] - T['tau_ov']) * 1e4, eo.std() * 1e4))


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        quick()
    else:
        print(__doc__)
        print('Usage: python 10_algorithmic_delivery_estimands.py quick | e4_coarseness | e5_confounder | e6_lognoise | e7_exclusion | e9_negative_control | e10_crossover | e11_lambda_bounds | e12_het_map | e13_simex | e14_asymmetric_shock | e15_K_selection')
    if len(sys.argv) > 1 and sys.argv[1] in globals() and sys.argv[1] != 'quick':
        globals()[sys.argv[1]]()
