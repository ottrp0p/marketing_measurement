"""Deep dive 22 — Price the prior (backlog BL75).

Where is an informative prior worth more than an extra dollar of excitation?

Builds on dive 20's cost-precision frontier  C * Var(c1) = kappa * S_eps(w) / 2,
i.e. Var = A / C with A = kappa*S_eps/2, and on dive 20's Psi ladder
(Psi = 1 with the response shape known, Psi ~ 10 with it free).

Self-contained.  `python 22_price_of_the_prior.py` runs everything;
`python 22_price_of_the_prior.py e1 e4` runs selected sections.

Sections
  e1   scalar shrinkage: MSE law, exchange rate, the bias ceiling
  e2   the dollar plane: C_eq, the constant exchange rate, the A/delta^2 cap
  e3   van Trees attainment identity (resolves dive 20's "inside the CRLB")
  e4   the continuous Psi ladder Psi(u) = (1+u)/(1+u-rho^2)   [MMM FIM]
  e5   dollar value of a shape prior + the half-value precision u_half
  e6   bias transfer from a mis-centred nuisance prior + its ceiling
  e7   shape-prior admissibility (break-even mis-centring)
  e8   negative / power controls
  e9   decision layer: deadband regret, bias vs variance at matched MSE
  e10  robustness sweep 1 — misspecification (parameterisation, bounded prior)
  e11  robustness sweep 2 — noise / horizon / seed sensitivity
  e12  ROI prior vs coefficient prior (Meridian-style reparameterisation)
"""
import sys, json, math
import numpy as np
from scipy.optimize import brentq, minimize, least_squares

RNG = np.random.default_rng(20260907)

# ============================== MMM MODEL CORE ==============================
# (same structural model as dives 20/21 so numbers are comparable)
TH0 = dict(beta=1.0, K=1.5, S=2.0, alpha=0.6)
M_MARGIN = 2.0
SIGMA = 0.05


def hill(a, K, S):
    a = np.maximum(a, 1e-12)
    return a ** S / (a ** S + K ** S)


def dhill(a, K, S):
    a = np.maximum(a, 1e-12)
    return S * (K ** S) * a ** (S - 1.0) / (a ** S + K ** S) ** 2


def d2hill(a, K, S, eps=1e-5):
    return (dhill(a + eps, K, S) - dhill(a - eps, K, S)) / (2 * eps)


def adstock(x, alpha, a0=None):
    a = np.empty_like(x, dtype=float)
    prev = (x[0] / (1 - alpha)) if a0 is None else a0
    for t in range(len(x)):
        prev = x[t] + alpha * prev
        a[t] = prev
    return a


def abar_opt(th=TH0, m=M_MARGIN):
    K, S, al, b = th['K'], th['S'], th['alpha'], th['beta']
    infl = K * ((S - 1) / (S + 1)) ** (1 / S) if S > 1 else 1e-6
    f = lambda a: m * b * dhill(a, K, S) - (1 - al)
    hi = infl
    while f(hi) > 0 and hi < 1e4:
        hi *= 1.5
    return brentq(f, infl, hi)


def seasonal_basis(T, harmonics=3, trend=1):
    t = np.arange(T)
    cols = [np.ones(T)]
    for d in range(1, trend + 1):
        cols.append(((t - t.mean()) / T) ** d)
    for k in range(1, harmonics + 1):
        cols.append(np.cos(2 * np.pi * k * t / 52.0))
        cols.append(np.sin(2 * np.pi * k * t / 52.0))
    return np.column_stack(cols)


def spend_path(T, period, amp, th=TH0, m=M_MARGIN, phase=0.0):
    """steady spend at the profit optimum plus a sinusoidal probe."""
    ab = abar_opt(th, m)
    x0 = ab * (1 - th['alpha'])
    t = np.arange(T)
    return x0 * (1.0 + amp * np.cos(2 * np.pi * t / period + phase))


PROBE_PERIODS = [11, 21, 34]      # dive-21 comb: coprime-ish, off the annual harmonics
PROBE_AMPS = [0.10, 0.10, 0.10]   # 12% spend CV in total


def main_design(T=156):
    """the canonical design used throughout this dive: a 3-tone comb probe."""
    return spend_path2(T, PROBE_PERIODS, PROBE_AMPS)


def mean_response(x, th, basis=None, gamma=None):
    a = adstock(x, th['alpha'])
    mu = th['beta'] * hill(a, th['K'], th['S'])
    if basis is not None:
        mu = mu + basis @ gamma
    return mu


def fim_structural(x, th=TH0, sigma=SIGMA, basis=None, params=('beta', 'alpha', 'K', 'S')):
    """Gaussian FIM for the structural MMM parameters (+ nuisance basis profiled out)."""
    eps = {'beta': 1e-5, 'alpha': 1e-5, 'K': 1e-5, 'S': 1e-5}
    cols = []
    for p in params:
        h = eps[p]
        tp, tm = dict(th), dict(th)
        tp[p] += h
        tm[p] -= h
        cols.append((mean_response(x, tp) - mean_response(x, tm)) / (2 * h))
    J = np.column_stack(cols)
    if basis is not None:
        # profile out the linear nuisance basis: residualise the score columns
        Q, _ = np.linalg.qr(basis)
        J = J - Q @ (Q.T @ J)
    return J.T @ J / sigma ** 2


ABAR_FIX = None   # set below: the fixed operating adstock anchoring the functional


def mroas_at(tt, abar, m=M_MARGIN):
    """marginal ROAS of a permanent +$1/wk at the CURRENT adstock level abar.
    NOTE: evaluated at the profit OPTIMUM this is identically 1 for every theta
    (that is the first-order condition), so its gradient vanishes and the functional
    is degenerate -- the decision functional must be anchored at an observed spend."""
    return m * tt['beta'] * dhill(abar, tt['K'], tt['S']) / (1 - tt['alpha'])


def xstar(tt, m=M_MARGIN):
    """profit-optimal steady weekly spend"""
    return abar_opt(tt, m) * (1 - tt['alpha'])


def func_grad(which='mroas', th=TH0, m=M_MARGIN, params=('beta', 'alpha', 'K', 'S'),
              abar=None, rel=0.85):
    """gradient of a DECISION functional wrt theta.
    which='mroas' : mROAS at a fixed operating adstock abar (default: rel x the
                    profit-optimal adstock of the true model -- an *observed* level,
                    held fixed as theta varies).
    which='xstar' : the profit-optimal steady spend.
    which='beta'  : the raw coefficient (for contrast).
    """
    if abar is None:
        abar = rel * abar_opt(th, m)
    if which == 'mroas':
        f = lambda tt: mroas_at(tt, abar, m)
    elif which == 'xstar':
        f = lambda tt: xstar(tt, m)
    else:
        f = lambda tt: tt['beta']
    g = []
    for p in params:
        h = 1e-4
        tp, tm = dict(th), dict(th)
        tp[p] += h
        tm[p] -= h
        g.append((f(tp) - f(tm)) / (2 * h))
    return np.array(g), f(th)


def mroas_grad(th=TH0, m=M_MARGIN, params=('beta', 'alpha', 'K', 'S')):
    return func_grad('mroas', th, m, params)


ABAR_FIX = 0.85 * abar_opt(TH0, M_MARGIN)


def spend_path2(T, periods, amps, th=TH0, m=M_MARGIN, phases=None):
    """multi-tone (comb) probe -- dive 21's design, needed to identify a 4-param shape."""
    ab = abar_opt(th, m)
    x0 = ab * (1 - th['alpha'])
    t = np.arange(T)
    if phases is None:
        phases = [0.0] * len(periods)
    z = np.ones(T)
    for P_, A_, ph in zip(periods, amps, phases):
        z = z + A_ * np.cos(2 * np.pi * t / P_ + ph)
    return x0 * z


# ======================= THE SHAPE-PRIOR LEDGER (core algebra) =======================
# Partition theta = (c, s): c = free parameters (no prior), s = shape/nuisance with a
# Gaussian prior of precision P_s, centred delta_s away from the truth.
# Decision functional  phi = g'theta  (here: mROAS).  Write
#     V_inf = g_c' I_cc^{-1} g_c                (variance with the shape KNOWN)
#     Itil  = I_ss - I_sc I_cc^{-1} I_cs        (profile / effective shape information)
#     d     = g_s - I_sc I_cc^{-1} g_c          (RESIDUAL SHAPE SENSITIVITY of the functional)
# Then, exactly:
#     V_post(P) = V_inf + d'(Itil+P)^{-1} d                       [posterior variance]
#     bias(P)   = - d'(Itil+P)^{-1} P delta_s                     [ceiling: -d'delta_s]
#     V_samp(P) = V_inf + d'(Itil+P)^{-1} Itil (Itil+P)^{-1} d     [sampling variance]
#     MSE       = V_samp + bias^2
# Psi(P) := V_post(P)/V_inf = 1 + d'(Itil+P)^{-1}d / V_inf  is dive 20's Psi ladder, now
# a continuous function of prior precision, with a SPECTRAL decomposition:
#     Psi(P) = 1 + sum_k rho_k^2/(1+u_k),  u_k = eig of Itil^{-1/2} P Itil^{-1/2},
#     rho_k^2 = (q_k' Itil^{-1/2} d)^2 / V_inf,   sum_k rho_k^2 = Psi(0)-1.
# Special case g_s = 0, scalar s:  Psi(u) = (1+u)/(1+u-rho^2), u = p/I_ss.

def ledger(I, g, ns, P=None, delta=None):
    """I: FIM (full). g: functional gradient. ns: number of trailing NUISANCE params.
    P: nuisance prior precision matrix (ns x ns). delta: nuisance mis-centring (ns,).
    Returns dict with V_inf, Itil, d, V_post, V_samp, bias, mse, Psi."""
    n = I.shape[0]
    nc = n - ns
    Icc, Ics, Iss = I[:nc, :nc], I[:nc, nc:], I[nc:, nc:]
    gc, gs = g[:nc], g[nc:]
    Iccinv_gc = np.linalg.solve(Icc, gc)
    V_inf = gc @ Iccinv_gc
    Itil = Iss - Ics.T @ np.linalg.solve(Icc, Ics)
    d = gs - Ics.T @ Iccinv_gc
    if P is None:
        P = np.zeros((ns, ns))
    M = Itil + P
    Minv_d = np.linalg.solve(M, d)
    V_post = V_inf + d @ Minv_d
    V_samp = V_inf + Minv_d @ (Itil @ Minv_d)
    bias = 0.0 if delta is None else -float(Minv_d @ (P @ np.asarray(delta)))
    return dict(V_inf=V_inf, Itil=Itil, d=d, V_post=V_post, V_samp=V_samp,
                bias=bias, mse=V_samp + bias ** 2, Psi=V_post / V_inf,
                Psi0=1 + d @ np.linalg.solve(Itil, d) / V_inf)


def spectral(I, g, ns):
    """rho_k^2 and the Itil eigen-basis: which shape directions the functional leaks into."""
    L = ledger(I, g, ns)
    Itil, d, V_inf = L['Itil'], L['d'], L['V_inf']
    w, Q = np.linalg.eigh(Itil)
    # whitened residual sensitivity
    z = (Q.T @ d) / np.sqrt(np.maximum(w, 1e-300))
    return w, Q, z ** 2 / V_inf          # rho_k^2 in the Itil eigenbasis


def report(name, d):
    print(f"\n--- {name} ---")
    for k, v in d.items():
        print(f"  {k:44s} {v}")


# ==================================================================== E1
def e1(nrep=200000):
    """Scalar Gaussian shrinkage: MSE law, exchange rate, the bias ceiling.

    y ~ N(theta, s2), prior N(mu0, tau2), posterior mean m = w*y + (1-w)*mu0.
    Claim (textbook, verified here as the base case):
        MSE(theta) = w^2 s2 + (1-w)^2 b^2 ,  b = theta - mu0
        prior beats MLE  <=>  b^2 < 2 tau^2 + s2
    """
    out = {}
    s2, tau2 = 1.0, 0.5
    w = tau2 / (tau2 + s2)
    for b in [0.0, 0.5, 1.0, 1.4142, 2.0]:
        y = RNG.normal(b, math.sqrt(s2), nrep)   # theta = b, mu0 = 0
        m = w * y
        mse_mc = np.mean((m - b) ** 2)
        se = np.std((m - b) ** 2, ddof=1) / math.sqrt(nrep)
        mse_th = w ** 2 * s2 + (1 - w) ** 2 * b ** 2
        out[f"b={b:.4f}  MSE mc / theory"] = f"{mse_mc:.5f} +- {se:.5f}  /  {mse_th:.5f}"
    # the ceiling: b^2 = 2 tau^2 + s2  -> MSE == s2 exactly
    bstar = math.sqrt(2 * tau2 + s2)
    mse_at_star = w ** 2 * s2 + (1 - w) ** 2 * bstar ** 2
    out["break-even b* = sqrt(2 tau^2 + s2)"] = f"{bstar:.6f}"
    out["MSE at b*  (should equal s2=1)"] = f"{mse_at_star:.10f}"
    # solve the crossing numerically as an independent check
    f = lambda b: w ** 2 * s2 + (1 - w) ** 2 * b ** 2 - s2
    out["numerical root of MSE(b)-s2"] = f"{brentq(f, 0.1, 10):.6f}"
    return out


# ==================================================================== E2
def e2():
    """The dollar plane.

    Dive 20 frontier: Var = A / C.  Information I_C = C / A.
    Prior precision p (= 1/tau^2).  Posterior-mean risk at mis-centring delta:
        MSE(C, p, delta) = (I_C + p^2 delta^2) / (I_C + p)^2
    Excitation-equivalent budget:  C_eq = A / MSE.
    Three prior-quality regimes:
      (i)   ORACLE (delta == 0 exactly):  C_eq = A (I_C+p)^2 / I_C  -- superlinear, unbounded.
      (ii)  CALIBRATED (E delta^2 = 1/p): E[MSE] = 1/(I_C+p) exactly, so
              C_eq = C + A*p    ->  EXCHANGE RATE dC/dp = A dollars per unit precision, CONSTANT.
      (iii) MIS-CENTRED by a fixed delta: p -> inf gives C_eq -> A/delta^2, a hard CAP, and the
              prior is worth NEGATIVE dollars once delta^2 > 2/p + A/C.
    """
    A = 0.25          # dollars * variance  (units: $ * (mROAS)^2)
    out = {}
    for C in [1.0, 4.0, 16.0]:
        IC = C / A
        for p in [1.0, 4.0, 16.0, 1e6]:
            for delta in [0.0, 0.1, 0.3]:
                mse = (IC + p ** 2 * delta ** 2) / (IC + p) ** 2
                Ceq = A / mse
                out[f"C={C:<5.1f} p={p:<9.1f} d={delta:.2f}  C_eq"] = f"{Ceq:12.4f}   value=${Ceq - C:10.4f}"
    out["--- (ii) CALIBRATED exchange rate: C + A*p vs A*(I_C+p) ---"] = ""
    for (C, p) in [(4.0, 7.0), (1.0, 0.5), (16.0, 40.0)]:
        IC = C / A
        mse_cal = 1.0 / (IC + p)          # = E_delta[MSE] with E[delta^2]=1/p (see e3)
        out[f"C={C:<5.1f} p={p:<5.1f}: C+A*p / A/E[MSE]"] = f"{C + A * p:.10f} / {A / mse_cal:.10f}"
    out["--- (i) ORACLE prior value at C=4 ---"] = ""
    for p in [1.0, 4.0, 16.0]:
        IC = 4.0 / A
        out[f"p={p:<5.1f}: oracle C_eq / calibrated C_eq"] = \
            f"{A * (IC + p) ** 2 / IC:.4f} / {4.0 + A * p:.4f}"
    out["--- cap check ---"] = ""
    for delta in [0.05, 0.1, 0.3, 1.0]:
        IC = 4.0 / A
        mse_inf = delta ** 2
        out[f"delta={delta:.2f}: cap A/delta^2 / limit of C_eq"] = \
            f"{A / delta ** 2:12.4f} / {A / mse_inf:12.4f}"
    out["--- sign flip: prior is worth negative dollars when ---"] = "delta^2 > 2/p + A/C"
    C, p, A_ = 4.0, 4.0, 0.25
    thr = math.sqrt(2 / p + A_ / C)
    for delta in [thr * 0.9, thr, thr * 1.1]:
        IC = C / A_
        mse = (IC + p ** 2 * delta ** 2) / (IC + p) ** 2
        out[f"delta={delta:.4f} (thr={thr:.4f}): C_eq - C"] = f"{A_ / mse - C:+.6f}"
    return out


# ==================================================================== E3
def e3(nrep=400000):
    """van Trees attainment identity.

    Two-parameter Gaussian linear model, theta = (c, s); FIM I; Gaussian prior with
    precision matrix P on s only (precision p, centre s0, mis-centring delta = s - s0).
    Penalised MLE:  thetahat = theta + J^{-1} score - J^{-1} P (theta - theta_prior),  J = I + P.

    IDENTITY (derived in the writeup, verified here):
        E_delta[ MSE_c ]  =  [J^{-1}]_cc     exactly, when E[delta^2] = 1/p.
    i.e. the prior-regularised estimator attains the van Trees bound 1/(I+P) on average
    over a CALIBRATED prior.  This resolves dive 20's "RMSE 0.229 inside CRLB 0.311":
    the CRLB is the wrong frontier for a biased estimator; 1/(I+P) is the right one.
    """
    out = {}
    for (Icc, Ics, Iss, p) in [(10.0, 6.0, 8.0, 3.0), (4.0, 3.5, 5.0, 12.0), (20.0, -9.0, 6.0, 1.0)]:
        I = np.array([[Icc, Ics], [Ics, Iss]])
        P = np.array([[0.0, 0.0], [0.0, p]])
        J = I + P
        Ji = np.linalg.inv(J)
        sandwich = Ji @ I @ Ji
        # Monte Carlo over delta ~ N(0, 1/p) and over the score
        L = np.linalg.cholesky(I)
        z = RNG.normal(size=(2, nrep))
        score = L @ z                                     # score ~ N(0, I)
        delta = RNG.normal(0.0, 1.0 / math.sqrt(p), nrep)  # calibrated prior
        dvec = np.vstack([np.zeros(nrep), delta])
        err = Ji @ score - Ji @ (P @ dvec)
        mse_c = np.mean(err[0] ** 2)
        se = np.std(err[0] ** 2, ddof=1) / math.sqrt(nrep)
        tag = f"I=({Icc},{Ics},{Iss}) p={p}"
        out[f"{tag}: E[MSE_c] mc"] = f"{mse_c:.6f} +- {se:.6f}   (z vs vanTrees = {(mse_c-Ji[0,0])/se:+.2f})"
        out[f"{tag}: [J^-1]_cc (van Trees)"] = f"{Ji[0, 0]:.6f}"
        out[f"{tag}: sandwich Var_c / bias^2 term"] = \
            f"{sandwich[0, 0]:.6f} / {p * Ji[0, 1] ** 2:.6f}  sum={sandwich[0,0] + p*Ji[0,1]**2:.10f}"
        out[f"{tag}: CRLB (unpenalised) [I^-1]_cc"] = f"{np.linalg.inv(I)[0, 0]:.6f}"
    return out


# ==================================================================== E4

# ==================================================================== E4
def e4():
    """The continuous Psi ladder for the DECISION functional (mROAS).

    Psi(P) = V_post(P)/V_inf = 1 + d'(Itil+P)^{-1}d / V_inf, verified against a direct
    matrix inverse, plus the spectral decomposition and the scalar special case
    Psi(u) = (1+u)/(1+u-rho^2).
    """
    out = {}
    # (a) algebraic check of the scalar special case on random FIMs
    errs = []
    for _ in range(200):
        Icc, Iss = RNG.uniform(0.5, 20, 2)
        rho = RNG.uniform(-0.98, 0.98)
        Ics = rho * math.sqrt(Icc * Iss)
        I = np.array([[Icc, Ics], [Ics, Iss]])
        g = np.array([1.0, 0.0])
        for u in [0.0, 0.1, 1.0, 10.0, 1000.0]:
            p = u * Iss
            L = ledger(I, g, 1, P=np.array([[p]]))
            errs.append(abs(L['Psi'] - (1 + u) / (1 + u - rho ** 2)) / L['Psi'])
    out["(a) scalar law: max rel err, 1000 random cells"] = f"{max(errs):.3e}"

    # (b) general identity V_post = g'(I+P)^-1 g on random 4-param FIMs with g_s != 0
    errs = []
    for _ in range(300):
        Araw = RNG.normal(size=(6, 4))
        I = Araw.T @ Araw + 0.05 * np.eye(4)
        g = RNG.normal(size=4)
        P = np.diag(RNG.uniform(0, 5, 3))
        Pfull = np.zeros((4, 4)); Pfull[1:, 1:] = P
        L = ledger(I, g, 3, P=P)
        direct = g @ np.linalg.solve(I + Pfull, g)
        errs.append(abs(L['V_post'] - direct) / abs(direct))
    out["(b) V_post identity: max rel err, 300 random 4-param cells"] = f"{max(errs):.3e}"

    # (c) the real MMM: functional = mROAS, free = beta, nuisance = (alpha, K, S)
    Tn, period, amp = 156, 26, 0.10
    x = main_design(Tn)
    B = seasonal_basis(Tn, harmonics=3, trend=1)
    params = ('beta', 'alpha', 'K', 'S')
    I = fim_structural(x, basis=B, params=params)
    sv = np.linalg.svd(I, compute_uv=False)
    out["(c) FIM svals (BL82 estimability audit)"] = " ".join(f"{s:.3e}" for s in sv)
    out["    condition number"] = f"{sv[0]/sv[-1]:.3e}"
    g, mroas0 = mroas_grad(params=params)
    L0 = ledger(I, g, 3)
    out["    mROAS / sd(mROAS) shape-known / shape-free"] = \
        f"{mroas0:.4f} / {math.sqrt(L0['V_inf']):.5f} / {math.sqrt(L0['V_post']):.5f}"
    out["    Psi(0) [= dive 20's ladder top]"] = f"{L0['Psi0']:.4f}"
    out["    residual shape sensitivity d (alpha,K,S)"] = " ".join(f"{v:+.4f}" for v in L0['d'])
    out["    raw g_s (alpha,K,S)"] = " ".join(f"{v:+.4f}" for v in g[1:])
    w, Q = np.linalg.eigh(L0['Itil'])
    zc = (Q.T @ L0['d']) / np.sqrt(np.abs(w))
    out["    Itil eigenvalues"] = " ".join(f"{v:.4e}" for v in w)
    out["    rho_k^2 per Itil eigendirection"] = " ".join(f"{v:.5f}" for v in zc ** 2 / L0['V_inf'])
    out["    sum rho_k^2 vs Psi0-1"] = f"{(zc**2/L0['V_inf']).sum():.6f} / {L0['Psi0']-1:.6f}"
    # (d) isotropic-in-Itil prior reproduces the scalar law exactly
    errs = []
    for u in [0.0, 0.1, 0.5, 1.0, 5.0, 100.0]:
        P = u * L0['Itil']
        L = ledger(I, g, 3, P=P)
        pred = 1 + (L0['Psi0'] - 1) / (1 + u)
        errs.append(abs(L['Psi'] - pred) / pred)
        out[f"    u={u:<7.2f} Psi (isotropic-in-Itil) / 1+(Psi0-1)/(1+u)"] = \
            f"{L['Psi']:.6f} / {pred:.6f}"
    out["    max rel err of the isotropic law"] = f"{max(errs):.3e}"
    return out


# ==================================================================== E5
def e5():
    """The dollar value of a shape prior, and the half-value precision.

    Dive 20: L* = m sigma sqrt(Psi T_r) is the optimal perpetual identification budget.
    V(u) = L*(0) - L*(u) = m sigma sqrt(T_r) [sqrt(Psi(0)) - sqrt(Psi(u))].
    With Psi(u) = 1 + (Psi0-1)/(1+u), the HALF-VALUE precision solves
        sqrt(Psi(u)) = (sqrt(Psi0)+1)/2  =>  u_half = (Psi0-1)/(r^2-1) - 1, r=(sqrt(Psi0)+1)/2.
    """
    out = {}
    m, sigma, Tr = M_MARGIN, SIGMA, 156.0
    scale = m * sigma * math.sqrt(Tr)
    out["dollar scale m*sigma*sqrt(T_r)"] = f"{scale:.5f}"

    def psi(u, P0):
        return 1 + (P0 - 1) / (1 + u)

    def uhalf(P0):
        r = (math.sqrt(P0) + 1) / 2
        return (P0 - 1) / (r ** 2 - 1) - 1

    for P0 in [1.5, 2.0, 5.0, 10.0, 20.0, 100.0]:
        uh = uhalf(P0)
        Vinf = scale * (math.sqrt(P0) - 1)
        Vh = scale * (math.sqrt(P0) - math.sqrt(psi(uh, P0)))
        out[f"Psi0={P0:<7.1f} u_half={uh:8.4f}"] = \
            f"V(inf)=${Vinf:8.4f}  V(u_half)=${Vh:8.4f}  ratio={Vh/Vinf:.8f}"
    out["--- diminishing returns: u for 50% / 90% / 99% of the value ---"] = ""
    for P0 in [2.0, 10.0, 100.0]:
        us = []
        for frac in [0.5, 0.9, 0.99]:
            f = lambda u: (math.sqrt(P0) - math.sqrt(psi(u, P0))) / (math.sqrt(P0) - 1) - frac
            us.append(brentq(f, 1e-12, 1e9))
        out[f"Psi0={P0:<7.1f}"] = f"u50={us[0]:.4f}  u90={us[1]:.4f}  u99={us[2]:.4f}   " \
                                  f"u90/u50={us[1]/us[0]:.2f}x  u99/u50={us[2]/us[0]:.1f}x"
    Tn = 156
    x = main_design(Tn)
    B = seasonal_basis(Tn, harmonics=3, trend=1)
    params = ('beta', 'alpha', 'K', 'S')
    I = fim_structural(x, basis=B, params=params)
    g, _ = mroas_grad(params=params)
    L0 = ledger(I, g, 3)
    P0 = L0['Psi0']
    out["--- MMM (T=156, 26-wk probe) ---"] = ""
    out["Psi0 / u_half / max prior value as a fraction of L*(0)"] = \
        f"{P0:.3f} / {uhalf(P0):.4f} / {1 - 1/math.sqrt(P0):.4f}"
    return out


# ==================================================================== E6
def e6(nrep=200):
    """Bias transfer from a mis-centred shape prior, and its ceiling.

        bias(phi) = - d'(Itil+P)^{-1} P delta ,  ceiling (P->inf) = - d' delta
    """
    out = {}
    Tn, period, amp = 156, 26, 0.10
    x = main_design(Tn)
    B = seasonal_basis(Tn, harmonics=3, trend=1)
    params = ('beta', 'alpha', 'K', 'S')
    I = fim_structural(x, basis=B, params=params)
    g, mroas0 = mroas_grad(params=params)
    L0 = ledger(I, g, 3)
    out["d (residual shape sensitivity)"] = " ".join(f"{v:+.4f}" for v in L0['d'])
    errs = []
    for _ in range(200):
        P = np.diag(RNG.uniform(0, 50, 3))
        delta = RNG.normal(0, 0.05, 3)
        L = ledger(I, g, 3, P=P, delta=delta)
        Pf = np.zeros((4, 4)); Pf[1:, 1:] = P
        dfull = np.concatenate([[0.0], delta])
        direct = -g @ np.linalg.solve(I + Pf, Pf @ dfull)
        errs.append(abs(L['bias'] - direct) / max(abs(direct), 1e-12))
    out["(a) bias identity max rel err (200 cells)"] = f"{max(errs):.3e}"
    for da in [0.02, 0.05, 0.10]:
        delta = np.array([da, 0.0, 0.0])
        row = []
        for u in [0.1, 1.0, 10.0, 1e6]:
            P = u * L0['Itil']
            row.append(ledger(I, g, 3, P=P, delta=delta)['bias'])
        out[f"delta_alpha={da:.2f}: bias(mROAS) at u=0.1/1/10/inf"] = \
            " ".join(f"{v:+.5f}" for v in row) + f"   ceiling={-L0['d'] @ delta:+.5f}"
        out[f"   as % of mROAS={mroas0:.3f}"] = \
            " ".join(f"{100*v/mroas0:+.2f}%" for v in row)

    Q, _ = np.linalg.qr(B)

    def fit(y, alpha0, p):
        def resid(z):
            th = dict(beta=z[0], alpha=z[1], K=z[2], S=z[3])
            r = y - mean_response(x, th)
            r = r - Q @ (Q.T @ r)
            return np.concatenate([r, [math.sqrt(p) * (z[1] - alpha0) * SIGMA]])
        z = least_squares(resid, [TH0['beta'], TH0['alpha'], TH0['K'], TH0['S']],
                          bounds=([0.05, 0.05, 0.2, 0.5], [5.0, 0.95, 6.0, 6.0])).x
        th = dict(beta=z[0], alpha=z[1], K=z[2], S=z[3])
        return mroas_at(th, ABAR_FIX)

    mu0 = mean_response(x, TH0)
    Itil_aa = L0['Itil'][0, 0]
    for da in [0.05]:
        for u in [1.0, 10.0]:
            p = u * Itil_aa
            vals = []
            for i in range(nrep):
                y = mu0 + np.random.default_rng(5000 + i).normal(0, SIGMA, Tn)
                vals.append(fit(y, TH0['alpha'] - da, p))
            vals = np.array(vals)
            bias_mc = vals.mean() - mroas0
            se = vals.std(ddof=1) / math.sqrt(len(vals))
            P = np.zeros((3, 3)); P[0, 0] = p
            pred = ledger(I, g, 3, P=P, delta=np.array([da, 0, 0]))['bias']
            out[f"(MC n={nrep}) da={da} u={u}: bias mc / pred"] = \
                f"{bias_mc:+.5f} +- {se:.5f}  /  {pred:+.5f}   (z={(bias_mc-pred)/se:+.2f})"
    return out


# ==================================================================== E7
def e7():
    """Admissibility: how wrong may a shape prior be and still pay?

        delta^2  <  2/p + 1/Itil   =   2 tau^2 + sd_data^2      (INDEPENDENT of d)
    Round-2 note: using the POSTERIOR variance instead of the sampling variance gives
    the wrong constant (1/p instead of 2/p).
    """
    out = {}
    Tn = 156
    x = main_design(Tn)
    B = seasonal_basis(Tn, harmonics=3, trend=1)
    params = ('beta', 'alpha')
    I = fim_structural(x, basis=B, params=params)
    g, mroas0 = mroas_grad(params=params)
    L0 = ledger(I, g, 1)
    Itil = L0['Itil'][0, 0]
    dsc = L0['d'][0]
    out["Itil (profile info on alpha) / sd_data(alpha)"] = f"{Itil:.4f} / {1/math.sqrt(Itil):.4f}"
    out["d (residual sensitivity of mROAS to alpha)"] = f"{dsc:+.4f}"
    out["Psi(0)"] = f"{L0['Psi0']:.4f}"
    base = L0['V_samp']
    for u in [0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1e4]:
        p = u * Itil
        f = lambda dd: ledger(I, g, 1, P=np.array([[p]]), delta=np.array([dd]))['mse'] - base
        dstar = brentq(f, 1e-10, 1e7)
        pred = math.sqrt(2 / p + 1 / Itil)
        tau = 1 / math.sqrt(p)
        z = dstar / math.sqrt(tau ** 2 + 1 / Itil)
        out[f"u={u:<9.1f} tau={tau:.4f}: delta* numeric / formula / z*"] = \
            f"{dstar:.6f} / {pred:.6f} / {z:.4f}"
    out["--- round-2 note: posterior-variance version gives the wrong constant ---"] = ""
    for u in [0.3, 3.0]:
        p = u * Itil
        f = lambda dd: (L0['V_inf'] + dsc ** 2 / (Itil + p) +
                        (dsc * p * dd / (Itil + p)) ** 2) - (L0['V_inf'] + dsc ** 2 / Itil)
        dpost = brentq(f, 1e-10, 1e7)
        out[f"u={u}: delta* from V_post / from V_samp"] = \
            f"{dpost:.6f} (formula sqrt(1/p+1/Itil)={math.sqrt(1/p+1/Itil):.6f}) / " \
            f"{math.sqrt(2/p+1/Itil):.6f}"
    return out


# ==================================================================== E8
def e8(nrep=400000):
    """Negative and power controls."""
    out = {}
    # N1: d = 0 exactly
    Icc = np.array([[5.0]])
    Ics = np.array([[2.0, -1.0]])
    Iss = np.array([[6.0, 1.0], [1.0, 4.0]])
    I = np.block([[Icc, Ics], [Ics.T, Iss]])
    gc = np.array([1.0])
    gs = (Ics.T @ np.linalg.solve(Icc, gc)).ravel()
    g = np.concatenate([gc, gs])
    L0 = ledger(I, g, 2)
    out["N1 d (should be 0)"] = " ".join(f"{v:+.3e}" for v in L0['d'])
    rows = []
    for u in [0.0, 1.0, 1e6]:
        L = ledger(I, g, 2, P=u * L0['Itil'], delta=np.array([0.5, -0.3]))
        rows.append((L['Psi'], L['mse'], L['bias']))
    out["N1 Psi / MSE / bias at u=0,1,1e6"] = \
        " | ".join(f"{a:.10f},{b:.10f},{c:+.2e}" for a, b, c in rows)
    # N2: prior along an Itil-orthogonal direction
    g = np.array([1.0, 0.3, 0.0])
    L0 = ledger(I, g, 2)
    Itil, dv = L0['Itil'], L0['d']
    y = np.linalg.solve(Itil, dv)              # (Itil)^-1 d
    n_orth = np.array([-y[1], y[0]]); n_orth /= np.linalg.norm(n_orth)
    n_par = y / np.linalg.norm(y)
    out["N2 n_orth . (Itil^-1 d)  (must be 0)"] = f"{n_orth @ y:+.3e}"
    out["N2 Psi with prior along n_orth (n'Itil^-1 d = 0)"] = \
        " ".join(f"{ledger(I,g,2,P=t*np.outer(n_orth,n_orth))['Psi']:.12f}"
                 for t in [0.0, 1.0, 100.0, 1e8])
    out["N2 (contrast) Psi with prior along n_par"] = \
        " ".join(f"{ledger(I,g,2,P=t*np.outer(n_par,n_par))['Psi']:.6f}"
                 for t in [0.0, 1.0, 100.0, 1e8])
    out["N3 Psi at P=0 (must equal Psi0)"] = \
        f"{ledger(I,g,2,P=np.zeros((2,2)))['Psi']:.12f} / {L0['Psi0']:.12f}"
    # N4 power near break-even
    Tn = 156
    x = main_design(Tn)
    B = seasonal_basis(Tn, harmonics=3, trend=1)
    Imm = fim_structural(x, basis=B, params=('beta', 'alpha'))
    gm, _ = mroas_grad(params=('beta', 'alpha'))
    Lm = ledger(Imm, gm, 1)
    Itm = Lm['Itil'][0, 0]
    p = 1.0 * Itm
    base = Lm['V_samp']
    dstar = brentq(lambda dd: ledger(Imm, gm, 1, P=np.array([[p]]),
                                     delta=np.array([dd]))["mse"] - base, 1e-10, 1e7)
    out["N4 break-even delta*_alpha at u=1"] = f"{dstar:.6f}"
    Lc = np.linalg.cholesky(Imm)
    J = Imm + np.array([[0.0, 0.0], [0.0, p]])
    Ji = np.linalg.inv(J)
    Iinv = np.linalg.inv(Imm)
    for mult in [0.6, 0.9, 1.0, 1.1, 1.5]:
        dd = dstar * mult
        z = RNG.normal(size=(2, nrep))
        e1_ = gm @ (Ji @ (Lc @ z) - Ji @ np.array([[0.0], [p * dd]]))
        z2 = RNG.normal(size=(2, nrep))
        e0_ = gm @ (Iinv @ (Lc @ z2))
        diff = np.mean(e1_ ** 2) - np.mean(e0_ ** 2)
        sed = math.sqrt(np.var(e1_ ** 2, ddof=1) / nrep + np.var(e0_ ** 2, ddof=1) / nrep)
        out[f"N4 delta={dd:.4f} ({mult:.2f}x): MSE_prior - MSE_flat"] = \
            f"{diff:+.4e} +- {sed:.2e}   (z={diff/sed:+.2f})"
    return out


def e9(nT=4000, nrep=3000):
    """Decision layer: is a dollar of bias as expensive as a dollar of variance?

    Dive 08's deadband: act only when |zhat| > h on the filtered deviation.  A prior
    contributes a PERSISTENT bias; excitation contributes TRANSIENT noise.  At matched
    MSE, which costs more decision regret?

    Model (dive 08 reduced form): true optimum z_t drifts as a random walk with
    innovation variance q; the analyst observes zhat_t = z_t + e_t; the action x_t is
    held until |zhat_t - x_t| > h, then set to zhat_t; per-period loss = (gamma/2)(x_t-z_t)^2
    plus a fixed cost K per move.
      arm A: e_t = b (constant bias),      MSE = b^2
      arm B: e_t ~ N(0, b^2) iid,          MSE = b^2
    """
    out = {}
    gamma, K, q = 1.0, 0.5, 0.01
    h = (12 * K * q / gamma) ** 0.25 - 0.5826 * math.sqrt(q)   # dive 08's law

    def run(mode, b, seed):
        rng = np.random.default_rng(seed)
        z = 0.0
        x = 0.0
        loss = 0.0
        moves = 0
        for t in range(nT):
            z += rng.normal(0, math.sqrt(q))
            e = b if mode == 'bias' else rng.normal(0, b)
            zhat = z + e
            if abs(zhat - x) > h:
                x = zhat
                moves += 1
            loss += 0.5 * gamma * (x - z) ** 2
        return loss / nT, moves * K / nT

    out["deadband h* (dive 08 law)"] = f"{h:.5f}"
    for b in [0.02, 0.05, 0.10, 0.20, 0.40]:
        rows = {}
        for mode in ['bias', 'none', 'noise']:
            bb = 0.0 if mode == 'none' else b
            mm = 'bias' if mode == 'bias' else 'noise'
            res = np.array([run(mm, bb, 900 + i) for i in range(12)])
            rows[mode] = (res[:, 0].mean(), res[:, 1].mean(),
                          (res[:, 0] + res[:, 1]).mean(),
                          (res[:, 0] + res[:, 1]).std(ddof=1) / math.sqrt(12))
        excess_bias = rows['bias'][2] - rows['none'][2]
        excess_noise = rows['noise'][2] - rows['none'][2]
        out[f"MSE=b^2, b={b:.2f}: excess regret bias / noise"] = \
            f"{excess_bias:.6f} / {excess_noise:.6f}   ratio pi={excess_bias/max(excess_noise,1e-12):.3f}"
        out[f"   (b={b:.2f}) move costs bias/noise/none"] = \
            f"{rows['bias'][1]:.5f} / {rows['noise'][1]:.5f} / {rows['none'][1]:.5f}"
    return out


# ==================================================================== E10


# ==================================================================== E10
def e10():
    """Robustness sweep 1 -- misspecification (reparameterisation, bounded prior,
    prior on the wrong shape parameter)."""
    out = {}
    Tn = 156
    x = main_design(Tn)
    B = seasonal_basis(Tn, harmonics=3, trend=1)
    params = ('beta', 'alpha', 'K', 'S')
    I = fim_structural(x, basis=B, params=params)
    g, mroas0 = mroas_grad(params=params)
    L0 = ledger(I, g, 3)
    a = TH0['alpha']
    dhl_da = math.log(2) / (a * math.log(a) ** 2)
    hl = math.log(2) / (-math.log(a))
    out["alpha / half-life / dhl_dalpha"] = f"{a:.3f} / {hl:.4f} wk / {dhl_da:.4f}"
    out["Psi0"] = f"{L0['Psi0']:.4f}"
    for sd_hl in [0.25, 0.5, 1.0, 2.0]:
        p_alpha = (dhl_da ** 2) / sd_hl ** 2
        P = np.zeros((3, 3)); P[0, 0] = p_alpha
        L = ledger(I, g, 3, P=P)
        out[f"(a) prior sd {sd_hl:.2f} wk on half-life -> sd_alpha={sd_hl/dhl_da:.4f}"] = \
            f"Psi={L['Psi']:.5f}  (L* multiplier {math.sqrt(L['Psi']/L0['Psi0']):.4f})"
    out["--- (b) uniform vs Gaussian prior on alpha ---"] = ""
    Q, _ = np.linalg.qr(B)
    mu0 = mean_response(x, TH0)

    def fit_bounded(y, lo, hi):
        def resid(z):
            th = dict(beta=z[0], alpha=z[1], K=z[2], S=z[3])
            r = y - mean_response(x, th)
            return r - Q @ (Q.T @ r)
        z = least_squares(resid, [TH0['beta'], TH0['alpha'], TH0['K'], TH0['S']],
                          bounds=([0.05, lo, 0.2, 0.5], [5.0, hi, 6.0, 6.0])).x
        th = dict(beta=z[0], alpha=z[1], K=z[2], S=z[3])
        return mroas_at(th, ABAR_FIX)

    sd_alpha_free = math.sqrt(np.linalg.inv(I)[1, 1])
    out["free sd(alpha) from FIM"] = f"{sd_alpha_free:.4f}"
    vals0 = []
    for i in range(150):
        y = mu0 + np.random.default_rng(7000 + i).normal(0, SIGMA, Tn)
        vals0.append(fit_bounded(y, 0.05, 0.95))
    vals0 = np.array(vals0)
    out["UNCONSTRAINED MC baseline sd(mROAS) / FIM prediction"] = \
        f"{vals0.std(ddof=1):.5f} +- {vals0.std(ddof=1)/math.sqrt(2*149):.5f} / " \
        f"{math.sqrt(L0['V_post']):.5f}"
    for w in [0.02, 0.05, 0.10]:
        p_eq = 3 / w ** 2
        P = np.zeros((3, 3)); P[0, 0] = p_eq
        Lg = ledger(I, g, 3, P=P)
        vals = []
        for i in range(150):
            y = mu0 + np.random.default_rng(7000 + i).normal(0, SIGMA, Tn)
            vals.append(fit_bounded(y, a - w, a + w))
        vals = np.array(vals)
        sd_mc = vals.std(ddof=1)
        se_sd = sd_mc / math.sqrt(2 * (len(vals) - 1))
        out[f"w={w:.2f} (w/sd_free={w/sd_alpha_free:.2f}): sd(mROAS) MC / Gaussian-matched"] = \
            f"{sd_mc:.5f} +- {se_sd:.5f} / {math.sqrt(Lg['V_post']):.5f}"
    out["--- (b2) MC estimate of the LADDER ITSELF (nonlinear check) ---"] = ""
    def fit_shape_known(y):
        def resid(z):
            th = dict(beta=z[0], alpha=TH0['alpha'], K=TH0['K'], S=TH0['S'])
            r = y - mean_response(x, th)
            return r - Q @ (Q.T @ r)
        z = least_squares(resid, [TH0['beta']], bounds=([0.05], [8.0])).x
        return mroas_at(dict(beta=z[0], alpha=TH0['alpha'], K=TH0['K'], S=TH0['S']), ABAR_FIX)
    vk = np.array([fit_shape_known(mu0 + np.random.default_rng(7000 + i).normal(0, SIGMA, Tn))
                   for i in range(150)])
    sd_known_mc, sd_free_mc = vk.std(ddof=1), vals0.std(ddof=1)
    out["sd(mROAS) MC shape-known / shape-free"] = \
        f"{sd_known_mc:.5f} +- {sd_known_mc/math.sqrt(298):.5f} / " \
        f"{sd_free_mc:.5f} +- {sd_free_mc/math.sqrt(298):.5f}"
    out["Psi0 MC / Psi0 FIM  (the FIM ladder is CONSERVATIVE)"] = \
        f"{(sd_free_mc/sd_known_mc)**2:.3f} / {L0['Psi0']:.3f}"
    out["FIM sd shape-known / MC sd shape-known"] = \
        f"{math.sqrt(L0['V_inf']):.5f} / {sd_known_mc:.5f}"
    out["--- (c) which shape parameter is worth pricing ---"] = ""
    for j, nm in enumerate(['alpha', 'K', 'S']):
        P = np.zeros((3, 3)); P[j, j] = 1e10
        L = ledger(I, g, 3, P=P)
        out[f"pin {nm} exactly -> Psi"] = \
            f"{L['Psi']:.4f}   (value captured " \
            f"{100*(math.sqrt(L0['Psi0'])-math.sqrt(L['Psi']))/(math.sqrt(L0['Psi0'])-1):.1f}%)"
    return out


# ==================================================================== E11
def e11():
    """Robustness sweep 2 -- design, noise, horizon, amplitude."""
    out = {}
    params = ('beta', 'alpha', 'K', 'S')
    g, _ = mroas_grad(params=params)
    for sigma in [0.02, 0.05, 0.10]:
        for Tn in [104, 156, 312]:
            x = main_design(Tn)
            B = seasonal_basis(Tn, harmonics=3, trend=1)
            I = fim_structural(x, sigma=sigma, basis=B, params=params)
            L = ledger(I, g, 3)
            uh = (L['Psi0'] - 1) / (((math.sqrt(L['Psi0']) + 1) / 2) ** 2 - 1) - 1
            out[f"sigma={sigma:.2f} T={Tn}: Psi0 / sd(mROAS) / u_half"] = \
                f"{L['Psi0']:8.3f} / {math.sqrt(L['V_post']):.5f} / {uh:.4f}"
    out["--- probe period sweep (T=156, sigma=0.05, amp=0.10) ---"] = ""
    for period in [13, 21, 26, 34, 52, 78]:
        x = spend_path(156, period, 0.10)
        B = seasonal_basis(156, harmonics=3, trend=1)
        I = fim_structural(x, basis=B, params=params)
        L = ledger(I, g, 3)
        sv = np.linalg.svd(I, compute_uv=False)
        out[f"period={period}: Psi0 / sd(mROAS) / cond(FIM)"] = \
            f"{L['Psi0']:10.3f} / {math.sqrt(L['V_post']):.5f} / {sv[0]/sv[-1]:.2e}"
    out["--- amplitude sweep (T=156, period=26) ---"] = ""
    for amp in [0.02, 0.05, 0.10, 0.20, 0.40]:
        x = spend_path(156, 26, amp)
        B = seasonal_basis(156, harmonics=3, trend=1)
        I = fim_structural(x, basis=B, params=params)
        L = ledger(I, g, 3)
        out[f"amp={amp:.2f}: Psi0 / sd(mROAS) / d"] = \
            f"{L['Psi0']:10.3f} / {math.sqrt(L['V_post']):.5f} / " + \
            " ".join(f"{v:+.3f}" for v in L['d'])
    return out


# ==================================================================== E12
def e12():
    """ROI prior vs coefficient prior (Meridian's reparameterisation), priced.

    A rank-one prior of precision p along a unit direction n gives, exactly,
        Var(P) = Var_flat - (g' I^{-1} n)^2 / (n' I^{-1} n + 1/p)
    so the value is governed by corr(g, n) under the sampling distribution.
    """
    out = {}
    Tn = 156
    x = main_design(Tn)
    B = seasonal_basis(Tn, harmonics=3, trend=1)
    params = ('beta', 'alpha', 'K', 'S')
    I = fim_structural(x, basis=B, params=params)
    Iinv = np.linalg.inv(I)
    g, mroas0 = mroas_grad(params=params)
    base = g @ Iinv @ g
    out["mROAS / sd(mROAS) flat"] = f"{mroas0:.5f} / {math.sqrt(base):.5f}"

    def var_dir(n, p):
        return base - (g @ Iinv @ n) ** 2 / (n @ Iinv @ n + 1.0 / p)

    def check(n, p):
        return g @ np.linalg.solve(I + p * np.outer(n, n), g)

    dirs = {
        'beta': (np.array([1.0, 0, 0, 0]), TH0['beta']),
        'alpha': (np.array([0, 1.0, 0, 0]), TH0['alpha']),
        'K': (np.array([0, 0, 1.0, 0]), TH0['K']),
        'mROAS': (g / np.linalg.norm(g), mroas0 / np.linalg.norm(g)),
    }
    errs = []
    for r in [0.5, 0.3, 0.2, 0.1, 0.05]:
        row = {}
        for nm, (n, scale_) in dirs.items():
            p = 1.0 / (r * scale_) ** 2
            v = var_dir(n, p)
            errs.append(abs(v - check(n, p)) / v)
            row[nm] = math.sqrt(max(v, 0))
        out[f"rel sd r={r:.2f}: sd(mROAS) for priors on " + "/".join(dirs)] = \
            " / ".join(f"{row[k]:.5f}" for k in dirs)
    out["rank-one identity max rel err"] = f"{max(errs):.3e}"
    out["--- L* multiplier (dollar value) at r=0.20 ---"] = ""
    r = 0.20
    for nm, (n, scale_) in dirs.items():
        p = 1.0 / (r * scale_) ** 2
        v = var_dir(n, p)
        out[f"   prior on {nm}"] = f"L* x {math.sqrt(v/base):.4f}   " \
                                   f"(saves {100*(1-math.sqrt(v/base)):.1f}% of the budget)"
    out["--- the governing law: value/V_flat = corr^2 * u/(1+u), u = p * n'I^-1 n ---"] = ""
    errs = []
    for nm, (n, scale_) in dirs.items():
        cc = (g @ Iinv @ n) / math.sqrt(base * (n @ Iinv @ n))
        row = []
        for u in [0.3, 1.0, 3.0, 10.0]:
            p = u / (n @ Iinv @ n)
            v = var_dir(n, p)
            pred = cc ** 2 * u / (1 + u)
            errs.append(abs((base - v) / base - pred))
            row.append(f"u={u}:{(base-v)/base:.4f}/{pred:.4f}")
        out[f"   {nm}: corr={cc:+.5f}  frac of Var removed (actual/law)"] = "  ".join(row)
    out["   max abs error of the corr^2 u/(1+u) law"] = f"{max(errs):.3e}"
    return out


# ==================================================================== E13
def e13(nrep=300):
    """ROUND 2 -- adversarial attacks on the round-1 ledger.

    A1  Nonlinearity: the bias-transfer law is a linearisation.  Compute the EXACT
        pseudo-true bias (fit the noiseless mean with alpha pinned at the prior centre)
        and compare with the linear ceiling -d'delta.
    A2  Calibration failure: the van Trees attainment identity assumes E[delta^2]=1/p.
        What if the analyst's stated tau is right on average but delta is heavy-tailed
        (or systematically larger)?  Price the miscalibration.
    A3  Experiment-derived priors are not zero-mean: dive 02's transport bias means
        E[delta] = t != 0.  Combined law and the effect on the cap.
    A4  Design envelope: the "value of the prior" V = L*(0)-L*(u) holds the design fixed.
        Re-optimise the probe given the prior and see whether the value grows.
    """
    out = {}
    Tn = 156
    x = main_design(Tn)
    B = seasonal_basis(Tn, harmonics=3, trend=1)
    params = ('beta', 'alpha', 'K', 'S')
    I = fim_structural(x, basis=B, params=params)
    g, mroas0 = func_grad('mroas', params=params)
    L0 = ledger(I, g, 3)
    Q, _ = np.linalg.qr(B)
    mu0 = mean_response(x, TH0)

    # ---- A1: exact pseudo-true bias with alpha pinned (the u -> inf limit)
    def pseudo_true(alpha_fixed):
        def resid(z):
            th = dict(beta=z[0], alpha=alpha_fixed, K=z[1], S=z[2])
            r = mu0 - mean_response(x, th)
            return r - Q @ (Q.T @ r)
        z = least_squares(resid, [TH0['beta'], TH0['K'], TH0['S']],
                          bounds=([0.05, 0.2, 0.5], [8.0, 8.0, 8.0])).x
        th = dict(beta=z[0], alpha=alpha_fixed, K=z[1], S=z[2])
        return mroas_at(th, ABAR_FIX), z

    out["A1 linear ceiling coefficient -d_alpha"] = f"{-L0['d'][0]:+.5f} per unit delta_alpha"
    for da in [0.005, 0.01, 0.02, 0.05, 0.10]:
        v, z = pseudo_true(TH0['alpha'] - da)
        exact = v - mroas0
        lin = -L0['d'][0] * da
        out[f"A1 delta_alpha={da:.3f}: exact / linear / ratio"] = \
            f"{exact:+.5f} / {lin:+.5f} / {exact/lin:.4f}   (pseudo-true beta,K,S=" + \
            " ".join(f"{q:.3f}" for q in z) + ")"

    # ---- A2: miscalibrated prior spread.  E[MSE] vs van Trees under kappa = E[delta^2]*p
    out["--- A2 calibration failure ---"] = ""
    Itil = L0['Itil']
    for u in [0.3, 1.0, 3.0]:
        P = u * Itil
        Lp = ledger(I, g, 3, P=P)
        # E[MSE] = V_samp + E[(d'(Itil+P)^-1 P delta)^2].  Take delta ~ N(0, k/p) isotropic
        M = np.linalg.solve(Itil + P, L0['d'])
        w = P @ M                        # bias = -w'delta
        flat = L0['V_samp']
        for k in [0.5, 1.0, 2.0, 4.0]:
            Sig = k * np.linalg.inv(P)
            emse = Lp['V_samp'] + w @ Sig @ w
            out[f"A2 u={u:<4.1f} k=E[d^2]p={k:.1f}: E[MSE]/V_post / vs flat"] = \
                f"{emse/Lp['V_post']:.4f} / {emse/flat:.4f}"
        out[f"A2 u={u:<4.1f} break-even k*"] = \
            f"{brentq(lambda kk: Lp['V_samp'] + kk*(w @ np.linalg.inv(P) @ w) - flat, 1e-9, 1e6):.4f}"

    # ---- A3: transport bias (non-zero-mean delta)
    out["--- A3 experiment-derived prior with transport bias t ---"] = ""
    for u in [1.0, 10.0]:
        P = u * Itil
        M = np.linalg.solve(Itil + P, L0['d'])
        w = P @ M
        Lp = ledger(I, g, 3, P=P)
        flat = L0['V_samp']
        for t in [0.0, 0.02, 0.05, 0.10]:
            tvec = np.array([t, 0.0, 0.0])
            Sig = np.linalg.inv(P)
            emse = Lp['V_samp'] + (w @ tvec) ** 2 + w @ Sig @ w
            out[f"A3 u={u:<5.1f} t={t:.2f}: E[MSE] / flat"] = \
                f"{emse:.5f} / {flat:.5f}   ({'PAYS' if emse < flat else 'HARMS'})"

    # ---- A4: design envelope -- re-optimise the probe amplitude split given the prior
    out["--- A4 design envelope (re-optimise the comb given the prior) ---"] = ""
    tot = sum(a ** 2 for a in PROBE_AMPS)      # keep total probe energy fixed

    def sd_for(split, u):
        amps = np.sqrt(np.maximum(split, 0) / max(np.sum(np.maximum(split, 0)), 1e-12) * tot)
        xx = spend_path2(Tn, PROBE_PERIODS, amps)
        II = fim_structural(xx, basis=B, params=params)
        LL = ledger(II, g, 3, P=u * (II[1:, 1:] - II[1:, :1] @ np.linalg.solve(II[:1, :1], II[:1, 1:])))
        return math.sqrt(LL['V_post'])
    for u in [0.0, 1.0, 10.0]:
        base_split = np.array([a ** 2 for a in PROBE_AMPS])
        best = minimize(lambda s: sd_for(np.abs(s), u), base_split, method='Nelder-Mead',
                        options=dict(maxiter=200, xatol=1e-4, fatol=1e-8))
        amps = np.sqrt(np.abs(best.x) / np.sum(np.abs(best.x)) * tot)
        out[f"A4 u={u:<5.1f}: sd fixed design / re-optimised / amps"] = \
            f"{sd_for(base_split,u):.5f} / {best.fun:.5f} / " + " ".join(f"{a:.3f}" for a in amps)
    return out


# ==================================================================== E14
def e14(nrep=20000):
    """ROUND 3 -- the second construction: DON'T TEST THE PRIOR, WIDEN IT.

    delta is unobservable, but the prior/data discrepancy D = shat_data - s0 is observed,
    with D = delta + noise, Var(noise) = 1/Itil.  Four rules:
       FLAT   : ignore the prior            (p = 0)
       FIXED  : use the stated precision p
       TEST   : use the stated p iff D^2 < 2 tau^2 + 2/Itil  (the plug-in admissibility test)
       EB     : widen the prior to tauhat^2 = max(tau^2, D^2 - 1/Itil)   [empirical Bayes]
    Compare frequentist risk (MSE of the functional) as a function of the TRUE delta.
    Claim: EB is uniformly safe -- its risk never exceeds the flat risk -- while TEST is not.
    """
    out = {}
    Tn = 156
    x = main_design(Tn)
    B = seasonal_basis(Tn, harmonics=3, trend=1)
    I = fim_structural(x, basis=B, params=('beta', 'alpha'))
    g, mroas0 = func_grad('mroas', params=('beta', 'alpha'))
    L0 = ledger(I, g, 1)
    Itil = L0['Itil'][0, 0]
    dsc = L0['d'][0]
    V_inf = L0['V_inf']
    sd_data = 1 / math.sqrt(Itil)
    out["Itil / sd_data(alpha) / d / Psi0"] = \
        f"{Itil:.3f} / {sd_data:.5f} / {dsc:+.4f} / {L0['Psi0']:.4f}"

    def risk_mc(delta, tau, rule, rng):
        p = 1.0 / tau ** 2
        # observed data-only estimate of alpha and the induced functional error
        # error of the functional = V_inf part (independent) + d * (alpha_hat_pen - alpha)
        eps = rng.normal(0, sd_data, nrep)          # alphahat_flat - alpha
        D = delta + eps                             # discrepancy vs the prior centre
        if rule == 'flat':
            pe = np.full(nrep, 0.0)
        elif rule == 'fixed':
            pe = np.full(nrep, p)
        elif rule == 'test':
            pe = np.where(D ** 2 < 2 * tau ** 2 + 2 / Itil, p, 0.0)
        elif rule == 'eb':
            tauh2 = np.maximum(tau ** 2, D ** 2 - 1 / Itil)
            pe = 1.0 / tauh2
        # penalised estimate of alpha: shrink alphahat_flat toward s0 = alpha - delta
        # alphahat_pen - alpha = (Itil*eps - pe*delta)/(Itil+pe)
        err_alpha = (Itil * eps - pe * delta) / (Itil + pe)
        # functional error: independent V_inf noise + d * err_alpha
        indep = rng.normal(0, math.sqrt(V_inf), nrep)
        err = indep + dsc * err_alpha
        return np.mean(err ** 2), np.std(err ** 2, ddof=1) / math.sqrt(nrep)

    tau = sd_data          # a prior as strong as the data (u = 1)
    out["prior sd tau (= sd_data, u=1)"] = f"{tau:.5f}"
    dstar = math.sqrt(2 * tau ** 2 + 1 / Itil)
    out["break-even |delta*|"] = f"{dstar:.5f}"
    rng = np.random.default_rng(4242)
    hdr = f"{'delta':>8} {'flat':>10} {'fixed':>10} {'test':>10} {'eb':>10}"
    out["risk table header"] = hdr
    for dmult in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        delta = dmult * dstar
        row = {}
        for rule in ['flat', 'fixed', 'test', 'eb']:
            r, s = risk_mc(delta, tau, rule, np.random.default_rng(1000 + int(dmult * 10)))
            row[rule] = (r, s)
        out[f"delta={delta:.4f} ({dmult:.1f}x d*)"] = \
            "  ".join(f"{row[k][0]:.5f}+-{row[k][1]:.5f}" for k in ['flat', 'fixed', 'test', 'eb'])
    out["--- worst-case (max over delta) risk ratio vs flat ---"] = ""
    grid = np.linspace(0, 6 * dstar, 40)
    worst = {}
    for rule in ['fixed', 'test', 'eb']:
        rr = []
        for delta in grid:
            r, _ = risk_mc(delta, tau, rule, np.random.default_rng(77))
            rr.append(r)
        flat, _ = risk_mc(0.0, tau, 'flat', np.random.default_rng(77))
        worst[rule] = (max(rr) / flat, grid[int(np.argmax(rr))] / dstar)
        out[f"   {rule}: max risk / flat risk (at delta/d*)"] = \
            f"{worst[rule][0]:.4f}  (at {worst[rule][1]:.2f}x d*)"
    out["--- average risk over delta ~ N(0, tau^2) (the calibrated case) ---"] = ""
    rng2 = np.random.default_rng(99)
    for rule in ['flat', 'fixed', 'test', 'eb']:
        ds = rng2.normal(0, tau, 400)
        rr = np.mean([risk_mc(d, tau, rule, np.random.default_rng(int(abs(d)*1e6) % 10**6))[0]
                      for d in ds])
        out[f"   {rule}"] = f"{rr:.6f}"
    return out


# ==================================================================== E15
def e15(nT=6000, nseeds=24):
    """ROUND 3 -- the decision layer, with the power control round 2 demanded.

    Dive 08's deadband: at MATCHED MSE, is a persistent bias cheaper or dearer than
    transient noise?  Controls:
      (i)  h = 0 (act always): bias and noise must cost the SAME (pi = 1).  [power control]
      (ii) quadratic-loss and move-cost components reported separately.
      (iii) Hill (nonlinear-profit) transport rather than the quadratic reduced form.
    """
    out = {}
    gamma, K, q = 1.0, 0.5, 0.01
    hstar = (12 * K * q / gamma) ** 0.25 - 0.5826 * math.sqrt(q)

    def run(mode, b, h, seed):
        rng = np.random.default_rng(seed)
        z = 0.0; xx = 0.0; ql = 0.0; moves = 0
        for t in range(nT):
            z += rng.normal(0, math.sqrt(q))
            e = b if mode == 'bias' else rng.normal(0, b)
            zhat = z + e
            if abs(zhat - xx) > h:
                xx = zhat; moves += 1
            ql += 0.5 * gamma * (xx - z) ** 2
        return ql / nT, moves * K / nT

    def cell(mode, b, h):
        r = np.array([run(mode, b, h, 900 + i) for i in range(nseeds)])
        tot = r[:, 0] + r[:, 1]
        return r[:, 0].mean(), r[:, 1].mean(), tot.mean(), tot.std(ddof=1) / math.sqrt(nseeds)

    out["deadband h* / q / K / gamma"] = f"{hstar:.5f} / {q} / {K} / {gamma}"
    out["EXACT INVARIANCE: a constant bias shifts zhat and x together, so the trigger "
        "statistic |zhat-x| is unchanged -> a biased estimate never causes churn."] = ""
    for h, lab in [(hstar, f"deadband h*={hstar:.4f}"), (0.0, "act-always h=0 [POWER CONTROL]")]:
        out[f"--- {lab} ---"] = ""
        base = cell('bias', 0.0, h)
        for b in [0.05, 0.10, 0.20, 0.40]:
            cb = cell('bias', b, h)
            cn = cell('noise', b, h)
            eb = cb[2] - base[2]; en = cn[2] - base[2]
            sed = math.sqrt(cb[3] ** 2 + cn[3] ** 2 + 2 * base[3] ** 2)
            tg = 'h*' if h > 0 else 'h0'
            out[f"  [{tg}] b={b:.2f}: excess bias / noise / pi"] = \
                f"{eb:.6f} / {en:.6f} / pi={eb/en:.3f} +- {sed/abs(en):.3f}"
            out[f"  [{tg}] b={b:.2f} quad bias/noise , move bias/noise"] = \
                f"{cb[0]-base[0]:.6f} / {cn[0]-base[0]:.6f}  ,  {cb[1]-base[1]:.6f} / {cn[1]-base[1]:.6f}"
    return out


# ==================================================================== E16
def e16(ndraw=2000):
    """ROUND 3 -- the estimability audit the clean-room demanded (BL82).

    The clean-room verifier reproduced every ladder number on the 4-parameter MMM FIM
    to 1e-13, but flagged that cond(I) = 3.8e5 with s_min = 3.0e-3, so Psi0 survives
    1e-14 jitter and does NOT survive 1e-4 relative jitter (it can go negative).
    Here: how much relative error in the Fisher matrix can each conclusion tolerate?
    Reported for the 4-param ladder, the 2-param (beta, alpha) ladder, and the
    direction law -- and for a longer record, which is the practical repair.
    """
    out = {}

    def jitter_psi(I, g, ns, rel, n=ndraw, seed=7):
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n):
            E = rng.normal(0, rel, I.shape)
            E = (E + E.T) / 2
            Ij = I * (1 + E)
            try:
                L = ledger(Ij, g, ns)
                vals.append(L['Psi0'])
            except np.linalg.LinAlgError:
                vals.append(np.nan)
        v = np.array(vals)
        ok = np.isfinite(v) & (v > 0)
        return v, ok

    for Tn, lab in [(156, 'T=156'), (312, 'T=312'), (520, 'T=520')]:
        x = main_design(Tn)
        B = seasonal_basis(Tn, harmonics=3, trend=1)
        for params, ns in [(('beta', 'alpha'), 1), (('beta', 'alpha', 'K', 'S'), 3)]:
            I = fim_structural(x, basis=B, params=params)
            g, _ = func_grad('mroas', params=params)
            sv = np.linalg.svd(I, compute_uv=False)
            L = ledger(I, g, ns)
            out[f"{lab} {len(params)}-param: Psi0 / cond / s_min"] = \
                f"{L['Psi0']:9.4f} / {sv[0]/sv[-1]:.3e} / {sv[-1]:.3e}"
            row = []
            for rel in [1e-8, 1e-6, 1e-4, 1e-3]:
                v, ok = jitter_psi(I, g, ns, rel)
                if ok.sum() == 0:
                    row.append(f"{rel:.0e}:DEAD")
                else:
                    row.append(f"{rel:.0e}:{np.median(v[ok]):.2f}"
                               f"[{np.percentile(v[ok],5):.2f},{np.percentile(v[ok],95):.2f}]"
                               f"({100*ok.mean():.0f}%ok)")
            out[f"   {lab} {len(params)}p Psi0 under relative FIM jitter"] = "  ".join(row)
    # the direction law is a ratio and should be far more robust
    out["--- robustness of the DIRECTION law (a ratio) under jitter ---"] = ""
    Tn = 156
    x = main_design(Tn)
    B = seasonal_basis(Tn, harmonics=3, trend=1)
    params = ('beta', 'alpha', 'K', 'S')
    I = fim_structural(x, basis=B, params=params)
    g, mroas0 = func_grad('mroas', params=params)
    rng = np.random.default_rng(11)
    for rel in [1e-6, 1e-4, 1e-3]:
        ratios = []
        for _ in range(400):
            E = rng.normal(0, rel, I.shape); E = (E + E.T) / 2
            Ij = I * (1 + E)
            try:
                Ii = np.linalg.inv(Ij)
                bs = g @ Ii @ g
                nb = np.array([1.0, 0, 0, 0])
                p_b = 1 / (0.2 * TH0['beta']) ** 2
                vb = bs - (g @ Ii @ nb) ** 2 / (nb @ Ii @ nb + 1 / p_b)
                p_r = 1 / (0.2 * mroas0) ** 2
                vr = bs - (g @ Ii @ g) ** 2 / (g @ Ii @ g + 1 / p_r)
                if vb > 0 and vr > 0 and bs > 0:
                    ratios.append((math.sqrt(vb / bs), math.sqrt(vr / bs)))
            except np.linalg.LinAlgError:
                pass
        R = np.array(ratios)
        out[f"   rel={rel:.0e}: L* mult beta / mROAS (median [5,95])"] = \
            f"{np.median(R[:,0]):.4f}[{np.percentile(R[:,0],5):.4f},{np.percentile(R[:,0],95):.4f}] / " \
            f"{np.median(R[:,1]):.4f}[{np.percentile(R[:,1],5):.4f},{np.percentile(R[:,1],95):.4f}]"
    return out


SECTIONS = dict(e1=e1, e2=e2, e3=e3, e4=e4, e5=e5, e6=e6, e7=e7, e8=e8,
                e9=e9, e10=e10, e11=e11, e12=e12, e13=e13, e14=e14, e15=e15, e16=e16)

if __name__ == '__main__':
    want = [a for a in sys.argv[1:] if a in SECTIONS] or list(SECTIONS)
    allout = {}
    for k in want:
        d = SECTIONS[k]()
        report(k, d)
        allout[k] = d
    tag = '_'.join(want) if len(want) < 4 else 'all'
    with open(f"22_results_{tag}.json", 'w') as f:
        json.dump(allout, f, indent=1)
