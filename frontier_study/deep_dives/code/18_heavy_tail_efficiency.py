"""Dive 18 — Semiparametric efficiency for lift under zero-inflated heavy tails.
Reproduces experiments E1–E14 of frontier_study/deep_dives/18_heavy_tail_efficiency.md.

Usage:  python 18_heavy_tail_efficiency.py quick   (~3 min: E1(60 reps), E2, E4, E5(n=200k), E8, E9, E11, E14)
        python 18_heavy_tail_efficiency.py all     (~15 min: adds E3, E5(n=1M), E7, E10, E12, E13 seed sweep, 200-rep E1)
Needs numpy + scipy.  Results are printed and written to 18_results_<mode>.json next to this file.
Note: exp5/E5_E6 uses CFG with lift_p; E8/E9 use CFG5 without lift_p (lift passed explicitly).
"""
import sys, json, time, os
import numpy as np
from scipy import stats, integrate
from scipy.special import expit

# ---------------- DGP ----------------
# Y = B * V ; B ~ Bern(p_i), V positive heavy-tailed. Strata: never / always / influenced (monotone).
# Effect models on the intensive margin:
#   kappa : influenced buyers' spend = kappa * (always-buyer draw)  (composition effect)
#   gamma : always-buyers' spend scaled by (1+gamma) under treatment (scale effect)
#   tail  : only top-q spenders (by latent draw) scaled by (1+g_tail) under treatment (tail effect)

def draw_V(rng, n, tail="lognormal", s=1.27, df=3, alpha=1.5, m_target=140.0):
    """Positive spend with mean ~ m_target (for finite-mean tails)."""
    if tail == "lognormal":
        z = rng.standard_normal(n)
        v = np.exp(s * z)
        v *= m_target / np.exp(s**2 / 2)
    elif tail == "logt":
        z = rng.standard_t(df, n) * s / np.sqrt(df / (df - 2))  # match log-sd s
        v = np.exp(z)
        # normalize by MC mean of the law (fixed seed constant)
        v *= m_target / _logt_mean(s, df)
    elif tail == "pareto":
        # Lomax/Pareto II with tail index alpha, scale chosen for mean m_target when alpha>1
        u = rng.random(n)
        xm = m_target * (alpha - 1) / alpha
        v = xm * u ** (-1.0 / alpha)
    elif tail == "loglogistic":
        # log V ~ logistic(0, s*sqrt(3)/pi) so log-sd = s; Fisher location info = 1/(3 scale^2)
        sc = s * np.sqrt(3) / np.pi
        v = np.exp(rng.logistic(0, sc, n))
        v *= m_target / _loglog_mean(sc)
    else:
        raise ValueError(tail)
    return v

_cache = {}
def _logt_mean(s, df):
    key = ("logt", s, df)
    if key not in _cache:
        r = np.random.default_rng(12345)
        z = r.standard_t(df, 4_000_000) * s / np.sqrt(df / (df - 2))
        _cache[key] = np.exp(z).mean()
    return _cache[key]

def _loglog_mean(sc):
    key = ("ll", sc)
    if key not in _cache:
        r = np.random.default_rng(54321)
        _cache[key] = np.exp(r.logistic(0, sc, 4_000_000)).mean()
    return _cache[key]

def simulate(rng, n_arm, p0=0.05, lift_p=0.05, kappa=1.0, gamma=0.0, tail="lognormal",
             s=1.27, df=3, alpha=1.5, m0=140.0, tail_q=None, g_tail=0.0,
             pre=False, sd_u=1.5, b_u=0.5, return_strata=False, K_pre=1):
    """Return dict with Y0, Y1 (control / treatment arms, n_arm each) and optionally P0,P1 (pre-period).
    Latent propensity u ~ N(0,1): purchase prob expit(a + sd_u u) (mean p0), spend log-shift b_u u.
    Influenced buyers (extra p1-p0 share) spend kappa * (a draw from the always-buyer spend law)."""
    p1 = p0 * (1 + lift_p)
    a = _logit_intercept(p0, sd_u)
    out = {}
    for arm, n in (("0", n_arm), ("1", n_arm)):
        u = rng.standard_normal(n)
        pa = expit(a + sd_u * u)
        always = rng.random(n) < pa
        infl = (~always) & (rng.random(n) < np.minimum(1.0, (p1 - p0) / (1 - pa)))
        base = draw_V(rng, n, tail, s, df, alpha, m0)
        shift = np.exp(b_u * u - 0.5 * b_u**2)
        # influenced buyers get a u-shift resampled from the always-buyers (so kappa=1 => F_I = F_A)
        if infl.any():
            idx = rng.choice(np.flatnonzero(always), infl.sum(), replace=True)
            shift = shift.copy(); shift[infl] = shift[idx]
        v = base * shift
        if arm == "0":
            Y = np.where(always, v, 0.0)
        else:
            vt = v.copy()
            vt[always] *= (1 + gamma)
            if tail_q is not None and g_tail != 0:
                thr = np.quantile(v[always], 1 - tail_q)
                vt[always & (v > thr)] *= (1 + g_tail)
            Y = np.where(always, vt, np.where(infl, kappa * v, 0.0))
        out["Y" + arm] = Y
        if return_strata:
            out["S" + arm] = np.where(always, 1, np.where(infl, 2, 0))
        if pre:
            P = np.zeros(n)
            for _k in range(K_pre):
                bp = rng.random(n) < pa
                vp = draw_V(rng, n, tail, s, df, alpha, m0) * np.exp(b_u * u - 0.5 * b_u**2)
                P += np.where(bp, vp, 0.0)
            out["P" + arm] = P
    return out

def m_always(p0=0.05, sd_u=1.5, b_u=0.5, m0=140.0):
    """E[V | always-buyer] = m0 * E[exp(b_u u - b_u^2/2) | always]."""
    a = _logit_intercept(p0, sd_u)
    x, w = np.polynomial.hermite_e.hermegauss(80); w = w / w.sum()
    pa = expit(a + sd_u * x)
    return m0 * (w * pa * np.exp(b_u * x - 0.5 * b_u**2)).sum() / (w * pa).sum()

def _logit_intercept(p, sd):
    # find a such that E[expit(a + sd*Z)] = p  (Gauss-Hermite)
    x, w = np.polynomial.hermite_e.hermegauss(60)
    w = w / w.sum()
    lo, hi = -15, 15
    for _ in range(60):
        a = (lo + hi) / 2
        if (w * expit(a + sd * x)).sum() < p:
            lo = a
        else:
            hi = a
    return (lo + hi) / 2

def true_tau(p0, lift_p, kappa, gamma, m0=140.0, tail_q=None, g_tail=0.0, tail="lognormal", s=1.27, df=3, alpha=1.5, sd_u=1.5, b_u=0.5, mA=None):
    """Population ATE (per user): (p1-p0) kappa mA + p0 gamma mA (+ tail-effect term by MC)."""
    p1 = p0 * (1 + lift_p)
    if mA is None: mA = m_always(p0, sd_u, b_u, m0)
    tau = (p1 - p0) * kappa * mA + p0 * gamma * mA
    if tail_q is not None and g_tail != 0:
        r = np.random.default_rng(999)
        d = simulate(r, 2_000_000, p0, 0.0, 1.0, 0.0, tail, s, df, alpha, m0, sd_u=sd_u, b_u=b_u)
        v = d["Y0"][d["Y0"] > 0]
        thr = np.quantile(v, 1 - tail_q)
        tau += p0 * g_tail * v[v > thr].mean() * tail_q
    return tau

# ---------------- estimators ----------------
def dim(Y0, Y1):
    t = Y1.mean() - Y0.mean()
    se = np.sqrt(Y1.var(ddof=1) / len(Y1) + Y0.var(ddof=1) / len(Y0))
    return t, se

def cuped(Y0, Y1, P0, P1):
    Y = np.concatenate([Y0, Y1]); P = np.concatenate([P0, P1])
    th = np.cov(Y, P)[0, 1] / P.var(ddof=1)
    A0 = Y0 - th * (P0 - P.mean()); A1 = Y1 - th * (P1 - P.mean())
    return dim(A0, A1)

def winsor_dim(Y0, Y1, c):
    return dim(np.minimum(Y0, c), np.minimum(Y1, c))

def hl_log_shift(V0, V1, nmax=None, rng=None):
    """Hodges-Lehmann two-sample shift of log V1 vs log V0: the d solving U(logV1 - d, logV0) = n0 n1 / 2
    (Mann-Whitney inversion by bisection; O(n log n) per step)."""
    a = np.sort(np.log(V0)); b = np.log(V1)
    na, nb = len(a), len(b)
    def U(d):
        # number of pairs with (b_j - d) > a_i
        return np.searchsorted(a, b - d, side="left").sum() - na * nb / 2
    lo, hi = np.median(b) - np.median(a) - 1.0, np.median(b) - np.median(a) + 1.0
    while U(lo) < 0: lo -= 1.0
    while U(hi) > 0: hi += 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if U(mid) > 0: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def median_log_shift(V0, V1):
    return np.median(np.log(V1)) - np.median(np.log(V0))

def restricted_ext(Y0, Y1):
    """Pure extensive-margin estimator: (p1-p0) * pooled positive mean.  Assumes F1+ = F0+."""
    n0, n1 = len(Y0), len(Y1)
    p0 = (Y0 > 0).mean(); p1 = (Y1 > 0).mean()
    V = np.concatenate([Y0[Y0 > 0], Y1[Y1 > 0]])
    m = V.mean()
    t = (p1 - p0) * m
    var = m**2 * (p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0) + (p1 - p0) ** 2 * V.var(ddof=1) / len(V)
    return t, np.sqrt(var), p0, p1, m

def restricted_scale(Y0, Y1, shift="median", rng=None):
    """Scale-family estimator: tau = m0 * (p1 (1+g) - p0), g from rank/median log shift on positives."""
    n0, n1 = len(Y0), len(Y1)
    p0 = (Y0 > 0).mean(); p1 = (Y1 > 0).mean()
    V0 = Y0[Y0 > 0]; V1 = Y1[Y1 > 0]
    d = hl_log_shift(V0, V1, rng=rng) if shift == "hl" else median_log_shift(V0, V1)
    g = np.exp(d) - 1
    # pooled m0: rescale treated positives back
    Vp = np.concatenate([V0, V1 / (1 + g)])
    m0 = Vp.mean()
    t = m0 * (p1 * (1 + g) - p0)
    return t, g, p0, p1, m0

def kappa_isolation(Y0, Y1, kappa_prior_mean, kappa_prior_sd, gamma=0.0):
    """tau(kappa) = m0[(p1-p0) kappa + p0 gamma]; combine prior on kappa with raw kappa-hat.
    Returns posterior mean/sd of tau and the raw kappa-hat with its se."""
    n0, n1 = len(Y0), len(Y1)
    p0 = (Y0 > 0).mean(); p1 = (Y1 > 0).mean()
    V0 = Y0[Y0 > 0]; m0 = V0.mean(); s0 = V0.std(ddof=1)
    m1 = Y1[Y1 > 0].mean(); s1 = Y1[Y1 > 0].std(ddof=1)
    dp = p1 - p0
    # raw kappa-hat from means (heavy-tailed):  kappa = (p1 m1 - p0 m0 - p0 gamma m0)/((p1-p0) m0)
    num = p1 * m1 - p0 * m0 * (1 + gamma)
    khat = num / (dp * m0)
    # delta-method se of khat: dominated by m1, m0 variance
    var_num = (p1**2 * s1**2 / (Y1 > 0).sum() + (p0 * (1 + gamma))**2 * s0**2 / (Y0 > 0).sum()
               + m1**2 * p1 * (1 - p1) / n1 + (m0 * (1 + gamma))**2 * p0 * (1 - p0) / n0)
    se_k = np.sqrt(var_num) / abs(dp * m0)
    # posterior on kappa (normal-normal)
    w = (1 / kappa_prior_sd**2) / (1 / kappa_prior_sd**2 + 1 / se_k**2)
    k_post = w * kappa_prior_mean + (1 - w) * khat
    sd_post = np.sqrt(1 / (1 / kappa_prior_sd**2 + 1 / se_k**2))
    # tau posterior (approx; dp, m0 uncertainty added)
    var_dp = p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0
    tau = m0 * (dp * k_post + p0 * gamma)
    var_tau = (m0 * dp) ** 2 * sd_post**2 + (m0 * k_post) ** 2 * var_dp + (dp * k_post) ** 2 * s0**2 / len(V0)
    return tau, np.sqrt(var_tau), khat, se_k, k_post, sd_post

# ---------------- bounds ----------------
def hahn_bound_nocov(Y0, Y1):
    return Y1.var() / len(Y1) + Y0.var() / len(Y0)

def hahn_bound_cov(Y0, Y1, X0, X1, nbins=50):
    """Hahn bound with covariate X via binning: E[s1^2(X)/n1 + s0^2(X)/n0] + Var(tau(X))/n."""
    X = np.concatenate([X0, X1])
    edges = np.unique(np.quantile(X, np.linspace(0, 1, nbins + 1)))
    b0 = np.clip(np.searchsorted(edges, X0, side="right") - 1, 0, len(edges) - 2)
    b1 = np.clip(np.searchsorted(edges, X1, side="right") - 1, 0, len(edges) - 2)
    n0, n1 = len(Y0), len(Y1)
    tot = 0.0
    taus = []; ws = []
    for k in range(len(edges) - 1):
        y0 = Y0[b0 == k]; y1 = Y1[b1 == k]
        if len(y0) < 2 or len(y1) < 2: continue
        w = (len(y0) + len(y1)) / (n0 + n1)
        tot += w * (y1.var() / n1 + y0.var() / n0)
        taus.append(y1.mean() - y0.mean()); ws.append(w)
    taus = np.array(taus); ws = np.array(ws)
    tb = (ws * taus).sum()
    tot += (ws * (taus - tb) ** 2).sum() / (n0 + n1)
    return tot

# ---------------- Round 3: rank-identified marginal-buyer basket ratio ----------------
def rank_kappa(Y0, Y1, gamma=0.0, use_hl=False):
    """Scale-mixture model: F1+ = w0 F_A(y/(1+gamma)) + w1 F_A(y/kappa), w0=p0/p1.  With gamma given,
    log kappa = (E log V1 - E log V0 - w0 log(1+gamma)) / w1  (exact under the model; tail-free).
    Returns kappa_hat, se(log kappa), and the tail-free relative lift L = (p1/p0)(w0 gamma + w1 kappa) - 1 ... i.e.
    L = (p1-p0)/p0 * kappa + gamma, with its se, plus tau_hat = mA_hat * p0 * L."""
    n0, n1 = len(Y0), len(Y1)
    p0 = (Y0 > 0).mean(); p1 = (Y1 > 0).mean()
    V0 = Y0[Y0 > 0]; V1 = Y1[Y1 > 0]
    l0 = np.log(V0); l1 = np.log(V1)
    w0 = p0 / p1; w1 = 1 - w0
    if use_hl:
        shift = hl_log_shift(V0, V1)
        # HL variance approx via normal-density plug-in on pooled log-sd
        s = np.r_[l0, l1].std(); var_shift = np.pi * s**2 / 3 * (1 / len(l0) + 1 / len(l1))
    else:
        shift = l1.mean() - l0.mean(); var_shift = l1.var(ddof=1) / len(l1) + l0.var(ddof=1) / len(l0)
    logk = (shift - w0 * np.log(1 + gamma)) / w1
    # se of log kappa: dominated by var_shift / w1^2 ; w1 uncertainty adds (logk)^2 var(w1)/w1^2
    var_w1 = (p0 / p1) ** 2 * (p0 * (1 - p0) / n0 / p0**2 + p1 * (1 - p1) / n1 / p1**2)
    var_logk = var_shift / w1**2 + logk**2 * var_w1 / w1**2
    kappa = np.exp(logk)
    dp = p1 - p0
    L = dp / p0 * kappa + gamma
    var_dp_over_p0 = (p1 / p0) ** 2 * (p1 * (1 - p1) / n1 / p1**2 + p0 * (1 - p0) / n0 / p0**2)  # var of p1/p0
    var_L = (dp / p0 * kappa) ** 2 * var_logk + kappa**2 * var_dp_over_p0
    mA = V0.mean()
    tau = mA * p0 * L
    # log-scale CI for the extensive part (dp/p0)*kappa when gamma=0: var log = var_logk + var(p1/p0)/(dp/p0)^2
    var_logE = var_logk + var_dp_over_p0 / (dp / p0) ** 2 if dp > 0 else np.inf
    return dict(kappa=kappa, se_logk=np.sqrt(var_logk), L=L, se_L=np.sqrt(var_L), tau=tau, se_tau=mA * p0 * np.sqrt(var_L), shift=shift, w1=w1,
                se_logE=np.sqrt(var_logE), E=dp / p0 * kappa)

def spec_test_positives(Y0, Y1):
    """Mann-Whitney test of F1+ = F0+ (tail-free specification test of the pure extensive-margin restriction).
    Returns z statistic."""
    from scipy.stats import mannwhitneyu
    V0 = Y0[Y0 > 0]; V1 = Y1[Y1 > 0]
    u = mannwhitneyu(V1, V0, alternative='two-sided')
    n0, n1 = len(V0), len(V1)
    mu = n0 * n1 / 2; sd = np.sqrt(n0 * n1 * (n0 + n1 + 1) / 12)
    return (u.statistic - mu) / sd, u.pvalue

def rank_kappa_post(Y0, Y1, prior_mu=0.0, prior_sd=0.5, gamma=0.0, ngrid=801):
    """Posterior over log kappa on a grid: likelihood from (shift, p0hat, p1hat) treated as independent normals,
    with shift ~ N(w1(kappa) log kappa + w0 log(1+gamma), var_shift), w1 = 1 - p0/p1 evaluated at the *true* p's,
    which are integrated over their sampling distributions by plugging a grid over dp as well (2-D grid, marginalised).
    Prior: log kappa ~ N(prior_mu, prior_sd^2).  Returns posterior median/sd of log kappa, of L, and tau."""
    n0, n1 = len(Y0), len(Y1)
    p0 = (Y0 > 0).mean(); p1 = (Y1 > 0).mean()
    V0 = Y0[Y0 > 0]; V1 = Y1[Y1 > 0]
    l0 = np.log(V0); l1 = np.log(V1)
    shift = l1.mean() - l0.mean(); var_shift = l1.var(ddof=1) / len(l1) + l0.var(ddof=1) / len(l0)
    dp_hat = p1 - p0; var_dp = p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0
    lk = np.linspace(prior_mu - 4 * prior_sd, prior_mu + 4 * prior_sd, ngrid)
    dps = dp_hat + np.sqrt(var_dp) * np.linspace(-4, 4, 161)
    dps = dps[dps > 0]
    LK, DP = np.meshgrid(lk, dps, indexing='ij')
    w1 = DP / (p0 + DP)
    mu_shift = w1 * LK + (1 - w1) * np.log(1 + gamma)
    loglik = -0.5 * (shift - mu_shift) ** 2 / var_shift - 0.5 * (DP - dp_hat) ** 2 / var_dp
    logprior = -0.5 * (LK - prior_mu) ** 2 / prior_sd**2
    post = np.exp(loglik + logprior - (loglik + logprior).max())
    post /= post.sum()
    L = DP / p0 * np.exp(LK) + gamma
    # marginals
    pk = post.sum(axis=1); cdf = np.cumsum(pk)
    lk_med = lk[np.searchsorted(cdf, 0.5)]; lk_sd = np.sqrt((pk * (lk - (pk * lk).sum()) ** 2).sum())
    Lf = L.ravel(); pf = post.ravel(); o = np.argsort(Lf); cL = np.cumsum(pf[o])
    L_med = Lf[o][np.searchsorted(cL, 0.5)]; L_lo = Lf[o][np.searchsorted(cL, 0.025)]; L_hi = Lf[o][np.searchsorted(cL, 0.975)]
    L_mean = (pf * Lf).sum(); L_sd = np.sqrt((pf * (Lf - L_mean) ** 2).sum())
    mA = V0.mean()
    return dict(logk_med=lk_med, logk_sd=lk_sd, L_med=L_med, L_mean=L_mean, L_sd=L_sd, L_lo=L_lo, L_hi=L_hi, tau_med=mA * p0 * L_med, tau_mean=mA * p0 * L_mean, tau_sd=mA * p0 * L_sd, mA=mA, p0=p0)

def rank_kappa_lognormal_I(Y0, Y1):
    """Second-log-moment corrected kappa: assumes F_I lognormal; kappa_arith = exp(mu_I - mu_A + (s_I^2 - s_A^2)/2),
    with mu_I, s_I^2 from the mixture's first two log-moments (tail-free but divided by w1)."""
    p0 = (Y0 > 0).mean(); p1 = (Y1 > 0).mean(); w0 = p0 / p1; w1 = 1 - w0
    l0 = np.log(Y0[Y0 > 0]); l1 = np.log(Y1[Y1 > 0])
    m1A, m2A = l0.mean(), (l0**2).mean()
    m1I = (l1.mean() - w0 * m1A) / w1; m2I = ((l1**2).mean() - w0 * m2A) / w1
    sI2 = max(m2I - m1I**2, 0.0); sA2 = m2A - m1A**2
    return np.exp(m1I - m1A + (sI2 - sA2) / 2), m1I - m1A, sI2, sA2

# ==================== from exp1.py ====================
BASE = dict(p0=0.05, lift_p=0.05, kappa=1.0, gamma=0.0, sd_u=2.5, b_u=0.7, K_pre=26)
mA = m_always(0.05, 2.5, 0.7)
TAU = true_tau(0.05, 0.05, 1, 0, sd_u=2.5, b_u=0.7)
print("mA", mA, "tau", TAU)

def run_E1(n_arm=200_000, reps=200, seed=0, gamma=0.0, kappa=1.0, **kw):
    rng = np.random.default_rng(seed)
    cfg = dict(BASE); cfg.update(kw); cfg['gamma'] = gamma; cfg['kappa'] = kappa
    tau = true_tau(cfg['p0'], cfg['lift_p'], kappa, gamma, sd_u=cfg['sd_u'], b_u=cfg['b_u'])
    rows = {k: [] for k in ["dim","cuped","w99","w999","w9999","ext","scale_med","scale_hl","hahn_se","hahn_cov_se","ext_se","dim_se","kiso1","kiso1_se","rank","post"]}
    for r in range(reps):
        d = simulate(rng, n_arm, pre=True, **cfg)
        Y0, Y1, P0, P1 = d['Y0'], d['Y1'], d['P0'], d['P1']
        t, se = dim(Y0, Y1); rows['dim'].append(t); rows['dim_se'].append(se)
        rows['cuped'].append(cuped(Y0, Y1, P0, P1)[0])
        Yall = np.r_[Y0, Y1]
        for q, k in [(0.99, 'w99'), (0.999, 'w999'), (0.9999, 'w9999')]:
            c = np.quantile(Yall[Yall > 0], q)
            rows[k].append(winsor_dim(Y0, Y1, c)[0])
        e = restricted_ext(Y0, Y1); rows['ext'].append(e[0]); rows['ext_se'].append(e[1])
        rows['scale_med'].append(restricted_scale(Y0, Y1, 'median')[0])
        rows['scale_hl'].append(restricted_scale(Y0, Y1, 'hl')[0])
        rows['hahn_se'].append(np.sqrt(hahn_bound_nocov(Y0, Y1)))
        rows['hahn_cov_se'].append(np.sqrt(hahn_bound_cov(Y0, Y1, P0, P1)))
        ki = kappa_isolation(Y0, Y1, 1.0, 0.25); rows['kiso1'].append(ki[0]); rows['kiso1_se'].append(ki[1])
        rows['rank'].append(rank_kappa(Y0, Y1)['tau']); rows['post'].append(rank_kappa_post(Y0, Y1)['tau_med'])
    out = {"n_arm": n_arm, "reps": reps, "tau": tau}
    for k in ["dim","cuped","w99","w999","w9999","ext","scale_med","scale_hl","kiso1","rank","post"]:
        a = np.array(rows[k])
        out[k] = dict(mean=a.mean(), bias=a.mean()-tau, sd=a.std(ddof=1), rmse=np.sqrt(((a-tau)**2).mean()),
                      sd_mcerr=a.std(ddof=1)/np.sqrt(2*(reps-1)), t_ratio=tau/a.std(ddof=1))
    for k in ["hahn_se","hahn_cov_se","ext_se","dim_se","kiso1_se"]:
        out[k] = float(np.mean(rows[k]))
    return out


# ==================== from exp2.py ====================
"""E2: the (1+s^2)e^{-s^2} law — tail-free restricted bounds vs Hahn as log-sd of positives grows.
   E12: HL efficiency check on log-logistic (known I_F)."""

def E2(n_arm=200_000, reps=60, seed=1, s_list=(0.5, 0.9, 1.27, 1.6, 2.0)):
    out = []
    for s in s_list:
        rng = np.random.default_rng(seed)
        # no u-heterogeneity in spend so log-sd of positives = s exactly
        cfg = dict(p0=0.05, lift_p=0.05, kappa=1.0, gamma=0.0, sd_u=2.5, b_u=0.0, s=s)
        tau = true_tau(0.05, 0.05, 1, 0, sd_u=2.5, b_u=0.0)
        est = {k: [] for k in ["dim", "ext", "scale_hl", "hahn"]}
        for r in range(reps):
            d = simulate(rng, n_arm, **cfg)
            Y0, Y1 = d['Y0'], d['Y1']
            est['dim'].append(dim(Y0, Y1)[0]); est['ext'].append(restricted_ext(Y0, Y1)[0])
            est['scale_hl'].append(restricted_scale(Y0, Y1, 'hl')[0]); est['hahn'].append(np.sqrt(hahn_bound_nocov(Y0, Y1)))
        sd = {k: np.std(v, ddof=1) for k, v in est.items() if k != 'hahn'}
        m = 140.0; p = 0.05; n = n_arm
        pred_ratio_ext = np.exp(-s**2)              # Var_ext / Var_hahn (lognormal, kappa=1)
        pred_ratio_scale_eff = (1 + s**2) * np.exp(-s**2)   # efficient scale-model bound
        pred_ratio_scale_hl = (1 + (np.pi / 3) * s**2) * np.exp(-s**2)  # HL attains pi/3 of I_F^-1
        out.append(dict(s=s, tau=tau, sd_dim=sd['dim'], sd_ext=sd['ext'], sd_scale_hl=sd['scale_hl'],
                        hahn_se=float(np.mean(est['hahn'])),
                        obs_ratio_ext=sd['ext']**2 / sd['dim']**2, pred_ratio_ext=pred_ratio_ext,
                        obs_ratio_scale=sd['scale_hl']**2 / sd['dim']**2, pred_ratio_scale_hl=pred_ratio_scale_hl,
                        pred_ratio_scale_eff=pred_ratio_scale_eff, mc_rel_err=1/np.sqrt(2*(reps-1))*2))
        print(out[-1])
    return out

def E12(n_arm=200_000, reps=100, seed=2):
    """HL log-shift variance vs efficient bound (n0+n1)/(n0 n1 I_F) on log-logistic (I_F = 1/(3 sc^2)) and lognormal (I_F = 1/s^2)."""
    res = {}
    for tail in ["loglogistic", "lognormal"]:
        rng = np.random.default_rng(seed)
        hl = []; md = []; npos = []
        for r in range(reps):
            d = simulate(rng, n_arm, p0=0.05, lift_p=0.0, kappa=1.0, gamma=0.05, tail=tail, s=1.27, sd_u=2.5, b_u=0.0)
            V0 = d['Y0'][d['Y0'] > 0]; V1 = d['Y1'][d['Y1'] > 0]
            hl.append(hl_log_shift(V0, V1)); md.append(median_log_shift(V0, V1)); npos.append((len(V0), len(V1)))
        n0 = np.mean([a for a, b in npos]); n1 = np.mean([b for a, b in npos])
        s = 1.27
        if tail == "loglogistic":
            sc = s * np.sqrt(3) / np.pi; I = 1 / (3 * sc**2)
            # density of logistic at 0: 1/(4 sc); int f^2 = 1/(6 sc)
            var_hl = (n0 + n1) / (12 * n0 * n1 * (1 / (6 * sc))**2)
            var_med = (n0 + n1) / (n0 * n1) / (4 * (1 / (4 * sc))**2)
        else:
            I = 1 / s**2
            var_hl = (n0 + n1) / (12 * n0 * n1) * 4 * np.pi * s**2
            var_med = (n0 + n1) / (n0 * n1) * np.pi * s**2 / 2
        bound = (n0 + n1) / (n0 * n1 * I)
        res[tail] = dict(true_shift=np.log(1.05), mean_hl=np.mean(hl), sd_hl=np.std(hl, ddof=1), pred_sd_hl=np.sqrt(var_hl),
                         mean_med=np.mean(md), sd_med=np.std(md, ddof=1), pred_sd_med=np.sqrt(var_med), eff_bound_sd=np.sqrt(bound),
                         ARE_hl=bound / np.var(hl, ddof=1), ARE_med=bound / np.var(md, ddof=1), mc_rel_err=1/np.sqrt(2*(reps-1)))
        print(tail, res[tail])
    return res


# ==================== from exp3.py ====================
"""E3: infinite-variance regime (Pareto tail index alpha<2): DIM vs restricted estimators; rate & coverage."""

def E3(alphas=(1.5, 1.8, 2.5), ns=(50_000, 200_000, 800_000), reps=100, seed=3):
    out = []
    for alpha in alphas:
        for n in ns:
            rng = np.random.default_rng(seed)
            tau = true_tau(0.05, 0.05, 1, 0, sd_u=2.5, b_u=0.0)  # mean of V is 140 regardless of tail
            e = {k: [] for k in ["dim", "ext", "scale_hl", "cov_dim", "cov_ext", "w99"]}
            for r in range(reps):
                d = simulate(rng, n, p0=0.05, lift_p=0.05, tail="pareto", alpha=alpha, sd_u=2.5, b_u=0.0)
                Y0, Y1 = d['Y0'], d['Y1']
                t, se = dim(Y0, Y1); e['dim'].append(t); e['cov_dim'].append(abs(t - tau) < 1.96 * se)
                te, see = restricted_ext(Y0, Y1)[:2]; e['ext'].append(te); e['cov_ext'].append(abs(te - tau) < 1.96 * see)
                e['scale_hl'].append(restricted_scale(Y0, Y1, 'hl')[0])
                Yall = np.r_[Y0, Y1]; c = np.quantile(Yall[Yall > 0], 0.99); e['w99'].append(winsor_dim(Y0, Y1, c)[0])
            a = np.array(e['dim'])
            out.append(dict(alpha=alpha, n_arm=n, tau=tau,
                            dim_sd=a.std(ddof=1), dim_mad_sd=1.4826 * np.median(np.abs(a - np.median(a))), dim_kurt=stats.kurtosis(a),
                            dim_cov=np.mean(e['cov_dim']), dim_bias=a.mean() - tau,
                            ext_sd=np.std(e['ext'], ddof=1), ext_cov=np.mean(e['cov_ext']), ext_bias=np.mean(e['ext']) - tau,
                            scale_sd=np.std(e['scale_hl'], ddof=1), scale_bias=np.mean(e['scale_hl']) - tau,
                            w99_sd=np.std(e['w99'], ddof=1), w99_bias=np.mean(e['w99']) - tau))
            print({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out[-1].items()})
    # rate exponents: log sd vs log n
    for alpha in alphas:
        rows = [o for o in out if o['alpha'] == alpha]
        ln = np.log([o['n_arm'] for o in rows])
        for k in ['dim_sd', 'dim_mad_sd', 'ext_sd', 'scale_sd', 'w99_sd']:
            sl = np.polyfit(ln, np.log([o[k] for o in rows]), 1)[0]
            print(f"alpha {alpha} {k} rate exponent {sl:.3f}  (theory DIM: -min(1-1/alpha,0.5) = {-min(1-1/alpha,0.5):.3f}; restricted: -0.5)")
    return out


# ==================== from exp4.py ====================
"""E4: the winsorization path — effect-location profile e(c), variance v(c), retention index R(c);
   flat-prior lemma regret = MSE/2; regret vs MSE under informative priors (Parikh et al. divergence)."""

def profiles(n=3_000_000, seed=4, p0=0.05, lift_p=0.05, sd_u=2.5, b_u=0.7):
    """Population quantities via one huge always-buyer sample: returns dict of c-grid, e(c) for three effect models, v(c)."""
    rng = np.random.default_rng(seed)
    d = simulate(rng, n, p0=p0, lift_p=0.0, sd_u=sd_u, b_u=b_u)
    V = d['Y0'][d['Y0'] > 0]  # always-buyer spend law F_A
    mA = V.mean(); p1 = p0 * (1 + lift_p); dp = p1 - p0
    cs = np.quantile(V, [0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999]); cs = np.r_[cs, np.inf]
    thr90 = np.quantile(V, 0.9)
    res = dict(c_quantiles=[0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999, 1.0], c=cs.tolist(), mA=mA)
    models = {
        "extensive_k1": dict(kappa=1.0, gamma=0.0, g_tail=0.0),      # marginal buyers like always-buyers
        "extensive_k05": dict(kappa=0.5, gamma=0.0, g_tail=0.0),     # marginal buyers spend half
        "scale_g": dict(kappa=1.0, gamma=0.05, g_tail=0.0),           # 5% scale effect on always-buyers, no extensive
        "tail_top10": dict(kappa=1.0, gamma=0.0, g_tail=0.5),        # top-decile spenders +50%
    }
    for name, m in models.items():
        e_c = []; v_c = []
        for c in cs:
            Wc = np.minimum(V, c)
            # tau_c = dp*E[min(kV,c)] (extensive; note scale/tail models use lift_p=0 => dp=0 for them below)
            if name.startswith("extensive"):
                tau_c = dp * np.minimum(m['kappa'] * V, c).mean()
                tau = dp * m['kappa'] * mA
            elif name == "scale_g":
                tau_c = p0 * (np.minimum((1 + m['gamma']) * V, c).mean() - Wc.mean()); tau = p0 * m['gamma'] * mA
            else:
                Vt = V.copy(); Vt[V > thr90] *= (1 + m['g_tail'])
                tau_c = p0 * (np.minimum(Vt, c).mean() - Wc.mean()); tau = p0 * (Vt.mean() - mA)
            e_c.append(tau_c / tau)
            v_c.append(p0 * (Wc**2).mean() - (p0 * Wc.mean())**2)   # per-user variance of min(Y,c) under control
        e_c = np.array(e_c); v_c = np.array(v_c)
        R = e_c / np.sqrt(v_c / v_c[-1])
        res[name] = dict(tau=tau, e_c=e_c.tolist(), v_c=v_c.tolist(), retention=R.tolist(), best_c_index=int(np.argmax(R)))
    return res

def regret_flat(b, sig):
    """Exact regret for go/no-go at threshold 0 with tau ~ flat prior, tauhat ~ N(tau+b, sig^2), loss |tau|."""
    f1 = integrate.quad(lambda d: d * stats.norm.cdf(-(d + b) / sig), 0, 40 * sig + abs(b))[0]
    f2 = integrate.quad(lambda d: -d * stats.norm.cdf((d + b) / sig), -40 * sig - abs(b), 0)[0]
    return f1 + f2

def regret_prior(mu, s, sig, atten=1.0, bias=0.0, thr=0.0):
    """Regret for decision go iff tauhat > thr; tau ~ N(mu, s^2); tauhat ~ N(atten*tau + bias, sig^2); loss |tau - thr|.
    (thr enters loss as the breakeven.)"""
    def integrand(t):
        pgo = 1 - stats.norm.cdf((thr - atten * t - bias) / sig)
        wrong = (1 - pgo) if t > thr else pgo
        return abs(t - thr) * wrong * stats.norm.pdf(t, mu, s)
    return integrate.quad(integrand, mu - 10 * s, mu + 10 * s, limit=200)[0]


# ==================== from exp5.py ====================
"""E5: composition attack — marginal buyers spend kappa x always-buyers; bias of restricted estimators vs kappa;
   E6: the rank-identified kappa estimator (Round 3) and the tail-free relative lift; E7: spec-test power."""

def E5_E6(n_arm=200_000, reps=100, seed=5, kappas=(0.3, 0.5, 1.0, 1.5), gamma=0.0):
    CFG = dict(p0=0.05, lift_p=0.05, sd_u=2.5, b_u=0.7)
    out = []
    for k in kappas:
        rng = np.random.default_rng(seed)
        tau = true_tau(0.05, 0.05, k, gamma, sd_u=2.5, b_u=0.7)
        L_true = 0.05 * k + gamma
        e = {n: [] for n in ["dim", "ext", "scale_hl", "kiso", "rank_k", "rank_L", "rank_seL", "rank_tau", "rank_setau", "rank_khat", "rank_selogk", "z_spec"]}
        for r in range(reps):
            d = simulate(rng, n_arm, kappa=k, gamma=gamma, **CFG)
            Y0, Y1 = d['Y0'], d['Y1']
            e['dim'].append(dim(Y0, Y1)[0]); e['ext'].append(restricted_ext(Y0, Y1)[0]); e['scale_hl'].append(restricted_scale(Y0, Y1, 'hl')[0])
            e['kiso'].append(kappa_isolation(Y0, Y1, 1.0, 0.25)[0])
            rk = rank_kappa(Y0, Y1, gamma=gamma)
            e['rank_tau'].append(rk['tau']); e['rank_setau'].append(rk['se_tau']); e['rank_L'].append(rk['L']); e['rank_seL'].append(rk['se_L'])
            e['rank_khat'].append(rk['kappa']); e['rank_selogk'].append(rk['se_logk'])
            e['z_spec'].append(spec_test_positives(Y0, Y1)[0])
        row = dict(kappa=k, gamma=gamma, tau=tau, L_true=L_true, n_arm=n_arm)
        for n in ["dim", "ext", "scale_hl", "kiso", "rank_tau"]:
            a = np.array(e[n]); row[n] = dict(bias=a.mean() - tau, sd=a.std(ddof=1), rmse=np.sqrt(((a - tau) ** 2).mean()), bias_over_sd=(a.mean() - tau) / a.std(ddof=1), bias_mcse=a.std(ddof=1) / np.sqrt(reps))
        a = np.array(e['rank_khat']); row['rank_kappa'] = dict(mean=a.mean(), median=np.median(a), sd=a.std(ddof=1), mean_se_logk=np.mean(e['rank_selogk']), sd_logk=np.std(np.log(a), ddof=1))
        a = np.array(e['rank_L']); row['rank_L'] = dict(mean=a.mean(), sd=a.std(ddof=1), mean_se=np.mean(e['rank_seL']), cover=np.mean(np.abs(a - L_true) < 1.96 * np.array(e['rank_seL'])))
        row['rank_tau']['mean_se'] = float(np.mean(e['rank_setau']))
        z = np.array(e['z_spec']); row['spec'] = dict(mean_z=z.mean(), power_5pct=np.mean(np.abs(z) > 1.96))
        out.append(row); print(json.dumps(row, default=float))
    return out


# ==================== from exp7.py ====================
"""E7: attacks on the rank-kappa estimator — (a) influenced buyers are a *truncated* (selected) part of F_A, not a scaled copy;
   (b) an unmodelled intensive effect gamma on always-buyers; (c) the joint (kappa,gamma) identified set (rank line vs moment line).
   Also heavier tails (Pareto alpha=1.5) as robustness for rank-kappa and coverage with log-scale CI."""

def simulate_trunc(rng, n_arm, q_trunc=0.5, **cfg):
    """Like simulate() but influenced buyers' spend is drawn from F_A truncated below its q_trunc quantile
    (marginal buyers are the *small* buyers by selection)."""
    d = simulate(rng, n_arm, kappa=1.0, gamma=0.0, return_strata=True, **cfg)
    Y1, S1 = d['Y1'], d['S1']
    VA = d['Y0'][d['Y0'] > 0]
    thr = np.quantile(VA, q_trunc)
    small = VA[VA <= thr]
    idx = np.flatnonzero(S1 == 2)
    Y1 = Y1.copy(); Y1[idx] = rng.choice(small, len(idx), replace=True)
    kappa_eff = small.mean() / VA.mean()
    return d['Y0'], Y1, kappa_eff

def E7(n_arm=200_000, reps=100, seed=7):
    CFG = dict(p0=0.05, lift_p=0.05, sd_u=2.5, b_u=0.7)
    out = {}
    # (a) truncation attack
    rows = []
    for q in (0.5, 0.8):
        rng = np.random.default_rng(seed)
        e = {k: [] for k in ["dim", "rank_tau", "rank_k", "keff", "se_tau", "cover_logE", "E"]}
        for r in range(reps):
            Y0, Y1, keff = simulate_trunc(rng, n_arm, q, **CFG)
            e['keff'].append(keff); e['dim'].append(dim(Y0, Y1)[0])
            rk = rank_kappa(Y0, Y1); e['rank_tau'].append(rk['tau']); e['rank_k'].append(rk['kappa']); e['se_tau'].append(rk['se_tau'])
            Etrue = 0.05 * keff; e['E'].append(rk['E'])
            e['cover_logE'].append(abs(np.log(rk['E']) - np.log(Etrue)) < 1.96 * rk['se_logE'])
        keff = np.mean(e['keff']); tau = true_tau(0.05, 0.05, keff, 0, sd_u=2.5, b_u=0.7)
        a = np.array(e['rank_tau'])
        rows.append(dict(q_trunc=q, kappa_eff=keff, tau=tau, dim_bias=np.mean(e['dim']) - tau, dim_sd=np.std(e['dim'], ddof=1),
                         rank_bias=a.mean() - tau, rank_sd=a.std(ddof=1), rank_bias_over_sd=(a.mean() - tau) / a.std(ddof=1),
                         rank_k_median=np.median(e['rank_k']), rank_rmse=np.sqrt(((a - tau) ** 2).mean()), cover_logE=np.mean(e['cover_logE'])))
        print(rows[-1])
    out['trunc'] = rows
    # (b) gamma attack: true gamma>0 but rank-kappa assumes gamma=0
    rows = []
    for g in (0.01, 0.03):
        rng = np.random.default_rng(seed)
        e = {k: [] for k in ["dim", "rank_tau", "rank_k", "rank_L"]}
        tau = true_tau(0.05, 0.05, 1.0, g, sd_u=2.5, b_u=0.7); Ltrue = 0.05 + g
        for r in range(reps):
            d = simulate(rng, n_arm, kappa=1.0, gamma=g, **CFG); Y0, Y1 = d['Y0'], d['Y1']
            e['dim'].append(dim(Y0, Y1)[0]); rk = rank_kappa(Y0, Y1); e['rank_tau'].append(rk['tau']); e['rank_k'].append(rk['kappa']); e['rank_L'].append(rk['L'])
        a = np.array(e['rank_tau'])
        # theory: misattributed shift: log kappa_hat -> log kappa + (w0/w1) log(1+g); L_hat = 0.05*kappa_hat vs Ltrue = 0.05 + g
        w0 = 1 / 1.05; w1 = 1 - w0
        k_pred = np.exp((w0 / w1) * np.log(1 + g)); L_pred = 0.05 * k_pred
        rows.append(dict(gamma=g, tau=tau, L_true=Ltrue, dim_bias=np.mean(e['dim']) - tau, rank_bias=a.mean() - tau, rank_sd=a.std(ddof=1),
                         rank_bias_over_sd=(a.mean() - tau) / a.std(ddof=1), rank_k_median=np.median(e['rank_k']), k_pred=k_pred,
                         rank_L_median=np.median(e['rank_L']), L_pred=L_pred))
        print(rows[-1])
    out['gamma'] = rows
    # (d) heavier tails: Pareto alpha=1.5 positives; rank-kappa vs dim vs ext, kappa=0.7
    rows = []
    for tail, kw in (("pareto", dict(alpha=1.5)), ("lognormal", {})):
        rng = np.random.default_rng(seed)
        e = {k: [] for k in ["dim", "rank_tau", "ext", "cover_logE", "rank_hl_tau"]}
        Etrue = 0.05 * 0.7
        for r in range(reps):
            d = simulate(rng, n_arm, kappa=0.7, gamma=0.0, tail=tail, **kw, **CFG); Y0, Y1 = d['Y0'], d['Y1']
            e['dim'].append(dim(Y0, Y1)[0]); e['ext'].append(restricted_ext(Y0, Y1)[0])
            rk = rank_kappa(Y0, Y1); e['rank_tau'].append(rk['tau']); e['cover_logE'].append(abs(np.log(rk['E']) - np.log(Etrue)) < 1.96 * rk['se_logE'])
            e['rank_hl_tau'].append(rank_kappa(Y0, Y1, use_hl=True)['tau'])
        # true tau: need mA for this tail -> use large-sample control mean
        dd = simulate(np.random.default_rng(99), 2_000_000, kappa=0.7, tail=tail, **kw, **CFG); mA = dd['Y0'][dd['Y0'] > 0].mean()
        tau = 0.05 * 0.05 * 0.7 * mA
        rows.append(dict(tail=tail, tau=tau, mA=mA, dim_bias=np.mean(e['dim']) - tau, dim_sd=np.std(e['dim'], ddof=1),
                         ext_bias=np.mean(e['ext']) - tau, ext_sd=np.std(e['ext'], ddof=1),
                         rank_bias=np.mean(e['rank_tau']) - tau, rank_sd=np.std(e['rank_tau'], ddof=1),
                         rank_hl_bias=np.mean(e['rank_hl_tau']) - tau, rank_hl_sd=np.std(e['rank_hl_tau'], ddof=1), cover_logE=np.mean(e['cover_logE'])))
        print(rows[-1])
    out['tails'] = rows
    return out


# ==================== from exp8.py ====================
"""E8: power / null control; E9: tail-located effects (the blind spot) and the exceedance test;
   E11: data-dependent caps pooled vs stratified (FPR under null); E14: p0 sweep of the gain law."""

def E8(n_arm=200_000, reps=200, seed=8, kappa=0.7):
    CFG = dict(p0=0.05, sd_u=2.5, b_u=0.7)
    out = {}
    for lift in (0.0, 0.05):
        rng = np.random.default_rng(seed)
        rej = {k: [] for k in ["dim", "ext", "rank_logE", "post", "w99"]}
        for r in range(reps):
            d = simulate(rng, n_arm, lift_p=lift, kappa=kappa, **CFG); Y0, Y1 = d['Y0'], d['Y1']
            t, se = dim(Y0, Y1); rej['dim'].append(t / se > 1.645)
            e = restricted_ext(Y0, Y1); rej['ext'].append(e[0] / e[1] > 1.645)
            Yall = np.r_[Y0, Y1]; c = np.quantile(Yall[Yall > 0], 0.99); tw, sew = winsor_dim(Y0, Y1, c); rej['w99'].append(tw / sew > 1.645)
            # rank estimator: one-sided test of E>0 is just the binomial test on dp when lift=0 (kappa undefined); use dp z + require E CI
            p0 = (Y0 > 0).mean(); p1 = (Y1 > 0).mean(); z = (p1 - p0) / np.sqrt(p1 * (1 - p1) / n_arm + p0 * (1 - p0) / n_arm)
            rej['rank_logE'].append(z > 1.645)
            po = rank_kappa_post(Y0, Y1) if p1 > p0 else None
            rej['post'].append((po is not None) and (po['L_lo'] > 0) )
        out[f"lift{lift}"] = {k: float(np.mean(v)) for k, v in rej.items()}
        out[f"lift{lift}"]["mc_se"] = float(np.sqrt(0.25 / reps))
        print(lift, out[f"lift{lift}"])
    return out

def E9(n_arm=200_000, reps=100, seed=9):
    CFG = dict(p0=0.05, sd_u=2.5, b_u=0.7)
    """Tail effect: only top-decile always-buyers spend +50% (no extensive effect). Which estimators see it?
    Exceedance test: P(Y > c) treated vs control at c = 90th pct of positives (tail-free)."""
    rng = np.random.default_rng(seed)
    tau = true_tau(0.05, 0.0, 1.0, 0.0, tail_q=0.1, g_tail=0.5, sd_u=2.5, b_u=0.7)
    e = {k: [] for k in ["dim", "ext", "scale_hl", "rank_tau", "z_exceed90", "z_exceed99", "w99"]}
    for r in range(reps):
        d = simulate(rng, n_arm, lift_p=0.0, tail_q=0.1, g_tail=0.5, **CFG); Y0, Y1 = d['Y0'], d['Y1']
        e['dim'].append(dim(Y0, Y1)[0]); e['ext'].append(restricted_ext(Y0, Y1)[0]); e['scale_hl'].append(restricted_scale(Y0, Y1, 'hl')[0])
        e['rank_tau'].append(rank_kappa(Y0, Y1)['tau'] if (Y1 > 0).mean() > (Y0 > 0).mean() else np.nan)
        Yall = np.r_[Y0, Y1]; V = Yall[Yall > 0]
        for q, k in ((0.9, 'z_exceed90'), (0.99, 'z_exceed99')):
            c = np.quantile(V, q); a1 = (Y1 > c).mean(); a0 = (Y0 > c).mean()
            e[k].append((a1 - a0) / np.sqrt(a1 * (1 - a1) / n_arm + a0 * (1 - a0) / n_arm))
        c = np.quantile(V, 0.99); e['w99'].append(winsor_dim(Y0, Y1, c)[0])
    out = dict(tau=tau)
    for k in ["dim", "ext", "scale_hl", "rank_tau", "w99"]:
        a = np.array(e[k]); a = a[np.isfinite(a)]; out[k] = dict(mean=a.mean(), sd=a.std(ddof=1), frac_of_tau=a.mean() / tau, t_mean=a.mean() / a.std(ddof=1))
    for k in ["z_exceed90", "z_exceed99"]:
        z = np.array(e[k]); out[k] = dict(mean_z=z.mean(), power=np.mean(z > 1.645))
    print(json.dumps(out, default=float))
    return out

def E11(n_arm=200_000, reps=200, seed=10):
    CFG = dict(p0=0.05, sd_u=2.5, b_u=0.7)
    """Null (lift=0): FPR of winsorized DIM with pooled cap vs per-arm caps (99th pct of positives), Pareto 1.5 and lognormal."""
    out = {}
    for tail in ("lognormal", "pareto"):
        rng = np.random.default_rng(seed)
        rej = {"pooled": [], "per_arm": [], "dim": []}
        for r in range(reps):
            d = simulate(rng, n_arm, lift_p=0.0, tail=tail, alpha=1.5, **CFG); Y0, Y1 = d['Y0'], d['Y1']
            Yall = np.r_[Y0, Y1]; c = np.quantile(Yall[Yall > 0], 0.99)
            t, se = winsor_dim(Y0, Y1, c); rej['pooled'].append(abs(t / se) > 1.96)
            c0 = np.quantile(Y0[Y0 > 0], 0.99); c1 = np.quantile(Y1[Y1 > 0], 0.99)
            t, se = dim(np.minimum(Y0, c0), np.minimum(Y1, c1)); rej['per_arm'].append(abs(t / se) > 1.96)
            t, se = dim(Y0, Y1); rej['dim'].append(abs(t / se) > 1.96)
        out[tail] = {k: float(np.mean(v)) for k, v in rej.items()}; print(tail, out[tail])
    return out

def E14(n_arm=200_000, reps=60, seed=12, p0s=(0.01, 0.05, 0.2), s=1.27):
    """Gain law across p0 (lognormal, b_u=0 so log-sd = s): Var_ext/Var_dim vs e^{-s^2}, Var_scaleHL/Var_dim vs (1+pi s^2/3)e^{-s^2}."""
    rows = []
    for p0 in p0s:
        rng = np.random.default_rng(seed)
        e = {k: [] for k in ["dim", "ext", "scale_hl"]}
        for r in range(reps):
            d = simulate(rng, n_arm, p0=p0, lift_p=0.05, sd_u=2.5, b_u=0.0, s=s); Y0, Y1 = d['Y0'], d['Y1']
            e['dim'].append(dim(Y0, Y1)[0]); e['ext'].append(restricted_ext(Y0, Y1)[0]); e['scale_hl'].append(restricted_scale(Y0, Y1, 'hl')[0])
        v = {k: np.var(a, ddof=1) for k, a in e.items()}
        rows.append(dict(p0=p0, ratio_ext=v['ext'] / v['dim'], pred_ext=np.exp(-s**2) * (1 - p0) , ratio_scale=v['scale_hl'] / v['dim'],
                         pred_scale=(1 + np.pi * s**2 / 3) * np.exp(-s**2), mc_rel_err=2 / np.sqrt(2 * (reps - 1))))
        print(rows[-1])
    return rows


# ==================== from exp10.py ====================
"""E10: decision regret under a prior over the marginal-buyer basket ratio kappa ~ U[0.3,1.2]:
   go iff L_hat > L0 (breakeven relative lift), loss |L - L0| when wrong.  Estimators of L: DIM, ext (kappa=1), rank-kappa exact,
   rank-kappa posterior (prior N(0,0.5^2) on log kappa), kappa-isolation with a misspecified tight prior (1 +- 0.25)."""

def E10(n_arm=200_000, reps=200, seed=13, L0=0.03, klo=0.3, khi=1.2):
    CFG = dict(p0=0.05, lift_p=0.05, sd_u=2.5, b_u=0.7)
    rng = np.random.default_rng(seed)
    names = ["dim", "ext", "rank", "post", "kiso"]
    L_hat = {k: [] for k in names}; Ltrue = []
    for r in range(reps):
        k = rng.uniform(klo, khi); Ltrue.append(0.05 * k)
        d = simulate(rng, n_arm, kappa=k, **CFG); Y0, Y1 = d['Y0'], d['Y1']
        p0 = (Y0 > 0).mean(); p1 = (Y1 > 0).mean(); mA = Y0[Y0 > 0].mean()
        base = p0 * mA
        L_hat['dim'].append(dim(Y0, Y1)[0] / base)
        L_hat['ext'].append((p1 - p0) / p0)
        rk = rank_kappa(Y0, Y1) if p1 > p0 else None
        L_hat['rank'].append(rk['L'] if rk else (p1 - p0) / p0)
        po = rank_kappa_post(Y0, Y1) if p1 > p0 else None
        L_hat['post'].append(po['L_med'] if po else (p1 - p0) / p0)
        ki = kappa_isolation(Y0, Y1, 1.0, 0.25); L_hat['kiso'].append(ki[0] / base)
    Ltrue = np.array(Ltrue)
    out = dict(n_arm=n_arm, reps=reps, L0=L0, oracle_regret=0.0)
    for k in names:
        a = np.array(L_hat[k])
        go = a > L0; wrong = go != (Ltrue > L0)
        reg = np.abs(Ltrue - L0) * wrong
        out[k] = dict(mean_regret=reg.mean(), regret_mcse=reg.std(ddof=1) / np.sqrt(reps), frac_wrong=wrong.mean(),
                      mse=((a - Ltrue) ** 2).mean(), bias=(a - Ltrue).mean(), rmse=np.sqrt(((a - Ltrue) ** 2).mean()))
    # always-go / never-go baselines
    for nm, go in (("always_go", np.ones(reps, bool)), ("never_go", np.zeros(reps, bool))):
        wrong = go != (Ltrue > L0); out[nm] = dict(mean_regret=(np.abs(Ltrue - L0) * wrong).mean())
    print(json.dumps(out, default=float))
    return out


# ==================== main ====================
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    t0 = time.time(); R = {}
    def log(k, v): R[k] = v; print(f"--- {k} done ({time.time()-t0:.0f}s)")
    log("E1", run_E1(reps=60 if mode == "quick" else 200))
    log("E2", E2(reps=60 if mode == "quick" else 200))
    log("E4", dict(profiles=profiles(n=1_000_000 if mode == "quick" else 3_000_000),
                   lemma=[dict(b=b, sig=s, regret=regret_flat(b, s), half_mse=(b*b+s*s)/2) for b, s in [(0,1),(0.5,1),(2,1),(0.3,0.2)]]))
    log("E5", E5_E6(reps=40 if mode == "quick" else 100))
    log("E8", E8(reps=100 if mode == "quick" else 200)); log("E9", E9(reps=50 if mode == "quick" else 100))
    log("E11", E11(reps=100 if mode == "quick" else 200)); log("E14", E14(reps=40 if mode == "quick" else 100))
    if mode == "all":
        log("E3", E3(reps=100)); log("E5_1M", E5_E6(n_arm=1_000_000, reps=40, seed=6, kappas=(0.5, 1.0)))
        log("E7", E7()); log("E10", dict(n200k=E10(reps=600), n1M=E10(n_arm=1_000_000, reps=300, seed=14))); log("E12", E12())
        R["E13_seed_sd"] = {sd: {k: run_E1(reps=60, seed=sd)[k]["sd"] for k in ["dim", "ext", "scale_hl", "rank"]} for sd in (101, 202, 303)}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"18_results_{mode}.json")
    json.dump(R, open(out, "w"), default=float, indent=1); print("saved", out, "total", round(time.time() - t0), "s")
