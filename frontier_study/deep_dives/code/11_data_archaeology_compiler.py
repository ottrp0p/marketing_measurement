"""
Dive 11 — The data-archaeology compiler (B5: O1/O3).
Part I : geo-imputation of a national channel as structured Berkson error;
         the leakage law into correctly-measured channels; latent-loading repair.
Part II: taxonomy misclassification as a channel-mixing matrix; audit design
         (certify-top-k + random rectifier) with Pareto line-item spends.
Part III: decision-loss ledger (archaeology VOI).

Usage:  python 11_data_archaeology_compiler.py quick     (~1 min headline numbers)
        python 11_data_archaeology_compiler.py e1 ... e12  (individual experiments)
"""
import sys, numpy as np
from numpy.linalg import lstsq, solve
from scipy.optimize import least_squares, minimize

# ----------------------------------------------------------------------------
# Part I: geo panel DGP
# ----------------------------------------------------------------------------
def hill(x, K, s):
    x = np.maximum(x, 1e-12)
    return x**s / (x**s + K**s)

def dhill(x, K, s):
    x = np.maximum(x, 1e-12)
    return s * K**s * x**(s-1) / (x**s + K**s)**2

def adstock(x, alpha):
    a = np.zeros_like(x, dtype=float)
    prev = np.zeros(x.shape[1:]) if x.ndim > 1 else 0.0
    for t in range(x.shape[0]):
        prev = x[t] + alpha * prev
        a[t] = prev
    return a

def national_paths(T, rho_SD, rng, seas_amp=0.4, ar=0.6, noise=0.25):
    """Two national spend paths S (TV) and D (Digital), per-capita units, with
    a shared Q4 seasonal ramp whose relative strength sets corr(S,D)."""
    t = np.arange(T)
    seas = 1 + seas_amp * np.cos(2*np.pi*(t-45)/52)  # peaks in Q4
    def ar1(n):
        e = rng.normal(size=n); out = np.zeros(n)
        for i in range(1, n): out[i] = ar*out[i-1] + e[i]*np.sqrt(1-ar**2)
        return out
    zS, zD = ar1(T), ar1(T)
    # mix: a common component with weight rho_SD
    zC = ar1(T)
    zS = np.sqrt(rho_SD)*zC + np.sqrt(1-rho_SD)*zS
    zD = np.sqrt(rho_SD)*zC + np.sqrt(1-rho_SD)*zD
    S = seas * np.exp(noise*zS); D = seas * np.exp(noise*zD)
    return S/S.mean(), D/D.mean()

def geo_shares(G, sig_eta, sig_zeta, rho_ez, rng):
    """Population shares p_g and relative share deviations eta (TV true), zeta (digital)."""
    pop = rng.lognormal(0, 0.8, size=G); p = pop/pop.sum()
    z1, z2 = rng.normal(size=G), rng.normal(size=G)
    eta = sig_eta * z1
    zeta = sig_zeta * (rho_ez*z1 + np.sqrt(1-rho_ez**2)*z2)
    # renormalise so population-weighted mean deviation is zero (shares sum to 1)
    eta = eta - (p*eta).sum(); zeta = zeta - (p*zeta).sum()
    return p, eta, zeta

def make_panel(G=50, T=104, rho_SD=0.6, sig_eta=0.3, sig_zeta=0.3, rho_ez=0.5,
               beta_tv=1.0, beta_d=1.0, sigma=0.5, alpha_tv=0.5, alpha_d=0.2,
               K_tv=None, s_tv=1.0, K_d=None, s_d=1.0, sig_e=0.0, rng=None,
               scale_tv=1.0, scale_d=1.0):
    """Per-capita geo panel. If K_* is None the response is linear in adstocked
    per-capita spend; otherwise Hill(K,s) with scale beta.  sig_e = time-varying
    share noise (Berkson) on the TV channel. Returns dict of arrays [T,G]."""
    rng = np.random.default_rng() if rng is None else rng
    S, D = national_paths(T, rho_SD, rng)
    S, D = scale_tv*S, scale_d*D
    p, eta, zeta = geo_shares(G, sig_eta, sig_zeta, rho_ez, rng)
    e = sig_e*rng.normal(size=(T, G)) if sig_e > 0 else np.zeros((T, G))
    x_tv_true = S[:, None]*(1+eta[None, :]+e)      # true per-capita TV spend
    x_tv_imp = np.repeat(S[:, None], G, axis=1)     # population-share imputation
    x_d = D[:, None]*(1+zeta[None, :])              # measured digital
    a_tv = adstock(x_tv_true, alpha_tv); a_d = adstock(x_d, alpha_d)
    f_tv = a_tv if K_tv is None else hill(a_tv, K_tv, s_tv)
    f_d = a_d if K_d is None else hill(a_d, K_d, s_d)
    mu = rng.normal(0, 1, size=G)
    tt = np.arange(T)
    seas = 0.3*np.sin(2*np.pi*tt/52 + 0.7)
    y = mu[None, :] + beta_tv*f_tv + beta_d*f_d + seas[:, None] + sigma*rng.normal(size=(T, G))
    return dict(y=y, S=S, D=D, p=p, eta=eta, zeta=zeta, x_tv_true=x_tv_true,
                x_tv_imp=x_tv_imp, x_d=x_d, T=T, G=G, alpha_tv=alpha_tv, alpha_d=alpha_d,
                K_tv=K_tv, s_tv=s_tv, K_d=K_d, s_d=s_d, beta_tv=beta_tv, beta_d=beta_d, e=e)

def fourier(T, k=3):
    t = np.arange(T); cols = [np.ones(T)]
    for j in range(1, k+1):
        cols += [np.sin(2*np.pi*j*t/52), np.cos(2*np.pi*j*t/52)]
    return np.column_stack(cols)

def demean_geo(A):  # remove geo fixed effects (within-geo mean over time)
    return A - A.mean(axis=0, keepdims=True)

def fit_linear(panel, tv_regressor, kf=3, alpha_known=True):
    """OLS with geo FE + Fourier seasonality; returns (b_tv, b_d)."""
    T, G = panel['T'], panel['G']
    a_tv = adstock(tv_regressor, panel['alpha_tv']); a_d = adstock(panel['x_d'], panel['alpha_d'])
    F = fourier(T, kf)
    y = demean_geo(panel['y']).ravel(order='F')
    X = np.column_stack([demean_geo(a_tv).ravel(order='F'), demean_geo(a_d).ravel(order='F'),
                         np.tile(F[:, 1:], (G, 1))])
    b = lstsq(X, y, rcond=None)[0]
    return b[0], b[1]

def fit_latent_loading(panel, kf=3, shrink=0.0):
    """Geo-specific slope on the imputed national TV path: y = mu_g + lam_g*a_S + b_d*a_d + seas.
    Population-weighted mean of lam_g estimates beta_tv; optional ridge shrinkage of lam_g toward
    their mean (hierarchical-prior analogue, precision = shrink)."""
    T, G = panel['T'], panel['G']
    a_S = adstock(panel['S'], panel['alpha_tv']); a_d = adstock(panel['x_d'], panel['alpha_d'])
    F = fourier(T, kf)
    y = demean_geo(panel['y']).ravel(order='F')
    aS_dm = a_S - a_S.mean()
    cols = [np.zeros((T*G,)) for _ in range(G)]
    Xg = np.zeros((T*G, G))
    for g in range(G):
        Xg[g*T:(g+1)*T, g] = aS_dm
    X = np.column_stack([Xg, demean_geo(a_d).ravel(order='F'), np.tile(F[:, 1:], (G, 1))])
    if shrink > 0:
        # penalty shrink * sum_g (lam_g - lam_bar)^2  == ridge on deviations; implement via augmented rows
        C = np.eye(G) - np.ones((G, G))/G
        aug = np.zeros((G, X.shape[1])); aug[:, :G] = np.sqrt(shrink)*C
        X = np.vstack([X, aug]); y = np.concatenate([y, np.zeros(G)])
    b = lstsq(X, y, rcond=None)[0]
    lam = b[:G]
    return (panel['p']*lam).sum(), b[G], lam

def leakage_formula(panel, kf=3):
    """Closed-form leakage law (derived in the dive, §3.1):
        bias(b_d)/beta_tv = [ sum_t s_t d_t * Cov_g(eta,zeta) ] / [ sum_t d_t^2 Var_g(zeta) + sum_t r_t^2 (1+zeta_bar)^2 ]
    where s_t, d_t are the within-geo-demeaned adstocked national paths (NOT residualised on the seasonal basis —
    geo-constant controls cannot absorb geo-heterogeneous seasonal amplitude) and r_t is d_t residualised on the
    geo-constant columns (imputed TV path + Fourier basis). Unweighted geo moments (OLS weights geos equally).
    Assumes sum_g eta_g = 0 (exact when shares are renormalised with equal weights; approximate otherwise)."""
    T = panel['T']; F = fourier(T, kf)
    aS = adstock(panel['S'], panel['alpha_tv']); aD = adstock(panel['D'], panel['alpha_d'])
    s = aS - aS.mean(); d = aD - aD.mean()
    C = np.column_stack([s, F[:, 1:]])
    r = d - C @ lstsq(C, d, rcond=None)[0]
    eta, zeta = panel['eta'], panel['zeta']
    cov = np.mean((eta-eta.mean())*(zeta-zeta.mean())); var = np.var(zeta); zb = zeta.mean()
    return (s*d).sum()*cov / ((d*d).sum()*var + (r*r).sum()*(1+zb)**2)

def leakage_exact(panel, kf=3):
    """Exact finite-sample omitted-variable bias (X'X)^{-1}X'z for the fitted linear design, b_d component."""
    T, G = panel['T'], panel['G']
    a_imp = adstock(panel['x_tv_imp'], panel['alpha_tv']); a_true = adstock(panel['x_tv_true'], panel['alpha_tv'])
    a_d = adstock(panel['x_d'], panel['alpha_d']); F = fourier(T, kf)
    X = np.column_stack([demean_geo(a_imp).ravel(order='F'), demean_geo(a_d).ravel(order='F'), np.tile(F[:, 1:], (G, 1))])
    z = demean_geo(a_true - a_imp).ravel(order='F')
    return lstsq(X, z, rcond=None)[0][:2]*panel['beta_tv']

# --- nonlinear (Hill) fits -------------------------------------------------
def fit_hill_model(panel, mode, kf=3, shrink=1.0, x_tv=None):
    """mode: 'plugin' (common beta,K; imputed x), 'hier' (geo beta_g, common K; imputed x),
    'latent' (common beta,K; geo loading 1+eta_g on x), 'oracle' (true x).
    Adstock and digital params known/linear; Hill slope s fixed at truth (curvature identification is dive 01's job).
    Returns dict with beta_tv, K_tv, b_d, and (for hier/latent) geo vectors."""
    T, G = panel['T'], panel['G']
    s_tv = panel['s_tv']
    F = fourier(T, kf)[:, 1:]
    a_d = demean_geo(adstock(panel['x_d'], panel['alpha_d']))
    y = panel['y']
    x_base = panel['x_tv_true'] if mode == 'oracle' else (panel['x_tv_imp'] if x_tv is None else x_tv)
    a_base = adstock(x_base, panel['alpha_tv'])  # [T,G]
    Fl = np.tile(F, (G, 1))
    def resid(theta):
        logK = theta[0]
        if mode in ('plugin', 'oracle'):
            f = hill(a_base, np.exp(logK), s_tv); lin_cols = [demean_geo(f).ravel(order='F')]
            pen = np.zeros(0)
        elif mode == 'hier':
            f = hill(a_base, np.exp(logK), s_tv); fd = demean_geo(f)
            Xg = np.zeros((T*G, G))
            for g in range(G): Xg[g*T:(g+1)*T, g] = fd[:, g]
            lin_cols = list(Xg.T); pen = np.zeros(0)
        elif mode == 'latent':
            eta = theta[1:1+G]
            f = hill(a_base*(1+eta[None, :]), np.exp(logK), s_tv); lin_cols = [demean_geo(f).ravel(order='F')]
            pen = np.sqrt(shrink)*eta  # weak ridge keeps eta identifiable in flat regions
        X = np.column_stack(lin_cols + [a_d.ravel(order='F'), Fl])
        yy = demean_geo(y).ravel(order='F')
        b = lstsq(X, yy, rcond=None)[0]
        r = yy - X @ b
        if mode == 'hier' and shrink > 0:
            bg = b[:G]; r = np.concatenate([r, np.sqrt(shrink)*(bg-bg.mean())])
        return np.concatenate([r, pen]), b
    def obj(theta): return resid(theta)[0]
    theta0 = np.array([np.log(np.median(a_base))] + ([0.0]*G if mode == 'latent' else []))
    sol = least_squares(obj, theta0, method='trf', max_nfev=400)
    r, b = resid(sol.x)
    out = dict(K_tv=np.exp(sol.x[0]))
    if mode == 'hier':
        out['beta_g'] = b[:G]; out['beta_tv'] = (panel['p']*b[:G]).sum(); out['b_d'] = b[G]
    else:
        out['beta_tv'] = b[0]; out['b_d'] = b[1]
    if mode == 'latent': out['eta_hat'] = sol.x[1:1+G]
    return out

def national_mroas(panel, fit, mode, S_level=None):
    """Marginal national response to +1 unit of national per-capita TV spend at the mean level,
    computed under the fitted model's own geo structure, vs truth."""
    p, G = panel['p'], panel['G']
    S0 = panel['S'].mean()/(1-panel['alpha_tv']) if S_level is None else S_level
    K, s = fit['K_tv'], panel['s_tv']
    if mode == 'hier':
        return (p*fit['beta_g']).sum()*dhill(S0, K, s) if False else (p*fit['beta_g']*dhill(S0, K, s)).sum()
    if mode == 'latent':
        eta = fit['eta_hat']; return fit['beta_tv']*(p*(1+eta)*dhill(S0*(1+eta), K, s)).sum()
    if mode == 'plugin':
        return fit['beta_tv']*dhill(S0, K, s)
    if mode == 'oracle':
        eta = panel['eta']; return fit['beta_tv']*(p*(1+eta)*dhill(S0*(1+eta), K, s)).sum()

def true_mroas(panel, S_level=None):
    p, eta = panel['p'], panel['eta']
    S0 = panel['S'].mean()/(1-panel['alpha_tv']) if S_level is None else S_level
    return panel['beta_tv']*(p*(1+eta)*dhill(S0*(1+eta), panel['K_tv'], panel['s_tv'])).sum()

# ----------------------------------------------------------------------------
# Part II: taxonomy misclassification
# ----------------------------------------------------------------------------
def line_items(N, alpha_pareto, n_ch, rng, ch_probs=None):
    s = (rng.pareto(alpha_pareto, size=N) + 1)  # Pareto(alpha) with x_min=1
    ch_probs = np.ones(n_ch)/n_ch if ch_probs is None else ch_probs
    c = rng.choice(n_ch, size=N, p=ch_probs)
    return s, c

def llm_label(c, s, M, rng, size_dependence=0.0):
    """Label each line by confusion matrix M[true, pred]; optional reduction of error for big lines
    (error rate multiplied by s^-size_dependence relative to median)."""
    N = len(c); n_ch = M.shape[0]
    rows = M[c].copy()                                   # [N, n_ch]
    if size_dependence > 0:
        f = np.minimum((s/np.median(s))**(-size_dependence), 1.0)[:, None]
        diag = rows[np.arange(N), c]
        rows = rows*f; rows[np.arange(N), c] = 1-(1-diag)*f[:, 0]
    rows = rows/rows.sum(1, keepdims=True)
    u = rng.random(N)[:, None]
    return (u > np.cumsum(rows, axis=1)).sum(1).clip(0, n_ch-1)

def channel_totals(s, lab, n_ch):
    return np.array([s[lab == k].sum() for k in range(n_ch)])

def audit_design(s, c, pred, n_ch, budget, strategy, rng, ppi=True):
    """Return estimated channel totals after spending `budget` audit labels.
    strategies: 'llm' (no audit), 'topk' (certify largest budget lines; rest LLM),
    'srs' (random labels, PPI rectifier), 'split' (half top-k certify, half random PPI on the tail),
    'pps' (probability proportional to size sample, Horvitz-Thompson difference estimator)."""
    N = len(s)
    if strategy == 'llm' or budget == 0:
        return channel_totals(s, pred, n_ch)
    if strategy == 'topk':
        idx = np.argsort(-s)[:budget]; lab = pred.copy(); lab[idx] = c[idx]
        return channel_totals(s, lab, n_ch)
    if strategy in ('matrix', 'topk_matrix'):
        # certify top-k (k = budget/2 for topk_matrix, 0 for matrix); SRS n_r from the tail to estimate the
        # confusion matrix M[true,pred] by counts; tail totals corrected by T_true = M^{-T} T_pred (matrix calibration).
        k = budget//2 if strategy == 'topk_matrix' else 0
        top = np.argsort(-s)[:k]; lab = pred.copy(); lab[top] = c[top]
        tail = np.setdiff1d(np.arange(N), top); n_r = budget - k
        samp = rng.choice(tail, size=min(n_r, len(tail)), replace=False)
        Mh = np.ones((n_ch, n_ch))*0.5  # Jeffreys-style smoothing
        for i in samp: Mh[c[i], pred[i]] += 1
        Mh = Mh/Mh.sum(1, keepdims=True)
        T_pred_tail = channel_totals(s[tail], pred[tail], n_ch)
        T_true_tail = solve(Mh.T, T_pred_tail)
        return channel_totals(s[top], c[top], n_ch) + T_true_tail
    if strategy in ('srs', 'split', 'pps'):
        if strategy == 'split':
            k = budget//2; top = np.argsort(-s)[:k]; lab = pred.copy(); lab[top] = c[top]
            tail = np.setdiff1d(np.arange(N), top); n_r = budget - k
        else:
            lab = pred.copy(); tail = np.arange(N); n_r = budget
        # difference estimator (PPI): total_pred + N_tail/n * sum_{sampled}(s*(1[c=k]-1[pred=k]))
        if strategy == 'pps':
            pr = s[tail]/s[tail].sum()
            samp = rng.choice(tail, size=n_r, replace=True, p=pr)
            corr = np.array([np.mean(s[samp]*((c[samp] == k).astype(float)-(pred[samp] == k))/pr[np.searchsorted(tail, samp)]) for k in range(n_ch)])
        else:
            samp = rng.choice(tail, size=min(n_r, len(tail)), replace=False)
            corr = np.array([len(tail)/len(samp)*np.sum(s[samp]*((c[samp] == k).astype(float)-(pred[samp] == k))) for k in range(n_ch)])
        return channel_totals(s, lab, n_ch) + corr
    raise ValueError(strategy)

# ----------------------------------------------------------------------------
# Part III: decision loss for 2-channel allocation with Hill response
# ----------------------------------------------------------------------------
def alloc_loss(beta_true, beta_hat, K=(1.0, 1.0), s=(1.0, 1.0), B=2.0):
    """Regret (revenue units) of allocating B across two channels using beta_hat instead of beta_true."""
    def rev(x, beta): return beta[0]*hill(x, K[0], s[0]) + beta[1]*hill(B-x, K[1], s[1])
    grid = np.linspace(0.01, B-0.01, 2001)
    x_star = grid[np.argmax(rev(grid, beta_true))]; x_hat = grid[np.argmax(rev(grid, beta_hat))]
    return rev(x_star, beta_true) - rev(x_hat, beta_true), x_star, x_hat

# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def e1(nrep=200, seed=1):
    """Linear leakage law: bias in b_d from imputing TV, vs closed form."""
    rng = np.random.default_rng(seed)
    rows = []
    for cfg in [dict(rho_SD=0.6, rho_ez=0.5), dict(rho_SD=0.9, rho_ez=0.8, sig_eta=0.5, sig_zeta=0.5),
                dict(rho_SD=0.6, rho_ez=0.0), dict(rho_SD=0.0, rho_ez=0.8), dict(rho_SD=0.6, rho_ez=-0.5)]:
        bd, bt, form, bd_or, ex = [], [], [], [], []
        for r in range(nrep):
            pn = make_panel(rng=rng, **cfg)
            b1, b2 = fit_linear(pn, pn['x_tv_imp']); bt.append(b1); bd.append(b2)
            bd_or.append(fit_linear(pn, pn['x_tv_true'])[1])
            form.append(leakage_formula(pn)); ex.append(leakage_exact(pn)[1])
        bd, bt, form, bd_or, ex = map(np.array, (bd, bt, form, bd_or, ex))
        rows.append((cfg, bt.mean()-1, bd.mean()-1, bd.std()/np.sqrt(nrep), form.mean(), bd_or.mean()-1,
                     np.corrcoef(bd-1, form)[0, 1]))
        print(f"{cfg}: bias b_tv={bt.mean()-1:+.3f}  bias b_d={bd.mean()-1:+.3f} (se {bd.std()/np.sqrt(nrep):.3f})  "
              f"closed-form={form.mean():+.3f}  exact-OVB={ex.mean():+.3f}  oracle b_d bias={bd_or.mean()-1:+.3f}  "
              f"corr per rep (bias,closed)={np.corrcoef(bd-1, form)[0,1]:.2f} (bias,exact)={np.corrcoef(bd-1, ex)[0,1]:.2f}")
    return rows

def e2(nrep=100, seed=2):
    """Latent-loading repair; variance cost as rho_SD -> 1."""
    rng = np.random.default_rng(seed)
    for rho in [0.0, 0.6, 0.9, 0.97, 0.995]:
        bd_pl, bd_ll, bt_ll, bd_or = [], [], [], []
        for r in range(nrep):
            pn = make_panel(rng=rng, rho_SD=rho, rho_ez=0.8, sig_eta=0.5, sig_zeta=0.5)
            bd_pl.append(fit_linear(pn, pn['x_tv_imp'])[1])
            bt, bd, lam = fit_latent_loading(pn); bd_ll.append(bd); bt_ll.append(bt)
            bd_or.append(fit_linear(pn, pn['x_tv_true'])[1])
        f = lambda a: (np.mean(a)-1, np.std(a))
        print(f"rho_SD={rho}: plugin b_d bias {f(bd_pl)[0]:+.3f} sd {f(bd_pl)[1]:.3f} | latent b_d bias {f(bd_ll)[0]:+.3f} sd {f(bd_ll)[1]:.3f} "
              f"| latent b_tv bias {f(bt_ll)[0]:+.3f} sd {f(bt_ll)[1]:.3f} | oracle b_d sd {f(bd_or)[1]:.3f}")

def e3(nrep=60, seed=3):
    """Hierarchical geo-beta model absorbs persistent misallocation in the LINEAR case; leakage returns with shrinkage."""
    rng = np.random.default_rng(seed)
    for shrink in [0.0, 1.0, 10.0, 100.0, 1e4]:
        bd = []
        for r in range(nrep):
            pn = make_panel(rng=rng, rho_SD=0.9, rho_ez=0.8, sig_eta=0.5, sig_zeta=0.5)
            bd.append(fit_latent_loading(pn, shrink=shrink)[1])
        print(f"shrink={shrink:g}: b_d bias {np.mean(bd)-1:+.3f} sd {np.std(bd):.3f}")

def e4(nrep=40, seed=4, sig_eta=0.5):
    """Hill response: national mROAS bias under plugin / hier / latent / oracle."""
    rng = np.random.default_rng(seed)
    res = {m: [] for m in ['plugin', 'hier', 'oracle']}; bd = {m: [] for m in res}  # 'latent' Hill is unidentified near the knee (e19)
    for r in range(nrep):
        pn = make_panel(rng=rng, rho_SD=0.9, rho_ez=0.8, sig_eta=sig_eta, sig_zeta=0.5, K_tv=1.5, s_tv=1.0,
                        beta_tv=3.0, beta_d=1.0, sigma=0.3)
        tm = true_mroas(pn)
        for m in res:
            fit = fit_hill_model(pn, m)
            res[m].append(national_mroas(pn, fit, m)/tm - 1); bd[m].append(fit['b_d']-1)
    for m in res:
        a = np.array(res[m]); b = np.array(bd[m])
        print(f"{m:7s}: national mROAS rel.bias {a.mean():+.3f} (se {a.std()/np.sqrt(nrep):.3f})  b_d bias {b.mean():+.3f} (se {b.std()/np.sqrt(nrep):.3f})")
    return res

def e5(nrep=40, seed=5):
    """Sweep sig_eta for Hill plugin bias (Sun et al. positive-bias check) and latent repair."""
    rng = np.random.default_rng(seed)
    for se in [0.0, 0.25, 0.5, 0.8]:
        out = {m: [] for m in ['plugin', 'hier']}
        for r in range(nrep):
            pn = make_panel(rng=rng, rho_SD=0.9, rho_ez=0.8, sig_eta=se, sig_zeta=0.5, K_tv=1.5, s_tv=1.0,
                            beta_tv=3.0, beta_d=1.0, sigma=0.3)
            tm = true_mroas(pn)
            for m in out:
                fit = fit_hill_model(pn, m); out[m].append(national_mroas(pn, fit, m)/tm-1)
        print(f"sig_eta={se}: " + "  ".join(f"{m} {np.mean(out[m]):+.3f}±{np.std(out[m])/np.sqrt(nrep):.3f}" for m in out))

def e6(nrep=40, seed=6):
    """No-free-lunch: imputed geo panel vs national model — precision of TV K and beta."""
    rng = np.random.default_rng(seed)
    Kg, Kn, bg, bn = [], [], [], []
    for r in range(nrep):
        pn = make_panel(rng=rng, rho_SD=0.6, rho_ez=0.0, sig_eta=0.0, sig_zeta=0.5, K_tv=1.5, s_tv=1.0,
                        beta_tv=3.0, beta_d=1.0, sigma=0.3)
        fg = fit_hill_model(pn, 'plugin'); Kg.append(fg['K_tv']); bg.append(fg['beta_tv'])
        # national model: population-weighted aggregate outcome, national spends, noise sigma/sqrt(G_eff)
        p = pn['p']; yN = (pn['y']*p[None, :]).sum(1)
        xdN = (pn['x_d']*p[None, :]).sum(1)
        pnN = dict(pn); pnN.update(y=yN[:, None], x_d=xdN[:, None], x_tv_imp=pn['S'][:, None], x_tv_true=pn['S'][:, None], G=1, p=np.array([1.0]), eta=np.array([0.0]))
        fn = fit_hill_model(pnN, 'plugin'); Kn.append(fn['K_tv']); bn.append(fn['beta_tv'])
    print(f"geo-imputed: K sd {np.std(Kg):.3f} (mean {np.mean(Kg):.3f}), beta sd {np.std(bg):.3f} | national: K sd {np.std(Kn):.3f} (mean {np.mean(Kn):.3f}), beta sd {np.std(bn):.3f}")

def e7(nrep=40, seed=7):
    """Time-varying share noise (Berkson) on TV: plugin bias in Hill model and in linear model."""
    rng = np.random.default_rng(seed)
    for sig_e in [0.0, 0.3, 0.6]:
        lin, hl, hl_or = [], [], []
        for r in range(nrep):
            pn = make_panel(rng=rng, rho_SD=0.6, rho_ez=0.0, sig_eta=0.0, sig_zeta=0.5, sig_e=sig_e)
            lin.append(fit_linear(pn, pn['x_tv_imp'])[0])
            pn2 = make_panel(rng=rng, rho_SD=0.6, rho_ez=0.0, sig_eta=0.0, sig_zeta=0.5, sig_e=sig_e, K_tv=1.5, s_tv=1.0, beta_tv=3.0, sigma=0.3)
            tm = true_mroas(pn2)
            # true mROAS averaged over the noise: E_e[(1+e) H'(S(1+e))]
            e = sig_e*rng.normal(size=20000); S0 = pn2['S'].mean()/(1-pn2['alpha_tv'])
            tm = 3.0*np.mean((1+e)*dhill(S0*(1+e), 1.5, 1.0))
            f = fit_hill_model(pn2, 'plugin'); hl.append(national_mroas(pn2, f, 'plugin')/tm-1)
        print(f"sig_e={sig_e}: linear b_tv bias {np.mean(lin)-1:+.3f}±{np.std(lin)/np.sqrt(nrep):.3f} | Hill plugin mROAS rel.bias {np.mean(hl):+.3f}±{np.std(hl)/np.sqrt(nrep):.3f}")

# --- Part II experiments ----------------------------------------------------
def e8(seed=8, N=2000, alpha=1.3, n_ch=3, err=0.10, nrep=200):
    """Channel-total error under LLM labelling; variance concentration in top lines."""
    rng = np.random.default_rng(seed)
    M = np.full((n_ch, n_ch), err/(n_ch-1)); np.fill_diagonal(M, 1-err)
    for a in [1.1, 1.3, 2.0, 3.0]:
        s, c = line_items(N, a, n_ch, rng)
        true = channel_totals(s, c, n_ch)
        srt = np.sort(s)[::-1]; cum = np.cumsum(srt**2)/np.sum(srt**2)
        q = [np.searchsorted(cum, f)/N for f in (0.5, 0.9, 0.99)]
        errs = []
        for r in range(nrep):
            pred = llm_label(c, s, M, rng); errs.append(channel_totals(s, pred, n_ch)/true-1)
        errs = np.array(errs)
        print(f"alpha={a}: top share of s^2 -> 50%:{q[0]*100:.1f}% lines, 90%:{q[1]*100:.1f}%, 99%:{q[2]*100:.1f}% | rel err sd per channel {errs.std(0).round(3)} | mean {errs.mean(0).round(3)}")

def e9(seed=9, N=2000, alpha=1.3, n_ch=3, err=0.10, nrep=300, asym=True, size_dep=0.0):
    """Audit strategies at equal budget: RMSE of channel totals."""
    rng = np.random.default_rng(seed)
    M = np.full((n_ch, n_ch), err/(n_ch-1)); np.fill_diagonal(M, 1-err)
    if asym:  # systematic flow: 15% of channel 0 lines mislabelled as channel 1
        M = np.eye(n_ch)*(1-err); M[0, 1] += err; M[1, 2] += err; M[2, 0] += err
    s, c = line_items(N, alpha, n_ch, rng); true = channel_totals(s, c, n_ch)
    for budget in [0, 50, 100, 200, 400]:
        out = {}
        for strat in (['llm'] if budget == 0 else ['topk', 'srs', 'pps', 'split', 'matrix', 'topk_matrix']):
            e = []
            for r in range(nrep):
                pred = llm_label(c, s, M, rng, size_dependence=size_dep)
                est = audit_design(s, c, pred, n_ch, budget, strat, rng); e.append(est/true-1)
            e = np.array(e); out[strat] = (np.sqrt((e**2).mean(0)).mean(), np.abs(e.mean(0)).mean())
        print(f"budget={budget}: " + "  ".join(f"{k}: rmse {v[0]:.3f} |bias| {v[1]:.3f}" for k, v in out.items()))

def e10(seed=10, nrep=300):
    """Decision-loss ledger: propagate (a) leakage bias in b_d, (b) taxonomy total error into allocation regret."""
    rng = np.random.default_rng(seed)
    beta = np.array([3.0, 1.0]); K = (1.5, 1.0); B = 2.5
    base = alloc_loss(beta, beta, K, B=B)[0]
    print("ledger (regret as % of optimal revenue):")
    def rev_opt():
        grid = np.linspace(0.01, B-0.01, 2001)
        return max(beta[0]*hill(grid, K[0], 1)+beta[1]*hill(B-grid, K[1], 1))
    R = rev_opt()
    cases = [('clean (sd 3%)', 0.03, 0.03, 0.0, 0.0),
             ('imputation leak, baseline (TV -7%, D +17%)', 0.03, 0.03, -0.07, 0.17),
             ('imputation leak, strong (TV -48%, D +79%)', 0.03, 0.03, -0.48, 0.79),
             ('taxonomy raw LLM (D -6% +-12%)', 0.04, 0.12, 0.0, -0.06),
             ('taxonomy certify top 5% (sd 1.3%)', 0.02, 0.013, 0.0, 0.0),
             ('Hill Berkson share noise 0.6 (TV +21%)', 0.03, 0.03, 0.21, 0.0)]
    for name, sd_tv, sd_d, bias_tv, bias_d in cases:
        L = []
        for r in range(nrep):
            bh = beta*np.array([1+bias_tv+sd_tv*rng.normal(), 1+bias_d+sd_d*rng.normal()])
            L.append(alloc_loss(beta, bh, K, B=B)[0])
        print(f"  {name:45s}: E[regret] {np.mean(L)/R*100:.2f}% (se {np.std(L)/np.sqrt(nrep)/R*100:.2f})")

def e11(nrep=60, seed=11):
    """Robustness: misspecified adstock in the fitted model (truth 0.5, fitted 0.3/0.7) — does the leakage law survive?"""
    rng = np.random.default_rng(seed)
    for a_fit in [0.3, 0.5, 0.7]:
        bd_pl, bd_ll, form = [], [], []
        for r in range(nrep):
            pn = make_panel(rng=rng, rho_SD=0.9, rho_ez=0.8, sig_eta=0.5, sig_zeta=0.5)
            pn2 = dict(pn); pn2['alpha_tv'] = a_fit
            bd_pl.append(fit_linear(pn2, pn['x_tv_imp'])[1]); bd_ll.append(fit_latent_loading(pn2)[1]); form.append(leakage_formula(pn2))
        print(f"alpha_fit={a_fit}: plugin b_d bias {np.mean(bd_pl)-1:+.3f}  formula {np.mean(form):+.3f}  latent b_d bias {np.mean(bd_ll)-1:+.3f}")

def e12(nrep=60, seed=12):
    """Power control: G and T sweep — is leakage detectable (Hausman-type test plugin vs latent b_d)?"""
    rng = np.random.default_rng(seed)
    for (G, T) in [(20, 52), (50, 104), (100, 156)]:
        for rho_ez in [0.0, 0.8]:
            diffs, sds = [], []
            for r in range(nrep):
                pn = make_panel(G=G, T=T, rng=rng, rho_SD=0.9, rho_ez=rho_ez, sig_eta=0.5, sig_zeta=0.5)
                d = fit_linear(pn, pn['x_tv_imp'])[1] - fit_latent_loading(pn)[1]; diffs.append(d)
            diffs = np.array(diffs)
            print(f"G={G},T={T},rho_ez={rho_ez}: mean(plugin-latent b_d) {diffs.mean():+.3f} sd {diffs.std():.3f}  |t|={abs(diffs.mean())/diffs.std():.1f}")

def leak_hill_ovb(pn, kf=3):
    """Exact OVB for the Hill case at the true K: omitted term beta*[H(a(1+eta))-H(a)] projected on the plug-in design."""
    T, G = pn['T'], pn['G']; F = fourier(T, kf)
    a = adstock(pn['x_tv_imp'], pn['alpha_tv']); at = adstock(pn['x_tv_true'], pn['alpha_tv'])
    Himp = hill(a, pn['K_tv'], pn['s_tv']); Htrue = hill(at, pn['K_tv'], pn['s_tv'])
    ad = adstock(pn['x_d'], pn['alpha_d'])
    X = np.column_stack([demean_geo(Himp).ravel(order='F'), demean_geo(ad).ravel(order='F'), np.tile(F[:, 1:], (G, 1))])
    z = demean_geo(Htrue-Himp).ravel(order='F')
    return lstsq(X, z, rcond=None)[0][:2]*pn['beta_tv']

def e13(nrep=100, seed=13):
    """Geo-plugin vs national aggregate vs geo-latent (linear): imputation manufactures the bias."""
    rng = np.random.default_rng(seed)
    for cfg in [dict(rho_SD=0.6, rho_ez=0.5), dict(rho_SD=0.9, rho_ez=0.8, sig_eta=0.5, sig_zeta=0.5)]:
        out = []
        for r in range(nrep):
            pn = make_panel(rng=rng, **cfg)
            bt, bd = fit_linear(pn, pn['x_tv_imp'])
            p = pn['p']; yP = (pn['y']*p).sum(1); xdP = (pn['x_d']*p).sum(1)
            pnP = dict(pn); pnP.update(y=yP[:, None], x_d=xdP[:, None], x_tv_imp=pn['S'][:, None], G=1)
            btP, bdP = fit_linear(pnP, pnP['x_tv_imp'])
            bl, bdl, _ = fit_latent_loading(pn)
            out.append([bt, bd, btP, bdP, bl, bdl])
        o = np.array(out); mu = o.mean(0)-1; sd = o.std(0)
        print(cfg)
        for i, nm in enumerate(['geo-plugin', 'national-pop', 'geo-latent']):
            print(f"  {nm:12s} b_tv {mu[2*i]:+.3f} (sd {sd[2*i]:.3f})  b_d {mu[2*i+1]:+.3f} (sd {sd[2*i+1]:.3f})")

def fit_linear_geoseason(panel, tv_regressor, kf=3):
    """OLS with geo FE + geo-specific Fourier seasonality (geo x season interactions)."""
    T, G = panel['T'], panel['G']
    a_tv = adstock(tv_regressor, panel['alpha_tv']); a_d = adstock(panel['x_d'], panel['alpha_d'])
    F = fourier(T, kf)[:, 1:]
    y = demean_geo(panel['y']).ravel(order='F')
    Xg = np.zeros((T*G, G*F.shape[1]))
    for g in range(G): Xg[g*T:(g+1)*T, g*F.shape[1]:(g+1)*F.shape[1]] = F
    X = np.column_stack([demean_geo(a_tv).ravel(order='F'), demean_geo(a_d).ravel(order='F'), Xg])
    b = lstsq(X, y, rcond=None)[0]
    return b[0], b[1]

def e14(nrep=60, seed=14):
    """Attack: do geo x season controls remove the leak? (partial fix; precision cost)"""
    rng = np.random.default_rng(seed)
    for rho in [0.0, 0.6, 0.9]:
        out = []
        for r in range(nrep):
            pn = make_panel(rng=rng, rho_SD=rho, rho_ez=0.8, sig_eta=0.5, sig_zeta=0.5)
            out.append([fit_linear(pn, pn['x_tv_imp'])[1], fit_linear_geoseason(pn, pn['x_tv_imp'])[1],
                        fit_latent_loading(pn)[1], fit_linear_geoseason(pn, pn['x_tv_true'])[1]])
        o = np.array(out)
        print(f"rho_SD={rho}: b_d bias plugin {o[:,0].mean()-1:+.3f} | +geo x season {o[:,1].mean()-1:+.3f} (sd {o[:,1].std():.3f}) | latent {o[:,2].mean()-1:+.3f} (sd {o[:,2].std():.3f}) | oracle+geoseason sd {o[:,3].std():.3f}")

def e15(seed=15, N=2000, alpha=1.3, n_ch=3, err=0.10, nrep=300, n_fam=200):
    """Attack: LLM errors clustered within campaign families (shared naming pattern). Certify by line vs by family mass."""
    rng = np.random.default_rng(seed)
    s, c = line_items(N, alpha, n_ch, rng); fam = rng.integers(0, n_fam, size=N)
    true = channel_totals(s, c, n_ch)
    fam_mass = np.bincount(fam, weights=s, minlength=n_fam)
    for budget in [0, 50, 100, 200]:
        res = {}
        for strat in (['llm'] if budget == 0 else ['topk_line', 'topk_family', 'srs']):
            e = []
            for r in range(nrep):
                # family-level error: whole family mislabelled to a random other channel w.p. err
                bad = rng.random(n_fam) < err; shift = rng.integers(1, n_ch, size=n_fam)
                pred = np.where(bad[fam], (c + shift[fam]) % n_ch, c)
                lab = pred.copy()
                if strat == 'topk_line':
                    idx = np.argsort(-s)[:budget]; lab[idx] = c[idx]
                elif strat == 'topk_family':
                    # audit one line per family in mass order (certifying a family fixes all its lines)
                    fams = np.argsort(-fam_mass)[:budget]; mask = np.isin(fam, fams); lab[mask] = c[mask]
                elif strat == 'srs':
                    samp = rng.choice(N, size=budget, replace=False)
                    corr = np.array([N/budget*np.sum(s[samp]*((c[samp] == k).astype(float)-(pred[samp] == k))) for k in range(n_ch)])
                    e.append((channel_totals(s, lab, n_ch)+corr)/true-1); continue
                e.append(channel_totals(s, lab, n_ch)/true-1)
            e = np.array(e); res[strat] = np.sqrt((e**2).mean(0)).mean()
        print(f"budget={budget}: " + "  ".join(f"{k} rmse {v:.3f}" for k, v in res.items()))

def e16(seed=16, N=2000, n_ch=3, err=0.10, nrep=400):
    """MSE law for certify-top-k: predicted tail variance sum_{i>k} s_i^2 * v vs empirical; audit-size rule."""
    rng = np.random.default_rng(seed)
    M = np.eye(n_ch)*(1-err); M[0, 1] += err; M[1, 2] += err; M[2, 0] += err
    for alpha in [1.3, 2.0, 3.0]:
        s, c = line_items(N, alpha, n_ch, rng); true = channel_totals(s, c, n_ch)
        order = np.argsort(-s); ss = s[order]; cc = c[order]
        print(f"alpha={alpha}:")
        for k in [0, 20, 50, 100, 200]:
            # predicted for channel 0: outflow var err(1-err) over true-0 tail lines + inflow var from true-2 tail lines; bias = err*(sum s2 - sum s0) over tail
            tail = slice(k, None)
            s0 = ss[tail][cc[tail] == 0]; s2 = ss[tail][cc[tail] == 2]
            var = err*(1-err)*((s0**2).sum() + (s2**2).sum()); bias = err*(s2.sum()-s0.sum())
            pred_rmse = np.sqrt(var + bias**2)/true[0]
            e = []
            for r in range(nrep):
                pred = llm_label(c, s, M, rng); lab = pred.copy(); idx = order[:k]; lab[idx] = c[idx]
                e.append(channel_totals(s, lab, n_ch)[0]/true[0]-1)
            e = np.array(e)
            print(f"   k={k:4d} ({k/N*100:.1f}% lines, {ss[:k].sum()/s.sum()*100:.0f}% spend): empirical rmse {np.sqrt((e**2).mean()):.4f} (bias {e.mean():+.4f})  predicted {pred_rmse:.4f} (bias {bias/true[0]:+.4f})")

def e17(nrep=15, seed=17):
    """Hill operating-point sweep: leak sign flips across the knee; plug-in vs hierarchical-beta national mROAS."""
    rng = np.random.default_rng(seed)
    for scale in [0.15, 0.3, 0.6, 1.0, 1.5, 3.0]:
        res = []
        for r in range(nrep):
            pn = make_panel(rng=rng, rho_SD=0.9, rho_ez=0.8, sig_eta=0.5, sig_zeta=0.5, K_tv=1.5, s_tv=1.0, beta_tv=3.0, beta_d=1.0, sigma=0.3, scale_tv=scale)
            fp = fit_hill_model(pn, 'plugin'); fh = fit_hill_model(pn, 'hier'); tm = true_mroas(pn)
            res.append([fp['b_d']-1, fh['b_d']-1, leak_hill_ovb(pn)[1], national_mroas(pn, fp, 'plugin')/tm-1, national_mroas(pn, fh, 'hier')/tm-1])
        res = np.array(res); se = res.std(0)/np.sqrt(nrep)
        print(f"a/K={scale*2/1.5:.2f}: b_d bias plugin {res[:,0].mean():+.3f}±{se[0]:.3f} hier {res[:,1].mean():+.3f}±{se[1]:.3f} | OVB@trueK {res[:,2].mean():+.3f} | mROAS rel.bias plugin {res[:,3].mean():+.3f}±{se[3]:.3f} hier {res[:,4].mean():+.3f}±{se[4]:.3f}")

def e18(nrep=100, seed=18, T=104, lines_per_week=60, err=0.10, alpha=1.3):
    """Time-varying mixing: weekly line items LLM-labelled; MMM (2 channels, linear, national) fitted on observed weekly channel spends.
    Compare no-audit vs certify top-k per week; measure beta bias/RMSE."""
    rng = np.random.default_rng(seed)
    n_ch = 2; M = np.array([[1-err, err], [err/2, 1-err/2]])
    out = {k: [] for k in ['raw', 'top5', 'top10']}
    for r in range(nrep):
        S_true = np.zeros((T, n_ch)); S_obs = {k: np.zeros((T, n_ch)) for k in out}
        for t in range(T):
            s, c = line_items(lines_per_week, alpha, n_ch, rng, ch_probs=np.array([0.6, 0.4]))
            pred = llm_label(c, s, M, rng)
            S_true[t] = channel_totals(s, c, n_ch); S_obs['raw'][t] = channel_totals(s, pred, n_ch)
            for k, nm in [(3, 'top5'), (6, 'top10')]:
                lab = pred.copy(); idx = np.argsort(-s)[:k]; lab[idx] = c[idx]; S_obs[nm][t] = channel_totals(s, lab, n_ch)
        beta = np.array([1.0, 2.0]); y = S_true @ beta + 20*rng.normal(size=T)
        for k in out:
            X = np.column_stack([S_obs[k], np.ones(T)]); b = lstsq(X, y, rcond=None)[0][:2]; out[k].append(b/beta-1)
    for k in out:
        o = np.array(out[k]); print(f"{k:5s}: beta rel.bias {o.mean(0).round(3)}  rmse {np.sqrt((o**2).mean(0)).round(3)}")

def e19(seed=19):
    """Identification failure of the latent-loading HILL model near the knee: profile SSR flat in K, eta_hat uncorrelated with eta."""
    rng = np.random.default_rng(seed)
    pn = make_panel(rng=rng, rho_SD=0.9, rho_ez=0.8, sig_eta=0.5, sig_zeta=0.5, K_tv=1.5, s_tv=1.0, beta_tv=3.0, beta_d=1.0, sigma=0.3)
    T, G = pn['T'], pn['G']; F = fourier(T)[:, 1:]; a_d = demean_geo(adstock(pn['x_d'], pn['alpha_d'])); Fl = np.tile(F, (G, 1))
    a_base = adstock(pn['x_tv_imp'], pn['alpha_tv']); yy = demean_geo(pn['y']).ravel(order='F')
    def res(K, eta, shrink=1.0):
        f = hill(a_base*(1+eta[None, :]), K, 1.0)
        X = np.column_stack([demean_geo(f).ravel(order='F'), a_d.ravel(order='F'), Fl]); b = lstsq(X, yy, rcond=None)[0]
        return np.concatenate([yy-X @ b, np.sqrt(shrink)*eta])
    print("SSR at truth (K=1.5, true eta):", (res(1.5, pn['eta'])**2).sum(), " at (K=1.5, eta=0):", (res(1.5, np.zeros(G))**2).sum())
    for K in [0.5, 1.0, 1.5, 2.0, 3.0]:
        sol = least_squares(lambda th: res(K, th), np.zeros(G), max_nfev=200)
        print(f"  K={K}: profiled SSR {(sol.fun**2).sum():.2f}  corr(eta_hat, eta)={np.corrcoef(sol.x, pn['eta'])[0,1]:+.2f}")

def e20(nrep=30, seed=20):
    """Refine: latent-loading repair with the imputed channel's adstock PROFILED (grid) instead of assumed."""
    rng = np.random.default_rng(seed); grid = [0.3, 0.4, 0.5, 0.6, 0.7]; picks, bds = [], []
    for r in range(nrep):
        pn = make_panel(rng=rng, rho_SD=0.9, rho_ez=0.8, sig_eta=0.5, sig_zeta=0.5); ssr, bd = [], []
        for a in grid:
            pn2 = dict(pn); pn2['alpha_tv'] = a; T, G = pn['T'], pn['G']
            bt, b, lam = fit_latent_loading(pn2)
            a_S = adstock(pn['S'], a); a_d = adstock(pn['x_d'], pn['alpha_d']); F = fourier(T)
            y = demean_geo(pn['y']).ravel(order='F'); aS_dm = a_S-a_S.mean(); Xg = np.zeros((T*G, G))
            for g in range(G): Xg[g*T:(g+1)*T, g] = aS_dm
            X = np.column_stack([Xg, demean_geo(a_d).ravel(order='F'), np.tile(F[:, 1:], (G, 1))])
            bb = lstsq(X, y, rcond=None)[0]; ssr.append(((y-X @ bb)**2).sum()); bd.append(b)
        i = int(np.argmin(ssr)); picks.append(grid[i]); bds.append(bd[i])
    print('alpha picked:', np.unique(picks, return_counts=True), ' b_d bias at picked alpha %+.3f (sd %.3f)' % (np.mean(bds)-1, np.std(bds)))

def quick():
    print("== E1 leakage law =="); e1(nrep=60)
    print("== E2 latent repair =="); e2(nrep=30)
    print("== E4 Hill mROAS =="); e4(nrep=10)
    print("== E9 audit =="); e9(nrep=100)
    print("== E10 ledger =="); e10(nrep=100)

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'quick'
    globals()[cmd]()
