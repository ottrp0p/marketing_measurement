"""Dive 19 - EXTEND of dive 16: stall diagnostics, multichannel stall geometry, adstock/confounding.
Usage: python 19_feedback_loop_inference_ext.py quick|all|E1..E14   (needs numpy, scipy).
quick ~2 min, all ~8 min on 4 cores.
"""
import sys, json, time
import numpy as np
from scipy import stats, special, optimize

# --------------------------------------------------------------------------------------
# 1. Single-channel Hill loop with optional adstock, demand confounder, exploration, AR(1) noise
# --------------------------------------------------------------------------------------
def hill(x, K, s=1.0):
    xs = np.power(np.maximum(x, 1e-12), s)
    return xs / (xs + K ** s)

def dhill(x, K, s=1.0):
    x = np.maximum(x, 1e-12)
    return s * K ** s * x ** (s - 1) / (x ** s + K ** s) ** 2

def profile_fit(X_a, Y, Kgrid, lam_grid, s, D=None):
    """X_a: (T,R) spend history; Y: (T,R); returns fitted (alpha,beta,K,lam,[gamma]) per run by profile OLS.
    D: optional observed demand (T,R) included as a regressor."""
    T, R = X_a.shape
    best = np.full(R, np.inf); out = {}
    for lam in lam_grid:
        # adstock path for this lam
        A = np.zeros_like(X_a)
        a = np.zeros(R)
        for t in range(T):
            a = X_a[t] + lam * a; A[t] = a
        for K in Kgrid:
            H = hill(A, K, s)
            cols = [np.ones((T, R)), H] + ([D] if D is not None else [])
            p = len(cols)
            Xm = np.stack(cols, axis=2)  # T,R,p
            XtX = np.einsum('tri,trj->rij', Xm, Xm)
            Xty = np.einsum('tri,tr->ri', Xm, Y)
            try:
                coef = np.linalg.solve(XtX, Xty[..., None])[..., 0]
            except np.linalg.LinAlgError:
                continue
            res = Y - np.einsum('tri,ri->tr', Xm, coef)
            rss = (res ** 2).sum(0)
            better = rss < best
            if not better.any():
                continue
            best = np.where(better, rss, best)
            for i, name in enumerate(['alpha', 'beta', 'gamma'][:p]):
                out.setdefault(name, np.zeros(R))[better] = coef[better, i]
            out.setdefault('K', np.zeros(R))[better] = K
            out.setdefault('lam', np.zeros(R))[better] = lam
    out['rss'] = best
    return out

def opt_spend(beta, K, lam, s, target=1.0, xmax=200.0):
    """Certainty-equivalent spend: long-run mROAS beta*H'(x/(1-lam))/(1-lam) = target; 0 if beta<=0."""
    if s == 1.0:  # closed form: beta*K/(a+K)^2/(1-lam) = target, a = x/(1-lam)
        a = np.sqrt(np.maximum(beta * K / ((1 - lam) * target), 0)) - K
        x = np.clip((1 - lam) * a, 0.05, xmax)
        return np.where(beta <= 0, 0.05, x)
    grid = np.linspace(0.05, xmax, 4000)
    prof = beta[None, :] * hill(grid[:, None] / (1 - lam[None, :]), K[None, :], s) - target * grid[:, None]
    x = grid[np.argmax(prof, axis=0)]
    return np.where(beta <= 0, 0.05, x)

def run_hill_loop(T=208, R=200, refresh=4, seed=0, beta=40.0, K=5.0, s=1.0, lam=0.0, sig=2.0,
                  explore=0.0, explore_decay=0.0, phi=0.0, gam=0.0, rho_d=0.8, sd_d=1.0, demand_obs=False,
                  ar1=0.0, T0=8, burn_amp=0.25, x0=None, Kgrid=None, lam_grid=None, fixed_spend=None,
                  impl_noise=0.0, log_states=True):
    """Returns dict of time series (per refresh) and final states."""
    rng = np.random.default_rng(seed)
    if Kgrid is None: Kgrid = np.geomspace(0.5, 60, 40)
    if lam_grid is None: lam_grid = np.array([lam]) if lam == 0 else np.linspace(0.0, 0.8, 9)
    # true optimum spend (long-run mROAS = 1)
    xt = opt_spend(np.array([beta]), np.array([K]), np.array([lam]), s)[0]
    if x0 is None: x0 = xt
    X = np.zeros((T, R)); Y = np.zeros((T, R)); Dm = np.zeros((T, R)); Xstar = np.full((T, R), np.nan)
    a = np.zeros(R); d = rng.normal(0, sd_d / np.sqrt(1 - rho_d ** 2), R); eps_prev = np.zeros(R)
    th = dict(alpha=np.zeros(R), beta=np.full(R, beta), K=np.full(R, K), lam=np.full(R, lam))
    xs_cur = np.full(R, x0)
    rec = dict(t=[], m_own=[], m_true=[], x=[], D_H=[], D_x=[], D_perp=[], beta=[], K=[], lam=[], alpha=[],
               m_true_star=[], loss_star=[])
    prof_opt = beta * hill(xt / (1 - lam), K, s) - xt
    for t in range(T):
        d = rho_d * d + rng.normal(0, sd_d, R)
        Dm[t] = d
        if t < T0:
            target = np.full(R, x0 * (1 + burn_amp * (-1) ** t))
        elif fixed_spend is not None:
            target = np.full(R, fixed_spend)
        else:
            if (t - T0) % refresh == 0:
                fit = profile_fit(X[:t], Y[:t], Kgrid, lam_grid, s, D=Dm[:t] if demand_obs else None)
                th = fit
                xs_cur = opt_spend(fit['beta'], fit['K'], fit['lam'], s)
                xs_cur = np.clip(xs_cur, 0.05 * x0, 20 * x0)
            target = xs_cur.copy()
        Xstar[t] = target
        v = explore * (1.0 if explore_decay == 0 else (max(t - T0 + 1, 1)) ** (-explore_decay))
        x = target + phi * d + (rng.normal(0, v, R) if v > 0 else 0) + (rng.normal(0, impl_noise, R) if impl_noise > 0 else 0)
        x = np.maximum(x, 0.02 * x0)
        a = x + lam * a
        e = ar1 * eps_prev + rng.normal(0, sig * np.sqrt(1 - ar1 ** 2), R); eps_prev = e
        y = beta * hill(a, K, s) + gam * d + e
        X[t] = x; Y[t] = y
        if t >= T0 and (t - T0) % refresh == 0 and log_states and fixed_spend is None:
            # model's own long-run mROAS at current spend vs truth
            A_ss = x / (1 - th['lam'])
            m_own = th['beta'] * dhill(A_ss, th['K'], s) / (1 - th['lam'])
            m_true = beta * dhill(x / (1 - lam), K, s) / (1 - lam)
            # information clocks
            Hf = np.stack([hill(X[u], th['K'], s) for u in range(t)])  # crude: at fitted K, no adstock
            rec['t'].append(t); rec['m_own'].append(m_own); rec['m_true'].append(m_true); rec['x'].append(x)
            rec['m_true_star'].append(beta * dhill(target / (1 - lam), K, s) / (1 - lam))
            rec['loss_star'].append(prof_opt - (beta * hill(target / (1 - lam), K, s) - target))
            rec['D_H'].append(((Hf - Hf.mean(0)) ** 2).sum(0)); rec['D_x'].append(((X[:t] - X[:t].mean(0)) ** 2).sum(0))
            rec['D_perp'].append(np.nansum((X[T0:t] - Xstar[T0:t]) ** 2, axis=0))
            for k in ['beta', 'K', 'lam', 'alpha']: rec[k].append(th[k])
    out = {k: np.array(v) for k, v in rec.items()}
    out['X'] = X; out['Y'] = Y; out['Dm'] = Dm; out['Xstar'] = Xstar; out['x_true_opt'] = xt; out['x0'] = x0
    return out

# --------------------------------------------------------------------------------------
# 2. Diagnostics from logs only
# --------------------------------------------------------------------------------------
def granger_stat(X, Y, p=4, start=8):
    """F-stat for H0: lagged y do not predict x_t given lagged x, per run. X,Y: (T,R)."""
    T, R = X.shape
    rows = range(start + p, T)
    F = np.zeros(R)
    for r in range(R):
        x = X[:, r]; y = Y[:, r]
        Z1 = np.stack([np.ones(len(rows))] + [np.array([x[t - k] for t in rows]) for k in range(1, p + 1)], axis=1)
        Z2 = np.stack([np.array([y[t - k] for t in rows]) for k in range(1, p + 1)], axis=1)
        xt = np.array([x[t] for t in rows])
        def rss(Z):
            c, *_ = np.linalg.lstsq(Z, xt, rcond=None); return ((xt - Z @ c) ** 2).sum()
        r1 = rss(Z1); r2 = rss(np.hstack([Z1, Z2]))
        n = len(rows); k1 = Z1.shape[1]; k2 = k1 + p
        F[r] = ((r1 - r2) / p) / max(r2 / (n - k2), 1e-300)
    return F

def laiwei_clock(X, s_of_x, t_list):
    """lambda_min of design (1, f(x)) vs log lambda_max, at times in t_list."""
    out = []
    for t in t_list:
        Z = np.stack([np.ones(t), s_of_x(X[:t])], axis=1)
        V = Z.T @ Z
        ev = np.linalg.eigvalsh(V)
        out.append((ev[0], np.log(ev[-1])))
    return np.array(out)

def echo_fa_theory(eps, se_vec, corr, k, nsim=200000, seed=1):
    """P(all k consecutive |m_j-1|<=eps) for a genuinely breakeven, non-stalled channel with Gaussian
    refresh errors of sd se_j and correlation matrix corr."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr + 1e-12 * np.eye(k))
    z = rng.normal(size=(nsim, k)) @ L.T * se_vec
    return (np.abs(z) <= eps).all(1).mean()

# --------------------------------------------------------------------------------------
# 3. Multichannel quadratic loop with budget coupling (local model of the Hill loop)
# --------------------------------------------------------------------------------------
def quad_loop(n=3, T=208, R=400, refresh=4, seed=0, b=None, c=None, B=None, sig=1.0, T0=12,
              V_explore=None, explore_scale=0.0, explore_decay=0.0, burn_amp=0.3, switch_sched=None):
    """y = alpha + sum_j (b_j x_j - c_j x_j^2/2) + eps ; allocator equalises fitted marginal returns under sum x = B.
    Exploration: budget-neutral xi ~ N(0, explore_scale^2 * V_explore) (V_explore has 1'V1 = 0), or a switch schedule."""
    rng = np.random.default_rng(seed)
    if b is None: b = np.array([3.0, 2.5, 2.0])[:n]
    if c is None: c = np.array([0.2, 0.5, 1.0])[:n]
    P = np.eye(n) - np.ones((n, n)) / n
    # true optimum under budget
    def alloc(bh, ch, B):
        # b_j - c_j x_j = lam, sum x = B  ->  x_j = (b_j - lam)/c_j ; lam = (sum b/c - B)/sum(1/c)
        lam = ((bh / ch).sum(-1) - B) / (1 / ch).sum(-1)
        return (bh - lam[..., None]) / ch
    if B is None: B = 10.0
    x_true = alloc(b, c, B)
    X = np.zeros((T, R, n)); Y = np.zeros((T, R))
    bh = np.tile(b, (R, 1)); ch = np.tile(c, (R, 1))
    xs_cur = np.tile(x_true, (R, 1))
    rec = dict(t=[], gdiff_err=[], x=[], loss=[], D_min=[], own_lam=[], true_lam_spread=[], g1_err=[])
    Lch = None if V_explore is None else np.linalg.cholesky(V_explore + 1e-10 * np.eye(n))
    for t in range(T):
        if t < T0:
            # burn-in: rotate budget-neutral pulses across channels
            pulse = np.zeros(n); j = t % n; pulse[j] = burn_amp * B / n; pulse -= pulse.sum() / n
            target = np.tile(x_true + pulse * (-1) ** (t // n), (R, 1))
        else:
            if (t - T0) % refresh == 0:
                # OLS on (1, x_j, -x_j^2/2)
                Z = np.concatenate([np.ones((t, R, 1)), X[:t], -X[:t] ** 2 / 2], axis=2)
                ZtZ = np.einsum('tri,trj->rij', Z, Z); Zty = np.einsum('tri,tr->ri', Z, Y[:t])
                coef = np.linalg.solve(ZtZ + 1e-9 * np.eye(2 * n + 1), Zty[..., None])[..., 0]
                bh = coef[:, 1:n + 1]; ch = np.maximum(coef[:, n + 1:], 0.02)
                xs_cur = np.clip(alloc(bh, ch, B), 0.02 * B, B)
                xs_cur = xs_cur * (B / xs_cur.sum(1, keepdims=True))
            target = xs_cur
        x = target.copy()
        sc = explore_scale * (1.0 if explore_decay == 0 else max(t - T0 + 1, 1) ** (-explore_decay))
        if Lch is not None and sc > 0:
            x = x + sc * (rng.normal(size=(R, n)) @ Lch.T)
        if switch_sched is not None:
            i, j, amp = switch_sched[(t - T0) % len(switch_sched)]
            sgn = rng.choice([-1, 1], R)
            x[:, i] += sgn * amp; x[:, j] -= sgn * amp
        x = np.maximum(x, 0.01)
        y = (b * x - c * x ** 2 / 2).sum(1) + rng.normal(0, sig, R)
        X[t] = x; Y[t] = y
        if t >= T0 and (t - T0) % refresh == 0:
            g_true = b - c * x; g_fit = bh - ch * x
            gd_true = g_true @ P.T; gd_fit = g_fit @ P.T  # feasible-direction gradients
            rec['t'].append(t); rec['gdiff_err'].append(gd_fit - gd_true); rec['x'].append(x)
            rec['g1_err'].append((g_fit - g_true).mean(1))  # budget-direction (total marginal value) error
            rec['loss'].append(((b * x_true - c * x_true ** 2 / 2).sum() - (b * x - c * x ** 2 / 2).sum(1)))
            rec['own_lam'].append(g_fit); rec['true_lam_spread'].append(gd_true)
    out = {k: np.array(v) for k, v in rec.items()}
    out['X'] = X; out['x_true'] = x_true; out['P'] = P; out['b'] = b; out['c'] = c
    return out

def switch_representation(Hinv_feas, n):
    """Find nonneg amplitudes a_ij^2 with sum a_ij^2 d_ij d_ij' = Hinv_feas (on feasible subspace)."""
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    cols = []
    for i, j in pairs:
        d = np.zeros(n); d[i] = 1; d[j] = -1
        cols.append(np.outer(d, d).ravel())
    A = np.stack(cols, axis=1)
    w, rnorm = optimize.nnls(A, Hinv_feas.ravel())
    return pairs, w, rnorm

# --------------------------------------------------------------------------------------
# 4. AR(1)-robust confidence sequence in the linear loop (mixture over phi grid)
# --------------------------------------------------------------------------------------
def nig_logm(XtX, Xty, yty, n, mu0, lam0, a0, b0):
    Ln = lam0 + XtX
    rhs = lam0 @ mu0 + Xty
    mun = np.linalg.solve(Ln, rhs)
    an = a0 + n / 2
    bn = b0 + 0.5 * (yty + mu0 @ lam0 @ mu0 - mun @ Ln @ mun)
    return (special.gammaln(an) - special.gammaln(a0) + a0 * np.log(b0) - an * np.log(bn)
            + 0.5 * (np.linalg.slogdet(lam0)[1] - np.linalg.slogdet(Ln)[1]) - (n / 2) * np.log(2 * np.pi))

def linear_loop_cs(T=104, R=400, g=10.0, ar1=0.6, sig=1.0, a=10.0, b=2.0, x0=5.0, v=1.0, T0=8, seed=0,
                   delta=0.05, phis=np.linspace(-0.2, 0.9, 12), check_every=1):
    """Smooth linear loop x_t = x0 + g(bhat-b) + xi; errors AR(1). Returns time-uniform coverage of beta for
    (i) naive fixed-time interval at T, (ii) iid-mixture CS, (iii) AR(1)-mixture CS (sup over phi grid)."""
    rng = np.random.default_rng(seed)
    mu0 = np.array([a, b]); lam0 = np.diag([1e-2, 1e-2]); a0, b0 = 2.0, 1.0
    cov_iid = np.ones(R, bool); cov_ar = np.ones(R, bool); cov_naive_T = np.zeros(R, bool)
    hw_iid = np.zeros(R); hw_ar = np.zeros(R); hw_naive = np.zeros(R)
    for r in range(R):
        x = np.zeros(T); y = np.zeros(T); e = 0.0; bhat = b
        for t in range(T):
            xt = x0 * (1 + 0.25 * (-1) ** t) if t < T0 else x0 + g * (bhat - b) + rng.normal(0, v)
            xt = max(xt, 0.05 * x0)
            e = ar1 * e + rng.normal(0, sig * np.sqrt(1 - ar1 ** 2))
            x[t] = xt; y[t] = a + b * xt + e
            if t >= T0 - 1:
                Z = np.stack([np.ones(t + 1), x[:t + 1]], 1)
                coef, *_ = np.linalg.lstsq(Z, y[:t + 1], rcond=None); bhat = coef[1]
            if t >= T0 and (t % check_every == 0 or t == T - 1):
                n = t + 1; Z = np.stack([np.ones(n), x[:n]], 1); yy = y[:n]
                # iid mixture
                XtX = Z.T @ Z; Xty = Z.T @ yy; yty = yy @ yy
                logm = nig_logm(XtX, Xty, yty, n, mu0, lam0, a0, b0)
                Rstar = (n / (2 * np.pi)) * np.exp(-2 * (logm + np.log(delta)) / n - 1)
                # RSS at true beta minimised over alpha
                res = yy - b * x[:n]; rss_b = ((res - res.mean()) ** 2).sum()
                if rss_b > Rstar: cov_iid[r] = False
                # AR mixture: for each phi, transformed data (exact conditional likelihood given y_0 handled by dropping first obs)
                logms = []; rss_bs = []
                for ph in phis:
                    ys = yy[1:] - ph * yy[:-1]; Zs = Z[1:] - ph * Z[:-1]
                    XtXs = Zs.T @ Zs; Xtys = Zs.T @ ys; ytys = ys @ ys
                    logms.append(nig_logm(XtXs, Xtys, ytys, n - 1, mu0, lam0, a0, b0))
                    rs = ys - b * Zs[:, 1]; wgt = Zs[:, 0]
                    # min over alpha of sum (rs - alpha*wgt)^2
                    al = (rs @ wgt) / (wgt @ wgt); rss_bs.append(((rs - al * wgt) ** 2).sum())
                logm_ar = special.logsumexp(logms) - np.log(len(phis))
                m = n - 1
                Rstar_ar = (m / (2 * np.pi)) * np.exp(-2 * (logm_ar + np.log(delta)) / m - 1)
                if min(rss_bs) > Rstar_ar: cov_ar[r] = False
                if t == T - 1:
                    coef, res_, *_ = np.linalg.lstsq(Z, yy, rcond=None)
                    resid = yy - Z @ coef; s2 = resid @ resid / (n - 2)
                    se = np.sqrt(s2 * np.linalg.inv(XtX)[1, 1])
                    cov_naive_T[r] = abs(coef[1] - b) <= 1.96 * se; hw_naive[r] = 1.96 * se
                    rss_min = resid @ resid
                    hw_iid[r] = np.sqrt(max(Rstar - rss_min, 0) * np.linalg.inv(XtX)[1, 1])
                    # AR width: profile over phi at the best phi (approx): use phi maximising logm
                    ph = phis[int(np.argmax(logms))]
                    ys = yy[1:] - ph * yy[:-1]; Zs = Z[1:] - ph * Z[:-1]
                    cs, *_ = np.linalg.lstsq(Zs, ys, rcond=None); rmin = ((ys - Zs @ cs) ** 2).sum()
                    hw_ar[r] = np.sqrt(max(Rstar_ar - rmin, 0) * np.linalg.inv(Zs.T @ Zs)[1, 1])
    return dict(cov_iid=cov_iid.mean(), cov_ar=cov_ar.mean(), cov_naive_T=cov_naive_T.mean(),
                hw_naive=np.median(hw_naive), hw_iid=np.median(hw_iid), hw_ar=np.median(hw_ar))

# --------------------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------------------
RES = {}
def log(name, **kw):
    RES[name] = kw
    print(f"[{name}]", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in kw.items()}, default=lambda o: np.round(o, 4).tolist() if hasattr(o, 'tolist') else str(o)))

def E1(R=300, T=208):
    """Three regimes of the single-channel Hill loop: stall, exogenous exploration, demand-chasing (confounded)."""
    cfg = dict(stall=dict(), explore=dict(explore=1.0), chase=dict(phi=1.5, gam=6.0), chase_obs=dict(phi=1.5, gam=6.0, demand_obs=True))
    res = {}
    for name, kw in cfg.items():
        o = run_hill_loop(T=T, R=R, seed=1, **kw)
        me = o['m_own'][-1] - o['m_true'][-1]
        mt_err = o['m_true'][-1] - 1.0  # true mROAS at the loop's spend vs the target
        res[name] = dict(D_x_T=float(np.median(o['D_x'][-1])), D_x_half=float(np.median(o['D_x'][len(o['t']) // 2])),
                         D_perp_T=float(np.median(o['D_perp'][-1])),
                         m_own_mean=float(o['m_own'][-1].mean()), m_own_sd=float(o['m_own'][-1].std()),
                         m_true_sd=float(o['m_true'][-1].std()), m_true_mean=float(o['m_true'][-1].mean()),
                         bias_m=float(np.mean(me)), rmse_m=float(np.sqrt(np.mean(me ** 2))),
                         x_bias=float(np.mean(o['x'][-1]) / o['x_true_opt'] - 1),
                         beta_q=np.percentile(o['beta'][-1], [5, 50, 95]).round(1).tolist())
        log('E1_' + name, **res[name])
    return res

def E2(R=300, T=208):
    """Diagnostic pair from logs only: Granger (y->x) feedback test and the Lai-Wei slope-eigenvalue clock."""
    cfg = dict(stall=dict(), stall_weekly=dict(refresh=1), explore=dict(explore=1.0), explore_weekly=dict(explore=1.0, refresh=1),
               chase=dict(phi=1.5, gam=6.0), chase_weekly=dict(phi=1.5, gam=6.0, refresh=1), fixed=dict(fixed_spend=None))
    out = {}
    for name, kw in cfg.items():
        if name == 'fixed':
            xt = opt_spend(np.array([40.0]), np.array([5.0]), np.array([0.0]), 1.0)[0]
            o = run_hill_loop(T=T, R=R, seed=2, explore=1.0, fixed_spend=xt, log_states=False)
        else:
            o = run_hill_loop(T=T, R=R, seed=2, **kw)
        F = granger_stat(o['X'], o['Y'], p=4, start=8)
        crit = stats.f.ppf(0.95, 4, T - 8 - 4 - 9)
        rej = float((F > crit).mean())
        # slope-direction eigenvalue clock
        Kt = 5.0
        lam_min_T = []; lam_min_half = []; loglmax = []
        for r in range(min(R, 100)):
            cl = laiwei_clock(o['X'][:, r:r + 1][:, 0], lambda x: hill(x, Kt), [T // 2, T])
            lam_min_half.append(cl[0, 0]); lam_min_T.append(cl[1, 0]); loglmax.append(cl[1, 1])
        out[name] = dict(granger_rej=rej, F_med=float(np.median(F)), lam_min_half=float(np.median(lam_min_half)),
                         lam_min_T=float(np.median(lam_min_T)), growth_ratio=float(np.median(np.array(lam_min_T) / np.array(lam_min_half))),
                         log_lam_max=float(np.median(loglmax)))
        log('E2_' + name, **out[name])
    return out

def E3(R=400, T=208):
    """Echo test: false-alarm rate for a genuinely breakeven channel at fixed spend with exogenous dispersion,
    vs detection under the stall; refresh-error correlation from overlapping windows."""
    xt = run_hill_loop(T=10, R=1, seed=0)['x_true_opt']
    # (a) non-stalled genuinely breakeven: fixed spend at the true optimum + exploration v=1 -> fit each 4 weeks, m_own at x_t
    o = run_hill_loop(T=T, R=R, seed=3)  # stalled loop
    # build a 'fixed + explore' variant by evaluating m_own at refreshes with fixed spend: reuse run with fixed_spend but logging
    rng = np.random.default_rng(5)
    Kgrid = np.geomspace(0.5, 60, 40)
    X = np.zeros((T, R)); Y = np.zeros((T, R)); mo = []; ts = []
    for t in range(T):
        x = xt * (1 + 0.25 * (-1) ** t) if t < 8 else xt + rng.normal(0, 1.0, R)
        x = np.maximum(x, 0.1); X[t] = x; Y[t] = 40 * hill(x, 5.0) + rng.normal(0, 2.0, R)
        if t >= 8 and (t - 8) % 4 == 0 and t >= 12:
            f = profile_fit(X[:t], Y[:t], Kgrid, np.array([0.0]), 1.0)
            mo.append(f['beta'] * dhill(x, f['K'])); ts.append(t)
    mo = np.array(mo)
    err = mo - 1.0
    mprime = f['beta'] * (-2 * f['K'] / (x + f['K']) ** 3)  # fitted d mROAS/dx at last refresh, per run
    se_by_t = err.std(1)
    C = np.corrcoef(err[-6:])
    res = {}
    for eps in [0.01, 0.02, 0.05]:
        for k in [1, 3, 6]:
            hit = (np.abs(err[-k:]) <= eps).all(0).mean()
            th = echo_fa_theory(eps, se_by_t[-k:], C[-k:, -k:] if k > 1 else np.ones((1, 1)), k)
            # refined: err_j = e (slow, common) + mprime_r * xi_j (iid, v=1)
            v_ex = 1.0; se_est = np.sqrt(max(se_by_t[-1] ** 2 - np.mean(mprime ** 2) * v_ex, 1e-6))
            rg = np.random.default_rng(0); e = rg.normal(0, se_est, (2000, 1)); mp = np.abs(mprime)[None, :] * np.sqrt(v_ex)
            pj = stats.norm.cdf((eps - e) / mp) - stats.norm.cdf((-eps - e) / mp)
            th2 = float((pj ** k).mean())
            res[f'FA_eps{eps}_k{k}'] = (float(hit), float(th), th2)
    # stall detection rate: fraction of stalled runs with |m_own-1|<=eps for last k refreshes
    for eps in [0.01, 0.02, 0.05]:
        res[f'stall_hit_eps{eps}_k6'] = float((np.abs(o['m_own'][-6:] - 1.0) <= eps).all(0).mean())
    res['se_m_last'] = float(se_by_t[-1]); res['mprime_rms'] = float(np.sqrt(np.mean(mprime**2))); res['mprime_true'] = float(40*(-2*5/(xt+5)**3)); res['corr_adjacent'] = float(C[-1, -2]); res['corr_lag5'] = float(C[-1, 0])
    log('E3', **res)
    return res

def E4(R=300, T=312):
    """Stall with adstock: does the stall and echo survive carryover; is lambda learned?"""
    res = {}
    for lam in [0.0, 0.5]:
        o = run_hill_loop(T=T, R=R, seed=4, lam=lam, refresh=4)
        n = len(o['t'])
        res[f'lam{lam}'] = dict(m_own_mean=float(o['m_own'][-1].mean()), m_own_sd=float(o['m_own'][-1].std()),
                                m_true_sd_half=float(o['m_true_star'][n // 2].std()), m_true_sd_T=float(o['m_true_star'][-1].std()),
                                loss_star_T=float(o['loss_star'][-1].mean()),
                                lam_hat_sd_half=float(o['lam'][n // 2].std()), lam_hat_sd_T=float(o['lam'][-1].std()),
                                lam_hat_mean_T=float(o['lam'][-1].mean()),
                                D_x_half=float(np.median(o['D_x'][n // 2])), D_x_T=float(np.median(o['D_x'][-1])),
                                x_sd_T=float(o['x'][-1].std()))
        log(f'E4_lam{lam}', **res[f'lam{lam}'])
        o2 = run_hill_loop(T=T, R=R, seed=4, lam=lam, refresh=4, explore=1.5, explore_decay=0.25)
        res[f'lam{lam}_explore'] = dict(m_true_sd_T=float(o2['m_true_star'][-1].std()), lam_hat_sd_T=float(o2['lam'][-1].std()),
                                        m_own_sd=float(o2['m_own'][-1].std()), loss_star_T=float(o2['loss_star'][-1].mean()))
        log(f'E4_lam{lam}_explore', **res[f'lam{lam}_explore'])
    return res

def E5(R=400, T=208):
    """Multichannel quadratic loop under budget: feasible-direction gradient error frozen vs exploration."""
    res = {}
    o = quad_loop(n=3, T=T, R=R, seed=5)
    n = len(o['t'])
    e = o['gdiff_err']
    res['stall'] = dict(gd_rmse_half=float(np.sqrt((e[n // 2] ** 2).mean())), gd_rmse_T=float(np.sqrt((e[-1] ** 2).mean())),
                        own_lam_spread=float(np.abs(o['own_lam'][-1] @ o['P'].T).mean()),
                        true_lam_spread=float(np.abs(o['true_lam_spread'][-1]).mean()),
                        loss_T=float(o['loss'][-1].mean()), x_sd_T=float(o['x'][-1].std(0).mean()),
                        g1_rmse_T=float(np.sqrt((o['g1_err'][-1] ** 2).mean())), g1_median_abs=float(np.median(np.abs(o['g1_err'][-1]))))
    log('E5_stall', **res['stall'])
    P = o['P']
    o2 = quad_loop(n=3, T=T, R=R, seed=5, V_explore=P, explore_scale=0.5)
    e = o2['gdiff_err']
    res['explore'] = dict(gd_rmse_half=float(np.sqrt((e[n // 2] ** 2).mean())), gd_rmse_T=float(np.sqrt((e[-1] ** 2).mean())),
                          own_lam_spread=float(np.abs(o2['own_lam'][-1] @ P.T).mean()), loss_T=float(o2['loss'][-1].mean()),
                          g1_rmse_T=float(np.sqrt((o2['g1_err'][-1] ** 2).mean())), g1_median_abs=float(np.median(np.abs(o2['g1_err'][-1]))))
    log('E5_explore', **res['explore'])
    # control: a non-budget-neutral pulse in the total (burn-in style) identifies the budget direction
    o3 = quad_loop(n=3, T=T, R=R, seed=5, V_explore=np.eye(3), explore_scale=0.5)  # NOT budget neutral: 1'V1 != 0
    res['explore_total'] = dict(g1_rmse_T=float(np.sqrt((o3['g1_err'][-1] ** 2).mean())), g1_median_abs=float(np.median(np.abs(o3['g1_err'][-1]))),
                                gd_rmse_T=float(np.sqrt((o3['gdiff_err'][-1] ** 2).mean())))
    log('E5_explore_total', **res['explore_total'])
    return res

def passive_design(V, scale, T=208, R=300, seed=0, b=None, c=None, B=10.0, sig=1.0, T0=12, burn_amp=0.3, Cw=None, refresh=4):
    """Spend fixed at the true optimum + exploration N(0, scale^2 V) (no loop); fit quadratic; return gradient error in feasible directions."""
    rng = np.random.default_rng(seed); n = len(b)
    P = np.eye(n) - np.ones((n, n)) / n
    lam = ((b / c).sum() - B) / (1 / c).sum(); x_true = (b - lam) / c
    L = np.linalg.cholesky(V + 1e-10 * np.eye(n))
    X = np.zeros((T, R, n)); Y = np.zeros((T, R))
    for t in range(T):
        if t < T0:
            pulse = np.zeros(n); pulse[t % n] = burn_amp * B / n; pulse -= pulse.mean()
            x = np.tile(x_true + pulse * (-1) ** (t // n), (R, 1))
        else:
            if Cw is not None and (t - T0) % refresh == 0:
                zeta = rng.multivariate_normal(np.zeros(n), Cw, size=R)  # exogenous 'wobble' matched to the loop's, independent of errors
            x = x_true + scale * (rng.normal(size=(R, n)) @ L.T) + (zeta if Cw is not None else 0)
        x = np.maximum(x, 0.01); X[t] = x; Y[t] = (b * x - c * x ** 2 / 2).sum(1) + rng.normal(0, sig, R)
    Z = np.concatenate([np.ones((T, R, 1)), X, -X ** 2 / 2], axis=2)
    ZtZ = np.einsum('tri,trj->rij', Z, Z); Zty = np.einsum('tri,tr->ri', Z, Y)
    coef = np.linalg.solve(ZtZ, Zty[..., None])[..., 0]
    bh = coef[:, 1:n + 1]; ch = coef[:, n + 1:]
    g_fit = bh - ch * x_true; g_true = b - c * x_true
    return (g_fit - g_true) @ P.T, x_true

def E6(R=300, T=208):
    """Curvature-whitened exploration law. (a) passive design (no loop): V ∝ H^-1 optimal, ratios tr(H)tr(H^-1)/(n-1)^2;
    (b) inside the CE loop, where the loop's own wobble is already H^-1-shaped and the optimal complement reverses."""
    b = np.array([3.0, 3.0, 3.5]); c = np.array([0.1, 0.4, 1.6]); n = 3
    P = np.eye(n) - np.ones((n, n)) / n
    H = np.diag(c); Hf = P @ H @ P; Hf_inv = np.linalg.pinv(Hf)
    res = {}
    designs = {'iso': P, 'Hinv': Hf_inv / np.trace(Hf_inv) * np.trace(P), 'H': Hf / np.trace(Hf) * np.trace(P)}
    tau = 0.2
    # water-filling complement: measure the loop's own (no-exploration) whitened wobble, fill its weak directions
    o0 = quad_loop(n=n, T=T, R=R, seed=6, b=b, c=c)
    xdev0 = o0['x'][-1] - o0['x_true']; Cw0 = np.cov(xdev0.T); Hs = np.sqrt(H)
    W0 = Hs @ Cw0 @ Hs; W0 = P @ W0 @ P
    ew, U = np.linalg.eigh(W0); ew = np.maximum(ew, 0)
    fill = U @ np.diag(np.where(ew > 1e-9, ew.max() - ew, 0.0)) @ U.T  # whitened complement (zero on 1-direction)
    Hsi = np.linalg.pinv(Hs @ P @ Hs) @ Hs  # pseudo H^{-1/2} on feasible subspace
    Vfill = np.linalg.pinv(P @ Hs @ P) @ fill @ np.linalg.pinv(P @ Hs @ P); Vfill = P @ Vfill @ P
    designs['fill'] = Vfill / np.trace(Vfill) * np.trace(P)
    for mode in ['passive', 'loop']:
        for name, V in designs.items():
            s2 = tau / (0.5 * np.trace(H @ V))
            if mode == 'passive':
                e, _ = passive_design(V, np.sqrt(s2), T=T, R=R, seed=6, b=b, c=c)
                extra = {}
            else:
                o = quad_loop(n=n, T=T, R=2 * R, seed=6, b=b, c=c, V_explore=V, explore_scale=np.sqrt(s2))
                e = o['gdiff_err'][-1]
                # loop wobble covariance (allocation deviation from the fixed target), whitened by H
                xdev = o['x'][-1] - o['x_true']; Cw = np.cov(xdev.T)
                extra = dict(wobble_whitened_eigs=np.linalg.eigvalsh(np.sqrt(H) @ Cw @ np.sqrt(H)).round(3).tolist(), realised_loss_T=float(o['loss'][-1].mean()))
            dl = 0.5 * np.einsum('ri,ij,rj->r', e, Hf_inv, e)
            rg = np.random.default_rng(1); boot = np.array([np.median(rg.choice(dl, len(dl))) for _ in range(300)])
            res[f'{mode}_{name}'] = dict(dec_loss=float(dl.mean()), se=float(dl.std() / np.sqrt(len(dl))), median=float(np.median(dl)),
                                         median_se=float(boot.std()), trim10=float(stats.trim_mean(dl, 0.1)), gd_rmse=float(np.sqrt((e ** 2).mean())), **extra)
            log(f'E6_{mode}_{name}', **res[f'{mode}_{name}'])
        for stat in ['dec_loss', 'median', 'trim10']:
            res[f'{mode}_iso_over_Hinv_{stat}'] = res[f'{mode}_iso'][stat] / res[f'{mode}_Hinv'][stat]
            res[f'{mode}_H_over_Hinv_{stat}'] = res[f'{mode}_H'][stat] / res[f'{mode}_Hinv'][stat]
            res[f'{mode}_fill_over_Hinv_{stat}'] = res[f'{mode}_fill'][stat] / res[f'{mode}_Hinv'][stat]
    # attack: is the loop's excess loss a feedback effect? passive design with an exogenous wobble of the loop's covariance
    for name in ['iso', 'Hinv']:
        V = designs[name]; s2 = tau / (0.5 * np.trace(H @ V))
        e, _ = passive_design(V, np.sqrt(s2), T=T, R=R, seed=6, b=b, c=c, Cw=Cw0)
        dl = 0.5 * np.einsum('ri,ij,rj->r', e, Hf_inv, e)
        res[f'passive_wobble_{name}'] = dict(dec_loss=float(dl.mean()), se=float(dl.std() / np.sqrt(R)))
        log(f'E6_passive_wobble_{name}', **res[f'passive_wobble_{name}'])
    ev = np.linalg.eigvalsh(Hf); ev = ev[ev > 1e-9]
    res['theory_iso_over_Hinv'] = float(ev.sum() * (1 / ev).sum() / (n - 1) ** 2)
    res['theory_H_over_Hinv'] = float((ev ** 2).sum() * (ev ** -2).sum() / (n - 1) ** 2)
    res['loop_noexplore_wobble_whitened_eigs'] = ew.round(3).tolist(); res['fill_design_whitened_eigs'] = np.linalg.eigvalsh(Hs @ designs['fill'] @ Hs).round(3).tolist()
    e0 = o0['gdiff_err'][-1]; res['loop_noexplore_dec_loss'] = float((0.5 * np.einsum('ri,ij,rj->r', e0, Hf_inv, e0)).mean())
    log('E6_summary', **{k: v for k, v in res.items() if not isinstance(v, dict)})
    return res

def E7():
    """Switch-representability of H^-1 exploration: NNLS over pairwise switch dictionary, random curvatures."""
    rng = np.random.default_rng(7)
    out = {}
    for n in [3, 4, 5, 6]:
        ok = 0; resid = []
        for trial in range(200):
            c = np.exp(rng.normal(0, 1.0, n))
            P = np.eye(n) - np.ones((n, n)) / n
            Hf_inv = np.linalg.pinv(P @ np.diag(c) @ P)
            pairs, w, rn = switch_representation(Hf_inv, n)
            resid.append(rn / np.linalg.norm(Hf_inv))
            if rn / np.linalg.norm(Hf_inv) < 1e-6: ok += 1
        out[f'n{n}'] = dict(frac_exact=ok / 200, med_rel_resid=float(np.median(resid)), max_rel_resid=float(np.max(resid)))
        log(f'E7_n{n}', **out[f'n{n}'])
    # closed-form check for diagonal H: a_ij^2 = ?  (report the NNLS weights for one case)
    c = np.array([0.1, 0.4, 1.6]); n = 3; P = np.eye(n) - np.ones((n, n)) / n
    Hf_inv = np.linalg.pinv(P @ np.diag(c) @ P)
    pairs, w, rn = switch_representation(Hf_inv, n)
    S = (1 / c).sum(); w_closed = [1 / (c[i] * c[j] * S) for i, j in pairs]
    out['example'] = dict(c=c.tolist(), pairs=[list(p) for p in pairs], w=w.round(4).tolist(), w_closed=np.round(w_closed, 4).tolist(), resid=float(rn))
    log('E7_example', **out['example'])
    # non-diagonal Hessians (cross-channel interactions) and the H-shaped design: representability fails generically
    for n in [3, 4, 5]:
        okH = 0; okHi = 0
        for trial in range(200):
            A = rng.normal(size=(n, n)); Hn = A @ A.T + n * np.eye(n)  # PD with off-diagonals
            P = np.eye(n) - np.ones((n, n)) / n
            for M, tag in [(np.linalg.pinv(P @ Hn @ P), 'Hi'), (P @ np.diag(np.exp(rng.normal(0, 1, n))) @ P, 'H')]:
                _, w, rn = switch_representation(M, n)
                if rn / np.linalg.norm(M) < 1e-6:
                    if tag == 'Hi': okHi += 1
                    else: okH += 1
        out[f'nondiag_n{n}'] = dict(frac_Hinv_nondiag_representable=okHi / 200, frac_H_diag_representable=okH / 200)
        log(f'E7_nondiag_n{n}', **out[f'nondiag_n{n}'])
    return out

def E8(R=300, T=312):
    """Hill multichannel (n=3) with budget: stall of true mROAS spread, and the equal-mROAS echo."""
    # simple 3-channel Hill loop via nonlinear LS per refit (slower; R small)
    rng = np.random.default_rng(8)
    beta = np.array([30.0, 25.0, 20.0]); K = np.array([3.0, 5.0, 8.0]); B = 15.0; sig = 2.0; T0 = 12; refresh = 8
    n = 3
    def resp(x, th):
        al = th[0]; bt = th[1:4]; Kk = np.exp(th[4:7])
        return al + (bt * hill(x, Kk)).sum(-1)
    def alloc(th):
        bt = th[1:4]; Kk = np.exp(th[4:7])
        # equalise beta_j H'(x_j) under sum x = B via 1-D search on lam
        def xs(lam):
            # solve beta K x^{-2}... for s=1: H'(x) = K/(x+K)^2 -> x = sqrt(beta K / lam) - K
            return np.maximum(np.sqrt(np.maximum(bt * Kk / lam, 1e-9)) - Kk, 0.02)
        lo, hi = 1e-3, 1e3
        for _ in range(60):
            mid = np.sqrt(lo * hi)
            if xs(mid).sum() > B: lo = mid
            else: hi = mid
        x = xs(np.sqrt(lo * hi)); return x * B / x.sum()
    th_true = np.concatenate([[0.0], beta, np.log(K)])
    x_true = alloc(th_true)
    m_true_spread = []; m_own_spread = []; loss = []; xsd = []
    for explore in [0.0, 1.0]:
        mts = []; mos = []; ls = []; mts_half = []
        for r in range(R // 3):
            X = np.zeros((T, n)); Y = np.zeros(T); th = th_true.copy(); xcur = x_true.copy()
            for t in range(T):
                if t < T0:
                    pulse = np.zeros(n); pulse[t % n] = 0.3 * B / n; pulse -= pulse.mean()
                    x = x_true + pulse * (-1) ** (t // n)
                else:
                    if (t - T0) % refresh == 0:
                        lb = np.array([-50, 0, 0, 0, np.log(0.3), np.log(0.3), np.log(0.3)]); ub = np.array([50, 500, 500, 500, np.log(100)] * 1 + [np.log(100)] * 2)
                        th0 = np.clip(th, lb + 1e-6, ub - 1e-6)
                        f = optimize.least_squares(lambda th_: Y[:t] - resp(X[:t], th_), th0, method='trf', bounds=(lb, ub), max_nfev=300)
                        th = f.x; xcur = alloc(th)
                    x = xcur.copy()
                    if explore > 0:
                        xi = rng.normal(0, explore * max(t - T0 + 1, 1) ** (-0.25), n); xi -= xi.mean(); x = x + xi
                    if t == T // 2:
                        mth = beta * dhill(xcur, K); mts_half.append(mth - mth.mean())
                x = np.maximum(x, 0.05); X[t] = x; Y[t] = resp(x, th_true) + rng.normal(0, sig)
            mt = beta * dhill(xcur, K); mo = th[1:4] * dhill(xcur, np.exp(th[4:7]))
            mts.append(mt - mt.mean()); mos.append(mo - mo.mean())
            ls.append(resp(x_true, th_true) - resp(xcur, th_true))
        mts = np.array(mts); mos = np.array(mos); mts_half = np.array(mts_half)
        m_true_spread.append((float(np.abs(mts_half).mean()), float(np.abs(mts).mean()))); m_own_spread.append(float(np.abs(mos).mean())); loss.append(float(np.mean(ls)))
    res = dict(true_mROAS_spread_stall=m_true_spread[0], own_mROAS_spread_stall=m_own_spread[0],
               true_mROAS_spread_explore=m_true_spread[1], own_mROAS_spread_explore=m_own_spread[1],
               loss_stall=loss[0], loss_explore=loss[1])
    log('E8', **res)
    return res

def E9(R=300, T=104):
    """AR(1)-corrected confidence sequence in the smooth linear loop."""
    res = {}
    for ar, TT in [(0.0, 104), (0.6, 104), (0.8, 156)]:
        o = linear_loop_cs(T=TT, R=R, ar1=ar, seed=9)
        res[f'ar{ar}_T{TT}'] = o; log(f'E9_ar{ar}_T{TT}', **o)
    return res

def E10(R=300, T=208):
    """Robustness sweep 1: stall diagnostics across refresh interval, noise scale, and Hill slope."""
    res = {}
    for name, kw in dict(refresh1=dict(refresh=1), refresh13=dict(refresh=13), sig4=dict(sig=4.0), s2=dict(s=2.0),
                         skew=dict()).items():
        o = run_hill_loop(T=T, R=R, seed=10, **kw)
        res[name] = dict(m_own_sd=float(o['m_own'][-1].std()), m_own_mean=float(o['m_own'][-1].mean()),
                         m_true_sd=float(o['m_true_star'][-1].std()), D_perp_T=float(np.median(o['D_perp'][-1])),
                         growth=float(np.median(o['D_x'][-1] / o['D_x'][len(o['t']) // 2])))
        log('E10_' + name, **res[name])
    return res

def E11(R=300, T=208):
    """Power control for the exogenous clock: detect stall vs exploration of decreasing size."""
    res = {}
    W = 26
    for v in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        o = run_hill_loop(T=T, R=R, seed=11, explore=v)
        # window clock: exogenous dispersion in the last W weeks per week
        Dw = np.nansum((o['X'][-W:] - o['Xstar'][-W:]) ** 2, axis=0) / W
        # naive clock: raw dispersion of spend in last W weeks
        Dr = o['X'][-W:].var(0, ddof=1)
        # threshold from the pricing law: v_min for target half-width w over horizon Th weeks
        res[f'v{v}'] = dict(D_perp_week_med=float(np.median(Dw)), D_raw_week_med=float(np.median(Dr)),
                            frac_below_0p05=float((Dw < 0.05).mean()), m_true_sd=float(o['m_true_star'][-1].std()),
                            loss_star=float(o['loss_star'][-1].mean()))
        log(f'E11_v{v}', **res[f'v{v}'])
    return res

def E12(R=300, T=208):
    """Confounded pseudo-learning: coverage of naive interval and of the mixture CS for mROAS under demand chasing (linear loop version)."""
    # linear version for exactness: y = a + b x + gam d + e ; x = x0 + g(bhat-b) + phi d + xi
    rng = np.random.default_rng(12)
    a, b, x0, sig = 10.0, 2.0, 5.0, 1.0
    res = {}
    for name, (phi, gam, v) in dict(explore=(0.0, 0.0, 1.0), chase=(1.0, 1.0, 0.3), chase_strong=(1.0, 2.0, 0.3)).items():
        cov = 0; covcs = 0; bias = []; hw = []; Dx = []
        mu0 = np.array([a, b]); lam0 = np.diag([1e-2, 1e-2])
        for r in range(R):
            x = np.zeros(T); y = np.zeros(T); d = 0.0; bhat = b; ok = True
            for t in range(T):
                d = 0.8 * d + rng.normal(0, 0.6)
                xt = x0 * (1 + 0.25 * (-1) ** t) if t < 8 else x0 + 3.0 * (bhat - b) + phi * d + rng.normal(0, v)
                xt = max(xt, 0.25); x[t] = xt; y[t] = a + b * xt + gam * d + rng.normal(0, sig)
                if t >= 7:
                    Z = np.stack([np.ones(t + 1), x[:t + 1]], 1); coef, *_ = np.linalg.lstsq(Z, y[:t + 1], rcond=None); bhat = coef[1]
                    if t >= 8:
                        n = t + 1; yy = y[:n]
                        logm = nig_logm(Z.T @ Z, Z.T @ yy, yy @ yy, n, mu0, lam0, 2.0, 1.0)
                        Rstar = (n / (2 * np.pi)) * np.exp(-2 * (logm + np.log(0.05)) / n - 1)
                        resb = yy - b * x[:n]; rssb = ((resb - resb.mean()) ** 2).sum()
                        if rssb > Rstar: ok = False
            covcs += ok
            Z = np.stack([np.ones(T), x], 1); coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
            resid = y - Z @ coef; se = np.sqrt(resid @ resid / (T - 2) * np.linalg.inv(Z.T @ Z)[1, 1])
            cov += abs(coef[1] - b) <= 1.96 * se; bias.append(coef[1] - b); hw.append(1.96 * se); Dx.append(((x - x.mean()) ** 2).sum())
        res[name] = dict(cov_naive=cov / R, cov_cs_uniform=covcs / R, bias=float(np.mean(bias)), hw_naive=float(np.median(hw)),
                         D_x=float(np.median(Dx)))
        log('E12_' + name, **res[name])
    return res

def E13(R=200, T=208):
    """Seed sensitivity of the headline stall/echo numbers."""
    out = []
    for sd in range(5):
        o = run_hill_loop(T=T, R=R, seed=100 + sd)
        out.append((float(o['m_own'][-1].mean()), float(o['m_own'][-1].std()), float(o['m_true'][-1].std()), float(np.median(o['D_x'][-1]))))
    out = np.array(out)
    res = dict(m_own_mean_range=[float(out[:, 0].min()), float(out[:, 0].max())], m_own_sd_range=[float(out[:, 1].min()), float(out[:, 1].max())],
               m_true_sd_range=[float(out[:, 2].min()), float(out[:, 2].max())], D_x_range=[float(out[:, 3].min()), float(out[:, 3].max())])
    log('E13', **res)
    return res

def E14(R=300, T=208):
    """Granger/eigen-clock power sweep: mixtures of exploration and demand-chasing."""
    res = {}
    for name, kw in dict(chase_weak=dict(phi=0.5, gam=2.0), chase_plus_explore=dict(phi=1.5, gam=6.0, explore=1.0),
                         stall_impl=dict(impl_noise=0.3)).items():
        o = run_hill_loop(T=T, R=R, seed=14, **kw)
        F = granger_stat(o['X'], o['Y'], p=4, start=8); crit = stats.f.ppf(0.95, 4, T - 8 - 4 - 9)
        me = o['m_own'][-1] - o['m_true'][-1]
        res[name] = dict(granger_rej=float((F > crit).mean()), growth=float(np.median(o['D_x'][-1] / o['D_x'][len(o['t']) // 2])),
                         bias_m=float(me.mean()), rmse_m=float(np.sqrt((me ** 2).mean())), m_true_sd=float(o['m_true'][-1].std()))
        log('E14_' + name, **res[name])
    return res

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'quick'
    t0 = time.time()
    allE = dict(E1=E1, E2=E2, E3=E3, E4=E4, E5=E5, E6=E6, E7=E7, E8=E8, E9=E9, E10=E10, E11=E11, E12=E12, E13=E13, E14=E14)
    quick = ['E1', 'E2', 'E3', 'E5', 'E6', 'E7', 'E11', 'E12']
    todo = quick if which == 'quick' else (list(allE) if which == 'all' else which.split(','))
    for e in todo:
        print(f'--- {e} ---'); allE[e]()
        print(f'  elapsed {time.time() - t0:.0f}s')
    json.dump(RES, open(f'19_results_{which}.json', 'w'), default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o), indent=1)
