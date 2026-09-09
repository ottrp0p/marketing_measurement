"""Deep dive 07 - B1: refresh stability & analyst degrees of freedom.

DGP: y_t = c0 + sum_j beta_j Hill(adstock(x_jt);K_j,s_j) + seasonal + eps.
Refresh k fits window [t_k-W+1, t_k], t_k = W + k*Delta; overlap w = (W-Delta)/W.

Theory tested:
  T1 (flip-rate law): corr(th_k, th_{k+1}) = w for asymptotically linear
     estimators with exogenous X, iid eps; pairwise ranking flip rate
     F(z,w) = 2[Phi(-z) - Phi2(-z,-z;w)], z = delta/sd(diff).
  T2 (cadence law): annualized null-pair flips ~ (52/pi)*sqrt(2/(W*Delta)).
  T3 (double counting): naive posterior-as-prior on moving windows shrinks
     width by ~sqrt(Delta/W) per effective pass; coverage collapses.
  T4 (corrected scheme): increment-likelihood + power-prior discount a=w
     reproduces moving-window information (EWMA weights, ESS=W) without
     double counting; adjacent-estimate corr also ~w (same flip rate, honest
     width); drift handled by predictive test + discount drop + var floor.
"""
import json, os, pathlib, sys, time
import numpy as np
from scipy import stats, optimize

RES = pathlib.Path("/tmp/d07/results")
RES.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ helpers
def phi2(h, k, rho):
    """P(U<=h, V<=k) std bivariate normal."""
    return stats.multivariate_normal.cdf([h, k], mean=[0, 0],
                                         cov=[[1, rho], [rho, 1]])

def flip_law(z, rho):
    """P(sign flip of adjacent-refresh estimates), mean z (in sd units), corr rho."""
    return 2.0 * (stats.norm.cdf(-z) - phi2(-z, -z, rho))

# ------------------------------------------------------------------ DGP
J = 3
TRUE = dict(
    c0=2.0, gc=0.30, gs=0.20, sig=0.25,
    beta=np.array([1.20, 1.00, 0.55]),
    K=np.array([1.00, 1.20, 0.80]),
    s=np.array([1.8, 1.6, 2.0]),
    alpha=np.array([0.55, 0.35, 0.70]),
    xbar=np.array([1.0, 0.9, 0.6]),   # mean weekly spend
)

def hill(a, K, s):
    a = np.maximum(a, 1e-12)
    from scipy.special import expit
    return expit(s * (np.log(a) - np.log(K)))

def adstock(x, alpha):
    a = np.zeros_like(x)
    c = 0.0
    for t in range(len(x)):
        c = x[t] + alpha * c
        a[t] = c
    return a

def gen_spend(T, rng, rho_x=0.5, cv=0.35):
    """AR(1) lognormal spend paths, exogenous."""
    X = np.zeros((T, J))
    for j in range(J):
        e = rng.standard_normal(T)
        u = np.zeros(T)
        u[0] = e[0]
        for t in range(1, T):
            u[t] = rho_x * u[t-1] + np.sqrt(1 - rho_x**2) * e[t]
        X[:, j] = TRUE["xbar"][j] * np.exp(cv * u - 0.5 * cv**2)
    return X

def gen_y(X, rng, beta_path=None, delayed_adstock=False):
    """beta_path: (T,J) time-varying truth or None (static)."""
    T = X.shape[0]
    contrib = np.zeros((T, J))
    for j in range(J):
        if delayed_adstock:   # delayed-peak: weights on lags 0..6 peaking at lag 2
            wts = np.array([0.5, 0.8, 1.0, 0.7, 0.4, 0.2, 0.1]); wts /= wts[0] + 0  # keep scale
            a = np.convolve(X[:, j], wts)[:T]
        else:
            a = adstock(X[:, j], TRUE["alpha"][j])
        h = hill(a, TRUE["K"][j], TRUE["s"][j])
        b = TRUE["beta"][j] if beta_path is None else beta_path[:, j]
        contrib[:, j] = b * h
    tt = np.arange(T)
    seas = TRUE["gc"] * np.cos(2 * np.pi * tt / 52) + TRUE["gs"] * np.sin(2 * np.pi * tt / 52)
    y = TRUE["c0"] + seas + contrib.sum(1) + TRUE["sig"] * rng.standard_normal(T)
    return y, contrib

def true_roas(X_win, beta=None):
    """ROAS_j = mean incremental contribution / mean spend over window (static truth)."""
    b = TRUE["beta"] if beta is None else beta
    out = np.zeros(J)
    for j in range(J):
        a = adstock(X_win[:, j], TRUE["alpha"][j])
        out[j] = b[j] * hill(a, TRUE["K"][j], TRUE["s"][j]).mean() / X_win[:, j].mean()
    return out

# ------------------------------------------------------------------ fitter
# param vector (15): c0, gc, gs, log beta(3), log K(3), log s(3), logit alpha(3)
NP_ = 3 + 4 * J
def pack(c0, gc, gs, beta, K, s, alpha):
    return np.concatenate([[c0, gc, gs], np.log(beta), np.log(K), np.log(s),
                           np.log(alpha / (1 - alpha))])
def unpack(v):
    c0, gc, gs = v[:3]
    beta = np.exp(v[3:3+J]); K = np.exp(v[6:6+J]); s = np.exp(v[9:9+J])
    alpha = 1 / (1 + np.exp(-v[12:12+J]))
    return c0, gc, gs, beta, K, s, alpha

V_TRUE = pack(TRUE["c0"], TRUE["gc"], TRUE["gs"], TRUE["beta"], TRUE["K"], TRUE["s"], TRUE["alpha"])

# weak base prior on transformed params
P0_MEAN = pack(1.5, 0.0, 0.0, np.ones(J)*0.8, np.ones(J), np.ones(J)*1.7, np.ones(J)*0.5)
P0_SD = np.concatenate([[3.0, 1.0, 1.0], np.ones(J)*1.5, np.ones(J)*1.0,
                        np.ones(J)*0.5, np.ones(J)*2.0])

def model_mean(v, X, t0):
    c0, gc, gs, beta, K, s, alpha = unpack(v)
    T = X.shape[0]
    mu = np.full(T, c0)
    tt = np.arange(t0, t0 + T)
    mu += gc * np.cos(2*np.pi*tt/52) + gs * np.sin(2*np.pi*tt/52)
    for j in range(J):
        a = adstock(X[:, j], alpha[j])
        mu += beta[j] * hill(a, K[j], s[j])
    return mu

def fit_map(X, y, t0, prior_mean, prior_prec_chol, sig=TRUE["sig"], v_init=None):
    """Penalized LS: residuals [ (y-mu)/sig ; L^T (v - m) ] with prior precision = L L^T."""
    def resid(v):
        r1 = (y - model_mean(v, X, t0)) / sig
        r2 = prior_prec_chol.T @ (v - prior_mean)
        return np.concatenate([r1, r2])
    v0 = v_init if v_init is not None else prior_mean.copy()
    sol = optimize.least_squares(resid, v0, method="trf", x_scale="jac",
                                 max_nfev=400)
    Jc = sol.jac
    H = Jc.T @ Jc                       # posterior precision (Laplace)
    cov = np.linalg.inv(H + 1e-9*np.eye(NP_))
    return sol.x, cov

def base_prior():
    m = P0_MEAN.copy()
    prec = np.diag(1.0 / P0_SD**2)
    return m, np.linalg.cholesky(prec)

def roas_hat(v, X_win):
    c0, gc, gs, beta, K, s, alpha = unpack(v)
    out = np.zeros(J)
    for j in range(J):
        a = adstock(X_win[:, j], alpha[j])
        out[j] = beta[j] * hill(a, K[j], s[j]).mean() / X_win[:, j].mean()
    return out

def pair_flips(R):
    """R: (K_refresh, J) ROAS estimates. count adjacent-refresh pairwise ranking flips."""
    flips, tot = 0, 0
    for k in range(1, R.shape[0]):
        for i in range(J):
            for j in range(i+1, J):
                s1 = np.sign(R[k-1, i] - R[k-1, j]); s2 = np.sign(R[k, i] - R[k, j])
                flips += (s1 != s2); tot += 1
    return flips, tot

# =================================================================== E1
def s1(n_rep=400, seed=0):
    """Linear probe: OLS on overlapping windows. Verify rho=omega and flip law.
    y = Xb + e, X AR(1) gaussian (3 cols + intercept), iid noise."""
    rng = np.random.default_rng(seed)
    W = 104
    out = {}
    for Delta in [4, 13, 26, 52, 104]:
        omega = max(0.0, (W - Delta) / W)
        K_ref = 8
        T = W + (K_ref - 1) * Delta
        b_true = np.array([0.5, 0.30, 0.28, 0.1])  # pair (1,2) near-null gap
        est = np.zeros((n_rep, K_ref, 4))
        for r in range(n_rep):
            X = np.column_stack([np.ones(T), gen_spend(T, rng)])
            y = X @ b_true + 0.8 * rng.standard_normal(T)
            for k in range(K_ref):
                lo = k * Delta; hi = lo + W
                Xw, yw = X[lo:hi], y[lo:hi]
                est[r, k] = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        # adjacent-refresh correlation of coefficient 1
        c1 = est[:, :, 1]
        rho_emp = np.corrcoef(c1[:, :-1].ravel(), c1[:, 1:].ravel())[0, 1]
        # flip stats for near-null pair (b1-b2) and clear pair (b1-b3)
        d12 = est[:, :, 1] - est[:, :, 2]
        d13 = est[:, :, 1] - est[:, :, 3]
        res_D = {}
        for name, d, gap in [("null_pair", d12, b_true[1]-b_true[2]),
                             ("clear_pair", d13, b_true[1]-b_true[3])]:
            sd = d.std()
            z = gap / sd
            f_emp = np.mean(np.sign(d[:, :-1]) != np.sign(d[:, 1:]))
            f_th = flip_law(z, omega)
            se = np.sqrt(f_emp*(1-f_emp)/(n_rep*(K_ref-1)))
            res_D[name] = dict(z=float(z), f_emp=float(f_emp), f_th=float(f_th), se=float(se))
        out[str(Delta)] = dict(omega=omega, rho_emp=float(rho_emp), pairs=res_D)
        print(f"Delta={Delta:3d} omega={omega:.3f} rho_emp={rho_emp:.3f} "
              f"null: emp={res_D['null_pair']['f_emp']:.4f} th={res_D['null_pair']['f_th']:.4f} "
              f"clear: emp={res_D['clear_pair']['f_emp']:.4f} th={res_D['clear_pair']['f_th']:.4f}")
    json.dump(out, open(RES/"s1.json", "w"), indent=1)


# =================================================================== E2/E3
def run_refresh_ind(T, Delta, W, rng, beta_path=None, delayed=False, warm=True):
    """Independent refits (base prior) across refreshes. Returns ROAS (K,J), truths, sds."""
    X = gen_spend(T, rng)
    y, _ = gen_y(X, rng, beta_path=beta_path, delayed_adstock=delayed)
    m0, L0 = base_prior()
    K_ref = (T - W) // Delta + 1
    Rh = np.zeros((K_ref, J)); Rt = np.zeros((K_ref, J))
    v_prev = None
    for k in range(K_ref):
        lo = k * Delta; hi = lo + W
        v, cov = fit_map(X[lo:hi], y[lo:hi], lo, m0, L0,
                         v_init=(v_prev if warm else None))
        v_prev = v.copy()
        Rh[k] = roas_hat(v, X[lo:hi])
        bt = None if beta_path is None else beta_path[lo:hi].mean(0)
        Rt[k] = true_roas(X[lo:hi], beta=bt)
    return Rh, Rt

def s2(n_rep=60, seed=1):
    """Nonlinear MMM: flip rates under IND refits vs flip-law prediction."""
    rng = np.random.default_rng(seed)
    W = 104
    out = {}
    for Delta in [13, 26, 52]:
        K_ref = 8
        T = W + (K_ref - 1) * Delta
        omega = (W - Delta) / W
        allR = np.zeros((n_rep, K_ref, J))
        for r in range(n_rep):
            Rh, Rt = run_refresh_ind(T, Delta, W, rng)
            allR[r] = Rh
        pairs = {}
        for (i, j) in [(0, 1), (0, 2), (1, 2)]:
            d = allR[:, :, i] - allR[:, :, j]
            rho_emp = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0, 1]
            # center: use empirical mean gap (pseudo-true, absorbs misfit bias)
            z = d.mean() / d.std()
            f_emp = np.mean(np.sign(d[:, :-1]) != np.sign(d[:, 1:]))
            f_th = flip_law(abs(z), omega)
            f_th_emp_rho = flip_law(abs(z), max(-0.999, rho_emp))
            se = np.sqrt(f_emp*(1-f_emp)/(n_rep*(K_ref-1)))
            pairs[f"{i}{j}"] = dict(z=float(z), rho_emp=float(rho_emp),
                                    f_emp=float(f_emp), f_th=float(f_th),
                                    f_th_emp_rho=float(f_th_emp_rho), se=float(se))
            print(f"D={Delta} pair {i}{j}: z={z:+.2f} rho={rho_emp:.3f} (omega={omega:.3f}) "
                  f"f_emp={f_emp:.3f}+-{se:.3f} f_th(omega)={f_th:.3f} f_th(rho)={f_th_emp_rho:.3f}")
        out[str(Delta)] = dict(omega=omega, pairs=pairs)
    json.dump(out, open(RES/"s2.json", "w"), indent=1)

def s3(n_rep=150, seed=2):
    """Cadence law: annualized null-pair flips ~ Delta^{-1/2} (linear probe, z=0)."""
    rng = np.random.default_rng(seed)
    W = 104
    out = {}
    b_true = np.array([0.5, 0.30, 0.30, 0.1])   # exact null pair (1,2)
    for Delta in [1, 2, 4, 8, 13, 26]:
        K_ref = max(8, int(52/Delta) + 2)
        T = W + (K_ref - 1) * Delta
        cnt = 0; tot = 0
        for r in range(n_rep):
            X = np.column_stack([np.ones(T), gen_spend(T, rng)])
            y = X @ b_true + 0.8 * rng.standard_normal(T)
            est = np.zeros((K_ref, 4))
            for k in range(K_ref):
                lo = k*Delta; hi = lo + W
                est[k] = np.linalg.lstsq(X[lo:hi], y[lo:hi], rcond=None)[0]
            d = est[:, 1] - est[:, 2]
            cnt += np.sum(np.sign(d[:-1]) != np.sign(d[1:])); tot += K_ref - 1
        f = cnt / tot
        fa_emp = f * (52 / Delta)
        fa_th = (52/np.pi) * np.sqrt(2/(W*Delta))
        fa_exact = flip_law(0.0, (W-Delta)/W) * 52/Delta
        se = np.sqrt(f*(1-f)/tot) * 52/Delta
        out[str(Delta)] = dict(f=float(f), fa_emp=float(fa_emp), fa_th=float(fa_th),
                               fa_exact=float(fa_exact), se=float(se))
        print(f"Delta={Delta:3d}: per-refresh f={f:.4f}, annualized={fa_emp:.2f}+-{se:.2f} "
              f"(asymptote {fa_th:.2f}, exact {fa_exact:.2f})")
    json.dump(out, open(RES/"s3.json", "w"), indent=1)



def model_mean_burn(v, X_full, t0_full, n_eval):
    """Model mean for the last n_eval rows of X_full, adstock warmed on full history."""
    mu_all = model_mean(v, X_full, t0_full)
    return mu_all[-n_eval:]

# =================================================================== E4: churn probe
def s4(n_rep=60, n_init=8, seed=3):
    """Decomposition rho = omega*(1-h): h = idiosyncratic (multi-basin) share.
    Probe: refit same window from n_init prior-drawn inits; within-window
    dispersion of ROAS-diff = churn variance."""
    rng = np.random.default_rng(seed)
    W, Delta, K_ref = 104, 13, 8
    T = W + (K_ref - 1) * Delta
    omega = (W - Delta) / W
    m0, L0 = base_prior()
    within = {p: [] for p in ["01", "02", "12"]}
    across = {p: [] for p in ["01", "02", "12"]}
    rho_num = {p: [] for p in ["01", "02", "12"]}
    allR = np.zeros((n_rep, K_ref, J))
    for r in range(n_rep):
        X = gen_spend(T, rng); y, _ = gen_y(X, rng)
        v_prev = None
        for k in range(K_ref):
            lo = k*Delta; hi = lo + W
            v, cov = fit_map(X[lo:hi], y[lo:hi], lo, m0, L0, v_init=v_prev)
            v_prev = v.copy()
            allR[r, k] = roas_hat(v, X[lo:hi])
        # churn probe on window 0 of this replicate
        Rp = np.zeros((n_init, J))
        for i in range(n_init):
            v0 = m0 + P0_SD * 0.5 * rng.standard_normal(NP_)
            v, cov = fit_map(X[:W], y[:W], 0, m0, L0, v_init=v0)
            Rp[i] = roas_hat(v, X[:W])
        for (i, j) in [(0,1),(0,2),(1,2)]:
            within[f"{i}{j}"].append(np.var(Rp[:, i]-Rp[:, j], ddof=1))
    out = {}
    for (i, j) in [(0,1),(0,2),(1,2)]:
        p = f"{i}{j}"
        d = allR[:, :, i] - allR[:, :, j]
        v_tot = d.var()
        v_eta = np.mean(within[p])
        h = min(1.0, v_eta / v_tot)
        rho_pred = omega * (1 - h)
        rho_emp = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0, 1]
        z = d.mean()/d.std()
        f_emp = np.mean(np.sign(d[:, :-1]) != np.sign(d[:, 1:]))
        f_pred = flip_law(abs(z), rho_pred)
        out[p] = dict(h=float(h), v_eta=float(v_eta), v_tot=float(v_tot),
                      rho_pred=float(rho_pred), rho_emp=float(rho_emp),
                      f_emp=float(f_emp), f_pred=float(f_pred))
        print(f"pair {p}: h={h:.3f} rho_pred={rho_pred:.3f} rho_emp={rho_emp:.3f} "
              f"f_emp={f_emp:.3f} f_pred(z,rho_pred)={f_pred:.3f}")
    json.dump(out, open(RES/"s4.json", "w"), indent=1)

# =================================================================== E5/E6: double counting & corrected scheme
def gauss_pow_prior(m1, C1, a, m0v, C0):
    """prior = N(m1,C1)^a * N(m0,C0)^(1-a) -> Gaussian (precision-weighted)."""
    P = a*np.linalg.inv(C1) + (1-a)*np.linalg.inv(C0)
    C = np.linalg.inv(P)
    m = C @ (a*np.linalg.solve(C1, m1) + (1-a)*np.linalg.solve(C0, m0v))
    return m, C

def run_refresh_scheme(X, y, T, Delta, W, scheme, floor_frac=0.0, drift_test=False,
                       a_disc=None, sig=None):
    """schemes: IND, NPP (naive post-as-prior, full window lik),
    CPP (increment lik, discount a=omega), FRZ (freeze media params after k=0).
    Returns Rh (K,J), widths (K,J) [posterior sd of ROAS diffs proxy: sd of log beta],
    flags (K,) drift-test fires."""
    if sig is None: sig = TRUE["sig"]
    m0, L0 = base_prior(); C0 = np.diag(P0_SD**2)
    omega = (W - Delta)/W
    a = omega if a_disc is None else a_disc
    K_ref = (T - W)//Delta + 1
    Rh = np.zeros((K_ref, J)); wid = np.zeros((K_ref, NP_)); flags = np.zeros(K_ref)
    v_prev, cov_prev = None, None
    for k in range(K_ref):
        lo = k*Delta; hi = lo + W
        if scheme == "IND" or k == 0:
            v, cov = fit_map(X[lo:hi], y[lo:hi], lo, m0, L0, sig=sig, v_init=v_prev)
        elif scheme == "NPP":
            Lp = np.linalg.cholesky(np.linalg.inv(cov_prev))
            v, cov = fit_map(X[lo:hi], y[lo:hi], lo, v_prev, Lp, sig=sig, v_init=v_prev)
        elif scheme == "CPP":
            aa = a
            if drift_test:
                burn0 = min(hi - Delta, 52)
                mu_pred = model_mean_burn(v_prev, X[hi-Delta-burn0:hi], hi-Delta-burn0, Delta)
                r = (y[hi-Delta:hi] - mu_pred)/sig
                # predictive chi2 with param-uncertainty inflation (approx: ignore)
                stat = np.sum(r**2)
                if stat > stats.chi2.ppf(0.99, Delta):
                    aa = 0.2; flags[k] = 1
            mp, Cp = gauss_pow_prior(v_prev, cov_prev, aa, m0, C0)
            if floor_frac > 0:
                dfloor = (floor_frac*P0_SD)**2
                Cp = Cp + np.diag(np.maximum(0, dfloor - np.diag(Cp)))
            Lp = np.linalg.cholesky(np.linalg.inv(Cp))
            burn = min(hi - Delta, 52)
            Xb = X[hi-Delta-burn:hi]
            def resid_inc(vv):
                r1 = (y[hi-Delta:hi] - model_mean_burn(vv, Xb, hi-Delta-burn, Delta))/sig
                r2 = Lp.T @ (vv - mp)
                return np.concatenate([r1, r2])
            sol = optimize.least_squares(resid_inc, v_prev, method="trf",
                                         x_scale="jac", max_nfev=400)
            Jc = sol.jac; cov = np.linalg.inv(Jc.T@Jc + 1e-9*np.eye(NP_))
            v = sol.x
        elif scheme == "FRZ":
            # refit only baseline params (c0,gc,gs): fix media at k=0 values
            vfix = v_prev.copy()
            def resid(b3):
                vv = vfix.copy(); vv[:3] = b3
                return (y[lo:hi] - model_mean(vv, X[lo:hi], lo))/sig
            sol = optimize.least_squares(resid, vfix[:3])
            v = vfix.copy(); v[:3] = sol.x; cov = cov_prev
        v_prev, cov_prev = v.copy(), cov.copy()
        Rh[k] = roas_hat(v, X[lo:hi]); wid[k] = np.sqrt(np.diag(cov))
    return Rh, wid, flags

def s5(n_rep=80, seed=4):
    """Double counting: NPP width collapse & coverage vs IND and CPP. Static truth."""
    rng = np.random.default_rng(seed)
    W, Delta, K_ref = 104, 13, 9
    T = W + (K_ref-1)*Delta
    res = {}
    for scheme in ["IND", "NPP", "CPP"]:
        wds, covr, Rall, Rtr = [], [], [], []
        for r in range(n_rep):
            rng_r = np.random.default_rng(seed*1000+r)
            X = gen_spend(T, rng_r); y, _ = gen_y(X, rng_r)
            Rh, wid, _ = run_refresh_scheme(X, y, T, Delta, W, scheme)
            Rall.append(Rh); wds.append(wid)
            Rt = np.array([true_roas(X[k*Delta:k*Delta+W]) for k in range(K_ref)])
            Rtr.append(Rt)
        Rall = np.array(Rall); wds = np.array(wds); Rtr = np.array(Rtr)
        # use log-beta width as canonical param width; coverage of log beta_0
        lb_hat = np.array([[v for v in row] for row in Rall])  # ROAS-space
        width_k = wds[:, :, 3].mean(0)  # sd of log beta_0 by refresh
        # coverage: |thetahat - true|<1.645 sd for log beta_0 needs stored v; approx via ROAS
        err = Rall - Rtr
        rmse_k = np.sqrt((err**2).mean((0, 2)))
        d = Rall[:, :, 0] - Rall[:, :, 1]
        f_emp = np.mean(np.sign(d[:, :-1]) != np.sign(d[:, 1:]))
        rho = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0, 1]
        res[scheme] = dict(width_by_refresh=width_k.tolist(),
                           rmse_by_refresh=rmse_k.tolist(),
                           flip01=float(f_emp), rho01=float(rho))
        print(f"{scheme}: width(logb0) k0={width_k[0]:.3f} k4={width_k[4]:.3f} "
              f"k8={width_k[8]:.3f} | rmse k0={rmse_k[0]:.3f} k8={rmse_k[8]:.3f} "
              f"| flip01={f_emp:.3f} rho01={rho:.3f}")
    json.dump(res, open(RES/"s5.json", "w"), indent=1)


# ============================================================ E5b: linear-Gaussian exact
def s6(n_rep=400, seed=5):
    """Exact conjugate updates, linear model: verifies the accounting theory
    (CPP a=omega == moving window; NPP collapse 1/sqrt(k+1)) without optimizer noise."""
    rng = np.random.default_rng(seed)
    W, Delta, K_ref = 104, 13, 9
    T = W + (K_ref-1)*Delta
    omega = (W-Delta)/W
    p = 4
    b_true = np.array([0.5, 0.30, 0.28, 0.1])
    sig = 0.8
    P0 = np.eye(p)/9.0            # base prior N(0, 9 I)
    res = {}
    for scheme in ["IND", "NPP", "CPP"]:
        widths = np.zeros((n_rep, K_ref)); errs = np.zeros((n_rep, K_ref, p))
        cover = np.zeros((n_rep, K_ref)); flips = []
        for r in range(n_rep):
            X = np.column_stack([np.ones(T), gen_spend(T, rng)])
            y = X @ b_true + sig*rng.standard_normal(T)
            m_prev, P_prev = None, None
            est = np.zeros((K_ref, p))
            for k in range(K_ref):
                lo, hi = k*Delta, k*Delta+W
                if scheme == "IND" or k == 0:
                    Xw = X[lo:hi]; Pl = Xw.T@Xw/sig**2
                    P = Pl + P0; m = np.linalg.solve(P, Xw.T@y[lo:hi]/sig**2)
                elif scheme == "NPP":
                    Xw = X[lo:hi]; Pl = Xw.T@Xw/sig**2
                    P = Pl + P_prev
                    m = np.linalg.solve(P, Xw.T@y[lo:hi]/sig**2 + P_prev@m_prev)
                elif scheme == "CPP":
                    Xn = X[hi-Delta:hi]; Pl = Xn.T@Xn/sig**2
                    Pp = omega*P_prev + (1-omega)*P0
                    P = Pl + Pp
                    m = np.linalg.solve(P, Xn.T@y[hi-Delta:hi]/sig**2 + omega*P_prev@m_prev)
                m_prev, P_prev = m, P
                C = np.linalg.inv(P)
                est[k] = m
                widths[r, k] = np.sqrt(C[1, 1])
                errs[r, k] = m - b_true
                cover[r, k] = abs(m[1]-b_true[1]) < 1.645*np.sqrt(C[1,1])
            d = est[:, 1] - est[:, 2]
            flips.append(np.sign(d[:-1]) != np.sign(d[1:]))
        flips = np.array(flips)
        res[scheme] = dict(width_k=widths.mean(0).tolist(),
                           rmse_k=np.sqrt((errs[:, :, 1]**2).mean(0)).tolist(),
                           cover_k=cover.mean(0).tolist(),
                           flip=float(flips.mean()))
        w = widths.mean(0); rm = np.sqrt((errs[:, :, 1]**2).mean(0)); cv = cover.mean(0)
        print(f"{scheme}: width k0={w[0]:.4f} k8={w[8]:.4f} | rmse k0={rm[0]:.4f} "
              f"k8={rm[8]:.4f} | cover90 k8={cv[8]:.3f} | flip={flips.mean():.3f}")
    json.dump(res, open(RES/"s6.json", "w"), indent=1)

# ============================================================ E7: churn-inflated CPP
def s7(n_rep=80, seed=6, infl_list=(0.0, 0.15, 0.3)):
    """CPP with churn/process-noise inflation: add (infl*P0_SD)^2 to propagated prior var.
    Calibration check: coverage of true ROAS by +-1.645 sd(ROAS) via delta method."""
    rng = np.random.default_rng(seed)
    W, Delta, K_ref = 104, 13, 9
    T = W + (K_ref-1)*Delta
    res = {}
    for infl in infl_list:
        wds, rmses, covs, fl = [], [], [], []
        for r in range(n_rep):
            rng_r = np.random.default_rng(seed*1000+r)
            X = gen_spend(T, rng_r); y, _ = gen_y(X, rng_r)
            Rh, wid, _ = run_refresh_scheme(X, y, T, Delta, W, "CPP", floor_frac=infl)
            Rt = np.array([true_roas(X[k*Delta:k*Delta+W]) for k in range(K_ref)])
            # delta-method-ish: use log beta width as ROAS log-sd proxy
            lw = wid[:, 3:6]
            covs.append(np.abs(np.log(np.maximum(Rh,1e-6)) - np.log(Rt)) < 1.645*lw)
            wds.append(wid[:, 3].mean()); rmses.append(np.sqrt(((Rh-Rt)**2).mean()))
            d = Rh[:, 0]-Rh[:, 1]; fl.append(np.sign(d[:-1]) != np.sign(d[1:]))
        cov90 = np.mean(covs); wm = np.mean(wds); rm = np.mean(rmses); f = np.mean(fl)
        res[str(infl)] = dict(cover90=float(cov90), width=float(wm), rmse=float(rm), flip=float(f))
        print(f"infl={infl}: cover90={cov90:.3f} width={wm:.3f} rmse={rm:.3f} flip01={f:.3f}")
    json.dump(res, open(RES/"s7.json", "w"), indent=1)


# ============================================================ E8: drift/break duel
def make_beta_path(T, regime, rng, q=0.01):
    bp = np.tile(TRUE["beta"], (T, 1)).astype(float)
    if regime == "drift":
        for j in range(J):
            lw = np.cumsum(q*rng.standard_normal(T))
            bp[:, j] = TRUE["beta"][j]*np.exp(lw - 0)
    elif regime == "break":
        tb = 104 + 4*13
        bp[tb:, 0] = TRUE["beta"][0]*0.45
    return bp

def s8(n_rep=60, seed=7, regimes=("static","drift","break"),
       schemes=(("IND",{}),("NPP",{}),("FRZ",{}),("CPP",dict(floor_frac=0.3)),
                ("CPPD",dict(floor_frac=0.3, drift_test=True))), tag="s8"):
    """Regime duel. Metrics: flip rate, RMSE vs CURRENT truth (last 4 refreshes),
    coverage, trap rate (final |log err| > 0.69), break end-error."""
    W, Delta, K_ref = 104, 13, 12
    T = W + (K_ref-1)*Delta
    res = {}
    for regime in regimes:
        for scheme, kw in schemes:
            fl, rmse_late, traps, covl, recov = [], [], [], [], []
            for r in range(n_rep):
                rng_r = np.random.default_rng(seed*1000+r)
                X = gen_spend(T, rng_r)
                bp = make_beta_path(T, regime, rng_r)
                y, _ = gen_y(X, rng_r, beta_path=bp)
                sch = "CPP" if scheme == "CPPD" else scheme
                Rh, wid, flags = run_refresh_scheme(X, y, T, Delta, W, sch, **kw)
                Rt = np.array([true_roas(X[k*Delta:k*Delta+W],
                               beta=bp[k*Delta:k*Delta+W].mean(0)) for k in range(K_ref)])
                d = Rh[:, 0]-Rh[:, 1]
                fl.append(np.sign(d[:-1]) != np.sign(d[1:]))
                err = np.log(np.maximum(Rh, 1e-6)) - np.log(Rt)
                rmse_late.append(np.sqrt((err[-4:]**2).mean()))
                covl.append(np.abs(err[-4:, 0]) < 1.645*wid[-4:, 3])
                traps.append(np.abs(err[-1, 0]) > 0.69)
                if regime == "break":
                    recov.append(np.abs(err[-1, 0]))
            key = f"{regime}/{scheme}"
            res[key] = dict(flip=float(np.mean(fl)), rmse_late=float(np.mean(rmse_late)),
                            rmse_se=float(np.std(rmse_late)/np.sqrt(n_rep)),
                            cover=float(np.mean(covl)), trap=float(np.mean(traps)),
                            recov=(float(np.mean(recov)) if recov else None))
            print(f"{key:14s} flip={np.mean(fl):.3f} rmse_late={np.mean(rmse_late):.3f}"
                  f"+-{np.std(rmse_late)/np.sqrt(n_rep):.3f} cover={np.mean(covl):.2f} "
                  f"trap={np.mean(traps):.2f}" + (f" |err_end|={np.mean(recov):.3f}" if recov else ""))
    json.dump(res, open(RES/f"{tag}.json", "w"), indent=1)


# ============================================================ E9: analyst df / selection churn
def s9(n_rep=50, R_sel=8, seed=8):
    """Robyn-like per-refresh model selection vs pre-registered fit vs front-average.
    Selection criterion: NRMSE + lam*RSSD(effect share vs spend share)."""
    rng = np.random.default_rng(seed)
    W, Delta, K_ref = 104, 13, 8
    T = W + (K_ref-1)*Delta
    omega = (W-Delta)/W
    m0, L0 = base_prior()
    lam = 0.5
    modes = ["prereg", "select", "average"]
    allR = {m: np.zeros((n_rep, K_ref, J)) for m in modes}
    for r in range(n_rep):
        rng_r = np.random.default_rng(seed*1000+r)
        X = gen_spend(T, rng_r); y, _ = gen_y(X, rng_r)
        v_prev = None
        for k in range(K_ref):
            lo, hi = k*Delta, k*Delta+W
            Xw, yw = X[lo:hi], y[lo:hi]
            # candidates
            cands, crits = [], []
            for i in range(R_sel):
                v0 = m0 + P0_SD*0.7*rng_r.standard_normal(NP_)
                v, cov = fit_map(Xw, yw, lo, m0, L0, v_init=v0)
                Rj = roas_hat(v, Xw)
                nrmse = np.sqrt(np.mean((yw - model_mean(v, Xw, lo))**2))/yw.std()
                _,_,_,beta,K,s,al = unpack(v)
                eff = np.array([beta[j]*hill(adstock(Xw[:,j],al[j]),K[j],s[j]).mean() for j in range(J)])
                eff = eff/max(eff.sum(),1e-9); sp = Xw.mean(0)/Xw.mean(0).sum()
                rssd = np.sqrt(((eff-sp)**2).sum())
                cands.append(Rj); crits.append(nrmse + lam*rssd)
            cands = np.array(cands)
            allR["select"][r, k] = cands[int(np.argmin(crits))]
            allR["average"][r, k] = cands.mean(0)
            v, cov = fit_map(Xw, yw, lo, m0, L0, v_init=v_prev)
            v_prev = v.copy()
            allR["prereg"][r, k] = roas_hat(v, Xw)
    out = {}
    for m in modes:
        d = allR[m][:, :, 0]-allR[m][:, :, 1]
        rho = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0, 1]
        f = np.mean(np.sign(d[:, :-1]) != np.sign(d[:, 1:]))
        se = np.sqrt(f*(1-f)/(n_rep*(K_ref-1)))
        out[m] = dict(flip=float(f), se=float(se), rho=float(rho), sd=float(d.std()))
        print(f"{m:8s}: flip01={f:.3f}+-{se:.3f} rho={rho:.3f} sd(d)={d.std():.3f}")
    json.dump(out, open(RES/"s9.json", "w"), indent=1)

# ============================================================ E10: hysteresis reporting layer
def s10(seed=9, n_rep=200):
    """Pre-registered hysteresis on reported ranking: flip reported only when the
    posterior z of the NEW ordering exceeds h. Uses IND estimates (worst case).
    Metrics vs h: reported flip rate, wrongness (reported order != true order)."""
    rng = np.random.default_rng(seed)
    W, Delta, K_ref = 104, 13, 12
    T = W + (K_ref-1)*Delta
    out = {}
    # simulate normal d-path directly with rho=omega*(1-h_churn) measured earlier (~0.55)
    # but do it on real nonlinear estimates for honesty:
    Ds, SDs, Ts = [], [], []
    for r in range(n_rep//4):
        rng_r = np.random.default_rng(seed*1000+r)
        X = gen_spend(T, rng_r); y, _ = gen_y(X, rng_r)
        Rh, Rt = run_refresh_ind(T, Delta, W, rng_r) if False else (None, None)
        # reuse scheme runner for speed
        Rh, wid, _ = run_refresh_scheme(X, y, T, Delta, W, "IND")
        Rtr = np.array([true_roas(X[k*Delta:k*Delta+W]) for k in range(K_ref)])
        Ds.append(Rh[:, 0]-Rh[:, 1]); SDs.append(np.sqrt(wid[:, 3]**2+wid[:, 4]**2))
        Ts.append(Rtr[:, 0]-Rtr[:, 1])
    Ds = np.array(Ds); SDs = np.array(SDs)*np.abs(Ds).mean()  # crude scale: log->abs
    Ts = np.array(Ts)
    sd_emp = Ds.std()
    for h in [0.0, 0.5, 1.0, 1.5]:
        rep_flips, wrong = [], []
        for r in range(Ds.shape[0]):
            rep = np.sign(Ds[r, 0]); fl = 0; wr = 0
            for k in range(1, K_ref):
                cand = np.sign(Ds[r, k])
                if cand != rep and abs(Ds[r, k]) > h*sd_emp:
                    rep = cand; fl += 1
                wr += (rep != np.sign(Ts[r, k]))
            rep_flips.append(fl/(K_ref-1)); wrong.append(wr/(K_ref-1))
        out[str(h)] = dict(rep_flip=float(np.mean(rep_flips)), wrong=float(np.mean(wrong)))
        print(f"h={h}: reported flips/refresh={np.mean(rep_flips):.3f} wrong-order rate={np.mean(wrong):.3f}")
    json.dump(out, open(RES/"s10.json", "w"), indent=1)

# ============================================================ E11: misspec + power control + cold start
def s11(seed=10):
    rng = np.random.default_rng(seed)
    W, Delta, K_ref = 104, 13, 8
    T = W + (K_ref-1)*Delta
    omega = (W-Delta)/W
    out = {}
    # (a) misspecified adstock (truth delayed-peak, model geometric)
    n_rep = 50
    allR = np.zeros((n_rep, K_ref, J))
    for r in range(n_rep):
        rng_r = np.random.default_rng(seed*1000+r)
        X = gen_spend(T, rng_r); y, _ = gen_y(X, rng_r, delayed_adstock=True)
        Rh, wid, _ = run_refresh_scheme(X, y, T, Delta, W, "IND")
        allR[r] = Rh
    d = allR[:, :, 0]-allR[:, :, 1]
    rho = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0, 1]
    z = d.mean()/d.std()
    f = np.mean(np.sign(d[:, :-1]) != np.sign(d[:, 1:]))
    fp = flip_law(abs(z), rho)
    out["misspec"] = dict(rho=float(rho), z=float(z), f_emp=float(f), f_pred=float(fp))
    print(f"misspec: rho={rho:.3f} z={z:.2f} f_emp={f:.3f} f_pred(rho_emp)={fp:.3f}")
    # (a2) drift-test false-fire under misspec (static truth, delayed adstock)
    fires = []
    for r in range(25):
        rng_r = np.random.default_rng(seed*2000+r)
        X = gen_spend(T, rng_r); y, _ = gen_y(X, rng_r, delayed_adstock=True)
        _,_,flags = run_refresh_scheme(X, y, T, Delta, W, "CPP", floor_frac=0.3, drift_test=True)
        fires.append(flags[1:].mean())
    out["misspec_fire"] = float(np.mean(fires))
    print(f"drift-test fire rate under misspec+static: {np.mean(fires):.3f}")
    fires = []
    for r in range(25):
        rng_r = np.random.default_rng(seed*3000+r)
        X = gen_spend(T, rng_r); y, _ = gen_y(X, rng_r)
        _,_,flags = run_refresh_scheme(X, y, T, Delta, W, "CPP", floor_frac=0.3, drift_test=True)
        fires.append(flags[1:].mean())
    out["wellspec_fire"] = float(np.mean(fires))
    print(f"drift-test fire rate well-spec+static: {np.mean(fires):.3f}")
    # (b) power control: strong design (cv=0.8, sig=0.08) -> big z, near-zero flips
    n_rep = 40
    allR = np.zeros((n_rep, K_ref, J))
    for r in range(n_rep):
        rng_r = np.random.default_rng(seed*4000+r)
        X = gen_spend(T, rng_r, cv=0.8)
        contrib = np.zeros((T, J))
        for j in range(J):
            a = adstock(X[:, j], TRUE["alpha"][j])
            contrib[:, j] = TRUE["beta"][j]*hill(a, TRUE["K"][j], TRUE["s"][j])
        tt = np.arange(T)
        y = TRUE["c0"] + TRUE["gc"]*np.cos(2*np.pi*tt/52) + TRUE["gs"]*np.sin(2*np.pi*tt/52) \
            + contrib.sum(1) + 0.08*rng_r.standard_normal(T)
        m0, L0 = base_prior()
        v_prev = None
        for k in range(K_ref):
            lo, hi = k*Delta, k*Delta+W
            v, cov = fit_map(X[lo:hi], y[lo:hi], lo, m0, L0, sig=0.08, v_init=v_prev)
            v_prev = v.copy()
            allR[r, k] = roas_hat(v, X[lo:hi])
    res_p = {}
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        d = allR[:, :, i]-allR[:, :, j]
        z = d.mean()/d.std()
        f = np.mean(np.sign(d[:, :-1]) != np.sign(d[:, 1:]))
        res_p[f"{i}{j}"] = dict(z=float(z), f=float(f))
        print(f"power control pair {i}{j}: z={z:+.1f} flip={f:.4f}")
    out["power"] = res_p
    # (c) cold-start rho (is warm-start inflating rho?)
    n_rep = 40
    allR = np.zeros((n_rep, K_ref, J))
    m0, L0 = base_prior()
    for r in range(n_rep):
        rng_r = np.random.default_rng(seed*5000+r)
        X = gen_spend(T, rng_r); y, _ = gen_y(X, rng_r)
        for k in range(K_ref):
            lo, hi = k*Delta, k*Delta+W
            v, cov = fit_map(X[lo:hi], y[lo:hi], lo, m0, L0, v_init=None)
            allR[r, k] = roas_hat(v, X[lo:hi])
    d = allR[:, :, 0]-allR[:, :, 1]
    rho_c = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0, 1]
    f_c = np.mean(np.sign(d[:, :-1]) != np.sign(d[:, 1:]))
    out["coldstart"] = dict(rho=float(rho_c), f=float(f_c))
    print(f"cold-start: rho01={rho_c:.3f} flip01={f_c:.3f}")
    json.dump(out, open(RES/"s11.json", "w"), indent=1)

if __name__ == "__main__":
    t0 = time.time()
    for stage in sys.argv[1:]:
        globals()[stage]()
    print(f"[{time.time()-t0:.1f}s]")


# ======================================================================
# ATTACK/REFINE MODULE: attack1 (rounds 2-3; imports the base module above when
# run standalone as originally executed: from d07 import *)
# ======================================================================
import numpy as np, json
# from d07 import *  (merged: base module is above)

def power_ctrl_multistart(seed=20, n_rep=30, n_start=5, cv=0.8, sig=0.08):
    """Power control with best-of-n_start fits (crush optimizer churn)."""
    W, Delta, K_ref = 104, 13, 8
    T = W + (K_ref-1)*Delta
    m0, L0 = base_prior()
    allR = np.zeros((n_rep, K_ref, J))
    for r in range(n_rep):
        rng_r = np.random.default_rng(seed*1000+r)
        X = gen_spend(T, rng_r, cv=cv)
        contrib = np.zeros((T, J))
        for j in range(J):
            a = adstock(X[:, j], TRUE["alpha"][j])
            contrib[:, j] = TRUE["beta"][j]*hill(a, TRUE["K"][j], TRUE["s"][j])
        tt = np.arange(T)
        y = TRUE["c0"] + TRUE["gc"]*np.cos(2*np.pi*tt/52) + TRUE["gs"]*np.sin(2*np.pi*tt/52) \
            + contrib.sum(1) + sig*rng_r.standard_normal(T)
        for k in range(K_ref):
            lo, hi = k*Delta, k*Delta+W
            best, bcost = None, np.inf
            for i in range(n_start):
                v0 = m0 + P0_SD*0.5*rng_r.standard_normal(NP_) if i else m0
                def resid(v):
                    r1 = (y[lo:hi]-model_mean(v, X[lo:hi], lo))/sig
                    r2 = L0.T @ (v-m0)
                    return np.concatenate([r1, r2])
                sol = optimize.least_squares(resid, v0, method="trf", x_scale="jac", max_nfev=600)
                if sol.cost < bcost: bcost, best = sol.cost, sol.x
            allR[r, k] = roas_hat(best, X[lo:hi])
    out = {}
    for (i, j) in [(0,1),(0,2),(1,2)]:
        d = allR[:, :, i]-allR[:, :, j]
        rho = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0,1]
        z = d.mean()/d.std()
        f = np.mean(np.sign(d[:, :-1]) != np.sign(d[:, 1:]))
        out[f"{i}{j}"] = dict(z=float(z), rho=float(rho), f=float(f), sd=float(d.std()))
        print(f"multistart power pair {i}{j}: z={z:+.2f} rho={rho:.3f} flip={f:.4f} sd={d.std():.4f}")
    json.dump(out, open(RES/"a1_power.json","w"), indent=1)

def fixed_functional(seed=21, n_rep=40):
    """Does evaluating ROAS at a FIXED reference spend path restore rho -> omega?"""
    W, Delta, K_ref = 104, 13, 8
    T = W + (K_ref-1)*Delta
    m0, L0 = base_prior()
    rngref = np.random.default_rng(999)
    Xref = gen_spend(W, rngref)
    allR = np.zeros((n_rep, K_ref, J)); allF = np.zeros((n_rep, K_ref, J))
    for r in range(n_rep):
        rng_r = np.random.default_rng(seed*1000+r)
        X = gen_spend(T, rng_r); y, _ = gen_y(X, rng_r)
        v_prev = None
        for k in range(K_ref):
            lo, hi = k*Delta, k*Delta+W
            v, cov = fit_map(X[lo:hi], y[lo:hi], lo, m0, L0, v_init=v_prev)
            v_prev = v.copy()
            allR[r, k] = roas_hat(v, X[lo:hi])
            allF[r, k] = roas_hat(v, Xref)
    for name, A in [("window-ROAS", allR), ("fixed-ROAS", allF)]:
        d = A[:, :, 0]-A[:, :, 1]
        rho = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0,1]
        f = np.mean(np.sign(d[:, :-1]) != np.sign(d[:, 1:]))
        print(f"{name}: rho01={rho:.3f} flip01={f:.3f}")

def nofloor_trap(seed=22, n_rep=40):
    """CPP floor=0 vs floor=0.3, static truth: lock-in trap rate + width."""
    W, Delta, K_ref = 104, 13, 12
    T = W + (K_ref-1)*Delta
    for ff in [0.0, 0.3]:
        traps, wid_end, covl = [], [], []
        for r in range(n_rep):
            rng_r = np.random.default_rng(seed*1000+r)
            X = gen_spend(T, rng_r); y, _ = gen_y(X, rng_r)
            Rh, wid, _ = run_refresh_scheme(X, y, T, Delta, W, "CPP", floor_frac=ff)
            Rt = np.array([true_roas(X[k*Delta:k*Delta+W]) for k in range(K_ref)])
            err = np.log(np.maximum(Rh,1e-6))-np.log(Rt)
            traps.append(abs(err[-1,0])>0.69); wid_end.append(wid[-1,3])
            covl.append(np.abs(err[-4:,0])<1.645*wid[-4:,3])
        print(f"floor={ff}: trap={np.mean(traps):.3f} width_end={np.mean(wid_end):.3f} cover={np.mean(covl):.3f}")

if False:  # standalone entry of merged attack module (disabled)
    import sys
    globals()[sys.argv[1]](**eval(sys.argv[2]) if len(sys.argv)>2 else {})


# ======================================================================
# ATTACK/REFINE MODULE: attack2 (rounds 2-3; imports the base module above when
# run standalone as originally executed: from d07 import *)
# ======================================================================
import numpy as np, json
# from d07 import *  (merged: base module is above)

def power2(seed=30, n_rep=40, cv=0.5, sig=0.08):
    """Proper power control: fixed-reference ROAS, stable hill, warm starts.
    Also linear big-gap control."""
    W, Delta, K_ref = 104, 13, 8
    T = W + (K_ref-1)*Delta
    m0, L0 = base_prior()
    rngref = np.random.default_rng(999); Xref = gen_spend(W, rngref)
    Rtrue = true_roas(Xref)
    print("true fixed-ref ROAS:", np.round(Rtrue, 3))
    allF = np.zeros((n_rep, K_ref, J))
    for r in range(n_rep):
        rng_r = np.random.default_rng(seed*1000+r)
        X = gen_spend(T, rng_r, cv=cv)
        contrib = np.zeros((T, J))
        for j in range(J):
            a = adstock(X[:, j], TRUE["alpha"][j])
            contrib[:, j] = TRUE["beta"][j]*hill(a, TRUE["K"][j], TRUE["s"][j])
        tt = np.arange(T)
        y = TRUE["c0"] + TRUE["gc"]*np.cos(2*np.pi*tt/52) + TRUE["gs"]*np.sin(2*np.pi*tt/52) \
            + contrib.sum(1) + sig*rng_r.standard_normal(T)
        v_prev = None
        for k in range(K_ref):
            lo, hi = k*Delta, k*Delta+W
            v, cov = fit_map(X[lo:hi], y[lo:hi], lo, m0, L0, sig=sig, v_init=v_prev)
            v_prev = v.copy()
            allF[r, k] = roas_hat(v, Xref)
    out = {}
    for (i, j) in [(0,1),(0,2),(1,2)]:
        d = allF[:, :, i]-allF[:, :, j]
        gap = Rtrue[i]-Rtrue[j]
        z = d.mean()/d.std()
        rho = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0,1]
        f = np.mean(np.sign(d[:, :-1]) != np.sign(d[:, 1:]))
        out[f"{i}{j}"] = dict(z=float(z), rho=float(rho), f=float(f), sd=float(d.std()), gap=float(gap))
        print(f"pair {i}{j}: true_gap={gap:+.3f} z={z:+.2f} rho={rho:.3f} flip={f:.4f} sd={d.std():.4f}")
    # linear big-gap control
    rng = np.random.default_rng(seed+1)
    b_true = np.array([0.5, 0.45, 0.15, 0.1]); n2 = 200
    cnt, tot = 0, 0
    for r in range(n2):
        X = np.column_stack([np.ones(T), gen_spend(T, rng)])
        y = X @ b_true + 0.25*rng.standard_normal(T)
        est = np.zeros((K_ref, 4))
        for k in range(K_ref):
            lo, hi = k*Delta, k*Delta+W
            est[k] = np.linalg.lstsq(X[lo:hi], y[lo:hi], rcond=None)[0]
        d = est[:, 1]-est[:, 2]
        cnt += np.sum(np.sign(d[:-1]) != np.sign(d[1:])); tot += K_ref-1
    dsd = None
    print(f"linear big-gap control: flip={cnt/tot:.5f} ({cnt}/{tot})")
    out["linear_biggap"] = dict(flip=cnt/tot)
    json.dump(out, open(RES/"a2_power.json","w"), indent=1)

if False:  # standalone entry of merged attack module (disabled)
    import sys; globals()[sys.argv[1]]()


# ======================================================================
# ATTACK/REFINE MODULE: attack3 (rounds 2-3; imports the base module above when
# run standalone as originally executed: from d07 import *)
# ======================================================================
import numpy as np, json
# from d07 import *  (merged: base module is above)

def gen_spend_pulsed(T, rng, amp=0.8):
    """Dive-01-style block pulses: per-channel square waves, L ~ 3/(1-alpha), staggered."""
    X = np.zeros((T, J))
    per = [8, 5, 12]   # approx 3/(1-alpha) rounded, distinct
    for j in range(J):
        ph = rng.integers(0, per[j])
        sq = (((np.arange(T)+ph)//per[j]) % 2)*2 - 1
        X[:, j] = TRUE["xbar"][j]*(1 + amp*sq)
        X[:, j] *= np.exp(0.05*rng.standard_normal(T))
    return X

def power3(seed=31, n_rep=40, sig=0.25):
    """Pulsed identifying design at FULL noise: prediction = flips collapse."""
    W, Delta, K_ref = 104, 13, 8
    T = W + (K_ref-1)*Delta
    m0, L0 = base_prior()
    rngref = np.random.default_rng(999); Xref = gen_spend(W, rngref)
    Rtrue = true_roas(Xref)
    allF = np.zeros((n_rep, K_ref, J))
    for r in range(n_rep):
        rng_r = np.random.default_rng(seed*1000+r)
        X = gen_spend_pulsed(T, rng_r)
        y, _ = gen_y(X, rng_r)
        v_prev = None
        for k in range(K_ref):
            lo, hi = k*Delta, k*Delta+W
            v, cov = fit_map(X[lo:hi], y[lo:hi], lo, m0, L0, sig=sig, v_init=v_prev)
            v_prev = v.copy()
            allF[r, k] = roas_hat(v, Xref)
    out = {}
    for (i, j) in [(0,1),(0,2),(1,2)]:
        d = allF[:, :, i]-allF[:, :, j]
        gap = Rtrue[i]-Rtrue[j]
        z = d.mean()/d.std(); rho = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0,1]
        f = np.mean(np.sign(d[:, :-1]) != np.sign(d[:, 1:]))
        se = np.sqrt(max(f*(1-f),1e-9)/(n_rep*(K_ref-1)))
        out[f"{i}{j}"] = dict(z=float(z), rho=float(rho), f=float(f), se=float(se), sd=float(d.std()), gap=float(gap))
        print(f"pulsed pair {i}{j}: true_gap={gap:+.3f} z={z:+.2f} rho={rho:.3f} flip={f:.4f}+-{se:.4f} sd={d.std():.4f}")
    json.dump(out, open(RES/"a3_pulsed.json","w"), indent=1)

if False:  # standalone entry of merged attack module (disabled)
    import sys; globals()[sys.argv[1]]()


# ======================================================================
# ATTACK/REFINE MODULE: attack4 (rounds 2-3; imports the base module above when
# run standalone as originally executed: from d07 import *)
# ======================================================================
import numpy as np, json
# from d07 import *  (merged: base module is above)

def anchored(seed=40, n_rep=50):
    """Anchored refresh: fix linearization point v_anchor = MAP of window 0;
    each refresh does ONE Gauss-Newton step from v_anchor on its window.
    Prediction: estimator linear in data => rho -> omega, flips -> flip_law(z, omega)."""
    W, Delta, K_ref = 104, 13, 8
    T = W + (K_ref-1)*Delta
    omega = (W-Delta)/W
    m0, L0 = base_prior()
    allA = np.zeros((n_rep, K_ref, J)); allM = np.zeros((n_rep, K_ref, J))
    for r in range(n_rep):
        rng_r = np.random.default_rng(seed*1000+r)
        X = gen_spend(T, rng_r); y, _ = gen_y(X, rng_r)
        va, _ = fit_map(X[:W], y[:W], 0, m0, L0)     # anchor
        v_prev = va.copy()
        for k in range(K_ref):
            lo, hi = k*Delta, k*Delta+W
            # one GN step from anchor
            eps = 1e-6
            mu0 = model_mean(va, X[lo:hi], lo)
            Jm = np.zeros((W, NP_))
            for p in range(NP_):
                vp = va.copy(); vp[p] += eps*max(abs(va[p]), 1.0)
                Jm[:, p] = (model_mean(vp, X[lo:hi], lo) - mu0)/(eps*max(abs(va[p]), 1.0))
            P0i = np.diag(1.0/P0_SD**2)
            A = Jm.T@Jm/TRUE["sig"]**2 + P0i
            b = Jm.T@(y[lo:hi]-mu0)/TRUE["sig"]**2 + P0i@(m0-va)
            v_lin = va + np.linalg.solve(A, b)
            allA[r, k] = roas_hat(v_lin, X[lo:hi])
            # full MAP for comparison (same data)
            v, _ = fit_map(X[lo:hi], y[lo:hi], lo, m0, L0, v_init=v_prev)
            v_prev = v.copy()
            allM[r, k] = roas_hat(v, X[lo:hi])
    out = {}
    for name, A_ in [("anchored", allA), ("full_map", allM)]:
        d = A_[:, :, 0]-A_[:, :, 1]
        rho = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0, 1]
        z = d.mean()/d.std()
        f = np.mean(np.sign(d[:, :-1]) != np.sign(d[:, 1:]))
        fl = flip_law(abs(z), omega)
        out[name] = dict(rho=float(rho), z=float(z), f=float(f), f_law_omega=float(fl), sd=float(d.std()))
        print(f"{name:9s}: rho01={rho:.3f} (omega={omega:.3f}) z={z:+.2f} "
              f"flip={f:.3f} law(z,omega)={fl:.3f} sd={d.std():.3f}")
    json.dump(out, open(RES/"a4_anchored.json","w"), indent=1)

if False:  # standalone entry of merged attack module (disabled)
    import sys; globals()[sys.argv[1]]()


# ======================================================================
# ATTACK/REFINE MODULE: attack5 (rounds 2-3; imports the base module above when
# run standalone as originally executed: from d07 import *)
# ======================================================================
import numpy as np, json
# from d07 import *  (merged: base module is above)

def decomp(seed=41, n_rep=60):
    """Exact decomposition for the anchored (linear-in-y) estimator:
    theta_k = E[theta_k|X] + C_k J_k' eps_k / sig^2.
    Measure: V_design = Var over k,r of g'E[theta|X]; V_noise = mean g'C J'J C g /sig^2...;
    rho_pred = (omega_eff*V_noise + rho_X*V_design)/(V_noise+V_design)."""
    W, Delta, K_ref = 104, 13, 8
    T = W + (K_ref-1)*Delta
    omega = (W-Delta)/W
    m0, L0 = base_prior()
    P0i = np.diag(1.0/P0_SD**2)
    sig = TRUE["sig"]
    gsel = None
    mean_part = np.zeros((n_rep, K_ref))     # g' E[theta|X]  (ROAS diff of mean)
    noise_var = np.zeros((n_rep, K_ref))     # g' C J'J C g /sig^2
    noise_cov = np.zeros((n_rep, K_ref-1))   # adjacent shared-noise covariance
    rng0 = np.random.default_rng(seed)
    for r in range(n_rep):
        rng_r = np.random.default_rng(seed*1000+r)
        X = gen_spend(T, rng_r)
        y0, _ = gen_y(X, rng_r)              # only for anchor fit
        va, _ = fit_map(X[:W], y0[:W], 0, m0, L0)
        mu_true_all = model_mean(V_TRUE, X, 0)  # noiseless truth mean
        Js, Cs, Es = [], [], []
        for k in range(K_ref):
            lo, hi = k*Delta, k*Delta+W
            eps_ = 1e-6
            mu0 = model_mean(va, X[lo:hi], lo)
            Jm = np.zeros((W, NP_))
            for p in range(NP_):
                vp = va.copy(); step = eps_*max(abs(va[p]), 1.0)
                vp[p] += step
                Jm[:, p] = (model_mean(vp, X[lo:hi], lo)-mu0)/step
            A = Jm.T@Jm/sig**2 + P0i
            C = np.linalg.inv(A)
            b_mean = Jm.T@(mu_true_all[lo:hi]-mu0)/sig**2 + P0i@(m0-va)
            v_mean = va + C@b_mean
            # gradient of g = ROAS0-ROAS1 at v_mean (numeric)
            g = np.zeros(NP_)
            for p in range(NP_):
                vp = v_mean.copy(); step = 1e-5*max(abs(v_mean[p]),1e-3)
                vp[p] += step
                Rp = roas_hat(vp, X[lo:hi]); Rm_ = roas_hat(v_mean, X[lo:hi])
                g[p] = ((Rp[0]-Rp[1]) - (Rm_[0]-Rm_[1]))/step
            Rm_ = roas_hat(v_mean, X[lo:hi])
            mean_part[r, k] = Rm_[0]-Rm_[1]
            M = C@Jm.T/sig**2       # theta = mean + M eps
            noise_var[r, k] = g@(M@M.T)@g * sig**2
            Js.append((M, g, lo, hi))
        for k in range(K_ref-1):
            M1, g1, lo1, hi1 = Js[k]; M2, g2, lo2, hi2 = Js[k+1]
            # shared eps indices: [lo2, hi1)
            s1 = slice(lo2-lo1, W); s2 = slice(0, hi1-lo2)
            noise_cov[r, k] = g1@(M1[:, s1]@M2[:, s2].T)@g2 * sig**2
    V_design = mean_part.var()
    rho_X = np.corrcoef(mean_part[:, :-1].ravel(), mean_part[:, 1:].ravel())[0, 1]
    V_noise = noise_var.mean()
    rho_noise = noise_cov.mean()/V_noise
    V_tot = V_design + V_noise
    rho_pred = (rho_noise*V_noise + rho_X*V_design)/V_tot
    print(f"V_design={V_design:.4f} (share {V_design/V_tot:.2f}) rho_X={rho_X:.3f}")
    print(f"V_noise={V_noise:.4f} (share {V_noise/V_tot:.2f}) rho_noise={rho_noise:.3f} (omega={omega:.3f})")
    print(f"rho_pred={rho_pred:.3f}  [attack4 measured rho=0.568]")
    json.dump(dict(V_design=float(V_design), rho_X=float(rho_X), V_noise=float(V_noise),
                   rho_noise=float(rho_noise), rho_pred=float(rho_pred)),
              open(RES/"a5_decomp.json","w"), indent=1)

if False:  # standalone entry of merged attack module (disabled)
    decomp()


# ======================================================================
# ATTACK/REFINE MODULE: attack6 (rounds 2-3; imports the base module above when
# run standalone as originally executed: from d07 import *)
# ======================================================================
import numpy as np, json
# from d07 import *  (merged: base module is above)

def spec_select(seed=50, n_rep=40):
    """M4 test: per-refresh spec selection among W in {78,104,130} x seas {on,off},
    chosen by in-window NRMSE, vs pre-registered spec (W=104, seas on)."""
    Delta, K_ref = 13, 8
    Wmax = 130
    T = Wmax + (K_ref-1)*Delta
    m0, L0 = base_prior()
    m0n = m0.copy(); L0n = np.linalg.cholesky(np.diag(np.concatenate([[1/9., 1e6, 1e6], 1/ P0_SD[3:]**2])))  # seas off: gc,gs pinned to 0
    specs = [(78, True), (104, True), (130, True), (78, False), (104, False), (130, False)]
    allR = {m: np.zeros((n_rep, K_ref, J)) for m in ["prereg", "select"]}
    pick_hist = np.zeros(len(specs))
    switch = 0; nsw = 0
    for r in range(n_rep):
        rng_r = np.random.default_rng(seed*1000+r)
        X = gen_spend(T, rng_r); y, _ = gen_y(X, rng_r)
        v_prev = None; last_pick = None
        for k in range(K_ref):
            hi = Wmax + k*Delta
            cands, crits = [], []
            for si, (Wl, seas) in enumerate(specs):
                lo = hi - Wl
                mm, LL = (m0, L0) if seas else (m0n, L0n)
                v, cov = fit_map(X[lo:hi], y[lo:hi], lo, mm, LL, v_init=v_prev)
                mu = model_mean(v, X[lo:hi], lo)
                p_eff = NP_ - (0 if seas else 2)
                aic = len(y[lo:hi])*np.log(np.mean((y[lo:hi]-mu)**2)) + 2*p_eff
                cands.append(roas_hat(v, X[hi-104:hi])); crits.append(aic)
                if Wl == 104 and seas:
                    allR["prereg"][r, k] = cands[-1]
                    v_prev = v.copy()
            pk = int(np.argmin(crits))
            pick_hist[pk] += 1
            if last_pick is not None:
                switch += (pk != last_pick); nsw += 1
            last_pick = pk
            allR["select"][r, k] = cands[pk]
    out = {}
    for m in ["prereg", "select"]:
        d = allR[m][:, :, 0]-allR[m][:, :, 1]
        rho = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0, 1]
        f = np.mean(np.sign(d[:, :-1]) != np.sign(d[:, 1:]))
        se = np.sqrt(f*(1-f)/(n_rep*(K_ref-1)))
        out[m] = dict(flip=float(f), se=float(se), rho=float(rho), sd=float(d.std()))
        print(f"{m:8s}: flip01={f:.3f}+-{se:.3f} rho={rho:.3f} sd={d.std():.3f}")
    out["spec_switch_rate"] = switch/max(nsw,1)
    out["pick_hist"] = (pick_hist/pick_hist.sum()).tolist()
    print(f"spec switch rate between adjacent refreshes: {switch/max(nsw,1):.3f}; picks {np.round(pick_hist/pick_hist.sum(),2)}")
    json.dump(out, open(RES/"a6_spec.json","w"), indent=1)

if False:  # standalone entry of merged attack module (disabled)
    spec_select()


# ======================================================================
# ATTACK/REFINE MODULE: attack7 (rounds 2-3; imports the base module above when
# run standalone as originally executed: from d07 import *)
# ======================================================================
import numpy as np, json
# from d07 import *  (merged: base module is above)

def spec_select2(seed=51, n_rep=40):
    """Near-tie selection: specs = alpha-prior camps {short: logit-alpha mean -1.5,
    long: +1.5, tight sd 0.5} x W {104,130}. Among specs within 2 AIC of best,
    pick uniformly at random (analyst arbitrariness). vs prereg (W=104, broad alpha)."""
    Delta, K_ref = 13, 8
    Wmax = 130
    T = Wmax + (K_ref-1)*Delta
    m0, L0 = base_prior()
    def camp_prior(mu_a):
        m = P0_MEAN.copy(); sd = P0_SD.copy()
        m[12:15] = mu_a; sd[12:15] = 0.5
        return m, np.linalg.cholesky(np.diag(1/sd**2))
    specs = []
    for mu_a in [-1.5, 1.5]:
        for Wl in [104, 130]:
            specs.append((Wl, mu_a))
    allR = {m: np.zeros((n_rep, K_ref, J)) for m in ["prereg", "select"]}
    switch, nsw = 0, 0
    tie_sizes = []
    for r in range(n_rep):
        rng_r = np.random.default_rng(seed*1000+r)
        X = gen_spend(T, rng_r); y, _ = gen_y(X, rng_r)
        v_prev = None; last_pick = None
        for k in range(K_ref):
            hi = Wmax + k*Delta
            cands, aics = [], []
            for (Wl, mu_a) in specs:
                lo = hi - Wl
                mm, LL = camp_prior(mu_a)
                v, cov = fit_map(X[lo:hi], y[lo:hi], lo, mm, LL)
                mu = model_mean(v, X[lo:hi], lo)
                aic = Wl*np.log(np.mean((y[lo:hi]-mu)**2)/1.0) - Wl*np.log(Wl)*0 + 2*NP_
                aic = Wl*np.mean((y[lo:hi]-mu)**2)/TRUE["sig"]**2/Wl*Wl  # chi2-scale per window... use mean sq/sig2 * n
                aic = np.sum((y[lo:hi]-mu)**2)/TRUE["sig"]**2 / Wl * 104  # normalized deviance per 104wk
                cands.append(roas_hat(v, X[hi-104:hi])); aics.append(aic)
            aics = np.array(aics)
            near = np.where(aics < aics.min() + 2.0)[0]
            tie_sizes.append(len(near))
            pk = int(rng_r.choice(near))
            if last_pick is not None:
                switch += (pk != last_pick); nsw += 1
            last_pick = pk
            allR["select"][r, k] = cands[pk]
            v, cov = fit_map(X[hi-104:hi], y[hi-104:hi], hi-104, m0, L0, v_init=v_prev)
            v_prev = v.copy()
            allR["prereg"][r, k] = roas_hat(v, X[hi-104:hi])
    out = {}
    for m in ["prereg", "select"]:
        d = allR[m][:, :, 0]-allR[m][:, :, 1]
        rho = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0, 1]
        f = np.mean(np.sign(d[:, :-1]) != np.sign(d[:, 1:]))
        se = np.sqrt(f*(1-f)/(n_rep*(K_ref-1)))
        out[m] = dict(flip=float(f), se=float(se), rho=float(rho), sd=float(d.std()))
        print(f"{m:8s}: flip01={f:.3f}+-{se:.3f} rho={rho:.3f} sd={d.std():.3f}")
    out["switch"] = switch/max(nsw,1); out["mean_tie"] = float(np.mean(tie_sizes))
    print(f"switch rate={switch/max(nsw,1):.3f} mean tie-set size={np.mean(tie_sizes):.2f}")
    json.dump(out, open(RES/"a7_spec2.json","w"), indent=1)

if False:  # standalone entry of merged attack module (disabled)
    spec_select2()
