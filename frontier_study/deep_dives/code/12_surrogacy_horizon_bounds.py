"""
Dive 12 — Long-run advertising effects under weak surrogacy:
the surrogate-index bias in a dynamic model, and the completely-monotone
tail bound (moment-problem LP) indexed by a maximum memory half-life.

Usage:
  python 12_surrogacy_horizon_bounds.py quick        # headline numbers (~4 min)
  python 12_surrogacy_horizon_bounds.py e1 ... e27   # individual experiments (prefix match)
Requires numpy + scipy + cvxpy (Clarabel).
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import linprog
from scipy import stats

rng_global = np.random.default_rng(0)

# ----------------------------------------------------------------------------
# 1. Structural impulse response: g(h) = sum_j w_j rho_j^h  (mixture of geometric)
#    "activation" component (fast, alpha) + "brand-stock" component (slow, rho_B)
# ----------------------------------------------------------------------------
def impulse(H, comps):
    """comps: list of (weight, rho[, lag]) ; returns g[0..H-1] (per unit spend)."""
    g = np.zeros(H)
    h = np.arange(H)
    for c in comps:
        w, rho = c[0], c[1]
        lag = c[2] if len(c) > 2 else 0
        kind = c[3] if len(c) > 3 else "geom"
        hh = h - lag
        m = hh >= 0
        if kind == "geom":
            g[m] += w * rho ** hh[m]
        elif kind == "erlang2":            # hump: (h+1) rho^h, normalised to sum 1 like geom would
            g[m] += w * (hh[m] + 1) * rho ** hh[m] * (1 - rho)
        elif kind == "negdip":             # short negative lobe (pull-forward): -w * rho^h
            g[m] -= w * rho ** hh[m]
    return g

def long_run_total(comps, L):
    return impulse(L, comps).sum()

def convolve_pulse(g, dx):
    """observed effect curve at t=0..len(g)-1 from spend pulse dx (len<=len(g))"""
    T = len(g)
    out = np.zeros(T)
    for k, d in enumerate(dx):
        if d == 0: continue
        out[k:] += d * g[:T - k]
    return out

# ----------------------------------------------------------------------------
# 2. Geo experiment simulator: y_gt = mu_g + d_gt + effect_t*W_g + eps
#    d_gt = rho_d d_{g,t-1} + eta ; per-week diff-in-means with pre-period control (CUPED-lite)
# ----------------------------------------------------------------------------
def lw_shrink(S, n):
    """Ledoit-Wolf (2004) shrinkage of a sample covariance toward its diagonal (simple version)."""
    d = np.diag(np.diag(S))
    if n <= 2: return d
    num = ((S - d) ** 2).sum()
    # rough LW: shrinkage intensity = min(1, (tr(S^2)/n)/||S-d||^2) proxy
    lam = min(1.0, (np.trace(S @ S) / n) / max(num, 1e-12))
    return lam * d + (1 - lam) * S

def simulate_experiment(effect_curve, G=100, pre=52, rho_d=0.6, s_eta=1.0, s_eps=1.0,
                        s_mu=2.0, rng=None, common_shock=0.0, t_noise_df=None, full_cov=True,
                        ancova=True):
    """returns tau_hat (H,), V (H x H covariance; diagonal if full_cov=False), raw data."""
    rng = rng or rng_global
    T = len(effect_curve)
    Ttot = pre + T
    W = np.zeros(G, int); W[rng.permutation(G)[:G // 2]] = 1
    mu = rng.normal(0, s_mu, G)
    d = np.zeros((G, Ttot))
    d[:, 0] = rng.normal(0, s_eta / np.sqrt(1 - rho_d ** 2), G)
    for t in range(1, Ttot):
        d[:, t] = rho_d * d[:, t - 1] + rng.normal(0, s_eta, G)
    if t_noise_df:
        eps = stats.t.rvs(t_noise_df, size=(G, Ttot), random_state=rng) * s_eps / np.sqrt(t_noise_df / (t_noise_df - 2))
    else:
        eps = rng.normal(0, s_eps, (G, Ttot))
    y = mu[:, None] + d + eps
    if common_shock:
        y += rng.normal(0, common_shock, Ttot)[None, :]
    y[:, pre:] += W[:, None] * effect_curve[None, :]
    ybar_pre = y[:, :pre].mean(1)
    if ancova:   # per-week ANCOVA on the pre-period mean, pooled slope (within arm)
        Yp = y[:, pre:]
        xc = ybar_pre - np.where(W == 1, ybar_pre[W == 1].mean(), ybar_pre[W == 0].mean())
        yc = Yp - np.where(W[:, None] == 1, Yp[W == 1].mean(0), Yp[W == 0].mean(0))
        theta = (xc[:, None] * yc).sum(0) / (xc ** 2).sum()
        adj = Yp - theta[None, :] * ybar_pre[:, None]
    else:
        adj = y[:, pre:] - ybar_pre[:, None]
    tr, co = adj[W == 1], adj[W == 0]
    tau_hat = tr.mean(0) - co.mean(0)
    if full_cov:
        St = np.cov(tr, rowvar=False); Sc = np.cov(co, rowvar=False)
        V = lw_shrink(St, tr.shape[0]) / tr.shape[0] + lw_shrink(Sc, co.shape[0]) / co.shape[0]
    else:
        V = np.diag(tr.var(0, ddof=1) / tr.shape[0] + co.var(0, ddof=1) / co.shape[0])
    return tau_hat, V, (y, W, pre)

# ----------------------------------------------------------------------------
# 3. The completely-monotone tail bound (LP over a grid of decay rates)
#    model: effect curve tau_t = sum_k dx_k g(t-k), g(h)=int rho^h dmu(rho), mu>=0 on [0,rho_max]
#    target: Theta_L = total incremental sales through week L (per pulse)
#    feasibility: |tau_hat_t - (A m)_t| <= z*se_t  for t<H   (box, simultaneous z)
# ----------------------------------------------------------------------------
def design_matrices(dx, H, L, rhos, kernels=("geom",)):
    """A: H x J map from weights to observed effect curve; c: J target coefficients."""
    cols, cvec = [], []
    for kind in kernels:
        for r in rhos:
            if kind == "geom":
                g = impulse(L, [(1.0, r)])
            elif kind == "erlang2":
                g = impulse(L, [(1.0, r, 0, "erlang2")])
            elif kind.startswith("lag"):          # delayed geometric, lag k
                g = impulse(L, [(1.0, r, int(kind[3:]))])
            full = convolve_pulse(g, dx)
            cols.append(full[:H]); cvec.append(full.sum())
    return np.array(cols).T, np.array(cvec)

def tail_bounds(tau_hat, se, dx, L, rho_max, z=2.0, J=300, kernels=("geom",), rho_min=0.0,
                extra_neg=None):
    """returns (lo, hi, feasible) for Theta_L. extra_neg: (psi, rho_neg) allows a negative
    geometric lobe with weight in [0, psi] (pull-forward sensitivity)."""
    H = len(tau_hat)
    rhos = np.linspace(rho_min, rho_max, J)
    A, c = design_matrices(dx, H, L, rhos, kernels)
    bounds = [(0, None)] * A.shape[1]
    if extra_neg is not None:
        psi, rneg = extra_neg
        g = impulse(L, [(1.0, rneg)]); full = convolve_pulse(g, dx)
        A = np.column_stack([A, -full[:H]]); c = np.append(c, -full.sum())
        bounds.append((0, psi))
    # |A m - tau| <= z se  ->  A m <= tau+z se ; -A m <= -(tau - z se)
    A_ub = np.vstack([A, -A])
    b_ub = np.concatenate([tau_hat + z * se, -(tau_hat - z * se)])
    out = []
    for sign in (1, -1):
        res = linprog(sign * c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.status != 0:
            return np.nan, np.nan, False
        out.append(sign * res.fun)
    return out[0], out[1], True

def simultaneous_z(H, alpha=0.05):
    """Sidak: P(all |N| <= z) = 1-alpha"""
    return stats.norm.ppf(1 - (1 - (1 - alpha) ** (1 / H)) / 2)

# ----------------------------------------------------------------------------
# 4. Surrogate index (ACIK) in the dynamic world: observational panel with
#    AR(1) demand (+noise); Y_L = sum_{t<=L} y_t regressed on (y_1..y_H)
# ----------------------------------------------------------------------------
def si_coefficients_exact(H, L, rho_d, s_eta=1.0, s_eps=1.0):
    """population regression coefficients of Y_L = sum_{t=1}^L y_t on y_1..y_H
    for y_t = d_t + eps_t, d AR(1) stationary."""
    vd = s_eta ** 2 / (1 - rho_d ** 2)
    t = np.arange(1, L + 1)
    C = vd * rho_d ** np.abs(t[:, None] - t[None, :]) + s_eps ** 2 * np.eye(L)
    S = C[:H, :H]
    cov_SY = C[:H, :].sum(1)
    gamma = np.linalg.solve(S, cov_SY)
    return gamma

def si_prediction(gamma, tau_curve):
    return gamma @ tau_curve[:len(gamma)]

def acik_r2_bound(tau_hat_L_si, r2_W_S, r2_Y_S):
    """ACIK (NBER 2019 eq 4.3): |bias| <= tau * sqrt((1-R2_{W|S})(1-R2_{Y|S}))"""
    return abs(tau_hat_L_si) * np.sqrt((1 - r2_W_S) * (1 - r2_Y_S))

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def half_life(rho): return np.log(0.5) / np.log(rho)
def rho_from_hl(hl): return 0.5 ** (1 / hl)

BASE = dict(alpha=0.5, rho_B=rho_from_hl(26), w_act=1.0, w_brand=None)  # brand weight set by phi

def base_comps(phi=0.4, L=104, alpha=0.5, rho_B=None):
    """activation weight 1; brand weight chosen so brand share of long-run (L) total = phi"""
    rho_B = rho_B or BASE["rho_B"]
    act_total = (1 - alpha ** L) / (1 - alpha)
    br_unit = (1 - rho_B ** L) / (1 - rho_B)
    w_b = phi / (1 - phi) * act_total / br_unit
    return [(1.0, alpha), (w_b, rho_B)]

# ============================================================================
# EXPERIMENTS
# ============================================================================
def e1_si_bias_closed_form(verbose=True):
    """SI bias = gamma'tau_{1:H} - tau_L  vs Monte Carlo of the full ACIK pipeline."""
    L, H, rho_d = 104, 13, 0.6
    comps = base_comps(phi=0.4, L=L)
    dx = np.array([1.0] * 4)                # 4-week pulse of unit spend
    g = impulse(L, comps); curve = convolve_pulse(g, dx)
    tau_L = curve.sum()
    gamma = si_coefficients_exact(H, L, rho_d)
    pred = si_prediction(gamma, curve)
    if verbose:
        print(f"E1  true tau_L={tau_L:.3f}  SI(H={H}) prediction={pred:.3f}  bias={pred-tau_L:+.3f} ({(pred/tau_L-1)*100:+.1f}%)")
        print("    gamma (last 4):", np.round(gamma[-4:], 3), " sum gamma:", round(gamma.sum(), 3))
    # MC pipeline: fit SI on panel, apply to experiment
    rng = np.random.default_rng(1)
    preds = []
    for rep in range(40):
        # observational panel: 300 geos x 1 window
        Gp = 300
        d = np.zeros((Gp, L)); d[:, 0] = rng.normal(0, 1 / np.sqrt(1 - rho_d ** 2), Gp)
        for t in range(1, L): d[:, t] = rho_d * d[:, t - 1] + rng.normal(0, 1, Gp)
        y = d + rng.normal(0, 1, (Gp, L))
        X = np.column_stack([np.ones(Gp), y[:, :H]]); YL = y.sum(1)
        gh = np.linalg.lstsq(X, YL, rcond=None)[0][1:]
        th, V, _ = simulate_experiment(curve[:H], G=200, rng=rng); se = np.sqrt(np.diag(V))
        preds.append(gh @ th)
    preds = np.array(preds)
    if verbose:
        print(f"    MC pipeline: mean SI = {preds.mean():.3f} ± {preds.std(ddof=1)/np.sqrt(len(preds)):.3f}  (closed form {pred:.3f})")
    return pred, tau_L, preds

def e2_crossing_horizon(verbose=True):
    """SI bias as function of H: sign flip at a crossing horizon."""
    L, rho_d = 104, 0.6
    dx = np.array([1.0] * 4)
    rows = []
    for phi in (0.0, 0.2, 0.4, 0.6):
        comps = base_comps(phi=phi, L=L) if phi > 0 else [(1.0, 0.5)]
        curve = convolve_pulse(impulse(L, comps), dx); tau_L = curve.sum()
        line = []
        for H in (2, 4, 6, 8, 13, 26, 52):
            gamma = si_coefficients_exact(H, L, rho_d)
            line.append((si_prediction(gamma, curve) / tau_L - 1) * 100)
        rows.append((phi, line))
        if verbose:
            print(f"E2  phi={phi:.1f}  SI bias % by H (2,4,6,8,13,26,52): " + " ".join(f"{b:+6.1f}" for b in line))
    return rows

def e3_tail_bound_noisefree(verbose=True):
    """Sharpness: with exact tau_{1:H} and rho_max >= true rho_B, is truth inside; how wide?"""
    L, H = 104, 13
    dx = np.array([1.0] * 4)
    for phi in (0.0, 0.2, 0.4, 0.6):
        comps = base_comps(phi=phi, L=L) if phi > 0 else [(1.0, 0.5)]
        curve = convolve_pulse(impulse(L, comps), dx); tau_L = curve.sum()
        tau_H = curve[:H]
        se = np.full(H, 1e-4)
        outs = []
        for hl in (13, 26, 52, 104):
            lo, hi, ok = tail_bounds(tau_H, se, dx, L, rho_from_hl(hl), z=1.0)
            outs.append((hl, lo, hi))
        if verbose:
            s = "  ".join(f"hl≤{hl}: [{lo:.2f},{hi:.2f}]" for hl, lo, hi in outs)
            print(f"E3  phi={phi:.1f} true={tau_L:.2f} H={H}  {s}")
    return

def e4_noisy_bounds(verbose=True, G=100, H=13, phi=0.4, hl_max=52, reps=200, seed=2, dx=None, L=104,
                    rho_d=0.6, z=None, kernels=("geom",)):
    """Coverage and width of the LP bound under geo-experiment noise."""
    rng = np.random.default_rng(seed)
    dx = np.array([1.0] * 4) if dx is None else dx
    comps = base_comps(phi=phi, L=L) if phi > 0 else [(1.0, 0.5)]
    curve = convolve_pulse(impulse(L, comps), dx); tau_L = curve.sum()
    z = simultaneous_z(H) if z is None else z
    los, his, cover, feas = [], [], 0, 0
    for r in range(reps):
        th, V, _ = simulate_experiment(curve[:H], G=G, rho_d=rho_d, rng=rng); se = np.sqrt(np.diag(V))
        lo, hi, ok = tail_bounds(th, se, dx, L, rho_from_hl(hl_max), z=z, kernels=kernels)
        if not ok: continue
        feas += 1; los.append(lo); his.append(hi); cover += (lo <= tau_L <= hi)
    los, his = np.array(los), np.array(his)
    res = dict(tau_L=tau_L, lo=los.mean(), hi=his.mean(), lo_sd=los.std(), hi_sd=his.std(),
               cover=cover / max(feas, 1), feas=feas / reps, width=(his - los).mean())
    if verbose:
        print(f"E4  G={G} H={H} phi={phi} hl_max={hl_max} z={z:.2f}: true={tau_L:.2f} "
              f"bounds [{res['lo']:.2f}±{res['lo_sd']:.2f}, {res['hi']:.2f}±{res['hi_sd']:.2f}] "
              f"width/true={res['width']/tau_L:.2f} coverage={res['cover']:.3f} feasible={res['feas']:.2f}")
    return res

def e5_power_sweep(verbose=True):
    """How many geos / how long a horizon make the bound decision-informative?"""
    print("E5  width/true of the LP bound (hl_max=52, phi=0.4):")
    for H in (8, 13, 26, 52):
        line = []
        for G in (50, 100, 200, 400, 1000):
            r = e4_noisy_bounds(verbose=False, G=G, H=H, reps=60, seed=3 + H)
            line.append(f"G={G}:{r['width']/r['tau_L']:.2f}(c{r['cover']:.2f})")
        print(f"    H={H:3d}  " + "  ".join(line))



# ----------------------------------------------------------------------------
# 3b. Ellipsoid (chi-square) version of the bound: convex QCQP via cvxpy/Clarabel
#     {m>=0 : (A m - tau)' V^{-1} (A m - tau) <= chi2_{H,1-alpha}} ; bounds on c'm
# ----------------------------------------------------------------------------
import cvxpy as cp
def make_blocks(H, weekly=13, block=13):
    """index groups: weeks 0..weekly-1 individually, then blocks of `block` weeks."""
    groups = [[t] for t in range(min(H, weekly))]
    t = weekly
    while t < H:
        groups.append(list(range(t, min(H, t + block)))); t += block
    return groups

def block_matrix(groups, H):
    B = np.zeros((len(groups), H))
    for i, g in enumerate(groups): B[i, g] = 1.0
    return B

def crit_radius(k, n_eff, alpha=0.05):
    """Hotelling T^2 critical value for k dims with covariance estimated on n_eff dof (chi2 if n_eff None)."""
    if n_eff is None or n_eff <= k + 1:
        return stats.chi2.ppf(1 - alpha, k)
    return k * (n_eff - 1) / (n_eff - k) * stats.f.ppf(1 - alpha, k, n_eff - k)

def tail_bounds_ellipsoid(tau_hat, V, dx, L, rho_max, alpha=0.05, J=200, kernels=("geom",),
                          rho_min=0.0, extra_neg=None, chi2=None, groups=None, n_eff=None):
    """Convex QCQP bound on Theta_L. groups: optional week-blocking (list of index lists);
    n_eff: dof of the covariance estimate (Hotelling correction)."""
    H = len(tau_hat)
    rhos = np.linspace(rho_min, rho_max, J)
    A, c = design_matrices(dx, H, L, rhos, kernels)
    ub = [None] * A.shape[1]
    if extra_neg is not None:
        psi, rneg = extra_neg
        g = impulse(L, [(1.0, rneg)]); full = convolve_pulse(g, dx)
        A = np.column_stack([A, -full[:H]]); c = np.append(c, -full.sum()); ub.append(psi)
    if np.ndim(V) == 1: V = np.diag(V)
    if groups is not None:
        B = block_matrix(groups, H)
        tau_hat, V, A = B @ tau_hat, B @ V @ B.T, B @ A
    k = len(tau_hat)
    Lw = np.linalg.cholesky(np.linalg.inv(V)); W = Lw.T
    r = chi2 if chi2 is not None else crit_radius(k, n_eff, alpha)
    m = cp.Variable(A.shape[1], nonneg=True)
    cons = [cp.sum_squares(W @ (A @ m - tau_hat)) <= r]
    for j, u in enumerate(ub):
        if u is not None: cons.append(m[j] <= u)
    out = []
    for sign in (1, -1):
        prob = cp.Problem(cp.Minimize(sign * (c @ m)), cons)
        try:
            prob.solve(solver="CLARABEL")
        except Exception:
            prob.solve(solver="SCS")
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return np.nan, np.nan, False
        out.append(sign * prob.value)
    return out[0], out[1], True

def e6_ellipsoid_vs_box(verbose=True, G=100, H=13, phi=0.4, hl_max=52, reps=100, seed=5, L=104):
    """Round-3 refinement: chi-square ellipsoid vs Sidak box."""
    rng = np.random.default_rng(seed)
    dx = np.array([1.0] * 4)
    comps = base_comps(phi=phi, L=L) if phi > 0 else [(1.0, 0.5)]
    curve = convolve_pulse(impulse(L, comps), dx); tau_L = curve.sum()
    z = simultaneous_z(H)
    box, ell, cb, ce = [], [], 0, 0
    for r in range(reps):
        th, V, _ = simulate_experiment(curve[:H], G=G, rng=rng); se = np.sqrt(np.diag(V))
        lo, hi, ok = tail_bounds(th, se, dx, L, rho_from_hl(hl_max), z=z)
        lo2, hi2, ok2 = tail_bounds_ellipsoid(th, V, dx, L, rho_from_hl(hl_max))
        if ok and ok2:
            box.append((lo, hi)); ell.append((lo2, hi2))
            cb += lo <= tau_L <= hi; ce += lo2 <= tau_L <= hi2
    box, ell = np.array(box), np.array(ell)
    n = len(box)
    if verbose:
        print(f"E6  G={G} H={H} phi={phi} hl_max={hl_max} true={tau_L:.2f}: "
              f"box [{box[:,0].mean():.2f},{box[:,1].mean():.2f}] w/true={(box[:,1]-box[:,0]).mean()/tau_L:.2f} cov={cb/n:.2f} | "
              f"ellipsoid [{ell[:,0].mean():.2f},{ell[:,1].mean():.2f}] w/true={(ell[:,1]-ell[:,0]).mean()/tau_L:.2f} cov={ce/n:.2f}")
    return dict(tau_L=tau_L, box=box, ell=ell, cov_box=cb / n, cov_ell=ce / n)

QUICK = ["e1_si_bias_closed_form", "e2_crossing_horizon", "e3_tail_bound_noisefree", "e6_ellipsoid_vs_box", "e9_profile_bound"]

def e2b_sign_flip(verbose=True):
    """SI bias sign depends on demand persistence rho_d vs effect persistence: crossing horizon."""
    L = 104; dx = np.array([1.0] * 4)
    print("E2b SI bias % (phi=0.2, hl_B=26) by H for several demand persistences rho_d:")
    for rho_d in (0.3, 0.6, 0.8, 0.9, 0.95):
        comps = base_comps(phi=0.2, L=L)
        curve = convolve_pulse(impulse(L, comps), dx); tau_L = curve.sum()
        line = []
        for H in (2, 4, 6, 8, 13, 26, 52):
            gamma = si_coefficients_exact(H, L, rho_d)
            line.append((si_prediction(gamma, curve) / tau_L - 1) * 100)
        print(f"    rho_d={rho_d:.2f}: " + " ".join(f"{b:+6.1f}" for b in line))
    print("    (phi=0, pure activation alpha=0.5):")
    for rho_d in (0.6, 0.9, 0.95):
        curve = convolve_pulse(impulse(L, [(1.0, 0.5)]), dx); tau_L = curve.sum()
        line = [(si_prediction(si_coefficients_exact(H, L, rho_d), curve) / tau_L - 1) * 100 for H in (2, 4, 6, 8, 13, 26, 52)]
        print(f"    rho_d={rho_d:.2f}: " + " ".join(f"{b:+6.1f}" for b in line))

def width_law(sigma_w, G, H, rho_max, D, L=104, z=1.96):
    """predicted upper-lower width of Theta_L bound in the small-tail regime:
    level of a ~constant slow component is pinned to +/- z*sigma_w*sqrt(4/G)/sqrt(H);
    its long-run total multiplies by L_eff = (1-rho^L)/(1-rho) and pulse mass D."""
    se_week = sigma_w * np.sqrt(4.0 / G)           # two arms of G/2 each
    L_eff = (1 - rho_max ** L) / (1 - rho_max)
    return 2 * z * se_week / np.sqrt(H) * L_eff * D

def e7_width_law(verbose=True):
    """Verify the price-of-the-long-run width law across (G, H, rho_max); ellipsoid bound."""
    L = 104; dx = np.array([1.0] * 4); D = dx.sum()
    comps = base_comps(phi=0.4, L=L)
    curve = convolve_pulse(impulse(L, comps), dx); tau_L = curve.sum()
    sigma_w = np.sqrt(1 / (1 - 0.6 ** 2) + 1 + 1 / 8 * 0)  # DiD: var(d)+var(eps) (pre-mean adj adds ~1/8 var, ignore)
    print(f"E7  width law check (true tau_L={tau_L:.2f}); ratio = observed ellipsoid width / predicted")
    rng = np.random.default_rng(11)
    rows = []
    for (G, H, hl) in [(100, 13, 52), (400, 13, 52), (100, 52, 52), (400, 52, 52), (100, 13, 26), (100, 13, 104), (1000, 26, 52), (200, 26, 26)]:
        ws = []
        for r in range(25):
            th, V, _ = simulate_experiment(curve[:H], G=G, rng=rng); se = np.sqrt(np.diag(V))
            lo, hi, ok = tail_bounds_ellipsoid(th, V, dx, L, rho_from_hl(hl))
            if ok: ws.append(hi - lo)
        ws = np.array(ws)
        pred = width_law(sigma_w, G, H, rho_from_hl(hl), D)
        rows.append((G, H, hl, ws.mean(), ws.std() / np.sqrt(len(ws)), pred))
        print(f"    G={G:4d} H={H:2d} hl_max={hl:3d}: width {ws.mean():6.2f}±{ws.std()/np.sqrt(len(ws)):.2f}  predicted {pred:6.2f}  ratio {ws.mean()/pred:.2f}")
    return rows

def e8_acik_bound_comparison(verbose=True, G=100, H=13, phi=0.4, reps=100, seed=7, L=104, rho_d=0.6):
    """ACIK partial-R2 sensitivity bound (their eq 4.3 with R2_{Y|W} <= in-sample) vs the moment bound."""
    rng = np.random.default_rng(seed)
    dx = np.array([1.0] * 4)
    comps = base_comps(phi=phi, L=L)
    curve = convolve_pulse(impulse(L, comps), dx); tau_L = curve.sum()
    gamma = si_coefficients_exact(H, L, rho_d)
    # R2_{Y|S} in the observational panel (population): 1 - resid var / var(Y_L)
    vd = 1 / (1 - rho_d ** 2); t = np.arange(1, L + 1)
    C = vd * rho_d ** np.abs(t[:, None] - t[None, :]) + np.eye(L)
    varY = C.sum(); covSY = C[:H, :].sum(1); S = C[:H, :H]
    r2_YS = covSY @ np.linalg.solve(S, covSY) / varY
    outs = []
    for r in range(reps):
        th, V, (y, W, pre) = simulate_experiment(curve[:H], G=G, rng=rng); se = np.sqrt(np.diag(V))
        si = gamma @ th
        # R2_{W|S}: regress W on the H surrogate weeks (adjusted) in the experiment
        adj = y[:, pre:] - y[:, :pre].mean(1)[:, None]
        X = np.column_stack([np.ones(G), adj]); b = np.linalg.lstsq(X, W, rcond=None)[0]
        r2_WS = 1 - ((W - X @ b) ** 2).sum() / ((W - W.mean()) ** 2).sum()
        acik_w = 2 * acik_r2_bound(si, r2_WS, r2_YS)
        lo, hi, ok = tail_bounds_ellipsoid(th, V, dx, L, rho_from_hl(52))
        outs.append((si, acik_w, r2_WS, hi - lo if ok else np.nan, (si - acik_w / 2 <= tau_L <= si + acik_w / 2)))
    o = np.array(outs, dtype=float)
    print(f"E8  true={tau_L:.2f}  SI={o[:,0].mean():.2f}±{o[:,0].std():.2f} (bias {o[:,0].mean()/tau_L-1:+.1%});  R2_W|S={o[:,2].mean():.3f} R2_Y|S={r2_YS:.3f}")
    print(f"    ACIK eq4.3 interval width {o[:,1].mean():.2f} (w/true {o[:,1].mean()/tau_L:.2f}), covers truth {o[:,4].mean():.2f};  moment bound width {np.nanmean(o[:,3]):.2f} (w/true {np.nanmean(o[:,3])/tau_L:.2f})")
    return o

CRIT_PROFILE = 6.63

def tail_bounds_profile(tau_hat, V, dx, L, rho_max, alpha=0.05, J=200, kernels=("geom",),
                        rho_min=0.0, extra_neg=None, groups=None, n_eff=None, crit=None):
    """Profile-likelihood-ratio bound: {Theta : Q(Theta) - Q_min <= chi2_1(1-alpha)} where
    Q(Theta) = min_{m>=0, c'm=Theta} (tau-Am)'V^{-1}(tau-Am).  Returns (lo, hi, ok, Qmin)."""
    H = len(tau_hat)
    rhos = np.linspace(rho_min, rho_max, J)
    A, c = design_matrices(dx, H, L, rhos, kernels)
    ub = [None] * A.shape[1]
    if extra_neg is not None:
        psi, rneg = extra_neg
        g = impulse(L, [(1.0, rneg)]); full = convolve_pulse(g, dx)
        A = np.column_stack([A, -full[:H]]); c = np.append(c, -full.sum()); ub.append(psi)
    if np.ndim(V) == 1: V = np.diag(V)
    if groups is not None:
        B = block_matrix(groups, H)
        tau_hat, V, A = B @ tau_hat, B @ V @ B.T, B @ A
    k = len(tau_hat)
    W = np.linalg.cholesky(np.linalg.inv(V)).T
    m = cp.Variable(A.shape[1], nonneg=True)
    ubc = [m[j] <= u for j, u in enumerate(ub) if u is not None]
    p0 = cp.Problem(cp.Minimize(cp.sum_squares(W @ (A @ m - tau_hat))), ubc)
    p0.solve(solver="CLARABEL")
    if p0.status not in ("optimal", "optimal_inaccurate"): return np.nan, np.nan, False, np.nan
    Qmin = p0.value
    if crit is None:
        crit = CRIT_PROFILE   # MC-calibrated (chi2_1 at 0.99 = 6.63): chi2_1(0.95) under-covers at the zero-tail boundary (E9b)
    cons = [cp.sum_squares(W @ (A @ m - tau_hat)) <= Qmin + crit] + ubc
    out = []
    for sign in (1, -1):
        prob = cp.Problem(cp.Minimize(sign * (c @ m)), cons)
        try: prob.solve(solver="CLARABEL")
        except Exception: prob.solve(solver="SCS")
        if prob.status not in ("optimal", "optimal_inaccurate"): return np.nan, np.nan, False, Qmin
        out.append(sign * prob.value)
    return out[0], out[1], True, Qmin

def e9_profile_bound(verbose=True, configs=None, reps=100, seed=13, L=104):
    """Profile-LR bound: width and coverage across configurations (the headline sweep)."""
    dx = np.array([1.0] * 4)
    configs = configs or [(100, 13, 0.4, 52), (100, 26, 0.4, 52), (100, 52, 0.4, 52), (400, 13, 0.4, 52),
                          (1600, 13, 0.4, 52), (1600, 52, 0.4, 52), (100, 13, 0.0, 52), (100, 13, 0.6, 52),
                          (100, 13, 0.4, 26), (100, 13, 0.4, 104), (400, 52, 0.4, 52), (400, 104, 0.4, 52)]
    print("E9  profile-LR bound (chi2_1 radius), blocked weeks, Hotelling-free:")
    res = []
    for (G, H, phi, hl) in configs:
        rng = np.random.default_rng(seed)
        comps = base_comps(phi=phi, L=L) if phi > 0 else [(1.0, 0.5)]
        curve = convolve_pulse(impulse(L, comps), dx); tau_L = curve.sum()
        groups = make_blocks(H)
        los, his, cov, n, sel = [], [], 0, 0, []
        for r in range(reps):
            th, V, _ = simulate_experiment(curve[:H], G=G, rng=rng)
            lo, hi, ok, Q = tail_bounds_profile(th, V, dx, L, rho_from_hl(hl), groups=groups)
            if ok:
                los.append(lo); his.append(hi); n += 1; cov += lo <= tau_L <= hi; sel.append(np.sqrt(V.sum()) / H)
        Leff = (1 - rho_from_hl(hl) ** L) / (1 - rho_from_hl(hl))
        w = (np.mean(his) - np.mean(los))
        res.append((G, H, phi, hl, tau_L, np.mean(los), np.mean(his), cov / n, w / tau_L))
        print(f"    G={G:4d} H={H:3d} phi={phi} hl_max={hl:3d}: true={tau_L:5.2f} [{np.mean(los):5.2f}±{np.std(los):.2f}, {np.mean(his):5.2f}±{np.std(his):.2f}] "
              f"w/true={w/tau_L:.2f} cov={cov/n:.3f} feas={n/reps:.2f}  (hi-true)/(se_level·Leff·D)={(np.mean(his)-tau_L)/(np.mean(sel)*Leff*4):.2f}")
    return res

def e9b_profile_calibration(verbose=True, reps=300, L=104):
    """Boundary undercoverage of the chi2_1 profile radius and the MC calibration."""
    dx = np.array([1.0] * 4)
    print("E9b profile-LR calibration (300 reps; one-sided failure rates, target <=0.025 each):")
    for phi, G, H in [(0.0, 100, 13), (0.0, 400, 26), (0.4, 100, 13), (0.2, 400, 26)]:
        comps = base_comps(phi=phi, L=L) if phi > 0 else [(1.0, 0.5)]
        curve = convolve_pulse(impulse(L, comps), dx); tau_L = curve.sum()
        groups = make_blocks(H)
        line = []
        for crit in (3.84, 6.63, 9.0):
            rng = np.random.default_rng(21); lf = hf = n = 0; ws = []
            for r in range(reps):
                th, V, _ = simulate_experiment(curve[:H], G=G, rng=rng)
                lo, hi, ok, Q = tail_bounds_profile(th, V, dx, L, rho_from_hl(52), groups=groups, crit=crit)
                if ok: n += 1; lf += lo > tau_L; hf += hi < tau_L; ws.append(hi - lo)
            line.append(f"crit={crit}: lo-fail {lf/n:.3f} hi-fail {hf/n:.3f} w/true {np.mean(ws)/tau_L:.2f}")
        print(f"    phi={phi} G={G} H={H}: " + " | ".join(line))

def e10_misspecification(verbose=True, reps=150, L=104, G=400, H=26, seed=31):
    """Attack: brand kernel is not geometric (Erlang-2 hump / delayed onset) or has a negative
    pull-forward lobe. Coverage of the geometric-dictionary bound; repair via enlarged dictionary."""
    dx = np.array([1.0] * 4)
    rho_B = rho_from_hl(26)
    # brand weight chosen so that brand share of L-total is 0.4 for each kernel
    act_total = (1 - 0.5 ** L) / (1 - 0.5)
    cases = {
        "geom (correct)": [(1.0, 0.5), (None, rho_B, 0, "geom")],
        "erlang2 hump": [(1.0, 0.5), (None, rho_B, 0, "erlang2")],
        "delayed onset (lag 8)": [(1.0, 0.5), (None, rho_B, 8, "geom")],
        "pull-forward dip": [(1.0, 0.5), (None, rho_B, 0, "geom"), (0.3, 0.7, 1, "negdip")],
    }
    print(f"E10 misspecified brand kernels (G={G}, H={H}, hl_max=52, brand share 0.4):")
    for name, comps in cases.items():
        # solve brand weight
        unit = impulse(L, [(1.0, comps[1][1], comps[1][2], comps[1][3])]).sum()
        neg = impulse(L, [c for c in comps if len(c) > 3 and c[3] == "negdip"]).sum() if len(comps) > 2 else 0.0
        w_b = 0.4 / 0.6 * (act_total + neg) / unit if False else 0.4 / 0.6 * act_total / unit
        comps = [comps[0], (w_b,) + tuple(comps[1][1:])] + comps[2:]
        curve = convolve_pulse(impulse(L, comps), dx); tau_L = curve.sum()
        groups = make_blocks(H)
        for dict_name, kernels, extra in [("geom dict", ("geom",), None), ("geom+erlang2+lag4/8/13", ("geom", "erlang2", "lag4", "lag8", "lag13"), None),
                                          ("geom + neg lobe psi=0.5", ("geom",), (0.5, 0.7))]:
            rng = np.random.default_rng(seed); los, his, cov, n = [], [], 0, 0
            for r in range(reps):
                th, V, _ = simulate_experiment(curve[:H], G=G, rng=rng)
                lo, hi, ok, Q = tail_bounds_profile(th, V, dx, L, rho_from_hl(52), groups=groups, kernels=kernels, extra_neg=extra, J=120)
                if ok: n += 1; los.append(lo); his.append(hi); cov += lo <= tau_L <= hi
            print(f"    {name:22s} | {dict_name:26s}: true={tau_L:5.2f} [{np.mean(los):5.2f},{np.mean(his):5.2f}] w/true={(np.mean(his)-np.mean(los))/tau_L:.2f} cov={cov/n:.3f}")

def e11_noise_robustness(verbose=True, reps=150, L=104, G=400, H=26, seed=41):
    """Attack: heavy-tailed (t3) geo-week noise; common (national) shocks; stronger demand persistence."""
    dx = np.array([1.0] * 4)
    comps = base_comps(phi=0.4, L=L)
    curve = convolve_pulse(impulse(L, comps), dx); tau_L = curve.sum()
    groups = make_blocks(H)
    print(f"E11 noise robustness (G={G}, H={H}, hl_max=52, true={tau_L:.2f}):")
    for name, kw in [("gaussian baseline", {}), ("t3 noise", dict(t_noise_df=3)), ("common shocks sd=1", dict(common_shock=1.0)),
                     ("rho_d=0.9 demand", dict(rho_d=0.9)), ("rho_d=0.95 demand", dict(rho_d=0.95))]:
        rng = np.random.default_rng(seed); los, his, cov, n = [], [], 0, 0
        for r in range(reps):
            th, V, _ = simulate_experiment(curve[:H], G=G, rng=rng, **kw)
            lo, hi, ok, Q = tail_bounds_profile(th, V, dx, L, rho_from_hl(52), groups=groups)
            if ok: n += 1; los.append(lo); his.append(hi); cov += lo <= tau_L <= hi
        print(f"    {name:20s}: [{np.mean(los):5.2f}±{np.std(los):.2f},{np.mean(his):5.2f}±{np.std(his):.2f}] w/true={(np.mean(his)-np.mean(los))/tau_L:.2f} cov={cov/n:.3f}")

# ----------------------------------------------------------------------------
# 5. Second construction: one-experiment de-confounding of the observational
#    distributed lag.  Panel: y = mu_g + sum_k g_k x_{t-k} + d_t + eps ;
#    x_t = xbar_g + a d_t + u_t.  OLS distributed lag has bias  s * b,
#    b = (a^2 v_d R + s_u^2 I)^{-1} r,  r_k = rho_d^k,  s = a v_d  (scalar, unknown sign).
#    Everything in b is estimable from the spend ACF; s is pinned by the experiment's
#    short-run curve; the panel's long lags then deliver the tail.
# ----------------------------------------------------------------------------
def simulate_panel(g, G_o=200, T_o=156, K=52, rho_d=0.6, a=0.5, s_u=1.0, s_eta=1.0, s_eps=1.0,
                   rng=None, lead=0, hill=None):
    """returns beta_hat (K+1), V_beta (cluster by geo), spend ACF estimates (rho, a2vd, su2).
    lead: spend anticipates demand by `lead` weeks (misspecification of b).
    hill: (K_half, xbar) -> effect uses Hill-saturated spend (nonlinearity misspecification)."""
    rng = rng or rng_global
    burn = len(g)
    T = burn + T_o
    d = np.zeros((G_o, T + lead)); d[:, 0] = rng.normal(0, s_eta / np.sqrt(1 - rho_d ** 2), G_o)
    for t in range(1, T + lead): d[:, t] = rho_d * d[:, t - 1] + rng.normal(0, s_eta, G_o)
    u = rng.normal(0, s_u, (G_o, T))
    xbar = 3.0
    x = xbar + a * d[:, lead:lead + T] + u
    x = np.maximum(x, 0.0)
    xeff = x if hill is None else hill(x)
    # effect: convolution of g with spend
    eff = np.zeros((G_o, T))
    for k in range(len(g)):
        eff[:, k:] += g[k] * xeff[:, :T - k]
    y = rng.normal(0, 2.0, G_o)[:, None] + eff + d[:, :T] + rng.normal(0, s_eps, (G_o, T))
    # distributed-lag regression on lags 0..K with geo FE (within transform), sample t>=burn
    rows_y, rows_x, geo = [], [], []
    for gi in range(G_o):
        Xg = np.column_stack([x[gi, burn - k:T - k] for k in range(K + 1)])
        yg = y[gi, burn:]
        rows_x.append(Xg - Xg.mean(0)); rows_y.append(yg - yg.mean()); geo.append(np.full(len(yg), gi))
    X = np.vstack(rows_x); Y = np.concatenate(rows_y); geo = np.concatenate(geo)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    e = Y - X @ beta
    meat = np.zeros((K + 1, K + 1))
    for gi in range(G_o):
        m = geo == gi; s = X[m].T @ e[m]; meat += np.outer(s, s)
    Vb = XtX_inv @ meat @ XtX_inv * G_o / (G_o - 1)
    # spend ACF (within geo)
    xc = x[:, burn:] - x[:, burn:].mean(1, keepdims=True)
    g0 = (xc * xc).mean(); g1 = (xc[:, 1:] * xc[:, :-1]).mean(); g2 = (xc[:, 2:] * xc[:, :-2]).mean()
    rho_hat = g2 / g1; a2vd = g1 / rho_hat; su2 = g0 - a2vd
    return beta, Vb, (rho_hat, a2vd, su2)

def confound_shape(K, rho, a2vd, su2):
    k = np.arange(K + 1)
    R = rho ** np.abs(k[:, None] - k[None, :])
    r = rho ** k
    return np.linalg.solve(a2vd * R + su2 * np.eye(K + 1), r)

def joint_profile_bound(tau_hat, V_e, dx, beta, Vb, b, L, rho_max, J=150, crit=None, groups=None,
                        use_exp=True, use_panel=True, free_s=True):
    """profile-LR bound on Theta_L combining experiment (tau ~ A_e m) and panel (beta ~ A_g m + s b)."""
    crit = CRIT_PROFILE if crit is None else crit
    H = len(tau_hat); K1 = len(beta)
    rhos = np.linspace(0.0, rho_max, J)
    A_e, c = design_matrices(dx, H, L, rhos)
    A_g = np.array([impulse(K1, [(1.0, r)]) for r in rhos]).T       # K+1 x J : g_k = rho^k
    if groups is not None:
        B = block_matrix(groups, H); tau_hat, V_e, A_e = B @ tau_hat, B @ V_e @ B.T, B @ A_e
    We = np.linalg.cholesky(np.linalg.inv(V_e)).T
    Wb = np.linalg.cholesky(np.linalg.inv(Vb)).T
    m = cp.Variable(J, nonneg=True); s = cp.Variable()
    terms = []
    if use_exp: terms.append(cp.sum_squares(We @ (A_e @ m - tau_hat)))
    if use_panel:
        resid = A_g @ m + (s * b if free_s else 0) - beta
        terms.append(cp.sum_squares(Wb @ resid))
    Q = sum(terms)
    p0 = cp.Problem(cp.Minimize(Q)); p0.solve(solver="CLARABEL")
    if p0.status not in ("optimal", "optimal_inaccurate"): return np.nan, np.nan, False, np.nan, np.nan
    Qmin = p0.value; shat = s.value if free_s and use_panel else np.nan
    out = []
    for sign in (1, -1):
        prob = cp.Problem(cp.Minimize(sign * (c @ m)), [Q <= Qmin + crit])
        try: prob.solve(solver="CLARABEL")
        except Exception: prob.solve(solver="SCS")
        if prob.status not in ("optimal", "optimal_inaccurate"): return np.nan, np.nan, False, Qmin, shat
        out.append(sign * prob.value)
    return out[0], out[1], True, Qmin, shat

def e12_panel_deconfounding(verbose=True, reps=40, L=104, G=100, H=13, seed=51, a=0.5, K=52, G_o=200, T_o=156, lead=0, hill=None, label=""):
    dx = np.array([1.0] * 4)
    comps = base_comps(phi=0.4, L=L)
    g = impulse(L, comps); curve = convolve_pulse(g, dx); tau_L = curve.sum()
    groups = make_blocks(H)
    rng = np.random.default_rng(seed)
    res = {k: [] for k in ("exp", "joint", "panel_naive", "panel_s")}; cov = {k: 0 for k in res}; n = 0; shats = []; Qs = []
    for r in range(reps):
        th, V, _ = simulate_experiment(curve[:H], G=G, rng=rng)
        beta, Vb, (rho_hat, a2vd, su2) = simulate_panel(g, G_o=G_o, T_o=T_o, K=K, a=a, rng=rng, lead=lead, hill=hill)
        b = confound_shape(K, rho_hat, a2vd, su2)
        outs = {
            "exp": joint_profile_bound(th, V, dx, beta, Vb, b, L, rho_from_hl(52), groups=groups, use_panel=False),
            "joint": joint_profile_bound(th, V, dx, beta, Vb, b, L, rho_from_hl(52), groups=groups),
            "panel_naive": joint_profile_bound(th, V, dx, beta, Vb, b, L, rho_from_hl(52), groups=groups, use_exp=False, free_s=False),
            "panel_s": joint_profile_bound(th, V, dx, beta, Vb, b, L, rho_from_hl(52), groups=groups, use_exp=False, free_s=True),
        }
        if all(o[2] for o in outs.values()):
            n += 1
            for k, o in outs.items():
                res[k].append((o[0], o[1])); cov[k] += o[0] <= tau_L <= o[1]
            shats.append(outs["joint"][4]); Qs.append(outs["joint"][3])
    print(f"E12{label} panel de-confounding (a={a}, lead={lead}, hill={'yes' if hill else 'no'}; G={G},H={H}; panel {G_o}x{T_o}, K={K}); true={tau_L:.2f}; n={n}; s_hat={np.mean(shats):.3f}±{np.std(shats):.3f} (true a·v_d={a/(1-0.36):.3f}); Qmin={np.mean(Qs):.1f} (dof≈{len(groups)+K+1-3})")
    for k in res:
        arr = np.array(res[k])
        print(f"    {k:12s}: [{arr[:,0].mean():5.2f}±{arr[:,0].std():.2f}, {arr[:,1].mean():5.2f}±{arr[:,1].std():.2f}] w/true={(arr[:,1]-arr[:,0]).mean()/tau_L:.2f} cov={cov[k]/n:.3f}")
    return res

# ---- experiments added in the completing run (attack/refine rounds) ----


def e13_power_control(reps=100, L=104, seeds=(101,)):
    """Negative/power control: can the bound discriminate phi=0 (no long-run) from phi=0.4?
    Under phi=0.4 truth: P(lo > Theta_L(phi=0)) ; under phi=0 truth: P(hi < Theta_L(phi=0.4))."""
    dx = np.array([1.0]*4)
    t0 = convolve_pulse(impulse(L, [(1.0,0.5)]), dx).sum()
    t4 = convolve_pulse(impulse(L, base_comps(0.4, L)), dx).sum()
    print(f"E13 power control: Theta_L(phi=0)={t0:.2f}, Theta_L(phi=0.4)={t4:.2f}")
    for (G,H) in [(100,13),(400,26),(1600,52),(1600,104),(6400,52)]:
        rng = np.random.default_rng(seeds[0]); groups = make_blocks(H)
        pw = pn = n = 0; sr=[]
        for r in range(reps):
            th,V,_ = simulate_experiment(convolve_pulse(impulse(L, base_comps(0.4,L)),dx)[:H], G=G, rng=rng)
            lo,hi,ok,Q = tail_bounds_profile(th,V,dx,L,rho_from_hl(52),groups=groups)
            th0,V0,_ = simulate_experiment(convolve_pulse(impulse(L,[(1.0,0.5)]),dx)[:H], G=G, rng=rng)
            lo0,hi0,ok0,Q0 = tail_bounds_profile(th0,V0,dx,L,rho_from_hl(52),groups=groups)
            if ok and ok0:
                n+=1; pw += lo > t0; pn += hi0 < t4
                # short-run sum test: reject phi=0 if observed 13..H-week sum exceeds ...
        print(f"    G={G:5d} H={H:3d}: P(lo>{t0:.1f} | phi=.4)={pw/n:.2f}   P(hi<{t4:.1f} | phi=0)={pn/n:.2f}")

def e14_si_with_pre_covariate(L=104, rho_d=0.6):
    """Does conditioning the SI on pre-period outcomes (X) remove phantom persistence? Population calc."""
    dx = np.array([1.0]*4); pre=52
    vd = 1/(1-rho_d**2)
    # joint of (pre-mean xbar, y_1..y_L): stack indices -pre..L-1
    T = pre + L; idx = np.arange(T)
    C = vd*rho_d**np.abs(idx[:,None]-idx[None,:]) + np.eye(T)
    Bm = np.zeros(T); Bm[:pre]=1/pre
    print("E14 SI bias % with pre-period-mean covariate vs without (phi=0.2 / phi=0):")
    for phi in (0.0, 0.2):
        comps = base_comps(phi,L) if phi>0 else [(1.0,0.5)]
        curve = convolve_pulse(impulse(L,comps),dx); tau_L=curve.sum()
        line=[]
        for H in (2,4,8,13,26,52):
            S = np.column_stack([Bm, np.eye(T)[:,pre:pre+H]])           # (T x (H+1)) selection
            Y = np.zeros(T); Y[pre:]=1
            Sig = S.T@C@S; cSY = S.T@C@Y
            gam = np.linalg.solve(Sig,cSY)[1:]                            # coefficients on y_1..y_H
            gam0 = si_coefficients_exact(H,L,rho_d)
            line.append(f"H={H}:{(gam@curve[:H]/tau_L-1)*100:+.1f}({(gam0@curve[:H]/tau_L-1)*100:+.1f})")
        print(f"    phi={phi}: "+"  ".join(line))

def e15_calibration_trueV(reps=300, L=104):
    """Attack on E9b: is the low-side under-coverage from covariance estimation? Use the true V."""
    dx=np.array([1.0]*4)
    print("E15 profile-LR one-sided failure with TRUE covariance (phi=0):")
    for (G,H) in [(100,13),(400,26)]:
        curve = convolve_pulse(impulse(L,[(1.0,0.5)]),dx); tau_L=curve.sum(); groups=make_blocks(H)
        # true V: per-week var = (var(d)+1)/ (G/2) *2 ... estimate by big MC of tau_hat
        rng=np.random.default_rng(7); ths=np.array([simulate_experiment(curve[:H],G=G,rng=rng)[0] for _ in range(2000)])
        Vtrue=np.cov(ths,rowvar=False)
        for crit in (3.84,6.63):
            for label,useV in (("est",None),("true",Vtrue)):
                rng=np.random.default_rng(21); lf=hf=n=0; ws=[]
                for r in range(reps):
                    th,V,_=simulate_experiment(curve[:H],G=G,rng=rng)
                    Vu = V if useV is None else useV
                    lo,hi,ok,Q=tail_bounds_profile(th,Vu,dx,L,rho_from_hl(52),groups=groups,crit=crit)
                    if ok: n+=1; lf+=lo>tau_L; hf+=hi<tau_L; ws.append(hi-lo)
                print(f"    G={G} H={H} crit={crit} V={label}: lo-fail {lf/n:.3f} hi-fail {hf/n:.3f} w/true {np.mean(ws)/tau_L:.2f}")

def e16_misspec_precise(reps=100, L=104):
    """Attack on E10: does the geometric-only dictionary still cover under Erlang/lagged truth when the
    experiment is precise (G=1600, H=52)?"""
    dx=np.array([1.0]*4); rho_B=rho_from_hl(26); act_total=(1-0.5**L)/(1-0.5)
    cases={"erlang2 hump":[(1.0,0.5),(None,rho_B,0,"erlang2")],
           "delayed onset (lag 8)":[(1.0,0.5),(None,rho_B,8,"geom")],
           "pull-forward dip":[(1.0,0.5),(None,rho_B,0,"geom"),(0.3,0.7,1,"negdip")]}
    print("E16 misspecified kernels at high precision (G=1600, H=52, hl_max=52):")
    for name,comps in cases.items():
        unit=impulse(L,[(1.0,comps[1][1],comps[1][2],comps[1][3])]).sum()
        w_b=0.4/0.6*act_total/unit; comps=[comps[0],(w_b,)+tuple(comps[1][1:])]+comps[2:]
        curve=convolve_pulse(impulse(L,comps),dx); tau_L=curve.sum(); H=52; G=1600; groups=make_blocks(H)
        for dict_name,kernels,extra in [("geom dict",("geom",),None),("geom+erlang2+lag4/8/13",("geom","erlang2","lag4","lag8","lag13"),None),("geom+neg lobe psi=.5",("geom",),(0.5,0.7))]:
            rng=np.random.default_rng(31); los=[];his=[];cov=0;n=0;Qs=[]
            for r in range(reps):
                th,V,_=simulate_experiment(curve[:H],G=G,rng=rng)
                lo,hi,ok,Q=tail_bounds_profile(th,V,dx,L,rho_from_hl(52),groups=groups,kernels=kernels,extra_neg=extra,J=120)
                if ok: n+=1;los.append(lo);his.append(hi);cov+=lo<=tau_L<=hi;Qs.append(Q)
            print(f"    {name:22s} | {dict_name:24s}: true={tau_L:5.2f} [{np.mean(los):5.2f},{np.mean(his):5.2f}] w/true={(np.mean(his)-np.mean(los))/tau_L:.2f} cov={cov/n:.3f} Qmin={np.mean(Qs):.1f} (k={len(groups)})")

def e17_panel_misspec(reps=30, L=104):
    """Attack on construction 3: spend anticipates demand (lead), Hill saturation, two-component demand."""
    hill = lambda x: 6.0*x/(x+3.0)
    print("E17 panel de-confounding under misspecification (joint bound):")
    for label,kw in [(" baseline",{}),(" lead=2",dict(lead=2)),(" lead=4",dict(lead=4)),(" hill",dict(hill=hill)),(" a=0 (no confounding)",dict(a=0.0)),(" a=1.5 (strong)",dict(a=1.5))]:
        e12_panel_deconfounding(reps=reps, label=label, **kw)

def simulate_panel2(g, G_o=200, T_o=156, K=52, a=0.5, s_u=1.0, rng=None):
    """two-component demand d = d_fast(rho .3) + d_slow(rho .95, sd .3): ACF is not AR(1)."""
    rng=rng or rng_global; burn=len(g); T=burn+T_o
    def ar(rho,s):
        d=np.zeros((G_o,T)); d[:,0]=rng.normal(0,s/np.sqrt(1-rho**2),G_o)
        for t in range(1,T): d[:,t]=rho*d[:,t-1]+rng.normal(0,s,G_o)
        return d
    d=ar(0.3,1.0)+ar(0.95,0.3)
    u=rng.normal(0,s_u,(G_o,T)); x=np.maximum(3.0+a*d+u,0)
    eff=np.zeros((G_o,T))
    for k in range(len(g)): eff[:,k:]+=g[k]*x[:,:T-k]
    y=rng.normal(0,2,G_o)[:,None]+eff+d+rng.normal(0,1,(G_o,T))
    rows_y,rows_x,geo=[],[],[]
    for gi in range(G_o):
        Xg=np.column_stack([x[gi,burn-k:T-k] for k in range(K+1)]); yg=y[gi,burn:]
        rows_x.append(Xg-Xg.mean(0)); rows_y.append(yg-yg.mean()); geo.append(np.full(len(yg),gi))
    X=np.vstack(rows_x); Y=np.concatenate(rows_y); geo=np.concatenate(geo)
    XtX_inv=np.linalg.inv(X.T@X); beta=XtX_inv@X.T@Y; e=Y-X@beta
    meat=np.zeros((K+1,K+1))
    for gi in range(G_o):
        m=geo==gi; s=X[m].T@e[m]; meat+=np.outer(s,s)
    Vb=XtX_inv@meat@XtX_inv*G_o/(G_o-1)
    xc=x[:,burn:]-x[:,burn:].mean(1,keepdims=True)
    g0=(xc*xc).mean(); g1=(xc[:,1:]*xc[:,:-1]).mean(); g2=(xc[:,2:]*xc[:,:-2]).mean()
    rho_hat=g2/g1; a2vd=g1/rho_hat; su2=g0-a2vd
    return beta,Vb,(rho_hat,a2vd,su2)

def e18_two_component_demand(reps=30, L=104, G=100, H=13, K=52):
    dx=np.array([1.0]*4); comps=base_comps(0.4,L); g=impulse(L,comps); curve=convolve_pulse(g,dx); tau_L=curve.sum()
    groups=make_blocks(H); rng=np.random.default_rng(61)
    res={k:[] for k in ("exp","joint","panel_naive")}; cov={k:0 for k in res}; n=0; Qs=[]
    for r in range(reps):
        th,V,_=simulate_experiment(curve[:H],G=G,rng=rng)
        beta,Vb,(rho_hat,a2vd,su2)=simulate_panel2(g,K=K,rng=rng)
        b=confound_shape(K,rho_hat,a2vd,su2)
        outs={"exp":joint_profile_bound(th,V,dx,beta,Vb,b,L,rho_from_hl(52),groups=groups,use_panel=False),
              "joint":joint_profile_bound(th,V,dx,beta,Vb,b,L,rho_from_hl(52),groups=groups),
              "panel_naive":joint_profile_bound(th,V,dx,beta,Vb,b,L,rho_from_hl(52),groups=groups,use_exp=False,free_s=False)}
        if all(o[2] for o in outs.values()):
            n+=1
            for k,o in outs.items(): res[k].append((o[0],o[1])); cov[k]+=o[0]<=tau_L<=o[1]
            Qs.append(outs["joint"][3])
    print(f"E18 two-component demand (AR(1) shape misspecified): true={tau_L:.2f} n={n} Qmin={np.mean(Qs):.1f} rho_hat~{rho_hat:.2f}")
    for k in res:
        arr=np.array(res[k]); print(f"    {k:12s}: [{arr[:,0].mean():5.2f}±{arr[:,0].std():.2f}, {arr[:,1].mean():5.2f}±{arr[:,1].std():.2f}] w/true={(arr[:,1]-arr[:,0]).mean()/tau_L:.2f} cov={cov[k]/n:.3f}")

def e19_sensitivity_curve(reps=60, L=104, G=400, H=26):
    """The M5 index: bound as a function of the surrogacy-violation parameter hl_max (max half-life)."""
    dx=np.array([1.0]*4); comps=base_comps(0.4,L); curve=convolve_pulse(impulse(L,comps),dx); tau_L=curve.sum()
    groups=make_blocks(H); short=curve[:H].sum()
    print(f"E19 sensitivity curve (G={G},H={H}); true={tau_L:.2f}; observed-window mass={short:.2f}")
    for hl in (4,8,13,26,52,104,208):
        rng=np.random.default_rng(71); los=[];his=[];n=0;cov=0
        for r in range(reps):
            th,V,_=simulate_experiment(curve[:H],G=G,rng=rng)
            lo,hi,ok,Q=tail_bounds_profile(th,V,dx,L,rho_from_hl(hl),groups=groups)
            if ok: n+=1;los.append(lo);his.append(hi);cov+=lo<=tau_L<=hi
        Leff=(1-rho_from_hl(hl)**L)/(1-rho_from_hl(hl))
        print(f"    hl_max={hl:3d} (L_eff={Leff:5.1f}): [{np.mean(los):5.2f}±{np.std(los):.2f},{np.mean(his):5.2f}±{np.std(his):.2f}] w/true={(np.mean(his)-np.mean(los))/tau_L:.2f} cov={cov/n:.2f} n={n}")

def e20_seed_sweep(reps=60, L=104):
    dx=np.array([1.0]*4); comps=base_comps(0.4,L); curve=convolve_pulse(impulse(L,comps),dx); tau_L=curve.sum()
    print("E20 seed sensitivity of the headline (G=400,H=26,hl_max=52) profile bound:")
    for seed in (1,2,3,4,5):
        rng=np.random.default_rng(seed); groups=make_blocks(26); los=[];his=[];cov=0;n=0
        for r in range(reps):
            th,V,_=simulate_experiment(curve[:26],G=400,rng=rng)
            lo,hi,ok,Q=tail_bounds_profile(th,V,dx,L,rho_from_hl(52),groups=groups)
            if ok: n+=1;los.append(lo);his.append(hi);cov+=lo<=tau_L<=hi
        print(f"    seed={seed}: [{np.mean(los):.2f},{np.mean(his):.2f}] w/true={(np.mean(his)-np.mean(los))/tau_L:.2f} cov={cov/n:.2f}")



def solve_safe(prob):
    try: prob.solve(solver="CLARABEL")
    except Exception: pass
    if prob.status not in ("optimal","optimal_inaccurate"):
        try: prob.solve(solver="SCS", max_iters=20000)
        except Exception: pass
    return prob.status in ("optimal","optimal_inaccurate")

def confound_shape_lead(K, rho, a2vd, su2, lead):
    """OLS bias shape when spend anticipates demand by `lead` weeks: x_t = a d_{t+lead}+u.
    Cov(x_{t-k}, d_t) = a v_d rho^{|k-lead|}."""
    k=np.arange(K+1); R=rho**np.abs(k[:,None]-k[None,:]); r=rho**np.abs(k-lead)
    return np.linalg.solve(a2vd*R+su2*np.eye(K+1), r)

def joint_bound2(tau_hat, V_e, dx, beta, Vb, bs, L, rho_max, J=150, crit=None, groups=None, free_kappa=False):
    """Generalised joint profile bound: panel beta ~ kappa*A_g m + sum_l s_l b_l  (bs: list of shapes).
    Multiple shapes = lead-profiling (s_l free for each candidate lead). kappa: panel/experiment scale."""
    crit = CRIT_PROFILE if crit is None else crit
    H=len(tau_hat); K1=len(beta); rhos=np.linspace(0.0,rho_max,J)
    A_e,c=design_matrices(dx,H,L,rhos); A_g=np.array([impulse(K1,[(1.0,r)]) for r in rhos]).T
    if groups is not None:
        B=block_matrix(groups,H); tau_hat,V_e,A_e=B@tau_hat,B@V_e@B.T,B@A_e
    We=np.linalg.cholesky(np.linalg.inv(V_e)).T; Wb=np.linalg.cholesky(np.linalg.inv(Vb)).T
    m=cp.Variable(J,nonneg=True); s=cp.Variable(len(bs)); Bmat=np.array(bs).T
    if free_kappa:
        # bilinear kappa*m is nonconvex; profile kappa over a grid instead
        best=None
        for kap in np.linspace(0.2,1.6,15):
            Q=cp.sum_squares(We@(A_e@m-tau_hat))+cp.sum_squares(Wb@(kap*(A_g@m)+Bmat@s-beta))
            p0=cp.Problem(cp.Minimize(Q))
            if solve_safe(p0) and (best is None or p0.value<best[0]): best=(p0.value,kap)
        Qmin,kap=best
        # union over kappa grid of profile intervals with Q<=Qmin+crit
        lo,hi=np.inf,-np.inf
        for kap2 in np.linspace(0.2,1.6,15):
            Q=cp.sum_squares(We@(A_e@m-tau_hat))+cp.sum_squares(Wb@(kap2*(A_g@m)+Bmat@s-beta))
            for sign in (1,-1):
                prob=cp.Problem(cp.Minimize(sign*(c@m)),[Q<=Qmin+crit])
                if solve_safe(prob):
                    v=sign*prob.value; lo=min(lo,v); hi=max(hi,v)
        return lo,hi,np.isfinite(lo),Qmin,kap
    Q=cp.sum_squares(We@(A_e@m-tau_hat))+cp.sum_squares(Wb@(A_g@m+Bmat@s-beta))
    p0=cp.Problem(cp.Minimize(Q))
    if not solve_safe(p0): return np.nan,np.nan,False,np.nan,np.nan
    Qmin=p0.value; out=[]
    for sign in (1,-1):
        prob=cp.Problem(cp.Minimize(sign*(c@m)),[Q<=Qmin+crit])
        if not solve_safe(prob): return np.nan,np.nan,False,Qmin,np.nan
        out.append(sign*prob.value)
    return out[0],out[1],True,Qmin,s.value

def e21_lead_repair(reps=25, L=104, G=100, H=13, K=52, leads_true=(0,2,4), cand=(0,1,2,3,4,6)):
    """Round-3: lead-profiled bias shapes. Panel shape family b_l for candidate leads; all s_l free."""
    dx=np.array([1.0]*4); comps=base_comps(0.4,L); g=impulse(L,comps); curve=convolve_pulse(g,dx); tau_L=curve.sum()
    groups=make_blocks(H)
    print(f"E21 lead-profiled de-confounding (candidate leads {cand}); true={tau_L:.2f}")
    for lt in leads_true:
        rng=np.random.default_rng(81); res={"single":[], "family":[]}; cov={k:0 for k in res}; n=0; Qs={"single":[], "family":[]}
        for r in range(reps):
            th,V,_=simulate_experiment(curve[:H],G=G,rng=rng)
            beta,Vb,(rho_hat,a2vd,su2)=simulate_panel(g,G_o=200,T_o=156,K=K,a=0.5,rng=rng,lead=lt)
            b0=[confound_shape_lead(K,rho_hat,a2vd,su2,0)]
            bf=[confound_shape_lead(K,rho_hat,a2vd,su2,l) for l in cand]
            o1=joint_bound2(th,V,dx,beta,Vb,b0,L,rho_from_hl(52),groups=groups)
            o2=joint_bound2(th,V,dx,beta,Vb,bf,L,rho_from_hl(52),groups=groups)
            if o1[2] and o2[2]:
                n+=1
                for k,o in (("single",o1),("family",o2)):
                    res[k].append((o[0],o[1])); cov[k]+=o[0]<=tau_L<=o[1]; Qs[k].append(o[3])
        for k in res:
            arr=np.array(res[k]); print(f"    true lead={lt} | {k:6s}: [{arr[:,0].mean():5.2f}±{arr[:,0].std():.2f},{arr[:,1].mean():5.2f}±{arr[:,1].std():.2f}] w/true={(arr[:,1]-arr[:,0]).mean()/tau_L:.2f} cov={cov[k]/n:.2f} Qmin={np.mean(Qs[k]):.0f} (n={n})")

def e22_hill_fair(reps=20, L=104, G=100, H=13, K=52):
    """Fair Hill attack: BOTH experiment and panel obey y = sum_k g_k hill(x_{t-k}); experiment effect is the
    finite difference hill(xbar+dx)-hill(xbar) at the geo operating point (xbar=3). Panel linear DL gives
    kappa*g (kappa = avg derivative). Test the free-kappa repair."""
    dx=np.array([1.0]*4); comps=base_comps(0.4,L); g=impulse(L,comps)
    hill=lambda x: 6.0*x/(x+3.0)
    dh=hill(3.0+1.0)-hill(3.0)            # per-week finite difference for a unit pulse at xbar=3
    curve=convolve_pulse(g,dx*dh); tau_L=curve.sum()
    groups=make_blocks(H); rng=np.random.default_rng(91)
    res={k:[] for k in ("exp","joint_k1","joint_freek")}; cov={k:0 for k in res}; n=0; kaps=[]
    for r in range(reps):
        th,V,_=simulate_experiment(curve[:H],G=G,rng=rng)
        beta,Vb,(rho_hat,a2vd,su2)=simulate_panel(g,G_o=200,T_o=156,K=K,a=0.5,rng=rng,hill=hill)
        b=[confound_shape(K,rho_hat,a2vd,su2)]
        # experiment side is in units of dh-scaled pulse: design uses dx*dh so that m is per-unit-of-hill-output
        o_e=joint_bound2(th,V,dx*dh,beta,Vb,b,L,rho_from_hl(52),groups=groups) if False else None
        # simpler: express everything per unit of experiment pulse (dx); panel scale kappa absorbs dh and derivative
        oe=tail_bounds_profile(th,V,dx,L,rho_from_hl(52),groups=groups)
        o1=joint_bound2(th,V,dx,beta,Vb,b,L,rho_from_hl(52),groups=groups)
        o2=joint_bound2(th,V,dx,beta,Vb,b,L,rho_from_hl(52),groups=groups,free_kappa=True)
        if oe[2] and o1[2] and o2[2]:
            n+=1
            for k,o in (("exp",oe),("joint_k1",o1),("joint_freek",o2)):
                res[k].append((o[0],o[1])); cov[k]+=o[0]<=tau_L<=o[1]
            kaps.append(o2[4])
    print(f"E22 fair Hill (experiment finite-difference truth={tau_L:.2f}; linear-g total would be {convolve_pulse(g,dx).sum():.2f}); kappa_hat={np.mean(kaps):.2f}±{np.std(kaps):.2f} (expected ~ E[hill'(x)]/dh ≈ {0.5/dh:.2f}); n={n}")
    for k in res:
        arr=np.array(res[k]); print(f"    {k:12s}: [{arr[:,0].mean():5.2f}±{arr[:,0].std():.2f},{arr[:,1].mean():5.2f}±{arr[:,1].std():.2f}] w/true={(arr[:,1]-arr[:,0]).mean()/tau_L:.2f} cov={cov[k]/n:.2f}")

def e23_joint_value_vs_experiment_size(reps=20, L=104):
    dx=np.array([1.0]*4); comps=base_comps(0.4,L); g=impulse(L,comps); curve=convolve_pulse(g,dx); tau_L=curve.sum()
    print(f"E23 value of the panel vs experiment size (true={tau_L:.2f}):")
    for (G,H) in [(100,13),(400,26),(1600,52)]:
        groups=make_blocks(H); rng=np.random.default_rng(111); res={"exp":[],"joint":[]}; cov={"exp":0,"joint":0}; n=0
        for r in range(reps):
            th,V,_=simulate_experiment(curve[:H],G=G,rng=rng)
            beta,Vb,(rho_hat,a2vd,su2)=simulate_panel(g,G_o=200,T_o=156,K=52,a=0.5,rng=rng)
            b=[confound_shape(52,rho_hat,a2vd,su2)]
            oe=tail_bounds_profile(th,V,dx,L,rho_from_hl(52),groups=groups); oj=joint_bound2(th,V,dx,beta,Vb,b,L,rho_from_hl(52),groups=groups)
            if oe[2] and oj[2]:
                n+=1
                for k,o in (("exp",oe),("joint",oj)): res[k].append((o[0],o[1])); cov[k]+=o[0]<=tau_L<=o[1]
        for k in res:
            arr=np.array(res[k]); print(f"    G={G:4d} H={H:2d} {k:5s}: [{arr[:,0].mean():5.2f},{arr[:,1].mean():5.2f}] w/true={(arr[:,1]-arr[:,0]).mean()/tau_L:.2f} cov={cov[k]/n:.2f}")

def e24_no_and_strong_confounding(reps=20, L=104):
    for a in (0.0, 1.5):
        try: e12_panel_deconfounding(reps=reps, a=a, label=f" a={a}")
        except Exception as ex: print("   solver failure", ex)



def confound_shape_within(K, rho, a2vd, su2, T_o, lead=0):
    """Exact finite-T within-transform bias shape: b = E[X~'X~]^{-1} E[X~'d~] for lag columns k=0..K,
    each column demeaned over the T_o sample rows. d AR(1)(rho), x_t = a d_{t+lead} + u_t."""
    N=T_o+K+lead; idx=np.arange(N)
    Cd=rho**np.abs(idx[:,None]-idx[None,:])          # unit-variance AR(1) correlation
    M=np.eye(T_o)-np.ones((T_o,T_o))/T_o
    # selection: column k picks rows (K+lead) - k + t  for t=0..T_o-1 (of x, whose d index is +lead)
    S=[]
    for k in range(K+1):
        s=np.zeros((T_o,N)); s[np.arange(T_o), K+lead-k+np.arange(T_o)+lead]=1; S.append(s)   # d index of x_{t-k} is t-k+lead
    Sd=np.zeros((T_o,N)); Sd[np.arange(T_o), K+lead+np.arange(T_o)]=1                          # d_t itself
    XX=np.zeros((K+1,K+1)); Xd=np.zeros(K+1)
    for k in range(K+1):
        Xd[k]=np.trace(M@S[k]@Cd@Sd.T)/T_o
        for j in range(K+1):
            XX[k,j]=np.trace(M@S[k]@Cd@S[j].T)/T_o
    # u part: x_{t-k} u-columns share u when k=j; demeaning: E[u~_k' u~_j] = su2*(delta_kj*(T_o-1)/T_o - overlap terms ~ -1/T_o)
    UU=np.zeros((K+1,K+1))
    for k in range(K+1):
        for j in range(K+1):
            ov = T_o-abs(k-j)   # number of shared u's between the two columns (shifted)
            UU[k,j] = (1.0 if k==j else 0.0) - ov/T_o**2 if k!=j else (1.0 - 1.0/T_o)
    # (off-diagonal: column k and j share u values at shifted positions: E[u~_{k,t} u~_{j,t}] = -ov/T_o^2 summed over t gives -ov/T_o... use per-row average)
    UU=np.where(np.eye(K+1)==1, 1.0-1.0/T_o, -(T_o-np.abs(np.subtract.outer(np.arange(K+1),np.arange(K+1))))/T_o**2)
    G=a2vd*XX+su2*UU
    return np.linalg.solve(G, Xd)        # per unit of a*v_d (scalar s absorbs a v_d)

def e25_within_shape(reps=60, L=104, G=400, H=26):
    dx=np.array([1.0]*4); comps=base_comps(0.4,L); g=impulse(L,comps); curve=convolve_pulse(g,dx); tau_L=curve.sum()
    groups=make_blocks(H); rng=np.random.default_rng(222); rows=[]
    b_inf=confound_shape(52,0.6,0.25/(1-0.36),1.0); b_w=confound_shape_within(52,0.6,0.25/(1-0.36),1.0,156)
    print("E25 within-transform shape vs infinite-T shape: b_k (k=0,1,2,5,10,20,40,52):")
    for k in (0,1,2,5,10,20,40,52): print(f"    k={k:2d}: b_inf={b_inf[k]:+.4f}  b_within={b_w[k]:+.4f}")
    print(f"    sum b_inf={b_inf.sum():.4f}  sum b_within={b_w.sum():.4f}")
    for r in range(reps):
        th,V,_=simulate_experiment(curve[:H],G=G,rng=rng)
        beta,Vb,(rho_hat,a2vd,su2)=simulate_panel(g,G_o=200,T_o=156,K=52,a=0.5,rng=rng)
        b1=[confound_shape(52,rho_hat,a2vd,su2)]; b2=[confound_shape_within(52,rho_hat,a2vd,su2,156)]
        o1=joint_bound2(th,V,dx,beta,Vb,b1,L,rho_from_hl(52),groups=groups)
        o2=joint_bound2(th,V,dx,beta,Vb,b2,L,rho_from_hl(52),groups=groups)
        rows.append((o1[0],o1[1],o1[3],o2[0],o2[1],o2[3]))
    R=np.array(rows)
    for name,i in (("AR(1) shape",0),("within shape",3)):
        lo,hi,Q=R[:,i],R[:,i+1],R[:,i+2]
        print(f"    {name}: [{lo.mean():.2f},{hi.mean():.2f}] w/true={(hi-lo).mean()/tau_L:.2f} cov={((lo<=tau_L)&(tau_L<=hi)).mean():.2f} hi-fail={(hi<tau_L).mean():.2f} Qmin={Q.mean():.1f}")


def e7b_width_law_refined(L=104):
    """Refined price-of-the-long-run law: hi - Theta_true vs the slow-direction prediction
    m_max*c_slow with m_max = sqrt(crit / a'V^{-1}a)  (a = in-window signature of the rho_max component)."""
    dx=np.array([1.0]*4); comps=base_comps(0.4,L); curve=convolve_pulse(impulse(L,comps),dx); tau_L=curve.sum()
    print("E7b refined width law (noise-free tau_hat, true V from 1500 MC draws):")
    for (G,H,hl) in [(100,13,52),(400,13,52),(1600,13,52),(400,26,52),(400,52,52),(400,26,26),(400,26,104)]:
        rng=np.random.default_rng(3); ths=np.array([simulate_experiment(curve[:H],G=G,rng=rng)[0] for _ in range(1500)]); Vt=np.cov(ths,rowvar=False)
        groups=make_blocks(H); B=block_matrix(groups,H); Vb=B@Vt@B.T
        lo,hi,ok,Q=tail_bounds_profile(curve[:H],Vt,dx,L,rho_from_hl(hl),groups=groups)
        rho=rho_from_hl(hl); Leff=(1-rho**L)/(1-rho)
        rhos=np.linspace(0,rho,200); A,c=design_matrices(dx,H,L,rhos); a=B@A[:,-1]; ctot=c[-1]
        m_max=np.sqrt(CRIT_PROFILE/(a@np.linalg.solve(Vb,a)))
        one=np.ones(len(groups)); se_gls=1/np.sqrt(one@np.linalg.solve(Vb,one))
        print(f"    G={G:4d} H={H:2d} hl={hl:3d}: hi-true={hi-tau_L:5.2f} true-lo={tau_L-lo:4.2f} | slow-dir m_max*c={m_max*ctot:5.2f} (ratio {(hi-tau_L)/(m_max*ctot):.2f}) | naive z*se_gls*Leff={2.575*se_gls*Leff:5.2f}")

def e15b_boundary_vs_dictionary(L=104, G=400, H=26, reps=300):
    """Low-side failure of the chi2_1 profile radius grows with dictionary richness (phi=0 truth)."""
    dx=np.array([1.0]*4); curve=convolve_pulse(impulse(L,[(1.0,0.5)]),dx); tau_L=curve.sum()
    rng=np.random.default_rng(7); ths=np.array([simulate_experiment(curve[:H],G=G,rng=rng)[0] for _ in range(2000)]); Vt=np.cov(ths,rowvar=False)
    groups=make_blocks(H)
    print("E15b low-side failure at crit=3.84 (true V, phi=0) vs hl_max of the dictionary:")
    for hl in (1,3,8,52):
        rng=np.random.default_rng(21); lf=0;n=0; zs=[]
        for r in range(reps):
            th,V,_=simulate_experiment(curve[:H],G=G,rng=rng)
            lo,hi,ok,Q=tail_bounds_profile(th,Vt,dx,L,rho_from_hl(hl),groups=groups,crit=3.84)
            if ok: n+=1; lf+=lo>tau_L
            zs.append((th.sum()-curve[:H].sum())/np.sqrt(Vt.sum()))
        zs=np.array(zs)
        print(f"    hl_max={hl:2d}: lo-fail {lf/n:.3f}   (unconstrained one-sided sum test z>1.96: {(zs>1.96).mean():.3f})")

def e16b_delayed_onset_counterexample(L=104, G=1600, reps=40):
    """Boundary of the CM assumption: brand component whose onset lies beyond the window."""
    dx=np.array([1.0]*4); act_total=2.0; rho_B=rho_from_hl(26)
    print("E16b counterexample: delayed-onset brand component (G=1600), geometric dictionary, brand share 0.4")
    for lag in (12,20,30):
        unit=impulse(L,[(1.0,rho_B,lag,"geom")]).sum(); w_b=0.4/0.6*act_total/unit
        comps=[(1.0,0.5),(w_b,rho_B,lag,"geom")]; curve=convolve_pulse(impulse(L,comps),dx); tau_L=curve.sum()
        for H in (26,52):
            groups=make_blocks(H); rng=np.random.default_rng(31); los=[];his=[];cov=0;n=0;Qs=[]
            for r in range(reps):
                th,V,_=simulate_experiment(curve[:H],G=G,rng=rng)
                lo,hi,ok,Q=tail_bounds_profile(th,V,dx,L,rho_from_hl(52),groups=groups,J=120)
                if ok: n+=1;los.append(lo);his.append(hi);cov+=lo<=tau_L<=hi;Qs.append(Q)
            print(f"    lag={lag:2d} H={H}: true={tau_L:.2f} [{np.mean(los):.2f},{np.mean(his):.2f}] cov={cov/n:.2f} Qmin={np.mean(Qs):.1f} (k={len(groups)}) in-window mass={curve[:H].sum():.2f}")

def e26_joint_coverage_audit(L=104, G=400, H=26, reps=60):
    """Attack on construction 3: high-side failures of the joint bound; AR(1) vs within shape; crit."""
    dx=np.array([1.0]*4); comps=base_comps(0.4,L); g=impulse(L,comps); curve=convolve_pulse(g,dx); tau_L=curve.sum()
    groups=make_blocks(H); rng=np.random.default_rng(222); rows=[]
    for r in range(reps):
        th,V,_=simulate_experiment(curve[:H],G=G,rng=rng)
        beta,Vb,(rho_hat,a2vd,su2)=simulate_panel(g,G_o=200,T_o=156,K=52,a=0.5,rng=rng)
        b_est=[confound_shape(52,rho_hat,a2vd,su2)]; b_true=[confound_shape(52,0.6,0.25/(1-0.36),1.0)]; b_w=[confound_shape_within(52,rho_hat,a2vd,su2,156)]
        o1=joint_bound2(th,V,dx,beta,Vb,b_est,L,rho_from_hl(52),groups=groups)
        o2=joint_bound2(th,V,dx,beta,Vb,b_true,L,rho_from_hl(52),groups=groups)
        o3=joint_bound2(th,V,dx,beta,Vb,b_est,L,rho_from_hl(52),groups=groups,crit=9.0)
        o4=joint_bound2(th,V,dx,beta,Vb,b_w,L,rho_from_hl(52),groups=groups)
        rows.append((o1[0],o1[1],o2[0],o2[1],o3[0],o3[1],o4[0],o4[1],o1[3]))
    R=np.array(rows)
    print(f"E26 joint-bound coverage audit (G={G},H={H}, true={tau_L:.2f}, Qmin={R[:,8].mean():.1f}):")
    for name,i in (("AR(1) shape, est, crit 6.63",0),("AR(1) shape, TRUE params",2),("AR(1) shape, est, crit 9.0",4),("within-T shape, est, crit 6.63",6)):
        lo,hi=R[:,i],R[:,i+1]; print(f"    {name:32s}: [{lo.mean():.2f},{hi.mean():.2f}] w/true={(hi-lo).mean()/tau_L:.2f} cov={((lo<=tau_L)&(tau_L<=hi)).mean():.2f} hi-fail={(hi<tau_L).mean():.2f}")

def e27_panel_qmin_source(L=104, G=400, H=26, reps=12):
    """Why is Qmin ~80 vs dof ~63? cluster-robust small-sample bias (p/G_o) vs shape: vary panel size."""
    dx=np.array([1.0]*4); comps=base_comps(0.4,L); g=impulse(L,comps); curve=convolve_pulse(g,dx); tau_L=curve.sum()
    groups=make_blocks(H)
    print("E27 Qmin of the joint fit vs panel size (within shape):")
    for G_o,T_o in ((200,156),(800,156),(200,400)):
        rng=np.random.default_rng(333); Qs=[]; cov=0; n=0
        for r in range(reps):
            th,V,_=simulate_experiment(curve[:H],G=G,rng=rng)
            beta,Vb,(rho_hat,a2vd,su2)=simulate_panel(g,G_o=G_o,T_o=T_o,K=52,a=0.5,rng=rng)
            b=[confound_shape_within(52,rho_hat,a2vd,su2,T_o)]
            o=joint_bound2(th,V,dx,beta,Vb,b,L,rho_from_hl(52),groups=groups)
            Qs.append(o[3]); cov+=o[0]<=tau_L<=o[1]; n+=1
        print(f"    G_o={G_o} T_o={T_o}: Qmin={np.mean(Qs):.1f}±{np.std(Qs):.1f} (dof≈63) cov={cov/n:.2f}")

QUICK = ["e1_si_bias_closed_form", "e2_crossing_horizon", "e3_tail_bound_noisefree", "e9_profile_bound", "e13_power_control", "e19_sensitivity_curve", "e12_panel_deconfounding", "e21_lead_repair"]

if __name__ == "__main__":
    which = sys.argv[1:] or ["quick"]
    t0 = time.time()
    fns = {k: v for k, v in globals().items() if k.startswith("e") and callable(v)}
    if which == ["quick"]:
        for name in QUICK: fns[name]()
    else:
        for w in which:
            for k, f in list(fns.items()):
                if k == w or k.startswith(w + "_"): f()
    print(f"[{time.time()-t0:.1f}s]")
