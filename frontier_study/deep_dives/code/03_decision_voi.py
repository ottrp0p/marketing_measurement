"""Deep dive 03 - Decision-calibrated uncertainty & value of information (M10).

Verifies:
  E1: joint posterior over 3-channel response curves from an observational fit
      (Laplace/FIM), with cross-channel correlation.
  E2: Bayes vs plug-in allocation; exact (MC) EVPI vs quadratic formula
      EVPI ~= 1/2 tr(Omega D Sigma D^T).
  E3: the decision-null direction: common multiplicative beta uncertainty is
      free under a fixed budget; contrast uncertainty is not; the null breaks
      when total budget is also chosen.
  E4: closed-form rank-1 EVSI vs nested-Monte-Carlo EVSI for candidate
      geo-experiment designs (operator forward map from dive 02).
  E5: design ranking, decision-aligned EVSI vs D-optimality (info gain).
  E6: greedy EVSI experiment schedule (feeds A5).

Model per dive 01/02: y_t = sum_j beta_j h(a_jt;K_j,S_j) + eps,
a_jt = x_jt + alpha_j a_j,t-1, h(a)=a^S/(a^S+K^S), sigma=0.05.
"""
import numpy as np
from scipy.optimize import minimize

np.seterr(all="ignore")
rng = np.random.default_rng(11)
SIGMA = 0.05
# channels: (beta, K, S, alpha), baseline spends
TH0 = np.array([[1.0, 1.5, 2.0, 0.6],
                [0.8, 1.0, 1.8, 0.3],
                [1.3, 2.5, 2.2, 0.8]])
XBAR = np.array([1.0, 0.8, 1.2])
J = 3
B_TOT = XBAR.sum()
T_OBS = 104


def hill(a, K, S):
    a = np.maximum(a, 1e-12)
    return a**S / (a**S + K**S)


def hill_d1(a, K, S):
    h = hill(a, K, S)
    return S * h * (1 - h) / a


def adstock(x, alpha, a_init):
    a = np.empty_like(x, dtype=float)
    prev = a_init
    for t in range(len(x)):
        prev = x[t] + alpha * prev
        a[t] = prev
    return a


# ---------- steady-state profit pieces (fixed total budget => maximize revenue)
def rev_ss(b, th):
    """Steady-state per-period revenue for constant spends b, params th (J x 4)."""
    a = b / (1 - th[:, 3])
    return float(np.sum(th[:, 0] * hill(a, th[:, 1], th[:, 2])))


def rev_ss_batch(b, ths):
    """ths: (N,J,4) -> (N,) revenues."""
    a = b[None, :] / (1 - ths[:, :, 3])
    return np.sum(ths[:, :, 0] * hill(a, ths[:, :, 1], ths[:, :, 2]), axis=1)


def mroas(b, th):
    """Marginal ROAS per channel at spends b (vector, length J)."""
    a = b / (1 - th[:, 3])
    return th[:, 0] * hill_d1(a, th[:, 1], th[:, 2]) / (1 - th[:, 3])


# ---------- simplex optimizer (2 free vars)
def opt_simplex(obj, b_start=None):
    """Maximize obj(b) over b>=0, sum b = B_TOT. Returns b."""
    x0 = (b_start if b_start is not None else XBAR)[:2]
    best, bestv = None, -np.inf
    for x_init in [x0, np.array([0.6, 0.6]), np.array([1.4, 0.9])]:
        res = minimize(lambda x: -obj(np.array([x[0], x[1], B_TOT - x[0] - x[1]])),
                       x_init, method="Nelder-Mead",
                       options=dict(xatol=1e-6, fatol=1e-10, maxiter=2000))
        if -res.fun > bestv:
            bestv, best = -res.fun, res.x
    b = np.array([best[0], best[1], B_TOT - best[0] - best[1]])
    return np.maximum(b, 0.0)


# ---------- E1: posterior from observational fit (Laplace) --------------------
def flat(th):
    return th.reshape(-1)


def unflat(v):
    return v.reshape(J, 4)


def obs_mean_path(v, X):
    th = unflat(v)
    T = X.shape[0]
    out = np.zeros(T)
    for j in range(J):
        a = adstock(X[:, j], th[j, 3], XBAR[j] / (1 - th[j, 3]))
        out += th[j, 0] * hill(a, th[j, 1], th[j, 2])
    return out


def build_posterior():
    # correlated AR(1) spends, +-10 % swings around XBAR
    ar, s_in = 0.8, 0.06
    Z = np.zeros((T_OBS, J))
    common = np.zeros(T_OBS)
    e = rng.standard_normal((T_OBS, J))
    ec = rng.standard_normal(T_OBS)
    for t in range(1, T_OBS):
        common[t] = ar * common[t - 1] + s_in * ec[t]
        Z[t] = ar * Z[t - 1] + s_in * e[t]
    mix = 0.7 * common[:, None] + 0.7 * Z          # shared demand cycle -> collinearity
    X = XBAR[None, :] * np.clip(1 + mix, 0.7, 1.3)
    v0 = flat(TH0)
    # numerical Jacobian of mean path
    Jac = np.zeros((T_OBS, 12))
    for k in range(12):
        d = 1e-5 * max(abs(v0[k]), 1e-3)
        vp, vm = v0.copy(), v0.copy()
        vp[k] += d
        vm[k] -= d
        Jac[:, k] = (obs_mean_path(vp, X) - obs_mean_path(vm, X)) / (2 * d)
    F = Jac.T @ Jac / SIGMA**2
    # weak prior ridge: sd ~ (50% of param scale)
    prior_prec = np.diag(1.0 / (0.5 * np.abs(v0)) ** 2)
    Sig = np.linalg.inv(F + prior_prec)
    return Sig, X


def sample_post(Sig, n, mean=None):
    m = flat(TH0) if mean is None else mean
    L = np.linalg.cholesky(Sig + 1e-12 * np.eye(12))
    out = np.empty((n, 12))
    filled = 0
    while filled < n:
        cand = m[None, :] + rng.standard_normal((2 * n, 12)) @ L.T
        th = cand.reshape(-1, J, 4)
        ok = (np.all(th[:, :, 0] > 0.05, axis=1) & np.all(th[:, :, 1] > 0.1, axis=1)
              & np.all(th[:, :, 2] > 0.3, axis=1)
              & np.all(th[:, :, 3] > 0.0, axis=1) & np.all(th[:, :, 3] < 0.95, axis=1))
        good = cand[ok]
        take = min(len(good), n - filled)
        out[filled:filled + take] = good[:take]
        filled += take
    return out


# ---------- decision-geometry objects ----------------------------------------
def decision_objects(Sig):
    """Bayes allocation b_bar (approx: plug-in at mean, refined), H, D, Omega."""
    b_bar = opt_simplex(lambda b: rev_ss(b, TH0))
    # H_j = -d m_j / d b_j  (numerical)
    H = np.zeros(J)
    for j in range(J):
        d = 1e-5
        bp, bm = b_bar.copy(), b_bar.copy()
        bp[j] += d
        bm[j] -= d
        H[j] = -(mroas(bp, TH0)[j] - mroas(bm, TH0)[j]) / (2 * d)
    # D = d m / d theta  (J x 12)
    v0 = flat(TH0)
    D = np.zeros((J, 12))
    for k in range(12):
        d = 1e-5 * max(abs(v0[k]), 1e-3)
        vp, vm = v0.copy(), v0.copy()
        vp[k] += d
        vm[k] -= d
        D[:, k] = (mroas(b_bar, unflat(vp)) - mroas(b_bar, unflat(vm))) / (2 * d)
    Hinv = 1.0 / H
    Omega = np.diag(Hinv) - np.outer(Hinv, Hinv) / Hinv.sum()
    return b_bar, H, D, Omega


def evpi_quad(Sig, D, Omega):
    return 0.5 * float(np.trace(Omega @ D @ Sig @ D.T))


# ---------- E2: exact EVPI + Bayes vs plug-in ---------------------------------
def exp_E2(Sig, n_outer=400, n_bayes=4000):
    print("=" * 72)
    print("E2: Bayes vs plug-in allocation; MC EVPI vs quadratic formula")
    ths = sample_post(Sig, n_bayes).reshape(-1, J, 4)
    b_plug = opt_simplex(lambda b: rev_ss(b, TH0))
    b_bayes = opt_simplex(lambda b: float(np.mean(rev_ss_batch(b, ths))), b_plug)
    ev_plug = float(np.mean(rev_ss_batch(b_plug, ths)))
    ev_bayes = float(np.mean(rev_ss_batch(b_bayes, ths)))
    # exact EVPI: per-draw oracle optimization
    ths_o = sample_post(Sig, n_outer).reshape(-1, J, 4)
    oracle = np.empty(n_outer)
    held = np.empty(n_outer)
    for i, th in enumerate(ths_o):
        bo = opt_simplex(lambda b: rev_ss(b, th), b_bayes)
        oracle[i] = rev_ss(bo, th)
        held[i] = rev_ss(b_bayes, th)
    evpi_mc = float(np.mean(oracle - held))
    se = float(np.std(oracle - held) / np.sqrt(n_outer))
    b_bar, H, D, Omega = decision_objects(Sig)
    evpi_q = evpi_quad(Sig, D, Omega)
    print(f"  b_plug  = {np.round(b_plug,4)}   E rev = {ev_plug:.5f}")
    print(f"  b_bayes = {np.round(b_bayes,4)}   E rev = {ev_bayes:.5f}")
    print(f"  Bayes - plug-in expected-revenue gap = {ev_bayes-ev_plug:.2e}")
    print(f"  EVPI (MC, oracle)   = {evpi_mc:.5f}  (+- {se:.5f})")
    print(f"  EVPI (quadratic tr) = {evpi_q:.5f}")
    print(f"  baseline revenue at b_bayes = {ev_bayes:.4f}; EVPI/rev = {evpi_mc/ev_bayes*100:.2f}%")
    return b_bayes, evpi_mc, evpi_q




def exp_E2b(Sig, n_outer=300):
    print("=" * 72)
    print("E2b: EVPI quadratic formula vs MC as posterior width shrinks")
    for s in (1.0, 0.5, 0.25):
        Sig_s = s**2 * Sig
        b_bar, H, D, Omega = decision_objects(Sig_s)
        q = evpi_quad(Sig_s, D, Omega)
        ths_o = sample_post(Sig_s, n_outer).reshape(-1, J, 4)
        dif = []
        for th in ths_o:
            bo = opt_simplex(lambda b: rev_ss(b, th), b_bar)
            dif.append(rev_ss(bo, th) - rev_ss(b_bar, th))
        mc, se_mc = float(np.mean(dif)), float(np.std(dif)/np.sqrt(n_outer))
        print(f"  scale {s:.2f}: EVPI_MC = {mc:.5f} (+-{se_mc:.5f})  quad = {q:.5f}"
              f"  ratio MC/quad = {mc/q:.2f}")

# ---------- E3: decision-null direction ---------------------------------------
def exp_E3(Sig, s_sd=0.3, n=300):
    print("=" * 72)
    print("E3: common multiplicative-beta error is decision-null (fixed budget)")
    b_bar = opt_simplex(lambda b: rev_ss(b, TH0))
    ths = sample_post(Sig, n).reshape(-1, J, 4)
    s_draw = rng.normal(0, s_sd, n)

    def evpi_mc(mod):
        dif = []
        for i, th in enumerate(ths):
            th2 = th.copy()
            mod(th2, s_draw[i])
            bo = opt_simplex(lambda b: rev_ss(b, th2), b_bar)
            dif.append(rev_ss(bo, th2) - rev_ss(b_bar, th2))
        return float(np.mean(dif)), float(np.std(dif) / np.sqrt(n))

    def none_mod(th2, s):
        pass

    def common_mod(th2, s):
        th2[:, 0] *= (1 + s)

    def contrast_mod(th2, s):
        th2[0, 0] *= (1 + s)
        th2[2, 0] *= (1 - s)

    for name, mod in [("baseline (no shock)", none_mod),
                      ("+ common beta x(1+s)", common_mod),
                      ("+ contrast beta 1up/3dn", contrast_mod)]:
        m, se = evpi_mc(mod)
        print(f"  {name:26s}: EVPI_MC = {m:.5f} (+-{se:.5f})")
    # free total budget: the null must break
    v_m = 1.0 / mroas(b_bar, TH0)[0]

    def profit(b, th):
        return v_m * rev_ss(b, th) - float(np.sum(b))

    def opt_free(th, b_start):
        res = minimize(lambda x: -profit(np.abs(x), th), b_start, method="Nelder-Mead",
                       options=dict(xatol=1e-6, fatol=1e-10, maxiter=3000))
        return np.abs(res.x)

    b0 = opt_free(TH0, XBAR.copy())
    for name, mod in [("baseline (no shock)", none_mod),
                      ("+ common beta x(1+s)", common_mod)]:
        dif = []
        for i, th in enumerate(ths):
            th2 = th.copy()
            mod(th2, s_draw[i])
            bo = opt_free(th2, b0.copy())
            dif.append(profit(bo, th2) - profit(b0, th2))
        print(f"  FREE-budget {name:26s}: EVPI_MC = {np.mean(dif):.5f}")


# ---------- experiment operator (dive 02) ------------------------------------
def exp_lift(th_j, xbar_e, delta, Te, P):
    W = Te + P
    x0 = np.full(W, xbar_e)
    x1 = x0.copy()
    x1[:Te] += delta
    b, K, S, al = th_j
    ai = xbar_e / (1 - al)
    a0 = adstock(x0, al, ai)
    a1 = adstock(x1, al, ai)
    return float(b * np.sum(hill(a1, K, S) - hill(a0, K, S)))


def lift_grad(j, xbar_e, delta, Te, P):
    """Gradient of lift wrt full 12-dim theta (nonzero only in channel j block)."""
    g = np.zeros(12)
    for k in range(4):
        d = 1e-5 * max(abs(TH0[j, k]), 1e-3)
        tp, tm = TH0[j].copy(), TH0[j].copy()
        tp[k] += d
        tm[k] -= d
        g[4 * j + k] = (exp_lift(tp, xbar_e, delta, Te, P)
                        - exp_lift(tm, xbar_e, delta, Te, P)) / (2 * d)
    return g


def se_of_design(Te, P):
    return 0.03 * np.sqrt(Te + P)   # noise accumulates over measurement window


def evsi_closed(Sig, D, Omega, g, se):
    DS = D @ Sig
    num = float(g @ DS.T @ Omega @ DS @ g)
    return 0.5 * num / (se**2 + float(g @ Sig @ g))


def dscore(Sig, g, se):
    return 0.5 * np.log1p(float(g @ Sig @ g) / se**2)


# ---------- E4: closed-form EVSI vs nested MC ---------------------------------
def evsi_nested_mc(Sig, design, ths, outer_idx, outer_noise, b_prior):
    """Split-sample nested MC: optimize on even half, evaluate on odd half (CRN)."""
    j, dl, Te, P = design
    se = se_of_design(Te, P)
    lifts = np.array([exp_lift(unflat(v)[j], XBAR[j], dl, Te, P) for v in ths])
    thJ = ths.reshape(-1, J, 4)
    n = len(ths)
    ev_idx = np.arange(0, n, 2)
    od_idx = np.arange(1, n, 2)
    vals = []
    for i, z in zip(outer_idx, outer_noise):
        L = lifts[i] + se * z
        w = np.exp(-0.5 * ((lifts - L) / se) ** 2)
        we = w[ev_idx] / w[ev_idx].sum()
        wo = w[od_idx] / w[od_idx].sum()
        b_post = opt_simplex(lambda b: float(we @ rev_ss_batch(b, thJ[ev_idx])), b_prior)
        vals.append(float(wo @ rev_ss_batch(b_post, thJ[od_idx]))
                    - float(wo @ rev_ss_batch(b_prior, thJ[od_idx])))
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals)))


def exp_E4(Sig, b_bayes, label="full-width", n_inner=4000, n_outer=200):
    b_bar, H, D, Omega = decision_objects(Sig)
    print("=" * 72)
    print(f"E4 ({label}): closed-form rank-1 EVSI vs nested-MC EVSI (split-sample, CRN)")
    designs = [(0, 0.5, 8, 0), (1, 0.4, 8, 0), (2, 0.6, 8, 0),
               (2, 0.6, 8, 8), (2, 1.2, 4, 0), (0, 1.0, 4, 8),
               (1, 0.8, 13, 0), (2, 0.3, 13, 8)]
    ths = sample_post(Sig, n_inner)
    thJ = ths.reshape(-1, J, 4)
    b_pr = opt_simplex(lambda b: float(np.mean(rev_ss_batch(b, thJ))), b_bayes)
    outer_idx = rng.integers(0, n_inner, n_outer)
    outer_noise = rng.standard_normal(n_outer)
    print(f"  {'design (j, dl, Te, P)':26s} {'EVSI_closed':>11s} {'EVSI_MC':>9s} {'+-':>7s}")
    pairs = []
    for dsn in designs:
        j, dl, Te, P = dsn
        se = se_of_design(Te, P)
        g = lift_grad(j, XBAR[j], dl, Te, P)
        ec = evsi_closed(Sig, D, Omega, g, se)
        em, ese = evsi_nested_mc(Sig, dsn, ths, outer_idx, outer_noise, b_pr)
        pairs.append((ec, em))
        print(f"  ch{j} dl={dl:.2f} Te={Te:2d} P={P:2d}     {ec:11.5f} {em:9.5f} {ese:7.5f}")
    ecs, ems = np.array(pairs).T
    from scipy.stats import spearmanr
    print(f"  pearson corr = {np.corrcoef(ecs, ems)[0,1]:.3f}   "
          f"spearman rank corr = {spearmanr(ecs, ems).statistic:.3f}")
    print(f"  median level ratio MC/closed = {np.median(ems/np.maximum(ecs,1e-12)):.2f}")
    return pairs


# ---------- E5: full design grid, EVSI vs D-optimality ------------------------
def exp_E5(Sig, D, Omega, evpi_ref):
    print("=" * 72)
    print("E5: design grid - decision-aligned EVSI vs D-optimality ranking")
    grid = []
    for j in range(J):
        for frac in (0.25, 0.5, 1.0):
            for Te in (4, 8):
                for P in (0, 8):
                    dl = frac * XBAR[j]
                    se = se_of_design(Te, P)
                    g = lift_grad(j, XBAR[j], dl, Te, P)
                    grid.append((j, frac, Te, P, evsi_closed(Sig, D, Omega, g, se),
                                 dscore(Sig, g, se)))
    grid = sorted(grid, key=lambda r: -r[4])
    print("  top-5 by EVSI:")
    for r in grid[:5]:
        print(f"    ch{r[0]} frac={r[1]:.2f} Te={r[2]} P={r[3]}: EVSI={r[4]:.5f} Dscore={r[5]:.3f}")
    gd = sorted(grid, key=lambda r: -r[5])
    print("  top-5 by D-score (info gain):")
    for r in gd[:5]:
        print(f"    ch{r[0]} frac={r[1]:.2f} Te={r[2]} P={r[3]}: EVSI={r[4]:.5f} Dscore={r[5]:.3f}")
    ev = np.array([r[4] for r in grid])
    ds = np.array([r[5] for r in grid])
    from scipy.stats import kendalltau
    tau = kendalltau(ev, ds).statistic
    best_ev = grid[0][4]
    ev_of_dbest = gd[0][4]
    print(f"  Kendall tau(EVSI, D-score) = {tau:.3f}")
    print(f"  EVSI of EVSI-best design   = {best_ev:.5f}  ({best_ev/evpi_ref*100:.0f}% of EVPI)")
    print(f"  EVSI of D-best design      = {ev_of_dbest:.5f}  "
          f"(value lost by D-optimality: {(1-ev_of_dbest/best_ev)*100:.0f}%)")
    return grid


# ---------- E6: greedy schedule ------------------------------------------------
def exp_E6(Sig, evpi0):
    print("=" * 72)
    print("E6: greedy EVSI experiment schedule (rank-1 posterior updates)")
    Sig_c = Sig.copy()
    for step in range(5):
        b_bar, H, D, Omega = decision_objects(Sig_c)
        best, bg, bse = None, None, None
        for j in range(J):
            for frac in (0.25, 0.5, 1.0):
                for Te in (4, 8):
                    for P in (0, 8):
                        dl = frac * XBAR[j]
                        se = se_of_design(Te, P)
                        g = lift_grad(j, XBAR[j], dl, Te, P)
                        e = evsi_closed(Sig_c, D, Omega, g, se)
                        if best is None or e > best[4]:
                            best, bg, bse = (j, frac, Te, P, e), g, se
        Sig_c = Sig_c - np.outer(Sig_c @ bg, bg @ Sig_c) / (bse**2 + bg @ Sig_c @ bg)
        rem = evpi_quad(Sig_c, D, Omega)
        print(f"  step {step+1}: run ch{best[0]} frac={best[1]:.2f} Te={best[2]} P={best[3]} "
          f"-> EVSI {best[4]:.5f}; residual EVPI {rem:.5f} ({rem/evpi0*100:.0f}%)")




# ---------- E7: exact nested-MC duels and controls ----------------------------
def _duel_vals(Sig_s, design, ths, thJ, b_pr, oi, on, se_mult=1.0):
    j, dl, Te, P = design
    se = se_of_design(Te, P) * se_mult
    lifts = np.array([exp_lift(unflat(v)[j], XBAR[j], dl, Te, P) for v in ths])
    n = len(ths)
    e_idx = np.arange(0, n, 2); o_idx = np.arange(1, n, 2)
    out = np.empty(len(oi))
    for t, (i, z) in enumerate(zip(oi, on)):
        L = lifts[i] + se * z
        w = np.exp(-0.5 * ((lifts - L) / se) ** 2)
        we = w[e_idx] / w[e_idx].sum(); wo = w[o_idx] / w[o_idx].sum()
        b_post = opt_simplex(lambda b: float(we @ rev_ss_batch(b, thJ[e_idx])), b_pr)
        out[t] = (float(wo @ rev_ss_batch(b_post, thJ[o_idx]))
                  - float(wo @ rev_ss_batch(b_pr, thJ[o_idx])))
    return out


def exp_E7(Sig, scale=1.0, label="full width", n_inner=4000, n_outer=600,
           duel=None, controls=()):
    """duel: (designA, designB) in absolute-delta units; controls: list of
    (name, design, se_mult)."""
    Sig_s = scale**2 * Sig
    print("=" * 72)
    print(f"E7 ({label}): exact nested-MC duel / controls")
    ths = sample_post(Sig_s, n_inner)
    thJ = ths.reshape(-1, J, 4)
    b_pr = opt_simplex(lambda b: float(np.mean(rev_ss_batch(b, thJ))))
    oi = rng.integers(0, n_inner, n_outer)
    on = rng.standard_normal(n_outer)
    if duel is not None:
        va = _duel_vals(Sig_s, duel[0], ths, thJ, b_pr, oi, on)
        vb = _duel_vals(Sig_s, duel[1], ths, thJ, b_pr, oi, on)
        d = va - vb
        print(f"  A {duel[0]}: {va.mean():.6f} (+-{va.std()/np.sqrt(n_outer):.6f})")
        print(f"  B {duel[1]}: {vb.mean():.6f} (+-{vb.std()/np.sqrt(n_outer):.6f})")
        print(f"  paired A-B = {d.mean():.6f} (+-{d.std()/np.sqrt(n_outer):.6f})")
    for name, dsn, sm in controls:
        v = _duel_vals(Sig_s, dsn, ths, thJ, b_pr, oi, on, sm)
        print(f"  {name:42s}: {v.mean():.5f} (+-{v.std()/np.sqrt(n_outer):.5f})")


if __name__ == "__main__":
    Sig, X = build_posterior()
    sd = np.sqrt(np.diag(Sig))
    names = [f"{p}{j+1}" for j in range(J) for p in ("b", "K", "S", "a")]
    print("E1: posterior marginal sd / |param|:")
    print("   ", {n: f"{s/max(abs(v),1e-9):.2f}" for n, s, v in zip(names, sd, flat(TH0))})
    C = Sig / np.outer(sd, sd)
    print(f"    max |cross-channel corr| = "
          f"{max(abs(C[i,k]) for i in range(4) for k in range(8,12)):.2f}")
    b_bayes, evpi_mc, evpi_q = exp_E2(Sig)
    exp_E2b(Sig)
    b_bar, H, D, Omega = decision_objects(Sig)
    exp_E3(Sig)
    exp_E4(Sig, b_bayes, label="full-width")
    exp_E4(0.25**2 * Sig, b_bayes, label="quarter-width LQ regime")
    grid = exp_E5(Sig, D, Omega, evpi_q)
    exp_E6(Sig, evpi_q)
    # exact duels: EVSI-best vs D-best at full and quarter width, plus power controls
    exp_E7(Sig, 1.0, "full width", duel=((2, 0.6, 8, 8), (1, 0.8, 8, 0)),
           controls=[("weak-pulse ch2 dl=0.06 Te=8 P=8", (2, 0.06, 8, 8), 1.0),
                     ("underpowered ch2 dl=0.6, se x6", (2, 0.6, 8, 8), 6.0),
                     ("tiny-short ch0 dl=0.1 Te=2 P=0", (0, 0.1, 2, 0), 1.0)])
    exp_E7(Sig, 0.25, "quarter width", duel=((2, 1.2, 8, 8), (1, 0.8, 8, 0)))
