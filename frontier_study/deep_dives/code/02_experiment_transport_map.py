"""Deep dive 02 - Transport map from lift tests to MMM parameters (M2).

Verifies:
  Exp A: closed-form window-capture factor C(alpha, T_e, P).
  Exp B: curvature/secant factor Q vs second-order prediction.
  Exp C: Monte Carlo calibration comparison (none / naive-avg / naive-marg / operator),
         incl. operating-point-mismatch variant, plus decision loss.
  Exp D: information content of scalar iROAS summary vs full experiment trajectory.

Model (as dive 01): y_t = beta*h(a_t;K,S) + eps,  a_t = x_t + alpha*a_{t-1},
h(a) = a^S/(a^S+K^S).  theta0 = (beta,K,S,alpha) = (1, 1.5, 2, 0.6),
mean spend 1 => steady-state adstock 2.5, sigma = 0.05.
"""
import numpy as np
from scipy.optimize import least_squares

np.seterr(all="ignore")
rng_global = np.random.default_rng(7)
BETA, K0, S0, ALPHA = 1.0, 1.5, 2.0, 0.6
THETA0 = np.array([BETA, K0, S0, ALPHA])
SIGMA = 0.05


def hill(a, K, S):
    a = np.maximum(a, 1e-12)
    return a**S / (a**S + K**S)


def hill_d1(a, K, S):
    h = hill(a, K, S)
    return S * h * (1 - h) / a


def _hill_d2_num(a, K, S, eps=1e-5):
    return (hill_d1(a + eps, K, S) - hill_d1(a - eps, K, S)) / (2 * eps)


def adstock(x, alpha, a_init=None):
    a = np.empty_like(x, dtype=float)
    prev = x[0] / (1 - alpha) if a_init is None else a_init
    for t in range(len(x)):
        prev = x[t] + alpha * prev
        a[t] = prev
    return a


def mean_path(x, th, a_init=None):
    b, K, S, al = th
    return b * hill(adstock(x, al, a_init), K, S)


def C_formula(alpha, Te, P):
    """Window-capture factor for a uniform pulse of length Te, window Te+P."""
    return 1 - alpha ** (P + 1) * (1 - alpha**Te) / (Te * (1 - alpha))


def experiment_lift(th, xbar_e, delta, Te, P, steady=True):
    """Exact model-implied total incremental lift over window [0, Te+P)."""
    W = Te + P
    x0 = np.full(W, xbar_e)
    x1 = x0.copy()
    x1[:Te] += delta
    a_init = xbar_e / (1 - th[3]) if steady else 0.0
    return float(np.sum(mean_path(x1, th, a_init) - mean_path(x0, th, a_init)))


# ---------------------------------------------------------------- Exp A
def exp_A():
    print("=" * 70)
    print("Exp A: window-capture factor C(alpha,Te,P) vs formula (linear regime)")
    rows = []
    for alpha in (0.3, 0.6, 0.8, 0.9):
        for (Te, P) in ((1, 0), (4, 0), (8, 0), (8, 4), (8, 26), (13, 8)):
            th = np.array([BETA, K0, S0, alpha])
            d = 1e-6
            L = experiment_lift(th, 1.0, d, Te, P)
            abar = 1.0 / (1 - alpha)
            mroas_inf = BETA * hill_d1(abar, K0, S0) / (1 - alpha)
            C_num = L / (d * Te) / mroas_inf
            C_th = C_formula(alpha, Te, P)
            rows.append((alpha, Te, P, C_num, C_th, abs(C_num - C_th)))
    err = max(r[5] for r in rows)
    for r in rows:
        print(f"  alpha={r[0]:.1f} Te={r[1]:2d} P={r[2]:2d}  C_num={r[3]:.6f}  C_formula={r[4]:.6f}")
    print(f"  max |C_num - C_formula| = {err:.2e}")
    return err


# ---------------------------------------------------------------- Exp B
def exp_B():
    print("=" * 70)
    print("Exp B: curvature factor Q exact vs 2nd-order prediction")
    Te, P = 8, 0
    for abar_rel in (0.5, 1.0, 5 / 3, 2.5):  # abar/K
        abar = abar_rel * K0
        xbar = abar * (1 - ALPHA)
        for drel in (0.1, 0.25, 0.5, 1.0):
            delta = drel * xbar
            L = experiment_lift(THETA0, xbar, delta, Te, P)
            # linearized lift with exact window weights
            x0 = np.full(Te + P, xbar)
            x1 = x0.copy()
            x1[:Te] += delta
            da = adstock(x1, ALPHA, xbar / (1 - ALPHA)) - adstock(x0, ALPHA, xbar / (1 - ALPHA))
            L_lin = BETA * hill_d1(abar, K0, S0) * float(np.sum(da))
            Q_exact = L / L_lin
            h1, h2 = hill_d1(abar, K0, S0), _hill_d2_num(abar, K0, S0)
            # weighted mean of da over the window, weights = da (2nd-order term)
            Q_pred = 1 + 0.5 * (h2 / h1) * float(np.sum(da**2) / np.sum(da))
            print(f"  abar/K={abar_rel:.2f} delta/xbar={drel:.2f}: Q_exact={Q_exact:.4f}  Q_2nd={Q_pred:.4f}")


# ---------------------------------------------------------------- Exp C
def make_obs(rng, T=104):
    """Business-as-usual national series: AR(1) log-spend, ±10%-ish (weakly identifying)."""
    u = np.zeros(T)
    for t in range(1, T):
        u[t] = 0.7 * u[t - 1] + rng.normal(0, 0.07)
    x = np.exp(u)
    x = x / x.mean()
    y = mean_path(x, THETA0) + rng.normal(0, SIGMA, T)
    return x, y


def fit(x, y, penalty, rng, n_start=6):
    """Multistart LS: rows = residuals/sigma plus penalty(theta) extra residuals."""
    def resid(p):
        b, K, S, al = np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), 1 / (1 + np.exp(-p[3]))
        th = np.array([b, K, S, al])
        r = (y - mean_path(x, th)) / SIGMA
        return np.concatenate([r, penalty(th)])
    best, bc = None, np.inf
    for _ in range(n_start):
        p0 = np.array([np.log(rng.uniform(0.3, 3)), np.log(rng.uniform(0.5, 4)),
                       np.log(rng.uniform(0.7, 4)), np.log(rng.uniform(0.2, 0.9) / (1 - rng.uniform(0.2, 0.9)))])
        p0[3] = rng.normal(0, 1)
        try:
            sol = least_squares(resid, p0, method="lm", max_nfev=4000)
        except Exception:
            continue
        if sol.cost < bc:
            bc, best = sol.cost, sol.x
    b, K, S, al = np.exp(best[0]), np.exp(best[1]), np.exp(best[2]), 1 / (1 + np.exp(-best[3]))
    return np.array([b, K, S, al])


def mroas_inf(th, xbar=1.0):
    abar = xbar / (1 - th[3])
    return th[0] * hill_d1(abar, th[1], th[2]) / (1 - th[3])


def opt_spend(th, margin=3.0):
    grid = np.linspace(0.05, 5, 800)
    prof = margin * th[0] * hill(grid / (1 - th[3]), th[1], th[2]) - grid
    return grid[np.argmax(prof)]


def profit(x, margin=3.0):
    return margin * BETA * hill(x / (1 - ALPHA), K0, S0) - x


def exp_C(n_rep=60, mismatch=False):
    tag = "operating-point MISMATCH (xbar_e=0.5)" if mismatch else "matched operating point"
    print("=" * 70)
    print(f"Exp C ({tag}): calibration comparison, {n_rep} Monte Carlo reps")
    Te, P, drel = 8, 0, 0.5
    xbar_e = 0.5 if mismatch else 1.0
    delta = drel * xbar_e
    L_true = experiment_lift(THETA0, xbar_e, delta, Te, P)
    se_L = 0.05 * abs(L_true)          # a well-powered test: 5% se on the lift
    spend_delta = delta * Te
    iroas_se = se_L / spend_delta
    m_true = mroas_inf(THETA0)
    xstar_true = opt_spend(THETA0)
    pmax = profit(xstar_true)
    print(f"  true mROAS_inf={m_true:.4f}  true avg ROAS={BETA*hill(2.5,K0,S0):.4f}  "
          f"true iROAS_exp={L_true/spend_delta:.4f}  (C={C_formula(ALPHA,Te,P):.3f})")

    modes = ["none", "naive-avg", "naive-marg", "operator"]
    res = {m: [] for m in modes}
    rng = np.random.default_rng(42)
    for rep in range(n_rep):
        x, y = make_obs(rng)
        L_hat = L_true + rng.normal(0, se_L)
        iroas_hat = L_hat / spend_delta

        def pen_none(th):
            return np.zeros(0)

        def pen_navg(th):  # Meridian-style: experiment iROAS as prior on model avg ROAS
            return np.array([(th[0] * hill(1.0 / (1 - th[3]), th[1], th[2]) - iroas_hat) / iroas_se])

        def pen_nmarg(th):  # experiment iROAS as estimate of steady-state marginal ROAS
            return np.array([(mroas_inf(th) - iroas_hat) / iroas_se])

        def pen_op(th):   # experiment operator: simulate the design inside the model
            return np.array([(experiment_lift(th, xbar_e, delta, Te, P) - L_hat) / se_L])

        xg = np.linspace(0.05, 2.5, 120)
        curve_true = BETA * hill(xg / (1 - ALPHA), K0, S0)
        for m, pen in zip(modes, [pen_none, pen_navg, pen_nmarg, pen_op]):
            th = fit(x, y, pen, rng)
            with np.errstate(all="ignore"):
                cv = th[0] * hill(xg / (1 - th[3]), th[1], th[2])
            cv = np.where(np.isfinite(cv), cv, 1e6)
            curve_err = float(np.max(np.abs(cv - curve_true)))
            res[m].append([mroas_inf(th), curve_err, abs(th[3] - ALPHA),
                           pmax - profit(opt_spend(th))])
    pred_marg_bias = 100 * (L_true / spend_delta / m_true - 1)
    print(f"  theory-predicted naive-marg bias = C*Q*rho - 1 = {pred_marg_bias:.1f}%")
    print(f"  {'mode':<11}{'mROAS med bias%':>16}{'mROAS IQR':>12}{'med curve err':>14}{'wrecked%':>9}{'med |dalpha|':>13}{'med profit loss':>16}")
    out = {}
    for m in modes:
        A = np.array(res[m])
        mb = 100 * (np.median(A[:, 0]) - m_true) / m_true
        iqr = np.subtract(*np.percentile(A[:, 0], [75, 25]))
        ce = np.median(A[:, 1])
        wr = 100 * np.mean(A[:, 1] > 0.5)  # curve off by >50% of sales ceiling anywhere
        da = np.median(A[:, 2])
        pl = np.median(A[:, 3])
        out[m] = (mb, iqr, ce, wr, da, pl)
        print(f"  {m:<11}{mb:>15.1f}%{iqr:>12.4f}{ce:>14.4f}{wr:>8.0f}%{da:>13.4f}{pl:>16.4f}")
    return out


# ---------------------------------------------------------------- Exp D
def exp_D():
    print("=" * 70)
    print("Exp D: Fisher information, scalar iROAS summary vs full trajectory")
    Te, P = 8, 6
    xbar_e, delta = 1.0, 0.5
    W = Te + P
    sig_e = 0.05  # per-period noise of the differenced treated-control series

    def traj(th):
        x0 = np.full(W, xbar_e)
        x1 = x0.copy()
        x1[:Te] += delta
        ai = xbar_e / (1 - th[3])
        return mean_path(x1, th, ai) - mean_path(x0, th, ai)

    eps = 1e-6
    Jt = np.zeros((W, 4))
    Js = np.zeros((1, 4))
    for i in range(4):
        tp, tm = THETA0.copy(), THETA0.copy()
        tp[i] += eps
        tm[i] -= eps
        Jt[:, i] = (traj(tp) - traj(tm)) / (2 * eps)
        Js[0, i] = (np.sum(traj(tp)) - np.sum(traj(tm))) / (2 * eps)
    F_traj = Jt.T @ Jt / sig_e**2
    F_scal = Js.T @ Js / (W * sig_e**2)  # scalar sum has variance W*sig^2
    ev_t = np.linalg.eigvalsh(F_traj)
    ev_s = np.linalg.eigvalsh(F_scal)
    print(f"  trajectory FIM eigenvalues: {np.array2string(ev_t, precision=3)}")
    print(f"  scalar     FIM eigenvalues: {np.array2string(ev_s, precision=3)}")
    print(f"  rank: trajectory={np.sum(ev_t > 1e-8 * ev_t[-1])}, scalar={np.sum(ev_s > 1e-8 * max(ev_s[-1],1))}")
    print(f"  trace ratio (traj/scalar): {np.trace(F_traj) / np.trace(F_scal):.1f}")
    se = np.sqrt(np.diag(np.linalg.inv(F_traj + 1e-9 * np.eye(4))))
    print(f"  trajectory CRLB SEs (beta,K,S,alpha): {np.array2string(se, precision=2)}")
    # per-parameter: information about alpha alone (profile: 1/CRLB with others known)
    for i, nm in enumerate(["beta", "K", "S", "alpha"]):
        print(f"    known-others info, {nm:5s}: traj={F_traj[i,i]:.1f}  scalar={F_scal[i,i]:.2f}  "
              f"ratio={F_traj[i,i]/max(F_scal[i,i],1e-12):.1f}")


if __name__ == "__main__":
    errA = exp_A()
    exp_B()
    exp_C(mismatch=False)
    exp_C(mismatch=True)
    exp_D()
    print("=" * 70)
    print(f"Exp A max formula error: {errA:.2e}  (should be ~1e-6 or less)")
