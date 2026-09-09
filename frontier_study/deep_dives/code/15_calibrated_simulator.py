"""
Dive 15 — The calibrated simulator: a certification harness binding a market simulator
to a library of held-out experiments, and the quantitative counterfactual bar.

Usage:  python 15_calibrated_simulator.py quick     (E1, E3, E4, E6, E8, E9, E11-E13: ~1.5 min)
        python 15_calibrated_simulator.py all       (everything, ~6 min; e2 ≈ 20 s, e5 ≈ 15 s, e7 ≈ 1 min, e10 ≈ 1 min)
        python 15_calibrated_simulator.py e5|e7|e10|e11 ...
Needs numpy + scipy.

Model (Section 1 of the writeup):
  library experiments e=1..E:  tau_hat_e = tau_e + eps_e,  eps_e ~ N(0, v_e), v_e known
  simulator prediction s_e;   calibration  tau_e = alpha + beta*s_e + u_e,  u_e ~ N(0, omega^2)
  omega^2 = the simulator's irreducible counterfactual residual variance after recalibration.
"""
import sys, numpy as np
from scipy import stats, optimize

rng_global = np.random.default_rng(0)

# ----------------------------------------------------------------------------------
# Core estimators
# ----------------------------------------------------------------------------------
def excess_variance(tau_hat, s, v, p=2):
    """Exactly unbiased excess-variance estimator of omega^2 from an OLS recalibration,
    tau_hat on (1, s), with known per-experiment sampling variances v.
    Returns omega2_hat, alpha_hat, beta_hat, residuals, leverage, sum_(1-h)v."""
    E = len(s)
    X = np.column_stack([np.ones(E), s])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ tau_hat
    r = tau_hat - X @ coef
    h = np.einsum('ij,jk,ik->i', X, XtX_inv, X)
    floor = np.sum((1 - h) * v)
    om2 = (r @ r - floor) / (E - p)
    return om2, coef[0], coef[1], r, h, floor

def certify_ucb(tau_hat, s, v, a=0.05, p=2):
    """Upper confidence bound on omega^2 via the chi-square profile
    (E-p)(om2_hat+vbar)/chi2_{a,E-p} - vbar  (exact under homoscedastic v)."""
    E = len(s)
    om2, *_ = excess_variance(tau_hat, s, v, p)
    vbar = v.mean()
    q = stats.chi2.ppf(a, E - p)
    return (E - p) * (om2 + vbar) / q - vbar, om2

def E_needed_naive(rho, nu, a=0.05, b=0.2, p=2):
    """Round-1 (WRONG) normal-approximation law: p + 2(z_a+z_b)^2 (rho+nu)^2/(1-rho)^2."""
    z = stats.norm.ppf(1 - a) + stats.norm.ppf(1 - b)
    return p + 2 * z**2 * (rho + nu)**2 / (1 - rho)**2

def kappa_gap(rho, nu):
    """Certification gap ratio kappa = (omega*^2 + vbar)/(omega^2 + vbar), in omega*^2 units."""
    return (1 + nu) / (rho + nu)

def E_needed(rho, nu, a=0.05, b=0.2, p=2):
    """Round-3 certification law (normal approx of the chi-square ratio test):
    E - p >= 2 (z_b + kappa z_a)^2 / (kappa - 1)^2  with kappa = (1+nu)/(rho+nu),
    i.e. 2 (z_b + kappa z_a)^2 (rho+nu)^2 / (1-rho)^2."""
    k = kappa_gap(rho, nu)
    z = stats.norm.ppf(1 - b) + k * stats.norm.ppf(1 - a)
    return p + 2 * z**2 / (k - 1)**2

def E_needed_exact(rho, nu, a=0.05, b=0.2, p=2, Emax=5000):
    """Exact (homoscedastic) law: smallest E with chi2_{1-b,E-p}/chi2_{a,E-p} <= kappa."""
    k = kappa_gap(rho, nu)
    for E in range(p + 1, Emax):
        if stats.chi2.ppf(1 - b, E - p) / stats.chi2.ppf(a, E - p) <= k:
            return E
    return np.inf

def gen_library(E, omega, vbar, rng, beta=1.0, alpha=0.0, s_sd=1.0, tail='normal',
                v_spread=0.0):
    s = rng.normal(0, s_sd, E)
    if tail == 'normal':
        u = rng.normal(0, omega, E)
    elif tail == 't3':
        u = rng.standard_t(3, E) * omega / np.sqrt(3)   # var = 3/(3-2)=3 -> scale to omega
    v = vbar * np.exp(rng.normal(0, v_spread, E) - v_spread**2 / 2) if v_spread > 0 else np.full(E, vbar)
    tau = alpha + beta * s + u
    tau_hat = tau + rng.normal(0, np.sqrt(v))
    return s, tau, tau_hat, v

# ----------------------------------------------------------------------------------
# E1: unbiasedness and sd of the excess-variance estimator; certification law power
# ----------------------------------------------------------------------------------
def e1(R=4000, seed=1):
    rng = np.random.default_rng(seed)
    print("\n=== E1: excess-variance estimator: bias / sd / chi2 UCB coverage ===")
    omega, vbar = 0.5, 0.8
    for E in [8, 15, 30, 60, 120]:
        est, ucb_cov = [], 0
        for _ in range(R):
            s, tau, th, v = gen_library(E, omega, vbar, rng)
            om2, *_ = excess_variance(th, s, v)
            ucb, _ = certify_ucb(th, s, v)
            est.append(om2); ucb_cov += (ucb >= omega**2)
        est = np.array(est)
        sd_pred = (omega**2 + vbar) * np.sqrt(2 / (E - 2))
        print(f"E={E:4d}  mean om2_hat={est.mean():.4f} (true {omega**2:.4f}, mc se {est.std()/np.sqrt(R):.4f})"
              f"  sd={est.std():.4f} pred={sd_pred:.4f}  P(om2_hat<0)={np.mean(est<0):.3f}"
              f"  UCB95 coverage={ucb_cov/R:.3f}")

# ----------------------------------------------------------------------------------
# E2: certification law — required E vs formula; optimal library precision
# ----------------------------------------------------------------------------------
def e2(R=3000, seed=2):
    rng = np.random.default_rng(seed)
    print("\n=== E2: certification law: E needed to certify omega^2 <= omega*^2 (a=.05, power .8) ===")
    om_star2 = 1.0
    print(" rho=om2/om*2  nu=vbar/om*2   E_naive(R1)  E_law(R3)  E_exact   power@E_exact(MC)   E_MC(power .8)")
    for rho in [0.25, 0.5, 0.7]:
        for nu in [0.1, 0.5, 1.0, 2.0]:
            En = int(np.ceil(E_needed_naive(rho, nu))); El = int(np.ceil(E_needed(rho, nu))); Ef = E_needed_exact(rho, nu)
            def power(E):
                c = 0
                for _ in range(R):
                    s, tau, th, v = gen_library(E, np.sqrt(rho * om_star2), nu * om_star2, rng)
                    ucb, _ = certify_ucb(th, s, v)
                    c += ucb <= om_star2
                return c / R
            pf = power(Ef)
            # bisection for E giving power .8
            lo, hi = 3, max(Ef * 3, 20)
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if power(mid) >= 0.8: hi = mid
                else: lo = mid
            print(f"   {rho:4.2f}       {nu:4.2f}        {En:5d}      {El:5d}     {Ef:5d}        {pf:.3f}              {hi:5d}")
    print("\n Optimal library precision: total geo-cost ∝ E(rho,nu)/nu (cost per experiment ∝ 1/v).")
    for rho in [0.25, 0.5, 0.7]:
        nus = np.linspace(0.05, 3, 300)
        cost = np.array([E_needed(rho, n) / n for n in nus])
        cost_naive = (rho + nus)**2 / nus
        print(f"   rho={rho}: argmin nu (R3 law) = {nus[np.argmin(cost)]:.2f}  [R1 law said nu=rho={rho}];  cost(nu*)/cost(nu=1) = {cost.min()/cost[np.argmin(abs(nus-1))]:.2f}, cost(nu=0.1)/cost(nu=1) = {cost[np.argmin(abs(nus-0.1))]/cost[np.argmin(abs(nus-1))]:.2f}")

# ----------------------------------------------------------------------------------
# E3: the counterfactual bar — EVSI of an experiment given a certified simulator
# ----------------------------------------------------------------------------------
def evsi_quadratic(V, vx, gamma):
    return V**2 / ((V + vx) * 2 * gamma)

def omega_star2(c, vx, gamma):
    return gamma * c + np.sqrt((gamma * c)**2 + 2 * gamma * c * vx)

def e3(R=200000, seed=3):
    rng = np.random.default_rng(seed)
    print("\n=== E3: the counterfactual bar: skip the experiment iff EVSI(experiment | simulator posterior) <= cost ===")
    gamma, vx = 1.0, 1.0
    print(" (a) quadratic-loss EVSI closed form vs Monte Carlo (prior N(m, V), experiment var vx)")
    for V in [0.25, 0.5, 1.0, 2.0]:
        tau = rng.normal(0.0, np.sqrt(V), R)
        th = tau + rng.normal(0, np.sqrt(vx), R)
        post_mean = V / (V + vx) * th
        loss_prior = np.mean((tau - 0) ** 2) / (2 * gamma)
        loss_post = np.mean((tau - post_mean) ** 2) / (2 * gamma)
        print(f"   V={V:4.2f}: EVSI_MC={loss_prior - loss_post:.4f}  closed={evsi_quadratic(V, vx, gamma):.4f}")
    print(" (b) omega*^2 solving EVSI=c, in units of vx, vs cost c (in units of vx/(2 gamma)):")
    for ctil in [0.05, 0.1, 0.25, 0.5, 1.0]:
        c = ctil * vx / (2 * gamma)
        print(f"   c~={ctil:5.2f}: omega*^2/vx = {omega_star2(c, vx, gamma)/vx:.3f}  (experiment-equivalents needed: {vx/omega_star2(c, vx, gamma):.2f})")
    print(" (c) threshold (go/no-go) decision: EVSI_thresh vs quadratic EVSI at the same V; loss = |tau - t0| on wrong side")
    t0 = 0.0
    for m in [0.0, 0.5, 1.0]:
        for V in [0.25, 0.5, 1.0, 2.0]:
            tau = rng.normal(m, np.sqrt(V), R)
            th = tau + rng.normal(0, np.sqrt(vx), R)
            pm = m + V / (V + vx) * (th - m)
            act_prior = m > t0
            loss_prior = np.mean(np.where(act_prior, np.maximum(t0 - tau, 0), np.maximum(tau - t0, 0)))
            act_post = pm > t0
            loss_post = np.mean(np.where(act_post, np.maximum(t0 - tau, 0), np.maximum(tau - t0, 0)))
            print(f"   m={m:.1f} V={V:4.2f}: EVSI_thresh={loss_prior-loss_post:.4f}   EVSI_quad(gamma=1)={evsi_quadratic(V, vx, 1):.4f}   ratio={(loss_prior-loss_post)/evsi_quadratic(V, vx, 1):.2f}")

# ----------------------------------------------------------------------------------
# E4: level fit does not constrain lift error (factual != counterfactual)
# ----------------------------------------------------------------------------------
def e4(R=20000, seed=4):
    rng = np.random.default_rng(seed)
    print("\n=== E4: factual (level) fit vs counterfactual (lift) error in a random-utility agent simulator ===")
    # truth: p(d) = Phi((mu + a d)/s); simulator: Phi(mu' + a' d), mu' matched to baseline level up to noise
    mu = rng.normal(-1.0, 0.5, R); a = 0.6; s = 1.5; d = 1.0
    p0 = stats.norm.cdf(mu / s); p1 = stats.norm.cdf((mu + a * d) / s)
    lift = p1 - p0
    mu_p = stats.norm.ppf(p0) + rng.normal(0, 0.05, R)       # good level calibration
    a_p = a / s * np.exp(rng.normal(0, 0.5, R))              # unknown sensitivity scale
    p0s = stats.norm.cdf(mu_p); p1s = stats.norm.cdf(mu_p + a_p * d)
    lev_err = p0s - p0; lift_err = (p1s - p0s) - lift
    print(f"   corr(level error, lift error) = {np.corrcoef(lev_err, lift_err)[0,1]:+.3f};  sd level err {lev_err.std():.4f}, sd lift err {lift_err.std():.4f}, mean lift {lift.mean():.4f}")
    # a simulator with WORSE level fit but right sensitivity
    mu_p2 = stats.norm.ppf(p0) + rng.normal(0, 0.3, R); a_p2 = a / s
    lift_err2 = (stats.norm.cdf(mu_p2 + a_p2 * d) - stats.norm.cdf(mu_p2)) - lift
    print(f"   worse level fit (sd 0.3 on latent) but right sensitivity: sd lift err {lift_err2.std():.4f} (vs {lift_err.std():.4f} with good level fit, wrong sensitivity)")

# ----------------------------------------------------------------------------------
# E5: homogeneity: which recalibration family transports across dose and baseline
# ----------------------------------------------------------------------------------
def lift_truth(mu, d, q, a, s):
    """Random-coefficient truth: fraction q responsive with sensitivity a, rest 0;
    latent normal heterogeneity sd s (probit scale)."""
    return q * (stats.norm.cdf((mu + a * d) / s) - stats.norm.cdf(mu / s))

def sim_homog(mu_lat, d, a_p):
    return stats.norm.cdf(mu_lat + a_p * d) - stats.norm.cdf(mu_lat)

def fit_family(fam, th, v, mu_lat, d, sim_pred):
    """Return a predictor f(mu_lat, d, sim_pred) fitted by weighted least squares."""
    w = 1 / v
    if fam == 'additive':      # tau = alpha + beta*sim
        X = np.column_stack([np.ones_like(sim_pred), sim_pred])
        coef = np.linalg.lstsq(X * np.sqrt(w)[:, None], th * np.sqrt(w), rcond=None)[0]
        return lambda ml, dd, sp: coef[0] + coef[1] * sp
    if fam == 'latent':        # tau = Phi(mu' + a'' d) - Phi(mu')   (one parameter a'')
        f = lambda a2: np.sum(w * (th - sim_homog(mu_lat, d, a2))**2)
        a2 = optimize.minimize_scalar(f, bounds=(1e-4, 5), method='bounded').x
        return lambda ml, dd, sp: sim_homog(ml, dd, a2)
    if fam == 'mixture':       # tau = q'[Phi(mu'+a''d) - Phi(mu')]
        f = lambda p: np.sum(w * (th - p[0] * sim_homog(mu_lat, d, p[1]))**2)
        p = optimize.minimize(f, x0=[0.5, 0.3], bounds=[(0.01, 1.5), (1e-4, 5)]).x
        return lambda ml, dd, sp: p[0] * sim_homog(ml, dd, p[1])
    if fam == 'mixture2':      # q'[Phi((mu'+a''d)/s') - Phi(mu'/s')] with mu' = s' * Phi^-1(p0): re-derive
        def pred(ml, dd, pr):
            p0 = stats.norm.cdf(ml)
            m = pr[2] * stats.norm.ppf(p0)
            return pr[0] * (stats.norm.cdf((m + pr[1] * dd) / pr[2]) - stats.norm.cdf(m / pr[2]))
        f = lambda p: np.sum(w * (th - pred(mu_lat, d, p))**2)
        p = optimize.minimize(f, x0=[0.5, 0.4, 1.2], bounds=[(0.01, 1.5), (1e-4, 5), (0.5, 4)]).x
        return lambda ml, dd, sp: pred(ml, dd, p)

def e5(R=300, seed=5, E=24, N_users=200000):
    rng = np.random.default_rng(seed)
    print("\n=== E5: recalibration families under homogeneous-agent simulators: transport across dose & baseline ===")
    a, s = 0.8, 1.6
    for q in [1.0, 0.4]:
        print(f"\n  truth: responsive fraction q={q}, sensitivity a={a}, latent sd s={s}; simulator homogeneous with a'=a (wrong scale)")
        print("  library: doses in [0.3, 1.0], baselines mu in [-1.5,-0.5]; targets: (i) in-range, (ii) dose 2.5, (iii) mu=-3 (rare baseline)")
        res = {f: {'in': [], 'dose': [], 'base': []} for f in ['additive', 'latent', 'mixture', 'mixture2']}
        for _ in range(R):
            mu = rng.uniform(-1.5, -0.5, E); d = rng.uniform(0.3, 1.0, E)
            p0 = stats.norm.cdf(mu / s); tau = lift_truth(mu, d, q, a, s)
            v = (p0 * (1 - p0) * 2) / N_users * np.ones(E)   # binomial two-arm variance
            th = tau + rng.normal(0, np.sqrt(v))
            mu_lat = stats.norm.ppf(p0)                        # simulator matches baseline level
            sp = sim_homog(mu_lat, d, a)
            targets = {'in': (rng.uniform(-1.5, -0.5), rng.uniform(0.3, 1.0)),
                       'dose': (-1.0, 2.5), 'base': (-3.0, 0.7)}
            for fam in res:
                f = fit_family(fam, th, v, mu_lat, d, sp)
                for k, (mt, dt) in targets.items():
                    truth = lift_truth(mt, dt, q, a, s)
                    ml = stats.norm.ppf(stats.norm.cdf(mt / s))
                    res[fam][k].append((f(ml, dt, sim_homog(ml, dt, a)) - truth) / truth)
        for fam in res:
            out = "   %-9s" % fam
            for k in ['in', 'dose', 'base']:
                e = np.array(res[fam][k])
                out += f"  {k}: rel.err mean {e.mean():+.3f} rmse {np.sqrt(np.mean(e**2)):.3f} |"
            print(out)

# ----------------------------------------------------------------------------------
# E6: contamination — the noise-floor test and its blind spot; the re-estimation test
# ----------------------------------------------------------------------------------
def e6(R=4000, seed=6):
    rng = np.random.default_rng(seed)
    print("\n=== E6: contamination tests ===")
    E, vbar = 30, 1.0
    print(" (a) noise-floor test: reject 'honest' if r'r < chi2_{.05,E-2} * sum(1-h)v. Rejection rate vs memorised fraction lam, omega")
    print("      predicted blind spot: undetectable when (1-lam)(1+om2/v) >~ 1 - z*sqrt(2/(E-2))")
    for omega in [0.0, 0.5, 1.0]:
        row = f"   omega={omega:.1f}: "
        for lam in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            rej = 0
            for _ in range(R):
                s, tau, th, v = gen_library(E, omega, vbar, rng)
                mem = rng.random(E) < lam
                s_c = np.where(mem, th, s)              # memoriser returns the published estimate
                _, _, _, r, h, floor = excess_variance(th, s_c, v)
                rej += (r @ r) < stats.chi2.ppf(0.05, E - 2) / (E - 2) * floor
            row += f" lam={lam:.1f}:{rej/R:.2f}"
        print(row + f"   [blind-spot lam < {omega**2/(omega**2+vbar):.2f} + O(E^-1/2)]")
    print(" (b) strategic evasion: memoriser adds noise N(0, v + om2) to its output -> passes the floor test")
    rej = 0; om_est = []
    for _ in range(R):
        s, tau, th, v = gen_library(E, 0.5, vbar, rng)
        s_c = th + rng.normal(0, np.sqrt(vbar + 0.25), E)
        om2, _, _, r, h, floor = excess_variance(th, s_c, v)
        rej += (r @ r) < stats.chi2.ppf(0.05, E - 2) / (E - 2) * floor; om_est.append(om2)
    print(f"   rejection {rej/R:.3f}; apparent omega^2 = {np.mean(om_est):.3f} (looks like an honest omega=0.5 simulator)")
    print(" (c) re-estimation test: D = MSE(s, tau') - MSE(s, tau_hat) - (v' - v), tau' an alternative estimate with corr(eps,eps')=r")
    print("      honest E[D]=0; memoriser E[D] = 2 lam (v - cov) = 2 lam v (1-r). One-sided t-test on D_e, level .05.")
    for r_ee in [0.5, 0.8]:
        row = f"   corr={r_ee}: "
        for lam in [0.0, 0.1, 0.2, 0.4, 1.0]:
            for omega in [0.5]:
                rej = 0
                for _ in range(R):
                    s = rng.normal(0, 1, E); tau = s + rng.normal(0, omega, E)
                    eps = rng.normal(0, 1, E); eps2 = r_ee * eps + np.sqrt(1 - r_ee**2) * rng.normal(0, 1, E)
                    th = tau + np.sqrt(vbar) * eps; th2 = tau + np.sqrt(vbar) * eps2
                    mem = rng.random(E) < lam
                    # honest simulator output recalibrated on the library (identical calibration for both)
                    s_c = np.where(mem, th, s)
                    D = (s_c - th2)**2 - (s_c - th)**2
                    t = D.mean() / (D.std(ddof=1) / np.sqrt(E))
                    rej += t > stats.t.ppf(0.95, E - 1)
                row += f" lam={lam:.1f}:{rej/R:.2f}"
        print(row)
    print(" (d) strategic memoriser (adds noise) under the re-estimation test:")
    for r_ee in [0.5, 0.8]:
        rej = 0
        for _ in range(R):
            s = rng.normal(0, 1, E); tau = s + rng.normal(0, 0.5, E)
            eps = rng.normal(0, 1, E); eps2 = r_ee * eps + np.sqrt(1 - r_ee**2) * rng.normal(0, 1, E)
            th = tau + eps; th2 = tau + eps2
            s_c = th + rng.normal(0, np.sqrt(1.25), E)
            D = (s_c - th2)**2 - (s_c - th)**2
            t = D.mean() / (D.std(ddof=1) / np.sqrt(E)); rej += t > stats.t.ppf(0.95, E - 1)
        print(f"   corr={r_ee}: rejection {rej/R:.2f}")

# ----------------------------------------------------------------------------------
# E7: distribution shift — coverage of prediction intervals with global vs local omega
# ----------------------------------------------------------------------------------
def e7(R=2000, seed=7, E=60):
    rng = np.random.default_rng(seed)
    print("\n=== E7: distribution shift: omega^2(x) = om0^2 + kappa x^2 (x = novelty covariate); PI coverage global vs kernel-local ===")
    om0, kappa, vbar = 0.4, 0.6, 0.5
    for xt in [0.0, 1.0, 2.0, 3.0]:
        cov_g = cov_l = 0; wid_g = wid_l = 0.0; neff = []
        for _ in range(R):
            x = np.abs(rng.normal(0, 1, E))           # library novelty; mostly near 0
            s = rng.normal(0, 1, E)
            om2x = om0**2 + kappa * x**2
            tau = s + rng.normal(0, np.sqrt(om2x)); th = tau + rng.normal(0, np.sqrt(vbar), E)
            om2, al, be, r, h, floor = excess_variance(th, s, np.full(E, vbar))
            # kernel-local excess variance around xt
            w = np.exp(-0.5 * ((x - xt) / 0.5)**2); w /= w.sum()
            om2_loc = max(np.sum(w * (r**2 - (1 - h) * vbar)) / (1 - np.sum(w**2)) , 0)
            neff.append(1 / np.sum(w**2))
            # target
            st = rng.normal(); taut = st + rng.normal(0, np.sqrt(om0**2 + kappa * xt**2))
            pred = al + be * st
            for om2_use, tag in [(max(om2, 0), 'g'), (om2_loc, 'l')]:
                V = om2_use + (om2_use + vbar) * (1 / E + (st - s.mean())**2 / np.sum((s - s.mean())**2))
                half = 1.96 * np.sqrt(V)
                hit = abs(taut - pred) <= half
                if tag == 'g': cov_g += hit; wid_g += half
                else: cov_l += hit; wid_l += half
        print(f"   target x={xt:.1f}: true om2={om0**2+kappa*xt**2:.2f} | global PI: cov {cov_g/R:.3f} halfwidth {wid_g/R:.2f} | local PI: cov {cov_l/R:.3f} halfwidth {wid_l/R:.2f} (n_eff {np.mean(neff):.1f})")

# ----------------------------------------------------------------------------------
# E8: library selection — selection on s harmless, selection on tau_hat (publication) biases
# ----------------------------------------------------------------------------------
def e8(R=3000, seed=8, E=30):
    rng = np.random.default_rng(seed)
    print("\n=== E8: selection into the library ===")
    omega, vbar = 0.5, 0.5
    for mode in ['none', 'on s (advertisers test promising channels)', 'on tau_hat>0 (only positive results kept)', 'on |t|>1.64 (only significant results)']:
        al, be, om = [], [], []
        for _ in range(R):
            s, tau, th, v = gen_library(E * 3, omega, vbar, rng)
            if mode == 'none': keep = np.arange(E)
            elif mode.startswith('on s'): keep = np.argsort(-s)[:E]
            elif mode.startswith('on tau_hat'): keep = np.where(th > 0)[0][:E]
            else: keep = np.where(np.abs(th / np.sqrt(v)) > 1.64)[0][:E]
            if len(keep) < 8: continue
            om2, a, b, *_ = excess_variance(th[keep], s[keep], v[keep])
            al.append(a); be.append(b); om.append(om2)
        print(f"   {mode:45s}: alpha {np.mean(al):+.3f} beta {np.mean(be):.3f} omega2 {np.mean(om):.3f}  (truth 0, 1, {omega**2})")

# ----------------------------------------------------------------------------------
# E9: robustness sweeps — heavy tails, heteroscedastic v, estimated v
# ----------------------------------------------------------------------------------
def e9(R=4000, seed=9):
    rng = np.random.default_rng(seed)
    print("\n=== E9: robustness of the excess-variance estimator and the UCB ===")
    omega, vbar, E = 0.5, 0.8, 30
    for tail, vs, df_v in [('normal', 0.0, None), ('t3', 0.0, None), ('normal', 0.8, None), ('normal', 0.0, 20), ('t3', 0.8, 20)]:
        est, cov = [], 0
        for _ in range(R):
            s, tau, th, v = gen_library(E, omega, vbar, rng, tail=tail, v_spread=vs)
            v_use = v * rng.chisquare(df_v, E) / df_v if df_v else v
            om2, *_ = excess_variance(th, s, v_use)
            ucb, _ = certify_ucb(th, s, v_use); est.append(om2); cov += ucb >= omega**2
        est = np.array(est)
        print(f"   tail={tail:6s} v-spread(log sd)={vs:.1f} v est df={df_v}: mean {est.mean():.3f} (true .25) sd {est.std():.3f} pred {(omega**2+vbar)*np.sqrt(2/(E-2)):.3f} UCB cov {cov/R:.3f}")
    print(" seed sensitivity of E1 headline (E=30): ")
    for sd in [11, 12, 13]:
        r2 = np.random.default_rng(sd); est = []
        for _ in range(R):
            s, tau, th, v = gen_library(30, omega, vbar, r2); est.append(excess_variance(th, s, v)[0])
        print(f"   seed {sd}: mean {np.mean(est):.4f} sd {np.std(est):.4f}")

# ----------------------------------------------------------------------------------
# E10: sequential certification — optional stopping inflates false certification; alpha-spending fixes it
# ----------------------------------------------------------------------------------
def prequential_u(th, s, v, E0=8):
    """Standardised prequential residuals u_E = (tau_hat_E - x_E' beta_{E-1}) / sqrt(1 + h_E),
    E = E0..; independent N(0, omega^2 + vbar) under the homoscedastic model (Gaussian prequential property)."""
    us = []
    for E in range(E0, len(s) + 1):
        X = np.column_stack([np.ones(E - 1), s[:E - 1]])
        XtX_inv = np.linalg.inv(X.T @ X); coef = XtX_inv @ X.T @ th[:E - 1]
        x = np.array([1.0, s[E - 1]]); hp = x @ XtX_inv @ x
        us.append((th[E - 1] - x @ coef) / np.sqrt(1 + hp))
    return np.array(us)

def e10(R=2000, seed=10, Emax=150):
    rng = np.random.default_rng(seed)
    print("\n=== E10: sequential certification as experiments accrue (looks at every E from 8 to 150) ===")
    om_star2 = 1.0; vbar = 0.5
    sig0 = om_star2 + vbar; sig1 = 0.5 * om_star2 + vbar     # e-process: H0 at the bar vs a design alternative rho=0.5
    print("   rules: naive (chi2 UCB at every look, a=.05) | e-process (prequential likelihood ratio at the bar vs rho=.5, certify when LR>=20) | 5 geometric looks (E=10,20,40,80,150) with a=.01 each")
    for omega2 in [1.0, 1.2, 0.5, 0.7]:
        cnt = {'naive': 0, 'e-process': 0, 'geo-looks': 0}; tm = {k: [] for k in cnt}
        for _ in range(R):
            s, tau, th, v = gen_library(Emax, np.sqrt(omega2), vbar, rng)
            u = prequential_u(th, s, v)
            lr = np.cumprod(np.sqrt(sig0 / sig1) * np.exp(u**2 * (1 / sig0 - 1 / sig1) / 2))
            hit = np.where(lr >= 20)[0]
            if len(hit): cnt['e-process'] += 1; tm['e-process'].append(hit[0] + 8)
            done_n = False
            for E in range(8, Emax + 1):
                ucb, _ = certify_ucb(th[:E], s[:E], v[:E], a=0.05)
                if not done_n and ucb <= om_star2: cnt['naive'] += 1; tm['naive'].append(E); done_n = True
                if E in (10, 20, 40, 80, 150):
                    ucb_g, _ = certify_ucb(th[:E], s[:E], v[:E], a=0.01)
                    if ucb_g <= om_star2 and E not in tm['geo-looks'] and (len(tm['geo-looks']) == 0 or True):
                        pass
            # geometric looks, first certification
            for E in (10, 20, 40, 80, 150):
                ucb_g, _ = certify_ucb(th[:E], s[:E], v[:E], a=0.01)
                if ucb_g <= om_star2: cnt['geo-looks'] += 1; tm['geo-looks'].append(E); break
        tag = 'FALSE cert' if omega2 >= om_star2 else 'power'
        print(f"   omega2={omega2:.1f} (bar 1.0): " + " | ".join(f"{k} {tag} {cnt[k]/R:.3f} (median E {np.median(tm[k]) if tm[k] else float('nan'):.0f})" for k in cnt))

# ----------------------------------------------------------------------------------
# E11: end-to-end decision duel — substitute the certified simulator vs always experiment vs never
# ----------------------------------------------------------------------------------
def e11(R=4000, seed=11):
    rng = np.random.default_rng(seed)
    print("\n=== E11: decision duel over a stream of channel decisions (LQ profit, gamma=1, experiment var vx=1) ===")
    vx, gamma, c = 1.0, 1.0, 0.1
    om_s2 = omega_star2(c, vx, gamma)
    print(f"   cost c={c} -> bar omega*^2={om_s2:.3f}")
    for omega2 in [0.1, 0.3, 0.5, 0.8, 1.5]:
        E = 40
        loss = {'never': [], 'always': [], 'sim-only': [], 'bar-rule': [], 'oracle-rule': []}
        for _ in range(R):
            s, tau, th, v = gen_library(E, np.sqrt(omega2), vx, rng)
            om2, al, be, r, h, floor = excess_variance(th, s, v)
            ucb, _ = certify_ucb(th, s, v)
            st = rng.normal(); taut = al + be * st + rng.normal(0, np.sqrt(omega2))  # truth generated with fitted line? no: use true line
            taut = st + rng.normal(0, np.sqrt(omega2))
            pred = al + be * st
            Vp = max(om2, 0) + (max(om2, 0) + vx) * (1 / E + (st - s.mean())**2 / np.sum((s - s.mean())**2))
            xhat = rng.normal(taut, np.sqrt(vx))
            # decisions
            loss['never'].append(taut**2 / (2 * gamma))                                   # x=0 (prior mean 0)
            loss['always'].append((taut - xhat)**2 / (2 * gamma) + c)
            loss['sim-only'].append((taut - pred)**2 / (2 * gamma))
            post = (pred / Vp + xhat / vx) / (1 / Vp + 1 / vx)
            if ucb <= om_s2: loss['bar-rule'].append((taut - pred)**2 / (2 * gamma))
            else: loss['bar-rule'].append((taut - post)**2 / (2 * gamma) + c)
            if omega2 <= om_s2: loss['oracle-rule'].append((taut - pred)**2 / (2 * gamma))
            else: loss['oracle-rule'].append((taut - post)**2 / (2 * gamma) + c)
        out = f"   omega2={omega2:.1f} (EE={vx/omega2:.1f}): " + "  ".join(f"{k} {np.mean(v_):.3f}±{np.std(v_)/np.sqrt(R):.3f}" for k, v_ in loss.items())
        print(out)

# ----------------------------------------------------------------------------------
# E12 (Round 3): the per-decision bar for threshold decisions — closed form and duel
# ----------------------------------------------------------------------------------
def psi(z):
    return stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z))

def evsi_threshold(m, V, vx, t0=0.0):
    """Preposterior EVSI for a go/no-go decision at threshold t0 with linear loss |tau-t0|:
    sigma_pp = V/sqrt(V+vx) (sd of the posterior-mean update); EVSI = sigma_pp * psi(|m-t0|/sigma_pp)."""
    spp = V / np.sqrt(V + vx)
    return spp * psi(np.abs(m - t0) / spp)

def e12(R=200000, seed=12):
    rng = np.random.default_rng(seed)
    print("\n=== E12: per-decision bar for threshold decisions ===")
    vx = 1.0
    print(" (a) closed form sigma_pp*psi(|m-t0|/sigma_pp) vs MC (t0=0)")
    for m in [0.0, 0.5, 1.0]:
        for V in [0.25, 1.0, 2.0]:
            tau = rng.normal(m, np.sqrt(V), R); th = tau + rng.normal(0, np.sqrt(vx), R)
            pm = m + V / (V + vx) * (th - m)
            lp = np.mean(np.where(m > 0, np.maximum(-tau, 0), np.maximum(tau, 0)))
            lq = np.mean(np.where(pm > 0, np.maximum(-tau, 0), np.maximum(tau, 0)))
            print(f"   m={m:.1f} V={V:.2f}: MC {lp-lq:.4f}  closed {evsi_threshold(m, V, vx):.4f}")
    print(" (b) duel on a stream of go/no-go decisions: global quadratic bar vs per-decision threshold bar vs oracle")
    c = 0.05; gamma = 1.0; E = 60
    om_s2 = omega_star2(c, vx, gamma)
    Rd = 20000
    for omega2 in [0.2, 0.5, 1.0, 2.0]:
        s, tau, th, v = gen_library(E, np.sqrt(omega2), vx, rng)
        om2, al, be, r, h, floor = excess_variance(th, s, v); ucb, _ = certify_ucb(th, s, v)
        loss = {'sim-only': 0.0, 'always': 0.0, 'global-bar': 0.0, 'per-decision': 0.0, 'oracle-per-dec': 0.0}
        nexp = {'global-bar': 0, 'per-decision': 0, 'oracle-per-dec': 0}
        for _ in range(Rd):
            st = rng.normal(); taut = st + rng.normal(0, np.sqrt(omega2)); pred = al + be * st
            Vp = max(om2, 0) + (max(om2, 0) + vx) * (1 / E + (st - s.mean())**2 / np.sum((s - s.mean())**2))
            xhat = rng.normal(taut, np.sqrt(vx))
            post = (pred / Vp + xhat / vx) / (1 / Vp + 1 / vx)
            L = lambda act: (max(-taut, 0) if act else max(taut, 0))
            loss['sim-only'] += L(pred > 0)
            loss['always'] += L(post > 0) + c
            run_g = ucb > om_s2
            loss['global-bar'] += (L(post > 0) + c) if run_g else L(pred > 0); nexp['global-bar'] += run_g
            run_p = evsi_threshold(pred, Vp, vx) > c
            loss['per-decision'] += (L(post > 0) + c) if run_p else L(pred > 0); nexp['per-decision'] += run_p
            Vo = omega2
            run_o = evsi_threshold(pred, Vo, vx) > c
            post_o = (pred / Vo + xhat / vx) / (1 / Vo + 1 / vx)
            loss['oracle-per-dec'] += (L(post_o > 0) + c) if run_o else L(pred > 0); nexp['oracle-per-dec'] += run_o
        print(f"   omega2={omega2:.1f} (global bar {om_s2:.2f}; certified={ucb<=om_s2}): " +
              "  ".join(f"{k} {v_/Rd:.4f}" for k, v_ in loss.items()) +
              "   | experiments run: " + ", ".join(f"{k} {v_/Rd:.2f}" for k, v_ in nexp.items()))

# ----------------------------------------------------------------------------------
# E13 (Round 3): robust (empirical fourth-moment) UCB under heavy tails; certification drag vs E
# ----------------------------------------------------------------------------------
def certify_ucb_robust(tau_hat, s, v, a=0.05, p=2):
    """Satterthwaite-matched chi-square UCB: keep the skewed chi-square shape but set the
    effective df from the empirical fourth moment of z_e = (r_e^2 - (1-h_e) v_e) E/(E-p);
    nu_eff = 2 (om2+vbar)^2 / Var_emp(om2_hat), capped at E-p."""
    E = len(s)
    om2, al, be, r, h, floor = excess_variance(tau_hat, s, v, p)
    vbar = v.mean()
    z_e = (r**2 - (1 - h) * v) * E / (E - p)
    var_emp = z_e.var(ddof=1) / E
    nu = min(max(2 * (om2 + vbar)**2 / var_emp, 2.0), E - p)
    return (om2 + vbar) * nu / stats.chi2.ppf(a, nu) - vbar, om2

def e13(R=4000, seed=13):
    rng = np.random.default_rng(seed)
    print("\n=== E13: robust UCB under heavy tails; certification drag ===")
    omega, vbar = 0.5, 0.8
    print(" (a) UCB95 coverage, chi-square vs Satterthwaite-matched chi-square UCB (a symmetric t-interval on om2_hat was tried first and under-covered even under normality: 0.875 at E=30)")
    for E in [30, 60, 120]:
        for tail, vs in [('normal', 0.0), ('t3', 0.0), ('t3', 0.8)]:
            c1 = c2 = 0; w1 = w2 = 0.0
            for _ in range(R):
                s, tau, th, v = gen_library(E, omega, vbar, rng, tail=tail, v_spread=vs)
                u1, _ = certify_ucb(th, s, v); u2, _ = certify_ucb_robust(th, s, v)
                c1 += u1 >= omega**2; c2 += u2 >= omega**2; w1 += u1; w2 += u2
            print(f"   E={E:3d} {tail:6s} v-spread {vs:.1f}: chi2 cov {c1/R:.3f} (mean UCB {w1/R:.2f}) | robust cov {c2/R:.3f} (mean UCB {w2/R:.2f})")
    print(" (b) certification drag: P(certify) and expected loss of the bar rule vs oracle as the library grows (omega2=0.2, bar 0.558, LQ)")
    vx, gamma, c = 1.0, 1.0, 0.1; om_s2 = omega_star2(c, vx, gamma); omega2 = 0.2
    for E in [10, 20, 40, 80, 160]:
        cert = 0; lb = lo = 0.0
        for _ in range(R):
            s, tau, th, v = gen_library(E, np.sqrt(omega2), vx, rng)
            om2, al, be, r, h, floor = excess_variance(th, s, v); ucb, _ = certify_ucb(th, s, v)
            st = rng.normal(); taut = st + rng.normal(0, np.sqrt(omega2)); pred = al + be * st
            Vp = max(om2, 0) + (max(om2, 0) + vx) * (1 / E + (st - s.mean())**2 / np.sum((s - s.mean())**2))
            xhat = rng.normal(taut, np.sqrt(vx)); post = (pred / Vp + xhat / vx) / (1 / Vp + 1 / vx)
            cert += ucb <= om_s2
            lb += (taut - pred)**2 / 2 if ucb <= om_s2 else (taut - post)**2 / 2 + c
            lo += (taut - pred)**2 / 2
        print(f"   E={E:3d}: P(certify) {cert/R:.3f}  bar-rule loss {lb/R:.4f}  oracle {lo/R:.4f}  drag {(lb-lo)/R:.4f}  [law E_exact for rho={omega2/om_s2:.2f}, nu={vx/om_s2:.2f}: {E_needed_exact(omega2/om_s2, vx/om_s2)}]")

# ----------------------------------------------------------------------------------
if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'quick'
    table = {'e1': e1, 'e2': e2, 'e3': e3, 'e4': e4, 'e5': e5, 'e6': e6, 'e7': e7, 'e8': e8, 'e9': e9, 'e10': e10, 'e11': e11, 'e12': e12, 'e13': e13}
    if arg == 'quick':
        for k in ['e1', 'e3', 'e4', 'e6', 'e8', 'e9', 'e11', 'e12', 'e13']: table[k]()
    elif arg == 'all':
        for k in table: table[k]()
    else:
        table[arg]()
