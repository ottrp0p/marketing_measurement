"""Deep dive 05 - A5/E6: always-on experiment-to-model fusion (closed loop).

The system under study: a Gaussian belief over the 12-dim MMM parameter vector
maintained by an extended Kalman filter with two measurement streams
(weekly aggregate revenue; geo-experiment readouts through dive 02's operator),
a decision-risk ledger rho_t = 1/2 tr(Omega D Sigma_t D^T), and a scheduler
that launches experiments when rho crosses a renewal-optimal threshold,
choosing the design by dive 04's LQ/gauss-u capture score.

Theory verified here:
  T1 (cadence law): with drift rate r = 1/2 tr(Omega D Q D^T) and capture
     fraction kappa, the optimal cadence is tau* = sqrt(c / (r (1/kappa - 1/2)))
     and steady-state excess loss ell* = 2 sqrt(c r (1/kappa - 1/2)).
  T2 (top-k portfolio ceiling): for any k readouts,
     kappa(A) <= sum of top-k eigenshares of A = Om^{1/2} Cov(u) Om^{1/2}.
  T3 (level/contrast split): near-constant spends make the observational
     stream inform mostly the decision-null level direction; drift accumulates
     in contrasts that only experiments (or spend perturbation) can drain.

Stages (run: python3 05_fusion_loop.py setup s1 s2 ...): state in /tmp/d05_state.pkl
"""
import importlib.util
import itertools
import pathlib
import os
import pickle
import sys
import time

import numpy as np

STATE = pathlib.Path(os.environ.get("D05_STATE", "/tmp/d05_state.pkl"))
_here = pathlib.Path(__file__).resolve().parent


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, _here / fn)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


d3 = _load("dive03", "03_decision_voi.py")
d4 = _load("dive04", "04_decision_voi_ext.py")

J, XBAR, TH0, B_TOT = d3.J, d3.XBAR, d3.TH0, d3.B_TOT
SIGY = d3.SIGMA          # weekly obs noise sd
SEW = 0.03               # per-week experiment readout noise (dives 02-04)
IB, IK, IS, IA = 0, 1, 2, 3   # per-channel param offsets in the (J,4) layout

# admissible box (flat 12-dim): [beta, K, S, alpha] x 3
LO = np.tile(np.array([0.05, 0.30, 1.05, 0.02]), J)
HI = np.tile(np.array([4.00, 5.00, 3.50, 0.95]), J)


def clipbox(v):
    return np.clip(v, LO, HI)


# ---------------------------------------------------------------- geometry ---
def make_omega_D(vm, b):
    """Omega (JxJ), D (Jx12), H at belief mean vm (flat) and spends b."""
    th = d3.unflat(vm)
    H = np.zeros(J)
    for j in range(J):
        d = 1e-5
        bp, bm = b.copy(), b.copy()
        bp[j] += d
        bm[j] -= d
        H[j] = -(d3.mroas(bp, th)[j] - d3.mroas(bm, th)[j]) / (2 * d)
    H = np.maximum(H, 1e-4)
    Hi = 1.0 / H
    Om = np.diag(Hi) - np.outer(Hi, Hi) / Hi.sum()
    D = np.zeros((J, 12))
    for k in range(12):
        d = 1e-5 * max(abs(vm[k]), 1e-3)
        vp, vn = vm.copy(), vm.copy()
        vp[k] += d
        vn[k] -= d
        D[:, k] = (d3.mroas(b, d3.unflat(vp)) - d3.mroas(b, d3.unflat(vn))) / (2 * d)
    return Om, D, H


def alloc_opt(th, b0):
    """Warm-started Nelder-Mead on the simplex (reference solver)."""
    from scipy.optimize import minimize
    def f(x):
        b3 = B_TOT - x[0] - x[1]
        if x[0] < 0 or x[1] < 0 or b3 < 0:
            return 1e6 + x[0] ** 2 + x[1] ** 2
        return -d3.rev_ss(np.array([x[0], x[1], b3]), th)
    res = minimize(f, np.asarray(b0[:2], float), method="Nelder-Mead",
                   options=dict(xatol=1e-5, fatol=1e-10, maxiter=500))
    b = np.array([res.x[0], res.x[1], B_TOT - res.x[0] - res.x[1]])
    return np.clip(b, 1e-6, None)


def proj_simplex(v, B=B_TOT):
    """Euclidean projection onto {b >= 0, sum b = B}."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - B
    rho = np.nonzero(u - css / (np.arange(len(v)) + 1) > 0)[0][-1]
    return np.maximum(v - css[rho] / (rho + 1.0), 0.0)


def alloc_pg(th, b0, iters=120, eta=0.5, floor=None):
    """Fast warm-started projected gradient ascent; grad rev_ss = mroas.
    floor: per-channel minimum spends (exploration floor)."""
    fl = np.zeros(J) if floor is None else floor
    Bf = B_TOT - fl.sum()
    b = np.maximum(b0.copy(), fl)
    for it in range(iters):
        g = d3.mroas(np.maximum(b, 1e-6), th)
        bn = fl + proj_simplex(b - fl + eta * g, Bf)
        if np.max(np.abs(bn - b)) < 1e-7:
            b = bn
            break
        b = bn
    return np.maximum(b, 1e-6)


# ------------------------------------------------------------- design menu ---
# (design tuple, cost multiplier): dual = two cells => 2x cost
MENU = [
    (("scalar", 0, 0.50 * XBAR[0], 8, 8), 1.0),
    (("scalar", 1, 0.50 * XBAR[1], 8, 8), 1.0),
    (("scalar", 2, 0.50 * XBAR[2], 8, 8), 1.0),
    (("scalar", 0, 1.00 * XBAR[0], 8, 8), 1.0),
    (("scalar", 1, 1.00 * XBAR[1], 8, 8), 1.0),
    (("scalar", 2, 1.00 * XBAR[2], 8, 8), 1.0),
    (("switch", (0, 0.50), (2, 0.60), 8, 8), 1.0),
    (("switch", (1, 0.40), (2, 0.50), 8, 8), 1.0),
    (("dual", (2, 0.60), (1, 0.40), 8, 8), 2.0),
]
WEAK = [(("scalar", 0, 0.05 * XBAR[0], 4, 0), 1.0),
        (("scalar", 1, 0.04 * XBAR[1], 4, 0), 1.0)]
W_EXP = 16  # weeks from launch to readout delivery


def stacked_readout(thJ, subset):
    """Concatenate readouts for a list of designs -> (L (n,m), se (m,))."""
    Ls, ses = [], []
    for d in subset:
        L, se = d4.readout(thJ, d)
        Ls.append(L)
        ses.append(se)
    return np.concatenate(Ls, 1), np.concatenate(ses)


def gauss_u_evsi(U, L, se, Om):
    """u-space Gaussian EVSI for (possibly multivariate) readout (dive 04 3.2)."""
    n = len(U)
    Uc = U - U.mean(0)
    Lc = L - L.mean(0)
    C_uL = Uc.T @ Lc / n
    S_L = Lc.T @ Lc / n + np.diag(se ** 2)
    K = np.linalg.solve(S_L, C_uL.T)
    return 0.5 * float(np.trace(Om @ C_uL @ K))


def design_scores(vm, Sig, b, rng, n=2000, menu=None):
    """kappa and EVSI_gauss for each menu design at the current belief."""
    menu = MENU if menu is None else menu
    thv = d3.sample_post(Sig, n, mean=vm)
    thJ = thv.reshape(n, J, 4)
    U = d4.u_stats(thJ, b)
    Om, D, H = make_omega_D(vm, b)
    evpi = d4.evpi_lq(U, Om)
    out = []
    for d, cm in menu:
        L, se = d4.readout(thJ, d)
        e = gauss_u_evsi(U, L, se, Om)
        out.append((d, cm, e, e / max(evpi, 1e-12)))
    return out, evpi


# ------------------------------------------------------- EKF closed loop -----
def h_parts(a, K, S):
    h = d3.hill(a, K, S)
    return h, h * (1 - h)


def obs_grad(vm, a, s):
    """Closed-form gradient of y_hat wrt flat params, given belief adstock a (J,)
    and sensitivity s = da/dalpha (J,). Returns (y_hat, g (12,))."""
    th = d3.unflat(vm)
    g = np.zeros(12)
    y = 0.0
    for j in range(J):
        b, K, S, al = th[j]
        h, hq = h_parts(a[j], K, S)
        y += b * h
        base = 4 * j
        g[base + IB] = h
        g[base + IK] = b * (-S / K) * hq
        g[base + IS] = b * hq * np.log(max(a[j], 1e-9) / K)
        g[base + IA] = b * (S * hq / max(a[j], 1e-9)) * s[j]
    return y, g


def cfg_default(**kw):
    cfg = dict(T=400, q=0.015, c=0.15, qf=None, drift="walk", jump_size=0.25,
               jump_rate=0.0, adapt_inflate=False, ingest="moment",
               n_score=1500, kappa_plan=0.40, menu=MENU, seed0=0,
               init_err=True, guard_G=100, jit_scale=0.03, b_floor_frac=0.0)
    cfg.update(kw)
    if cfg["qf"] is None:
        cfg["qf"] = cfg["q"]
    return cfg


def law(r, c, kappa):
    tau = np.sqrt(c / (r * (1 / kappa - 0.5)))
    ell = 2 * np.sqrt(c * r * (1 / kappa - 0.5))
    p = np.sqrt(r * c / (kappa * (1 - kappa / 2)))
    return tau, ell, p


def run_loop(policy, seed, cfg, Sig0, collect=False):
    """policy: never | cadence:<tau> | trigger | trigger_lqc | trigger_rand
    Returns per-week mean regret, cost rate, counts; CRN across policies."""
    T, q, c, qf = cfg["T"], cfg["q"], cfg["c"], cfg["qf"]
    r_init = np.random.default_rng([seed, 17])
    r_tru = np.random.default_rng([seed, 1])
    r_obs = np.random.default_rng([seed, 2])
    r_jit = np.random.default_rng([seed, 3])
    r_exp = np.random.default_rng([seed, 4])
    r_scr = np.random.default_rng([seed, 5])
    exp_noise = r_exp.standard_normal((60, 2))
    jumps = (r_tru.random(T) < cfg["jump_rate"])
    jump_sgn = r_tru.standard_normal((T, J))

    # truth and belief init
    v_true = d3.flat(TH0).copy()
    L0 = np.linalg.cholesky(Sig0 + 1e-12 * np.eye(12))
    z_init = r_init.standard_normal(12)
    v_mean = clipbox(d3.flat(TH0) + (L0 @ z_init if cfg["init_err"] else 0.0))
    Sig = Sig0.copy()
    Qf = np.zeros(12)
    Qf[[0, 4, 8]] = qf ** 2
    infl = 1.0
    z2_ewma = 1.0

    th_t = d3.unflat(v_true)
    a_true = XBAR / (1 - th_t[:, IA])
    thb = d3.unflat(v_mean)
    a_bel = XBAR / (1 - thb[:, IA])
    s_bel = XBAR / (1 - thb[:, IA]) ** 2
    jit = np.zeros(J)

    b = alloc_opt(d3.unflat(v_mean), XBAR)
    b_orc = alloc_opt(d3.unflat(v_true), XBAR)

    in_flight = None      # (deliver_week, design, cost_mult, L_true_mean, se)
    n_exp, cost_tot, k_exp = 0, 0.0, 0
    probes = np.zeros(J)
    regret = np.zeros(T)
    trace = {"rho": [], "lev": [], "con": [], "z": []} if collect else None
    rr_idx = 0            # round-robin pointer for cadence policy
    p_trig = cfg.get("p_trig", None)
    last_probe = np.zeros(J)   # week each channel was last touched by a test

    def touched(dsg):
        if dsg[0] == "scalar":
            return [dsg[1]]
        return [dsg[1][0], dsg[2][0]]

    for t in range(T):
        # ---- truth drift
        eps = r_tru.standard_normal(J)
        v_true[[0, 4, 8]] += q * eps
        if cfg["drift"] == "jumpwalk" and jumps[t]:
            v_true[[0, 4, 8]] += cfg["jump_size"] * jump_sgn[t]
        v_true = clipbox(v_true)
        th_t = d3.unflat(v_true)

        # ---- belief predict
        Sig[np.diag_indices(12)] += Qf * infl
        # ---- spends this week
        jit = 0.9 * jit + cfg["jit_scale"] * r_jit.standard_normal(J)
        x = np.clip(b * (1 + jit), 0.01, None)
        # ---- world
        a_true = x + th_t[:, IA] * a_true
        y = float(np.sum(th_t[:, IB] * d3.hill(a_true, th_t[:, IK], th_t[:, IS]))
                  + SIGY * r_obs.standard_normal())
        # ---- belief adstock + sensitivity, EKF obs update
        thb = d3.unflat(v_mean)
        s_bel = a_bel + thb[:, IA] * s_bel
        a_bel = x + thb[:, IA] * a_bel
        yh, g = obs_grad(v_mean, a_bel, s_bel)
        Svar = float(g @ Sig @ g) + SIGY ** 2
        z = (y - yh) / np.sqrt(Svar)
        Kg = (Sig @ g) / Svar
        v_mean = clipbox(v_mean + Kg * (y - yh))
        Sig = Sig - np.outer(Kg, g @ Sig)
        Sig = 0.5 * (Sig + Sig.T)
        z2_ewma = 0.98 * z2_ewma + 0.02 * z * z
        if cfg["adapt_inflate"]:
            infl = float(np.clip(z2_ewma ** 2, 1.0, 40.0))

        # ---- experiment delivery
        if in_flight is not None and t >= in_flight[0]:
            _, dsg, cm, Lmean, se = in_flight
            L_obs = Lmean + se * exp_noise[k_exp % 60, :len(se)]
            k_exp += 1
            n = cfg["n_score"]
            thv = d3.sample_post(Sig, n, mean=v_mean)
            thJ = thv.reshape(n, J, 4)
            Lm, _ = d4.readout(thJ, dsg)
            if cfg["ingest"] == "moment":
                Tc = thv - thv.mean(0)
                Lc = Lm - Lm.mean(0)
                C_tL = Tc.T @ Lc / n
                S_L = Lc.T @ Lc / n + np.diag(se ** 2)
                Kmat = np.linalg.solve(S_L, C_tL.T).T
                v_mean = clipbox(v_mean + Kmat @ (L_obs - Lm.mean(0)))
                Sig = Sig - Kmat @ C_tL.T
            else:  # linearized
                g_ops = np.zeros((len(se), 12))
                mu0, _ = d4.readout(v_mean.reshape(1, J, 4), dsg)
                for kk in range(12):
                    dd = 1e-4 * max(abs(v_mean[kk]), 1e-3)
                    vp = v_mean.copy()
                    vp[kk] += dd
                    mup, _ = d4.readout(vp.reshape(1, J, 4), dsg)
                    g_ops[:, kk] = (mup[0] - mu0[0]) / dd
                S_L = g_ops @ Sig @ g_ops.T + np.diag(se ** 2)
                Kmat = Sig @ g_ops.T @ np.linalg.inv(S_L)
                v_mean = clipbox(v_mean + Kmat @ (L_obs - mu0[0]))
                Sig = Sig - Kmat @ g_ops @ Sig
            Sig = 0.5 * (Sig + Sig.T)
            w, V = np.linalg.eigh(Sig)
            Sig = (V * np.maximum(w, 1e-10)) @ V.T
            in_flight = None

        # ---- reallocate + oracle + regret
        fl = cfg["b_floor_frac"] * XBAR if cfg["b_floor_frac"] > 0 else None
        b = alloc_pg(d3.unflat(v_mean), b, floor=fl)
        b_orc = alloc_pg(th_t, b_orc)
        regret[t] = d3.rev_ss(b_orc, th_t) - d3.rev_ss(b, th_t)

        # ---- scheduler (every 2 weeks)
        if t % 2 == 0 and policy != "never":
            # probe-point ledger: corner-blindness fix (h'(0)=0 under Hill S>1
            # zeroes D's column at b_j=0, hiding dark-channel risk)
            b_led = (np.maximum(b, 0.15 * XBAR)
                     if policy in ("trigger_pp", "trigger_lqcg") else b)
            Om, D, H = make_omega_D(v_mean, b_led)
            rho = 0.5 * float(np.trace(Om @ D @ Sig @ D.T))
            if collect:
                Vu = D @ Sig @ D.T
                trace["rho"].append(rho)
                trace["lev"].append(float(np.ones(J) @ Vu @ np.ones(J)) / J ** 2)
                trace["con"].append(rho)
            launch, forced_j = False, None
            if policy == "trigger_lqcg":
                # coverage guard as a CONSTRAINT: force a probe on any channel
                # that is stale or dark, regardless of the believed risk rho
                stale = t - last_probe
                bad = [(stale[j], j) for j in range(J)
                       if stale[j] > cfg["guard_G"]
                       or (b[j] < 0.2 * XBAR[j] and stale[j] > 30)]
                if bad:
                    forced_j = max(bad)[1]
            if policy.startswith("cadence"):
                tau = float(policy.split(":")[1])
                launch = (t >= (n_exp + 1) * tau)
            elif policy.startswith("trigger"):
                launch = (rho >= p_trig) or (forced_j is not None)
            if launch and in_flight is None and t < T - W_EXP:
                if forced_j is not None:
                    cand = [(cm, d0) for d0, cm in cfg["menu"]
                            if forced_j in touched(d0)]
                    cm, dsg = min(cand, key=lambda x: x[0])
                elif policy.startswith("cadence") or policy in ("trigger",
                                                               "trigger_pp"):
                    dsg, cm = cfg["menu"][rr_idx % len(cfg["menu"])]
                    rr_idx += 1
                elif policy == "trigger_rand":
                    ii = r_scr.integers(len(cfg["menu"]))
                    dsg, cm = cfg["menu"][ii]
                else:  # trigger_lqc / trigger_lqcg (no guard firing)
                    scores, _ = design_scores(v_mean, Sig, b, r_scr,
                                              n=cfg["n_score"],
                                              menu=cfg["menu"])
                    best = int(np.argmax([e / cm
                                          for (d0, cm, e, kap) in scores]))
                    dsg, cm = cfg["menu"][best]
                Lmean, se = d4.readout(th_t.reshape(1, J, 4), dsg)
                in_flight = (t + W_EXP, dsg, cm, Lmean[0], se)
                for jj in touched(dsg):
                    last_probe[jj] = t
                    probes[jj] += 1
                n_exp += 1
                cost_tot += cfg["c"] * cm
        if collect and policy == "never" and t % 2 == 0:
            Om, D, H = make_omega_D(v_mean, b)
            Vu = D @ Sig @ D.T
            trace["rho"].append(0.5 * float(np.trace(Om @ Vu)))
            trace["lev"].append(float(np.ones(J) @ Vu @ np.ones(J)) / J ** 2)
            trace["z"].append(z)

    out = dict(mean_regret=float(regret[50:].mean()),
               loss=float(regret[50:].mean() + cost_tot / T),
               cost_rate=cost_tot / T, n_exp=n_exp, probes=probes,
               late_regret=float(regret[-100:].mean()),
               regret=regret if collect else None, trace=trace)
    return out


# ================================================================= stages ====
def stage_setup(S):
    t0 = time.time()
    Sig0, X = d3.build_posterior()
    b_pr = alloc_opt(TH0, XBAR)
    Om, D, H = make_omega_D(d3.flat(TH0), b_pr)
    Q = np.zeros(12)
    Q[[0, 4, 8]] = 1.0     # unit q^2; scale later
    r_unit = 0.5 * float(np.trace(Om @ D @ np.diag(Q) @ D.T))  # r = r_unit * q^2
    S.update(Sig0=Sig0, b_pr=b_pr, Om=Om, D=D, H=H, r_unit=r_unit)
    print(f"setup: b_pr={b_pr.round(3)} r_unit={r_unit:.4f} "
          f"(r=q^2*{r_unit:.4f}/wk)  H={H.round(3)}  {time.time()-t0:.1f}s")
    # validate fast allocator vs reference on perturbed draws
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(60):
        thv = clipbox(d3.flat(TH0) * (1 + 0.3 * rng.standard_normal(12)))
        th = d3.unflat(thv)
        b1 = alloc_pg(th, XBAR, iters=400)
        b2 = alloc_opt(th, b1)
        worst = max(worst, d3.rev_ss(b2, th) - d3.rev_ss(b1, th))
    print(f"  alloc_pg vs NM: worst revenue shortfall {worst:.2e}")
    for q in (0.0075, 0.015, 0.03):
        for c in (0.05, 0.15, 0.45):
            tau, ell, p = law(r_unit * q * q, c, 0.40)
            print(f"  q={q} c={c}: tau*={tau:6.1f}wk ell*={ell:.5f} p*={p:.5f}")


def stage_s1(S):
    """Portfolio ceiling (T2), submodularity audit, greedy vs optimal."""
    rng = np.random.default_rng(51)
    n = 6000
    thv = d3.sample_post(S["Sig0"], n)
    thJ = thv.reshape(n, J, 4)
    U = d4.u_stats(thJ, S["b_pr"])
    Om = S["Om"]
    evpi = d4.evpi_lq(U, Om)
    ev, eff, Om12, W = d4.spectrum(U, Om)
    shares = ev / ev.sum()
    print(f"s1: EVPI_LQ={evpi:.5f} spectrum={ev.round(5)} shares={shares.round(3)}")

    designs = [d for d, cm in MENU]
    f = {}
    for k in range(1, 4):
        for A in itertools.combinations(range(len(designs)), k):
            L, se = stacked_readout(thJ, [designs[i] for i in A])
            f[A] = gauss_u_evsi(U, L, se, Om)
    # split-half stability of f on singles
    half = n // 2
    for A in [(0,), (6,), (8,)]:
        L, se = stacked_readout(thJ[:half], [designs[i] for i in A])
        f1 = gauss_u_evsi(U[:half], L, se, Om)
        print(f"  stability {A}: full={f[A]:.5f} half={f1:.5f}")
    # T2 ceiling check for all subsets
    viol_ceil = 0
    for A, val in f.items():
        cap = ev[:len(A)].sum() / ev.sum()
        if val / evpi > cap + 1e-9:
            viol_ceil += 1
    print(f"  T2 ceiling violations: {viol_ceil}/{len(f)}")
    # diminishing returns audit: f(A+d) - f(A) vs f(d) - f(empty)
    worst, n_viol, n_checks = 0.0, 0, 0
    worst_case = None
    for A in list(f):
        for dnew in range(len(designs)):
            if dnew in A or len(A) > 2:
                continue
            Au = tuple(sorted(A + (dnew,)))
            gain_late = f[Au] - f[A]
            gain_early = f[(dnew,)]
            n_checks += 1
            ratio = gain_late / max(gain_early, 1e-12)
            if gain_late > gain_early + 1e-6:
                n_viol += 1
                if ratio > worst:
                    worst, worst_case = ratio, (A, dnew)
    print(f"  submodularity: {n_viol}/{n_checks} DR violations; "
          f"worst late/early gain ratio={worst:.3f} at {worst_case}")
    # greedy vs optimal, k=1..3 (unit costs; dual charged 2 -> compare per cost)
    for k in (1, 2, 3):
        best = max((v, A) for A, v in f.items() if len(A) == k)
        # greedy
        cur, curv = (), 0.0
        for _ in range(k):
            cand = [(f[tuple(sorted(cur + (i,)))], i)
                    for i in range(len(designs)) if i not in cur]
            v, i = max(cand)
            cur = tuple(sorted(cur + (i,)))
            curv = v
        print(f"  k={k}: greedy={curv:.5f} ({curv/evpi:.3f}) "
              f"opt={best[0]:.5f} ({best[0]/evpi:.3f}) ratio={curv/best[0]:.4f} "
              f"greedy_set={cur} opt_set={best[1]}")
    S["f_static"] = f
    S["evpi0"] = evpi
    S["spec0"] = ev


def stage_s2(S):
    """No-experiment closed loop: filter calibration, rho growth, T3 split."""
    cfg = cfg_default()
    seeds = range(24)
    rhos, levs, zs, regs = [], [], [], []
    t0 = time.time()
    for sd in seeds:
        out = run_loop("never", sd, cfg, S["Sig0"], collect=True)
        rhos.append(out["trace"]["rho"])
        levs.append(out["trace"]["lev"])
        zs.extend(out["trace"]["z"])
        regs.append(out["regret"])
    rho = np.array(rhos).mean(0)
    lev = np.array(levs).mean(0)
    zs = np.array(zs)
    regs = np.array(regs)
    r = S["r_unit"] * cfg["q"] ** 2
    wk = 2 * np.arange(len(rho))
    print(f"s2 ({time.time()-t0:.0f}s): z mean={zs.mean():.3f} sd={zs.std():.3f} "
          f"(want ~0,1)")
    for w in (0, 25, 50, 100, 150, 199):
        pred = rho[0] + r * wk[w]
        print(f"  wk={wk[w]:3d}: rho={rho[w]:.5f} linear-pred={pred:.5f} "
              f"lev={lev[w]:.5f} mean_regret={regs[:, wk[w]].mean():.5f}")
    # does rho predict realized regret? (quadratic theory: E[regret] ~ rho)
    late = regs[:, 200:].mean()
    rho_late = rho[100:].mean()
    print(f"  late mean regret={late:.5f} vs mean rho={rho_late:.5f} "
          f"ratio={late/rho_late:.2f}")
    S["s2"] = dict(rho=rho, lev=lev, r=r)


def _duel(policies, seeds, cfg, Sig0, label):
    res = {p: [] for p in policies}
    t0 = time.time()
    for sd in seeds:
        for p in policies:
            res[p].append(run_loop(p, sd, cfg, Sig0))
    print(f"{label} ({time.time()-t0:.0f}s):")
    base = np.array([o["loss"] for o in res[policies[0]]])
    for p in policies:
        li = np.array([o["loss"] for o in res[p]])
        ne = np.mean([o["n_exp"] for o in res[p]])
        d = li - base
        print(f"  {p:16s} loss={li.mean():.5f}+-{li.std()/np.sqrt(len(li)):.5f} "
              f"med={np.median(li):.5f} nexp={ne:5.1f} "
              f"dloss={d.mean():+.5f}+-{d.std()/np.sqrt(len(d)):.5f} "
              f"dmed={np.median(d):+.5f}")
    return res


def stage_s3(S):
    """Cadence grid in the CLEAN-start regime: empirical tau* vs law."""
    cfg = cfg_default(init_err=False, T=500)
    r = S["r_unit"] * cfg["q"] ** 2
    tau_l, ell_l, p_l = law(r, cfg["c"], cfg["kappa_plan"])
    print(f"s3(clean): law tau*={tau_l:.1f} ell*={ell_l:.5f} p*={p_l:.5f}")
    pols = ["never"] + [f"cadence:{t}" for t in (15, 25, 40, 60, 90, 140, 220)]
    S["s3"] = _duel(pols, range(24), cfg, S["Sig0"], "s3 cadence grid (clean)")


def stage_s4(S, sub="realistic"):
    """Policy duel; sub in {realistic, clean}."""
    ie = (sub == "realistic")
    cfg = cfg_default(init_err=ie)
    r = S["r_unit"] * cfg["q"] ** 2
    tau_l, _, p = law(r, cfg["c"], cfg["kappa_plan"])
    cfg["p_trig"] = p
    pols = ["never", f"cadence:{tau_l:.0f}", "trigger", "trigger_pp",
            "trigger_lqc", "trigger_lqcg", "trigger_rand"]
    S[f"s4_{sub}"] = _duel(pols, range(32), cfg, S["Sig0"],
                           f"s4 policy duel ({sub} start)")


def stage_s5(S, sub="0"):
    """Scaling sweep over (q, c) with trigger policy; sub selects q index."""
    rows = S.get("s5", [])
    for q in [(0.0075, 0.015, 0.03)[int(sub)]]:
        for c in (0.05, 0.15, 0.45):
            cfg = cfg_default(q=q, c=c)
            r = S["r_unit"] * q * q
            _, ell, p = law(r, c, cfg["kappa_plan"])
            cfg["p_trig"] = p
            cfg["init_err"] = False
            cfg["T"] = 500
            losses, nexps, reg = [], [], []
            for sd in range(16):
                o = run_loop("trigger", sd, cfg, S["Sig0"])
                o0 = run_loop("never", sd, cfg, S["Sig0"])
                losses.append(o["loss"])
                reg.append(o0["loss"])
                nexps.append(o["n_exp"])
            rows.append((q, c, float(np.median(losses)),
                         np.std(losses) / np.sqrt(16), ell,
                         np.mean(nexps), float(np.median(reg))))
            print(f"s5: q={q} c={c} med_loss={rows[-1][2]:.5f}"
                  f"(sd {rows[-1][3]:.5f}) law_ell*={ell:.5f} "
                  f"nexp={rows[-1][5]:.1f} never_med={rows[-1][6]:.5f}")
    S["s5"] = rows


def stage_s6(S):
    """Ingestion: moment-matched vs linearized (paired)."""
    cfg = cfg_default()
    r = S["r_unit"] * cfg["q"] ** 2
    _, _, p = law(r, cfg["c"], cfg["kappa_plan"])
    cfg["p_trig"] = p
    diffs = []
    for sd in range(20):
        cfg["ingest"] = "moment"
        om = run_loop("trigger", sd, cfg, S["Sig0"])
        cfg["ingest"] = "linear"
        ol = run_loop("trigger", sd, cfg, S["Sig0"])
        diffs.append(ol["loss"] - om["loss"])
    d = np.array(diffs)
    print(f"s6 ingestion: linear - moment loss = {d.mean():+.5f}"
          f"+-{d.std()/np.sqrt(len(d)):.5f}")
    S["s6"] = d


def stage_s7(S):
    """Robustness: filter-Q misspecification and burst drift."""
    r0 = S["r_unit"] * 0.015 ** 2
    _, _, p = law(r0, 0.15, 0.40)
    for tag, kw in [("qf=q/4", dict(qf=0.015 / 4)), ("qf=4q", dict(qf=0.06)),
                    ("bursts", dict(q=0.003, drift="jumpwalk", jump_rate=0.01)),
                    ("bursts+inflate", dict(q=0.003, drift="jumpwalk",
                                            jump_rate=0.01, adapt_inflate=True))]:
        cfg = cfg_default(**kw)
        cfg["p_trig"] = p
        ls, ns = [], []
        for sd in range(16):
            o = run_loop("trigger", sd, cfg, S["Sig0"])
            ls.append(o["loss"])
            ns.append(o["n_exp"])
        print(f"s7 {tag:15s}: loss={np.mean(ls):.5f}+-"
              f"{np.std(ls)/np.sqrt(16):.5f} nexp={np.mean(ns):.1f}")


def stage_s8(S):
    """Negative/power controls: zero drift; weak-design menu."""
    r0 = S["r_unit"] * 0.015 ** 2
    _, _, p = law(r0, 0.15, 0.40)
    cfg = cfg_default(q=1e-6)
    cfg["p_trig"] = p
    ls, ns = [], []
    for sd in range(16):
        o = run_loop("trigger", sd, cfg, S["Sig0"])
        ls.append(o["loss"])
        ns.append(o["n_exp"])
    print(f"s8 zero-drift trigger: loss={np.mean(ls):.5f} nexp={np.mean(ns):.2f}")
    cfg = cfg_default(menu=WEAK)
    cfg["p_trig"] = p
    ls2, ns2 = [], []
    for sd in range(16):
        o = run_loop("trigger", sd, cfg, S["Sig0"])
        ls2.append(o["loss"])
        ns2.append(o["n_exp"])
    cfgb = cfg_default()
    l0 = [run_loop("never", sd, cfgb, S["Sig0"])["loss"] for sd in range(16)]
    print(f"s8 weak-menu trigger: loss={np.mean(ls2):.5f}+-"
          f"{np.std(ls2)/np.sqrt(16):.5f} nexp={np.mean(ns2):.1f} "
          f"(never={np.mean(l0):.5f})")


def stage_s9(S):
    """Attack control: does a spend floor alone (no experiments) kill traps?"""
    r = S["r_unit"] * 0.015 ** 2
    _, _, p = law(r, 0.15, 0.40)
    for tag, kw, pol in [("never", {}, "never"),
                         ("never+floor15", dict(b_floor_frac=0.15), "never"),
                         ("never+floor30", dict(b_floor_frac=0.30), "never"),
                         ("trigger+floor15", dict(b_floor_frac=0.15), "trigger")]:
        cfg = cfg_default(**kw)
        cfg["p_trig"] = p
        ls = []
        for sd in range(32):
            ls.append(run_loop(pol, sd, cfg, S["Sig0"])["loss"])
        ls = np.array(ls)
        print(f"s9 {tag:16s}: loss={ls.mean():.5f}+-{ls.std()/np.sqrt(32):.5f} "
              f"med={np.median(ls):.5f}")


def stage_s3b(S):
    """Cadence law under near-zero jitter (obs stream cannot see contrasts)."""
    cfg = cfg_default(init_err=False, T=500, jit_scale=0.004)
    r = S["r_unit"] * cfg["q"] ** 2
    tau_l, ell_l, _ = law(r, cfg["c"], cfg["kappa_plan"])
    print(f"s3b(no-jitter): law tau*={tau_l:.1f} ell*={ell_l:.5f}")
    pols = ["never"] + [f"cadence:{t}" for t in (15, 25, 40, 60, 90, 140, 220)]
    S["s3b"] = _duel(pols, range(24), cfg, S["Sig0"], "s3b cadence (no jitter)")


STAGES = dict(setup=stage_setup, s1=stage_s1, s2=stage_s2, s3=stage_s3,
              s3b=stage_s3b, s4=stage_s4, s5=stage_s5, s6=stage_s6,
              s7=stage_s7, s8=stage_s8, s9=stage_s9)

if __name__ == "__main__":
    S = pickle.load(open(STATE, "rb")) if STATE.exists() else {}
    for arg in sys.argv[1:]:
        print(f"===== {arg} =====")
        if ":" in arg:
            name, sub = arg.split(":")
            STAGES[name](S, sub)
        else:
            STAGES[arg](S)
        pickle.dump(S, open(STATE, "wb"))
