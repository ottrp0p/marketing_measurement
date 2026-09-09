"""Deep dive 04 - EXTEND of dive 03 (A4): where the EVSI power plateau ends.

F1: LQC ("LQ-conditional") EVSI estimator - exact *nonlinear* conditioning via
    importance weights (as in nested MC) but with the inner optimization replaced
    by dive 03's closed-form LQ decision layer. No nested optimization.
    Validated against rank-1 closed form and exact nested MC.
F2: effective-rank theory of the plateau: decision-relevant covariance
    A = Omega^{1/2} Cov(u) Omega^{1/2}, u = marginal-ROAS vector at the prior
    optimum. Plateau <=> effrank(A) ~ 1 and all readouts align with the top
    contrast.
F3: boundary: block-independent / variance-equalized posteriors (effrank -> 2)
    - channel choice matters, then a hard single-readout ceiling appears.
F4: the fix at the boundary: multivariate readouts (trajectory; simultaneous
    two-channel experiments) - LQC handles these natively.
F5: free total budget: Omega_free = v * diag(1/H) has no 1-null direction;
    level information rehabilitates; rankings change.

Model, posterior, operator all imported from code/03_decision_voi.py.
"""
import importlib.util
import pathlib
import pickle
import sys
import time

import numpy as np

STATE = pathlib.Path("/tmp/d04_state.pkl")   # stage-to-stage scratch state

_p = pathlib.Path(__file__).resolve().parent / "03_decision_voi.py"
_spec = importlib.util.spec_from_file_location("dive03", _p)
d3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d3)

rng = np.random.default_rng(4)
J = d3.J
XBAR = d3.XBAR
TH0 = d3.TH0
SEW = 0.03          # per-week readout noise sd; scalar se = SEW*sqrt(Te+P) as in dive 03


# ---------- vectorized experiment operator (per-week lift paths) --------------
def lift_paths(thJ, j, delta, Te, P):
    """thJ: (n,J,4) -> (n, Te+P) per-week incremental-revenue path for a
    uniform pulse delta on channel j over Te weeks, observed Te+P weeks."""
    b, K, S, al = thJ[:, j, 0], thJ[:, j, 1], thJ[:, j, 2], thJ[:, j, 3]
    xb = XBAR[j]
    W = Te + P
    a0 = xb / (1 - al)
    a1 = a0.copy()
    out = np.empty((len(b), W))
    for t in range(W):
        a0 = xb + al * a0
        a1 = xb + (delta if t < Te else 0.0) + al * a1
        out[:, t] = b * (d3.hill(a1, K, S) - d3.hill(a0, K, S))
    return out


def readout(thJ, design):
    """design -> (Lmat (n,m), se_vec (m,)).
    kinds: ("scalar", j, dl, Te, P) - windowed total lift, se = SEW*sqrt(W)
           ("traj",   j, dl, Te, P) - per-week path, se = SEW per week
           ("dual", (j1,dl1), (j2,dl2), Te, P) - two simultaneous scalar tests
    """
    kind = design[0]
    if kind == "scalar":
        _, j, dl, Te, P = design
        L = lift_paths(thJ, j, dl, Te, P).sum(1, keepdims=True)
        return L, np.array([SEW * np.sqrt(Te + P)])
    if kind == "traj":
        _, j, dl, Te, P = design
        return lift_paths(thJ, j, dl, Te, P), np.full(Te + P, SEW)
    if kind == "dual":
        _, (j1, d1), (j2, d2), Te, P = design
        L1 = lift_paths(thJ, j1, d1, Te, P).sum(1)
        L2 = lift_paths(thJ, j2, d2, Te, P).sum(1)
        se = SEW * np.sqrt(Te + P)
        return np.stack([L1, L2], 1), np.array([se, se])
    if kind == "switch":
        # budget-neutral reallocation pulse: +d1 on j1, -d2 on j2, ONE readout
        _, (j1, d1), (j2, d2), Te, P = design
        L = (lift_paths(thJ, j1, d1, Te, P)
             + lift_paths(thJ, j2, -d2, Te, P)).sum(1, keepdims=True)
        return L, np.array([SEW * np.sqrt(Te + P)])
    raise ValueError(kind)


# ---------- LQC estimator -----------------------------------------------------
def u_stats(thJ, b_bar):
    """Decision-sufficient statistic: marginal ROAS vector at the prior optimum."""
    a = b_bar[None, :] / (1 - thJ[:, :, 3])
    return (thJ[:, :, 0] * d3.hill_d1(a, thJ[:, :, 1], thJ[:, :, 2])
            / (1 - thJ[:, :, 3]))


def lqc_evsi(U, Lmat, se_vec, Omega, oi, on, se_mult=1.0):
    """LQ-conditional EVSI: exact IS conditioning, quadratic decision layer.
    Split-half cross product debiases the self-normalized-IS noise term.
    oi: (n_outer,) indices of outer 'true' draws; on: (n_outer, m_max) N(0,1)."""
    n, m = Lmat.shape
    se = se_vec * se_mult
    ev = np.arange(0, n, 2)
    od = np.arange(1, n, 2)
    ubar = U.mean(0)
    vals = np.empty(len(oi))
    for t, i in enumerate(oi):
        ell = Lmat[i] + se * on[t, :m]
        logw = -0.5 * np.sum(((Lmat - ell[None, :]) / se[None, :]) ** 2, axis=1)
        w = np.exp(logw - logw.max())
        we = w[ev] / w[ev].sum()
        wo = w[od] / w[od].sum()
        ue = we @ U[ev] - ubar
        uo = wo @ U[od] - ubar
        vals[t] = 0.5 * float(ue @ Omega @ uo)
    return float(vals.mean()), float(vals.std() / np.sqrt(len(oi)))


def evpi_lq(U, Omega):
    Uc = U - U.mean(0)
    return 0.5 * float(np.einsum("ij,jk,ik->", Uc, Omega, Uc) / len(U))


def _opt_fast(obj, b_start):
    """Warm-started single-run Nelder-Mead on the simplex (2 free vars)."""
    from scipy.optimize import minimize
    B = d3.B_TOT
    res = minimize(lambda x: -obj(np.array([x[0], x[1], B - x[0] - x[1]])),
                   b_start[:2], method="Nelder-Mead",
                   options=dict(xatol=1e-5, fatol=1e-9, maxiter=600))
    b = np.array([res.x[0], res.x[1], B - res.x[0] - res.x[1]])
    return np.maximum(b, 0.0)


# ---------- generalized nested MC (multivariate readouts, real optimization) --
def nested_mc(thJ, Lmat, se_vec, b_pr, oi, on, se_mult=1.0):
    n, m = Lmat.shape
    se = se_vec * se_mult
    ev = np.arange(0, n, 2)
    od = np.arange(1, n, 2)
    vals = np.empty(len(oi))
    for t, i in enumerate(oi):
        ell = Lmat[i] + se * on[t, :m]
        logw = -0.5 * np.sum(((Lmat - ell[None, :]) / se[None, :]) ** 2, axis=1)
        w = np.exp(logw - logw.max())
        we = w[ev] / w[ev].sum()
        wo = w[od] / w[od].sum()
        b_post = _opt_fast(
            lambda b: float(we @ d3.rev_ss_batch(b, thJ[ev])), b_pr)
        vals[t] = (float(wo @ d3.rev_ss_batch(b_post, thJ[od]))
                   - float(wo @ d3.rev_ss_batch(b_pr, thJ[od])))
    return float(vals.mean()), float(vals.std() / np.sqrt(len(oi)))


# ---------- decision-relevant spectrum (F2) ----------------------------------
def spectrum(U, Omega):
    """A = Om^{1/2} Cov(u) Om^{1/2}: eigenvalues, effective rank, and the
    principal decision contrasts c_k (in u-space)."""
    lam, V = np.linalg.eigh(Omega)
    lam = np.maximum(lam, 0)
    Om12 = V @ np.diag(np.sqrt(lam)) @ V.T
    Su = np.cov(U.T)
    A = Om12 @ Su @ Om12
    ev, W = np.linalg.eigh(A)
    ev = ev[::-1]
    W = W[:, ::-1]
    eff = ev.sum() ** 2 / (ev ** 2).sum()
    return ev, eff, Om12, W


def design_grid(fracs=(0.25, 0.5, 1.0), tes=(4, 8), ps=(0, 8)):
    return [("scalar", j, f * XBAR[j], Te, P)
            for j in range(J) for f in fracs for Te in tes for P in ps]


def fmt(d):
    if d[0] == "scalar":
        return f"ch{d[1]} dl={d[2]:.2f} Te={d[3]} P={d[4]}"
    if d[0] == "traj":
        return f"TRAJ ch{d[1]} dl={d[2]:.2f} Te={d[3]} P={d[4]}"
    tag = d[0].upper()
    sgn = "-" if d[0] == "switch" else ""
    return (f"{tag} ch{d[1][0]}&ch{d[2][0]} dl=({d[1][1]:.2f},{sgn}{d[2][1]:.2f}) "
            f"Te={d[3]} P={d[4]}")


# =============================================================================
def run_grid_lqc(thJ, U, Omega, designs, oi, on, extras=()):
    out = []
    for d in designs:
        Lm, sev = readout(thJ, d)
        e, se_e = lqc_evsi(U, Lm, sev, Omega, oi, on)
        out.append((d, e, se_e))
    for name, d, sm in extras:
        Lm, sev = readout(thJ, d)
        e, se_e = lqc_evsi(U, Lm, sev, Omega, oi, on, se_mult=sm)
        out.append(((name, d), e, se_e))
    return out


MC_SUB = [0, 10, 17, 22, 25, 34, 35]      # grid indices spot-checked by nested MC
EXTRAS = [("weak-pulse ch2 dl=0.06", ("scalar", 2, 0.06, 8, 8), 1.0),
          ("underpowered se x6    ", ("scalar", 2, 0.6, 8, 8), 6.0),
          ("tiny-short ch0        ", ("scalar", 0, 0.1, 2, 0), 1.0)]


def save_state(**kw):
    st = pickle.load(open(STATE, "rb")) if STATE.exists() else {}
    st.update(kw)
    pickle.dump(st, open(STATE, "wb"))


def load_state():
    return pickle.load(open(STATE, "rb"))


def stage_setup(n_inner=6000, n_outer=1200):
    """Posterior, draws, decision objects for baseline + the two F3 posteriors."""
    Sig, X = d3.build_posterior()
    ths = d3.sample_post(Sig, n_inner)
    thJ = ths.reshape(-1, J, 4)
    b_bar, H, D, Omega = d3.decision_objects(Sig)
    b_pr = d3.opt_simplex(lambda b: float(np.mean(d3.rev_ss_batch(b, thJ))), b_bar)
    U = u_stats(thJ, b_pr)
    oi = rng.integers(0, n_inner, n_outer)
    on = rng.standard_normal((n_outer, 16))
    st = dict(Sig=Sig, thJ=thJ, b_pr=b_pr, H=H, D=D, Omega=Omega, U=U,
              oi=oi, on=on, evpi=evpi_lq(U, Omega))
    # F3 posteriors
    b_bar0 = d3.opt_simplex(lambda b: d3.rev_ss(b, TH0))
    for tag, Sg in [("bd", make_sigma_bd(Sig)),
                    ("eq", make_sigma_bd(Sig, True, b_bar0))]:
        thJ2 = d3.sample_post(Sg, n_inner).reshape(-1, J, 4)
        _, H2, D2, Om2 = d3.decision_objects(Sg)
        bp2 = d3.opt_simplex(lambda b: float(np.mean(d3.rev_ss_batch(b, thJ2))), b_bar0)
        U2 = u_stats(thJ2, bp2)
        st.update({f"Sig_{tag}": Sg, f"thJ_{tag}": thJ2, f"b_pr_{tag}": bp2,
                   f"Omega_{tag}": Om2, f"U_{tag}": U2,
                   f"evpi_{tag}": evpi_lq(U2, Om2)})
    pickle.dump(st, open(STATE, "wb"))
    print(f"setup done. EVPI_LQ base={st['evpi']:.5f} "
          f"bd={st['evpi_bd']:.5f} eq={st['evpi_eq']:.5f}")


def stage_f1():
    st = load_state()
    thJ, U, Omega, oi, on, evpi = (st["thJ"], st["U"], st["Omega"],
                                   st["oi"], st["on"], st["evpi"])
    print("=" * 78)
    print("F1: LQC estimator vs rank-1 closed form (full width)")
    print(f"  EVPI_LQ (sample) = {evpi:.5f}")
    designs = design_grid()
    t0 = time.time()
    res = run_grid_lqc(thJ, U, Omega, designs, oi, on, EXTRAS)
    t_lqc = time.time() - t0
    closed = []
    for d in designs:
        _, j, dl, Te, P = d
        g = d3.lift_grad(j, XBAR[j], dl, Te, P)
        closed.append(d3.evsi_closed(st["Sig"], st["D"], Omega, g,
                                     d3.se_of_design(Te, P)))
    lq = np.array([r[1] for r in res[:len(designs)]])
    cl = np.array(closed)
    from scipy.stats import kendalltau, spearmanr
    print(f"  grid of {len(designs)}: corr(LQC, rank-1 closed): "
          f"pearson {np.corrcoef(lq, cl)[0, 1]:.3f} "
          f"spearman {spearmanr(lq, cl).statistic:.3f} "
          f"kendall {kendalltau(lq, cl).statistic:.3f}")
    print(f"  LQC grid time {t_lqc:.1f}s "
          f"({len(designs)+3} designs x {len(oi)} outer draws)")
    for (name, d), e, se_e in res[len(designs):]:
        print(f"  control {name:26s} LQC = {e:9.5f} (+-{se_e:.5f})")
    powered = [r for r in res[:len(designs)]
               if r[0][2] >= 0.4 * XBAR[r[0][1]] and r[0][3] >= 8]
    pe = np.array([r[1] for r in powered])
    print(f"  plateau check, powered designs (dl>=0.4 xbar, Te=8): n={len(pe)}, "
          f"max/min LQC = {pe.max()/pe.min():.2f}, "
          f"kappa in [{pe.min()/evpi:.2f}, {pe.max()/evpi:.2f}]")
    save_state(lq=lq, cl=cl, res=res)


def stage_mc(which):
    """Nested-MC spot checks, chunked to fit the call budget. which in {0,1,2}."""
    st = load_state()
    thJ, b_pr, oi, on = st["thJ"], st["b_pr"], st["oi"], st["on"]
    designs = design_grid()
    chunk = MC_SUB[which * 3:(which + 1) * 3]
    n_mc = 240
    mc = st.get("mc", {})
    for k in chunk:
        t0 = time.time()
        Lm, sev = readout(thJ, designs[k])
        mc[k] = nested_mc(thJ, Lm, sev, b_pr, oi[:n_mc], on[:n_mc])
        print(f"  nested-MC {fmt(designs[k]):28s} = {mc[k][0]:.5f} "
              f"(+-{mc[k][1]:.5f})  [{time.time()-t0:.0f}s]")
    save_state(mc=mc)


def stage_f1_compare():
    st = load_state()
    lq, cl, mc = st["lq"], st["cl"], st["mc"]
    designs = design_grid()
    from scipy.stats import spearmanr
    print("  " + f"{'design':30s} {'LQC':>9s} {'rank-1':>9s} {'nestedMC':>9s}")
    for k in MC_SUB:
        print(f"  {fmt(designs[k]):30s} {lq[k]:9.5f} {cl[k]:9.5f} "
              f"{mc[k][0]:9.5f} (+-{mc[k][1]:.5f})")
    lqs = np.array([lq[k] for k in MC_SUB])
    cls = np.array([cl[k] for k in MC_SUB])
    mcs = np.array([mc[k][0] for k in MC_SUB])
    print(f"  corr(LQC, MC):    pearson {np.corrcoef(lqs, mcs)[0,1]:.3f} "
          f"spearman {spearmanr(lqs, mcs).statistic:.3f}  "
          f"median MC/LQC = {np.median(mcs/lqs):.2f}")
    print(f"  corr(rank-1, MC): pearson {np.corrcoef(cls, mcs)[0,1]:.3f} "
          f"spearman {spearmanr(cls, mcs).statistic:.3f}  "
          f"median MC/closed = {np.median(mcs/cls):.2f}")


def stage_f2():
    st = load_state()
    thJ, U, Omega, evpi, res = st["thJ"], st["U"], st["Omega"], st["evpi"], st["res"]
    print("=" * 78)
    print("F2: the plateau explained - decision-relevant spectrum and alignment")
    ev, eff, Om12, W = spectrum(U, Omega)
    print(f"  eig(A) = {np.round(ev, 5)};  effective rank = {eff:.2f};  "
          f"top-contrast share lam1/tr = {ev[0]/ev.sum():.2f}")
    c1 = Om12 @ W[:, 0]
    c2 = Om12 @ W[:, 1]
    print(f"  principal contrast c1 (channels) = {np.round(c1 / np.abs(c1).max(), 2)}")
    s1 = (U - U.mean(0)) @ c1
    s2 = (U - U.mean(0)) @ c2
    print("  alignment of scalar readouts with the principal contrast:")
    for d, e, _ in res[:36]:
        if d[0] == "scalar" and d[3] == 8 and d[4] == 8:
            Lm, _ = readout(thJ, d)
            r1 = np.corrcoef(Lm[:, 0], s1)[0, 1]
            r2 = np.corrcoef(Lm[:, 0], s2)[0, 1]
            print(f"    {fmt(d):28s} corr(L, c1'u) = {r1:+.2f}  "
                  f"corr(L, c2'u) = {r2:+.2f}   kappa = {e/evpi:.2f}")


def make_sigma_bd(Sig, equalize=False, b_ref=None):
    """Zero cross-channel blocks; optionally rescale blocks to equalize each
    channel's decision-relevant variance (engineered rank-2 stress test)."""
    Sbd = np.zeros_like(Sig)
    for j in range(J):
        s = slice(4 * j, 4 * j + 4)
        Sbd[s, s] = Sig[s, s]
    if not equalize:
        return Sbd
    ths = d3.sample_post(Sbd, 4000).reshape(-1, J, 4)
    U = u_stats(ths, b_ref)
    v = U.var(0)
    scale = v.mean() / v
    out = np.zeros_like(Sig)
    for j in range(J):
        s = slice(4 * j, 4 * j + 4)
        out[s, s] = Sig[s, s] * scale[j]
    return out


def stage_f2b():
    """(i) Moment-matched Gaussian formula in u-space vs LQC: how much of the
    value is linear conditioning on the *nonlinear* readout; (ii) the share
    ceiling under independent-channel posteriors: kappa_j <= s_j = share of
    tr(Omega Cov u) owned by channel j."""
    st = load_state()
    oi, on = st["oi"], st["on"]
    print("=" * 78)
    print("F2b: Gaussian-in-u formula vs LQC; share-ceiling law")
    thJ, U, Omega, evpi = st["thJ"], st["U"], st["Omega"], st["evpi"]
    print("  baseline posterior (powered designs, Te=8, P=8):")
    for j, dl in [(0, 0.5), (0, 1.0), (1, 0.8), (2, 0.6), (2, 1.2)]:
        d = ("scalar", j, dl, 8, 8)
        Lm, sev = readout(thJ, d)
        L = Lm[:, 0]
        c = np.cov(U.T, L)[:J, J] / np.sqrt(L.var() + sev[0] ** 2)
        e_g = 0.5 * float(c @ Omega @ c)
        e_l, _ = lqc_evsi(U, Lm, sev, Omega, oi, on)
        print(f"    {fmt(d):26s} gauss-u={e_g:.5f}  LQC={e_l:.5f}  "
              f"nonlinear share = {(e_l-e_g)/e_l:+.2f}")
    for tag, name in [("bd", "block-diag raw"), ("eq", "block-diag equalized")]:
        U2, Om2, thJ2, evpi2 = (st[f"U_{tag}"], st[f"Omega_{tag}"],
                                st[f"thJ_{tag}"], st[f"evpi_{tag}"])
        Su = np.cov(U2.T)
        tr = float(np.trace(Om2 @ Su))
        shares = np.array([Om2[j, j] * Su[j, j] for j in range(J)])
        # off-diagonal Omega x diagonal-ish Su -> cross terms small; report anyway
        print(f"  [{name}] channel shares s_j = {np.round(shares/tr, 3)} "
              f"(sum {shares.sum()/tr:.2f} of trace)")
        best = st[f"best_{tag}"]
        for j in range(J):
            print(f"    ch{j}: kappa_LQC = {best[j]/evpi2:.2f}  vs ceiling s_j = "
                  f"{shares[j]/tr:.2f}")


def stage_f3():
    st = load_state()
    oi, on = st["oi"], st["on"]
    print("=" * 78)
    print("F3: the boundary - independent-channel posteriors (effrank -> 2)")
    best_by = {}
    for tag, name in [("bd", "block-diag (raw)"), ("eq", "block-diag equalized")]:
        thJ, U, Omega, evpi = (st[f"thJ_{tag}"], st[f"U_{tag}"],
                               st[f"Omega_{tag}"], st[f"evpi_{tag}"])
        ev, eff, _, _ = spectrum(U, Omega)
        print(f"  [{name}] eig(A)={np.round(ev, 5)} effrank={eff:.2f} "
              f"EVPI_LQ={evpi:.5f}")
        best = {}
        for j in range(J):
            d = ("scalar", j, 1.0 * XBAR[j], 8, 8)
            Lm, sev = readout(thJ, d)
            e, se_e = lqc_evsi(U, Lm, sev, Omega, oi, on)
            best[j] = e
            print(f"    powered {fmt(d):26s} LQC={e:.5f}  kappa={e/evpi:.2f}")
        print(f"    channel-choice ratio (best/worst powered) = "
              f"{max(best.values())/min(best.values()):.2f}")
        best_by[tag] = best
    save_state(best_bd=best_by["bd"], best_eq=best_by["eq"])


def stage_f3mc():
    """MC confirmation on raw block-diag: best-channel vs worst-channel design."""
    st = load_state()
    thJ, b_pr, oi, on = st["thJ_bd"], st["b_pr_bd"], st["oi"], st["on"]
    best = st["best_bd"]
    jb = max(best, key=best.get)
    jw = min(best, key=best.get)
    n_mc = 240
    for j in (jb, jw):
        Lm, sev = readout(thJ, ("scalar", j, 1.0 * XBAR[j], 8, 8))
        m, s = nested_mc(thJ, Lm, sev, b_pr, oi[:n_mc], on[:n_mc])
        print(f"  nested-MC confirm [bd raw] ch{j}: {m:.5f} (+-{s:.5f})")


def stage_f4():
    st = load_state()
    oi, on = st["oi"], st["on"]
    print("=" * 78)
    print("F4: the fix at the boundary - richer readouts (LQC native)")
    thJ, U, Omega, evpi = st["thJ"], st["U"], st["Omega"], st["evpi"]
    for d in [("scalar", 2, 0.6, 8, 8), ("traj", 2, 0.6, 8, 8),
              ("dual", (2, 0.6), (1, 0.4), 8, 8)]:
        Lm, sev = readout(thJ, d)
        e, se_e = lqc_evsi(U, Lm, sev, Omega, oi, on)
        print(f"  [baseline]   {fmt(d):44s} LQC={e:.5f} kappa={e/evpi:.2f}")
    thJ2, U2, Omega2, evpi2 = (st["thJ_eq"], st["U_eq"], st["Omega_eq"],
                               st["evpi_eq"])
    best = st["best_eq"]
    hi = max(best.values())
    print(f"  [rank-2 eq]  best single-scalar kappa = {hi/evpi2:.2f}")
    for d in [("dual", (0, 1.0 * XBAR[0]), (2, 1.0 * XBAR[2]), 8, 8),
              ("dual", (0, 1.0 * XBAR[0]), (1, 1.0 * XBAR[1]), 8, 8),
              ("dual", (1, 1.0 * XBAR[1]), (2, 1.0 * XBAR[2]), 8, 8),
              ("traj", 2, 1.0 * XBAR[2], 8, 8)]:
        Lm, sev = readout(thJ2, d)
        e, se_e = lqc_evsi(U2, Lm, sev, Omega2, oi, on)
        print(f"  [rank-2 eq]  {fmt(d):44s} LQC={e:.5f} kappa={e/evpi2:.2f}")


def stage_f4mc():
    st = load_state()
    thJ2, b_pr2, oi, on = st["thJ_eq"], st["b_pr_eq"], st["oi"], st["on"]
    best = st["best_eq"]
    j_best = max(best, key=best.get)
    n_mc = 240
    for d in (("dual", (0, 1.0 * XBAR[0]), (2, 1.0 * XBAR[2]), 8, 8),
              ("scalar", j_best, 1.0 * XBAR[j_best], 8, 8)):
        Lm, sev = readout(thJ2, d)
        m, s = nested_mc(thJ2, Lm, sev, b_pr2, oi[:n_mc], on[:n_mc])
        print(f"  nested-MC confirm [eq] {fmt(d):44s}: {m:.5f} (+-{s:.5f})")


def stage_f5():
    st = load_state()
    thJ, U, Omega, oi, on = st["thJ"], st["U"], st["Omega"], st["oi"], st["on"]
    print("=" * 78)
    print("F5: free total budget - the level direction is live again")
    H = st["H"]
    b_bar = d3.opt_simplex(lambda b: d3.rev_ss(b, TH0))
    v_m = 1.0 / d3.mroas(b_bar, TH0)[0]
    Omega_free = v_m * np.diag(1.0 / H)
    evpi_fix = evpi_lq(U, Omega)
    evpi_free = evpi_lq(U, Omega_free)
    print(f"  EVPI_LQ fixed budget = {evpi_fix:.5f};  free budget = "
          f"{evpi_free:.5f} (x{evpi_free/evpi_fix:.1f})")
    designs = design_grid()
    e_fix, e_free = [], []
    for d in designs:
        Lm, sev = readout(thJ, d)
        e_fix.append(lqc_evsi(U, Lm, sev, Omega, oi, on)[0])
        e_free.append(lqc_evsi(U, Lm, sev, Omega_free, oi, on)[0])
    e_fix, e_free = np.array(e_fix), np.array(e_free)
    from scipy.stats import kendalltau, spearmanr
    print(f"  ranking agreement fixed vs free: spearman "
          f"{spearmanr(e_fix, e_free).statistic:.3f} "
          f"kendall {kendalltau(e_fix, e_free).statistic:.3f}")
    print(f"  best design fixed:  {fmt(designs[int(np.argmax(e_fix))]):28s} "
          f"EVSI={e_fix.max():.5f} kappa={e_fix.max()/evpi_fix:.2f}")
    print(f"  best design free:   {fmt(designs[int(np.argmax(e_free))]):28s} "
          f"EVSI={e_free.max():.5f} kappa={e_free.max()/evpi_free:.2f}")
    powered = [i for i, d in enumerate(designs)
               if d[2] >= 0.4 * XBAR[d[1]] and d[3] >= 8]
    pf = e_free[powered]
    print(f"  free-budget plateau check over powered designs: "
          f"max/min = {pf.max()/pf.min():.2f}, kappa in "
          f"[{pf.min()/evpi_free:.2f}, {pf.max()/evpi_free:.2f}]")


def stage_f6():
    """The switch experiment: align a single scalar readout with the top
    decision contrast c1 (a reallocation direction, not a channel)."""
    st = load_state()
    thJ, U, Omega, evpi, oi, on = (st["thJ"], st["U"], st["Omega"],
                                   st["evpi"], st["oi"], st["on"])
    print("=" * 78)
    print("F6: the switch experiment - one readout aligned with the top contrast")
    ev, eff, Om12, W = spectrum(U, Omega)
    print(f"  scalar-readout bound: kappa <= lam1/tr(A) = {ev[0]/ev.sum():.2f}; "
          f"best single-channel achieved 0.36")
    c1 = Om12 @ W[:, 0]
    s1 = (U - U.mean(0)) @ c1
    cands = [("switch", (0, 0.5), (2, 0.6), 8, 8),
             ("switch", (0, 0.5), (1, 0.4), 8, 8),
             ("switch", (0, 1.0), (2, 1.2), 8, 8),
             ("switch", (2, 0.6), (0, 0.5), 8, 8)]     # wrong-sign control
    for d in cands:
        Lm, sev = readout(thJ, d)
        r1 = np.corrcoef(Lm[:, 0], s1)[0, 1]
        e, se_e = lqc_evsi(U, Lm, sev, Omega, oi, on)
        print(f"    {fmt(d):44s} corr(L,c1'u)={r1:+.2f}  LQC={e:.5f} "
              f"kappa={e/evpi:.2f}")


def stage_f6mc():
    st = load_state()
    thJ, b_pr, oi, on = st["thJ"], st["b_pr"], st["oi"], st["on"]
    n_mc = 240
    for d in (("switch", (0, 0.5), (2, 0.6), 8, 8),
              ("switch", (0, 1.0), (2, 1.2), 8, 8),
              ("scalar", 0, 0.5, 8, 8)):
        Lm, sev = readout(thJ, d)
        m, s = nested_mc(thJ, Lm, sev, b_pr, oi[:n_mc], on[:n_mc])
        print(f"  nested-MC confirm {fmt(d):44s}: {m:.5f} (+-{s:.5f})")


STAGES = dict(setup=stage_setup, f1=stage_f1, f6=stage_f6, f6mc=stage_f6mc,
              mc0=lambda: stage_mc(0), mc1=lambda: stage_mc(1),
              mc2=lambda: stage_mc(2), f1c=stage_f1_compare,
              f2=stage_f2, f2b=stage_f2b, f3=stage_f3, f3mc=stage_f3mc,
              f4=stage_f4, f4mc=stage_f4mc, f5=stage_f5)

if __name__ == "__main__":
    t00 = time.time()
    for s in sys.argv[1:]:
        STAGES[s]()
    print(f"[stage(s) {sys.argv[1:]} done in {time.time()-t00:.0f}s]")
