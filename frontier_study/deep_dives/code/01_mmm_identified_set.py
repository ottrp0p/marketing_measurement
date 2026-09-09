"""
Deep dive 01: The identified set of MMM parameters (M1).
Numerical verification of identification/design theory for
  y_t = beta * Hill(Adstock_alpha(x_t); K, S) + eps_t,  eps ~ N(0, sigma^2)

Experiments:
  A: FIM eigenstructure across canonical spend designs
  B: delta^4 scaling of ridge information for small-amplitude single-level designs
  C: block-length sweep for two-level designs (settling-time vs transition-count tradeoff)
  D: MLE recovery Monte Carlo across designs
  E: range-coverage failure (max adstock << K) -> beta/K^S ridge
  F: annealed budget-constrained design optimization vs heuristics
"""
import numpy as np
from scipy.optimize import least_squares
rng_global = np.random.default_rng(0)

# ---------- model ----------
def adstock(x, alpha, a0=None):
    a = np.empty_like(x, dtype=float)
    prev = x[0] / (1 - alpha) if a0 is None else a0   # start at steady state of first value
    for t in range(len(x)):
        prev = x[t] + alpha * prev
        a[t] = prev
    return a

def hill(a, K, S):
    aS = np.power(np.maximum(a, 1e-12), S)
    return aS / (aS + K**S)

def mean_response(x, th):
    beta, K, S, alpha = th
    return beta * hill(adstock(x, alpha), K, S)

TH0 = np.array([1.0, 1.5, 2.0, 0.6])   # beta, K, S, alpha ; xbar=1 -> abar=2.5
SIGMA = 0.05
NAMES = ["beta", "K", "S", "alpha"]

def jacobian(x, th, rel=1e-6):
    J = np.zeros((len(x), 4))
    f0 = mean_response(x, th)
    for i in range(4):
        d = rel * max(abs(th[i]), 1e-3)
        tp = th.copy(); tp[i] += d
        J[:, i] = (mean_response(x, tp) - f0) / d
    return J

def fim(x, th=TH0, sigma=SIGMA):
    J = jacobian(x, th)
    return J.T @ J / sigma**2

def eigs(F):
    return np.sort(np.linalg.eigvalsh(F))

def crlb_se(F, ridge=1e-12):
    try:
        C = np.linalg.inv(F + ridge*np.eye(4))
        return np.sqrt(np.maximum(np.diag(C), 0))
    except np.linalg.LinAlgError:
        return np.full(4, np.inf)

def dcrit(F):
    ev = eigs(F)
    return np.prod(np.maximum(ev, 1e-300)) ** 0.25   # det^(1/p)

# ---------- designs (all mean ~ 1.0, T weeks) ----------
def d_constant(T, rng): return np.ones(T)
def d_lognormal(T, rng):
    x = rng.lognormal(mean=-0.125, sigma=0.5, size=T); return x / x.mean()
def d_ar1(T, rng):
    z = np.zeros(T); phi=0.8
    for t in range(1,T): z[t] = phi*z[t-1] + rng.normal(0, 0.3)
    x = np.exp(z); return x/x.mean()
def d_sine10(T, rng):
    return 1 + 0.10*np.sin(2*np.pi*np.arange(T)/8)
def d_blocks2(T, rng, L=8, lo=0.2, hi=1.8):
    x = np.where((np.arange(T)//L) % 2 == 0, hi, lo); return x/x.mean()
def d_onoff(T, rng, L=8):
    x = np.where((np.arange(T)//L) % 2 == 0, 2.0, 0.0); return x/x.mean()
def d_blocks3(T, rng, L=10):
    lv = [0.0, 1.0, 2.0]
    x = np.array([lv[(t//L) % 3] for t in range(T)], float); return x/x.mean()

DESIGNS = {
    "constant":            d_constant,
    "iid lognormal":       d_lognormal,
    "AR(1) lognormal":     d_ar1,
    "sine +/-10%":         d_sine10,
    "2-level blocks L=8":  d_blocks2,
    "on/off blocks L=8":   d_onoff,
    "3-level blocks L=10": d_blocks3,
}

def experiment_A(T=156):
    print("="*80); print(f"EXPERIMENT A: FIM eigenstructure, T={T}, theta0={TH0}, sigma={SIGMA}")
    rng = np.random.default_rng(1)
    print(f"{'design':22s} {'log10 eigenvalues (asc)':32s} {'D-crit':>10s}  SE(beta,K,S,alpha)")
    for name, fn in DESIGNS.items():
        x = fn(T, rng)
        F = fim(x)
        ev = eigs(F)
        se = crlb_se(F)
        lev = " ".join(f"{np.log10(max(e,1e-300)):6.2f}" for e in ev)
        ses = " ".join(f"{s:8.3g}" for s in se)
        print(f"{name:22s} {lev:32s} {dcrit(F):10.3g}  {ses}")

def experiment_B(T=156):
    print("="*80); print("EXPERIMENT B: small-amplitude scaling, x = 1 + delta*sin(2 pi t/8)")
    print("Prediction: lambda_min(FIM) ~ delta^4  (ridge resolved only at 2nd order)")
    deltas = np.array([0.025, 0.05, 0.1, 0.2, 0.4])
    lmins, l2s = [], []
    for d in deltas:
        x = 1 + d*np.sin(2*np.pi*np.arange(T)/8)
        ev = eigs(fim(x))
        lmins.append(ev[0]); l2s.append(ev[1])
    lmins = np.array(lmins); l2s = np.array(l2s)
    for d, l1, l2 in zip(deltas, lmins, l2s):
        print(f"  delta={d:5.3f}  lambda_min={l1:10.4g}  lambda_2={l2:10.4g}")
    s_min = np.polyfit(np.log(deltas), np.log(lmins), 1)[0]
    s_2   = np.polyfit(np.log(deltas), np.log(l2s), 1)[0]
    print(f"  log-log slope lambda_min vs delta: {s_min:.3f}   (theory: 4)")
    print(f"  log-log slope lambda_2   vs delta: {s_2:.3f}   (theory: 2)")

def experiment_C(T=156):
    print("="*80); print("EXPERIMENT C: two-level block design, D-criterion vs block length L")
    print(f"(adstock settling time 1/(1-alpha) = {1/(1-TH0[3]):.1f} weeks)")
    best = (None, -np.inf)
    for L in [1,2,3,4,6,8,10,13,16,20,26,39,52,78]:
        x = d_blocks2(T, None, L=L)
        Dc = dcrit(fim(x))
        lm = eigs(fim(x))[0]
        print(f"  L={L:3d}  D-crit={Dc:10.4g}  lambda_min={lm:10.4g}")
        if Dc > best[1]: best = (L, Dc)
    print(f"  best L = {best[0]}")
    return best[0]

def fit_mle(x, y, rng, n_starts=6):
    best = None
    for _ in range(n_starts):
        p0 = np.array([np.exp(rng.normal(0,0.5)), np.exp(rng.normal(0.4,0.5)),
                       np.exp(rng.normal(0.7,0.4)), rng.uniform(0.1,0.9)])
        try:
            r = least_squares(lambda p: mean_response(x, p) - y, p0,
                              bounds=([1e-3,1e-3,0.2,0.0],[50,50,8,0.99]),
                              method="trf", max_nfev=2000)
            if best is None or r.cost < best.cost: best = r
        except Exception: pass
    return best.x

def roas_curve_err(th_hat, grid):
    """max abs error of steady-state response curve beta*h(x/(1-a)) over spend grid"""
    def curve(th): return th[0]*hill(grid/(1-th[3]), th[1], th[2])
    return np.max(np.abs(curve(th_hat) - curve(TH0)))

def experiment_D(T=156, nrep=120):
    print("="*80); print(f"EXPERIMENT D: MLE recovery, {nrep} Monte Carlo reps")
    rng = np.random.default_rng(7)
    grid = np.linspace(0.2, 1.6, 15)   # spend range around budget
    sel = ["constant", "AR(1) lognormal", "sine +/-10%", "2-level blocks L=8", "3-level blocks L=10"]
    print(f"{'design':22s} " + " ".join(f"RMSE({n}):>9" for n in NAMES) + "  curve_maxerr(med)")
    for name in sel:
        x = DESIGNS[name](T, np.random.default_rng(1))
        errs, cerrs = [], []
        mu = mean_response(x, TH0)
        for r in range(nrep):
            y = mu + rng.normal(0, SIGMA, T)
            th = fit_mle(x, y, rng)
            errs.append(th - TH0)
            cerrs.append(roas_curve_err(th, grid))
        errs = np.array(errs)
        rmse = np.sqrt((errs**2).mean(0))
        print(f"{name:22s} " + " ".join(f"{v:9.3g}" for v in rmse) + f"   {np.median(cerrs):9.3g}")

def experiment_E(T=156):
    print("="*80); print("EXPERIMENT E: range coverage failure — spend scaled so max adstock << K")
    rng = np.random.default_rng(3)
    for scale in [1.0, 0.3, 0.1]:
        x = d_blocks2(T, None, L=8) * scale
        a = adstock(x, TH0[3])
        F = fim(x)
        ev = eigs(F)
        # ridge direction = eigvec of smallest eigenvalue
        w = np.linalg.eigh(F)[1][:, 0]
        print(f"  scale={scale:4.1f}  max(a)/K={a.max()/TH0[1]:5.2f}  "
              f"log10 ev={[f'{np.log10(max(e,1e-300)):.1f}' for e in ev]}  "
              f"ridge dir (b,K,S,al)={np.round(w,2)}")
    # profile-likelihood ridge demo at scale=0.1: vary K, refit beta with S,alpha fixed at truth
    x = d_blocks2(T, None, L=8) * 0.1
    mu = mean_response(x, TH0)
    print("  profile RSS along the beta/K^S ridge (S,alpha at truth), noiseless:")
    for Kv in [0.8, 1.5, 3.0, 6.0]:
        # best beta given K: closed form linear fit
        h = hill(adstock(x, TH0[3]), Kv, TH0[2])
        b = (h @ mu) / (h @ h)
        rss = np.sum((mu - b*h)**2)
        print(f"    K={Kv:4.1f} -> beta*={b:7.3f}  beta*/ (K^-S ratio-adj) rss={rss:9.3g}  beta*K^S={b*Kv**TH0[2]:8.3f}")


def experiment_F(T=96, iters=6000):
    """Annealed budget-constrained design optimization vs heuristics (budget-preserving moves)."""
    print("="*80)
    print(f"EXPERIMENT F: annealed design optimization, T={T}, budget mean=1, levels 0..2.5 step 0.5")
    rng = np.random.default_rng(11)
    levels = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    x = np.tile([2.0]*8 + [0.0]*8, T//16).astype(float)   # exact mean 1.0
    def obj(x): return np.log(max(dcrit(fim(x)), 1e-300))
    cur = obj(x); bestx, bestv = x.copy(), cur
    Temp0 = 0.3
    for it in range(iters):
        Temp = Temp0 * (1 - it/iters) + 1e-3
        prop = x.copy()
        if rng.random() < 0.5:      # swap two slots (budget preserved)
            i, j = rng.integers(0, T, 2); prop[i], prop[j] = prop[j], prop[i]
        else:                        # move one slot up, another down (budget preserved)
            i, j = rng.integers(0, T, 2)
            li = np.argmin(np.abs(levels - prop[i])); lj = np.argmin(np.abs(levels - prop[j]))
            if li < len(levels)-1 and lj > 0 and i != j:
                prop[i] = levels[li+1]; prop[j] = levels[lj-1]
            else: continue
        v = obj(prop)
        if v > cur or rng.random() < np.exp((v-cur)/Temp):
            x, cur = prop, v
            if v > bestv: bestx, bestv = x.copy(), v
    for nm, xd in [("2-level 0.2/1.8 L=8", d_blocks2(T,None,L=8)),
                   ("on/off 0/2 L=8", np.tile([2.0]*8+[0.0]*8, T//16)),
                   ("3-level L=10", d_blocks3(T,None,L=10)),
                   ("AR(1) lognormal", d_ar1(T, np.random.default_rng(1)))]:
        F = fim(xd)
        print(f"  {nm:24s} D-crit={dcrit(F):10.4g}  lambda_min={eigs(F)[0]:9.4g}  SE={np.round(crlb_se(F),4)}")
    F = fim(bestx)
    print(f"  {'ANNEALED':24s} D-crit={dcrit(F):10.4g}  lambda_min={eigs(F)[0]:9.4g}  SE={np.round(crlb_se(F),4)}")
    print("  annealed design (units of 0.5):")
    print("   ", "".join(str(int(round(v*2))) for v in bestx))
    lv, ct = np.unique(bestx, return_counts=True)
    print("  level histogram:", dict(zip(lv.tolist(), ct.tolist())))
    return bestx

def experiment_G(T=156):
    """Transients-as-dose-ranging: remove adstock transients from a 2-level block design
    and watch the smallest FIM eigenvalue collapse ~1000x at equal sample size."""
    print("="*80)
    print("EXPERIMENT G: is identification carried by the adstock transient?")
    x = d_blocks2(T, None, L=13, lo=0.2, hi=1.8)   # L=13: ~6 transient + ~7 settled weeks/block
    J = jacobian(x, TH0)
    keep_settled = np.array([(t % 13) >= 6 for t in range(T)])
    keep_trans   = np.array([(t % 13) < 6 or (t % 13) >= 12 for t in range(T)])
    for nm, mask in [("all obs", np.ones(T, bool)),
                     ("settled-only", keep_settled),
                     ("transient-heavy", keep_trans)]:
        Jm = J[mask]; F = Jm.T @ Jm / SIGMA**2
        print(f"  {nm:16s} n={mask.sum():3d}  lambda_min={eigs(F)[0]:9.4g}  "
              f"D-crit={dcrit(F):9.4g}  SE={np.round(crlb_se(F),4)}")

if __name__ == "__main__":
    experiment_A()
    experiment_B()
    experiment_C()
    experiment_E()
    experiment_F()
    experiment_G()
    experiment_D()
