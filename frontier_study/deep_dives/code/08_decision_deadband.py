# Dive 08 core: decision-framed uncertainty — filtered deadband law
# State: z random walk (var q per refresh), obs y = z + eps (var v), flow loss (g/2) z^2,
# impulse: reset z (or zhat) to 0 at fixed cost K. Long-run average cost.
import numpy as np
from scipy.stats import norm

# ---------- Kalman steady state (random walk + white obs noise) ----------
def kalman_ss(q, v):
    P = (-q + np.sqrt(q * q + 4 * q * v)) / 2.0  # P(P+q)=qv
    lam = (P + q) / (P + q + v)
    sig_f2 = (P + q) ** 2 / (P + q + v)  # filtered-increment variance (claim: == q)
    return P, lam, sig_f2

# ---------- DP: average-cost impulse control, fully observed random walk ----------
def dp_threshold(K, q, g=1.0, zmax_mult=3.0, dz=0.02, iters=4000, tol=1e-9):
    """Relative value iteration. Observe z; either act (pay K, z->0) or hold; pay (g/2)z^2; z += w."""
    hstar_guess = (6 * K * q / g) ** 0.25
    zmax = hstar_guess * 1.8 + zmax_mult * np.sqrt(q) + 2.0
    z = np.arange(-zmax, zmax + dz / 2, dz)
    n = len(z)
    # transition kernel row for w ~ N(0,q), truncated
    wmax = 5.0 * np.sqrt(q)
    nw = int(np.ceil(2 * wmax / dz)) | 1
    w = (np.arange(nw) - nw // 2) * dz
    pw = norm.pdf(w, 0, np.sqrt(q)) * dz
    pw /= pw.sum()
    flow = 0.5 * g * z ** 2
    V = flow.copy()
    i0 = n // 2  # z=0 index
    gain = 0.0
    for it in range(iters):
        # E[V(z+w)] with reflecting-ish clamp at edges (edges are deep in act region anyway)
        EV = np.convolve(V, pw[::-1], mode="same")
        # fix edges: clamp indexing
        pad = nw // 2
        Vp = np.concatenate([np.full(pad, V[0]), V, np.full(pad, V[-1])])
        EV = np.array([np.dot(Vp[i:i + nw], pw) for i in range(n)]) if it == 0 else EV
        hold = flow + EV
        act = K + flow[i0] + EV[i0]
        Vn = np.minimum(hold, act)
        gain = Vn[i0]
        Vn = Vn - gain
        sp = np.max(np.abs(Vn - V))
        V = Vn
        if sp < tol:
            break
    acted = hold > act + 1e-12
    # threshold: smallest |z| where acting is optimal (upper side)
    up = z[(z > 0) & acted[z > 0] if False else (z > 0)]
    mask_up = acted[z > 0]
    hstar = up[mask_up][0] if mask_up.any() else np.nan
    return hstar, gain / 1.0, z, acted, it

# proper EV via padded dot each iteration (slower but correct) -- vectorized version
def dp_threshold_fast(K, q, g=1.0, dz=0.02, iters=6000, tol=1e-8, zmax=None):
    hstar_guess = (6 * K * q / g) ** 0.25
    if zmax is None:
        zmax = hstar_guess * 1.8 + 3.0 * np.sqrt(q) + 1.0
    z = np.arange(-zmax, zmax + dz / 2, dz)
    n = len(z)
    wmax = 5.0 * np.sqrt(q)
    nw = int(np.ceil(2 * wmax / dz)) | 1
    w = (np.arange(nw) - nw // 2) * dz
    pw = norm.pdf(w, 0, np.sqrt(q)) * dz
    pw /= pw.sum()
    pad = nw // 2
    flow = 0.5 * g * z ** 2
    V = flow.copy()
    i0 = int(np.argmin(np.abs(z)))
    it_used = iters
    for it in range(iters):
        Vp = np.concatenate([np.full(pad, V[0]), V, np.full(pad, V[-1])])
        # EV[i] = sum_j Vp[i+j]*pw[j]
        EV = np.correlate(Vp, pw, mode="valid")
        hold = flow + EV
        act = K + flow[i0] + EV[i0]
        Vn = np.minimum(hold, act)
        gain = Vn[i0]
        Vn -= gain
        sp = np.max(np.abs(Vn - V))
        V = Vn
        if sp < tol:
            it_used = it
            break
    acted = hold > act + 1e-10
    zpos = z[z > 1e-9]
    mpos = acted[z > 1e-9]
    hstar = zpos[mpos][0] if mpos.any() else np.nan
    return hstar, gain, it_used

# ---------- simulator: partially observed loop with pluggable trigger policy ----------
def simulate(policy, K, q, v, g=1.0, T=200_000, seed=0, noise="gauss", rho=0.0,
             ou_phi=1.0, burn=2000):
    """policy(zhat, P, ctx) -> bool act. Truth z: z' = ou_phi*z + w. Obs noise may be AR(1) with corr rho.
    Returns average cost per step (flow + K*acts), action rate, mean sq true deviation."""
    rng = np.random.default_rng(seed)
    P, lam, _ = kalman_ss(q, v)
    z = 0.0
    zhat = 0.0
    e_prev = 0.0
    cost = 0.0
    n_act = 0
    msd = 0.0
    cnt = 0
    sq = np.sqrt(q)
    sv = np.sqrt(v)
    for t in range(T):
        # observe
        if noise == "gauss":
            e = rng.normal(0, sv)
        elif noise == "t3":
            e = rng.standard_t(3) * sv / np.sqrt(3)  # var-matched
        if rho != 0.0:
            e = rho * e_prev + np.sqrt(1 - rho ** 2) * e
            e_prev = e
        y = z + e
        # steady-state filter (predict step for RW: zhat unchanged, P->P+q handled in lam)
        zhat = zhat + lam * (y - zhat)
        if policy(zhat, P):
            # reset: move allocation by zhat
            z = z - zhat
            zhat = 0.0
            cost += K
            n_act += 1
        if t >= burn:
            cost_flow = 0.5 * g * z * z
            cost += cost_flow
            msd += z * z
            cnt += 1
        else:
            cost = 0.0  # restart cost accounting after burn (act costs pre-burn dropped)
            n_act = 0
        # evolve
        z = ou_phi * z + rng.normal(0, sq)
    return cost / cnt, n_act / cnt, msd / cnt

def band_policy(h):
    return lambda zhat, P: abs(zhat) > h

def raw_band_policy(h, q, v):
    # acts on raw innovation-free naive statistic: emulate by using unshrunk obs deviation
    # implemented separately in simulate_raw
    pass

def simulate_raw(h, K, q, v, g=1.0, T=200_000, seed=0, burn=2000):
    """Naive: track y-based deviation with no filtering: dev accumulates observed y each refresh."""
    rng = np.random.default_rng(seed)
    z = 0.0
    cost = 0.0
    n_act = 0
    cnt = 0
    sq, sv = np.sqrt(q), np.sqrt(v)
    for t in range(T):
        y = z + rng.normal(0, sv)
        if abs(y) > h:
            z = z - y  # believes y, moves by y
            cost += K
            n_act += 1
        if t >= burn:
            cost += 0.5 * g * z * z
            cnt += 1
        else:
            cost = 0.0
            n_act = 0
        z += rng.normal(0, sq)
    return cost / cnt, n_act / cnt

def myopic_policy(K, q, g, horizon_steps):
    """Act when NPV of estimated gain over given horizon exceeds K: (g/2) zhat^2 * H > K."""
    thr = np.sqrt(2 * K / (g * horizon_steps))
    return band_policy(thr)

# ============================================================================
# Experiment drivers (dive 08). Run: python 08_decision_deadband.py [quick]
# Reproduces headline numbers; seeds fixed. See 08_decision_deadband.md.
# ============================================================================
BETA = 0.5826  # Broadie-Glasserman-Kou discrete-monitoring constant

def hstar_law(K, q, g):
    """Composite deadband law (derived + verified in dive 08)."""
    return (12 * K * q / g) ** 0.25 - BETA * np.sqrt(q)

def dp_thr_mixed(K0, kap, q=1., g=1., dz=0.01, iters=8000, tol=1e-9):
    """Average-cost DP threshold with action cost K0 + kap*|z| (reset to 0)."""
    guess = max((12 * max(K0, 1e-3) * q / g) ** 0.25, (6 * kap * q / g) ** (1 / 3.), 1.)
    zmax = guess * 2.5 + 4 * np.sqrt(q)
    z = np.arange(-zmax, zmax + dz / 2, dz)
    wmax = 5 * np.sqrt(q); nw = int(np.ceil(2 * wmax / dz)) | 1
    w = (np.arange(nw) - nw // 2) * dz
    pw = norm.pdf(w, 0, np.sqrt(q)) * dz; pw /= pw.sum(); pad = nw // 2
    flow = 0.5 * g * z ** 2; V = flow.copy(); i0 = int(np.argmin(np.abs(z)))
    for it in range(iters):
        Vp = np.concatenate([np.full(pad, V[0]), V, np.full(pad, V[-1])])
        EV = np.correlate(Vp, pw, mode="valid")
        hold = flow + EV
        act = K0 + kap * np.abs(z) + flow[i0] + EV[i0]
        Vn = np.minimum(hold, act); Vn -= Vn[i0]
        if np.max(np.abs(Vn - V)) < tol: V = Vn; break
        V = Vn
    acted = hold > act + 1e-10
    zp = z[z > 1e-9]; mp = acted[z > 1e-9]
    return zp[mp][0]

def sim_corr_noise(h, rho, q, v, K, g=1., filt="white", T=200000, seed=0, burn=2000):
    """A1: AR(1) refresh noise; filt in {white, aug}. Returns (avgcost, actrate, incvar)."""
    rng = np.random.default_rng(seed)
    sq, se = np.sqrt(q), np.sqrt(v * (1 - rho ** 2))
    if filt == "white":
        P, lam, _ = kalman_ss(q, v)
        z = zh = e = 0.; cost = 0.; na = cnt = 0; incs = []
        for t in range(T):
            e = rho * e + rng.normal(0, se)
            y = z + e
            zh_new = zh + lam * (y - zh); incs.append(zh_new - zh); zh = zh_new
            if abs(zh) > h: z -= zh; zh = 0.; cost += K; na += 1
            if t >= burn: cost += 0.5 * g * z * z; cnt += 1
            else: cost = 0.; na = 0
            z += rng.normal(0, sq)
        return cost / cnt, na / cnt, np.var(incs[burn:])
    F = np.array([[1., 0.], [0., rho]]); Q = np.diag([q, v * (1 - rho ** 2)]); H = np.array([1., 1.])
    P2 = np.eye(2)
    for _ in range(2000):
        Pp = F @ P2 @ F.T + Q; S = H @ Pp @ H
        Kk = Pp @ H / S
        P2n = (np.eye(2) - np.outer(Kk, H)) @ Pp
        if np.max(abs(P2n - P2)) < 1e-12: P2 = P2n; break
        P2 = P2n
    Pp = F @ P2 @ F.T + Q; S = H @ Pp @ H; Kk = Pp @ H / S
    z = e = 0.; s = np.zeros(2); cost = 0.; na = cnt = 0; incs = []
    for t in range(T):
        e = rho * e + rng.normal(0, se)
        y = z + e
        sp = F @ s; zh_old = s[0]
        s = sp + Kk * (y - H @ sp); incs.append(s[0] - zh_old)
        if abs(s[0]) > h: z -= s[0]; s[0] = 0.; cost += K; na += 1
        if t >= burn: cost += 0.5 * g * z * z; cnt += 1
        else: cost = 0.; na = 0
        z += rng.normal(0, sq)
    return cost / cnt, na / cnt, np.var(incs[burn:])

def mmm_transport(Kc=0.02, eta=0.02, tau=0.10, seeds=6, T=4000):
    """E10: 2-channel Hill MMM; returns dict of policy losses. gamma,q,v calibrated at start."""
    from scipy.optimize import minimize_scalar
    B = 2.0; b10, b20 = 3., 2.
    hill = lambda x, Kh, s: x ** s / (x ** s + Kh ** s)
    rev = lambda x1, b1, b2: b1 * hill(x1, 1., 1.2) + b2 * hill(B - x1, 1., 1.2)
    xstar = lambda b1, b2: minimize_scalar(lambda x: -rev(x, b1, b2), bounds=(0.05, B - 0.05), method="bounded").x
    rng = np.random.default_rng(7); xs0 = xstar(b10, b20); dx = 1e-3
    gam = -(rev(xs0 + dx, b10, b20) - 2 * rev(xs0, b10, b20) + rev(xs0 - dx, b10, b20)) / dx ** 2
    qs = [xstar(b10 * np.exp(rng.normal(0, eta)), b20 * np.exp(rng.normal(0, eta))) - xs0 for _ in range(300)]
    vs = [xstar(b10 * np.exp(rng.normal(0, tau)), b20 * np.exp(rng.normal(0, tau))) - xs0 for _ in range(300)]
    q_x, v_x = np.var(qs), np.var(vs)
    hs = hstar_law(Kc, q_x, gam)
    P, lam, _ = kalman_ss(q_x, v_x)
    def run(policy_h):
        out = []
        for sd in range(seeds):
            r = np.random.default_rng(100 + sd)
            b1, b2 = b10, b20; xc = xs0; zh = 0.; loss = 0.
            for t in range(T):
                b1 *= np.exp(r.normal(0, eta)); b2 *= np.exp(r.normal(0, eta))
                xs = xstar(b1, b2)
                xhat = xstar(b1 * np.exp(r.normal(0, tau)), b2 * np.exp(r.normal(0, tau)))
                zh = zh + lam * ((xhat - xc) - zh)
                if abs(zh) > policy_h: xc += zh; zh = 0.; loss += Kc
                loss += rev(xs, b1, b2) - rev(xc, b1, b2)
            out.append(loss / T)
        return np.mean(out), np.std(out) / np.sqrt(seeds)
    return {"h*": hs, "gamma": gam, "q_x": q_x, "v_x": v_x,
            "protocol": run(hs), "act_always": run(0.), "myopic": run(np.sqrt(2 * Kc / gam)),
            "ci_rule": run(1.96 * np.sqrt(P))}

if __name__ == "__main__":
    import sys
    quick = "quick" in sys.argv
    K, q, g = 30., 1., 1.
    hs = hstar_law(K, q, g)
    print("h*(K=30,q=1,g=1) =", hs)
    print("E1 DP check:", dp_thr_mixed(K, 0.), "(formula %.3f)" % hs)
    print("E2 identity: sig_f2 =", kalman_ss(1., 100.)[2], "(should be q=1)")
    c = [simulate(band_policy(hs), K, q, 10., T=50000 if quick else 150000, seed=s)[0] for s in range(4)]
    print("E3 protocol cost v=10:", np.mean(c), "+/-", np.std(c) / 2)
    print("A1 corr-noise:", sim_corr_noise(hs, 0.5, q, 10., K, filt="aug", T=50000 if quick else 200000))
    if not quick:
        print("E10 MMM transport:", mmm_transport())
