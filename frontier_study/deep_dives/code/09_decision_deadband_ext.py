"""Dive 09 — EXTEND of dive 08: multichannel budget-coupled filtered deadband.
Ellipsoid law 4M tr(QM)+8MQM = Sigma/K (exact ergodic solution), tiered pairwise-transfer rule,
cadence separation, and nonlinear MMM transport.  Run:  python 09_decision_deadband_ext.py <exp>
where <exp> in {e1,e2,e3,e3b,e4,e4b,e4c,e5,e6,e7,e8,e8b1,e8b2,e9,e10,quick}.
Library section first; experiment drivers below (each mirrors the numbers quoted in the writeup).
"""

import numpy as np
from numpy.linalg import eigh, inv, cholesky
from scipy.optimize import brentq

BETA_BGK = 0.5826  # -zeta(1/2)/sqrt(2pi)

# ---------- tangent-space basis for budget constraint ----------
def tangent_basis(n):
    """Orthonormal basis (n x (n-1)) of {z: sum z = 0}."""
    B = np.zeros((n, n-1))
    for k in range(1, n):
        v = np.zeros(n); v[:k] = 1.0; v[k] = -k
        B[:, k-1] = v/np.linalg.norm(v)
    return B

# ---------- steady-state multivariate Kalman (local level) ----------
def kf_steady(Q, V, iters=5000, tol=1e-13):
    m = Q.shape[0]; P = Q.copy()   # P = filtered (posterior) covariance
    for _ in range(iters):
        Pp = P + Q                 # predictive
        S = Pp + V
        G = Pp @ inv(S)            # gain
        Pn = Pp - G @ Pp
        if np.max(np.abs(Pn - P)) < tol: P = Pn; break
        P = Pn
    Pp = P + Q; G = Pp @ inv(Pp + V)
    return P, G

# ---------- ellipsoid law: solve 4 M tr(AM) + 8 M A M = Sigma / K ----------
def sqrtm_sym(A):
    w, U = eigh(A); return U @ np.diag(np.sqrt(w)) @ U.T
def isqrtm_sym(A):
    w, U = eigh(A); return U @ np.diag(1/np.sqrt(w)) @ U.T

def ellipsoid_M(A, Sigma, K):
    """Exact solution via whitening: u = A^{1/2} w, S = A^{1/2} Sigma A^{1/2}/K,
    then Mt = f(S) with eigenvalues m_i(4T + 8 m_i) = s_i, T = sum m_i."""
    Ah = sqrtm_sym(A); Aih = isqrtm_sym(A)
    S = Ah @ Sigma @ Ah / K
    s, U = eigh(S)
    def m_of_T(T):
        return (-4*T + np.sqrt(16*T**2 + 32*s)) / 16.0
    f = lambda T: m_of_T(T).sum() - T
    T = brentq(f, 1e-12, 1e6)
    m = m_of_T(T)
    Mt = U @ np.diag(m) @ U.T
    M = Aih @ Mt @ Aih
    g_cont = 2*K*np.trace(A @ M)   # continuous-time ergodic cost (excl. filter floor)
    return M, g_cont

def iso_radius(mdim, K, q, gamma):
    return (4*(mdim+2)*q*K/gamma)**0.25

# ---------- policies ----------
class EllipsoidPolicy:
    def __init__(self, M, A=None, dt=1.0, bgk=True):
        self.M = M; self.A = A; self.dt = dt; self.bgk = bgk and (A is not None)
        self.MAM = M @ A @ M if A is not None else None
    def trigger(self, u):
        Phi = u @ self.M @ u
        if not self.bgk: return Phi >= 1.0
        if Phi <= 0: return False
        corr = 2*BETA_BGK*np.sqrt(self.dt)*np.sqrt(u @ self.MAM @ u / Phi)
        return Phi + corr >= 1.0

class QuadPolicy:  # generic quadratic-form trigger u'Wu >= 1 without BGK
    def __init__(self, W): self.W = W
    def trigger(self, u): return u @ self.W @ u >= 1.0

class BoxPolicy:   # act if any |u_i| >= h_i (in given coordinates via matrix C: c = C u)
    def __init__(self, C, h): self.C = C; self.h = h
    def trigger(self, u): return np.any(np.abs(self.C @ u) >= self.h)

class AlwaysPolicy:
    def trigger(self, u): return True

class NeverPolicy:
    def trigger(self, u): return False

# ---------- simulation ----------
def simulate(policy, Q, V, Sigma, K, T=200000, seed=0, filt=True, P=None, G=None,
             noise='gauss', df=3, return_series=False, warm=2000):
    """z_{k+1}=z_k+w, y=z+eps; filter; act => x_c += zhat (z -= zhat, zhat=0).
    Cost per period: 0.5 z'Sigma z (+K if act). Returns long-run average cost and act rate."""
    rng = np.random.default_rng(seed)
    m = Q.shape[0]
    if P is None: P, G = kf_steady(Q, V)
    Lq = cholesky(Q); Lv = cholesky(V + 1e-15*np.eye(m))
    z = np.zeros(m); zh = np.zeros(m)
    W = rng.standard_normal((T, m)) @ Lq.T
    E = rng.standard_normal((T, m))
    if noise == 't':
        scale = np.sqrt(rng.chisquare(df, size=(T,1))/df)
        E = E/scale * np.sqrt((df-2)/df)   # variance-matched
    E = E @ Lv.T
    cost = 0.0; acts = 0; n = 0
    for k in range(T):
        z = z + W[k]
        y = z + E[k]
        if filt:
            zp = zh; zh = zp + G @ (y - zp)
        else:
            zh = y
        act = policy.trigger(zh)
        c = 0.0
        if act:
            z = z - zh; zh = np.zeros(m); c += K; acts += 1
        c += 0.5*z @ Sigma @ z
        if k >= warm:
            cost += c; n += 1
    return cost/n, acts/T

def simulate_fast_iso(policy_fn, Q, V, Sigma, K, T, seed, P=None, G=None, warm=2000):
    """Vectorised-free but jit-light variant used when trigger is cheap: same as simulate."""
    return simulate(policy_fn, Q, V, Sigma, K, T, seed, P=P, G=G, warm=warm)

# ---------- decision-framed outputs ----------
def por(zh, P, Sigma):
    """P(moving to zhat hurts | data), z~N(zh,P): gain = zh'Sigma z - 0.5 zh'Sigma zh."""
    mu = 0.5*zh @ Sigma @ zh
    sd = np.sqrt(zh @ Sigma @ P @ Sigma @ zh)
    from scipy.stats import norm
    return norm.cdf(-mu/sd) if sd > 0 else 0.0

# ---------- numba fast simulator ----------
import numba as nb

@nb.njit(cache=True)
def _sim_core(W, E, G, Sigma, K, ptype, Mq, MAM, bgk_c, Cbox, hbox, warm):
    T, m = W.shape
    z = np.zeros(m); zh = np.zeros(m)
    cost = 0.0; acts = 0; n = 0
    for k in range(T):
        for i in range(m): z[i] += W[k, i]
        # filter update: zh = zh + G (y - zh), y = z + E
        innov = np.empty(m)
        for i in range(m): innov[i] = z[i] + E[k, i] - zh[i]
        upd = G @ innov
        for i in range(m): zh[i] += upd[i]
        # trigger
        act = False
        if ptype == 0:      # quadratic form (+BGK if bgk_c>0)
            Phi = 0.0
            for i in range(m):
                for j in range(m): Phi += zh[i]*Mq[i, j]*zh[j]
            if bgk_c > 0 and Phi > 0:
                q2 = 0.0
                for i in range(m):
                    for j in range(m): q2 += zh[i]*MAM[i, j]*zh[j]
                Phi += bgk_c*np.sqrt(q2/Phi)
            act = Phi >= 1.0
        elif ptype == 1:    # box in coordinates C
            c = Cbox @ zh
            for i in range(c.shape[0]):
                if abs(c[i]) >= hbox[i]: act = True
        elif ptype == 2:    # always
            act = True
        c0 = 0.0
        if act:
            for i in range(m): z[i] -= zh[i]; zh[i] = 0.0
            c0 += K; acts += 1
        for i in range(m):
            for j in range(m): c0 += 0.5*z[i]*Sigma[i, j]*z[j]
        if k >= warm:
            cost += c0; n += 1
    return cost/n, acts/T

def sim_fast(Q, V, Sigma, K, T=400000, seed=0, ptype=0, Mq=None, MAM=None, bgk_c=0.0,
             Cbox=None, hbox=None, noise='gauss', df=3, warm=2000, P=None, G=None, Vnoise=None):
    rng = np.random.default_rng(seed); m = Q.shape[0]
    if G is None: P, G = kf_steady(Q, V)
    Lq = cholesky(Q); Vn = V if Vnoise is None else Vnoise
    Lv = cholesky(Vn + 1e-15*np.eye(m))
    W = rng.standard_normal((T, m)) @ Lq.T
    E = rng.standard_normal((T, m))
    if noise == 't':
        scale = np.sqrt(rng.chisquare(df, size=(T, 1))/df)
        E = E/scale*np.sqrt((df-2)/df)
    E = E @ Lv.T
    if Mq is None: Mq = np.eye(m)
    if MAM is None: MAM = np.eye(m)
    if Cbox is None: Cbox = np.eye(m)
    if hbox is None: hbox = np.ones(m)
    return _sim_core(np.ascontiguousarray(W), np.ascontiguousarray(E), np.ascontiguousarray(G),
                     np.ascontiguousarray(Sigma), float(K), int(ptype), np.ascontiguousarray(Mq),
                     np.ascontiguousarray(MAM), float(bgk_c), np.ascontiguousarray(Cbox),
                     np.ascontiguousarray(hbox, dtype=np.float64), int(warm))

def ellipsoid_run(Q, V, Sigma, K, M, T=400000, seed=0, bgk=True, scale=1.0, **kw):
    """Run ellipsoid policy u'(M/scale^2)u>=1 (scale multiplies the linear size)."""
    Ms = M/scale**2
    MAM = Ms @ Q @ Ms
    return sim_fast(Q, V, Sigma, K, T, seed, ptype=0, Mq=Ms, MAM=MAM,
                    bgk_c=(2*BETA_BGK if bgk else 0.0), **kw)

def predicted_cost(Q, V, Sigma, K, M):
    P, G = kf_steady(Q, V)
    return 2*K*np.trace(Q @ M) + 0.5*np.trace(Sigma @ P), 0.5*np.trace(Sigma @ P)

# ---------- 2-D discrete-time average-cost DP (fully observed reduced problem) ----------
from scipy.signal import fftconvolve
def dp2d(Q, Sigma, K, L=14.0, d=0.2, iters=3000, tol=1e-7, verbose=False):
    """Relative value iteration for W(u)=min{ c(u)+E W(u+w), K + c(0) + E W(w) } - g.
    Actions: hold or full reset to 0. Returns grid, W, inaction mask, g."""
    x = np.arange(-L, L+1e-9, d); n = len(x)
    X, Y = np.meshgrid(x, x, indexing='ij'); U = np.stack([X, Y], -1)
    c = 0.5*np.einsum('abi,ij,abj->ab', U, Sigma, U)
    # kernel
    Qi = inv(Q); r = int(np.ceil(4.5*np.sqrt(np.max(np.diag(Q)))/d))
    kx = np.arange(-r, r+1)*d; KX, KY = np.meshgrid(kx, kx, indexing='ij'); KU = np.stack([KX, KY], -1)
    ker = np.exp(-0.5*np.einsum('abi,ij,abj->ab', KU, Qi, KU)); ker /= ker.sum()
    W = c.copy(); i0 = n//2
    pad = r
    for it in range(iters):
        # continuation expectation with padding = action value (valid if band inside grid)
        EW0_guess = None
        Wp = np.pad(W, pad, mode='edge')  # temporary; refine below
        EW = fftconvolve(Wp, ker, mode='valid')
        act_val = K + 0.0 + EW[i0, i0]
        # redo padding with action value for outside-grid states
        Wp = np.pad(W, pad, mode='constant', constant_values=act_val)
        EW = fftconvolve(Wp, ker, mode='valid')
        act_val = K + EW[i0, i0]
        hold = c + EW
        Wn = np.minimum(hold, act_val)
        g = Wn[i0, i0]
        Wn = Wn - g
        diff = np.max(np.abs(Wn - W)); W = Wn
        if verbose and it % 200 == 0: print(it, g, diff)
        if diff < tol: break
    inaction = hold < act_val
    return x, W, inaction, g, it

def dp2d_eval(mask_fn, Q, Sigma, K, L=14.0, d=0.2, iters=6000, tol=1e-8):
    """Average cost of policy 'hold iff mask_fn(U) True' by relative value iteration."""
    x = np.arange(-L, L+1e-9, d); n = len(x)
    X, Y = np.meshgrid(x, x, indexing='ij'); U = np.stack([X, Y], -1)
    c = 0.5*np.einsum('abi,ij,abj->ab', U, Sigma, U)
    hold_mask = mask_fn(U)
    Qi = inv(Q); r = int(np.ceil(4.5*np.sqrt(np.max(np.diag(Q)))/d))
    kx = np.arange(-r, r+1)*d; KX, KY = np.meshgrid(kx, kx, indexing='ij'); KU = np.stack([KX, KY], -1)
    ker = np.exp(-0.5*np.einsum('abi,ij,abj->ab', KU, Qi, KU)); ker /= ker.sum()
    W = c.copy(); i0 = n//2
    for it in range(iters):
        Wp = np.pad(W, r, mode='edge'); EW = fftconvolve(Wp, ker, mode='valid'); act_val = K + EW[i0, i0]
        Wp = np.pad(W, r, mode='constant', constant_values=act_val)
        EW = fftconvolve(Wp, ker, mode='valid'); act_val = K + EW[i0, i0]
        Wn = np.where(hold_mask, c + EW, act_val)
        g = Wn[i0, i0]; Wn = Wn - g
        diff = np.max(np.abs(Wn - W)); W = Wn
        if diff < tol: break
    return g, it

def quad_mask(Wm, bgk_A=None, dt=1.0):
    """hold iff u'Wu (+BGK) < 1."""
    def f(U):
        Phi = np.einsum('abi,ij,abj->ab', U, Wm, U)
        if bgk_A is None: return Phi < 1.0
        MAM = Wm @ bgk_A @ Wm
        q2 = np.einsum('abi,ij,abj->ab', U, MAM, U)
        corr = 2*BETA_BGK*np.sqrt(dt)*np.sqrt(np.where(Phi > 0, q2/np.maximum(Phi, 1e-300), 0.0))
        return Phi + corr < 1.0
    return f

def best_scale(shape_W, Q, Sigma, K, scales, **kw):
    """Evaluate hold-region u'(W/s^2)u<1 (no BGK; scale search absorbs it) and return best (s, g)."""
    res = []
    for s in scales:
        g, _ = dp2d_eval(quad_mask(shape_W/s**2), Q, Sigma, K, **kw); res.append((s, g))
    return min(res, key=lambda r: r[1]), res

# ---------- DP with pairwise-transfer actions (n=3 channels, m=2 tangent coords) ----------
from scipy.ndimage import map_coordinates
def dp2d_pairwise(Q, Sigma, K0, c, L=14.0, d=0.2, iters=400, tol=1e-7, tgrid=None, allow_pair=True, verbose=False):
    """Actions: hold; full reset (cost K0+3c) to 0; pairwise transfer along d_ij (cost K0+2c) to best point on line.
    Coordinates: tangent basis B (3x2); d_ij = B^T (e_i - e_j)."""
    B = tangent_basis(3)
    dirs = [B.T @ (np.eye(3)[i] - np.eye(3)[j]) for i, j in [(0, 1), (0, 2), (1, 2)]]
    dirs = [v/np.linalg.norm(v) for v in dirs]
    x = np.arange(-L, L+1e-9, d); n = len(x)
    X, Y = np.meshgrid(x, x, indexing='ij'); U = np.stack([X, Y], -1)
    cst = 0.5*np.einsum('abi,ij,abj->ab', U, Sigma, U)
    Qi = inv(Q); r = int(np.ceil(4.5*np.sqrt(np.max(np.diag(Q)))/d))
    kx = np.arange(-r, r+1)*d; KX, KY = np.meshgrid(kx, kx, indexing='ij'); KU = np.stack([KX, KY], -1)
    ker = np.exp(-0.5*np.einsum('abi,ij,abj->ab', KU, Qi, KU)); ker /= ker.sum()
    if tgrid is None: tgrid = np.linspace(-1.5*L, 1.5*L, 301)
    W = cst.copy(); i0 = n//2
    Kf = K0 + 3*c; Kp = K0 + 2*c
    # precompute interpolation coords for each dir: shape (2, ntl, n, n)
    coords = []
    for v in dirs:
        px = (X[None] + tgrid[:, None, None]*v[0] + L)/d
        py = (Y[None] + tgrid[:, None, None]*v[1] + L)/d
        coords.append(np.stack([px, py]))
    for it in range(iters):
        Wp = np.pad(W, r, mode='edge'); EW = fftconvolve(Wp, ker, mode='valid'); full_val = Kf + EW[i0, i0]
        Wp = np.pad(W, r, mode='constant', constant_values=full_val)
        EW = fftconvolve(Wp, ker, mode='valid'); full_val = Kf + EW[i0, i0]
        Hval = cst + EW                       # value of holding = post-action value at u
        best = np.minimum(Hval, full_val); which = np.where(Hval <= full_val, 0, 1)
        if allow_pair:
            for di, cd in enumerate(coords):
                # value after moving to u + t v then holding there: Hval interpolated; outside grid -> full_val
                vals = map_coordinates(Hval, cd.reshape(2, -1), order=1, mode='constant', cval=full_val).reshape(cd.shape[1:])
                pv = Kp + vals.min(axis=0)
                upd = pv < best; best = np.where(upd, pv, best); which = np.where(upd, 2+di, which)
        g = best[i0, i0]; Wn = best - g
        diff = np.max(np.abs(Wn - W)); W = Wn
        if verbose and it % 20 == 0: print(it, g, diff)
        if diff < tol: break
    return x, W, which, g, it

def dp2d_eval_actions(action_fn, Q, Sigma, K0, c, L=12.0, d=0.2, iters=2000, tol=1e-8):
    """Evaluate a fixed policy: action_fn(U)->int array (0 hold, 1 full reset, 2+k pair transfer along dirs[k]
    with projection target). Returns average cost."""
    B = tangent_basis(3)
    dirs = [B.T @ (np.eye(3)[i] - np.eye(3)[j]) for i, j in [(0, 1), (0, 2), (1, 2)]]
    dirs = [v/np.linalg.norm(v) for v in dirs]
    x = np.arange(-L, L+1e-9, d); n = len(x)
    X, Y = np.meshgrid(x, x, indexing='ij'); U = np.stack([X, Y], -1)
    cst = 0.5*np.einsum('abi,ij,abj->ab', U, Sigma, U)
    Qi = inv(Q); r = int(np.ceil(4.5*np.sqrt(np.max(np.diag(Q)))/d))
    kx = np.arange(-r, r+1)*d; KX, KY = np.meshgrid(kx, kx, indexing='ij'); KU = np.stack([KX, KY], -1)
    ker = np.exp(-0.5*np.einsum('abi,ij,abj->ab', KU, Qi, KU)); ker /= ker.sum()
    A = action_fn(U); Kf = K0 + 3*c; Kp = K0 + 2*c
    # projection targets for pair actions (Sigma-orthogonal projection onto line span(v))
    targets = []
    for v in dirs:
        t = -(U @ Sigma @ v)/(v @ Sigma @ v)
        tgt = U + t[..., None]*v[None, None, :]
        targets.append(np.stack([(tgt[..., 0]+L)/d, (tgt[..., 1]+L)/d]))
    W = cst.copy(); i0 = n//2
    for it in range(iters):
        Wp = np.pad(W, r, mode='edge'); EW = fftconvolve(Wp, ker, mode='valid'); full_val = Kf + EW[i0, i0]
        Wp = np.pad(W, r, mode='constant', constant_values=full_val)
        EW = fftconvolve(Wp, ker, mode='valid'); full_val = Kf + EW[i0, i0]
        Hval = cst + EW
        Wn = np.where(A == 0, Hval, full_val)
        for k, tg in enumerate(targets):
            pv = Kp + map_coordinates(Hval, tg.reshape(2, -1), order=1, mode='constant', cval=full_val).reshape(Hval.shape)
            Wn = np.where(A == 2+k, pv, Wn)
        g = Wn[i0, i0]; Wn = Wn - g
        diff = np.max(np.abs(Wn - W)); W = Wn
        if diff < tol: break
    return g, it

# ---------- joint cadence–band problem (1-D, calendar time) ----------
@nb.njit(cache=True)
def _cad_core(Tfine, delta, nsub, q, v, gamma, K, cr, h, lam, seed):
    np.random.seed(seed)
    z = 0.0; zh = 0.0; flow = 0.0; nact = 0; nref = 0
    sq = np.sqrt(q*delta); sv = np.sqrt(v)
    for k in range(Tfine):
        z += sq*np.random.standard_normal()
        if (k+1) % nsub == 0:
            y = z + sv*np.random.standard_normal()
            zh = zh + lam*(y - zh); nref += 1
            if abs(zh) >= h:
                z -= zh; zh = 0.0; nact += 1
        flow += 0.5*gamma*z*z*delta
    Ttot = Tfine*delta
    return flow/Ttot, nact*K/Ttot, nref*cr/Ttot, nact/Ttot

def cadence_sim(Delta, h, q=1.0, v=10.0, gamma=1.0, K=30.0, cr=1.0, Ttot=200000.0, delta=0.05, seed=0):
    nsub = int(round(Delta/delta)); Tfine = int(Ttot/delta)
    qd = q*Delta; P = (-qd + np.sqrt(qd*qd + 4*qd*v))/2; lam = (P+qd)/(P+qd+v)
    f, a, r, rate = _cad_core(Tfine, delta, nsub, q, v, gamma, K, cr, h, lam, seed)
    return f+a+r, f, a, r, rate

def cadence_pred(Delta, q=1.0, v=10.0, gamma=1.0, K=30.0, cr=1.0):
    qd = q*Delta; P = (-qd + np.sqrt(qd*qd + 4*qd*v))/2
    hc = (12*K*q/gamma)**0.25; hd = hc - BETA_BGK*np.sqrt(qd)
    g = np.sqrt(K*q*gamma/3) + 0.075*gamma*qd + 0.5*gamma*P + cr/Delta
    return g, hd, P

# ---------- cadence with overlap-correlated refresh noise (AR(1), rho = 1 - Delta/W) ----------
def kf_aug_steady(qd, v, rho):
    F = np.diag([1.0, rho]); Qs = np.diag([qd, (1-rho**2)*v]); H = np.array([[1.0, 1.0]]); R = 1e-9
    P = Qs.copy()
    for _ in range(20000):
        Pp = F @ P @ F.T + Qs; S = H @ Pp @ H.T + R; G = Pp @ H.T / S
        Pn = Pp - G @ H @ Pp
        if np.max(np.abs(Pn-P)) < 1e-14: P = Pn; break
        P = Pn
    Pp = F @ P @ F.T + Qs; S = H @ Pp @ H.T + R; G = (Pp @ H.T / S).ravel()
    return P, G

@nb.njit(cache=True)
def _cad_corr_core(Tfine, delta, nsub, q, v, gamma, K, cr, h, rho, G0, G1, seed, white_lam):
    np.random.seed(seed)
    z = 0.0; e = 0.0; sz = 0.0; se = 0.0; flow = 0.0; nact = 0; nref = 0
    sq = np.sqrt(q*delta); sv = np.sqrt(v); s1 = np.sqrt(1-rho*rho)
    for k in range(Tfine):
        z += sq*np.random.standard_normal()
        if (k+1) % nsub == 0:
            e = rho*e + s1*sv*np.random.standard_normal()
            y = z + e; nref += 1
            if white_lam > 0:
                sz = sz + white_lam*(y - sz)
            else:
                # predict: sz same, se -> rho*se ; innovation
                pe = rho*se; inn = y - sz - pe
                sz = sz + G0*inn; se = pe + G1*inn
            if abs(sz) >= h:
                z -= sz; sz = 0.0; nact += 1
        flow += 0.5*gamma*z*z*delta
    Ttot = Tfine*delta
    return flow/Ttot, nact*K/Ttot, nref*cr/Ttot, nact/Ttot

def cadence_corr_sim(Delta, h, rho, q, v, gamma, K, cr, Ttot=300000.0, delta=0.25, seed=0, white=False):
    nsub = int(round(Delta/delta)); Tfine = int(Ttot/delta); qd = q*Delta
    if white:
        P = (-qd + np.sqrt(qd*qd + 4*qd*v))/2; lam = (P+qd)/(P+qd+v)
        f, a, r, rate = _cad_corr_core(Tfine, delta, nsub, q, v, gamma, K, cr, h, rho, 0.0, 0.0, seed, lam)
    else:
        P, G = kf_aug_steady(qd, v, rho)
        f, a, r, rate = _cad_corr_core(Tfine, delta, nsub, q, v, gamma, K, cr, h, rho, G[0], G[1], seed, 0.0)
    return f+a+r, f, a, r, rate


# ======================= EXPERIMENT DRIVERS =======================
def e1():
    import numpy as np, time; 
    np.set_printoptions(precision=4, suppress=True)
    # ---- E1: vector Muth identity Cov(dzhat)=Q, anisotropic Q,V, m=3 ----
    rng=np.random.default_rng(1)
    m=3
    Bq=rng.standard_normal((m,m)); Q=Bq@Bq.T/m + 0.2*np.eye(m)
    Bv=rng.standard_normal((m,m)); V=3*(Bv@Bv.T/m) + 0.5*np.eye(m)
    P,G=kf_steady(Q,V)
    print("Q=\n",Q); 
    # algebraic check: Cov(dzh) = G (P+Q+V) G'
    Pp=P+Q; C=G@(Pp+V)@G.T; print("G(Pp+V)G' - Q max abs:",np.max(np.abs(C-Q)))
    # MC
    T=400000; Lq=cholesky(Q); Lv=cholesky(V)
    W=rng.standard_normal((T,m))@Lq.T; E=rng.standard_normal((T,m))@Lv.T
    z=np.zeros(m); zh=np.zeros(m); dz=np.empty((T,m))
    for k in range(T):
        z=z+W[k]; y=z+E[k]; zn=zh+G@(y-zh); dz[k]=zn-zh; zh=zn
    Cemp=np.cov(dz[5000:].T); print("MC Cov(dzhat)=\n",Cemp,"\nrel err",np.max(np.abs(Cemp-Q))/np.max(np.abs(Q)))
    print("P=\n",P)


def e2():
    import numpy as np, time; 
    np.set_printoptions(precision=4, suppress=True)
    K=30.0; T=600000
    # ---- E2: isotropic m=2, sweep scale of radius; v-invariance across V ----
    m=2; Q=np.eye(m); Sigma=np.eye(m)
    M,g=ellipsoid_M(Q,Sigma,K); R=(1/M[0,0])**0.5
    print("iso m=2: R*=%.3f  g_cont=%.4f"%(R,g))
    t0=time.time()
    for v in [1.0, 10.0, 100.0]:
        V=v*np.eye(m); P,G=kf_steady(Q,V); pred,floor=predicted_cost(Q,V,Sigma,K,M)
        row=[]
        for sc in [0.6,0.8,0.9,1.0,1.1,1.2,1.5]:
            cs=[ellipsoid_run(Q,V,Sigma,K,M,T=T,seed=s,scale=sc)[0] for s in range(3)]
            row.append((sc,np.mean(cs),np.std(cs)/np.sqrt(3)))
        best=min(row,key=lambda r:r[1])
        print("v=%5.1f pred=%.3f (floor %.3f) | "%(v,pred,floor)+" ".join("%.1f:%.3f±%.3f"%r for r in row)+" | argmin %.1f"%best[0])
    print("time",time.time()-t0)
    # act rate check at scale 1, v=1
    V=np.eye(m); c,a=ellipsoid_run(Q,V,Sigma,K,M,T=T,seed=0)
    print("act rate obs %.4f pred m q/R*^2 = %.4f"%(a, m*1.0/R**2))


def e3():
    import numpy as np, time; 
    np.set_printoptions(precision=4, suppress=True)
    K=30.0
    # ---- E3a: DP truth for isotropic m=2 ----
    Q=np.eye(2); S=np.eye(2)
    t0=time.time(); x,W,ina,g,it=dp2d(Q,S,K,L=12,d=0.2,iters=4000); print("iso DP g=%.4f iters=%d t=%.1fs"%(g,it,time.time()-t0))
    # effective radius of inaction region: area
    area=ina.sum()*0.2**2; Rdp=np.sqrt(area/np.pi)
    M,gc=ellipsoid_M(Q,S,K); R=(1/M[0,0])**0.5
    print("DP radius %.3f ; law R*-beta*sqrt(q) = %.3f ; R* = %.3f ; g_cont %.4f"%(Rdp,R-BETA_BGK,R,gc))
    # axis cut radius
    i0=len(x)//2; row=ina[i0]; print("axis half-width", (row.sum()*0.2)/2)


def e3b():
    import numpy as np, time; 
    np.set_printoptions(precision=4, suppress=True)
    K=30.0
    def study(Q,S,label,L=14,d=0.2):
        print("=== ",label); print("Q=",Q.tolist()," Sigma=",S.tolist())
        x,W,ina,gdp,it=dp2d(Q,S,K,L=L,d=d); print("DP optimal g=%.4f"%gdp)
        M,gc=ellipsoid_M(Q,S,K); print("Riccati ellipsoid M=",M.tolist()," g_cont=%.4f"%gc)
        # 1. Riccati ellipsoid with analytic BGK (no tuning)
        g1,_=dp2d_eval(quad_mask(M,bgk_A=Q),Q,S,K,L=L,d=d); print("Riccati+BGK (zero-tuned): g=%.4f  (+%.2f%% vs DP)"%(g1,100*(g1/gdp-1)))
        # 1b. Riccati with best scale
        (s,g1b),_=best_scale(M,Q,S,K,np.linspace(0.80,0.95,7),L=L,d=d); print("Riccati best-scale s=%.3f g=%.4f (+%.2f%%)"%(s,g1b,100*(g1b/gdp-1)))
        scales=np.linspace(0.5,1.6,23)
        # 2. sphere in loss-whitened coords (AL isotropic with avg q): W = Sigma / R^2 where R from qbar
        Sh=sqrtm_sym(S); Qt=Sh@Q@Sh; qbar=np.trace(Qt)/2; R=iso_radius(2,K,qbar,1.0)
        (s,g2),_=best_scale(S/R**2,Q,S,K,scales,L=L,d=d); print("Loss-ellipsoid (AL sphere in loss coords) best s=%.2f g=%.4f (+%.2f%%)"%(s,g2,100*(g2/gdp-1)))
        # 3. Mahalanobis-of-drift ellipsoid u'Q^{-1}u
        (s,g3),_=best_scale(inv(Q)/R**2,Q,S,K,scales,L=L,d=d); print("Drift-Mahalanobis best s=%.2f g=%.4f (+%.2f%%)"%(s,g3,100*(g3/gdp-1)))
        # 4. plain sphere in raw coords
        (s,g4),_=best_scale(np.eye(2)/R**2,Q,S,K,scales,L=L,d=d); print("Raw sphere best s=%.2f g=%.4f (+%.2f%%)"%(s,g4,100*(g4/gdp-1)))
        # 5. box in raw coords, best common half-width
        res=[]
        for h in np.linspace(2.0,7.0,26):
            g,_=dp2d_eval(lambda U: np.all(np.abs(U)<h,axis=-1),Q,S,K,L=L,d=d); res.append((h,g))
        h,g5=min(res,key=lambda r:r[1]); print("Box (per-channel scalar bands) best h=%.2f g=%.4f (+%.2f%%)"%(h,g5,100*(g5/gdp-1)))
        # region geometry: DP boundary vs ellipsoid: sample DP region principal axes
        X,Y=np.meshgrid(x,x,indexing='ij'); pts=np.stack([X[ina],Y[ina]],1)
        C=np.cov(pts.T); w,U=eigh(C); print("DP-region second-moment axes ratio %.3f, angle %.1f deg"%(np.sqrt(w[1]/w[0]), np.degrees(np.arctan2(U[1,1],U[0,1]))))
        w2,U2=eigh(M); print("Riccati ellipsoid axes ratio %.3f, angle %.1f deg"%(np.sqrt(w2[1]/w2[0]), np.degrees(np.arctan2(U2[1,0],U2[0,0]))))
        return gdp,g1,g2,g3,g4,g5
    t0=time.time()
    study(np.eye(2),np.diag([1.0,4.0]),"A: isotropic drift, anisotropic loss 1:4")
    study(np.array([[1.0,0.7],[0.7,1.0]]),np.diag([1.0,4.0]),"B: correlated drift rho=0.7, loss 1:4 (non-commuting)")
    study(np.array([[1.0,0.0],[0.0,0.1]]),np.array([[1.0,0.8],[0.8,1.0]]),"C: drift 10:1, correlated loss 0.8")
    print("time",time.time()-t0)


def e4():
    import numpy as np, time; 
    np.set_printoptions(precision=4, suppress=True)
    # n=3 channels; tangent coords; isotropic drift/loss in channel space -> isotropic in tangent coords
    Q=np.eye(2); S=np.eye(2)
    for (K0,c) in [(30,0),(20,10/3.),(10,20/3.),(0,10),(15,15)]:
        t0=time.time()
        x,W,wh,gf,_=dp2d_pairwise(Q,S,K0,c,allow_pair=False,L=12,d=0.2)
        x,W,wh2,gp,it=dp2d_pairwise(Q,S,K0,c,allow_pair=True,L=12,d=0.2)
        frac_pair=(wh2>=2).sum()/max((wh2>=1).sum(),1)
        print("K0=%4.1f c=%5.2f (full %.1f, pair %.1f): full-only g=%.4f | with pairwise g=%.4f (-%.2f%%) | pair share of action region %.2f | it %d t=%.0fs"%(K0,c,K0+3*c,K0+2*c,gf,gp,100*(1-gp/gf),frac_pair,it,time.time()-t0))


def e4b():
    import numpy as np; 
    Q=np.eye(2); S=np.eye(2)
    B=tangent_basis(3); dirs=[B.T@(np.eye(3)[i]-np.eye(3)[j]) for i,j in [(0,1),(0,2),(1,2)]]; dirs=[v/np.linalg.norm(v) for v in dirs]
    for (K0,c) in [(10,20/3.),(0,10)]:
        x,W,wh,g,it=dp2d_pairwise(Q,S,K0,c,allow_pair=True,L=12,d=0.2)
        X,Y=np.meshgrid(x,x,indexing='ij'); U=np.stack([X,Y],-1)
        # residual after best (smallest residual) pairwise projection
        res=np.min([np.abs(U[...,0]*(-v[1])+U[...,1]*v[0]) for v in dirs],axis=0)   # distance to nearest transfer line
        rad=np.sqrt(X**2+Y**2)
        act=wh>=1; pair=wh>=2; full=wh==1
        print("K0=%.0f c=%.2f: g=%.4f"%(K0,c,g))
        print("  residual-to-line among PAIR actions: max %.2f ; among FULL actions: min %.2f"%(res[pair].max(), res[full].min()))
        print("  radius at first action along a transfer line (theta=0): %.2f ; along 30deg (between lines): %.2f"%(
            rad[act & (res<0.11)].min(), rad[act & (np.abs(np.degrees(np.arctan2(Y,X))%60-30)<3)].min()))
        # inaction-region shape: min radius by angle
        ang=np.degrees(np.arctan2(Y,X))%60
        for a in [0,10,20,30]:
            sel=act&(np.abs(ang-a)<2.5); print("   angle %2d: first-action radius %.2f"%(a,rad[sel].min()))
        # compare to iso R*-beta
        M,_=ellipsoid_M(Q,S,K0+3*c); print("  full-cost law radius %.2f ; pair-cost law radius %.2f"%((1/M[0,0])**0.5-BETA_BGK,(1/ellipsoid_M(Q,S,K0+2*c)[0][0,0])**0.5-BETA_BGK))


def e4c():
    import numpy as np; 
    Q=np.eye(2); S=np.eye(2)
    B=tangent_basis(3); dirs=[B.T@(np.eye(3)[i]-np.eye(3)[j]) for i,j in [(0,1),(0,2),(1,2)]]; dirs=[v/np.linalg.norm(v) for v in dirs]
    def heuristic(K0,c,mode):
        Kf=K0+3*c; Kp=K0+2*c
        Mf,_=ellipsoid_M(Q,S,Kf); Mp,_=ellipsoid_M(Q,S,Kp); Rf=(1/Mf[0,0])**0.5
        rc=Rf*np.sqrt(1-np.sqrt(max(1-c/Kf,0)))
        def f(U):
            rad2=U[...,0]**2+U[...,1]**2; rad=np.sqrt(rad2)
            res=np.stack([np.abs(U[...,0]*(-v[1])+U[...,1]*v[0]) for v in dirs]); k=res.argmin(0); rmin=res.min(0)
            Rf_d=Rf-BETA_BGK; Rp_d=(1/Mp[0,0])**0.5-BETA_BGK
            A=np.zeros(rad.shape,int)
            if mode=='full_only':
                A[rad>=Rf_d]=1; return A
            pair_ok=(rmin<rc)
            # pair trigger: the along-line component exceeds pair-cost radius
            along=np.sqrt(np.maximum(rad2-rmin**2,0))
            trig_pair=pair_ok&(along>=Rp_d)
            trig_full=(rad>=Rf_d)&(~pair_ok)
            A[trig_full]=1; A[trig_pair]=2+k[trig_pair]
            # if outside full radius but pair_ok: pair anyway
            A[(rad>=Rf_d)&pair_ok&(A==0)]=2+k[(rad>=Rf_d)&pair_ok&(A==0)]
            return A
        return f, rc
    for (K0,c) in [(20,10/3.),(10,20/3.),(0,10),(15,15)]:
        x,W,wh,gdp,_=dp2d_pairwise(Q,S,K0,c,allow_pair=True,L=12,d=0.2)
        f,rc=heuristic(K0,c,'tier'); gh,_=dp2d_eval_actions(f,Q,S,K0,c)
        f0,_=heuristic(K0,c,'full_only'); g0,_=dp2d_eval_actions(f0,Q,S,K0,c)
        print("K0=%4.1f c=%5.2f: DP %.4f | tiered heuristic (r_c=%.2f) %.4f (+%.2f%%) | full-only ellipsoid %.4f (+%.2f%%)"%(K0,c,gdp,rc,gh,100*(gh/gdp-1),g0,100*(g0/gdp-1)))


def e5():
    import numpy as np, time; 
    from scipy.stats import chi2
    np.set_printoptions(precision=4, suppress=True)
    K=30.0; T=600000; seeds=range(6)
    Q=np.array([[1.0,0.7],[0.7,1.0]]); S=np.diag([1.0,4.0]); m=2
    M,gc=ellipsoid_M(Q,S,K)
    def run_all(v):
        V=v*np.eye(m); P,G=kf_steady(Q,V); pred,floor=predicted_cost(Q,V,S,K,M)
        out={}
        def rep(fn): 
            r=[fn(s) for s in seeds]; return np.mean(r), np.std(r)/np.sqrt(len(r))
        out['ellipsoid+BGK (protocol)']=rep(lambda s: ellipsoid_run(Q,V,S,K,M,T=T,seed=s)[0])
        Mn,_=ellipsoid_M(Q+V,S,K); out['naive: Riccati with Q+V']=rep(lambda s: ellipsoid_run(Q,V,S,K,Mn,T=T,seed=s)[0])
        # raw (unfiltered) ellipsoid: use G=I
        out['raw ellipsoid on y (no filter)']=rep(lambda s: sim_fast(Q,V,S,K,T,s,ptype=0,Mq=M,MAM=M@Q@M,bgk_c=2*BETA_BGK,G=np.eye(m),P=P)[0])
        # chi2 CI rule
        Wc=inv(P)/chi2.ppf(0.95,m); out['CI rule: zh\'P^-1 zh > chi2_95']=rep(lambda s: sim_fast(Q,V,S,K,T,s,ptype=0,Mq=Wc,MAM=Wc,bgk_c=0.0)[0])
        # per-coordinate scalar 08 bands (box) using marginal q_i, gamma_i
        h=np.array([(12*K*Q[i,i]/S[i,i])**0.25-BETA_BGK*np.sqrt(Q[i,i]) for i in range(m)])
        out['box of per-channel dive-08 bands']=rep(lambda s: sim_fast(Q,V,S,K,T,s,ptype=1,Cbox=np.eye(m),hbox=h)[0])
        out['act-always']=rep(lambda s: sim_fast(Q,V,S,K,T,s,ptype=2)[0])
        # oracle best scale of ellipsoid
        sc=[(x,np.mean([ellipsoid_run(Q,V,S,K,M,T=T,seed=s,scale=x)[0] for s in seeds])) for x in [0.85,0.9,0.95,1.0,1.05,1.1,1.15]]
        out['oracle scale sweep argmin']=min(sc,key=lambda r:r[1])
        print("v=%g  predicted cont+floor=%.3f (floor %.3f)"%(v,pred,floor))
        for k,val in out.items(): print("   %-38s %s"%(k, "%.3f ± %.3f"%val if k[:6]!='oracle' else "scale %.2f cost %.3f"%val))
    t0=time.time()
    for v in [1.0,10.0]: run_all(v)
    print("time",time.time()-t0)


def e6():
    import numpy as np, time; 
    from scipy.stats import chi2
    K=30.0; T=400000; seeds=range(4); v=1.0
    print("isotropic q=gamma=1, K=30, v=1: sphere law vs best box vs scalar-law box vs chi2 CI, by dimension m")
    for m in [1,2,3,5,8]:
        Q=np.eye(m); S=np.eye(m); V=v*np.eye(m); P,G=kf_steady(Q,V)
        M,gc=ellipsoid_M(Q,S,K); R=(1/M[0,0])**0.5
        rep=lambda fn: np.mean([fn(s) for s in seeds])
        g_e=rep(lambda s: ellipsoid_run(Q,V,S,K,M,T=T,seed=s)[0])
        # sphere scale sweep (check argmin)
        sw=[(x,rep(lambda s: ellipsoid_run(Q,V,S,K,M,T=T,seed=s,scale=x)[0])) for x in [0.9,1.0,1.1]]
        h08=(12*K)**0.25-BETA_BGK
        g_box08=rep(lambda s: sim_fast(Q,V,S,K,T,s,ptype=1,Cbox=np.eye(m),hbox=h08*np.ones(m))[0])
        bx=[(h,rep(lambda s: sim_fast(Q,V,S,K,T,s,ptype=1,Cbox=np.eye(m),hbox=h*np.ones(m))[0])) for h in np.linspace(2.0,6.0,17)]
        hb,g_box=min(bx,key=lambda r:r[1])
        Wc=inv(P)/chi2.ppf(0.95,m); g_ci=rep(lambda s: sim_fast(Q,V,S,K,T,s,ptype=0,Mq=Wc,MAM=Wc)[0])
        floor=0.5*np.trace(S@P)
        print("m=%d R*=%.2f sphere %.3f (argmin scale %.1f) | box@scalar-law h=%.2f: %.3f (+%.1f%%) | best box h=%.2f: %.3f (+%.1f%%) | chi2 CI %.3f (+%.1f%%) | floor %.2f"%(
            m,R,g_e,min(sw,key=lambda r:r[1])[0],h08,g_box08,100*(g_box08/g_e-1),hb,g_box,100*(g_box/g_e-1),g_ci,100*(g_ci/g_e-1),floor))


def e7():
    import numpy as np, time; 
    q=1.;v=10.;gamma=1.;K=30.
    t0=time.time()
    print("E7a: band argmin vs Delta (cr=1): scale multipliers on h_d")
    for D in [0.25,0.5,1.0,2.0,4.0]:
        g,hd,P=cadence_pred(D,q,v,gamma,K,1.0)
        row=[]
        for s in [0.7,0.85,1.0,1.15,1.3]:
            cs=[cadence_sim(D,hd*s,q,v,gamma,K,1.0,seed=sd)[0] for sd in range(3)]; row.append((s,np.mean(cs)))
        b=min(row,key=lambda r:r[1])
        print(" Delta=%.2f h_d=%.2f pred=%.3f | "%(D,hd,g)+" ".join("%.2f:%.3f"%r for r in row)+" | argmin %.2f"%b[0])
    print("E7b: Delta argmin vs cr at h=h_d(Delta); K in {30,120}")
    Ds=[0.25,0.5,1.0,2.0,4.0,8.0]
    for K_ in [30.,120.]:
      for cr in [0.25,1.0,4.0,16.0]:
        obs=[];pred=[]
        for D in Ds:
            g,hd,P=cadence_pred(D,q,v,gamma,K_,cr); pred.append(g)
            obs.append(np.mean([cadence_sim(D,hd,q,v,gamma,K_,cr,seed=sd)[0] for sd in range(3)]))
        io=int(np.argmin(obs)); ip=int(np.argmin(pred))
        Dstar=(4*cr/(gamma*np.sqrt(q*v)))**(2/3)
        print(" K=%4.0f cr=%5.2f: obs argmin Delta=%.2f (cost %.3f) | pred argmin %.2f | closed-form Delta*=%.2f | obs costs: %s"%(K_,cr,Ds[io],obs[io],Ds[ip],Dstar," ".join("%.2f"%o for o in obs)))
    print("time",time.time()-t0)


def e9():
    import numpy as np, time; 
    from scipy.stats import norm
    np.set_printoptions(precision=4, suppress=True)
    K=30.; T=400000; seeds=range(4)
    # ---- E9a: vector PoR formula MC ----
    rng=np.random.default_rng(3)
    S=np.diag([1.,4.]); P=np.array([[0.8,0.3],[0.3,1.5]]); zh=np.array([1.2,-0.7])
    z=rng.multivariate_normal(zh,P,size=2_000_000); gain=z@S@zh-0.5*zh@S@zh
    print("PoR MC %.5f  formula %.5f"%(np.mean(gain<0), norm.cdf(-0.5*zh@S@zh/np.sqrt(zh@S@P@S@zh))))
    # ---- E9b: misspecified shape: truth anisotropic case B, policy assumes isotropic drift / ignores loss anisotropy ----
    Q=np.array([[1.0,0.7],[0.7,1.0]]); S=np.diag([1.0,4.0]); V=10*np.eye(2); m=2
    M,_=ellipsoid_M(Q,S,K)
    rep=lambda fn: (np.mean([fn(s) for s in seeds]), np.std([fn(s) for s in seeds])/2)
    base=rep(lambda s: ellipsoid_run(Q,V,S,K,M,T=T,seed=s)[0])
    Mq=ellipsoid_M(np.trace(Q)/2*np.eye(2),S,K)[0]; wq=rep(lambda s: ellipsoid_run(Q,V,S,K,Mq,T=T,seed=s)[0])
    Ms=ellipsoid_M(Q,np.trace(S)/2*np.eye(2),K)[0]; ws=rep(lambda s: ellipsoid_run(Q,V,S,K,Ms,T=T,seed=s)[0])
    Mb=ellipsoid_M(np.trace(Q)/2*np.eye(2),np.trace(S)/2*np.eye(2),K)[0]; wb=rep(lambda s: ellipsoid_run(Q,V,S,K,Mb,T=T,seed=s)[0])
    print("E9b (v=10): correct %.3f±%.3f | drift assumed isotropic %.3f (+%.1f%%) | loss assumed isotropic %.3f (+%.1f%%) | both isotropic %.3f (+%.1f%%)"%(base[0],base[1],wq[0],100*(wq[0]/base[0]-1),ws[0],100*(ws[0]/base[0]-1),wb[0],100*(wb[0]/base[0]-1)))
    # parameter error sweep: K, Q scale wrong by factor f
    for f in [0.5,2.0]:
        Mk=ellipsoid_M(Q,S,K*f)[0]; c=rep(lambda s: ellipsoid_run(Q,V,S,K,Mk,T=T,seed=s)[0])
        Mqq=ellipsoid_M(Q*f,S,K)[0]; c2=rep(lambda s: ellipsoid_run(Q,V,S,K,Mqq,T=T,seed=s)[0])
        print("   K mis-set x%.1f: %.3f (+%.1f%%) | Q mis-set x%.1f: %.3f (+%.1f%%)"%(f,c[0],100*(c[0]/base[0]-1),f,c2[0],100*(c2[0]/base[0]-1)))
    # ---- E9c: heavy-tailed refresh noise (t3, variance matched) ----
    ht=rep(lambda s: ellipsoid_run(Q,V,S,K,M,T=T,seed=s,noise='t',df=3)[0])
    sw=[(x,np.mean([ellipsoid_run(Q,V,S,K,M,T=T,seed=s,scale=x,noise='t',df=3)[0] for s in seeds])) for x in [0.9,1.0,1.1,1.2]]
    print("E9c t3 noise: %.3f±%.3f (+%.1f%% vs gauss) argmin scale %.1f"%(ht[0],ht[1],100*(ht[0]/base[0]-1),min(sw,key=lambda r:r[1])[0]))
    # ---- E9d: negative control K->0 and v-invariance of argmin in anisotropic case ----
    for Kt in [0.3,3.0]:
        Mt,_=ellipsoid_M(Q,S,Kt); w,_=eigh(Mt); print("   K=%.1f semi-axes %s (K=30: %s)"%(Kt,1/np.sqrt(w),1/np.sqrt(eigh(M)[0])))
    for v in [0.1,100.]:
        Vv=v*np.eye(2); sw=[(x,np.mean([ellipsoid_run(Q,Vv,S,K,M,T=T,seed=s,scale=x)[0] for s in seeds])) for x in [0.85,0.925,1.0,1.075,1.15]]
        print("   v=%.1f scale sweep: %s argmin %.3f"%(v," ".join("%.3f:%.3f"%r for r in sw),min(sw,key=lambda r:r[1])[0]))
    # ---- E9e: seed sensitivity of headline duel number (protocol at v=10), 12 seeds ----
    c=[ellipsoid_run(Q,V,S,K,M,T=T,seed=s)[0] for s in range(12)]; print("E9e seed sd/mean = %.4f (12 seeds)"%(np.std(c)/np.mean(c)))


def e10():
    import numpy as np, time; 
    q=1/13.; v=10.; gamma=1.; K=30.; W=104.
    print("E10: cadence law with overlap-correlated refresh noise rho=1-Delta/W (W=104 wk), q=1/13 per wk, v=10, K=30")
    Ds=[1,2,4,8,13,26]
    for cr in [0.05,0.2,0.8]:
        rows=[]
        for D in Ds:
            rho=max(0.,1-D/W); qd=q*D
            hd=(12*K*q/gamma)**0.25-BETA_BGK*np.sqrt(qd)
            Pw=(-qd+np.sqrt(qd*qd+4*qd*v))/2; Pa,_=kf_aug_steady(qd,v,rho)
            pred_w=np.sqrt(K*q*gamma/3)+0.075*gamma*qd+0.5*gamma*Pw+cr/D
            pred_a=np.sqrt(K*q*gamma/3)+0.075*gamma*qd+0.5*gamma*Pa[0,0]+cr/D
            oc=np.mean([cadence_corr_sim(D,hd,rho,q,v,gamma,K,cr,seed=s)[0] for s in range(3)])
            ow=np.mean([cadence_corr_sim(D,hd,rho,q,v,gamma,K,cr,seed=s,white=True)[0] for s in range(3)])
            oi=np.mean([cadence_corr_sim(D,hd,0.0,q,v,gamma,K,cr,seed=s,white=True)[0] for s in range(3)])
            rows.append((D,rho,pred_w,pred_a,oi,oc,ow))
        print(" cr=%.2f"%cr)
        print("   Delta rho | pred(indep) pred(aug) | obs indep-noise | obs corr+augKF | obs corr+whiteKF")
        for r in rows: print("   %4d %.2f | %.3f %.3f | %.3f | %.3f | %.3f"%r)
        print("   argmin Delta: indep %d, corr+aug %d, corr+white %d"%(Ds[int(np.argmin([r[4] for r in rows]))],Ds[int(np.argmin([r[5] for r in rows]))],Ds[int(np.argmin([r[6] for r in rows]))]))


E8_SRC = r"""
import numpy as np, time; 
from scipy.optimize import minimize
np.set_printoptions(precision=4, suppress=True)
# ---- E8: nonlinear Hill MMM transport, n=4 channels, budget 1, m=3 tangent coords ----
n=4; Bt=tangent_basis(n)
Kh=np.array([0.30,0.20,0.25,0.35]); sh=np.array([0.8,1.0,0.7,0.9]); beta0=np.array([1.0,0.8,0.9,1.1])
def hill(x,Kk,s): return x**s/(x**s+Kk**s)
def rev(x,beta): return np.sum(beta*hill(x,Kh,sh))
def grad(x,beta):
    return beta*sh*Kh**sh*x**(sh-1)/(x**sh+Kh**sh)**2
def hess_diag(x,beta,eps=1e-5):
    return (grad(x+eps,beta)-grad(x-eps,beta))/(2*eps)
def xstar(beta,x0=None,iters=30):
    x=np.ones(n)/n if x0 is None else x0.copy()
    for _ in range(iters):
        g=Bt.T@grad(x,beta); H=-Bt.T@np.diag(hess_diag(x,beta))@Bt
        if np.min(np.linalg.eigvalsh(H))<=1e-8: step=0.05*g
        else: step=np.linalg.solve(H,g)
        xn=np.clip(x+Bt@step,1e-4,1); xn/=xn.sum()
        if np.max(np.abs(xn-x))<1e-10: x=xn; break
        x=xn
    return x
def hess_tangent(x,beta,eps=1e-4):
    H=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            e_i=np.eye(n)[i]*eps; e_j=np.eye(n)[j]*eps
            H[i,j]=(rev(x+e_i+e_j,beta)-rev(x+e_i-e_j,beta)-rev(x-e_i+e_j,beta)+rev(x-e_i-e_j,beta))/(4*eps*eps)
    return -Bt.T@H@Bt
eta=0.02; tau=0.10; Kc=0.02
rng=np.random.default_rng(0)
x0=xstar(beta0); Gam=hess_tangent(x0,beta0); print("x*0",x0,"Gamma eig",np.linalg.eigvalsh(Gam))
# calibrate Q (drift of x*) and V (estimation noise of x*) in tangent coords by MC at the operating point
dQ=[];dV=[]
for _ in range(400):
    b1=beta0*np.exp(eta*rng.standard_normal(n)); dQ.append(Bt.T@(xstar(b1,x0)-x0))
    bh=beta0*np.exp(tau*rng.standard_normal(n)); dV.append(Bt.T@(xstar(bh,x0)-x0))
Q=np.cov(np.array(dQ).T); V=np.cov(np.array(dV).T)
print("Q eig",np.linalg.eigvalsh(Q),"V eig",np.linalg.eigvalsh(V)," v/q ratio ~",np.trace(V)/np.trace(Q))
M,gc=ellipsoid_M(Q,Gam,Kc); P,G=kf_steady(Q,V)
w,U=eigh(M); print("ellipsoid semi-axes (budget share units):",1/np.sqrt(w)," gain eig",np.linalg.eigvalsh(G))
Sh=sqrtm_sym(Gam); qbar=np.trace(Sh@Q@Sh)/3; Rsph=iso_radius(3,Kc,qbar,1.0); Msph=Gam/Rsph**2
Mnaive,_=ellipsoid_M(Q+V,Gam,Kc)
hbox=np.array([(12*Kc*Q[i,i]/Gam[i,i])**0.25-BETA_BGK*np.sqrt(Q[i,i]) for i in range(3)])
def run(policy, seed, T=4000):
    rng=np.random.default_rng(100+seed)
    beta=beta0.copy(); xc=x0.copy(); zh=np.zeros(3); regret=0.; acts=0; xs=x0.copy()
    for t in range(T):
        beta=beta*np.exp(eta*rng.standard_normal(n))
        xs=xstar(beta,xs); bh=beta*np.exp(tau*rng.standard_normal(n)); xh=xstar(bh,xs)
        y=Bt.T@(xh-xc)
        if policy=='raw': zh=y
        else: zh=zh+G@(y-zh)
        act=False
        if policy=='always': act=True
        elif policy=='never': act=False
        elif policy in('ellipsoid','sphere','naive','raw'):
            Mm={'ellipsoid':M,'sphere':Msph,'naive':Mnaive,'raw':M}[policy]
            Phi=zh@Mm@zh; MAM=Mm@Q@Mm
            act = Phi + (2*BETA_BGK*np.sqrt(zh@MAM@zh/Phi) if Phi>0 else 0) >= 1
        elif policy=='box': act=np.any(np.abs(zh)>=hbox)
        elif policy=='myopic': act = 0.5*zh@Gam@zh >= Kc
        elif policy=='ci':
            from scipy.stats import chi2; act = zh@inv(P)@zh >= chi2.ppf(0.95,3)
        if act:
            xc=np.clip(xc+Bt@zh,1e-4,1); xc=xc/xc.sum(); zh=np.zeros(3); regret+=Kc; acts+=1
        regret+= rev(xs,beta)-rev(xc,beta)
    return regret/T, acts/T
t0=time.time()
for pol in ['ellipsoid','sphere','box','naive','raw','myopic','ci','always','never']:
    r=[run(pol,s) for s in range(8)]; c=[x[0] for x in r]; a=[x[1] for x in r]
    print("%-10s regret/period %.5f ± %.5f  act rate %.3f"%(pol,np.mean(c),np.std(c)/np.sqrt(8),np.mean(a)))
print("time",time.time()-t0)

"""

def e8():
    exec(E8_SRC, globals())
def e8b1():
    src = E8_SRC.split("t0=time.time()\nfor pol")[0]
    exec(src, globals()); exec(E8B1_SRC, globals())

E8B1_SRC = r"""
np.set_printoptions(precision=3)
print("Q eig",np.linalg.eigvalsh(Q),"V eig",np.linalg.eigvalsh(V))
# add scaled-ellipsoid policies by monkeypatching M via closure
def run_scaled(scale,seed,T=4000):
    global M
    M0=M; M=M0/scale**2
    try: return run('ellipsoid',seed,T)
    finally: M=M0
seeds=range(8)
base=np.array([run('ellipsoid',s)[0] for s in seeds])
print("ellipsoid mean %.5f ± %.5f"%(base.mean(),base.std()/np.sqrt(len(seeds))))
for pol in ['sphere','box','ci','naive','myopic','raw','always']:
    r=np.array([run(pol,s)[0] for s in seeds]); d=r-base
    print("%-9s %.5f | paired diff vs ellipsoid %+.5f ± %.5f (ratio %.2f)"%(pol,r.mean(),d.mean(),d.std()/np.sqrt(len(seeds)),r.mean()/base.mean()))

"""

def quick():
    """Smoke test of headline numbers (~30 s)."""
    K=30.0; Q=np.array([[1.0,0.7],[0.7,1.0]]); S=np.diag([1.0,4.0])
    x,W,ina,gdp,it=dp2d(Q,S,K,L=14,d=0.2); M,gc=ellipsoid_M(Q,S,K)
    g1,_=dp2d_eval(quad_mask(M,bgk_A=Q),Q,S,K,L=14,d=0.2)
    print("case B: DP %.4f  ellipsoid+BGK %.4f  gap %.3f%%  g_cont %.4f offset %.3f (-0.165 tr(SQ)=%.3f)"%(gdp,g1,100*(g1/gdp-1),gc,gdp-gc,-0.165*np.trace(S@Q)))
    x,W,wh,gf,_=dp2d_pairwise(np.eye(2),np.eye(2),0.0,10.0,allow_pair=False,L=12,d=0.2)
    x,W,wh,gp,_=dp2d_pairwise(np.eye(2),np.eye(2),0.0,10.0,allow_pair=True,L=12,d=0.2)
    print("pairwise K0=0,c=10: full-only %.4f with-pair %.4f saving %.2f%%"%(gf,gp,100*(1-gp/gf)))
    V=10*np.eye(2); c=np.mean([ellipsoid_run(Q,V,S,K,M,T=400000,seed=s)[0] for s in range(3)])
    print("duel v=10 protocol cost %.3f (writeup: 13.21)"%c)

if __name__ == "__main__":
    import sys
    exp = sys.argv[1] if len(sys.argv) > 1 else "quick"
    globals()[exp]()
