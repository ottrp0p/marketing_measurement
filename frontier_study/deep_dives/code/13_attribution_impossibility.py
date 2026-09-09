"""
Dive 13 — An impossibility theorem for causal attribution (queue item C1 / M8).

Core objects (n channels, latent confounder U ~ N(0,1) unless stated):
  e_j(U)     = sigmoid(a_j + g_j U)                    exposure propensity of channel j (g_j = targeting slope)
  p(z, U)    = P(Y(z)=1 | U)                            potential-outcome model (logit or reach-overlap)
  c(S, A)    = E[Y(1_A) | Z = 1_S]                      the cross-journey counterfactual table (A ⊆ S used)
  v_obs(S)   = c(S, S) = E[Y | Z = 1_S]                 the OBSERVATIONAL journey game (what DDA/Shao-Li see)
  v_star(A)  = E[Y(1_A)]                                the full-exposure interventional game
  v_pop(A)   = E[Y(Z ∧ 1_A)] = Σ_S P(S) c(S, S∩A)       the realized-exposure game; Sh(v_pop) is the causal target
  tau_j      = E[Y(Z) − Y(Z − e_j)] = Σ_S P(S)[c(S,S) − c(S,S∖j)]   removal effect (population)
  DDA_j      = Σ_S P(S) Sh_j(v_obs|_S)                  journey-level data-driven attribution (Shapley within journey)
All population quantities are computed EXACTLY by Gauss–Hermite quadrature over U; Monte-Carlo experiments
use finite user panels.  Run `python 13_attribution_impossibility.py quick` (≈2 min) or `e1` … `e16`.
"""
import sys, itertools, time
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit
from scipy.optimize import least_squares, linprog

# ----------------------------------------------------------------------------- set algebra
def subsets(n):
    return list(range(1 << n))

def popcount(m):
    return bin(m).count("1")

def members(m, n):
    return [j for j in range(n) if (m >> j) & 1]

def mobius(v, n):
    """Harsanyi dividends d(T) of a game v indexed by bitmask (v[0] is v(∅))."""
    d = np.array(v, dtype=float).copy()
    for j in range(n):
        for m in range(1 << n):
            if (m >> j) & 1:
                d[m] -= d[m ^ (1 << j)]
    return d

def shapley(v, n, players=None):
    """Shapley values of game v (bitmask indexed) for the players in `players` (default all).
    Uses Sh_j = Σ_{T∋j} d(T)/|T|.  If players ⊂ [n] the game is restricted to that subset (others fixed absent)."""
    if players is None:
        players = list(range(n))
    d = mobius(v, n)
    out = np.zeros(n)
    for m in range(1, 1 << n):
        mem = members(m, n)
        if all(j in players for j in mem):
            for j in mem:
                out[j] += d[m] / len(mem)
    return out

def shapley_weights(n):
    """w[j][A] = Shapley weight of coalition A for player j (A ∌ j): |A|!(n−|A|−1)!/n!."""
    from math import factorial
    W = np.zeros((n, 1 << n))
    for j in range(n):
        for A in range(1 << n):
            if not (A >> j) & 1:
                k = popcount(A)
                W[j, A] = factorial(k) * factorial(n - k - 1) / factorial(n)
    return W

# ----------------------------------------------------------------------------- the DGP
class DGP:
    """n channels, latent U, logit exposure e_j(U)=σ(a_j+g_j U), outcome model p(z,U)."""
    def __init__(self, a, g, beta0, beta, theta, theta2=None, model="logit", ugrid=None, uw=None,
                 udist="normal", reach_alpha=None, beta2=0.0):
        self.a = np.asarray(a, float); self.g = np.asarray(g, float); self.n = len(a)
        self.beta0, self.beta, self.beta2 = beta0, beta, beta2
        self.theta = np.asarray(theta, float)
        self.theta2 = np.zeros((self.n, self.n)) if theta2 is None else np.asarray(theta2, float)
        self.model = model
        self.reach_alpha = reach_alpha
        if ugrid is None:
            if udist == "normal":
                x, w = hermegauss(120); self.u, self.w = x, w / w.sum()
            elif udist == "bimodal":  # 50/50 mixture N(±1.2, 0.5²)
                x, w = hermegauss(120); w = w / w.sum()
                self.u = np.concatenate([1.2 + 0.5 * x, -1.2 + 0.5 * x]); self.w = np.concatenate([w, w]) / 2
            elif udist == "skew":     # standardized exponential
                x, w = np.polynomial.laguerre.laggauss(120); w = w / w.sum()
                self.u = x - 1.0; self.w = w
        else:
            self.u, self.w = np.asarray(ugrid), np.asarray(uw) / np.sum(uw)
        self.N = 1 << self.n
        self._cache()

    def e(self):
        return expit(self.a[None, :] + self.g[None, :] * self.u[:, None])          # (Q, n)

    def p(self, zmask):
        """P(Y(1_A)=1 | U) on the grid for bitmask A."""
        z = np.array([(zmask >> j) & 1 for j in range(self.n)], float)
        if self.model == "logit":
            lin = self.beta0 + self.beta * self.u + self.beta2 * self.u ** 2 + z @ self.theta + 0.5 * z @ self.theta2 @ z
            return expit(lin)
        elif self.model == "probit":
            from scipy.stats import norm
            lin = self.beta0 + self.beta * self.u + z @ self.theta + 0.5 * z @ self.theta2 @ z
            return norm.cdf(lin)
        elif self.model == "reach":  # Y = 1 − (1−base(U)) Π_j (1 − q_j(U))^{z_j}: independent persuasion chances
            base = expit(self.beta0 + self.beta * self.u)
            q = expit(self.theta[None, :] + (self.reach_alpha if self.reach_alpha is not None else 0.0) * self.u[:, None])
            keep = np.prod((1 - q) ** z[None, :], axis=1)
            return 1 - (1 - base) * keep

    def _cache(self):
        n, Q = self.n, len(self.u)
        E = self.e()
        # P(Z=1_S | U) for every S
        PZ = np.ones((Q, self.N))
        for S in range(self.N):
            for j in range(n):
                PZ[:, S] *= E[:, j] if (S >> j) & 1 else (1 - E[:, j])
        self.PZu = PZ
        self.PS = self.w @ PZ                                        # P(Z = 1_S)
        self.Pu = np.stack([self.p(A) for A in range(self.N)], 1)   # (Q, N): p(1_A, U)
        # cross-journey table c(S, A) = E[Y(1_A) | Z=1_S]
        self.c = (PZ * self.w[:, None]).T @ self.Pu / self.PS[:, None]   # (N_S, N_A)
        self.v_obs = np.array([self.c[S, S] for S in range(self.N)])
        self.v_star = self.w @ self.Pu
        self.v_pop = np.array([sum(self.PS[S] * self.c[S, S & A] for S in range(self.N)) for A in range(self.N)])
        self.EY = self.PS @ self.v_obs
        self.EY0 = self.PS @ self.c[:, 0]

    # --- attribution rules
    def sh_obs(self):   return shapley(self.v_obs, self.n)
    def sh_star(self):  return shapley(self.v_star, self.n)
    def sh_pop(self):   return shapley(self.v_pop, self.n)
    def bias_game(self): return self.v_obs - self.v_star
    def tau(self):
        return np.array([sum(self.PS[S] * (self.c[S, S] - self.c[S, S & ~(1 << j)]) for S in range(self.N))
                         for j in range(self.n)])
    def dda(self):
        """Journey-level Shapley on the observational game: Σ_S P(S) Sh_j(v_obs restricted to S)."""
        out = np.zeros(self.n)
        for S in range(1, self.N):
            out += self.PS[S] * shapley(self.v_obs, self.n, players=members(S, self.n))
        return out
    def dda_bias_games(self):
        """b_S(A) = c(A,A) − c(S,A) for A ⊆ S; returns Σ_S P(S) Sh_j(b_S)."""
        out = np.zeros(self.n)
        for S in range(1, self.N):
            bS = np.array([self.c[A, A] - self.c[S, A] if (A & ~S) == 0 else 0.0 for A in range(self.N)])
            out += self.PS[S] * shapley(bS, self.n, players=members(S, self.n))
        return out
    def last_touch_like(self):
        """'Equal split among touched channels' of every conversion — the linear rule most tools default to."""
        out = np.zeros(self.n)
        for S in range(1, self.N):
            mem = members(S, self.n)
            for j in mem:
                out[j] += self.PS[S] * self.v_obs[S] / len(mem)
        return out
    def total_incremental(self):
        return self.EY - self.EY0

    # --- finite-sample simulation
    def sample(self, N, rng):
        u = rng.choice(self.u, size=N, p=self.w) if len(self.u) < 1000 else None
        u = rng.standard_normal(N) if u is None else u
        E = expit(self.a[None, :] + self.g[None, :] * u[:, None])
        Z = (rng.random((N, self.n)) < E).astype(int)
        S = Z @ (1 << np.arange(self.n))
        # potential outcome under realized z
        pz = np.zeros(N)
        for A in range(self.N):
            idx = S == A
            if idx.any():
                zz = np.array([(A >> j) & 1 for j in range(self.n)], float)
                if self.model == "logit":
                    pz[idx] = expit(self.beta0 + self.beta * u[idx] + zz @ self.theta + 0.5 * zz @ self.theta2 @ zz)
                elif self.model == "reach":
                    base = expit(self.beta0 + self.beta * u[idx])
                    q = expit(self.theta[None, :] + (self.reach_alpha or 0.0) * u[idx, None])
                    pz[idx] = 1 - (1 - base) * np.prod((1 - q) ** zz[None, :], axis=1)
        Y = (rng.random(N) < pz).astype(int)
        return S, Y

def dda_from_sample(S, Y, n, min_cell=1):
    N = 1 << n
    cnt = np.bincount(S, minlength=N).astype(float)
    conv = np.bincount(S, weights=Y, minlength=N)
    v = np.where(cnt >= min_cell, conv / np.maximum(cnt, 1), np.nan)
    # fill unobserved cells with overall rate (as tools do)
    v = np.where(np.isnan(v), conv.sum() / cnt.sum(), v)
    PS = cnt / cnt.sum()
    out = np.zeros(n)
    for m in range(1, N):
        if PS[m] > 0:
            out += PS[m] * shapley(v, n, players=members(m, n))
    return out, v, PS

# ----------------------------------------------------------------------------- baseline DGP
def base_dgp(n=3, g=(0.4, 1.0, 1.6), theta=(0.5, 0.5, 0.0), beta=1.0, a=(-0.5, -1.0, -1.5), beta0=-2.5, **kw):
    """Three channels: 1 = broad (weak targeting), 2 = mid, 3 = 'retargeting' (strong targeting) and causally NULL."""
    return DGP(a=a[:n], g=g[:n], beta0=beta0, beta=beta, theta=theta[:n], **kw)

def fmt(x, p=4):
    return np.array2string(np.asarray(x), precision=p, suppress_small=True)

# ============================================================================= experiments
def e1():
    """E1 — exact decomposition identities and the journey-level lemma."""
    print("E1: decomposition identities (exact quadrature)")
    D = base_dgp()
    sh_obs, sh_star, sh_pop, shb = D.sh_obs(), D.sh_star(), D.sh_pop(), shapley(D.bias_game(), D.n)
    print("  Sh(v_obs)          ", fmt(sh_obs))
    print("  Sh(v_star)+Sh(b)   ", fmt(sh_star + shb), " max|diff| = %.2e" % np.max(np.abs(sh_obs - sh_star - shb)))
    dda, ddab = D.dda(), D.dda_bias_games()
    print("  DDA                ", fmt(dda))
    print("  Sh(v_pop)+ΣP(S)Sh(b_S)", fmt(sh_pop + ddab), " max|diff| = %.2e" % np.max(np.abs(dda - sh_pop - ddab)))
    print("  Sh(v_pop) (target) ", fmt(sh_pop), " sum=%.5f  total incremental=%.5f" % (sh_pop.sum(), D.total_incremental()))
    print("  tau (removal)      ", fmt(D.tau()), " sum=%.5f" % D.tau().sum())
    print("  DDA sum=%.5f  = E[Y]-v_obs(∅)=%.5f ; excess over incremental = E[Y0]-E[Y0|Z=∅] = %.5f"
          % (dda.sum(), D.EY - D.v_obs[0], D.EY0 - D.c[0, 0]))
    # journey-level lemma: with g=0, DDA == Sh(v_pop) exactly
    D0 = base_dgp(g=(0, 0, 0))
    print("  [g=0] DDA - Sh(v_pop) max|diff| = %.2e ; Sh(v_star) vs Sh(v_pop):" % np.max(np.abs(D0.dda() - D0.sh_pop())),
          fmt(D0.sh_star()), fmt(D0.sh_pop()))
    return D

def e2():
    """E2 — the null channel: causally zero, credited positive; credit grows with its targeting slope."""
    print("E2: null-channel credit vs targeting slope g3 (theta3 = 0)")
    print("  g3   Sh_pop3   DDA3    Sh_obs3   DDA3/DDA_total  Sh_obs rank of ch3")
    for g3 in [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5]:
        D = base_dgp(g=(0.4, 1.0, g3))
        dda, so = D.dda(), D.sh_obs()
        rank = int(np.argsort(-so).tolist().index(2)) + 1
        print("  %.1f  %+.4f  %+.4f  %+.4f    %.3f            %d" % (g3, D.sh_pop()[2], dda[2], so[2], dda[2] / dda.sum(), rank))

def first_order(D):
    """First-order (in the targeting slopes g) prediction of the bias games.
    P(Z=1_S|U)/P(S) ≈ 1 + U·L(S), L(S)=Σ_k g_k (s_k − ē_k), ē_k = σ(a_k)  (Gaussian U, log-linear tilt)
    b(S)  = Cov(p(1_S,U), P(Z=1_S|U))/P(S) ≈ β_S L(S),  β_S := E[U p(1_S,U)] (outcome U-slope under exposure S)
    b_S(A) = c(A,A) − c(S,A) ≈ β_A [L(A) − L(S)] = −β_A Σ_{k∈S∖A} g_k.
    Returns (Sh(b) first-order, DDA-bias first-order, Sh(b) constant-slope closed form β̄ g_j, DDA closed form β̄ g_j ē_j)."""
    n, N = D.n, D.N
    ebar = expit(D.a)
    L = np.array([sum(D.g[k] * (((S >> k) & 1) - ebar[k]) for k in range(n)) for S in range(N)])
    betaS = np.array([np.sum(D.w * D.u * D.Pu[:, S]) for S in range(N)])
    b1 = betaS * L
    shb1 = shapley(b1, n)
    dda1 = np.zeros(n)
    for S in range(1, N):
        bS = np.array([betaS[A] * (L[A] - L[S]) if (A & ~S) == 0 else 0.0 for A in range(N)])
        dda1 += D.PS[S] * shapley(bS, n, players=members(S, n))
    W = shapley_weights(n)
    bbar_j = np.array([sum(W[j, B] * betaS[B | (1 << j)] for B in range(N) if not (B >> j) & 1) for j in range(n)])
    # DDA closed form: g_j · Σ_S P(S) 1{j∈S} · (Shapley-weighted β over A⊆S with j∈A)
    dd_j = np.zeros(n)
    for S in range(1, N):
        mem = members(S, n); Ws = shapley_weights(len(mem))
        for jj, j in enumerate(mem):
            others = [k for k in mem if k != j]
            tot = 0.0
            for r in range(1 << len(others)):
                B = sum(1 << others[i] for i in range(len(others)) if (r >> i) & 1)
                Bloc = sum(1 << [mem.index(k) for k in others][i] for i in range(len(others)) if (r >> i) & 1)
                tot += Ws[jj, Bloc] * betaS[B | (1 << j)]
            dd_j[j] += D.PS[S] * tot
    return shb1, dda1, bbar_j * D.g, dd_j * D.g, betaS

def e3():
    """E3 — first-order law: Sh_j(b) ≈ Sh_j(β_S·L(S)); with exposure-invariant outcome slope β̄: Sh_j(b) ≈ β̄ g_j and
    DDA_j − Sh_j(v_pop) ≈ β̄ g_j ē_j.  Ratios observed/predicted as the targeting scale → 0."""
    print("E3: first-order targeting-bias law (ratios exact/first-order; 'cs' = constant-slope closed form)")
    print("  cs = own-slope closed forms: Sh_j(b) ≈ g_j·β̄_j (β̄_j Shapley-weighted outcome slope on coalitions ∋ j); DDA bias_j ≈ g_j·E[β̄_j(Z); j∈Z]")
    print("  scale | Sh(b) exact          | ratio to 1st-order   | ratio to cs          || DDA bias exact       | ratio 1st-order      | ratio cs")
    for scale in [0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]:
        g = np.array([0.4, 1.0, 1.6]) * scale
        D = base_dgp(g=tuple(g))
        shb = shapley(D.bias_game(), D.n); ddab = D.dda() - D.sh_pop()
        shb1, dda1, cs1, cs2, betaS = first_order(D)
        print("  %.3f | %s | %s | %s || %s | %s | %s" % (scale, fmt(shb, 5), fmt(shb / shb1, 3), fmt(shb / cs1, 3), fmt(ddab, 5), fmt(ddab / dda1, 3), fmt(ddab / cs2, 3)))
    D = base_dgp(); _, _, _, _, betaS = first_order(D)
    print("  outcome U-slope β_S by exposure set (∅ … 123):", fmt(betaS, 4), " — slope varies %.1f× across S" % (betaS.max() / betaS.min()))
    # linear-probability outcome (constant slope) check: use a DGP where p is linear in U → closed form should be tight
    print("  constant-slope DGP (probit-linear approx via small β): ")
    for scale in [0.1, 0.4, 1.0]:
        D = DGP(a=[-0.5, -1.0, -1.5], g=np.array([0.4, 1.0, 1.6]) * scale, beta0=0.0, beta=0.3, theta=[0.3, 0.3, 0.0])  # p≈0.5 region: p(1−p) nearly flat
        shb = shapley(D.bias_game(), D.n); ddab = D.dda() - D.sh_pop()
        shb1, dda1, cs1, cs2, betaS = first_order(D)
        print("   scale=%.1f  Sh(b)/cs=%s   DDAbias/cs=%s   (β_S spread %.2f×)" % (scale, fmt(shb / cs1, 3), fmt(ddab / cs2, 3), betaS.max() / betaS.min()))

def e4(verbose=True):
    """E4 — the impossibility swap.  (An earlier discrete-U version did not converge to an exact match with the
    total-lift constraint; the continuous-U construction in E15 does, so E4 delegates to it.)"""
    return e15()

def e5():
    """E5 — Γ-sensitivity bounds on Sh_j(v_pop): global box, channelwise, monotone; robustness values."""
    print("E5: cross-journey odds-ratio sensitivity bounds on the causal Shapley credit (benchmark DGP)")
    D = base_dgp()
    gl = [1.0, 1.25, 1.5, 2.0, 3.0, 5.31]
    los, his, truth, Gtrue = sens_bounds(D, gl)
    Gk = true_channel_gammas(D)
    print("  truth Sh(v_pop)=%s  DDA (=Γ=1 point)=%s ; true global Γ*=%.2f ; true channelwise Γ_k*=%s" % (fmt(truth), fmt(D.dda()), Gtrue, fmt(Gk, 2)))
    print("  global box:")
    for G, l, h in zip(gl, los, his):
        print("    Γ=%.2f  [%s , %s]  covers=%s" % (G, fmt(l), fmt(h), np.all((l - 1e-9 <= truth) & (truth <= h + 1e-9))))
    print("  channelwise Γ_k (scaled copies of Γ_k*):")
    for sc in [0.25, 0.5, 1.0]:
        Gv = Gk ** sc
        (l,), (h,), _, _ = sens_bounds(D, [Gv], mode="channelwise")
        print("    Γ_k=%s  [%s , %s]  covers=%s  width=%s" % (fmt(Gv, 2), fmt(l), fmt(h), np.all((l - 1e-9 <= truth) & (truth <= h + 1e-9)), fmt(h - l)))
    (l,), (h,), _, _ = sens_bounds(D, [Gk], mode="channelwise", monotone=True)
    print("  channelwise + monotone (positive selection):  [%s , %s]  covers=%s width=%s" % (fmt(l), fmt(h), np.all((l - 1e-9 <= truth) & (truth <= h + 1e-9)), fmt(h - l)))
    for mode, mono in [("box", False), ("box", True), ("channelwise", False), ("channelwise", True)]:
        rv = robustness_values(D, mode=mode, monotone=mono)
        print("  robustness values [%s%s]: Γ where each channel's interval touches 0: %s ; Γ where DDA's top-2 ordering becomes ambiguous: %.2f"
              % (mode, ", monotone" if mono else "", fmt(rv["zero"], 2), rv["rank"]))

def shapley_coef(D):
    """coef[j,S,A]: Sh_j(v_pop) = Σ_{S,A} coef[j,S,A]·c(S,A)  (A ⊆ S)."""
    n, N = D.n, D.N
    W = shapley_weights(n)
    coef = np.zeros((n, N, N))
    for j in range(n):
        for B in range(N):
            if (B >> j) & 1: continue
            for S in range(N):
                coef[j, S, S & (B | (1 << j))] += W[j, B] * D.PS[S]
                coef[j, S, S & B] -= W[j, B] * D.PS[S]
    return coef

def sens_bounds(D, gammas, mode="box", monotone=False):
    """Sharp cell-wise bounds on Sh_j(v_pop) given the observational law.
    Sensitivity model on the unidentified cells c(S,A), A ⊊ S:
      mode='box':         odds c(S,A) ∈ odds c(A,A) · [1/Γ, Γ]                  (one global Γ)
      mode='channelwise': odds c(S,A) ∈ odds c(A,A) · Π_{k∈S∖A} [1/Γ_k, Γ_k]      (gammas = list of vectors)
      monotone=True:      positive selection only — c(S,A) ≥ c(A,A) (upper factor only)
    Returns lists lo, hi (per Γ), truth, and the true per-cell odds ratios' max."""
    n, N = D.n, D.N
    coef = shapley_coef(D)
    truth = np.einsum("jsa,sa->j", coef, D.c)
    los, his = [], []
    for G in gammas:
        Gv = np.full(n, float(G)) if mode == "box" else np.asarray(G, float)
        lo = np.zeros(n); hi = np.zeros(n)
        for j in range(n):
            for S in range(N):
                for A in range(N):
                    k = coef[j, S, A]
                    if k == 0: continue
                    m = D.c[A, A]
                    if A == S:
                        lo[j] += k * m; hi[j] += k * m; continue
                    fac = np.prod([Gv[q] for q in members(S & ~A, n)]) if mode == "channelwise" else Gv[0]
                    odds = m / (1 - m)
                    ch = (odds * fac) / (1 + odds * fac)
                    cl = m if monotone else (odds / fac) / (1 + odds / fac)
                    if k > 0: lo[j] += k * cl; hi[j] += k * ch
                    else:     lo[j] += k * ch; hi[j] += k * cl
        los.append(lo); his.append(hi)
    Gt = 1.0
    for S in range(N):
        for A in range(N):
            if (A & ~S) == 0 and A != S:
                o1 = D.c[S, A] / (1 - D.c[S, A]); o2 = D.c[A, A] / (1 - D.c[A, A])
                Gt = max(Gt, o1 / o2, o2 / o1)
    return los, his, truth, Gt

def true_channel_gammas(D):
    """Smallest per-channel Γ_k such that every unidentified cell satisfies the channelwise model: fit log-odds
    ratios r(S,A) = log[odds c(S,A)/odds c(A,A)] ≤ Σ_{k∈S∖A} log Γ_k by LP (minimise Σ log Γ_k)."""
    n, N = D.n, D.N
    rows, rhs = [], []
    for S in range(N):
        for A in range(N):
            if (A & ~S) == 0 and A != S:
                r = abs(np.log(D.c[S, A] / (1 - D.c[S, A])) - np.log(D.c[A, A] / (1 - D.c[A, A])))
                row = np.array([1.0 if (((S & ~A) >> k) & 1) else 0.0 for k in range(n)])
                rows.append(-row); rhs.append(-r)
    res = linprog(np.ones(n), A_ub=np.array(rows), b_ub=np.array(rhs), bounds=[(0, None)] * n)
    return np.exp(res.x)

def robustness_values(D, mode="box", monotone=False):
    grid = np.concatenate([np.linspace(1.0, 1.5, 51), np.linspace(1.5, 8, 131)])
    gam = grid if mode == "box" else [np.full(D.n, g) for g in grid]
    los, his, truth, _ = sens_bounds(D, gam, mode=mode, monotone=monotone)
    los, his = np.array(los), np.array(his)
    zero = np.full(D.n, np.inf)
    for j in range(D.n):
        idx = np.where(los[:, j] <= 0)[0]
        if len(idx): zero[j] = grid[idx[0]]
    point = los[0]                       # Γ=1: the point-identified (DDA) allocation
    order = np.argsort(-point); t1, t2 = order[0], order[1]
    idx = np.where(los[:, t1] <= his[:, t2])[0]
    rank = grid[idx[0]] if len(idx) else np.inf
    return {"zero": zero, "rank": rank, "grid": grid, "lo": los, "hi": his, "truth": truth, "point": point}

def e6():
    """E6 — interaction structure: Shapley of realized lift vs removal effects vs equal-split, reach-overlap model."""
    print("E6: unconfounded reach-overlap model — allocation rules disagree through interactions alone")
    for q in [(-3.0, -3.0, -3.0), (-1.5, -1.5, -1.5), (-0.5, -0.5, -0.5), (0.5, -1.0, -2.5)]:
        D = DGP(a=[0.0, 0.0, 0.0], g=[0, 0, 0], beta0=-2.0, beta=1.0, theta=q, model="reach")
        d = mobius(D.v_pop, D.n)
        over = sum((popcount(T) - 1) * d[T] for T in range(1, D.N))
        print("  q=%s: Sh(v_pop)=%s tau=%s eq-split=%s | Σtau−total=%+.4f (=Σ(|T|−1)D(T)=%+.4f) total=%.4f"
              % (fmt(expit(np.array(q)), 2), fmt(D.sh_pop()), fmt(D.tau()), fmt(D.last_touch_like()), D.tau().sum() - D.total_incremental(), over, D.total_incremental()))
    print("  logit synergy: theta2_12=+1.0")
    D = DGP(a=[0, 0, 0], g=[0, 0, 0], beta0=-2.5, beta=1.0, theta=[0.5, 0.5, 0.5], theta2=[[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    print("   Sh(v_pop)=%s tau=%s Σtau−total=%+.4f" % (fmt(D.sh_pop()), fmt(D.tau()), D.tau().sum() - D.total_incremental()))

def holdout_design_matrix(n, holdouts):
    """Row per holdout set H (bitmask): the arm exposes users to Z ∧ 1_{H^c}, identifying v_pop(H^c) = Σ_{T ⊆ H^c} D(T).
    Columns index T ≠ ∅ (Möbius basis)."""
    cols = [T for T in range(0, 1 << n)]          # includes T=∅ (the intercept v_pop(∅))
    M = np.zeros((len(holdouts), len(cols)))
    for r, H in enumerate(holdouts):
        Hc = ((1 << n) - 1) & ~H
        for ci, T in enumerate(cols):
            M[r, ci] = 1.0 if (T & ~Hc) == 0 else 0.0
    return M, cols

def shapley_target_vectors(n, cols):
    V = np.zeros((n, len(cols)))
    for j in range(n):
        for ci, T in enumerate(cols):
            if T and (T >> j) & 1: V[j, ci] = 1.0 / popcount(T)
    return V

def identified(M, V, cols, kcap=None, tol=1e-9):
    """Is each Shapley target identified from the design, optionally under a k-way interaction cap (D(T)=0, |T|>k)?"""
    keep = [ci for ci, T in enumerate(cols) if (kcap is None or popcount(T) <= kcap)]
    Mk, Vk = M[:, keep], V[:, keep]
    # v in rowspace(M) iff v ⟂ null(M): check residual of least squares
    out = []
    for j in range(V.shape[0]):
        x, res, rk, sv = np.linalg.lstsq(Mk.T, Vk[j], rcond=None)
        out.append(np.linalg.norm(Mk.T @ x - Vk[j]) < tol)
    return np.array(out), np.linalg.matrix_rank(Mk), len(keep)

def e7():
    """E7 — identification ladder: which holdout designs identify the causal Shapley allocation?"""
    print("E7: identification of Sh(v_pop) from holdout cells (Möbius rank conditions)")
    for n in [2, 3, 4, 5]:
        full = (1 << n) - 1
        singles = [1 << j for j in range(n)]
        pairs = [(1 << i) | (1 << j) for i in range(n) for j in range(i + 1, n)]
        designs = {
            "no-holdout+global": [0, full],
            "+singles": [0, full] + singles,
            "+singles+pairs": [0, full] + singles + pairs,
            "complement-singles (expose only j)": [0, full] + [full & ~(1 << j) for j in range(n)],
            "expose only j + singles": [0, full] + singles + [full & ~(1 << j) for j in range(n)],
        }
        print("  n=%d (unknown dividends: %d)" % (n, full))
        for name, H in designs.items():
            M, cols = holdout_design_matrix(n, H); V = shapley_target_vectors(n, cols)
            line = "    %-36s cells=%2d :" % (name, len(H))
            for kcap in [None, 2, 3] if n >= 3 else [None, 2]:
                idn, rk, ncol = identified(M, V, cols, kcap)
                line += "  cap=%s: %s (rank %d/%d)" % (kcap if kcap else "∞", "ID" if idn.all() else "no", rk, ncol)
            print(line)
    # minimal designs under pairwise cap via greedy search, n=3,4
    for n in [3, 4]:
        full = (1 << n) - 1
        cand = [H for H in range(1, full)]  # all proper nonempty holdouts
        chosen = [0, full]
        M, cols = holdout_design_matrix(n, chosen); V = shapley_target_vectors(n, cols)
        while not identified(M, V, cols, 2)[0].all():
            best = None
            for H in cand:
                if H in chosen: continue
                M2, _ = holdout_design_matrix(n, chosen + [H])
                rk = identified(M2, V, cols, 2)[1]
                if best is None or rk > best[0]: best = (rk, H)
            chosen.append(best[1]); M, cols = holdout_design_matrix(n, chosen)
        print("  n=%d greedy minimal design under pairwise cap: %d cells (holdouts %s)" % (n, len(chosen), [members(H, n) for H in chosen]))

def e8(N_total=2_000_000, reps=200, seed=1):
    """E8 — power: precision of experimentally identified Sh(v_pop) vs a single-channel lift test, same total N."""
    print("E8: sampling precision of experimental Shapley (n=3, pairwise-cap design) vs single lift test, N=%d" % N_total)
    D = base_dgp()  # confounded — experiments don't care
    n = D.n; full = D.N - 1
    rng = np.random.default_rng(seed)
    design = [0, full] + [1 << j for j in range(n)] + [3, 5, 6]     # global, none, singles, pairs → 8 cells
    M, cols = holdout_design_matrix(n, design); V = shapley_target_vectors(n, cols)
    keep = [ci for ci, T in enumerate(cols) if popcount(T) <= 2]
    Mk, Vk = M[:, keep], V[:, keep]
    A = Vk @ np.linalg.pinv(Mk)          # Sh_j = Σ_cells A[j,cell] * v_pop(H^c)
    d = mobius(D.v_pop, n)
    print("  dividends of v_pop by order: 1st %s 2nd %s 3rd %s" % (fmt(d[[1,2,4]],5), fmt(d[[3,5,6]],5), fmt(d[[7]],5)))
    # exact cell means
    cell_means = np.array([D.v_pop[full & ~H] for H in design])
    est = []
    for r in range(reps):
        Ncell = N_total // len(design)
        ybar = rng.binomial(Ncell, cell_means) / Ncell
        est.append(A @ ybar)
    est = np.array(est)
    truth = D.sh_pop()
    print("  truth Sh(v_pop)=%s ; exp-Shapley mean=%s sd=%s" % (fmt(truth), fmt(est.mean(0)), fmt(est.std(0))))
    # single-channel lift test with same N split in two arms: tau_j
    for j in range(n):
        m1, m0 = D.v_pop[full], D.v_pop[full & ~(1 << j)]
        sd = np.sqrt(m1 * (1 - m1) / (N_total / 2) + m0 * (1 - m0) / (N_total / 2))
        print("  single lift test ch%d: tau=%.5f sd=%.5f  → Shapley sd/tau sd = %.2f" % (j + 1, m1 - m0, sd, est.std(0)[j] / sd))
    print("  design weights |A| row sums:", fmt(np.abs(A).sum(1), 2), "(variance amplification vs a 2-arm test)")
    return est

def e9(Ns=(1e4, 1e5, 1e6), reps=30, seed=2):
    """E9 — finite-sample DDA: sampling noise vs confounding bias; when does bias dominate?"""
    print("E9: finite-sample journey-level DDA (bias vs noise)")
    D = base_dgp()
    truth, popbias = D.sh_pop(), D.dda() - D.sh_pop()
    print("  truth Sh(v_pop)=%s ; population DDA bias=%s" % (fmt(truth), fmt(popbias)))
    rng = np.random.default_rng(seed)
    for N in Ns:
        ests = []
        for r in range(reps):
            S, Y = D.sample(int(N), rng)
            ests.append(dda_from_sample(S, Y, D.n)[0])
        ests = np.array(ests)
        print("  N=%8d  mean DDA=%s  sd=%s  |bias|/sd=%s" % (N, fmt(ests.mean(0)), fmt(ests.std(0)), fmt(np.abs(popbias) / ests.std(0), 1)))

def e10(N=200_000, reps=200, seed=3):
    """E10 — negative/power control: a global-holdout Hausman test of 'DDA total = incremental total'."""
    print("E10: power control — global holdout vs DDA total (N=%d users, half held out)" % N)
    for g3, label in [((0, 0, 0), "no confounding (null)"), ((0.4, 1.0, 1.6), "benchmark confounding"), ((0.2, 0.5, 0.8), "half confounding")]:
        D = base_dgp(g=g3)
        rng = np.random.default_rng(seed)
        tstats = []
        for r in range(reps):
            S, Y = D.sample(N // 2, rng)                       # exposed arm
            dda, v, PS = dda_from_sample(S, Y, D.n)
            dda_total = dda.sum()                              # = ȳ − v̂(∅)
            # holdout arm: nobody exposed → Y(0)
            u = rng.standard_normal(N // 2)
            Y0 = (rng.random(N // 2) < expit(D.beta0 + D.beta * u)).astype(int)
            incr = Y.mean() - Y0.mean()
            # se via delta: var(ȳ)+var(v̂∅)+var(Ȳ0) (ȳ common cancels in the difference dda_total − incr = Ȳ0 − v̂(∅))
            n0 = np.sum(S == 0)
            se = np.sqrt(Y0.var() / len(Y0) + v[0] * (1 - v[0]) / max(n0, 1))
            tstats.append((dda_total - incr) / se)
        t = np.array(tstats)
        print("  %-24s mean t=%.2f  sd=%.2f  reject@5%%: %.2f" % (label, t.mean(), t.std(), np.mean(np.abs(t) > 1.96)))

def e11():
    """E11 — robustness sweep of the first-order law across link, confounder distribution, effect heterogeneity."""
    print("E11: robustness of the first-order bias operator and the own-slope closed form (ratios exact/predicted)")
    configs = {"logit, normal U": dict(), "probit link": dict(model="probit"), "bimodal U": dict(udist="bimodal"),
               "skewed U": dict(udist="skew"), "U-dependent persuasion (reach, α=1)": "reach"}
    for name, kw in configs.items():
        for scale in [0.1, 0.25, 1.0]:
            g = np.array([0.4, 1.0, 1.6]) * scale
            if kw == "reach":
                D = DGP(a=[-0.5, -1, -1.5], g=g, beta0=-2.5, beta=1.0, theta=[-1.5, -1.5, -8.0], model="reach", reach_alpha=1.0)
            else:
                D = base_dgp(g=tuple(g), **kw)
            shb = shapley(D.bias_game(), D.n); ddab = D.dda() - D.sh_pop()
            shb1, dda1, cs1, cs2, betaS = first_order(D)
            print("  %-36s scale=%.2f  Sh(b)/1st=%s  Sh(b)/cs=%s | DDAbias/1st=%s  DDAbias/cs=%s" % (name, scale, fmt(shb / shb1, 2), fmt(shb / cs1, 2), fmt(ddab / dda1, 2), fmt(ddab / cs2, 2)))

def e12(seeds=range(5)):
    """E12 — seed sensitivity of the MC experiments (E9 at N=1e5, E10 benchmark)."""
    print("E12: seed sensitivity")
    D = base_dgp()
    for s in seeds:
        rng = np.random.default_rng(100 + s)
        ests = np.array([dda_from_sample(*D.sample(100_000, rng), D.n)[0] for _ in range(20)])
        print("  seed %d: DDA mean=%s sd=%s" % (s, fmt(ests.mean(0)), fmt(ests.std(0))))

def e13():
    """E13 — exchangeable targeting: is the bias a uniform shift (rank-preserving)?  How much heterogeneity flips ranks?"""
    print("E13: exchangeable vs heterogeneous targeting")
    for g in [1.0, 1.5, 2.0]:
        D = base_dgp(a=(-1.0, -1.0, -1.0), g=(g, g, g), theta=(0.6, 0.4, 0.2))
        shb = shapley(D.bias_game(), D.n); ddab = D.dda() - D.sh_pop()
        print("  g=%.1f (equal a): Sh(b)=%s spread=%.1e | DDA bias=%s spread=%.1e" % (g, fmt(shb), np.ptp(shb), fmt(ddab), np.ptp(ddab)))
    D = base_dgp(a=(-0.5, -1.0, -1.5), g=(1.2, 1.2, 1.2), theta=(0.6, 0.4, 0.2))
    print("  equal g, unequal a: Sh(b)=%s | DDA bias=%s (reach enters through e_j)" % (fmt(shapley(D.bias_game(), D.n)), fmt(D.dda() - D.sh_pop())))
    # rank flip threshold: ch1 true effect 0.6 > ch2 0.4; raise g2 until DDA ranks 2 above 1
    for g2 in np.linspace(1.0, 3.0, 9):
        D = base_dgp(a=(-1.0, -1.0, -1.0), g=(1.0, g2, 1.0), theta=(0.6, 0.4, 0.2))
        sp, dd = D.sh_pop(), D.dda()
        print("   g2=%.2f  truth ranks %s  DDA ranks %s  DDA=%s" % (g2, np.argsort(-sp) + 1, np.argsort(-dd) + 1, fmt(dd)))

def e14():
    """E14 — the exchange rate: how much targeting buys the same DDA credit as real effect."""
    print("E14: ∂DDA_j/∂g_j vs ∂DDA_j/∂theta_j at the benchmark (finite differences)")
    D = base_dgp(); h = 1e-3
    for j in range(D.n):
        g = np.array([0.4, 1.0, 1.6]); th = np.array([0.5, 0.5, 0.0])
        gp = g.copy(); gp[j] += h; thp = th.copy(); thp[j] += h
        dg = (base_dgp(g=tuple(gp)).dda()[j] - D.dda()[j]) / h
        dt = (base_dgp(theta=tuple(thp)).dda()[j] - D.dda()[j]) / h
        dt_true = (base_dgp(theta=tuple(thp)).sh_pop()[j] - D.sh_pop()[j]) / h
        print("  ch%d: ∂DDA/∂g=%.4f  ∂DDA/∂θ=%.4f  ∂Sh_pop/∂θ=%.4f  → targeting/effect exchange rate = %.2f" % (j + 1, dg, dt, dt_true, dg / dt))

def e15():
    """E15 — monotone swap: impossibility persists under monotone targeting AND monotone effects (n=2, Gaussian U),
    with the total lift also matched (so a global holdout cannot separate A from B)."""
    print("E15: swap with sign restrictions (g>0, beta>0, theta>=0), continuous Gaussian U, matched total lift")
    parA = dict(a=[-0.5, -1.5], g=[0.6, 1.8], beta0=-2.0, beta=1.2)
    DA = DGP(theta=[0.8, 0.0], **parA); totA = DA.total_incremental()
    TA = np.stack([DA.PS * (1 - DA.v_obs), DA.PS * DA.v_obs], 1)
    def unpack(x):
        return dict(a=[x[0], x[1]], g=[np.exp(x[2]), np.exp(x[3])], beta0=x[4], beta=np.exp(x[5]), beta2=x[7]), np.array([0.0, np.exp(x[6])])
    best = None; rng = np.random.default_rng(1)
    for t in range(30):
        x0 = np.array([-0.5, -1.5, np.log(0.6), np.log(1.8), -2.0, np.log(1.2), np.log(0.8), 0.0]) + rng.normal(0, 0.5, 8)
        def resid(x):
            par, th = unpack(x); D = DGP(theta=th, **par)
            return np.concatenate([(np.stack([D.PS * (1 - D.v_obs), D.PS * D.v_obs], 1) - TA).ravel() * 100, [(D.total_incremental() - totA) * 100]])
        r = least_squares(resid, x0, max_nfev=3000)
        if best is None or r.cost < best.cost: best = r
    par, th = unpack(best.x); DB = DGP(theta=th, **par)
    TB = np.stack([DB.PS * (1 - DB.v_obs), DB.PS * DB.v_obs], 1)
    print("  max|table diff|=%.1e |Δtotal|=%.1e ; B params: a=%s g=%s beta0=%.3f beta=%.3f beta2=%.3f theta=%s"
          % (np.max(np.abs(TA - TB)), abs(DB.total_incremental() - totA), fmt(par["a"], 3), fmt(par["g"], 3), par["beta0"], par["beta"], par["beta2"], fmt(th, 3)))
    for name, D in [("A", DA), ("B", DB)]:
        print("  %s: Sh(v_pop)=%s total=%.4f  DDA=%s  Sh(v_obs)=%s" % (name, fmt(D.sh_pop()), D.total_incremental(), fmt(D.dda()), fmt(D.sh_obs())))
    return np.max(np.abs(TA - TB)), DA, DB

def e16():
    """E16 — sharpness: where does the truth sit inside the Γ* intervals (box vs channelwise vs monotone)?"""
    print("E16: bound sharpness — position of truth within the interval at the TRUE sensitivity parameters")
    for g in [(0.2, 0.5, 0.8), (0.4, 1.0, 1.6), (0.8, 2.0, 3.2), (1.0, 1.0, 1.0)]:
        D = base_dgp(g=g)
        _, _, truth, Gt = sens_bounds(D, [1.0]); Gk = true_channel_gammas(D)
        (lb,), (hb,), _, _ = sens_bounds(D, [Gt])
        (lc,), (hc,), _, _ = sens_bounds(D, [Gk], mode="channelwise")
        (lm,), (hm,), _, _ = sens_bounds(D, [Gk], mode="channelwise", monotone=True)
        print("  g=%s Γ*=%.2f Γ_k*=%s" % (fmt(g, 1), Gt, fmt(Gk, 2)))
        for name, l, h in [("box", lb, hb), ("channelwise", lc, hc), ("chan+monotone", lm, hm)]:
            print("    %-14s width=%s  position of truth=%s  covers=%s" % (name, fmt(h - l, 4), fmt((truth - l) / (h - l), 2), np.all((l - 1e-9 <= truth) & (truth <= h + 1e-9))))

def layered_design(n, m):
    """Cells with ≤ m channels held out or ≤ m channels exposed (plus global holdout / no holdout)."""
    full = (1 << n) - 1
    H = set([0, full])
    for X in range(1 << n):
        if popcount(X) <= m: H.add(X); H.add(full & ~X)
    return sorted(H)

def e17():
    """E17 — the layered-design theorem: under a k-way interaction cap, the layer m=k−1 design identifies Sh(v_pop)."""
    print("E17: minimal layer m identifying Sh(v_pop) under a k-way cap (cells in the layered design vs 2^n)")
    for n in [3, 4, 5, 6, 7, 8]:
        line = "  n=%d:" % n
        for k in range(2, min(n, 7) + 1):
            found = None
            for m in range(0, n):
                H = layered_design(n, m); M, cols = holdout_design_matrix(n, H); V = shapley_target_vectors(n, cols)
                if identified(M, V, cols, k)[0].all():
                    found = (m, len(H)); break
            line += "  cap k=%d → m=%d (%d cells of %d)" % (k, found[0], found[1], 1 << n)
        print(line)
    # closed form check: under pairwise cap Sh_j = ½[v({j}) − v(∅) + v([n]) − v([n]∖j)]
    D = DGP(a=[-0.5, -1, -1.5], g=[0.4, 1.0, 1.6], beta0=-2.5, beta=1.0, theta=[0.5, 0.5, 0.3])
    v = D.v_pop; full = D.N - 1
    cf = np.array([0.5 * (v[1 << j] - v[0] + v[full] - v[full & ~(1 << j)]) for j in range(D.n)])
    print("  benchmark-with-3-active: closed form ½(solo+removal)=%s vs Sh(v_pop)=%s (diff = 3-way dividend/6 = %s)" % (fmt(cf, 5), fmt(D.sh_pop(), 5), fmt(mobius(v, D.n)[7] / 6, 5)))

def e18(N_total=2_000_000, reps=200, seed=4):
    """E18 — attack on the pairwise-cap design: bias when 3-way dividends are present (logit with 3 active channels + synergy)."""
    print("E18: pairwise-cap design under true 3-way interaction (bias vs sampling sd at N=%d)" % N_total)
    for th2, label in [(0.0, "logit, no explicit synergy"), (1.0, "pairwise synergy θ12=1"), (2.0, "strong synergy θ12=θ23=2")]:
        t2 = np.zeros((3, 3))
        if th2: t2[0, 1] = t2[1, 0] = th2
        if th2 > 1.5: t2[1, 2] = t2[2, 1] = th2
        D = DGP(a=[-0.5, -1, -1.5], g=[0.4, 1.0, 1.6], beta0=-2.5, beta=1.0, theta=[0.5, 0.5, 0.3], theta2=t2)
        n = 3; full = 7
        design = layered_design(n, 1)
        M, cols = holdout_design_matrix(n, design); V = shapley_target_vectors(n, cols)
        keep = [ci for ci, T in enumerate(cols) if popcount(T) <= 2]
        A = V[:, keep] @ np.linalg.pinv(M[:, keep])
        cell = np.array([D.v_pop[full & ~H] for H in design])
        est_mean = A @ cell
        rng = np.random.default_rng(seed); Nc = N_total // len(design)
        sd = np.std([A @ (rng.binomial(Nc, cell) / Nc) for _ in range(reps)], axis=0)
        d = mobius(D.v_pop, n)
        print("  %-28s truth=%s  cap-design estimand=%s  bias=%s  sd=%s  3-way dividend=%.5f (2nd: %s)"
              % (label, fmt(D.sh_pop(), 4), fmt(est_mean, 4), fmt(est_mean - D.sh_pop(), 4), fmt(sd, 4), d[7], fmt(d[[3, 5, 6]], 4)))

def e19():
    """E19 — attack: does a PARAMETRIC single-index family identify the allocation at n=3 (over-identified by counting)?
    Search for a swap (ch3 null ↔ ch1 null) inside (i) the rigid logit/logit/Gaussian family, (ii) the same family
    with a 2-component mixture for U.  Residual ≫ 0 means the parametric family is identified by functional form."""
    print("E19: parametric identification at n=3 by functional form?  (min table mismatch of a ch1-null DGP vs the ch3-null truth)")
    DA = base_dgp(theta=(0.5, 0.5, 0.0))
    TA = np.stack([DA.PS * (1 - DA.v_obs), DA.PS * DA.v_obs], 1); totA = DA.total_incremental()
    x, w = hermegauss(60); w = w / w.sum()
    def build(x_, mix):
        a = x_[:3]; g = np.exp(x_[3:6]); b0 = x_[6]; b = np.exp(x_[7]); th = np.array([0.0, np.exp(x_[8]), np.exp(x_[9])])
        if mix:
            mu, ls, lp = x_[10], x_[11], x_[12]; pm = expit(lp); sd = np.exp(ls)
            u = np.concatenate([x, mu + sd * x]); ww = np.concatenate([(1 - pm) * w, pm * w])
            # standardize U to mean 0 var 1 (β absorbs scale)
            m = ww @ u; v = ww @ (u - m) ** 2; u = (u - m) / np.sqrt(v)
            return DGP(a=a, g=g, beta0=b0, beta=b, theta=th, ugrid=u, uw=ww)
        return DGP(a=a, g=g, beta0=b0, beta=b, theta=th)
    for mix in [False, True]:
        rng = np.random.default_rng(7); best = None
        for t in range(25 if not mix else 25):
            x0 = np.concatenate([[-0.5, -1, -1.5], np.log([0.4, 1, 1.6]), [-2.5, 0.0], np.log([0.5, 0.5])]) + rng.normal(0, 0.4, 10)
            if mix: x0 = np.concatenate([x0, [rng.normal(0, 1.5), rng.normal(0, 0.5), rng.normal(0, 1)]])
            def resid(x_):
                D = build(x_, mix)
                return np.concatenate([(np.stack([D.PS * (1 - D.v_obs), D.PS * D.v_obs], 1) - TA).ravel() * 100, [(D.total_incremental() - totA) * 100]])
            r = least_squares(resid, x0, max_nfev=2000)
            if best is None or r.cost < best.cost: best = r
        D = build(best.x, mix)
        err = np.max(np.abs(np.stack([D.PS * (1 - D.v_obs), D.PS * D.v_obs], 1) - TA))
        print("  %-28s max|table diff|=%.1e  |Δtotal|=%.1e  B: Sh(v_pop)=%s (A: %s)  DDA_B=%s DDA_A=%s"
              % ("rigid family" if not mix else "mixture-U family", err, abs(D.total_incremental() - totA), fmt(D.sh_pop()), fmt(DA.sh_pop()), fmt(D.dda()), fmt(DA.dda())))

def e20(N_total=2_000_000, reps=300, seed=5):
    """E20 — design economics: the 'expose-only-j' layered design vs the 'holdouts of size ≤2' design at equal N:
    precision of Sh_j and forgone conversions (cost of the intervention)."""
    print("E20: two pairwise-cap designs at equal N=%d — precision and forgone lift" % N_total)
    D = DGP(a=[-0.5, -1, -1.5, -2.0], g=[0.4, 1.0, 1.6, 0.8], beta0=-2.5, beta=1.0, theta=[0.5, 0.5, 0.3, 0.4])
    n, full = 4, 15
    pairs = [(1 << i) | (1 << j) for i in range(n) for j in range(i + 1, n)]
    designs = {"layered m=1 (global, none, single holdouts, single exposures)": layered_design(n, 1),
               "holdouts of size ≤2 (global, none, singles, pairs)": [0, full] + [1 << j for j in range(n)] + pairs}
    rng = np.random.default_rng(seed)
    for name, H in designs.items():
        M, cols = holdout_design_matrix(n, H); V = shapley_target_vectors(n, cols)
        keep = [ci for ci, T in enumerate(cols) if popcount(T) <= 2]
        A = V[:, keep] @ np.linalg.pinv(M[:, keep])
        cell = np.array([D.v_pop[full & ~h] for h in H]); Nc = N_total // len(H)
        est = np.array([A @ (rng.binomial(Nc, cell) / Nc) for _ in range(reps)])
        forgone = sum(Nc * (D.v_pop[full] - D.v_pop[full & ~h]) for h in H)
        print("  %-64s cells=%d  sd(Sh)=%s  forgone conversions=%.0f (%.2f%% of the panel's lift)"
              % (name, len(H), fmt(est.std(0), 5), forgone, 100 * forgone / (N_total * (D.v_pop[full] - D.v_pop[0]))))
    # optimal (Neyman-style) allocation for the layered design: minimise Σ_j Var(Sh_j) s.t. Σ N_c = N
    H = layered_design(n, 1); M, cols = holdout_design_matrix(n, H); V = shapley_target_vectors(n, cols)
    keep = [ci for ci, T in enumerate(cols) if popcount(T) <= 2]; A = V[:, keep] @ np.linalg.pinv(M[:, keep])
    print("  cell weights |A| row sums (layered): %s ; 3rd-order dividends of this DGP: %s" % (fmt(np.abs(A).sum(1), 2), fmt(mobius(D.v_pop, n)[[7, 11, 13, 14]], 5)))
    cell = np.array([D.v_pop[full & ~h] for h in H]); var_c = cell * (1 - cell)
    wgt = np.sqrt((A ** 2).sum(0) * var_c); Nopt = N_total * wgt / wgt.sum()
    sd_opt = np.sqrt(((A ** 2) * (var_c / Nopt)[None, :]).sum(1)); sd_eq = np.sqrt(((A ** 2) * (var_c / (N_total / len(H)))[None, :]).sum(1))
    print("  layered design, optimal cell allocation: shares=%s  sd(Sh)=%s vs equal %s" % (fmt(Nopt / N_total, 2), fmt(sd_opt, 5), fmt(sd_eq, 5)))

def e21(n=6, Ns=(3e4, 1e5, 1e6), reps=20, seed=6):
    """E21 — attack on the journey-level lemma: sparse cells at n=6 (64 journeys) — finite-sample DDA bias with NO confounding."""
    print("E21: unconfounded n=%d DDA with sparse journey cells (cell-filling bias)" % n)
    D = DGP(a=[-0.5, -1, -1.5, -2, -2.5, -3], g=[0] * 6, beta0=-2.5, beta=1.0, theta=[0.5, 0.5, 0.3, 0.3, 0.2, 0.0])
    truth = D.sh_pop()
    print("  truth Sh(v_pop)=%s" % fmt(truth))
    rng = np.random.default_rng(seed)
    for N in Ns:
        ests = np.array([dda_from_sample(*D.sample(int(N), rng), n)[0] for _ in range(reps)])
        print("  N=%8d  mean=%s  sd=%s  bias/truth=%s  empty cells≈%.0f%%" % (N, fmt(ests.mean(0)), fmt(ests.std(0)), fmt((ests.mean(0) - truth) / np.maximum(truth, 1e-9), 2),
              100 * np.mean(np.bincount(D.sample(int(N), rng)[0], minlength=64) < 30)))

def e22():
    """E22 — the reach gap without confounding: Shao–Li population-game Shapley vs journey-level DDA (= Sh(v_pop))."""
    print("E22: unconfounded — population-game Shapley Sh(v_obs)=Sh(v*) vs journey-level DDA=Sh(v_pop), equal per-exposure effects, unequal reach")
    for a in [(-0.5, -0.5, -0.5), (-0.5, -1.5, -2.5), (0.5, -1.0, -3.0)]:
        D = DGP(a=a, g=[0, 0, 0], beta0=-2.5, beta=1.0, theta=[0.5, 0.5, 0.5])
        so, dd = D.sh_obs(), D.dda()
        print("  reach=%s  Sh(v*) shares=%s  Sh(v_pop) shares=%s  (levels %s vs %s)" % (fmt(expit(np.array(a)), 2), fmt(so / so.sum(), 3), fmt(dd / dd.sum(), 3), fmt(so, 4), fmt(dd, 4)))

def quick():
    t0 = time.time()
    for f in [e1, e2, e3, e5, e6, e7, e17, e18, e13, e14, e11, e22]:
        f(); print()
    e9(Ns=(1e5,), reps=10); print()
    e10(N=100_000, reps=40); print()
    e8(N_total=2_000_000, reps=100); print()
    e15(); print(); e16(); print(); e19(); print(); e20(); print(); e21(); print(); e12(seeds=range(2))
    print("done in %.0fs" % (time.time() - t0))

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "quick"
    globals()[cmd]()
