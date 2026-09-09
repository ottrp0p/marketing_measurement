"""Dive 16 - Inference in the measurement-allocation feedback loop (C4 / M7).
Self-contained: linear closed loop (loop engine), Hill closed loop, first-order theory check, experiment runners.
Usage: python 16_feedback_loop_inference.py quick|all|E1..E13|theory   (needs numpy, scipy). all ~3 min; theory ~1 min.
"""
import sys
import numpy as np
from scipy import stats, special

def nig_log_marginal(Sxx, n, mu0, lam0, a0, b0):
    XtX, Xty, yty = Sxx
    Ln = lam0[None] + XtX
    rhs = (lam0 @ mu0)[None] + Xty
    mun = np.linalg.solve(Ln, rhs[..., None])[..., 0]
    an = a0 + n / 2
    bn = b0 + 0.5 * (yty + mu0 @ lam0 @ mu0 - np.einsum('ri,rij,rj->r', mun, Ln, mun))
    logdet0 = np.linalg.slogdet(lam0)[1]
    logdetn = np.linalg.slogdet(Ln)[1]
    return (special.gammaln(an) - special.gammaln(a0) + a0 * np.log(b0) - an * np.log(bn)
            + 0.5 * (logdet0 - logdetn) - (n / 2) * np.log(2 * np.pi))

def cs_radius(logm, n, delta):
    return (n / (2 * np.pi)) * np.exp(-2 * (logm + np.log(delta)) / n - 1)

def run_loop(T, R, g, policy='linear', sig=1.0, a=10.0, b=2.0, x0=5.0, xi_sd=1.0, T0=8,
             seed=0, skew=0.0, band=0.0, x_hi=None, x_lo=None, refresh=1, rho=1.0,
             xi_decay=0.0, delta=0.05, prior=None, record=None, ar1=0.0, hetero=0.0, bref=None, burn_amp=0.2, skew_sign=1.0, hold_noise=True):
    rng = np.random.default_rng(seed)
    if prior is None:
        prior = dict(mu0=np.array([a, b]), lam0=np.diag([1e-2, 1e-2]), a0=2.0, b0=1.0)
    if bref is None: bref = b
    n = 0
    S1 = np.zeros(R); Sx = np.zeros(R); Sy = np.zeros(R); Sxx = np.zeros(R); Sxy = np.zeros(R); Syy = np.zeros(R)
    bhat_used = np.full(R, bref); lo_used = np.full(R, -np.inf); hi_used = np.full(R, np.inf)
    x_prev = np.full(R, x0); e_prev = np.zeros(R); base_prev = np.full(R, x0)
    if x_hi is None: x_hi = x0 * 1.5
    if x_lo is None: x_lo = x0 * 0.5
    cover_naive = np.ones(R, bool); cover_cs = np.ones(R, bool)
    miss_naive_t = np.full(R, -1); miss_cs_t = np.full(R, -1)
    hist = {} if record is None else {k: [] for k in record}
    xs = np.zeros((T, R)); bh = se = hw = rss = Dxx = miss = missc = None
    # burn-in-prior CS state
    P1 = np.zeros(R); Px = np.zeros(R); Py = np.zeros(R); Pxx = np.zeros(R); Pxy = np.zeros(R); Pyy = np.zeros(R)
    bprior = None; cover_csb = np.ones(R, bool); hwb = np.full(R, np.inf); missb = np.zeros(R, bool)
    for t in range(1, T + 1):
        if t <= T0:
            target = np.full(R, x0 * (1 + burn_amp * (-1) ** t))
        elif policy == 'linear':
            target = x0 + g * (bhat_used - bref)
        elif policy == 'deadband':
            up = bhat_used > bref + band; dn = bhat_used < bref - band
            target = np.where(up, x_hi, np.where(dn, x_lo, x_prev))
        elif policy == 'citrig':
            up = lo_used > bref; dn = hi_used < bref
            target = np.where(up, x_hi, np.where(dn, x_lo, x_prev if hold_noise else base_prev))
        elif policy == 'rw':
            target = x_prev
        elif policy == 'deadband_base':
            up = bhat_used > bref + band; dn = bhat_used < bref - band
            target = np.where(up, x_hi, np.where(dn, x_lo, base_prev))
        else:
            raise ValueError
        xsd = xi_sd * (t ** (-xi_decay)) if xi_decay > 0 else xi_sd
        xi = rng.normal(0, xsd, R) if xsd > 0 else 0.0
        base_prev = rho * target + (1 - rho) * x_prev
        x = base_prev + xi
        x = np.maximum(x, 0.05 * x0)
        if skew > 0:
            k = 4.0 / skew ** 2
            e = skew_sign * (rng.gamma(k, 1.0, R) - k) / np.sqrt(k)
        else:
            e = rng.normal(0, 1, R)
        if ar1:
            e = ar1 * e_prev + np.sqrt(1 - ar1 ** 2) * e
        e_prev = e
        sd = sig * (1 + hetero * (x - x0) / x0) if hetero else sig
        y = a + b * x + sd * e
        xs[t - 1] = x
        n += 1; S1 += 1; Sx += x; Sy += y; Sxx += x * x; Sxy += x * y; Syy += y * y
        if t > T0:
            P1 += 1; Px += x; Py += y; Pxx += x * x; Pxy += x * y; Pyy += y * y
        if t == T0:
            # vague NIG prior -> burn-in posterior (per rep): mu_n, Lam_n, a_n, b_n
            lam_v = np.diag([1e-4, 1e-4]); mu_v = np.array([0.0, 0.0]); a_v = 1.0; b_v = 0.5
            XtX0 = np.stack([np.stack([S1, Sx], -1), np.stack([Sx, Sxx], -1)], -2)
            Ln0 = lam_v[None] + XtX0; mun0 = np.linalg.solve(Ln0, np.stack([Sy, Sxy], -1)[..., None])[..., 0]
            an0 = a_v + n / 2
            bn0 = b_v + 0.5 * (Syy - np.einsum('ri,rij,rj->r', mun0, Ln0, mun0))
            bprior = (mun0, Ln0, an0, bn0)
        if n >= 3:
            xbar = Sx / n; ybar = Sy / n
            Dxx = Sxx - n * xbar ** 2; Dxy = Sxy - n * xbar * ybar
            bh = Dxy / np.maximum(Dxx, 1e-12); ah = ybar - bh * xbar
            rss = np.maximum(Syy - n * ybar ** 2 - bh * Dxy, 1e-12)
            s2 = rss / (n - 2)
            se = np.sqrt(s2 / np.maximum(Dxx, 1e-12))
            tq = stats.t.ppf(0.975, n - 2)
            miss = np.abs(bh - b) > tq * se
            miss_naive_t[miss & cover_naive] = t
            cover_naive &= ~miss
            XtX = np.stack([np.stack([S1, Sx], -1), np.stack([Sx, Sxx], -1)], -2)
            Xty = np.stack([Sy, Sxy], -1)
            logm = nig_log_marginal((XtX, Xty, Syy), n, **prior)
            Rstar = cs_radius(logm, n, delta)
            inv_bb = S1 / np.maximum(S1 * Sxx - Sx ** 2, 1e-12)
            hw = np.sqrt(np.maximum(Rstar - rss, 0) * inv_bb)
            missc = (np.abs(bh - b) > hw) | (Rstar < rss)
            miss_cs_t[missc & cover_cs] = t
            cover_cs &= ~missc
            if bprior is not None and P1[0] >= 3:
                mun0, Ln0, an0, bn0 = bprior; m = int(P1[0])
                XtXp = np.stack([np.stack([P1, Px], -1), np.stack([Px, Pxx], -1)], -2)
                Lnp = Ln0 + XtXp
                rhsp = np.einsum('rij,rj->ri', Ln0, mun0) + np.stack([Py, Pxy], -1)
                munp = np.linalg.solve(Lnp, rhsp[..., None])[..., 0]
                anp = an0 + m / 2
                bnp = bn0 + 0.5 * (Pyy + np.einsum('ri,rij,rj->r', mun0, Ln0, mun0) - np.einsum('ri,rij,rj->r', munp, Lnp, munp))
                logmb = (special.gammaln(anp) - special.gammaln(an0) + an0 * np.log(bn0) - anp * np.log(bnp)
                         + 0.5 * (np.linalg.slogdet(Ln0)[1] - np.linalg.slogdet(Lnp)[1]) - (m / 2) * np.log(2 * np.pi))
                Rb = (m / (2 * np.pi)) * np.exp(-2 * (logmb + np.log(delta)) / m - 1)
                xb = Px / m; yb = Py / m; Dxxp = Pxx - m * xb ** 2; Dxyp = Pxy - m * xb * yb
                bhp = Dxyp / np.maximum(Dxxp, 1e-12); rssp = np.maximum(Pyy - m * yb ** 2 - bhp * Dxyp, 1e-12)
                inv_bbp = P1 / np.maximum(P1 * Pxx - Px ** 2, 1e-12)
                hwb = np.sqrt(np.maximum(Rb - rssp, 0) * inv_bbp)
                missb = (np.abs(bhp - b) > hwb) | (Rb < rssp)
                cover_csb &= ~missb
                bhp_last = bhp
            if t <= T0 or (t - T0) % refresh == 0:
                bhat_used = bh.copy(); lo_used = bh - tq*se; hi_used = bh + tq*se
            if record:
                snap = {'bh': bh, 'se': se, 'hw': hw, 'hwb': hwb, 'bhp': bhp_last if bprior is not None and P1[0]>=3 else bh, 'x': x, 'Dxx': Dxx, 'miss': miss, 'missc': missc}
                for k in record: hist[k].append(snap[k].copy())
        x_prev = x
    return dict(bh=bh, ah=ah, se=se, hw=hw, rss=rss, Dxx=Dxx, miss_final=miss, missc_final=missc,
                cover_naive=cover_naive, cover_cs=cover_cs, cover_csb=cover_csb, hwb=hwb, missb_final=missb, miss_naive_t=miss_naive_t, miss_cs_t=miss_cs_t,
                xs=xs, hist=hist)

def summarize(o, b=2.0):
    R = len(o['bh']); d = {}
    d['bias'] = o['bh'].mean() - b; d['bias_se'] = o['bh'].std() / np.sqrt(R)
    d['sd'] = o['bh'].std(); d['mean_se'] = o['se'].mean()
    d['cov_final_naive'] = 1 - o['miss_final'].mean(); d['cov_final_cs'] = 1 - o['missc_final'].mean()
    d['cov_unif_naive'] = o['cover_naive'].mean(); d['cov_unif_cs'] = o['cover_cs'].mean()
    d['hw_naive'] = (1.96 * o['se']).mean(); d['hw_cs'] = o['hw'].mean()
    d['Dxx'] = o['Dxx'].mean(); d['cov_unif_csb'] = o['cover_csb'].mean(); d['hw_csb'] = np.median(o['hwb'])
    return d

# ============================ Hill closed loop ============================
def H(x, K, s): return x**s / (x**s + K**s)
def Hp(x, K, s): return s * x**(s-1) * K**s / (x**s + K**s)**2
def xstar(b, K, s):
    # solve b*Hp(x)=1 numerically (vectorised) by bisection on log x
    lo = np.full_like(b, -3.0); hi = np.full_like(b, 6.0)
    for _ in range(50):
        mid = 0.5*(lo+hi); x = np.exp(mid)
        f = b*Hp(x, K, s) - 1.0    # decreasing in x beyond the inflection (s=1: everywhere)
        lo = np.where(f > 0, mid, lo); hi = np.where(f > 0, hi, mid)
    return np.exp(0.5*(lo+hi))

def run_hill(T, R, s=1.0, a=20.0, b=40.0, K=5.0, sig=2.0, T0=8, burn_amp=0.3, refresh=4, rho=1.0,
             xi_sd=0.0, seed=0, delta=0.05, Kgrid=None, prior=None, x_floor=0.5, x_cap=60.0, gain_mult=1.0, xi_decay=0.0):
    rng = np.random.default_rng(seed)
    if Kgrid is None: Kgrid = np.exp(np.linspace(np.log(0.5), np.log(60.0), 80))
    nK = len(Kgrid)
    if prior is None: prior = dict(mu0=np.array([a, b]), lam0=np.diag([1e-2, 1e-3]), a0=2.0, b0=sig**2)
    xopt = float(xstar(np.array([b]), np.array([K]), s)[0])
    # running sums per K: X_K = [1, H(x;K)]
    S1 = np.zeros(R); Sy = np.zeros(R); Syy = np.zeros(R)
    Sh = np.zeros((R, nK)); Shh = np.zeros((R, nK)); Shy = np.zeros((R, nK))
    xs = np.zeros((T, R)); x_prev = np.full(R, xopt); cumloss = np.zeros(R); popt = b*H(xopt,K,s)-xopt
    bh_used = np.full(R, b); Kh_used = np.full(R, K)
    cov_naive = np.ones(R, bool); cov_cs = np.ones(R, bool)
    out_hist = []
    lam0 = prior['lam0']; mu0 = prior['mu0']; a0 = prior['a0']; b0 = prior['b0']
    logdet0 = np.linalg.slogdet(lam0)[1]
    m_true = None
    for t in range(1, T+1):
        if t <= T0:
            target = np.full(R, xopt * (1 + burn_amp * (-1)**t))
        else:
            target = xopt + gain_mult * (xstar(bh_used, Kh_used, s) - xopt)
        xsd = xi_sd * (t ** (-xi_decay)) if xi_decay > 0 else xi_sd
        xi = rng.normal(0, xsd, R) if xsd > 0 else 0.0
        x = np.clip(rho*target + (1-rho)*x_prev + xi, x_floor, x_cap)
        y = a + b*H(x, K, s) + sig*rng.normal(0, 1, R)
        xs[t-1] = x; x_prev = x
        if t > T0: cumloss += popt - (b*H(x,K,s)-x)
        h = H(x[:, None], Kgrid[None, :], s)
        S1 += 1; Sy += y; Syy += y*y; Sh += h; Shh += h*h; Shy += h*y[:, None]
        n = t
        if n >= 4:
            # OLS per K
            hbar = Sh/n; ybar = (Sy/n)[:, None]
            Dhh = Shh - n*hbar**2; Dhy = Shy - n*hbar*ybar
            bK = Dhy/np.maximum(Dhh, 1e-12); aK = ybar - bK*hbar
            rssK = np.maximum(Syy[:, None] - n*ybar**2 - bK*Dhy, 1e-12)
            iK = np.argmin(rssK, axis=1); rr = np.arange(R)
            bh = bK[rr, iK]; Kh = Kgrid[iK]; ah = aK[rr, iK]; rss = rssK[rr, iK]
            # NIG marginal per K, then mixture over K (uniform on grid)
            XtX = np.stack([np.stack([np.broadcast_to(S1[:, None], Sh.shape), Sh], -1), np.stack([Sh, Shh], -1)], -2)
            Xty = np.stack([np.broadcast_to(Sy[:, None], Sh.shape), Shy], -1)
            Ln = lam0[None, None] + XtX
            rhs = (lam0 @ mu0)[None, None] + Xty
            mun = np.linalg.solve(Ln, rhs[..., None])[..., 0]
            an = a0 + n/2
            bn = b0 + 0.5*(Syy[:, None] + mu0 @ lam0 @ mu0 - np.einsum('rki,rkij,rkj->rk', mun, Ln, mun))
            logdetn = np.linalg.slogdet(Ln)[1]
            logmK = (special.gammaln(an) - special.gammaln(a0) + a0*np.log(b0) - an*np.log(bn)
                     + 0.5*(logdet0 - logdetn) - (n/2)*np.log(2*np.pi))
            logm = special.logsumexp(logmK, axis=1) - np.log(nK)
            Rstar = (n/(2*np.pi)) * np.exp(-2*(logm + np.log(delta))/n - 1)
            # projections of {RSS<=R} onto mROAS at current x (x of this period)
            def proj(Rthr):
                Hp_now = Hp(x[:, None], Kgrid[None, :], s)
                slack = Rthr[:, None] - rssK
                ok = slack >= 0
                inv_bb = S1[:, None]/np.maximum(S1[:, None]*Shh - Sh**2, 1e-12)
                hw = np.sqrt(np.maximum(slack, 0)*inv_bb)
                lo = np.where(ok, (bK - hw)*Hp_now, np.inf).min(1); hi = np.where(ok, (bK + hw)*Hp_now, -np.inf).max(1)
                return lo, hi
            qF = stats.f.ppf(0.95, 1, max(n-3, 1))
            Rn = rss*(1 + qF/max(n-3, 1))
            lo_n, hi_n = proj(Rn); lo_c, hi_c = proj(Rstar)
            m_true = b*Hp(x, K, s)
            miss_n = (m_true < lo_n) | (m_true > hi_n); miss_c = (m_true < lo_c) | (m_true > hi_c)
            cov_naive &= ~miss_n; cov_cs &= ~miss_c
            if t <= T0 or (t - T0) % refresh == 0:
                bh_used = bh.copy(); Kh_used = Kh.copy()
            if t == T:
                m_hat = bh*Hp(x, Kh, s)
                out_hist = dict(bh=bh, Kh=Kh, ah=ah, m_hat=m_hat, m_true=m_true, lo_n=lo_n, hi_n=hi_n, lo_c=lo_c, hi_c=hi_c,
                                miss_n=miss_n, miss_c=miss_c, Dhh=Dhh[rr, iK])
    out_hist.update(cov_naive=cov_naive, cov_cs=cov_cs, xs=xs, xopt=xopt, cumloss=cumloss)
    return out_hist

def summ(o):
    d = {}
    d['mean_mtrue'] = o['m_true'].mean(); d['bias_m'] = (o['m_hat']-o['m_true']).mean(); d['sd_m'] = (o['m_hat']-o['m_true']).std()
    d['cov_fin_naive'] = 1-o['miss_n'].mean(); d['cov_fin_cs'] = 1-o['miss_c'].mean()
    d['cov_unif_naive'] = o['cov_naive'].mean(); d['cov_unif_cs'] = o['cov_cs'].mean()
    d['hw_naive'] = ((o['hi_n']-o['lo_n'])/2).mean(); d['hw_cs'] = np.median((o['hi_c']-o['lo_c'])/2)
    d['sd_xT'] = o['xs'][-1].std(); d['mean_xT'] = o['xs'][-1].mean(); d['xopt'] = o['xopt']
    d['loss/period'] = np.mean((o['xs'][-1]-o['xopt'])**2)  # placeholder scaled later
    return d

# ============================ first-order theory check (E2) ============================
def theory_check():
    # First-order theory check with common random numbers. T=40, T0=8, x0=5, exploration xi_sd=1, skew kappa3.
    T, T0, R = 40, 8, 40000
    def stats_at(g, skew, seed=77):
        o = run_loop(T=T, R=R, g=g, xi_sd=1.0, seed=seed, skew=skew, T0=T0)
        return o['bh'].mean()-2.0, o['bh'].var(), o
    for skew in [0.0, 2.0]:
        print(f'--- skew={skew} (kappa3={skew})')
        b0, v0, o0 = stats_at(0.0, skew)
        for g in [0.5, 1.0, 2.0, 4.0]:
            bg, vg, og = stats_at(g, skew)
            print(f'g={g}: dbias={bg-b0:+.5f} (se~{np.sqrt(v0/R):.5f})  dVar/g={(vg-v0)/g:+.5f}  dVar/g^2={(vg-v0)/g**2:+.5f}')
        # theory for dVar/dg at g=0 from the exploration design (xi paths at g=0)
        xs = o0['xs']  # T x R
        xi = xs - 5.0
        tot = 0.0
        for r in range(2000):
            z = xi[:, r]; zb = z - z.mean(); D0 = (zb**2).sum()
            # w[t,s] for t<s, s>T0: (z_t - mean(z[:s-1]))/S_{s-1}
            W = np.zeros((T, T))
            for s in range(T0, T):  # s index 0-based: period s+1; uses obs 0..s-1
                m = z[:s].mean(); S = ((z[:s]-m)**2).sum()
                W[:s, s] = (z[:s]-m)/S
            k3 = skew
            EN0N1 = -(k3/T) * sum(zb[s]*W[s, s+1:].sum() for s in range(T))
            EN0sqD1 = 2*k3 * sum(zb[t]**2 * (zb[t+1:]*W[t, t+1:]).sum() for t in range(T))
            tot += 2*(EN0N1/D0**2 - EN0sqD1/D0**3)
        print(f'theory dVar/dg at g=0 (avg over designs): {tot/2000:+.5f}')

# ============================ experiment runners ============================
def E1():
    print('E1 smooth linear feedback with exploration (xi=1): bias/coverage vs gain g, T=104')
    for g in [0,1,3,10,30,100]:
        d=summarize(run_loop(T=104,R=8000,g=g,seed=1)); print(f"  g={g:3d} bias={d['bias']:+.4f}±{d['bias_se']:.4f} sd={d['sd']:.4f} mean_se={d['mean_se']:.4f} cov_fixedT={d['cov_final_naive']:.3f} cov_unif_naive={d['cov_unif_naive']:.3f} cov_unif_CS={d['cov_unif_cs']:.4f} Dxx={d['Dxx']:.0f}")
def E2():
    theory_check()
    return

    print('E2 first-order theory check with common random numbers (T=40,T0=8,xi=1,x0=5,sigma=1)')
    import subprocess; print(open('e_theory_out.txt').read() if False else '  (see e_theory.py; run separately)')
def E3():
    print('E3 skew sign x gain (xi=1, T=104): O(g) variance term ∝ kappa3*g flips the naive coverage direction')
    for sgn in [1,-1]:
        for g in [0,10,30]:
            d=summarize(run_loop(T=104,R=8000,g=g,skew=2.0,skew_sign=sgn,seed=42)); print(f"  skew_sign={sgn:+d} g={g:3d} cov_fixedT={d['cov_final_naive']:.3f} sd/mean_se={d['sd']/d['mean_se']:.3f} bias={d['bias']:+.4f}")
def E4():
    print('E4 threshold policies, x in {2.5,7.5}, xi=1: hold branch drifting (RW: last realised spend incl. noise) vs fixed base level')
    for T in [104,208,416]:
        for lab,kw in [('pure random walk',dict(policy='rw')),('citrig RW-hold',dict(policy='citrig')),('citrig base-hold',dict(policy='citrig',hold_noise=False)),('deadband.1 RW-hold',dict(policy='deadband',band=0.1)),('deadband.1 base-hold',dict(policy='deadband_base',band=0.1))]:
            d=summarize(run_loop(T=T,R=8000,g=0,seed=43,**kw)); print(f"  T={T} {lab:20s} cov_fixedT={d['cov_final_naive']:.3f} sd/mean_se={d['sd']/d['mean_se']:.3f} bias={d['bias']:+.4f} cov_unif_naive={d['cov_unif_naive']:.3f} cov_unif_CS-B={d.get('cov_unif_csb',float('nan')):.4f}")
def E5():
    print('E5 stall: self-driven loop (xi=0) Dxx plateau vs T (g=10) and manifold identity')
    for T in [30,104,416,832]:
        o=run_loop(T=T,R=2000,g=10,xi_sd=0.0,seed=7); print(f"  T={T:4d} medDxx={np.median(o['Dxx']):.1f} sd_b={o['bh'].std():.4f} mean_se={o['se'].mean():.3f} cov_fixedT={1-o['miss_final'].mean():.3f} hw_naive={1.96*o['se'].mean():.3f} hw_CS={o['hw'].mean():.3f}")
    o=run_loop(T=832,R=4000,g=10,xi_sd=0.0,seed=21); xinf=o['xs'][-1]
    print(f"  manifold |ahat-a+(bhat-b)x_inf| = {np.abs((o['ah']-10)+(o['bh']-2)*xinf).mean():.4f} vs |bhat-b|*x0 = {(np.abs(o['bh']-2)*5).mean():.4f}; sd(x_inf)={xinf.std():.4f} = g*sd(bhat)={10*o['bh'].std():.4f}")
    print('  stall vs gain (T=416):')
    for g in [1,3,10,30,100]:
        o=run_loop(T=416,R=4000,g=g,xi_sd=0.0,seed=22); print(f"   g={g:4d} sd_b={o['bh'].std():.4f} sd_x={o['xs'][-1].std():.3f} medDxx={np.median(o['Dxx']):.1f} cov_fixedT={1-o['miss_final'].mean():.3f} loss/period={g*o['bh'].var()/2:.4f}")
def E6():
    print('E6 decaying exploration xi_sd = t^-d restores learning iff d<=1/4 (Keskin-Zeevi dispersion), g=10')
    for d in [0.0,0.25,0.5,1.0]:
        for T in [104,832]:
            o=run_loop(T=T,R=2000,g=10,xi_sd=1.0,xi_decay=d,seed=23); print(f"  decay={d} T={T}: sd_b={o['bh'].std():.4f} medDxx={np.median(o['Dxx']):.1f} hw_naive={1.96*o['se'].mean():.3f} hw_CS={o['hw'].mean():.3f}")
def E7():
    print('E7 peeking vs adaptivity decomposition (weekly looks): fixed design | smooth g=10 | CI-triggered')
    for T in [26,52,104,208,416]:
        row=[]
        for pol,g in [('linear',0),('linear',10),('citrig',0)]:
            o=run_loop(T=T,R=4000,g=g,policy=pol,seed=51); row.append((1-o['miss_final'].mean(),o['cover_naive'].mean(),o['cover_csb'].mean()))
        print(f"  T={T:3d} "+' | '.join(f'fixedT={a:.3f} unif={b:.3f} CS-B unif={c:.4f}' for a,b,c in row))
def E8():
    print('E8 CS width: vague-prior CS, burn-in-prior CS-B, naive, Bonferroni-over-looks; g=3 xi=1')
    for T in [26,52,104,208,416,832]:
        o=run_loop(T=T,R=3000,g=3,seed=61); rho2=(o['hw']/o['se'])**2
        print(f"  T={T:3d} hw_naive={1.96*o['se'].mean():.3f} hw_CS={o['hw'].mean():.3f} (rho^2={rho2.mean():.1f}) hw_CS-B={o['hwb'].mean():.3f} ratioB={o['hwb'].mean()/(1.96*o['se']).mean():.2f} bonf_ratio={stats.norm.ppf(1-0.025/(T-2))/1.96:.2f} covB_unif={o['cover_csb'].mean():.4f}")
def E9():
    print('E9 CS-B validity under adaptive policies and misspecification (uniform coverage), T=104')
    for lab,kw in [('linear g=10 xi=1',dict(g=10)),('linear g=100 xi=.3',dict(g=100,xi_sd=0.3)),('self g=10 xi=0',dict(g=10,xi_sd=0.0)),('citrig',dict(policy='citrig')),('deadband .1',dict(policy='deadband',band=0.1)),
                   ('ar1=0.3',dict(g=10,ar1=0.3)),('ar1=0.6',dict(g=10,ar1=0.6)),('hetero=1',dict(g=10,hetero=1.0)),('skew=2',dict(g=10,skew=2.0))]:
        o=run_loop(T=104,R=4000,seed=62,**{**dict(g=0),**kw}); print(f"  {lab:20s} CS-B unif={o['cover_csb'].mean():.4f} CS unif={o['cover_cs'].mean():.4f} naive fixedT={1-o['miss_final'].mean():.3f} naive unif={o['cover_naive'].mean():.3f}")
def E10():
    print('E10 power control: first exclusion of b+delta (g=10, xi=1, T=104); CS-B vs naive')
    o=run_loop(T=104,R=4000,g=10,seed=54,record=['bh','hw','hwb','bhp','se'])
    bh=np.array(o['hist']['bh']); se=np.array(o['hist']['se']); hwb=np.array(o['hist']['hwb']); bhp=np.array(o['hist']['bhp'])
    for dlt in [0.0,0.2,0.4,0.8]:
        ec=(np.abs(bhp-(2+dlt))>hwb); en=(np.abs(bh-(2+dlt))>1.96*se)
        print(f"  delta={dlt}: CS-B ever-excludes={ec.any(0).mean():.3f} median week={np.median(np.where(ec.any(0),ec.argmax(0)+3,999)):.0f} | naive ever-excludes={en.any(0).mean():.3f} median week={np.median(np.where(en.any(0),en.argmax(0)+3,999)):.0f}")
def E11():
    print('E11 pricing law: exogenous exploration cost to reach naive half-width w=0.15 by T; theory rho^2 s^2/(2 g w^2)')
    w=0.15
    for g in [1,10]:
        print(f'  g={g} theory={3.84/(2*g*w**2):.1f}')
        for T in [52,104,208,416]:
            for v in np.exp(np.linspace(np.log(0.003),np.log(4),40)):
                o=run_loop(T=T,R=800,g=g,xi_sd=np.sqrt(v),seed=71+g)
                if np.median(1.96*o['se'])<=w: break
            xs=o['xs'][8:]; print(f"   T={T:3d}: v*={v:.3f} exo cost={(T-8)*v/(2*g):.1f} total deviation cost={np.median(((xs-5.0)**2).sum(0))/(2*g):.1f} medDxx={np.median(o['Dxx']):.0f}")
def E12():
    print('E12 Hill loop (b=40,K=5,s=1,sig=2, refresh 4): stall of mROAS learning; policy echo; exploration schedules')
    for T in [52,104,416,832]:
        o=run_hill(T=T,R=1000,seed=2); d=summ(o); print(f"  no-explore T={T:3d}: sd(m_true-m_hat)={d['sd_m']:.3f} medDhh={np.median(o['Dhh']):.3f} cov_fixedT={d['cov_fin_naive']:.3f} hw_naive={d['hw_naive']:.3f} hw_CS={d['hw_cs']:.3f} loss/period={o['cumloss'].mean()/(T-8):.3f}")
    o=run_hill(T=416,R=1000,seed=81); print(f"  policy echo T=416: m_hat at x_T mean={o['m_hat'].mean():.4f} sd={o['m_hat'].std():.4f}; true mROAS at x_T sd={o['m_true'].std():.3f}, 5-95%={np.quantile(o['m_true'],[.05,.95])}; bhat 5-95%={np.quantile(o['bh'],[.05,.95])}, Khat 5-95%={np.quantile(o['Kh'],[.05,.95])}")
    for lab,kw in [('none',{}),('const1.0',dict(xi_sd=1.0)),('const1.5',dict(xi_sd=1.5)),('decay c=2 t^-1/4',dict(xi_sd=2.0,xi_decay=0.25)),('decay c=2 t^-1/2',dict(xi_sd=2.0,xi_decay=0.5))]:
        r=[]
        for T in [104,832]:
            o=run_hill(T=T,R=1000,seed=5,**kw); d=summ(o); r.append((o['cumloss'].mean(),o['cumloss'].std()/np.sqrt(1000),d['sd_m'],d['hw_naive'],d['hw_cs'],d['cov_fin_naive'],d['cov_unif_cs']))
        print(f"  {lab:18s} T104: loss={r[0][0]:6.1f}±{r[0][1]:.1f} sd_m={r[0][2]:.3f} hw_n={r[0][3]:.3f} hw_CS={r[0][4]:.3f} | T832: loss={r[1][0]:6.1f}±{r[1][1]:.1f} sd_m={r[1][2]:.3f} hw_n={r[1][3]:.3f} hw_CS={r[1][4]:.3f} cov_fixedT={r[1][5]:.3f} CS_unif={r[1][6]:.3f}")
def E13():
    print('E13 Hill realism sweep (stall persists?): sd_m at T=104 vs 416')
    for lab,kw in [('base r4',dict()),('refresh 1',dict(refresh=1)),('refresh 13',dict(refresh=13)),('rho .5',dict(rho=0.5)),('sig 1',dict(sig=1.0)),('s=2',dict(s=2.0)),('gain x2',dict(gain_mult=2.0))]:
        r=[]
        for T in [104,416]:
            o=run_hill(T=T,R=800,seed=81,**kw); d=summ(o); r.append((d['sd_m'],np.median(o['Dhh']),d['cov_fin_naive']))
        print(f"  {lab:10s} T104: sd_m={r[0][0]:.3f} Dhh={r[0][1]:.3f} cov={r[0][2]:.3f} | T416: sd_m={r[1][0]:.3f} Dhh={r[1][1]:.3f} cov={r[1][2]:.3f}")

if __name__=='__main__':
    arg=sys.argv[1] if len(sys.argv)>1 else 'quick'
    allE={f'E{i}':globals()[f'E{i}'] for i in range(1,14)}
    if arg=='quick': [allE[k]() for k in ['E1','E3','E4','E5','E7','E9','E12']]
    elif arg=='all': [f() for f in allE.values()]
    else: allE[arg]()
