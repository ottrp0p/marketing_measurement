"""Dive 06 base: geo-level Hill MMM, aggregation, DP noise.
Micro model: y_gt = b_g + beta*H(a_gt;K,s) + eps,  a_gt = adstock_alpha(x_gt)
H(a) = a^s/(a^s+K^s). Stable shares: x_gt = w_g * X_t.
Naive national fit: Y_t = B + beta*H(A_t;K,s)+e on A_t=adstock(X_t).
Mixture (share-calibrated) fit: Y_t = B + beta*sum_g H(adstock(x_g);K,s)+e.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

def adstock(x, alpha):  # a_t = x_t + alpha a_{t-1}
    return lfilter([1.0],[1.0,-alpha], x, axis=-1)

def hill(a, K, s):
    a = np.maximum(a, 1e-12)
    return a**s/(a**s + K**s)

def hill_d1(a, K, s):
    a = np.maximum(a,1e-12)
    return s*K**s*a**(s-1)/(a**s+K**s)**2

TRUE = dict(alpha=0.5, K=1.0, s=2.0, beta=100.0, B=500.0)  # beta = incremental sales at full saturation per geo-sum unit

def make_shares(G, cv, rng):
    """Geo shares with mean 1/G and coefficient of variation cv (lognormal)."""
    if cv <= 0: return np.ones(G)/G
    sig2 = np.log(1+cv**2)
    w = rng.lognormal(-sig2/2, np.sqrt(sig2), G)
    return w/w.sum()  # renormalize (cv approx preserved for moderate cv)

def make_spend(T, rng, lo=8.0, hi=45.0):
    """National spend path with block pulsing + AR noise (identifying per dive 01)."""
    X = np.empty(T); level = (lo+hi)/2
    t = 0
    while t < T:
        L = rng.integers(4, 9)
        level = rng.uniform(lo, hi)
        X[t:t+L] = level
        t += L
    X *= np.exp(0.08*rng.standard_normal(T))
    return X

def simulate(G=20, T=104, cv=0.3, sigma_geo=None, rng=None, shares=None, X=None,
             Kg_cv=0.0, tv_shares=0.0):
    """Returns dict with geo spend x (G,T), geo outcome y (G,T), national X_t, Y_t.
    sigma_geo: noise sd per geo-week. Kg_cv: heterogeneity in K across geos.
    tv_shares: sd of AR(1) log-share wiggle over time (share instability)."""
    p = TRUE; rng = rng or np.random.default_rng(0)
    w = shares if shares is not None else make_shares(G, cv, rng)
    X = X if X is not None else make_spend(T, rng)
    if tv_shares > 0:
        e = np.zeros((G,T))
        for t in range(1,T): e[:,t] = 0.9*e[:,t-1] + tv_shares*rng.standard_normal(G)
        Wt = w[:,None]*np.exp(e); Wt /= Wt.sum(0, keepdims=True)
    else:
        Wt = np.repeat(w[:,None], T, axis=1)
    x = Wt * X[None,:]                      # geo spend
    a = adstock(x, p['alpha'])              # per-geo adstock
    Kg = p['K']*np.exp(Kg_cv*rng.standard_normal(G)-Kg_cv**2/2) if Kg_cv>0 else np.full(G,p['K'])
    mu_g = p['B']/G + p['beta']/G * G*hill(a, Kg[:,None], p['s'])/G  # per-geo mean; total beta split evenly
    # define: total incremental = beta * mean_g hill(a_g) * G / G ... keep total = beta*sum_g hill(a_g)/G? No:
    # Convention: Y_t = B + (beta/G)*sum_g G*? -- simplest: y_gt = B/G + beta_g*hill(a_gt), beta_g=beta/G? then
    # full-saturation national incremental = beta. Use that.
    mu_g = p['B']/G + (p['beta']/G)*hill(a, Kg[:,None], p['s'])
    if sigma_geo is None: sigma_geo = 0.05*p['beta']/G   # per-geo noise
    y = mu_g + sigma_geo*rng.standard_normal((G,T))
    return dict(x=x, y=y, X=X, Y=y.sum(0), A=adstock(X, p['alpha']), w=w, Wt=Wt,
                sigma_geo=sigma_geo, Kg=Kg)

def nll_national(theta, A_or_X, Y, mixture_x=None, alpha_free=True):
    """theta = [B, beta, log K, log s, logit-ish alpha]. If mixture_x given (G,T geo spend),
    model is B + (beta/G)*sum_g H(adstock(x_g)); else single Hill on adstock(X)."""
    B, beta, lK, ls, aa = theta
    alpha = 1/(1+np.exp(-aa)); K = np.exp(lK); s = np.exp(ls)
    if mixture_x is not None:
        a = adstock(mixture_x, alpha); G = mixture_x.shape[0]
        mu = B + (beta/G)*hill(a, K, s).sum(0)
    else:
        a = adstock(A_or_X, alpha)
        mu = B + beta*hill(a, K, s)
    r = Y - mu
    return 0.5*np.sum(r**2)   # LS == Gaussian MLE profile

def fit(X, Y, mixture_x=None, x0=None, restarts=8, rng=None):
    rng = rng or np.random.default_rng(1)
    best = None
    base = x0 if x0 is not None else np.array([Y.mean()*0.8, (Y.max()-Y.min())*2, np.log(0.02 if mixture_x is not None else 0.5), np.log(2.0), 0.0])
    # note K scale differs: per-geo adstock ~ X*w/(1-alpha) small; init smartly below
    if mixture_x is not None:
        a0 = adstock(mixture_x, 0.5); base[2] = np.log(np.median(a0))
    else:
        a0 = adstock(X, 0.5); base[2] = np.log(np.median(a0))
    for i in range(restarts):
        th0 = base + (0 if i==0 else rng.standard_normal(5)*np.array([0.1*abs(base[0]),0.3*abs(base[1]),0.5,0.3,0.5]))
        try:
            r = minimize(nll_national, th0, args=(X, Y, mixture_x), method='Nelder-Mead',
                         options=dict(maxiter=6000, xatol=1e-8, fatol=1e-10))
            r = minimize(nll_national, r.x, args=(X, Y, mixture_x), method='BFGS')
        except Exception: continue
        if best is None or r.fun < best.fun: best = r
    B, beta, lK, ls, aa = best.x
    return dict(B=B, beta=beta, K=np.exp(lK), s=np.exp(ls), alpha=1/(1+np.exp(-best.x[4])), fun=best.fun, x=best.x)

def mroas_national(fitres, Xstar, w=None, G=None, mixture=False, alpha=None):
    """Marginal ROAS d(total incr sales)/d(national spend) at steady-state spend Xstar.
    Steady state adstock multiplier 1/(1-alpha)."""
    al = alpha if alpha is not None else fitres['alpha']
    m = 1/(1-al)
    if mixture:
        a = w*Xstar*m   # per-geo steady adstock
        return (fitres['beta']/G)*np.sum(hill_d1(a, fitres['K'], fitres['s'])*w)*m
    else:
        return fitres['beta']*hill_d1(Xstar*m, fitres['K'], fitres['s'])*m

def mroas_true(Xstar, w, Kg=None):
    p = TRUE; m = 1/(1-p['alpha']); G = len(w)
    K = Kg if Kg is not None else np.full(G, p['K'])
    a = w*Xstar*m
    return (p['beta']/G)*np.sum(hill_d1(a, K, p['s'])*w)*m

# ============ penalized MAP fit (Meridian-style weak priors) ============
SIG2 = None  # fixed known noise variance (national); realistic: estimated, keep known for cleanliness
def make_post(X,Y,mixture_x,sig2):
    if mixture_x is not None:
        aref=adstock(mixture_x,0.5); lK0=np.log(np.median(aref))
    else:
        aref=adstock(X,0.5); lK0=np.log(np.median(aref))
    lb0=np.log(4*Y.std())
    def nlp(th):
        B,lbeta,lK,ls,aa=th
        beta=np.exp(lbeta); alpha=1/(1+np.exp(-aa)); K=np.exp(lK); s=np.exp(ls)
        if mixture_x is not None:
            a=adstock(mixture_x,alpha); G=mixture_x.shape[0]
            mu=B+(beta/G)*hill(a,K,s).sum(0)
        else:
            a=adstock(X,alpha); mu=B+beta*hill(a,K,s)
        nll=0.5*np.sum((Y-mu)**2)/sig2
        pen=0.5*((ls-np.log(1.5))/0.5)**2+0.5*((lK-lK0)/1.0)**2+0.5*((lbeta-lb0)/1.0)**2+0.5*aa**2
        return nll+pen
    return nlp,lK0,lb0
def fitmap(X,Y,mixture_x=None,sig2=None,restarts=6,rng=None):
    rng=rng or np.random.default_rng(7)
    nlp,lK0,lb0=make_post(X,Y,mixture_x,sig2)
    best=None
    th0=np.array([Y.mean()*0.85,lb0,lK0,np.log(1.5),0.0])
    for i in range(restarts):
        t=th0+(0 if i==0 else rng.standard_normal(5)*np.array([0.05*abs(th0[0]),0.5,0.7,0.3,0.5]))
        r=minimize(nlp,t,method='Nelder-Mead',options=dict(maxiter=8000,fatol=1e-9))
        r=minimize(nlp,r.x,method='BFGS')
        if best is None or r.fun<best.fun: best=r
    B,lbeta,lK,ls,aa=best.x
    # Laplace SE via numerical Hessian
    from scipy.optimize import approx_fprime
    n=5; H=np.zeros((n,n)); e=1e-4
    g0=approx_fprime(best.x,nlp,e)
    for j in range(n):
        dx=np.zeros(n); dx[j]=e
        H[:,j]=(approx_fprime(best.x+dx,nlp,e)-g0)/e
    H=(H+H.T)/2
    try: cov=np.linalg.inv(H); se=np.sqrt(np.maximum(np.diag(cov),0))
    except Exception: se=np.full(n,np.nan); cov=None
    return dict(B=B,beta=np.exp(lbeta),K=np.exp(lK),s=np.exp(ls),alpha=1/(1+np.exp(-aa)),
                x=best.x,se=se,cov=cov,fun=best.fun)

# ============ E7: privacy-granularity frontier ============
def fitmap_gen(Yobs, spend_x, mode, sig2cell, rng):
    G,T=spend_x.shape
    X=spend_x.sum(0)
    aref=adstock(spend_x,0.5); lK0=np.log(np.median(aref))
    lKn0=np.log(np.median(adstock(X,0.5)))
    Ynat=Yobs.sum(0) if mode=='geo' else Yobs
    lb0=np.log(4*Ynat.std())
    def nlp(th):
        B,lbeta,lK,ls,aa=th
        beta=np.exp(lbeta);alpha=1/(1+np.exp(-aa));K=np.exp(lK);s=np.exp(ls)
        if mode=='nat':
            mu=B+beta*hill(adstock(X,alpha),K,s); r=Yobs-mu
        elif mode=='mix':
            mu=B+(beta/G)*hill(adstock(spend_x,alpha),K,s).sum(0); r=Yobs-mu
        else:
            mu=B/G+(beta/G)*hill(adstock(spend_x,alpha),K,s); r=Yobs-mu
        nll=0.5*np.sum(r**2)/sig2cell
        lKc = lKn0 if mode=='nat' else lK0
        pen=0.5*((ls-np.log(1.5))/0.5)**2+0.5*((lK-lKc)/1.0)**2+0.5*((lbeta-lb0)/1.0)**2+0.5*aa**2
        return nll+pen
    best=None
    th0=np.array([Ynat.mean()*0.85, lb0, lKn0 if mode=='nat' else lK0, np.log(1.5),0.0])
    for i in range(4):
        t=th0+(0 if i==0 else rng.standard_normal(5)*np.array([0.05*abs(th0[0]),0.4,0.6,0.3,0.5]))
        r=minimize(nlp,t,method='Nelder-Mead',options=dict(maxiter=6000,fatol=1e-9))
        r=minimize(nlp,r.x,method='BFGS')
        if best is None or r.fun<best.fun: best=r
    B,lbeta,lK,ls,aa=best.x
    return dict(B=B,beta=np.exp(lbeta),K=np.exp(lK),s=np.exp(ls),alpha=1/(1+np.exp(-aa)))
def e7_frontier(b, SEEDS=6):
    out={'nat':[], 'mix':[], 'geo':[]}
    for seed in range(SEEDS):
        rng=np.random.default_rng(500+seed)
        d=simulate(G=20,T=104,cv=0.5,sigma_geo=1.0,rng=rng)
        Xop=d['X'].mean()
        ygeo=d['y']+(rng.laplace(0,b,(20,104)) if b>0 else 0)
        ynat=d['Y']+(rng.laplace(0,b,104) if b>0 else 0)
        for mode,Yo,s2 in [('nat',ynat,20+2*b**2),('mix',ynat,20+2*b**2),('geo',ygeo,1+2*b**2)]:
            f=fitmap_gen(Yo,d['x'],mode,s2,rng)
            mix = mode in ('mix','geo')
            m1=(mroas_national(f,Xop,w=d['w'],G=20,mixture=True) if mix else mroas_national(f,Xop))/mroas_true(Xop,d['w'])-1
            m2=(mroas_national(f,2*Xop,w=d['w'],G=20,mixture=True) if mix else mroas_national(f,2*Xop))/mroas_true(2*Xop,d['w'])-1
            out[mode].append([m1,m2,f['s']])
    for mode in out:
        a=np.array(out[mode])
        r1=np.sqrt(np.mean(a[:,0]**2)); r2=np.sqrt(np.mean(a[:,1]**2)); rs=np.sqrt(np.mean((a[:,2]-2)**2))
        print(f"b={b:5.1f} {mode:4s} RMSE@1x={r1*100:6.1f}% RMSE@2x={r2*100:6.1f}% RMSE(s)={rs:6.3f} mean_s={a[:,2].mean():5.2f}")

# ============ E5n: share-drift bias at realistic noise ============
def e5_drift_noise(tv=0.30, cv=0.5, seeds=24, seed0=900):
    res={'naive':[],'mix':[]}
    for seed in range(seeds):
        rng=np.random.default_rng(seed0+seed)
        d=simulate(G=20,T=104,cv=cv,sigma_geo=1.0,rng=rng,tv_shares=tv)
        Xop=d['X'].mean()
        fn=e3b.fitmap(d['X'],d['Y'],sig2=20.0); fm=e3b.fitmap(d['X'],d['Y'],mixture_x=d['x'],sig2=20.0)
        for k,f,mix in [('naive',fn,False),('mix',fm,True)]:
            m1=(mroas_national(f,Xop,w=d['w'],G=20,mixture=mix) if mix else mroas_national(f,Xop))/mroas_true(Xop,d['w'])-1
            res[k].append(m1)
    for k in res:
        a=np.array(res[k]); print(f"{k:5s} mROAS@1x bias={a.mean()*100:5.1f}% ± {a.std()/np.sqrt(seeds)*100:3.1f} RMSE={np.sqrt(np.mean(a**2))*100:5.1f}%")

# ============ E1b: closed-form drift laws ============
def e1b_drift_laws():
    G=4000
    for s in [1.5,2.0,3.0]:
        K=1.0
        for c in [0.1,0.2,0.3]:
            w=np.concatenate([np.full(G//2,1-c),np.full(G//2,1+c)])  # symmetric, exact cv=c
            us=np.geomspace(0.3,3.0,41)
            Hbar=np.array([np.mean(hill(w*u,K,s)) for u in us])
            lg=np.log(Hbar/(1-Hbar))
            seff=np.gradient(lg,np.log(us))
            pred=s-c**2*s**3*hill(us,K,s)*(1-hill(us,K,s))
            err=np.max(np.abs(seff-pred))
            # K drift: u* where lg=0
            i=np.argmin(np.abs(lg)); # interpolate
            u0=np.interp(0,lg,us)
            print(f"s={s} c={c}: max|s_eff - pred|={err:.4f} (drop scale {c**2*s**3/4:.3f});  log u* = {np.log(u0):.4f} vs pred c^2/2={c**2/2:.4f}")

# ============ E6: DP width law + LS efficiency ============
def fisher_num(sig,b,N=200001,L=None):
    L=L or (10*sig+14*b)
    z=np.linspace(-L,L,N); dz=z[1]-z[0]
    # convolve via FFT of densities
    g=np.exp(-0.5*(z/sig)**2)/(sig*np.sqrt(2*np.pi))
    l=np.exp(-np.abs(z)/b)/(2*b)
    from numpy.fft import rfft,irfft
    f=irfft(rfft(g)*rfft(l),N)*dz
    f=np.fft.fftshift(f); f=np.maximum(f,1e-300)
    lf=np.log(f); sc=np.gradient(lf,dz)
    mask=f>1e-14
    return np.trapz((sc[mask]**2)*f[mask], z[mask])
def e6_dp():
    print("(ii) ARE: efficiency of Gaussian-LS (var sig^2+2b^2) vs exact convolved MLE")
    for r in [0.25,0.5,1.0,2.0,4.0,8.0]:
        sig=1.0;b=r*sig
        Ie=fisher_num(sig,b); Ig=1/(sig**2+2*b**2)
        print(f"  b/sig={r:4.2f}: I_exact={Ie:.4f} 1/(s2+2b2)={Ig:.4f} LS efficiency={Ig/Ie*100:5.1f}%  (limit 50%)")
# (i) width law on the IDENTIFIED subproblem: (B,beta) linear given true (K,s,alpha)
    print("(i) width law, linear subfit sd(beta) across 40 seeds:")
    res={}
    signat=1.0*np.sqrt(20)
    for b in [0.0,signat,2*signat,4*signat]:
        est=[]
        for seed in range(40):
            rng=np.random.default_rng(300+seed)
            d=simulate(G=20,T=104,cv=0.3,sigma_geo=1.0,rng=rng)
            h=hill(adstock(d['X'],TRUE['alpha']),20.0,2.0)  # national-equiv K=20 at cv->0... use naive pseudo-true basis
            # use exact aggregate regressor to avoid basis misspec: mean_g hill(per-geo adstock)
            hb=hill(adstock(d['x'],TRUE['alpha']),TRUE['K'],TRUE['s']).sum(0)/20
            Z=np.c_[np.ones(104),hb]
            Ydp=d['Y']+(rng.laplace(0,b,104) if b>0 else 0)
            beta=np.linalg.lstsq(Z,Ydp,rcond=None)[0][1]
            est.append(beta)
        sd=np.std(est); res[b]=sd
        pred=np.sqrt((signat**2+2*b**2)/(signat**2+2*0**2))
        print(f"  b={b:6.2f}: sd(beta)={sd:6.2f} ratio={sd/res[0.0]:5.2f} predicted={pred:5.2f}")


# ============ dispatch ============
if __name__ == '__main__':
    import sys as _sys
    cmd = _sys.argv[1] if len(_sys.argv) > 1 else 'help'
    if cmd == 'drift_laws': e1b_drift_laws()
    elif cmd == 'drift_noise': e5_drift_noise()
    elif cmd == 'dp': e6_dp()
    elif cmd == 'frontier': e7_frontier(float(_sys.argv[2]) if len(_sys.argv) > 2 else 3.0)
    else: print('usage: 06_aggregation_privacy.py {drift_laws|drift_noise|dp|frontier [b]}')
