"""Dive 17 — The human-mechanism panel: design theory for a longitudinal single-source panel
(attention -> memory -> purchase). Simulator, estimators and all experiments E1-E15.
Usage: python 17_human_mechanism_panel.py quick   (~1 min)   |  all (~5 min)  |  E5 E6 ... (named)
Requires numpy + scipy."""
import sys, time, json
import numpy as np
from scipy import optimize, stats

def flight_schedule(T, on=6, off=7, start=4, mu=3.0):
    """weekly expected impressions per exposed panelist"""
    m = np.zeros(T)
    t = start
    while t < T:
        m[t:t+on] = mu; t += on+off
    return m

def simulate(N=2000, T=104, lam=0.90, kappa=0.08, sig_eta=0.15, sig_m=0.5,
             sig_e=0.5, b=1.0, d=0.0, f=0.5, alpha0=-2.0, sig_alpha=0.6,
             rho_alpha_m=0.5, mu=3.0, on=6, off=7, hold_frac=0.5, seed=0,
             att_mean=2.0, cond_delta=0.0, waves=None, lam_fn=None, bounded=False, rotate=False, psi=0.0, lam_mix=None):
    """Single-source panel simulator.
    A_it attentive seconds; M_it latent memory (trait m_i + AR(1) transient);
    S_it survey = M + e at wave weeks (optional conditioning: survey adds cond_delta to M);
    Y_it brand purchases: occasions ~ Poisson(f), choice prob expit(alpha_i + b*M + d*A)."""
    rng = np.random.default_rng(seed)
    Z = (rng.random(N) < 1-hold_frac).astype(float)      # 1 = exposed, 0 = holdout
    sched = flight_schedule(T, on, off, mu=mu)
    # individual traits: memory trait m_i and purchase propensity alpha_i, correlated
    z1 = rng.standard_normal(N); z2 = rng.standard_normal(N)
    m_i = sig_m*z1
    alpha_i = alpha0 + sig_alpha*(rho_alpha_m*z1 + np.sqrt(1-rho_alpha_m**2)*z2)
    att_prop = rng.gamma(2.0, att_mean/2.0, N)               # attention seconds/impression
    A = np.zeros((N,T)); M = np.zeros((N,T)); Y = np.zeros((N,T)); S = np.full((N,T), np.nan)
    x = rng.standard_normal(N)*sig_eta/np.sqrt(1-lam**2)      # transient, stationary start
    if waves is None: waves = []
    wset = set(waves)
    R = np.zeros((N,T), bool)                                   # surveyed indicator (rotating design)
    lam_i = np.full(N, lam) if lam_mix is None else rng.choice(lam_mix[0], N, p=lam_mix[1])
    Yprev = np.zeros(N)
    for t in range(T):
        imp = rng.poisson(sched[t]*Z)
        A[:,t] = rng.gamma(imp+1e-9, att_prop)*(imp>0)      # attentive seconds this week
        l = lam_i if lam_fn is None else lam_fn(x)
        if psi: x = x + psi*Yprev
        if bounded:
            # saturating encoding: increments shrink as memory approaches cap
            x = l*x + kappa*A[:,t]*np.exp(-np.maximum(x,0)/1.0) + rng.standard_normal(N)*sig_eta
        else:
            x = l*x + kappa*A[:,t] + rng.standard_normal(N)*sig_eta
        if t in wset:
            surv = (rng.random(N) < 0.5) if rotate else np.ones(N, bool)
            R[:,t] = surv
            S[surv,t] = (m_i + x + rng.standard_normal(N)*sig_e)[surv]
            x = x + cond_delta*surv                             # survey refreshes memory
        M[:,t] = m_i + x
        occ = rng.poisson(f, N)
        p = 1/(1+np.exp(-(alpha_i + b*M[:,t] + d*A[:,t])))
        Y[:,t] = rng.binomial(occ, p); Yprev = Y[:,t]
    return dict(Z=Z, A=A, M=M, S=S, Y=Y, R=R, waves=list(waves), sched=sched, m_i=m_i, alpha_i=alpha_i, att_mean=att_mean, lam_i=lam_i)

# ---------- estimators ----------
def simplex_fit(S, waves, groups=None):
    """Moment estimator: cross-sectional covariances between wave pairs
    c(k) = V_m + V_x * phi^k (k = lag in waves). Profile grid on phi; returns phi, V_m, V_x, reliability.
    Demeans by wave x group (group = randomized exposure arm) so arm-driven common shocks drop out."""
    W = len(waves); X = S[:, waves].copy()
    if groups is None: groups = np.zeros(X.shape[0])
    for g in np.unique(groups):
        idx = groups==g; X[idx] = X[idx] - X[idx].mean(0, keepdims=True)
    C = X.T @ X / X.shape[0]
    lags = {}
    for i in range(W):
        for j in range(i, W):
            lags.setdefault(j-i, []).append(C[i,j])
    ks = np.array(sorted(k for k in lags if k>=1)); ck = np.array([np.mean(lags[k]) for k in ks])
    c0 = np.mean(lags[0])
    best=None
    for phi in np.linspace(0.01, 0.999, 990):
        Xd = np.column_stack([np.ones_like(ks, float), phi**ks])
        beta, res, *_ = np.linalg.lstsq(Xd, ck, rcond=None)
        r = ck - Xd@beta; sse = r@r
        if best is None or sse < best[0]: best=(sse, phi, beta)
    sse, phi, (Vm, Vx) = best
    rel = (Vm+Vx)/c0
    return dict(phi=phi, Vm=Vm, Vx=Vx, rel=rel, c0=c0, ck=ck, ks=ks)

def simplex_fit_nofe(S, waves, groups=None):
    """3-wave version without trait: c(k)=Vx phi^k -> phi = c2/c1 (uses all pairs)."""
    W=len(waves); X=S[:,waves].copy()
    if groups is None: groups=np.zeros(X.shape[0])
    for g in np.unique(groups):
        idx=groups==g; X[idx]=X[idx]-X[idx].mean(0,keepdims=True)
    C=X.T@X/X.shape[0]
    c1=np.mean([C[i,i+1] for i in range(W-1)]); c2=np.mean([C[i,i+2] for i in range(W-2)])
    return c2/c1

def impulse_response(out, Z, sched, T, tmax=30):
    """Experimental impulse response: contrast exposed-holdout means of `out` per week, deconvolved
    against the schedule (least squares on kernel h[0..tmax])."""
    diff = out[Z==1].mean(0) - out[Z==0].mean(0)   # T
    # design: diff_t = sum_k h_k * sched[t-k]
    X = np.zeros((T, tmax+1))
    for k in range(tmax+1):
        X[k:,k] = sched[:T-k]
    ok = ~np.isnan(diff)
    h, *_ = np.linalg.lstsq(X[ok], diff[ok], rcond=None)
    return h, diff

def geo_fit(h, tmax=None):
    """fit c*a^k to kernel h by LS -> pseudo-true adstock"""
    k = np.arange(len(h)) if tmax is None else np.arange(tmax+1)
    def sse(a):
        g = a**k; c = (g@h[:len(k)])/(g@g); return ((h[:len(k)]-c*g)**2).sum()
    r = optimize.minimize_scalar(sse, bounds=(0.0,0.999), method='bounded')
    a=r.x; g=a**k; c=(g@h[:len(k)])/(g@g)
    return a, c

# ---------- base configuration (dive 17) ----------
BASE = dict(N=3000, T=104, lam=0.90, kappa=0.012, sig_eta=0.15, sig_m=1.0, sig_e=0.7,
            b=0.25, d=0.0, f=0.5, alpha0=-2.0, sig_alpha=1.0, rho_alpha_m=0.5, mu=3.0, on=6, off=7,
            hold_frac=0.5, att_mean=2.0)
WAVES8 = list(range(8, 104, 8))

def mem_kernel(kappa, lam, L=60):
    return kappa*lam**np.arange(L)

def ir_fit_S(sim, waves=None, L=60):
    """Experimental (κ, λ) from wave contrasts of S: NLS of contrast on convolution of expected attention path."""
    waves = sim['waves'] if waves is None else waves
    Sw = sim['S'][:, waves]; Z = sim['Z']
    n1 = (~np.isnan(Sw[Z==1])).sum(0); n0 = (~np.isnan(Sw[Z==0])).sum(0)
    diff = np.nanmean(Sw[Z==1],0) - np.nanmean(Sw[Z==0],0)
    se = np.sqrt(np.nanvar(Sw[Z==1],0)/n1 + np.nanvar(Sw[Z==0],0)/n0)
    EA = sim['sched']*sim.get('att_mean', 2.0)
    T = sim['S'].shape[1]
    def pred(p):
        conv = np.convolve(EA, mem_kernel(p[0], p[1], L))[:T]; return conv[waves]
    r = optimize.least_squares(lambda p: (pred(p)-diff)/se, [0.01, 0.8], bounds=([0,0],[1,0.999]))
    J = r.jac; cov = np.linalg.pinv(J.T@J)
    return dict(kappa=r.x[0], lam=r.x[1], se=np.sqrt(np.diag(cov)), diff=diff, diff_se=se)

def ir_fit_Y(sim, L=30):
    """Weekly purchase contrast (exposed - holdout) and LS deconvolution into a kernel h_Y[0..L]."""
    Y = sim['Y']; Z = sim['Z']; T = Y.shape[1]
    diff = Y[Z==1].mean(0) - Y[Z==0].mean(0)
    X = np.zeros((T, L+1)); EA = sim['sched']*sim.get('att_mean',2.0)
    for k in range(L+1): X[k:, k] = EA[:T-k]
    h, *_ = np.linalg.lstsq(X, diff, rcond=None)
    return dict(diff=diff, h=h, X=X)

def shape_transfer(sim, kap, lam, direct=True, cuped_weeks=0, ret_q=False):
    """Matched-filter (shape-transfer) estimator. Weekly purchase contrast diff_t = c*shape_t + d*EA_t,
    shape_t = expected exposure-driven memory path (conv of E[A] with the fitted kernel).
    Estimated by WLS on weekly contrasts; se computed EXACTLY from the person-level linear statistic
    (the estimator is a difference of arm means of q_i = sum_t w_t Y_it), optionally CUPED-adjusted with
    the person's pre-campaign purchase total (first `cuped_weeks` weeks, before any flight)."""
    Y = sim['Y']; Z = sim['Z']; T = Y.shape[1]; EA = sim['sched']*sim.get('att_mean',2.0)
    shape = np.convolve(EA, mem_kernel(kap, lam, 60))[:T]
    X = np.column_stack([shape] + ([EA] if direct else []))
    v = Y[Z==1].var(0)/(Z==1).sum() + Y[Z==0].var(0)/(Z==0).sum()
    Wt = 1/np.maximum(v, 1e-12)
    A = np.linalg.inv((X.T*Wt)@X) @ (X.T*Wt)            # p x T : beta = A @ diff
    Q = Y @ A.T                                          # N x p person-level statistics
    if cuped_weeks:
        pre = Y[:, :cuped_weeks].sum(1)
        for j in range(Q.shape[1]):
            th = np.cov(Q[:,j], pre)[0,1]/pre.var(); Q[:,j] = Q[:,j] - th*(pre-pre.mean())
    beta = Q[Z==1].mean(0) - Q[Z==0].mean(0)
    se = np.sqrt(Q[Z==1].var(0)/(Z==1).sum() + Q[Z==0].var(0)/(Z==0).sum())
    out = dict(beta=beta, se=se, shape=shape, X=X)
    if ret_q: out['Q']=Q
    return out

def total_effect(sim, cuped_weeks=0):
    Y=sim['Y']; Z=sim['Z']; q=Y[:, cuped_weeks:].sum(1)
    if cuped_weeks:
        pre=Y[:,:cuped_weeks].sum(1); th=np.cov(q,pre)[0,1]/pre.var(); q=q-th*(pre-pre.mean())
    d=q[Z==1].mean()-q[Z==0].mean(); se=np.sqrt(q[Z==1].var()/(Z==1).sum()+q[Z==0].var()/(Z==0).sum())
    return d, se

def survey_instrument(sim, lam_hat, kap_hat=None, window=None):
    """Rotating-survey instrument. For each wave w with next wave w+D:
    memory bump delta identified from S at w+D: E[S|R_w=1]-E[S|R_w=0] = delta*lam^D (randomized, so a plain contrast);
    purchase response: contrast of purchases in weeks w+1..w+D-1 (or window) between R_w=1 and R_w=0, within-person
    (person FE via demeaning over waves). beta_Y = purchase contrast / (delta * sum_{k=1}^{D-1} lam^k)."""
    S=sim['S']; Y=sim['Y']; R=sim['R']; waves=sim['waves']; N=S.shape[0]
    Wn=len(waves); D=waves[1]-waves[0]
    # --- delta ---
    num=0.; den=0.; dvar=0.
    d_list=[]; v_list=[]
    for j in range(Wn-1):
        w=waves[j]; wn=waves[j+1]
        s=S[:,wn]; ok=~np.isnan(s); r=R[:,w]
        a=s[ok&r]; c=s[ok&~r]
        d_list.append(a.mean()-c.mean()); v_list.append(a.var()/len(a)+c.var()/len(c))
    d_arr=np.array(d_list); v_arr=np.array(v_list); wts=1/v_arr
    dS=(wts*d_arr).sum()/wts.sum(); dS_se=np.sqrt(1/wts.sum())
    delta=dS/lam_hat**D; delta_se=dS_se/lam_hat**D
    # --- purchase response, within-person: Yw_ij = purchases in (w, w+D) ---
    Yw=np.zeros((N,Wn)); Rw=np.zeros((N,Wn))
    for j,w in enumerate(waves):
        hi=min(w+D, Y.shape[1]) if window is None else min(w+1+window, Y.shape[1])
        Yw[:,j]=Y[:,w+1:hi].sum(1); Rw[:,j]=R[:,w]
    # demean within person and within wave (two-way FE via double demeaning, balanced panel)
    Yd=Yw-Yw.mean(1,keepdims=True)-Yw.mean(0,keepdims=True)+Yw.mean()
    Rd=Rw-Rw.mean(1,keepdims=True)-Rw.mean(0,keepdims=True)+Rw.mean()
    g=(Rd*Yd).sum()/(Rd*Rd).sum()
    res=Yd-g*Rd
    # cluster-by-person se
    u=(Rd*res).sum(1); g_se=np.sqrt((u**2).sum())/(Rd*Rd).sum()
    kk=np.arange(1,D); gsum=(lam_hat**kk).sum()
    beta=g/(delta*gsum); beta_se=beta*np.sqrt((g_se/g)**2+(delta_se/delta)**2) if g!=0 else np.nan
    return dict(delta=delta, delta_se=delta_se, g=g, g_se=g_se, beta=beta, beta_se=beta_se)

OUT={}
def rep(fn, seeds):
    return np.array([fn(s) for s in seeds])

def E1(nrep=40):
    """SNR ladder at N=3000, 2 years: a-path (kappa, lam) from wave contrasts vs purchase total effect."""
    rows=[]
    for s in range(nrep):
        sim=simulate(waves=WAVES8, seed=s, **BASE)
        f=ir_fit_S(sim); te=total_effect(sim); te4=total_effect(sim,4)
        st=shape_transfer(sim,f['kappa'],f['lam'],direct=False)
        # per-wave memory contrast t
        tw=(f['diff']/f['diff_se'])
        rows.append([f['kappa'],f['se'][0],f['lam'],f['se'][1],te[0],te[1],te4[0],te4[1],st['beta'][0],st['se'][0],np.abs(tw).mean(),tw.max()])
    r=np.array(rows)
    o=dict(kappa=(r[:,0].mean(),r[:,0].std(),r[:,1].mean()), lam=(r[:,2].mean(),r[:,2].std(),r[:,3].mean()),
           total=(r[:,4].mean(),r[:,4].std(),r[:,5].mean(),(r[:,4]/r[:,5]).mean()), total_cuped4=(r[:,6].mean(),r[:,6].std(),r[:,7].mean(),(r[:,6]/r[:,7]).mean()),
           c_mf=(r[:,8].mean(),r[:,8].std(),r[:,9].mean(),(r[:,8]/r[:,9]).mean()), wave_t_mean=r[:,10].mean(), wave_t_max=r[:,11].mean(),
           lam_t=(r[:,2]/r[:,3]).mean(), kappa_t=(r[:,0]/r[:,1]).mean())
    print('E1', json.dumps(o, default=float, indent=0)); OUT['E1']=o

def E2(nrep=20):
    """Observational quasi-simplex (tracking-only) vs experimental lambda; heterogeneity masquerading as persistence."""
    rows=[]
    for s in range(nrep):
        sim=simulate(waves=WAVES8, seed=s, **BASE)
        Z=sim['Z']
        all_=simplex_fit(sim['S'],WAVES8)            # naive: pooled, wave-demeaned only
        arm=simplex_fit(sim['S'],WAVES8,Z)           # arm x wave demeaned
        exp_=simplex_fit(sim['S'][Z==1],WAVES8)      # exposed only
        hold=simplex_fit(sim['S'][Z==0],WAVES8)      # holdout only (no ad-driven heterogeneity)
        f=ir_fit_S(sim)
        rows.append([all_['phi']**(1/8),arm['phi']**(1/8),exp_['phi']**(1/8),hold['phi']**(1/8),f['lam'],hold['rel'],exp_['rel']])
        # same but with homogeneous attention (no persistent attention heterogeneity)
    r=np.array(rows)
    # homogeneous attention control
    rows2=[]
    for s in range(nrep):
        p=dict(BASE); 
        sim=simulate(waves=WAVES8, seed=s, **p)
        # replace attention heterogeneity: re-simulate with gamma shape large -> approx homogeneous; emulate by att prop fixed
        rows2.append(0)
    o=dict(naive=(r[:,0].mean(),r[:,0].std()), arm_demeaned=(r[:,1].mean(),r[:,1].std()), exposed_only=(r[:,2].mean(),r[:,2].std()),
           holdout_only=(r[:,3].mean(),r[:,3].std()), experimental=(r[:,4].mean(),r[:,4].std()), rel_hold=r[:,5].mean(), rel_exp=r[:,6].mean())
    print('E2', json.dumps(o, default=float)); OUT['E2']=o

def E3():
    """Wave-spacing design law: single burst (6 weeks), then W waves at spacing Delta; Fisher se(lam) vs Delta; Box-Lucas check."""
    lam=0.9; N=3000; sigS=1.30
    res={}
    for W in [2,3,4,6]:
        best=None; curve=[]
        for Delta in range(1,27):
            taus=np.array([k*Delta for k in range(W)])   # lags after burst end (first wave at burst end)
            D0=1.0
            X=np.column_stack([lam**taus, D0*taus*lam**(taus-1.0)])
            cov=np.linalg.inv(X.T@X)*(2*sigS**2/N)/D0**2*D0**2
            # scale: contrast D0*lam^tau with noise var 2 sigS^2/N per wave; se(lam)= sqrt(cov[1,1]) * (noise/D0)
            se_lam=np.sqrt(cov[1,1]); curve.append((Delta,se_lam))
            if best is None or se_lam<best[1]: best=(Delta,se_lam)
        res[W]=dict(best_Delta=best[0], best_se=best[1], curve=curve)
        print('E3 W',W,'best Delta',best[0],'se(lam) x D0/1 =',round(best[1],4), 'Box-Lucas tau*=1/(-ln lam)=',round(1/(-np.log(lam)),2),
              'se at Delta=8',round(dict(curve)[8],4),'at 2',round(dict(curve)[2],4),'at 20',round(dict(curve)[20],4))
    # MC check for W=3 at several Delta with the simulator (single burst schedule)
    mc={}
    for Delta in [3,6,10,16]:
        ests=[]
        for s in range(30):
            p=dict(BASE); p['on']=6; p['off']=200
            waves=[10+k*Delta for k in range(3)]
            sim=simulate(waves=waves, seed=s, **p); sim['sched']=flight_schedule(104,6,200,4,3.0)
            f=ir_fit_S(sim, waves=waves); ests.append(f['lam'])
        mc[Delta]=(np.mean(ests),np.std(ests))
        print('E3 MC W=3 Delta',Delta,'lam mean/sd',np.round(mc[Delta],4))
    OUT['E3']=dict(theory={k:(v['best_Delta'],v['best_se']) for k,v in res.items()}, mc=mc)

def E4(nrep=30):
    """Rotating-survey instrument: identifies conditioning delta and beta_Y independent of ads; compare to ads (matched filter) and to naive within-person OLS of Y on S."""
    for delta in [0.0, 0.3]:
        rows=[]
        for s in range(nrep):
            sim=simulate(waves=WAVES8, seed=s, rotate=True, cond_delta=delta, **BASE)
            f=ir_fit_S(sim); si=survey_instrument(sim,f['lam'])
            st=shape_transfer(sim,f['kappa'],f['lam'],direct=False,cuped_weeks=4)
            # naive within-person OLS: purchases in (w,w+8) on S_w, person+wave FE
            S=sim['S']; Y=sim['Y']; N=S.shape[0]; Wn=len(WAVES8)
            Sw=S[:,WAVES8]; Yw=np.column_stack([Y[:,w+1:w+8].sum(1) for w in WAVES8])
            ok=~np.isnan(Sw); 
            # two-way FE with missing: person-demean over observed, then wave-demean
            Sd=np.where(ok,Sw-np.nanmean(Sw,1,keepdims=True),0); Yd=np.where(ok,Yw-np.array([Yw[i,ok[i]].mean() for i in range(N)])[:,None],0)
            Sd=np.where(ok,Sd-np.nansum(Sd,0)/ok.sum(0),0); Yd=np.where(ok,Yd-np.nansum(Yd,0)/ok.sum(0),0)
            b_ols=(Sd*Yd).sum()/(Sd*Sd).sum()/7.0     # per-week purchase per memory unit
            rows.append([si['delta'],si['delta_se'],si['beta'],si['beta_se'],st['beta'][0],st['se'][0],b_ols,f['lam']])
        r=np.array(rows)
        # truth beta_Y
        sim=simulate(waves=WAVES8, seed=0, **BASE); p=1/(1+np.exp(-(sim['alpha_i'][:,None]+BASE['b']*sim['M']))); truth=(BASE['f']*BASE['b']*p*(1-p)).mean()
        o=dict(delta_true=delta, delta=(r[:,0].mean(),r[:,0].std(),r[:,1].mean()), beta_survey=(r[:,2].mean(),r[:,2].std(),r[:,3].mean(),(r[:,2]/r[:,3]).mean() if delta>0 else None),
               beta_ads=(r[:,4].mean(),r[:,4].std(),r[:,5].mean(),(r[:,4]/r[:,5]).mean()), beta_ols=(r[:,6].mean(),r[:,6].std()), truth=truth, lam=(r[:,7].mean(),r[:,7].std()))
        print('E4', json.dumps(o, default=float)); OUT[f'E4_{delta}']=o

def E3b(nrep=30):
    """Waves vs people at fixed budget: 2-year cost per person = c_fix + W*c_wave (rotating: half surveyed per wave)."""
    c_fix=300.0; c_wave=8.0; budget=3000*(c_fix+12*0.5*c_wave)
    o={}
    for W,Delta in [(4,24),(6,16),(12,8),(24,4)]:
        N=int(budget/(c_fix+W*0.5*c_wave)); waves=list(range(Delta, 104, Delta))[:W]
        rows=[]
        for s in range(nrep):
            p=dict(BASE); p['N']=N
            sim=simulate(waves=waves, seed=s, rotate=True, cond_delta=0.3, **p)
            f=ir_fit_S(sim); si=survey_instrument(sim,f['lam'])
            rows.append([f['lam'],f['kappa'],si['beta'],si['beta_se']])
        r=np.array(rows)
        o[W]=dict(N=N, lam_sd=r[:,0].std(), lam_mean=r[:,0].mean(), kappa_sd=r[:,1].std(), beta_survey_sd=r[:,2].std(), beta_survey_t=(r[:,2]/r[:,3]).mean())
        print('E3b W',W,'N',N, {k:round(v,4) for k,v in o[W].items()})
    OUT['E3b']=o

def E5(nrep=40):
    """Shape-transfer test: power to detect a direct (lag-0) purchase path vs pure memory mediation; type I at d=0.
    d chosen so the direct path carries a share of the total purchase lift."""
    o={}
    for N in [3000, 10000, 30000]:
        for dshare in [0.0, 0.3, 0.5]:
            # calibrate d: direct lift per attentive second on logit; total memory lift ~ b*E[M lift]. choose d so that
            # direct share of the total effect in logit units ~ dshare (memory logit lift ~ b*0.3 avg during flight; A mean ~6/wk in flight)
            d = 0.0 if dshare==0 else (dshare/(1-dshare))*BASE['b']*0.30/6.0
            tstats=[]; shares=[]
            for s in range(nrep):
                p=dict(BASE); p['N']=N; p['d']=d
                sim=simulate(waves=WAVES8, seed=s, **p)
                f=ir_fit_S(sim); st=shape_transfer(sim,f['kappa'],f['lam'],direct=True,cuped_weeks=4)
                tstats.append(st['beta'][1]/st['se'][1])
                tot=st['beta'][0]*st['shape'].sum()+st['beta'][1]*(sim['sched']*2).sum()
                shares.append(st['beta'][1]*(sim['sched']*2).sum()/tot if tot!=0 else np.nan)
            t=np.array(tstats); o[f'N{N}_share{dshare}']=dict(d=d, reject=float((np.abs(t)>1.96).mean()), t_mean=float(t.mean()), share_est=float(np.nanmedian(shares)))
            print('E5 N',N,'dshare',dshare,'d',round(d,5),'reject',o[f'N{N}_share{dshare}']['reject'],'t',round(t.mean(),2),'share_est',round(np.nanmedian(shares),2))
    OUT['E5']=o

def E6(nrep=40):
    """Two-instrument equality test (ads vs survey refresh) as a test of memory mediation.
    Alternatives: direct path (d>0), purchase->memory feedback (psi>0), and survey demand effect (survey affects purchase directly: emulate by d_survey)."""
    o={}
    for N in [3000, 10000]:
        for label,kw in [('pure',{}),('direct',dict(d=(0.5)*BASE['b']*0.30/6.0)),('feedback',dict(psi=0.3))]:
            rows=[]
            for s in range(nrep):
                p=dict(BASE); p['N']=N; p.update(kw)
                sim=simulate(waves=WAVES8, seed=s, rotate=True, cond_delta=0.3, **p)
                f=ir_fit_S(sim); si=survey_instrument(sim,f['lam']); st=shape_transfer(sim,f['kappa'],f['lam'],direct=False,cuped_weeks=4)
                diff=st['beta'][0]-si['beta']; se=np.sqrt(st['se'][0]**2+si['beta_se']**2)
                rows.append([st['beta'][0],si['beta'],diff/se,f['lam']])
            r=np.array(rows)
            o[f'N{N}_{label}']=dict(beta_ads=r[:,0].mean(), beta_survey=r[:,1].mean(), reject=float((np.abs(r[:,2])>1.96).mean()), z_mean=r[:,2].mean(), lam=r[:,3].mean())
            print('E6',N,label,{k:round(v,4) for k,v in o[f'N{N}_{label}'].items()})
    OUT['E6']=o

def E7():
    """Sales-fitted geometric adstock vs memory decay: pseudo-true alpha as a function of the direct-path share omega of the impulse response.
    Kernel h_k = omega*1[k=0] + (1-omega)*(1-lam)*lam^k (unit mass). LS fit of c*a^k."""
    o={}
    for lam in [0.7,0.8,0.9,0.95]:
        row=[]
        for omega in [0,0.1,0.2,0.3,0.5,0.7]:
            k=np.arange(120); h=(1-omega)*(1-lam)*lam**k; h[0]+=omega
            a,c=geo_fit(h); row.append((omega,a))
        o[lam]=row; print('E7 lam',lam,[(w,round(a,3)) for w,a in row], 'half-life true',round(np.log(.5)/np.log(lam),2), 'fitted at omega=.3',round(np.log(.5)/np.log(dict(row)[0.3]),2))
    # MMM-style check: fit geometric adstock to aggregate weekly sales from the simulator with d>0 (direct path), profile grid on alpha
    rows=[]
    for s in range(10):
        p=dict(BASE); p['N']=20000; p['d']=(0.5)*BASE['b']*0.30/6.0; p['hold_frac']=0.0
        sim=simulate(waves=[], seed=s, **p)
        y=sim['Y'].mean(0); x=sim['sched']*2.0   # aggregate sales, national spend proxy
        best=None
        for a in np.linspace(0.3,0.98,69):
            ad=np.zeros_like(x); 
            for t in range(len(x)): ad[t]=x[t]+(a*ad[t-1] if t>0 else 0)
            X=np.column_stack([np.ones_like(ad),ad]); bb,res,*_=np.linalg.lstsq(X,y,rcond=None); sse=((y-X@bb)**2).sum()
            if best is None or sse<best[0]: best=(sse,a)
        rows.append(best[1])
    print('E7 MMM adstock fit with 50% direct share: alpha mean/sd',np.mean(rows).round(3),np.std(rows).round(3),'vs lam 0.90')
    o['mmm_alpha']=(float(np.mean(rows)),float(np.std(rows)))
    OUT['E7']=o


def E7b():
    """MMM adstock fit control: d=0 (pure memory) with same flighted spend -> does the sales-only profile recover lam?"""
    o={}
    for label,d in [('pure',0.0),('direct50',(0.5)*BASE['b']*0.30/6.0)]:
        rows=[]
        for s in range(10):
            p=dict(BASE); p['N']=20000; p['d']=d; p['hold_frac']=0.0
            sim=simulate(waves=[], seed=s, **p); y=sim['Y'].mean(0); x=sim['sched']*2.0
            prof=[]
            for a in np.linspace(0.3,0.98,69):
                ad=np.zeros_like(x)
                for t in range(len(x)): ad[t]=x[t]+(a*ad[t-1] if t>0 else 0)
                X=np.column_stack([np.ones_like(ad),ad]); bb,*_=np.linalg.lstsq(X,y,rcond=None); prof.append(((y-X@bb)**2).sum())
            prof=np.array(prof); a_grid=np.linspace(0.3,0.98,69); rows.append(a_grid[prof.argmin()])
        o[label]=(float(np.mean(rows)),float(np.std(rows))); print('E7b',label,'alpha',np.round(o[label],3))
    OUT['E7b']=o

def E8(nrep=20):
    """Attack: observational within-person OLS of purchases on measured memory vs the randomized instruments,
    under (i) measurement error alone, (ii) purchase->memory feedback psi, (iii) both. Also reports within-person reliability."""
    o={}
    for label,kw in [('base',{}),('feedback',dict(psi=0.3)),('feedback_hi',dict(psi=0.6))]:
        rows=[]
        for s in range(nrep):
            p=dict(BASE); p.update(kw)
            sim=simulate(waves=WAVES8, seed=s, rotate=True, cond_delta=0.3, **p)
            f=ir_fit_S(sim); si=survey_instrument(sim,f['lam']); st=shape_transfer(sim,f['kappa'],f['lam'],direct=False,cuped_weeks=4)
            S=sim['S']; Y=sim['Y']; M=sim['M']; N=S.shape[0]
            Sw=S[:,WAVES8]; Mw=M[:,WAVES8]; Yw=np.column_stack([Y[:,w+1:w+8].sum(1) for w in WAVES8]); ok=~np.isnan(Sw)
            def twfe(Xw):
                Xd=np.where(ok,Xw-np.array([Xw[i,ok[i]].mean() for i in range(N)])[:,None],0)
                Xd=np.where(ok,Xd-np.nansum(Xd,0)/ok.sum(0),0); return Xd
            Sd=twfe(Sw); Md=twfe(np.where(ok,Mw,np.nan)); Yd=twfe(np.where(ok,Yw,np.nan))
            b_S=(Sd*Yd).sum()/(Sd*Sd).sum()/7; b_M=(Md*Yd).sum()/(Md*Md).sum()/7
            rel_within=(Md*Md).sum()/(Sd*Sd).sum()
            rows.append([b_S,b_M,rel_within,si['beta'],st['beta'][0]])
        r=np.array(rows)
        o[label]=dict(ols_S=r[:,0].mean(), ols_trueM=r[:,1].mean(), within_rel=r[:,2].mean(), survey_iv=r[:,3].mean(), ads_iv=r[:,4].mean())
        print('E8',label,{k:round(v,4) for k,v in o[label].items()},'truth~0.0152')
    OUT['E8']=o

def E9(nrep=20):
    """Attrition: per-wave dropout hazard depends on memory trait m_i (brand-engaged stay) or on alpha_i (buyers stay); bias in lam, kappa, beta_ads."""
    o={}
    for label,kind,strength in [('none','m',0.0),('on_memory','m',0.6),('on_purchase','a',0.6),('on_purchase_strong','a',1.2)]:
        rows=[]
        for s in range(nrep):
            sim=simulate(waves=WAVES8, seed=s, **BASE); rng=np.random.default_rng(1000+s)
            N=BASE['N']; trait=sim['m_i'] if kind=='m' else (sim['alpha_i']-BASE['alpha0'])
            haz=1/(1+np.exp(-(-2.2 - strength*trait)))    # ~10%/wave baseline, lower for high trait
            alive=np.ones(N,bool); S=sim['S'].copy(); Y=sim['Y'].astype(float).copy()
            for w in WAVES8:
                drop=alive & (rng.random(N)<haz); alive&=~drop
                S[~alive,w]=np.nan; Y[~alive,w:]=np.nan
            sim2=dict(sim); sim2['S']=S
            f=ir_fit_S(sim2)
            # total effect on completers only (naive) 
            Z=sim['Z']; q=np.nansum(Y,1); comp=alive
            d=q[(Z==1)&comp].mean()-q[(Z==0)&comp].mean(); 
            rows.append([f['lam'],f['kappa'],d,comp.mean()])
        r=np.array(rows); o[label]=dict(lam=r[:,0].mean(),kappa=r[:,1].mean(),total_completers=r[:,2].mean(),retention=r[:,3].mean())
        print('E9',label,{k:round(v,4) for k,v in o[label].items()})
    OUT['E9']=o

def E10(nrep=20):
    """Misspecification: (a) saturating encoding, (b) heterogeneous decay (fast/slow mixture), (c) attention expectation misestimated (EA x1.3), (d) survey error x2."""
    o={}
    cases=[('base',{},1.0,1.0),('bounded',dict(bounded=True),1.0,1.0),('lam_mix',dict(lam_mix=([0.7,0.97],[0.5,0.5])),1.0,1.0),
           ('EA_x1.3',{},1.3,1.0),('sig_e_x2',dict(sig_e=1.4),1.0,1.0),('sig_e_half',dict(sig_e=0.35),1.0,1.0)]
    for label,kw,ea_mult,_ in cases:
        rows=[]
        for s in range(nrep):
            p=dict(BASE); p.update(kw)
            sim=simulate(waves=WAVES8, seed=s, rotate=True, cond_delta=0.3, **p)
            sim['att_mean']=2.0*ea_mult
            f=ir_fit_S(sim); si=survey_instrument(sim,f['lam']); st=shape_transfer(sim,f['kappa'],f['lam'],direct=False,cuped_weeks=4)
            # implied memory lift at flight end: kappa*sum lam^k*EA
            rows.append([f['lam'],f['se'][1],f['kappa'],st['beta'][0],si['beta'],f['kappa']*6*ea_mult*((f['lam']**np.arange(6)).sum())])
        r=np.array(rows)
        # truth of memory lift at flight end from the simulator
        p=dict(BASE); p.update(kw); sim=simulate(waves=WAVES8, seed=0, **p); Z=sim['Z']; ml=(sim['M'][Z==1].mean(0)-sim['M'][Z==0].mean(0))[[9,22,35,48]].mean()
        o[label]=dict(lam=r[:,0].mean(),lam_sd=r[:,0].std(),kappa=r[:,2].mean(),beta_ads=r[:,3].mean(),beta_survey=r[:,4].mean(),lift_fit=r[:,5].mean(),lift_true=ml)
        print('E10',label,{k:round(v,4) for k,v in o[label].items()})
    OUT['E10']=o

def E11(nrep=12):
    """Power ladder: category purchase frequency f and panel size N -> t-stats for (lam, beta_ads, beta_survey, total effect) at 2 years."""
    o={}
    for f_ in [0.05, 0.2, 0.5, 1.5]:
        for N in [1000, 3000, 10000]:
            rows=[]
            for s in range(nrep):
                p=dict(BASE); p['N']=N; p['f']=f_
                sim=simulate(waves=WAVES8, seed=s, rotate=True, cond_delta=0.3, **p)
                fi=ir_fit_S(sim); si=survey_instrument(sim,fi['lam']); st=shape_transfer(sim,fi['kappa'],fi['lam'],direct=False,cuped_weeks=4); te=total_effect(sim,4)
                rows.append([fi['lam']/fi['se'][1], st['beta'][0]/st['se'][0], si['beta']/si['beta_se'], te[0]/te[1], sim['Y'].sum(1).mean()/2])
            r=np.array(rows); o[f'f{f_}_N{N}']=dict(t_lam=r[:,0].mean(), t_ads=r[:,1].mean(), t_survey=r[:,2].mean(), t_total=r[:,3].mean(), purch_per_yr=r[:,4].mean())
            print('E11 f',f_,'N',N,{k:round(v,2) for k,v in o[f'f{f_}_N{N}'].items()})
    OUT['E11']=o

def E12():
    """Cost model: per-person 2-year cost = recruit + device/attention + purchase capture + W/2 survey completes (rotating) ; N for target t-stats via sqrt scaling from E11."""
    prices=dict(recruit=40, attention_device_per_yr=120, purchase_capture_per_yr=60, survey_complete=8, incentive_per_yr=60)
    def cost(N, years=2, W=12, rotate=True):
        per=prices['recruit']+years*(prices['attention_device_per_yr']+prices['purchase_capture_per_yr']+prices['incentive_per_yr'])+W*(0.5 if rotate else 1)*prices['survey_complete']
        return N*per, per
    o={}
    for N in [1000,3000,10000,30000]:
        c,per=cost(N); o[N]=dict(total=c, per_person=per); print('E12 N',N,'per-person 2y $',per,'total $',f'{c:,.0f}')
    OUT['E12']=dict(prices=prices, costs=o)


def ir_fit_S2(sim, waves=None, L=80):
    """Round-3 refinement: two-component memory kernel k1*l1^j + k2*l2^j from the wave contrasts."""
    waves = sim['waves'] if waves is None else waves
    Sw = sim['S'][:, waves]; Z = sim['Z']
    n1 = (~np.isnan(Sw[Z==1])).sum(0); n0 = (~np.isnan(Sw[Z==0])).sum(0)
    diff = np.nanmean(Sw[Z==1],0) - np.nanmean(Sw[Z==0],0)
    se = np.sqrt(np.nanvar(Sw[Z==1],0)/n1 + np.nanvar(Sw[Z==0],0)/n0)
    EA = sim['sched']*sim.get('att_mean', 2.0); T=sim['S'].shape[1]
    def pred(p):
        ker=p[0]*p[1]**np.arange(L)+p[2]*p[3]**np.arange(L); return np.convolve(EA, ker)[:T][waves]
    best=None
    for init in [[0.01,0.6,0.005,0.95],[0.005,0.8,0.005,0.98],[0.01,0.5,0.002,0.9]]:
        r = optimize.least_squares(lambda p: (pred(p)-diff)/se, init, bounds=([0,0,0,0],[1,0.9,1,0.999]))
        if best is None or r.cost<best.cost: best=r
    return best.x

def shape_transfer_kernel(sim, ker, cuped_weeks=4):
    EA=sim['sched']*sim.get('att_mean',2.0); T=sim['Y'].shape[1]; shape=np.convolve(EA,ker)[:T]
    Y=sim['Y']; Z=sim['Z']; X=np.column_stack([shape,EA]); v=Y[Z==1].var(0)/(Z==1).sum()+Y[Z==0].var(0)/(Z==0).sum(); W=1/v
    A=np.linalg.inv((X.T*W)@X)@(X.T*W); Q=Y@A.T
    if cuped_weeks:
        pre=Y[:,:cuped_weeks].sum(1)
        for j in range(2): th=np.cov(Q[:,j],pre)[0,1]/pre.var(); Q[:,j]-=th*(pre-pre.mean())
    beta=Q[Z==1].mean(0)-Q[Z==0].mean(0); se=np.sqrt(Q[Z==1].var(0)/(Z==1).sum()+Q[Z==0].var(0)/(Z==0).sum())
    return beta, se, shape

def E5b(nrep=60):
    """Attack + refinement: shape misspecification (heterogeneous decay) -> spurious direct path under the one-exponential shape;
    the two-exponential kernel fitted from the memory contrasts removes it."""
    o={}
    for label,kw in [('base',{}),('lam_mix',dict(lam_mix=([0.7,0.97],[0.5,0.5])))]:
        for N in [3000,30000]:
            ts=[];ts2=[]
            for s in range(nrep):
                p=dict(BASE); p['N']=N; p.update(kw)
                sim=simulate(waves=WAVES8, seed=s, **p)
                f=ir_fit_S(sim); st=shape_transfer(sim,f['kappa'],f['lam'],direct=True,cuped_weeks=4); ts.append(st['beta'][1]/st['se'][1])
                x=ir_fit_S2(sim); ker=x[0]*x[1]**np.arange(80)+x[2]*x[3]**np.arange(80)
                beta,se,_=shape_transfer_kernel(sim,ker); ts2.append(beta[1]/se[1])
            ts=np.array(ts); ts2=np.array(ts2)
            o[f'{label}_N{N}']=dict(rej_1exp=float((np.abs(ts)>1.96).mean()), t_1exp=float(ts.mean()), rej_2exp=float((np.abs(ts2)>1.96).mean()), t_2exp=float(ts2.mean()))
            print('E5b',label,N,o[f'{label}_N{N}'])
    OUT['E5b']=o

def E13(nrep=16):
    """Mediated-share point estimate from the two instruments: share = beta_survey * sum(shape) / total purchase effect."""
    o={}
    d50=0.5*BASE['b']*0.30/6.0
    for label,kw in [('pure',{}),('direct20',dict(d=d50))]:
        for N in [10000,30000]:
            sh=[]; ds=[]
            for s in range(nrep):
                p=dict(BASE); p['N']=N; p.update(kw)
                sim=simulate(waves=WAVES8, seed=s, rotate=True, cond_delta=0.3, **p)
                f=ir_fit_S(sim); si=survey_instrument(sim,f['lam']); st=shape_transfer(sim,f['kappa'],f['lam'],direct=True,cuped_weeks=4)
                EA=sim['sched']*2.0; tot=st['beta'][0]*st['shape'].sum()+st['beta'][1]*EA.sum()
                sh.append(si['beta']*st['shape'].sum()/tot); ds.append(st['beta'][1]*EA.sum()/tot)
            o[f'{label}_N{N}']=dict(share_median=float(np.median(sh)), share_iqr=[float(x) for x in np.percentile(sh,[25,75])], direct_share_median=float(np.median(ds)))
            print('E13',label,N,o[f'{label}_N{N}'])
    OUT['E13']=o

def E14(nrep=40):
    """Pre-period covariate (CUPED) gain on the total purchase effect: flights start at week 30, outcome weeks 30-129."""
    import types
    global flight_schedule
    fs=flight_schedule
    res={0:[],4:[],13:[],26:[]}
    for s in range(nrep):
        p=dict(BASE); p['T']=130
        globals()['flight_schedule']=lambda T,on=6,off=7,start=30,mu=3.0: fs(T,on,off,30,mu)
        sim=simulate(waves=list(range(30,130,8)), seed=s, **p)
        globals()['flight_schedule']=fs
        Y=sim['Y']; Z=sim['Z']; q=Y[:,30:].sum(1)
        for pre in res:
            qq=q.copy()
            if pre:
                c=Y[:,:pre].sum(1); th=np.cov(qq,c)[0,1]/c.var(); qq=qq-th*(c-c.mean())
            d=qq[Z==1].mean()-qq[Z==0].mean(); se=np.sqrt(qq[Z==1].var()/(Z==1).sum()+qq[Z==0].var()/(Z==0).sum()); res[pre].append(d/se)
    o={pre:(float(np.mean(v)), float(np.mean(v)/np.mean(res[0]))) for pre,v in res.items()}
    print('E14 t by pre-period weeks (t, gain):',o); OUT['E14']=o

def E15():
    """Direct-share calibration: true direct share of the 2-year purchase lift for d = d50 (simulate with and without b)."""
    d=0.5*BASE['b']*0.30/6.0; tot=[];dir_=[]
    for s in range(6):
        p=dict(BASE); p['N']=30000; p['d']=d
        a=simulate(waves=WAVES8, seed=s, **p); p['d']=0.0; b_=simulate(waves=WAVES8, seed=s, **p)
        def lift(sim): Y=sim['Y'];Z=sim['Z']; return Y[Z==1].sum(1).mean()-Y[Z==0].sum(1).mean()
        tot.append(lift(a)); dir_.append(lift(a)-lift(b_))
    o=dict(total_lift=float(np.mean(tot)), direct_part=float(np.mean(dir_)), direct_share=float(np.mean(dir_)/np.mean(tot)))
    print('E15',o); OUT['E15']=o

QUICK=['E1','E2','E3','E7','E7b','E8','E9','E10','E12','E14','E15']
ALL=QUICK+['E3b','E4','E5','E5b','E6','E11','E13']

if __name__=='__main__':
    args=sys.argv[1:] or ['quick']
    names = QUICK if args==['quick'] else (ALL if args==['all'] else args)
    for e in names:
        t0=time.time(); globals()[e](); print(e,'done',round(time.time()-t0,1),'s', flush=True)
    json.dump(OUT, open('dive17_results.json','w'), default=float, indent=1)
