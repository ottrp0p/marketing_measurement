"""Deep dive 20 — The price of identification: the spectrum and the cost of MMM
experimental design.  (Backlog BL1 "seasonality-orthogonal perturbation design"
+ BL2 "the price of identification".)

Self-contained.  `python 20_price_of_identification.py` runs everything;
`python 20_price_of_identification.py e5 e6` runs selected sections.

Sections
  e1   second-order cost law                     e8   optimal identification energy
  e2   exchange-rate law (hyperbola)             e9   profit-optimal path is non-identifying
  e3   matched-cost frequency sweep + control    e10  robustness sweeps
  e4   fixed-reference VIF (Round-1, refuted)    e11  dive-01 block-length reconciliation
  e5   Fejer leakage margin rule                 e12  two channels
  e6   nuisance-basis sweep                      r3   the Psi ladder (+ Monte Carlo)
  e7   duration-amplitude                        r4   coloured residuals / operational rule
"""
import sys, json
import numpy as np
from scipy.optimize import brentq, least_squares

# ============================== MODEL CORE ==============================
TH0 = dict(beta=1.0, K=1.5, S=2.0, alpha=0.6)
M_MARGIN = 2.0
SIGMA = 0.05

def hill(a, K, S):
    a = np.maximum(a, 1e-12)
    return a**S/(a**S + K**S)

def dhill(a, K, S):
    a = np.maximum(a, 1e-12)
    return S*(K**S)*a**(S-1.0)/(a**S + K**S)**2

def d2hill(a, K, S, eps=1e-5):
    return (dhill(a+eps,K,S)-dhill(a-eps,K,S))/(2*eps)

def d3hill(a, K, S, eps=1e-3):
    return (dhill(a+eps,K,S)-2*dhill(a,K,S)+dhill(a-eps,K,S))/eps**2

def adstock(x, alpha, a0=None, burn=True):
    a = np.empty_like(x, dtype=float)
    prev = (x[0]/(1-alpha)) if a0 is None else a0
    for t in range(len(x)):
        prev = x[t] + alpha*prev
        a[t] = prev
    return a

def abar_opt(th=TH0, m=M_MARGIN):
    """steady-state profit-maximising adstock: m*beta*h'(a) = (1-alpha)"""
    from scipy.optimize import brentq
    K,S,al,b = th['K'],th['S'],th['alpha'],th['beta']
    infl = K*((S-1)/(S+1))**(1/S) if S>1 else 1e-6
    f = lambda a: m*b*dhill(a,K,S) - (1-al)
    hi = infl
    while f(hi) > 0 and hi < 1e4: hi *= 1.5
    return brentq(f, infl, hi)

def kappa(th=TH0, m=M_MARGIN):
    """curvature of weekly profit in adstock units: m*beta*|h''(abar*)|"""
    ab = abar_opt(th,m)
    return m*th['beta']*abs(d2hill(ab, th['K'], th['S']))

def profit_weekly(x, th=TH0, m=M_MARGIN, a0=None):
    a = adstock(x, th['alpha'], a0=a0)
    return m*th['beta']*hill(a, th['K'], th['S']) - x

def seasonal_basis(T, harmonics=3, trend=1):
    t = np.arange(T)
    cols = [np.ones(T)]
    for d in range(1, trend+1):
        cols.append(((t-t.mean())/T)**d)
    for k in range(1, harmonics+1):
        cols.append(np.cos(2*np.pi*k*t/52.0))
        cols.append(np.sin(2*np.pi*k*t/52.0))
    return np.column_stack(cols)

def spline_basis(T, n_knots=8, deg=3, trend=0):
    """natural-ish B-spline time basis (Meridian-style flexible baseline)"""
    from scipy.interpolate import BSpline
    t = np.linspace(0,1,T)
    knots = np.concatenate([np.zeros(deg), np.linspace(0,1,n_knots), np.ones(deg)])
    cols=[]
    n_basis = len(knots)-deg-1
    for i in range(n_basis):
        c = np.zeros(n_basis); c[i]=1
        cols.append(BSpline(knots, c, deg, extrapolate=False)(t))
    B = np.nan_to_num(np.column_stack(cols))
    return B

def resid_maker(Z):
    Q,_ = np.linalg.qr(Z)
    return lambda v: v - Q@(Q.T@v)

def fim(x, th=TH0, sigma=SIGMA, Z=None, params=('beta','K','S','alpha')):
    """Fisher information for selected structural params, nuisance Z profiled out."""
    T = len(x)
    if Z is None: Z = seasonal_basis(T)
    R = resid_maker(Z)
    cols=[]
    for p in params:
        h = 1e-5*max(1.0, abs(th[p]))
        tp = dict(th); tp[p]=th[p]+h
        tm = dict(th); tm[p]=th[p]-h
        mp = tp['beta']*hill(adstock(x,tp['alpha']), tp['K'], tp['S'])
        mm = tm['beta']*hill(adstock(x,tm['alpha']), tm['K'], tm['S'])
        cols.append(R((mp-mm)/(2*h)))
    J = np.column_stack(cols)
    return J.T@J/sigma**2

def cost_of_design(x, th=TH0, m=M_MARGIN, xbar=None):
    """exact expected weekly profit loss vs constant spend at xbar (mean-matched)."""
    if xbar is None: xbar = np.mean(x)
    pi_pert = profit_weekly(x, th, m).mean()
    pi_flat = profit_weekly(np.full(len(x), xbar), th, m).mean()
    return pi_flat - pi_pert

# ============================== HARNESS ==============================
th=TH0; m=M_MARGIN; sig=SIGMA
AB=abar_opt(); XB=AB*(1-th['alpha']); KAP=kappa()
BURN=52; TW=364; T=BURN+TW
FREQ_CLEAN=[4,7,13,14,28,91,182]      # periods dividing TW, not seasonal harmonics
FREQ_SEAS=[52.0,26.0,52/3]

def build(u_w):
    """u_w: perturbation on the analysis window (len TW). Burn-in is the PERIODIC
    extension of the design, so the window is in periodic steady state and
    mean(a)=abar exactly -- removing a spurious first-order edge term."""
    u_w = np.asarray(u_w, float); u_w = u_w - u_w.mean()
    pre = u_w[-BURN:] if BURN <= len(u_w) else np.tile(u_w, BURN//len(u_w)+1)[-BURN:]
    return XB + np.concatenate([pre, u_w])

def sine_design(period, amp):
    tw=np.arange(TW); return build(amp*np.sin(2*np.pi*tw/period))

def stats(x, Z=None, thv=th):
    """exact cost on window + FIM on window."""
    if Z is None: Z=seasonal_basis(TW, harmonics=3, trend=1)
    pi_p=profit_weekly(x, thv, m)[BURN:]
    pi_f=profit_weekly(np.full(len(x), XB), thv, m)[BURN:]
    C=float((pi_f-pi_p).sum())
    a=adstock(x, thv['alpha'])[BURN:]
    return C, float(np.var(a)), a

def fim_window(x, Z=None, thv=th, params=('beta','K','S','alpha')):
    if Z is None: Z=seasonal_basis(TW, harmonics=3, trend=1)
    R=resid_maker(Z); cols=[]
    for p in params:
        h=1e-5*max(1.0,abs(thv[p]))
        tp=dict(thv); tp[p]+=h; tm=dict(thv); tm[p]-=h
        mp=tp['beta']*hill(adstock(x,tp['alpha']),tp['K'],tp['S'])[BURN:]
        mm=tm['beta']*hill(adstock(x,tm['alpha']),tm['K'],tm['S'])[BURN:]
        cols.append(R((mp-mm)/(2*h)))
    J=np.column_stack(cols)
    return J.T@J/sig**2

def local_fim(x, Z=None, thv=th, free=('c1','c2','alpha')):
    """FIM in the LOCAL parametrisation: mu = c0 + c1*(a-abar) + 0.5*c2*(a-abar)^2, a=a(alpha).
    Uses the true nonlinear model's derivatives w.r.t. these local coordinates."""
    if Z is None: Z=seasonal_basis(TW, harmonics=3, trend=1)
    R=resid_maker(Z)
    al=thv['alpha']; a=adstock(x,al)[BURN:]; abar=a.mean()
    c1=thv['beta']*dhill(abar,thv['K'],thv['S'])
    d={'c1': a-abar, 'c2': 0.5*(a-abar)**2}
    # d mu / d alpha at fixed (c1,c2): via da/dalpha
    hh=1e-6
    ap=adstock(x,al+hh)[BURN:]; am=adstock(x,al-hh)[BURN:]
    dada=(ap-am)/(2*hh)
    c2=thv['beta']*d2hill(abar,thv['K'],thv['S'])
    d['alpha']=(c1+c2*(a-abar))*dada
    J=np.column_stack([R(d[k]) for k in free])
    return J.T@J/sig**2

def mroas(thd, xbar=XB):
    a=xbar/(1-thd['alpha'])
    return m*thd['beta']*dhill(a,thd['K'],thd['S'])/(1-thd['alpha'])

Zs = seasonal_basis(TW, harmonics=3, trend=1)
QZ, _ = np.linalg.qr(Zs)
GAMMA = np.array([10.0, 0.5, 0.8, 0.3, 0.25, -0.15, 0.1, 0.05])
AL = th['alpha']
C1 = th['beta']*dhill(AB, th['K'], th['S'])
C2 = th['beta']*d2hill(AB, th['K'], th['S'])
GLOC = np.array([m/(1-AL), 0.0, m*C2*(AB/(1-AL))/(1-AL) + m*C1/(1-AL)**2])   # corrected (Round 3)


def cost_exact(x, xb=None, thv=th):
    xb = XB if xb is None else xb
    return float((profit_weekly(np.full(len(x), xb), thv, m)
                  - profit_weekly(x, thv, m))[BURN:].sum())


def amp_for_cost(per, Ct):
    lo, hi = 1e-5, 0.88*XB
    for _ in range(60):
        mid = .5*(lo+hi)
        if cost_exact(sine_design(per, mid)) < Ct: lo = mid
        else: hi = mid
    return .5*(lo+hi)


def grad_struct(thd=th, xbar=None):
    xbar = XB if xbar is None else xbar
    g = []
    for p in ('beta', 'K', 'S', 'alpha'):
        h = 1e-6*max(1, abs(thd[p])); tp = dict(thd); tp[p] += h; tm = dict(thd); tm[p] -= h
        g.append((mroas(tp, xbar)-mroas(tm, xbar))/(2*h))
    return np.array(g)
GSTR = grad_struct()


def se_ladder(x):
    """se(mROAS) under three knowledge levels: calibrated / local-3 / structural-4."""
    a = adstock(x, AL)[BURN:]
    se1 = (m/(1-AL))*np.sqrt(sig**2/(TW*np.var(a)))
    V3 = np.linalg.inv(local_fim(x, Z=Zs, free=('c1', 'c2', 'alpha')))
    V4 = np.linalg.inv(fim_window(x, Z=Zs))
    return float(se1), float(np.sqrt(GLOC@V3@GLOC)), float(np.sqrt(GSTR@V4@GSTR))


# ---------------------------------------------------------------- E1
def e1():
    print("E1  second-order cost law:  weekly profit loss = 0.5*kappa*Var(a)")
    for per in (8, 13, 26, 52):
        for amp in (0.05, 0.20, 0.50):
            x = sine_design(per, amp*XB); a = adstock(x, AL)[BURN:]
            c = cost_exact(x)/TW
            print(f"  P={per:3d} A={amp:.2f}xbar  Var(a)={np.var(a):.5f}  cost/wk={c:.6f}"
                  f"  pred={0.5*KAP*np.var(a):.6f}  ratio={c/(0.5*KAP*np.var(a)):.4f}")


# ---------------------------------------------------------------- E2
def e2():
    print("E2  exchange-rate law:  C * Var(c1_hat | alpha known) = kappa*sigma^2/2 = %.4e" % (KAP*sig**2/2))
    th_c = KAP*sig**2/2
    for per in (7, 13, 28, 91):
        for amp in (0.05, 0.20, 0.50):
            x = sine_design(per, amp*XB); C = cost_exact(x)
            v1 = 1/local_fim(x, Z=Zs, free=('c1',))[0, 0]
            v2 = np.linalg.inv(local_fim(x, Z=Zs, free=('c1', 'alpha')))[0, 0]
            print(f"  P={per:4d} A={amp:.2f} C={C:8.4f} | alpha known {C*v1/th_c:6.3f}"
                  f"  | fixed-ref, alpha free {C*v2/th_c:8.3f}")


# ---------------------------------------------------------------- E3
def e3():
    print("E3  matched-cost frequency sweep (C=1.0) and the seasonal-harmonic negative control")
    for per in (4, 7, 13, 14, 21.4117647, 28, 45.5, 52, 91, 182):
        x = sine_design(per, amp_for_cost(per, 1.0))
        s1, s3, s4 = se_ladder(x)
        tag = "   <-- SEASONAL HARMONIC (negative control)" if per in (52, 26) else ""
        print(f"  P={per:9.3f}  se_calib={s1:.4f}  se_local3={s3:12.4f}  se_struct4={s4:14.4f}{tag}")


# ---------------------------------------------------------------- E4
def vif_fixed_ref(omega, thd=th):
    """Round-1 closed form for the *fixed-reference* slope.  Exact against the FIM,
    but it prices the wrong functional (see Round 2)."""
    al = thd['alpha']; ab = abar_opt(thd, m)
    hp = dhill(ab, thd['K'], thd['S']); hpp = d2hill(ab, thd['K'], thd['S'])
    r = (hpp/hp)*ab/(1-al)
    Hc = 1/(1-al*np.exp(-1j*omega)); phi = np.angle(np.exp(-1j*omega)*Hc)
    return 1 + (np.cos(phi)+r/abs(Hc))**2/np.sin(phi)**2


def e4():
    print("E4  Round-1 closed form (fixed-reference slope) vs FIM, and the curvature index rho")
    for per in (4, 7, 13, 14, 28, 91):
        w = 2*np.pi/per; x = sine_design(per, 0.05*XB)
        F = local_fim(x, Z=Zs, free=('c1', 'alpha'))
        num = np.linalg.inv(F)[0, 0]*F[0, 0]
        print(f"  P={per:4d}  closed={vif_fixed_ref(w):8.3f}  FIM={num:8.3f}  rel.err={abs(vif_fixed_ref(w)-num)/num:.4f}")
    print("  rho = |1 - S(1-2H*)| identity check across operating points:")
    for f in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        ab = AB*f; hp = dhill(ab, th['K'], th['S']); hpp = d2hill(ab, th['K'], th['S'])
        Hs = hill(ab, th['K'], th['S'])
        print(f"    a/a*={f:.2f}  H={Hs:.3f}  a|h''|/h'={ab*abs(hpp)/hp:.4f}  |1-S(1-2H)|={abs(1-th['S']*(1-2*Hs)):.4f}")


# ---------------------------------------------------------------- E5 / E6
def fejer(d, T):
    d = np.atleast_1d(d).astype(float)
    return np.where(np.abs(np.sin(d/2)) < 1e-14, 1.0,
                    np.sin(T*d/2)**2/(T**2*np.maximum(np.sin(d/2)**2, 1e-300)))


def e5():
    print("E5  leakage: c1-information lost vs distance (Fourier bins) from the annual harmonic")
    w1 = 2*np.pi/52.0; binw = 2*np.pi/TW
    for d in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 1.6, 2.0, 3.0):
        w = w1 + d*binw; per = 2*np.pi/w
        x = sine_design(per, 0.05*XB); a = adstock(x, AL)[BURN:]
        I = float(local_fim(x, Z=Zs, free=('c1',))[0, 0]); I0 = np.var(a)*TW/sig**2
        pred = sum(fejer(w-k*w1, TW)[0] for k in (1, 2, 3)) + fejer(w, TW)[0]
        print(f"  d={d:5.2f} bins  loss={1-I/I0:8.5f}  Fejer pred={pred:8.5f}")
    print("  margin rule (calibrated against the 3-harmonic sum): d >= 1.15/(pi*sqrt(eps)) bins"
          f"  -> {1.15/(np.pi*np.sqrt(0.05)):.2f} bins at eps=0.05")


def e6():
    print("E6  nuisance-basis sweep: fraction of c1-information retained")
    bases = {
        'Fourier k<=3 + linear trend': seasonal_basis(TW, 3, 1),
        'Fourier k<=6 + linear trend': seasonal_basis(TW, 6, 1),
        'B-spline baseline,  8 knots': spline_basis(TW, 8),
        'B-spline baseline, 20 knots': spline_basis(TW, 20),
        'B-spline baseline, 52 knots': spline_basis(TW, 52),
        'weekly-of-year dummies (52)': np.column_stack([(np.arange(TW) % 52 == j).astype(float) for j in range(52)]),
    }
    pers = [4, 7, 13, 14, 28, 52, 91, 182]
    for name, Z in bases.items():
        line = []
        for per in pers:
            x = sine_design(per, 0.05*XB); a = adstock(x, AL)[BURN:]
            line.append(float(local_fim(x, Z=Z, free=('c1',))[0, 0])/(np.var(a)*TW/sig**2))
        print(f"  {name:29s} dim={Z.shape[1]:3d} | " + " ".join(f"P{p}:{v:6.3f}" for p, v in zip(pers, line)))


# ---------------------------------------------------------------- E7 / E8
def duty(per, amp, Te):
    u = np.zeros(TW); n = int(round(Te/per))*per if Te >= per else Te
    n = int(n); u[:n] = amp*np.sin(2*np.pi*np.arange(n)/per)
    return build(u)


def e7():
    print("E7  duration-amplitude at fixed cost C=2.0 (period-14 probe, T_e weeks of the window)")
    for Te in (28, 56, 112, 182, 364):
        lo, hi = 1e-4, 0.85*XB
        for _ in range(70):
            mid = .5*(lo+hi)
            if cost_exact(duty(14, mid, Te)) < 2.0: lo = mid
            else: hi = mid
        x = duty(14, .5*(lo+hi), Te)
        print(f"  T_e={Te:4d}  amp={100*.5*(lo+hi)/XB:5.1f}% of spend  C={cost_exact(x):.4f}"
              f"  I(c1)={local_fim(x,Z=Zs,free=('c1',))[0,0]:10.1f}"
              f"  I(c2)={local_fim(x,Z=Zs,free=('c2',))[0,0]:10.1f}")


def e8(Psi=9.87):
    print(f"E8  optimal identification energy;  C* = 0.5*m*sigma*sqrt(Psi*T_r),  L* = 2C*   (Psi={Psi})")
    for Tr in (104, 260, 520):
        Cs = 0.5*m*sig*np.sqrt(Psi*Tr)
        print(f"  T_r={Tr:4d}  C*={Cs:7.4f} ({100*Cs/(Tr*XB):.2f}% of spend)  L*={2*Cs:7.4f} ({100*2*Cs/(Tr*XB):.2f}% of spend)")


# ---------------------------------------------------------------- E9
def e9():
    print("E9  the profit-optimal spend path under multiplicative demand seasonality")
    def infoc1(xx):
        aa = adstock(xx, AL)[BURN:]; v = aa-aa.mean(); vr = v-QZ@(QZ.T@v)
        return float(vr@vr/sig**2)
    for A in (0.0, 0.15, 0.30, 0.50):
        t = np.arange(T); s = 1+A*np.cos(2*np.pi*t/52.0)
        astar = np.array([brentq(lambda a: m*th['beta']*s[i]*dhill(a, th['K'], th['S'])-(1-AL),
                                 AB*0.3, 40) for i in range(T)])
        xo = np.maximum(astar-AL*np.concatenate([[astar[0]], astar[:-1]]), 1e-4)
        u = 0.10*XB*np.sin(2*np.pi*np.arange(TW)/21.4117647); u -= u.mean()
        xm = xo.copy(); xm[BURN:] += u; xm[:BURN] += u[-BURN:]
        print(f"  A={A:.2f}  sd(x)/mean={xo[BURN:].std()/xo[BURN:].mean():.3f}"
              f"  I(c1): optimal path={infoc1(xo):9.1f}   optimal+probe={infoc1(xm):9.1f}")


# ---------------------------------------------------------------- E10
def e10():
    print("E10a probe period chosen from an ASSUMED alpha; realised fixed-ref VIF at true alpha")
    ws = np.linspace(0.02, np.pi-0.02, 3000)
    for al_a in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
        tha = dict(th); tha['alpha'] = al_a
        w_a = ws[np.argmin([vif_fixed_ref(w, tha) for w in ws])]
        best = min(vif_fixed_ref(w) for w in ws)
        print(f"  assumed alpha={al_a:.1f} -> P={2*np.pi/w_a:6.2f} wk | realised {vif_fixed_ref(w_a):7.3f}"
              f" vs best {best:6.3f}  (+{100*(vif_fixed_ref(w_a)/best-1):5.1f}%)")


# ---------------------------------------------------------------- E11
def e11():
    print("E11  D-optimal block half-length vs record length (settling-time vs resolution)")
    def blk(T_, L):
        n = int(np.ceil(T_/L))
        return np.concatenate([[2*XB]*L+[0.0]*L for _ in range(n)])[:T_]
    for T_ in (104, 208, 416, 832):
        Z = seasonal_basis(T_, 3, 1); R = resid_maker(Z)
        best = (-1e18, None)
        for L in range(1, min(T_//3, 120)+1):
            x = blk(T_, L); cols = []
            for p in ('beta', 'K', 'S', 'alpha'):
                h = 1e-5; tp = dict(th); tp[p] += h; tm = dict(th); tm[p] -= h
                mp = tp['beta']*hill(adstock(x, tp['alpha']), tp['K'], tp['S'])
                mn = tm['beta']*hill(adstock(x, tm['alpha']), tm['K'], tm['S'])
                cols.append(R((mp-mn)/(2*h)))
            J = np.column_stack(cols); sgn, ld = np.linalg.slogdet(J.T@J/sig**2)
            if sgn > 0 and ld > best[0]: best = (ld, L)
        print(f"  T={T_:4d}  L*={best[1]:3d} wk   [3/(1-alpha) = {3/(1-AL):.1f}]")


# ---------------------------------------------------------------- E12
def e12():
    print("E12  two channels: c1-information per dollar of forgone profit ~ 2/(kappa_j sigma^2)")
    R = resid_maker(Zs)
    def setup(K, S, al, beta):
        t = dict(beta=beta, K=K, S=S, alpha=al); ab = abar_opt(t, m)
        return t, ab, ab*(1-al), m*beta*abs(d2hill(ab, K, S))
    for (t, ab, xb, kap) in (setup(1.5, 2.0, 0.6, 1.0), setup(2.5, 2.0, 0.3, 1.6)):
        for amp in (0.05, 0.10):
            uw = amp*xb*np.sin(2*np.pi*np.arange(TW)/21.4117647); uw -= uw.mean()
            x = xb + np.concatenate([uw[-BURN:], uw])      # periodic burn-in
            C = cost_exact(x, xb=xb, thv=t)
            a = adstock(x, t['alpha'])[BURN:]; v = R(a-a.mean()); I = float(v@v/sig**2)
            print(f"  K={t['K']} alpha={t['alpha']} kappa={kap:.4f} amp={amp:.2f}: C={C:.4f}"
                  f"  I/C={I/C:9.1f}  pred 2/(kappa sigma^2)={2/(kap*sig**2):9.1f}  ratio={(I/C)/(2/(kap*sig**2)):.3f}")


# ---------------------------------------------------------------- R3
def r3(nrep=250, seed=99):
    print("R3  the Psi ladder at matched cost C=1.0, and Monte-Carlo validation")
    for per in (4, 7, 13, 21.4117647, 28, 45.5, 91, 182):
        x = sine_design(per, amp_for_cost(per, 1.0)); s1, s3, s4 = se_ladder(x)
        print(f"  P={per:9.3f}  se_calib={s1:.4f}  Psi3={(s3/s1)**2:8.3f}  Psi4={(s4/s1)**2:12.3f}")
    rng = np.random.default_rng(seed)
    x = sine_design(21.4117647, 0.10*XB); a = adstock(x, AL)[BURN:]; ab = a.mean()
    mu = th['beta']*hill(a, th['K'], th['S']); s1, s3, s4 = se_ladder(x)
    E3, E4 = [], []
    for _ in range(nrep):
        y = mu + rng.normal(0, sig, TW)
        def f3(p):
            cc1, cc2, al = p; aa = adstock(x, np.clip(al, .05, .95))[BURN:]; aab = aa.mean()
            r = y-(cc1*(aa-aab)+0.5*cc2*(aa-aab)**2); return r-QZ@(QZ.T@r)
        E3.append(m*least_squares(f3, [C1, C2, AL], xtol=1e-13, ftol=1e-13).x[0]/(1-least_squares(f3, [C1, C2, AL], xtol=1e-13, ftol=1e-13).x[2]))
        def f4(p):
            b, K, S, al = p; aa = adstock(x, al)[BURN:]
            r = y-b*hill(aa, K, S); return r-QZ@(QZ.T@r)
        q = least_squares(f4, [1., 1.5, 2., .6], bounds=([1e-3, 1e-3, .1, .02], [1e4, 1e3, 30, .97]),
                          xtol=1e-13, ftol=1e-13).x
        E4.append(mroas(dict(beta=q[0], K=q[1], S=q[2], alpha=q[3])))
    tr = mroas(th)
    for nm, E, pred in (('local3', np.array(E3), s3), ('struct4', np.array(E4), s4)):
        s = E.std(ddof=1)
        print(f"  MC {nm:8s} se={s:.4f}(+-{s/np.sqrt(2*(len(E)-1)):.4f}) FIM={pred:.4f}"
              f" bias={E.mean()-tr:+.4f} rmse={np.sqrt(np.mean((E-tr)**2)):.4f}")


# ---------------------------------------------------------------- R4
def r4():
    print("R4a  generalised exchange law under AR(1) residuals: constant = kappa*S(omega)/2")
    x = sine_design(21.4117647, 0.10*XB); a = adstock(x, AL)[BURN:]; ab = a.mean()
    mu = th['beta']*hill(a, th['K'], th['S']); C = cost_exact(x)
    X = np.column_stack([Zs, a-ab, 0.5*(a-ab)**2]); w = 2*np.pi/21.4117647
    for rho in (0.0, 0.6, 0.9):
        S = (1-rho**2)/abs(1-rho*np.exp(-1j*w))**2 if rho > 0 else 1.0
        vals = []
        for seed in range(5):
            rng = np.random.default_rng(1000+seed); es = []
            for _ in range(300):
                if rho == 0: e = rng.normal(0, sig, TW)
                else:
                    z = rng.normal(0, 1, TW); ww = np.empty(TW); ww[0] = z[0]
                    for t in range(1, TW): ww[t] = rho*ww[t-1]+np.sqrt(1-rho**2)*z[t]
                    e = sig*ww
                es.append(np.linalg.lstsq(X, mu+e, rcond=None)[0][-2])
            vals.append(C*np.var(es, ddof=1)/(KAP*sig**2/2))
        print(f"  rho={rho:.1f}: measured inflation {np.mean(vals):.3f} +- {np.std(vals,ddof=1)/np.sqrt(5):.3f}"
              f"   predicted S(omega)={S:.3f}")
    print("\nR4b  joint optimum under coloured residuals:  minimise S(omega;rho) * se_struct4(P)^2")
    binw = 2*np.pi/TW
    def adm(p):
        ww = 2*np.pi/p
        return (all(abs(ww-2*np.pi*k/52.0)/binw >= 1.6 for k in range(1, 7))
                and TW/p >= 4 and p >= 4/(1-AL))
    cands = [p for p in np.arange(6, 120, 0.5) if adm(p)]
    se4 = {p: se_ladder(sine_design(p, 0.10*XB))[2] for p in cands}
    for rho in (0.0, 0.3, 0.5, 0.7, 0.85):
        J = [((1-rho**2)/abs(1-rho*np.exp(-1j*2*np.pi/p))**2 if rho > 0 else 1.0)*se4[p]**2 for p in cands]
        print(f"  AR(1) rho={rho:.2f}: optimal admissible probe period = {cands[int(np.argmin(J))]:6.1f} wk")


SECTIONS = dict(e1=e1, e2=e2, e3=e3, e4=e4, e5=e5, e6=e6, e7=e7, e8=e8, e9=e9,
                e10=e10, e11=e11, e12=e12, r3=r3, r4=r4)

if __name__ == '__main__':
    want = sys.argv[1:] or list(SECTIONS)
    for k in want:
        print("\n" + "="*78)
        SECTIONS[k]()
