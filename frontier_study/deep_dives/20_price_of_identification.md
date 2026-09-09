# Deep dive 20 — The price of identification: the spectrum and the cost of MMM experimental design

**Queue item:** backlog **BL1** (seasonality-orthogonal perturbation design) + **BL2** (the price of identification), both spawned by dive 01 · **Date:** 2026-09-04 · **Code:** `code/20_price_of_identification.py`, results `code/20_*.json`

---

## 1. Problem statement

Dive 01 established *which* spend paths identify an MMM (block pulsing at half-length $\approx 3/(1-\alpha)$, amplitude-⁴ scaling of curvature information, transients as free dose-ranging) and closed with two admitted gaps, logged as BL1 and BL2:

> "No controls/seasonality/trend in the simulations. Real designs must also be near-orthogonal to seasonal harmonics… collinearity with seasonality will degrade everything above."
> "Quantify short-run revenue forgone by a D-optimal pulsing plan vs the revenue-optimal constant plan — the cost–precision frontier a CMO chooses on."

This dive answers both, jointly, because they turn out to be the same question. Formally:

Let weekly sales be
$$y_t \;=\; z_t'\gamma \;+\; \beta\,h\!\big(a_t;K,S\big) \;+\; \varepsilon_t,\qquad a_t = x_t + \alpha a_{t-1},\qquad h(a)=\frac{a^S}{a^S+K^S},$$
with $z_t$ a nuisance basis (intercept, trend, seasonal harmonics or a flexible baseline) and $\varepsilon$ a stationary process with normalised spectral density $S_\varepsilon(\omega)$ ($S_\varepsilon \equiv \sigma^2$ for white noise). Weekly profit is $\pi_t = m\,y_t - x_t$; the myopically optimal constant spend $\bar x^*$ solves $m\beta h'(\bar a^*) = 1-\alpha$ with $\bar a = \bar x/(1-\alpha)$.

A **probing design** is a mean-zero perturbation $u$ added to $\bar x^*$. Two questions:

1. **(BL1, spectral)** Which frequencies should $u$ occupy, given that the nuisance basis annihilates a subspace and that finite records leak?
2. **(BL2, economic)** What does the perturbation cost in forgone profit, what does it buy in posterior precision, and what is the optimal amount?

Estimand throughout: the **marginal ROAS at the operating point**, $\mathrm{mROAS} = m\beta h'(\bar a^*)/(1-\alpha)$ — the number a budget decision actually consumes (dives 03, 08, 09).

Working point for all numerics: $(\beta,K,S,\alpha)=(1,1.5,2,0.6)$, $m=2$, $\sigma=0.05$; then $\bar a^*=2.1804$, $\bar x^*=0.8722$, $h''(\bar a^*)=-0.15731$, $\kappa \equiv m\beta|h''(\bar a^*)| = 0.31463$, weekly channel profit $0.4854$. Analysis window $T_w = 364$ weeks (7 years) preceded by a 52-week burn-in carrying the periodic extension of the design.

---

## 2. Prior state

**Established elsewhere (and the literature scan confirms these are prior art).**

- *Frequency-domain input design.* The Fisher information for a dynamic system is an integral of the input spectrum against transfer-function sensitivities (Mehra 1974; Goodwin & Payne 1977; Ljung 1999 Ch. 13); the feasible set is a moment set, so optima are realisable as finite multisines. "Least costly identification experiment for control" (Bombois et al. 2006) minimises input **power** subject to a frequency-wise control-performance LMI, solved as an SDP in the spectrum's moment vector; Rojas et al. (2008, Thm 3) prove least-costly and traditional designs are duals. Hjalmarsson's (2009) *cost of complexity* gives the nearest existing "excitation energy per unit accuracy" law. **None of this is stated for a profit cost, and none of it has been transported to marketing.**
- *Ds-optimality with nuisance parameters* — maximise the determinant of the Schur complement of the nuisance block — is Atkinson & Cox (1974), textbook in classical DoE, essentially absent from dynamic input design.
- *Leakage.* The expected periodogram is the true spectrum convolved with the Fejér kernel: main lobe $1/T$, sidelobes decaying $1/f^2$. Harmonic-suppressed multisines (Schoukens–Pintelon–Guillaume) deliberately leave nuisance lines unexcited. **No source in any field states a frequency-separation margin rule as a design constraint.**
- *Cost of learning.* Little (1966) already chooses experiment size "to minimize the cost of imperfect information plus the cost of experimentation" in a quadratic advertising-response model; Pekelman & Tse (1980) and Kolsarici, Vakratsas & Naik (2020) give the adaptive-control advertising version (probe increment $\propto$ posterior variance of effectiveness, empirically confirmed on 8 brands). den Boer & Zwart (2014) prove controlled-variance pricing attains $O(T^{1/2+\epsilon})$ regret with dispersion $\propto t^{-1/4}$, using exactly the second-order Taylor argument used below. Feit & Berman (2019) give a closed-form profit-maximising test size $n^*\approx\frac12\sqrt N (s/\sigma)$ and, in their eq. (9), a closed-form map from posterior precision to deployment profit — **an existing price-of-identification frontier, for a two-arm i.i.d. test.**
- *Marketing pulsing theory* (Sasieni 1989; Mahajan & Muller 1986; Feinberg 2001; Aravindakshan & Naik 2015) is entirely **profit-driven**, never identification-driven; Feinberg proves that in continuous time no finite pulsing frequency can be optimal, so any finite probing period must come from discreteness, memory, or — as here — identification.
- *MMM-specific.* Dew, Padilla & Shchetkina (2024) show saturation and time-varying effectiveness are not separately identifiable from standard MMM data, worst when media is autocorrelated, and propose an intervention-based *maximal-separation* experiment (model discrimination, time domain, no cost accounting). Heusch (2026) recovers $(\alpha,\lambda,\beta)$ structurally from geo go-darks. Chan & Perry (2017) and the Meridian docs name "insufficient variation in media spend" as the binding constraint. Recast's practitioner note (Feb 2026) is the closest prose statement of BL1 — "consider flighting your spend both with and against your seasonality… These are real risks" — with no math, no optimum, and no quantification of the cost.
- Internally: dive 01 (identified set, block design), 02 (transport operator), 03/04 (EVSI, the decision plateau, LQC), 08/09 (deadband, decision loss), 11 (leakage of imputed channels into measured ones), 16/19 (the certainty-equivalent stall — the *dynamic* cousin of §6 below).

**The gap this dive fills.** No one has written the MMM design problem in the frequency domain at all; no one has projected a probing spectrum onto a seasonal-harmonic (or spline) nuisance subspace; no one has stated the leakage margin as a rule; and no one has priced MMM identification in dollars against posterior standard error. The organising frame borrowed from process control is **plant-friendly identification**: probing a running plant while protecting product quality — where the "product quality" being protected is revenue.

---

## 3. Round 1 — the constructions

### 3.1 The same-filter theorem and the exchange rate *(derived here; the $\beta$ half is nearly a tautology, see §4.1)*

Write $u$ for the mean-zero spend perturbation and $\tilde a = a - \bar a$ for the resulting adstock deviation, so $\tilde a = H_\alpha u$ with $H_\alpha(\omega) = (1-\alpha e^{-i\omega})^{-1}$.

**Cost.** Because we perturb around the profit optimum, the linear term vanishes and
$$C_{\text{week}} \;=\; \tfrac12\,\kappa\,\mathrm{Var}(a) + O(\mathbb E \tilde a^3),\qquad \kappa = m\beta|h''(\bar a^*)|. \tag{1}$$

**Information.** For the local slope $c_1 = \beta h'(\bar a^*)$, the score is $\tilde a$, so after projecting out the nuisance basis $Z$,
$$I_{c_1} \;=\; \frac{\|M_Z\tilde a\|^2}{\sigma^2} \;=\; \frac{T_w \mathrm{Var}(a)}{\sigma^2}\,\big(1-\nu\big), \tag{2}$$
with $\nu$ the fraction of *filtered* energy absorbed by $Z$.

Cost and information are the same quadratic form in the same filtered signal. Hence, on the clean band ($\nu = 0$), for **total** window cost $C$:

> **Theorem 1 (exchange rate / price-of-precision hyperbola).**
> $$\boxed{\,C \cdot \mathrm{Var}(\hat c_1)\;=\;\tfrac12\,\kappa\,\sigma^2\,}$$
> independently of probing frequency, amplitude and duration. Equivalently, halving the standard error costs exactly four times as much profit. In the generalised form derived in §5.2, $\sigma^2 \to S_\varepsilon(\omega_{\text{probe}})$.

The immediate corollaries are the useful part:

- **Frequency invariance (at matched cost).** Probing at 4 weeks and at 182 weeks buys *identical* precision per dollar for the linear effect. The economics are frequency-blind; only the nuisance projection is not.
- **Duration–amplitude invariance.** Total excitation energy $E = T_e\,\mathrm{Var}(a)$ is the sole design scalar for $c_1$: a 4-week go-dark and a two-year 5% wiggle of equal cost are equally informative about mROAS.
- **The cheap-channel law.** Information per dollar forgone is $I/C = 2/(\kappa_j\sigma^2)$ — *inversely proportional to the channel's profit curvature*. Flat channels (the ones whose ROAS nobody can pin down) are the **cheap** ones to probe.
- **Curvature is the exception.** The score for $c_2 = \beta h''$ is $\tfrac12\tilde a^2$, so $I_{c_2}\propto T_e\mathrm{Var}(a)^2 = E^2/T_e$: at fixed cost, *concentrate*. This is dive 01's "amplitude is everything" with an economic exchange rate attached, and it is why go-darks exist.

### 3.2 The nuisance collision and the leakage margin *(derived here; unclaimed in any field per the literature scan)*

The nuisance basis annihilates energy at its own frequencies exactly, and nearby energy partially. For a rectangular record of length $T$, a probe at distance $\Delta$ from a nuisance harmonic loses the Fejér fraction
$$F_T(\Delta)=\frac{\sin^2(T\Delta/2)}{T^2\sin^2(\Delta/2)}\;\le\;\frac{4}{T^2\Delta^2},$$
summed over harmonics. In Fourier-bin units ($\text{bin}=2\pi/T$) the envelope is $1/(\pi^2 d^2)$, so a naive margin rule is $d \ge 1/(\pi\sqrt\epsilon)$ bins. The clean-room verifier showed this single-sidelobe bound is violated when three harmonics' sidelobes add; the calibrated rule is

> **Rule L (leakage margin).** Keep the probe **$d \ge 1.15/(\pi\sqrt\epsilon)$ Fourier bins** from every nuisance harmonic to lose less than $\epsilon$ of its information ($\approx 1.6$ bins at $\epsilon = 0.05$), **or** place it at an exact Fourier frequency of the estimation window (integer cycles), where leakage is identically zero at any separation $\ge 1$ bin.

### 3.3 The confounding index and the (later refuted) closed form

Round 1 also produced a closed form for the inflation of $\mathrm{Var}(\hat c_1)$ when $\alpha$ is estimated jointly. With $r = \frac{h''(\bar a)}{h'(\bar a)}\cdot\frac{\bar a}{1-\alpha}$ and $\phi_H(\omega) = \arg\!\big(e^{-i\omega}H_\alpha(\omega)\big)$,
$$\mathrm{VIF}(\omega)\;=\;1+\frac{\big(\cos\phi_H + r/|H_\alpha|\big)^2}{\sin^2\phi_H}. \tag{3}$$
It matches the Fisher information to 0.04–0.8% (E4), and in the long-memory limit it collapses to a clean statement in terms of a single **curvature index**
$$\rho \;=\; \frac{\bar a\,|h''(\bar a)|}{h'(\bar a)}\;=\;\big|1-S(1-2H^*)\big|,\qquad H^*=h(\bar a^*),$$
namely $\mathrm{VIF}^*=1+4\rho(\rho-1)$ at $\omega^*\tau=\sqrt{1-1/\rho}$ for $\rho>1$, and $\mathrm{VIF}^*=1$ (perfect separation) for $\rho<1$. For $S=2$ the threshold $\rho=1$ is exactly $H^*=1/2$, i.e. $\bar a^* = K$:

> **Observation (verified as an identity, E4).** Operating *below* half-saturation makes carryover and effectiveness separable; operating *above* it — where a profit-maximising firm sits whenever margins are healthy — makes them confounded, and the confounding grows with margin ($\mathrm{VIF}^*$ rises $5.9 \to 21.2$ as spend goes from $1\times$ to $3\times$ the optimum).

**This closed form did not survive Round 2 as a statement about the decision.** §4.2 explains why, and what replaces it.

---

## 4. Round 2 — the attack

### 4.1 "The invariance is a tautology"

Fair, for the linear coefficient. Once cost and information are written as quadratic forms in the same filtered signal, $I/C$ being frequency-free is close to immediate. Two defences: (i) it is *not* a tautology that the exact nonlinear cost obeys it — E1 verifies the second-order law to $\le 0.5\%$ up to $\pm$50% spend swings, and the clean-room verifier found the worst case over commensurate periods to be $+0.41\%$; (ii) it is not a tautology which *other* parameters break it, and they do (§3.1 curvature, §4.2 carryover, §5.2 coloured noise). The load-bearing content is the asymmetry, not the invariance.

### 4.2 The attack that landed: the closed form prices the wrong functional

Monte Carlo (250 reps) of the local estimator gave a variance inflation of $\approx 1.3$ where eq. (3) predicted $4.6$, and at a 7-week period MC gave $0.98$ against a predicted $12.1$ — an inflation below one, which is impossible for the quantity eq. (3) claims to describe.

Diagnosis: eq. (3) is the inflation for **the slope at a fixed reference adstock level**. The decision functional is the slope at the *fitted* operating point, $\mathrm{mROAS} = m\hat c_1/(1-\hat\alpha)$, whose gradient carries an extra term $m\,c_2\,\partial\bar a/\partial\alpha/(1-\alpha)$ that **cancels most of the confounding**: raising $\hat\alpha$ moves the operating point up the curve, and the reported marginal effect is far better determined than either ingredient. Numerically the corrected gradient is $(5,\,0,\,-1.788)$ against the $(5,\,0,\,+2.5)$ implied by the naive delta method — different in sign.

This is the same phenomenon dives 03/04 found in a different coordinate system: **quantities individually unidentified can combine into a well-determined decision functional.** Eq. (3) survives only as an exact result about a non-decision parameter, and is reported as such.

### 4.3 Other attacks run

| Attack | Result |
|---|---|
| Does the cost law survive large amplitude? | Yes to $\pm 50\%$ of spend ($\le 0.5\%$; verifier $+0.41\%$ worst over commensurate periods, $-3.5\%$ at a 3-week period where the sampled sinusoid is degenerate) |
| Does the estimator attain the frontier? | **No.** The bounded structural MLE is materially biased ($+0.107$ on an mROAS of $1.31$) and beats the CRLB in variance ($0.202$ vs $0.311$): RMSE $0.229$. The frontier is an information statement; a prior-regularised MMM trades along it (§5.3) |
| Is the "resolution" story for dive 01's block length right? | **Refuted, my own conjecture.** $L^*$ is $8$–$10$ weeks at $T = 104, 208, 416, 832$ — flat, i.e. a settling-time effect. Dive 01's explanation stands |
| Is a designed probe more efficient than managerial noise? | **No, per dollar.** Log-normal managerial noise at $cv = 10\%$ delivers $2604$ information units per profit dollar vs $2473$ for the designed probe. Its defect is endogeneity, not inefficiency (§6) |
| Do heavy tails break the exchange law? | No: $t_3$ errors give $0.965\pm0.023$ |
| Does residual autocorrelation? | **Yes, and lawfully** — this became a construction (§5.2) |
| Wrong assumed $\alpha$ when choosing the period? | Benign downward, dangerous upward: $\le +5\%$ loss for $\alpha_{\text{assumed}} \le 0.7$, $+233\%$ at $0.9$ |

---

## 5. Round 3 — refinement and the second construction

### 5.1 The $\Psi$ ladder: a price of ignorance, not a price of frequency

Replace eq. (3) with a directly computed inflation of the decision functional,
$$\Psi(\text{design},\text{knowledge}) \;=\;\frac{\mathrm{Var}(\widehat{\mathrm{mROAS}})}{\mathrm{Var}(\widehat{\mathrm{mROAS}}\mid \alpha,\ \text{curve shape known})},$$
so that the frontier becomes
$$\boxed{\;C\cdot\mathrm{Var}\big(\widehat{\mathrm{mROAS}}\big)\;=\;\Big(\tfrac{m}{1-\alpha}\Big)^{2}\,\tfrac12\,\kappa\,S_\varepsilon(\omega)\;\cdot\;\Psi\;}$$

Matched-cost sweep at $C = 1.0$ (0.44% of the window's spend), white residuals:

| probe period (wk) | 4 | 7 | 13 | 21.4 | 28 | 45.5 | **52** | 91 | 182 |
|---|---|---|---|---|---|---|---|---|---|
| $\mathrm{se}$, calibrated | 0.0992 | 0.0992 | 0.0992 | 0.0992 | 0.0992 | 0.0992 | 0.0992 | 0.0992 | 0.0992 |
| $\Psi_3$ ($c_1,c_2,\alpha$ free) | 17.0 | 4.69 | 1.97 | 1.35 | 1.20 | 1.08 | $10^{27}$ | 1.06 | 1.19 |
| $\Psi_4$ (full Hill free) | $3\times10^{9}$ | 12.1 | 10.7 | 10.1 | 9.98 | 9.87 | $10^{16}$ | 9.94 | 10.5 |

Three readings, all verified:

1. **The calibrated standard error is exactly frequency-invariant** — 0.0992 to four significant figures across a 45× range of periods (verifier: spread 0.004%). Theorem 1, cleanly visible.
2. **For the full Hill model the frequency choice is nearly decision-irrelevant** — $\Psi_4$ varies by 10% over periods 13–182 weeks. The **decision plateau of dives 03/04 reappears in the frequency domain.** What is *not* second-order is landing on a nuisance harmonic ($\Psi \to 10^{16}$) and probing faster than the memory ($\Psi_4 = 3\times 10^9$ at a 4-week period, where carryover becomes unidentifiable).
3. **The ladder is the real lever.** Going from "full Hill free" ($\Psi\approx 9.9$) to "$\alpha$ and shape known" ($\Psi = 1$) cuts the standing standard error by $\sqrt{9.9}\approx 3.1\times$ at fixed cost. A one-time shape-fixing experiment (dive 01 §3.5's transient, Heusch's go-dark, Dew et al.'s maximal-separation test) is therefore worth a factor of ten in the perpetual identification budget — a concrete VOI case that plugs directly into dive 02's transport operator and dive 03's EVSI.

Monte-Carlo check at a 21.4-week period, 10% amplitude, 250 reps: local-3 $\mathrm{se}_{\rm MC} = 0.1186\pm0.0053$ against FIM $0.1134$ (agrees within $1\sigma$); structural-4 $\mathrm{se}_{\rm MC} = 0.2022\pm0.0091$ with bias $+0.107$, RMSE $0.229$, against CRLB $0.311$.

### 5.2 Second construction: the noise spectrum restores a frequency optimum

Theorem 1 generalises. Replacing $\sigma^2$ by the residual spectral density at the probe frequency,
$$C\cdot\mathrm{Var}(\hat c_1)\;=\;\tfrac12\,\kappa\,S_\varepsilon(\omega_{\text{probe}}),\qquad S_\varepsilon(\omega)=\sigma^2\frac{1-\rho^2}{|1-\rho e^{-i\omega}|^2}\ \text{ for AR(1)}.$$
Measured inflation (5 seeds × 300 reps) at $\omega = 0.2934$: $3.077\pm0.080$ against a predicted $3.029$ for $\rho = 0.6$, and $2.215\pm0.057$ against $2.185$ (asymptotic) / $2.258$ (exact finite-sample sandwich, verifier) for $\rho = 0.9$.

This is what makes the design problem non-degenerate. MMM residuals are strongly positively autocorrelated, so the noise floor is *low-frequency-heavy* — pushing probes fast — while parameter separation pushes them slow. Minimising $S_\varepsilon(\omega)\Psi_4(\omega)$ over admissible periods:

| residual AR(1) $\rho$ | 0.0 | 0.3 | 0.5 | 0.7 | 0.85 |
|---|---|---|---|---|---|
| optimal admissible probe period (wk) | 84 | 84 | 84 | 12 | 11 |

**A sharp crossover between $\rho = 0.5$ and $\rho = 0.7$.** Within the admissible band the total loss varies only by 1.0–2.6×, so this is a second-order refinement — but it is the parameter that decides whether "flight annually-ish" or "flight quarterly-ish" is right, and it is estimable from any fitted MMM's residuals.

### 5.3 The price of identification

Certainty-equivalent budgeting maps an mROAS error $e$ into a spend error $e(1-\alpha)^2/\kappa$ and hence a per-period regret $\mathrm{Var}(\widehat{\mathrm{mROAS}})(1-\alpha)^2/(2\kappa)$. Substituting the frontier, **$\kappa$ and $(1-\alpha)$ both cancel**:
$$L(C)\;=\;\underbrace{C}_{\text{experiment}}\;+\;\underbrace{\frac{T_r\,m^2\sigma^2\Psi}{4C}}_{\text{residual misallocation}} \quad\Longrightarrow\quad \boxed{\;C^*=\tfrac12 m\sigma\sqrt{\Psi\,T_r},\qquad L^*=2C^*=m\sigma\sqrt{\Psi\,T_r}\;}$$

> **Theorem 2 (price of identification).** The minimum total price of identifying a channel is **margin × residual sales noise × $\sqrt{\text{decision horizon} \times \Psi}$**, split exactly half between money spent probing and money lost to residual ignorance. It does **not** depend on the response curvature, on the adstock rate, or on how the excitation is split between duration and amplitude.

Verified: closed form vs numeric argmin agree to 0.4% (theory $C^*=1.7289$, $L^*=3.4578$ at $\Psi = 4.599$, $T_r = 260$; the verifier reproduced the closed form as the exact argmin to six decimals).

In practitioner units, per channel, over a 5-year decision horizon:

| knowledge state | $\Psi$ | probing cost $C^*$ | total price $L^*$ |
|---|---|---|---|
| fully calibrated ($\alpha,K,S$ known from experiments) | 1.0 | **0.36% of media spend** | 0.71% |
| local shape free ($c_1,c_2,\alpha$) | 1.06 | 0.37% | 0.73% |
| full Hill free ($\beta,K,S,\alpha$) | 9.87 | **1.12% of media spend** | 2.23% |

Two things to notice. First, these are **an order of magnitude below** the practitioner rules of thumb for geo holdouts ("10–20% of the test region's budget", "pause spend in 10–15% of regions for 4–8 weeks"), because a continuous low-amplitude probe pays the second-order cost while a go-dark pays a large one. Second, the *uncertainty* on these numbers is dominated by $\sigma/\bar\pi$ and by $\Psi$, not by the theory: $L^*\propto m\sigma$ exactly, so an advertiser with twice the residual noise pays twice as much, and one who knows their curve shape pays $3.1\times$ less.

### 5.4 The design recipe

Admissibility, then optimisation:

1. **Never probe on a nuisance harmonic.** At $52/k$ weeks the information is *identically zero* — the strongest negative control in this dive (se $\to 10^{6}$–$10^{12}$ at matched cost).
2. **Rule L:** stay $\ge 1.6$ Fourier bins ($\epsilon = 5\%$) from every harmonic, or land on an exact Fourier frequency of the estimation window.
3. **Match the probe to the baseline specification** (E6, the most actionable table in the dive):

| nuisance basis | what it annihilates |
|---|---|
| Fourier $k\le3$ + trend | periods 52 only (and $\ge 182$ partially: 0.85 retained) |
| Fourier $k\le6$ | also period 13 — i.e. **quarterly flighting** |
| B-spline baseline, 8 knots | periods $\ge 91$ (0.16, 0.001 retained) |
| B-spline baseline, 52 knots | period 28 (0.000) and half of period 14 |
| weekly-of-year dummies | **every probe whose period divides 52**: 4, 13, 26, 52 all → 0.000 |

  A flexible baseline is a low-pass nuisance filter with cutoff $\approx 2\times$ the knot spacing; weekly seasonal dummies are a comb filter that eats exactly the monthly/quarterly/annual cycles media planners naturally use.
4. **Keep $\ge 4$ cycles in the estimation window** and **period $\ge 4\tau$** ($\tau=1/(1-\alpha)$), or carryover stops being identified.
5. **Within the admissible set**, pick the period minimising $S_\varepsilon(\omega)\Psi(\omega)$ — long ($\sim$18 months) for white-ish residuals, short ($\sim$3 months) for strongly autocorrelated ones.
6. **Split the budget by parameter:** a continuous low-amplitude probe for mROAS (duration-invariant), plus occasional short deep excursions for curve shape ($I_{c_2}$ concentrates).

Applied to the working example with a 7-year window and annual seasonality: the best admissible period is **84 weeks**, giving $\mathrm{se}(\mathrm{mROAS}) = 0.261$ against $0.381$ for conventional 13-week quarterly flighting — a $1.46\times$ standard-error gain, i.e. **2.1× less profit forgone for the same precision.** A 17-week ("quarterly-ish") flight sits $0.36$ bins from the third annual harmonic and **retains only 43% of its information**.

---

## 6. The design theorem behind MMM's identification problem

The sharpest result in the dive is also the simplest to state. Let demand seasonality be multiplicative, $s_t = 1+A\cos(2\pi t/52)$, and let the firm set spend quasi-statically optimally, $m s_t\beta h'(a_t^*) = 1-\alpha$. Then the entire spend path lives in the span of the seasonal nuisance basis, and after projection:

| seasonal amplitude $A$ | 0.00 | 0.15 | 0.30 | 0.50 |
|---|---|---|---|---|
| $\mathrm{sd}(x)/\mathrm{mean}(x)$ | 0.000 | 0.064 | 0.134 | 0.255 |
| $I(c_1)$ from the optimal path | 0.0 | **0.0** | **0.0** | 16.7 |
| $I(c_1)$, same path + 10% probe | 2617 | 2617 | 2617 | 2633 |

> **The profit-maximising media plan is, to first order, exactly non-identifying** — not because its variation is small (a 25% coefficient of variation is *more* spend variation than most advertisers have) but because optimal variation is, by construction, a deterministic function of the same seasonal demand signal the model must control for. The residual $16.7$ at $A=0.5$ is pure Hill nonlinearity leaking into harmonics $k\ge4$.

This is a stronger and more precise statement than the field's usual "insufficient variation in media spend" (Chan & Perry 2017; the Meridian docs), and it explains why more history does not help: the *rate* of information accrual from optimal spend is zero, not small. It is also the static counterpart of dive 16/19's certainty-equivalent stall — there the loop freezes at an uninformative spend *level*; here the loop freezes on an uninformative spend *subspace*.

**The honest counterweight (E9b).** Real managers do not optimise exactly. Log-normal deviations at $cv = 10\%$ generate $I(c_1) = 1633\pm30$ at a cost of $0.627$ — **2604 information units per profit dollar, slightly better than the designed probe's 2473.** Undesigned managerial noise is not inefficient; it is *endogenous* (it responds to demand, per Kolsarici et al.'s empirically confirmed $\Delta \propto \sigma^2_\beta$ rule) and *uncontrolled* (its frequency content is whatever it happens to be, including the harmonics). The case for design is identification-validity and harmonic avoidance, not thermodynamic efficiency — and dive 19's BL68 (no log-only detector separates exogenous from demand-chasing variation) is exactly why that distinction cannot be recovered after the fact.

---

## 7. Devil's advocate, per headline claim

**"$C\cdot\mathrm{Var}(\hat c_1)=\kappa\sigma^2/2$ exactly."** It is second-order in amplitude and asymptotic in record length. The verifier found $+0.72\%$ at 38% amplitude and $-3.5\%$ at a 3-week period; it is exact only for commensurate periods. And it prices *information*, not any particular estimator's RMSE — §4.3 shows the estimator actually used in practice sits off the frontier.

**"$L^*=m\sigma\sqrt{\Psi T_r}$, the price of identification."** The concept is Little (1966) and Feit & Berman (2019); only the object — a frontier for an adstock+saturation response with a nuisance baseline, denominated in percent of media spend — is new. It also assumes a stationary world with a single decision at the end. With drift (dives 05, 07, 08) the horizon $T_r$ is effectively capped at the drift time-scale, which *raises* the amortised price; with a deadband policy (dives 08/09) the regret is not quadratic in the estimate but piecewise, which lowers it. Both corrections are first-order in the answer and neither is computed here.

**"Frequency choice is nearly decision-irrelevant ($\Psi_4$ flat within 10%)."** True at this operating point and this $\alpha$; the operating-point sweep (E10b) shows $\Psi$-type inflation rising from 1.8 to 21 as spend goes from $0.75\times$ to $3\times$ optimal, and I did not re-run the full $\Psi_4$ frequency profile at those points. The claim should be read as: *at a well-run channel near its optimum, after excluding harmonics and sub-memory periods, frequency is second-order.*

**"The profit-optimal path is non-identifying."** It depends on the seasonality being exactly in the control basis. If the demand signal is imperfectly known to the modeller (the realistic case), the optimal path is *not* in the span, and it does carry information — which the model will then partly misattribute. The clean-room verifier also flagged an ambiguity: under multiplicative seasonality the natural score regressor is $s_t(a_t-\bar a)$ rather than $(a_t-\bar a)$; the quoted numbers use the latter. The qualitative claim is robust to this, the exact zeros are not.

**"Rule L: 1.6 bins."** Calibrated to three harmonics plus intercept and trend at $T=364$. More harmonics, or a spline baseline, widen it; the general statement is that the required margin grows like $\sqrt{\text{number of nearby nuisance lines}}/(\pi\sqrt\epsilon)$.

**"Continuous probing is 10× cheaper than go-darks."** The comparison is against practitioner rules of thumb I could not verify in any peer-reviewed or vendor-published source; the literature scan found no citable estimate of incrementality-testing cost as a fraction of revenue. Treat the 0.36–1.12% figures as *derived here for this model*, and the comparison as indicative.

---

## 8. Independent verification

A clean-room subagent, given only the model specification and the numerical claims (no code), re-implemented everything independently. It reproduced **C0 (all five constants), C1, C2 (analytic and MC), C3 (0.099177–0.099181 across periods 4–182), C4 (both spectral values and both Monte Carlos), C5 (all four leakage pairs), C6 (closed form = numeric argmin to 6 d.p., plus an independent re-derivation of the algebra showing $\kappa$ and $(1-\alpha)$ cancel), C7 (0.000 / 0.048 / 16.65 / 2616.4) and C8 (662.6 vs 22.16, ratio 29.9)**.

It raised six qualifications, all adopted above:

1. C1's 0.4% bound holds only for periods commensurate with the window ($+1.28\%$ at $P=77.7$, $-3.52\%$ at $P=3$).
2. C2's exactness degrades to $+0.72\%$ at 38% amplitude; 300 MC reps give ~8% precision on a variance and cannot adjudicate a 0.5% claim (its 4000-rep run: $0.989\pm0.022$).
3. C4's spectral law is asymptotic: the exact finite-sample sandwich gives $2.258$ vs $2.185$ at $\rho=0.9$ ($T=364$).
4. **C5's margin corollary was wrong as stated**: at $d = 1.42$ the measured loss is $0.0576 > 0.05$ because three sidelobes add. Rule L now reads $1.15/(\pi\sqrt\epsilon)$.
5. **C8's mechanism was wrong**: the factor-30 gain from concentrating a probe is *not* all the $C^2/T_e$ law, which predicts $364/28 = 13$. Decomposing $I(c_2)$ at $T_e = 28$ into the windowed-$2\omega$ component ($277.2$) and a rectangular burst-envelope component ($385.4$), the $2\omega$ part alone gives ratio $12.5$ — the law — and the remainder is a *window-contrast* term that exists only when $T_e < T_w$. The design advice survives; the stated rationale is now split in two. (Logged as BL78.)
6. C7's score regressor is ambiguous under multiplicative seasonality (see §7).

---

## 9. Limitations and failure modes

- **Single channel.** The multichannel results here are minimal (E12's cheap-channel law, verified to 0.2%; budget-neutral switches cost within 2.5% of total-budget pulses in the separable case, consistent with dive 19's separability finding). The $n$-channel spectral design — how many channels a 52-week harmonic comb can accommodate with mutually orthogonal probes — is not done.
- **The frontier is an information statement.** The estimator that MMM practitioners actually run is a prior-regularised Bayesian fit whose RMSE sits *inside* the CRLB frontier at the price of bias ($0.229$ vs $0.311$ here). Pricing the prior in frontier units is open (BL75), and it is the honest form of the catalogue's M7 concern.
- **Quasi-static optimisation.** §6 uses a myopic period-by-period optimum; a truly forward-looking advertiser with carryover would deviate from it, generating some information for free.
- **Sinusoidal probes only.** Real plans are bounded below by zero and quantised by flighting practice. Plant-friendly identification has an established answer (minimum-crest-factor multisines, Rivera et al.) that is not transported here (BL77).
- **No drift.** Everything is stationary. Dives 05/07/08 show the world is not, and drift caps the effective $T_r$.
- **$\Psi$ and $S_\varepsilon(\omega)$ are design inputs nobody measures.** The recipe consumes a residual spectrum and a knowledge state; both are estimable from a fitted MMM, but the estimator (and its own instability, dive 07) is not built (BL74).

---

## 10. Next steps for a future session

1. **The $n$-channel spectral comb.** Assign each channel its own admissible probe frequency; derive the packing constraint (how many channels fit between the harmonics of a 52-week season at a given leakage tolerance and record length), the cross-channel Gram penalty, and the budget-neutral vs total-budget split. Merges with dive 19's BL70 (the budget direction is unidentified under neutral variation — a *total*-budget probe at an admissible frequency is exactly the escape, and Theorem 2 now prices it).
2. **Price the prior.** Express the prior-regularised Bayesian MMM's realised RMSE on the $C$–precision plane and find where an informative prior is worth more than an extra dollar of excitation. This is the decision-theoretic version of the M7 debate and it needs the frontier as its axis.
3. **Estimate $\Psi$ and $S_\varepsilon$ from a real fitted model**, then run the admissibility screen on an actual advertiser's flighting calendar. The single most checkable prediction of this dive is that common flighting periods (13 and 17 weeks) sit on or near annual harmonics and lose 40–100% of their identifying content.
4. **Join to dive 03/04's EVSI.** $L^* = m\sigma\sqrt{\Psi T_r}$ is a scalar summary of the whole design problem; the $\Psi$ ladder says a shape-fixing experiment is worth $\sqrt{10}$ on the perpetual budget. Compute that as an EVSI and compare against dive 04's dual-experiment (+47%) and switch (+22%) margins — this is the missing cost side of BL7.
5. **Drift-capped horizon.** Replace $T_r$ by the effective horizon implied by dives 05/07/08's drift rate $q$ and re-derive $C^*$; the prediction is a *standing* probing rate rather than a one-off budget, which is the always-on version of dive 05's scheduler.
6. **Non-sinusoidal, non-negative, planner-feasible probes** (crest-factor-constrained multisines under $x\ge0$ and monthly planning granularity), and the resulting loss against the sinusoidal ideal.
7. **The window-contrast term** in curvature information (verifier's C8 decomposition) — derive it, since it appears to dominate the practical case for short deep excursions.

---

## 11. Sources

- Internal: dive 01 (BL1/BL2 spawned here; block-length rule reconciled in E11), 02 (transport operator — the $\Psi$ ladder is its VOI case), 03/04 (the decision plateau, which reappears in the frequency domain), 08/09 (decision loss, certainty-equivalent budgeting), 11 (nuisance leakage), 16/19 (the certainty-equivalent stall — §6's dynamic cousin); catalogues `../02_open_questions.md` §M1, §M2, `../03_mmm_adoption_barriers.md` §O7, §B1.
- Input design: [Mohan, Mithun & Bhikkaji 2017, arXiv:1706.03982](https://arxiv.org/pdf/1706.03982) · [Bombois, Scorletti, Gevers, Van den Hof & Hildebrand 2006, Automatica 42(10)](https://perso.uclouvain.be/michel.gevers/PublisMig/C130.pdf) · [Rojas, Agüero, Welsh & Goodwin 2008, Automatica 44(11)](https://www.sciencedirect.com/science/article/abs/pii/S0005109808002604) · [Rojas, Welsh, Goodwin & Feuer 2007, Automatica 43(6)](https://webee.technion.ac.il/people/feuer/JournalPapers/56_Robust_Optimal_Experiment.pdf) · [Hjalmarsson 2009, EJC 15(4)](https://www.kth.se/social/upload/525e5cdaf276545124e66218/hjalmarsson_ejc_2009.pdf) · Mehra 1974, IEEE TAC 19(6); Goodwin & Payne 1977; Ljung 1999 Ch. 13 · Atkinson & Cox 1974, JRSS-B ($D_s$-optimality) · [optimal relevant-subset designs, arXiv:2106.09633](https://arxiv.org/pdf/2106.09633) · Schoukens, Pintelon & Guillaume (harmonic-suppressed multisines) · [Pintelon & Schoukens, leakage reduction / LPM](https://perso.uclouvain.be/michel.gevers/PublisMig/LPMCDC-ECC2011.pdf) · [Subba Rao & Yang 2020, arXiv:2007.00363](https://arxiv.org/pdf/2007.00363) · [Rivera et al., plant-friendly identification](https://www.sciencedirect.com/science/article/abs/pii/S0959152408001352)
- Cost of learning: [Little 1966, Oper. Res. 14(6)](https://pubsonline.informs.org/doi/10.1287/opre.14.6.1075) · [Pekelman & Tse 1980, Oper. Res. 28(2)](https://pubsonline.informs.org/doi/10.1287/opre.28.2.321) · [Kolsarici, Vakratsas & Naik 2020, JMR 57(3)](https://smith.queensu.ca/centres/scotiabank/docs/kolsarici-et-al-2020.pdf) · [den Boer & Zwart 2014, Mgmt Sci 60(3)](https://ir.cwi.nl/pub/22667/22667D.pdf) · [Keskin & Zeevi 2014, Oper. Res. 62(5)](https://pubsonline.informs.org/doi/pdf/10.1287/opre.2014.1294) · [Aghion, Bolton, Harris & Jullien 1991, ReStud 58(4)](https://academic.oup.com/restud/article-abstract/58/4/621/1545777) · [Feit & Berman 2019, Mktg Sci 38(6)](https://arxiv.org/pdf/1811.00457)
- Marketing / MMM: [Dew, Padilla & Shchetkina 2024, arXiv:2408.07678](https://arxiv.org/abs/2408.07678) · [Heusch 2026, arXiv:2608.21128](https://arxiv.org/abs/2608.21128) · [Chan & Perry 2017, Google](https://services.google.com/fh/files/misc/challenges_and_opportunities_in_media_mix_modeling.pdf) · [Jin, Wang, Sun, Chan & Koehler 2017](https://research.google/pubs/pub46001/) · [Meridian: data collection & variation](https://developers.google.com/meridian/docs/pre-modeling/collect-data) · [Sasieni 1989, Mktg Sci 8(4)](https://pubsonline.informs.org/doi/abs/10.1287/mksc.8.4.358) · [Mahajan & Muller 1986, Mktg Sci 5(2)](https://pubsonline.informs.org/doi/10.1287/mksc.5.2.89) · [Feinberg 2001, Mgmt Sci 47(11)](https://pubsonline.informs.org/doi/10.1287/mnsc.47.11.1476.10246) · [Aravindakshan & Naik 2015, Oper. Res.](https://prasadnaik.faculty.ucdavis.edu/wp-content/uploads/sites/422/2016/11/OR2015.pdf) · [Recast, managing multicollinearity (2026)](https://getrecast.com/understand-and-manage-multicollinearity/) — the closest prose statement of BL1

*Code:* `code/20_price_of_identification.py` (run `python 20_price_of_identification.py` for all sections); results `code/20_*.json`.
