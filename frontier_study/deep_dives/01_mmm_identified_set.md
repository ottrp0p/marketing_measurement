# Deep dive 01 — The identified set of MMM parameters (M1)

**Queue item:** A1 (M1, `02_open_questions.md`) · **Date:** 2026-08-31 · **Code:** `code/01_mmm_identified_set.py`

## 1. Problem statement

For the canonical single-channel Bayesian MMM likelihood (Jin et al. 2017),

$$y_t = \beta\, h\big(a_t; K, S\big) + \varepsilon_t,\qquad a_t = x_t + \alpha\, a_{t-1},\qquad h(a;K,S) = \frac{a^S}{a^S + K^S},\qquad \varepsilon_t \sim \mathcal N(0,\sigma^2),$$

characterize how the identifiability of $\theta = (\beta, K, S, \alpha)$ depends on the spend path $\{x_t\}_{t=1}^T$, and derive a design theory: which budget-neutral perturbation schedules maximize information about saturation curvature *separately from* carryover decay?

Two notions used throughout:

- **Sharp identified set** $\Theta_I(x) = \{\theta : \mu_t(\theta; x) = \mu_t(\theta_0; x)\ \forall t\}$ — parameters producing identical mean paths under design $x$.
- **Practical identification** — the eigenstructure of the Fisher information $F(x,\theta_0) = J^\top J / \sigma^2$, $J_{ti} = \partial \mu_t/\partial\theta_i$. A near-zero eigenvalue is a likelihood ridge: point-identified in principle, useless in practice.

Working point for all numerics: $\theta_0 = (\beta, K, S, \alpha) = (1, 1.5, 2, 0.6)$, mean spend $\bar x = 1$ (so steady-state adstock $\bar a = \bar x/(1-\alpha) = 2.5$, i.e., the channel operates above its half-saturation point), $\sigma = 0.05$, $T = 156$ weeks unless noted.

## 2. Prior state

**Established.** Jin et al. (2017) demonstrate empirically that with weekly national data the posterior for $(\alpha, K, S)$ is prior-dominated; Chan & Perry (2017) show five equally well-fitting models disagreeing on allocation by up to 50% (see `../03_mmm_adoption_barriers.md` §M1). Heusch (2026, [arXiv:2608.21128](https://arxiv.org/abs/2608.21128)) shows geo-experiment differencing recovers $(\alpha, \lambda, \beta)$ structurally from the *temporal* variation an experiment induces — evidence that the right variation identifies, without saying which variation. In the dose-response literature (sigmoid Emax ≡ Hill), locally D-optimal designs for the 4-parameter model need ≥4 support points, and when the max dose is far below ED50 only the ratio Emax/ED50-type composite is identified. That static theory has never been transported to the *dynamic* MMM setting, where "dose" $a_t$ is a filtered functional of the design and cannot be set freely. That transport, plus the amplitude-scaling law and the transient-identification result below, is the contribution here.

## 3. Theory developed in this dive

### 3.1 Constant spend: rank-1 information (formalized folklore)

If $x_t \equiv \bar x$, then $a_t \to \bar a$ and the mean path is the single number $c_0 = \beta h(\bar a; K, S)$. The sharp identified set is the 3-dimensional manifold $\{\theta : \beta h(\bar x/(1-\alpha); K, S) = c_0\}$ — one equation in four unknowns. Equivalently $F$ has rank 1 (all gradient columns are constant sequences, hence collinear). *Status: trivially derivable but worth stating — it is the exact limit that business-as-usual near-constant spend approaches.* **Verified (Exp. A):** the constant design's FIM eigenvalues are numerically $(0, 0, 10^{-10.6}, 10^{5.0})$; CRLB standard errors are $10^4$–$10^6$.

### 3.2 The amplitude law: small perturbations leave a δ²-slow ridge (derived here; novel for MMM)

Write $x_t = \bar x(1 + \delta u_t)$ with $u_t$ bounded, mean-zero, and $\delta$ small. Then $a_t = \bar a(1 + \delta v_t)$ with $v_t = (1-\alpha)\sum_{j\ge0}\alpha^j u_{t-j}$, and

$$\mu_t = \beta h(\bar a) + \delta\, \bar a\beta h'(\bar a)\, v_t + \tfrac12 \delta^2 \bar a^2 \beta h''(\bar a)\, v_t^2 + O(\delta^3).$$

Order by order, the design reveals: at $O(1)$ the level $c_0 = \beta h(\bar a)$; at $O(\delta)$ the filter shape (hence $\alpha$, from any temporally rich $u$) and the slope composite $c_1 = \beta h'(\bar a)$; at $O(\delta^2)$ the curvature composite $c_2 = \beta h''(\bar a)$. To first order, $(c_0, c_1, \alpha)$ are 3 constraints on 4 parameters: **the identified set is a one-dimensional ridge through $(\beta, K, S)$-space regardless of the frequency content of the perturbation.** The ridge is resolved only by the curvature term, whose contribution to the Fisher quadratic form scales as $\delta^4$:

$$\lambda_{\min}\big(F\big) = \Theta(T\,\delta^4/\sigma^2), \qquad \lambda_{2}\big(F\big) = \Theta(T\,\delta^2/\sigma^2),$$

so the standard error along the ridge scales as $\sigma/(\delta^2\sqrt T)$ and the sample size needed to reach fixed precision scales as $\delta^{-4}$. Generic point identification (the map $(\beta,K,S)\mapsto(c_0,c_1,c_2)$ at fixed $\bar a$ is generically invertible) coexists with catastrophic practical weakness — this is, we believe, the precise mathematical form of the "flat ridges" Jin et al. observed.

**Verified (Exp. B):** for $x = 1 + \delta\sin(2\pi t/8)$, log-log regression of FIM eigenvalues on $\delta \in [0.025, 0.4]$ gives slope **3.990** for $\lambda_{\min}$ (theory: 4) and **2.020** for $\lambda_2$ (theory: 2). Consequence in absolute terms: a ±10% sinusoidal spend wiggle at $T=156$ has $\lambda_{\min} = 0.0028$; matching the two-level block design below ($\lambda_{\min} \approx 99$) would take ≈ 5.5 million weeks. Business-as-usual spend "variation" is not identification.

### 3.3 Range coverage: below the knee, only $\beta/K^S$ exists (transported from Emax theory)

If $\max_t a_t \ll K$, then $h(a) = (a/K)^S\big(1 + O((a/K)^S)\big)$, so the mean path depends on $(\beta, K)$ only through $\beta/K^S$ (with $S, \alpha$ separately identified from shape and dynamics). The identified set contains, to that order, the curve $\{(\beta\lambda^S,\, \lambda K,\, S,\, \alpha) : \lambda > 0\}$: the saturation point is pure extrapolation. **Verified (Exp. E):** scaling the block design so $\max a/K = 0.29$ drops $\lambda_{\min}$ from $10^{2}$ to $10^{-3}$, the ridge eigenvector rotates into the $(\beta, K)$ plane $(-0.74, -0.66, 0.07, 0.00)$, and a noiseless profile fit reproduces the data essentially exactly ($\mathrm{RSS} \le 3.4\times10^{-4}$) for any $K \in [0.8, 6]$ with $\beta$ sliding from 0.33 to 15.0 along $\beta/K^S \approx$ const (0.52, 0.44, 0.42, 0.42). Design implication: identification of *saturation* requires spend excursions that push adstock at least to the neighborhood of $K$ — you cannot learn the ceiling from below the knee.

### 3.4 Block-length theory: the settling-time / transition-count trade-off (derived here)

For a square wave alternating $x_{hi}/x_{lo}$ with half-period $L$, the steady-state adstock separation between block ends is (derived by solving the two-cycle fixed point):

$$\Delta a(L) = \frac{(x_{hi}-x_{lo})}{1-\alpha}\cdot\underbrace{\frac{1-\alpha^L}{1+\alpha^L}}_{A(L)},$$

so short blocks attenuate the effective dose separation by $A(L) \to 0$ as $L \to 0$ (killing curvature information, per §3.2 the attenuated separation acts like a small $\delta$), while long blocks waste observations: transitions carry the $O(1)$-per-event information about $\alpha$ and (see §3.5) about curvature, and their number is $T/L$. D-optimality therefore has an interior optimum. Rule of thumb from requiring $\alpha^L \le 0.05$: $L^* \gtrsim 3/\ln(1/\alpha) \approx 3/(1-\alpha)$ for $\alpha$ near 1. **Verified (Exp. C):** with $1/(1-\alpha) = 2.5$ weeks, D-criterion peaks at $L^* = 8$–10 (≈ 3–4 settling times); $\lambda_{\min}$ is 0.65 at $L{=}1$, 99 at $L{=}8$, 3.1 at $L{=}78$.

### 3.5 The adstock transient is a free dose-ranging sweep (novel observation, numerically established)

Static dose-response theory says a 4-parameter sigmoid needs ≥4 distinct dose levels. A two-level *dynamic* design nonetheless achieves full-rank, well-conditioned information — because every transition drags $a_t$ through a continuum of intermediate values along the known exponential path, tracing an arc of the Hill curve. The transient is not a nuisance to discard; it is where the curvature identification lives. **Verified (Exp. G):** in a 2-level block design ($L{=}13$), keeping only the 84 settled observations gives $\lambda_{\min} = 0.079$, while keeping 84 transient-heavy observations gives $\lambda_{\min} = 53.4$ — a **~700× collapse at equal sample size** when transients are removed (full series: 85.2). Practical corollary: MMM fitting pipelines that model only steady-state relationships (or geo-test analyses that discard "ramp-up" weeks as contamination) are throwing away most of the curvature information an experiment bought. This dovetails with Heusch (2026): his structural estimator works precisely because it keeps the experiment's temporal transient.

### 3.6 Optimized budget-neutral designs (prototype)

Simulated annealing over spend sequences on levels $\{0, 0.5, \dots, 2.5\}$ with exactly preserved mean spend (swap and transfer moves), maximizing $\det(F)^{1/4}$. **Result (Exp. F, $T{=}96$):** the annealed design beats the best heuristic (3-level blocks) by ×1.23 on D-criterion and beats realistic business-as-usual AR(1) lognormal spend by **×5.0** (and by ×3–4 on parameter SEs). The optimizer's solution is interpretable: a multi-level pulse plan — long full-off runs (35/96 weeks at zero, giving clean decay reads), sustained high blocks at 2–2.5 (probing past the knee), and a minority of intermediate levels (dose-ranging) — i.e., it rediscovers "on/off pulsing + dose ranging + settle-time blocks" as the optimal grammar. Headline SEs at $T{=}96$, $\sigma{=}0.05$: $(\beta, K, S, \alpha)$ SE $= (0.022, 0.065, 0.115, 0.009)$ vs $(0.087, 0.190, 0.453, 0.037)$ for AR(1) spend.

### 3.7 MLE recovery confirms the FIM story end-to-end

**Exp. D** (120 Monte Carlo MLE fits, multistart least squares, $T{=}156$): constant spend — response curve unrecoverable (median max curve error 0.20, i.e., 20% of the sales ceiling); ±10% sine — parameter estimates diverge (RMSE(K) = 32); AR(1) — mediocre (RMSE(S) = 0.38); 2-level blocks — RMSE $(\beta,K,S,\alpha) = (0.021, 0.087, 0.086, 0.011)$ and median max curve error **0.012**, matching CRLB predictions closely. The local FIM analysis is a reliable guide to global estimation behavior in this regime.

## 4. The design recipe (operational summary)

An "identification-first media plan," budget-neutral by construction:

1. Pulse in blocks of length $\ge 3/(1-\alpha)$ weeks (use a prior guess for $\alpha$; 6–10 weeks covers most channels).
2. Use 3+ spend levels including full-off windows and excursions to ~2–2.5× normal spend; ensure peak adstock plausibly reaches or exceeds the suspected half-saturation $K$ (§3.3).
3. Never discard transition weeks from the analysis (§3.5).
4. Amplitude is everything: information about saturation grows as (excursion size)⁴ (§3.2). Two big pulses beat fifty small wiggles by orders of magnitude.
5. If a spend schedule can be chosen freely, anneal the FIM directly (§3.6, code provided); expect ~5× D-criterion over business-as-usual paths.

## 5. Limitations and failure modes

- **No controls/seasonality/trend in the simulations.** Real designs must also be near-orthogonal to seasonal harmonics and control variation; collinearity with seasonality will degrade everything above (spawned as backlog item below).
- **Single channel.** Multi-channel designs must also decorrelate channels from each other; the budget constraint couples them.
- **Local analysis.** FIM eigenstructure is local at $\theta_0$; the D-optimal design depends on the unknown $\theta$ (classic locally-optimal-design circularity). Exp. D shows the local story holds globally in this regime, but a Bayesian/minimax design version is the right production answer.
- **Geometric adstock and exact Hill assumed.** Delayed-peak adstock adds parameters; misspecification of the functional form is not addressed — the "identified set" here is within-model.
- **Exogenous designed spend.** This is a theory of designed perturbation; it does not fix endogeneity of historical spend (that is what geo differencing à la Heusch is for — the two compose naturally).
- The δ⁴ law is derived for a single operating level; mixed regimes interpolate between the δ⁴ and O(1) cases.

## 6. Next steps for a future session

1. **Seasonality-orthogonal design** — re-run the annealer with a seasonal + trend nuisance basis in the model; derive frequency-domain placement rules (pulse energy away from seasonal harmonics); quantify how much D-criterion survives.
2. **Multi-channel coupled design** under a total-budget constraint: does the optimizer stagger channels' pulses (time-division multiplexing)? Conjecture: yes, with overlap only where interaction terms need identifying.
3. **Bayesian/minimax design** to break the locally-optimal circularity: maximize expected log-det under the Jin-et-al-style prior; connect directly to A3/M10 (value of information) — the FIM machinery here is the inner loop that dive will need.
4. **The price of identification:** compute short-run revenue forgone by the D-optimal plan vs the revenue-optimal constant plan under $\theta_0$, giving a $-cost vs posterior-precision frontier a CMO can choose from.
5. **Compose with geo differencing:** apply §3.4–3.6 to the treated-minus-control series of a geo experiment (Heusch's setting) — optimal within-experiment spend schedules, not just optimal deltas.
6. Formalize §3.5 as a theorem: conditions on $(L, \alpha, S)$ under which a two-level dynamic design's FIM is nonsingular while its static-level counterpart's is singular.

## Sources

- [Jin et al. 2017, Bayesian Methods for MMM with Carryover and Shape Effects](https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/b20467a5c27b86c08cceed56fc72ceadb875184a.pdf) · [Chan & Perry 2017](https://research.google/pubs/challenges-and-opportunities-in-media-mix-modeling/)
- [Heusch 2026, Structural Estimation of MMM Parameters from Geo-Experiments, arXiv:2608.21128](https://arxiv.org/abs/2608.21128)
- Dose-response optimal design (sigmoid Emax): [Dette et al., optimal designs for dose response curves](https://arxiv.org/pdf/1603.04500) · [adaptive optimal designs for sigmoid Emax](https://www.sciencedirect.com/science/article/abs/pii/S0378375813002267)
- Input design for nonlinear dynamic systems: [D-optimal input design for nonlinear FIR-type systems](https://arxiv.org/pdf/1703.08401) · [optimal experiment design for pharmacokinetic modeling](https://pmc.ncbi.nlm.nih.gov/articles/PMC11996619/)
