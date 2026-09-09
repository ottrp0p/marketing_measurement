# Dive 12 — Long-run effects under weak surrogacy: the price of the long run

**Queue item:** B6 / M5 (long-run effects under weak surrogacy). **Status:** partially verified (three constructions; the core algebra and one construction's key mechanism clean-room reproduced; coverage claims are Monte-Carlo, 20–300 reps). **Spanned two runs:** an earlier run built the simulator, the surrogate-index algebra, the three generations of the bound (box → ellipsoid → profile-LR) and the panel construction, then was interrupted before any attack rounds or writeup; this run re-ran everything, ran the attack/refine rounds (E13–E27), the clean-room verification, and wrote up. Code: `code/12_surrogacy_horizon_bounds.py` (`quick` ≈ 4 min; individual experiments `e1`…`e27`; needs numpy, scipy, cvxpy/Clarabel).

---

## 1. Problem statement

A geo experiment runs a spend pulse and reads weekly incremental sales for H weeks (typically 8–13, occasionally 26). The decision quantity is the *long-run* total Θ_L = Σ_{t<L} τ_t over L ≈ 104 weeks, because brand advertising's memory/brand-stock channel keeps paying after the surrogates (short-run sales) have gone quiet. Catalog item M5 asks for the missing tool: partial identification of Θ_L from a short experiment plus an observational panel, with bounds indexed by a *surrogacy-violation parameter*, in the spirit of Cinelli–Hazlett for confounding.

Formalization (explicit assumptions):

- **Effect dynamics.** Per-unit impulse response g(h), h = 0,1,…; effect curve of a pulse dx is the convolution τ_t = Σ_k dx_k g(t−k). Benchmark truth: g(h) = 0.5^h + w_B ρ_B^h — a fast "activation" component (1-week half-life) plus a slow "brand-stock" component (26-week half-life) whose weight w_B is set so the brand share of Θ_L is φ (φ = 0.4 benchmark; φ = 0 is the "no long run" control). Pulse dx = (1,1,1,1), L = 104. Θ_L = 13.32 at φ = 0.4, 8.00 at φ = 0.
- **Experiment.** G geos, half treated; y_gt = μ_g + d_gt + W_g τ_t + ε_gt with d an AR(1)(ρ_d = 0.6, unit innovations) geo demand state and ε iid N(0,1); 52 pre-weeks; per-week ANCOVA on the pre-period mean gives τ̂_{0:H−1} with a Ledoit–Wolf-shrunk H×H covariance V̂. Per-week se ≈ 1.6·√(4/G).
- **Surrogate index (ACIK).** Observational panel y_t = d_t + ε_t (no treatment); surrogates S = (y_1,…,y_H); Y_L = Σ_{t≤L} y_t; population linear index γ = Var(S)^{−1}Cov(S, Y_L); the SI estimate of the long-run effect is γ'τ_{1:H}.
- **The surrogacy-violation parameter.** Complete monotonicity with a cap: g(h) = ∫ρ^h dμ(ρ), μ ≥ 0 supported on [0, ρ_max], ρ_max = 0.5^{1/hl_max}. The single index **hl_max** (maximum memory half-life) plays the role of Cinelli–Hazlett's R². hl_max = 0 is full surrogacy (nothing after the window); hl_max = ∞ is no restriction beyond monotone decay.
- **Observational panel for construction 3.** y = μ_g + Σ_k g_k x_{t−k} + d_t + e_t with spend x_t = x̄ + a·d_t + u_t (spend chases demand with unknown a; u iid), K = 52 lags, geo fixed effects, T_o = 156 weeks, G_o = 200 geos.

## 2. Prior state

Literature subagent (this run) and catalogs: the surrogate-index identification (Athey–Chetty–Imbens–Kang) needs Y ⊥ W | S plus comparability; general sensitivity machinery now exists — Fan, Manzanares, Park & Qi 2026 (copula-indexed identified sets), Imbens–Kallus–Mao–Wang (persistent confounding via data combination), Chen–Ritzwoller (efficiency), Kallus–Mao (surrogates as efficiency aids, no surrogacy) — but nothing specialized to a dynamic panel with an AR(1) demand state and a brand-stock channel, and no closed form for the bias. Complete-monotone extrapolation is a functional-analysis topic (Brown & Grabovsky 2024) never applied to lift tests; LP shape-restriction bounds are standard in the MTE literature (Mogstad–Santos–Torgovitsky 2018) but not for impulse responses. Experiment-grounded de-confounding of observational estimates exists (Kallus–Puli–Shalit 2018; Rosenman et al. shrinkage) but nobody parametrizes the confounding bias *shape* from the spend autocorrelation. Empirical anchors: weekly carryover ≈ 0.9 (Shapiro–Hitsch–Tuchman 2021; Dubé–Hitsch–Manchanda 2005, half-life ≈ 6 weeks), Binet–Field's 60/40 qualitative long/short split, Lodish et al.'s BehaviorScan long-term multipliers. Practitioner geo-lift tools handle the long run with heuristic "cooldown" windows. All three constructions below therefore appear unclaimed as posed.

## 3. Constructions

### 3.1 Construction 1 — the surrogate-index bias identity (established here, clean-room exact)

Because the SI is linear and the treatment enters the surrogates additively,

  **SI − Θ_L = (γ − 1)'τ_{1:H} − Σ_{t>H} τ_t.**

Two terms with opposite signs and different physics:

- **Bypass tail** −Σ_{t>H}τ_t: the part of the effect that never touches the surrogates. Negative, grows with φ and hl_B, shrinks with H.
- **Phantom persistence** (γ−1)'τ_{1:H}: the SI learned, on the observational panel, that a high y_t predicts high future y because *demand* is persistent (γ_t > 1); it then reads the treatment-induced shift in S as a demand shift and extrapolates it. Positive whenever demand is persistent, and it is a surrogacy violation *even when the long-run effect is zero* — the surrogate is a noisy proxy for the state driving Y_L, and W shifts the proxy without shifting the state. Conditioning on the pre-period mean does not help (E14: identical bias to three decimals) because the AR(1) state is Markov given y_1.

Numbers (E1/E2/E2b, exact; clean-room reproduced to all printed digits): φ = 0.4, H = 13: Θ_L = 13.32, SI = 9.63, bias −27.7% (Σγ = 14.12, last γ's 1.02, 1.06, 1.22, 1.82); MC of the whole pipeline (300-geo panel fit + 200-geo experiment, 40 reps) 9.46 ± 0.44. Bias by H at φ = 0.4: −69% (H=2), −35% (4), −28% (13), −19% (26), −8% (52). The sign flips with demand persistence: at φ = 0.2, H = 4 the bias is −17% for ρ_d = 0.6 but **+116% for ρ_d = 0.9 and +295% for ρ_d = 0.95**; with φ = 0 (no long run at all) and ρ_d = 0.9 the SI over-states the H = 4 effect by +163%. There is a *crossing horizon* H* where the two terms cancel and the SI is accidentally right (H* ≈ 8 at φ = 0.2, ρ_d = 0.9) — right for the wrong reasons, and it moves with (φ, hl_B, ρ_d), none of which the analyst observes.

Comparison with the ACIK partial-R² style interval (E8, G = 100, H = 13): the interval built from (R²_{W|S} = 0.44, R²_{Y|S} = 0.13) has width 0.99×truth but covers only 81% — the R² calculus does not know the bias has a tail term outside the surrogate span.

### 3.2 Construction 2 — the completely-monotone tail bound indexed by hl_max

Feasible set: {μ ≥ 0 on [0, ρ_max]}; the observed window pins Aμ ≈ τ̂; the target c'μ = Θ_L is linear in μ. Three generations (the attack history is in §4):

1. **Box LP** (Round 1): |τ̂_t − (Aμ)_t| ≤ z·se_t simultaneously (Šidák z), two LPs. Conservative: coverage 1.00, width 3.8×truth at G = 100, H = 13 (E4/E6).
2. **Ellipsoid QCQP** (Round 2): (τ̂ − Aμ)'V̂^{−1}(τ̂ − Aμ) ≤ χ²_H; marginal gain (3.7×), and Hotelling inflation with V̂ estimated on 50 geos in 13 dimensions.
3. **Profile-LR** (Round 3, the one used): Q(Θ) = min_{μ≥0, c'μ=Θ} (τ̂ − Aμ)'V̂^{−1}(τ̂ − Aμ); interval {Θ : Q(Θ) − Q_min ≤ crit}, weeks beyond 13 blocked into 13-week bins to keep V̂ well-conditioned. χ²₁(0.95) = 3.84 *under-covers on the low side* (6–9% one-sided failures at φ = 0); crit = 6.63 is MC-calibrated (≤3.3% one-sided everywhere tested, E9b).

**Noise-free identified set** (E3, clean-room exact): at H = 13, φ = 0.4 the window alone leaves [13.25, 13.32] if hl_max = 26, [13.25, 14.77] if hl_max = 52, [13.25, 15.53] if hl_max = 104 — the shape restriction is *very* informative when the curve is known exactly. The whole difficulty is noise.

**The price of the long run** (E9, 100 reps, hl_max = 52; brackets are mean bounds ± sd across reps):

| G | H | bound | width/truth | coverage |
|---|---|---|---|---|
| 100 | 13 | [6.1 ± 1.5, 42.0 ± 10.4] | 2.69 | 1.00 |
| 100 | 52 | [6.7 ± 1.8, 30.9 ± 7.9] | 1.82 | 1.00 |
| 400 | 13 | [7.6 ± 0.8, 31.1 ± 5.3] | 1.76 | 1.00 |
| 400 | 26 | [7.7 ± 1.2, 25.9 ± 4.7] | 1.36 | 1.00 |
| 400 | 52 | [7.7 ± 1.3, 23.5 ± 3.8] | 1.19 | 0.99 |
| 1600 | 13 | [8.5 ± 0.7, 23.5 ± 3.4] | 1.12 | 1.00 |
| 1600 | 52 | [9.2 ± 0.9, 19.3 ± 1.8] | 0.76 | 1.00 |

Seed sweep (E20, five seeds, G = 400/H = 26): width/truth 1.36–1.45, coverage 0.98–1.00. Truth 13.32.

**Width law** (E7b, noise-free τ̂ with the true V). The unidentified direction is the ρ_max component: it adds an almost-constant level ℓ = m·D per observed week (D = pulse mass) and a total m·D·L_eff, L_eff = (1−ρ_max^L)/(1−ρ_max) (= 56.6 at hl_max = 52). Holding everything else fixed, the profile admits m ≤ m_max = √(crit / a'V^{−1}a) with a the in-window signature of that component, so

  **hi − Θ_true ≈ κ · m_max · c_slow, κ ∈ [1.1, 1.6]** (measured 1.30, 1.42, 1.57, 1.24, 1.15, 1.09, 1.30 across seven (G,H,hl_max) configurations),

where κ > 1 is the slack from re-fitting the fast components. The earlier run's cruder law (z·se_level·L_eff·D) has a spurious factor D and overshoots ~3×; the naive z·se_GLS·L_eff is within ±30%. Consequences: the upper bound scales like 1/√G and roughly 1/√H_eff through se, and **linearly in L_eff(hl_max)** — the sensitivity curve.

**Sensitivity curve — the M5 deliverable** (E19, G = 400, H = 26, 60 reps): upper bound 13.0 (hl_max = 4), 15.1 (8), 17.2 (13), 21.2 (26), 25.9 (52), 29.7 (104), 32.2 (208), i.e. **hi ≈ 11.5 + 0.235·L_eff(hl_max)** with the lower bound flat at 7.7 (it does not depend on hl_max at all). Coverage 0.35 at hl_max = 4 (the truth has a 26-week component), 0.87 at 8, ≥0.98 from 13 on. An analyst reports the upper bound as a function of the memory half-life they are willing to entertain and reads off the breakdown half-life at which the long-run ROAS crosses their decision threshold — exactly the Cinelli–Hazlett plot for surrogacy.

**Asymmetry of the two sides.** The lower bound is *not* limited by the long run: at H = 13 it sits 5–8 below truth and improves only ~√G-slowly (7.9 → 6.1 → 5.1 for G = 100 → 400 → 1600, E7b) and barely with H, because a short window cannot tell a 13-week from a 26-week half-life (in-window shapes differ by ρ^{12}: 0.53 vs 0.73), and the lower extreme reassigns the slow mass to the fastest rate the noise allows. So: **G buys the upper bound, H buys the lower bound's rate resolution, and neither buys the other.**

**Power control** (E13, 100 reps): P(lower bound excludes the φ = 0 total 8.0 | φ = 0.4 truth) = 0.12 (G=100,H=13), 0.39 (400,26), 0.93 (1600,52), 1.00 (6400,52); P(upper bound excludes 13.3 | φ = 0 truth) = 0.00, 0.03, 0.47, 1.00. The instrument discriminates — but *establishing* a long-run effect needs ~1600 geo-equivalents × 52 weeks, and *ruling one out* needs 4× more. This is the quantitative content of "short geo tests cannot bound the long run": they can, at a price nobody pays.

### 3.3 Construction 3 — one experiment de-confounds the whole observational lag structure

Observational distributed-lag OLS with spend chasing demand has probability-limit bias **s·b** where the scalar s = a·Var(d) (unknown sign and size) multiplies a *known shape* b = (a²Var(d)·R + σ_u²I)^{−1} r, R_{kj} = ρ_d^{|k−j|}, r_k = ρ_d^k — every ingredient of b is estimable from the spend autocovariances (γ̂₀, γ̂₁, γ̂₂ give ρ̂_d, a²V̂ar(d), σ̂_u²). The experiment, which has no s, pins the short lags; the panel's 52 lags, corrected by ŝ·b, deliver the tail. Joint profile-LR: Q = (τ̂ − A_eμ)'V_e^{−1}(·) + (β̂ − A_gμ − s·b)'V_b^{−1}(·) over (μ ≥ 0, s free).

Numbers (E12, G = 100/H = 13 experiment + 200×156 panel, 40 reps): experiment alone [6.0, 43.3] (width 2.80×truth); **joint [10.1, 15.4] (0.39×, coverage 1.00) — 7× tighter**; naive panel [15.0, 17.1] (biased, coverage 0.00); panel alone with s free: unbounded (s·b at ρ_d = 0.6 is collinear with dictionary mass near ρ = 0.6 — the experiment is what breaks the collinearity). ŝ = 0.69 ± 0.26 vs true 0.78. Value persists with better experiments (E23): joint width 0.25× at (400, 26) and 0.18× at (1600, 52) vs 1.40× and 0.75× alone. Strong confounding a = 1.5 (E24): joint [12.6, 15.5], coverage 1.00.

## 4. Attack-and-refine history

**Round 1 (earlier run, reconstructed from the code):** SI identity; box-LP bound; width law with the D factor. **Attack 1:** the box is Šidák-conservative and the ellipsoid inherits Hotelling inflation; both push width ≥ 3.7×. **Refine 1:** profile-LR with blocked weeks; χ²₁ radius under-covers at φ = 0 → MC-calibrated crit 6.63 (E9b). The run ended here.

**Attack 2 (this run), on construction 2.**
- *Is the under-coverage a V̂ artifact?* No: with the true covariance the one-sided low failure at χ²₁(0.95) is still 6–7% (E15). It grows with the richness of the dictionary — 2.7% with hl_max = 1 week, 4.7% (3), 5.0% (8), 6.7% (52) at G = 400/H = 26, while the unconstrained one-sided sum test fails 1.3% (E15b). Diagnosis: the profile statistic at a cone boundary is a chi-bar-square in which the nuisance cone over-fits noise (Q_min too low), not a χ²₁; the calibrated radius stands, and the exact boundary distribution is left open (BL37).
- *Misspecified kernels.* Erlang-2 hump, 8-week delayed onset, and a pull-forward negative lobe are all covered by the geometric-only dictionary at G = 400/H = 26 (coverage 0.993–1.00, E10) **and** at G = 1600/H = 52 (0.99–1.00, E16), with widths 0.7–1.0×; enlarging the dictionary (Erlang + lags) costs 1.5–2.5× width for no coverage gain, and the Q_min misfit statistic barely moves (13.5–15.9 vs k = 16) — the bound's width is dominated by the tail extrapolation, which does not care about in-window shape.
- *Counterexample (E16b).* A brand component whose onset lies beyond the window: lag 30 with H = 26 → coverage 0.47, and Q_min (11.2, k = 14) does *not* flag it; with H = 52 the onset is inside the window, Q_min rises to 22.9 and coverage returns to 0.95. This is the boundary of the assumption, not of the estimator: no window can bound what starts after it. The honest statement of hl_max is "the response has begun by the end of the window and decays no slower than hl_max."
- *Noise.* t₃ shocks, national common shocks: unchanged (E11). Demand persistence ρ_d = 0.9/0.95 doubles–triples the width because the per-week se doubles; coverage holds.
- *Width law.* The old law overshoots 3× (spurious D); the slow-direction law is within κ = 1.1–1.6 (E7b).

**Attack 3, on construction 3.**
- *Spend anticipates demand* (x_t = x̄ + a·d_{t+ℓ} + u): with the ℓ = 0 shape the joint bound is confidently wrong ([26, 30] vs 13.3, coverage 0) — but loudly: Q_min = 1,600–2,600 against ≈63 dof (E17). **Refine:** a *family* of shapes b_ℓ = (·)^{−1}r_ℓ, r_{ℓ,k} = ρ^{|k−ℓ|}, ℓ ∈ {0,1,2,3,4,6}, each with its own free scalar; coverage 0.88–0.92 at true leads 0/2/4 with width 0.44–0.46× (vs 0.38× for the single correct shape), Q_min back to 68–75 (E21).
- *Saturation.* A naive Hill attack mixed estimands (linear experiment vs saturated panel). The fair version — both obey Σ g_k·Hill(x_{t−k}), the experiment measures the finite difference at the operating point — gives the joint bound with fixed scale coverage 0.85 (E22): the panel's linear lag coefficients are Hill's average derivative times g, a *common* scale across lags. **Refine:** a free panel/experiment scale κ (profiled on a grid): coverage 1.00 at width 1.49× — still 4× tighter than the experiment alone (5.67×); κ̂ = 1.50 ± 0.29 vs ≈1.2 expected.
- *Coverage audit* (E26, 60 reps, G = 400/H = 26): high-side failures 10% with the AR(1) shape (15% even with the *true* (ρ_d, a²Var d, σ_u²)), so the shape itself is off. **Refine — fixed effects bend the shape:** the within transformation over T_o = 156 weeks subtracts (1/T_o)Σ_j Cov(x_{t−k}, d_j) ≈ a·Var(d)·(1+ρ)/((1−ρ)T_o) from every lag's covariance with the omitted state — a *non-decaying* term that the infinite-T shape misses and that maps through (X'X)^{−1} into a level of ≈ −0.003 per lag (per unit of s) at k = 10…52, cutting Σb from 1.18 to 1.02. **Clean-room verified** by direct simulation (4000 geos × 6 seeds: −0.0029 mean over k ≥ 10, Σ = 1.016 ± 0.007) and by an independent analytic plim (Σ = 1.022). Using the exact within-T shape restores coverage to 0.95 (high-side failures 3%) at width 0.23× (E25/E26).
- *Two-component demand* (AR(1) shape genuinely wrong, E18): joint [11.8, 17.4], coverage 0.90, Q_min 74 — graceful degradation, flagged weakly; the shape-family idea generalizes (open).
- *Residual misfit.* Q_min ≈ 80 vs ≈63 dof persists after the shape fix; it falls to 71 with G_o = 800 but not with T_o = 400 (E27) — consistent with the small-sample downward bias of CR0 cluster-robust covariance at 53 parameters/200 clusters, plus something unexplained (≈+8). Coverage is unaffected.

**Round 4:** re-running the audit with the within-T shape produced no further revision; the lead family and κ profile are additive fixes. Stop.

## 5. Independent verification

Clean-room agent (given claims only): Claim A (SI identity, all γ's, −27.7%, +163%, +2.0%) exact to every printed digit, with a 20k-geo MC confirming γ; Claim B (four noise-free identified sets) exact; Claim C (infinite-T shape b and the within-transform level term) exact by formula and reproduced by simulation, including the analytic within plim (their first attempt omitted the cross-lag u covariance and got −0.005/Σ = 0.88; the corrected derivation agrees). No discrepancies of substance.

## 6. Devil's advocate (against each headline)

- *"SI is biased −28%."* Under this DGP, with linear SI and no covariates. A nonlinear SI, or one trained on a panel that contains past campaigns (so S carries treatment variation), changes γ; the identity still holds but γ does. The phantom term also assumes the panel's demand persistence equals the experiment's — comparability, which ACIK assume anyway.
- *"Width 1.4× truth at 400 geos × 26 weeks."* The noise level (per-week se ≈ 0.16 on a peak effect of ~2) is one calibration; with cleaner geos or better covariates every width shrinks ∝ se. But the *ratio* to what a naive analyst reports (the 26-week sum ± CI) is the point, and that ratio is set by L_eff/H.
- *"hi is linear in L_eff."* Measured on a 7-point grid with the lower bound flat; the slope mixes z·se_ℓ, κ and the truth's own brand level, so it is empirical, not derived.
- *"Joint bound 7× tighter."* Requires (i) spend exogenous except through the AR(1) state — spend that anticipates demand is repaired only if the lead lies in the candidate family; (ii) a common effect scale across geos/time between panel and experiment (κ absorbs saturation but not heterogeneity); (iii) the fixed-effects shape at the right T_o. Under two-component demand it under-covers (0.90). And crit 6.63 was calibrated on construction 2's boundary geometry, not the joint problem's.
- *"Misspecified kernels are covered."* At the tested noise levels; the E16b counterexample shows the assumption's edge is *onset timing*, which the geometric dictionary cannot see and Q_min cannot flag.

## 7. Limitations and failure modes

Linear-in-spend effects (Hill handled only as a scale); a single pulse (multi-pulse designs would identify rates better — link dive 01); additive Gaussian-ish geo noise; the crit was MC-calibrated for one boundary geometry; the profile relies on a 200–300-point rate grid (fine at these tolerances); construction 3 assumes the panel's confounding runs only through the demand state that the spend ACF reveals; no competitive response, no price co-movement.

## 8. Next steps for a future session

1. **Exact boundary distribution of the profile-LR at the CM cone** (BL37): derive the chi-bar-square weights as a function of dictionary richness; replace the MC-calibrated 6.63 with a computed radius.
2. **Design for the lower bound:** the lower bound is rate-resolution-limited; test two-pulse/switch designs (dive 04's machinery) and longer post-periods with *fewer* geos at matched cost — the (G, H) frontier for the identified set.
3. **Shape families for construction 3:** AR(2)/two-state demand, lead+lag families, and a Q_min-driven family-selection rule with a derived null (links BL4/BL30).
4. **Decision layer:** feed [lo, hi](hl_max) into dive 08's PoR/deadband as an interval-valued posterior — what does a deadband policy do with a set instead of a distribution?
5. **Field pilot:** any advertiser with ≥1 geo test and a 3-year weekly spend panel can compute ŝ·b and Q_min today; a Q_min explosion (E17) would be the first field detection of anticipatory spend.
6. **Onset-timing sensitivity:** a second index (maximum onset lag) alongside hl_max, making E16b's failure an explicit assumption in the report.

## 9. Sources

- Internal: dives 01 (design/identification), 02 (transport), 04 (switch designs), 08/09 (decision layer); catalogs `../02_open_questions.md` §M5, `../03_mmm_adoption_barriers.md`.
- [Athey, Chetty, Imbens & Kang, "The Surrogate Index" (NBER w26463)](https://www.nber.org/papers/w26463) · [Fan, Manzanares, Park & Qi 2026, arXiv 2603.00580](https://arxiv.org/abs/2603.00580) · [Imbens, Kallus, Mao & Wang, arXiv 2202.07234](https://arxiv.org/abs/2202.07234) · [Chen & Ritzwoller, arXiv 2107.14405](https://arxiv.org/abs/2107.14405) · [Kallus & Mao, JRSS-B 2025](https://academic.oup.com/jrsssb/article/87/2/480/7829031) · [Yang, Eckles, Dhillon & Aral, arXiv 2010.15835](https://arxiv.org/abs/2010.15835) · [Proximal surrogate index, arXiv 2601.17712](https://arxiv.org/html/2601.17712)
- [Brown & Grabovsky 2024, extrapolation of completely monotone functions, arXiv 2401.15178](https://arxiv.org/abs/2401.15178) · [Mogstad, Santos & Torgovitsky 2018 / ivmte](https://github.com/jkcshea/ivmte)
- [Kallus, Puli & Shalit 2018, arXiv 1810.11646](https://arxiv.org/abs/1810.11646) · [Rosenman, Basse, Owen & Baiocchi 2023, arXiv 2002.06708](https://arxiv.org/abs/2002.06708) · [Gordon, Zettelmeyer, Bhargava & Chapsky 2019](https://www.kellogg.northwestern.edu/faculty/gordon_b/files/fb_comparison.pdf)
- [Shapiro, Hitsch & Tuchman 2021, Econometrica](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17674) · [Dubé, Hitsch & Manchanda 2005, QME](https://link.springer.com/article/10.1007/s11129-005-0334-2) · [Lodish et al. 1995, JMR](https://journals.sagepub.com/doi/abs/10.1177/002224379503200201) · Binet & Field 2013, *The Long and the Short of It* · [Google Ads conversion-lift cooldown guidance](https://support.google.com/google-ads/answer/14102986)

*Code:* `code/12_surrogacy_horizon_bounds.py`. (`_main12.txt` / `_run12.sh` in `code/` are leftovers of the interrupted run and can be deleted; this session lacked delete permission.)
