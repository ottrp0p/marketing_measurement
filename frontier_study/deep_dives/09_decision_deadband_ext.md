# Dive 09 — EXTEND of dive 08: the multichannel, budget-coupled deadband (B3; absorbs BL21, advances BL24-adjacent tiering and dive 05's cadence thread)

**Item:** B3 (EXTEND run). Thread chosen: dive 08's next-steps #1 (multichannel radial band, BL21) and #4 (joint cadence–band theory), because the scalar law was the most complete result in the program and its vector version is what any real reallocation needs. **Status of headline results:** the ellipsoid law and its optimality are **verified** (QVI algebra + 2-D dynamic programming + clean-room re-derivation and re-implementation); the tiered pairwise-transfer rule is **verified in structure, approximate in constant** (clean-room: threshold sharp, r_c 5.5% low); cadence separation is **verified**, the 2/3 refresh-cost exponent is **verified for independent refresh noise and shown to fail under overlap-correlated noise**; the nonlinear-MMM transport is **partially verified** (one 4-channel configuration, flat basin). The discrete-time cost constant −0.165·tr(ΣQ) is **conjecture** (numerically supported to ~3%).

---

## 1. Problem statement

Dive 08 solved the scalar problem: when the optimum drifts as a random walk and is seen only through noisy refreshes, act on the *Kalman-filtered* deviation when it leaves a band h\* = (12Kq/γ)^{1/4} − 0.5826√q, independent of posterior width. Real reallocation is a vector across n channels sharing a budget, the fixed cost K is paid once per reallocation *event* (a meeting, a re-brief, a credibility charge), and channels differ in curvature and drift. Three questions:

1. **Shape.** With n channels, deviation z ∈ R^n constrained to Σz = 0 (budget), curvature Γ (n×n Hessian of expected profit at the incumbent allocation) and drift covariance Q, what region of filtered deviations should trigger a reallocation? The naive candidates are a *box* (dive 08's scalar band per channel), a *sphere* (Alvarez–Lippi 2014's radial rule with an average q), a *Mahalanobis/χ² region* (what a CI-minded analyst would use), or an ellipsoid of some orientation.
2. **Partial moves.** If the cost has a per-channel component (each changed line item costs c on top of the meeting cost K₀), when is it optimal to move money between just two channels and leave the rest alone?
3. **Cadence.** Refreshes cost c_r each. How does the refresh interval Δ interact with the band, and what is the optimal Δ?

Formal setup. Work in the (n−1)-dimensional tangent space of the budget simplex with orthonormal basis B (n×(n−1)); u = Bᵀz. Between refreshes u_{k+1} = u_k + w_k, w ~ N(0, Q); refresh k delivers y_k = u_k + ε_k, ε ~ N(0, V); flow loss per period ½uᵀΣu with Σ = BᵀΓB; a reallocation resets the filtered deviation to zero at cost K (Q2 generalizes the cost). Objective: long-run average cost.

## 2. Prior state (literature subagent, confirmed)

- **Multi-product menu cost** (Alvarez & Lippi 2014, Econometrica): for n *independent, identically volatile* gaps and a single fixed cost the inaction region is a ball in Σgap², with small-cost threshold ȳ = √(2(n+2)σ²ψ/B); frequency N = nσ²/ȳ; kurtosis 3n/(n+2). No anisotropic closed form; per-product costs (Bonomo et al. 2023; Bhattarai–Schoenle 2014; Midrigan 2011) are treated numerically or with a continuum of products.
- **Anisotropic fixed-cost asymptotics** (Altarovici–Muhle-Karbe–Soner 2015, F&S; Atkinson–Wilmott 1995 ansatz): leading-order no-trade region for multi-asset portfolios with fixed costs is an ellipsoid {ρᵀMρ < 1} with M solving 4M·Tr(AM) + 8MAM = Σ. This is the key piece of prior art for Q1 — derived there as a *small-cost asymptotic* in a wealth problem.
- **Discrete monitoring in several dimensions** (Gobet–Menozzi 2010): shift the boundary inward along its normal by 0.5826·|nᵀσ|√Δ — the multidimensional Broadie–Glasserman–Kou.
- **Observation + menu costs** (Alvarez–Lippi–Paciello 2011, QJE; Reis 2006): with costly *perfect* observation the review interval scales as (cost)^{1/2}.
- **Marketing/OR application:** none found (confirmed gap): no paper applies inaction bands or fixed-cost impulse control to budget reallocation across channels or to "when to act on an MMM refresh."
- **Vector Muth identity:** not stated in vector form anywhere found; follows from the Riccati fixed point (proved below).

Internal: dive 08 (scalar law, v-invariance, PoR, protocol); dive 07 (refresh-noise correlation ρ ≈ ω = 1 − Δ/W for overlapping windows, lower for rotating nonlinear MMMs); dive 05 (cadence law τ\*).

## 3. Constructions

### 3.1 The vector identity: estimation noise never enters the band (any dimension)

Steady-state filter for the vector local-level model: P̄ = P + Q (predictive), G = P̄(P̄+V)⁻¹, Δû = G·innovation, so Cov(Δû) = P̄(P̄+V)⁻¹P̄. The Riccati fixed point P = P̄ − P̄(P̄+V)⁻¹P̄ gives **Cov(Δû) = P̄ − P = Q exactly**, for every V. The filtered optimum is a random walk with the *true* drift covariance, in every direction. (E1: algebraic residual 4×10⁻¹⁴; MC over 4×10⁵ steps, anisotropic 3-D Q and V, max relative error 0.2%.) Consequently the reduction of dive 08 §3.2 holds verbatim in R^m: the partially observed problem is *exactly* fully observed impulse control of the martingale û, plus a policy-independent floor ½tr(ΣP).

### 3.2 The ellipsoid law (derived here as an exact ergodic solution; asymptotic prior art in Altarovici et al.)

Ansatz for the relative value function W(u) = K[2Φ − Φ²], Φ = uᵀMu. Then ½tr(Q∇²W) = K[2(1−Φ)tr(QM) − 4uᵀMQMu], and the HJB inside the region, ½tr(Q∇²W) + ½uᵀΣu = g, holds identically in u iff

  **4M·tr(QM) + 8MQM = Σ/K,  g = 2K·tr(QM).**

Verification of the full QVI: value matching W(Φ=1) = K = W(0)+K; smooth fit ∇W = 2K(1−Φ)Mu = 0 on Φ = 1; W ≤ K everywhere; outside, ½uᵀΣu = K[4uᵀMQMu + 2Φ tr(QM)] ≥ 2K tr(QM) = g whenever Φ ≥ 1 because MQM ⪰ 0. So for the *ergodic* LQ-Brownian problem the ellipsoid is not merely leading-order — it is the exact continuation region (modulo the standard verification theorem). **Closed-form solution by whitening:** S = Q^{1/2}ΣQ^{1/2}/K = U diag(s)Uᵀ; M̃ = U diag(m)Uᵀ with m_i(4T + 8m_i) = s_i, T = Σm_i (one scalar root-find); M = Q^{−1/2}M̃Q^{−1/2}. Checks: m = 1 returns (12Kq/γ)^{1/4}; isotropic m dims returns the Alvarez–Lippi radius R\* = (4(m+2)Kq/γ)^{1/4}. Scaling: K and Q enter only through the product KQ (M scales as 1/√f under Q → fQ or K → fK), exactly as in one dimension.

**Discrete refreshes:** shrink the boundary inward along its normal by β√(nᵀQn), β = 0.5826 (Gobet–Menozzi). In closed form the trigger becomes

  **act iff Φ(û) + 2β·√(ûᵀMQMû / Φ(û)) ≥ 1.**

**Discrete-time cost:** the flow cost is charged at the start of each period rather than integrated; the resulting average cost is g_disc ≈ 2K tr(QM) + ½tr(ΣP) − κ·tr(ΣQ) with κ measured at 0.154–0.164 (1-D, q from 1 to 1/8) and 0.158 (2-D case B), trending toward the conjectured κ = ¼ − β²/4 = 0.165 as step/band → 0 (¼ is the exact "start-of-period vs integrated" correction; the β²/4 is the second-order monitoring penalty). **Conjecture, not derived.**

### 3.3 Decision-framed outputs in vector form

With u | data ~ N(û, P) the gain from moving by û is ûᵀΣu − ½ûᵀΣû ~ N(½ûᵀΣû, ûᵀΣPΣû), so **PoR = Φ_N(−½ûᵀΣû / √(ûᵀΣPΣû))** (MC: 0.30281 vs 0.30327 at n = 2×10⁶). Expected gain of acting Ĝ = ½ûᵀΣû; the registered threshold is now a *matrix* M rather than a scalar, but the act/hold statistic is still one number, Φ(û)+BGK term vs 1. Note that PoR is no longer a monotone function of the trigger statistic when Σ, P, M are not proportional, so "act when PoR ≤ α\*" is no longer identical to the optimal rule — the org should register M, and report PoR as a companion, not as the trigger.

### 3.4 Second construction: the tiered (pairwise-transfer) rule under per-channel costs

Cost K₀ + c·(#channels changed). For n = 3 the tangent plane is 2-D and a pairwise transfer i↔j moves u along the projected direction of e_i − e_j (three lines at 0°, 60°, 120°) at cost K₀ + 2c versus K₀ + 3c for a full reset. Solving the DP with both action types (E4) shows a sharp structure: **the pairwise move is chosen iff the perpendicular distance r from û to the nearest transfer line is below a threshold r_c** — the max residual among pairwise states and the min residual among full-reset states coincide to two decimals (1.63/1.64; 2.12/2.12). Derivation of the threshold: leaving residual r behind costs W(r) − W(0) under the relative value function; with W = K_full(2Φ − Φ²) and Φ = r²/R\*², pairwise pays iff W(r) < c, i.e.

  **r_c = R\*·√(1 − √(1 − c/K_full)),  R\* = (4(m+2)K_full q/γ)^{1/4}.**

Predicted 1.61 / 2.00 vs DP 1.63 / 2.12 (c = 6.67 / 10 at K_full = 30). The tiered heuristic policy — full-cost ellipsoid trigger; pairwise-cost trigger on the along-line component when r < r_c; choose pairwise iff r < r_c — lands within 0.07–0.65% of the DP optimum across four cost splits (E4c), while the full-reset ellipsoid alone is 2–14% worse.

### 3.5 Third construction: cadence separation

Calendar time, drift rate q, refresh every Δ at cost c_r with per-refresh noise v. Because the band is v-invariant and (after BGK) Δ-invariant in calendar units, the total cost rate separates:

  **cost(Δ, h) = √(Kqγ/3) [band, K only] + γP(Δ)/2 + c_r/Δ + κ'γqΔ [cadence, no K],**

P(Δ) = [−qΔ + √(q²Δ² + 4qΔv)]/2 the filtered variance at refresh, κ' ≈ 0.075–0.09 (the ¼ within-interval error growth net of the −κ start-of-period offset). Hence **Δ\* is independent of K and h\* is independent of c_r.** Small-Δ closed form with independent refresh noise: Δ\* ≈ (4c_r/(γ√(qv)))^{2/3} — elasticity 2/3 in refresh cost (measured local elasticity 0.68–0.72 across c_r ∈ [0.01, 10]), against ½ for ALP's costly-perfect-observation; and Δ\* ∝ v^{−1/3} (≈ −0.3 to −0.44 measured): *noisier refreshes should be run more often, not less.*

## 4. Attack-and-refine history

**Round 1 (construct).** Wrote the vector identity, the ellipsoid QVI, the whitening solver, and a vectorized simulator. First simulations put the ellipsoid's cost 10% *above* prediction with act rate 26% above the continuous formula. Two bookkeeping errors, not theory: (i) the flow cost was being charged on the pre-reset state (adding ≈ ½R\*² per action — overshoot charged at the boundary); dive 08's convention (charge after the decision) restored agreement; (ii) the act-rate prediction must use R\* (the BGK-shrunk discrete trigger *reproduces* the continuous barrier at R\*), giving mq/R\*² = 0.0913 vs 0.0908 observed.

**Round 2 (attack).** Six attacks:
- *Is the ellipsoid actually optimal when Q and Σ do not commute?* Built a 2-D average-cost DP (relative value iteration, FFT Gaussian kernel, off-grid = action value). Across three anisotropic configurations — isotropic drift/loss 1:4; drift correlation 0.7 with loss 1:4 (non-commuting); drift 10:1 with loss correlation 0.8 — the zero-tuned Riccati ellipsoid with the closed-form BGK shrink is within **0.00–0.01%** of the DP optimum (E3b). The DP region's second-moment axis ratios (1.73, 1.72, 2.29) match the *BGK-shrunk* ellipsoid (the shrink removes β√q from each semi-axis, so the ratio exceeds the continuous 1.60/1.66/2.15) and its orientation (−171.5° vs −166.4°; 140.9° vs 143.5°). Attack fails.
- *Does shape matter economically?* This attack largely succeeds and reframes the headline. Best-scaled alternatives: the loss-ellipsoid (Alvarez–Lippi sphere in loss-whitened coordinates) loses only 0.4%; a raw sphere 4–6%; a per-channel box 5–8% with a common half-width, but **only ~1% when each channel gets dive 08's scalar band with its own (q_i, γ_i)** (E5). In m dimensions, isotropic, the box's best-case penalty grows slowly: 0.1, 1.2, 1.8, 2.3, 2.3% at m = 1, 2, 3, 5, 8 (E6). The problem is as flat in *shape* as dive 08 found it in *scale*. What is *not* flat: the Mahalanobis (drift-whitened) shape loses 5–17%; the χ²/CI trigger loses 31–107% in the isotropic sweep and 29% (v=1)/9% (v=10) in case B; not filtering costs 1.4×/3.7× at v = 1/10; the "posterior-width" Riccati (Q+V in place of Q) costs 5%/45%.
- *Per-channel costs break the full-reset assumption.* Confirmed and quantified: 2.0, 6.3, 12.5, 7.7% savings for (K₀, c) = (20, 3.3), (10, 6.7), (0, 10), (15, 15) (E4). Led to §3.4.
- *The predicted cost is off.* The continuous g overstates the discrete-time cost by a term linear in tr(ΣQ) (E11); §3.2's κ formula is the refinement; clean-room measured the same 0.63 ratio to ¼ independently.
- *Cadence law under overlap-correlated refresh noise.* This attack succeeds decisively. With ρ(Δ) = 1 − Δ/W (W = 104 weeks; dive 07's ω) and a correctly specified augmented filter, the floor barely moves with Δ (4.38–4.41 at K=30, q=1/13 per week, v=10) because more refreshes of the *same window information* do not average the error; **Δ\* jumps from weekly (independent noise) to 8–13 weeks (quarterly)** for every refresh cost tried, and a white filter under correlated noise adds 12–25% (E10). The 2/3 law is the *independent-noise upper bound* on the value of refreshing faster.
- *The CI rule ties again.* In the MMM transport (v/q ≈ 26) the χ² trigger ties the ellipsoid, exactly as dive 08's CI rule tied at v=10 — coincidence of scales, as the E5/E6 sweeps show (2× worse at v=1, 31–107% worse in the isotropic sweep).

**Round 3 (refine).** Residual rule with closed-form r_c and the tiered heuristic (≤0.65% from DP); κ and κ' constants; cadence formula generalized to the floor-with-correlation form Δ\* = argmin[c_r/Δ + γP_zz(Δ; q, v, ρ(Δ))/2]; the 4-channel Hill transport (E8). **Round 4 (consolidation):** parameter-error sweep (K or Q wrong by 2× either way costs 3.4–3.8%; assuming isotropic drift 0.5%, isotropic loss 1.9%, both 2.7% — E9b), t₃ refresh noise +7.9% with argmin scale 0.9 (E9c), K→0 collapse of the ellipsoid (semi-axes 4.95/2.98 → 1.57/0.94 — E9d), v-invariance of the argmin at v = 0.1 and 100 in the anisotropic case (E9d), seed sd/mean 0.35% over 12 seeds (E9e). No material revision → stop. Clean-room verification (below) forced one clarification (the cadence formula must carry the O(γqΔ) term explicitly; it does) and softened r_c to "≈".

## 5. Numerical evidence (16 experiments; SEs from seed replication; DP results grid-converged at d = 0.2/0.1/0.05)

**Law verification.** E2: isotropic m = 2, argmin of the scale sweep at 1.00 for v ∈ {1, 10, 100} (noise sd up to 2.1× the band radius); act rate 0.0908 vs 0.0913. E3a: DP radius 4.05 vs law 4.10 (R\* − β√q). E3b (anisotropic):

| configuration | DP optimum | Riccati ellipsoid + BGK (zero-tuned) | best loss-ellipsoid | best drift-Mahalanobis | best raw sphere | best box |
|---|---|---|---|---|---|---|
| A: Q = I, Σ = diag(1,4) | 7.4496 | 7.4503 (+0.01%) | +0.37% | +5.2% | +5.2% | +6.4% |
| B: Q corr 0.7, Σ = diag(1,4) | 7.0383 | 7.0391 (+0.01%) | +0.46% | +11.0% | +4.5% | +5.0% |
| C: Q = diag(1, 0.1), Σ corr 0.8 | 3.3034 | 3.3034 (+0.00%) | +0.38% | +17.0% | +6.3% | +8.4% |

**Policy duel under estimation noise (case B, K = 30, ± SE over 6 seeds):**

| policy | v = 1 | v = 10 |
|---|---|---|
| **ellipsoid + BGK on filtered û (protocol)** | **8.444 ± 0.005** | **13.212 ± 0.006** |
| box of per-channel dive-08 bands (own q_i, γ_i) | 8.532 ± 0.004 | 13.308 ± 0.009 |
| Riccati with Q+V ("posterior width" band) | 8.904 ± 0.005 | 19.159 ± 0.020 |
| CI rule: ûᵀP⁻¹û > χ²₂(0.95) | 10.881 ± 0.004 | 14.370 ± 0.016 |
| ellipsoid on raw y (no filter) | 11.736 ± 0.008 | 48.521 ± 0.012 |
| act-always | 31.406 | 36.189 |

**Dimension sweep (E6, isotropic, v = 1):** sphere law 3.31/5.75/7.71/10.89/14.66 at m = 1/2/3/5/8; best box +0.1/+1.2/+1.8/+2.3/+2.3%; box at the scalar-law width +0.0/+1.3/+2.2/+3.5/+4.7%; χ² CI +107/+74/+59/+44/+31%.

**Tiered rule (E4–E4c):** savings vs full-reset-only 2.0/6.3/12.5/7.7%; DP threshold sharp (pair max 2.12 = full min 2.12); tiered heuristic +0.07/+0.41/+0.65/+0.38% vs DP.

**Cadence (E7, E10):** band argmin at h_d(Δ) for all Δ ∈ [0.25, 4]; Δ argmin identical for K = 30 and 120 at every c_r; formula vs simulation within 0.5% at Δ ≤ 2 (5.588 vs 5.611 ± 0.01); overlap-correlated noise moves Δ\* from 1 to 13 weeks.

**Nonlinear MMM transport (E8).** Four concave-Hill channels, budget 1, log-RW effectiveness drift η = 0.02, posterior noise τ = 0.10, per-event cost 0.02 (Γ, Q, V calibrated numerically at the operating point; v/q ≈ 26, gains 0.17–0.19; ellipsoid semi-axes 3.0–4.8% of budget). True regret per period, 8 paired seeds × 4000 periods: **protocol 0.00299 ± 0.00056**; sphere 0.97×, box 1.04×, CI 0.95× (all within ±1 SE of the protocol, paired); Q+V "naive" 2.96×; myopic 4.1×; unfiltered 5.6×; act-always 7.0×. Oracle scale sweep: paired differences at 0.8/1.0/1.25 are −0.00016 ± 0.00016 / 0 / +0.00065 ± 0.00027 — the closed form sits in the flat basin.

**Independent verification (clean-room subagent, given claims only).** Claim 1: sympy confirms the QVI algebra exactly (residual of ½tr(Q∇²W) + ½uᵀΣu − g is the quadratic form of Σ − K(4M tr(QM) + 8MQM)); own DP for case B: g_DP = 7.0384/7.0390/7.0389 at d = 0.2/0.1/0.05, closed-form policy gap 0.011–0.013% (**confirmed**), plain Φ ≥ 1 without the BGK term 6.7% worse; offset ratio to −¼tr(ΣQ) = 0.63 in both 1-D and 2-D (**matches κ**). Claim 2: own pairwise DP: saving 12.58%, threshold 2.1221/2.1224 (**structure confirmed**), r_c formula 5.5% low (**"≈"**). Claim 3: Δ\* = 1 for both K (**separation confirmed**); the agent flagged that a formula without the O(γqΔ) term picks Δ = 2 — reconciled: the term is in §3.5 (κ' ≈ 0.075–0.09) and the agent's simulated costs (5.606/5.643/6.153 at Δ = 1/2/4) match ours (5.611/5.645/6.113).

**Negative/power controls.** K → 0 collapses the region (E9d); the DP shape instrument separates axis ratio 1.73 from 1.00 at grid resolution; the BGK term is detected at 6.7% (clean-room); the CI-vs-protocol gap at v = 1 is 380 SE.

## 6. Limitations and failure modes

- **Interior optimum.** The tangent-space treatment ignores nonnegativity and per-channel caps; near a bound the region is truncated and the reset target is no longer "0."
- **Curvature drifts.** Γ, Q, V were calibrated once at the operating point; in the MMM transport the true Hessian drifts with β. A re-calibrated (adaptive-M) version was not tested; the flat basin (±25% scale costs ≤ 0.0007) is the mitigation.
- **Shape gains are small.** The ellipsoid is exactly optimal, but a per-channel box with dive 08's scalar bands is within ~1% (m=2) to ~5% (m=8). The decisive content is the *filter*, the *scale* (KQ product, (m+2)^{1/4} growth), and the *tiering* — not the ellipse. The χ² CI rule can tie by coincidence at v/q ≈ 25–30 and is 2× worse elsewhere.
- **Tiering was solved for n = 3.** For general n the action set is all subsets; the residual rule suggests a greedy "add channels until the marginal W-reduction < c," untested.
- **Cadence law needs the noise correlation structure.** Under dive 07's ω-correlation the 2/3 law is wrong by an order of magnitude in Δ\*; real MMMs sit between the independent and ω-correlated cases (dive 07 measured ρ ≈ 0.5 ≪ ω for nonlinear fits because of information-matrix rotation — which, ironically, makes frequent refreshes *more* valuable than the overlap arithmetic suggests).
- **Ergodic criterion.** Discounting changes the region (the discounted problem is not exactly quadratic); at business discount rates over quarterly steps the difference is second order (Alvarez–Lippi: ∂ȳ/∂r → 0).
- **Gaussian everything**, as in dive 08; t₃ noise costs +8%.

**Devil's advocate.** *Against the ellipsoid law:* it is Altarovici et al.'s Riccati equation with a change of notation; the "exactness" is only the observation that the LQ ergodic problem has no wealth effects. Answer: conceded on the equation; the contributions are the exactness statement, the whitening closed form, the budget-tangent-space application, the composite discrete/partial-observation policy verified against DP at 0.01%, and the honest finding that its economic edge over boxes is small. *Against the DP as ground truth:* it is a grid DP with an off-grid convention; three grid resolutions and an independent implementation agree to 0.01%. *Against the tiered rule:* n = 3 only; r_c is 5% off; the value function used to derive it is the full-reset one. Conceded — the structure (sharp residual threshold) is the finding, the constant is approximate. *Against cadence separation:* it assumes v does not depend on Δ; in practice a longer Δ means more new data per refresh and a slightly smaller v — that only strengthens the quarterly conclusion under correlated noise. *Against the MMM transport:* one configuration, paired SEs still ±20%, and the CI rule tied. Conceded: transport is partially verified; what it shows robustly is the 3–7× losses of the wrong statistics (Q+V width, myopic, unfiltered, act-always), not the fine ranking among filtered band shapes. *Against the whole program:* an org will not register a matrix. Answer: it registers one scalar statistic per refresh (Φ + BGK term vs 1) and one PoR; the matrix lives in the appendix with the posterior.

## 7. Protocol delta (design artifact, extends dive 08 §7)

Register additionally: (vi) the tangent basis and the curvature matrix Γ (model Hessian at the incumbent allocation), the drift covariance Q̂ and the derived matrix M (whitening solution) — or, if the org prefers per-channel bands, the per-channel scalar bands with their documented ≤5% penalty; (vii) the per-line-item change cost c and the derived residual threshold r_c, so that "move money between Search and Social only" is a pre-registered, priced option rather than an ad-hoc compromise; (viii) the refresh cadence Δ chosen from argmin[c_r/Δ + γP_zz(Δ; q̂, v̂, ρ̂(Δ))/2] with ρ̂ the measured refresh-error correlation (dive 07's anchored-estimator decomposition) — under overlapping-window refreshes expect quarterly, not monthly, and say so in the registration. Each refresh reports: expected gain ½ûᵀΣû, PoR, the trigger statistic vs 1, and ACT-FULL / ACT-PAIR(i,j) / HOLD.

## 8. Next steps for a future session

1. **General-n tiering:** derive/test the greedy subset rule (add channels while marginal W-reduction > c) against a lattice DP for n = 4–5; connect to Bonomo et al.'s two-constant policy in the continuum limit.
2. **Adaptive M:** re-calibrate (Γ, Q, V) from the fitted model each refresh in the Hill transport; test whether the flat basin survives large β drift, and whether M-churn itself becomes a stability problem (dive 07).
3. **Cadence with measured correlation (BL22 link):** estimate ρ̂(Δ) and q̂ from refresh history with the anchored-estimator decomposition; feed the floor-with-correlation cadence rule; test end-to-end on dive 07's nonlinear MMM where ρ ≈ 0.5 ≪ ω.
4. **Prove κ = ¼ − β²/4:** second-order discrete-monitoring expansion of the ergodic cost (Howison–Steinberg matched asymptotics) — small but clean.
5. **Trust as a stock (BL24)** now has a vector state: does tiering (smaller, cheaper moves) slow trust erosion relative to full resets?

## Sources

- Internal: dive 08 (scalar law, v-invariance, PoR, protocol), dive 07 (ρ = ω refresh-error correlation, rotation), dive 05 (cadence law), dives 03/04 (decision-loss machinery); catalogs `../03_mmm_adoption_barriers.md` §B5/§M9, `../02_open_questions.md`.
- [Alvarez & Lippi 2014, Econometrica](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA10662) ([WP](https://www.eief.it/files/2013/02/wp-02-price-setting-with-menu-cost-for-multi-product-firms.pdf)) · [Altarovici, Muhle-Karbe & Soner 2015](https://arxiv.org/abs/1306.2802) (ellipsoid Riccati 4M Tr(AM)+8MAM = Σ) · Atkinson & Wilmott 1995, Math. Finance · [Cai, Rosenbaum & Tankov 2017](https://arxiv.org/abs/1510.04295) · [Baccarin 2009](https://www.researchgate.net/publication/222399780) · [Bonomo, Carvalho, Kryvtsov, Ribon & Rigato 2023](https://www.econ.puc-rio.br/api/uploads/adm/trabalhos/files/td687.pdf) · [Bhattarai & Schoenle 2014](https://ideas.repec.org/a/eee/moneco/v66y2014icp178-192.html) · Midrigan 2011, Econometrica
- [Gobet & Menozzi 2010](https://arxiv.org/abs/0706.4042) (multidimensional discrete-monitoring shift) · Broadie–Glasserman–Kou 1997 · Howison & Steinberg 2007
- [Alvarez, Lippi & Paciello 2011, QJE](https://www.nber.org/system/files/working_papers/w15852/w15852.pdf) (observation + menu costs; τ\* ∝ cost^{1/2}) · Reis 2006, RES · [de Silva, Hyndman & Snyder 2010](https://robjhyndman.com/papers/vists.pdf) (vector innovations form); Harvey 1989; Durbin & Koopman 2012
- Dive 08's decision-communication sources (Joslyn & LeClerc 2012; Burgeno & Joslyn 2020/2023; Dietvorst et al. 2015/2018) carry over.

*Code:* `code/09_decision_deadband_ext.py` (library + drivers e1–e10; `python 09_decision_deadband_ext.py quick` reproduces the case-B DP gap, the pairwise saving, and the duel headline in ~30 s).
