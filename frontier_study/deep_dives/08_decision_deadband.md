# Dive 08 — Decision-framed uncertainty outputs: the filtered deadband law (B2, absorbs BL20)

**Item:** B2 — M9(comm)/B5/H8, decision-framed uncertainty outputs. **Status of headline results:** the deadband law and its v-invariance are **verified** (DP + simulation + independent clean-room re-derivation); the nonlinear-MMM transport is **partially verified** (one configuration, flat-basin agreement); the organizational protocol template is **design/speculation** grounded in the verified math.

> **Novelty correction (post-writeup arXiv audit, same day):** an adversarial prior-art search found the continuous-time core of the invariance result is **prior art**. Baley & Blanco (AEJ: Macro 2019) derive a Dixit quartic-root inaction band applied to a Kalman-filtered markup gap in a menu-cost model, with the estimate's volatility equal to the fundamental's (noise-independent) in steady state; they credit the constant-uncertainty case to the Appendix of Alvarez–Lippi–Paciello (QJE 2011) ("the information friction has no effects in steady state"). The identity Var(Δẑ)=q is essentially Muth (1960) (local-level model ⟹ optimal forecast is a random walk with the permanent-shock variance). BGK corrections to discretely monitored decision boundaries also exist (Howison et al., math/0511106; Gobet–Menozzi, 0706.4042). **What remains plausibly unclaimed (~75% confidence):** the discrete-periodic-observation composite formula (quartic + BGK on the filtered state, verified to 0.4%), the γP/2-floor + shrinkage cost decomposition, and the entire MMM application layer (PoR reporting, derived α\*, the protocol). Sections below should be read with §3.1–3.2 relabeled from "derived here" to "rediscovered; known in continuous time."

---

## 1. Problem statement

The adoption catalogs say the field's honest outputs — posterior intervals — lose meetings (B5: "precision theater"; M9: wide intervals read as "the model doesn't know"). B2 asks for the *interface layer*: outputs framed in decision units (probability-of-regret, expected loss of reallocation, answer-at-your-risk-tolerance) plus a decision protocol an org could pre-register. Dive 07 ended with exactly this hook (BL20): its 1σ reporting hysteresis cut reported flips 78% but had no optimality theory — "derive h\* from flip-cost vs staleness-cost."

Formal problem. An org operates allocation x_c. The true optimum x\*(θ_t) drifts; let z_k = x\*(θ_k) − x_c be the deviation at refresh k (scalar; the multichannel case is §6). Between refreshes z_{k+1} = z_k + w_k, w ~ N(0, q) (q = per-refresh variance of the drifting optimum). Each MMM refresh delivers a noisy estimate y_k = z_k + ε_k, ε ~ N(0, v) (v = posterior variance mapped to allocation units). Operating off-optimum costs flow loss (γ/2)z² per refresh period (γ = curvature of expected profit at the optimum). Reallocating costs a fixed K — execution, coordination, and the credibility cost of visibly changing the recommendation (Dietvorst et al. 2015: algorithms lose trust faster than humans when seen to err; Burgeno & Joslyn 2020: forecast inconsistency costs trust). Objective: minimize long-run average cost. Question: **what should the model report each refresh, and when should the org act?**

## 2. Prior state

Four literatures each hold one piece; the literature subagent confirmed the combination is unclaimed:

- **Menu-cost/hysteresis math** (Barro 1972; Dixit 1991; Stokey 2009): for a *perfectly observed* Brownian state, fixed cost + quadratic flow loss gives a deadband with the quartic-root law — with flow loss B·z², half-width (6σ²K/B)^{1/4}. No estimation noise anywhere.
- **Partially observed impulse control** (Mazziotto–Stettner–Szpirglas–Zabczyk 1988; Bensoussan 1992): existence theorems on the filter state only; no closed forms. Closest economics: Alvarez–Lippi–Paciello 2011 (costly *perfect* observation), Åström–Bernhardsson 2002 (event triggers, no estimation noise in the optimality proof).
- **Expected-loss decision rules in experimentation** (Stucchio/VWO 2015: ship when expected loss < "threshold of caring"; Feit & Berman 2019 Test & Roll): decision-framed, but static — no dynamics, no adjustment cost, and the thresholds are elicited, not derived.
- **Decision-framed communication works** (Joslyn & LeClerc 2012: probability formats improve incentivized decisions and buffer trust against forecast error; Burgeno & Joslyn 2023: probabilistic framing mitigates the inconsistency penalty). Current MMM tools don't do it: Recast ships intervals/simulations, Meridian ships point-optimal budgets + credible intervals, PyMC-Marketing ships CVaR *objectives* — none ships an act/hold statistic with a derived threshold. Pathak–Jeunen–Lambert 2026 build regret distributions but retrospectively, as an audit.

Internal: dive 03's machinery prices information in decision units; dive 07 supplies the refresh-noise facts (adjacent-refresh error correlation ρ ≈ 0.5; flip churn) and the unpriced hysteresis.

## 3. The construction

### 3.1 An exact identity: estimation noise never enters the band

Steady-state Kalman filter for the random walk + white noise pair: P solves P(P+q) = qv, gain λ = (P+q)/(P+q+v), filtered mean ẑ_k. The one-step increments of ẑ have variance λ²(P+q+v) = (P+q)²/(P+q+v). Substituting qv = P(P+q):

  q(P+q+v) = q(P+q) + P(P+q) = (P+q)²  ⟹  **Var(Δẑ) = q, exactly, for every v.**

The filtered optimum is a random walk with the *same* variance rate as the true optimum, no matter how noisy the refreshes. Estimation noise is entirely absorbed by shrinkage (λ < 1) and by a stationary residual z − ẑ with variance P = (−q + √(q² + 4qv))/2.

### 3.2 Separation, and the composite deadband law

Acting by ẑ (set x_c ← x_c + ẑ) subtracts a *known* quantity from z, so the filter is unaffected by the control; expected flow cost decomposes as (γ/2)E[z²|data] = (γ/2)(ẑ² + P), and P is policy-independent. The partially observed problem therefore reduces **exactly** to fully observed impulse control of the martingale ẑ — which by §3.1 is a random walk with rate q. (This exactness is special to the reset-by-ẑ action structure; general partial-observation impulse control has no separation theorem.)

Continuous-time renewal-reward on the reduced problem (BM rate q, absorb at ±h, reset to 0): E[cycle] = h²/q, E[∫(γ/2)z²dt] = γh⁴/(12q), so average cost g(h) = γh²/12 + Kq/h², minimized at h = (12Kq/γ)^{1/4}. Discrete refreshes overshoot the boundary; the Broadie–Glasserman–Kou correction for discretely monitored barriers (β = ζ(½)/√(2π) ≈ 0.5826) enters additively. The **composite filtered deadband law**:

  **h\* = (12Kq/γ)^{1/4} − 0.5826·√q,  applied to the filtered deviation ẑ — independent of posterior width v.**

Corollaries, all verified below: minimum average cost ≈ γh_c²/12 + Kq/h_c² + γP/2 with h_c = h\* + β√q (the only place v appears is the floor γP/2); act rate ≈ q/h_c²; cadence-invariance (in calendar units q/γ is refresh-rate-free, so only the β√q term shrinks with faster refreshing).

### 3.3 The decision-framed output layer

Three org-facing statistics per refresh, replacing intervals (all algebra, MC-checked):

1. **Expected gain of reallocating now:** Ĝ = (γ/2)ẑ². Act iff Ĝ ≥ G\* = (γ/2)h\*² (continuous limit: G\* = √(3γKq) — the demanded gain scales with the *square root* of the switching cost, not the cost itself).
2. **Probability-of-regret:** with z|data ~ N(ẑ, P), the gain from moving is γẑ(z − ẑ/2), so PoR = P(moving hurts) = **Φ(−|ẑ|/(2√P))**. Verified by MC: predicted 0.20711, observed 0.20725 (n = 2×10⁶).
3. **Derived risk tolerance:** the band maps to α\* = Φ(−h\*/(2√P)) — "act when PoR ≤ α\*" is *identical* to the deadband policy (PoR is monotone in |ẑ|). The tolerance is derived from (K, q, γ, v), not elicited. E.g. the E10 configuration gives α\* = 0.13: reallocate when there's ≤13% chance the move hurts.

A pre-registerable protocol template is in §7.

## 4. Attack-and-refine history

**Round 1 (construct).** First version used the Dixit constant 6 with loss (γ/2)z² — DP thresholds drifted 0.79→1.12 of "theory" across K. Two errors found and fixed: the constant is 12 for this loss normalization, and discrete monitoring shifts the band. After both corrections the DP/law ratio is 0.994–1.000 for K ≥ 30 (E1b), free-fit β = 0.599 ± 0.015 vs BGK 0.5826.

**Round 2 (attack).** Four attacks:
- *Correlated refresh noise* (dive 07's ρ ≈ 0.5 from overlapping windows) breaks the white-noise filter's martingale property: measured Var(Δẑ) = 0.82 ≠ q under misspecification. **Refinement:** the identity generalizes — for *any* correctly specified stationary linear-Gaussian observation structure, ẑ is a martingale and z − ẑ is stationary while Var(z) grows qk, forcing Var(Δẑ) = q. Verified with an augmented-state (z, AR(1) noise) filter: increment variance 1.000, argmin band unchanged at h\*, and average cost predicted by the floor formula to 0.3% (5.09 predicted, 5.085 ± 0.012 observed). Misspecified white filtering cost only +2.6% at ρ = 0.5, v = 10 — the law degrades gracefully.
- *Mean-reverting truth* (OU, φ < 1): near-unit-root drift (φ ≥ 0.98) leaves the argmin at h\*; strong reversion (φ = 0.9) widens it ~17% (free mean reversion adds option value to waiting) though acting still beats never-acting (2.08 vs 2.63). **Scope condition, not refutation.**
- *Cost structure:* if the reallocation cost is proportional (κ|move|) rather than fixed, the exponent must become 1/3. DP log-log slopes came out 0.295 (fixed) and 0.42 (proportional) vs "theory" 0.25/0.333 — apparent refutation, resolved in Round 3.
- *Heavy tails:* t₃ refresh noise (variance-matched) costs +8.9% with the linear filter; argmin band unmoved.

**Round 3 (refine).** The exponent "anomaly" is the composite law's own prediction: fitting log h vs log K with the additive −β√q term present yields effective slopes 0.292 and 0.40 over these ranges — matching 0.295/0.42. Proportional-cost *level* also matches: h = (6κq/γ)^{1/3} − β√q, ratio 0.996 at κ = 100. The end-to-end nonlinear MMM test (E10) then confirmed transport. **Round 4 (consolidation)** — PoR MC check, seed sweep (sd 0.5% of mean over 24 seeds), K→0 control — produced no material revision. Stopped per protocol (two consecutive stable rounds).

## 5. Numerical evidence (15 experiments; SEs from seed replication)

**Law verification.** E1/E1b: DP thresholds match the composite law within 0.4% for K ≥ 30 across q ∈ {0.25, 1} (worst case K=1: −3.6%; asymptotic regime is K ≫ γq/6). E2: Var(Δẑ) = q across a 3×3 (q,v) grid, MC within 0.5%. E4: empirically optimal band stays at 3.6–3.8 (h\* = 3.77) while v spans 0→100 — i.e., noise σ up to 2.7× the band itself — and the cost decomposition g₀ + γP/2 matches observed 3.004/3.31/4.35/7.78 vs predicted 3.00/3.31/4.35/7.76. Predicted act rate q/h_c² = 0.0526 vs observed 0.0518–0.0521.

**Policy duel (K=30, q=1, γ=1).** Long-run average cost (± SE over 8 seeds):

| policy | v=1 | v=10 |
|---|---|---|
| **filtered band h\* (protocol)** | **3.307 ± 0.006** | **4.354 ± 0.006** |
| filtered band, naive width from (q+v) | 3.414 ± 0.005 | 5.494 ± 0.010 |
| raw band on y (no filter) | 3.609 ± 0.005 | 11.388 ± 0.011 |
| act-always (chase the model) | 30.31 | 31.35 |
| myopic "expected loss > K" (Stucchio-style) | 6.338 ± 0.017 | 7.387 ± 0.020 |
| CI rule: act when |ẑ| > 1.96√P | 6.864 ± 0.006 | 4.468 ± 0.010 |

Readings: filtering is most of the value under noise (raw triggering is 2.6× worse at v=10); the *statistical-significance* heuristic is right only by coincidence — its width scales with estimation noise, which has nothing to do with the drift/cost economics (near-optimal at v=10 where 1.96√P ≈ h\* by accident, 2.1× worse at v=1); the myopic expected-loss rule has the wrong exponent (√K vs K^{1/4}) and waits ~2× too long here.

**Flatness (E5).** cost(mh\*)/cost(h\*): 1.64, 1.17, 1.00, 1.26, 1.86 at m = 0.5, 0.7, 1, 1.5, 2. Flat enough that a pre-registered approximate band loses little; wrong-by-2× is real money.

**Power/negative controls (E6).** The instrument discriminates the true law from the (q+v)-scaled alternative at 141σ; K→0 collapses the DP band to ~0.3 (≈ step scale) as it must.

**Nonlinear MMM transport (E10).** Two-channel Hill revenue, log-RW effectiveness drift (η=0.02), posterior noise τ=0.10, reallocation cost 0.02; (γ, q_x, v_x) calibrated numerically at the operating point, giving h\* = 0.064 (3.2% of budget) and λ = 0.17 (heavy shrinkage: v_x/q_x = 30). True-regret flow per refresh: protocol 0.0032 ± 0.0008; act-always 0.0209 ± 0.0002 (6.5× worse); myopic 0.0059 ± 0.0010; oracle band sweep has its (flat) argmin at 0.75h\* (0.0030 ± 0.0003), statistically indistinguishable from h\*. The closed form lands in the basin.

**Independent verification.** A clean-room subagent, given only the claims, re-derived the identity algebraically (exact; residual 10⁻¹⁴), re-implemented DP + CRN simulation from scratch: thresholds 3.760/7.160 vs formula 3.773/7.163; argmin at v=1 and v=100 both in 3.7–3.9; floors +0.3116 ± 0.0067 and +4.7573 ± 0.0233 vs predicted +0.309/+4.756. All three claims CONFIRMED; its one caveat is §6's first limitation.

## 6. Limitations and failure modes

- **Policy class.** v-invariance is proven for band-on-ẑ policies; no proof that a non-band policy can't beat it in the partially observed problem (symmetry + convexity make this very plausible; the reduction argument covers it *given* reset-by-ẑ actions).
- **Scalar deviation.** Real reallocation is a vector; Alvarez–Lippi 2014 suggests the radial (χ²) analogue but it is untested here (backlog BL21). E10's budget constraint made the 2-channel case scalar.
- **Parameters must be estimated.** q (optimum drift rate) is confounded with refresh noise in observed refresh-to-refresh jitter — dive 07's rotation wobble would inflate a naive q̂, widening the band spuriously. γ comes from the fitted model's curvature (misspecification risk, links BL10); K is an organizational quantity nobody currently measures (BL22). The fourth-root blunts all of this — a 2× error in K/q/γ moves h\* by 19% and costs ≲5% (E5) — but it is mitigation, not immunity.
- **Strong mean reversion** of the optimum (φ ≲ 0.9 per refresh) widens the true band ~17%+; the law is an approximation for the slow-drift regime it was built for.
- **Misspecified noise structure** costs little at ρ = 0.5 (+2.6%) but was not swept to extreme ρ; heavy tails cost +9% with the linear filter (a robust filter should recover part — BL23).
- **The credibility cost K is modeled as fixed and known.** If churny recommendations *themselves* degrade K (trust is a stock, not a flow price), the problem becomes a different control problem entirely.

**Devil's advocate.** *Against the law:* it is three asymptotic approximations stacked (small-cost band, BGK shift, LQ loss); the free-fit β = 0.599 ≠ 0.5826 shows residual structure; everything Gaussian. Answer: the stack is verified end-to-end at 0.4% by two independent implementations, and the t₃/OU/AR(1) attacks bound the damage. *Against v-invariance:* it sounds like "uncertainty doesn't matter," which would be misread; v matters enormously — through shrinkage (λ = 0.17 in E10) and the floor γP/2 — just not through the band. The counterintuitive content is precisely the deliverable. *Against the duel:* heuristic rankings depend on parametrization; the CI rule can tie (and did at v=10). Answer: the point is structural (wrong scaling variable), not the particular gap. *Against E10:* one configuration, drift/noise calibrated at the initial point, flat basin means many bands "work." Conceded — transport is *partially* verified. *Against the whole framing:* maybe orgs reject deadbands ("why do we pay for a model we mostly don't act on?"). That is B5's psychology again — but now the *inaction* has a derived price tag (option value of waiting), which is a different meeting.

## 7. The pre-registerable protocol (design artifact)

At engagement start, register: (i) loss curvature source (model Hessian at incumbent allocation); (ii) q̂ estimation procedure (refresh-history variance decomposition, estimation noise removed); (iii) K elicitation (execution cost + agreed credibility charge); (iv) the derived h\*, G\* = (γ/2)h\*², α\* = Φ(−h\*/(2√P)); (v) the filter (correct noise correlation structure per dive 07). Each refresh, the model reports exactly three lines: **expected annualized gain from moving to the filtered target; probability the move hurts (PoR); ACT/HOLD vs the registered threshold.** Intervals go to an appendix. HOLD is reported as "the expected gain does not yet cover the registered cost of switching" — inaction becomes a priced decision, not an admission of ignorance. This kills the B5 dynamic in the only way consistent with honesty: the wide posterior never reaches the meeting as a naked interval; it arrives already multiplied by the decision it implies. Dive 07's 1σ reporting hysteresis is the degenerate (K→reporting-only) case; BL20 is resolved by G\*.

## 8. Next steps for a future session

1. **Multichannel radial band (BL21):** deadband on the Mahalanobis norm ‖ẑ‖_Γ with the Alvarez–Lippi √n scaling; does the χ² analogue of the composite law hold with budget coupling?
2. **Estimating q under rotation wobble (BL22, links BL18):** variance-decompose refresh jitter into drift + estimation noise using dive 07's anchored-estimator trick; test how q̂ bias propagates to h\* and realized cost.
3. **Trust as a stock:** replace fixed K with a state variable degraded by acting-and-regretting (Dietvorst dynamics); does a deadband survive, and how much wider?
4. **Sequential refinement:** the band assumes refresh cadence is exogenous; joint (Δ, h) optimization couples this dive to dive 05's cadence law τ\* = √(c/(r(1/κ−½))) — one unified inaction theory.
5. **Field-facing artifact:** turn §7 into a one-page worked template with the E10 numbers as the running example.

## Sources

- Internal: dives 03/04 (decision-loss machinery), 05 (cadence law), 07 (refresh noise structure, ρ≈0.5, hysteresis hook); catalogs `../03_mmm_adoption_barriers.md` §B5/§M9, `../02_open_questions.md`.
- Prior art found in post-writeup audit: [Baley & Blanco 2019, AEJ:Macro](http://www.isaacbaley.com/uploads/6/7/3/5/6735245/firmuncertainty_baleyblanco.pdf) (quartic band on filtered gap) · Muth 1960 (filtered-RW variance identity) · [Howison et al., math/0511106](https://arxiv.org/abs/math/0511106) / [Gobet–Menozzi 0706.4042](https://arxiv.org/abs/0706.4042) (BGK corrections to decision boundaries)
- Deadband math: [Dixit 1991](https://academic.oup.com/restud/article-abstract/58/1/141/1518971) · Stokey, *The Economics of Inaction* (2009) · [Barro 1972](https://academic.oup.com/restud/article-abstract/39/1/17/1518557) · [Miller & Orr 1966](https://breakingdownfinance.com/finance-topics/risk-management/miller-orr-model/) · [Harrison–Sellke–Taylor 1983](https://doi.org/10.1137/0315007) · [Alvarez & Lippi 2014] · [Altarovici–Muhle-Karbe–Soner 2015](https://link.springer.com/content/pdf/10.1007/s00780-015-0261-3.pdf) · [Guasoni & Muhle-Karbe primer](https://arxiv.org/abs/1207.7330) · Broadie–Glasserman–Kou 1997 (discrete barrier correction)
- Partial observation: [Mazziotto et al. 1988](https://www.researchgate.net/publication/227322485_On_Impulse_Control_with_Partial_Observation) · Bensoussan 1992 · [Alvarez–Lippi–Paciello 2011](https://academic.oup.com/qje/article-abstract/126/4/1909/1924043) · Åström & Bernhardsson 2002
- Decision rules & communication: [VWO SmartStats (Stucchio)](https://vwo.com/downloads/VWO_SmartStats_technical_whitepaper.pdf) · [Robinson, peeking critique](http://varianceexplained.org/r/bayesian-ab-testing/) · [Feit & Berman, Test & Roll](https://pubsonline.informs.org/doi/10.1287/mksc.2019.1194) · [Joslyn & LeClerc 2012](https://www.apa.org/pubs/journals/features/xap-18-1-126.pdf) · [Burgeno & Joslyn 2020](https://journals.ametsoc.org/view/journals/wcas/12/4/WCAS-D-19-0074.1.xml) / [2023](https://journals.ametsoc.org/view/journals/wcas/15/3/WCAS-D-22-0064.1.xml) · [Dietvorst et al. 2015](https://marketing.wharton.upenn.edu/wp-content/uploads/2016/10/Dietvorst-Simmons-Massey-2014.pdf) / [2018](https://faculty.wharton.upenn.edu/wp-content/uploads/2016/08/Dietvorst-Simmons-Massey-2018.pdf) · [Pathak–Jeunen–Lambert 2026](https://arxiv.org/abs/2604.25977) · [PyMC-Marketing risk assessment](https://www.pymc-marketing.io/en/latest/notebooks/mmm/mmm_allocation_assessment.html) · [Meridian optimization outputs](https://developers.google.com/meridian/docs/user-guide/generate-optimization-results-output) · [Recast on uncertainty](https://getrecast.com/how-to-manage-uncertainty-in-media-mix-modeling/)

*Code:* `code/08_decision_deadband.py` (core library + experiment drivers; `python 08_decision_deadband.py quick` smoke-tests the headline numbers).
