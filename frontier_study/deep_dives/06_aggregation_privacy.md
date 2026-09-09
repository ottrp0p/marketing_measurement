# Dive 06 — MMM under aggregation and privacy noise (M9)

**Queue item:** A6 (M9). **Code:** `code/06_aggregation_privacy.py` (entry points: `drift_laws`, `drift_noise`, `dp`, `frontier [b]`). **Status of headline claims:** verified by simulation; drift laws additionally clean-room verified by an independent agent.

---

## 1. Problem statement

Post-cookie MMM consumes aggregated and privacy-noised data. The catalog (M9, `../02_open_questions.md`) poses two halves: (a) covariates aggregate over heterogeneous units — ecological bias interacting with nonlinear Hill transforms creates Jensen-gap bias; (b) outcomes carry calibrated DP noise with known mechanism, and nobody has priced ε in posterior width.

Formal setup. Geos g=1..G, weeks t=1..T. Micro model
y_gt = B/G + (β/G)·H(a_gt; K, s) + ε_gt, ε_gt ~ N(0, σ_geo²),
with a_gt the geometric adstock (rate α) of geo spend x_gt, and H(a) = a^s/(a^s + K^s). Geo spend shares w_g (Σw=1) with cross-geo coefficient of variation c; stable shares mean x_gt = w_g·X_t (adstock commutes with static shares, so a_gt = w_g·A_t exactly). The national modeler sees X_t and Y_t = Σ_g y_gt and fits the same functional form to the aggregate. A DP curator may release outcomes with additive Laplace(b) noise per released cell, b = Δ/ε known (parallel composition across geos — disjoint users — so per-cell scale b is the same whether cells are national-week or geo-week).

Defaults: G=20, T=104, α=0.5, K=1, s=2, β=100, B=500, σ_geo=1 (national noise σ_nat=σ_geo√G≈4.47), identifying block-pulse spend design per dive 01, lognormal shares. Estimands: the Hill shape (K, s) — the transport currency of dive 02 — and marginal ROAS (mROAS) at operating point 1× and counterfactuals 0.25×–4×.

## 2. Prior state

From the literature subagent: van Garderen–Lee–Pesaran (J. Econometrics 2000) show no aggregate model in aggregate variables reproduces an aggregated nonlinear micro model; the wedge depends on the cross-sectional regressor distribution. Stoker (1984) / Blundell–Stoker give the negative identification result: recovering micro parameters from aggregates requires distributional movement ("completeness") — with frozen shares the aggregate identifies only the mixture, not the micro curve. Google's GBHMMM (Sun et al. 2017) and Meridian docs make the operational claim that geo variation is what identifies Hill saturation. On DP: Evans–King (APSR/PA 2023, ext. 2025) treat known DP noise as classical measurement error; Gong (2022) shows MC-EM/ABC gives exact inference under a known mechanism; Karwa–Vadhan (2018) bound the price of privacy for CIs; Delaney et al. (PoPETS 2024) formalize the discrete-Laplace noise of the Attribution Reporting API. Two gaps the subagent confirmed as unclaimed: (1) closed-form Hill-parameter bias under geo-share heterogeneity; (2) noise-aware MMM likelihoods for DP ad aggregates. This dive fills (1), and shows (2) is worth less than the DP-inference literature's machinery suggests.

## 3. Novel constructions

### 3.1 Closed-form Hill drift laws under share dispersion

Write u = a/K, so logit H = s·log u exactly — the Hill family is the *linear* family in (logit, log) space, which makes the aggregation distortion analyzable. Expanding the mixture H̄(u) = E_w[H(wu)] to second order in share dispersion and pushing through the logit gives, with ψ(u) = s·((s−1)−(s+1)u^s)/(1+u^s):

- **Local slope law:** s_eff(u) ≡ d logit H̄/d log u = s − c²·s³·H(u)(1−H(u)) + O(c⁴). (The derivative dψ/dv = −2s²/(1+v)², v=u^s, collapses the mess to this one-liner.) Flattening is maximal at the half-saturation point: Δs = −c²s³/4.
- **Half-saturation drift law:** the fitted half-sat point drifts up by Δlog K̃ = +c²/2 + O(c⁴), *independent of s* at leading order.
- β is unbiased at leading order (confirmed: β̂ ∈ [99.0, 100.5] across the entire c-sweep).

Verification (E1b): pointwise, max |s_eff − prediction| ≤ 5–10% of the drop scale at c=0.3 (i.e., an O(c⁴) remainder), and log u* matches c²/2 to three decimals at c≤0.2. Projected-fit version (E2, noiseless T=520 pseudo-true fits): measured K̃ = 22.48±2.08 at c=0.5 vs law's 22.62; s̃ = 1.75±0.07 vs the design-averaged law. Clean-room agent independently re-derived both and found a strengthening: for symmetric two-point shares the K-drift law is **exact at all orders** (log u* = −½log(1−c²), s-independent), while the slope law degrades when c²s² ≳ 1 (e.g., s=8, c=0.5 predicts s_eff = −24 vs actual 0.64) — a validity boundary, safely outside marketing-realistic grids (s ≤ 4, c ≤ 0.3ish at DMA level... c can reach 0.8+ across US DMAs; treat the laws as first-order guides there).

### 3.2 The benign-projection result: stable dispersion is not the problem

The alarm implicit in M9 — "Jensen gap biases β" — is **refuted** in its decision-relevant form. With stable shares, the aggregate curve is a *fixed* mixture, and the LS projection onto a single Hill matches its level and slope on-support:

- mROAS bias at operating point: −0.0% to −0.1% for c up to 0.8 (E2, pseudo-true), and it stays ≤0.4% with heterogeneous geo half-sats (K_g lognormal, cv 0.5) and even with **targeted** shares w_g ∝ 1/K_g (spend concentrated in responsive geos) (E4).
- The bias lives elsewhere: deep counterfactuals (mROAS at 4× spend: +10.3%±3.0 at c=0.3, +28.3%±7.7 at c=0.8; at 0.25×: −3 to −7%) and the *parameters themselves* — K̃ +13% and s̃ −0.25 at c=0.5 — which is what poisons dive-02-style experiment transport and "headroom to saturation" narratives, both of which consume (K, s) directly.

So for the modal national decision (small reallocation at the operating point), stable-share ecological bias is a non-problem; for shape transport and budget-doubling/halving counterfactuals it is real and now has a formula.

### 3.3 The reframe: aggregation bias is a share-*instability* problem, and spend-only calibration fixes it

When shares drift over time (AR(1) log-share wiggle, sd 0.15–0.30 — think geo mix shifting across two years), the national curve is a *moving* mixture no single Hill can track, and bias invades the operating point itself:

- Pseudo-true (E5): naive mROAS@1x bias +5.3%±4.5 (drift 0.15), +10.1%±8.4 (drift 0.30); @2x errors up to ±65%.
- Realistic noise, T=104, 24 seeds (E5n): naive **+10.9%±3.0** (RMSE 18.4%); fresh-seed replication +10.0%±3.7 (RMSE 20.6%); with equal mean shares (c=0, drift 0.30) **+20.9%±2.7** — the drift, not the static dispersion, is the mechanism (R4 control).

**Construction — the share-calibrated mixture estimator:** fit Y_t = B + (β/G)·Σ_g H(adstock(x_gt); K, s) using only the *national* outcome plus the advertiser's own geo *spend* (always observable — it's the advertiser's data; no geo outcomes, no user-level data, no privacy exposure). Same parameter count, correct estimand. Results: bias +0.3%±2.2 / −2.9%±3.7 across the drift scenarios, RMSE 10.7–18.5% vs naive 18.4–24.9% (×1.7–2.3 better where drift is present), and it is *exact* in the pseudo-true limit (0.0% at both drift levels). Negative control (E8a): at c=0 the mixture and naive estimators coincide to four decimals, as they must.

### 3.4 Pricing ε: width law, a bounded prize for noise-awareness, and the granularity frontier

**Width law (E6c):** with known Laplace(b) DP noise on outcomes, the sampling/posterior width of linear-in-β fits scales as √(σ² + 2b²): fixed-design ratios 1.70 / 3.00 / 5.82 vs predicted 1.73 / 3.00 / 5.74 at b/σ_nat = 1/2/4 (400 reps; clean-room confirmed at 2.2% MC precision). Posterior-width inflation = √(1 + 2b²/σ²) — that is the whole price of ε for a national release: with weekly national conversions at natural noise sd ~100 and Δ=1, even ε=0.1/week inflates widths by ×1.01. **National-level DP is effectively free.**

**Bounded prize for exact noise-aware likelihoods (E6b):** the asymptotic efficiency of variance-inflated Gaussian LS relative to the exact convolved-density MLE is 100%/98.9%/90.5%/75.5%/63.7%/57.0% at b/σ = 0.25/0.5/1/2/4/8, with limit exactly 50% (clean-room: 51.9% at b/σ=30, 50.6% at 100). So the Gong/Evans–King exact-inference machinery buys *at most* a factor 2 in information over the two-line fix "inflate σ² by 2b²" — worth having at b ≫ σ, ignorable otherwise. This is, to our knowledge, the first such bound stated for the DP-MMM setting (derived here; the 50% limit follows from the convolution tending to Laplace, whose location MLE has info 1/b² vs LS's 1/2b²).

**Granularity frontier (E7) — the dive's synthesis.** The curator chooses release granularity: national-week cells (little DP noise on the aggregate, but ecological limits) vs geo-week cells (G× more cells ⇒ full identification, but per-cell signal β/G against the same per-cell noise b). RMSE of mROAS@1x, 6 seeds/cell, c=0.5:

| b (per-cell Laplace) | national naive | national mixture | geo release |
|---|---|---|---|
| 0 | 12.5% | 12.9% | **3.6%** |
| 1 | 13.2% | 13.7% | **6.4%** |
| 3 | 15.5% | 16.1% | **13.5%** |
| 10 | 27.4% | 27.4% | **38.2%** |
| 30 | 53.9% | **50.9%** | 129.6% |

(RMSE@2x and RMSE(s) show the same crossover; geo release recovers s ≈ 1.94–1.97 against a prior centered at 1.5 — the data overpower the prior — while national fits stay prior-dominated.) **Rule of thumb: release at geo granularity while per-cell DP noise b ≲ σ_nat = σ_geo√G; beyond that, coarse release + share-calibrated mixture wins.** At G=50 the geo tolerance shrinks (already behind at b=3 on mROAS@1x: 16.6% vs 10.7% for the mixture), consistent with per-cell SNR ∝ 1/(G·b): finer panels are more privacy-fragile. Pleasing symmetry: cross-geo dispersion is the *enemy* of the national fit (drift laws) and the *friend* of the geo fit (it spreads geos across operating points, which is what identifies the curve).

## 4. Attack-and-refine history

**Round 1 (construct):** drift laws + mixture estimator + DP width law, verified in the pseudo-true/noiseless regime.

**Round 2 (attack) — three breaks.** (i) *Identifiability crisis:* under realistic noise the unpenalized mixture (and naive) fit ran to a degenerate ridge — β̂ = 5972±2910 vs true 100 (β→∞, K→∞ with β·(a/K)^s finite mimics a power curve). National likelihoods at T=104 cannot pin the Hill shape at all; the ecological bias question is second-order to identification. (ii) *Benignness:* the planned "Jensen-gap indictment" of naive fitting collapsed at the operating point (§3.2) — the honest result is a reframe, not a correction of a large bias. (iii) *Instrument confound:* the first width-law experiment showed sd(β̂) *falling* as DP noise rose (20.3→12.1 against a predicted 5.7× rise) — prior shrinkage masquerading as precision, plus design randomness across seeds.

**Round 3 (refine):** penalized MAP with identical Meridian-style weak priors for both estimators (fair fight) — mixture recovers s = 1.99±0.12 vs naive 1.90±0.15 with the pseudo-true bias direction confirmed; width law re-tested on a fixed design and identified sub-fit (verified); the drift regime (E5/E5n) identified as where the mixture pays; frontier experiment built.

**Round 4 (attack again — no material revision, convergence):** fresh 24-seed replication of the drift headline (+10.0 vs +10.9); cv=0 drift control isolating the mechanism; G=50 frontier direction check; prior-center sweep. All confirmed Round 3's claims; only sharpenings emerged.

## 5. Honest negatives and checklist accounting

- **At realistic national SNR the drift laws are subdominant to sampling error and prior pull.** 24-seed naive fits give s = 1.98±0.08 where the pseudo-true value is 1.75: a finite-sample upward pull cancels the aggregation drift here by coincidence. The laws are asymptotic/high-SNR statements — they matter for long panels, refresh averaging (bias persists while noise averages), and transport, not for any single T=104 fit.
- **National shape recovery is prior-dominated either way** (E9a): centering the s-prior at 1.0/3.0 drags the mixture posterior to 1.77/2.90. The mixture moves the *target* to the right place; it cannot conjure identification the national likelihood lacks. Geo-level data can (E7, b small).
- **The mixture does not dominate everywhere:** its 4×-extrapolation RMSE (1.03) is *worse* than naive (0.67) at stable shares — unbiased shape + extrapolation variance can lose to biased-but-flat. Its win condition is share drift and/or transport use.
- **K-drift law is design-local** (E9b): holds at α=0.3/0.5; at α=0.7 the design sits deep in saturation, the crossing point is unvisited, and the measured drift (0.116) overshoots the law (0.064).

Checklist: 18 labeled experiments (E1, E1b, E2, E2b, E3, E3b, E4a/b, E5a/b, E5n, E6b/c, E7 ×5 noise levels + G=50, E8a/b/c, E9a/b, R4 controls); robustness sweeps: fresh seed batch, prior-center, DGP α, G; negative/power controls: E8a (c=0 estimator coincidence), E6c fixed-design analytic match (ratio 0.93 constant), E8b (the s-drift is *not* detectable in one realistic fit — a power statement); uncertainty: all headline numbers above carry ±sem or MC-precision notes (E7 cells are 6-seed, so RMSEs carry ~±30% relative MC error — the ×3.5–10 gaps at b∈{0,30} dwarf that; the b=3 near-tie is genuinely a tie within error).

**Devil's advocate.** *Against §3.2 (benignness):* it is proven only for single-channel, correctly-specified-family, additive-noise worlds; cross-channel confounding or non-Hill micro curves could reopen operating-point bias — untested here. *Against §3.3 (drift headline):* the "true" benchmark evaluates mROAS at mean shares; under drift the instantaneous truth wobbles around that, so a few points of the naive "bias" could be benchmark convention — but the mixture faces the same benchmark and lands at ~0, so the naive–mixture *gap* is real. The AR(1) exogenous drift is also the kindest case; endogenous drift (spend chasing demand) would confound both estimators. *Against §3.4 (frontier):* the geo arm uses variance-inflated Gaussian LS, not the exact likelihood — by the ARE bound this costs it at most √2 in width at b≫σ, which could shift the crossover left by a factor ≲1.4 but cannot erase the regime change; the parallel-composition premise (per-cell b independent of granularity) is standard for user-level DP but would fail for event-level budgets split across cells. *Against the ARE bound itself:* it is asymptotic and location-only; small-T or shape-parameter efficiency could differ (the clean-room agent's exact-density implementation matched, but both are asymptotic calculations).

## 6. Limitations and failure modes

Single channel throughout — no cross-channel spend correlation, no baseline covariates. Shares treated as exogenous; endogenous geo targeting correlated with geo-level demand shocks is the dangerous real-world case and is untested (connects to the endogenous-spend ROAS-inflation benchmark in the resource library). DP treated as additive Laplace on outcomes only — DP-noised *covariates* (SKAN/ARA-style noisy attributed conversions used as inputs) are classical measurement error through a nonlinear link and were deliberately scoped out. The frontier assumes the curator can be bargained with about granularity; real APIs fix it. Optimizer is MAP+Laplace, not full NUTS; dive 05's lesson (filter-class artifacts) suggests checking the prior-domination claims under full MCMC.

## 7. Next steps for a future session

1. **DP-noised covariates (the SKAN/ARA case):** known discrete-Laplace noise on *inputs* through the Hill link — Berkson vs classical decomposition, corrected estimating equations, and whether the 50% ARE bound has an analogue. The literature gap is confirmed unclaimed.
2. **Endogenous share drift:** replace AR(1) drift with spend-chasing-demand allocation; measure how much of the mixture estimator's win survives confounding (links BL10/BL3).
3. **The frontier as a mechanism-design statement:** given ε and a decision loss, derive the optimal release granularity G* in closed form from the per-cell SNR ∝ 1/(G·b) scaling; pitch as "what advertisers should ask aggregation APIs for."
4. **Drift-law transport correction:** dive 02's operator calibration consumes (K, s); apply Δlog K = c²/2, Δs = −c²s³H(1−H) as *analytic de-biasing* of nationally-fit parameters before transport, and test end-to-end against the dive-02 pipeline.
5. **Full-MCMC check of prior domination** (shares dive 05's NUTS-vs-EKF question; one joint compute budget could serve both BL13 and this).

## Sources

- Internal: dives 01 (identifying designs), 02 (transport consumes K,s), 03/04 (decision-loss framing); catalogs `../02_open_questions.md` M9, `../03_mmm_adoption_barriers.md` §O3.
- [van Garderen, Lee & Pesaran 2000](https://ideas.repec.org/p/cam/camdae/9803.html) · [Blundell & Stoker, Heterogeneity and Aggregation](https://web.mit.edu/tstoker/www/Blundell_Stoker_Handbook.pdf) · [Sun et al. 2017, GBHMMM (Google)](https://research.google/pubs/geo-level-bayesian-hierarchical-media-mix-modeling/) · [Meridian: geo vs national](https://developers.google.com/meridian/docs/pre-modeling/geo-selection-national-data)
- [Evans, King, Schwenzfeier & Thakurta 2023, APSR](https://gking.harvard.edu/dp/) · [Evans & King 2025, extensions to nonlinear transformations](https://gking.harvard.edu/dpd2/) · [Gong 2022, exact inference under known DP mechanisms](https://arxiv.org/abs/1909.12237) · [Karwa & Vadhan 2018](https://arxiv.org/abs/1711.03908) · [Delaney et al. 2024, DP ad conversion measurement](https://arxiv.org/abs/2403.15224) · [ARA summary-report noise](https://arxiv.org/pdf/2311.13586)
