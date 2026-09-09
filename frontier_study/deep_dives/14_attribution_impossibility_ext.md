# Dive 14 — EXTEND of 13: the experimental route to causal Shapley — a proof of the design theorem, the holdout-only design, and the precision–lift–aliasing frontier

**Queue item:** C2 (EXTEND run; extends dive 13 §5 / BL42, chosen over dive 12's BL37 because dive 13 left a conjecture verified only to n = 8 and a design rule with no cost theory — both provable and computable in one session). **Status:** verified (the theorem is proved below and reproduced clean-room by exhaustive rank computation, 1,070 cases, 0 mismatches; the estimator, aliasing law, add-up restriction, minimum-variance weights and the κ law were each reproduced clean-room to ≤ 1e−14 / < 0.01%; the finite-user results are Monte Carlo with stated sds). Code: `code/14_attribution_impossibility_ext.py` (`quick` ≈ 7 s reproduces the exact results; `e5`, `e13` ≈ 3–5 min; `e6`, `e7` ≈ 2 min each; needs numpy + scipy).

---

## 1. Problem statement

Dive 13 showed that observational attribution cannot be causal, that the only unidentified object is the off-diagonal of the cross-journey table, and that randomized multi-cell holdouts identify the causal Shapley allocation Sh(v_pop) if the design is rich enough: under a *k-way interaction cap* the "layered design" L_m (all cells with ≤ m channels held out or ≤ m channels exposed) identifies Sh iff m ≥ min(⌈k/2⌉, ⌊n/2⌋) — **verified by rank computation for n ≤ 8, proved only for k = 2**. It also left the economics open: which cells, how many users in each, what the design costs in forgone lift, and what happens when the cap is wrong.

Formalization (explicit assumptions):

- **Game.** v(A) = v_pop(A) = E[Y(Z ∧ 1_A)]: the conversion rate of a random user slice in which every channel outside A is suppressed (ghost-ad / PSA style; "cell A"). Harsanyi dividends D(T), v(A) = Σ_{T⊆A} D(T). Causal Shapley Sh_j = Σ_{T∋j} D(T)/|T|.
- **Interaction cap k** (Grabisch k-additivity): D(T) = 0 for |T| > k. k = 1 is the additive (ghost-ads) model; k = 2 is pairwise.
- **Two-layer design B(m_b, m_t)** = {A : |A| ≤ m_b or |N∖A| ≤ m_t}: expose-at-most-m_b cells ("bottom") plus hold-out-at-most-m_t cells ("top"). Dive 13's L_m = B(m, m); the ghost-ads menu is B(0, 1); the "holdout-only" design B(0, k) never deprives a user of more than k channels.
- **Identification.** A linear functional of D is identified iff its coefficient vector lies in the row space of the incidence matrix X[A, T] = 1[T ⊆ A] restricted to |T| ≤ k. Cells are unconfounded by randomization; sampling is binomial with cell mean v(A), variance s_A² = v(A)(1 − v(A)); cell A receives share f_A of N users.
- **Cost.** Lift forgone per user in cell A: ℓ_A = [v(N) − v(A)]/[v(N) − v(∅)] (fraction of the slice's incremental conversions given up); total ℓ = Σ_A f_A ℓ_A.
- **Aliasing.** For an estimator with cell weights c, the *leak* of a dividend T with |T| > k into Ŝh_j is Σ_A c_{jA} 1[T ⊆ A] − 1[j ∈ T]/|T|.
- **Numerical DGPs.** Dive 13's n = 3 benchmark (U ~ N(0,1), logit targeting a = (−0.5, −1, −1.5), g = (0.4, 1, 1.6), logit outcome β₀ = −2.5, β = 1, θ = (0.5, 0.5, 0)); an n = 5 logistic DGP with pairwise synergies (θ₁₂ = 0.6, θ₂₃ = −0.4, θ₁₄ = 0.3) and an optional 3-way term θ₃·z₁z₂z₃; an n = 4 version; monotone random games for scaling sweeps. Logistic potential outcomes generate dividends of *every* order, so the caps are genuinely misspecified in these DGPs (n = 5: |D₁| ≤ 0.022, |D₂| ≤ 0.018, |D₃| ≤ 0.0008 without and 0.009 with θ₃ = 0.8, |D₄| ≤ 0.0006).

## 2. Prior state

Literature subagent (this run) plus dive 13: k-additive games and the Shapley/Möbius formulas are Grabisch 1997 / Grabisch–Marichal–Roubens 2000; k-additive weighted-LS approximations of Shapley (Pelegrina–Kolpaczki–Hüllermeier, AAAI 2026) state explicitly that "there exists no approximation guarantee" when only a subset of coalitions is observed; Mayer–Wüthrich 2025 show paired sampling is exact when interactions have order ≤ 2; Fumagalli et al.'s OddSHAP (ICML 2026) show Shapley depends only on the odd part f(S) − f(S^c) — the closest existing statement to "bottom and top layers enter additively". Zhou–Mee–Hamers–Zheng (JASA 2025) recover exact Shapley from fractional factorials under a low-order truncation, but the layered designs here are not regular fractions (they are Hamming-weight shells), so classical resolution/aliasing tables do not apply. Marketing multi-cell designs (ghost ads; Gordon et al. 2019; Waisman–Gordon 2025) vary intensity of one campaign, never channel subsets; Test & Roll (Feit–Berman 2019) prices the opportunity cost of a single holdout. The subagent found none of: (a) the condition m_b + m_t ≥ min(k, n−1); (b) an aliasing coefficient for truncated Möbius fits on shell designs; (c) a precision–lift frontier for attribution experiments; (d) a lack-of-fit test of the cap from over-identified cells.

## 3. Constructions

### 3.1 Construction 1 — the identification theorem, proved (established here; clean-room 1,070/1,070)

**Theorem.** Under a k-way interaction cap, the causal Shapley allocation is identified from B(m_b, m_t) **iff m_b + m_t ≥ min(k, n − 1)**. (Dive 13's L_m statement is the case m_b = m_t = m: 2m ≥ min(k, n−1) ⟺ m ≥ min(⌈k/2⌉, ⌊n/2⌋).)

*Proof.* Fix j and let G = Stab(j) ⊂ S_n. The design and the cap are G-invariant and Sh_j is G-invariant, so Sh_j is identified iff it lies in the span of the G-averaged cell functionals (average any representation over G). Write σ_t := Σ_{T∋j,|T|=t} D(T), ω_t := Σ_{T∌j,|T|=t} D(T). G-averaging a bottom cell of size s gives a combination of {σ_t, ω_t : t ≤ s} that is triangular in s (a cell containing j is the only one contributing σ_s), so the bottom layer spans exactly {D_∅, σ_t, ω_t : t ≤ m_b}. The top layer is equivalent (invertible Möbius transform on the holdout lattice) to Q(H) := Σ_{T⊇H} D(T) = Σ_{B⊆H} (−1)^{|H∖B|} v((N∖H) ∪ B), |H| ≤ m_t; G-averaging gives, for H ∋ j, **R_h := Σ_{H∋j,|H|=h} Q(H) = Σ_t C(t−1, h−1) σ_t** (h = 1..m_t), and for H ∌ j, Σ_t C(t−1, h) σ_t + Σ_t C(t, h) ω_t (h = 0..m_t). Modulo the bottom layer the unknowns are u_t = σ_t (t = m_b+1..k) and w_t = ω_t (t = m_b+1..min(k, n−1); ω_n does not exist) and the target is (1/t on u, 0 on w).

*Sufficiency.* If m_b + m_t ≥ n − 1 the design is the full factorial (every size s satisfies s ≤ m_b or s ≥ n − m_t). Otherwise k − m_b ≤ m_t and the rows R_h, h = 1..m_t, restricted to u are the polynomials C(t−1, h−1) of degrees 0..m_t−1 evaluated at the k − m_b ≤ m_t points t = m_b+1..k; polynomials of degree ≤ m_t−1 interpolate any values on ≤ m_t points, so (1/t) is reached with zero w-component.

*Necessity.* Let m_b + m_t < min(k, n−1). Any representation Σ a_h R_h + Σ b_h R'_h of the target must vanish on the w-coordinates: Σ_h b_h C(t, h) = 0 at ≥ m_t + 1 distinct points t, and this is a polynomial of degree ≤ m_t, so b ≡ 0. The u-coordinates then require the degree-≤(m_t − 1) polynomial Σ a_h C(t−1, h−1) to equal 1/t at k − m_b ≥ m_t + 1 points, impossible because the m_t-th divided difference of 1/t over any m_t + 1 positive points is (−1)^{m_t}/Π t_i ≠ 0 while that of a degree-≤(m_t−1) polynomial is 0. ∎

*Explicit witness* (E1): n = 4, k = 3, L₁: dividends D₀₁ = 1, D₀₂ = −1, D₀₃ = 1, D₁₃ = −2, D₂₃ = 1, D₀₂₃ = −1, D₁₂₃ = 1 make every L₁ cell exactly 0 while Sh = (1/6, −1/6, 0, 0). Exhaustive rank check n = 3…8, all (k, m_b, m_t): 1,042 cases, 0 mismatches (clean-room: 1,070 cases n = 3…7, 0 mismatches).

**Corollary (holdout-only designs).** B(0, k) — global holdout, no holdout, and all holdouts of size ≤ k, so that no user is ever deprived of more than k channels — identifies Sh under a k-way cap. For k = 2 the closed form is

  Sh_j = [v(N) − v(N∖j)] − ½ Σ_{i≠j} [v(N) − v(N∖i) − v(N∖j) + v(N∖{i,j})],

i.e. the removal effect minus half the sum of pairwise removal interactions. Dive 13's "expose only j" cells are therefore not *necessary* for identification (E2, exact to 1e−15) — what they buy is precision (§3.4).

### 3.2 Construction 2 — the closed-form estimator and the aliasing law (established here; clean-room exact)

The sufficiency proof is an estimator. With r = k − m_b and λ solving the r×r binomial-Vandermonde system Σ_h λ_h C(t−1, h−1) = 1/t (t = m_b+1..k),

  **Ŝh_j = Σ_{t≤m_b} σ̂_t/t + Σ_{h=1}^{r} λ_h [R̂_h − Σ_{t≤m_b} C(t−1, h−1) σ̂_t],**

where σ̂_t comes from Möbius inversion of the bottom cells and R̂_h from alternating sums of top cells. For (m_b, m_t, k) = (1, 1, 2): λ = ½ and Ŝh_j = ½(solo + removal), dive 13's rule; for (2, 1, 3): Ŝh_j = ⅔D̂_j + ⅙Σ_i D̂_ij + ⅓[v(N) − v(N∖j)]; for (0, 2, 2): λ = (1, −½), the corollary above. Exact on random k-additive games for nine (n, m_b, m_t, k) combinations up to n = 8 (max error 4e−15; E2).

**Aliasing law.** Apply the cap-k estimator to a game with a dividend of order t' > k on a set T. Its coefficient on σ_{t'} is p(t'), the degree-(r−1) polynomial interpolating 1/t at t = m_b+1..k, so for j ∈ T the leak is the polynomial-extrapolation error of 1/t,

  leak = p(t') − 1/t' = (−1)^{r+1} Π_{t=m_b+1}^{k}(t' − t) / (t' Π_{t=m_b+1}^{k} t),

and for j ∉ T the leak is exactly zero (top rows containing j see only T ∋ j; bottom cells contain no set larger than m_b). At t' = k + 1:

  **|leak| = m_b! (k − m_b)!/(k+1)! = 1/((k+1)·C(k, m_b))**, sign (−1)^{k−m_b+1},

minimized by the *balanced* design m_b = ⌊k/2⌋ and worst (1/(k+1)) for the one-sided designs B(0, k) and B(k, 0). Verified for k = 2…5, all m_b, n ≤ 7 (sd across sets 0 to machine precision; E3): k = 2: 1/6 (balanced) vs 1/3 (one-sided); k = 3: 1/12 vs 1/4; k = 4: 1/30 vs 1/5. Higher orders extrapolate: for B(0, 3) the leak of order 5 and 6 dividends is 0.8 and 1.67 (polynomial extrapolation blows up), for B(3, 0) it is −1/t' (the bottom-only design simply misses them), for B(1, 2)/B(2, 1) it stays ≤ 1/3 (E3, all matching the formula). Summing over j ∈ T, a single order-(k+1) dividend moves the *total* credit by (−1)^{k−m_b+1} D(T)/C(k, m_b) — the add-up gap below is a guaranteed, sign-known diagnostic of cap violation.

### 3.3 Construction 3 — the one over-identifying restriction is "credits add up" (derived here; clean-room exact)

Every exactly-capped design B(m_b, m_t) with m_b + m_t = k ≤ n − 2 has rank(X) = #cells − 1 (E14, verified n = 4…8 for seven (m_b, m_t, k) families; clean-room n = 4…7): the cell means obey exactly one linear restriction under the cap, and it is

  **Σ_j Ŝh_j = v(N) − v(∅)** — the closed-form credits sum to the global-holdout lift

(the weight vector of Σ_j Ŝh_j − [v(N) − v(∅)] spans null(Xᵀ), cos = 1.000000 in every case). For the 2n + 2 design it reads Σ_i [v(N∖i) − v({i})] = (n − 2)[v(N) − v(∅)]. Origin: in the S_n-symmetric sector the top layer supplies m_t + 1 equations for the m_t unknown symmetric dividend sums s_{m_b+1..k} — one more than needed. Consequences: (i) the *add-up test* — the df = 1 lack-of-fit χ² of the cap is the standardized add-up gap, which under a single order-(k+1) dividend has mean D(T)/C(k, m_b) (E15: gap +0.0037 vs D₁₂₃/2 = +0.0045 at n = 5, the shortfall being other 3-/4-way dividends); (ii) the unbiased-under-cap estimators form a *line* c(α) = c_closed + α ν, on which variance is quadratic and aliasing is affine.

### 3.4 Construction 4 — minimum-variance weights, the precision–lift frontier and the κ law (derived here; clean-room < 0.01%)

**Minimum-variance weights.** Along the line c(α) the BLUE (c = Σ⁻¹X(XᵀΣ⁻¹X)⁺w) cuts the closed form's Σ_A c_A² from 1.000 to 0.437 (n = 4), 0.429 (n = 5), 0.443 (n = 8) for B(1, 1); from 4.0 to 0.86 (n = 5) for the holdout-only B(0, 2); from 15.9 to 2.27 for B(0, 3) at n = 6 (E4, E14). The cheap version — *subtract the add-up gap equally from every channel* — captures 85–99% of the gain (0.500, 0.480, 0.469; E14), so an analyst can implement it by hand: compute ½(solo + removal), compare the sum with the global-holdout lift, spread the difference.

**Frontier.** For a fixed design, tr Var_BLUE(f) is matrix-convex in the allocation f, so min_f tr V + μ·ℓ(f) is a convex program (L-optimal design with a lift penalty). At n = 5, k = 2 (E5): moving from the variance-optimal allocation (54% of the slice's lift forgone) to 10% forgone costs 4.3× in variance for the full factorial and for B(1, 1) alike; the holdout-only B(0, 2) sits 2.8–3.7× above the full-factorial frontier at every lift level and the bottom-heavy B(2, 0) 3–5×; B(1, 1) is within 11% of the global optimum over all 32 cells at every lift level, and at k = 3 the 22-cell B(1, 2) *is* the global optimum (the convex program puts zero mass on the ten pairs-exposed cells; E5, identical to 4 decimals). Neyman allocation gains only 3–10% over equal allocation (E8).

**The κ law.** As the lift forgone ℓ → 0 (mass concentrating on the no-holdout cell) the frontier obeys

  **N · ℓ · tr V → κ := min over cap-unbiased C of (Σ_{A≠N} ‖C_{·A}‖₂ s_A √ℓ_A)²**

(Cauchy–Schwarz: the optimal shares are g_A ∝ ‖C_A‖ s_A/√ℓ_A; the no-holdout cell is free). Verified: the frontier product converges to the group-L1 value to three decimals in 17 (n, k, design) cases (E13; clean-room 0.71041 vs 0.71044). κ is the *lift price of a Shapley allocation*: to reach a target Σ_j Var(Ŝh_j) = δ² one needs N·ℓ ≥ κ/δ² user-lift units. At n = 5, k = 2: κ = 1.05 (full), 1.11 (B(1, 1)), 2.91 (B(0, 2)), 5.01 (B(2, 0)); n = 6: 1.29, 1.50, 7.26, 16.3. Example: five channels, Sh_j ≈ 0.01–0.035, target sd 0.002 per channel → δ² = 2e−5 → N·ℓ ≈ 55,000: with 5M users in the test population the layered design costs 1.1% of the slice's lift, the holdout-only design 2.9%.

**Scaling in n** (E9, monotone pairwise game, Neyman allocation): tr V·N for B(1, 1) grows 0.81 → 6.1 from n = 3 to 8 (≈ n²: 2n + 2 cells, ½-weights), while the holdout-only B(0, 2) grows 0.81 → 156 (≈ n⁴: C(n, 2) cells each entering with weight ½) — 25× worse at n = 8, 3.5× at n = 5. Holdout-only Shapley is affordable only for n ≤ 4.

## 4. Attack-and-refine history

**Round 1** proved the theorem, derived the closed-form estimator and the aliasing law, and proposed the holdout-only design B(0, k) as the commercially realistic alternative to "expose only j" cells.

**Round 2 (attack).**
- *The closed form is not the estimator to use.* Every exactly-capped design turned out to be over-identified by one (rank = cells − 1, E14) — the closed form ignores the restriction and pays 2.3× (B(1, 1)) to 7× (B(0, 3)) in variance relative to the BLUE (E4). Dive 13's "½(solo + removal) costs the precision of one lift test" was therefore an *overstatement of the cost*.
- *The aliasing law describes the wrong estimator.* Under the BLUE the leak of an order-(k+1) dividend is no longer confined to T's members: at n = 5 with a genuine 3-way dividend D₁₂₃ = +0.009 the BLUE's bias is +0.0002–0.0003 on channels 1–3 but **−0.0013 and −0.0011 on channels 4 and 5, the latter causally null (5.7 sd at N = 4M for θ₃ = 1.5)**, whereas the closed form biases channels 1–3 by +0.0013 each and channel 5 by exactly 0 (E15; E6 confirms by Monte Carlo: B(1, 1) BLUE bias −0.0013/−0.0011 at sd 0.0003–0.0004). Variance-optimality transfers misspecification onto innocent channels.
- *Holdout-only precision collapses with n* (E9: 25× at n = 8) and its high-order aliasing explodes (order-5 leak 0.8, order-6 leak 1.67 for B(0, 3); E3) — polynomial extrapolation from top layers only is numerically unstable.
- *The frontier optimizer is fragile.* Minimizing tr V + μ·ℓ collapses at large μ onto f_N = 1 where XᵀΣ⁻¹X is singular and the pseudo-inverse silently returns tr V = 0 (found independently by the verifier); trustworthy limits need ℓ fixed or the group-L1 form.
- *Is any of this needed at all?* The ghost-ads menu B(0, 1) with an additive cap is what advertisers run today: under pairwise synergy it mis-allocates by 25–100% of a channel's credit (E6: bias +0.0087 on Sh₁ = 0.036, −0.0084 on Sh₃ = 0.009, −0.0047 on the null channel; 12–29 sd) while every identified design is unbiased to within sd (≤ 1.2e−4).

**Round 3 (refine).**
- Reframed the estimator family as the line c(α) = c_closed + α ν with ν the add-up restriction (Construction 3): the closed form (α = 0), the add-up-projected estimator (nearly BLUE, hand-computable) and the BLUE are three points on it, with a variance–aliasing trade-off that the analyst chooses explicitly; the add-up gap itself is the df = 1 specification test.
- Replaced "which design" by the frontier and the κ law (Construction 4), which price every design in the same currency (users × lift forgone per unit of Shapley variance) and show B(1, 1) within 5–16% of the global optimum while B(0, k) is 2.8–5.6× off.
- Added the cap-raising alternative: at n = 5 the 22-cell B(1, 2) under a 3-way cap has the same variance as the 12-cell B(1, 1) under a pairwise cap (tr V·N 2.29 vs 2.20; E11) and is immune to 3-way dividends (bias ≤ 1.7e−4 at θ₃ = 0.8, E6) — robustness to the next interaction order is nearly free in precision at n = 5 (and B(2, 2) under a 4-way cap is *more* precise than B(1, 1) at n = 7: 3.68 vs 4.40) at the price of 2–4× more cells.
- Attacked the MSE conclusion with a prior on higher-order dividends (E11): with τ₃ = 0.001 (a tenth of a main effect) the pairwise designs lose to the 3-way designs at N ≥ 2M; at τ₃ = 0.003 already at N = 2e5. The cap should be set by the add-up test, not assumed.

**Round 4** re-ran the attacks against the refined constructions: the verifier reproduced the theorem, estimator, aliasing law, restriction, BLUE numbers and κ, and added the corollary that the add-up gap under a single order-(k+1) dividend is D(T)/C(k, m_b) (folded into §3.2). Nothing else moved; rounds 3→4 produced no material revision.

## 5. Field protocol (derived; operational)

1. Choose k from the add-up test of a pilot (or start with k = 2). Run B(⌊k/2⌋, ⌈k/2⌉) — for k = 2 the 2n + 2 cells of dive 13; for k = 3 the 2 + 2n + C(n, 2) cells of B(1, 2) — never the holdout-only design for n ≥ 5.
2. Allocate: nearly all users to the no-holdout cell; spread the lift budget ℓ over the other cells with g_A ∝ ‖C_A‖ s_A/√ℓ_A (plug in pilot rates); size by N·ℓ ≥ κ/δ².
3. Estimate three ways and report all: closed form (structured aliasing, null channels protected), add-up-projected (near-BLUE), BLUE. Report the add-up gap with its χ²₁ p-value; if it rejects, the sign of the gap gives the sign of the missing interaction, and the remedy is B(1, 2) next wave, not a different estimator.
4. Feed [Ŝh_j ± sd] into dive 08/09's act/hold layer.

## 6. Verification and controls

- **Exactness.** Theorem: 1,042 cases (n ≤ 8) + clean-room 1,070 cases (n ≤ 7), 0 mismatches. Estimator: ≤ 4e−15 on random capped games (nine configurations). Aliasing: matches the formula to machine precision for k = 2…5, all m_b, orders k+1…k+3. Restriction: rank = cells − 1 and cos(add-up, ν) = 1.000000 in all ten configurations. BLUE Σc² and add-up-projected Σc²: reproduced to 3 decimals. κ: group-L1 program vs frontier limit within 0.3% in 17 cases; clean-room 0.71041 vs 0.71044 (< 0.01%).
- **Monte Carlo** (E6; binomial cells, N = 2–4M users, 40–60 reps; sd of a reported sd ≈ 10%): empirical sds match the predicted sds (0.00034–0.00043 vs 0.00037–0.00038); biases of identified designs ≤ 1.7e−4 when the cap holds; under θ₃ = 0.8 the k = 2 designs show the predicted biases and B(1, 2) does not. Seeds (E8: 5 seeds × 30 reps, N = 2.4M): means within 1.2e−4 of truth, sds 0.00037–0.00049.
- **Negative/power controls.** E10: non-identified designs have row-space residuals 0.12–0.35 and min-norm pseudo-estimates err by 0.07–0.38 on unit-scale games while identified ones err by 0 — the rank test discriminates. E7: the cap lack-of-fit test rejects at 2–3/60 under the null (5% nominal) and at 49/60 (df = 1 add-up test, B(1, 1)), 57/60 (B(1, 2)), 60/60 (full) against D₁₂₃ = 0.009 at N = 2M; 18–29/60 against 0.0042. E6: the additive ghost-ads menu is biased 20–30 sd under pairwise synergy.
- **Robustness sweeps.** Base rate β₀ ∈ {−4, −2.5, −1} (E8: B(1, 1) vs B(0, 2) sd ratio 1.7–1.8 at every rate; Neyman gain 3–10%); n = 3…8 scaling (E9); prior on higher-order dividends τ₃ ∈ {0, 0.001, 0.003} × N ∈ {2e5, 2e6, 2e7} (E11); competing design (E12: KernelSHAP/permutation-style size-uniform allocation over all 32 cells is 18% less precise than the 12-cell B(1, 1) at equal lift).
- **Uncertainty on headline numbers.** Exact quantities (theorem, estimator, aliasing, restriction, BLUE weights, κ) carry only floating-point error (≤ 1e−12); κ's frontier check has optimizer error ≤ 0.3%. MC biases carry sd ≈ 0.00006 (60 reps at cell sd 0.0005); the design-comparison ratios (2.3×, 3.7×, 25×) are exact given the game and vary ±15% across the games tried (E5/E8/E9/E11).

## 7. Devil's advocate

- *"The theorem is a rank fact anyone could compute."* Dive 13 did compute it and could not prove it; the proof's content is that the Stab(j)-symmetric sector reduces identification to interpolating 1/t by polynomials — which is also what yields the estimator and the exact aliasing coefficients, so it is the one lemma that produces the whole design theory.
- *"Cap assumptions are arbitrary."* They are testable: every exactly-capped design carries the add-up test with 82–100% power against a 3-way dividend equal to 40% of a main effect at N = 2M, and raising the cap costs ≤ 4% variance at n = 5. The honest residual is that a cap is a functional-form assumption on v_pop and the test has df = 1 — a 3-way dividend that happens to cancel in the symmetric sector is invisible (measure-zero, but there).
- *"Nobody can run 12–22 cells across platforms."* True today (the E4/E6 mechanism problem of the catalog); this dive prices the alternatives rather than removing the obstacle. Within one walled garden (one platform's placements as "channels") the design is implementable now. And the frontier says the cell *count* is not the binding constraint — the lift budget ℓ is, and it can be made small (1–3% of the slice's lift) by concentrating mass on the no-holdout cell.
- *"BLUE vs closed form is a false dilemma — just use the higher cap."* At n ≥ 7 the cap-4 design has 58 cells; at n = 5 the choice is indeed nearly free. The line c(α) matters precisely when cells are scarce.
- *"The κ law is an asymptotic in ℓ → 0 that no one operates at."* The frontier product is within 3% of κ already at ℓ = 5–10% (E13), the regime an always-on program would run.
- *"Precision comparisons depend on the game."* Ratios vary ±15% across the six games used; the orderings (full ≈ balanced ≪ one-sided) never changed, and the n-scaling exponents are structural (cell counts and weight row-sums).

## 8. Limitations and failure modes

Sets not paths (frequency and order add cells but do not change the algebra; BL44); binary outcomes (revenue changes s_A only); perfect suppression in cells (partial compliance rescales v(A) by a cell-specific factor and breaks the cap algebra unless compliance is logged); no interference between cells; pilot-rate plug-ins for Neyman and κ (two-stage designs would fix this); the κ law is for the trace criterion (a max-variance criterion changes the group norm); the aliasing law is proved for the exactly-determined estimator — the BLUE's aliasing is design- and variance-dependent and only computed; the df = 1 result is verified to n = 8, with a proof only that the symmetric sector contributes one restriction; the cost model counts lift forgone, not platform coordination cost or the option value of the information (dive 03/04's EVSI machinery would close that loop).

## 9. Next steps for a future session

1. **Test & Roll for attribution:** join the κ law to dive 03/04's decision-value machinery — choose ℓ (and k) to maximize EVSI minus κ-priced lift forgone; the multi-channel Test & Roll (links BL7).
2. **Prove df = 1 in general** via S_n-isotypic decomposition of the shell design (multiplicities of S^{(n−i,i)} in cell space vs capped dividend space).
3. **Compliance-aware cells:** partial suppression with logged exposure — does the cap algebra survive with an exposure-weighted incidence matrix, and what replaces the add-up restriction?
4. **Sequential/adaptive design:** always-on B(1, 1) with the add-up statistic as a trigger to open B(1, 2) cells (dive 05's scheduler with cap misspecification as the drift).
5. **Odd-part connection:** OddSHAP's f(S) − f(S^c) suggests pairing each bottom cell with its complementary top cell; check whether paired allocation (f_A = f_{N∖A}) is the frontier-optimal symmetry and whether it simplifies κ.
6. **Frequency/pacing extension** of the cell algebra (BL31/BL44) and the decision layer (BL41).

## 10. Sources

- Internal: dive 13 (objects, benchmark, layered design), dives 03/04 (EVSI), 05 (scheduler), 08/09 (decision layer); catalogs `../02_open_questions.md` §M8, `../03_mmm_adoption_barriers.md`.
- [Grabisch 1997, k-order additive fuzzy measures](https://www.sciencedirect.com/science/article/pii/S0165011497001681) · Grabisch, Marichal & Roubens 2000, Math. OR · [Pelegrina, Kolpaczki & Hüllermeier, AAAI 2026, arXiv 2502.04763](https://arxiv.org/abs/2502.04763) · [Mayer & Wüthrich 2025, arXiv 2508.12947](https://arxiv.org/abs/2508.12947) · [Fumagalli et al., OddSHAP, ICML 2026, arXiv 2602.01399](https://arxiv.org/abs/2602.01399) · [Fumagalli, Witter & Musco, PolySHAP, arXiv 2601.18608](https://arxiv.org/abs/2601.18608) · [Musco & Witter 2025, arXiv 2410.01917](https://arxiv.org/abs/2410.01917) · [Covert & Lee 2021, Improving KernelSHAP](https://proceedings.mlr.press/v130/covert21a.html) · [Bordt & von Luxburg 2023](https://arxiv.org/abs/2209.04012)
- [Zhou, Mee, Hamers & Zheng, JASA 2025, Shapley via fractional factorial designs](https://www.tandfonline.com/doi/abs/10.1080/01621459.2025.2529027) · [Zhou et al., JASA 2024, order-of-addition designs, arXiv 2309.08923](https://arxiv.org/abs/2309.08923) · Dasgupta, Pillai & Rubin 2015, JRSS-B · [Egami & Imai 2019, AMIE](https://imai.fas.harvard.edu/research/files/int.pdf)
- [Johnson, Lewis & Nubbemeyer 2017, Ghost Ads](https://journals.sagepub.com/doi/10.1509/jmr.15.0297) · [Gordon et al. 2019](https://pubsonline.informs.org/doi/10.1287/mksc.2018.1135) · [Waisman & Gordon 2025, multicell experiments, arXiv 2302.13857](https://arxiv.org/abs/2302.13857) · [Berman 2018](https://pubsonline.informs.org/doi/10.1287/mksc.2018.1104) · [Singal et al. 2022](https://pubsonline.informs.org/doi/10.1287/mnsc.2021.4263) · [Feit & Berman 2019, Test & Roll](https://arxiv.org/abs/1811.00457) · [Fair effect attribution in parallel experiments, arXiv 2210.08338](https://arxiv.org/abs/2210.08338)

*Code:* `code/14_attribution_impossibility_ext.py` (`quick` ≈ 7 s: E1–E4, E10, E14, E15; `e5`/`e13` ≈ 3–5 min; `e6`/`e7` ≈ 2 min; `e8`, `e9`, `e11`, `e12` < 1 min each).
