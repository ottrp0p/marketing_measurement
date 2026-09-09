# Dive 10 — Estimands under algorithmic delivery: cross-score logging identifies the fixed-policy creative effect (B4 / M3)

**Item:** B4 — M3, "Estimands stable under algorithmic delivery." **Status of headline results:** the divergence-bias identity, the sufficiency theorem for cross-score cell logs, the cross-score estimator, and the negative control are **verified** (population-limit algebra + finite-sample MC + independent clean-room re-implementation to ~1% of the divergence bias). The bias–noise crossover and the overlap-policy precision claim are **verified by simulation** (one DGP family, three sample sizes). The Λ-bounds negative result is **verified for the DGP family** (worst-case bounds; sharper bounds are not ruled out). SIMEX repair of noisy logs is **partially verified** (works, but extrapolant-dependent). The logging-specification hierarchy in §3.5 is **design grounded in the theorem**, not a field result.

---

## 1. Problem statement

Platform A/B tests of creatives route each arm through its own adaptive delivery (auction + pacing + per-creative response prediction). Braun & Schwartz's *divergent delivery* shows that the realized audiences of arms A and B differ systematically, so the platform-reported "A beats B" conflates creative quality with the algorithm's audience choice. The catalog asks (M3): define and identify "the effect of creative A vs B under a **fixed** delivery policy π" from data generated under adaptive policies π_A, π_B, where the logging policy is a black box with **unlogged propensities**; and (B4) find **what partial logging suffices**.

Formalization (explicit assumptions). Users i have covariates X (rich; only a coarse projection is advertiser-visible). Potential outcomes Y(0) (unexposed) and Y(1,a) for creative a ∈ {A,B}; write δ_a(X) = E[Y(1,a) − Y(0) | X]. Arms are randomized across users (Z ∈ {control, A, B}). Within arm a, the platform exposes user i with an unlogged, unknown probability π_a(X_i) — a function of the platform's per-creative score m_a(X) (predicted response), of pacing/competition, and of anything else it uses. Exclusion restriction: creative identity affects Y only through exposure (Y(0) does not depend on the arm). Define

- ITT_a = E[π_a(X) δ_a(X)] — the effect of creative a **under its own delivery** (what the test measures); ITTdiff = ITT_A − ITT_B;
- **τ(π) = E[π(X)(δ_A(X) − δ_B(X))]** — the creative effect under fixed delivery π. Natural choices: π = π_B ("A in B's audience"), π = π_A, and the **overlap policy** π_ov = 2π_Aπ_B/(π_A+π_B) (the audience both algorithms would reach; the delivery analogue of Li–Morgan–Zaslavsky overlap weights).

τ(π) is what a creative decision needs: "if I ship A instead of B, the algorithm will re-target, but creative quality is the part that transports." Note π_B itself is a legitimate fixed policy: the platform would deliver the winning creative with its own optimizer, so "A under B's audience" answers "would swapping the creative *for the same audience* help?"

## 2. Prior state

Literature subagent (full report in the run log; key sources in §9):

- **Divergent delivery is established and quantified.** Braun & Schwartz (*J. Marketing* 2025; SSRN 2019 w/ Eckles) coined it and showed two *identical* PSA ads drift apart on gender (57.0% vs 50.8% female) from delivery noise alone. Burtch, Moakler, Gordon, Zhang & Hill (Meta, arXiv 2508.21251) quantified it across 181,890 A/B tests + 3,204 lift tests: lift tests (one live arm vs holdout) show no imbalance; A/B tests (two separately optimized live arms) do; harmonizing campaign configuration reduces but never eliminates it. Ali et al. (CSCW 2019) give the mechanism (ML relevance layer skews delivery under maximally inclusive targeting).
- **The only formal identification result requires an extra arm.** Pal & Susarla (arXiv 2605.23706, 2026) add a third arm in which the algorithm sees the treatment's metadata but users see the control creative, decomposing total effect into natural direct (creative) and indirect (algorithm) effects — point identification without sequential ignorability, but requiring platform cooperation to build the arm. They also show why naive covariate adjustment on realized audience is biased (post-treatment mediator).
- **OPE tools exist but unmapped.** Estimated behavior policies (Hanna–Niekum–Stone 2019; Hirano–Imbens–Ridder 2003), OPE under unobserved confounding (Namkoong et al. 2020; Kallus & Zhou 2018/2020 Λ-sensitivity), slate OPE (Swaminathan et al. 2017). The subagent found **no paper** applying these to divergent delivery, and no formal "what logging suffices" statement — the gap is confirmed.
- Waisman & Gordon (*Mgmt Sci* 2023) is the closest econometric template (multi-cell designs for intensity), not creative identity.

Internal: dive 02's transport machinery (this is a transport problem across delivery policies); dive 06's Jensen/aggregation algebra (the coarse-cell bias here has the same shape).

## 3. The construction

### 3.1 The divergence-bias identity (exact)

For any fixed policy π and the randomized A/B/holdout design, ITTdiff is identified and

  **τ(π_B) − ITTdiff = E[(π_B − π_A) δ_A]**,  more generally  τ(π) − ITTdiff = E[(π − π_A)δ_A] − E[(π − π_B)δ_B].

The A/B test's bias for the fixed-policy estimand is a *covariance between the delivery difference and the creative-effect heterogeneity*. Two corollaries: (i) if δ_A is constant (no heterogeneity in the *new* creative's effect) the bias is exactly E[π_B − π_A]·δ_A — pure reach difference, sign known; (ii) if delivery does not diverge (π_A = π_B) there is no bias whatever the heterogeneity. Divergence *plus* heterogeneity is required, and this is exactly the case the algorithm manufactures: it diverges *because* it detects heterogeneity in predicted response.

Verified numerically to float precision (residual 1e-19; clean-room agent: 1e-19 to 3e-19 across seeds).

### 3.2 Sufficiency theorem for cross-score cell logs

Let C be a logged discrete cell variable (a function of X). Within cell c define r_a(c) = E[π_a | c] (reach, identified from exposure logs), ITT_a(c) = E[π_a δ_a | c] (identified from the arm-vs-holdout contrast within c), and the reach-normalized effect δ̃_a(c) = ITT_a(c)/r_a(c) = E_{π_a}[δ_a | c] (the π_a-weighted mean of δ_a in the cell). The **cell estimator** is

  τ̂(π) = Σ_c p_c · π(c) · [δ̃_A(c) − δ̃_B(c)],  with π(c) = E[π | c] (r_B(c) for target π_B; 2r_Ar_B/(r_A+r_B) for π_ov).

**Exact error decomposition** (derived here, verified to 3 decimals in E5): for target π_B,

  τ̂ − τ(π_B) = −Σ_c p_c { E[π_B δ_A | c] − r_B(c)·E_{π_A}[δ_A | c] }.

The B-term is exact (E[π_B δ_B|c] is directly identified); all error sits in the *cross term* E[π_B δ_A | c]. Hence:

**Theorem (ratio-balancing sufficiency).** τ̂(π_B) = τ(π_B) if, within every cell, **either** (a) the delivery ratio π_B(X)/π_A(X) is constant, **or** (b) δ_A(X) is constant, **or** more generally Cov_c(π_B/π_A, δ_A) = 0 under the π_A-weighted measure. Condition (a) holds whenever π_a(X) = g_a(m_A(X), m_B(X)) — delivery depends on X only through the two logged creative scores — so **the pair of per-creative scores (m_A, m_B), logged for every user in every arm, is a sufficient logging statistic. Neither score alone is.** Condition (a) is *weaker* than Rosenbaum–Rubin balancing: unlogged shocks that hit both arms' logits identically (pacing, time-of-day, competition common to both creatives) cancel in the ratio, so they need not be logged. What must not exist is an unlogged shock that is **arm-asymmetric and correlated with δ_A** (E14 shows this is the true boundary).

Why a single score fails: within an m_A cell, π_B still varies with m_B, and m_B is correlated with δ_A (creative effects share heterogeneity; scores share the baseline). The cross term E[π_B δ_A | m_A] therefore does not factor. E1/E4: with cells on m_A only the estimator lands at −3.4 vs truth −1.7 (CTR regime; clean-room agent: −3.44 ± 0.10); cells on m_B only land on τ(π_A) (+16.4) — the wrong estimand with the wrong sign.

Why this doesn't need propensities: the estimator uses reach *rates* and ITT *rates* by cell, both identified by randomization; π_a's individual values never appear. This is the M3 answer: the unlogged propensity is not the object that has to be logged — the *sorting variable* is. Any coarsening of (m_A, m_B) that preserves ratio-balancing to the needed tolerance suffices (E4: K = 40 quantile bins per score leaves 1% of the divergence bias).

### 3.3 Two variants and their assumption ladders

1. **Holdout version (above)**: needs a no-ad control cell (Meta's multi-cell lift design). Requires only ratio-balancing. Robust to unlogged confounders of *exposure vs Y(0)* because Y(0) enters only through the randomized arm-vs-holdout contrast (E5 CONV: error ≤ 0.02 under a strong competition confounder).
2. **No-holdout version**: δ̃_a(c) replaced by the within-cell exposed-minus-unexposed contrast. Requires *full* balancing (D ⊥ (Y(0), δ) | cell) — fails when unlogged competition selects high-baseline users into exposure (E5 CONV: error −0.12 on a truth of 0.48, i.e. 25%). In the CTR regime (Y(0) ≈ 0) the two coincide.

### 3.4 The overlap policy π_ov — the robust default

τ(π_ov) with π_ov = 2π_Aπ_B/(π_A+π_B): per-cell weights π_ov(c)/r_a(c) ≤ 2 are bounded, so (E10) its sampling sd equals ITTdiff's (0.95 vs 0.92 ×1e-4 at N = 3M) while τ(π_B) costs +27% sd (1.17), and (E15) its coarseness bias is negligible at every K (0.12 → 0.005 vs 1.69 → 0.07 for π_B). It also degrades most gracefully under asymmetric shocks (E14). Interpretation: "the creative effect on the audience both optimizers would reach." When the question is "which creative is better *as a creative*," π_ov is the recommended headline estimand; τ(π_B)/τ(π_A) answer the sharper "swap for the same audience" question at a precision cost.

### 3.5 Logging-specification hierarchy (design, from the theorem)

| Logged | Identifies | Assumption |
|---|---|---|
| L0: arm toplines | ITTdiff only | — (τ(π) not identified, sign of bias unknown) |
| L1: advertiser segments + holdout | Λ-bounds on τ(π_B) | within-segment ratio divergence ≤ Λ — **vacuous in practice** (E11) |
| L2: per-user bucketed (m_A, m_B) at decision time + holdout | τ(π) for any π expressible in scores | ratio-balancing (arm-symmetric unlogged shocks OK) |
| L2′: L2 without holdout | same | full balancing (no Y(0) confounding of exposure) |
| L2 + eligibility bit (was user in A's eligible set?) | identified part of τ(π) + reported unidentified mass | positivity failure made explicit (E7) |
| L2 + period id (or within-period ranks) | τ(π) under learning/non-stationary delivery | scores comparable within period (E8) |

The scores are bucketed quantiles — privacy-compatible, revealing neither the model nor propensities — and the platform already computes both scores for every auction. Not the standard Meta A/B log; the closest existing artifact is Meta's own configuration-harmonization guidance.

## 4. Attack-and-refine history

**Round 1 (construct).** Identity + cell estimator + IF standard errors. First results (conversion regime, N = 2–3M): ITTdiff overstates τ(π_B) by 67% (1.13 vs 0.67 ×1e-4); post-stratifying exposed users on advertiser segments does nothing (1.12); cross-score cells converge (0.85 → 0.68 for K = 3 → 40). But sampling sd ≈ 2–5 ×1e-4 swamped everything — the whole creative test is underpowered on conversions at 1M users. Added a CTR regime (p0 ≈ 0, δ = CTR): here ITTdiff = +7.8 but τ(π_B) = −1.8 and τ(π_A) = +16.5 — **the ranking of creatives flips sign with the fixed policy**, M3's "ill-defined" claim made concrete.

**Round 2 (attack).** Nine attacks:
1. *Coarseness* (E4): K = 10 leaves 36% of the divergence bias, K = 20 14%, K = 40 5% (1% in the clean-room's seed). Bias falls ~K^{−1.4}.
2. *Unlogged competition confounder* (E5): holdout version robust (ratio-balancing), no-holdout fails in the conversion regime. Exact error decomposition reproduces the holdout error to 3 decimals — confirms the theorem's error term is the whole story.
3. *Noisy logs* (E6): if the logged score is a noisier version of the delivery score, attenuation is severe: reliability 0.91 erases 18% of the correction, 0.79 erases 38%, 0.48 erases 70%. **Serious failure mode.**
4. *Positivity* (E7): when A's delivery excludes users at random w.r.t. δ, the estimator extrapolates correctly; when exclusion is correlated with δ (excluded = low-δ_A users) it lands at −0.6 — neither the truth (−1.8) nor the identified part (+3.1).
5. *Learning delivery* (E8): score noise decaying 0.8 → 0.1 over 12 periods plus arm-level refit shocks. Period-specific cells: limit within 0.3 of truth (K = 15). Pooled cells ignoring period: fine without shared shocks, wrong sign with them (+0.59 vs −1.16).
6. *Negative control* (E9): identical creatives (δ_B ≡ δ_A) but B scored with noisier predictions (a cold-start creative). ITTdiff = +5.5 ×1e-4 — a spurious "creative effect" 70% the size of the real effect scale — while the cell estimator gives 0.08 (clean-room: 0.077 ± 0.001).
7. *Segments-only Λ-bounds* (E11): realized within-segment divergence Λ95 = 5.7 (max 219); at Λ = 1.5 the bound is [−18.6, 47.5] — width 66 on an effect of scale 8. Adding the effect driver's tercile doesn't help (Λ95 = 5.1). **Clean negative result:** advertiser-side segments cannot rescue the estimand via sensitivity bounds; score noise alone makes within-segment delivery ratios wildly dispersed.
8. *Heterogeneity map* (E12): sign flips appear for effect-heterogeneity scale ≥ 0.5 with weakly correlated creative effects; estimator tracks truth within 0.2 everywhere; τ(π_ov) is stable (6.2–8.6) across the map.
9. *Bias–noise crossover* (E10): CTR outcomes, bias/sd = 2.4 at N = 0.3M, 8.3 at 3M; conversion outcomes, 0.05 → 0.16 — crossover near N ≈ 10^8.

**Round 3 (refine).** (a) SIMEX on the logged scores against attack 3: quadratic extrapolation recovers 85%/82%/62%/36% of the attenuation at reliability 0.96/0.91/0.79/0.60; a "reliability-ratio" extrapolant derived from the Gaussian heuristic did *worse* (67%/61%/53%/49%). (b) One extra bit — an eligibility flag — against attack 4: the estimator then returns the identified part (3.18 vs 3.09) and reports the unidentified B-mass exactly (0.066 = true). (c) Period id in the cell key against attack 5.

**Round 4 (attack the refinements).** (a) SIMEX's answer depends on the extrapolant by up to 1.7 ×1e-4 — it is a mitigation, not identification; the real fix is logging the score *used at decision time* (then reliability = 1 by construction). (b) *Arm-asymmetric* unlogged shocks correlated with δ (E14: competition hits A's auctions only, and competition correlates with the effect driver) break ratio-balancing: error up to −7.9 for τ(π_B), −1.5 for τ(π_ov); shocks uncorrelated with δ remain harmless (≤ 0.25). This is the precise boundary of the theorem and is stated as such in §3.2. (c) K-selection (E15): at N = 1M/CTR the RMSE-optimal K ≈ 30 for π_B (bias 0.14, sd 1.84); π_ov is K-insensitive.

**Round 5.** No material revision: the theorem's statement, the estimand recommendation (π_ov headline, π_B/π_A as sharper questions), the logging spec, and the failure boundary were unchanged by Round 4. Stopped.

## 5. Experiments and verification

All numbers ×1e-4 (absolute lift in outcome probability), CTR regime unless noted; seed sweep = 6 seeds at N = 600k, K = 40 population limits (mean ± sd across seeds).

| # | Experiment | Headline number | Uncertainty |
|---|---|---|---|
| E1 | Identity + estimator zoo (conversion) | ITTdiff/τ(π_B) = 1.70; post-strat no help | ± 0.03 (seeds) |
| E2 | Finite-N coverage (N = 1M, 60 reps) | IF-se 4.85 vs MC sd 4.53; 95% CI cover 0.93 | binomial ± 0.03 |
| E3 | Sign flip (CTR) | ITTdiff +7.99, τ(π_B) −1.69, τ(π_A) +16.68, τ(π_ov) +7.10 | ± 0.06–0.08 |
| E4 | Coarseness | residual bias fraction at K = 40: 0.010 | ± 0.000 |
| E5 | Confounder sweep, holdout vs no-holdout | CONV: 0.02 vs −0.12 error (truth 0.48) | single seed |
| E6 | Logged-score reliability | 0.91 → 18% of correction lost; 0.79 → 38% | single seed; monotone |
| E7 | Positivity + eligibility bit | flagged est 3.18 vs identified part 3.09; missed mass 0.066 = truth | single seed |
| E8 | Learning delivery | period-cells within 0.3 of truth (3 seeds); pooled+shared-shock wrong sign | 3 seeds |
| E9 | **Negative control** | ITTdiff 5.48 ± 0.03 spurious; cell 0.075 ± 0.002 | 6 seeds |
| E10 | Bias/sd crossover | CTR: 2.4 (0.3M) → 8.3 (3M); CONV: 0.05 → 0.16 | 20 reps/N |
| E11 | Λ-bounds (segments) | Λ95 = 5.70 ± 0.01; width at Λ = 1.5: 66.2 ± 0.1 | 6 seeds |
| E12 | Heterogeneity map | flips for het ≥ 0.5 & corr ≤ 0.5; est−truth ≤ 0.2 | grid |
| E13 | SIMEX | quad recovers 85/82/62/36% | extrapolant-dependent (±1.7) |
| E14 | Asymmetric shocks | π_B error to −7.9, π_ov to −1.5; uncorrelated shocks ≤ 0.25 | grid |
| E15 | K selection | RMSE-optimal K ≈ 30 (π_B); π_ov K-insensitive | 12 reps |

Robustness sweeps: coarseness K, confounder strength × regime, logged-score reliability, heterogeneity × correlation, seed sweep (6). Power/negative control: E9 (null creatives, non-null delivery → estimator nulls it, ITTdiff doesn't) and E10 (detectability vs N).

**Independent clean-room verification.** A subagent given only the DGP spec and the claims (no code) re-implemented everything in numpy: identity residual ≤ 3e-19; ITTdiff 8.03 ± 0.12, τ(π_B) −1.65 ± 0.09, τ(π_A) 16.74 ± 0.11 (sign flip confirmed); cross-score K = 40 relative residual 1.0% (≤ 5% claimed); m_A-only −3.44, m_B-only +16.38 (both far off); negative control ITTdiff 5.50 ± 0.01, cell 0.077 ± 0.001. Discrepancies: none substantive. One useful correction: finite-sample sd at N = 600k is ≈ 2.4 ×1e-4 (I had said ~3); its single draw of the cell estimator was 1.6 sd from truth, its ITTdiff draw 2.7 sd from the population ITTdiff, and its 10-replicate mean was −0.41 ± 0.79 (1.5 se from −1.59). The agent's bottom line is adopted verbatim in §6: at N = 600k the estimator can reject ITTdiff-sized values (~3 sd) but cannot sign τ(π_B).

**Devil's advocate, per headline claim.**
- *"Cross-scores suffice."* Only under ratio-balancing. Real auctions have arm-asymmetric unlogged components (creative format changes inventory eligibility, bid landscapes, frequency caps), and E14 shows correlated asymmetric shocks can produce errors larger than the effect. The theorem tells you what to log; it cannot certify that the platform's unlogged residual is arm-symmetric. A specification test (does adding more logged features move τ̂?) is the practical guard, not a proof.
- *"Sign flips."* Manufactured by het = 1, corr = 0.5 in a lognormal-heterogeneity DGP; at het = 0.25 there is no flip, only a 30% bias. Whether real creative-effect heterogeneity is large enough for flips is an empirical question this dive cannot answer; Pal & Susarla's live campaign found the algorithmic channel moved composition (+2.27 pp female share) while the visible creative did not, which is consistent with meaningful but not necessarily sign-flipping divergence.
- *"Negative control."* The spurious 5.5 requires B's scores to be substantially noisier (0.8 vs 0.3 log-sd). That is plausible for cold-start creatives, but for two mature creatives the pure-delivery ITTdiff is ~0 by symmetry (E9 row 1: 0.01). The control demonstrates the mechanism, not its typical size.
- *"Λ-bounds are vacuous."* These are worst-case bounds that ignore the distribution of δ_A within segment; a Dorn–Guo-style sharp bound using quantiles of δ̃_A could be tighter. Also the realized Λ is inflated by my independent-score-noise assumption; if platform score noise is highly correlated across creatives, within-segment ratios are far less dispersed. The result is "segments + worst-case Λ cannot work," not "no segment-based bound can."
- *"Overlap policy is free."* Precision parity was measured in one DGP family where reaches are similar (0.26 each). With very different reaches (one creative eligible for far more inventory), π_ov shrinks toward the smaller reach and answers a narrower question.
- *"Conversion tests are noise-dominated until 10^8."* Depends on base rate 2% and lift 15%; at 10% base rates the crossover is ~25× lower. The point is the *ratio* bias/sd, which the identity makes computable ex ante from reach divergence and δ heterogeneity — not the particular N.

## 6. Limitations and failure modes

- **Needs the platform to log both creatives' scores per auction.** This does not exist today; it is a proposal (like Pal & Susarla's third arm, but cheaper: a log field, not an arm). Until then the practical advice is Meta's (harmonize configuration) plus reporting τ(π_ov)-style reasoning only qualitatively.
- **Ratio-balancing is untestable from the logs alone.** Arm-asymmetric, δ-correlated unlogged delivery inputs (E14) bias the estimator without warning.
- **Noisy or re-scored logs attenuate** (E6): the score must be the one used at decision time; SIMEX is a partial patch.
- **Positivity** needs the eligibility bit or the estimand silently extrapolates (E7).
- **Non-stationarity** needs period keys (E8); a bandit whose scores depend on realized in-arm outcomes is covered as long as the logged score is the delivery score, but the *shared* refit shocks must be keyed.
- **Power.** For conversion outcomes at typical N the fixed-policy estimand — like ITTdiff itself — is far below detectability; the estimator is a bias fix, not a precision gain. Its use case is CTR/engagement outcomes and very large tests.
- **DGP scope.** One family (lognormal heterogeneity, logistic delivery on standardized scores, Bernoulli outcomes). Not tested: multiplicative (relative) creative effects, multi-exposure/frequency, budget coupling across arms (E2's closed loop), interference across advertisers (M4).

## 7. What is established vs derived vs speculative

- *Established (prior art):* divergent delivery and its scale; mediator bias of naive audience adjustment; three-arm decomposition (Pal & Susarla); Λ-sensitivity OPE machinery.
- *Derived here (verified):* the divergence-bias identity; the exact error decomposition; the ratio-balancing sufficiency theorem for cross-score logs and the failure of single-score logs; the negative control mechanism (asymmetric score noise ⇒ spurious ITT); the bias/noise crossover structure; vacuity of worst-case segment Λ-bounds in this DGP family; π_ov's precision/coarseness robustness (empirical).
- *Speculative / design:* the logging spec hierarchy as a platform product; SIMEX as an interim repair; the recommendation to report τ(π_ov) as the headline creative effect.

## 8. Next steps for a future session

1. **Sharp Λ-bounds (BL29):** replace the worst-case bound with Dorn–Guo quantile-balancing sharp bounds using the within-segment distribution of exposed outcomes; test whether any advertiser-side bound is informative at Λ95 ≈ 5.
2. **Specification test for ratio-balancing:** a Hausman-style comparison of τ̂ across nested cell keys (scores; scores × placement; scores × time) with a derived null distribution — the falsifiability hook M3 lacks (links BL4).
3. **Budget-coupled arms (E2):** A/B arms share a budget/pacing controller; π_A depends on B's performance. Does ratio-balancing survive, and what extra key (pacing multiplier) restores it?
4. **Multiplicative-effect DGP and frequency:** the identity holds for additive δ; with frequency-dependent response the "fixed policy" must fix a *frequency distribution* — define τ(π) over exposure counts and check whether cross-scores + logged frequency suffice.
5. **Pal–Susarla bridge:** their NDE equals τ(π_ctrl-creative) in this notation; show the third arm is a special case of L2 logging (scores from an arm that never shows A) and compare precision of "extra arm" vs "extra log field" at matched budget.
6. **Field pilot design:** a within-platform test where an advertiser has both scores exported at bucket granularity (some clean rooms expose "estimated action rate" deciles) — the first real τ(π_ov) estimate.

## 9. Sources

- Internal: dive 02 (transport across policies), dive 06 (aggregation/Jensen algebra), catalogs `../02_open_questions.md` §M3/§E2, `../03_mmm_adoption_barriers.md`.
- [Braun & Schwartz 2025, *J. Marketing*, "Where A-B Testing Goes Wrong"](https://journals.sagepub.com/doi/10.1177/00222429241275886) ([SSRN 3896024](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3896024)) · [Burtch, Moakler, Gordon, Zhang & Hill 2025, arXiv 2508.21251](https://arxiv.org/abs/2508.21251) · [Ali et al. 2019, arXiv 1904.02095](https://arxiv.org/abs/1904.02095) · [Pal & Susarla 2026, arXiv 2605.23706](https://arxiv.org/html/2605.23706) (recency flagged by the subagent; abstract-level verification only) · [Gordon, Zettelmeyer, Bhargava & Chapsky 2019](https://pubsonline.informs.org/doi/10.1287/mksc.2018.1135) · [Waisman & Gordon 2023, arXiv 2302.13857](https://arxiv.org/abs/2302.13857) · [Johnson, Lewis & Nubbemeyer, Ghost Ads](https://conference.nber.org/confer/2016/EoDs16/Johnson_Lewis_Nubbemeyer.pdf)
- OPE / sensitivity: [Hirano, Imbens & Ridder 2003](https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00442) · [Hanna, Niekum & Stone 2019](https://arxiv.org/abs/1806.01347) · [Namkoong et al. 2020](https://arxiv.org/abs/2003.05623) · [Kallus & Zhou 2018](https://arxiv.org/abs/1805.08593) / [2020](https://proceedings.neurips.cc/paper/2020/hash/fd4f21f2556dad0ea8b7a5c04eabebda-Abstract.html) · [Swaminathan et al. 2017](https://www.cs.cornell.edu/~adith/docs/Slates.pdf) · Rosenbaum & Rubin 1983 (balancing scores) · Li, Morgan & Zaslavsky 2018 (overlap weights) · Cook & Stefanski 1994 (SIMEX) · Dorn & Guo 2022 (sharp Λ-bounds)

*Code:* `code/10_algorithmic_delivery_estimands.py` (`python 10_algorithmic_delivery_estimands.py quick` reproduces the sign flip, cell estimator, negative control and finite-N numbers in ~1 min; `e4_coarseness` … `e15_K_selection` run the individual experiments).
