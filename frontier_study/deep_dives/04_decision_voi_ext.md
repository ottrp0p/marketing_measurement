# Deep dive 04 — EXTEND of dive 03: the anatomy and the edge of the EVSI power plateau (M10, BL5, BL6)

**Queue item:** A4 (EXTEND; strongest thread = dive 03 next-steps 1, 3, 4) · **Date:** 2026-08-31 · **Code:** `code/04_decision_voi_ext.py` (staged: run `python3 04_decision_voi_ext.py setup f1 mc0 mc1 mc2 f1c f2 f2b f3 f3mc f4 f4mc f5 f6 f6mc`; state in `/tmp/d04_state.pkl`)

## 1. Problem statement

Dive 03 found that among well-powered single-channel pulse experiments, exact EVSI for the budget-allocation decision is nearly design-invariant (the "power plateau"), that its own rank-1 decision-aligned closed form badly misranks designs (Spearman 0.33 vs exact MC), and diagnosed the gap as a "nonlinear conditioning bonus no gradient criterion can score." Three questions were left open: **(BL5)** where does the plateau end — is there any design whose exact EVSI beats a powered pulse by a material factor? **(BL6)** is there a cheap corrected scorer that ranks designs like exact nested MC does? And (dive 03 step 4) does a free total budget change the answers? Setup, model, posterior, and experiment operator are exactly dive 03's (J=3 channels, 12-dim Laplace posterior from a 104-week collinear observational fit, scalar geo-lift readouts through dive 02's operator, per-week readout noise 0.03).

## 2. Prior state

From dive 03 (internal): the LQ regret identity EVSI(d) ≈ ½·gᵀΣDᵀΩDΣg/(se²+gᵀΣg); exact nested-MC machinery with split-sample debiasing; the plateau (E5/E7) and the misranking of the rank-1 score. From the literature: [Strong, Oakley & Brennan 2015](https://pubmed.ncbi.nlm.nih.gov/25810269/) ([earlier open version](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3980703/)) estimate EVSI by nonparametric regression of *net benefit* on a low-dimensional summary of the data, for **discrete** treatment choices in health economics; the [voi R package](https://search.r-project.org/CRAN/refmans/voi/html/evsi.html) implements it. No transport of this idea to continuous budget allocation (a simplex-constrained continuum of actions) exists; goal-oriented OED (Attia et al.; GoBOED, cited in dive 03) scores designs by gradient criteria and inherits exactly the rank-1 failure documented there.

## 3. Constructions developed in this dive

### 3.1 The LQC estimator (novel; = Strong–Oakley regression EVSI transported to continuous allocation)

The obstruction to fast EVSI in dive 03 was the inner **optimization** per outer draw, not the inner *integration*. But dive 03's own LQ analysis says the decision layer consumes θ only through the marginal-ROAS vector at the incumbent optimum, u(θ) = m(b̄,θ) ∈ ℝᴶ, with quadratic value ½ΔuᵀΩΔu, Ω = H⁻¹ − H⁻¹𝟙𝟙ᵀH⁻¹/𝟙ᵀH⁻¹𝟙. So: keep the conditioning **exact** and make only the decision layer quadratic. With a shared posterior sample {θᵢ}, precompute uᵢ and readout means Lᵢ (both cheap forward maps, no refitting); for each simulated outcome ℓ form self-normalized importance weights wᵢ ∝ exp(−½‖(Lᵢ−ℓ)/se‖²) and

$$\widehat{\text{EVSI}}_{\text{LQC}}(d) = \mathbb E_\ell\Big[\tfrac12\,(\hat u^e_\ell-\bar u)^\top\, \Omega\,(\hat u^o_\ell-\bar u)\Big],\qquad \hat u^{e/o}_\ell=\textstyle\sum w_i u_i \text{ over even/odd halves},$$

the even/odd cross-product removing the self-normalized-IS noise bias (the LQ analog of dive 03's split-sample debiasing). Cost: O(n) vector ops per outer draw — **no nested optimization**. Multivariate readouts (trajectories, simultaneous experiments) come for free through the weight kernel. Relative to Strong–Oakley: they regress the per-action net benefit (finite actions); LQC regresses the *decision-sufficient statistic* u and closes the continuum of actions analytically through Ω. EVPI_LQ = ½·E[(u−ū)ᵀΩ(u−ū)] is the same functional at perfect information; κ(d) = EVSI/EVPI_LQ is the **capture fraction**.

### 3.2 The u-space Gaussian score, and a re-diagnosis of dive 03's E7b (novel)

Linear-Gaussian conditioning *in u-space with exact nonlinear moments* gives the closed form

$$\text{EVSI}_{\text{gauss-u}}(d)=\tfrac12\,c^\top\Omega\,c,\qquad c=\frac{\operatorname{Cov}(u,L)}{\sqrt{\operatorname{Var}(L)+se^2}},$$

with Cov(u,L), Var(L) sample moments of the shared draw (microseconds per design). §4-F2b shows this matches LQC within ±6% on powered designs. **Consequence:** dive 03's diagnosis was wrong in an instructive way. The rank-1 θ-space formula fails not because posterior conditioning is non-Gaussian ("nonlinear conditioning bonus") but because its ingredients gᵀΣDᵀ are *linearizations of the two forward maps*: over an MMM-realistic posterior, Cov(L,u) ≠ gᵀΣDᵀ. Replace linearized moments by exact sample moments and Gaussian conditioning is already enough. This resolves **BL6** negatively-and-better: no Hessian/sigma-point machinery is needed; the correction is a covariance estimate, not a higher-order expansion.

### 3.3 Plateau anatomy: an alignment bound and a share-ceiling law (derived here)

Let A = Ω^{1/2}·Cov(u)·Ω^{1/2} with eigenvalues λ₁ ≥ λ₂ (λ₃ = 0 since Ω𝟙 = 0): the **decision-relevant spectrum**. For any *scalar* readout, Var(E[u|ℓ]) is rank one, and Cauchy–Schwarz gives

$$\kappa_{\text{scalar}}(d)\;\le\;\rho^2(d)\,\frac{\lambda_1}{\lambda_1+\lambda_2},\qquad \rho^2 = \text{(squared correlation of the standardized readout with its best-aligned contrast)},$$

with equality iff the readout is perfectly aligned with the top eigencontrast c₁. Under an independent-channels posterior, a channel-j readout can move only u_j, giving the sharper **share ceiling** κ_j ≤ s_j = Ω_jj·Var(u_j)/tr(Ω·Cov u). The plateau is therefore not flatness but a **ceiling structure**: powered designs sit near their ceilings; ceilings of the conventional design family (single-channel pulses) happen to be comparable (s_j ≈ 0.25–0.43). Two escape routes follow immediately: add readouts (κ of a k-readout design is bounded by the top-k share sum), or **realign one readout with c₁** — and c₁ = (1, −0.33, −0.67) across channels is a *reallocation* direction, which no single-channel test can align with, but a **budget-neutral switch experiment** (+δ on one channel, −δ′ on another, one incrementality readout) can. Both are tested below.

## 4. Verification (all numbers from `code/04_decision_voi_ext.py`, CRN outer draws shared across all designs; nested MC uses 240 outer × 6000 inner draws with warm-started inner optimization)

**F1 — LQC is validated; the rank-1 score is refuted as a ranker.** Baseline posterior, 36-design grid + 3 weak controls in 4.7 s. Against exact nested MC on 7 CRN-matched designs spanning the grid: LQC Pearson **0.974**, Spearman **1.000**, median MC/LQC = 1.33 (the LQ conservatism factor, consistent with dive 03's E2b ratio at this width). The rank-1 closed form on the same designs: Pearson 0.31, Spearman 0.32, level ratio 2.4. Controls: weak pulse 0.00106, 6×-noise 0.00131, tiny-short 0.00066 vs ~0.006–0.009 for powered designs — the 16σ power cliff reproduced.

**F1b — the plateau is a band, not a line.** Among the 12 powered designs (δ ≥ 0.4x̄, Te = 8), LQC max/min = **1.65** (κ from 0.22 to 0.36). Dive 03's "tie" was real but an artifact of dueling two designs that both sit mid-band; the full grid has structure the two-design duel couldn't see (MC agrees: 0.00701 vs 0.00929 across the band ends it measured).

**F2 — anatomy.** Baseline spectrum: λ = (0.0257, 0.0182), **effective rank 1.94**, top-contrast share λ₁/tr = 0.59. The initial conjecture (plateau ⇔ effective rank ≈ 1) is **rejected by this run's own numbers** — decision uncertainty is genuinely two-dimensional here. The plateau instead comes from correlation spreading: every powered channel readout loads on *both* contrasts (e.g. ch0: corr(L, c₁ᵀu) = +0.78, corr(L, c₂ᵀu) = +0.28), so all single-channel κ land in a compressed band.

**F2b — the re-diagnosis, verified.** gauss-u vs LQC on five powered designs: differences −2% to +6%. The "nonlinear conditioning bonus" is ≈ 0 *in u-space*; the θ-space formula's error was moment error, as claimed in §3.2.

**F3 — share ceiling, verified.** Block-diagonalized posterior (channels independent; effrank 1.98): κ_LQC = (0.26, 0.23, 0.29) vs ceilings s_j = (0.32, 0.25, 0.43) — all below, with the shortfall largest for ch2 (α = 0.8; its 16-week window truncates the long adstock tail, ρ² lowest). Variance-equalized variant: κ = (0.35, 0.31, 0.19) vs s = (0.41, 0.34, 0.25). Channel-choice ratios 1.25 and 1.85. Nested MC on the raw-block pair declared best/worst by LQC returned a statistical tie (0.00806 vs 0.00842) — within-band orderings of ~1.2× are below both methods' resolution; only the band structure itself is established.

**F4 — where the plateau ends, route 1: more readouts.** Trajectory readouts are worthless for this decision: κ 0.29 → 0.30 (baseline), 0.19 → 0.19 (rank-2 posterior) — the rank-4 within-channel information dive 02 found does not project onto cross-channel contrasts. Simultaneous two-channel designs break the ceiling exactly as the share law predicts: baseline dual ch2&ch1 κ = **0.52** vs 0.29 scalar; equalized posterior duals κ = 0.48–0.65 vs 0.35 best single. Nested-MC confirmation (CRN): dual 0.01497 ± 0.00106 vs best single 0.01015 ± 0.00084 — **+47%**, the ">2×" regime dive 03 asked about exists across *experiment count*, never within single-scalar designs.

**F5 — free budget: conjecture refuted.** EVPI_LQ rises ×4.0 (the level direction is live, per dive 03's E3), but design rankings barely move (Spearman 0.86 vs fixed-budget), the best design is unchanged, and the powered-design band is *tighter* (max/min 1.26). Reason: every powered readout carries level information; the level direction is the easiest to inform, so it differentiates designs even less. Dive 03's step-4 conjecture (D-optimality and ROI-level calibration rehabilitate under free budgets) is **refuted** in this model family.

**F6 — where the plateau ends, route 2: the switch experiment (the operational novelty).** A budget-neutral reallocation pulse (+0.5 on ch0, −0.6 on ch2, one scalar readout) achieves corr(L, c₁ᵀu) = 0.83 and κ = **0.47**, vs 0.36 for the best single-channel design and the 0.59 scalar bound; sign-reversed switch identical (0.47), as theory requires. Nested MC: switch **0.01231 ± 0.00106** vs best conventional 0.01011 ± 0.00086 (CRN; +22%, LQC predicted +31%). **New failure mode found:** the switch is nonmonotone in size — doubling it (+1.0/−1.2) *collapses* κ to 0.21 (MC confirms: 0.00556), because large opposed pulses drive the two Hill responses into asymmetric saturation and the readout decorrelates from the contrast it was built to measure. Switch designs must be sized, not maxed.

## 5. Limitations and failure modes

All results are within dive 03's frame: correctly specified model, Laplace posterior centered at truth, steady-state per-period revenue, se(d) assumed rather than derived from geo counts (cost fairness between one dual experiment and one single experiment is therefore unaddressed — a dual is plausibly ~2× the cost, which would erase its EVSI advantage; that is BL7's question, still open). LQC inherits the LQ decision layer: its levels are ~25% conservative at full width (MC/LQC 1.33) and its within-band orderings (≲1.3×) are not confirmable by MC at feasible outer counts; only cross-band claims (power cliff, dual +47%, switch +22%, oversized-switch collapse −45%) are MC-established. The switch experiment as modeled reads out net incremental revenue of a reallocation across geos; operationally that requires the two channels' geo assignments to coincide, which real ad platforms only partially allow. Effective-rank and share numbers are properties of this 3-channel truth and will vary; the *laws* (alignment bound, share ceiling, count-additivity) are the portable content.

## 6. Next steps for a future session

1. **Optimal switch sizing (new):** derive the collapse point of §4-F6 analytically — the δ at which Cov(L_switch, c₁ᵀu) turns over, as a function of Hill curvature at the two operating points; produce a sizing rule (conjecture: δ* ≈ the size where the two channels' second-order lift terms are equal and opposite).
2. **The k-experiment portfolio law:** κ(portfolio) ≈ sum of captured shares suggests greedy share-hunting; formalize and test a portfolio version of dive 03's E6 scheduler using LQC as the scorer (feeds A5 directly — LQC is fast enough to be the production scorer in the always-on loop).
3. **Power–cost frontier (BL7):** replace se(d) with a geo-count power model and charge duals/switches their true cost; the dual-vs-single and switch-vs-single margins here (+47%, +22%) are the numbers that cost model must beat.
4. **LQC under misspecification (links BL3):** posterior centered off-truth via a wrong-adstock operator calibration; does LQC still rank designs correctly when u(θ) is computed under the wrong model?
5. Formalize §3.3 equality conditions and the count-additivity bound (κ of k independent readouts ≤ Σ top-k eigenshare) as propositions with proofs.

## Sources

- [Strong, Oakley & Brennan 2015, regression-based EVSI from the PSA sample, *Medical Decision Making*](https://pubmed.ncbi.nlm.nih.gov/25810269/) · [Strong et al., non-parametric regression EVSI (open access)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3980703/) · [voi R package, evsi()](https://search.r-project.org/CRAN/refmans/voi/html/evsi.html)
- [Heath et al. 2019, EVSI by moment matching](https://journals.sagepub.com/doi/10.1177/0272989X19837983) · [Li, Jalal & Heath 2024, Gaussian-approximation EVSI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11492544/) (the u-space gauss formula of §3.2 is the allocation-decision analog of this family)
- Internal: dive 03 (LQ regret identity, nested-MC machinery, the plateau result being extended), dive 02 (operator forward map), dive 01 (design/FIM machinery); catalogs `../02_open_questions.md` M10, `../03_mmm_adoption_barriers.md` §B5/§H8
