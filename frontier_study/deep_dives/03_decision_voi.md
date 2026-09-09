# Deep dive 03 — Decision-calibrated uncertainty & value of information (M10)

**Queue item:** A3 (M10, `02_open_questions.md`) · **Date:** 2026-08-31 · **Code:** `code/03_decision_voi.py`

## 1. Problem statement

An MMM posterior is only worth what it changes about the budget. M10 asks: (a) is Bayes-optimal allocation under the joint posterior materially better than the plug-in allocation everyone actually computes, and (b) which next experiment maximally reduces expected budget-decision regret? Formalize both, derive closed-form decision-value criteria, and test the natural construction — decision-aligned experiment selection — against exact preposterior Monte Carlo.

**Setup.** $J=3$ channels, model per dives 01–02: $y_t = \sum_j \beta_j h(a_{jt};K_j,S_j)+\varepsilon_t$, geometric adstock $\alpha_j$, Hill $h$, $\sigma=0.05$. Truth $\theta_0$: ch1 $(1.0,1.5,2.0,0.6)$, ch2 $(0.8,1.0,1.8,0.3)$, ch3 $(1.3,2.5,2.2,0.8)$; baseline spends $\bar x=(1.0,0.8,1.2)$, fixed total budget $B=3$. Steady-state revenue $R(b,\theta)=\sum_j \beta_j h(b_j/(1-\alpha_j))$; with $B$ fixed the decision is $b^\star=\arg\max_b \mathbb E_\pi[R(b,\theta)]$ on the simplex. Decision value of information is measured in expected revenue per period.

**Posterior.** Laplace posterior from a realistic observational fit: $T=104$ weeks, correlated AR(1) spends (±10 % swings, shared demand cycle), $\Sigma = (\mathcal J^\top\mathcal J/\sigma^2 + \Lambda_{\text{prior}})^{-1}$ over the 12-dim $\theta$, weak 50 %-scale prior. Resulting widths are MMM-realistic: marginal relative sds 11–49 %, max cross-channel correlation **0.71** (the "rank-correlated uncertainty" M10 names).

**Experiment model.** Candidate designs $d=(j,\delta,T_e,P)$ read out a scalar lift $\hat L\sim\mathcal N(\mathcal E_d(\theta), se^2)$ through dive 02's operator forward map $\mathcal E_d$, with $se = 0.03\sqrt{T_e+P}$ (window-accumulated noise; assumption, stated not derived).

## 2. Prior state

**Established.** [Test & Roll (Feit & Berman 2019)](https://pubsonline.informs.org/doi/10.1287/mksc.2019.1194) solves profit-maximizing test sizing exactly for the two-arm normal-normal case; M10's multi-channel curve-level version is open. EVSI computation is a mature literature in health economics — [nested Monte Carlo tutorials](https://www.sciencedirect.com/science/article/pii/S1098301518322010), [moment matching (Heath et al. 2019)](https://journals.sagepub.com/doi/10.1177/0272989X19837983), [Gaussian approximations for nonlinear models (Li et al. 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11492544/) — but has not been connected to MMM experiment scheduling. Goal-oriented OED ([Attia et al. 2018](https://arxiv.org/abs/1802.06517); [GoBOED, arXiv:2605.26093](https://arxiv.org/abs/2605.26093)) optimizes designs for a quantity of interest or decision layer rather than parameters, in inverse-problem settings; the message "parameter uncertainty ≠ decision quality" is the same one derived here, never instantiated for media budgets. Dive 01 built the FIM machinery for MMM designs (D-criterion); dive 02 built $\mathcal E_d(\theta)$ and flagged this dive as its consumer.

## 3. Theory developed in this dive

### 3.1 Quadratic regret identity and the EVPI trace formula

At an interior optimum the marginal ROAS $m_j(b_j,\theta)=\beta_j h'(b_j/(1-\alpha_j))/(1-\alpha_j)$ are equalized. Perturb $\theta$ by $d\theta$: the induced shift in per-channel marginal ROAS is $u = D\,d\theta$ with $D = \partial m/\partial\theta \in\mathbb R^{J\times 12}$ evaluated at $(\bar b,\bar\theta)$. Solving the displaced first-order conditions on the simplex with curvatures $H_j = -\partial m_j/\partial b_j>0$ gives the regret of holding $\bar b$:

$$R \approx \tfrac12\, u^\top \Omega\, u, \qquad \Omega = H^{-1} - \frac{H^{-1}\mathbf 1\mathbf 1^\top H^{-1}}{\mathbf 1^\top H^{-1}\mathbf 1},$$

hence (derived here, standard LQ preposterior logic in a new instantiation)

$$\text{EVPI} \approx \tfrac12\operatorname{tr}\!\big(\Omega\, D\,\Sigma\, D^\top\big), \qquad \text{EVSI}(d) \approx \tfrac12\,\frac{g^\top\Sigma D^\top \Omega\, D\,\Sigma\, g}{se^2 + g^\top\Sigma g},\quad g=\nabla_\theta\mathcal E_d(\bar\theta),$$

the EVSI form being the exact rank-1 posterior-covariance update for a scalar readout — a *decision-aligned c-optimality* score, computable in microseconds per design, vs the information-gain (D-)score $\tfrac12\log(1+g^\top\Sigma g/se^2)$. Note $\Omega\mathbf 1 = 0$: only cross-channel *contrasts* of marginal ROAS carry decision value.

### 3.2 A decision-null theorem (exact, not just second-order)

**Proposition.** Under a fixed total budget, a common multiplicative error in all effectiveness parameters ($\beta_j \to \beta_j(1+s)$, any $s>-1$) is exactly decision-null: it rescales $R(b,\theta)$ by $(1+s)$ and leaves $\arg\max_b$ unchanged. *Proof: revenue is linear in $\beta$; the shock factors out of the objective.* ∎

Corollary via dive 02: the industry-default calibration target — the average-ROI *level*, which is what Meridian-style iROAS→ROI priors chiefly move — is nearly worthless for fixed-budget allocation. Decision value lives in contrasts and curvature. (The null breaks when total budget is also chosen: then the level sets the spend margin.)

### 3.3 The two halves of M10, weighed

M10(a) is the stochastic program (Bayes vs plug-in allocation); M10(b) is the information problem (which experiment next). §4 shows (a) is worth ~0.03 % of revenue while the information gap (EVPI) is ~1.8 % — sixty-fold. The Bayes-allocation refinement is a rounding error; the experiment scheduler is the asset. This inverts the emphasis in M10's own phrasing.

## 4. Verification (all numbers from `code/03_decision_voi.py`, seeds fixed)

**E2 — plug-in ≈ Bayes; EVPI is large.** $b_{\text{plug}}=(1.027,0.919,1.054)$ vs $b_{\text{Bayes}}=(0.988,0.941,1.071)$; expected-revenue gap **6.4×10⁻⁴** (0.027 % of 2.381). Exact EVPI by per-draw oracle optimization: **0.0420 ± 0.0036** = **1.76 % of revenue**. Quadratic formula: 0.0151.

**E2b — the trace formula is verified asymptotically and conservative at realistic width.** Shrinking $\Sigma$ by scale $s\in\{1,\tfrac12,\tfrac14\}$: MC/quad ratio = **3.37 → 1.69 → 1.27** (quad scales exactly as $s^2$). The LQ theory is correct locally; at MMM-realistic widths, higher-order value it can't see is ~⅔ of the total.

**E3 — decision-null direction confirmed.** With CRN, adding a 30 %-sd common multiplicative $\beta$ shock moves exact EVPI 0.0444→0.0474 (within noise), while an equal-magnitude $\beta$-contrast shock (ch1 up/ch3 down) moves it to **0.0689 (+55 %)**. Free-budget variant: the common shock now adds real value (0.304→0.414), as predicted.

**E4 — closed-form EVSI vs exact nested MC** (split-sample inner optimization to kill winner's-curse bias, CRN outer draws, ESS 500–2600). Quarter-width (LQ regime): Pearson **0.945**, Spearman 0.93, level ratio MC/closed 1.16 — the formula is right where its assumptions hold. Full width: levels understated ~2.2×, rank correlation collapses (Spearman 0.33).

**E5/E7 — the headline negative result.** The closed form predicts decision-aligned selection beats D-optimality decisively (Kendall τ(EVSI, D-score) = 0.23; D-best design forfeits 42 % of decision value; at quarter width τ = 0.84, predicted forfeit 17 %). **Exact nested MC refutes the forfeit at both widths.** Full-width duel, EVSI-best (ch3, δ=½x̄, 8 wk + 8 wk post) vs D-best (ch2, δ=x̄, 8 wk, no post): 0.00721±0.00040 vs 0.00758±0.00055, paired diff −0.00037±0.00067 — a tie. Quarter-width duel: 0.000155 vs 0.000157, paired −2±13 ×10⁻⁶ — a tie, with the D-best design beating its own rank-1 prediction (0.000157 actual vs 0.000121 predicted). Diagnosis: a large-pulse experiment is *nonlinearly* informative — conditioning on its outcome constrains the response surface beyond the gradient direction $g$, and this off-direction information happens to fill most of the decision-relevant gap that design shifts open. No gradient-based criterion (c- or D-) can score it.

**E7c — but power discriminates enormously (MC sensitivity control).** Same machinery, weak designs: tenth-size pulse **0.00108**, 6× noise **0.00134**, tiny-short test **0.00079** — vs 0.00721 for the strong design (paired diff 0.00612±0.00038, ~16σ). The instrument is sharp; the plateau among well-powered designs is real.

**The operational finding (novel, verified within the tested design family):** EVSI has a **power plateau**. Signal-to-noise determines decision value almost entirely; among adequately powered single-channel pulse designs, the choices practitioners agonize over — which channel, pulse size, post-period — moved exact EVSI by less than MC resolution (~10 %), at both realistic and tight posterior widths. One well-powered experiment recovers **15–18 % of EVPI**. Reconciliation with dive 01: design determines *identification of the full parameter vector* (there, λ_min swings ×700), but the decision consumes only $J-1$ contrast directions, which almost any powered readout informs through the posterior's correlation structure.

**E6 — greedy scheduler** (rank-1 updates, LQ scoring): runs ch3 → ch2 → ch1, residual quadratic EVPI 69 %→51 %→36 %, then saturates (34 %, 31 %) — diminishing returns after one experiment per channel; feeds A5's always-on loop.

## 5. Limitations and failure modes

The plateau is demonstrated for scalar-readout, single-channel uniform-pulse designs with a correctly specified model and a Gaussian (Laplace) posterior centered at truth; richer readouts (trajectories — dive 02 showed rank-4 content), multi-channel simultaneous designs, or misspecification-driven posterior skew could break it. $se(d)$ is assumed, not derived from a geo-count/power model, so "cost of power" trade-offs (Test & Roll's territory) are outside scope. The Laplace posterior truncated to the admissible region is not the true posterior of the observational fit. EVPI/EVSI are per-period steady-state revenue units; converting to dollars needs a horizon and discounting. Nested-MC EVSI, though split-sample-debiased, retains the importance-sampling approximation of the conditional posterior.

## 6. Next steps for a future session

1. **Where does the plateau end?** Trajectory readouts (rank-4) vs scalar, simultaneous multi-channel designs, and misspecified-model posteriors — find a regime where design choice moves exact EVSI by >2×, or conjecture there isn't one within powered designs.
2. **Power–cost frontier:** replace assumed $se(d)$ with a geo-assignment power model (number of geos, test length, interference), making EVSI-minus-cost the objective — the true Test & Roll generalization M10 asks for.
3. **Beyond-gradient EVSI scoring:** a cheap second-order (Hessian/sigma-point) correction to the rank-1 formula that captures the nonlinear conditioning bonus E7b exposed; validate against nested MC.
4. **Variable-budget decision:** with total spend free, level uncertainty is live again (E3) — redo the design comparison there; conjecture: ROI-level calibration and D-optimality both rehabilitate partially.
5. **Feed A5:** wrap E6's scheduler + E7's exact evaluator into the closed-loop architecture (uncertainty-driven scheduling, automated ingestion via dive 02's operator).

## Sources

- [Feit & Berman 2019, Test & Roll: Profit-Maximizing A/B Tests, *Marketing Science*](https://pubsonline.informs.org/doi/10.1287/mksc.2019.1194) · [arXiv version](https://arxiv.org/pdf/1811.00457)
- [Strong & Oakley et al., EVSI via nested Monte Carlo: a tutorial, *Value in Health*](https://www.sciencedirect.com/science/article/pii/S1098301518322010) · [Heath et al. 2019, EVSI by moment matching](https://journals.sagepub.com/doi/10.1177/0272989X19837983) · [Li, Jalal & Heath 2024, Gaussian-approximation EVSI for nonlinear models](https://pmc.ncbi.nlm.nih.gov/articles/PMC11492544/)
- [Attia, Alexanderian & Saibaba 2018, Goal-oriented optimal design of experiments, arXiv:1802.06517](https://arxiv.org/abs/1802.06517) · [GoBOED: goal-driven Bayesian OED for decision-making, arXiv:2605.26093](https://arxiv.org/abs/2605.26093)
- Internal: dive 01 (FIM/design machinery, D-criterion results), dive 02 (operator forward map $\mathcal E_d$, rank-1 vs rank-4 readouts); catalogs `../02_open_questions.md` M10, `../03_mmm_adoption_barriers.md` §B5, §H8
