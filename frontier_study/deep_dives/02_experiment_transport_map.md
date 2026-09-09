# Deep dive 02 — A formal transport map from lift tests to MMM parameters (M2)

**Queue item:** A2 (M2, `02_open_questions.md`) · **Date:** 2026-08-31 · **Code:** `code/02_experiment_transport_map.py`

## 1. Problem statement

An incrementality experiment measures a finite-horizon ITT effect of a specific spend delta, at a specific operating point, in a specific subpopulation, over a specific window. An MMM parameter is a steady-state structural quantity for the target population. Calibration practice equates the two. Formalize the gap: instantiate the Pearl–Bareinboim selection diagram for experiment→MMM transport, derive the exact map between the experimental functional and the model functionals, characterize the correction terms, and construct a calibration mechanism that is correct by construction. Model as dive 01: $y_t = \beta h(a_t;K,S)+\varepsilon_t$, $a_t = x_t + \alpha a_{t-1}$, Hill $h$; working point $\theta_0=(1,1.5,2,0.6)$, $\bar x = 1$, $\sigma=0.05$.

Define the **experiment functional** for design $d = (\bar x_e, \{\delta_t\}_{t<T_e}, W = T_e + P)$ (baseline spend $\bar x_e$, pulse $\delta$ for $T_e$ periods, measurement window extending $P$ periods past pulse end):

$$\mathcal E_d(\theta) \;=\; \beta_e \sum_{t \in W} \Big[ h\big(a_t(x_e+\delta)\big) - h\big(a_t(x_e)\big) \Big], \qquad \text{iROAS}(d) = \frac{\mathcal E_d(\theta)}{\sum_t \delta_t}.$$

The **model functionals** a business consumes: steady-state marginal ROAS $m_\infty(\theta) = \beta_\star h'(\bar a_\star)/(1-\alpha)$ and average ROAS $\beta_\star h(\bar a_\star)/\bar x_\star$.

## 2. Prior state

**Established.** Meridian calibrates by using an experiment's iROAS point estimate/SE as the mean/sd of a prior on the model's *ROI (average ROAS)* parameter, with docs conceding "there is no single formula to translate an experiment result into a prior" ([Meridian calibration docs](https://developers.google.com/meridian/docs/advanced-modeling/roi-priors-and-calibration)). [PyMC-Marketing's lift-test likelihood](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_roas.html) is the strongest prior art: a lift test enters as a Normal likelihood on the *static saturation-curve secant* $\beta[s(x+\Delta x)-s(x)]$ — exact on curvature, but with no adstock dynamics, no finite-window truncation, and no population selection; [Orduz's simulation study](https://juanitorduz.github.io/mmm_roas/) compares ROAS-parametrized calibration. Practitioner literature documents the symptom without the mechanism: "reconciliation headaches rather than trust-building calibration" when experiment KPI/geo/window don't match the model (`../03_mmm_adoption_barriers.md` §O7), and published warnings that experiment-derived priors "transfer bias across non-comparable populations" (§M5). The [Pearl–Bareinboim transportability calculus](https://projecteuclid.org/journals/statistical-science/volume-29/issue-4/External-Validity-From-Do-Calculus-to-Transportability-Across-Populations/10.1214/14-STS486.full) has never been instantiated for this problem. Methodologically, the fix constructed below is a form of **indirect inference** (Gouriéroux–Monfort–Renault): match the model-simulated value of the experiment's estimator, not a renamed parameter — apparently never stated in the MMM calibration literature.

## 3. Theory developed in this dive

### 3.1 The selection diagram, and why all calibration is parametric transport

Domains: $\Pi_\star$ (target: national, business-as-usual spend) and $\Pi_e$ (experiment: subpopulation × time window). Shared causal graph $X_t \to A_t \to Y_t$ with $U \to X_t$, $U \to Y_t$ (demand confounding; severed by $do(X)$ in $\Pi_e$). Selection nodes: $S_\beta \to Y$ (geo effectiveness differs), $S_x \to X$ (operating point differs), $S_g \to Y$ (seasonality/demand differs).

**Negative result (direct from established theory, but worth stating sharply):** because $S_\beta$ points directly into the outcome mechanism, the causal effect of spend is *not nonparametrically transportable* from $\Pi_e$ to $\Pi_\star$ — no do-calculus derivation exists. Every experiment→MMM calibration therefore rides entirely on **parametric invariance assumptions**: the standard (usually implicit) one is *shape invariance* — $(K, S, \alpha)$ are $S$-node-free (shared across domains), while $\beta$ and seasonality carry the $S$-nodes. Under shape invariance the experiment is informative about $\theta$, and all transport is $\theta$-mediated: experiment → likelihood on $\theta$ → any target functional. There is no license for scalar-to-scalar transport (iROAS → model ROI) — that shortcut is what the correction factors below price.

### 3.2 The transport equation (derived here)

Writing $\Delta a_t$ for the adstock difference induced by the pulse and $Q \equiv \sum_t \Delta h_t / (h'(\bar a_e)\sum_t \Delta a_t)$, the identity

$$\boxed{\;\text{iROAS}(d) \;=\; m_\infty(\theta)\;\cdot\;\underbrace{\rho}_{\text{selection}}\;\cdot\;\underbrace{C(\alpha; T_e, P)}_{\text{window capture}}\;\cdot\;\underbrace{Q(\delta, \bar a_e; \theta)}_{\text{curvature/secant}}\;}$$

holds exactly (Q absorbs the remainder by definition; the content is the closed forms), with:

- **Window-capture factor** (closed form derived here; the $T_e=1$ case is folklore): for a uniform pulse,
$$C(\alpha; T_e, P) = 1 - \frac{\alpha^{P+1}\,(1-\alpha^{T_e})}{T_e\,(1-\alpha)}.$$
**Verified (Exp. A):** matches brute-force simulation to $9\times10^{-7}$ across $\alpha \in \{0.3,\dots,0.9\}$, $T_e \in \{1,\dots,13\}$, $P \in \{0,\dots,26\}$. Magnitudes matter: a 4-week test on a channel with $\alpha = 0.8$, analyzed only through test end ($P=0$), has $C = 0.41$ — the experiment *observes 41% of the incremental sales it caused*. At $\alpha=0.9$: $C=0.23$. Common practice (short tests, high-carryover channels, no post-period) bakes a 2.5–4× understatement of the long-run effect into the "ground truth" before any modeling starts.

- **Curvature/secant factor**: second-order expansion derived here for the dynamic case, $Q \approx 1 + \frac{h''(\bar a_e)}{2h'(\bar a_e)}\cdot\frac{\sum_t \Delta a_t^2}{\sum_t \Delta a_t}$ (transient-weighted — the static secant is the $\alpha\to 0$ special case). **Verified (Exp. B):** accurate to <2% for $\delta/\bar x \le 0.25$; for big tests exact evaluation is required (at $\bar a/K = 1.67$, $\delta = \bar x$: $Q_{\text{exact}} = 0.51$ vs 2nd-order 0.17). Sign flips below the Hill inflection $a^\dagger = K((S-1)/(S+1))^{1/S}$: low-spend-geo tests can *over*state the marginal slope ($Q>1$).

- **Selection factor** $\rho = \beta_e h'(\bar a_e) / (\beta_\star h'(\bar a_\star))$: effectiveness ratio × operating-point slope ratio. Even with $\beta_e = \beta_\star$, running the test in low-spend geos ($\bar a_e < \bar a_\star$) makes $\rho > 1$ — the experiment sits on a steeper part of the curve than the national book.

At $\theta_0$ with a $+50\%$, 8-week, $P=0$ test: iROAS $= 0.219$ while $m_\infty = 0.389$ and avg ROAS $= 0.735$. The three numbers a practitioner might equate differ by up to **3.4×** in a fully standard setting.

### 3.3 The operator fix: calibrate on the design, not on a renamed scalar

**Construction (novel for MMM; indirect-inference in spirit).** Add to the MMM posterior the likelihood term
$$\hat L \;\sim\; \mathcal N\big(\mathcal E_d(\theta),\; \text{se}^2\big),$$
where $\mathcal E_d(\theta)$ *simulates the experiment's actual design inside the model* — baseline path, delta path, adstock transient, window truncation, experiment-geo operating point. No correction factors are needed because nothing is transported scalar-to-scalar; $C$, $Q$, $\rho$ are computed implicitly and exactly at every posterior draw. PyMC-Marketing's saturation-secant likelihood is the static ($\alpha=0$, full-window, no-selection) special case. Cost: the experiment report must carry its design metadata, not just (iROAS, se) — see §5.

### 3.4 Verification: the four calibration modes head-to-head (Exp. C)

60 Monte Carlo reps; weakly-identifying business-as-usual national series ($T=104$, AR(1) spend ±10%) plus one well-powered experiment (se = 5% of lift); modes: uncalibrated, **naive-avg** (Meridian-style iROAS→ROI prior), **naive-marg** (iROAS→$m_\infty$ prior), **operator**. Multistart least squares per rep.

| mode | $m_\infty$ median bias | $m_\infty$ IQR | med. max curve err | wrecked-curve reps | med. profit loss |
|---|---|---|---|---|---|
| uncalibrated | −7.4% | 0.108 | 0.262 | 3% | 0.0057 |
| naive-avg | +4.0% | 0.301 | 0.228 | **30%** | **0.1368** |
| naive-marg | **−43.0%** | 0.016 | 0.352 | 12% | 0.0212 |
| operator | −3.8% | 0.093 | **0.124** | 0% | **0.0021** |

Findings, each individually checkable:

1. **The theory predicts the naive bias to within a point:** predicted naive-marg bias $= C\cdot Q\cdot\rho - 1 = -43.8\%$; observed −43.0%.
2. **Naive calibration is *precisely wrong*:** naive-marg has the tightest spread (IQR 0.016, ~7× tighter than anything else) centered 43% off truth. Calibration bought confidence, not accuracy — the quantitative form of the "precision theater" failure (`03_mmm_adoption_barriers.md` §B5), now produced *by the sophisticated method*.
3. **Naive-avg creates likelihood conflict, not bias:** the iROAS (0.219) contradicts the observational sales level ($\beta h(\bar a)=0.735$) that $T=104$ weeks of data pin down; the data win, but the fit escapes along the ridge — 30% of reps end with a wrecked response curve (off by >50% of the sales ceiling somewhere) and median profit loss **65× the operator's**. This is the mechanism behind practitioners' "reconciliation headaches": the two evidence sources are made to disagree *by the transport error itself*.
4. **Operator calibration dominates everywhere:** halves the uncalibrated curve error, zero wrecked reps, near-zero decision loss.

**Operating-point mismatch variant** (test run in geos at 50% of national spend): naive-marg flips to **+67.3%** bias (theory: +72.5%) — same pipeline, opposite sign, depending on where the test happened to run. The operator's curve error *improves* to 0.048 (vs 0.124 matched): through the correct map, an off-operating-point experiment is *more* informative (it probes the curve where it bends); through the naive map it is more toxic. Population/operating-point diversity in a testing program is an asset or a landmine depending solely on the transport machinery.

### 3.5 The scalar summary throws away the experiment's identifying power (Exp. D)

Fisher information at $\theta_0$ from one experiment (8-week pulse + 6 post-periods, per-period differenced noise 0.05): the **full treated-minus-control trajectory** has rank-4 FIM (eigenvalues $2.7\times10^{-5}, 0.059, 7.8, 577$) — one experiment alone carries information about all four parameters, including a standalone CRLB SE of **0.20 for $\alpha$** (its ramp-up and decay are visible). The **scalar iROAS summary** of the same experiment has rank-1 FIM: it constrains one direction in $\theta$-space and contains *zero* information about $\alpha$ separately. Corollary for joint calibration: $k$ scalar summaries constrain $k$ directions — with shape invariance you need ≥3 well-separated designs plus the observational level to point-identify $(\beta,K,S,\alpha)$; trajectory-level calibration needs one. This is dive 01 §3.5 wearing different clothes: the experiment's transient is a free dose-ranging sweep, and the industry's reporting convention (a single iROAS number) discards it.

### 3.6 Joint posteriors over multiple experiments (corollary, no separate sim)

Because each experiment enters as a likelihood row on the *shared* $\theta$, the posterior over any vector of derived ROAS quantities is automatically jointly correlated — the "joint, correlated posterior over the ROAS vector" M2 asks for is not an extra construction but a consequence of refusing to convert experiments into independent per-channel scalar priors. Two experiments on the same channel at different deltas jointly constrain curvature ($K,S$) in a way two independent ROI priors cannot even express (they would simply conflict, and §3.4.3 shows what conflict does).

## 4. Operational construction: the calibration header

For $\mathcal E_d(\theta)$ to be computable, an experiment result must ship with its design. Minimal metadata standard (the "calibration header") — proposed here as a concrete artifact an org or tool could adopt:

```
experiment:
  channel: <model channel id>          kpi: <must match model y definition>
  baseline_spend_path: [per-period, experiment geos]   # not just the mean
  delta_spend_path:    [per-period]
  window: {start, end}                 post_period_len: P
  population: {geo_ids or share-of-country weights, baseline spend share}
  estimate: {L_hat (total incremental KPI), se}        # totals, NOT iROAS
  trajectory: [per-period treated-minus-control, se_t] # optional; §3.5 says: include it
```

Reporting the *total lift with design attached* instead of iROAS makes the result transport-safe; reporting the per-period trajectory upgrades a rank-1 constraint to rank-4. Every field is something the experimenter already has.

## 5. Limitations and failure modes

- **Shape invariance is an assumption, not a result.** If $(K,S,\alpha)$ differ between experiment geos and the nation, the operator transports incorrectly too — but unlike the naive map it makes the assumption *explicit and testable* (two experiments in different geo strata over-determine the shared shape; disagreement is a specification test). Hierarchical $\beta_g$ partially absorbs $S_\beta$; nothing absorbs $S_K$ without multi-stratum experiments.
- **Model misspecification transfers.** $\mathcal E_d(\theta)$ is exact under the model family; if the true adstock is delayed-peak and the model is geometric, operator calibration matches the experiment by distorting $\theta$ (indirect inference under misspecification: pseudo-true values). Naive calibration is *also* wrong in this case; neither is robust. Quantifying this is a next step.
- **Experiment internal validity is taken as given** ($\hat L$ unbiased for the ITT under $do(X)$); contamination, spillover, and control-market disruption (§O7) are upstream of this dive.
- **The $C$ formula assumes geometric adstock and a uniform pulse**; non-uniform $\delta_t$ needs the general per-impulse sum (trivial numerically, no closed form given here). Sales-lag dynamics beyond adstock (pipeline/consideration delays) would enter $C$ identically and are not modeled.
- Exp. C uses MAP-style point fits, not full posteriors; the "precisely wrong" IQR finding should be re-confirmed with MCMC posterior widths (expected to be, if anything, starker).

## 6. Next steps for a future session

1. **Misspecification robustness:** simulate delayed-peak adstock truth fit with geometric-adstock operator calibration; measure pseudo-true bias; try a sandwich/inflated-se operator likelihood as a robustness patch.
2. **Multi-stratum shape-invariance test:** two experiments in different geo strata + shared-shape model → posterior predictive check as a formal test of the transport assumption (the field has no specification test for calibration at all).
3. **Trajectory calibration in a real stack:** prototype `add_experiment_operator()` against PyMC-Marketing's API surface; contribute the dynamic generalization of their static secant likelihood.
4. **Feed A3 (M10/VOI):** $\mathcal E_d(\theta)$ is exactly the forward map a value-of-information design chooser needs — pick the next experiment's $d$ to maximize expected posterior decision value; dive 01's FIM machinery + this dive's operator are the two inner loops.
5. **Multi-channel:** simultaneous experiments on interacting channels; does the operator handle cross-channel contamination terms (treated-geo budget reallocations) that scalar iROAS cannot express?
6. Formalize §3.4.3 (conflict → ridge escape) analytically: when a calibration constraint contradicts the observational level constraint, characterize the direction of the compromise fit as a function of the two constraint strengths.

## Sources

- [Meridian: calibrate treatment priors](https://developers.google.com/meridian/docs/advanced-modeling/roi-priors-and-calibration) · [Meridian ROI/mROI parametrizations](https://developers.google.com/meridian/docs/advanced-modeling/roi-mroi-contribution-parameterizations)
- [PyMC-Marketing: lift-test likelihood calibration](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_roas.html) · [Orduz, MMM and experimental calibration simulation study](https://juanitorduz.github.io/mmm_roas/)
- [Pearl & Bareinboim 2014, External Validity: From Do-Calculus to Transportability](https://projecteuclid.org/journals/statistical-science/volume-29/issue-4/External-Validity-From-Do-Calculus-to-Transportability-Across-Populations/10.1214/14-STS486.full) · [Bareinboim & Pearl, transportability of causal and statistical relations](https://dl.acm.org/doi/10.5555/2900423.2900462)
- Gouriéroux, Monfort & Renault 1993, "Indirect Inference," *J. Applied Econometrics* 8:S85–S118 (methodological ancestor of §3.3)
- Internal: dive 01 (`01_mmm_identified_set.md`) §3.2, §3.5; catalogs `../02_open_questions.md` M2, `../03_mmm_adoption_barriers.md` §O7, §B5, §M5
