# Open Questions and Key Next Steps in Marketing Measurement

**What has not yet been developed or well understood — August 2026**

Companion to `01_state_of_the_field.md`. Each entry states the open problem, why it is genuinely open (not just unimplemented), what a solution would look like, and who is closest. Ordered within each part roughly by importance.

---

## Part I — Engineering open problems

### E1. Cross-platform experiment orchestration

Every platform runs its own lift RCTs against its own holdouts; no infrastructure exists to run one coherent randomization across Meta, Google, TikTok, and retail media. Clean rooms do not interoperate (ADH ≠ AMC ≠ Meta), so an advertiser cannot even *join* results, let alone coordinate designs. **What a solution looks like:** an advertiser-side meta-experimentation control plane — a unified holdout registry, cross-platform ghost-bid coordination, and standardized experiment metadata (IAB interoperability efforts are embryonic). **Who is closest:** nobody credible; incrementality vendors (Haus, Measured) approximate it with geo designs precisely because geography is the only randomization unit platforms cannot fragment.

### E2. Measuring a system that is itself learning

Delivery algorithms optimize each experimental arm separately (divergent delivery), and value-based bidding creates a closed loop: measurement → spend → data → measurement. Meta has quantified divergence across ~182K experiments and offers configuration mitigations, but there is no engineering framework for experimenting *through* an adaptive delivery layer — constraining or logging the optimizer's behavior per arm so the estimand stays fixed. **What a solution looks like:** delivery-system-aware experiment designs with logged bandit state/propensities as a first-class platform feature. This is the engineering twin of open problem M3 below.

### E3. Differential privacy budget management as a systems discipline

Cookie Monster (SOSP 2024) established on-device DP budgeting, but production questions remain: scheduling scarce privacy budget across campaigns and queries, budget-aware query optimizers for clean rooms, DP composition across *multiple* clean rooms and APIs touching the same users, and observability for noisy pipelines — SRM detection under DP noise is essentially unsolved. **What a solution looks like:** the DP equivalent of a query planner plus monitoring stack, treating epsilon like a metered cloud resource.

### E4. MPC attribution at ad-scale cost

IPA-style protocols carry orders-of-magnitude compute/bandwidth overhead versus plaintext joins; malicious-secure, billion-event daily attribution at acceptable cloud cost has not been demonstrated. TEE-based designs scale but concentrate trust. Hybrid TEE+MPC architectures are unbuilt. **Who is closest:** Meta's fbpcs (scale-out private lift) and the IETF DAP/Prio ecosystem.

### E5. Bayesian MMM at true hierarchy scale

Meridian-class models handle ~50–200 geos × tens of channels. National retailers want store × SKU × channel — 10^5–10^6 cells — with honest posteriors. Requires distributed/sharded HMC or scalable variational inference with calibrated uncertainty, amortized inference for daily refresh, and automated identifiability diagnostics. Nobody has published production MMM at that granularity with defensible uncertainty.

### E6. Always-on experiment-to-model fusion (closed-loop calibration)

Lift-test-calibrated priors are today manual and point-in-time. The obvious next system — one that continuously *schedules* geo/budget-split experiments chosen to maximally reduce MMM posterior uncertainty (optimal experiment design in the loop), then folds results back automatically — is a described ambition, not a shipped system. This is the engineering realization of open problem M10 (value-of-information theory).

### E7. Interference-aware experimentation as a platform feature

Switchback and cluster designs are bespoke per company. General platform support — automatic exposure-graph construction, cluster-quality monitoring, bias-variance-optimal design selection, analysis under unknown interference — exists only in research code.

### E8. Signal-loss-robust measurement

Estimators that jointly model consent-driven missingness, SKAdNetwork-style censoring, DP noise, and modeled conversions — with the missingness mechanisms treated explicitly rather than patched by MMP heuristics — are just appearing (2026 arXiv work) and far from production.

### E9. Validated market digital twins

LLM-agent simulators show behavioral realism in narrow tasks, but no one has demonstrated a calibrated simulator whose *counterfactual* predictions match held-out geo-experiment results at scale — the necessary bar before simulation can substitute for any experiment. The obvious build: a calibration pipeline binding a simulator to a library of past RCTs (the PIE pattern, applied to simulation).

### E10. Long-horizon measurement infrastructure

Persistent multi-quarter holdouts that survive identity churn, re-consent, and cross-device movement; surrogate-index pipelines at ads scale. All production systems today measure days to weeks; the profit evidence says most value accrues later.

---

## Part II — Human-understanding open problems

### H1. The exposure → memory → future-choice causal chain has never been measured end to end

We infer long-term effects through fragile statistical conventions (adstock decay, econometric extrapolation) rather than observing the mechanism: attention at exposure → encoding into memory structures linked to category entry points → retrieval at a purchase occasion years later. **What a solution looks like:** longitudinal single-source panels linking attention measurement, CEP-based brand tracking, and purchase behavior on the same humans over years. These barely exist and are proprietary where they do. This is arguably the field's deepest empirical gap — everything in Part I ultimately proxies for this chain.

### H2. For whom does advertising work?

Everything from Lewis-Rao to the eBay experiment says response is concentrated in small, hard-to-predict subpopulations, yet the industry reports campaign-level averages. Uplift/HTE estimation is unstable at realistic sample sizes. Open: a credible science of *who* responds — stable, replicable moderators of ad response (category involvement, purchase cycle position, brand usage) rather than post-hoc segment mining.

### H3. Attention lacks a shared construct; encoding is unmeasured at scale

Lumen, Adelaide, and Amplified Intelligence measure different things (gaze probability, media-quality proxy, active/passive attention) and disagree at placement level; ARF validation shows predictive value but no convergent validity standard. The psychologically decisive variable — encoding into memory linked to a CEP — has no scaled instrument. Open: an attention-to-encoding measurement standard, and settling whether attention can function as a trading currency.

### H4. Why does "interesting" beat "dull," mechanistically?

The outcome-level fact is quantified (dull ads need ~2–2.6x the media budget), and right-brain/character/story features correlate with effectiveness (Wood's *Lemon*/*Look out*), but the cognitive mechanism is not: which creative features cause encoding, via what attentional and emotional pathways, with what interaction with media context? Emotion instruments (facial coding, System-1 scores) predict recall and correlate with long-term share in vendor databases but lack adversarial, independent validation against sales.

### H5. Mindset metrics ↔ behavior causal ordering

Which survey metrics *lead* market share, at what lag, and which merely track it? Regression to the mean contaminates most tracking narratives (the Ehrenberg-Bass critique cuts against everyone, including brand-lift vendors). CEP-based tracking is progress; the causal ordering problem is unresolved.

### H6. The brand lift study validity crisis — and the largest untapped dataset

Sub-1% survey response rates, consent-biased samples, recall-as-attention confounds, platforms grading their own homework. Millions of platform lift studies have accumulated; there is no public meta-analysis reconciling them with econometric or experimental sales outcomes. Whoever builds that reconciliation owns a foundational resource.

### H7. A theory of *when* synthetic consumers are valid

Empirical human-LLM alignment is improving fast, but there is no principled account of which constructs LLMs can simulate (structured trade-offs, WTP) versus cannot (identity, culture, humor, novelty — precisely where advertising lives). Unresolved: training contamination, distribution shift for genuinely new products, and homogeneity that understates the human variance which is itself the central measurement obstacle. The decisive missing demonstration: synthetic ad pre-testing that predicts in-market sales lift out of sample.

### H8. Decision-making under measurement conflict (organizational epistemics)

How should a CMO weight a Bayesian MMM, a conflicting geo test, and a brand tracker? Formal frameworks for measurement-as-decision-support — value of information, pre-registered decision rules, incentive design so no channel grades its own homework — are nearly absent from practice, and "attribution politics" is documented anecdotally but barely studied academically.

### H9. B2B measurement science

Buying groups (6–12 people), multi-year cycles (95-5), and account-level outcomes have no equivalent of the Dirichlet benchmarks; B2B brand effects on pipeline are asserted more than causally measured. An Ehrenberg-Bass-style empirical-generalization program for B2B does not yet exist.

### H10. Creative equity accrual and the cost of churn

Case evidence says consistent creative platforms compound returns over years, but measurement systems reset with campaigns. No standard exists for measuring distinctive-asset equity accrual or the penalty of creative-strategy churn.

---

## Part III — Mathematical and statistical open problems

### M1. The identified set of MMM parameters

For y_t = β·Hill(Adstock_α(x_t); K, S) + controls + ε, no one has characterized the sharp identified set for (α, K, S, β) as a functional of the spend-path design {x_t}. We know empirically the likelihood has flat ridges; missing are (a) formal conditions on spend variation — frequency content, on/off structure, range coverage — under which the response curve is point- versus set-identified, and (b) an optimal-design theory: which perturbation schedule maximizes Fisher information about curvature separately from decay? Heusch (2026) shows geo experiments can separate them; the general design-of-experiments problem for response-curve identification is open.

### M2. A formal transport map from lift tests to MMM parameters

Calibration today treats an experiment's iROAS as a prior on the model's ROAS. But an experiment measures a finite-horizon ITT effect of a specific spend delta at a specific point on the saturation curve under a specific competitive state; the MMM parameter is a steady-state structural quantity. The Pearl-Bareinboim transportability calculus has never been instantiated for this problem: write the selection diagram, derive when the experimental functional identifies the model functional, and characterize the correction terms (adstock truncation bias, saturation-point mismatch). Handling multiple simultaneous experiments as a *joint, correlated* posterior over the ROAS vector is likewise unsolved in practice.

### M3. Estimands stable under algorithmic delivery

Divergent delivery shows "the effect of creative A vs. B" is ill-defined when the delivery algorithm mediates exposure differently per arm. Open: define and identify "the effect of creative under a *fixed* delivery policy π" from data generated under adaptive policies π_A, π_B — off-policy evaluation where the logging policy is a black-box, non-stationary auction/pacing system with unlogged propensities. Operational mitigation exists (Meta); identification theory does not.

### M4. Auction-equilibrium interference bounds

When advertiser i experiments, treated spend shifts bid landscapes for everyone else — SUTVA violated across advertisers. Single-advertiser mechanics are handled (Waisman-Nair-Misra, Gui-Nair); open is the market-level problem: derive the bias of standard lift estimators as a function of auction thickness/overlap, design platform-run mechanisms (budget-split with equilibrium correction) whose estimands equal the advertiser's best-response-relevant effect, and characterize what is measurable in principle when all advertisers experiment simultaneously.

### M5. Long-run effects under weak surrogacy

The surrogate index requires that treatment affects long-run outcomes only through the surrogates — almost surely false for brand advertising, whose memory/brand-stock channel bypasses short-run behavior. Open: partial identification of long-run ROAS combining a short experiment with an observational panel, with bounds indexed by a surrogacy-violation parameter — a Cinelli-Hazlett-style sensitivity analysis for surrogacy, not yet developed.

### M6. Semiparametric efficiency for lift under heavy tails

Lewis-Rao establishes the power problem; the floor is uncharacterized. What is the semiparametric efficiency bound for the ATE of ad exposure with zero-inflated, heavy-tailed outcomes, and how close do CUPED + latent stratification + trimmed influence-function estimators get? Is there a principled theory trading small identified bias (trimming, as in Trimmed Match) against variance to minimize *decision regret* rather than MSE?

### M7. Inference in the measurement-allocation feedback loop

Hadad et al. solve inference under a known adaptive assignment rule. In practice budgets adapt to *interim estimates* (MMM readings, attribution dashboards), making assignment depend on past estimators, not just past outcomes. Open: valid confidence sequences for ROAS under endogenous, estimator-dependent adaptive designs — the statistical twin of engineering problem E2.

### M8. An impossibility theorem for causal attribution

Characterize the class of journey-value functions for which Shapley attribution equals an average of per-touch causal effects; conversely, prove that no observational attribution rule can jointly satisfy efficiency, symmetry, and a counterfactual-consistency axiom under confounded exposure. Berman shows equilibrium divergence; a clean impossibility theorem would settle "can attribution ever be causal?" definitively.

### M9. MMM under aggregation and privacy noise

Post-cookie MMM consumes aggregated, differentially private, delayed, and modeled conversions. Open: identification and efficient estimation when (a) covariates aggregate over heterogeneous units — ecological bias interacting with nonlinear Hill transforms creates Jensen-gap bias in β — and (b) outcomes carry calibrated DP noise with known mechanism. No one has derived the correction terms or priced privacy in ROAS posterior width.

### M10. Decision-calibrated uncertainty and value of information

Optimizing budget against posterior-mean response curves is not Bayes-optimal under the full posterior (nonlinearity + constraints), and MMM posteriors are over-confident under misspecification. Open: (a) tractable Bayes-optimal budget allocation under a joint posterior over multi-channel response curves — a stochastic program with rank-correlated uncertainty; (b) value-of-information theory for the next experiment: which geo test, at what spend delta, maximally reduces expected budget-decision regret? Test & Roll solves the two-arm case; the multi-channel, curve-level problem is open.

### M11. Causal model averaging

Stacking weights validated on held-out *sales predictions* do not validate *causal* functionals — good fit ≠ good ROAS. Open: principled ensembling of causal models, e.g., stacking against held-out experimental lift results rather than predictive scores (PIE gestures at this).

---

## Cross-cutting grand challenges

Three programs would each reorganize the field if executed:

**1. The unified causal measurement system.** Instantiate transportability calculus end to end: a single Bayesian causal model in which every lift RCT, geo experiment, brand lift study, and observational panel enters through an explicit selection diagram, with joint posteriors over channel response curves, decision-calibrated uncertainty, and an experiment scheduler driven by value of information (M2 + M10 + E6). All components exist separately; the composition does not.

**2. The human-mechanism panel.** A longitudinal single-source measurement asset linking attention, memory/CEP tracking, and purchases on the same individuals over years (H1 + H3 + H5), sufficient to validate — or overturn — the adstock conventions, the surrogacy assumptions (M5), and the long-term heuristics (60:40) the industry currently runs on.

**3. The calibrated simulator.** An LLM-agent (or hybrid structural) market simulator held to the counterfactual bar: predictions validated against a held-out library of real experiments before any measurement claim (E9 + H7). If achieved, it collapses the cost of the experiments everything else depends on; every current validity failure (homogeneity, contamination, identity flattening) is a named, attackable subproblem.
