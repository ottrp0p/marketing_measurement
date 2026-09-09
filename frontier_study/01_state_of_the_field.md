# Frontier Marketing Measurement: State of the Field

**A comprehensive study of current capabilities — August 2026**

This study synthesizes a survey of recent papers, white papers, books, arXiv preprints, and big-tech publications (with emphasis on 2020–2026 work layered on foundational results) across three lenses: engineering for scale, human understanding and marketing wisdom, and the deep mathematics and statistics of measurement. A companion document, `02_open_questions.md`, details what remains unsolved. Annotated source libraries live in `resources/`.

---

## Executive synthesis: where the field stands

Marketing measurement in 2026 has consolidated around a single organizing idea: **experiments are the ground truth, and everything else is calibrated to them.** The decade's defining empirical results — Lewis & Rao's proof that purchase variance swamps plausible ad effects, Blake/Nosko/Tadelis's eBay null, and Gordon et al.'s demonstrations across hundreds of Facebook RCTs that even sophisticated observational methods (propensity scores, double ML with thousands of covariates) fail to recover experimental lift — killed the hope that clever statistics on observational data could substitute for randomization. What replaced that hope is a *triangulation architecture*: platform lift RCTs and geo experiments provide causal anchors; Bayesian media mix models (MMM) provide the cross-channel, always-on connective tissue, with experiment results injected as priors or likelihood terms; attribution survives only as a fast, tactical signal explicitly understood to be non-causal.

Three convergent forces shaped this settlement. First, **privacy destroyed user-level joins** (ATT, SKAdNetwork, cookie deprecation-then-reversal, clean rooms, differentially private reporting APIs), pushing measurement back toward aggregate methods — a renaissance for MMM and geo experimentation. Second, **open-source tooling democratized the frontier**: Google Meridian, Meta Robyn, PyMC-Marketing, GeoLift, and Trimmed Match put what was once proprietary consulting IP into every analyst's hands, with GPU-accelerated Bayesian inference (JAX/NumPyro/TFP) making daily-refresh hierarchical MMM feasible. Third, **the delivery algorithms became part of the measurement problem**: ML-optimized ad delivery contaminates the very experiments meant to measure it (divergent delivery), and value-based bidding creates feedback loops between measurement and spend that no current framework fully handles.

Meanwhile the human side of the field carries a mature but uncomfortable body of wisdom: advertising is a weak, memory-refreshing force whose effects are small (mean short-run elasticity ~0.12), concentrated in light and out-of-market buyers, and mostly realized beyond the quarterly window where measurement systems look. Creative quality and attention — not targeting — are the largest controllable levers, yet they are the least well measured. And the newest instrument, LLM-simulated consumers, is improving on quantitative mimicry faster than on the identity, culture, and novelty dimensions where advertising actually operates.

The rest of this document details current capabilities under each lens.

---

## Part I — Engineering for scale

### 1. Experimentation platforms are mature infrastructure

Running tens of thousands of concurrent, trustworthy experiments is a solved engineering problem. The canonical architecture — deterministic hash-based randomization, layered/overlapping experiment universes, config-as-experiment, automated metric pipelines, sample-ratio-mismatch guardrails — is documented across Microsoft's ExP platform papers, the Kohavi/Tang/Xu book *Trustworthy Online Controlled Experiments*, Meta's PlanOut (experiments-as-programs DSL), and Netflix's science-centric platform rebuild. Airbnb (ERF), Uber (XP, ~1,000+ concurrent experiments), Spotify (Confidence, productized externally), and warehouse-native commercial platforms (Statsig, Eppo) represent the current dominant architectures. CUPED variance reduction is table stakes; interleaving gives 10–100x sensitivity for ranking surfaces and runs as stage one of two-stage funnels at Netflix and Airbnb.

### 2. Ad lift measurement runs at platform scale with logged counterfactuals

The ghost ads design (Johnson, Lewis & Nubbemeyer) — identify control users who *would have* seen the ad and log the counterfactual impression rather than serving a PSA — is the foundational pattern, extended to ghost bidding for DSPs and now productized across retail media networks. Meta Conversion Lift runs thousands of RCTs per year. The frontier development is treating the accumulated experiment library as *training data*: PIE (Predicted Incrementality by Experimentation) trains ML models on ~5,000 past lift RCTs to predict incrementality without running new experiments. Budget-split designs (LinkedIn, DoorDash, Amazon's asymmetric splits) solve the auction-interference pathology that makes naive user-split ad experiments invalid under budget constraints. Netflix-style persistent rotating holdouts provide always-on incrementality for owned channels.

### 3. Geo experimentation has an open-source, industrial toolchain

Google's lineage (Vaver & Koehler's geo-based regression → time-based regression → Trimmed Match with power-optimal pairing) and Meta's GeoLift (augmented synthetic control with simulation-based market selection) are open source and constitute the de facto standard for advertiser-run incrementality testing; vendors (Haus, Measured, INCRMNTAL) have industrialized always-on geo testing on top of them.

### 4. Bayesian MMM is democratized and GPU-accelerated

The methodological core is Google's 2017 trio (adstock + Hill saturation Bayesian MMM; geo-level hierarchical pooling; reach/frequency extensions), now shipped as Meridian (TFP, GPU NUTS, experiment-calibrated ROI priors, explicit causal DAG framing). Robyn takes a deliberately non-Bayesian route (ridge + Nevergrad evolutionary search) trading uncertainty quantification for automation. PyMC-Marketing's lift-test likelihood calibration — adding experiment results as additional likelihood terms on the saturation curve — is current best practice for experiment-MMM fusion. The JAX/GPU speedup (order of magnitude over CPU Stan) is the engineering unlock behind daily-refresh Bayesian MMM. Production accounts exist from HelloFresh, Netflix, Lyft (Symphony: closed-loop LTV → budget → bid automation), and Uber (Orbit structural time series, time-varying-coefficient MMM).

### 5. Privacy-preserving measurement is deployed but fragmented

The deployed stack includes: Google's Attribution Reporting API with TEE-based aggregation services enforcing contribution budgets and adding calibrated noise; Apple's SKAdNetwork → AdAttributionKit crowd-anonymity postbacks (spawning an entire MMP modeling sub-industry around censored conversions); Meta's MPC-based Private Lift (open-sourced fbpcf/fbpcs); the Mozilla+Meta Interoperable Private Attribution (IPA) protocol proposal and IETF DAP standardization of Prio-style secure aggregation; and mainstream clean rooms (Ads Data Hub, Amazon Marketing Cloud, AWS Clean Rooms with GA differential privacy). Formal DP accounting for ads measurement is an active literature (per-user DP accounting, attribution-rule-aware DP, Cookie Monster's on-device budget management). The pieces work; the ecosystem does not interoperate.

### 6. ML measurement systems and marketplace designs are commodity-to-mature

Delayed-feedback conversion modeling (Chapelle 2014 and successors) runs in every major bidder. Uplift modeling has industrial-scale benchmarks (Criteo, 25M rows) and production libraries (CausalML, EconML). Switchback experiments have rigorous minimax design theory and production playbooks (DoorDash); graph-cluster randomization runs at Meta scale. Amazon's AuctionGym provides the reference simulator for offline policy evaluation. LLM-agent market simulators are the 2024–2026 frontier — behaviorally promising, not yet validated as measurement instruments.

---

## Part II — Human understanding and marketing wisdom

### 1. The empirical laws replicate; their strategic corollaries are contested

The Ehrenberg-Bass canon (Sharp's *How Brands Grow*, Romaniuk's mental availability program) rests on genuinely law-like regularities replicated across categories, countries, and decades: double jeopardy (small brands have fewer, slightly less loyal buyers), NBD-Dirichlet purchase structure, light-buyer dominance, and growth through penetration rather than frequency. Category entry points (CEPs) now operationalize mental availability measurement, displacing funnel metrics in modern brand tracking. The live controversy is not the regularities but the strategic corollaries (never target, always maximize reach): Oxford's Felipe Thomaz argues from ~1,000 campaigns that mechanical reach-maximization yields mediocre outcomes, and the fair 2026 reading is that the laws constrain strategy while leaving wide room for creative and contextual judgment.

### 2. The long term dominates profit and eludes measurement

Binet & Field's brand/activation distinction (and its 60:40 heuristic, refined by category in *Effectiveness in Context*) remains the organizing framework, with known caveats (databank selection bias, correlational logic). The largest recent econometric synthesis, Profit Ability 2 (£1.8bn spend, 141 brands), finds ~60% of advertising's profit payback accrues beyond ~13 weeks — meaning short-window evaluation systematically undervalues advertising. But the mechanics of long-term measurement are shaky: Peter Cain's "adstock illusion" critique shows that stretching decay parameters to capture brand effects is statistically spurious, and proposes decomposing sales into transitory versus evolving-baseline components instead. No mainstream method observes memory formation directly; the long term is inferred, not measured.

### 3. Advertising is a weak force, and the evidence agrees

Ehrenberg's ATR(N) reinforcement model — ads mostly refresh memory and nudge propensities rather than persuade — is largely vindicated by the modern evidence: mean advertising elasticities of ~0.12 short-run / ~0.24 long-run (Sethuraman-Tellis-Briesch meta-analysis), declining over decades; near-null persuasion effects in massive political-ad experiments; TV elasticity distributions (Shapiro-Hitsch-Tuchman, 288 brands) with medians far below the published literature, quantifying publication bias. Recency planning (continuous presence beats frequency bursts) and non-monotonic frequency response (retargeting wearout) follow directly from the memory-refresh model.

### 4. Creative and attention are the biggest levers and the weakest instruments

Across Nielsen/NCS/Kantar decompositions, creative quality is the largest controllable driver of ad-driven sales (~50%), and consistent creative platforms compound over years. Peter Field's *Extraordinary Cost of Dull* quantifies the penalty: dull ads need ~2–2.6x more media spend for equal impact. Attention measurement (Nelson-Field, Lumen, Adelaide) established that viewability is not attention and attention predicts outcomes better than viewability — but the vendors measure different constructs, disagree at placement level, and the ARF's validation program shows the field has measurement-of-measurement problems. Emotion measurement is in a similar state: System-1-style behavioral proxies have commercial validation; neuro hardware (EEG/fMRI) has never cleared the replication bar at scale.

### 5. Human variance is the central measurement obstacle

Lewis & Rao's result deserves restating as a fact about people rather than statistics: individual purchase behavior is so variable relative to plausible ad effects that even 25 large experiments with millions of users could not reliably distinguish highly profitable from money-losing campaigns. Response is wildly heterogeneous and concentrated (eBay: returns concentrated in infrequent, loyalty-poor users), targeting makes observational data selection-soaked, and the nudge literature's post-correction shrinkage warns against vendor effect-size claims generally.

### 6. Synthetic consumers are advancing fast on a narrow front

Silicon sampling (Argyle et al.), homo silicus (Horton), LLM conjoint/WTP estimation (Brand-Israeli-Ngwe), and Stanford's interview-grounded generative agents (85% of humans' own test-retest reliability on GSS items) mark rapid progress; Hewitt et al. report LLMs predicting social-science experimental results at forecaster-panel accuracy. The counter-literature is equally strong: LLMs reason unlike humans in strategic settings, flatten within-group variance, misportray identity groups, exhibit mode collapse and social-desirability bias. The practitioner consensus circa 2026: synthetic respondents for piloting, instrument testing, and augmentation — not for tracking, representativeness claims, or go/no-go decisions. Nobody has demonstrated synthetic ad pre-testing that predicts in-market sales lift out of sample.

### 7. Measurement is organizationally and politically embedded

Triangulation doctrine is now orthodoxy, and its leading practitioners say plainly that the biggest blockers are organizational, not methodological: channel teams graded on ROAS defend last-click, platforms grade their own homework, and incrementality adoption redistributes budget power. Bayesian priors institutionalize judgment inside the model — and create a new politics of whose prior counts. B2B remains measured with B2C tools despite buying groups, multi-year cycles (the 95-5 rule), and account-level outcomes that break person-level attribution.

---

## Part III — The mathematics and statistics

### 1. Estimation theory is mature and commoditized

For aggregate and panel causal questions the machinery is essentially settled: double/debiased ML (Neyman orthogonality + cross-fitting), causal forests and generalized random forests for heterogeneous effects, the modern synthetic-control family (augmented SC, synthetic difference-in-differences, matrix completion for causal panels, conformal inference for SC), and Bayesian structural time series (CausalImpact). Off-policy evaluation (clipped IPS, doubly robust and shrinkage variants, from Bottou et al. onward) and valid inference after adaptive sampling (Hadad et al.'s adaptively-weighted AIPW) are solved in useful generality. The binding constraints have moved from estimation to *identification and design*.

### 2. MMM's identification problem is precisely diagnosed, partially solved

Chan & Perry catalogued the pathologies (selection bias, targeting feedback, model degrees of freedom); Jin et al. demonstrated the core weak-identification result — with typical national time series, the likelihood is nearly flat along ridges trading off adstock decay, Hill-curve shape, and effectiveness, so ROAS posteriors are heavily prior-driven. The partial solutions: hierarchical geo-level pooling multiplies effective sample size; search-query-volume backdoor adjustment de-biases paid-search coefficients; experiment-calibrated priors/likelihoods (Zhang et al., Meridian, PyMC-Marketing) inject causal anchors; and the newest frontier work (Heusch 2026) estimates adstock, saturation, and effectiveness *structurally from geo-experimental time series* — recovering full response curves rather than a single ROAS point. Transformer-based MMM (Google's NNN) trades identifiability for predictive fit and its causal status remains contested.

### 3. The negative results are load-bearing

The field's most important theorems are impossibility-flavored: Lewis-Rao's power economics (implied R² of ads on sales ~5e-6; ROI confidence intervals spanning ±100pp at realistic scale); Gordon et al.'s repeated demonstrations that observational methods cannot reproduce RCT lift even with rich covariates ("Close Enough?", 663 RCTs); Rossi's critique showing most marketing instruments are weak or invalid; Gelman-Carlin Type S/M logic explaining why underpowered lift tests systematically exaggerate; and quantified p-hacking/false discovery in commercial A/B testing (Berman et al.). These results discipline the whole field.

### 4. Interference and equilibrium are formalized, not conquered

Exposure mappings with Horvitz-Thompson estimation (Aronow-Samii), robustness under misspecified mappings (Sävje), graph-cluster randomization (Eckles-Karrer-Ugander), two-sided marketplace bias analysis (Johari et al.), minimax switchback design (Bojinov et al.), and ad-specific auction-interference results (Waisman-Nair-Misra's bandit inference in RTB auctions; Gui-Nair's throttling quasi-randomization; parallel-experimentation interference) give a real theory. Paid-organic cannibalization and cross-channel spillovers are empirically established at platform scale. What is missing is market-level theory: bias bounds as a function of auction thickness when many advertisers experiment simultaneously.

### 5. Attribution is mathematically clean and causally humbled

Shapley attribution has tidy axiomatics and efficient approximations; Markov-chain removal effects are the graph counterpart. The settled verdict: both allocate correlational credit, and no axiom system implies counterfactual incrementality when the coalition value function is estimated from targeting-shaped journeys. Berman's game-theoretic treatment reframes attribution as an incentive contract shaping publisher behavior — the deepest available answer to why last-touch persists and misleads. Incrementality bidding is formalized (Lewis-Wong); causal deep MTA exists but lacks experimental validation at scale.

### 6. Decision theory is entering measurement

Test & Roll (Feit-Berman) recasts A/B testing as Bayesian profit maximization with closed-form (often small) optimal samples — the cleanest exemplar of decision-focused measurement. Kasy-Sautmann exploration sampling optimizes adaptive designs for policy choice rather than estimation. Stacking (Yao et al.) offers principled model combination for M-open MMM ensembles. Surrogate indices (Athey-Chetty-Imbens-Kang) are the standard answer to short experiments versus long-term outcomes. Advanced identification machinery — proximal causal inference for unmeasured demand confounding, Cinelli-Hazlett sensitivity analysis, Manski-style partial identification, transportability calculus (Bareinboim-Pearl) — is available and underused; it constitutes the formal skeleton the experiment-MMM fusion literature implicitly needs.

---

## The through-line

Across all three lenses, 2020–2026 is the story of **convergence at the seams**. Experiments, MMM, and attribution are being fused into single Bayesian causal systems (Meridian-style calibration, PIE, structural geo-experiment estimation); privacy engineering is forcing measurement into aggregate and noisy channels where that fusion is the only viable strategy; and the human-side evidence (weak force, long horizons, creative dominance, heterogeneity) defines what any honest fused system must be able to represent. The open problems — detailed in `02_open_questions.md` — concentrate almost exactly at those seams.
