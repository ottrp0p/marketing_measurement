# Why Most Companies Haven't Adopted Sophisticated MMM

**The business-logic, mathematical, and operational barriers — August 2026**

Companion to `01_state_of_the_field.md` (what the frontier can do) and `02_open_questions.md` (what nobody can do yet). This document answers a different question: given that Meridian-class Bayesian MMM, experiment calibration, and geo-testing tooling are free and public, why does the majority of the market still run last-click dashboards and simple regressions? Sources are in `resources/mmm_adoption_barriers.md`.

---

## The adoption picture in numbers

The headline pattern across every 2024–2026 survey: *claimed* MMM usage is high, *sophisticated and acted-upon* MMM is rare, and last-click attribution remains the operating system of most marketing organizations.

An HBR Analytic Services study (Google-sponsored, March 2026) found 87% of marketing organizations "use MMM data for insights" — but only 28% can turn those insights into action, and only 22% qualify as "leaders" who gather *and* act on MMM effectively. A Kantar/Meta survey (2025, n=1,935 decision-makers at companies spending $1M+/yr on digital) found 34% prioritize MMM over other measurement, a third don't use MMM at all, only 31% of users in-house their model, and 55% frequently encounter conflicting results across measurement tools. The IAB's State of Data 2026 reports 60–75% of buy-side marketers say every leading measurement approach underperforms on rigor, timeliness, or trust. Meanwhile EMARKETER/Snap (2024) found 78.4% of marketers still use last-click attribution while only 21.5% believe it reflects long-term business impact — and 53% say *leadership* is its biggest believer. No survey measures Bayesian/hierarchical MMM adoption specifically (itself a data gap); triangulating the "28% can act" ceiling against a total advertiser population dominated by companies below any viable spend threshold, sophisticated MMM penetration is plausibly single-digit.

The synthesis that emerges from three research passes: **the software was never the main cost, and every barrier that remains is multiplicative with the others.** The tooling wave of 2023–2026 (Meridian, PyMC-Marketing, Robyn, Recast/Haus/Sellforte/Mutinex-class SaaS) cut price and cadence by 10x — and the bottleneck moved almost entirely into the buyer's data, org chart, and epistemics.

---

## Part 1 — Business-logic barriers

### B1. The unit-economics floor: MMM doesn't pencil out below ~$1–5M in annual media spend

Legacy vendors historically required ~$5M minimum media spend; even the new SaaS tier ($30K–$150K/yr, Recast at ~$2–5K/mo) plus data-prep costs only clears a hurdle rate at seven-figure budgets, because MMM's payoff is a few percent of allocation improvement on the spend base. Below ~$1M annual spend the model can't even be estimated stably (insufficient variation, too-short history). Since the *numerical* majority of companies are SMBs below this line, majority non-adoption is at the bottom of the market simply rational economics — a point usually missed in "why won't they modernize" discourse. Google cutting its minimum incrementality-test budget from ~$100K to $5K is a direct attack on this floor.

### B2. Refresh-cadence economics: a quarterly deliverable in a daily-decision world

Traditional MMM refreshes quarterly or semi-annually while budget decisions happen daily. A six-figure deliverable that is stale on arrival loses the daily argument to free, real-time platform dashboards — and makes the renewal business case collapse, which is why so many engagements are one-offs. Funnel's diagnosis is representative: "if the insights aren't part of that [daily] process, they get ignored."

### B3. Platforms grade their own homework — and keep winning anyway

The conflict of interest is a decade old on the record ("fox guarding the henhouse," Digiday 2017). The sharper 2025 finding, from Haus's meta-analysis of 640 incrementality experiments: Meta's 7-day-click attribution mis-states incrementality systematically (Advantage+ campaigns over-credit themselves ~12 points because the algorithm finds users who would have converted anyway) — *and advertiser behavior didn't change in response to the experimental evidence*. Note also that much of the "independent" tooling (Robyn, Meridian) and adoption research (Kantar/Meta, HBR/Google) is platform-subsidized.

### B4. Incentive misalignment: the model's correct answer hurts specific named people

Channel teams and agencies are graded on platform ROAS; an MMM that reclassifies branded search or retargeting as non-incremental takes budget and headcount justification from them. The canonical case: eBay's experiment showed branded paid search essentially non-incremental in 2015 — and branded search remains a staple line item a decade later. Vendor incentives compound it: under renewal pressure, vendors run many model variants and present the one that confirms leadership's priors (documented by Recast, whose interest is adverse to that practice). The MSI/Robyn working paper formalizes this as "managerial bias" and HiPPO interference. Failed engagements are never published — so the market cannot learn from failure or price its risk.

### B5. Precision theater: organizations prefer a wrong number with decimals over a right number with error bars

Last-click is epistemically inferior but *operationally* superior: real-time, free, ad-level granular, and confidently pointy. 77% of marketers call it "the easiest — but not best — way"; they use it anyway. The 53%-of-leadership-believes-last-click finding inverts the usual story: executives, where budget authority lives, are the constituency for the bad metric. Against a dashboard reading "ROAS 4.27," an honest posterior reading "1.2–3.8" loses the meeting.

### B6. Conflicting-results paralysis

Advertisers run ~3.8 measurement solutions simultaneously; 55% frequently get conflicting answers. In practice the triangulation stack lets every stakeholder cite the tool that flatters them, and the asymmetric risk of action (reallocation has a visible, attributable downside; status-quo opportunity cost is owned by no one) makes "we need more accuracy first" the perpetual answer.

### B7. Measurement bought as budget defense, not decision input

Much MMM is purchased to prove marketing's value to the CFO — vendors literally market it as budget-defense. A model bought to defend a budget will not be allowed to conclude the budget should shrink, which selects for the cherry-picking dynamic of B4. Meanwhile flat marketing budgets (Gartner: 7.7% of revenue) and short CMO tenures make a multi-quarter measurement rebuild with uncertain political payoff a poor personal bet for the median CMO. The deeper trap: firms avoid extreme reallocations, so when the model recommends a big change, the recommendation itself reads as evidence the model is broken; when it recommends small changes, it wasn't worth the money. Either way the engagement disappoints.

### B8. The actionability gap is 20 years old and method-independent

The HBR/Google barrier list — 46% slow processes, 45% expertise, 41% silos — is essentially identical to the ANA's 2006 accountability findings (20% "can measure ROI but cannot act on it"). The organizational impediments predate Bayesian MMM by two decades and have survived every methodological upgrade. The one-line summary of the business-logic research: **MMM is a decision technology being bought by organizations that don't want their decisions changed.**

---

## Part 2 — Mathematical barriers

These are the math facts that translate directly into lost organizational trust — distinct from the open research problems in `02_open_questions.md`, though several are their adoption-side shadows.

### M1. Weak identification: the data can't answer the question, so priors do

Google's own foundational papers say it plainly: with weekly national data, the likelihood is nearly flat along ridges trading off adstock decay, saturation shape, and effectiveness — "prior distributions have a big impact on the posteriors" (Jin et al. 2017), and five equally plausible models fit to identical data disagree on optimal allocation and achievable sales by up to 50% (Chan & Perry 2017). A CMO who understands this concludes, not unreasonably, that the model launders assumptions. Sophistication doesn't remove the flatness; it makes it explicit — which is honest and commercially fatal.

### M2. Small T, big p — structurally, not fixably

~104–156 usable weekly observations against 20+ channels × (coefficient + adstock + saturation) + trend + seasonality + controls. Waiting doesn't help: older data is non-stationary (channel effectiveness three years ago is irrelevant). The statistical fix — hierarchical geo-level modeling — is exactly what creates operational barrier O3 below.

### M3. Adversarial design matrix: collinear, seasonal, endogenous spend

Channels ramp together for Q4; budgets chase anticipated demand; algorithmic buying chases performance. Chan & Perry call the resulting selection bias "perhaps the largest hurdle"; Recast concedes "there is no way to eliminate multicollinearity entirely" — credit for the same revenue can go to linear TV *or* CTV with equal fit. A 2026 benchmark paper builds endogenous-spend synthetic data (0.46 spend–lagged-sales correlation) precisely because random-spend simulators "remove the hard part."

### M4. Model multiplicity and analyst degrees of freedom

Two competent analysts get materially different ROAS from identical data, and no statistic arbitrates. Robyn makes this literal: the analyst *selects one model from a Pareto front* of thousands; candidates "often yield totally incompatible results" (one rates TV highly, another poorly). Robyn's decomp.RSSD selection criterion — penalizing models whose attribution deviates from spend shares — is, per Recast's critique, "effectively optimizing to not telling the marketers that they were wrong." Once stakeholders witness the same data yield rankings that depend on who ran the model, MMM output reads as negotiable opinion. This is the mathematical seed of the public "MMM wars" (Bayesian vs. frequentist vendor camps, e.g., Recast vs. Aryma Labs).

### M5. The validation impossibility: good fit ≠ good ROAS

The quantity MMM estimates — incremental sales caused by each channel — is never observed. Holdout predictive accuracy validates forecasting, not the causal decomposition used for budgets; Google's Meridian docs concede the causal inference "cannot be directly assessed" and that a deliberately overfit model may even be preferable. The accepted fix — experiment calibration — is expensive, slow, per-channel, and itself contested (published practitioner arguments that experiment-derived priors transfer bias across non-comparable populations). Enterprises adopt systems they can audit; MMM has no unit test, and the honest answer to "how do we know it's right?" is "run a multi-month experiment program" — at which point many companies conclude they should run the experiments and skip the model.

### M6. Refresh instability: the ROAS that flipped

Weakly identified models are unstable under small data perturbations. Documented case: a Robyn deployment where Facebook's estimated ROAS went from 0.75 to 0.2 over five refreshes, driven partly by the moving modeling window. Channel rankings flipping between quarters is the single most visible trust-destroyer for executives — and the industry's quiet fix, freezing coefficients between refreshes, is (per Recast) "effectively just lying about the model results."

### M7. Prior elicitation: the model demands beliefs business users cannot supply

Bayesian MMM converts the identification deficit into a demand for informative priors. But marketers can't express beliefs as distributions over Hill-curve slopes; priors that feel right one at a time can be jointly incoherent; defaults silently drive results; and every informative prior invites the "you assumed the answer" objection — which rival vendors weaponize ("whose prior?" as a manipulation-surface argument). The prior-free alternative is not neutral either: Robyn's ridge regression biases new, short-history channels toward zero ROI. There is no assumption-free seat at the table, but only one camp's assumptions are visible, and visible assumptions lose procurement reviews.

### M8. Complexity vs. legibility: wrong-but-auditable beats right-but-opaque

A modern stack (Weibull adstock → Hill saturation → hierarchical geo pooling → time-varying coefficients → NUTS) cannot be traced by any CMO, finance partner, or procurement reviewer. Simple regression MMM is wrong but legible; hierarchical Bayesian MMM is (maybe) righter but unauditable — and legibility usually wins budget meetings. The "black box" objection is prominent enough that vendors publish standing rebuttals.

### M9. Uncertainty communication: wide intervals read as "the model doesn't know"

Honest posteriors under collinearity are wide — mathematically the correct output, organizationally heard as failure. Recast's co-founder states the dilemma exactly: emphasize the uncertainty and the business user says "this doesn't help me, I need an answer"; hide it and you've become the false-precision vendor you replaced. Interval-driven budget processes barely exist; the market rewards vendors who report false precision (the demand side of B5).

### M10. The Bayesian workflow is a specialist craft marketing teams don't have

Competent Meridian/PyMC operation means divergence diagnosis, R-hat interpretation, prior-predictive checks, reparameterization. 45% of organizations cite lack of expertise as an MMM barrier; even PyMC Labs' own benchmark hit convergence trouble at enterprise scale (11% of parameters with R̂>1.1; Meridian "unsuccessful sample"). And the tools measurably disagree: cross-framework recovery benchmarks show ~2x accuracy gaps and tool-specific documented biases, so a company committing to sophistication must first adjudicate a methods dispute between Meta, Google, and PyMC — run, in each case, by an interested party.

### How the math compounds

Weak identification (M1–M3) guarantees analyst discretion matters (M4), which guarantees refresh instability and cross-tool disagreement (M6, M10) — which stakeholders *will observe*. The only honest responses — informative priors (M7) and wide intervals (M9) — each create their own credibility attack surface. The only external fix — experiments (M5) — is costly and contested. Executing any of it requires scarce specialists (M10) producing artifacts leadership can't audit (M8). The median organization's rational response: last-click plus occasional lift tests, or a legible-but-crude regression.

---

## Part 3 — Operational barriers

### O1. Data archaeology: 70–80% of the project is finding and cleaning the data

MMM needs 2–3 years of clean weekly spend/impression history across every channel — scattered across platforms, agencies, and finance systems, under chaotic campaign taxonomies. Circana (an incumbent provider) reports data collection/cleaning at ~30% of project time with mature data and 70%+ with poor data; independent practitioners put "collecting and cleaning data" at 80% of the job; PyMC Labs concedes data wrangling "consumes the majority of MMM efforts." The IAB's 2025 *Modernizing MMM* guide names incomplete/inconsistent/stale data the primary obstacle and prescribes what amounts to a data-governance engineering program as the prerequisite.

### O2. Offline and retail-media channels are the weakest links

TV, OOH, radio, sponsorships, retail media, CTV, and podcasts lack standardized measurement and often lack usable spend feeds at all, forcing proxy methods and aggregation that hide exactly the nuance the model needs. Missing Amazon/retail conversion data is named by vendors as a core enterprise blocker.

### O3. Geo-level data — required for the *sophisticated* tier — is a step-change harder

Hierarchical geo MMM is the statistical answer to small-T (M2), and Meridian requires 2+ years of weekly *geo-level* data across the top 50–100 DMAs. But many channels simply don't report geo-level spend (national TV buys, sponsorships, some walled gardens), forcing population-proportional imputation; geo aggregation choices change results; per-geo variance complicates MCMC convergence. The model class that is statistically superior demands exactly the data enterprises are least able to produce — the cleanest single example of the barriers compounding.

### O4. Talent: scarce, slow to hire, and more than one person

Bayesian + causal + time-series + marketing-domain + software skills rarely co-occur; hiring cycles run 3–6 months; PyMC Labs argues one hire can't cover the cross-disciplinary depth. In-house builds take ~2 data scientists × 6 months (~$125K; Meta's own estimate was 12–22 weeks) plus permanent maintenance, with "the model dies when its author quits" the most common failure mode — Deloitte research found in-house MMM structures collapsing before reaching production. Per a 2026 peer-reviewed comparison: Robyn is runnable in under a day (with corresponding misuse risk and no geo hierarchy), Meridian takes days and solid data-science skill, PyMC-Marketing takes up to two weeks and real Bayesian expertise — for a prototype, not a production system. Hence only 31% of MMM users in-house.

### O5. No MLOps for MMM: outputs don't reach planning workflows

The recurring failure points (per vendor post-mortems): stale data pipelines, siloed teams, model complexity requiring a data scientist as interpreter, missing feedback loops, undefined ownership. Legacy annual/semi-annual refreshes are stale before decisions; modern weekly-refresh demands automated pipelines with validation checkpoints — "a significant technical and organizational lift" (IAB). Even Google's flagship shows operational fragility: Meridian's issue tracker documents recurring production MCMC convergence failures — firefighting most marketing analytics teams cannot staff. And teams respond to counterintuitive results by challenging the model rather than testing the insight, because no pre-agreed decision protocol exists.

### O6. Timelines exceed organizational patience

First builds: 1–4 months for SaaS onboarding (entirely dependent on the customer's data condition), 3+ months for custom builds — inside enterprise approval cycles that can run 8+ months. The documented failure cascade: data worse than assessed → missed deadlines → implausible first results → investigation loops → cost overruns → generic outputs → patience exhausted. The "free open-source" path routes through certified-partner consultancies (the Meridian partner ecosystem is itself the tell that the lift sits in data, priors, calibration, and workflow — not code).

### O7. The experiment program the sophisticated tier depends on is itself out of reach

Experiment-calibrated MMM needs a steady stream of geo tests: practitioners prescribe 2–3+ per year, a named owner fluent in both modeling and experimentation, geo-targetable media, weekly geo-level sales data, and a 2–3 year commitment — while geo experiments only make economic sense at roughly $1M+/month spend. Holdouts collide with commercial politics (sales and regional stakeholders resist going dark in their markets); mid-test disruptions in control markets invalidate results; and calibration only works when the experiment's KPI, geography, and window match the model's — mismatches create "reconciliation headaches rather than trust-building calibration." Most organizations instead test episodically, "when there's budget anxiety."

---

## Overall synthesis: one stack, three floors

Reading the three parts together, non-adoption is over-determined, and differently at each tier of the market:

**Bottom of the market (the numerical majority):** rational economics. Below ~$1M spend the model is neither estimable (M2) nor worth its cost (B1). No amount of tooling progress changes this; only radically cheaper experiments (Google's $5K tests) and fully-automated SaaS move the line.

**Mid-market:** the cost floor is now crossable, but platform dashboards are free, instant, and pointy (B5), the data archaeology is unbudgeted (O1), the talent doesn't exist in-house (O4), and channel-team incentives defend ROAS (B4). The math's honest outputs — priors and intervals — are competitive disadvantages against confident false precision (M7, M9).

**Enterprise:** can afford everything and mostly does buy MMM — but buys it as budget defense (B7), staffs it in a silo (B8/O5), can't produce the geo-level data its sophisticated tier requires (O3), can't run the calibrating experiment program (O7), watches conflicting tools and refresh flips erode trust (B6, M6), and has no decision protocol for acting on results. The 87%-use/28%-act gap is the measurement of exactly this.

The deepest common root spans all three lenses: **sophisticated MMM's defining feature is that it makes uncertainty and assumptions explicit, and the modern marketing organization is institutionally structured to punish explicit uncertainty** — in vendor selection, in budget meetings, in career incentives. The technical frontier (`02_open_questions.md`) will not fix this; the fixes that would are decision-theoretic and organizational: decision-framed outputs (probability-of-regret rather than intervals), pre-registered model specs and decision protocols, incentive separation between measurement and channel ownership, and cheap-enough experiments that trust can be bought empirically rather than argued statistically.

*Evidence quality note:* adoption statistics, cost structures, last-click persistence, platform self-grading bias, and tool instability cases are survey-, experiment-, or document-evidenced; the "wide intervals lose meetings" and CMO-tenure mechanisms are consistent multi-source practitioner testimony rather than surveyed fact; and the absence of published failure post-mortems (itself documented) means the failure base rate is unknowable — several key surveys (Kantar/Meta, HBR/Google) are platform-sponsored and should be weighted accordingly.
