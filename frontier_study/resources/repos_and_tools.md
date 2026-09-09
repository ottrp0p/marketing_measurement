# Resource Library — Repos, Tools, Datasets

The working codebases of frontier marketing measurement. All are clone-able starting points for experiments in this repo.

## Media mix modeling

- **Google Meridian** — https://github.com/google/meridian · docs: https://developers.google.com/meridian — Hierarchical geo-level Bayesian MMM (TensorFlow Probability, GPU NUTS), reach/frequency, paid-search causal correction, experiment-calibrated ROI priors. The current reference implementation.
- **Meta Robyn** — https://github.com/facebookexperimental/Robyn — Ridge regression + Nevergrad evolutionary hyperparameter search + budget allocator. Automation-first, non-Bayesian; good foil to Meridian.
- **PyMC-Marketing** — https://github.com/pymc-labs/pymc-marketing · docs: https://www.pymc-marketing.io — Bayesian MMM + CLV; lift-test likelihood calibration (https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_roas.html) is the cleanest open implementation of experiment-MMM fusion.
- **LightweightMMM** (superseded by Meridian, still instructive) — https://github.com/google/lightweight_mmm — Pioneered NumPyro/JAX GPU sampling for MMM.
- **Uber Orbit** — https://github.com/uber/orbit — Bayesian structural time series / KTR; the time-varying-coefficient machinery.

## Geo experimentation

- **Meta GeoLift** — https://github.com/facebookincubator/GeoLift · methodology: https://facebookincubator.github.io/GeoLift/docs/Methodology/ — Augmented synthetic control geo lift with simulation-based power analysis and market selection.
- **google/trimmed_match** — https://github.com/google/trimmed_match — Robust iROAS estimation + power-optimal geo pairing design.
- **google/matched_markets** — https://github.com/google/matched_markets — Time-based-regression geo experiment design/analysis.
- **CausalImpact** — https://github.com/google/CausalImpact (R) · Python: https://github.com/WillianFuks/tfcausalimpact — Bayesian structural time-series counterfactual for single-market interventions.

## Causal inference & uplift

- **EconML** — https://github.com/py-why/EconML — Microsoft's DML/DR-learner/Deep IV/policy-learning library (PyWhy ecosystem).
- **DoWhy** — https://github.com/py-why/dowhy — Causal graphs, identification, refutation tests; the assumptions-first companion to EconML.
- **CausalML** — https://github.com/uber/causalml — Uber's uplift modeling and CATE library (meta-learners, uplift trees).
- **grf** — https://github.com/grf-labs/grf · https://grf-labs.github.io/grf/ — Generalized random forests: causal/IV/quantile forests, policy learning.
- **DoubleML** — https://github.com/DoubleML/doubleml-for-py — Reference double machine learning implementation; literature index: https://docs.doubleml.org/stable/literature/literature.html
- **synthdid** — https://github.com/synth-inference/synthdid — Synthetic difference-in-differences (Arkhangelsky et al.) reference code.
- **augsynth** — https://github.com/ebenmichael/augsynth — Augmented synthetic control (Ben-Michael, Feller & Rothstein).

## Experimentation & simulation

- **PlanOut** — https://github.com/facebookarchive/planout — Meta's experiments-as-programs DSL (archived, conceptually foundational).
- **AuctionGym** — https://github.com/amzn/auction-gym — Amazon's ad-auction simulator for offline evaluation of bidding/measurement policies with counterfactual ground truth.
- **genagents** — https://github.com/joonspk-research/genagents — Stanford's interview-grounded generative agents (the "1,000 people" paper) — starting point for synthetic-consumer experiments.
- **fbpcf / fbpcs** — https://github.com/facebookresearch/fbpcf · https://github.com/facebookresearch/fbpcs — Meta's MPC private computation framework (Private Lift).

## Datasets & benchmarks

- **Criteo Uplift Prediction Dataset** — https://ailab.criteo.com/criteo-uplift-prediction-dataset/ — 25M rows; the standard large-scale uplift/incrementality benchmark.
- **Criteo large-scale ITE benchmark** — https://github.com/criteo-research/large-scale-ITE-UM-benchmark (paper: https://arxiv.org/abs/2111.10106).

## Specs & standards worth tracking

- **WICG Attribution Reporting API** — https://github.com/WICG/attribution-reporting-api — Chrome/Android private attribution spec + TEE aggregation service design.
- **IETF PPM/DAP** — https://datatracker.ietf.org/doc/draft-ietf-ppm-dap/ — Distributed (Prio-style) secure aggregation standard.
- **IPA protocol** — https://eprint.iacr.org/2023/437 — Meta+Mozilla MPC attribution proposal.
