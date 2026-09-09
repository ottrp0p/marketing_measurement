# Resource Library — Engineering for High-Scale Measurement

Annotated bibliography. Direct PDF links given where freely available. See also `repos_and_tools.md` for the open-source codebases.

## Experimentation platforms & infrastructure

1. **Trustworthy Online Controlled Experiments** — Kohavi, Tang & Xu (2020, book). The canonical engineering reference for A/B platforms at scale. https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/D97B26382EB0EB2DC2019A7A7B518F59
2. **The Anatomy of a Large-Scale Online Experimentation Platform** — Gupta et al., Microsoft (ICSE-SEIP 2018). Microsoft ExP architecture; the experimentation maturity model. https://exp-platform.com/large-scale/
3. **Designing and Deploying Online Field Experiments (PlanOut)** — Bakshy, Eckles & Bernstein, Facebook (WWW 2014). Experiments-as-programs DSL underlying modern assignment services. https://arxiv.org/abs/1409.3174 · PDF: https://arxiv.org/pdf/1409.3174
4. **Engineering for a Science-Centric Experimentation Platform** — Diamantopoulos et al., Netflix (ICSE-SEIP 2020). Netflix platform rebuild: reproducible stats engines, democratized metrics. https://arxiv.org/abs/1910.03878 · PDF: https://arxiv.org/pdf/1910.03878
5. **Reimagining Experimentation Analysis at Netflix** — Netflix TechBlog (2019). Notebook-based extensible analysis architecture. https://netflixtechblog.com/reimagining-experimentation-analysis-at-netflix-71356393af21
6. **Interleaving in Online Experiments at Netflix** — Netflix TechBlog (2017). 100x-sensitivity two-stage interleaving funnel in production. https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55
7. **Large-Scale Validation and Analysis of Interleaved Search Evaluation** — Chapelle, Joachims et al. (TOIS 2012). Foundational evidence for interleaving's sensitivity advantage. PDF: https://www.cs.cornell.edu/~tj/publications/chapelle_etal_12a.pdf
8. **Harnessing Interleaving and Counterfactual Evaluation for Airbnb Search Ranking** — Airbnb (2025). Modern production interleaving + off-policy evaluation stack. https://arxiv.org/abs/2508.00751
9. **Scaling Airbnb's Experimentation Platform (ERF)** — Airbnb Engineering (2018). Declarative metric config + Airflow/Spark experiment reporting. https://medium.com/airbnb-engineering/https-medium-com-jonathan-parks-scaling-erf-23fd17c91166
10. **Under the Hood of Uber's Experimentation Platform** — Uber Engineering (2018). ~1,000+ concurrent experiments; A/B + bandits + sequential (mSPRT) monitoring. https://eng.uber.com/xp/
11. **Supercharging A/B Testing at Uber** — Uber Engineering (2023). Second-gen platform: interference guardrails, automated rollouts. https://www.uber.com/blog/supercharging-a-b-testing-at-uber/
12. **Confidence: An Experimentation Platform from Spotify** — Spotify Engineering (2023). Internal platform productized; assignment "coordinate systems." https://engineering.atspotify.com/2023/8/coming-soon-confidence-an-experimentation-platform-from-spotify
13. **CUPED: Improving the Sensitivity of Online Controlled Experiments** — Deng, Xu, Kohavi & Walker (WSDM 2013). The variance-reduction technique every platform now ships. PDF: https://robotics.stanford.edu/~ronnyk/2013-02CUPEDImprovingSensitivityOfControlledExperiments.pdf
14. **Top Challenges from the First Practical Online Controlled Experiments Summit** — Gupta et al., 13 companies (SIGKDD Explorations 2019). Cross-industry consensus on open experimentation-infra problems. PDF: https://www.kdd.org/exploration_files/June_2019_-_2.TopChallengesInPracticalOnlineControlledExperiments_.pdf

## Ad lift, ghost ads, budget-split

15. **Ghost Ads: Improving the Economics of Measuring Online Ad Effectiveness** — Johnson, Lewis & Nubbemeyer (JMR 2017). Foundational counterfactual-logging design replacing PSA holdouts. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2620078 · PDF: https://conference.nber.org/confer/2016/EoDs16/Johnson_Lewis_Nubbemeyer.pdf
16. **A Comparison of Approaches to Advertising Measurement** — Gordon, Zettelmeyer, Bhargava & Chapsky (Marketing Science 2019). RCTs vs. observational methods on 15 big Facebook experiments. https://pubsonline.informs.org/doi/10.1287/mksc.2018.1135 · PDF: https://www.kellogg.northwestern.edu/faculty/gordon_b/files/fb_comparison.pdf
17. **Consumer Heterogeneity and Paid Search Effectiveness** — Blake, Nosko & Tadelis (Econometrica 2015). The landmark eBay paid-search shutoff experiment. PDF: https://faculty.haas.berkeley.edu/stadelis/BNT_ECMA_rev.pdf
18. **Designing Experiments to Measure Incrementality on Facebook** — Liu et al. (2018). Practical design guide for platform conversion-lift RCTs. https://arxiv.org/abs/1806.02588 · PDF: https://arxiv.org/pdf/1806.02588
19. **Predicted Incrementality by Experimentation (PIE)** — Gordon, Moakler & Zettelmeyer (2023/2025; NBER w35044). ML trained on ~5,000 lift RCTs to predict incrementality without new experiments. https://arxiv.org/abs/2304.06828 · PDF: https://arxiv.org/pdf/2304.06828
20. **Characterizing and Minimizing Divergent Delivery in Meta Advertising Experiments** — Burtch, Moakler, Gordon et al. (2025). 182K experiments; how delivery algorithms contaminate ad A/B tests. https://arxiv.org/abs/2508.21251 · PDF: https://arxiv.org/pdf/2508.21251
21. **Randomization and the Pernicious Effects of Limited Budgets on Auction Experiments** — Basse, Soufiani & Lambert (AISTATS 2016). Why naive user-split ad experiments break under budget constraints. https://arxiv.org/abs/1605.09171 · PDF: https://arxiv.org/pdf/1605.09171
22. **Trustworthy Online Marketplace Experimentation with Budget-split Design** — Liu et al., LinkedIn (KDD 2021). Production budget-split infrastructure for ads-marketplace experiments. https://arxiv.org/abs/2012.08724 · PDF: https://arxiv.org/pdf/2012.08724
23. **Valid and Unobtrusive Measurement of Returns to Advertising through Asymmetric Budget Split** — Amazon-affiliated authors (2022). Always-on ROAS measurement via small asymmetric budget perturbations. https://arxiv.org/abs/2207.00206
24. **How DoorDash Ads Uses Budget A/B Experimentation** — DoorDash Engineering (2024). Budget-split experimentation in a production retail-media system. https://careersatdoordash.com/blog/doordash-ads-uses-budget-a-b-experimentation/
25. **Media Measurement and the Assisted Own Goal** — Runge et al. (2026). Argues attribution/MMM must anchor to individual-level incrementality. https://arxiv.org/abs/2607.09608
26. **Privacy-Robust Incrementality Measurement under Signal Loss** — (2026). Frontier work on lift estimation under ATT/consent-driven signal loss. https://arxiv.org/abs/2606.03878
27. **Incrementality-Focused Messaging Measurement: All the Time, Everywhere** — Beaumont, Netflix (2021). Always-on persistent-holdout design for owned channels. https://medium.com/notificationsblog/incrementality-focused-messaging-measurement-all-the-time-everywhere-94ef0c229368

## Geo experiments

28. **Measuring Ad Effectiveness Using Geo Experiments** — Vaver & Koehler, Google (2011). The original geo-based-regression framework. https://research.google/pubs/measuring-ad-effectiveness-using-geo-experiments/
29. **Estimating Ad Effectiveness using Geo Experiments in a Time-Based Regression Framework** — Kerman, Wang & Vaver, Google (2017). TBR for few-geo cases; ships in matched_markets. https://research.google/pubs/estimating-ad-effectiveness-using-geo-experiments-in-a-time-based-regression-framework/
30. **Robust Causal Inference for Incremental Return on Ad Spend (Trimmed Match)** — Chen & Au, Google (AOAS 2022). Robust iROAS estimator for randomized paired geo experiments. https://arxiv.org/abs/1908.02922 · PDF: https://arxiv.org/pdf/1908.02922
31. **Trimmed Match Design for Randomized Paired Geo Experiments** — Chen et al., Google (2021). Power-optimal geo pairing, open-sourced. https://arxiv.org/abs/2105.07060 · PDF: https://arxiv.org/pdf/2105.07060
32. **Optimized Supergeo Design** — (2025). Scalable interference-aware geo partitioning. https://arxiv.org/abs/2506.20499

## Open-source MMM & Bayesian scaling

33. **Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects** — Jin, Wang, Sun, Chan & Koehler, Google (2017). The adstock+Hill Bayesian MMM every modern package descends from. PDF: https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/b20467a5c27b86c08cceed56fc72ceadb875184a.pdf
34. **Geo-level Bayesian Hierarchical Media Mix Modeling** — Sun et al., Google (2017). Hierarchical pooling across geos — the scale trick behind Meridian. https://research.google/pubs/geo-level-bayesian-hierarchical-media-mix-modeling/
35. **Bayesian Hierarchical MMM Incorporating Reach and Frequency** — Google (2023). Methodological core of Meridian's R&F treatment. https://research.google/pubs/bayesian-hierarchical-media-mix-model-incorporating-reach-and-frequency-data/
36. **Packaging Up Media Mix Modeling: Robyn's Open-Source Approach** — Runge, Skokan, Zhou & Pauwels (2024). Design rationale for democratized MMM tooling. https://arxiv.org/abs/2403.14674 · PDF: https://arxiv.org/pdf/2403.14674
37. **MCMC for Big Datasets: How Much Faster Is JAX and GPU?** — PyMC Labs (2022). The benchmark quantifying the JAX/GPU NUTS speedup enabling daily MMM refresh. https://www.pymc-labs.com/blog-posts/pymc-stan-benchmark
38. **Orbit: Bayesian Forecasting for Time Series** — Uber (2020). Production Bayesian structural time series used in Uber marketing science. https://arxiv.org/abs/2004.08492 · PDF: https://arxiv.org/pdf/2004.08492
39. **Bayesian Time-Varying Coefficient Model with Applications to MMM** — Ng, Wang et al., Uber (2021). Time-varying-coefficient MMM (KTR). https://arxiv.org/abs/2106.03322
40. **Bayesian Media Mix Modeling using PyMC3, for Fun and Profit** — HelloFresh Engineering (2020). Honest production MMM engineering account. https://engineering.hellofresh.com/bayesian-media-mix-modeling-using-pymc3-for-fun-and-profit-2bd4667504e6
41. **Building Lyft's Marketing Automation Platform (Symphony)** — Lyft Engineering (2019). Closed-loop LTV → budget → bid automation architecture. https://eng.lyft.com/lyft-marketing-automation-b43b7b7537cc
42. **Engineering to Scale Paid Media Campaigns** — Netflix TechBlog (2023). Netflix's marketing-tech stack for campaign scale. https://netflixtechblog.com/engineering-to-scale-paid-media-campaigns-84ba018fb3fa

## Privacy-preserving measurement

43. **Attribution Reporting API + Aggregation Service (TEE)** — WICG/Google (2021+). The deployed TEE+DP aggregate measurement architecture. https://github.com/WICG/attribution-reporting-api/blob/main/AGGREGATION_SERVICE_TEE.md · https://privacysandbox.google.com/private-advertising/aggregation-service/how-it-works
44. **Interoperable Private Attribution (IPA)** — Case, Jain, Koshelev, Masny et al., Meta+Mozilla (2023). The reference MPC protocol for cross-site attribution. https://eprint.iacr.org/2023/437
45. **Privacy-Preserving Attribution for Advertising** — Mozilla (2022). Browser-vendor framing of IPA/PPA. https://blog.mozilla.org/en/mozilla/privacy-preserving-attribution-for-advertising/
46. **fbpcf — Private Computation Framework (Private Lift)** — Meta (2021+). Production MPC lift measurement. https://github.com/facebookresearch/fbpcf · scale-out: https://github.com/facebookresearch/fbpcs
47. **Cookie Monster: Efficient On-device Budgeting for DP Ad-Measurement** — Columbia/Meta (SOSP 2024). Individual-DP budget management for ARA-class systems. https://arxiv.org/abs/2405.16719
48. **Differentially Private Ad Conversion Measurement** — Delaney et al., Google (PoPETs 2024). Formal DP framework for attribution rules and contribution bounding. PDF: https://petsymposium.org/popets/2024/popets-2024-0044.pdf
49. **Online Advertising Measurement via Per-User Differential Privacy** — Google (2024). Better per-user DP accounting for ads reporting. https://arxiv.org/abs/2406.02463 · PDF: https://arxiv.org/pdf/2406.02463
50. **IETF Distributed Aggregation Protocol (DAP)** — IETF PPM WG (2022–2026). Standardizing Prio-style secure aggregation. https://datatracker.ietf.org/doc/draft-ietf-ppm-dap/
51. **Ads Data Hub privacy checks** — Google (docs). k-aggregation + difference-check + noise engineering in a production clean room. https://developers.google.com/ads-data-hub/guides/privacy-checks
52. **AWS Clean Rooms Differential Privacy** — AWS (2024, docs). GA differential privacy in a general-purpose clean room. https://docs.aws.amazon.com/clean-rooms/latest/userguide/differential-privacy.html
53. **AdAttributionKit** — Apple (2024+, docs). Apple's crowd-anonymity postback attribution framework (SKAN successor). https://developer.apple.com/app-store/ad-attribution/ · comparison: https://www.adjust.com/blog/adattributionkit/

## ML systems, uplift, marketplaces, simulation

54. **Modeling Delayed Feedback in Display Advertising** — Chapelle, Criteo (KDD 2014). Foundational conversion-delay model for streaming training loops. PDF: http://wnzhang.net/share/rtb-papers/delayed-feedback.pdf
55. **A Large-Scale Benchmark for Uplift Modeling** — Diemert et al., Criteo (AdKDD 2018). 25M-row uplift benchmark. PDF: http://papers.adkdd.org/2018/papers/adkdd18-diemert-large-scale.pdf · data: https://ailab.criteo.com/criteo-uplift-prediction-dataset/ · ITE successor: https://arxiv.org/abs/2111.10106
56. **Interpretable Deep Learning Model for Online Multi-touch Attribution** — Yao et al. (2020). Additive-hazard deep MTA; representative of deployed deep attribution. https://arxiv.org/abs/2004.00384
57. **Deep Neural Net with Attention for Multi-channel Multi-touch Attribution** — Li et al. (2018). Attention-based journey attribution. https://arxiv.org/abs/1809.02230
58. **Design and Analysis of Switchback Experiments** — Bojinov, Simchi-Levi & Zhao (Mgmt Sci 2023). Optimal switchback design theory. https://arxiv.org/abs/2009.00148 · PDF: https://arxiv.org/pdf/2009.00148
59. **Switchback Experiments under Geometric Mixing** — Hu & Wager (2022). Sharper switchback analysis under carryover. https://arxiv.org/abs/2209.00197
60. **Switchback Tests and Randomized Experimentation under Network Effects at DoorDash** — DoorDash (2018+). Production switchback playbook. https://careersatdoordash.com/blog/switchback-tests-and-randomized-experimentation-under-network-effects-at-doordash/ · cluster-robust SEs: https://careersatdoordash.com/blog/cluster-robust-standard-error-in-switchback-experiments/
61. **Graph Cluster Randomization** — Ugander, Karrer, Backstrom & Kleinberg (KDD 2013). Foundational network-interference experiment design. https://arxiv.org/abs/1305.6979
62. **Network Experimentation at Scale** — Karrer et al., Facebook (2020). Meta's production cluster-experimentation system. https://arxiv.org/abs/2012.08591 · PDF: https://arxiv.org/pdf/2012.08591
63. **Reducing Interference Bias in Online Marketplace Experiments Using Cluster Randomization** — Holtz et al., Airbnb (Mgmt Sci 2024). Real-marketplace quantification of interference-bias reduction. https://pubsonline.informs.org/doi/10.1287/mnsc.2020.01157
64. **AuctionGym** — Jeunen et al., Amazon (AdKDD 2022 best paper). Reference ad-auction simulator for offline policy/measurement evaluation. https://github.com/amzn/auction-gym · https://www.amazon.science/blog/amazon-scientists-win-best-paper-award-for-ad-auction-simulator
65. **LLM-Based Multi-Agent System for Simulating Marketing and Consumer Behavior** — (2025). State of the art in LLM-agent marketing simulation. https://arxiv.org/abs/2510.18155 · field survey: https://arxiv.org/abs/2501.08579
