# Resource Library — Mathematics & Statistics of Marketing Measurement

Annotated bibliography of the identification, estimation, design, and decision-theory literature.

## Causal inference core

1. **Double/Debiased Machine Learning for Treatment and Structural Parameters** — Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins (2016/2018). Neyman orthogonality + cross-fitting: the recipe underlying ML-based lift estimation. https://arxiv.org/abs/1608.00060 · PDF: https://arxiv.org/pdf/1608.00060
2. **Estimation and Inference of Heterogeneous Treatment Effects using Random Forests** — Wager & Athey (JASA 2018). Honest causal forests: pointwise-valid CATE inference for targeting/uplift. https://arxiv.org/abs/1510.04342
3. **Generalized Random Forests (grf)** — Athey, Tibshirani & Wager (2019). Production implementation: causal/IV/quantile forests, policy learning. https://grf-labs.github.io/grf/
4. **Synthetic Difference-in-Differences** — Arkhangelsky, Athey, Hirshberg, Imbens & Wager (AER 2021). Unit+time reweighting; the modern default for geo rollout evaluation. https://arxiv.org/abs/1812.09970 · PDF: https://arxiv.org/pdf/1812.09970
5. **The Augmented Synthetic Control Method** — Ben-Michael, Feller & Rothstein (JASA 2021). Bias-corrected SC when pre-treatment fit is imperfect. https://arxiv.org/abs/1811.04170
6. **Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects** — Abadie (JEL 2021). Authoritative user's guide to SC assumptions and pitfalls. https://www.aeaweb.org/articles?id=10.1257/jel.20191450
7. **Matrix Completion Methods for Causal Panel Data Models** — Athey, Bayati, Doudchenko, Imbens & Khosravi (JASA 2021). Low-rank factor imputation unifying SC and fixed effects for geo panels. https://arxiv.org/abs/1710.10251
8. **Inferring Causal Impact Using Bayesian Structural Time-Series Models** — Brodersen, Gallusser, Koehler, Remy & Scott (AoAS 2015). CausalImpact: the workhorse single-market counterfactual method. https://arxiv.org/abs/1506.00356
9. **An Exact and Robust Conformal Inference Method for Counterfactual and Synthetic Controls** — Chernozhukov, Wüthrich & Zhu (JASA 2021). Finite-sample-valid tests fixing SC's inference gap. https://arxiv.org/abs/1712.09089
10. **Causal Inference and the Data-Fusion Problem** — Bareinboim & Pearl (PNAS 2016). The complete calculus for combining experiments + observational data across domains. https://www.pnas.org/doi/abs/10.1073/pnas.1510507113

## MMM identification & calibration

11. **Challenges and Opportunities in Media Mix Modeling** — Chan & Perry, Google (2017). The canonical diagnosis of MMM's selection-bias and data problems. https://research.google/pubs/challenges-and-opportunities-in-media-mix-modeling/ · PDF: https://research.google.com/pubs/archive/2d0395bc7d4d13ddedef54d744ba7748e8ba8dd1.pdf
12. **Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects** — Jin, Wang, Sun, Chan & Koehler (2017). Defines adstock+Hill Bayesian MMM and demonstrates its weak identification. PDF: https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/b20467a5c27b86c08cceed56fc72ceadb875184a.pdf
13. **Geo-level Bayesian Hierarchical Media Mix Modeling** — Sun, Wang, Jin, Chan & Koehler (2017). Partial pooling across geos: the main lever against MMM data poverty. https://research.google/pubs/geo-level-bayesian-hierarchical-media-mix-modeling/
14. **Bias Correction for Paid Search in Media Mix Modeling** — Chen et al., Google (2018). Backdoor adjustment with search-query volume to de-confound search ROAS. https://arxiv.org/abs/1807.03292
15. **Media Mix Model Calibration With Bayesian Priors** — Zhang, Wurm et al., Google (2024). How lift experiments become ROI priors (the Meridian calibration method). https://research.google/pubs/media-mix-model-calibration-with-bayesian-priors/ · PDF: https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/a09f404fdc3107fafb7a52cc5af6a80e4d0fda2b.pdf
16. **Structural Estimation of MMM Parameters from Geo-Experiments** — Heusch (2026). Frontier: recovers adstock, saturation, effectiveness structurally from experiments. https://arxiv.org/abs/2608.21128
17. **Bayesian Time-Varying Coefficient Model with Applications to MMM** — Ng et al. (2021). Handles drifting channel effectiveness. https://arxiv.org/abs/2106.03322
18. **Packaging Up Media Mix Modeling (Robyn)** — Runge et al. (2024). Documents — and lets you critique — Robyn's ridge+evolutionary methodology. https://arxiv.org/abs/2403.14674
19. **NNN: Next-Generation Neural Networks for Marketing Mix Modeling** — Google (2025). Transformer-based MMM; the deep-learning frontier and its identifiability trade-offs. https://arxiv.org/abs/2504.06212
20. **Meridian: MMM as Causal Inference (docs)** — Google (2024–26). The most explicit production statement of MMM's causal assumptions. https://developers.google.com/meridian/docs/causal-inference/about-mmm-causal-inference-methodology
21. **Trimmed Match (analysis + design)** — Chen & Au; Chen, Longfils & Remy, Google (AoAS 2022; 2021). Robust iROAS estimation/design for paired geo tests. https://arxiv.org/abs/1908.02922 · https://arxiv.org/abs/2105.07060
22. **Optimized Supergeo Design** — (2025). Scalable interference-aware geo partitioning. https://arxiv.org/abs/2506.20499
23. **Time-Based Regression geo framework** — Kerman, Wang & Vaver, Google (2017). The TBR framework behind Google's geo-lift tooling. https://research.google/pubs/estimating-ad-effectiveness-using-geo-experiments-in-a-time-based-regression-framework/

## Experiments vs. observational methods; effect sizes

24. **A Comparison of Approaches to Advertising Measurement** — Gordon, Zettelmeyer, Bhargava & Chapsky (Marketing Science 2019). The key paper: observational methods fail against RCT ground truth. https://pubsonline.informs.org/doi/10.1287/mksc.2018.1135 · PDF: https://gwern.net/doc/statistics/causality/2019-gordon.pdf
25. **Close Enough? A Large-Scale Exploration of Non-Experimental Approaches** — Gordon, Moakler & Zettelmeyer (Marketing Science 2023). 663 RCTs: even DML with rich covariates can't reproduce experimental lift. https://arxiv.org/abs/2201.07055
26. **Predicted Incrementality by Experimentation (PIE)** — Gordon, Moakler & Zettelmeyer (NBER w35044). Experiments as training data for predicting incrementality. https://arxiv.org/abs/2304.06828
27. **Consumer Heterogeneity and Paid Search Effectiveness** — Blake, Nosko & Tadelis (Econometrica 2015). The eBay null: founding document of incrementality skepticism. https://www.nber.org/papers/w20171 · PDF: https://faculty.haas.berkeley.edu/stadelis/BNT_ECMA_rev.pdf
28. **The Unfavorable Economics of Measuring the Returns to Advertising** — Lewis & Rao (QJE 2015). Why ad-ROI confidence intervals are enormous even at huge n. https://academic.oup.com/qje/article-abstract/130/4/1941/1914592 · PDF: https://gwern.net/doc/economics/advertising/2015-lewis.pdf
29. **Ghost Ads** — Johnson, Lewis & Nubbemeyer (JMR 2017). Counterfactual-exposure logging: cheap, unbiased ad experiments. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2620078
30. **TV Advertising Effectiveness and Profitability: Generalizable Results from 288 Brands** — Shapiro, Hitsch & Tuchman (Econometrica 2021). Distribution of TV elasticities; publication bias quantified. https://www.econometricsociety.org/publications/econometrica/2021/07/01/tv-advertising-effectiveness-and-profitability-generalizable · companion PDF: https://www.nber.org/system/files/working_papers/w27684/w27684.pdf
31. **Where A/B Testing Goes Wrong: Divergent Delivery** — Braun & Schwartz (Journal of Marketing 2025). Delivery algorithms confound creative comparisons in platform experiments. https://journals.sagepub.com/doi/10.1177/00222429241275886
32. **Characterizing and Minimizing Divergent Delivery** — Meta (2025). Meta's quantification and mitigation. https://arxiv.org/abs/2508.21251
33. **The Minimal Persuasive Effects of Campaign Contact in General Elections** — Kalla & Broockman (APSR 2018). 49 field experiments: persuasion ≈ 0; benchmark for realistic priors. https://www.cambridge.org/core/journals/american-political-science-review/article/abs/minimal-persuasive-effects-of-campaign-contact-in-general-elections-evidence-from-49-field-experiments/753665A313C4AB433DBF7110299B7433
34. **A 2-Million-Person, Campaign-Wide Field Experiment** — Aggarwal, Coppock et al. (2023). Campaign-scale RCT with precise near-null persuasion estimates. PDF: https://alexandercoppock.com/aggarwal_etal_2023.pdf

## Interference & marketplaces

35. **Experimental Design in Two-Sided Platforms: An Analysis of Bias** — Johari, Li, Liskovich & Weintraub (Mgmt Sci 2022). Formal model of why marketplace A/B tests are first-order biased. https://arxiv.org/abs/2002.05670
36. **Interference, Bias, and Variance in Two-Sided Marketplace Experimentation** — Li, Johari et al. (WWW 2022). Design guidance mapping bias/variance across randomization schemes. https://arxiv.org/abs/2104.12222
37. **Design and Analysis of Switchback Experiments** — Bojinov, Simchi-Levi & Zhao (Mgmt Sci 2023). Minimax-optimal switchback design under carryover. https://arxiv.org/abs/2009.00148
38. **Design and Analysis of Experiments in Networks** — Eckles, Karrer & Ugander (JCI 2017). Graph-cluster randomization: the standard interference-mitigation design. https://arxiv.org/abs/1404.7530
39. **Causal Inference with Misspecified Exposure Mappings** — Sävje (2021+). What exposure-mapping estimators recover when the mapping is wrong. https://arxiv.org/abs/2103.06471
40. **Estimating Average Causal Effects Under General Interference** — Aronow & Samii (AoAS 2017). Exposure mappings + Horvitz-Thompson under known interference. https://arxiv.org/abs/1305.6156
41. **Online Causal Inference for Advertising in Real-Time Bidding Auctions** — Waisman, Nair & Misra (Marketing Science). Sequential ad-effect inference exploiting auction mechanics. https://arxiv.org/abs/1908.08600
42. **Auction Throttling and Causal Inference of Online Advertising Effects** — Gui, Nair et al. (2021). Throttling as quasi-randomization under auction interference. https://arxiv.org/abs/2112.15155
43. **Parallel Experimentation and Competitive Interference** — (2019). Simultaneous advertiser experiments interfering through shared auctions. https://arxiv.org/abs/1903.11198
44. **Competition and Crowd-Out for Brand Keywords in Sponsored Search** — Simonov, Nosko & Rao (Marketing Science 2018). Quantifies paid-organic cannibalization. https://pubsonline.informs.org/doi/10.1287/mksc.2017.1065
45. **Reducing Interference Bias Using Cluster Randomization (Airbnb)** — Holtz et al. (Mgmt Sci 2024). Meta-experiment measuring interference bias directly. https://pubsonline.informs.org/doi/10.1287/mnsc.2020.01157

## Attribution theory

46. **Beyond the Last Touch: Attribution in Online Advertising** — Berman (Marketing Science 2018). Attribution as incentive contract; formalizes attribution ≠ incrementality. https://pubsonline.informs.org/doi/10.1287/mksc.2018.1104 · PDF: https://www.ron-berman.com/papers/attribution.pdf
47. **Shapley Meets Uniform: An Axiomatic Framework for Attribution** — Besbes et al. (WWW 2019). Axiomatic analysis of Shapley credit and its approximations. https://dl.acm.org/doi/10.1145/3308558.3313731
48. **Mapping the Customer Journey: Graph-Based Online Attribution Modeling** — Anderl, Becker, von Wangenheim & Schumann (IJRM 2016). The Markov-chain removal-effect framework. https://www.sciencedirect.com/science/article/abs/pii/S0167811616300349
49. **CausalMTA: Eliminating User Confounding Bias for Causal Multi-touch Attribution** — Alibaba (KDD 2022). Deep MTA with explicit deconfounding. https://arxiv.org/abs/2201.00689
50. **Incrementality Bidding & Attribution** — Lewis & Wong (2018/2022). Formal objective for bidding on causal lift rather than attributed conversions. https://arxiv.org/abs/2208.12809

## Bandits, adaptive inference, off-policy evaluation, decision theory

51. **Counterfactual Reasoning and Learning Systems** — Bottou et al. (JMLR 2013). Founding paper of counterfactual (IPS) evaluation of ad systems. https://arxiv.org/abs/1209.2355
52. **Doubly Robust Policy Evaluation and Optimization** — Dudík, Erhan, Langford & Li (2014/15). The DR estimator underlying modern off-policy evaluation. https://arxiv.org/abs/1503.02834
53. **Confidence Intervals for Policy Evaluation in Adaptive Experiments** — Hadad, Hirshberg, Zhan, Wager & Athey (PNAS 2021). Valid inference after bandit-driven data collection. https://arxiv.org/abs/1911.02768
54. **Adaptive Treatment Assignment in Experiments for Policy Choice** — Kasy & Sautmann (Econometrica 2021). Exploration sampling: design optimized for the decision, not estimation. PDF: https://maxkasy.github.io/home/files/papers/adaptiveexperimentspolicy.pdf
55. **Test & Roll: Profit-Maximizing A/B Tests** — Feit & Berman (Marketing Science 2019). Closed-form Bayesian decision-theoretic test sizing. https://pubsonline.informs.org/doi/10.1287/mksc.2019.1194 · PDF: https://www.ron-berman.com/papers/testandroll.pdf

## Advanced identification, sensitivity, model uncertainty, critiques

56. **An Introduction to Proximal Causal Learning** — Tchetgen Tchetgen, Ying, Cui, Shi & Miao (2020). Identification with proxies for unmeasured demand confounders — high potential for MMM. https://arxiv.org/abs/2009.10982
57. **Making Sense of Sensitivity: Extending Omitted Variable Bias** — Cinelli & Hazlett (JRSS-B 2020). Robustness values / partial-R² bounds — the right audit tool for MMM coefficients. PDF: https://carloscinelli.com/files/Cinelli%20and%20Hazlett%20(2020)%20-%20Making%20Sense%20of%20Sensitivity.pdf
58. **The E-Value** — VanderWeele & Ding (Ann Intern Med 2017). Minimum confounding strength needed to explain away an estimate. https://www.acpjournals.org/doi/10.7326/M16-2607
59. **Even the Rich Can Make Themselves Poor: IV Methods in Marketing** — Rossi (Marketing Science 2014). The definitive critique of instrument-based identification in marketing. https://pubsonline.informs.org/doi/10.1287/mksc.2014.0860
60. **Deep IV: A Flexible Approach for Counterfactual Prediction** — Hartford, Lewis, Leyton-Brown & Taddy (ICML 2017). Neural two-stage IV when instruments exist. PDF: https://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf
61. **Surrogate Index / Surrogate Score** — Athey, Chetty, Imbens & Kang (2016; REStud 2025). Long-term outcomes from short experiments. https://arxiv.org/abs/1603.09326 · NBER w26463 PDF: https://www.nber.org/system/files/working_papers/w26463/w26463.pdf
62. **Using Stacking to Average Bayesian Predictive Distributions** — Yao, Vehtari, Simpson & Gelman (Bayesian Analysis 2018). Principled model combination for M-open settings like MMM ensembles. https://arxiv.org/abs/1704.02030
63. **Inference on Counterfactual Distributions** — Chernozhukov, Fernández-Val & Melly (Econometrica 2013). Uniform inference on distributional/quantile treatment effects. PDF: https://www.mit.edu/~vchern/papers/counterfactual_2012Nov1.pdf
64. **Beyond Power Calculations: Type S and Type M Errors** — Gelman & Carlin (2014). Why significant results from underpowered lift tests systematically exaggerate. PDF: https://sites.stat.columbia.edu/gelman/research/published/retropower_final.pdf
65. **p-Hacking and False Discovery in A/B Testing** — Berman, Pekelis, Scott & Van den Bulte (2018; Mgmt Sci 2021). Empirical optional stopping and false discovery in commercial experimentation. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3204791 · FDR paper PDF: https://ron-berman.com/papers/fdr.pdf
66. **Partial Identification in Econometrics** — Tamer (2010, survey). Entry point to Manski-style bounds — the honest fallback when ad effects aren't point-identified. PDF: https://scholar.harvard.edu/files/tamer/files/pie.pdf
