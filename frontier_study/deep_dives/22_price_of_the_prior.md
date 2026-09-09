# 22 — Price the prior: what an assumption is worth in excitation dollars

*Backlog BL75 (spawned by dive 20). Run of 2026-09-07. Full-depth protocol: four construct→attack→refine rounds, 16 numerical experiments, two robustness sweeps, four negative/power controls, an independent clean-room verification (two passes).*

---

## 1. Problem statement

Dive 20 established the cost–precision frontier for MMM identification: deliberately perturbing media spend costs profit dollars $C$ and buys Fisher information, with

$$C\cdot\operatorname{Var}(\hat c_1) \;=\; \tfrac12\,\kappa\,S_\varepsilon(\omega) \;\equiv\; A \qquad\Longrightarrow\qquad \operatorname{Var} = A/C,$$

and the optimal perpetual identification budget $L^\* = m\sigma\sqrt{\Psi\,T_r}$, where $\Psi$ is a *ladder* factor: $\Psi=1$ when the response shape (adstock, saturation) is known and $\Psi\approx 10$ when it is free.

Dive 20 also left a puzzle. The estimator MMM practitioners actually run is prior-regularised, and its realised RMSE ($0.229$) sat *inside* the frequentist CRLB ($0.311$) by accepting bias ($+0.107$). The frontier is an information statement; the Bayesian MMM appears to violate it. So:

1. Where does the prior-regularised MMM actually sit on the cost–precision plane, and what is the correct frontier for a biased estimator?
2. What is a prior *worth*, in the same dollars the frontier prices excitation in?
3. How wrong may a prior be before it costs more than it saves — and, since the mis-centring is unobservable, what should a practitioner actually *do*?
4. Priors in MMM are placed on very different objects — Meridian puts them on ROI, LightweightMMM on adstock decay, everyone else on channel coefficients. Which placement is worth the most dollars?

The dive answers all four. The core object is new: an exact **shape-prior ledger** that turns dive 20's discrete $\Psi$ ladder into a continuous function of prior precision, with matching bias and admissibility laws.

---

## 2. Prior state

**Established elsewhere (the literature scan confirms these are prior art, and the dive does not claim them).**

- *Prior effective sample size.* Morita, Thall & Müller (2008) define a prior's ESS by curvature matching; in the normal known-variance case $\mathrm{ESS}=\sigma^2/\tilde\sigma^2$ — literally, a prior of variance $\tilde\sigma^2$ carries the information of $\sigma^2/\tilde\sigma^2$ observations. Multiplying that by a marginal cost-per-observation is exactly the "dollar value of a calibrated prior" of §4.1. **§4.1 is a bridge, not a theorem.** Neuenschwander et al. (2020) show most ESS definitions fail a predictive-consistency axiom; §4.1's does satisfy additivity by construction (the information adds).
- *Shrinkage risk.* For $X\sim N(\theta,\sigma^2)$ with prior $N(\mu_0,\tau^2)$, the posterior mean beats the MLE in MSE iff $(\theta-\mu_0)^2 < 2\tau^2+\sigma^2$. Textbook (Efron–Morris, Casella 1985). §4.5's admissibility law is the *nuisance-parameter* analogue, which is not textbook.
- *van Trees / Bayesian CRB.* Gill & Levit (1995): the bound $1/(I_{\rm data}+I_{\rm prior})$ holds for **any** estimator, including biased ones. This is the resolution of dive 20's puzzle (§4.2) — the classical CRLB was simply the wrong frontier.
- *MMM practice.* Meridian reparameterises $\beta_{g,m}$ as a deterministic function of $\mathrm{ROI}_m$ and the shape parameters so that the prior sits on ROI, explicitly noting that the induced prior on $\beta$ is then *not* independent of $(\alpha_m,\mathrm{ec}_m,\mathrm{slope}_m)$. LightweightMMM places `lag_weight ~ Beta(2,1)` on adstock decay, chosen "based on simulated studies that produce stable results" — i.e. for MCMC stability, not calibration. Robyn injects lift tests as a third GA objective with weights its own docs call "subjective". Meridian ships a *prior–posterior shift* diagnostic but attaches no threshold to it. §4.6 and §5.2 supply the missing theory and the missing threshold.
- *Weak identification of shape.* Dew, Padilla & Shchetkina (2024): saturation and time-varying effectiveness are not separately identifiable from observational spend. Heusch (2026, arXiv:2608.21128) reports a posterior ridge with $\mathrm{corr}(\lambda,\beta)=-0.68$ whose *implied response curve* over the probed range is nevertheless tight. §4.3's residual sensitivity $d$ is exactly the object that makes that possible, and §4.4 makes it quantitative.

**What appears genuinely unclaimed** (the literature agent's verdict, which I take as the honest novelty boundary): a continuous $\Psi$ ladder as a Schur-complement function of nuisance-prior precision; a dollar-denominated bridge from ESS to an excitation-cost frontier; a "price of a mis-centred prior" in those same dollars; and the decision-layer asymmetry of §5.3. Nothing found prices priors on *different directions in parameter space* against each other.

**Internal:** dive 20 (the frontier, $L^\*$, the $\Psi$ ladder — this dive is its Bayesian completion), dive 21 (the comb design used here; BL82's estimability audit, which §6.4 discharges for this dive), dives 08/09 (the deadband, used in §5.3), dive 16/19 (frozen learning), dive 02 (transport, the source of the mis-centring priced in §5.1-A3), dive 11 (leakage). Catalogues `../02_open_questions.md` §M1, §M2, §M7; `../03_mmm_adoption_barriers.md` §O7, §B1.

---

## 3. Formalisation

Gaussian model, parameter vector $\theta$ partitioned as $(c,s)$: $c$ is a free block carrying no prior; $s$ is the **shape** block (adstock $\alpha$, Hill $K$, $S$) carrying a Gaussian prior of precision matrix $P$ centred at $s_0$. Write $\delta = s_{\rm true}-s_0$ for the **mis-centring** (unobservable). Let $I$ be the Fisher information with blocks $I_{cc},I_{cs},I_{ss}$. The estimator is the penalised MLE / posterior mode; in the linear-Gaussian case

$$\hat\theta = \theta + J^{-1}u - J^{-1}P_{\rm full}(\theta-\theta_{\rm prior}),\qquad J=I+P_{\rm full},\quad u\sim N(0,I).$$

The object of interest is a scalar **decision functional** $\phi=g'\theta$. Throughout, $\phi=\mathrm{mROAS}$ — the marginal return on a permanent $+\$1$/wk, evaluated at a *fixed observed* adstock level.

> **A degeneracy worth recording.** Evaluated at the profit *optimum*, $\mathrm{mROAS}\equiv 1$ for every $\theta$ — that is the first-order condition — so its gradient vanishes identically and the functional is degenerate. This is dive 20's "the profit-maximising plan is exactly non-identifying" in functional form, and it broke the dive's first parameterisation. The decision functional must be anchored at an *observed* spend level, not at the model's own optimum. Here $\bar a_0 = 0.85\,\bar a^\*$, giving $\mathrm{mROAS}=1.290$.

**Model and design.** Same structural MMM as dives 20/21: $\mu_t = \beta\,h(a_t;K,S)$, geometric adstock $\alpha$, $\theta_0=(\beta,K,S,\alpha)=(1.0,1.5,2.0,0.6)$, margin $m=2$, $\sigma=0.05$, $T=156$ weeks, a 3-harmonic seasonal basis plus linear trend profiled out. The canonical design is dive 21's comb: three tones at periods $\{11,21,34\}$ weeks, $10\%$ amplitude each ($12\%$ total spend CV). Steady weekly spend $0.872$; three-year media spend $136.1$ in revenue units.

**Assumptions, stated explicitly.** (A1) Gaussian likelihood, local (FIM) approximation — attacked in §5.1-A7 and found *conservative*. (A2) Gaussian prior — attacked in §5.1-A7 with a uniform prior. (A3) $\delta$ independent of the data. (A4) The design is fixed while the prior varies — attacked in §5.1-A4 and found second-order. (A5) The decision loss is quadratic in $\phi$ — replaced by dive 08's deadband in §5.3, which changes the answer materially.

---

## 4. Round 1 — the constructions

### 4.1 The dollar exchange rate (a bridge, not a theorem)

On dive 20's frontier, information from excitation is $I_C = C/A$. A prior of precision $p$ on the *target* adds $p$. Three regimes, all exact (`e2`):

| prior quality | risk | excitation-equivalent budget $C_{\rm eq}=A/\mathrm{MSE}$ |
|---|---|---|
| oracle ($\delta\equiv 0$) | $I_C/(I_C+p)^2$ | $A(I_C+p)^2/I_C$ — superlinear, unbounded |
| **calibrated** ($\mathbb E\delta^2=1/p$) | $1/(I_C+p)$ | $\;C + A\,p\;$ — **exchange rate exactly $A$ $/unit precision, constant** |
| mis-centred by fixed $\delta$ | $(I_C+p^2\delta^2)/(I_C+p)^2$ | $\to A/\delta^2$ as $p\to\infty$ — a **hard cap** |

Verified to 10 decimal places at three $(C,p)$ cells; the cap reproduced at four $\delta$; and the sign flip — the prior is worth *negative* dollars once $\delta^2 > 2/p + A/C$ — bracketed numerically ($+0.294 / 0.000 / -0.281$ at $0.9\times$, $1\times$, $1.1\times$ the threshold).

**Honest labelling.** The middle row is Morita–Thall–Müller's $\mathrm{ESS}=\sigma^2/\tilde\sigma^2$ multiplied by dive 20's marginal cost of information. The *bridge* is new; the mathematics is not. The cap $A/\delta^2$ and the sign flip follow from the textbook shrinkage crossover. §§4.3–4.7 are where this dive's own content is.

### 4.2 The van Trees attainment identity (resolving dive 20)

**Claim.** If $\delta$ is random with $\mathbb E[\delta\delta']=P^{-1}$ (a *calibrated* prior) and independent of the score, then

$$\mathbb E_\delta\!\left[\mathrm{MSE}(g'\hat\theta)\right] \;=\; g'(I+P_{\rm full})^{-1}g \qquad\textbf{exactly.}$$

*Proof.* $\mathbb E[\mathrm{MSE}] = g'J^{-1}IJ^{-1}g + g'J^{-1}P\,\mathbb E[\delta\delta']\,PJ^{-1}g = g'J^{-1}(I+P)J^{-1}g = g'J^{-1}g$. $\square$

Monte Carlo at three $(I,P)$ cells, $4\times10^5$ draws: $z=-0.24,\;+0.39,\;-0.11$ against the claim (`e3`). The clean-room verifier reproduced it algebraically to $6.7\times10^{-12}$ over 300 random cells and by six independent $10^6$-draw MC replications ($z\in[-2.11,+1.42]$).

**This dissolves dive 20's puzzle.** The classical CRLB bounds unbiased estimators only. The right frontier for a prior-regularised MMM is the van Trees bound $1/(I+P)$, and the penalised MLE attains it exactly — *provided the prior is calibrated*. Dive 20's "RMSE $0.229$ inside CRLB $0.311$" was not an anomaly and not an achievement; it was the estimator sitting on a different, lower frontier that it had purchased with an assumption. What §4.5 then asks is what happens when that assumption is false.

### 4.3 The shape-prior ledger

Now the substance. Partition as above and define

$$\boxed{\;\tilde I \;=\; I_{ss}-I_{sc}I_{cc}^{-1}I_{cs},\qquad d \;=\; g_s - I_{sc}I_{cc}^{-1}g_c,\qquad V_\infty = g_c'I_{cc}^{-1}g_c\;}$$

$\tilde I$ is the *profile* information the data carry about the shape; $d$ is the **residual shape sensitivity** — how much the decision functional still depends on the shape *after the free parameters have re-absorbed everything they can*. Then, exactly:

$$V_{\rm post}(P) = V_\infty + d'(\tilde I+P)^{-1}d,\qquad
\mathrm{bias}(P) = -\,d'(\tilde I+P)^{-1}P\,\delta,$$
$$V_{\rm samp}(P) = V_\infty + d'(\tilde I+P)^{-1}\tilde I(\tilde I+P)^{-1}d,\qquad
\mathrm{MSE}=V_{\rm samp}+\mathrm{bias}^2 .$$

*Derivation.* Block-inverting $I+P_{\rm full}$ gives $\Sigma_{ss}=(\tilde I+P)^{-1}$, $\Sigma_{cs}=-I_{cc}^{-1}I_{cs}\Sigma_{ss}$, $\Sigma_{cc}=I_{cc}^{-1}+I_{cc}^{-1}I_{cs}\Sigma_{ss}I_{sc}I_{cc}^{-1}$; collecting $g'\Sigma g$ collapses the three terms into $V_\infty + d'\Sigma_{ss}d$. The bias follows from $g'J^{-1}e_s = d'(\tilde I+P)^{-1}$, and $V_{\rm samp}=g'J^{-1}(J-P_{\rm full})J^{-1}g$. $\square$

Verified: posterior identity to $6.4\times10^{-16}$ (300 random 4-parameter cells with $g_s\neq0$), bias identity to $4.9\times10^{-14}$ (200 cells). Clean-room: $6.3\times10^{-15}$ and $6.3\times10^{-14}$ on its own implementation.

**Everything about a shape prior enters through one vector $d$ and one matrix $\tilde I$.** Two consequences are worth stating on their own:

- **The bias ceiling is $-d'\delta$.** However confident the shape prior, the harm it can do to the decision functional is bounded by the residual sensitivity dotted into the mis-centring. (Clean-room caveat: this limit requires $P$ *nonsingular*; with a rank-deficient shape prior only the components of $\delta$ in $\mathrm{range}(P)$ are pinned.)
- **If $d=0$, a shape prior is worth exactly nothing and can do exactly no harm** — for any $P$, any $\delta$. Verified to 12 decimals including at $P=10^9$ and $|\delta|\sim100$ (`e8`-N1; clean-room confirms, and adds that the *sampling* variance is also unchanged, not just the posterior variance). This is the formal content of the Heusch/Dew observation that a posterior ridge between $\lambda$ and $\beta$ can coexist with a tightly identified response curve: what matters is not how badly the shape is identified but how much of it survives re-fitting the free parameters.

### 4.4 The continuous $\Psi$ ladder

Define $\Psi(P) = V_{\rm post}(P)/V_\infty = 1 + d'(\tilde I+P)^{-1}d/V_\infty$. This *is* dive 20's ladder, now continuous in prior precision. Two forms:

**Spectral.** Whitening by $\tilde I$ and diagonalising,
$$\Psi(P) = 1 + \sum_k \frac{\rho_k^2}{1+u_k},\qquad \sum_k\rho_k^2 = \Psi_0-1,$$
with $u_k$ the eigenvalues of $\tilde I^{-1/2}P\tilde I^{-1/2}$ and $\rho_k^2$ the squared whitened residual sensitivity in direction $k$. **A prior pays only in directions the functional actually leaks into.** Verified: prior precision along a direction with $n'\tilde I^{-1}d=0$ leaves $\Psi$ unchanged to 12 decimals even at $t=10^8$; along $\tilde I^{-1}d$ it drives $\Psi\to1$ (`e8`-N2).

**Isotropic (the one-number ladder).** For $P=u\tilde I$ — a prior as strong as the data's own shape information, scaled by $u$ —
$$\boxed{\;\Psi(u) \;=\; 1+\frac{\Psi_0-1}{1+u}\;}$$
exact to $7.9\times10^{-16}$ across $u\in[0,100]$ on the MMM FIM; $9.8\times10^{-16}$ over 400 random cells (clean-room).

The scalar special case ($g_s=0$, one nuisance) is $\Psi(u')=(1+u')/(1+u'-\rho^2)$ with $u'=p/I_{ss}$, $\rho^2=I_{cs}^2/(I_{cc}I_{ss})$ — verified to $2.0\times10^{-15}$. **Clean-room caveat, adopted:** this form requires the *free block to be one-dimensional*, not merely the nuisance to be scalar. With $\dim c>1$ the correct general form carries a cosine factor, $\Psi = 1+\rho^2\cos^2\theta/(1-\rho^2+u')$ with $\cos^2\theta=(I_{cs}'I_{cc}^{-1}g_c)^2/\big[(I_{cs}'I_{cc}^{-1}I_{cs})(g_c'I_{cc}^{-1}g_c)\big]$, and the uncorrected formula over-states $\Psi$ by up to $69\%$ in random draws. The isotropic form above is the safe general statement.

**On the MMM.** $\Psi_0=19.804$, $\sqrt{V_\infty}=0.1187$, $\mathrm{sd}(\mathrm{mROAS})=0.5283$ free, $d=(+6.478,-0.229,+0.113)$ against a raw gradient $g_s=(+3.226,+0.359,+0.588)$. Note $|d_\alpha|>|g_{s,\alpha}|$: re-fitting $\beta$ *amplifies* the functional's dependence on adstock rather than absorbing it. Per-eigendirection $\rho_k^2 = (1.535,\,5.256,\,12.013)$ summing to $\Psi_0-1=18.804$ to $2\times10^{-14}$. All independently reproduced by the clean-room verifier to 5–6 significant figures.

### 4.5 Admissibility: how wrong may a prior be?

**Claim (scalar nuisance, arbitrary $g$).** The penalised estimator beats the flat-prior estimator in frequentist MSE of $\phi$ iff

$$\boxed{\;\delta^2 \;<\; \frac{2}{p} + \frac{1}{\tilde I} \;=\; 2\tau^2 + \mathrm{sd}_{\rm data}^2\;}$$

**independent of $d$ and of $V_\infty$.** *Proof.* $\mathrm{MSE}-V_\infty = d^2[\tilde I + p^2\delta^2]/(\tilde I+p)^2$ versus $d^2/\tilde I$; $d^2$ cancels, leaving $\tilde I p\delta^2 < 2\tilde I + p$. $\square$

Verified numerically at seven $u$; formula and numeric root agree to 6 decimals. Clean-room: relative error $0$ to $7.6\times10^{-11}$ over ten random cells, and the *independence* demonstrated directly — $d\in\{0.01,\,6.478,\,500\}\times V_\infty\in\{10^{-4},\,0.0141,\,100\}$ all give the identical root $0.01210018957$.

Two readings:

- **As a $z$-score.** The prior–data discrepancy has sd $\sqrt{\tau^2+1/\tilde I}$, so the criterion is $z^2 < (2\tau^2+\sigma_d^2)/(\tau^2+\sigma_d^2) \in (1,2]$: **$z<\sqrt2$ for a prior much looser than the data, $z<1$ for one much tighter.** Measured: $z^\*=1.382,\,1.330,\,1.225,\,1.118,\,1.045,\,1.005,\,1.000$ at $u=0.1\to10^4$.
- **As a calibration slack.** Writing $k=p\,\mathbb E[\delta^2]$ — how many times more mis-centred the prior is than it advertises — the criterion is $\boxed{k < 2+u}$. Exact to 10 decimals (clean-room, 8 cells). A prior may be **2–5× more wrong than it claims and still pay**; a correctly calibrated prior has $k=1<2+u$ always and therefore *can never hurt*. Only the *second moment* of $\delta$ enters, so this holds for any mis-centring distribution.

**On the MMM** (functional mROAS, nuisance $\alpha$, $\tilde I=247.9$, $\mathrm{sd}_{\rm data}(\alpha)=0.0635$): break-even $|\delta_\alpha^\*| = 0.291,\,0.176,\,0.110,\,0.082,\,0.070,\,0.064$ at $u=0.1,\,0.3,\,1,\,3,\,10,\,100$. In words: **an adstock prior centred more than $0.064$ away from the truth in $\alpha$ (about $0.44$ weeks of half-life at $\alpha=0.6$) is worse than no prior at all, no matter how tight** — and if it is loose, it may be off by up to $0.29$.

### 4.6 The direction law: which parameter deserves the prior?

Drop the free/nuisance split. For a rank-one prior of precision $p$ along any direction $n$:

$$\boxed{\;\frac{\operatorname{Var}_{\rm flat}-\operatorname{Var}(p)}{\operatorname{Var}_{\rm flat}} \;=\; \mathrm{corr}^2(\phi,n)\cdot\frac{u}{1+u},\qquad u = p\,(n'I^{-1}n)\;}$$

where $\mathrm{corr}$ is between $g'\hat\theta$ and $n'\hat\theta$ under the sampling distribution. Exact to $3.3\times10^{-16}$ (clean-room: $1.7\times10^{-11}$ over 400 random cells; the law is invariant to rescaling $n$, so the "unit direction" requirement is vacuous).

The consequence is sharp and practical. At matched relative strength $u$, **the value of a prior is proportional to $\mathrm{corr}^2$ between the decision functional and the direction the prior constrains.** On the MMM:

| prior placed on | $\mathrm{corr}(\mathrm{mROAS},\cdot)$ | $L^\*$ multiplier at $r=20\%$ | budget saved | value as % of 3-yr media spend |
|---|---|---|---|---|
| coefficient $\beta$ | $+0.309$ | $0.951$ | $4.9\%$ | $0.20\%$ |
| adstock $\alpha$ | $+0.756$ | $0.935$ | $6.5\%$ | $0.27\%$ |
| saturation $K$ | $-0.340$ | $0.941$ | $5.9\%$ | $0.24\%$ |
| **ROI / mROAS** | $+1.000$ | $\mathbf{0.439}$ | $\mathbf{56.1\%}$ | $\mathbf{2.29\%}$ |

Every number independently reproduced by the clean-room verifier to 4–5 significant figures.

**This is a derivation of Meridian's design choice, with a price on it.** An ROI prior at $20\%$ relative sd is worth $11.4\times$ a coefficient prior of the same relative tightness ($2.29\%$ vs $0.20\%$ of media spend), and the ratio of value is exactly $\mathrm{corr}^2 = 0.096$. It also says something Meridian does not: **the coefficient prior — the thing most MMM stacks actually regularise — is the single least valuable direction of the four**, worth less than the adstock prior that LightweightMMM sets for MCMC stability.

### 4.7 The half-value precision, and the dollar ladder

Since $L^\* = m\sigma\sqrt{\Psi T_r}$, the dollar value of a shape prior of strength $u$ is $V(u)=m\sigma\sqrt{T_r}\left[\sqrt{\Psi_0}-\sqrt{\Psi(u)}\right]$. Setting $V(u)=V(\infty)/2$ gives, in closed form,

$$u_{1/2} = \frac{\Psi_0-1}{r^2-1}-1,\quad r=\frac{\sqrt{\Psi_0}+1}{2}
\qquad\Longleftrightarrow\qquad
u_{1/2} = \frac{3s+1}{s+3},\;\; s=\sqrt{\Psi_0},$$

the second (prettier) form found independently by the clean-room verifier and agreeing with mine to $10^{-12}$. Ratio $V(u_{1/2})/V(\infty)=0.500000000000$ at six values of $\Psi_0$.

$$\begin{array}{lccccccc}
\Psi_0 & 1.5 & 2 & 5 & 10 & 20 & 100 & \to\infty\\
u_{1/2} & 1.106 & 1.188 & 1.472 & 1.702 & 1.929 & 2.385 & \mathbf{3}\\
u_{90\%}/u_{50\%} & 9.0 & 9.1 & 9.7 & 10.5 & 11.6 & 15.5 & \\
u_{99\%}/u_{50\%} & 73 & 101 & 109 & 120 & 139 & 220 &
\end{array}$$

**$u_{1/2}<3$ always** (the clean-room's closed form makes the bound obvious). So: *half the dollar value of any shape prior is bought by a prior no stronger than 3× the data's own information about the shape, and the last 1% costs 100–220× more precision than the first 50%.* Elicitation effort should be aimed at getting a shape prior *roughly* right, never at making it tight.

**The MMM dollar ladder** ($m=2$, $\sigma=0.05$, $T_r=156$ wk, 3-yr media spend $136.1$):

| shape knowledge | $\Psi$ | $L^\*$ | as % of 3-yr media spend |
|---|---|---|---|
| free | $19.80$ | $5.558$ | $4.09\%$ |
| prior at $u=u_{1/2}=1.93$ | $7.43$ | $3.404$ | $2.50\%$ |
| prior at $u=1$ | $10.40$ | $4.028$ | $2.96\%$ |
| known exactly | $1.00$ | $1.249$ | $0.92\%$ |

**A perfect shape prior is worth $3.17\%$ of three-year media spend**; a merely decent one ($u\approx2$) captures half of that. For comparison, dive 20's fully-calibrated single-channel figure was $0.36\%$ and its shape-free figure $1.12\%$ — this dive's comb design at $\sigma=0.05$ sits higher because it must identify four structural parameters from three tones.

---

## 5. Rounds 2–4 — attack and refine

### 5.1 Round 2: seven attacks

**A1 — The bias law is a linearisation.** Computing the *exact* pseudo-true bias (fitting the noiseless mean with $\alpha$ pinned at the prior centre and $\beta,K,S$ free) against the linear ceiling $-d'\delta$:

| $\delta_\alpha$ | 0.005 | 0.01 | 0.02 | 0.05 | 0.10 |
|---|---|---|---|---|---|
| exact/linear | 0.954 | 0.939 | 0.908 | 0.817 | **0.672** |
| pseudo-true $K$ | 1.467 | 1.432 | 1.355 | 1.010 | **0.200** |

So the linear law over-states the harm by 5–33%, i.e. **it errs conservative** (real break-even mis-centrings are *larger* than §4.5 says). The mechanism is substantively interesting: a wrong adstock prior is partly absorbed by the *saturation* parameters — at $\delta_\alpha=0.10$, $K$ collapses from $1.5$ to $0.20$. The damage to mROAS at the observed spend is limited precisely because the model distorts the response curve elsewhere. **The response curve away from the operating point takes the damage the decision functional avoids** — which is exactly where budget re-allocation reads it.

Independently, a Monte-Carlo penalised nonlinear fit ($n=200$, $\delta_\alpha=0.05$) gives bias $-0.150\pm0.020$ vs predicted $-0.157$ at $u=1$ ($z=+0.38$, agrees) and $-0.230\pm0.009$ vs $-0.285$ at $u=10$ ($z=+5.94$, disagrees) — the same nonlinear deflation, now with an error bar on it.

**A2 — Calibration failure.** If $\tau$ is stated honestly but $\mathbb E[\delta^2]$ is $k$ times larger, the break-even is $k^\*=2+u$ exactly: $2.3,\,3.0,\,5.0$ measured at $u=0.3,\,1,\,3$. Adopted into §4.5.

**A3 — Experiment-derived priors are not zero-mean.** Dive 02's transport problem means an experiment-derived prior carries a systematic offset $t$, not just spread. Pricing it (`e13`-A3): at $u=1$ a transport bias of $t=0.10$ in $\alpha$ still pays ($\mathbb E[\mathrm{MSE}]=0.252$ vs flat $0.279$); at $u=10$ the same $t$ **flips the sign** ($0.385$ vs $0.279$, HARMS). *A tight prior needs an unbiased experiment; a loose one does not.* This is the quantitative form of Meridian's own warning that translating an experiment's SE into a prior SD "is not a precise formula."

**A4 — Design envelope.** §4.7 holds the design fixed while the prior varies; but the optimal probe should change once you have a prior. Re-optimising the comb's amplitude split by Nelder–Mead at $u=0,1,10$ improves $\mathrm{sd}(\mathrm{mROAS})$ by only $2.4\%,\,2.5\%,\,2.9\%$. **The value of a prior is design-invariant to second order** — an envelope result that licenses the whole §4.7 accounting.

**A5 — Estimability (the clean-room's attack).** $\mathrm{cond}(I)=3.77\times10^5$, $s_{\min}=3.0\times10^{-3}$ for the 4-parameter FIM. §6.4 quantifies the damage; it is the dive's most serious limitation.

**A6 — Wrong variance in the admissibility rule.** Round 1 derived §4.5 from the *posterior* variance, giving $\delta^2<1/p+1/\tilde I$. The frequentist MSE requires the *sandwich* variance, which changes the constant to $2/p$: at $u=0.3$, $\delta^\*=0.132$ (wrong) vs $0.176$ (right); at $u=3$, $0.073$ vs $0.082$. The Bayesian's own variance accounting is 25–33% too pessimistic about what a prior may get away with. Corrected throughout.

**A7 — Gaussian/FIM approximation and non-Gaussian priors.** Against a fully nonlinear bounded-MLE Monte Carlo ($n=150$):

| | FIM | MC |
|---|---|---|
| $\mathrm{sd}(\mathrm{mROAS})$, shape known | $0.1187$ | $0.1236\pm0.0072$ |
| $\mathrm{sd}(\mathrm{mROAS})$, shape free | $0.5283$ | $0.9166\pm0.0531$ |
| implied $\Psi_0$ | $19.80$ | $\mathbf{55.0}$ |

The FIM is accurate (4%) when the shape is pinned and badly optimistic (74%) when it is free, so **the FIM ladder under-states $\Psi_0$ by $2.8\times$ — the ledger is conservative about the prior's value.** Reassuringly the *fraction* of the budget a perfect shape prior saves, $1-1/\sqrt{\Psi_0}$, moves only from $0.775$ to $0.865$; the headline is robust even though $\Psi_0$ is not. Separately, a variance-matched Gaussian is a good stand-in for a uniform prior only while the bound is tight: MC $0.363$ vs predicted $0.353$ at half-width $w=0.31\,\mathrm{sd}_{\rm free}$, but $0.557$ vs $0.383$ at $w=0.79\,\mathrm{sd}_{\rm free}$ and $0.726$ vs $0.438$ at $w=1.57$.

### 5.2 Round 3, first refinement: don't test the prior — widen it

§4.5 gives a criterion in the *unobservable* $\delta$. What is observable is the prior–data discrepancy $D=\hat s_{\rm flat}-s_0 = \delta + \text{noise}$, $\operatorname{Var}(\text{noise})=1/\tilde I$. Four rules:

- **FLAT** — ignore the prior.
- **FIXED** — use the stated $p=1/\tau^2$.
- **TEST** — use $p$ iff $D^2 < 2\tau^2+2/\tilde I$ (the plug-in version of §4.5, since $\mathbb E[D^2]=\delta^2+1/\tilde I$). This is the missing threshold for Meridian's prior–posterior shift check.
- **EB** — *widen* the prior to $\hat\tau^2 = \max(\tau^2,\;D^2-1/\tilde I)$ and use $\hat p = 1/\hat\tau^2$.

At $\tau=\mathrm{sd}_{\rm data}(\alpha)$ ($u=1$, $\delta^\*=0.110$):

| | worst-case risk / FLAT | gain captured when calibrated |
|---|---|---|
| FIXED | $25.4$ (unbounded, $\propto\delta^2$) | $100\%$ |
| TEST | $1.53$ (at $1.38\delta^\*$) | $42.7\%$ |
| **EB** | $\mathbf{1.32}$ (at $1.54\delta^\*$) | $\mathbf{66.7\%}$ |

**EB dominates TEST on both axes simultaneously.** The clean-room verifier reproduced this independently ($1.322$ vs $1.528$ worst case; $66.7\%$ vs $42.7\%$ gain; $\sim100\sigma$ separation at $N=8\times10^6$) and showed it is stable across $u\in[0.25,10]$ — EB worst case $1.185$–$1.432$, TEST $1.398$–$1.590$, gain captured $66.5$–$66.7\%$ vs $42.7$–$42.9\%$.

The operational recommendation is therefore *not* the natural one. Do not run a hypothesis test on your prior and discard it if it fails. **Widen it by the observed prior–data discrepancy**: $\hat\tau^2=\max(\tau^2, D^2-1/\tilde I)$. It is a two-line change to any MMM stack that reports both a prior and a shape estimate, it never costs more than $1.32\times$ the flat-prior risk at any mis-centring, and it keeps two thirds of the benefit when the prior is honest.

### 5.3 Round 3, second refinement: the decision layer

MSE is not the decision object. Under dive 08's deadband — act only when $|\hat z - x| > h^\*$, with $h^\*=(12Kq/\gamma)^{1/4}-0.5826\sqrt q$ — a prior contributes a *persistent* bias while excitation dollars buy away *transient* noise. At matched MSE, which costs more?

**An exact invariance.** A constant bias $b$ shifts $\hat z$ and (once the band is crossed) $x$ by the same amount, so the trigger statistic $|\hat z - x|$ is unchanged. **A biased estimate never causes churn.** Confirmed: excess move cost from bias $\le 10^{-4}$ at every $b$, while noise adds up to $0.204$ per period.

Define $\pi = $ (excess regret from a bias $b$) / (excess regret from iid noise of sd $b$). Measured at $q=0.01$, $K=0.5$, $\gamma=1$, $h^\*=0.437$, 24 seeds × 6000 periods:

| $b$ | 0.05 | 0.10 | 0.20 | 0.40 |
|---|---|---|---|---|
| $\pi$ (deadband) | $0.91\pm0.40$ | $0.583\pm0.069$ | $\mathbf{0.291\pm0.015}$ | $\mathbf{0.302\pm0.004}$ |
| $\pi$ ($h=0$, **power control**) | $0.997\pm0.003$ | $0.997\pm0.003$ | $0.997\pm0.003$ | $0.997\pm0.003$ |

The act-always control returns $\pi=1$ to three decimals at every $b$ — the instrument discriminates, and the deadband result is not an artefact. A curiosity in the decomposition: at $b=0.10$ the noise arm's *tracking* loss is slightly **negative** ($-0.0015$) — noise dithers the deadband into moving sooner — so the entire excess is churn.

**Consequence.** Under a deadband policy, bias is 2–3.4× cheaper than equal-MSE variance, so the decision-admissible mis-centring is $\delta^2 < (2\tau^2+\mathrm{sd}_{\rm data}^2)/\pi$, i.e. **$1.3$–$1.9\times$ wider in $\delta$ than the MSE rule of §4.5**. *Derived here, not verified end-to-end:* this step assumes regret is proportional to MSE with the measured proportionality $\pi$, which the deadband simulation supports for the reduced-form model but which has not been run through the nonlinear Hill loop.

### 5.4 Round 4: no material revision

Round 4 was the clean-room verification plus the estimability audit. It produced three caveats — the $\dim c = 1$ condition on the scalar $\rho^2$ form, the nonsingularity of $P$ in the bias ceiling, and the conditioning problem of §6.4 — and one transcription error entirely of my own making (see §6.1). **It revised none of the constructions.** With round 3 having added constructions without revising round 1's identities, and round 4 having added only caveats, the iteration terminates.

---

## 6. Verification

### 6.1 Independent clean-room verification

A verification subagent was given the ten claims as prose statements, with no access to the code, and re-implemented them from scratch.

- **First pass:** all ten confirmed on random matrices — C1 ($6.3\times10^{-15}$), C2 ($6.3\times10^{-14}$), C3 ($3.7\times10^{-15}$, MC $z=-1.02$), C4 ($6.7\times10^{-12}$, six MC reps $z\in[-2.11,+1.42]$), C5a ($9.8\times10^{-16}$), C6 (exact, plus a better closed form), C7 ($7.6\times10^{-11}$), C7b (10 decimals), C8 ($1.7\times10^{-11}$), C9 (exact), C10 (reproduced, $\sim100\sigma$).
- **It also caught a real error.** The MMM Fisher matrix I transcribed into the verification prompt was fabricated, not exported — it failed the internal consistency check $g'I^{-1}g = \Psi_0 V_\infty$ ($3.222$ vs $0.279$), and the verifier ruled out rounding, permutation, reparameterisation and scaling before declaring it wrong. On the *actual* exported matrix (second pass) every headline number reproduced: $\sqrt{V_\infty}=0.118708$, $d=(6.478210,-0.228905,0.113044)$, $\Psi_0=19.8044126$, $\tilde I$ eigenvalues, $\rho_k^2$, the pinning fractions ($44.5\%/7.7\%/4.4\%$) and all four direction-law multipliers — to $10^{-13}$ on the identities and to the stated precision on the reported figures.
- **Three caveats adopted:** $\dim c=1$ for the $\rho^2$ form (§4.4), $P$ nonsingular for the bias ceiling (§4.3), and the conditioning warning (§6.4).

### 6.2 Experiment inventory (16)

`e1` scalar shrinkage MC · `e2` the dollar plane and the cap · `e3` van Trees attainment (MC) · `e4` the ladder: scalar law, general identity, MMM FIM, spectral decomposition, isotropic law · `e5` half-value precision and diminishing returns · `e6` bias transfer + penalised nonlinear MC · `e7` admissibility break-even + the round-2 constant correction · `e8` negative and power controls (N1 $d=0$; N2 orthogonal direction; N3 $P=0$; N4 MC power near break-even) · `e9` deadband regret, first pass · `e10` **robustness sweep 1** (reparameterisation to half-life, uniform vs Gaussian prior, nonlinear MC ladder, which shape parameter to price) · `e11` **robustness sweep 2** ($\sigma$, $T$, probe period, amplitude) · `e12` the direction law and ROI vs coefficient priors · `e13` round-2 attacks A1–A4 · `e14` the EB/TEST/FIXED/FLAT risk comparison · `e15` decision layer with the act-always power control · `e16` **estimability audit** under FIM jitter.

### 6.3 Negative and power controls (four)

1. **$d=0$** (`e8`-N1): $\Psi=1.000000000000$ and bias $\le3\times10^{-14}$ at $P$ up to $10^9$ with $|\delta|\sim100$. A shape prior is worth exactly zero when the functional does not leak into the shape.
2. **Orthogonal direction** (`e8`-N2): $\Psi$ invariant to 12 decimals for $n'\tilde I^{-1}d=0$; the parallel direction drives $\Psi\to1$.
3. **MC power near break-even** (`e8`-N4): at $u=1$, $\delta^\*=1.938$, $4\times10^5$ draws — $z=-56.4,\,-16.2,\,-1.97,\,+18.4,\,+91.5$ at $0.6,0.9,1.0,1.1,1.5\times\delta^\*$. The instrument signs the value correctly at $\pm10\%$ of the break-even and is (correctly) indecisive at it.
4. **Act-always deadband** (`e15`): $\pi=0.997\pm0.003$ at every $b$, against $0.29$–$0.58$ with the band on.

### 6.4 The estimability audit (BL82, discharged for this dive)

Dive 21 established that FIM-based standard errors need an $s_{\min}/s_{\max}$ check. Perturbing every FIM entry by relative noise and re-computing $\Psi_0$ (2000 draws per cell):

| model | $\mathrm{cond}(I)$ | $\Psi_0$ | at jitter $10^{-6}$ | at $10^{-4}$ | at $10^{-3}$ |
|---|---|---|---|---|---|
| 2-param $(\beta,\alpha)$, $T{=}156$ | $40.6$ | $13.01$ | $[13.01,13.01]$ | $[13.00,13.03]$ | $[12.88,13.15]$ |
| 4-param, $T{=}156$ | $3.77\times10^5$ | $19.80$ | $[19.77,19.85]$ | $\mathbf{[12.95,26.82]}$, 2% fail | $[16.57,19.86]$ |
| 4-param, $T{=}312$ | $5.10\times10^5$ | $23.95$ | $[23.81,24.10]$ | $[9.50,37.22]$, 4% fail | $[16.67,22.12]$ |

**Verdict.** The 2-parameter ladder is a hard number ($\pm1\%$ even at $10^{-3}$ FIM error). The 4-parameter $\Psi_0=19.8$ is a $\pm35\%$ quantity: it requires the Fisher matrix to be known to about six significant figures, and a longer record does not fix it. The *direction law* — a ratio — is far more robust: the ROI-prior $L^\*$ multiplier stays at $0.439\,[0.398,0.512]$ under $10^{-4}$ jitter and $0.456\,[0.440,0.473]$ under $10^{-3}$, while the coefficient-prior multiplier degrades to $[0.844,1.160]$. **The qualitative conclusion — an ROI prior is worth an order of magnitude more than a coefficient prior — survives; the exact value of $\Psi_0$ does not.**

### 6.5 Robustness sweeps

**Design and noise** (`e11`): $\Psi_0$ is exactly $\sigma$-invariant (it is a design object; $28.13$ at $T{=}104$ for $\sigma=0.02,0.05,0.10$ alike) while $\mathrm{sd}(\mathrm{mROAS})$ scales linearly in $\sigma$ and $u_{1/2}$ moves only over $[1.93,2.04]$ across nine $(\sigma,T)$ cells. Single-tone designs are near-singular for a 4-parameter shape (cond $10^7$–$10^9$, $\Psi_0$ in the thousands) and their numbers should not be quoted — an instance of §6.4's rule.

**Reparameterisation and prior family** (`e10`): a prior expressed on the adstock *half-life* transfers by the delta method ($\partial\mathrm{hl}/\partial\alpha = 4.427$ at $\alpha=0.6$); a $0.25$-week prior on half-life gives $\Psi=13.48$ ($L^\*$ multiplier $0.825$), a $1$-week prior only $18.97$ ($0.979$). Pinning one shape parameter exactly captures $44.5\%$ (adstock $\alpha$), $7.7\%$ ($K$) and $4.4\%$ ($S$) of the total available value — **adstock is where the shape prior belongs, and the three single-pin fractions sum to $0.566<1$, so they are strongly subadditive.** Uniform-vs-Gaussian: see §5.1-A7.

---

## 7. Limitations and failure modes

1. **The 4-parameter ladder is numerically fragile** (§6.4). Any application of $\Psi(u)$ to a real fitted MMM must report $s_{\min}/s_{\max}$ and a jitter interval, or it is reporting noise.
2. **The FIM ladder under-states the prior's value by $2.8\times$** in $\Psi_0$ (§5.1-A7). The *fraction* of budget saved is robust ($0.78$ vs $0.87$); the absolute $\Psi_0$ is not. All headline dollar figures here use the conservative FIM value.
3. **The bias law is linear and over-states harm by 5–33%** over $\delta_\alpha\in[0.005,0.10]$ (§5.1-A1). Break-evens in §4.5 are therefore conservative. The nonlinear mechanism — saturation absorbing a wrong adstock — means the damage relocates to the response curve *away* from the operating point, which is invisible to the functional studied here and is exactly what budget re-allocation reads. **This dive does not price that.**
4. **$\delta$ is unobservable.** §4.5 is a criterion in a quantity nobody has. §5.2 is the operational answer, but EB still costs up to $1.32\times$ the flat risk in the worst case, and its $\hat\tau^2$ estimator is noisy at small $\tilde I$.
5. **Everything assumes the model class is right.** A prior that is perfectly centred on the truth *of a misspecified model* is not covered by any law here. Heusch (2026) shows an observational MMM reporting ROAS $10.6\times$ against truth $4.2\times$ even with oracle controls; no prior on $\alpha$ helps with that.
6. **The decision layer is a reduced form.** $\pi$ was measured in dive 08's linear-quadratic deadband, not in the nonlinear Hill loop; §5.3's widening factor is derived, not verified end-to-end.
7. **Single channel.** Dive 21's channel-count tax means the multichannel version has $n$ shape blocks and a prior on one channel's adstock may leak into another's mROAS. Untouched.

---

## 8. Devil's advocate

*Against "a calibrated prior is worth exactly $A p$ dollars."* This is Morita–Thall–Müller's ESS with a price tag. If someone objects that the dive's headline is a repackaging, they are right about that row of the table, and §2 says so. The defence is that the surrounding structure — which *direction* the prior goes on, how the value saturates, when it turns negative, and what to do when you cannot check — is where the content is.

*Against "the ROI prior is worth $11.4\times$ the coefficient prior."* The comparison equalises *relative sd*, and relative sd is not a neutral yardstick across parameters with different natural scales; a $20\%$ prior on $\beta$ may be much harder or much easier to elicit than a $20\%$ prior on ROI. The scale-free statement is the direction law itself (value $\propto \mathrm{corr}^2\cdot u/(1+u)$, exact to $3\times10^{-16}$), and $\mathrm{corr}(\mathrm{mROAS},\beta)=0.309$ is a property of *this* design at *this* operating point. A design that identified $\beta$ orthogonally to the shape would raise it. What does not move is the ordering: the ROI direction has $\mathrm{corr}=1$ by construction, so no other direction can beat it at matched $u$.

*Against "$u_{1/2}<3$, so don't bother tightening a shape prior."* $u$ is measured against the *data's own* profile information $\tilde I$, which for a badly-designed spend path is nearly zero — and then $u=3$ can still mean an absurdly precise prior in natural units. The statement is scale-free but not effort-free.

*Against "$\delta^2<2\tau^2+\mathrm{sd}_{\rm data}^2$."* It is a criterion in expectation over the data, applied by a practitioner who sees one dataset; a prior that is admissible in expectation can still be the thing that ruined this particular quarter's allocation. And §5.1-A6 showed that the natural Bayesian derivation of it gives the wrong constant, so the rule is easy to get wrong by 30% in the direction of excess caution.

*Against "$\pi\approx0.3$, so bias is cheap."* Only under a deadband, only for a bias small relative to the band, and only when the loss really is quadratic around the operating point. A bias large enough to move the operating point across the Hill knee changes the local curvature, and then the reduced form does not apply. The power control shows the effect is real, not that it generalises.

*Against "EB dominates TEST."* EB was compared under a Gaussian noise model with the *correct* $\tilde I$. If $\tilde I$ is itself mis-estimated — and §6.4 says it can be, badly — $\hat\tau^2=\max(\tau^2,D^2-1/\tilde I)$ inherits that error, and EB's safety margin is not guaranteed. TEST has the same exposure but a coarser dependence.

*Against the whole dive.* Every number here comes from one synthetic structural model with one design. The clean-room verified the *mathematics*, not the *model*. The single most valuable next step is not more theory (§9.1).

---

## 9. Next steps for a future session

1. **Estimate $\tilde I$, $d$ and $\Psi_0$ on a real fitted MMM.** Every quantity in the ledger is computable from a fitted model's Hessian, and §6.4 gives the audit that must accompany it. This merges with BL74 and with dive 21's next-step 4 (BL81, auditing a real flighting calendar) — the same Hessian answers both, and it is the program's most checkable outstanding prediction. Prediction: real advertisers' $d$ is dominated by the adstock component, and the coefficient priors their stacks actually use are worth under $10\%$ of an ROI prior.
2. **Ship the EB widening rule as a diagnostic.** $\hat\tau^2=\max(\tau^2, D^2-1/\tilde I)$ is a two-line addition to Meridian's prior–posterior shift check, which currently reports a shift with no threshold. Write the threshold, the operating characteristic, and the multi-channel version.
3. **Price the *response-curve* damage, not just the functional damage.** §5.1-A1 showed a wrong adstock prior is absorbed by saturation, relocating the error to spend levels away from the operating point. Since budget re-allocation reads exactly that region, the decision-relevant functional is $x^\*(\theta)$ or the curve over a re-allocation range, not mROAS at the current point. Redo §4.6's table for $\phi=x^\*$; the prediction is that the ROI prior's advantage *shrinks*, because ROI at the current point does not pin the curvature that sets the optimum.
4. **The multichannel ledger.** With $n$ channels, $d$ becomes a matrix and cross-channel shape leakage (dive 11) enters. Does the direction law survive as a trace criterion? Does dive 21's exact comb decomposition make the ladders separate per channel?
5. **Join to dive 03/04's EVSI.** §4.7 says a perfect shape prior is worth $3.17\%$ of media spend and dive 20 says a shape-fixing experiment is worth $\sqrt{10}$ on the perpetual budget. Those are the same quantity computed two ways; reconcile them and compare against dive 04's dual-experiment margin (+47%) and dive 21's phase-management value ($1.09\times$ vs $15.5\times$). This finally closes BL7's cost side.
6. **A prior on the *design* rather than the parameters.** §5.1-A4 found the design envelope is second-order for a *fixed* comb, but the choice of which channel gets which bin (BL80) interacts with per-channel $\Psi_j$, and $\Psi_j$ now depends on each channel's prior. The assignment problem should be solved with the ladder inside it.
7. **Verify $\pi$ in the nonlinear loop** and re-derive §5.3's widening factor end-to-end, including the case where the bias moves the operating point across the Hill knee.

---

## 10. Sources

- **Internal:** dive 20 (the frontier, $L^\*$, the $\Psi$ ladder, and the "inside the CRLB" puzzle this dive resolves), dive 21 (the comb design; BL82's estimability rule, discharged in §6.4), dives 08/09 (the deadband and $h^\*$, used in §5.3), dives 16/19 (frozen learning; the certainty-equivalent stall), dive 02 (transport, the source of §5.1-A3's offset), dive 11 (cross-channel leakage), dives 03/04 (EVSI, the missing join). Catalogues `../02_open_questions.md` §M1, §M2, §M7; `../03_mmm_adoption_barriers.md` §O7, §B1.
- **Prior effective sample size:** [Morita, Thall & Müller 2008, *Biometrics* 64:595–602](https://web.ma.utexas.edu/users/pmueller/pap/MTM08.pdf) (ESS $=\sigma^2/\tilde\sigma^2$ in the normal case; Beta $\alpha+\beta$; power priors) · [Neuenschwander, Weber, Schmidli & O'Hagan 2020, *Biometrics*](https://onlinelibrary.wiley.com/doi/abs/10.1111/biom.13252) ([preprint](https://arxiv.org/abs/1907.04185)) — predictive consistency, the correctness axiom for any additive ESS claim.
- **Bayesian bounds and shrinkage risk:** [Gill & Levit 1995, *Bernoulli* 1(1/2):59–79](https://projecteuclid.org/journals/bernoulli/volume-1/issue-1-2/Applications-of-the-van-Trees-inequality/bj/1186078362.full) (van Trees holds for biased estimators — the frontier §4.2 uses) · [Rioul et al., arXiv:1902.08582](https://arxiv.org/abs/1902.08582) (a family of Bayesian CRBs; for log-concave priors a bound exists that does not depend on the prior's Fisher information) · Casella 1985, *Amer. Statistician* (the $(\theta-\mu_0)^2<\sigma^2(1+w)/(1-w)$ crossover) · Stein 1956; Efron & Morris.
- **MMM priors in practice:** [Meridian — ROI/mROI parameterisations](https://developers.google.com/meridian/docs/advanced-modeling/roi-mroi-contribution-parameterizations) · [Meridian — ROI priors and calibration](https://developers.google.com/meridian/docs/advanced-modeling/roi-priors-and-calibration) · [Meridian — setting custom priors from past experiments](https://developers.google.com/meridian/docs/advanced-modeling/set-custom-priors-past-experiments) (the "not a precise formula" warning priced in §5.1-A3) · [Meridian — prior/posterior shift check](https://developers.google.com/meridian/docs/post-modeling/quality-checks#prior-posterior-shift) (the diagnostic §5.2 supplies a threshold for) · [Zhang et al., *Media Mix Model Calibration with Bayesian Priors*, Google](https://research.google/pubs/media-mix-model-calibration-with-bayesian-priors/) · Robyn's tri-objective (NRMSE, DECOMP.RSSD, MAPE) with self-declared "subjective" weights · LightweightMMM `lag_weight ~ Beta(2,1)`.
- **Identification and criticism:** [Heusch 2026, arXiv:2608.21128](https://arxiv.org/abs/2608.21128) (ROAS $10.61\times$ vs true $4.20\times$; $8.41\times$ even with oracle controls; the $\lambda$–$\beta$ ridge at $\mathrm{corr}=-0.68$ with a tight response curve over the probed range — the empirical face of §4.3's $d$) · [Dew, Padilla & Shchetkina 2024, arXiv:2408.07678](https://arxiv.org/abs/2408.07678) (saturation vs time-varying effectiveness not separately identified) · [Chan & Perry 2017, Google](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/2d0395bc7d4d13ddedef54d744ba7748e8ba8dd1.pdf) · [Mutinex on opaque priors](https://mutinex.co/insights/bayesian-priors/) · a widely-cited 327-dataset / 1,241-model calibration study reporting that good experiments improve accuracy 6–16% while bad ones degrade it 10–40% — **primary source unverified, cited here only as directionally consistent with §5.1-A3, not as evidence.**

*Code:* `code/22_price_of_the_prior.py` (run `python 22_price_of_the_prior.py` for all sections, or name them, e.g. `e4 e12 e16`); merged results `code/22_results.json`; intermediate per-section JSON in `code/_scratch22/`. Sections `e6`, `e10` and `e13` involve nonlinear Monte-Carlo fits and take several minutes each.
