# Deep dive 21 — The $n$-channel spectral comb: harmonic collisions, phase as a design variable, and the price of budget neutrality

**Queue item:** backlog **BL73** (the $n$-channel spectral comb), absorbing **BL70** (the budget-direction escape, from dive 19) · **Date:** 2026-09-06 · **Code:** `code/21_spectral_comb.py`

---

## 1. Problem statement

Dive 20 solved the single-channel design problem in the frequency domain: at matched profit cost the achievable $\mathrm{se}(\mathrm{mROAS})$ is frequency-invariant (Theorem 1, $C\cdot\mathrm{Var}(\hat c_1)=\tfrac12\kappa S_\varepsilon(\omega)$), the total price of identifying a channel is $L^*=m\sigma\sqrt{\Psi T_r}$ (Theorem 2), and probe energy must avoid the seasonal nuisance harmonics (Rule L). It closed by admitting the obvious gap:

> "**Single channel.** … The $n$-channel spectral design — how many channels a 52-week harmonic comb can accommodate with mutually orthogonal probes — is not done."

This dive does it. Formally, let weekly sales be

$$y_t \;=\; z_t'\gamma \;+\; \sum_{j=1}^{n}\beta_j\,h_j\!\big(a_{jt};K_j,S_j\big)\;+\;\varepsilon_t,\qquad a_{jt}=x_{jt}+\alpha_j a_{j,t-1},\qquad h(a)=\frac{a^S}{a^S+K^S},$$

with $z_t$ = intercept + centred linear trend + Fourier harmonics $k\le3$ at period 52, and $\varepsilon\sim N(0,\sigma^2)$. A **comb design** gives channel $j$ its own probe $x_{jt}=\bar x_j^*\big(1+A_j\cos(\omega_j t+\phi_j)\big)$ with $\omega_j = 2\pi m_j/T$ on the DFT grid of the analysis window.

Four questions:

1. **Orthogonality.** Do channel-specific tones stay orthogonal once each passes through its *own* adstock filter, and once the response is *saturated*?
2. **Packing.** How many channels fit, given the seasonal harmonics, the memory constraint, and whatever the nonlinearity does?
3. **The Gram penalty.** What does a collision cost, and what — if anything — defeats it?
4. **Budget neutrality.** If the plan holds total spend fixed, what is identified and what is not, and what does the missing direction cost to buy? (dive 19's BL70.)

Working point: six channels $(K,S,\alpha,H^*)$ = A (1.5, 2.0, 0.60, 0.68), B (2.0, 1.6, 0.30, 0.55), C (1.0, 2.4, 0.75, 0.75), D (3.0, 1.3, 0.45, 0.60), E (0.8, 3.0, 0.20, 0.70), F (2.5, 2.0, 0.55, 0.50), each with margin $m=2$ and $\beta$ solved from the myopic first-order condition, so **true $\mathrm{mROAS}_j = 1$ for every channel** and errors are directly comparable. $T = 364$ weeks, $\sigma=0.05$.

---

## 2. Prior state

**Established elsewhere — must not be claimed here** (literature scan; the aircraft/process-control line is the load-bearing prior art and it is *not* cited anywhere in the marketing literature):

- **Assigning each input its own disjoint set of interleaved DFT-grid harmonics is a 30-year-old technique.** Morelli's NASA/AIAA orthogonal optimised multisine designs ([NTRS 20210018115, 2021](https://ntrs.nasa.gov/api/citations/20210018115/downloads/Morelli_Practical_FTI_Paper_v1%200706.pdf)) assign disjoint index sets $K_j$ over a shared base period, "each frequency index … assigned to only one of the inputs, to preserve mutual orthogonality," and explicitly discuss the trade between record length, input count and per-channel richness. **The spectral comb as a concept is a transplant, not an invention.** What their setting lacks is a seasonal nuisance comb to avoid and a saturating response that generates harmonics.
- **Simultaneous multi-input excitation dominates one-at-a-time**, with a known variance structure: Gevers, Miskovic, Bonvin & Karimi, *Automatica* 42(4) 2006 — for structures sharing input and noise parameters, adding an excited input reduces every parameter's covariance.
- **Convex/LMI design of MIMO input spectra**, and the fact that robust-optimal spectra collapse to finite sinusoid sums: Rojas, Welsh, Goodwin & Feuer, *Automatica* 43 (2007); Bombois et al. (2006) least-costly identification.
- **Amplitude- and crest-factor-constrained ("plant-friendly") multisines, including the MIMO extension** (Rivera et al.; Lee et al.) — the loss from non-negativity is a known, quantified phenomenon.
- **The MMM collinearity problem and the prescription "vary channels independently"** are established: Chan & Perry (2017) diagnose it precisely ("the data contains little information … when one of the ad channels moves independently"); Dew, Padilla & Shchetkina (2024) propose spend manipulations "that pin down model form"; Meridian's docs and Recast's practitioner note ("spending higher on linear TV one week and then pulsing up CTV the following") state the informal, single-pair, ad-hoc version.
- Internally: dive 01 (identified set, block design), 02 (transport operator), 03/04 (the decision plateau, LQC), 09 (multichannel ellipsoid deadband; presumes $1'g$ known), 11 (nuisance leakage between channels), 16/19 (the certainty-equivalent stall; BL70), and above all **dive 20**, whose Theorems 1 and 2 this dive extends to $n$ channels.

**The gap.** No source combines the seasonal-harmonic constraint with comb packing; no source treats the budget-simplex constraint as a rank condition on a frequency-domain design; and — the thing that turns out to matter most — no source asks what a *saturating* response does to comb orthogonality, i.e. where the nonlinear distortion products of channel $j$ land relative to channel $k$'s fundamental. The verification subagent confirmed it could find nothing on adstock-filtered comb orthogonality either.

---

## 3. Round 1 — the constructions

### 3.1 Heterogeneous adstock is free *(derived here; corollary-grade, but unclaimed)*

Geometric adstock is LTI, hence diagonal in the Fourier basis: the steady-state response to a tone at $\omega_j$ is a tone at $\omega_j$ with gain $|H_j(\omega_j)|$ and lag $\arg H_j(\omega_j)$, $H_j(\omega)=(1-\alpha_je^{-i\omega})^{-1}$. Distinct on-grid tones over a full window are exactly orthogonal at any phases. Therefore:

> **Proposition 1.** In steady state, on-grid comb tones at distinct bins produce exactly orthogonal adstock deviations **regardless of how different the channels' memories are**. Channel-specific decay rates cost nothing.

Measured (S1): with six adstock rates spanning $\alpha\in[0.20,0.75]$ at bins (19, 23, 25, 29, 31, 37), max $|$off-diagonal correlation$|$ = $2.5\times10^{-15}$ (clean-room: $7.5\times10^{-16}$).

Two riders, both exact:

- **The trend is the only linear leak.** Adding a centred linear trend to $z_t$ raises the coupling to $6.1\times10^{-4}$ (VIF $=1.000001$). For pure on-grid cosines the induced pairwise correlation is a *constant, bin-independent* number — because $\sum_{t<T} t\cos(2\pi m t/T)=-T/2$ for every integer $m\ne0$. The clean-room verifier sharpened my $-6/T^2$ to the exact **$-6/(T^2-7)$** (12 digits at $T = 52,104,208,364,520$; $-4.528678\times10^{-5}$ at $T=364$). Seasonal harmonics, being on-grid themselves, leak exactly zero.
- **The probe switch-on transient is benign.** If the *baseline* spend is already in steady state and only the probe switches on at $t=0$ (the realistic case), max coupling is $7.5\times10^{-3}$, VIF $1.0002$. My first version of this attack — starting the whole channel from $a=0$ — gave a correlation of 0.39 and was simply the wrong model of a running advertiser.

### 3.2 Comb optimality: the $n$-channel problem decomposes exactly *(derived here)*

Let $v_j = M_Z(H_j\!\star\!u_j)$ be channel $j$'s nuisance-projected adstock deviation. Then $\mathrm{Var}(\hat c_{1j})=\sigma^2/\|P_{-j}^{\perp}v_j\|^2 \ge \sigma^2/\|v_j\|^2$ with equality iff $v_j\perp\mathrm{span}\{v_k\}_{k\ne j}$; and the second-order profit cost is $C_j=\tfrac12\kappa_j\|v_j\|^2$ (separable across channels, since the response is additive). Hence:

> **Theorem 1 ($n$-channel frontier).** For **any** probing design, $C_j\cdot\mathrm{Var}(\hat c_{1j})\ \ge\ \tfrac12\kappa_j\,S_\varepsilon(\omega_j)$ for every $j$, with equality **iff** channel $j$'s excitation is orthogonal (after nuisance projection) to every other channel's. An orthogonal comb therefore attains dive 20's single-channel frontier for **all $n$ channels simultaneously**, and every collinearity is a strict Pareto loss.

Verified (S10): the six-channel comb gives $C_j\mathrm{Var}(\hat c_{1j})/(\tfrac12\kappa_j\sigma^2) \in [1.00010, 1.00071]$ — max deviation 0.071%, and **exactly 1.000000 to $10^{-12}$ when the trend column is dropped** (clean-room). The decision-functional version is equally sharp: each channel's $\mathrm{se}(\mathrm{mROAS})$ in the six-channel comb equals its value when probed *alone* to $1.1\times10^{-5}$ (S10, "additivity").

The economic corollary is immediate and is the dive's most transferable number:

> **Corollary (the channel-count tax).** Because dive 20's $L^*=m\sigma\sqrt{\Psi T_r}$ contains no channel-specific quantity — it is margin × *aggregate sales noise* × $\sqrt{\text{horizon}\times\Psi}$, and I re-verify here that $C\cdot\mathrm{Var}$ is invariant to scaling a channel by any factor $s$ (S16: ratio 1.000527 for $s=0.25\ldots4$) — the price of identification is the **same absolute number for a large channel and a small one**. Splitting a fixed media budget into more channels therefore multiplies the identification bill without changing the denominator.

At the working point ($\sigma/\text{weekly sales}$ as calibrated, $T_r=260$):

| channels | price at $\Psi=1$ (shape calibrated) | price at $\Psi=9.87$ (full Hill free) |
|---|---|---|
| 1 | 0.09% of media spend | 0.28% |
| 6 | 0.53% | 1.65% |
| 12 | 1.05% | 3.30% |
| 20 | **1.75%** | **5.50%** |

Fragmentation, not scale, is what makes measurement expensive. (Uncertainty: these inherit dive 20's $L^*\propto m\sigma$ exactly, so they scale linearly in the advertiser's residual sales noise and are only as good as $\Psi$; see §7.)

### 3.3 The harmonic-collision phase law *(derived here; the dive's main theorem)*

Proposition 1 is a statement about the *linear* design matrix. The response is not linear. Expanding $h_j(\bar a_j+\tilde a_j)$ with $\tilde a_j=B_j\cos(\omega_jt+\psi_j)$, $\psi_j=\phi_j+\arg H_j(\omega_j)$:

$$\tfrac12 c_{2j}\tilde a_j^2=\frac{c_{2j}B_j^2}{4}\big[1+\cos(2\omega_jt+2\psi_j)\big],\qquad
\tfrac16 c_{3j}\tilde a_j^3=\frac{c_{3j}B_j^3}{24}\Big[3\cos(\omega_jt+\psi_j)/2\cdot\ldots+\cos(3\omega_jt+3\psi_j)\Big].$$

So channel $j$ occupies not one bin but the arithmetic ladder $m_j, 2m_j, 3m_j,\dots$, with the $p$-th line carrying phase $p\psi_j$. If channel $k$'s fundamental sits on channel $j$'s $p$-th line, the two are two cosines at the *same* frequency, and their correlation is $\cos(p\psi_j-\psi_k)$ — a pure phase quantity. Hence:

> **Theorem 2 (harmonic-collision phase law).** In the local Volterra model, if $\omega_k=p\,\omega_j$ then
> $$\boxed{\ \mathrm{VIF}(\hat c_{1k})\;=\;\mathrm{VIF}(\hat c_{pj})\;=\;\frac{1}{\sin^{2}\!\big(p\,\psi_j-\psi_k\big)}\ }$$
> The design is **non-identifying** iff $\psi_k\equiv p\psi_j \pmod \pi$ and **exactly orthogonal** iff $\psi_k=p\psi_j\pm\pi/2$ (quadrature). Only the phase difference matters — not the amplitudes, not the adstock rates, not the curvature.

Verified (S3) to max **relative** error $3.75\times10^{-4}$ ($p=2$) and $1.98\times10^{-4}$ ($p=3$) over $\phi_k\in[0,\pi]$, with values sweeping 1.007 → 148 → 172; **exact to $10^{-10}$ with the trend column dropped** (clean-room). The verifier's methodological note is adopted: the deviation is relative, not absolute — near the pole the absolute error reaches 47 — and the sweep should be published in degrees because $d\mathrm{VIF}/d\phi\approx2500$ there.

The design consequence is the surprising part. Naively, a comb must avoid octaves; the theorem says **it need not**, because phase is a free design variable that nulls the collision. Capacity is a phase-feasibility question, not a frequency-exclusion one. Collision constraints $\psi_k-p\psi_j\equiv\pi/2\ (\mathrm{mod}\ \pi)$ form a linear system over $\mathbb{R}/\pi\mathbb{Z}$; if the collision graph is a forest the system is always solvable, and the compatibility of chained constraints is automatic mod $\pi$ (e.g. $m_l=2m_k=4m_j$ generates $4\psi_j+3\pi/2$ and $4\psi_j+\pi/2$, congruent mod $\pi$).

### 3.4 Packing capacity and a record-length number-theory rule *(derived here)*

Admissibility for bin $m$: (i) $\ge4$ cycles in the window, $m\ge4$; (ii) period $\ge4\tau_j$, i.e. $m\le T/(4\tau_j)$ (dive 20's carryover-identification floor); (iii) $r\,m$ is not a seasonal bin $pT/52$, for $r=1,\dots,q$ — protecting the fundamental *and its distortion products* up to order $q$, which §3.3 shows are where the shape information lives.

Constraint (iii) has a number-theoretic character nobody has noticed because no other field has a seasonal comb:

> **Rule R (record length).** Write $k=T/52$. Since the seasonal bins are the multiples of $k$, $rm\equiv0 \pmod k$ reduces to $m\equiv 0\pmod k$ whenever $\gcd(r,k)=1$. **If $k$ is coprime to $6$ — a 5-, 7-, 11- or 13-year window — protecting the 2nd and 3rd harmonics costs nothing at all.** If $k$ is even, second-harmonic protection additionally excises $m=k/2,3k/2,\ldots$; if $3\mid k$, third-harmonic protection excises $m=k/3,2k/3,\ldots$.

Enumeration (S5, $\tau=2.5$, $P=3$): $|\mathcal A|$ for $q=1,2,3$ is $(20,20,20)$ at $T=260$, $(30,30,30)$ at $T=364$, $(51,51,51)$ at $T=572$ — free — against $(14,13,13)$ at $T=208$, $(25,24,23)$ at $T=312$ and $(56,54,52)$ at $T=624$. The verifier confirmed the theorem and correctly flagged that **the converse is false**: $k=2,3,9,15$ also cost nothing at these settings, because the extra excluded bins fall below the $m\ge4$ floor. State it as sufficient, not iff.

Capacity itself (S5, $T=364$, $P=3$, $q=2$): **30 admissible bins**, of which the largest subset containing no $(m,2m)$ or $(m,3m)$ pair has **20** (exact ILP; clean-room reproduced by branch-and-bound *and* an independent MILP). The harmonic-free fraction is a stable $\approx2/3$ of $|\mathcal A|$ across $\tau$. Scaling: 14 / 20 / 26 / 33 / 40 / 46 phase-free channels at 5 / 7 / 9 / 11 / 13 / 15 years.

> **Finding (a negative one, and the most useful).** Comb capacity is **not** the binding constraint for any real advertiser. Twenty channels fit a 7-year record with no phase management at all, and thirty with it. The binding constraints are the *cost* (§3.2's channel-count tax) and the *phase* (§3.3) — not the number of available frequencies.

### 3.5 Budget neutrality: an exact rank theorem and an exact escape law *(derived here; the literature scan found this white space)*

Media plans are usually written under a fixed total. Let $\Phi:c\mapsto\sum_j c_j\,(H_j\!\star\!u_j)$ be the map from slopes to fitted mean, and impose $\sum_j u_{jt}=0$ for all $t$.

> **Theorem 3 (budget-neutral rank).** If all channels share one adstock rate, $\Phi(\mathbf 1)=H\star\sum_j u_j=0$, so the all-ones direction lies in $\ker\Phi$: **the total marginal value $\sum_j c_{1j}$ is unidentified under budget-neutral variation, no matter how many frequencies or contrast patterns are used.** With heterogeneous adstock, $\Phi(\mathbf 1)=\sum_j (H_j-\bar H)\star u_j$, generically nonzero.

Verified (S6): with $\alpha$ equal across four channels, adding 1, 2, 4 budget-neutral tones with orthogonal contrast patterns gives rank 1, 2, 3 out of 4 and $\|P_{\text{null}}\mathbf 1\|/\|\mathbf 1\| = 1.000$ throughout — the rank saturates at $n-1$ and the sum direction never leaves the null space. With distinct $\alpha$, rank 4 is reached; the verifier's correction is adopted, that **two** tones suffice (each contributes cos and sin), not four as I first wrote.

The quantitative version is exact. For $n=2$ with $u_2=-u_1=-A\cos\omega t$, writing $z_1=AH_1$, $z_2=-AH_2$, the Gram is $G_{ij}=\tfrac T2\mathrm{Re}(\bar z_iz_j)$, $\det G=(\tfrac T2)^2\mathrm{Im}(\bar z_1z_2)^2$, and $\mathbf 1'G^{-1}\mathbf 1=(2/T)|z_1-z_2|^2/\mathrm{Im}(\bar z_1z_2)^2$. Hence:

> **Theorem 4 (budget-neutral escape law).**
> $$\boxed{\ \mathrm{se}\big(\hat c_{11}+\hat c_{12}\big)\;=\;\sigma\sqrt{\tfrac2T}\;\frac{\big|H_1(\omega)+H_2(\omega)\big|}{A\,\big|\mathrm{Im}\big(\overline{H_1(\omega)}H_2(\omega)\big)\big|},\qquad
> \mathrm{Im}(\bar H_1H_2)=\frac{(\alpha_1-\alpha_2)\sin\omega}{\big|1+\alpha_1\alpha_2-\alpha_1e^{i\omega}-\alpha_2e^{-i\omega}\big|^2}. }$$

Verified (S7) to $\le3.6\times10^{-3}$ relative across eight $(\alpha_1,\alpha_2,\omega)$ cells (clean-room: $2.1$–$6.9\times10^{-4}$, and **exact to machine precision without the trend column**).

The identity has a reading that no amount of intuition produced first:

> **The escape from budget neutrality is bought with adstock *phase lag*, not adstock *gain*.** $\mathrm{Im}(\bar H_1H_2)$ is the area of the parallelogram spanned by the two transfer functions in the complex plane. At $\omega=0$ every $H_j=1/(1-\alpha_j)$ is **real**, so the area vanishes: **the total marginal value is unidentified at DC even when the memories are wildly different.** It vanishes again at Nyquist. There is an interior optimum.

Measured $\mathrm{Im}(\bar H_1H_2)$ for $(\alpha_1,\alpha_2)=(0.6,0.3)$: $6.6\times10^{-5}$ at a 364,000-week period, 0.0660 at 364 weeks, 0.827 at 19 weeks, 0.0987 at 3 weeks, $1.2\times10^{-4}$ at 2 weeks. At matched cost the optimal budget-neutral probe period is 8.6–20.5 weeks over the $\alpha$-pairs tested — and it frequently sits *inside* the $4\tau_{\max}$ carryover floor, so the constrained optimum is the floor (see §7).

**Pricing it (BL70's actual ask).** In dive-20 frontier units, define $\Lambda \equiv C\cdot\mathrm{Var}/(\tfrac12\kappa\sigma^2)$. Then $\Lambda_{\text{neutral}}=|H_1+H_2|^2(|H_1|^2+|H_2|^2)/\mathrm{Im}(\bar H_1H_2)^2$ and the total-budget analogue replaces $|H_1+H_2|$ by $|H_1-H_2|$. A single channel's own $c_1$ sits at $\Lambda=|H|^2\approx3.4$.

| $(\alpha_1,\alpha_2)$ | $\Lambda_{\text{neutral}}$ (13 wk) | $\Lambda_{\text{total}}$ (13 wk) | ratio |
|---|---|---|---|
| (0.60, 0.30) | 71.5 | 3.96 | **18×** |
| (0.80, 0.20) | 17.1 | 4.21 | 4× |
| (0.70, 0.50) | 84.7 | 2.85 | 30× |
| (0.90, 0.60) | 24.8 | 2.38 | 10× |
| (0.65, 0.60) | 1 196 | 2.63 | **454×** |
| (0.50, 0.45) | 2 193 | 3.57 | **615×** |

> A **total-budget** probe learns the aggregate marginal value at essentially the same price as a single channel's own coefficient ($\Lambda\approx3$–4 against a reference 3.4). A **budget-neutral** probe learns it for 4–615× more, the penalty exploding as the channels' memories converge. Dive 19's BL70 asked whether a non-neutral budget pulse is worth pricing against never learning $1'g$: it is, and the exchange rate is $\Lambda_{\text{neutral}}/\Lambda_{\text{total}}$, computable from a fitted MMM's adstock rates alone.

### 3.6 Second construction: the harmonic readout *(derived here)*

Theorem 2 says a comb puts channel $j$'s curvature on a specific, predictable line. That line can simply be **read off**, with no curve fitting at all. With $B_j$ the adstock-deviation amplitude and $Y(\cdot)$ the DFT of the nuisance-projected response,

$$\hat c_{2j}=\frac{4}{B_j^2}\,\mathrm{Re}\!\big[Y(2\omega_j)e^{-2i\psi_j}\big],\qquad
\hat c_{3j}=\frac{24}{B_j^3}\,\mathrm{Re}\!\big[Y(3\omega_j)e^{-3i\psi_j}\big],\qquad
\mathrm{se}(\hat c_{2j})=\frac{4\sigma}{B_j^2}\sqrt{\tfrac2T}.$$

Verified (S12–S13): noiseless recovery of $c_2$ to 0.1–2% and $c_3$ to 0.5–12% across three channels and two amplitudes; the se formula matched by a **20,000-rep** Monte Carlo to 0.05% at every amplitude (0.43380 ± 0.00217 vs 0.43402; 0.02711 ± 0.00014 vs 0.02713) — the clean-room's 40,000-rep run agrees and its warning is adopted, that my original 400-rep check (0.453) could not adjudicate a claim at this level.

Two riders:

- **The ratio $R=|Y(3\omega)|/|Y(2\omega)| = (|c_3/c_2|)B/6$ is a saturation *shape* test** requiring neither $\beta$ nor a fitted curve. Recovered $|c_3/c_2|$: 0.777 vs 0.781 (A), 1.399 vs 1.403 (C), 1.360 vs 1.397 (E). Channel F sits exactly at the Hill inflection where $c_3=0$: the test returns $R=0.0002$, implied ratio 0.0058 — **the shape test carries its own null control and passes it.**
- **Amplitude is everything, and the modulus estimator is a trap.** The $2\omega$ line's SNR is $|c_2|B^2/(4\sigma\sqrt{2/T})$: **0.36 at a 10% probe** — below the noise floor. A DFT-modulus estimator is then catastrophically biased (MC mean $-0.563$ against a truth of $-0.157$); only the phase-known linear projection is unbiased ($-0.1586$ at every amplitude). Reaching $z=2$ on $c_2$ needs $B^*=\sqrt{8\sigma\sqrt{2/T}/|c_2|}$ = **23.5% of mean spend**. This is dive 20's "concentrate for curvature" with a closed form and a number attached, and it is the cheapest route up dive 20's $\Psi$ ladder — the shape information comes free from the *same* probe, provided it is deep enough.

---

## 4. Round 2 — the attack

### 4.1 The attack that landed: the closed-form quadrature angle is the wrong angle

Theorem 2 lives in the local Volterra model with one curvature parameter per channel. In the **full structural Hill model** every channel-$A$ parameter contributes to the $2\omega_A$ line — and $\partial\mu/\partial\alpha_A$ contributes it at a *different phase* than $\partial\mu/\partial\beta_A$ or $\partial\mu/\partial S_A$ (measured phases $+0.350$ vs $+2.300$ vs $-0.841$). The quadrature loading matrix has singular values (0.768, 0.285): **channel $A$'s second-harmonic content spans a 2-D quadrature subspace, so no single phase for channel $B$ can be orthogonal to all of it.**

Consequences, measured (S4, bins 19 and 38, 10% amplitude, converged 4001-point sweep):

| phase choice | $\mathrm{se}(\mathrm{mROAS}_B)$ relative to a clean comb |
|---|---|
| naive $\phi_B=0$ ("start both probes in January") | **15.53×** |
| random phase, median | 7.97× (IQR 3.7–15.0) |
| local-theory quadrature $\phi_B=0.9650$ | **4.27×** |
| numerically optimal $\phi_B=1.5267$ | **1.085×** |
| worst phase | unbounded (pole) |

The closed form recovers only about half the available gain in log terms; a one-dimensional numerical line search on the FIM recovers essentially all of it. The theorem survives as an exact statement about the local model and as the *reason* phase works; it does not survive as a formula you can plug in.

My own grid statistics also failed. The clean-room verifier showed that the "maximum" I reported (332×) was a 49-point-grid artefact; at 401 / 4001 / 50001 points it reads 687 / 895 / 1181, because the VIF has a genuine pole. **Adopted: never summarise a near-pole function by a grid maximum.** The minimum (1.085) and median (7.97) converge and are the reportable numbers.

### 4.2 The attack that half-landed: on/off flighting has much bigger harmonics than saturation does

Real flights are on/off, and a square wave carries a 3rd harmonic at **32.5%** of its fundamental and a 5th at 18.4% — against the saturation response's $2\omega$ line at 2.0–15.2% over 5–40% probe amplitudes. So the dominant harmonic contamination in a realisable design is the *waveform*, not the response nonlinearity, and it is 2–20× larger.

But the cost is small, and for an instructive reason. The square wave's harmonics are *known*: they are part of the regressor computed from actual spend, so they create a Gram correlation, not an omitted-variable bias. Measured (S9b): $\mathrm{se}_B$ inflation of **1.12–1.23×** for a $3\times$ collision and **1.36–1.41×** for a $5\times$ one. And the asymmetry matters:

> Saturation collisions are **phase-defeasible** (1.09× to unbounded, a swing of two-plus orders of magnitude under the designer's control). Waveform collisions are **not** (min ≈ median ≈ max), because the harmonic's phase is locked to the fundamental by the waveform — but they are small.

Non-negativity, separately, is a non-issue at realistic amplitudes: a sinusoid clipped at zero spend generates no harmonics at all until the amplitude exceeds 100% of mean spend, and even at 130% (clipping 25% of weeks) the 2nd and 3rd harmonics are 5.9% and 4.6%. Rivera's plant-friendly problem bites only for go-darks.

### 4.3 Attacks that failed

| Attack | Result |
|---|---|
| Multiplicative demand seasonality creates sidebands at $\omega_j\pm\omega_s$ that collide across channels | **Refuted.** Placing channel $B$ exactly on channel $A$'s upper or lower sideband changes $\mathrm{se}(\mathrm{mROAS}_B)$ by $<2\%$ at seasonal amplitudes 0.0 → 0.8 (0.1565 → 0.1408); seasonality slightly *helps*. The sidebands are deterministic and modelled. |
| Record truncation puts the probes off-grid | **Benign.** Truncating to 75% of the design window leaves max $|$off-diagonal$|$ = 0.13 and maxVIF = **1.020**. The comb is robust to ragged records. |
| Misspecified $\alpha$ corrupts the designed phase | **Survives with degradation.** $\alpha$ error $\pm0.05$: se ratio unchanged at 1.10. $+0.10$: 1.63×. $+0.20$: 2.85×. Still far better than the naive 15.5× — phase management is worth doing on a rough prior. |
| The probe switch-on transient destroys orthogonality | **Refuted** (§3.1) once the baseline is correctly modelled as already in steady state. |
| Off-grid channel separation | Channel-to-channel **Rule C**: correlation 0.985 at 0.05 bins, 0.674 at 0.25, $\approx0$ at 0.5 and at every integer $\ge1$. Keep $\ge\tfrac12$ bin, or land on exact integers. |

### 4.4 The attack that reframed the dive: a full nonlinear Monte Carlo

Fitting the actual model (150 reps × 4 designs, calibrated-shape MMM with $\beta_j,\alpha_j$ free and $K_j,S_j$ known, all designs cost-matched) gave (S15):

| design | median RMSE ratio vs the compliant comb |
|---|---|
| comb (rule-compliant) | 1.00× |
| **synchronised 13-week flight (real practice)** | **11.2×** (max 20.2×) |
| comb with 2 octave collisions | **1.03×** (max 1.29×) |
| comb, random phases | 0.96× |

The octave collision — the dive's centrepiece — costs essentially **nothing** here. That is not a refutation; it is the mechanism showing itself:

> **The harmonic-collision penalty is exactly the price of not knowing your saturation curve.** The collision confounds channel $j$'s *curvature* with channel $k$'s *effectiveness*. If the curve shape is calibrated ($\Psi=1$), there is nothing to confound and the penalty is 1.03×. If the full Hill is free ($\Psi\approx10$), the penalty runs from 1.09× to unbounded depending on phase. The whole phenomenon lives inside dive 20's $\Psi$ ladder.

---

## 5. Round 3 — refinement

### 5.1 The synchronised design is not "worse", it is singular

My first negative control reported that six channels flighted on a common 13-week cycle give $\mathrm{se}(\mathrm{mROAS})$ of 119–2388 against 0.12–1.17 for the comb — a 700–2400× penalty. The verifier demolished the number and improved the claim: those figures are `numpy.linalg.pinv` at default `rcond` on an **exactly singular** information matrix, and are pure floating-point noise (they move to 63/9.7/202/… at `rcond=1e-11`, to 0.26/0.042/… at `1e-6`, and mpmath at 60–300 digits returns quadratic forms growing like $10^{\text{dps}}$ and flipping sign).

The correct statement is stronger and provable. $T/28=13$, so **every** spend path, adstock, response and Jacobian column of a bin-28 design is exactly 13-periodic; all 24 structural columns lie in a 13-dimensional subspace. Reproduced independently (S14): rank 19 of 32 under the full Hill (nullity 13, $s_{\min}/s_{\max}=1.9\times10^{-12}$), rank 18 of 20 under the calibrated-shape model, and every channel's $\mathrm{mROAS}$ gradient has a large component *inside* the null space ($\|P_{\text{null}}g\|/\|g\|$ up to 0.821 and 0.694). The comb, by contrast, is full rank in both parametrisations with zero leak.

> **Synchronised flighting does not degrade multi-channel MMM identification. It destroys it.** The $11.2\times$ RMSE penalty in §4.4 is a *lower bound* manufactured by the optimizer's damping and its start at the truth — the statistical truth is non-identification. This is the multichannel form of dive 20 §6's "the profit-maximising media plan is exactly non-identifying," and of dives 16/19's frozen-learning stall: a plan that moves all channels on one clock has one degree of freedom where the model needs $n$.

**Methodological rule adopted for the whole program:** any standard error reported from a Fisher matrix must be accompanied by an estimability check ($s_{\min}/s_{\max}$, and the mROAS gradient's projection on the numerical null space). `pinv` silently manufactures finite standard errors for non-identified functionals, and dives 09, 16 and 19 all report pinv-based numbers that have not had this check.

### 5.2 What the design recipe becomes

1. **Choose the record length before the design.** $T=52k$ with $\gcd(k,6)=1$ — 5, 7, 11 or 13 years — makes protection of the 2nd and 3rd distortion products free.
2. **Admissible bins:** $4\le m\le T/(4\tau_j)$, and $m,2m,3m$ all off the seasonal comb $\{pT/52\}$. At $T=364$, $\tau=2.5$, $P=3$ this is 30 bins.
3. **Assign one bin per channel**, integer, distinct; keep $\ge\tfrac12$ bin of separation if any bin must be fractional (Rule C). Prefer a $2\times/3\times$-free subset (20 of the 30 available) — but do not treat this as a hard constraint, because
4. **Set the phases.** Where a harmonic collision is unavoidable, choose $\phi_k$ by a one-dimensional numerical minimisation of $\mathrm{se}(\mathrm{mROAS}_k)$ on the FIM, *not* by the closed-form quadrature angle (1.09× vs 4.27× vs 15.5× naive). This step costs nothing and is, as far as the literature scan reaches, entirely absent from both marketing and system-identification practice for this purpose.
5. **Never synchronise.** A common flighting clock across channels is not a weak design, it is a rank-deficient one (§5.1).
6. **Budget the amplitude by parameter.** ~10% continuous amplitude buys mROAS at the frontier; **~24%** is what the $2\omega$ line needs to identify curvature at $z=2$, and the curvature is what collapses $\Psi$ from ~10 to ~1 (a $3.1\times$ standing precision gain, dive 20).
7. **If the plan must be budget-neutral,** accept that the aggregate marginal value is identified only through adstock *phase* contrast: not at all if the memories match, and never at DC. Buy it instead with one non-neutral total-budget tone at an admissible bin, which is 4–615× cheaper (§3.5).
8. **Read the harmonics.** The $2\omega_j$ and $3\omega_j$ lines are a free, model-free curvature estimate and saturation-shape test on every channel (§3.6). Reserve them: do not let another channel's fundamental sit there.

---

## 6. Completion checklist

Sixteen numbered sections, all reproducible from `code/21_spectral_comb.py`. Robustness sweeps: probe amplitude (5/10/20/40%), collision order ($p=2,3,5$), adstock-pair sweep (9 values), $\alpha$ misspecification ($\pm0.05/0.10/0.20$), seasonal amplitude (0→0.8), record truncation (100%→75%), channel scale ($0.25\times$→$4\times$), record length ($k=2$–15), $\tau$ (1.25–5.0), phase-grid convergence (49→50,001 points), Monte-Carlo size (400→40,000 reps). Negative/power controls: seasonal-harmonic bin, duplicated bin, synchronised design, the $c_3=0$ null in the shape test, and the trend-free ablation that turns four "approximate" laws exact. Uncertainty on every headline number is stated inline (MC standard errors, grid-convergence ranges, or the clean-room's independent value).

---

## 7. Devil's advocate, per headline claim

**"Heterogeneous adstock is free (Proposition 1)."** It is a one-line corollary of LTI diagonality once you write it down, and Morelli's flight-test literature has been exploiting the underlying fact since the 1990s. Its content here is only that *marketing's* extra structure — channel-specific memory — does not break the transplant. It is also a statement about the first-order design matrix only; §3.3 exists precisely because the saturated response is not in that matrix.

**"The comb attains the frontier for all $n$ channels simultaneously (Theorem 1)."** True for the *local slope* under an *additive* channel response. Synergy (cross-channel terms in $h$) would put intermodulation products at $\omega_i\pm\omega_j$ and break block-diagonality; I did not simulate a synergistic response at all. The cost side, at least, survives synergy, since the comb makes $\mathrm{Cov}(a_i,a_j)=0$ so the cross-curvature term drops out — but that is an argument, not a verified result.

**"The channel-count tax: 20 channels pay 1.75–5.5% of media spend."** This inherits every assumption of dive 20's Theorem 2 — stationarity, a single decision at the horizon, quadratic certainty-equivalent regret — and dive 20's own devil's advocate notes both the drift correction (raises it) and the deadband correction (lowers it). It also assumes each channel is probed to its own optimum independently, which the block-diagonal FIM justifies but a real budget process would not. And $\Psi$ is not measured for any real advertiser (BL74).

**"VIF $=1/\sin^2(p\psi_j-\psi_k)$ (Theorem 2)."** Exact in the local Volterra model, verified to $4\times10^{-4}$ relative. It is *not* the operative quantity in the full structural model, where the $p$-th-harmonic subspace is 2-D (§4.1) and the closed-form angle is 4× worse than the numerical optimum. It also assumes the collision is with a single curvature parameter; with $K$, $S$ and $\alpha$ all free the "curvature" is a three-dimensional object.

**"Phase is worth two orders of magnitude."** Only when the saturation curve is being estimated. The nonlinear Monte Carlo with a calibrated shape shows a 1.03× penalty for two octave collisions (§4.4) — i.e. for the many advertisers who fix their response curves by prior or by experiment, this whole construction is a rounding error. The honest scope is: *phase management matters exactly in proportion to $\Psi-1$.*

**"Budget-neutral variation cannot identify the total marginal value (Theorem 3)."** Exactly true for the stated model. Three escapes I did not price: functional-form identification through the saturation itself (dive 19 measured rmse 5.6 at the stall, so it is weak but nonzero); channel-specific *seasonality* in the baseline, which makes the "sum" direction time-varying; and heterogeneous, non-geometric memory kernels, where $\mathrm{Im}(\bar H_1H_2)$ need not vanish at DC — a delayed-peak (Erlang) adstock has a phase lag at $\omega\to0^+$ that geometric adstock does not. That last one may substantially weaken the DC null and should be checked before the rule is used.

**"The optimal budget-neutral period is 8.6–20.5 weeks."** In four of the six $\alpha$-pairs tested the unconstrained optimum sits *inside* the $4\tau_{\max}$ carryover floor, so the constrained answer is the floor, not the interior optimum. The interior optimum is a statement about a model with adstock known; the floor is what applies when it is not.

**"Synchronised flighting is exactly singular."** It is exactly singular *for a period that divides the record* — $T/28=13$. A 13-week flight in a 365-week record is merely very ill-conditioned rather than exactly rank-deficient, and real flighting calendars drift. The qualitative claim (one clock cannot identify $n$ channels) is robust; the exact zeros are an artefact of commensurability, exactly as dive 20 §6's exact zeros were.

**"The harmonic readout gives curvature for free."** At the amplitudes advertisers actually run, it gives nothing: SNR 0.36 at a 10% probe. It also requires the phase $\psi_j$, hence $\alpha_j$; the modulus estimator that avoids that requirement is badly biased at low SNR. And $\mathrm{se}=4\sigma\sqrt{2/T}/B^2$ assumes white residuals — dive 20 §5.2 showed MMM residuals are strongly autocorrelated, so the operative noise is $S_\varepsilon(2\omega_j)$, not $\sigma^2$, which for red noise makes the $2\omega$ line *cheaper* than this formula says. I did not compute that correction.

---

## 8. Independent verification

A clean-room subagent, given only the model specification and ten numerical claims (no code), re-implemented everything from scratch — closed-form circulant steady-state adstock, complex-step and analytic Jacobians, cross-checked against an mpmath arbitrary-precision build.

**Reproduced:** C1 ($7.5\times10^{-16}$), C3 and C4 (max relative error $3.76\times10^{-4}$ / $2.00\times10^{-4}$, and the whole VIF table 3.084 / 1.501 / 1.009 / 1.479 / 117.63 / 148.24 / 3.084), C5's headline numbers (0.320363, 0.154381, 15.532, 4.2684, and the quadrature angle 0.9650), C6 (0.310737 / 0.121649 / 1.518609 / 0.092642 and both corollaries), C7 (1.000528 … 1.000264, identical to six digits), C8's noiseless recovery, C9(a) (identical to six digits, including the cross-check that channels A and B match the two-channel run — as orthogonality requires), C10 (30 admissible bins, 20-element maximum harmonic-free subset, confirmed by exact branch-and-bound *and* an independent ILP, plus the coprimality enumeration for $k=2$–20).

**Six corrections raised and all adopted:**

1. The trend coupling is exactly **$-6/(T^2-7)$**, not $-6/T^2$ (12 digits at five window lengths); and it is exactly bin-independent, also with the full basis ($-4.606684\times10^{-5}$).
2. C3/C4's tolerance is **relative**, not absolute (max absolute error 47 near the pole); the laws are **exact to $10^{-10}$ once the trend column is dropped**, which also explains the residuals in C2, C6 and C7. One cause, five symptoms.
3. **C5's sweep statistics were grid artefacts.** Converged values: min 1.0853 at $\phi_B=1.52670$ (not 1.10 at 1.505), median 8.028 (not 8.45), and the maximum is a pole whose measured value tracks the grid (332 / 687 / 895 / 1181 at 49 / 401 / 4001 / 50001 points).
4. C6's second corollary is weaker than I stated: **two** budget-neutral tones suffice for full rank, not four.
5. C8's Monte-Carlo agreement was overclaimed: 400 reps carry a 3.5% standard error on an SD and cannot adjudicate a 4% gap. At 40,000 reps the theory is confirmed to 0.01%/0.05%. The verifier additionally established the **modulus-estimator bias at low SNR** ($+0.561$ against a truth of $-0.157$ at 10% amplitude), which became §3.6's main practical caveat.
6. **C9(b) was wrong in a way that strengthens the result:** the synchronised design's Fisher matrix is exactly singular (13-periodicity ⇒ nullity $\ge11$; mROAS gradients leak 0.19–0.78 into the null space), and my 837/119/2388/… were `pinv` noise at default `rcond`. Rewritten as total loss of identification (§5.1), with a general estimability rule adopted for the program.

The verifier also correctly narrowed Rule R from an "iff" to a sufficient condition ($k=2,3,9,15$ also cost nothing, because the extra exclusions fall below the $m\ge4$ floor).

---

## 9. Limitations and failure modes

- **Additive channels only.** No synergy, no cross-channel saturation, no shared pacing. Intermodulation at $\omega_i\pm\omega_j$ is the obvious next collision class and is entirely uninvestigated.
- **Sinusoids and square waves only.** Crest-factor-constrained, monthly-quantised, non-negative multisines (BL77) are still not transported; §4.2 only shows that the naive square wave's harmonics are large but cheap.
- **White residuals throughout.** Dive 20's $S_\varepsilon(\omega)$ generalisation should apply bin-by-bin across the comb, which would make the *choice of which channel gets which bin* a real optimisation (cheap channels to noisy bins). Not done.
- **Geometric adstock only.** The DC null in Theorem 4 is a property of a filter family whose transfer function is real at $\omega=0$; delayed-peak kernels may break it (§7).
- **The phase optimum is numerical.** I have a theorem for the local model and a line search for the real one, with no closed form for the full-model optimal angle. That is the least satisfying part of the dive.
- **No real data.** Every number is from the working point. The single most checkable prediction — that advertisers whose channels share one flighting calendar have rank-deficient MMMs — is untested against any actual spend calendar.

---

## 10. Next steps for a future session

1. **Intermodulation under synergy.** With a cross-channel term the comb's distortion products land at $\omega_i\pm\omega_j$, and the seasonal comb interacts. Derive the sum-and-difference exclusion set, and check whether the phase law generalises (it should: the product of two tones has phase $\psi_i\pm\psi_j$). This is the direct continuation of dive 19's BL71 (switch cones under synergy).
2. **Bin assignment as an optimisation.** With coloured residuals, $\Psi_j(\omega)$ from dive 20, and per-channel $\tau_j$, choosing *which* channel gets *which* admissible bin is an assignment problem with a computable objective $\sum_j \kappa_j S_\varepsilon(\omega_j)\Psi_j(\omega_j)$. Solve it; the prediction is that low-memory channels take the high bins and noisy-band channels get pushed off the red end of the spectrum.
3. **A closed form for the full-model optimal phase.** §4.1 shows the $p$-th harmonic subspace is 2-D with singular values (0.768, 0.285); the optimal angle should be the one orthogonal to the dominant left singular vector, which is computable from $\partial\arg H/\partial\alpha$. Deriving it would turn step 4 of the recipe from a line search into a formula.
4. **Audit a real flighting calendar.** Take an actual advertiser's weekly spend matrix, compute the rank and the mROAS-gradient null-space leak of its implied Fisher matrix, and report how close to singular real plans are. This is the most checkable claim in the dive and needs no new theory. Merges with dive 20's next-step 3.
5. **The estimability audit of dives 09, 16 and 19.** §5.1's rule — every FIM standard error needs $s_{\min}/s_{\max}$ and a null-space leak — has not been applied retroactively. Some `pinv`-based numbers elsewhere in this program may be manufactured.
6. **Price the phase.** The phase choice is worth $1.09\times$ vs $15.5\times$ in standard error at $\Psi\approx10$ and $1.03\times$ at $\Psi=1$. Express that as an EVSI in dive 03/04's machinery and compare against the shape-fixing experiment ($\sqrt{10}$) — i.e. is it cheaper to *manage the phase* or to *calibrate the curve*?
7. **Delayed-peak adstock and the DC null.** Check whether Theorem 4's $\omega=0$ null survives an Erlang/Weibull carryover kernel; if it does not, budget-neutral variation is less crippled than §3.5 claims and $\Lambda_{\text{neutral}}$ needs recomputing.

---

## 11. Sources

- Internal: dive 20 (Theorems 1 and 2, Rule L, the $\Psi$ ladder — this dive is its $n$-channel extension), dive 19 (BL70, the budget direction; BL71, switch cones), dive 16 (frozen learning), dive 11 (cross-channel leakage), dives 03/04 (the decision plateau, which reappears again in §4.4), dive 09 (the multichannel ellipsoid, which presumes $1'g$ known), dive 01 (block design, amplitude and curvature). Catalogues `../02_open_questions.md` §M1, §M2, §M7; `../03_mmm_adoption_barriers.md` §O7, §B1 (the channel-count tax is a quantitative form of the "measurement is too expensive for mid-size advertisers" barrier).
- Multi-input design: [Morelli 2021, NASA/AIAA (NTRS 20210018115)](https://ntrs.nasa.gov/api/citations/20210018115/downloads/Morelli_Practical_FTI_Paper_v1%200706.pdf) — the orthogonal interleaved-harmonic multisine, the direct prior art · [Gevers, Miskovic, Bonvin & Karimi 2006, *Automatica* 42(4)](https://www.sciencedirect.com/science/article/abs/pii/S0005109806000148) · [Rojas, Welsh, Goodwin & Feuer 2007, *Automatica* 43(6)](https://webee.technion.ac.il/people/feuer/JournalPapers/56_Robust_Optimal_Experiment.pdf) · [Bombois et al. 2006, *Automatica* 42(10)](https://perso.uclouvain.be/michel.gevers/PublisMig/C130.pdf) · [finite-sample MIMO multisine identification, arXiv:2510.26929](https://arxiv.org/abs/2510.26929) · [Rivera et al., plant-friendly identification](https://www.researchgate.net/publication/250441444_Constrained_minimum_crest_factor_multisine_signals_for_Plant-Friendly_identification_of_highly_interactive_systems) · Schroeder 1970 (crest-factor phases); Schoukens–Pintelon–Guillaume (harmonic-suppressed and odd-odd multisines — the nonlinear-distortion analogue of §3.6) · Zhu, *Multivariable System Identification for Process Control*.
- MMM: [Chan & Perry 2017, Google](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/2d0395bc7d4d13ddedef54d744ba7748e8ba8dd1.pdf) · [Dew, Padilla & Shchetkina 2024, arXiv:2408.07678](https://arxiv.org/html/2408.07678) · [Meridian, budget optimization scenarios](https://developers.google.com/meridian/docs/user-guide/budget-optimization-scenarios) · [Recast, managing multicollinearity](https://getrecast.com/understand-and-manage-multicollinearity/).

*Code:* `code/21_spectral_comb.py` (run `python 21_spectral_comb.py` for all sections, or name sections e.g. `S3 S7`; **S15**, the nonlinear Monte Carlo, takes several minutes at the default 150 reps).
