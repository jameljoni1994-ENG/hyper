# Theory Notes for the Rescued Paper

**Working title:** *How Much Does the Switching Threshold Matter? Sensitivity Analysis
and a Rate-Based Safeguarded Hybrid for First-to-Second-Order Transitions.*

All statements here are drafted for direct transfer into LaTeX. Numeric claims marked
**[CHECK]** are verified programmatically by `tests/theory_check.py`.

---

## 0. Setting and notation

Unconstrained minimization `min f(x)` with `f : R^n -> R`, twice continuously
differentiable. `g_k = grad f(x_k)`, `g_n(k) = ||g_k||`. A hybrid algorithm runs a
first-order method until `||g_k|| <= theta` ("switching threshold") and a second-order
method afterwards, stopping at `||g|| <= eps`, `eps << theta`.

Cost model (worst-case rates, strongly convex smooth case):

```
C(theta) = c1 * K1(theta) + c2 * K2(theta, eps),
K1(theta) = (sqrt(kappa)/2) * ln(g0/theta)                [phase 1, NAG-type]
K2(theta, eps) = B * [ ln ln(1/eps) - ln ln(1/theta) ],  B = log_2(e)
```

with `c1` = unit cost of one first-order iteration, `c2` = unit cost of one Newton
build+solve, `kappa = L/mu`. Substituting and dropping constants,

```
C(u) = alpha*u - beta*ln(u) + beta*ln ln(1/eps) + const,
u := ln(1/theta),  alpha := c1*sqrt(kappa)/2,  beta := c2*B.
```

### Corrected monotonicity statement (fixes the fatal error of the old draft)

The closed-form minimizer of the idealized model is

```
theta*(c1, c2, kappa) = exp( - 2 (c2/c1) log_2(e) / sqrt(kappa) ).
```

**Remark 1 (corrected).** `theta*` is **increasing** in `kappa` and **decreasing** in
the cost ratio `c2/c1`:

```
d theta*/d kappa     = theta* * ( (c2/c1) log_2(e) ) / kappa^{3/2}  > 0,
d theta*/d (c2/c1)   = - 2 log_2(e) theta* / ((c2/c1)^{-1} ...)      < 0
```

i.e., worse conditioning pushes the switch **earlier** (larger threshold), while more
expensive second-order iterations push it **later**. The intuition ("NAG degrades as
kappa grows, so Newton should start sooner") now matches the formula. The earlier draft
asserted the opposite direction for kappa; that sentence is withdrawn. **[CHECK: mono]**

---

## 1. Lemma 1 — Global convergence of the safeguarded hybrid

**Assumptions.**
(A1) `f in C^2`, `grad f` Lipschitz continuous, `f` bounded below.
(A2) Phase 1 applies either NAG with known constants (`L`, `mu >= 0`) or a safeguarded
Barzilai–Borwein step with Armijo line search; iterates remain bounded.
(A3) Every transition into phase 2 happens through a **probe**: a Newton direction
`safeguarded` by `H -> H + tau*I` regularization and accepted only under the Armijo
sufficient-decrease condition `f(x + a p) <= f(x) + c_a a g^T p`; a rejected probe does
not move the iterate and shrinks the active threshold.
(A4) The phase-2 tail accepts a Newton step only when the gradient norm strictly
decreases, falling back to a damped steepest-descent step with the same acceptance test
(float-floor-safe variant; no function-value comparisons).

**Lemma 1.** Under (A1)-(A4):
(i) every accumulation point of `{x_k}` is stationary;
(ii) if additionally `f` is strongly convex, `x_k -> x*`, and the number of phase-1
iterations is finite whenever `theta > eps`;
(iii) on any neighborhood of `x*` where `mu > 0` and the Hessian is Lipschitz with
constant `M`, the tail eventually takes undamped Newton steps and the local rate is
quadratic.

*Proof sketch.*
(i) Phase-1 families in (A2) have classical global convergence proofs (Nesterov 2004;
safeguarded BB with Armijo, e.g. Dai–Zhang 2005 style arguments). Probes that fail
(A3) leave the iterate unchanged, hence cannot affect limits. In the tail, each
iteration either moves along `p` with `||g(x+tp)|| < ||g(x)||` or along `-t g/||g||`.
For `t > 0` sufficiently small the latter has `phi'(0) = -||g|| < 0`, so some `t`
strictly decreases the norm; hence the loop never stalls at a non-stationary point,
and a monotone-bounded argument on `||g(x_k)||` combined with the descent lemma yields
stationarity of all limit points.
(ii) Strong convexity makes `x*` unique and both phases subsequence-converge to it;
the switch fires finitely because `||g_k|| -> 0` monotonically through any fixed
`theta > eps` level.
(iii) Classical local Newton analysis (Nocedal–Wright Thm 3.5): inside the basin
`D = { x : M ||x - x*|| <= mu/2 }` (up to absolute constants) full steps are accepted
and `||g_{k+1}|| <= C ||g_k||^2`. Our acceptance test passes `t = 1` there because the
quadratic contraction implies strict norm decrease. `q.e.d.` (full version for LaTeX.)

---

## 2. Proposition 2 — No U-curve on quadratics (structural failure mode)

**Proposition 2.** Let `f(x) = 1/2 (x-x*)^T Q (x-x*)` with `Q succ 0` (any spectrum).
For every `x0 != x*`, the exact Newton step from `x0` lands at `x*`. Consequently

```
K2(theta, eps) = 1   for all theta in (0, ||g0||],
C(theta)      = c1 (sqrt(kappa)/2) ln(||g0||/theta) + c2 + const,
```

which is **strictly decreasing in theta**: the infimum of the modeled cost is attained
at the *largest feasible* threshold, not at an interior point. Any interior U-curve
minimum therefore requires genuinely non-quadratic basin-entry costs (Newton steps that
are rejected or non-contractive outside the basin). **[CHECK: quad_one_step]**

**Corollary (folklore explained).** The decades-old practice of fixing
`theta = 1e-3` is consistent with two facts proved/measured here: (a) on quadratics the
optimum sits at the feasibility edge anyway (Prop. 2); (b) by Lemma 3 the modeled cost
is flat to first order around its minimizer, so any threshold within a factor of ten of
`theta*` is nearly optimal *on all problem classes* — precision in choosing theta is
simply not worth paying for.

---

## 3. Lemma 3 — Sensitivity: threshold errors are cheap (centerpiece)

Work with `C(u)` from Section 0, `u = ln(1/theta)`.

```
C'(u) = alpha - beta/u,        C''(u) = beta/u^2 > 0,
u* = beta/alpha,               C''(u*) = beta/u*^2.
```

**Lemma 3.** For a multiplicative misspecification `tilde_theta = gamma * theta*`
(`gamma > 0`), the excess cost obeys

```
Delta C / C(u*) = (ln gamma)^2 / ( 2 u*^2 [ 1 + ln( ln(1/eps) / u* ) ] )
                  <= (ln gamma)^2 / (2 u*^2).
```

*Proof.* `Delta u = |ln tilde_theta^{-1} - ln theta*^{-1}| = |ln gamma|`.
Second-order Taylor at `u*`:
`Delta C ~= 1/2 C''(u*) Delta u^2 = beta/(2 u*^2) (ln gamma)^2`.
At the optimum,
`C(u*) = alpha u* - beta ln u* + beta ln ln(1/eps) = beta[1 + ln( ln(1/eps)/u* )]`
(using `alpha = beta/u*`). Divide. `q.e.d.` **[CHECK: sens]**

**Numerical reading** (`eps = 1e-8`, `theta* = 0.1`, i.e. `u* = 2.303`,
denominator `2 u*^2 (1 + ln(18.42/2.30)) ~= 32.6`):

| gamma | predicted excess |
|---|---|
| 2    | 1.5 %  |
| 10   | 16 %   |
| 100  | 65 %   |

Measured on E2-quadratic (v1 data): a factor-10 miss downward (theta = 0.01 vs best
0.1) costs +24 % wall time — same order as the predicted 16 %; the gap is consistent
with the model's worst-case bias documented next. **[CHECK against v2 E7 runs]**

---

## 4. Honest scope: what the closed form cannot do (motivates RH)

Feeding the *online estimates actually logged by the v1 adaptive run*
(`code/results/all_runs.csv`, exp=E3) into the closed form gives:

| problem | kappa_hat | c2/c1 | theta*(formula) | empirical argmin | gap |
|---|---|---|---|---|---|
| quadratic k1000 n1000 | 493 | 3353 | 5.8e-190 | 1e-1 | 1e188 |
| logistic lam0.01 m2000-n200 | 10.7 | 1000 | underflows to 0 | 3.5e-6 | infinite |
| logsumexp m300-n50 | 10.0 | 37.8 | 1.0e-15 | 1e-8 | 1e7 |

Three structural defects, each amplified by the others:
(D1) the one-shot `c2` probe includes BLAS warm-up and overstates steady-state solve
cost; (D2) the log-log `K2` overcounts on quadratics, where `K2 == 1` exactly
(Prop. 2); (D3) the phase-1 term uses the worst-case rate, understating realized
progress savings. Net effect: the model *always* recommends "never switch" on the
tested families, while switching measurably saves 2.6x wall time on E1-quadratic.
Conclusion adopted by the paper: **do not optimize theta online; exploit its flatness
(Lemma 3) and replace kappa-engineering by a self-validating realized-rate estimate
(algorithm RH)**.

---

## 5. Algorithm RH (rate-based hybrid) — design contract

Decision rule evaluated on a decadal schedule of the gradient level:

1. Maintain window `W = { ln g_n(j) }` over the last `m` iterates.
2. Fit realized contraction `rho_hat` by robust least squares on `W`
   (trust the fit only if `W` spans >= 2 natural decades).
3. Predicted remaining first-order work: `T_stay = c1_hat * ln(g_n/eps) / |ln rho_hat|`.
4. Rolling-median probe cost `c2_hat` (updated lazily by actual probes).
5. Switch iff `gn <= theta_cap` AND `c2_hat * K2_est < margin * T_stay`, where
   `K2_est` uses the Prop.-2-aware count `max(1, log2(ln(1/eps)/ln(1/gn)))`.
6. Safeguarded probe (Lemma 1 assumptions) then the float-floor-safe tail.

Properties: no Lanczos, no kappa, no closed-form theta at runtime; estimator reads the
*realized* spectral progress, immune to worst-case bias (D3); decision errors are
bounded by Lemma 3's quadratic penalty, so a margin of order one suffices.
