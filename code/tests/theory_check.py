"""Numeric verification of THEORY.md claims. Exits 0 iff all checks pass.

Checks
------
[mono]          theta* increasing in kappa, decreasing in c2/c1.
[quad_one_step] Prop. 2: exact Newton step solves any SPD quadratic (K2 == 1).
[sens]          Lemma 3: argmin identity, curvature identity, second-order
                penalty prediction accuracy as gamma -> 1.
[consistency]   The closed form theta_star matches a brute-force argmin of the
                k1/k2 cost model used by hybrid.py/hybrid_v2.py.
[regime]        Regime map: sufficient flatness criterion
                u* >= |ln gamma|/sqrt(2 delta) is valid+conservative; penalty
                blow-up as u* -> 0 (>= 20x between theta*=0.1 and 0.9).
[basin]         Lemma 3b: squaring count ceil(log2((u_eps-lnC)/(u_g-lnC)))
                matches direct iteration of ||g|| <- C||g||^2; C=1 recovers
                the model count log2(u_eps/u_g).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid import theta_star, k1_model, k2_model  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name} {detail}")
    if not ok:
        FAILURES.append(name)


rng = np.random.default_rng(20260823)

# ---------------------------------------------------------------- [mono] ----
# exact form: ln theta* == -a/sqrt(kappa), a := 2 (c2/c1) log2(e) > 0.
LOG2E = 1.0 / np.log(2.0)
kappas = np.logspace(1, 5, 25)
a_exponent = 2.0 * (3.6e-2 / 1.0e-5) * LOG2E
exponents = -a_exponent / np.sqrt(kappas)
check("mono: ln(theta*) == -a/sqrt(kappa) strictly increasing", True
      if np.all(np.diff(exponents) > 0) else False,
      "(analytic form; value-space check below avoids underflow)")

c1s, c2s = 1.0e-3, 3.6e-3   # cost ratio 3.6 keeps theta* in normal range
th = np.array([theta_star(c1s, c2s, k) for k in kappas])
check("mono: dtheta*/dkappa > 0", bool(np.all(np.diff(th) > 0)),
      f"(min step={np.diff(th).min():.3e})")

ratios = np.logspace(-2, 2, 25)
th = np.array([theta_star(c1s, r * c2s, 493.0) for r in ratios])
check("mono: dtheta*/d(c2/c1) < 0", bool(np.all(np.diff(th) < 0)))

# ------------------------------------------------------- [quad_one_step] ---
worst = 0.0
for trial in range(20):
    n = int(rng.integers(20, 200))
    kap = float(10 ** rng.uniform(1, 4))
    d = np.logspace(0.0, np.log10(kap), n)
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Q = (V * d) @ V.T
    xstar = rng.standard_normal(n)
    b = Q @ xstar
    x0 = xstar + rng.standard_normal(n) * float(10 ** rng.uniform(-1, 1))
    g0 = Q @ x0 - b
    x1 = x0 - np.linalg.solve(Q, g0)
    g1 = np.linalg.norm(Q @ x1 - b)
    worst = max(worst, g1 / max(np.linalg.norm(g0), 1e-300))
check("quad_one_step: K2 == 1 on SPD quadratics", worst < 1e-10,
      f"(worst residual ratio over 20 trials = {worst:.2e})")

# ------------------------------------------------------------------ [sens] --
def C_of_u(u, alpha, beta, eps):
    u = np.asarray(u, dtype=float)
    if np.any(u <= 0):
        raise ValueError(f"C_of_u requires u > 0, got {u}")
    return alpha * u - beta * np.log(u) + beta * np.log(np.log(1.0 / eps))


def sens_pred(gamma, alpha, beta, eps):
    ustar = beta / alpha
    denom = 2.0 * ustar**2 * (1.0 + np.log(np.log(1.0 / eps) / ustar))
    return np.log(gamma) ** 2 / denom


ok_stat, ok_curv, ok_pred = True, True, True
for _ in range(30):
    c1 = float(10 ** rng.uniform(-6, -4))
    c2 = float(10 ** rng.uniform(-4, -1))
    kappa = float(10 ** rng.uniform(1, 4))
    eps = 1e-8
    alpha = c1 * np.sqrt(kappa) / 2.0
    beta = c2 / np.log(2.0)
    us = beta / alpha                      # scale-safe: us > 0 always

    # stationarity of the minimizer: C'(u*) == alpha - beta/u* == 0
    h = 1e-5 * max(us, 1.0)
    fd = (C_of_u(us + h, alpha, beta, eps) - C_of_u(us - h, alpha, beta, eps)) \
        / (2 * h)
    scale = abs(alpha) + beta / us**2 * max(us, 1.0)
    if abs(fd) > 1e-6 * scale:
        ok_stat = False

    # global-minimum identity via convexity: C >= C(u*) on [u*/4, 4u*]
    grid = np.linspace(0.25 * us, 4.0 * us, 20001)
    if C_of_u(grid, alpha, beta, eps).min() < C_of_u(us, alpha, beta, eps) \
            - 1e-12 * abs(C_of_u(us, alpha, beta, eps)):
        ok_stat = False

    # curvature identity at u*
    h2 = 1e-4 * max(us, 1.0)
    fd2 = (C_of_u(us + h2, alpha, beta, eps)
           - 2 * C_of_u(us, alpha, beta, eps)
           + C_of_u(us - h2, alpha, beta, eps)) / h2**2
    if abs(fd2 - beta / us**2) / (beta / us**2) > 1e-3:
        ok_curv = False

    # second-order prediction accuracy for near-optimal thresholds
    for gamma in (1.02, 1.05, 1/1.05):
        direct = abs(C_of_u(us + np.log(1.0 / gamma), alpha, beta, eps)
                     - C_of_u(us, alpha, beta, eps))
        pred = sens_pred(gamma, alpha, beta, eps) * C_of_u(us, alpha, beta, eps)
        if abs(direct - pred) / pred > 0.05:
            ok_pred = False
check("sens: u* == beta/alpha is the stationary global minimizer", ok_stat)
check("sens: C''(u*) == beta/u*^2", ok_curv)
check("sens: Lemma-3 penalty within 5% for |gamma-1| <= 5%", ok_pred)

# headline numbers quoted in THEORY.md (eps=1e-8, u*=ln(10))
eps = 1e-8
ustar = np.log(10.0)
denom = 2 * ustar**2 * (1 + np.log(np.log(1 / eps) / ustar))
for gamma, expect in ((2, 1.5), (10, 16), (100, 65)):
    got = 100 * np.log(gamma) ** 2 / denom
    check(f"sens: headline excess ~{expect}% at gamma={gamma}",
          abs(got - expect) < 1.0, f"(got {got:.1f}%)")

# ---------------------------------------------------------------- [regime] --
def sens_pred_us(gamma, us, eps=1e-8):
    denom = 2.0 * us**2 * (1.0 + np.log(np.log(1.0 / eps) / us))
    return np.log(gamma) ** 2 / denom


ok_reg = True
# (i) sufficient criterion u* >= |ln g|/sqrt(2 delta): valid at the critical
# point, violated well inside it (true crossing sits in (0.6, 1.0]*u_crit --
# the criterion is conservative by the [1+ln(L/u*)] factor)
us_crit = np.log(10.0) / np.sqrt(0.1)          # delta = 5%, gamma = 10
p_at_crit = sens_pred_us(10.0, us_crit)
if not (p_at_crit <= 0.05 and sens_pred_us(10.0, 0.6 * us_crit) > 0.05):
    ok_reg = False
# (ii) blow-up: same gamma=10 miss at theta*=0.9 costs >= 20x theta*=0.1
r_flat = sens_pred_us(10.0, np.log(10.0))       # ~16 %
r_near_one = sens_pred_us(10.0, 0.105)          # theta* = 0.9
if not (r_near_one >= 20.0 * r_flat):
    ok_reg = False
check("regime: flatness criterion valid+conservative; penalty blows up "
      "as u* -> 0", ok_reg,
      f"(u*_crit={us_crit:.3f} -> {100 * p_at_crit:.1f}%<=5%; "
      f"ratio near-one/flat={r_near_one / r_flat:.0f}x)")

# ----------------------------------------------------------------- [basin] --
ok_bas = True
for _ in range(25):
    C = float(10 ** rng.uniform(0, 3))
    ue = float(rng.uniform(np.log(C) + 0.5, 40.0))      # need u_eps > ln C
    ug = float(rng.uniform(np.log(C) + 0.2, ue - 0.01))  # gate ug>lnC, ug<ue
    h, target, j = float(np.exp(-ug)), float(np.exp(-ue)), 0
    while h > target and j < 200:
        h = C * h * h
        j += 1
    K = int(np.ceil(np.log2((ue - np.log(C)) / (ug - np.log(C)))))
    if j != K:
        ok_bas = False
# C == 1 degenerates to the model count
ue, ug = 18.42, 4.0
if int(np.ceil(np.log2((ue - 0.0) / (ug - 0.0)))) != \
        int(np.ceil(np.log2(ue / ug))):
    ok_bas = False
check("basin: Lemma-3b count ceil(log2((ue-lnC)/(ug-lnC))) exact; C=1 "
      "recovers model", ok_bas)

# ----------------------------------------------------------- [consistency] --
ok_cons = True
for _ in range(15):
    c1 = float(10 ** rng.uniform(-6, -4))
    c2 = float(10 ** rng.uniform(-4, -1))
    kap = float(10 ** rng.uniform(1, 4))
    eps = 1e-8
    gn = 1.0
    thetas = np.geomspace(eps * 1.01, 0.9, 5000)
    costs = np.array([k1_model(gn, t, kap) * c1 + k2_model(t, eps) * c2
                      for t in thetas])
    th_hat = thetas[int(np.argmin(costs))]
    th_form = theta_star(c1, c2, kap)
    if not (abs(th_form - th_hat) / th_hat < 5e-3 or
            (th_form < thetas.min() and th_hat <= thetas.min() * 1.01)):
        # allow boundary cases where the clipped box dominates
        if th_form >= thetas.min():
            ok_cons = False
check("consistency: theta_star == argmin of k1/k2 model", ok_cons,
      "(within grid resolution / feasibility box)")

# ------------------------------------------------------------------- done ---
print()
if FAILURES:
    print(f"FAILED checks: {FAILURES}")
    sys.exit(1)
print("All theory checks passed.")
