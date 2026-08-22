"""RH: rate-based safeguarded hybrid (v2 contribution).

Replaces the v1 kappa-engineering (Lanczos + closed-form theta*) by an
economic decision on *realized* progress:

  1. realized contraction rho_hat from a sliding-window log-linear fit of
     ||grad|| (self-validating: trusted only when the window spans enough);
  2. predicted remaining first-order work  T_stay = c1_hat*ln(gn/eps)/|ln rho_hat|;
  3. expected second-order entry cost      T_sw   = c2_hat * K2_est(gn, eps),
     K2_est counts squarings with floor 1 (Prop. 2: quadratics need exactly 1);
  4. switch iff gn <= theta_cap and T_sw < margin * T_stay;
  5. the switch itself is a safeguarded Newton probe (Lemma 1 assumptions);
     rejected probes are measured, fed into c2_hat, and trigger a cooldown --
     never a permanent latch like the v1 allow_switch flag.

No Lanczos, no Hessian-vector products during phase 1, no theta* formula at
runtime. See THEORY.md section 5 for the design contract.
"""
import time

import numpy as np

from methods import _backtrack, _budget_ok, _BB, _bb_descent, _newton_dir
from hybrid import _fo_init, newton_tail


def _rho_fit(log_gn):
    """Realized per-step contraction from least squares on ln||g||.

    Returns (rho_hat, span) where span = ln(max/min) inside the window.
    """
    y = np.asarray(log_gn, float)
    j = np.arange(len(y), dtype=float)
    A = np.vstack([j, np.ones_like(j)]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    slope = float(coef[0])
    span = float(y.max() - y.min())
    return float(np.exp(-slope)), span


def rate_hybrid(pb, x0, eps=1e-8, max_time=600.0,
                win=30, span_trust=3.0,
                check_every=10, margin=1.5, cap_frac=0.5,
                backoff_max=200, fo_mode="auto", name="RH"):
    """First-order until the economic test fires, then safeguarded Newton.

    fo_mode: "auto" -> NAG-AR when constants unknown (else constant-momentum
    NAG); "bb" -> force the v1-style safeguarded BB step (ablation).
    """
    from methods import _AdaptiveNAG
    x_cur = np.asarray(x0, float).copy()
    y = x_cur.copy()
    t0 = time.perf_counter()
    hist = []
    fe = ge = he = hv = 0
    it1 = it2 = 0
    known, alpha, beta = _fo_init(pb)
    if not known and fo_mode == "auto":
        anag = _AdaptiveNAG()
        y_old = y.copy()
        known = True          # handled by the anag branch below
        alpha = None
    else:
        anag = None
    bb = _BB()
    g0n = float(np.linalg.norm(pb.grad(x_cur)))
    ge += 1
    theta_cap = cap_frac * g0n

    win_gn = []                 # recent ln||g||
    step_times = []
    c1_hat = None
    probe_costs = []            # measured build+solve wall times
    c2_hat = None
    rho_hat, span = None, None
    cooldown_until = -1
    switched_at = None
    probes_failed = 0
    decisions = 0
    fired = 0

    # pending carries (y_old, y, f_y, g, gn) already computed by the anag
    # branch so the loop head does not re-evaluate the gradient.
    pending = None
    f_y = pb.f(x_cur); fe += 1

    while True:
        if pending is not None:
            y_old, y, f_y, g, gn = pending
            pending = None
        else:
            g = pb.grad(y)
            ge += 1
            gn = float(np.linalg.norm(g))
        hist.append((time.perf_counter() - t0, gn))
        if gn <= eps or not _budget_ok(it1 + it2, t0, int(9e9), max_time):
            break

        # ------------- cheap decision layer (scalar regression only) --------
        win_gn.append(np.log(max(gn, 1e-300)))
        if len(win_gn) > win:
            win_gn.pop(0)
        can_decide = (it1 >= 5 and len(win_gn) >= 8 and it1 >= cooldown_until)
        if can_decide and (it1 % check_every == 0 or gn <= theta_cap):
            rho_hat, span = _rho_fit(win_gn)
            trusted = bool(span >= span_trust and np.isfinite(rho_hat))
            if trusted and c1_hat:
                u_ep = np.log(1.0 / eps)
                u_g = np.log(1.0 / max(gn, 1e-300))
                if 0.0 < u_g < u_ep:
                    k2_est = max(1.0, np.log(u_ep / u_g) / np.log(2.0))
                else:
                    k2_est = 1.0   # far from / at solution: entry-step cost only
                t_stay = c1_hat * u_ep / max(abs(np.log(rho_hat)), 1e-12)
                t_sw = (c2_hat if c2_hat else 0.0) * k2_est
                decisions += 1
                # no c2 data yet: the first probe IS the measurement
                fire = gn <= theta_cap and (
                    c2_hat is None or t_sw < margin * t_stay)
                if fire:
                    fired += 1
                    ts_probe = time.perf_counter()
                    gx = pb.grad(x_cur)
                    ge += 1
                    H = pb.hess(x_cur)
                    he += 1
                    p = _newton_dir(H, gx)
                    f_x = pb.f(x_cur)
                    fe += 1
                    a, fe = _backtrack(pb, x_cur, f_x, gx, p)
                    probe_cost = time.perf_counter() - ts_probe
                    probe_costs.append(probe_cost)
                    probe_costs[:] = sorted(probe_costs)[-10:]
                    c2_hat = float(np.median(probe_costs[-10:]))
                    if a == 0.0:
                        probes_failed += 1
                        cooldown_until = it1 + min(
                            5 * (2 ** min(probes_failed, 8)), backoff_max)
                        continue
                    x_cur = x_cur + a * p
                    switched_at = float(gn)
                    break

        # -------------------- first-order phase step ------------------------
        ts_step = time.perf_counter()
        if anag is not None:
            fe, y_new = anag.step(pb, y_old, y, f_y, g, fe)
            f_y2 = pb.f(y_new); fe += 1
            g2 = pb.grad(y_new); ge += 1
            pending = (y, y_new, f_y2, g2, float(np.linalg.norm(g2)))
            x_cur = y
        elif not known or alpha is None:
            f_y = pb.f(y)
            fe += 1
            a, fe, x_new = _bb_descent(pb, y, f_y, g, bb, fe)
            y = x_new
            x_cur = x_new
        else:
            x_new = y - alpha * g
            y = x_new + beta * (x_new - x_cur)
            x_cur = x_new
        it1 += 1
        step_times.append(time.perf_counter() - ts_step)
        if len(step_times) > 20:
            step_times.pop(0)
        c1_hat = float(np.mean(step_times))

    # ----------------------------- phase 2 tail -----------------------------
    gn_f, xf = gn, x_cur
    if switched_at is not None:
        s2 = newton_tail(pb, x_cur, eps, max_iter=2000,
                         max_time=max(1.0, max_time - (time.perf_counter() - t0)))
        it2 = s2["iters"]
        he += s2["hevals"]
        ge += s2["gevals"]
        fe += s2["fevals"]
        hist += s2["history"]
        xf, gn_f = s2["xfinal"], s2["gnorm"]
    out = dict(method=name, iters=it1 + it2, fevals=fe, gevals=ge, hevals=he,
               hvevals=hv, time_s=time.perf_counter() - t0,
               gnorm=float(gn_f), fval=float(pb.f(xf)),
               converged=bool(gn_f <= eps), history=hist,
               phase1_iters=it1, phase2_iters=it2,
               theta_used=(theta_cap if fired else None),
               switched=bool(switched_at is not None),
               switched_at_gnorm=switched_at, probes_failed=probes_failed,
               kappa_hat=None, c1_est=c1_hat, c2_est=c2_hat,
               rho_hat=rho_hat, decisions=decisions, probes_fired=fired)
    return out
