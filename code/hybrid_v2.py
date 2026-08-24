"""RH: rate-based safeguarded hybrid (v2 contribution).

Replaces the v1 kappa-engineering (Lanczos + closed-form theta*) by an
economic decision on *realized* progress. Two firing arms:

  A. VALIDATED ECONOMICS.
     1. realized contraction rho_hat from a sliding-window log-linear fit
        of ||grad||. Trust requires a FULL window, genuine descent
        (0 < rho_hat < 1), a minimal signal span (span_trust natural-log
        units -- SNR against noise, NOT progress magnitude: demanding a
        large span here is anti-correlated with switch-worthiness, since
        a wide span inside a fixed window means first-order progress is
        already excellent), and line-fit quality
        resid <= max(0.15*span, 0.05) (rejects rising transients, whose
        |ln rho_hat| would understate remaining work by orders of
        magnitude);
     2. predicted remaining first-order work
        T_stay = c1_hat*ln(gn/eps)/|ln rho_hat|;
     3. expected entry cost T_sw = c2_eff*K2_est(gn, eps); K2_est counts
        squarings with floor 1 (Prop. 2: quadratics need exactly 1).
        c2_eff is a PESSIMISTIC PRIOR c2_prior*c1_hat until the first
        probe is measured (rolling median thereafter). Asymmetry
        rationale: Lemma 3 makes lateness cheap ((ln gamma)^2 penalty,
        regime map) while earliness cost 6x wall time on logistic in the
        pre-prior pilot;
     4. fire iff gn <= theta_cap and T_sw < margin * T_stay.

  B. DESPERATION MEASUREMENT. If the estimator never validates but the
     gradient has sat under theta_cap for streak_need consecutive
     checkpoints, fire a MEASUREMENT probe anyway (safeguarded per
     Lemma 1; microscopic acceptances feed c2_hat and trigger the
     escalating cooldown). Covers slow-flat regimes where no 30-point
     window ever looks like clean exponential descent (logsumexp-type).

  5. the switch itself is a safeguarded Newton probe (Lemma 1 assumptions;
     two-stage acceptance: strict gradient-norm decrease, else Armijo on f
     -- the norm arm is float-floor-safe). A probe fails ONLY when neither
     arm certifies any step; such probes are measured, fed into c2_hat,
     and trigger a cooldown -- they never open the tail. Tiny-but-certified
     steps are accepted (near-flat curvature produces correct Newton
     directions at pathological lengths); the tail's stagnation monitor
     judges whether tail mode actually pays and bails otherwise.

The switch is an experiment, NOT a latch: after a successful probe the same
loop continues in 'tail' mode under a stagnation monitor -- every 50 tail
iterations the gradient norm must sit below 0.98 * ref_gn (reference
snapshot: anchored at tail entry, refreshed to the running minimum at each
checkpoint), otherwise the algorithm bails back to 'fo' mode with an
escalating cooldown min(50*2^giveups, 500) and resets its rate estimator.

No Lanczos, no Hessian-vector products during phase 1, no theta* formula at
runtime. See THEORY.md section 5 for the design contract.
"""
import time

import numpy as np

from methods import _backtrack, _budget_ok, _BB, _bb_descent, _newton_dir
from hybrid import _fo_init


def _rho_fit(log_gn):
    """Realized per-step contraction from least squares on ln||g||.

    Returns (rho_hat, span, resid): span = ln(max/min) inside the window,
    resid = std of the fit residuals (line quality / oscillation gauge).
    """
    y = np.asarray(log_gn, float)
    j = np.arange(len(y), dtype=float)
    A = np.vstack([j, np.ones_like(j)]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    slope = float(coef[0])
    span = float(y.max() - y.min())
    resid = float(np.std(y - (coef[0] * j + coef[1])))
    return float(np.exp(-slope)), span, resid


def rate_hybrid(pb, x0, eps=1e-8, max_time=600.0,
                win=30, span_trust=0.5,
                check_every=10, margin=1.5, cap_frac=0.5,
                c2_prior=100.0, streak_need=5,
                backoff_max=200, fo_mode="auto", name="RH",
                verbose=False):
    """First-order until the economic test fires, then safeguarded Newton.

    fo_mode: "auto" -> NAG-AR when constants unknown (else constant-momentum
    NAG); "bb" -> force the v1-style safeguarded BB step (ablation).

    Single-loop implementation over modes 'fo' | 'tail': the switch is an
    experiment, not a latch; monitored re-entry with escalating cooldown.
    """
    from methods import _AdaptiveNAG
    x_cur = np.asarray(x0, float).copy()
    y = x_cur.copy()
    t0 = time.perf_counter()
    hist = []
    fe = ge = he = hv = 0
    it1 = it2 = giveups = 0
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

    win_gn = []                 # recent ln||g|| (fo mode only)
    step_times = []
    c1_hat = None
    probe_costs = []            # measured build+solve wall times
    c2_hat = None
    rho_hat, span, resid = None, None, None
    cooldown_until = -1
    switched_at = None          # recorded once: the first successful probe
    probes_failed = 0
    decisions = 0
    fired = 0
    cap_streak = 0              # consecutive checkpoints with gn <= theta_cap

    # pending carries (y_old, y, f_y, g, gn) already computed so the loop
    # head does not re-evaluate the gradient.
    pending = None
    f_y = pb.f(x_cur); fe += 1
    mode = "fo"                 # 'fo' | 'tail'
    ref_gn = np.inf             # tail-stagnation reference snapshot
    tail_since_check = 0

    while True:
        # ------------------------- loop-top state -------------------------
        if pending is not None:
            y_old, y, f_y, g, gn = pending
            pending = None
        elif mode == "fo":
            g = pb.grad(y); ge += 1
            gn = float(np.linalg.norm(g))
        else:                   # tail: gradient at the true iterate
            g = pb.grad(x_cur); ge += 1
            gn = float(np.linalg.norm(g))
        hist.append((time.perf_counter() - t0, gn))
        if gn <= eps or not _budget_ok(it1 + it2, t0, int(9e9), max_time):
            break

        if mode == "fo":
            # ---------- cheap decision layer (scalar regression) ----------
            win_gn.append(np.log(max(gn, 1e-300)))
            if len(win_gn) > win:
                win_gn.pop(0)
            cap_streak = cap_streak + 1 if gn <= theta_cap else 0
            can_decide = (it1 >= 5 and len(win_gn) >= win
                          and it1 >= cooldown_until)
            if can_decide and (it1 % check_every == 0 or gn <= theta_cap):
                rho_hat, span, resid = _rho_fit(win_gn)
                trusted = bool(span >= span_trust and np.isfinite(rho_hat)
                               and 0.0 < rho_hat < 1.0
                               and resid <= max(0.15 * span, 0.05))
                desperate = bool(not trusted and cap_streak >= streak_need)
                if (trusted or desperate) and c1_hat:
                    u_ep = np.log(1.0 / eps)
                    u_g = np.log(1.0 / max(gn, 1e-300))
                    if 0.0 < u_g < u_ep:
                        k2_est = max(1.0, np.log(u_ep / u_g) / np.log(2.0))
                    else:
                        k2_est = 1.0  # far from / at solution: entry cost only
                    decisions += 1
                    if verbose:
                        print(f"    [RH-dec] it1={it1} gn={gn:.3e} "
                              f"trusted={trusted} desperate={desperate} "
                              f"rho={rho_hat:.3f} span={span:.2f} "
                              f"resid={resid:.3f} streak={cap_streak} "
                              f"c2={c2_hat}", flush=True)
                    if desperate and not trusted:
                        # estimator never validated, but the gradient has
                        # camped under the cap: probe to MEASURE (c2_hat
                        # is learned whatever the Armijo outcome).
                        fire = True
                    else:
                        t_stay = c1_hat * u_ep / max(abs(np.log(rho_hat)),
                                                     1e-12)
                        c2_eff = c2_hat if c2_hat is not None \
                            else c2_prior * c1_hat
                        t_sw = c2_eff * k2_est
                        fire = gn <= theta_cap and t_sw < margin * t_stay
                    if fire:
                        fired += 1
                        ts_probe = time.perf_counter()
                        gx = pb.grad(x_cur); ge += 1
                        H = pb.hess(x_cur); he += 1
                        p = _newton_dir(H, gx)
                        # Two-stage probe acceptance, mirroring the tail:
                        # strict gradient-NORM decrease first (certifiable
                        # down to the float gradient floor, where
                        # Armijo-on-f cannot resolve decreases of order
                        # ||g||^2 against f-evaluation noise), else classic
                        # backtracking Armijo on f.
                        a, gnx, t_try = 0.0, float(np.linalg.norm(gx)), 1.0
                        while t_try > 1e-12:
                            g_try = pb.grad(x_cur + t_try * p); fe += 1
                            if np.isfinite(g_try).all() and float(
                                    np.linalg.norm(g_try)) < \
                                    (1 - 1e-12) * gnx:
                                a = t_try
                                break
                            t_try *= 0.5
                        if a == 0.0:
                            f_x = pb.f(x_cur); fe += 1
                            a, fe = _backtrack(pb, x_cur, f_x, gx, p)
                        probe_cost = time.perf_counter() - ts_probe
                        probe_costs.append(probe_cost)
                        probe_costs[:] = sorted(probe_costs)[-10:]
                        c2_hat = float(np.median(probe_costs[-10:]))
                        if verbose:
                            print(f"    [RH-probe] it1={it1} gn={gn:.3e} "
                                  f"a={a:.3e} certified={a > 0.0} "
                                  f"cost={probe_cost:.4f}s", flush=True)
                        if a < 1e-12:
                            # Neither arm certified any step: genuine stall
                            # signature -- measure, cool down, stay in fo.
                            # Tiny-but-certified steps ARE accepted: near-flat
                            # curvature yields correct Newton directions at
                            # pathological lengths; the tail's stagnation
                            # monitor, not this gate, polices productivity.
                            probes_failed += 1
                            cooldown_until = it1 + min(
                                5 * (2 ** min(probes_failed, 8)), backoff_max)
                            continue
                        x_cur = x_cur + a * p
                        if switched_at is None:
                            switched_at = float(gn)
                        mode = "tail"
                        ref_gn = np.inf      # anchored at first tail loop-top
                        tail_since_check = 0
                        cap_streak = 0
                        continue

            # ---------------------- first-order step ----------------------
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
        else:
            # -------------- tail step (norm-decrease Newton) --------------
            if ref_gn == np.inf:
                ref_gn = gn         # anchor the reference at tail entry
            moved = False
            H = pb.hess(x_cur); he += 1
            p = _newton_dir(H, g)
            if np.isfinite(p).all():
                t = 1.0
                while t > 1e-16:
                    g_try = pb.grad(x_cur + t * p); ge += 1
                    if np.isfinite(g_try).all() and \
                            float(np.linalg.norm(g_try)) < (1 - 1e-12) * gn:
                        x_cur = x_cur + t * p
                        moved = True
                        break
                    t *= 0.5
            if not moved:
                pg = g / max(gn, 1e-300)
                t = 1.0
                while t > 1e-16:
                    g_try = pb.grad(x_cur - t * pg); ge += 1
                    if np.isfinite(g_try).all() and \
                            float(np.linalg.norm(g_try)) < (1 - 1e-12) * gn:
                        x_cur = x_cur - t * pg
                        moved = True
                        break
                    t *= 0.5
            it2 += 1
            tail_since_check += 1
            if tail_since_check >= 50:
                tail_since_check = 0
                if gn > 0.98 * ref_gn:
                    # stagnation: the basin hypothesis failed -- bail to fo.
                    giveups += 1
                    mode = "fo"
                    cooldown_until = it1 + min(
                        50 * (2 ** min(giveups, 9)), 500)
                    win_gn = []      # rate estimator stale across modes
                    cap_streak = 0
                    y = x_cur.copy()
                    g_bail = pb.grad(x_cur); ge += 1
                    gn_bail = float(np.linalg.norm(g_bail))
                    if anag is not None:
                        y_old = x_cur.copy()
                        anag.px = None
                        f_bail = pb.f(x_cur); fe += 1
                    else:
                        y_old = y.copy()
                        f_bail = 0.0  # unused by CM/BB branches
                    pending = (y_old, y.copy(), f_bail, g_bail, gn_bail)
                    ref_gn = np.inf
                else:
                    ref_gn = min(ref_gn, gn)

    # ------------------------------- output --------------------------------
    xf = x_cur
    out = dict(method=name, iters=it1 + it2, fevals=fe, gevals=ge, hevals=he,
               hvevals=hv, time_s=time.perf_counter() - t0,
               gnorm=float(gn), fval=float(pb.f(xf)),
               converged=bool(gn <= eps), history=hist,
               phase1_iters=it1, phase2_iters=it2,
               theta_used=(theta_cap if fired else None),
               switched=bool(switched_at is not None),
               switched_at_gnorm=switched_at, probes_failed=probes_failed,
               kappa_hat=None, c1_est=c1_hat, c2_est=c2_hat,
               rho_hat=rho_hat, decisions=decisions, probes_fired=fired,
               giveups=giveups)
    return out
