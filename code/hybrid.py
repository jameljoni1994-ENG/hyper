"""Hybrid first/second-order switching — core contribution of the study.

Corrected cost model (vs. the original draft):
  * phase 2 count uses a basin-aware term: Newton's quadratic regime starts
    only below a certified radius; otherwise extra safeguarded steps are paid;
  * theta* lives in a feasibility box [theta_lo, theta_hi] and a "never
    switch" decision guards loose-accuracy regimes;
  * kappa is estimated online by Lanczos extremes of Hessian-vector products,
    replacing the dimensionally inconsistent secant formula of the draft;
  * the switch itself is safeguarded: a rejected Newton probe defers the
    transition instead of risking divergence.
"""
import time
import numpy as np

from methods import _backtrack, _budget_ok, _BB, _bb_descent, _newton_dir

LOG2E = 1.0 / np.log(2.0)


# ------------------------------------------------------------ cost models ----

def theta_star(c1, c2, kappa):
    """Model-optimal switching threshold (clipped later to feasibility box)."""
    return float(np.exp(-2.0 * (c2 / max(c1, 1e-300)) * LOG2E / np.sqrt(kappa)))


def k1_model(g_now, target, kappa):
    """First-order iteration estimate from gradient level g_now down to target."""
    ratio = max(g_now, target) / max(target, 1e-300)
    return (np.sqrt(kappa) / 2.0) * np.log(max(ratio, np.exp(1e-12)))


def k2_model(theta, eps, c_basin=1.0):
    """Basin-aware Newton count: linear entry phase + loglog squaring phase."""
    u_th = np.log(1.0 / max(min(theta, 1.0 / c_basin), eps))
    u_ep = np.log(1.0 / eps)
    sq = max(np.log(u_ep) - np.log(max(u_th, 1e-12)), 0.0) / np.log(2.0)
    basin = 0.0 if c_basin * theta <= 1.0 else np.log(max(c_basin * theta, 1.0000001))
    return basin + sq


# ------------------------------------------------------- Lanczos estimator ---

def lanczos_extremes(pb, x, m=25):
    """Extreme eigenvalue estimates of H(x) from m Hessian-vector products."""
    n = pb.n
    rng = np.random.default_rng(11)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    V = np.zeros((n, min(m, n)))
    alpha, beta_l = [], []
    V[:, 0] = v
    w = pb.hv(x, v)
    a = float(v @ w)
    alpha.append(a)
    w -= a * v
    cols = 1
    for j in range(1, min(m, n)):
        b = float(np.linalg.norm(w))
        if b < 1e-14:
            break
        beta_l.append(b)
        V[:, j] = w / b
        w = pb.hv(x, V[:, j])
        a = float(V[:, j] @ w)
        alpha.append(a)
        w -= a * V[:, j]
        if j > 0:
            w -= b * V[:, j - 1]
        # reorthogonalize (cheap, keeps Lanczos stable)
        w -= V[:, :j + 1] @ (V[:, :j + 1].T @ w)
        cols = j + 1
    T = np.diag(alpha[:cols])
    if len(beta_l):
        idx = np.arange(cols - 1)
        T[idx, idx + 1] = beta_l[:cols - 1]
        T[idx + 1, idx] = beta_l[:cols - 1]
    ev = np.linalg.eigvalsh(T)
    return float(ev[-1]), float(ev[0])


def kappa_estimate(pb, x, m=25, lo=10.0, hi=1e8):
    lmax, lmin = lanczos_extremes(pb, x, m)
    kap = lmax / max(lmin, 1e-16)
    return float(np.clip(kap, lo, hi)), lmax, lmin


# --------------------------------------------------------- hybrid solvers ----

def _fo_init(pb):
    known = pb.L is not None and getattr(pb, "mu", 0.0) and pb.mu > 0
    if known:
        kap = pb.L / pb.mu
        return True, 1.0 / pb.L, (np.sqrt(kap) - 1.0) / (np.sqrt(kap) + 1.0)
    return False, None, None


def hybrid_fixed(pb, x0, theta, eps=1e-8, max_time=600.0, name=None):
    """NAG until ||grad|| <= theta, then safeguarded Newton until eps."""
    nm = name or f"Hybrid-fixed(th={theta:g})"
    x_cur = np.asarray(x0, float).copy()
    y = x_cur.copy()
    t0 = time.perf_counter()
    hist = []
    fe = ge = he = hv = it1 = it2 = 0
    known, alpha, beta = _fo_init(pb)
    bb = _BB()
    probes_failed = 0
    switched_at = None
    while True:
        g = pb.grad(y); ge += 1
        gn = float(np.linalg.norm(g))
        hist.append((time.perf_counter() - t0, gn))
        if gn <= eps or not _budget_ok(it1 + it2, t0, int(9e9), max_time):
            break
        if gn > theta:
            # ---- phase 1: accelerated (or safeguarded-BB) step ----
            if not known:
                f_y = pb.f(y); fe += 1
                a, fe, x_new = _bb_descent(pb, y, f_y, g, bb, fe)
                y = x_new
            else:
                x_new = y - alpha * g
                y = x_new + beta * (x_new - x_cur)
            x_cur = x_new
            it1 += 1
            continue
        # ---- attempt switch at the true iterate ----
        gx = pb.grad(x_cur); ge += 1
        H = pb.hess(x_cur); he += 1
        p = _newton_dir(H, gx)
        f_x = pb.f(x_cur); fe += 1
        a, fe = _backtrack(pb, x_cur, f_x, gx, p)
        if a == 0.0:
            probes_failed += 1
            theta *= 0.125
            continue
        switched_at = float(np.linalg.norm(gx))
        x_cur = x_cur + (a if a > 0 else 1e-8) * p
        break
    # ---- phase 2: pure Newton with Armijo ----
    s2 = None
    if switched_at is not None:
        s2 = newton_tail(pb, x_cur, eps, max_iter=2000,
                         max_time=max(1.0, max_time - (time.perf_counter() - t0)))
    if s2 is not None:
        it2 = s2["iters"]
        he += s2["hevals"]; ge += s2["gevals"]; fe += s2["fevals"]
        hist += s2["history"]
        xf, gn_f = s2["xfinal"], s2["gnorm"]
    else:
        xf = x_cur
        gn_f = float(np.linalg.norm(pb.grad(xf))); ge += 1
    total_time = time.perf_counter() - t0
    out = dict(method=("Hybrid-fixed" if name is None else nm),
               iters=it1 + it2, fevals=fe, gevals=ge, hevals=he,
               hvevals=hv, time_s=total_time, gnorm=float(gn_f),
               fval=float(pb.f(xf)), converged=bool(gn_f <= eps),
               history=hist, phase1_iters=it1, phase2_iters=it2,
               theta_used=float(theta), switched=bool(switched_at is not None),
               switched_at_gnorm=switched_at, probes_failed=probes_failed,
               kappa_hat=None)
    return out


def newton_tail(pb, x, eps, max_iter=2000, max_time=600.0):
    """Newton tail that accepts only gradient-norm decrease (float-floor safe).

    Near a minimizer with large |f| the Armijo test on function values
    degenerates into a rounding lottery, so this tail never compares f: it
    takes Newton steps unconditionally and keeps them only when they shrink
    ||grad||, falling back to a damped gradient step (which provably reduces
    the norm for small enough t).
    """
    x = np.asarray(x, float).copy()
    t0, it, fe, ge, he = time.perf_counter(), 0, 0, 0, 0
    hist = []
    while True:
        g = pb.grad(x); ge += 1
        gn = float(np.linalg.norm(g))
        hist.append((time.perf_counter() - t0, gn))
        if gn <= eps or it >= max_iter or (time.perf_counter() - t0) > max_time:
            return dict(xfinal=x, gnorm=gn, iters=it, fevals=fe, gevals=ge,
                        hevals=he, history=[(a + 0.0, b) for a, b in hist])
        H = pb.hess(x); he += 1
        p = _newton_dir(H, g)
        moved = False
        if np.isfinite(p).all():
            t = 1.0
            while t > 1e-16:
                g_try = pb.grad(x + t * p); ge += 1
                if np.isfinite(g_try).all() and \
                        float(np.linalg.norm(g_try)) < (1 - 1e-12) * gn:
                    x = x + t * p
                    moved = True
                    break
                t *= 0.5
        if not moved:
            pg = g / max(gn, 1e-300)
            t = 1.0
            while t > 1e-16:
                g_try = pb.grad(x - t * pg); ge += 1
                if np.isfinite(g_try).all() and \
                        float(np.linalg.norm(g_try)) < (1 - 1e-12) * gn:
                    x = x - t * pg
                    moved = True
                    break
                t *= 0.5
        it += 1


# ------------------------------------------------------ adaptive hybrid ------

def hybrid_adaptive(pb, x0, eps=1e-8, max_time=900.0,
                    update_freq=25, lan_m=25, margin=1.10,
                    theta_floor_mult=50.0, name="Hybrid-adaptive"):
    """Adaptive-threshold hybrid.

    Online estimates:
      c1  rolling mean wall-time of a first-order step;
      c2  measured cost of one dense Newton build+solve (single probe);
      kap Lanczos-based kappa estimate refreshed every `update_freq` steps.
    The threshold is recomputed BEFORE any switch decision and clipped to a
    feasibility box; a 'never switch' guard compares predicted costs.
    """
    x_cur = np.asarray(x0, float).copy()
    y = x_cur.copy()
    t0 = time.perf_counter()
    hist = []
    fe = ge = he = hv = 0
    it1 = it2 = 0
    known, alpha, beta = _fo_init(pb)
    bb = _BB()
    g0n = float(np.linalg.norm(pb.grad(x_cur))); ge += 1
    c1_est, c2_est = None, None
    step_times = []
    kap_hat = None
    theta_cur = None
    allow_switch = True
    switched_at = None
    probes_failed = 0
    theta_log = []
    while True:
        g = pb.grad(y); ge += 1
        gn = float(np.linalg.norm(g))
        hist.append((time.perf_counter() - t0, gn))
        if gn <= eps or not _budget_ok(it1 + it2, t0, int(9e9), max_time):
            break
        do_update = (it1 % update_freq == 0) or \
                    (gn <= (theta_cur if theta_cur else np.inf) * 4.0)
        if do_update and allow_switch:
            kap_hat, _, _ = kappa_estimate(pb, x_cur, lan_m)
            hv += lan_m
            if c2_est is None and gn <= 0.5 * g0n:
                ts = time.perf_counter()
                H = pb.hess(x_cur); he += 1
                try:
                    np.linalg.solve(H, g)
                except np.linalg.LinAlgError:
                    pass
                c2_est = max(time.perf_counter() - ts, 1e-9)
            if c1_est and c2_est and kap_hat:
                th_raw = theta_star(c1_est, c2_est, kap_hat)
                th_lo = theta_floor_mult * eps
                th_hi = max(gn / 4.0, th_lo * 2.0)
                theta_cur = float(np.clip(th_raw, th_lo, th_hi))
                # never-switch guard
                k_sw = k1_model(gn, theta_cur, kap_hat) * c1_est + \
                       k2_model(theta_cur, eps) * c2_est
                k_stay = k1_model(gn, eps, kap_hat) * c1_est
                if k_sw > margin * k_stay:
                    allow_switch = False
                theta_log.append((gn, theta_cur, bool(allow_switch)))
        # ---- first-order phase step (no switching before estimates exist) ----
        if (not allow_switch) or theta_cur is None or gn > theta_cur:
            ts_step = time.perf_counter()
            if not known:
                f_y = pb.f(y); fe += 1
                a, fe, x_new = _bb_descent(pb, y, f_y, g, bb, fe)
                y = x_new
            else:
                x_new = y - alpha * g
                y = x_new + beta * (x_new - x_cur)
            x_cur = x_new
            it1 += 1
            step_times.append(time.perf_counter() - ts_step)
            if len(step_times) > 20:
                step_times.pop(0)
            c1_est = float(np.mean(step_times))
            continue
        # ---- switch probe at the true iterate ----
        gx = pb.grad(x_cur); ge += 1
        H = pb.hess(x_cur); he += 1
        p = _newton_dir(H, gx)
        f_x = pb.f(x_cur); fe += 1
        a, fe = _backtrack(pb, x_cur, f_x, gx, p)
        if a == 0.0:
            probes_failed += 1
            theta_cur = max(theta_cur * 0.125, theta_floor_mult * eps)
            continue
        x_cur = x_cur + a * p
        switched_at = float(gn)
        break
    # ---- phase 2 tail ----
    gn_f, xf = gn, x_cur
    if switched_at is not None:
        s2 = newton_tail(pb, x_cur, eps, max_iter=2000,
                         max_time=max(1.0, max_time - (time.perf_counter() - t0)))
        it2 = s2["iters"]
        he += s2["hevals"]; ge += s2["gevals"]; fe += s2["fevals"]
        hist += s2["history"]
        xf, gn_f = s2["xfinal"], s2["gnorm"]
    out = dict(method=name, iters=it1 + it2, fevals=fe, gevals=ge, hevals=he,
               hvevals=hv, time_s=time.perf_counter() - t0, gnorm=float(gn_f),
               fval=float(pb.f(xf)), converged=bool(gn_f <= eps),
               history=hist, phase1_iters=it1, phase2_iters=it2,
               theta_used=(theta_cur if theta_cur else None),
               switched=bool(switched_at is not None),
               switched_at_gnorm=switched_at, probes_failed=probes_failed,
               kappa_hat=kap_hat, c1_est=c1_est, c2_est=c2_est,
               theta_log=str(theta_log)[-2000:])
    return out
