"""Optimization methods compared in the hybrid-switching study.

Every solver shares the interface
    solver(pb, x0, eps=1e-8, max_iter=..., max_time=...) -> stats dict
with keys: method, iters, fevals, gevals, hevals, hvevals,
time_s, gnorm, fval, converged, history [(t, ||g||), ...].

Cost accounting: fevals/gevals count objective/gradient calls; hevals counts
dense Hessian formations + linear solves; hvevals counts Hessian-vector
products. Wall time is measured with perf_counter.
"""
import time
import numpy as np

# ----------------------------------------------------------------- helpers ---

def _stats(name, pb, x, t0, it, fe, ge, he, hv, hist, gnorm):
    return dict(method=name, iters=it, fevals=fe, gevals=ge, hevals=he,
                hvevals=hv, time_s=time.perf_counter() - t0,
                gnorm=float(gnorm), fval=float(pb.f(x)),
                converged=None, history=hist)


def _budget_ok(it, t0, max_iter, max_time):
    return it < max_iter and (time.perf_counter() - t0) < max_time


def _backtrack(pb, x, f_x, g, d, c1=1e-4, dec=0.5, nmax=50, fe=0):
    """Backtracking Armijo along direction d (descent assumed)."""
    t, gd = 1.0, float(g @ d)
    for _ in range(nmax):
        if pb.f(x + t * d) <= f_x + c1 * t * gd:
            return t, fe + 1
        t *= dec
        fe += 1
    return 0.0, fe + nmax


class _BB:
    """Safeguarded Barzilai-Borwein step-size state for unknown-L problems."""

    def __init__(self):
        self.px = None
        self.pg = None

    def alpha(self, pb, x, g, fe):
        a_bb = None
        if self.px is not None:
            s = x - self.px
            yv = g - self.pg
            ss, sy, yy = s @ s, s @ yv, yv @ yv
            if ss > 0 and yy > 0 and sy != 0:
                a1 = ss / max(sy, 1e-300)
                a2 = sy / max(yy, 1e-300)
                cand = a1 if (fe % 2) else a2
                if np.isfinite(cand) and 1e-12 < abs(cand) < 1e12:
                    a_bb = float(cand)
        self.px, self.pg = x.copy(), g.copy()
        return a_bb


def _newton_dir(H, g):
    """Ridge-safeguarded Newton direction with sanity bounds."""
    n = len(g)
    s = float(np.abs(np.diag(H)).max()) if len(H) else 1.0
    s = s if s > 0 else 1.0
    scale = max(np.linalg.norm(g) / s, 1.0)
    for reg in (0.0, 1e-12 * s, 1e-8 * s, 1e-6 * s, 1e-4 * s):
        try:
            p = -np.linalg.solve(H + reg * np.eye(n), g)
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(p).all() and (g @ p) < 0 \
                and np.linalg.norm(p) <= 1e8 * scale:
            return p
    return -g


def _bb_descent(pb, x, f_x, g, bb, fe, dec=0.5):
    """One BB-gradient step with Armijo safeguard; returns (alpha, fe, x_new)."""
    a_bb = bb.alpha(pb, x, g, fe)
    d = -g
    t0 = min(1.0, a_bb) if a_bb else 1.0
    gd = float(g @ d)
    t = max(min(t0, 1e10), 1e-16)
    for _ in range(60):
        xn = x + t * d
        if np.all(np.isfinite(xn)):
            fn = pb.f(xn)
            if np.isfinite(fn) and fn <= f_x + 1e-4 * t * gd:
                return t, fe + 1, xn
        fe += 1
        t *= dec
    return 1e-8, fe, x - 1e-8 * g


# ------------------------------------------------------------------- GD ------

def gd(pb, x0, eps=1e-8, max_iter=int(5e5), max_time=120.0):
    x = np.asarray(x0, float).copy()
    t0, it, fe, ge, hv = time.perf_counter(), 0, 0, 0, 0
    hist = []
    known = pb.L is not None and getattr(pb, "mu", 0.0)
    alpha = 2.0 / (pb.L + pb.mu) if known else 1.0
    bb = _BB()
    while True:
        g = pb.grad(x); ge += 1
        gn = float(np.linalg.norm(g))
        hist.append((time.perf_counter() - t0, gn))
        if gn <= eps or not _budget_ok(it, t0, max_iter, max_time):
            s = _stats("GD", pb, x, t0, it, fe, ge, 0, hv, hist, gn)
            s["converged"] = gn <= eps
            return s
        if not known:                       # safeguarded BB variant
            f_x = pb.f(x); fe += 1
            a, fe, x = _bb_descent(pb, x, f_x, g, bb, fe)
        else:
            x -= alpha * g
        it += 1


# ------------------------------------------------------------- NAG (t-sched) -

def nag_t(pb, x0, eps=1e-8, max_iter=int(5e5), max_time=120.0):
    x = np.asarray(x0, float).copy()
    y = x.copy(); t = 1.0
    t0, it, fe, ge, hv = time.perf_counter(), 0, 0, 0, 0
    hist = []
    known = pb.L is not None and pb.L > 0
    alpha = 1.0 / pb.L if known else 1.0
    bb = _BB()
    while True:
        g = pb.grad(y); ge += 1
        gn = float(np.linalg.norm(g))
        hist.append((time.perf_counter() - t0, gn))
        if gn <= eps or not _budget_ok(it, t0, max_iter, max_time):
            s = _stats("NAG-t", pb, x, t0, it, fe, ge, 0, hv, hist, gn)
            s["converged"] = gn <= eps
            return s
        if not known:
            f_y = pb.f(y); fe += 1
            a, fe, x_new = _bb_descent(pb, y, f_y, g, bb, fe)
            y = x_new                      # no momentum without constants
        else:
            x_new = y - alpha * g
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y = x_new + ((t - 1.0) / t_new) * (x_new - x)
            t = t_new
        x = x_new
        it += 1


# ------------------------------------------------------- NAG constant-momentum

def nag_cm(pb, x0, kappa=None, eps=1e-8, max_iter=int(5e5), max_time=120.0):
    if pb.mu is None or pb.mu <= 0 or pb.L is None or kappa == 0:
        return nag_t(pb, x0, eps, max_iter, max_time)
    kap = kappa if kappa is not None else pb.L / pb.mu
    beta = (np.sqrt(kap) - 1.0) / (np.sqrt(kap) + 1.0)
    alpha = 1.0 / pb.L
    x = np.asarray(x0, float).copy()
    y = x.copy()
    t0, it, fe, ge = time.perf_counter(), 0, 0, 0
    hist = []
    while True:
        g = pb.grad(y); ge += 1
        gn = float(np.linalg.norm(g))
        hist.append((time.perf_counter() - t0, gn))
        if gn <= eps or not _budget_ok(it, t0, max_iter, max_time):
            s = _stats("NAG-CM", pb, x, t0, it, fe, ge, 0, 0, hist, gn)
            s["converged"] = gn <= eps
            return s
        x_new = y - alpha * g
        y = x_new + beta * (x_new - x)
        x = x_new
        it += 1


# ------------------------------------------------------------- Newton + LS ---

def newton_ls(pb, x0, eps=1e-8, max_iter=2000, max_time=600.0, name="Newton"):
    x = np.asarray(x0, float).copy()
    t0, it, fe, ge, he = time.perf_counter(), 0, 0, 0, 0
    hist = []
    while True:
        g = pb.grad(x); ge += 1
        gn = float(np.linalg.norm(g))
        hist.append((time.perf_counter() - t0, gn))
        if gn <= eps or not _budget_ok(it, t0, max_iter, max_time):
            s = _stats(name, pb, x, t0, it, fe, ge, he, 0, hist, gn)
            s["converged"] = gn <= eps
            return s
        H = pb.hess(x); he += 1
        p = _newton_dir(H, g)
        f_x = pb.f(x); fe += 1
        a, fe = _backtrack(pb, x, f_x, g, p)
        if a > 0:
            x = x + a * p
        else:
            ag, fe = _backtrack(pb, x, f_x, g, -g)
            t = ag if ag > 0 else 1e-6 / max(gn, 1e-300)
            x = x - t * g
        it += 1


# --------------------------------------------------------------- L-BFGS ------

def lbfgs(pb, x0, mem=10, eps=1e-8, max_iter=50000, max_time=300.0):
    x = np.asarray(x0, float).copy()
    t0, it, fe, ge = time.perf_counter(), 0, 0, 0
    S, Y, hist = [], [], []
    tail_t = None      # adaptive step size for the float-floor regime
    tail_fail = 0
    tail_run = 0       # consecutive float-floor (tail-mode) steps
    best_gn, best_it = np.inf, 0

    def _done():
        nonlocal ge
        g_f = pb.grad(x); ge += 1
        gn_f = float(np.linalg.norm(g_f))
        hist.append((time.perf_counter() - t0, gn_f))
        s = _stats("L-BFGS", pb, x, t0, it, fe, ge, 0, 0, hist, gn_f)
        s["converged"] = gn_f <= eps
        return s

    while True:
        g = pb.grad(x); ge += 1
        gn = float(np.linalg.norm(g))
        hist.append((time.perf_counter() - t0, gn))
        if gn <= eps or not _budget_ok(it, t0, max_iter, max_time):
            s = _stats("L-BFGS", pb, x, t0, it, fe, ge, 0, 0, hist, gn)
            s["converged"] = gn <= eps
            return s
        if gn < 0.995 * best_gn:
            best_gn, best_it = gn, it
        elif it - best_it >= 500:
            return _done()
        q = g.copy()
        alphas = []
        for s_, y_ in zip(reversed(S), reversed(Y)):
            a = (s_ @ q) / (s_ @ y_)
            alphas.append(a)
            q -= a * y_
        r = q
        if Y:
            gamma = S[-1] @ Y[-1] / (Y[-1] @ Y[-1])
            r = gamma * q
        for (s_, y_), a in zip(zip(S, Y), reversed(alphas)):
            b = (y_ @ r) / (s_ @ y_)
            r += (a - b) * s_
        p = -r
        if g @ p >= 0 or not np.isfinite(p).all():
            p = -g
        f_x = pb.f(x); fe += 1
        a, fe = _backtrack(pb, x, f_x, g, p, dec=0.5)
        use_tail = False
        if a > 0.0:
            x_try = x + a * p
            # Micro-step: displacement below positional resolution. Near |f|
            # large enough that any decrease is below one ULP of f, Armijo
            # degenerates into a rounding lottery; switch to a gradient tail
            # that never compares f values.
            if np.linalg.norm(x_try - x) <= 1e-11 * (1.0 + np.linalg.norm(x)):
                use_tail = True
            else:
                x_new = x_try
                tail_t = None
        else:
            use_tail = True
        if use_tail:
            tail_run += 1
            if tail_run >= 150:
                # f-comparisons are dead here; hand over to the Newton tail.
                return _newton_tail(pb, x, "L-BFGS", eps, t0, it, fe, ge,
                                    0, 0, hist, max_iter, max_time)
        else:
            tail_run = 0
        if use_tail:
            gamma = (S[-1] @ Y[-1] / (Y[-1] @ Y[-1])) if Y else 1.0
            if tail_t is None:
                tail_t = float(np.clip(gamma, 1e-14, 1e14))
            x_new = x - tail_t * g
        g_new = pb.grad(x_new); ge += 1
        gn_new = float(np.linalg.norm(g_new))
        if use_tail:
            if np.isfinite(gn_new) and gn_new < 0.99 * gn:
                tail_fail = 0
            else:
                tail_t *= 0.25          # no progress -> shrink safeguarded step
            if not np.isfinite(gn_new) or gn_new > 4.0 * gn:
                tail_fail += 1
                if tail_fail >= 40:
                    return _done()
                it += 1
                continue                # reject exploding step, retry smaller
        s_ = x_new - x; y_ = g_new - g
        if np.linalg.norm(s_) > 1e-11 * (1.0 + np.linalg.norm(x)) and \
           s_ @ y_ > 1e-10 * np.linalg.norm(s_) * np.linalg.norm(y_):
            S.append(s_); Y.append(y_)
            if len(S) > mem:
                S.pop(0); Y.pop(0)
        x = x_new
        it += 1


# ------------------------------------------- Trust region (Steihaug-CG) ------

def tr_steihaug(pb, x0, eps=1e-8, max_iter=5000, max_time=600.0,
                Delta0=None, eta=0.15, name="TR"):
    x = np.asarray(x0, float).copy()
    Delta = Delta0 if Delta0 else 0.1 * max(1.0, np.linalg.norm(x))
    t0, it, fe, ge, hv = time.perf_counter(), 0, 0, 0, 0
    hist = []
    best_gn, best_it = np.inf, 0
    while True:
        g = pb.grad(x); ge += 1
        gn = float(np.linalg.norm(g))
        hist.append((time.perf_counter() - t0, gn))
        if gn <= eps or not _budget_ok(it, t0, max_iter, max_time):
            s = _stats(name, pb, x, t0, it, fe, ge, 0, hv, hist, gn)
            s["converged"] = gn <= eps
            return s
        if gn < 0.995 * best_gn:
            best_gn, best_it = gn, it
        elif it - best_it >= 200:
            return _newton_tail(pb, x, name, eps, t0, it, fe, ge, 0, hv,
                                hist, max_iter, max_time)
        # --- truncated CG inner solve ---
        z = np.zeros_like(g); r = g.copy(); d = -r.copy()
        rr = r @ r
        hit = False
        gtol = 0.1 * gn
        for _ in range(min(2 * len(x), 300)):
            Hd = pb.hv(x, d); hv += 1
            dd = float(d @ Hd)
            if dd <= 0:                                    # negative curvature
                tau = _boundary(z, d, Delta)
                z = z + tau * d; hit = True; break
            a = rr / dd
            z_new = z + a * d
            if np.linalg.norm(z_new) >= Delta:             # boundary truncation
                tau = _boundary(z, d, Delta)
                z = z + tau * d; hit = True; break
            z, r = z_new, r + a * Hd
            rr_new = r @ r
            if np.sqrt(rr_new) < gtol:
                break
            d = -r + (rr_new / rr) * d
            rr = rr_new
        if np.linalg.norm(z) < 1e-14:                      # stalled inner solve
            Delta *= 8.0
            it += 1
            continue
        pred = -(g @ z + 0.5 * z @ pb.hv(x, z)); hv += 1
        f_x = pb.f(x); fe += 1
        f_new = pb.f(x + z); fe += 1
        if -float(g @ z) <= 1e-12 * (abs(f_x) + 1.0):
            rho = 1.0                       # predicted gain below float noise
        else:
            rho = (f_x - f_new) / pred if pred > 0 else -np.inf
        if rho > eta and np.isfinite(f_new):
            x = x + z
        if rho < 0.25:
            Delta *= 0.25
        elif rho > 0.75 and (hit or np.linalg.norm(z) >= 0.99 * Delta):
            Delta = min(2.0 * Delta, 1e6)
        it += 1


def _boundary(z, d, Delta):
    zd, dd = float(z @ d), float(d @ d)
    disc = max(zd * zd + dd * (Delta * Delta - z @ z), 0.0)
    return (-zd + np.sqrt(disc)) / dd if dd > 0 else 0.0


def _newton_tail(pb, x, name, eps, t0, it, fe, ge, he, hv, hist,
                 max_iter, max_time, tail_t=1.0):
    """Safeguarded Newton steps for the float-floor regime.

    Near a minimizer with large |f|, any decrease is below one ULP of f, so
    f-comparing mechanisms (Armijo / rho tests) degenerate into rounding
    lotteries and the outer loop stagnates just above eps.  The tail takes
    Newton directions unconditionally and accepts only steps that shrink the
    gradient norm; if no Newton multiple works it falls back to a damped
    gradient step, which provably reduces ||grad|| for small enough t.  It
    never evaluates or compares f.
    """
    while True:
        g = pb.grad(x); ge += 1
        gn = float(np.linalg.norm(g))
        hist.append((time.perf_counter() - t0, gn))
        if gn <= eps or not _budget_ok(it, t0, max_iter, max_time):
            s = _stats(name, pb, x, t0, it, fe, ge, he, hv, hist, gn)
            s["converged"] = gn <= eps
            return s
        moved = False
        H = pb.hess(x); he += 1
        p = _newton_dir(H, g)
        if np.isfinite(p).all():
            t = min(tail_t, 1.0)
            while t > 1e-16:
                g_try = pb.grad(x + t * p); ge += 1
                if np.isfinite(g_try).all() and \
                        float(np.linalg.norm(g_try)) < (1 - 1e-12) * gn:
                    x = x + t * p
                    tail_t = min(2.0 * t, 1.0)
                    moved = True
                    break
                t *= 0.5
        if not moved:
            t = 1.0
            pg = g / max(gn, 1e-300)
            while t > 1e-16:
                g_try = pb.grad(x - t * pg); ge += 1
                if np.isfinite(g_try).all() and \
                        float(np.linalg.norm(g_try)) < (1 - 1e-12) * gn:
                    x = x - t * pg
                    moved = True
                    break
                t *= 0.5
        it += 1


# ------------------------- Adaptive cubic regularization (ARC, dense-lite) ---

def arc(pb, x0, eps=1e-8, max_iter=20000, max_time=900.0, sigma0=1.0, name="ARC"):
    x = np.asarray(x0, float).copy()
    sigma = sigma0
    t0, it, fe, ge, he = time.perf_counter(), 0, 0, 0, 0
    hist = []
    lam_tol = 1e-10
    noise_run = 0
    best_gn, best_it = np.inf, 0
    while True:
        g = pb.grad(x); ge += 1
        gn = float(np.linalg.norm(g))
        hist.append((time.perf_counter() - t0, gn))
        if gn <= eps or not _budget_ok(it, t0, max_iter, max_time):
            s = _stats(name, pb, x, t0, it, fe, ge, he, 0, hist, gn)
            s["converged"] = gn <= eps
            return s
        if gn < 0.995 * best_gn:
            best_gn, best_it = gn, it
        elif it - best_it >= 200:
            return _newton_tail(pb, x, name, eps, t0, it, fe, ge, he, 0,
                                hist, max_iter, max_time)
        H = pb.hess(x); he += 1
        lmin = float(np.linalg.eigvalsh(H)[0])
        shift = max(0.0, -lmin) + lam_tol
        # solve phi(t)=||p(t)|| - t/sigma = 0 on [shift, big] by bisection
        def pt(t):
            try:
                return -np.linalg.solve(H + t * np.eye(len(x)), g)
            except np.linalg.LinAlgError:
                return -g
        lo, hi = shift, shift + max(sigma * gn, 1.0)
        p_lo = pt(lo)
        if p_lo @ g >= 0:
            p = -g
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                p_m = pt(mid)
                if not np.isfinite(p_m).all():
                    hi = mid
                    continue
                if p_m @ g >= 0 or np.linalg.norm(p_m) > mid / sigma:
                    lo = mid
                else:
                    hi = mid
            p = pt(0.5 * (lo + hi))
            if not np.isfinite(p).all() or g @ p >= 0:
                p = -g / max(gn, 1e-300)
        pred = -(g @ p + 0.5 * p @ (H @ p) + (sigma / 3.0) * np.linalg.norm(p) ** 3)
        f_x = pb.f(x); fe += 1
        f_new = pb.f(x + p); fe += 1
        if -float(g @ p) <= 1e-12 * (abs(f_x) + 1.0):
            rho_k = 1.0                     # predicted gain below float noise
            noise_run += 1
            if noise_run >= 150:
                return _newton_tail(pb, x, name, eps, t0, it, fe, ge, he,
                                    0, hist, max_iter, max_time)
        else:
            rho_k = (f_x - f_new) / pred if pred > 0 else -np.inf
            noise_run = 0
        if rho_k >= 0.1 and np.isfinite(f_new):
            x = x + p
        if rho_k < 0.25:
            sigma = min(sigma * 2.0, 1e12)
        elif rho_k > 0.75:
            sigma = max(sigma / 2.0, 1e-12)
        it += 1


# --------------------------------------------- Catalyst-style accelerator ----

def catalyst(pb, x0, rho=None, eps=1e-8, max_iter=int(3e5), max_time=300.0):
    """One-step accelerated proximal point (Catalyst flavour): CM update with
    effective constants L+rho and mu_eff=rho."""
    L = pb.L if pb.L else 1.0
    mu = pb.mu if pb.mu and pb.mu > 0 else 0.0
    r = rho if rho else max(mu, L / 100.0)
    Leff, meff = L + r, r
    beta = (np.sqrt(Leff / meff) - 1.0) / (np.sqrt(Leff / meff) + 1.0) if meff > 0 else 0.9
    alpha = 1.0 / Leff
    x = np.asarray(x0, float).copy()
    y = x.copy()
    t0, it, fe, ge = time.perf_counter(), 0, 0, 0
    hist = []
    while True:
        g = pb.grad(y); ge += 1
        gn = float(np.linalg.norm(g))
        hist.append((time.perf_counter() - t0, gn))
        if gn <= eps or not _budget_ok(it, t0, max_iter, max_time):
            s = _stats("Catalyst", pb, x, t0, it, fe, ge, 0, 0, hist, gn)
            s["converged"] = gn <= eps
            return s
        x_new = y - alpha * g
        y = x_new + beta * (x_new - x)
        x = x_new
        it += 1
