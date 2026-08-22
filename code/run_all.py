"""Run experiments E1-E6 with per-run CSV checkpoints and resume support.

Usage:
    python run_all.py --smoke            # fast sanity pass (<2 min)
    python run_all.py                    # full suite (~105 min budget)
    python run_all.py --exp E1 --exp E2  # selected stages only
"""
import argparse
import csv
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import make_problem                      # noqa: E402
import methods as M                                    # noqa: E402
from hybrid import hybrid_fixed, hybrid_adaptive       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
HISTDIR = os.path.join(RESULTS, "hist")
CSV_PATH = os.path.join(RESULTS, "all_runs.csv")

FIELDS = ["exp", "problem", "cfg", "seed", "method", "theta",
          "iters", "phase1_iters", "phase2_iters",
          "fevals", "gevals", "hevals", "hvevals",
          "time_s", "gnorm", "fval", "converged", "switched",
          "switched_at_gnorm", "probes_failed", "kappa_hat",
          "c1_est", "c2_est", "error"]


# ------------------------------------------------------------- bookkeeping --

def _ensure_dirs():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(HISTDIR, exist_ok=True)


def load_done():
    done = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done.add(_key(r["exp"], r["problem"], r["cfg"], r["method"],
                              r["theta"], int(float(r["seed"]))))
    return done


def _key(exp, problem, cfg, method, theta, seed):
    return f"{exp}|{problem}|{cfg}|{method}|{theta}|{seed}"


def save_result(row):
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
        fh.flush()
        os.fsync(fh.fileno())
    hist = row.get("history") or []
    if hist:
        k = _key(row["exp"], row["problem"], row["cfg"], row["method"],
                 str(row.get("theta", "")), int(row["seed"]))
        arr = np.array([[t, g] for t, g in hist], dtype=float)
        np.savez_compressed(os.path.join(HISTDIR, k.replace("|", "__") + ".npz"),
                            tg=arr)


# ------------------------------------------------------------------ runner --

SOLVERS = {
    "GD": lambda pb, x0, eps, mt: M.gd(pb, x0, eps=eps, max_time=mt),
    "NAG-t": lambda pb, x0, eps, mt: M.nag_t(pb, x0, eps=eps, max_time=mt),
    "NAG-CM": lambda pb, x0, eps, mt: M.nag_cm(
        pb, x0, kappa=(pb.L / pb.mu if (pb.L and getattr(pb, "mu", 0) and pb.mu > 0) else None),
        eps=eps, max_time=mt),
    "Newton": lambda pb, x0, eps, mt: M.newton_ls(pb, x0, eps=eps, max_time=mt),
    "L-BFGS": lambda pb, x0, eps, mt: M.lbfgs(pb, x0, eps=eps, max_time=mt),
    "TR": lambda pb, x0, eps, mt: M.tr_steihaug(pb, x0, eps=eps, max_time=mt),
    "ARC": lambda pb, x0, eps, mt: M.arc(pb, x0, eps=eps, max_time=mt),
    "Catalyst": lambda pb, x0, eps, mt: M.catalyst(pb, x0, eps=eps, max_time=mt),
}


def initial_point(problem_kind, n, seed):
    if problem_kind == "rosenbrock":
        return np.where(np.arange(n) % 2 == 0, -1.2, 1.0)
    rng = np.random.default_rng(9000 + seed)
    return rng.standard_normal(n)


def execute(exp, kind, cfg, seed, method, theta=None, eps=1e-8, mt=120.0):
    pb = make_problem(kind, **cfg)
    x0 = initial_point(kind, pb.n, seed)
    t_start = time.perf_counter()
    try:
        if method == "Hybrid-fixed":
            s = hybrid_fixed(pb, x0, theta=theta, eps=eps, max_time=mt)
        elif method == "Hybrid-adaptive":
            s = hybrid_adaptive(pb, x0, eps=eps, max_time=min(mt, 900.0))
        else:
            s = SOLVERS[method](pb, x0, eps, mt)
    except Exception:
        err = traceback.format_exc(limit=3)
        print(f"  !! FAIL {exp}/{kind}/{cfg}/{method}: {err}", flush=True)
        base = dict(exp=exp, problem=kind, cfg=cfg_str(cfg), seed=seed,
                    method=method, theta=(theta or ""), error=err.splitlines()[-1])
        return base
    wall = time.perf_counter() - t_start
    row = dict(exp=exp, problem=kind, cfg=cfg_str(cfg), seed=seed,
               method=method, theta=(theta if theta is not None else ""),
               iters=s["iters"], phase1_iters=s.get("phase1_iters", ""),
               phase2_iters=s.get("phase2_iters", ""),
               fevals=s["fevals"], gevals=s["gevals"], hevals=s["hevals"],
               hvevals=s["hvevals"], time_s=round(s["time_s"], 3),
               gnorm=s["gnorm"], fval=s["fval"],
               converged=int(bool(s["converged"])),
               switched=int(bool(s.get("switched", False))),
               switched_at_gnorm=s.get("switched_at_gnorm") or "",
               probes_failed=s.get("probes_failed", 0),
               kappa_hat=s.get("kappa_hat") or "",
               c1_est=s.get("c1_est") or "", c2_est=s.get("c2_est") or "",
               error="", history=s.get("history"))
    print(f"  [{wall:7.2f}s] {exp} {kind} {cfg_str(cfg)} {method}"
          f"{' th=' + str(theta) if theta is not None else ''} seed={seed}"
          f" -> it={row['iters']} t={row['time_s']}s conv={row['converged']}",
          flush=True)
    return row


def cfg_str(cfg):
    parts = [f"{k}{v}" for k, v in sorted(cfg.items())]
    return "-".join(parts) if parts else "default"


# ---------------------------------------------------------------- job lists -

def build_jobs(smoke=False):
    """Yield job dicts in execution order."""
    J = []

    def add(exp, kind, cfg, seed, method, theta=None, eps=1e-8, mt=120.0):
        J.append(dict(exp=exp, kind=kind, cfg=cfg, seed=seed, method=method,
                      theta=theta, eps=eps, mt=mt))

    if smoke:
        quads = [(50, 100.0)]
        fo = ["GD", "NAG-t", "NAG-CM", "L-BFGS", "Catalyst",
              ("Hybrid-fixed", 1e-3), "Hybrid-adaptive"]
        so = ["Newton", "TR", "ARC"]
        for n, kap in quads:
            for meth in fo + so:
                nm, th = (meth if isinstance(meth, tuple) else (meth, None))
                for sd in (0,):
                    add("E1", "quadratic", dict(n=n, kappa=kap), sd, nm,
                        theta=th, eps=1e-6, mt=25.0)
        for th in (1e-1, 1e-3, 1e-5):
            add("E2", "quadratic", dict(n=50, kappa=100.0), 0, "Hybrid-fixed",
                theta=th, eps=1e-6, mt=25.0)
        add("E2", "quadratic", dict(n=50, kappa=100.0), 0, "Hybrid-adaptive",
            eps=1e-6, mt=25.0)
        add("E4", "rosenbrock", dict(n=10), 0, "NAG-t", eps=1e-6, mt=25.0)
        add("E4", "rosenbrock", dict(n=10), 0, "Newton", eps=1e-6, mt=25.0)
        add("E4", "rosenbrock", dict(n=10), 0, "Hybrid-fixed", theta=1e-3,
            eps=1e-6, mt=25.0)
        add("E4", "logsumexp", dict(n=20, m=100), 0, "NAG-t", eps=1e-6, mt=25.0)
        add("E4", "logsumexp", dict(n=20, m=100), 0, "Hybrid-fixed", theta=1e-3,
            eps=1e-6, mt=25.0)
        add("E5", "logistic", dict(n=50, m=200, lam=0.01), 0, "NAG-t",
            eps=1e-6, mt=25.0)
        add("E5", "logistic", dict(n=50, m=200, lam=0.01), 0, "Newton",
            eps=1e-6, mt=25.0)
        add("E5", "logistic", dict(n=50, m=200, lam=0.01), 0, "Hybrid-adaptive",
            eps=1e-6, mt=25.0)
        return J

    # ---------------- full suite ----------------
    all_meth_fo = ["GD", "NAG-t", "NAG-CM", "L-BFGS", "Catalyst",
                   ("Hybrid-fixed", 1e-3), "Hybrid-adaptive"]
    all_meth_so = ["Newton", "TR", "ARC"]

    # E1 validation suite on quadratics
    for n in (100, 1000):
        for kap in (1e2, 1e3, 1e4):
            for sd in (0, 1, 2):
                for meth in all_meth_fo + all_meth_so:
                    nm, th = meth if isinstance(meth, tuple) else (meth, None)
                    add("E1", "quadratic", dict(n=n, kappa=kap), sd, nm,
                        theta=th, eps=1e-8, mt=90.0)

    # E2 threshold sensitivity U-curve (+E3 theory anchors via adaptive runs)
    e2_probs = [
        ("quadratic", dict(n=1000, kappa=1e3)),
        ("logistic", dict(n=200, m=2000, lam=0.01)),
        ("logsumexp", dict(n=50, m=300)),
    ]
    theta_grid = list(np.geomspace(1e-1, 1e-8, 12))
    for kind, cfg in e2_probs:
        for th in theta_grid:
            add("E2", kind, cfg, 0, "Hybrid-fixed", theta=float(th),
                eps=1e-8, mt=180.0)
        add("E3", kind, cfg, 0, "Hybrid-adaptive", eps=1e-8, mt=300.0)

    # E4 nonconvex / non-quadratic
    for n_rb in (50, 100):
        for sd in (0, 1):
            for meth in ["NAG-t", "Newton", "L-BFGS", "TR", "ARC",
                         ("Hybrid-fixed", 1e-3), "Hybrid-adaptive"]:
                nm, th = meth if isinstance(meth, tuple) else (meth, None)
                add("E4", "rosenbrock", dict(n=n_rb), sd, nm, theta=th,
                    eps=1e-8, mt=150.0)
    for sd in (0, 1):
        for meth in ["NAG-t", "Newton", "TR", "ARC",
                     ("Hybrid-fixed", 1e-3), "Hybrid-adaptive"]:
            nm, th = meth if isinstance(meth, tuple) else (meth, None)
            add("E4", "logsumexp", dict(n=50, m=500), sd, nm, theta=th,
                eps=1e-8, mt=150.0)

    # E5 logistic regression at scale
    for sd in (0, 1, 2):
        for meth in all_meth_fo + all_meth_so:
            nm, th = meth if isinstance(meth, tuple) else (meth, None)
            add("E5", "logistic", dict(n=1000, m=5000, lam=0.01), sd, nm,
                theta=th, eps=1e-8, mt=300.0)

    return J


# -------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exp", action="append", default=[],
                    help="restrict to given stage(s), e.g. --exp E1")
    ap.add_argument("--budget-min", type=float, default=110.0)
    args = ap.parse_args()

    _ensure_dirs()
    done = load_done()
    jobs = build_jobs(args.smoke)
    if args.exp:
        jobs = [j for j in jobs if j["exp"] in set(args.exp)]
    todo = [j for j in jobs if _key(j["exp"], j["kind"], cfg_str(j["cfg"]),
                                    j["method"],
                                    str(j["theta"] if j["theta"] is not None
                                        else ""),
                                    j["seed"]) not in done]
    total = len(jobs)
    print(f"[run_all] jobs={total} done={total-len(todo)} todo={len(todo)} "
          f"smoke={args.smoke}", flush=True)
    t_all = time.perf_counter()
    started_new = 0
    for i, j in enumerate(todo, 1):
        elapsed = time.perf_counter() - t_all
        remaining_known = elapsed / max(started_new, 1) * (len(todo) - i + 1) \
            if started_new else 0.0
        if started_new and elapsed + remaining_known > args.budget_min * 60 \
                and i > 1:
            print(f"[run_all] budget guard hit ({elapsed/60:.1f} min used); "
                  f"stopping early at job {i}/{len(todo)}", flush=True)
            break
        row = execute(**j)
        save_result(row)
        started_new += 1
    mins = (time.perf_counter() - t_all) / 60.0
    print(f"[run_all] finished session: {started_new} runs in {mins:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
