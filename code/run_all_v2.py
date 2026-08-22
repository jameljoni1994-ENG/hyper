"""V2 experiment suite: E6 (real data), E7 (safe-region U-curves), E8 (head-to-head).

Writes to results/all_runs_v2.csv + results/hist_v2/ -- never touches v1 files.
Resume-safe via unique (exp,problem,cfg,seed,method,theta) keys.

Usage:
    python run_all_v2.py --smoke          # sanity pass (<3 min)
    python run_all_v2.py --exp E7         # selected stage(s)
    python run_all_v2.py                  # everything (~budget-guarded)
"""
import argparse
import csv
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import make_problem                       # noqa: E402
from problems_real import LibsvmLogistic, DATASETS      # noqa: E402
import methods as M                                     # noqa: E402
from hybrid import hybrid_fixed, hybrid_adaptive        # noqa: E402
from hybrid_v2 import rate_hybrid                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
HISTDIR = os.path.join(RESULTS, "hist_v2")
DATADIR = os.path.join(ROOT, "data")
CSV_PATH = os.path.join(RESULTS, "all_runs_v2.csv")

FIELDS = ["exp", "problem", "cfg", "seed", "method", "theta",
          "iters", "phase1_iters", "phase2_iters",
          "fevals", "gevals", "hevals", "hvevals",
          "time_s", "gnorm", "fval", "converged", "switched",
          "switched_at_gnorm", "probes_failed", "rho_hat",
          "decisions", "probes_fired", "error"]


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
        np.savez_compressed(os.path.join(HISTDIR, k.replace("|", "__")
                                         + ".npz"), tg=arr)


def cfg_str(cfg):
    parts = [f"{k}{v}" for k, v in sorted(cfg.items())]
    return "-".join(parts) if parts else "default"


def make_problem_v2(kind, **kw):
    """Extends make_problem with real LIBSVM datasets."""
    if kind in DATASETS:
        nf = DATASETS[kind][0]
        return LibsvmLogistic(DATADIR, kind, nf)
    return make_problem(kind, **kw)


def initial_point(problem_kind, n, seed):
    if problem_kind == "rosenbrock":
        return np.where(np.arange(n) % 2 == 0, -1.2, 1.0)
    rng = np.random.default_rng(9000 + seed)
    return rng.standard_normal(n) * (0.01 if problem_kind != "quadratic" else 1.0)


SOLVERS = {
    "GD": lambda pb, x0, eps, mt: M.gd(pb, x0, eps=eps, max_time=mt),
    "NAG-t": lambda pb, x0, eps, mt: M.nag_t(pb, x0, eps=eps, max_time=mt),
    "NAG-CM": lambda pb, x0, eps, mt: M.nag_cm(
        pb, x0, kappa=(pb.L / pb.mu if (pb.L and getattr(pb, "mu", 0)
                                        and pb.mu > 0) else None),
        eps=eps, max_time=mt),
    "Newton": lambda pb, x0, eps, mt: M.newton_ls(pb, x0, eps=eps,
                                                  max_time=mt),
    "Newton-CG": lambda pb, x0, eps, mt: M.newton_cg(pb, x0, eps=eps,
                                                     max_time=mt),
    "L-BFGS": lambda pb, x0, eps, mt: M.lbfgs(pb, x0, eps=eps, max_time=mt),
    "TR": lambda pb, x0, eps, mt: M.tr_steihaug(pb, x0, eps=eps,
                                                max_time=mt),
    "ARC": lambda pb, x0, eps, mt: M.arc(pb, x0, eps=eps, max_time=mt),
    "Catalyst": lambda pb, x0, eps, mt: M.catalyst(pb, x0, eps=eps,
                                                   max_time=mt),
}


def execute(exp, kind, cfg, seed, method, theta=None, eps=1e-8, mt=120.0):
    pb = make_problem_v2(kind, **cfg)
    x0 = initial_point(kind, pb.n, seed)
    t_start = time.perf_counter()
    try:
        if method == "Hybrid-fixed":
            s = hybrid_fixed(pb, x0, theta=theta, eps=eps, max_time=mt)
        elif method == "Hybrid-adaptive":
            s = hybrid_adaptive(pb, x0, eps=eps, max_time=min(mt, 900.0))
        elif method == "RH":
            s = rate_hybrid(pb, x0, eps=eps, max_time=min(mt, 900.0))
        else:
            s = SOLVERS[method](pb, x0, eps, mt)
    except Exception:
        err = traceback.format_exc(limit=3)
        print(f"  !! FAIL {exp}/{kind}/{cfg_str(cfg)}/{method}: "
              f"{err.splitlines()[-1]}", flush=True)
        return dict(exp=exp, problem=kind, cfg=cfg_str(cfg), seed=seed,
                    method=method, theta=(theta or ""),
                    error=err.splitlines()[-1])
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
               rho_hat=s.get("rho_hat") if s.get("rho_hat") is not None else "",
               decisions=s.get("decisions", ""),
               probes_fired=s.get("probes_fired", ""),
               error="", history=s.get("history"))
    print(f"  [{time.perf_counter() - t_start:7.2f}s] {exp} {kind} "
          f"{cfg_str(cfg)} {method}"
          f"{' th=' + str(theta) if theta is not None else ''} seed={seed}"
          f" -> it={row['iters']} t={row['time_s']}s conv={row['converged']}",
          flush=True)
    return row


# ---------------------------------------------------------------- job lists --

def build_jobs(smoke=False):
    J = []

    def add(exp, kind, cfg, seed, method, theta=None, eps=1e-8, mt=120.0):
        J.append(dict(exp=exp, kind=kind, cfg=cfg, seed=seed, method=method,
                      theta=theta, eps=eps, mt=mt))

    e8_methods = ["NAG-CM", "L-BFGS", "Newton", "Newton-CG", "TR",
                  ("Hybrid-fixed", 1e-3), "Hybrid-adaptive", "RH"]

    if smoke:
        add("E6", "a9a", {}, 0, "NAG-CM", eps=1e-6, mt=25.0)
        add("E6", "a9a", {}, 0, "RH", eps=1e-6, mt=25.0)
        add("E6", "mushrooms", {}, 0, "Newton-CG", eps=1e-6, mt=25.0)
        add("E7", "quadratic", dict(n=200, kappa=1e3), 0,
            "Hybrid-fixed", theta=1e-3, eps=1e-6, mt=25.0)
        add("E8", "rosenbrock", dict(n=50), 0, "RH", eps=1e-6, mt=25.0)
        add("E8", "logistic", dict(n=500, m=3000, lam=0.01), 0, "NAG-CM",
            eps=1e-6, mt=25.0)
        return J

    # ---- E6: real LIBSVM data, 10 seeds -------------------------------
    e6_methods = ["GD", "NAG-t", "NAG-CM", "L-BFGS", "Catalyst",
                  "Newton", "Newton-CG", "TR", "ARC",
                  ("Hybrid-fixed", 1e-3), "Hybrid-adaptive", "RH"]
    for ds in ("a9a", "mushrooms", "w8a"):
        for sd in range(10):
            for meth in e6_methods:
                nm, th = meth if isinstance(meth, tuple) else (meth, None)
                add("E6", ds, {}, sd, nm, theta=th, eps=1e-8, mt=120.0)

    # ---- E7: extended-threshold safe region, 10 seeds ------------------
    e7_probs = [
        ("quadratic", dict(n=1000, kappa=1e3)),
        ("logistic", dict(n=200, m=2000, lam=0.01)),
        ("logsumexp", dict(n=50, m=300)),
        ("a9a", {}),
    ]
    theta_grid = [0.5, 0.3, 0.2, 0.1] + \
        list(np.geomspace(1e-2, 1e-8, 11))
    for kind, cfg in e7_probs:
        for th in theta_grid:
            for sd in range(10):
                add("E7", kind, cfg, sd, "Hybrid-fixed", theta=float(th),
                    eps=1e-8, mt=180.0)

    # ---- E8: head-to-head with 10 seeds ---------------------------------
    e1_cfgs = [dict(n=n, kappa=k) for n in (100, 1000)
               for k in (1e2, 1e3, 1e4)]
    for cfg in e1_cfgs:
        for sd in range(10):
            for meth in e8_methods:
                nm, th = meth if isinstance(meth, tuple) else (meth, None)
                add("E8", "quadratic", cfg, sd, nm, theta=th, eps=1e-8,
                    mt=90.0)
    for cfg in (dict(n=50), dict(n=100)):
        for sd in range(10):
            for meth in e8_methods:
                nm, th = meth if isinstance(meth, tuple) else (meth, None)
                add("E8", "rosenbrock", cfg, sd, nm, theta=th, eps=1e-8,
                    mt=150.0)
    for sd in range(10):
        for meth in e8_methods:
            nm, th = meth if isinstance(meth, tuple) else (meth, None)
            add("E8", "logsumexp", dict(n=50, m=500), sd, nm, theta=th,
                eps=1e-8, mt=150.0)
    for sd in range(10):
        for meth in e8_methods:
            nm, th = meth if isinstance(meth, tuple) else (meth, None)
            add("E8", "logistic", dict(n=1000, m=5000, lam=0.01), sd,
                nm, theta=th, eps=1e-8, mt=300.0)

    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exp", action="append", default=[])
    ap.add_argument("--budget-min", type=float, default=110.0)
    args = ap.parse_args()

    _ensure_dirs()
    done = load_done()
    jobs = build_jobs(args.smoke)
    if args.exp:
        keep = set(args.exp)
        jobs = [j for j in jobs if j["exp"] in keep]
    todo = [j for j in jobs if _key(j["exp"], j["kind"], cfg_str(j["cfg"]),
                                    j["method"],
                                    str(j["theta"] if j["theta"] is not None
                                        else ""), j["seed"]) not in done]
    total = len(jobs)
    print(f"[run_all_v2] jobs={total} done={total - len(todo)} "
          f"todo={len(todo)} smoke={args.smoke}", flush=True)
    t_all = time.perf_counter()
    started_new = 0
    for i, j in enumerate(todo, 1):
        elapsed = time.perf_counter() - t_all
        est_per_job = elapsed / started_new if started_new else 0.0
        if started_new and elapsed + est_per_job > args.budget_min * 60:
            print(f"[run_all_v2] budget guard hit ({elapsed / 60:.1f} min); "
                  f"stopping early at job {i}/{len(todo)}", flush=True)
            break
        row = execute(**j)
        save_result(row)
        started_new += 1
    mins = (time.perf_counter() - t_all) / 60.0
    print(f"[run_all_v2] finished session: {started_new} runs in "
          f"{mins:.1f} min", flush=True)


if __name__ == "__main__":
    main()
