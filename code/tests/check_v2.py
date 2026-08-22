"""Quick functional test of the v2 additions: newton_cg and rate_hybrid."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems import Quadratic, Rosenbrock, Logistic, LogSumExp   # noqa: E402
from methods import newton_ls, newton_cg                          # noqa: E402
from hybrid import hybrid_fixed                                   # noqa: E402
from hybrid_v2 import rate_hybrid                                 # noqa: E402

PY = sys.version_info
ok = True


def report(name, cond, detail):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    ok = ok and bool(cond)


def run_one(solver, pb, x0, eps=1e-8, mt=60.0, **kw):
    t0 = time.perf_counter()
    s = solver(pb, x0, eps=eps, max_time=mt, **kw)
    return s, time.perf_counter() - t0


# ------------------------------------------------------------- Newton-CG ----
pb = Quadratic(n=200, kappa=1e3)
rng = np.random.default_rng(3)
x0 = rng.standard_normal(pb.n)
s_ncg, _ = run_one(newton_cg, pb, x0)
s_nls, _ = run_one(newton_ls, pb, x0)
report("newton_cg converges on quadratic", s_ncg["converged"],
       f"(gn={s_ncg['gnorm']:.2e}, hv={s_ncg['hvevals']}, "
       f"t={s_ncg['time_s']:.3f}s)")
report("newton_ls comparable iters on quadratic",
       s_nls["iters"] <= 5 and s_ncg["iters"] <= 30,
       f"(dense it={s_nls['iters']}, cg it={s_ncg['iters']})")

pbl = Logistic(n=100, m=500, lam=0.01)
x0 = rng.standard_normal(pbl.n) * 0.01
s_ncg, _ = run_one(newton_cg, pbl, x0)
report("newton_cg converges on logistic", s_ncg["converged"],
       f"(gn={s_ncg['gnorm']:.2e}, hv={s_ncg['hvevals']})")

# ------------------------------------------------------------------- RH -----
cases = [
    ("quadratic-k1000-n200", Quadratic(n=200, kappa=1e3), rng.standard_normal(200)),
    ("logistic-100x500", Logistic(n=100, m=500, lam=0.01), None),
    ("logsumexp-50x300", LogSumExp(n=50, m=300),
     np.random.default_rng(7).standard_normal(50)),
]
for nm, pb, x0 in cases:
    if x0 is None:
        x0 = np.random.default_rng(5).standard_normal(pb.n) * 0.01
    s_rh, _ = run_one(rate_hybrid, pb, x0)
    s_hf, _ = run_one(hybrid_fixed, pb, x0, theta=1e-3)
    print(f"  RH  {nm:<24} conv={int(bool(s_rh['converged']))} "
          f"sw={int(bool(s_rh['switched']))} it={s_rh['iters']:>6} "
          f"t={s_rh['time_s']:.3f}s rho={s_rh['rho_hat']} "
          f"c2={s_rh['c2_est']}")
    print(f"  HF  {nm:<24} conv={int(bool(s_hf['converged']))} "
          f"it={s_hf['iters']:>6} t={s_hf['time_s']:.3f}s")
    report(f"RH converges on {nm}", s_rh["converged"],
           f"(gn={s_rh['gnorm']:.2e})")

# RH must switch early on a quadratic (Prop. 2: economics favor it)
pbq = Quadratic(n=300, kappa=1e3)
x0 = np.random.default_rng(11).standard_normal(pbq.n)
s_rh, _ = run_one(rate_hybrid, pbq, x0)
report("RH switches on quadratic (early probe wins)", s_rh["switched"],
       f"(phase1={s_rh['phase1_iters']}, phase2={s_rh['phase2_iters']})")

# ------------------------------------------------- NAG-AR / RH on Rosenbrock --
from methods import nag_adaptive                                    # noqa: E402

pbr = Rosenbrock(n=50)
x0 = np.where(np.arange(pbr.n) % 2 == 0, -1.2, 1.0)
s_ar, _ = run_one(nag_adaptive, pbr, x0, mt=30.0)
s_bb, _ = run_one(lambda pb, xx, eps, max_time: rate_hybrid(
    pb, xx, eps=eps, max_time=max_time, fo_mode="bb"),
    pbr, x0, mt=30.0)
print(f"  NAG-AR rosenbrock-n50  conv={int(bool(s_ar['converged']))} "
      f"it={s_ar['iters']:>6} t={s_ar['time_s']:.1f}s gn={s_ar['gnorm']:.2e}")
print(f"  RH-bb  rosenbrock-n50  conv={int(bool(s_bb['converged']))} "
      f"it={s_bb['phase1_iters'] + s_bb['phase2_iters']:>6} "
      f"t={s_bb['time_s']:.1f}s gn={s_bb['gnorm']:.2e}")
report("NAG-AR beats BB-style phase 1 on Rosenbrock",
       s_ar["gnorm"] < s_bb["gnorm"],
       f"(AR gn={s_ar['gnorm']:.2e} vs BB gn={s_bb['gnorm']:.2e})")
s_rrh, _ = run_one(rate_hybrid, pbr, x0, mt=45.0)
print(f"  RH     rosenbrock-n50  conv={int(bool(s_rrh['converged']))} "
      f"sw={int(bool(s_rrh['switched']))} "
      f"p1={s_rrh['phase1_iters']} p2={s_rrh['phase2_iters']} "
      f"t={s_rrh['time_s']:.1f}s gn={s_rrh['gnorm']:.2e}")
report("RH(auto) converges on Rosenbrock", s_rrh["converged"],
       f"(gn={s_rrh['gnorm']:.2e})")

print()
print("ALL OK" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
