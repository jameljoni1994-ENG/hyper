"""Paper statistics dump: aggregates all_runs_v2.csv into LaTeX-ready tables."""
import csv
import statistics as st
from collections import defaultdict

PATH = r"C:\Users\Windows.11\Desktop\hyper\results\all_runs_v2.csv"
rows = list(csv.DictReader(open(PATH, encoding="utf-8")))


def med(rs):
    v = [float(r["time_s"]) for r in rs if r.get("time_s")]
    return st.median(v) if v else float("nan")


def label(r):
    return r["method"] + ("@" + r["theta"] if r["theta"] else "")


print("=" * 70)
print("E7  median time_s by problem x theta  (Hybrid-fixed)")
print("=" * 70)
by = defaultdict(dict)
for r in rows:
    if r["exp"] == "E7" and r["method"] == "Hybrid-fixed":
        key = {"a9a": "a9a (real)",
               "logistic": "logistic lam=.01 m2000-n200",
               "logsumexp": "logsumexp m300-n50",
               "quadratic": "quadratic kappa=1e3 n=1000"}[r["problem"]]
        by[key].setdefault(float(r["theta"]), []).append(r)
for p, d in sorted(by.items()):
    print("\n" + p)
    best = min(st.median([float(x["time_s"]) for x in v]) for v in d.values())
    for th in sorted(d):
        ts = [float(x["time_s"]) for x in d[th]]
        cv = sum(int(x["converged"]) for x in d[th])
        ratio = st.median(ts) / best if best > 0 else float("nan")
        print(f"  theta={th:>10.2e} conv={cv:>2}/10 med={st.median(ts):8.3f}s "
              f"x_best={ratio:6.2f}")

print()
print("=" * 70)
print("E8  median time_s by family x method")
print("=" * 70)


def fam_of(r):
    p, c = r["problem"], r["cfg"]
    if p == "quadratic":
        kap = float(c.split("kappa")[1].split("-")[0])
        return f"quad k={kap:.0e}"
    return p


groups = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["exp"] == "E8":
        groups[fam_of(r)][label(r)].append(r)

order = ["GD", "NAG-t", "NAG-CM", "L-BFGS", "Catalyst", "Newton",
         "Newton-CG", "TR", "ARC", "Hybrid-fixed@0.001",
         "Hybrid-adaptive", "RH", "RH-BB"]
for fam in sorted(groups):
    print("\n" + fam)
    g = groups[fam]
    for m in order:
        if m in g:
            rs = g[m]
            cv = sum(int(x["converged"]) for x in rs)
            sw = sum(int(x["switched"]) for x in rs) if "switched" in rs[0] else "-"
            print(f"  {m:<22} n={len(rs):>3} conv={cv:>3} sw={sw:>3} "
                  f"med={med(rs):8.3f}s")
    for m in sorted(set(g) - set(order)):
        rs = g[m]
        print(f"  {m:<22} n={len(rs):>3} med={med(rs):8.3f}s")

print()
print("=" * 70)
print("E6  (partial)  median time_s by dataset x method")
print("=" * 70)
g6 = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["exp"] == "E6":
        g6[r["problem"]][label(r)].append(r)
for ds in sorted(g6):
    print("\n" + ds)
    g = g6[ds]
    for m in order:
        if m in g:
            rs = g[m]
            cv = sum(int(x["converged"]) for x in rs)
            print(f"  {m:<22} n={len(rs):>3} conv={cv:>3} med={med(rs):8.3f}s")

print()
print("=" * 70)
print("RH spot facts (any exp)")
print("=" * 70)
rh = [r for r in rows if r["method"] in ("RH", "RH-BB")]
print(f"RH-family rows present: {len(rh)}")
for r in rh[:40]:
    print(f"  {r['exp']} {r['problem']:<11} {r['cfg']:<24} seed={r['seed']} "
          f"{r['method']:<5} it={r['iters']:>7} t={r['time_s']:>7}s "
          f"conv={r['converged']} sw={r['switched']} "
          f"sw@={r['switched_at_gnorm']}")
