"""Publication-quality figures + summary tables from results/all_runs.csv.

Outputs into results/figs/:
  pp_<exp>.png        Dolan-More performance profiles, two panels:
                      wall time (floored at 0.1 ms) and gradient evaluations
  conv_*.png          loglog convergence curves from hist/*.npz traces
  E2_ucurve.png       Hybrid-fixed wall time vs theta (failures kept, marked)
  tables.md           per-experiment summary tables
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = r"C:\Users\Windows.11\Desktop\New folder\results"
FIG = os.path.join(RES, "figs")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.35, "lines.linewidth": 1.6,
})

METHODS = ["GD", "NAG-t", "NAG-CM", "L-BFGS", "Catalyst",
           "Hybrid-fixed", "Hybrid-adaptive", "Newton", "TR", "ARC"]
STYLE = dict(zip(METHODS, [
    ("#7f7f7f", "o"), ("#1f77b4", "s"), ("#aec7e8", "D"),
    ("#2ca02c", "v"), ("#17becf", "^"), ("#d62728", "P"),
    ("#ff7f0e", "X"), ("#9467bd", "p"), ("#8c564b", "<"), ("#e377c2", ">"),
]))
T_FLOOR = 1e-4          # timer floor: ratios involving <0.1 ms are noise

df = pd.read_csv(os.path.join(RES, "all_runs.csv"),
                 dtype={"theta": str}, keep_default_na=False, na_values=[""],
                 encoding="utf-8-sig")
df["seed"] = df["seed"].astype(int)
df["converged"] = df["converged"].astype(bool)


def styl(m):
    c, mk = STYLE.get(m.split("(")[0], ("k", "."))
    return c, mk


# ------------------------------------------------- performance profiles -----
def _instance_matrix(sub, metric):
    """rows=instances, cols=methods; inf when failed/missing."""
    methods = [m for m in METHODS if m in set(sub["method"])]
    rows = []
    for _, g in sub.groupby(["problem", "cfg", "seed"], observed=True):
        rec = {}
        for m in methods:
            r = g[g["method"] == m]
            ok = len(r) == 1 and bool(r["converged"].iloc[0])
            rec[m] = float(r[metric].iloc[0]) if ok else np.inf
        if any(np.isfinite(v) for v in rec.values()):
            rows.append(rec)
    return methods, pd.DataFrame(rows)


def perf_profile(sub, title, fname):
    methods_t, T = _instance_matrix(sub, "time_s")
    methods_g, G = _instance_matrix(sub, "gevals")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1),
                             constrained_layout=True)
    for ax, M, metric, lab in [(axes[0], T, "time_s", "wall time (s)"),
                               (axes[1], G, "gevals",
                                r"gradient evaluations")]:
        vals = M.clip(lower=T_FLOOR if metric == "time_s" else 1)
        best = vals.min(axis=1)
        R = vals.div(best, axis=0)
        finite = R.replace([np.inf, -np.inf], np.nan).to_numpy().ravel()
        tau_max = float(np.nanmax(finite)) * 1.3 if np.isfinite(finite).any() else 10.0
        tau_max = float(np.clip(tau_max, 10, 2e5))
        tau_max = 10 ** np.ceil(np.log10(tau_max))   # round up to a decade
        taus = np.logspace(0, np.log10(tau_max), 240)
        n_inst = len(M)
        for m in methods_t:
            rho = [(R[m] <= t).mean() for t in taus]
            c, mk = styl(m)
            ax.step(taus, rho, where="post", label=m, color=c)
            k = max(len(taus) // 8, 1)
            ax.plot(taus[::k], rho[::k], mk, ms=4, color=c, alpha=.9)
        ax.set_xscale("log")
        ax.set_xlim(1, tau_max)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(r"$\tau$  (ratio to best method)")
        ax.set_ylabel(r"$\rho(\tau)$   [%d instances]" % n_inst)
        ax.set_title(lab)
        ax.axvline(1.0, color="k", lw=.6, alpha=.5)
        ax.axhline(1.0, color="k", lw=.6, alpha=.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=5,
               frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.savefig(os.path.join(FIG, fname), dpi=200)
    plt.close(fig)


# ------------------------------------------------------ convergence plots ---
def hist_path(exp, problem, cfg, method, theta, seed):
    th = "" if theta in (None, "") else theta
    key = "|".join([exp, problem, cfg, method, th, str(seed)])
    return os.path.join(RES, "hist", key.replace("|", "__") + ".npz")


def conv_plot(items, title, fname):
    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    for label, path in items:
        if not os.path.exists(path):
            print("  [missing hist]", os.path.basename(path))
            continue
        tg = np.load(path)["tg"]
        t, g = tg[:, 0], tg[:, 1]
        m = g > 0
        t, g = t[m], g[m]
        if len(t) == 0:
            continue
        c, mk = styl(label)
        ls = "--" if ("Hybrid" in label or label == "Newton") else "-"
        me = max(len(t) // 14, 1)
        ax.loglog(t, g, ls, color=c,
                  marker=mk, ms=3.5, markevery=(me // 2, me))
    ax.axhline(1e-8, color="k", lw=.9, alpha=.6)
    ax.annotate(r"$\varepsilon=10^{-8}$", xy=(0.99, 1.6e-8),
                xycoords=("axes fraction", "data"), ha="right", fontsize=8)
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel(r"$\|\nabla f(x)\|$")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="best", framealpha=0.85)
    fig.savefig(os.path.join(FIG, fname), dpi=200)
    plt.close(fig)


e1 = df[df["exp"] == "E1"]
perf_profile(e1[e1["cfg"].str.endswith("n100")],
             "E1a  quadratics $n{=}100$: "
             "$\\kappa\\in\\{10^2,10^3,10^4\\}$, 3 seeds", "pp_E1a_n100.png")
perf_profile(e1[e1["cfg"].str.endswith("n1000")],
             "E1b  quadratics $n{=}1000$: "
             "$\\kappa\\in\\{10^2,10^3,10^4\\}$, 3 seeds", "pp_E1b_n1000.png")
os.remove(os.path.join(FIG, "pp_E1.png"))
e4 = df[df["exp"] == "E4"]
perf_profile(e4[e4["problem"] == "rosenbrock"],
             "E4a  Rosenbrock $n\\in\\{50,100\\}$", "pp_E4a_rosenbrock.png")
perf_profile(e4[e4["problem"] == "logsumexp"],
             "E4b  LogSumExp $m{=}500$, $n{=}50$ (degenerate Hessian)",
             "pp_E4b_logsumexp.png")
perf_profile(df[df["exp"] == "E5"],
             "E5  logistic regression $n{=}1000$, $m{=}5000$", "pp_E5.png")

conv_plot(
    [("GD", hist_path("E1", "quadratic", "kappa1000.0-n100", "GD", None, 0)),
     ("NAG-t", hist_path("E1", "quadratic", "kappa1000.0-n100", "NAG-t", None, 0)),
     ("NAG-CM", hist_path("E1", "quadratic", "kappa1000.0-n100", "NAG-CM", None, 0)),
     ("L-BFGS", hist_path("E1", "quadratic", "kappa1000.0-n100", "L-BFGS", None, 0)),
     ("Catalyst", hist_path("E1", "quadratic", "kappa1000.0-n100", "Catalyst", None, 0)),
     ("Newton", hist_path("E1", "quadratic", "kappa1000.0-n100", "Newton", None, 0)),
     ("TR", hist_path("E1", "quadratic", "kappa1000.0-n100", "TR", None, 0)),
     ("ARC", hist_path("E1", "quadratic", "kappa1000.0-n100", "ARC", None, 0)),
     ("Hybrid-fixed", hist_path("E1", "quadratic", "kappa1000.0-n100", "Hybrid-fixed", "0.001", 0)),
     ("Hybrid-adaptive", hist_path("E1", "quadratic", "kappa1000.0-n100", "Hybrid-adaptive", None, 0))],
    "E1  quadratic $\\kappa{=}10^3$, $n{=}100$, seed 0",
    "conv_E1_quad.png")

conv_plot(
    [("NAG-t", hist_path("E4", "rosenbrock", "n100", "NAG-t", None, 0)),
     ("L-BFGS", hist_path("E4", "rosenbrock", "n100", "L-BFGS", None, 0)),
     ("Newton", hist_path("E4", "rosenbrock", "n100", "Newton", None, 0)),
     ("TR", hist_path("E4", "rosenbrock", "n100", "TR", None, 0)),
     ("ARC", hist_path("E4", "rosenbrock", "n100", "ARC", None, 0)),
     ("Hybrid-fixed", hist_path("E4", "rosenbrock", "n100", "Hybrid-fixed", "0.001", 0)),
     ("Hybrid-adaptive", hist_path("E4", "rosenbrock", "n100", "Hybrid-adaptive", None, 0))],
    "E4a  Rosenbrock $n{=}100$", "conv_E4_rosenbrock_n100.png")

conv_plot(
    [("NAG-t", hist_path("E4", "logsumexp", "m500-n50", "NAG-t", None, 0)),
     ("Newton", hist_path("E4", "logsumexp", "m500-n50", "Newton", None, 0)),
     ("TR", hist_path("E4", "logsumexp", "m500-n50", "TR", None, 0)),
     ("ARC", hist_path("E4", "logsumexp", "m500-n50", "ARC", None, 0)),
     ("Hybrid-fixed", hist_path("E4", "logsumexp", "m500-n50", "Hybrid-fixed", "0.001", 0)),
     ("Hybrid-adaptive", hist_path("E4", "logsumexp", "m500-n50", "Hybrid-adaptive", None, 0))],
    "E4b  LogSumExp $m{=}500$ (rank-deficient Hessian)",
    "conv_E4_logsumexp_m500.png")

# ------------------------------------------------------------ E2 U-curve ----
e2 = df[df["exp"] == "E2"].copy()
e2["theta_f"] = e2["theta"].astype(float)
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9),
                         constrained_layout=True)
for ax, (prob, lab) in zip(axes, [
        ("quadratic", "quadratic $\\kappa{=}10^3,n{=}1000$"),
        ("logistic", "logistic $n{=}200,m{=}2000$"),
        ("logsumexp", "LogSumExp $n{=}50,m{=}300$")]):
    g = e2[e2["problem"] == prob]
    med = g.groupby("theta_f").agg(t=("time_s", "median"),
                                   ok=("converged", "mean"))
    ax.plot(med.index, med["t"], "-", color="#1f77b4", zorder=2)
    ax.plot(med.index, med["t"], "o", ms=4, color="#1f77b4", zorder=3)
    bad = med[med["ok"] < 1]
    if len(bad):
        ax.plot(bad.index, bad["t"], "x", ms=9, mew=2.2,
                color="#d62728", zorder=4, label="some seeds fail")
        ax.legend(loc="upper right", framealpha=.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta$")
    ax.set_title(lab)
axes[0].set_ylabel("wall time (s)  [median]")
fig.suptitle("E2  Hybrid-fixed sensitivity to switching threshold "
             "$\\theta$ ($\\times$ = at least one seed missed $\\varepsilon$)",
             fontsize=10)
fig.savefig(os.path.join(FIG, "E2_ucurve.png"), dpi=200)
plt.close(fig)

# ------------------------------------------------------------- tables -------
def md_table(tab):
    cols = list(tab.columns)
    out = ["| method | " + " | ".join(cols) + " |",
           "|" + "---|" * (len(cols) + 1)]
    for idx, row in tab.iterrows():
        out.append("| %s | " % idx + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


lines = ["# Summary tables\n"]
for exp, sub in df.groupby("exp"):
    lines.append(f"\n## {exp}\n")
    tab = sub.groupby("method").agg(
        runs=("converged", "size"),
        conv_rate=("converged", "mean"),
        med_time=("time_s", "median"),
        mean_time=("time_s", "mean"),
        med_iters=("iters", "median"),
    ).round(3)
    tab["conv_rate"] = (100 * tab["conv_rate"]).round(1)
    lines.append(md_table(tab))
with open(os.path.join(RES, "tables.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))

print("figures written to", FIG)
for f in sorted(glob.glob(os.path.join(FIG, "*"))):
    print("  ", os.path.basename(f), "%.1f KB" % (os.path.getsize(f) / 1024))
