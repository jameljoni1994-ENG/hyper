"""V2 analysis: tables + figures from results/all_runs_v2.csv.

Outputs into results/figs_v2/:
  E7_ucurve_<prob>.png   Hybrid-fixed wall time vs threshold theta
  E6_profile.png         Dolan-More performance profile (wall time), real data
  E8_switch_hist.png     RH switch points vs problem class
and results/tables_v2.md with per-stage summaries.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figs_v2")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.35, "lines.linewidth": 1.6,
})

T_FLOOR = 1e-4

df = pd.read_csv(os.path.join(RES, "all_runs_v2.csv"),
                 dtype={"theta": str}, keep_default_na=False,
                 na_values=[""], encoding="utf-8-sig")
df["seed"] = df["seed"].astype(int)
df["converged"] = df["converged"].astype(bool)

lines = []
def emit(s=""):
    print(s)
    lines.append(s)


# ---------------------------------------------------------------- E7 U-curves
emit("## E7 — safe region: median wall time vs threshold\n")
e7 = df[df.exp == "E7"].copy()
if len(e7):
    e7["theta_f"] = e7["theta"].astype(float)
    fig, axes = plt.subplots(1, e7.problem.nunique(),
                             figsize=(3.1 * e7.problem.nunique(), 2.9),
                             squeeze=False)
    for ax, (prob, gdf) in zip(axes[0], e7.groupby("problem")):
        agg = gdf.groupby("theta_f").agg(
            med_t=("time_s", "median"),
            fail=("converged", lambda s: 1 - s.mean()))
        ax.plot(agg.index, np.maximum(agg.med_t, T_FLOOR), "o-", color="#d62728")
        for th, row in agg.iterrows():
            if row.fail > 0:
                ax.annotate(f"{int(round(100 * row.fail))}%",
                            (th, max(row.med_t, T_FLOOR)),
                            textcoords="offset points", xytext=(0, 5),
                            fontsize=7, color="#d62728")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"threshold $\theta$")
        ax.set_ylabel("median wall time [s]")
        ax.set_title(prob)
        mono = bool(np.all(np.diff(np.maximum(agg.med_t.values, T_FLOOR))
                           >= -1e-12))
        emit(f"- **{prob}**: monotone nonincreasing toward eps-anchor: "
             f"{mono}; failure rates "
             f"{ {round(th, 6): round(f, 2) for th, f in agg.fail.items() if f > 0} }")
    fig.suptitle("E7: no interior optimum on quadratics (Prop. 2); "
                 "mild dips allowed elsewhere", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "E7_ucurves.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
else:
    emit("- (no E7 rows yet)")

# ------------------------------------------------------------- E6 profile ----
emit("\n## E6 — real LIBSVM data\n")
e6 = df[df.exp == "E6"].copy()
if len(e6):
    rows = []
    for (prob, meth), gdf in e6.groupby(["problem", "method"]):
        rows.append(dict(problem=prob, method=meth,
                         med_t=round(float(gdf.time_s.median()), 3),
                         med_it=float(gdf.iters.median()),
                         conv=round(float(gdf.converged.mean()), 2),
                         sw=round(float(gdf.switched.astype(bool).mean()), 2)
                         if gdf.method.iloc[0].startswith(("RH", "Hybrid"))
                         else ""))
    tab = pd.DataFrame(rows).sort_values(["problem", "med_t"])
    emit(tab.to_string(index=False))

    methods_present = sorted(e6.method.unique())
    # performance profile: fraction of problems solved within tau x best time
    grid = np.logspace(0, 3, 60)
    wins = {}
    best = e6.groupby(["problem", "seed"])["time_s"].min()
    key = e6.set_index(["problem", "seed"]).index
    rel = (e6.set_index(["problem", "seed"]).time_s
           / best.reindex(key).values).values
    e6["rel_t"] = np.maximum(rel, T_FLOOR / e6.time_s.clip(lower=T_FLOOR))
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for meth in methods_present:
        r = e6[e6.method == meth]["rel_t"].values
        frac = [(r <= t * np.ones_like(r)).mean() for t in grid]
        ax.plot(grid, frac, label=meth, drawstyle="steps-post")
    ax.set_xscale("log"); ax.set_xlabel(r"$\tau$ (time multiple of best)")
    ax.set_ylabel("fraction of runs solved")
    ax.set_title("E6 performance profile (wall time)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "E6_profile.png"), dpi=200)
    plt.close(fig)
else:
    emit("- (no E6 rows yet)")

# ------------------------------------------------------------------- E8 ------
emit("\n## E8 — head-to-head with 10 seeds\n")
e8 = df[df.exp == "E8"].copy()
if len(e8):
    rows = []
    for (prob, cfg, meth), gdf in e8.groupby(["problem", "cfg", "method"]):
        rows.append(dict(suite=f"{prob}:{cfg}", method=meth,
                         med_t=round(float(gdf.time_s.median()), 3),
                         conv=round(float(gdf.converged.mean()), 2),
                         med_gn=float(gdf.gnorm.median())))
    tab = pd.DataFrame(rows).sort_values(["suite", "med_t"])
    emit(tab.to_string(index=False))

    rh = e8[(e8.method == "RH") & e8.switched.astype(bool)]
    if len(rh):
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        groups = []
        labels = []
        for (prob, cfg), gdf in rh.groupby(["problem", "cfg"]):
            groups.append(np.log10(gdf.switched_at_gnorm.astype(float)))
            labels.append(f"{prob}\n{cfg}")
        ax.boxplot(groups, tick_labels=labels)
        ax.set_ylabel(r"$\log_{10}\|\nabla f\|$ at switch")
        ax.set_title("RH realized switch points")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "E8_switch_hist.png"), dpi=200)
        plt.close(fig)
else:
    emit("- (no E8 rows yet)")

with open(os.path.join(RES, "tables_v2.md"), "w", encoding="utf-8") as fh:
    fh.write("# V2 result tables\n\n" + "\n".join(lines) + "\n")
print(f"\nwrote {os.path.join(RES, 'tables_v2.md')} and figures in {FIG}")
