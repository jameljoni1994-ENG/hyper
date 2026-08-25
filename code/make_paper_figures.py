"""Publication figures for the paper -> results/figs/*.pdf (vector).

Run AFTER experiments finish (or anytime; missing data is skipped).
Palette: Okabe-Ito (colorblind safe). Fonts: serif / CM math.
"""
import csv
import glob
import math
import os
import statistics as st
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\Windows.11\Desktop\hyper"
FIGD = os.path.join(ROOT, "results", "figs")
HIST = os.path.join(ROOT, "results", "hist_v2")
os.makedirs(FIGD, exist_ok=True)

OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00",
      "pink": "#CC79A7", "black": "#000000"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "cm", "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "legend.fontsize": 7.5,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

rows = list(csv.DictReader(open(
    os.path.join(ROOT, "results", "all_runs_v2.csv"), encoding="utf-8")))

# canonical E7 grid (excludes the one orphan theta=1e-3 row)
E7_GRID = [0.5, 0.3, 0.2, 0.1] + list(np.geomspace(1e-2, 1e-8, 11))


def med(rs):
    v = [float(r["time_s"]) for r in rs]
    return st.median(v)


def save(fig, name):
    fig.savefig(os.path.join(FIGD, name))
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ F1 E7 --
PROBS = [("quadratic", "quadratic $\\kappa{=}10^3$, $n{=}1000$",
          ("kappa1000.0-n1000",), True),
         ("logistic", "logistic $\\lambda{=}0.01$, $m{=}2000,n{=}200$",
          ("lam0.01-m2000-n200",), False),
         ("logsumexp", "log-sum-exp $m{=}300,n{=}50$", ("m300-n50",), True),
         ("a9a", "a9a (LIBSVM, real)", ("default",), True)]

fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.0))
for ax, (prob, title, cfgs, logy) in zip(axes.ravel(), PROBS):
    pts = defaultdict(list)
    for r in rows:
        if (r["exp"] == "E7" and r["problem"] == prob
                and any(c in r["cfg"] for c in cfgs)
                and float(r["theta"]) in E7_GRID):
            pts[float(r["theta"])].append(float(r["time_s"]))
    ths = sorted(pts)
    ys = [st.median(pts[t]) for t in ths]
    ns = [len(pts[t]) for t in ths]
    ax.plot(ths, ys, "-o", lw=1.2, ms=3.2, color=OI["blue"],
            mfc=OI["blue"], zorder=3)
    i = int(np.argmin(ys))
    ax.plot(ths[i], ys[i], "*", ms=11, color=OI["verm"], zorder=4)
    ax.axvline(1e-3, color="0.55", ls="--", lw=0.8)
    ax.text(1.15e-3, ax.get_ylim()[0], "", fontsize=6)
    lo, hi = ths[i] / 10, min(ths[i] * 10, max(ths))
    ax.axvspan(lo, hi, color=OI["yellow"], alpha=0.18, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log" if logy else "linear")
    ax.set_title(title, fontsize=8.5, pad=9)
    ax.set_xlabel(r"threshold $\theta$")
    bad = max(ys) / ys[i]
    ax.annotate(f"$\\times${bad:.0f}", xy=(ths[0], ys[0]),
                xytext=(5, -2), textcoords="offset points",
                fontsize=7.5, color=OI["verm"])
    print(f"F1 {prob}: n-seeds {min(ns)}-{max(ns)}, best th={ths[i]:.1e}, "
          f"worst/best={bad:.1f}")
axes[0, 0].text(1.05e-3, axes[0, 0].get_ylim()[1] * 0.55, r"folklore $10^{-3}$",
                rotation=90, fontsize=6.5, color="0.35", va="top")
for ax in axes[:, 0]:
    ax.set_ylabel(r"median wall time [s]")
fig.tight_layout(w_pad=1.4)
save(fig, "F1_e7_ucurves.pdf")

# ------------------------------------------------------------ F2 theory ---
eps = 1e-8
L = math.log(1 / eps)


def C(u, alpha=1.0, beta=2.302585093):
    u = np.asarray(u, float)
    return alpha * u - beta * np.log(u) + beta * math.log(L)


u_star = 2.302585093
uu = np.linspace(0.06, 12, 500)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5))


def _band(target=1.05):
    lo, hi = 0.06, u_star
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if C(mid) / C(u_star) > target:
            lo = mid
        else:
            hi = mid
    ul = 0.5 * (lo + hi)
    lo, hi = u_star, 12.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if C(mid) / C(u_star) < target:
            lo = mid
        else:
            hi = mid
    return ul, 0.5 * (lo + hi)


u_lo5, u_hi5 = _band()
a1.axvspan(u_lo5, u_hi5, color=OI["yellow"], alpha=0.18, lw=0, zorder=0)
print(f"F2(a): +5% band u=[{u_lo5:.3f},{u_hi5:.3f}] -> "
      f"theta in [{math.exp(-u_hi5):.4f},{math.exp(-u_lo5):.4f}] "
      f"(x{math.exp(-u_lo5)/0.1:.1f} / /{0.1/math.exp(-u_hi5):.1f})")
a1.plot(uu, C(uu) / C(u_star), lw=1.4, color=OI["blue"], zorder=3)
a1.axvline(u_star, color=OI["verm"], ls="--", lw=0.9, zorder=4)
a1.plot([u_star], [1.0], "*", ms=10, color=OI["verm"], zorder=4)
a1.axhline(1.05, color="0.6", ls=":", lw=0.8)
a1.text(9.3, 1.06, "+5%", fontsize=7, color="0.35")
a1.set_xlabel(r"$u=\ln(1/\theta)$")
a1.set_ylabel(r"$C(u)/C(u^\star)$")
a1.set_title(r"(a) model cost ($\alpha{=}1,\beta{=}\ln 10$, $\epsilon{=}10^{-8}$)",
             pad=2)
CURVES = [(0.105, OI["verm"], "0.105"), (0.5, OI["orange"], "0.50"),
          (2.303, OI["green"], "2.30"), (7.28, OI["blue"], "7.28")]


def excess(us, g):
    return 100 * np.log(g) ** 2 / (2 * us ** 2 * (1 + np.log(L / us)))


for us, col, lbl in CURVES:
    g = np.geomspace(0.02, 100, 800)
    a2.semilogx(g, excess(us, g), lw=1.3, color=col,
                label="$u^{\\star}{=}" + lbl + "$")
a2.axhline(5, color="0.6", ls=":", lw=0.8)
a2.text(45, 6.2, "5%", fontsize=7, color="0.35")
a2.set_yscale("log")
a2.set_ylim(0.05, 30000)
v10 = {us: float(excess(us, 10.0)) for us, _, _ in CURVES}
print("F2(b): exact excess at gamma=10:",
      {f"{k:g}": round(v, 1) for k, v in v10.items()})
for us, col, _ in CURVES:
    if us in (0.105, 2.303):
        a2.plot([10], [v10[us]], "o", ms=3, color=col, zorder=5)
a2.annotate(r"$\approx$3900%", xy=(10, v10[0.105]), xytext=(-46, -1),
            textcoords="offset points", fontsize=6.5, color=OI["verm"])
a2.annotate("16%", xy=(10, v10[2.303]), xytext=(4, -13),
            textcoords="offset points", fontsize=6.5, color=OI["green"])
a2.set_xlabel(r"misspecification factor $\gamma$")
a2.set_ylabel(r"excess cost [%]")
a2.set_title("(b) sensitivity regime map (Lemma 3)", fontsize=8.5, pad=9)
a2.legend(loc="upper left", ncol=2, columnspacing=0.9)
fig.tight_layout(w_pad=1.6)
save(fig, "F2_theory_penalty.pdf")

# ------------------------------------------------------------- F3 basin ---
fig, ax = plt.subplots(figsize=(3.3, 2.5))
ue = L
for Cv, col in [(1.0, OI["green"]), (10.0, OI["blue"]), (50.0, OI["verm"])]:
    gate = math.log(Cv)
    ug = np.linspace(gate * 1.001 + 0.05, ue - 0.3, 400)
    K2 = np.ceil(np.log2((ue - gate) / (ug - gate)))
    lbl = (f"$C={{1}}$" if Cv == 1 else f"$C={Cv:.0f}$")
    ax.plot(ug, K2, lw=1.4, color=col, label=lbl)
    ax.axvline(gate, color=col, ls=":", lw=0.7)
ax.set_xlabel(r"entry level $u_g=\ln(1/g_n)$")
ax.set_ylabel(r"$K_2^{\rm exact}$ squarings")
ax.legend()
fig.tight_layout()
save(fig, "F3_basin_gate.pdf")

# ------------------------------------------------------- F4 family map ----
def fam_of(r):
    p, c = r["problem"], r["cfg"]
    if p == "quadratic":
        kap = float(c.split("kappa")[1].split("-")[0])
        return f"quad\n$k{{=}}10^{{{int(round(math.log10(kap)))}}}$"
    return {"rosenbrock": "rosen-\nbrock", "logsumexp": "log-sum-\nexp",
            "logistic": "logistic"}[p]


METHODS = ["GD", "NAG-t", "NAG-CM", "L-BFGS", "Catalyst", "Newton",
           "Newton-CG", "TR", "ARC", "Hybrid-fixed@0.001",
           "Hybrid-adaptive", "RH", "RH-BB"]
PRETTY = {"Hybrid-fixed@0.001": "Hybrid fixed $\\theta{=}10^{-3}$",
          "Hybrid-adaptive": "Hybrid adaptive"}
groups = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["exp"] == "E8":
        groups[fam_of(r)][r["method"] +
                          ("@" + r["theta"] if r["theta"] else "")].append(r)
fams = sorted(groups, key=lambda s: ("quad" not in s, s))
meth_present = [m for m in METHODS if sum(m in groups[f] for f in fams) >= 2]
M = len(meth_present)
Z = np.full((M, len(fams)), np.nan)
for j, f in enumerate(fams):
    best = min(med(groups[f][m]) for m in meth_present if m in groups[f])
    for i, m in enumerate(meth_present):
        if m in groups[f]:
            Z[i, j] = med(groups[f][m]) / best
fig, ax = plt.subplots(figsize=(5.4, 0.28 * M + 1.2))
im = ax.imshow(Z, aspect="auto", cmap="YlGnBu_r",
               norm=matplotlib.colors.LogNorm(vmin=max(np.nanmin(Z), 1),
                                              vmax=np.nanmax(Z)))
ax.set_xticks(range(len(fams)), fams)
labels = [PRETTY.get(m, "$\\theta$".join(m.split("@"))
                     .replace("$\\theta$0.001", "")
                     if False else PRETTY.get(m, m)) for m in meth_present]
ax.set_yticks(range(M), [PRETTY.get(m, m.replace("@", " ")) for m in meth_present])
for i in range(M):
    for j in range(len(fams)):
        if np.isfinite(Z[i, j]):
            ax.text(j, i, f"{Z[i,j]:.1f}" if Z[i, j] >= 10 else
                    (f"{Z[i,j]:.2f}" if Z[i, j] < 1.1 else f"{Z[i,j]:.1f}"),
                    ha="center", va="center", fontsize=6.6,
                    color="white" if Z[i, j] > 8 else "black")
cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03)
cb.set_label("$\\times$ slower than best", fontsize=7.5)
fig.tight_layout()
save(fig, "F4_e8_family_map.pdf")

# ------------------------------------------------------ F5 trajectories ---
def load_hist(exp, prob, cfgsub, method, theta="", seed="0"):
    pat = os.path.join(HIST, f"{exp}__{prob}__*{cfgsub}*__{method}__"
                             f"{theta}__{seed}.npz")
    hits = glob.glob(pat)
    return np.load(hits[0])["tg"] if hits else None


PANELS = [
    ("quad $\\kappa{=}10^3$, $n{=}1000$",
     [("E8", "quadratic", "kappa1000.0-n1000", "NAG-CM", "", OI["sky"], "NAG-CM"),
      ("E8", "quadratic", "kappa1000.0-n1000", "L-BFGS", "", OI["green"], "L-BFGS"),
      ("E8", "quadratic", "kappa1000.0-n1000", "Hybrid-fixed", "0.001",
       OI["verm"], "Hybrid $\\theta{=}10^{-3}$")]),
    ("rosenbrock $n{=}100$",
     [("E8", "rosenbrock", "n100", "L-BFGS", "", OI["green"], "L-BFGS"),
      ("E8", "rosenbrock", "n100", "Newton", "", OI["blue"], "Newton"),
      ("E8", "rosenbrock", "n100", "Hybrid-fixed", "0.001",
       OI["verm"], "Hybrid $\\theta{=}10^{-3}$")]),
    ("log-sum-exp $m{=}500,n{=}50$",
     [("E8", "logsumexp", "m500-n50", "L-BFGS", "", OI["green"], "L-BFGS"),
      ("E8", "logsumexp", "m500-n50", "TR", "", OI["pink"], "TR"),
      ("E8", "logsumexp", "m500-n50", "Hybrid-fixed", "0.001",
       OI["verm"], "Hybrid $\\theta{=}10^{-3}$")]),
    ("a9a (real)",
     [("E6", "a9a", "default", "NAG-CM", "", OI["sky"], "NAG-CM"),
      ("E6", "a9a", "default", "L-BFGS", "", OI["green"], "L-BFGS"),
      ("E6", "a9a", "default", "Hybrid-fixed", "0.001",
       OI["verm"], "Hybrid $\\theta{=}10^{-3}$")]),
]
fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.0))
for ax, (title, specs) in zip(axes.ravel(), PANELS):
    for exp, prob, cfg, meth, th, col, lab in specs:
        d = load_hist(exp, prob, cfg, meth, th)
        if d is None:
            print("  missing:", exp, prob, cfg, meth, th)
            continue
        ax.semilogy(d[:, 0], np.maximum(d[:, 1], 1e-14), lw=1.1,
                    color=col, label=lab)
        ax.set_xlim(left=0)
    ax.set_xlabel("time [s]")
    ax.set_title(title, fontsize=8.5, pad=9)
axes[0, 0].set_ylabel(r"$\|g\|$")
axes[1, 0].set_ylabel(r"$\|g\|$")
axes[0, 0].legend(fontsize=6.2, loc="lower left")
axes[0, 1].legend(fontsize=6.2, loc="upper right")
fig.tight_layout(w_pad=1.4)
save(fig, "F5_trajectories.pdf")

# --------------------------------------------------------- F6 E6 profile --
g6 = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["exp"] == "E6":
        g6[r["problem"]][r["method"] +
                         ("@" + r["theta"] if r["theta"] else "")].append(r)
SHOW = ["GD", "NAG-t", "NAG-CM", "L-BFGS", "Catalyst", "Newton-CG", "TR",
        "ARC", "Hybrid-fixed@0.001", "Hybrid-adaptive", "RH"]
datasets = sorted(g6)
cols = [OI["sky"], OI["green"], OI["blue"], OI["verm"], OI["pink"],
        OI["orange"], "0.4"]
fig, ax = plt.subplots(figsize=(6.8, 3.0))
W = 0.8 / max(len(SHOW), 1)
for i, ds in enumerate(datasets):
    base = min(med(v) for v in g6[ds].values())
    for j, m in enumerate(SHOW):
        if m not in g6[ds]:
            continue
        v = med(g6[ds][m]) / base
        ax.bar(i + (j - len(SHOW) / 2) * W, min(v, 200), W * 0.92,
               color=cols[j % len(cols)],
               label=(m if i == 0 else None))
        if v > 200:
            ax.text(i + (j - len(SHOW) / 2) * W, 205, "^", ha="center",
                    fontsize=6, rotation=90)
ax.set_yscale("log")
ax.set_xticks(range(len(datasets)),
              [d + "\n(real)" for d in datasets])
ax.set_ylabel("median time / dataset best")
ax.axhline(1, color="0.6", ls=":", lw=0.8)
leg = ax.legend(ncol=4, fontsize=6.4, loc="upper center",
                bbox_to_anchor=(0.5, 1.18))
out = os.path.join(FIGD, "F6_e6_profile.pdf")
fig.savefig(out, bbox_inches="tight",
            bbox_extra_artists=(leg,))
plt.close(fig)
print("wrote F6_e6_profile.pdf")

print("done.")


