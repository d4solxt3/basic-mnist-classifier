"""Paper figures (static). Each figure is drawn only if its results file exists.

Design: fixed categorical order (validated palette, see dataviz skill), one
colour per lens across all figures, thin 2px lines, recessive grid, direct
labels (contrast relief), one y-axis per panel.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = __file__.rsplit("/scripts/", 1)[0]
R = os.path.join(ROOT, "results")
F = os.path.join(ROOT, "paper", "figures")
os.makedirs(F, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR = {"J-lens": SLOT[0], "logit": SLOT[1], "regress": SLOT[2], "J++": SLOT[3], "sketch": SLOT[4],
         "shaped": SLOT[5], "identity-cv": SLOT[6], "no-cv": SLOT[7]}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "lines.linewidth": 2})


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(F, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def grid(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- per-layer
main, band = load("readout_main.json"), load("band.json")
if main:
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    grid(ax)
    if band:
        lo, hi = band["band"]
        ax.axvspan(lo - 0.5, hi + 0.5, color=GRID, alpha=0.6, lw=0)
        ax.text((lo + hi) / 2, 0.98, "band", transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=INK2, fontsize=8)
    for key, lab in (("J-lens|raw", "J-lens"), ("logit|raw", "logit"), ("regress|raw", "regress")):
        if key not in main["results"]:
            continue
        c = main["results"][key]["per_layer@10"]
        xs = sorted(int(k) for k in c)
        ys = [c[str(x)] for x in xs]
        ax.plot(xs, ys, color=COLOR[lab], marker="o", ms=3)
        ax.text(xs[-1] + 0.4, ys[-1], {"J-lens": "J-lens", "logit": "logit lens", "regress": "regression lens"}[lab],
                color=INK, va="center", fontsize=8)
    ax.set_xlabel("layer (block output)")
    ax.set_ylabel("recall@10 (single layer)")
    ax.set_title("Latent-intermediate recall by layer, Qwen3-0.6B", loc="left", fontsize=10, color=INK)
    ax.set_xlim(-0.5, 30)
    save(fig, "fig_layers")

# ---------------------------------------------------------------- J++ bars
panels = [(t, load(f"jpp_compare_{t}.json")) for t in ("main", "main_all")]
panels = [(t, d) for t, d in panels if d]
if panels:
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2), sharey=True, squeeze=False)
    order = ["logit", "regress", "J-lens", "R-lens", "Pooled", "Pooled-LRP", "J-filt", "J++", "Routed", "Routed-LRP",
             "SS-J-filt", "SS-J++", "Semi-J-filt", "Semi-J++"]
    for ax, (t, d) in zip(axes[0], panels):
        names = [n for n in order if f"{n}|raw|filter" in d["results"]]
        means = [d["results"][f"{n}|raw|filter"]["item@10"]["macro"]["mean"] for n in names]
        lo = [d["results"][f"{n}|raw|filter"]["item@10"]["macro"]["lo"] for n in names]
        hi = [d["results"][f"{n}|raw|filter"]["item@10"]["macro"]["hi"] for n in names]
        y = np.arange(len(names))[::-1]
        cols = [COLOR["J++"] if n == "J++" else (SLOT[0] if n.startswith(("SS", "Semi", "Routed")) else "#b9b8b2")
                for n in names]
        ax.barh(y, means, height=0.62, color=cols, edgecolor=SURF, linewidth=2)
        ax.errorbar(means, y, xerr=[np.array(means) - lo, np.array(hi) - means], fmt="none", ecolor=INK2,
                    elinewidth=1, capsize=2)
        for yi, m in zip(y, means):
            ax.text(m + 0.01, yi, f"{m:.2f}", va="center", fontsize=7.5, color=INK)
        ax.set_yticks(y, names)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("held-out recall@10 (item-weighted)")
        ax.set_title({"main": "correctness-filtered items", "main_all": "all items"}[t] +
                     f"  (n={d['n_held']})", loc="left", fontsize=9.5, color=INK)
    fig.suptitle("Head-to-head under the J++ protocol (Readout Filtering, best of 7 layers)", x=0.01, ha="left",
                 fontsize=10, color=INK)
    save(fig, "fig_jpp")

# ---------------------------------------------------------------- learning curves
cur = load("curves.json")
if cur:
    lab = {"exact": "J-lens", "sk16": "sketch", "sk16w": "shaped", "sk16id": "identity-cv", "sk16none": "no-cv"}
    nice = {"J-lens": "exact", "sketch": "sketch + online CV", "shaped": "shaped sketch + online CV",
            "identity-cv": "sketch + identity CV", "no-cv": "sketch, no CV"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for tag, name in lab.items():
        rows = sorted([v for v in cur.values() if v["tag"] == tag], key=lambda v: v["vjp_rows"])
        if not rows:
            continue
        x = [v["vjp_rows"] for v in rows]
        axes[0].plot(x, [v["recall10_all"]["mean"] for v in rows], color=COLOR[name], marker="o", ms=3,
                     label=nice[name])
        er = [(v["vjp_rows"], np.mean(list(v["dict_err"].values()))) for v in rows if "dict_err" in v]
        if er:
            axes[1].plot(*zip(*er), color=COLOR[name], marker="o", ms=3, label=nice[name])
    for ax, yl in zip(axes, ("recall@10 (all items, best of 7 layers)", "rel. error of A·J vs held-out reference")):
        ax.set_xscale("log")
        ax.set_xlabel("backward VJP rows (compute)")
        ax.set_ylabel(yl)
        grid(ax)
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Estimator efficiency: lens quality vs compute", x=0.01, ha="left", fontsize=10, color=INK)
    save(fig, "fig_curves")

# ---------------------------------------------------------------- swaps
sw = load("swaps_main.json")
if sw:
    exps = [("probe-swap", "probe-swap (two-hop)"), ("flexgen", "flexible generalization")]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
    for ax, (exp, title) in zip(axes, exps):
        grid(ax)
        for lens, lab_ in (("jlens", "J-lens"), ("logit", "logit"), ("regress", "regress")):
            for var, ls in (("main", "-"), ("random_null", ":")):
                pts = [(sc, sw["results"][f"{exp}|{var}|{lens}|{sc}"]["success"][0]) for sc in sw["scales"]
                       if f"{exp}|{var}|{lens}|{sc}" in sw["results"]]
                if pts:
                    ax.plot(*zip(*pts), color=COLOR[lab_], ls=ls, marker="o" if var == "main" else None, ms=3,
                            label=f"{lab_} {'swap' if var == 'main' else 'random null'}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("edit scale")
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    axes[0].set_ylabel("swap success (rank-1)")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "fig_swaps")

# ---------------------------------------------------------------- ignition
ig = load("ignition_main.json")
if ig:
    fig, ax = plt.subplots(figsize=(6, 3))
    grid(ax)
    for key, lab_, ls in (("jlens|country", "J-lens", "-"), ("logit|country", "logit", "-"),
                          ("jlens|scrambled", "J-lens", ":")):
        if key in ig["summary"]:
            ax.plot(ig["layers"], ig["summary"][key]["median_width"], color=COLOR[lab_], ls=ls, marker="o", ms=3,
                    label=key.replace("|", ", "))
    ax.axhline(0.8, color=INK2, lw=1, ls="--")
    ax.text(ig["layers"][0], 0.81, "linear mixing", fontsize=7.5, color=INK2, va="bottom")
    ax.set_xlabel("layer")
    ax.set_ylabel("median 10→90% width (α)")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("Ignition: sharpness of the concept switch", loc="left", fontsize=10, color=INK)
    save(fig, "fig_ignition")
print("done")
