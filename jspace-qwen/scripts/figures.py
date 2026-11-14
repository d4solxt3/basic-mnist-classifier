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
OURS = ("SS-", "Semi-", "Routed", "B-lens", "JB-mix")
if panels:
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.6), sharey=True, squeeze=False)
    order = ["logit", "regress", "J-lens", "R-lens", "J-filt", "J++", "SS-J++", "Semi-J++", "Routed",
             "Routed-LRP", "B-lens", "JB-mix"]
    for ax, (t, d) in zip(axes[0], panels):
        names = [n for n in order if n in d.get("selected", {})]
        keys = [d["selected"][n] for n in names]
        labels = [k.split("|")[0] for k in keys]
        means = [d["results"][k]["item@10"]["macro"]["mean"] for k in keys]
        lo = [d["results"][k]["item@10"]["macro"]["lo"] for k in keys]
        hi = [d["results"][k]["item@10"]["macro"]["hi"] for k in keys]
        y = np.arange(len(names))[::-1]
        cols = [COLOR["J++"] if n == "J++" else (SLOT[0] if n.startswith(OURS) else "#b9b8b2") for n in names]
        ax.barh(y, means, height=0.62, color=cols, edgecolor=SURF, linewidth=2)
        ax.errorbar(means, y, xerr=[np.array(means) - lo, np.array(hi) - means], fmt="none", ecolor=INK2,
                    elinewidth=1, capsize=2)
        for yi, m in zip(y, means):
            ax.text(m + 0.012, yi, f"{m:.3f}", va="center", fontsize=7.5, color=INK)
        ax.set_yticks(y, labels)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("held-out recall@10 (macro over items)")
        ax.set_title({"main": "correctness-filtered items", "main_all": "all items"}[t] +
                     f"  (n={d['n_held']})", loc="left", fontsize=9.5, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#b9b8b2", COLOR["J++"], SLOT[0])]
    axes[0][-1].legend(handles, ["baseline / prior work", "J++", "ours"], frameon=False, fontsize=7.5,
                       loc="lower right")
    fig.suptitle("Round 1: head-to-head under the J++ protocol (Readout Filtering, best of 7 layers, split seed 0)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    save(fig, "fig_jpp")

# ---------------------------------------------------------------- round 2 (multi-split)
r2 = load("jpp_splits_round2.json")
if r2 and "confirmatory" in r2:
    conf = r2["confirmatory"]
    seeds = [s_ for s_ in r2["seeds"] if s_ != 0]
    order = ["JppB", "RB-mix", "Routed-J++-a200", "Bagged-J++", "JB-mix", "Routed-LRP-a200", "Routed-LRP-s0.5",
             "R-lens", "J-lens", "logit"]
    names = [n for n in order if n in conf]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(names) + 1.0))
    ax.axvline(0, color=INK2, lw=1)
    for yi, n in zip(np.arange(len(names))[::-1], names):
        ds = [r2["per_seed"][str(s_)]["results"][n]["diff_vs_jpp"]["diff"] for s_ in seeds]
        c = SLOT[0] if n in ("JppB", "RB-mix", "Routed-J++-a200", "Bagged-J++") else "#8d8c87"
        ax.scatter(ds, [yi] * len(ds), s=22, color=c, alpha=0.55, edgecolor=SURF, linewidth=1, zorder=3)
        ax.scatter([conf[n]["mean_diff"]], [yi], s=70, marker="|", color=INK, linewidth=2, zorder=4)
    def lab(n):
        ties = sum(r2["per_seed"][str(s_)]["results"][n]["diff_vs_jpp"]["diff"] == 0 for s_ in seeds)
        return f"{n}  ({conf[n]['wins']}/{conf[n]['n']}" + (f", {ties} tie)" if ties else ")")
    ax.set_yticks(np.arange(len(names))[::-1], [lab(n) for n in names])
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("held-out recall@10 minus J++ (paired, same split)")
    ax.set_title("Round 2: five fresh splits (dots), mean (bar); wins vs J++ in brackets", loc="left",
                 fontsize=9.5, color=INK)
    hs = [plt.Line2D([], [], ls="", marker="o", ms=5, color=SLOT[0], alpha=0.55),
          plt.Line2D([], [], ls="", marker="o", ms=5, color="#8d8c87", alpha=0.55),
          plt.Line2D([], [], ls="", marker="|", ms=9, mew=2, color=INK)]
    ax.legend(hs, ["pre-registered candidate (one split)", "re-evaluated lens (one split)", "mean over splits"],
              frameon=False, fontsize=7.5, loc="upper left")
    save(fig, "fig_round2")

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
swm = load("swaps_midband.json")
if sw:
    panels_sw = [(sw, f"workspace band (layers {sw['band'][0]}–{sw['band'][-1]})")]
    if swm:
        panels_sw.append((swm, f"mid-depth band (layers {swm['band'][0]}–{swm['band'][-1]})"))
    fig, axes = plt.subplots(1, len(panels_sw), figsize=(4.3 * len(panels_sw), 3.2), sharey=True, squeeze=False)
    for ax, (d, title) in zip(axes[0], panels_sw):
        grid(ax)
        for lens, lab_ in (("jlens", "J-lens"), ("logit", "logit"), ("regress", "regress")):
            for var in ("main", "random_null"):
                ks = [f"flexgen|{var}|{lens}|{sc}" for sc in d["scales"]]
                pts = [(d["results"][k]["median_edit_norm"], d["results"][k]["success"][0], sc)
                       for k, sc in zip(ks, d["scales"]) if k in d["results"]]
                if not pts:
                    continue
                xs, ys, scs = zip(*pts)
                if var == "main":
                    ax.plot(xs, ys, color=COLOR[lab_], marker="o", ms=4, label=f"{lab_} swap")
                    for x_, y_, sc in pts:
                        ax.annotate(f"×{sc:g}", (x_, y_), xytext=(3, 3), textcoords="offset points", fontsize=6.5,
                                    color=INK2)
                else:
                    ax.plot(xs, ys, color=COLOR[lab_], ls=":", lw=1.5, alpha=0.8,
                            label=f"{lab_} random-target null")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("median edit norm")
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    axes[0][0].set_ylabel("swap success (f(B) at rank 1)")
    axes[0][0].set_ylim(-0.02, 0.62)
    axes[0][0].legend(frameon=False, fontsize=6.8, loc="upper left", ncol=2)
    fig.suptitle("Flexible generalisation: swapping a function's argument along lens directions (60 scored trials)",
                 x=0.01, y=1.03, ha="left", fontsize=10, color=INK)
    save(fig, "fig_swaps")

# ---------------------------------------------------------------- ignition
ist = load("ignition_main_stats.json")
if ist and "linear_jlens|country" in ist["per_layer"]:
    Ls = ist["layers"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax in (a1, a2):
        grid(ax)
        ax.axvspan(19.5, 24.5, color=GRID, alpha=0.6, lw=0)
        ax.text(22, 0.98, "band", transform=ax.get_xaxis_transform(), ha="center", va="top", color=INK2, fontsize=7.5)
    for key, lab_, c, ls in (("jlens|country", "J-lens, actual", COLOR["J-lens"], "-"),
                             ("linear_jlens|country", "J-lens, linear control", COLOR["J-lens"], ":"),
                             ("logit|country", "logit lens, actual", COLOR["logit"], "-"),
                             ("linear_logit|country", "logit lens, linear control", COLOR["logit"], ":")):
        ys = [r[0] for r in ist["per_layer"][key]]
        a1.plot(Ls, ys, color=c, ls=ls, marker="o" if ls == "-" else None, ms=3, label=lab_)
    a1.set_xlabel("layer")
    a1.set_ylabel("median 10→90% width in α")
    a1.set_title("Country pairs: actual vs linear-interpolation control", loc="left", fontsize=9.5, color=INK)
    a1.legend(frameon=False, fontsize=7, loc="lower left")
    a1.set_ylim(0, 0.8)
    for kind, lab_, c in (("country", "countries (660 trials)", SLOT[0]), ("scrambled", "unrelated nouns (60)", SLOT[6]),
                          ("idiom", "associated nouns (60)", SLOT[3])):
        r = ist["contrasts"][f"jlens: actual-linear|{kind}"]
        m = [x[0] for x in r]
        lo = [x[1] for x in r]
        hi = [x[2] for x in r]
        a2.fill_between(Ls, lo, hi, color=c, alpha=0.15, lw=0)
        a2.plot(Ls, m, color=c, marker="o", ms=3, label=lab_)
    a2.axhline(0, color=INK2, lw=1)
    a2.set_xlabel("layer")
    a2.set_ylabel("actual − linear width (J-lens)")
    a2.set_title("Sharpening done by the network (negative = sharper)", loc="left", fontsize=9.5, color=INK)
    a2.legend(frameon=False, fontsize=7, loc="lower left")
    save(fig, "fig_ignition")
print("done")
