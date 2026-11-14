"""Paper figures in one publication style. Each figure is drawn only if its results exist.

Style (shared by every figure):
* Computer Modern text and maths, so figure text matches the LaTeX paper; 8 pt type.
* Figures are drawn at their printed size (full width = 6.5 in, the arXiv text width), so
  type and line weights are the same in every figure. PDF (vector) for LaTeX, 300-dpi PNG
  for Markdown and the doc.
* One colour per lens across all figures, from a fixed categorical order that passes the
  colour-vision-deficiency checks (dataviz reference palette); prior work in neutral grey.
* Thin 1.3 pt lines, recessive grid and axes, panel letters, direct labels where they fit.

python scripts/figures.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = __file__.rsplit("/scripts/", 1)[0]
R = os.path.join(ROOT, "results")
F = os.path.join(ROOT, "paper", "figures")
os.makedirs(F, exist_ok=True)

FULL = 6.5                              # printed width (in) = arXiv text width
INK, INK2, MUTED, GRID = "#1a1a19", "#52514e", "#9a998f", "#e4e3de"
GREY = "#8d8c87"                         # prior work / baselines
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LENS = {"J-lens": SLOT[0], "logit": SLOT[1], "regress": SLOT[2], "J++": INK,
        "sketch": SLOT[0], "shaped": SLOT[6], "identity-cv": SLOT[2], "no-cv": SLOT[1], "exact": INK}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": True,
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.3, "lines.markersize": 3.2, "patch.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
})
DASH = r"$-$"


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(F, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(F, f"{name}.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def panel(ax, letter, title):
    if not letter:
        ax.set_title(title, loc="left", color=INK, pad=4)
        return
    ax.set_title(rf"$\mathbf{{({letter})}}$  {title}", loc="left", color=INK, pad=4)


def band_shade(ax, lo=20, hi=24, label=True):
    ax.axvspan(lo - 0.5, hi + 0.5, color="#f1f0ec", lw=0, zorder=0)
    if label:
        ax.text((lo + hi) / 2, 0.985, "band", transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=INK2, fontsize=6.8)


# ================================================================ Figure: readout by layer
main, band = load("readout_main.json"), load("band.json")
if main:
    fig, ax = plt.subplots(figsize=(FULL, 2.25))
    grid(ax)
    if band:
        band_shade(ax, *band["band"])
    for key, col, lab in (("J-lens|raw", "J-lens", "J-lens"), ("logit|raw", "logit", "logit lens"),
                          ("regress|raw", "regress", "regression lens")):
        if key not in main["results"]:
            continue
        c = main["results"][key]["per_layer@10"]
        xs = sorted(int(k) for k in c)
        ys = [100 * c[str(x)] for x in xs]
        ax.plot(xs, ys, color=LENS[col], marker="o", ms=2.6, zorder=3)
        nudge = {"logit": 0.9, "regress": -0.9}.get(col, 0.0)
        ax.text(xs[-1] + 0.5, ys[-1] + nudge, lab, color=LENS[col], va="center", fontsize=7.5)
    ax.set_xlim(-0.6, 30.5)
    ax.set_xticks(range(0, 27, 2))
    ax.set_xlabel(r"layer $\ell$ (block output)")
    ax.set_ylabel("recall@10 (%, single layer)")
    save(fig, "fig_layers")

# ================================================================ Figure: round 1 dot-and-whisker
FAM = ["J++", "Routed-LRP", "JB-mix", "R-lens", "Semi-J++", "J-filt", "J-lens", "SS-J++", "Routed", "logit",
       "B-lens", "regress"]
NICE = {"logit": "logit lens", "regress": "regression lens", "J-lens": "J-lens", "R-lens": "R-lens (LRP)",
        "J-filt": "J-filt (plain experts)", "J++": "J++", "SS-J++": "SS-J++ (future tokens)",
        "Semi-J++": "Semi-J++", "Routed": "routed, plain experts", "Routed-LRP": "routed LRP experts",
        "B-lens": "broadcast lens", "JB-mix": "JB-mix"}
OURS = ("SS-", "Semi-", "Routed", "B-lens", "JB-mix")
r1 = [(t, load(f"jpp_compare_{t}.json")) for t in ("main", "main_all")]
r1 = [(t, d) for t, d in r1 if d]
if r1:
    d0 = r1[0][1]
    fams = [f for f in FAM if f in d0["selected"]]
    fams.sort(key=lambda f: -d0["results"][d0["selected"][f]]["item@10"]["macro"]["mean"])
    fig, axes = plt.subplots(1, len(r1), figsize=(FULL, 2.9), sharey=True, squeeze=False)
    y = np.arange(len(fams))[::-1]
    for k, (ax, (t, d)) in enumerate(zip(axes[0], r1)):
        grid(ax, "x")
        jm = d["results"][d["selected"]["J++"]]["item@10"]["macro"]["mean"]
        ax.axvline(100 * jm, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=1)
        for yi, f in zip(y, fams):
            if f not in d["selected"]:
                continue
            m = d["results"][d["selected"][f]]["item@10"]["macro"]
            c = INK if f == "J++" else (SLOT[0] if f.startswith(OURS) else GREY)
            ax.plot([100 * m["lo"], 100 * m["hi"]], [yi, yi], color=c, lw=1.1, solid_capstyle="butt", zorder=2)
            ax.plot(100 * m["mean"], yi, marker="D" if f == "J++" else "o", ms=4 if f == "J++" else 3.6, color=c,
                    mec="white", mew=0.5, zorder=3)
        ax.set_yticks(y, [NICE[f] for f in fams])
        ax.set_xlabel("held-out recall@10 (%)")
        n = d["n_held"]
        panel(ax, "ab"[k], {"main": f"correctness-filtered items (n = {n})",
                            "main_all": f"all items (n = {n})"}[t])
    handles = [Line2D([], [], color=GREY, marker="o", lw=1.1, ms=3.6, label="baseline / prior work"),
               Line2D([], [], color=INK, marker="D", lw=1.1, ms=4, label="J++ (dashed: its score)"),
               Line2D([], [], color=SLOT[0], marker="o", lw=1.1, ms=3.6, label="ours")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.58, -0.05))
    fig.subplots_adjust(bottom=0.2, wspace=0.08)
    save(fig, "fig_jpp")

# ================================================================ Figures: rounds 2 and 3 (forest plots, bridge)
LAB2 = {"Bagged-J++": "Bagged-J++", "RB-mix": "RB-mix", "JppB": "JppB", "Routed-J++-a200": "Routed-J++",
        "RLB-mix": "RLB-mix", "Routed-LRP-a200": r"routed LRP ($\kappa$=200)",
        "Routed-LRP-s0.5": r"routed LRP ($\lambda$=0.5)", "R-lens": "R-lens", "JB-mix": "JB-mix",
        "J-lens": "J-lens", "logit": "logit lens", "LRP-B-lens": "LRP broadcast lens"}


def forest(ax, res, seeds, cands, others, letter, title, min_wins):
    conf = res["confirmatory"]

    def diffs(n):
        return [100 * res["per_seed"][str(s)]["results"][n]["diff_vs_jpp"]["diff"] for s in seeds]

    rows = cands + [None] + others
    y = np.arange(len(rows))[::-1].astype(float)
    ax.axvline(0, color=INK, lw=0.7, zorder=1)
    ax.axhspan(y[len(cands) - 1] - 0.5, y[0] + 0.5, color="#eef4fc", lw=0, zorder=0)
    ticks, labels = [], []
    for yi, n in zip(y, rows):
        if n is None:
            continue
        ds = diffs(n)
        c = SLOT[0] if n in cands else GREY
        ax.plot([min(ds), max(ds)], [yi, yi], color=c, lw=0.9, alpha=0.7, zorder=2)
        ax.scatter(ds, [yi] * len(ds), s=8 if len(ds) > 8 else 9, color=c, alpha=0.45 if len(ds) > 8 else 0.55,
                   edgecolor="none", zorder=3)
        ax.plot(conf[n]["mean_diff"] * 100, yi, marker="|", ms=9, mew=1.8, color=INK if n in cands else INK2,
                zorder=4)
        ties = sum(x == 0 for x in ds)
        w = f"{conf[n]['wins']}/{conf[n]['n']}" + (f" +{ties} tie" if ties else "")
        meets = n in cands and conf[n]["mean_diff"] > 0 and conf[n]["wins"] >= min_wins
        ax.text(1.02, yi, w, transform=ax.get_yaxis_transform(), va="center", fontsize=7,
                color=INK if meets else INK2, family="cmb10" if meets else "cmr10")
        ticks.append(yi)
        labels.append(LAB2[n])
    ax.text(1.02, y[0] + 0.95, "wins", transform=ax.get_yaxis_transform(), fontsize=7, color=INK2)
    ax.set_yticks(ticks, labels)
    ax.set_ylim(y[-1] - 0.7, y[0] + 1.25)
    grid(ax, "x")
    ax.set_xlabel(r"held-out recall@10 minus J++ (points, same split)")
    ax.text(0.01, y[0] + 0.95, "pre-registered candidates", color=SLOT[0], fontsize=6.8,
            transform=ax.get_yaxis_transform())
    panel(ax, letter, title)
    leg = [Line2D([], [], ls="", marker="o", ms=3, color=INK2, alpha=0.6, label="one split"),
           Line2D([], [], ls="", marker="|", ms=8, mew=1.8, color=INK, label="mean")]
    ygap = y[len(cands)]
    y0, y1 = ax.get_ylim()
    ax.legend(handles=leg, loc="center left", ncol=2, handletextpad=0.3, columnspacing=1.0,
              bbox_to_anchor=(0.0, (ygap - y0) / (y1 - y0)))


def bridge(bx, per, letter, title, ylim):
    held = {n: float(np.mean(v)) for n, v in per.items()}
    steps = [("J-lens", "R-lens", "+LRP"), ("R-lens", "J++", "+weights"), ("J++", "Bagged-J++", "+bagging")]
    grid(bx)
    bx.plot(0, held["J-lens"], marker="o", ms=5, color=SLOT[0], zorder=4)
    bx.text(0, held["J-lens"] - 0.5, f"{held['J-lens']:.1f}", ha="center", va="top", fontsize=7)
    prev_right = 0.0
    for i, (a, b, _) in enumerate(steps, start=1):
        lo, hi = held[a], held[b]
        dlt = np.array(per[b]) - np.array(per[a])
        bx.bar(i, hi - lo, bottom=lo, width=0.55, color=SLOT[0], alpha=0.85, edgecolor="white", lw=0.6, zorder=3)
        bx.plot([prev_right, i - 0.275], [lo, lo], color=MUTED, lw=0.6, ls=":", zorder=2)
        prev_right = i + 0.275
        bx.text(i, hi + 0.25, f"+{hi - lo:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
        bx.text(i, hi + 1.05, f"[{dlt.min():+.1f}, {dlt.max():+.1f}]".replace("-", DASH), ha="center",
                va="bottom", fontsize=5.8, color=INK2)
    last = len(steps) + 1
    bx.plot([prev_right, last], [held["Bagged-J++"]] * 2, color=MUTED, lw=0.6, ls=":")
    bx.plot(last, held["Bagged-J++"], marker="o", ms=5, color=INK, zorder=4)
    bx.text(last, held["Bagged-J++"] + 0.35, f"{held['Bagged-J++']:.1f}", ha="center", va="bottom", fontsize=7)
    bx.set_xticks(np.arange(last + 1), ["J-lens", *[s[2] for s in steps], "Bagged\nJ++"], fontsize=6.4)
    bx.set_ylim(*ylim)
    bx.set_ylabel("mean held-out recall@10 (%)")
    panel(bx, letter, title)
    bx.text(0.0, -0.3, f"axis starts at {ylim[0]:g}%; brackets: range over splits", transform=bx.transAxes,
            fontsize=6, color=INK2)


r2, r3 = load("jpp_splits_round2.json"), load("jpp_splits_round3.json")
if r2 and "confirmatory" in r2:
    s2 = [s for s in r2["seeds"] if s != 0]
    fig, ax = plt.subplots(figsize=(4.4, 2.85))
    forest(ax, r2, s2, ["Bagged-J++", "RB-mix", "JppB", "Routed-J++-a200"],
           ["Routed-LRP-a200", "Routed-LRP-s0.5", "R-lens", "JB-mix", "J-lens", "logit"],
           "", "round 2: difference from J++ on five fresh splits", 4)
    save(fig, "fig_round2")
if r3 and "confirmatory" in r3:
    s3 = list(r3["seeds"])
    fig = plt.figure(figsize=(FULL, 2.85))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.5)
    ax = fig.add_subplot(gs[0])
    forest(ax, r3, s3, ["Bagged-J++", "RB-mix", "RLB-mix"],
           ["Routed-LRP-a200", "R-lens", "J-lens", "LRP-B-lens", "logit"], "a", "round 3: twenty fresh splits", 16)
    bx = fig.add_subplot(gs[1])
    names_b = ("J-lens", "R-lens", "J++", "Bagged-J++")
    per = {n: [100 * r2["per_seed"][str(s)]["results"][n]["held@10"] for s in s2] +
              [100 * r3["per_seed"][str(s)]["results"][n]["held@10"] for s in s3] for n in names_b}
    bridge(bx, per, "b", f"where J++'s gain comes from ({len(s2) + len(s3)} splits)", (32.5, 43.5))
    save(fig, "fig_round3")

# ================================================================ Figure: causal swaps
sw, swm = load("swaps_main.json"), load("swaps_midband.json")
wins_ = [(lo, hi, load(f"swaps_win{lo}_{hi}.json")) for lo, hi in ((3, 7), (7, 11), (11, 15), (15, 19))]
wins_ = [(lo, hi, d) for lo, hi, d in wins_ if d]
if sw:
    pans = [(sw, f"band 20{DASH}24 (5 layers)")] + ([(swm, f"layers 7{DASH}20 (14 layers)")] if swm else [])
    ncol = len(pans) + (1 if wins_ else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(FULL, 2.35), squeeze=False)
    axes = axes[0]
    for k, (ax, (d, title)) in enumerate(zip(axes, pans)):
        grid(ax)
        for lens, col, lab_ in (("jlens", "J-lens", "J-lens"), ("logit", "logit", "logit lens"),
                                ("regress", "regress", "regression lens")):
            pts = [(d["results"][f"flexgen|main|{lens}|{sc}"]["median_edit_norm"],
                    100 * d["results"][f"flexgen|main|{lens}|{sc}"]["success"][0], sc)
                   for sc in d["scales"] if f"flexgen|main|{lens}|{sc}" in d["results"]]
            if not pts:
                continue
            xs_, ys_, _ = zip(*pts)
            ax.plot(xs_, ys_, color=LENS[col], marker="o", label=lab_, zorder=3)
            if lens == "jlens":
                for x_, y_, sc in pts:
                    ax.annotate(rf"$\times${sc:g}", (x_, y_), xytext=(2, 3), textcoords="offset points",
                                fontsize=6, color=INK2)
            nul = [(d["results"][f"flexgen|random_null|{lens}|{sc}"]["median_edit_norm"],
                    100 * d["results"][f"flexgen|random_null|{lens}|{sc}"]["success"][0])
                   for sc in d["scales"] if f"flexgen|random_null|{lens}|{sc}" in d["results"]]
            if nul:
                ax.plot(*zip(*nul), color=LENS[col], ls=(0, (1, 1.5)), lw=0.9, alpha=0.8)
        ax.set_xscale("log", base=2)
        ax.set_ylim(-2, 55)
        ax.set_xlabel("median edit norm")
        if k == 0:
            ax.set_ylabel(r"swap success (%, $f(B)$ at rank 1)")
        panel(ax, "abc"[k], title)
    hs = [Line2D([], [], color=LENS["J-lens"], marker="o", label="J-lens"),
          Line2D([], [], color=LENS["logit"], marker="o", label="logit lens"),
          Line2D([], [], color=LENS["regress"], marker="o", label="regression lens"),
          Line2D([], [], color=INK2, ls=(0, (1, 1.5)), lw=0.9, label="unrelated-target null")]
    if wins_:
        hs += [Line2D([], [], color=INK2, marker="o", ms=3, label=r"(c) scale $\times$1"),
               Line2D([], [], color=INK2, ls=(0, (3, 1.5)), marker="s", ms=3, label=r"(c) scale $\times$2")]
    fig.legend(handles=hs, loc="lower center", ncol=len(hs), bbox_to_anchor=(0.5, -0.07), fontsize=6.6,
               columnspacing=1.2, handlelength=1.8)
    if wins_:
        ax = axes[-1]
        grid(ax)
        allw = wins_ + [(20, 24, sw)]
        for lens, col, lab_ in (("jlens", "J-lens", "J-lens"), ("logit", "logit", "logit lens")):
            for sc, ls, mk in ((1.0, "-", "o"), (2.0, (0, (3, 1.5)), "s")):
                xs_, ys_, lo_, hi_ = [], [], [], []
                for lo, hi, d in allw:
                    key = f"flexgen|main|{lens}|{sc}"
                    if key in d["results"]:
                        m, l_, h_ = d["results"][key]["success"]
                        xs_.append((lo + hi) / 2)
                        ys_.append(100 * m)
                        lo_.append(100 * l_)
                        hi_.append(100 * h_)
                if xs_:
                    ax.plot(xs_, ys_, color=LENS[col], ls=ls, marker=mk, ms=3)
                    if sc == 1.0:
                        ax.fill_between(xs_, lo_, hi_, color=LENS[col], alpha=0.12, lw=0)
        nx, ny = [], []
        for lo, hi, d in allw:
            key = "flexgen|random_null|jlens|1.0"
            if key in d["results"]:
                nx.append((lo + hi) / 2)
                ny.append(100 * d["results"][key]["success"][0])
        if nx:
            ax.plot(nx, ny, color=LENS["J-lens"], ls=(0, (1, 1.5)), lw=0.9, alpha=0.8)
        ax.set_xticks([(lo + hi) / 2 for lo, hi, _ in allw], [f"{lo}{DASH}{hi}" for lo, hi, _ in allw],
                      fontsize=6.5)
        ax.set_ylim(-2, 55)
        ax.set_xlabel("5-layer window")
        panel(ax, "abc"[len(pans)], "equal-width windows")
    fig.subplots_adjust(wspace=0.3, bottom=0.22)
    save(fig, "fig_swaps")

# ================================================================ Figure: ignition
ist = load("ignition_main_stats.json")
if ist and "linear_jlens|country" in ist["per_layer"]:
    Ls = ist["layers"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.4))
    for ax in (a1, a2):
        grid(ax)
        band_shade(ax)
    for key, col, lab_, ls in (("jlens|country", "J-lens", "J-lens", "-"),
                               ("linear_jlens|country", "J-lens", "J-lens, linear control", (0, (1.2, 1.2))),
                               ("logit|country", "logit", "logit lens", "-"),
                               ("linear_logit|country", "logit", "logit lens, linear control", (0, (1.2, 1.2)))):
        ys = [r[0] for r in ist["per_layer"][key]]
        a1.plot(Ls, ys, color=LENS[col], ls=ls, marker="o" if ls == "-" else None, ms=2.6, label=lab_)
    a1.set_ylim(0, 0.8)
    a1.set_xlabel(r"layer $\ell$")
    a1.set_ylabel(r"median 10$\to$90% width in $\alpha$")
    panel(a1, "a", "country pairs: actual vs linear control")
    a1.legend(loc="lower left", fontsize=6.5)
    for kind, lab_, c in (("country", "countries (660 trials)", SLOT[0]),
                          ("scrambled", "unrelated nouns (60)", SLOT[6]),
                          ("idiom", "associated nouns (60)", SLOT[3])):
        r = ist["contrasts"][f"jlens: actual-linear|{kind}"]
        a2.fill_between(Ls, [x[1] for x in r], [x[2] for x in r], color=c, alpha=0.15, lw=0)
        a2.plot(Ls, [x[0] for x in r], color=c, marker="o", ms=2.6, label=lab_)
    a2.axhline(0, color=INK, lw=0.6)
    a2.set_ylim(-0.36, 0.16)
    a2.set_xlabel(r"layer $\ell$")
    a2.set_ylabel(r"actual $-$ control width (J-lens)")
    panel(a2, "b", "sharpening done by the network")
    a2.legend(loc="lower left", fontsize=6.5)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig_ignition")

# ================================================================ Figure: sketch estimators
cur, rho = load("curves.json"), load("rho.json")
if cur:
    lab = {"exact": "exact", "sk16": "sketch", "sk16w": "shaped", "sk16id": "identity-cv", "sk16none": "no-cv"}
    nice = {"exact": "exact estimator", "sketch": "sketch + online CV", "shaped": "shaped sketch + online CV",
            "identity-cv": "sketch + identity CV", "no-cv": "sketch, no CV"}
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    for tag, name in lab.items():
        rows = sorted([v for v in cur.values() if isinstance(v, dict) and v.get("tag") == tag],
                      key=lambda v: v["vjp_rows"])
        if not rows:
            continue
        ls = (0, (3, 1.5)) if name == "exact" else "-"
        x = [v["vjp_rows"] for v in rows]
        axes[0].plot(x, [100 * v["recall10_all"]["mean"] for v in rows], color=LENS[name], marker="o", ms=2.6,
                     ls=ls, label=nice[name])
        er = [(v["vjp_rows"], np.mean(list(v["dict_err"].values()))) for v in rows if "dict_err" in v]
        if er:
            axes[1].plot(*zip(*er), color=LENS[name], marker="o", ms=2.6, ls=ls, label=nice[name])
    for k, (ax, yl, t) in enumerate(zip(axes, ("recall@10 (%, all items)", r"relative error of $AJ$"),
                                        ("lens quality vs compute", "dictionary error vs compute"))):
        ax.set_xscale("log")
        ax.set_xlabel("backward VJP rows (compute)")
        ax.set_ylabel(yl)
        grid(ax)
        panel(ax, "ab"[k], t)
    axes[1].set_yscale("log")
    axes[0].legend(loc="lower right", fontsize=6.5)
    if rho:
        axes[1].text(0.98, 0.95, rf"predicted shaping factor $\rho(A)$ = {rho['rho_ideal']:.3f}",
                     transform=axes[1].transAxes, ha="right", va="top", fontsize=6.5, color=INK2)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig_curves")
print("done")
