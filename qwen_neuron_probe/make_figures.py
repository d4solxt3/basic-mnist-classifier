"""Draw the manuscript figures from outputs/*.json (vector PDF, paper/figures/).

fig_stage1_null.pdf   - per axis: 200 permutation maxima of |d| vs the candidate's |d|
fig_stage2_dose.pdf   - per axis: target vs random-control shift by dose (with the 2 SD bar),
                        and distinct_2 by dose (with the 0.45 guard and the unedited level)
Run: python make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = ROOT / "paper" / "figures"
AXES = ["syc", "unc", "cau", "per"]
TITLES = {"syc": "Sycophancy", "unc": "Uncertainty", "cau": "Caution", "per": "Persona"}

# Validated categorical slots 1-2 (light mode) and text/neutral inks.
TARGET, RANDOM = "#2a78d6", "#eb6834"
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"

# STIXGeneral ships with matplotlib (Times-like, matches the newtx body font of the paper).
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def stage1_figure():
    fig, axs = plt.subplots(1, 4, figsize=(6.3, 1.85), sharex=True, sharey=True)
    for ax, a in zip(axs, AXES):
        s = load(f"stage1_{a}.json")
        st = s["stats"]
        null = st["null_max_abs_d"]
        cand = st["candidate"]["abs_d"]
        ax.hist(null, bins=[1.0 + 0.1 * i for i in range(15)], color=TARGET, alpha=0.85,
                edgecolor="white", linewidth=0.6, zorder=2)
        ax.axvline(st["null_q99"], color=NEUTRAL, lw=0.9, ls=(0, (3, 2)), zorder=3)
        ax.axvline(cand, color=RANDOM, lw=1.6, zorder=4)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_title(f"{TITLES[a]} ({s['verdict']})", color=INK, pad=4)
        ymax = 75
        ax.set_ylim(0, ymax)
        right = cand > 4.5
        x_lab = cand - 0.15 if right else max(cand, st["null_q99"]) + 0.15
        ax.text(x_lab, ymax * 0.9, f"candidate\n$|d|={cand:.3f}$",
                ha="right" if right else "left", va="top", color=INK, fontsize=7)
        if a == AXES[0]:
            ax.text(st["null_q99"] + 0.12, ymax * 0.5, "99th pct.\nof null", ha="left", va="center",
                    color=INK2, fontsize=6.5)
        ax.set_xlim(1.0, 8.2)
        ax.set_xticks([2, 4, 6, 8])
    axs[0].set_ylabel("permutations")
    fig.supxlabel(r"whole-network maximum $|d|$ under shuffled labels ($B = 200$)", fontsize=8, y=0.02)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(FIG / "fig_stage1_null.pdf")
    plt.close(fig)


def stage2_figure():
    fig, axs = plt.subplots(2, 4, figsize=(6.3, 3.55), sharex=True, sharey="row")
    for j, a in enumerate(AXES):
        s = load(f"stage2_{a}.json")
        doses = s["doses"]
        o, g = s["ood_delta_logp"], s["distinct_2"]["summary"]
        thr = s["decision"]["threshold"]
        tgt = [o["target"][str(d)]["shift"]["mean"] for d in doses]
        rnd = [o["random"][str(d)]["shift"]["mean"] for d in doses]
        top = axs[0, j]
        top.axhline(0, color=INK2, lw=0.6, zorder=1)
        top.axhline(thr, color=NEUTRAL, lw=0.9, ls=(0, (3, 2)), zorder=1)
        top.text(doses[0] - 0.12, thr - 0.8, f"2 SD = {thr:.3f}", color=INK2, fontsize=6.5, va="top")
        top.plot(doses, rnd, color=RANDOM, lw=1.6, marker="s", ms=4.2, mec="white", mew=0.7, zorder=3,
                 label="random unit, same layer (mean of 3 seeds)")
        top.plot(doses, tgt, color=TARGET, lw=1.6, marker="o", ms=4.6, mec="white", mew=0.7, zorder=4,
                 label="Stage 1 candidate")
        top.set_title(TITLES[a] + ("$^{\\dagger}$" if s["decision"]["exploratory"] else ""), color=INK, pad=4)
        top.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        top.set_ylim(-9, 23)
        if j == 0:  # direct labels on the first panel (legend carries them for the rest)
            top.text(doses[0] - 0.12, tgt[0] + 2.2, "candidate", color=INK, fontsize=6.5, va="bottom")
            top.text(doses[-1] + 0.1, rnd[-1] - 1.2, "random", color=INK, fontsize=6.5, va="top", ha="right")

        bot = axs[1, j]
        bt = [g["target"][str(d)]["mean"] for d in doses]
        bts = [g["target"][str(d)]["sd"] for d in doses]
        br = [g["random"][str(d)]["mean"] for d in doses]
        brs = [g["random"][str(d)]["sd"] for d in doses]
        base = g["baseline"]["mean"]
        bot.axhline(0.45, color=NEUTRAL, lw=0.9, ls=(0, (3, 2)), zorder=1)
        bot.axhline(base, color=INK2, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
        bot.errorbar(doses, br, yerr=brs, color=RANDOM, lw=1.6, marker="s", ms=4.2, mec="white", mew=0.7,
                     elinewidth=0.8, capsize=1.8, zorder=3)
        bot.errorbar(doses, bt, yerr=bts, color=TARGET, lw=1.6, marker="o", ms=4.6, mec="white", mew=0.7,
                     elinewidth=0.8, capsize=1.8, zorder=4)
        bot.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        bot.set_ylim(0, 1.0)
        bot.set_xticks(doses)
        bot.set_xlim(0.35, 1.65)
        if j == 0:
            bot.text(0.38, 0.45 + 0.02, "guard 0.45", color=INK2, fontsize=6.5, va="bottom")
            bot.text(0.38, base + 0.02, "unedited", color=INK2, fontsize=6.5, va="bottom")
    axs[0, 0].set_ylabel(r"shift in $\Delta\log P$")
    axs[1, 0].set_ylabel(r"distinct\_2".replace("\\_", "_"))
    fig.supxlabel(r"dose $\alpha$ (perturbation norm in units of median residual norm $\tilde\rho_\ell$)",
                  fontsize=8, y=0.015)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.6)
    fig.subplots_adjust(top=0.86, bottom=0.13)
    fig.savefig(FIG / "fig_stage2_dose.pdf")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    stage1_figure()
    stage2_figure()
    print("wrote", sorted(p.name for p in FIG.glob("*.pdf")))
