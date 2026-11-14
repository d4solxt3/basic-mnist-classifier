"""Generate every empirical number in the manuscript from outputs/*.json.

Writes paper/generated/*.tex (table rows + macros). Formatting rule, applied uniformly:
statistics to 3 decimal places, p_FWER as value plus exceedance count, wall times to 0.1 s,
memory to 0.01 GB. Run: python make_tables.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
GEN = ROOT / "paper" / "generated"
AXES = ["syc", "unc", "cau", "per"]
NAMES = {"syc": "1. Sycophancy", "unc": "2. Uncertainty", "cau": "3. Caution", "per": "4. Persona"}
MACRO = {"syc": "Syc", "unc": "Unc", "cau": "Cau", "per": "Per"}


def f3(x: float) -> str:
    s = f"{x:.3f}"
    return "0.000" if s == "-0.000" else s


def m(x: float) -> str:  # math-mode number with a proper minus sign
    return f"${f3(x)}$"


def pm(d: dict) -> str:
    return f"${f3(d['mean'])} \\pm {f3(d['sd'])}$"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    s1 = {a: load(f"stage1_{a}.json") for a in AXES}
    s2 = {a: load(f"stage2_{a}.json") for a in AXES}
    res = load("resources.json")
    macros = []

    def mac(name, val):
        macros.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    # ---------------------------------------------------------------- Table 3 (Stage 1)
    rows = []
    for a in AXES:
        st = s1[a]["stats"]
        c = st["candidate"]
        passes = "Yes" if st["candidate_passes"] else "No"
        rows.append(
            f"{NAMES[a]} & $({c['layer']}, {c['neuron']})$ & {m(c['cohen_d'])} & {m(st['null_q99'])} & "
            f"${f3(c['p_fwer'])}$ ({c['null_exceedances']}/{st['n_permutations']}) & {m(c['col_norm'])} & "
            f"{st['n_passing_units']:,} & {passes} \\\\".replace(",", "{,}")
        )
        k = MACRO[a]
        mac(f"{k}Unit", f"$({c['layer']}, {c['neuron']})$")
        mac(f"{k}Layer", f"${c['layer']}$")
        mac(f"{k}D", m(c["cohen_d"]))
        mac(f"{k}Null", m(st["null_q99"]))
        mac(f"{k}NullNinetyFive", m(st["null_q95"]))
        mac(f"{k}P", f"${f3(c['p_fwer'])}$")
        mac(f"{k}Exceed", f"{c['null_exceedances']}/{st['n_permutations']}")
        mac(f"{k}NPass", f"{st['n_passing_units']:,}".replace(",", "{,}"))
        mac(f"{k}NActive", f"{st['n_active']:,}".replace(",", "{,}"))
        mac(f"{k}MinP", f"${f3(st['min_p_fwer'])}$")
        mac(f"{k}Verdict", s1[a]["verdict"])
        mac(f"{k}MeanPos", m(c["mean_pos"]))
        mac(f"{k}MeanNeg", m(c["mean_neg"]))
        mac(f"{k}FirePos", f"${f3(c['fire_pos'])}$")
        mac(f"{k}FireNeg", f"${f3(c['fire_neg'])}$")
        t = st["top_abs_d_unit"]
        mac(f"{k}TopDUnit", f"$({t['layer']}, {t['neuron']})$")
        mac(f"{k}TopD", m(t["cohen_d"]))
    (GEN / "stage1_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- Table 4 (Stage 2, decision dose)
    rows = []
    for a in AXES:
        r = s2[a]
        k = str(r["report_dose"])
        o, g, dec = r["ood_delta_logp"], r["distinct_2"]["summary"], r["decision"]
        u = r["unit"]
        tag = "" if not dec["exploratory"] else "$^{\\dagger}$"
        tgt, rnd, base = o["target"][k], o["random"][k], o["baseline"]
        effect = "Yes" if dec["causal_effect"] else "No"
        rows.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{NAMES[a]}{tag}, unit $({u['layer']}, {u['neuron']})$, "
                    f"verdict: {dec['verdict']}}}}} \\\\")
        rows.append(f"\\quad Unedited baseline & {pm(base)} & $0.000$ & --- & {pm(g['baseline'])} & --- \\\\")
        rows.append(f"\\quad Target neuron & {pm(tgt)} & {m(tgt['shift']['mean'])} & "
                    f"{m(tgt['mean'] - rnd['mean'])} & {pm(g['target'][k])} & {effect} \\\\")
        rows.append(f"\\quad Random control & {pm(rnd)} & {m(rnd['shift']['mean'])} & $0.000$ & "
                    f"{pm(g['random'][k])} & --- \\\\")
        if a != AXES[-1]:
            rows.append("\\addlinespace")
        km = MACRO[a]
        mac(f"{km}BaseMean", m(base["mean"]))
        mac(f"{km}BaseSD", m(base["sd"]))
        mac(f"{km}Shift", m(tgt["shift"]["mean"]))
        mac(f"{km}RandShift", m(rnd["shift"]["mean"]))
        mac(f"{km}Threshold", m(dec["threshold"]))
        mac(f"{km}TgtMinusRand", m(tgt["mean"] - rnd["mean"]))
        mac(f"{km}DistTgt", m(g["target"][k]["mean"]))
        mac(f"{km}DistBase", m(g["baseline"]["mean"]))
        mac(f"{km}DistRand", m(g["random"][k]["mean"]))
        mac(f"{km}StageTwoVerdict", dec["verdict"])
        mac(f"{km}Rho", m(r["rho_layer"]))
        mac(f"{km}PushValue", m(tgt["push_value"]))
        crit_names = {"stage1_p_lt_alpha": "CritStageOne", "shift_ge_2sd": "CritShift",
                      "beats_random": "CritRandom", "distinct2_ok": "CritFluency"}
        for crit, ok in dec["criteria"].items():
            mac(f"{km}{crit_names[crit]}", "met" if ok else "not met")
    (GEN / "stage2_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- dose sweep table
    rows = []
    dose_name = {"0.5": "Half", "1.0": "One", "1.5": "OneHalf"}
    for a in AXES:
        r = s2[a]
        o, g = r["ood_delta_logp"], r["distinct_2"]["summary"]
        for i, dose in enumerate(r["doses"]):
            k = str(dose)
            km, dn = MACRO[a], dose_name[k]
            mac(f"{km}Shift{dn}", m(o["target"][k]["shift"]["mean"]))
            mac(f"{km}RandShift{dn}", m(o["random"][k]["shift"]["mean"]))
            mac(f"{km}DistTgt{dn}", m(g["target"][k]["mean"]))
            mac(f"{km}DistRand{dn}", m(g["random"][k]["mean"]))
            lead = NAMES[a] if i == 0 else ""
            rows.append(f"{lead} & ${dose}$ & {m(o['target'][k]['shift']['mean'])} & "
                        f"{m(o['random'][k]['shift']['mean'])} & {pm(g['target'][k])} & {pm(g['random'][k])} \\\\")
        if a != AXES[-1]:
            rows.append("\\addlinespace")
    (GEN / "dose_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- in-distribution table
    rows = []
    for a in AXES:
        ind = s2[a]["in_distribution"]
        e, s = ind["elicitation_on_neg"], ind["suppression_on_pos"]
        rows.append(f"{NAMES[a]} & {pm(e['target_shift'])} & {pm(e['random_shift'])} & "
                    f"{pm(s['target_shift'])} & {pm(s['random_shift'])} \\\\")
        km = MACRO[a]
        mac(f"{km}Elicit", m(e["target_shift"]["mean"]))
        mac(f"{km}ElicitRand", m(e["random_shift"]["mean"]))
        mac(f"{km}Suppress", m(s["target_shift"]["mean"]))
        mac(f"{km}SuppressRand", m(s["random_shift"]["mean"]))
    (GEN / "indist_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- resources table
    gb = res["weight_bytes"] / 1e9
    st1 = res["stage1_axes"]
    rows = [
        f"Model weights (BF16, {res['n_params']:,} parameters) & {gb:.2f}~GB & --- \\\\".replace(",", "{,}"),
        f"Model load & --- & {res['model_load_s']:.1f}~s \\\\",
    ]
    for a in AXES:
        rows.append(f"Stage 1: {NAMES[a][3:]} & --- & {st1[a]['wall_s']:.1f}~s \\\\")
    rows.append(f"Stage 1: full run incl.\\ model load (peak RSS) & {res['stage1_peak_rss_gb']:.2f}~GB & "
                f"{res['stage1_total_s']:.1f}~s \\\\")
    rows.append(f"Stage 2: full run, {res['stage2_generation_runs']} generation runs $\\times$ "
                f"{res['stage2_prompts_per_run']} prompts (peak RSS) & {res['stage2_peak_rss_gb']:.2f}~GB & "
                f"{res['stage2_total_s']:.1f}~s \\\\")
    (GEN / "resources_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    mac("WeightGB", f"{gb:.2f}")
    mac("NParams", f"{res['n_params']:,}".replace(",", "{,}"))
    mac("StageOneTotal", f"{res['stage1_total_s']:.1f}")
    mac("StageTwoTotal", f"{res['stage2_total_s']:.1f}")
    mac("StageTwoRuns", str(res["stage2_generation_runs"]))
    env = s2["per"]["env"]
    mac("ModelCommit", "\\texttt{" + env["model_commit"][:12] + "}")
    mac("TorchVersion", env["torch"].replace("+", "\\texttt{+}"))
    mac("TransformersVersion", env["transformers"])

    # ---------------------------------------------------------------- token pieces (Appendix A)
    rows = []
    for a in AXES:
        tm = s2[a]["token_map"]
        split = [f"\\texttt{{{w}}}$\\to$" + "+".join(f"\\texttt{{{p}}}" for p in ps)
                 for kind in ("target_pieces", "neutral_pieces") for w, ps in tm[kind].items() if len(ps) > 1]
        rows.append(f"{NAMES[a]} & {', '.join(split) if split else 'none (all single tokens)'} \\\\")
    (GEN / "token_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    (GEN / "macros.tex").write_text("% Generated by make_tables.py from outputs/*.json. Do not edit.\n"
                                    + "\n".join(macros) + "\n", encoding="utf-8")
    write_runlog_block(macros)
    print(f"wrote {len(list(GEN.glob('*.tex')))} files, {len(macros)} macros to {GEN}")


def write_runlog_block(macros: list[str]) -> None:
    """Keep one current copy of every manuscript number in RUNLOG.md (replaced on each run)."""
    start, end = "<!-- manuscript-numbers:start -->", "<!-- manuscript-numbers:end -->"
    lines = [start, "## Numbers printed in the manuscript", "",
             "Written by `make_tables.py` from `outputs/*.json` (3-decimal rounding; the only derived quantity is "
             "`*TgtMinusRand` = mean target ΔlogP − mean random-control ΔlogP). Table rows: `paper/generated/*.tex`.",
             "", "| Macro | Value |", "|---|---|"]
    for mline in macros:
        name = mline.split("{", 1)[1].split("}", 1)[0]
        val = mline[len("\\newcommand{") + len(name) + 2:-1]
        lines.append(f"| `{name}` | `{val}` |")
    for f in sorted(GEN.glob("*_rows.tex")):
        lines += ["", f"### Table rows: `paper/generated/{f.name}`", "", "```latex", f.read_text(encoding="utf-8").rstrip(), "```"]
    lines.append(end)
    block = "\n".join(lines)
    log = ROOT / "RUNLOG.md"
    text = log.read_text(encoding="utf-8")
    if start in text:
        text = text[:text.index(start)] + block + text[text.index(end) + len(end):]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    log.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
