"""Extract the 80-pair benchmark (Appendix B) and per-axis token sets (Appendix A) from
paper/main.tex into datasets/, and run structural checks. Usage: python extract_benchmark.py ."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
tex = (root / "paper" / "main.tex").read_text(encoding="utf-8")


def tex_to_text(s: str) -> str:
    s = s.strip()
    s = s.replace("---", "\u2014").replace("\\&", "&")
    if "\\" in s or "{" in s:
        raise ValueError(f"Unhandled LaTeX in cell: {s!r}")
    return s


def env_body(label: str) -> str:
    """Return the tabular(x) body of the table carrying \\label{label}."""
    i = tex.index(f"\\label{{{label}}}")
    start = tex.index("\\midrule", i) + len("\\midrule")
    end = tex.index("\\bottomrule", start)
    return tex[start:end]


def rows(body: str) -> list[list[str]]:
    out = []
    for raw in body.split("\\\\"):
        raw = raw.strip()
        if raw:
            out.append([c.strip() for c in raw.split(" & ")])
    return out


report = {}

# --- structure -------------------------------------------------------------
labels = re.findall(r"\\label\{([^}]+)\}", tex)
refs = re.findall(r"\\ref\{([^}]+)\}", tex)
cites = [c.strip() for grp in re.findall(r"\\cite\{([^}]+)\}", tex) for c in grp.split(",")]
bibitems = re.findall(r"\\bibitem\{([^}]+)\}", tex)
begins = Counter(re.findall(r"\\begin\{(\w+)\}", tex))
ends = Counter(re.findall(r"\\end\{(\w+)\}", tex))
report["sections"] = re.findall(r"\\(section|subsection)\{([^}]+)\}", tex)
report["duplicate_labels"] = [l for l, n in Counter(labels).items() if n > 1]
report["undefined_refs"] = sorted(set(refs) - set(labels))
report["unreferenced_labels"] = sorted(set(labels) - set(refs))
report["undefined_cites"] = sorted(set(cites) - set(bibitems))
report["uncited_bibitems"] = sorted(set(bibitems) - set(cites))
report["unbalanced_envs"] = {k: (begins[k], ends[k]) for k in set(begins) | set(ends) if begins[k] != ends[k]}

# --- placeholders ----------------------------------------------------------
pend = {}
for lab in ("tab:stage1_schema", "tab:stage2_schema"):
    body = env_body(lab)
    pend[lab] = body.count("[pending]")
report["pending_placeholders"] = pend
report["pending_total_in_file"] = tex.count("[pending]")

# numbers stated as results outside the [pending] schema
vram = rows(env_body("tab:vram"))
report["tab_vram_rows"] = vram

# --- token sets ------------------------------------------------------------
axes = {"1": "syc", "2": "unc", "3": "cau", "4": "per"}
token_sets = {}
for r in rows(env_body("tab:tokens")):
    axis_num = r[0].split(".")[0]
    grab = lambda c: [t.strip() for t in re.search(r"\\texttt\{([^}]*)\}", c).group(1).split(",")]
    token_sets[axes[axis_num]] = {"target_tokens": grab(r[1]), "neutral_tokens": grab(r[2])}
report["token_sets"] = token_sets

# --- benchmark extraction ----------------------------------------------------
out_dir = root / "datasets"
out_dir.mkdir(exist_ok=True)
checks = {}
for num, code in axes.items():
    pairs = rows(env_body(f"tab:axis{num}"))
    assert len(pairs) == 20, (code, len(pairs))
    recs, diffs, bad_q = [], [], []
    for idx, (k, pos, neg) in enumerate(pairs, start=1):
        assert int(k) == idx, (code, k, idx)
        pos_t, neg_t = tex_to_text(pos), tex_to_text(neg)
        for pol, txt in (("pos", pos_t), ("neg", neg_t)):
            recs.append({"id": f"{code}_{pol}{idx:02d}", "axis": code, "label": 1 if pol == "pos" else 0,
                         "pair": idx, "prompt": txt})
            if not txt.endswith("?"):
                bad_q.append(f"{code}_{pol}{idx:02d}")
        # word count after splitting em-dash joins, as the paper's "1-3 words" claim implies
        wc = lambda t: len(t.replace("\u2014", " ").split())
        diffs.append((idx, wc(pos_t), wc(neg_t), abs(wc(pos_t) - wc(neg_t))))
    with open(out_dir / f"{code}.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    checks[code] = {
        "n_prompts": len(recs),
        "unique_prompts": len({r["prompt"] for r in recs}),
        "not_ending_in_qmark": bad_q,
        "word_diff_max": max(d[3] for d in diffs),
        "pairs_over_3_words": [(i, p, n) for i, p, n, d in diffs if d > 3],
        "pairs_within_1_3": sum(1 for d in diffs if 1 <= d[3] <= 3),
        "pairs_zero_diff": sum(1 for d in diffs if d[3] == 0),
    }
report["benchmark_checks"] = checks

with open(out_dir / "token_sets.json", "w", encoding="utf-8") as f:
    json.dump(token_sets, f, indent=2)

# --- OOD suite keyword audit (paper claims zero persona/dual-use keywords) ----
kw = re.compile(r"personalit|persona|villain|crypto|secur|safety|hack|weapon|attack|kill", re.I)
ood = [json.loads(l) for l in open(root / "prompts.jsonl", encoding="utf-8") if l.strip()]
report["ood_keyword_hits"] = [(o["id"], o["prompt"]) for o in ood if kw.search(o["prompt"])]

print(json.dumps(report, indent=1, ensure_ascii=False))
