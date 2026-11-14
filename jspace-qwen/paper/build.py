"""Assemble paper.md from the fragments (the fragments are the source of truth).

python paper/build.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["head.md", "intro.md", "related_work.md", "methods_head.md", "method_broadcast.md",
         "method_readouts.md", "method_estimators.md", "method_negative.md", "method_candidates.md", "setup.md",
         "results_repro.md", "results_jpp.md", "results_round2.md", "results_swaps.md", "results_pending.md", "limitations.md",
         "conclusion.md", "references.md"]
parts = []
for name in ORDER:
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        continue
    text = open(path).read()
    text = re.sub(r"^\s*<!--.*?-->\s*", "", text, flags=re.S)
    parts.append(text.strip())
open(os.path.join(HERE, "paper.md"), "w").write("\n\n".join(parts) + "\n")
print("wrote paper/paper.md:", sum(len(p.split()) for p in parts), "words")
