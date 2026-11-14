"""jspace: a compact, CPU-friendly reproduction toolkit for the Jacobian lens.

Reproduces the estimator of Gurnee et al. (2026), "Verbalizable Representations
Form a Global Workspace in Language Models" (reference code:
github.com/anthropics/jacobian-lens, Apache-2.0), and adds:

* randomized, unbiased sketch estimators of the average Jacobian with control
  variates (``jspace.fitting``), including a readout-aware importance sketch;
* calibrated (per-token standardised) lens readouts (``jspace.lens``);
* readout and causal-swap evaluations with matched controls and bootstrap CIs
  (``jspace.evals``, ``jspace.interventions``).
"""

from jspace.model import LensModel, load_model  # noqa: F401
from jspace.lens import Lens  # noqa: F401
