# J++ Lens improvement sprint (Qwen3-0.6B)

Hypothesis-first sprint on the J++ Lens (`safety-research/jpp_lens` @ `d7efc0d`).

- `PLAN.md`: scope, the evaluation suite, the execution plan and the Experiment 1 pre-registration
  (thresholds fixed before any Exp 1 code).
- `RUNLOG.md`: append-only log of every run.

## Code

- `harness.py`: the four metric families. M1 readout recall@10 / MRR by layer stratum. M2 false
  positives on matched controls and WikiText. M3 transport alignment on held-out WikiText.
- `swap_check.py`: M4 causal swaps.
- `jppsearch.py`: J++'s weight search, the same algorithm as jpp_lens with an exact analytic
  gradient. It adds an optional ridge toward a reference weighting and custom starts.
- `sprintfit.py`: fitting inputs and fitted maps.
- `baseline.py`: E0.2 metric_before on the tuning splits.
- `diag_spectral.py`: E0.1 label-free split-half reliability of expert deviations by singular band
  (the Exp 1 pre-check).
- `collect_heldout.py`: held-out WikiText activations (records 7000-7127).
- `fidelity/`: E0.1 checks against jpp_lens itself.
  - `jpp_side.py` + `compare_lrp.py`: the LRP expert estimator.
  - `weights_ours.py` + `weights_jpp.py`: the weight search.
  - `search_noise.py`: how much held-out recall moves between two runs of the same search algorithm.

## Environment

jpp_lens is cloned at `/home/user/safety-research/jpp_lens`. Its CPU tests run with:

```
uv sync --extra dev
uv run --no-sync pytest -q
```

That environment needs torch from the host venv (see RUNLOG E0.0). The PyTorch CPU wheel index is
blocked by the network policy here.
