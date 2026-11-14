# J-space on Qwen3-0.6B: a CPU-budget reproduction of the Jacobian lens, with sketch estimators

Reproduction and extension of Gurnee et al. (2026), *Verbalizable Representations Form a Global
Workspace in Language Models* ([paper](https://transformer-circuits.pub/2026/workspace/index.html),
[reference code](https://github.com/anthropics/jacobian-lens)), on a small open-weight Qwen model,
under a CPU-only budget (4 cores, 15 GB RAM).

- [`PROMPT.md`](PROMPT.md): the original request, what was underspecified about it, and the
  improved research spec this project executes.
- [`paper/`](paper/): the paper draft.

## Status

| Component | State |
|---|---|
| Exact J-lens estimator (reference reduction) | done; matches brute-force autograd to ~1e-7 |
| Sketch estimators (Haar probes; none / identity / online control variate; readout-aware shaping) | done; unbiased, error matches theory |
| Readout eval (6 released sets, recall@k, correctness filter, bootstrap CIs, Readout Filtering, calibrated readouts) | done; smoke-tested |
| Regression-lens baseline, per-token calibration | done; smoke-tested |
| Clamp-swap causal evals with logit-lens / random-null / answer-swap controls | done; clamp verified exact |
| Ignition (embedding interpolation) | done; smoke-tested |
| Experiments on Qwen3-0.6B | **waiting on network access to huggingface.co** |

## Layout

```
jspace/            library
  model.py         HF wrapper, recording/editing hooks, fp32 unembedding
  fitting.py       exact + sketch estimators of E[dh_final/dh_l]
  lens.py          logit / J / regression lenses, per-token calibration
  evals.py         readout evals (recall@k, bootstrap, paired differences)
  interventions.py clamp swap
  corpus.py        WikiText-103 fitting corpus, activation collection
scripts/
  test_estimators.py   correctness tests (no downloads)
  bench.py             CPU throughput at Qwen3-0.6B shape (no downloads)
  make_smoke_fixture.py tiny offline model + tokenizer for end-to-end smoke tests
  fit.py collect.py readout_eval.py swaps.py ignition.py
data/              prompt sets from anthropics/jacobian-lens (Apache-2.0, see LICENSE-jacobian-lens)
results/           JSON results (committed); lenses/ holds fitted matrices (not committed)
```

## Reproduce

```bash
pip install torch "transformers>=5.5" datasets numpy scipy matplotlib
python scripts/test_estimators.py                      # correctness, ~1 min
python scripts/fit.py --method exact --n 48 --tag exact --snapshots 1 2 4 8 16 32 48
python scripts/collect.py --n 256 --offset 5000
python scripts/readout_eval.py --lens jlens=lenses/exact_final.pt --regress --calib --filter --tag main
```

Measured cost on this machine (bf16 with AMX): an exact prompt (1,024 VJPs over 128 tokens) takes
about 3.2 min; a K = 16 sketch prompt takes about 2.5 s.
