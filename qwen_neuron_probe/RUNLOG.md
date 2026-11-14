# RUNLOG

Machine-written log of every executed run. Numbers here are copied by the scripts from `outputs/*.json`; nothing in this file is typed by hand.

## Stage 1 discovery (2026-11-14T01:59:26+00:00)

Command: `python discover_neurons.py --config config.yaml`

- utc: `2026-11-14T01:58:34+00:00`
- repo_commit: `3b399ad52c566b27beb5e4b008a62920c88a89b8`
- model_commit: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- python: `3.13.18`
- torch: `2.15.0+cpu`
- transformers: `5.23.0`
- numpy: `2.5.4`
- device: `cpu`
- cpu: `x86_64`
- torch_threads: `4`
- cuda_available: `False`
- dtype: `bfloat16`; parameters: `1720574976`; weight bytes: `3441149952`
- model load: `2.3 s`; Stage 1 total wall time: `55.2 s`; peak RSS: `4.115 GB`

Candidate = highest composite score S among units with p_FWER < 0.01 (top-S unit if none pass).

| Axis | Candidate | Cohen's d | Null 99% |d| | p_FWER (exceed/B) | ‖w_down‖ | # passing units | min p_FWER | Verdict | Wall s |
|---|---|---|---|---|---|---|---|---|---|
| syc | L13 N5085 | 7.738349 | 2.043599 | 0.000000 (0/200) | 1.810718 | 11145 | 0.000000 | pass | 15.2 |
| unc | L19 N566 | 4.366607 | 2.136885 | 0.000000 (0/200) | 1.640751 | 747 | 0.000000 | pass | 11.4 |
| cau | L0 N100 | 1.757638 | 1.899894 | 0.060000 (12/200) | 1.532066 | 0 | 0.060000 | refuted | 12.1 |
| per | L19 N3950 | -6.064071 | 2.081447 | 0.000000 (0/200) | 1.758680 | 5576 | 0.000000 | pass | 12.9 |
