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

## Stage 2 causal steering (2026-11-14T02:30:03+00:00)

Command: `python run_stage2.py --config config.yaml`

- utc: `2026-11-14T02:03:05+00:00`
- repo_commit: `237296845fa5cc50d90e1f6f36cbceeab14cfbab`
- model_commit: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- python: `3.13.18`
- torch: `2.15.0+cpu`
- transformers: `5.23.0`
- numpy: `2.5.4`
- device: `cpu`
- cpu: `x86_64`
- torch_threads: `4`
- cuda_available: `False`
- doses: `[0.5, 1.0, 1.5]` (decision dose α = 1.0); seeds: `[0, 1, 2]`; decoding: T=0.7, top_p=0.9, 64 new tokens
- Stage 2 total wall time: `1620.0 s`; peak RSS: `5.853 GB`
- ρ_ℓ (median post-layer residual ℓ2 norm on P_OOD) saved to `outputs/calibration.json`

OOD ΔlogP values are means over the 16 OOD prompts (SD across prompts); random control = mean over 3 seeds, each with its own random unit in layer ℓ*. distinct_2 = mean over seeds.

| Axis | Unit | Baseline ΔlogP ± SD | Target ΔlogP | Target shift | Random shift | d2 base | d2 target | d2 random | Elicit shift (neg) | Suppress shift (pos) | Verdict | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syc | L13 N5085 | 5.035637 ± 5.766296 | 2.964903 | -2.070734 | -6.454043 | 0.916824 | 0.538195 | 0.413755 | 14.303770 | -9.308071 | refuted (Stage 2) | 344.6 |
| unc | L19 N566 | -7.404608 ± 9.953942 | -1.076411 | 6.328197 | 2.240910 | 0.916824 | 0.814014 | 0.727720 | 14.329837 | -3.501294 | refuted (Stage 2) | 411.1 |
| cau | L0 N100 | -0.565112 ± 10.400734 | -1.657874 | -1.092762 | -1.502835 | 0.916824 | 0.366406 | 0.689199 | 2.429108 | 1.189938 | refuted (Stage 1) | 351.4 |
| per | L19 N3950 | -17.271407 ± 9.347778 | -16.279142 | 0.992265 | 4.826099 | 0.916824 | 0.855140 | 0.864519 | -2.765946 | -1.632722 | refuted (Stage 2) | 456.4 |

## Hardware (2026-11-14T02:34:14+00:00)

Read from `lscpu` and `/proc/meminfo` on the machine that executed both stages.

- CPU model: `Intel(R) Xeon(R) Processor @ 2.10GHz`
- logical CPUs: `4`
- amx_bf16: `True`; avx512_bf16: `True`
- MemTotal: `16473776 kB` (`15.7 GiB`)
- NVIDIA GPU present: `False`

<!-- manuscript-numbers:start -->
## Numbers printed in the manuscript

Written by `make_tables.py` from `outputs/*.json` (3-decimal rounding; the only derived quantity is `*TgtMinusRand` = mean target ΔlogP − mean random-control ΔlogP). Table rows: `paper/generated/*.tex`.

| Macro | Value |
|---|---|
| `\SycUnit` | `$(13, 5085)$` |
| `\SycLayer` | `$13$` |
| `\SycD` | `$7.738$` |
| `\SycNull` | `$2.044$` |
| `\SycNullNinetyFive` | `$1.804$` |
| `\SycP` | `$0.000$` |
| `\SycExceed` | `0/200` |
| `\SycNPass` | `11{,}145` |
| `\SycNActive` | `116{,}882` |
| `\SycMinP` | `$0.000$` |
| `\SycVerdict` | `pass` |
| `\SycMeanPos` | `$-0.018$` |
| `\SycMeanNeg` | `$-0.265$` |
| `\SycFirePos` | `$0.300$` |
| `\SycFireNeg` | `$0.000$` |
| `\SycTopDUnit` | `$(13, 5085)$` |
| `\SycTopD` | `$7.738$` |
| `\UncUnit` | `$(19, 566)$` |
| `\UncLayer` | `$19$` |
| `\UncD` | `$4.367$` |
| `\UncNull` | `$2.137$` |
| `\UncNullNinetyFive` | `$1.880$` |
| `\UncP` | `$0.000$` |
| `\UncExceed` | `0/200` |
| `\UncNPass` | `747` |
| `\UncNActive` | `120{,}686` |
| `\UncMinP` | `$0.000$` |
| `\UncVerdict` | `pass` |
| `\UncMeanPos` | `$29.528$` |
| `\UncMeanNeg` | `$1.686$` |
| `\UncFirePos` | `$1.000$` |
| `\UncFireNeg` | `$0.950$` |
| `\UncTopDUnit` | `$(19, 566)$` |
| `\UncTopD` | `$4.367$` |
| `\CauUnit` | `$(0, 100)$` |
| `\CauLayer` | `$0$` |
| `\CauD` | `$1.758$` |
| `\CauNull` | `$1.900$` |
| `\CauNullNinetyFive` | `$1.792$` |
| `\CauP` | `$0.060$` |
| `\CauExceed` | `12/200` |
| `\CauNPass` | `0` |
| `\CauNActive` | `114{,}393` |
| `\CauMinP` | `$0.060$` |
| `\CauVerdict` | `refuted` |
| `\CauMeanPos` | `$0.017$` |
| `\CauMeanNeg` | `$0.012$` |
| `\CauFirePos` | `$1.000$` |
| `\CauFireNeg` | `$1.000$` |
| `\CauTopDUnit` | `$(0, 100)$` |
| `\CauTopD` | `$1.758$` |
| `\PerUnit` | `$(19, 3950)$` |
| `\PerLayer` | `$19$` |
| `\PerD` | `$-6.064$` |
| `\PerNull` | `$2.081$` |
| `\PerNullNinetyFive` | `$1.874$` |
| `\PerP` | `$0.000$` |
| `\PerExceed` | `0/200` |
| `\PerNPass` | `5{,}576` |
| `\PerNActive` | `120{,}463` |
| `\PerMinP` | `$0.000$` |
| `\PerVerdict` | `pass` |
| `\PerMeanPos` | `$0.221$` |
| `\PerMeanNeg` | `$0.771$` |
| `\PerFirePos` | `$1.000$` |
| `\PerFireNeg` | `$1.000$` |
| `\PerTopDUnit` | `$(23, 3812)$` |
| `\PerTopD` | `$-6.802$` |
| `\SycBaseMean` | `$5.036$` |
| `\SycBaseSD` | `$5.766$` |
| `\SycShift` | `$-2.071$` |
| `\SycRandShift` | `$-6.454$` |
| `\SycThreshold` | `$11.533$` |
| `\SycTgtMinusRand` | `$4.383$` |
| `\SycDistTgt` | `$0.538$` |
| `\SycDistBase` | `$0.917$` |
| `\SycDistRand` | `$0.414$` |
| `\SycStageTwoVerdict` | `refuted (Stage 2)` |
| `\SycRho` | `$171.960$` |
| `\SycPushValue` | `$94.968$` |
| `\SycCritStageOne` | `met` |
| `\SycCritShift` | `not met` |
| `\SycCritRandom` | `met` |
| `\SycCritFluency` | `met` |
| `\UncBaseMean` | `$-7.405$` |
| `\UncBaseSD` | `$9.954$` |
| `\UncShift` | `$6.328$` |
| `\UncRandShift` | `$2.241$` |
| `\UncThreshold` | `$19.908$` |
| `\UncTgtMinusRand` | `$4.087$` |
| `\UncDistTgt` | `$0.814$` |
| `\UncDistBase` | `$0.917$` |
| `\UncDistRand` | `$0.728$` |
| `\UncStageTwoVerdict` | `refuted (Stage 2)` |
| `\UncRho` | `$633.604$` |
| `\UncPushValue` | `$386.166$` |
| `\UncCritStageOne` | `met` |
| `\UncCritShift` | `not met` |
| `\UncCritRandom` | `met` |
| `\UncCritFluency` | `met` |
| `\CauBaseMean` | `$-0.565$` |
| `\CauBaseSD` | `$10.401$` |
| `\CauShift` | `$-1.093$` |
| `\CauRandShift` | `$-1.503$` |
| `\CauThreshold` | `$20.801$` |
| `\CauTgtMinusRand` | `$0.410$` |
| `\CauDistTgt` | `$0.366$` |
| `\CauDistBase` | `$0.917$` |
| `\CauDistRand` | `$0.689$` |
| `\CauStageTwoVerdict` | `refuted (Stage 1)` |
| `\CauRho` | `$18.029$` |
| `\CauPushValue` | `$11.768$` |
| `\CauCritStageOne` | `not met` |
| `\CauCritShift` | `not met` |
| `\CauCritRandom` | `met` |
| `\CauCritFluency` | `not met` |
| `\PerBaseMean` | `$-17.271$` |
| `\PerBaseSD` | `$9.348$` |
| `\PerShift` | `$0.992$` |
| `\PerRandShift` | `$4.826$` |
| `\PerThreshold` | `$18.696$` |
| `\PerTgtMinusRand` | `$-3.834$` |
| `\PerDistTgt` | `$0.855$` |
| `\PerDistBase` | `$0.917$` |
| `\PerDistRand` | `$0.865$` |
| `\PerStageTwoVerdict` | `refuted (Stage 2)` |
| `\PerRho` | `$633.604$` |
| `\PerPushValue` | `$-360.272$` |
| `\PerCritStageOne` | `met` |
| `\PerCritShift` | `not met` |
| `\PerCritRandom` | `not met` |
| `\PerCritFluency` | `met` |
| `\SycShiftHalf` | `$0.965$` |
| `\SycRandShiftHalf` | `$-1.866$` |
| `\SycDistTgtHalf` | `$0.891$` |
| `\SycDistRandHalf` | `$0.874$` |
| `\SycShiftOne` | `$-2.071$` |
| `\SycRandShiftOne` | `$-6.454$` |
| `\SycDistTgtOne` | `$0.538$` |
| `\SycDistRandOne` | `$0.414$` |
| `\SycShiftOneHalf` | `$-2.759$` |
| `\SycRandShiftOneHalf` | `$-5.328$` |
| `\SycDistTgtOneHalf` | `$0.359$` |
| `\SycDistRandOneHalf` | `$0.349$` |
| `\UncShiftHalf` | `$3.295$` |
| `\UncRandShiftHalf` | `$0.746$` |
| `\UncDistTgtHalf` | `$0.903$` |
| `\UncDistRandHalf` | `$0.914$` |
| `\UncShiftOne` | `$6.328$` |
| `\UncRandShiftOne` | `$2.241$` |
| `\UncDistTgtOne` | `$0.814$` |
| `\UncDistRandOne` | `$0.728$` |
| `\UncShiftOneHalf` | `$9.343$` |
| `\UncRandShiftOneHalf` | `$3.691$` |
| `\UncDistTgtOneHalf` | `$0.295$` |
| `\UncDistRandOneHalf` | `$0.374$` |
| `\CauShiftHalf` | `$-0.832$` |
| `\CauRandShiftHalf` | `$-0.541$` |
| `\CauDistTgtHalf` | `$0.876$` |
| `\CauDistRandHalf` | `$0.909$` |
| `\CauShiftOne` | `$-1.093$` |
| `\CauRandShiftOne` | `$-1.503$` |
| `\CauDistTgtOne` | `$0.366$` |
| `\CauDistRandOne` | `$0.689$` |
| `\CauShiftOneHalf` | `$-2.451$` |
| `\CauRandShiftOneHalf` | `$0.520$` |
| `\CauDistTgtOneHalf` | `$0.339$` |
| `\CauDistRandOneHalf` | `$0.084$` |
| `\PerShiftHalf` | `$-0.267$` |
| `\PerRandShiftHalf` | `$1.649$` |
| `\PerDistTgtHalf` | `$0.909$` |
| `\PerDistRandHalf` | `$0.923$` |
| `\PerShiftOne` | `$0.992$` |
| `\PerRandShiftOne` | `$4.826$` |
| `\PerDistTgtOne` | `$0.855$` |
| `\PerDistRandOne` | `$0.865$` |
| `\PerShiftOneHalf` | `$3.219$` |
| `\PerRandShiftOneHalf` | `$6.507$` |
| `\PerDistTgtOneHalf` | `$0.597$` |
| `\PerDistRandOneHalf` | `$0.557$` |
| `\SycElicit` | `$14.304$` |
| `\SycElicitRand` | `$10.153$` |
| `\SycSuppress` | `$-9.308$` |
| `\SycSuppressRand` | `$-2.072$` |
| `\UncElicit` | `$14.330$` |
| `\UncElicitRand` | `$7.955$` |
| `\UncSuppress` | `$-3.501$` |
| `\UncSuppressRand` | `$-0.483$` |
| `\CauElicit` | `$2.429$` |
| `\CauElicitRand` | `$2.460$` |
| `\CauSuppress` | `$1.190$` |
| `\CauSuppressRand` | `$3.345$` |
| `\PerElicit` | `$-2.766$` |
| `\PerElicitRand` | `$0.549$` |
| `\PerSuppress` | `$-1.633$` |
| `\PerSuppressRand` | `$-3.435$` |
| `\WeightGB` | `3.44` |
| `\NParams` | `1{,}720{,}574{,}976` |
| `\StageOneTotal` | `55.2` |
| `\StageTwoTotal` | `1620.0` |
| `\StageTwoRuns` | `75` |
| `\ModelCommit` | `\texttt{70d244cc86cc}` |
| `\TorchVersion` | `2.15.0\texttt{+}cpu` |
| `\TransformersVersion` | `5.23.0` |

### Table rows: `paper/generated/dose_rows.tex`

```latex
1. Sycophancy & $0.5$ & $0.965$ & $-1.866$ & $0.891 \pm 0.009$ & $0.874 \pm 0.036$ \\
 & $1.0$ & $-2.071$ & $-6.454$ & $0.538 \pm 0.009$ & $0.414 \pm 0.086$ \\
 & $1.5$ & $-2.759$ & $-5.328$ & $0.359 \pm 0.011$ & $0.349 \pm 0.257$ \\
\addlinespace
2. Uncertainty & $0.5$ & $3.295$ & $0.746$ & $0.903 \pm 0.013$ & $0.914 \pm 0.011$ \\
 & $1.0$ & $6.328$ & $2.241$ & $0.814 \pm 0.024$ & $0.728 \pm 0.247$ \\
 & $1.5$ & $9.343$ & $3.691$ & $0.295 \pm 0.058$ & $0.374 \pm 0.184$ \\
\addlinespace
3. Caution & $0.5$ & $-0.832$ & $-0.541$ & $0.876 \pm 0.035$ & $0.909 \pm 0.013$ \\
 & $1.0$ & $-1.093$ & $-1.503$ & $0.366 \pm 0.058$ & $0.689 \pm 0.160$ \\
 & $1.5$ & $-2.451$ & $0.520$ & $0.339 \pm 0.033$ & $0.084 \pm 0.065$ \\
\addlinespace
4. Persona & $0.5$ & $-0.267$ & $1.649$ & $0.909 \pm 0.003$ & $0.923 \pm 0.009$ \\
 & $1.0$ & $0.992$ & $4.826$ & $0.855 \pm 0.031$ & $0.865 \pm 0.030$ \\
 & $1.5$ & $3.219$ & $6.507$ & $0.597 \pm 0.012$ & $0.557 \pm 0.133$ \\
```

### Table rows: `paper/generated/indist_rows.tex`

```latex
1. Sycophancy & $14.304 \pm 5.960$ & $10.153 \pm 6.902$ & $-9.308 \pm 5.465$ & $-2.072 \pm 5.808$ \\
2. Uncertainty & $14.330 \pm 4.303$ & $7.955 \pm 5.079$ & $-3.501 \pm 2.814$ & $-0.483 \pm 2.159$ \\
3. Caution & $2.429 \pm 4.263$ & $2.460 \pm 3.735$ & $1.190 \pm 4.213$ & $3.345 \pm 3.888$ \\
4. Persona & $-2.766 \pm 2.486$ & $0.549 \pm 2.840$ & $-1.633 \pm 1.632$ & $-3.435 \pm 1.484$ \\
```

### Table rows: `paper/generated/resources_rows.tex`

```latex
Model weights (BF16{,} 1{,}720{,}574{,}976 parameters) & 3.44~GB & --- \\
Model load & --- & 2.3~s \\
Stage 1: Sycophancy & --- & 15.2~s \\
Stage 1: Uncertainty & --- & 11.4~s \\
Stage 1: Caution & --- & 12.1~s \\
Stage 1: Persona & --- & 12.9~s \\
Stage 1: full run incl.\ model load (peak RSS) & 4.11~GB & 55.2~s \\
Stage 2: full run, 75 generation runs $\times$ 16 prompts (peak RSS) & 5.85~GB & 1620.0~s \\
```

### Table rows: `paper/generated/stage1_rows.tex`

```latex
1. Sycophancy & $(13{,} 5085)$ & $7.738$ & $2.044$ & $0.000$ (0/200) & $1.811$ & 11{,}145 & Yes \\
2. Uncertainty & $(19{,} 566)$ & $4.367$ & $2.137$ & $0.000$ (0/200) & $1.641$ & 747 & Yes \\
3. Caution & $(0{,} 100)$ & $1.758$ & $1.900$ & $0.060$ (12/200) & $1.532$ & 0 & No \\
4. Persona & $(19{,} 3950)$ & $-6.064$ & $2.081$ & $0.000$ (0/200) & $1.759$ & 5{,}576 & Yes \\
```

### Table rows: `paper/generated/stage2_rows.tex`

```latex
\multicolumn{6}{@{}l}{\emph{1. Sycophancy, unit $(13, 5085)$, verdict: refuted (Stage 2)}} \\
\quad Unedited baseline & $5.036 \pm 5.766$ & $0.000$ & --- & $0.917 \pm 0.005$ & --- \\
\quad Target neuron & $2.965 \pm 2.519$ & $-2.071$ & $4.383$ & $0.538 \pm 0.009$ & No \\
\quad Random control & $-1.418 \pm 1.185$ & $-6.454$ & $0.000$ & $0.414 \pm 0.086$ & --- \\
\addlinespace
\multicolumn{6}{@{}l}{\emph{2. Uncertainty, unit $(19, 566)$, verdict: refuted (Stage 2)}} \\
\quad Unedited baseline & $-7.405 \pm 9.954$ & $0.000$ & --- & $0.917 \pm 0.005$ & --- \\
\quad Target neuron & $-1.076 \pm 7.324$ & $6.328$ & $4.087$ & $0.814 \pm 0.024$ & No \\
\quad Random control & $-5.164 \pm 6.148$ & $2.241$ & $0.000$ & $0.728 \pm 0.247$ & --- \\
\addlinespace
\multicolumn{6}{@{}l}{\emph{3. Caution$^{\dagger}$, unit $(0, 100)$, verdict: refuted (Stage 1)}} \\
\quad Unedited baseline & $-0.565 \pm 10.401$ & $0.000$ & --- & $0.917 \pm 0.005$ & --- \\
\quad Target neuron & $-1.658 \pm 1.364$ & $-1.093$ & $0.410$ & $0.366 \pm 0.058$ & No \\
\quad Random control & $-2.068 \pm 3.790$ & $-1.503$ & $0.000$ & $0.689 \pm 0.160$ & --- \\
\addlinespace
\multicolumn{6}{@{}l}{\emph{4. Persona, unit $(19, 3950)$, verdict: refuted (Stage 2)}} \\
\quad Unedited baseline & $-17.271 \pm 9.348$ & $0.000$ & --- & $0.917 \pm 0.005$ & --- \\
\quad Target neuron & $-16.279 \pm 7.973$ & $0.992$ & $-3.834$ & $0.855 \pm 0.031$ & No \\
\quad Random control & $-12.445 \pm 6.120$ & $4.826$ & $0.000$ & $0.865 \pm 0.030$ & --- \\
```

### Table rows: `paper/generated/token_rows.tex`

```latex
1. Sycophancy & none (all single tokens) \\
2. Uncertainty & \texttt{Predicting}$\to$\texttt{Predict}+\texttt{ing}, \texttt{Approximately}$\to$\texttt{Approx}+\texttt{imately} \\
3. Caution & none (all single tokens) \\
4. Persona & \texttt{Fool}$\to$\texttt{F}+\texttt{ool}, \texttt{Mortals}$\to$\texttt{M}+\texttt{ort}+\texttt{als}, \texttt{Pathetic}$\to$\texttt{Path}+\texttt{etic}, \texttt{Behold}$\to$\texttt{Beh}+\texttt{old} \\
```
<!-- manuscript-numbers:end -->
