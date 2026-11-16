# J++ sprint run log (append-only)

One row per run, appended by `sprint/harness.py:runlog_append` or by hand. Rows are never edited or
deleted; corrections are new rows that reference the old one. "Held-out" = items not used to fit any
label-dependent weight on that split. In-sample (fitting-split) numbers appear only as overfitting
diagnostics, never as headline results.

| date (UTC) | experiment_id | hypothesis | metric_before | metric_after | control_check | interpretation |
|---|---|---|---|---|---|---|
| 2026-11-16 02:43 | E0.0-env | jpp_lens CPU test suite passes in this container | n/a | 264 passed in 37 s (jpp_lens @ d7efc0d) | uv sync could not fetch torch 2.13 CPU (download.pytorch.org denied by network policy); used host torch 2.15.0, all other deps from uv.lock | baseline environment OK; deviation = torch minor version |
| 2026-11-16 02:43 | E0.1-fidelity-LRP | our cached LRP expert Jacobians use the same estimator as jpp_lens ExpertJacobianRelPTrainer (lrp_mode=rlens) | n/a | one WikiText prompt (48 tok), layers 7 and 21, same router: per-cluster row sums rel. error 0.0, counts identical | target layer 27 in both; same kept positions (skip 16, drop last) | cached experts (32 prompts, all 27 layers) are faithful J++ inputs |
| 2026-11-16 02:43 | E0.1-fidelity-search | our J++ weight search reproduces jpp_lens fit_expert_weights | n/a | same starts, same step-0 surrogate (0.154042) and gradient; final iterate differs: L4 train 0.271 (ours) vs 0.285 (ref), L21 0.279 vs 0.279 (cos w 0.89) | difference persists with the reference's summation order -> float-level chaos of the 200-step Adam search, not an algorithmic difference | use the fast analytic-gradient search (same algorithm) for all arms; quantify search noise vs the reference (E0.1-search-noise) |
| 2026-11-16 02:43 | E0.1-spectral-diag | Exp-1 premise: expert-specific deviations contain noisy output directions separable by pooled-SVD bands (label-free test) | n/a | split-half reliability of expert deviations by band [0,16)/[16,64)/[64,256)/[256,1024): L4 0.89/0.90/0.87/0.84, L7 0.91/0.90/0.89/0.83, L14 0.94/0.94/0.93/0.87, L21 0.97/0.94/0.92/0.92; S/N >= 9.4 everywhere | coordinate-aligned outliers (max/u_k/ 0.54-0.64) are the TOP singular directions, not the tail | noise mechanism of H1 not supported: deviations are reproducible in every band; the tail band holds 17-20% of deviation energy vs 4-6% of pooled energy at early layers (only lead for Exp 1) |
