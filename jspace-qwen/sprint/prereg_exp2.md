# Experiment 2 pre-registration: self-supervised KL and anti-false-positive weighting

Written and committed while Experiment 1's confirmation stage was still running, before Exp 1's
verdict and before any Exp 2 code. Exp 2 code runs only after Exp 1's verdict (one hypothesis at a
time). Exp 2 does not depend on Exp 1's outcome: every arm modifies J++'s own 8-expert weighting.
Amendments go at the bottom, dated.

## Hypothesis H2 (as given)

Replacing or regularising J++'s supervised recall@10 surrogate with a self-supervised next-token KL
alignment, KL(unembed(h_27) ‖ unembed(J_l(w) h_l)), plus a false-positive penalty, will:
- eliminate J++'s fitting/held-out overfitting;
- improve generalisation to a task held out from weight fitting.

Baseline facts (E0.2, tuning splits):
- J++'s early-stratum fitting-split recall@10 is 0.190 against 0.154 held-out, a gap of 3.6 points.
- The late-stratum gap is 9.7 points (0.290 vs 0.193).
- At late layers J++ trails the plain LRP lens on held-out items (0.193 vs 0.207).
- Matched-control FP@10 at the early layers is 0.36% for J++ against 0.11% for the R-lens.

**Our prediction before running: refuted on cross-task generalisation, confirmed on overfitting.**
- The KL objective pulls the map toward the model's next-token distribution (a tuned-lens
  objective), which the latent-intermediate tasks do not reward. KL-only weights should read out
  intermediates less well than J++.
- Regularisation should shrink the fitting/held-out gap. But Bagged-J++ already halved J++'s
  overfitting without raising held-out recall (paper §5.4).

## Objectives

All arms use J++'s 8 LRP experts per layer, unit-L2 signed weights w, and the same Adam search
(200 steps, lr 0.05, starts, exact score every 5 steps).

- **Surr(w).** J++'s surrogate: macro over evals of sigmoid(hit margin / (0.1 · spread)) on the
  fitting pairs.
- **FalseSurr(w).** The same surrogate on matched-control pairs within the fitting items. For each
  fitting (item i, intermediate x), the controls are up to 5 other fitting items j of the same
  eval:
  - no shared candidate token, and x does not appear in j's prompt;
  - a "false hit" is x ranked in the top 10 of j's readout.

  This is the training-side version of metric M2a, on the same scale as Surr.
- **KL(w).** Mean over N = 256 WikiText positions (records 5000–5255, seed 11) of
  KL(softmax(unembed(h_27)) ‖ softmax(unembed(J_l(w) h_l))), over the full vocabulary, unmasked.
  The model's own next-token distribution is the target. It is computed exactly from cached expert
  logits, with an analytic gradient through the final RMSNorm.

## Arms

| Arm | Objective | Labels |
|---|---|---|
| B0 J++ | Surr | fitting items |
| B1 KL-only | −KL (split-independent; fitted once per layer) | none |
| **B2 (primary)** | Surr − γ·FalseSurr − β·KL, with γ = 1 fixed a priori (a false hit costs as much as a hit) and β tuned | fitting items |
| B3 (secondary) | Surr − β·KL (same β) | fitting items |
| B4 (secondary) | Surr − γ·FalseSurr | fitting items |

- The returned iterate maximises the same combination with exact scores in place of surrogates
  (ties broken by the surrogate). B1 returns its lowest-KL iterate.
- B1's starts are uniform, the best single expert by KL, and the R-lens weighting (w ∝ n_e).

## Hyperparameter

- β ∈ {0.03, 0.1, 0.3} (KL in nats; Surr ∈ [0, 1]).
- Chosen on tuning splits 0–2, all seven layers, as the value with the best mean held-out best-of-7
  recall@10 of B2. Ties within 0.2 points go to the largest β.

## Evaluation

**Primary: leave-one-task-out (the control the hypothesis names).**
- For each macro eval (5 folds) and each of 4 replicates (20 fold-replicates), fit on 47 items
  sampled stratified from the other four macro evals plus order-ops (seed 300 + 10·fold + replicate).
- Score held-out best-of-7 recall@10 on every scored item of the left-out eval, plus the strata.
- Arms: B0, B1, B2.

**Standard splits.**
- Seeds 100–109, all arms. B0 is reloaded from Exp 1's A0 weights on the same seeds; the search is
  deterministic, so this is the same J++.
- Full M1–M3: held-out recall/MRR by stratum and best-of-7, FP@10 matched and WikiText, transport.
- Overfitting gap = fitting-split minus held-out recall@10, averaged over the three strata.

**M4** only if confirm criteria 1–4 hold: B2 against B0 maps from seed 100, layer sets as in Exp 1,
with the Amendment-1 null rule.

## Criteria

**Confirm H2** if all hold:
1. LOTO: mean held-out-task best-of-7 recall@10 of B2 − B0 is at least +2.0 points, and B2 wins at
   least 14 of 20 fold-replicates.
2. Overfitting: on the standard splits, B2's mean gap is at most half of B0's.
3. No in-distribution loss: the standard-split held-out best-of-7 recall@10 change B2 − B0 is at
   least −0.5 points.
4. False positives: the mean change in matched FP@10 (early stratum) is at most +0.5 points.

**Refute H2** if any hold:
- the LOTO mean of B2 − B0 is below +0.5 points;
- B2 wins 10 or fewer of 20 fold-replicates;
- the standard-split held-out best-of-7 change is below −2.0 points.

**Otherwise inconclusive.** Partial outcomes are named, for example "reduces overfitting without
generalisation gain" when only criterion 2 holds.

**Secondary (reported, no verdict):**
- B1 against B0: we predict at least 2 points lower held-out recall and a gap of about 0;
- B3 and B4, which attribute any B2 effect to its KL or false-positive term;
- the KL and transport metrics of every arm on held-out WikiText.

## Amendments

**Amendment 1 (2026-11-16, before any Exp 2 run; from the component checks only).** The exact
full-vocabulary KL made one B2 search take about 15 minutes per layer, which puts the whole
experiment near 15 CPU-hours. The KL is therefore computed through an exact decomposition,
KL_n = Σ p log p − (u·m_n)/r_n + logsumexp_v z_nv, in which the first two terms are exact over the
full vocabulary.

During the search, the log-partition is estimated from each position's 8,192 tokens with the
largest expert-logit norm plus 16,384 tail tokens sampled once per position and weighted by
|tail|/16,384.

Validation (`sprint/test_exp2_kl.py`), at layers 4 and 21:
- error ≤ 0.0002 nats at the J++, uniform and R-lens weightings;
- error ≤ 0.038 nats over 16 test weight vectors (KL range 2.8–29.7 nats).

Every reported KL, and B0's KL, is the exact full-vocabulary value at the final weights. Nothing
else changes.
