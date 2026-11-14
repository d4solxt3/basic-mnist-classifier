# Round 3 of the J++ comparison: pre-registration

Written and committed after round 2 was analysed. At commit time, no number has been computed for
the new lens (the LRP broadcast lens, below) or for any round-3 split.

## What round 2 showed

Round 2 (`results/jpp_splits_round2.json`, `prereg_round2.md`) was scored on split seeds 1–5.

- **Bagged-J++** gained +0.010 over J++ (4/5 wins).
- **RB-mix**, two signed weights per layer over the R-lens and the broadcast lens, gained
  +0.006 (4/5 wins, 1 tie).
- Both met the pre-registered win criterion, by margins well inside a single split's paired
  interval.
- RB-mix's gains came mostly from multihop.
- Across splits, J++ led the R-lens by only 0.6 points. At this scale the LRP backward pass
  carries nearly all of J++'s advantage over the J-lens.

Two questions follow:

1. **Do the round-2 winners hold up over many more splits?** Five splits that share most of
   their held-out items are weak evidence.
2. **Does an LRP version of the broadcast lens help more?** The R-lens is the plain J-lens
   with LRP backward rules, and LRP is where J++'s gain comes from. The broadcast lens in
   RB-mix still uses the plain backward pass.

## New lens (fixed now)

**LRP-B-lens.** The broadcast lens (targets p′ > p only, §3.1) computed with the same LRP
backward rules as the R-lens. It uses exactly the B-lens's configuration, so the only
difference is the backward rule:

    python scripts/fit.py --method exact --n 8 --dim-batch 8 --targets later --lrp --tag lrp_later

## Candidates (fixed now)

1. **Bagged-J++**: as in round 2, with 8 bootstrap resamples.
2. **RB-mix**: as in round 2, signed weights over unit-normalised [R-lens, B-lens].
3. **RLB-mix**: signed weights over unit-normalised [R-lens, LRP-B-lens], fitted with the same
   search as RB-mix.

Also reported, with no criterion: J++, R-lens, J-lens, Routed-LRP-a200, B-lens, LRP-B-lens.

## Protocol

- **Splits.** Twenty new stratified 20%/80% splits, seeds 6–25, with the same `split_items`
  rule. All 20 are confirmatory. No other seed is run.
- **Fixed settings.** Unchanged from round 2:
  - correctness-filtered item pool (250 items);
  - every label-dependent weight refitted on each split's fitting items only;
  - the faithful weight-search port (`fit_expert_weights_v2`) with J++'s settings (200 Adam
    steps, lr 0.05, T = 0.1, three starts);
  - the expert fallback, Readout Filtering, best rank over layers 4, 7, 11, 14, 18, 21, 24,
    and macro item recall@10.
- **Primary endpoint.** For each candidate, the mean over the 20 splits of the paired held-out
  difference vs J++ on the same split.
- **Win criterion.** A candidate beats J++ only if its mean difference is positive *and* it wins
  (difference > 0) on at least 16 of the 20 splits, the same 80% rate as round 2's 4/5. Ties
  count as non-wins.
- **Reporting, for every candidate whatever the outcome:**
  - the mean, range and number of wins, ties and losses;
  - each split's paired bootstrap interval.

  As in round 2, the held-out sets overlap, so the splits are not independent and we compute no
  pooled p-value.

## Secondary analysis (no criterion)

**Out-of-pool items.** The released items that fail the correctness filter (the model's answer
is wrong) were never scored in round 2. On each split, the lenses fitted on that split's
fitting items are also scored on all of these items. We report the mean paired difference vs
J++ over splits.

These items are a different population: the model may never compute the intermediate. In the
round-1 all-items panel they were part of the held-out set for J++, the R-lens and the J-lens.
They have not been scored for the candidates above.

## What we have seen before committing this

- All round-1 and round-2 results.
- No number from the LRP-B lens, which has not been fitted.
- No number from seeds 6–25.

A smoke test of the code path will use the plain B-lens as a stand-in for the LRP-B lens, a
single layer and a few Adam steps. Its numbers are not results and will not be used.
