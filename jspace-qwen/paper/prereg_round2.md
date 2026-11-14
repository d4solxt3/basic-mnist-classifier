# Round 2 of the J++ comparison: pre-registration

Written and committed before any round-2 number was computed.

## What round 1 showed

Round 1 (`results/jpp_compare_main.json`, split seed 0, correctness-filtered items, Readout
Filtering, best of seven layers) gave held-out recall@10 of:

- J++: 0.374
- Routed-LRP-a200: 0.373
- JB-mix: 0.368
- R-lens: 0.365
- J-lens: 0.332
- B-lens alone: 0.217

No lens beat J++. The B-lens ranks some intermediates the other lenses miss (e.g. "Earth" at
rank 3 vs 12–16; "January" at 5 vs 8–13). This motivates combining it with the strongest maps.
The round-1 held-out split has now been looked at, so it cannot serve as an unbiased test for
anything designed after it.

## Candidates (fixed now)

1. **JppB.** Two-stage. First fit the J++ map. Then fit per-layer signed weights over the
   normalised [J++, B-lens] maps with J++'s own search on the same fitting items.
2. **RB-mix.** Per-layer signed weights over the normalised [R-lens, B-lens] maps.
3. **Routed-J++-a200.** Routed LRP experts shrunk toward the J++ map (rescaled to the pooled
   map's norm), with λ_e = n_e/(n_e + 200). κ = 200 is carried over from round 1, not tuned.
4. **Bagged-J++.** J++ weights refitted on 8 bootstrap resamples of the fitting items
   (stratified by eval), then averaged and renormalised. This targets J++'s fit/held-out gap
   (0.500 vs 0.374 in round 1).

Re-evaluated alongside: J++, R-lens, J-lens, logit lens, JB-mix, Routed-LRP-a200,
Routed-LRP-s0.5, B-lens.

## Protocol

- **Splits.** Fresh stratified 20%/80% splits with seeds 1–5, using the same `split_items` rule
  as J++. Seed 0 is re-run as a reproduction check and is excluded from the confirmatory
  summary.
- **Refitting.** On every split, every label-dependent component (J++ weights and every
  candidate's weights) is refitted on that split's fitting items only.
- **Fixed settings.** J++'s settings are unchanged: 200 Adam steps, lr 0.05, T = 0.1, three
  starts, the expert fallback, Readout Filtering, the strict rank rule.
- **Primary endpoint.** For each candidate, the mean over seeds 1–5 of the paired held-out
  difference in macro item recall@10 vs J++ on the same split.
- **Reporting.**
  - The mean and range of that difference.
  - The number of splits on which the candidate beats J++.
  - Each split's paired bootstrap interval.
- **Win criterion.** A candidate "beats J++" only if its mean difference is positive *and* it
  wins on at least 4 of the 5 splits.
- **Interpretation.**
  - Held-out sets of different splits overlap (each is 80% of the same 250 items), so the five
    differences are not independent. We do not pool them into a single p-value.
  - All candidates are reported, whatever the outcome.
