# The research prompt, improved

## Original prompt

> use the anthropic jspace paper to reproduce that paper on a qwen model, try improve it in any way
> you can, make this prompt better and execute it, also formalise this into the draft of something
> publishable.

## What was wrong with it

The original asks for something reasonable but leaves the important decisions unmade:

- **Which claims?** The paper (Gurnee et al., 2026, *Verbalizable Representations Form a Global
  Workspace in Language Models*) makes a methods claim (the Jacobian lens reads out
  latent content better than the logit lens) and several "global workspace" claims (the readable
  content is causally used, is confined to a mid-depth band, is flexibly routed, shows ignition-like
  dynamics, has limited capacity, ...). A reproduction has to say which of these it tests, and how.
- **Which Qwen, and with what compute?** "A Qwen model" ranges from 0.6B to several hundred
  billion parameters. The core cost, d_model backward passes per fitting prompt, decides
  what is feasible.
- **Improve relative to what?** By November 2026 the original lens had already been improved
  (the R-Lens; the J++ Lens of Ayonrinde & Lindsey, 2026, with Jacobian filtering, an LRP backward
  pass and Readout Filtering), and the core findings had been replicated on Qwen 3.6 27B. An
  "improvement" has to be new relative to that work, and measured with the same metric.
- **What counts as success?** No metrics, controls, or failure criteria are given, which makes the
  write-up unfalsifiable.

## Improved prompt

> **Goal.** Test whether the central claims of Gurnee et al. (2026) hold in a *small*
> open-weight model, and make the method cheaper and more robust, under a strict CPU-only budget
> (4 cores, 15 GB RAM, no GPU).
>
> **Model.** Qwen3-0.6B (28 layers, d_model = 1024; post-trained checkpoint, thinking disabled
> for chat prompts). If budget remains, a second size (Qwen3-1.7B) for a two-point scale
> comparison.
>
> **Reproduction (use the authors' released code conventions and prompt sets).**
> 1. Fit the Jacobian lens with the exact reference estimator (one-hot cotangents summed over
>    target positions, averaged over source positions; the first 16 positions skipped; 128-token
>    pretraining-style sequences). Check numerically that the implementation matches
>    `anthropics/jacobian-lens`.
> 2. *Lens quality:* recall@k (k = 1, 10) of latent intermediates on the six released readout
>    evals (multihop, multilingual, poetry, association, typo, order-ops). Score only items the
>    model answers correctly, report 95% bootstrap CIs over items, and compare to the logit lens.
> 3. *Workspace band:* locate the depth range where readouts are interpretable, and say how the
>    band was chosen *before* using it in causal experiments.
> 4. *Causal use:* clamp-swap interventions along unit-normalised lens directions
>    (probe-swap two-hop prompts, flexible-generalization). Always run three controls: the
>    *logit-lens* direction (does the Jacobian transport matter?), a *matched-norm random
>    concept* (is any edit enough?), and the *answer swap* ceiling (is the edit strong enough
>    at all?).
> 5. *Dynamics:* the ignition experiment (embedding interpolation between two concepts).
>    Measure whether the J-space readout switches more sharply than the input mix.
>
> **Improvements (each must be new relative to the J++/R-Lens work, and evaluated on the same
> metric as the baseline).**
> - *Cost:* the exact estimator needs d_model backward passes per prompt. Test an unbiased
>   randomised estimator (orthogonal random cotangent sketches, with the exactly known identity
>   path, and optionally an independent pilot estimate, used as control variates). Report lens
>   quality against the number of backward passes, compared with the exact estimator at equal
>   compute.
> - *Readout calibration:* the lens's mid-layer logits contain a context-independent prior
>   (frequent tokens rank high everywhere). Test per-token standardisation of lens logits against
>   corpus statistics as a zero-inference-cost readout. Compare it to raw ranking and to J++'s
>   Readout Filtering.
>
> **Reporting.** Pre-state metrics; include negative and null results. Every number in the
> write-up must come from a script and a results file in the repository. State plainly what was
> not reproduced and why (e.g. experiments that need a larger model or a GPU).
>
> **Deliverables.** (i) code and results in this repository; (ii) a paper draft in workshop
> format (abstract, intro, related work, methods, results, limitations), with figures.

## How the execution departs from this spec

Each departure from the improved prompt above, and the reason for it. The results are in
[`paper/paper.md`](paper/paper.md) and the arXiv-format version in [`paper/arxiv/`](paper/arxiv/).

**Done as specified**
- Exact J-lens estimator, checked against brute-force autograd (relative error ~1e-7).
- Recall@1 and recall@10 on the released readout sets, with the correctness filter and
  item-bootstrap intervals, against the logit lens.
- A workspace band chosen by a rule fixed before any causal experiment (layers 20–24).
- The three swap controls (logit-lens directions, random target, answer-swap ceiling).
- The sketch estimator with the exact identity path and an online control variate.
- Per-token readout calibration, compared with raw ranking and Readout Filtering.

**Departures**
- **No second model size.** The spec asked for Qwen3-1.7B "if budget remains". The budget went
  instead to the J++ comparison, which proved more informative. An exact 1.7B lens would cost
  about 3× the 0.6B fits.
- **Order-ops is reported separately.** It is not in the macro, which follows J++'s five-evaluation
  protocol so that J++ numbers compare directly. Only 18 of 55 order-ops items survive the
  correctness filter.
- **The random control is a random target, not a matched-norm random concept.** Every lens uses
  unit-normalised directions at the same scales. The edit norm is reported for every condition, so
  edits of matched size can be compared directly.
- **Probe-swap is uninformative.** The model answers only 8 of 83 two-hop prompts correctly, so
  flexible generalisation (60 scored trials) carries the causal results.
- **The ignition baseline changed.** "Sharper than the input mix" was first operationalised as a
  fixed reference width (0.8). That is invalid, because the rank-share statistic is non-linear. It
  was replaced by a linear-interpolation control at each layer, run on the same trials.
- **Calibration hurt rather than helped.** This is reported as a negative result, alongside two
  other readout ideas that did not work.
- **Fitting budget.** The lenses average 32 prompts (8 for the broadcast lens), against 1,000 for
  the released lenses. Recall is flat from 8 to 32 prompts, but J++'s experts are the most
  budget-sensitive component.

**Added beyond the spec**
- A full J++ reimplementation and head-to-head, with two implementation pitfalls documented.
- The broadcast lens.
- Label-free and routed variants.
- Two pre-registered multi-split tests of improvements over J++ (rounds 2 and 3).
- A weights-only predictor ρ(A) of the benefit of readout-aware probe shaping. For Qwen3-0.6B
  ρ(A) = 0.764 (`results/rho.json`).

**Not available to us.** The J-space paper's text could not be fetched from the compute
environment. The implementation follows the released code (`anthropics/jacobian-lens`,
`safety-research/jpp_lens`) line by line where possible, and states the choices that code does
not pin down.
