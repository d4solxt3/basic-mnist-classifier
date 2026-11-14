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

Recorded at the end of the project so the reader can compare plan and outcome: see
[`paper/paper.md`](paper/paper.md), §Limitations.
