<!-- Draft fragment; merged into paper.md. Pre-registrations: paper/prereg_round2.md, paper/prereg_round3.md. -->

### 3.6 Candidate improvements to J++

J++ chooses eight signed weights per layer to maximise recall on about 50 labelled items. Its
fitting-split recall far exceeds its held-out recall (§5.2). We therefore designed candidates
that either stabilise the weight search or feed it better maps. All of them use J++'s own search
(exact recall@10 with a sigmoid-margin surrogate, three starts, 200 Adam steps) on the same
fitting items. Maps entering a mix are first scaled to unit Frobenius norm, so the weights
compare directions rather than scales.

*Bagged-J++* refits the J++ weights on B = 8 bootstrap resamples of the fitting items, drawn
separately within each evaluation. It averages the B weight vectors and renormalises the average
to unit length. Averaging reduces the variance of a search whose objective, recall over a few
dozen items, is a step function of the weights.

*RB-mix* fits two signed weights per layer over the R-lens and the B-lens (§3.1). It has fewer
weights than J++ (2 against 8). Its two maps differ in kind rather than in cluster: total
influence through the LRP graph, and influence on later positions only.

*JppB* is two-stage. It first fits J++, then fits two weights over the J++ map and the B-lens.

*Routed-J++* routes each residual to its own LRP expert (§3.3), shrunk toward the fitted J++ map
instead of the pooled map, with λ_e = n_e/(n_e + 200).

*RLB-mix* (round 3) is RB-mix with the B-lens replaced by an *LRP broadcast lens*: the broadcast
estimator of §3.1 run through the R-Lens's LRP backward rules. At this scale the LRP rules account
for nearly all of J++'s gain over the J-lens (§5.3). The question is whether they help the
broadcast map too.

Round 2 fixed the first four candidates, and round 3 fixed RLB-mix, each in a committed
pre-registration before any of its numbers were computed.
