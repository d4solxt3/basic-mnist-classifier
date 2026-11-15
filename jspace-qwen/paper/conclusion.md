<!-- Draft fragment; merged into paper.md. -->

## 7 Conclusion

On a 0.6B model, the Jacobian lens reproduces as a better reader than the logit lens and a better
causal handle. The clean late-band workspace of large models does not reproduce.

- **Causal leverage.** J-lens directions move a function's argument most effectively from layer 3 to
  layer 19. That is earlier than the band where readouts peak, and swaps lose effect inside that band.
- **Sharpening.** The network sharpens competing concepts into a categorical choice, but
  progressively with depth, not as an abrupt event in a band.
- **J++.** Its gain over the J-lens shrinks from about 20 points to about 7, two-thirds of it from
  the LRP backward pass.
- **Improving on J++.** Two pre-registered rounds found nothing that beats it under its own protocol.
  Changes that curb its overfitting match it, and a label-free routed lens comes within two points.
- **Sketch estimators.** The planned online control variate diverged, as a variance recursion
  predicts, and a shrunk version is stable. Probe shaping cut sketch noise by the factor a
  weights-only formula predicts, but did not improve recall. With independent prompts, no sketch can
  beat the exact estimator at equal cost.

The second round overturned the first round's small wins. On benchmarks this small, a single split,
or even five overlapping ones, is not enough to claim an improvement.
