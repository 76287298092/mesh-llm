# Partition localization: layer22 row101

Sourcefbf877158ef17d21d10dd0154236ea84e434b739. Same128-token raw fixture,
RTX5090, driver615.71.09, features-greedy PTX SHA256
961d1652408eeb9ec8d72aa9c14d32ced2c2f0efb3e8e2a32dd0d5cdcc0c4a05.
FP8exact, split-Koff, MLPworkspaceon, GPUgreedyoff.

Exact attention: every logical row at all64layers has identical SHA256 under
whole-prefix and token submissions; all existing strict checks pass.
Online attention: all rows through layer21 match. The first mismatch is
layer22,row101(zero-based). Layer23 also differs only at row101; layer24
spreads the difference across later rows. Layer22 is GDN, so the first observed
layer-output mismatch occurs after a GDN block, not an attention block.
This localizes the failure but does not yet identify its operation.

Both profiles retain exactly the previous trial's whole and token final-state
hashes, confirming the added row captures did not change those results.
Profile/control output and complete state checks still pass in both modes.
Online's strict partition gate remains failed, as expected; the diagnostic
script accepts exit1 only to save that failure and requires valid audit output.
No throughput claim. Ninfer stayed inactive; ComfyUI remained resident.

Next: capture existing layer22 stage boundaries and verify profile/control
equality, then run the first differing projection with identical quantized
inputs against an independent reference. No arithmetic change or promotion.
