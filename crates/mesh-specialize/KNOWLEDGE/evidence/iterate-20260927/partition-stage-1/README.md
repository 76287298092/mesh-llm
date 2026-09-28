# Layer22 stages: exact numerical checks pass, memory gate fails

Sourceff387d8ce, retained features-greedy PTX,128-token retained fixture.
FP8exact/attentionexact, MLPworkspaceon, split-Koff, GPUgreedyoff.
All64layer row hashes, every captured layer22 stage, whole/token logits and
complete state, and profile/control outputs/state agree. The observed layer22
path uses ordinary MLP allocation to expose stage buffers.

The overall report is NOT a pass: `memory_released=false`. Free memory dropped
from32219725824 to31165579264bytes. Process sampling records a separate
`target/release/mesh-llm` process appearing on GPU0 with1000MiB allocation.
It was not started or stopped by this trial. This external allocation confounds
the global free-memory check; no memory-leak clearance is claimed. The script
stopped before online mode. Numerical diagnostics remain usable; candidate
localization is dispatched separately in partition-stage-2. Ninfer stayedinactive
and ComfyUI remainedresident. No timing/quality/serving qualification claim.
