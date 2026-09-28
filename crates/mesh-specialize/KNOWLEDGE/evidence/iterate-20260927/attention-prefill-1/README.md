# Online attention: failed128-token partition qualification

Source238ac8a7de94d70d112bd0e55c93fd18a3e02574, unchanged features-greedy PTX
961d1652408eeb9ec8d72aa9c14d32ced2c2f0efb3e8e2a32dd0d5cdcc0c4a05.
FP8exact, split-Koff, MLPworkspaceon, GPUgreedyoff. Same retained raw128-token
prompt used by previous exact prefill checks. RTX5090,driver615.71.09.

Exact attention passes all gates. Online attention passes profile/control
logits and complete state equality, but fails strict whole-prefix versus
token-by-token equivalence:238354prefill BF16 logits differ, raw normalized
L2=0.055244, KL=0.034233, TV=0.103237; complete state hashes differ. The same
teacher token98094 was used for subsequent decode, which also differs. Greedy
prefill and teacher-decode winners still agree. First nonzero *last-row* hidden
drift is at layer25(relativeL2=0.0092872). Earlier rows were not captured; this
is not proof the first divergent operation is at layer25.

The script exited1 at the unchanged gate, before512-token cases.128-prefill
medians were153.963exact/166.184online tokens/s; these timings are CPU-contended,
fixed-order and shared-GPU, and do not qualify a performance claim. Ninfer was
inactive before/after; ComfyUI remained resident. No unrelated job was stopped.

Source inspection identifies a possible downstream confound: multirow NVFP4
uses native FP32 MMA accumulation (`nvfp4_linear`), whereas one row uses integer
accumulation (`nvfp4_decode_exact`). The attention change could expose a rounding
boundary in that existing dispatch. This is only a hypothesis, not a measured
root cause. Capture all row/layer boundaries and compare the earliest differing
operator on identical inputs before changing any arithmetic or qualification gate.

Full BF16 logits remain in ignored local/remote trial directories, with committed
SHA256 manifests. Preserve this failure. Online attention remains experimental;
512-token timing, full-model sanitizers and longer-answer quality have not run.
