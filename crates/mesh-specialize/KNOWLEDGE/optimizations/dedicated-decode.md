# Dedicated decode continuation

The first candidate transposes the existing NVFP4 tensor-core operands for m=1:
weights occupy the 16-row A operand and the one-token activation occupies column0
of B. This produces sixteen output channels per warp instead of eight, reuses the
existing packed loads and scale format, and retains K64 iteration order. No A16
policy or checkpoint representation change accompanies the experiment. Independent
signed/tail fixtures include output widths1/13/35/17 and K16/80/5120/17408; the
full-model reference and partition check decide acceptance before timing.

Status: candidate under qualification. The source, PTX and measured reports will
be recorded after the Carrack trial. Baseline is the retained September27 result:
20.071 short decode,18.146 after128 inputs,136.830 prefill tokens/s.

The transpose trial at `782e87cba81e23570fc3f43d8faecfbfaaa4666d`, PTX
`eddf54396f334ebde79c0311de2ad757759faea234bbf6cf450ed1b3d731bcc2`, passes
independent components, original-MMA FP32/BF16 equality, and exact full-model
hidden/logits/state. Its short median20.233 and128-prefix median18.281 barely
improve the fresh control20.103/18.121. NVFP4 event time is18.214ms versus18.627ms
in the prior profile. This is not the hoped-for useful gain; keep its source as
an experiment while testing a more parallel grouped-dot candidate. Ninfer
restored13:38:30 EDT, PID3284931, HTTP200; ComfyUI448118 unchanged.

The grouped-dot candidate computes each sixteen-value NVFP4 group with exact
signed integer dot products using DP4A. E2M1 values are integers/2 and group
scales are integers/512. Group dots fit signed32, their activation-scale product
fits signed32, and the fully scaled sum fits signed64 below2^58. A warp owns one
output channel; K groups distribute across lanes. Final conversion/scaling keeps
the independent oracle's order. Original-MMA FP32 differences are recorded, not
used as an equality requirement: its floating-point accumulation order differs.
Independent BF16-reference equality and unchanged full-model gates remain required.

The [raw first-candidate evidence](../evidence/iterate-20260927/decode-round/)
includes fresh baseline controls and restoration checks. The [trial wrapper](../evidence/iterate-20260927/run-round.sh)
requires an exact clean revision and PTX hash and restores the user service on exit.
Authentication-bearing journal lines are omitted from durable evidence.
