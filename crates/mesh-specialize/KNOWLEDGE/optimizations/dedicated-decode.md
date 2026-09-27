# Dedicated decode continuation

The first candidate transposes the existing NVFP4 tensor-core operands for m=1:
weights occupy the 16-row A operand and the one-token activation occupies column0
of B. This produces sixteen output channels per warp instead of eight, reuses the
existing packed loads and scale format, and retains K64 iteration order. No A16
policy or checkpoint representation change accompanies the experiment. Independent
signed/tail fixtures include output widths1/13/35/17 and K16/80/5120/17408; the
full-model reference and partition check decide acceptance before timing.

Status: grouped-dot decode retained after complete qualification. Baseline is the retained September27 result:
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

## Retained grouped-dot result

Code `24aa1b142` and PTX
`64cf82c90ba3d25299e61693af46b8523aba122cd989810274a7b2786687cda9`
pass all 16,388 independent structured outputs (256 FP4 value pairs, 16,129
scale pairs and three maximum-width outputs), including exact FP32/BF16 bits.
The full two-token, 64-layer hidden/logit oracle and whole/token state comparison
remain exact. Initial medians are 25.313 short decode, 22.388 after 128 inputs,
and 136.914 prefill tokens/s. Fresh controls earlier in the same session were
20.103, 18.121 and 137.024 respectively. NVFP4 event time falls from the earlier
18.627 ms to 8.232 ms; total short-decode GPU time is 30.341 ms. These are fixed
raw-token trials with eight outputs, not a matched Ninfer serving comparison.

The JIT reports 38 registers and zero local/shared bytes; offline PTXAS reports
39 registers with no stack/spills. macOS checks pass, and Linux passes 271 library
and 25 validation tests plus Clippy. The first Linux lint pass caught excessive
fixture argument counts and a raw-pointer argument-array assignment; the final
host harness groups fixture slices and reconstructs the control argument vector.
The [initial raw result](../evidence/iterate-20260927/exact-round/) records the
model/evidence hashes, selected GPU, clocks and service restoration. Ninfer was
healthy again at 13:53:10 EDT, PID 3292174; ComfyUI PID 448118 remained unchanged.
Final sanitizer and repeated timing evidence follows after its restoration check.

The [final qualification round](../evidence/iterate-20260927/exact-qualified/)
passes normal, memcheck, racecheck and synccheck with zero reported errors/hazards.
Repeated medians are 25.260 short decode, 22.366 after 128 inputs and 136.619
prefill tokens/s. Both profiles again pass exact control/state and memory release.
Ninfer restored at 13:57:40 EDT, PID 3293651, HTTP 200; ComfyUI unchanged.
This closes the first dedicated-decode improvement. Longer prefill is next; MTP
remains unimplemented. The earlier transposed MMA remains a recorded experiment,
not the selected resident decode path.
