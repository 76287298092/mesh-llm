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
