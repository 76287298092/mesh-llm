# Candidate compilation checkpoint

Both dedicated workers used GPT-6-Astra with low reasoning. Parent registered
separate A16 head and NVFP4 pipeline candidates and their independent references.
The retained first Clippy run failed three A16 reference style checks. Parent
replaced divisibility expressions and chunks_exact with the required standard
methods; the second run passes. 276 host tests passed before that repair and
all A16 reference tests passed afterward. Just PTX compilation and console-print
policy pass. GPU execution and resident integration remain unqualified.

PTX target is sm_120a with nightly-2026-09-25. Artifact SHA256 is
`e8beed4f5bba069809a7ff3d80398bea6f3980aa77a9a9f7352c4753f6ec401a`.
Local artifact: `target/specialize/iterate-20260927/features-head-pipeline.ptx`.
Compilation includes the candidate source and registration in this evidence's
containing commit. Existing profiles and retained baseline PTX are unchanged.

Copied test logs omit blank lines at EOF; raw logs remain in the ignored trial directory.
