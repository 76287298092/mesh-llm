# Fixed-producer NVFP4 operator checks

At e68e93d95, revised PTX83f52bb4... passes the unchanged19 cases normally and
under memcheck, racecheck and synccheck, all zero errors/hazards. Raw FP32 and
BF16 output bits match the old native baseline in every case. Reference limits
remain unchanged. RTX5090, driver615.71.09; Ninfer inactive and ComfyUI preserved.
CUDA reports37registers,4608shared bytes,112local bytes. This is32 fewer local
bytes than the original candidate, not elimination. Emitted PTX local accesses
store/read the final array of four output tuples. Four explicit output stores
are a possible follow-up. No kernel edit was made on that observation yet.

278 host tests, host/Linux Clippy, Just PTX and Linux tool builds pass. Copied
logs omit blank lines at EOF; original logs remain in the ignored trial directory.
These are operator checks only. Model comparison remains a separate trial.
