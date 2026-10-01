# Retained evidence verification

The parent rechecked all 20 required N=1..5 execution-mode cells after copying
trial C from Carrack. Normal, memcheck, synccheck, and N=1 racecheck use trial B;
N=2..5 racecheck use trial C. Every selected report passes the unchanged retained
`target-batch-selected-20260930.jq` against the retained fixture and pinned request
metadata. Every selected result records `exit=0 gate_failed=0`. Every sanitizer
cell has a final zero-error or zero-hazard summary. The four C summaries report
zero hazards, errors, and warnings.

The copied four C JSON reports match their remote SHA256 hashes. The retained
filter and fixture match the identities in `hashes.txt`, and the filter matches
the original parent copy byte for byte. The retained retry runner passes
`bash -n`. Each C per-case artifact/PTX/executable/fixture hash check passes.

The source manifest contains 341 entries. All pass in both
`target-batch-racecheck-source-check-before-20261001.log` and
`target-batch-racecheck-source-check-after-verified-20261001.log`. The initial
postcheck used the wrong manifest path and failed before checking any source;
its original diagnostic remains in
`target-batch-racecheck-source-check-after-20261001.log`.

The original four B racecheck reports remain empty, with exit 124 and failed
gates. Neither those timeouts nor trial A's failures were rewritten.

`runner.log` and `units-after.txt` record both authorized services restored
active. The parent independently checked both services active and received
HTTP 200 with `{"status":"ok"}` from `http://localhost:1235/health`.

This closes the bounded target-batch comparison and sanitizer checkpoint only.
Native MTP execution, full-state rejection/EOS recovery, model-quality gates,
and matched whole-model performance remain open. No runtime code, executable,
PTX, arithmetic tolerance, or admission flag changed during this verification.
