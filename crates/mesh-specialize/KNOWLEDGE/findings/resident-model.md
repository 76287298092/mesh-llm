# Resident full-model connection

Status: implementation pending live qualification. No full-model performance claim.

The decoder follows the compiled 64-layer schedule with persistent checkpoint
weights and one session state arena. A transaction advances the sequence cursor
only after all layers, the final normalization, vocabulary projection and greedy
selection succeed. A partially failed step poisons the session and rejects reuse.
The LM head selects the last hidden row on device. Intermediate hidden rows remain
on device except for optional read-only observers in the correctness trial.

An independent CPU composition loads the original quantized checkpoint one layer
at a time and saves every layer's BF16 output and the final logits. It runs while
Ninfer remains online. GPU qualification consumes this saved reference, verifies
its model/weights identity and extents, compares every layer and final logits with
the existing fixed 1% relative L2 and 0.9999 cosine budgets, checks greedy choice,
and compares full-batch against token-by-token logits and complete state hashes.
An injected failure checks rejection of partially updated sessions. Reference
files and trial reports are created without replacing existing evidence.

Commands, after `just specialize-tools-build`:

```sh
target/release/xtask specialize qwen-model-reference --artifact ARTIFACT --tokens 248044 --output REFERENCE
target/release/xtask specialize qwen-model-check --artifact ARTIFACT --reference REFERENCE --ptx target/specialize/probes.ptx --device 0 --output REPORT
```

CPU references are bounded to 1..17 tokens. This comparison uses our explicit
quantized arithmetic contract. It is not upstream BF16 quality validation.
The trial includes layer readbacks, state hashing and repeated independent
sessions. Its elapsed time is not model prefill or decode throughput. No device
kernel or PTX instruction site changes in this stage. Current performance, peak
memory, usable long context, tokenizer/chat serving, MTP and ABI integration
remain unmeasured or incomplete.

Exact source revisions and live results will be recorded after the first trial.
