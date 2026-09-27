# Integration corrections

These are summaries of integration failures, not reconstructed raw logs. Some local check logs were rerun at the same path; remote numbered failure logs are retained here.

- F02 CPU reference loop-index Clippy warnings were fixed with scale iterators. A parent broad replacement briefly changed a test variable; corrected before passing tests.
- Parent projection probe initially had too-narrow entrypoint visibility and obsolete constant-size chunk iteration; corrected before GPU execution.
- F04 test fixture required an explicit `Vec<u16>` collection type. An unused kernel local was removed after PTX compilation warned.
- F05 CPU test loop used an indexed accumulator; replaced with an iterator without changing equations.
- F06 tie fixture encoded 3/1024 as `0x3ac0`; correct BF16 encoding is `0x3b40`. Decoder iteration then received the Clippy-required iterator form.
- F08 parent review required exclusive stream borrowing for manual capture abort. Linux validation includes the driver at another module path, requiring relative visibility. Clippy requested a collapsed cleanup conditional.

These fixes do not establish model throughput improvements. GPU evidence covers only the named synthetic components and graph replay.
