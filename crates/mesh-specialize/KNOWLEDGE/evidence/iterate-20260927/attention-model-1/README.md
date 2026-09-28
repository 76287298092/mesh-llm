# Online attention: short natural-prompt comparison

Source `238ac8a7de94d70d112bd0e55c93fd18a3e02574`; unchanged `features-greedy.ptx` SHA256 `961d1652408eeb9ec8d72aa9c14d32ced2c2f0efb3e8e2a32dd0d5cdcc0c4a05`. RTX5090 GPU0, driver615.71.09. FP8 exact, split-K off, MLP workspace on, GPU greedy off in both modes. Default attention remains exact.

Each prompt generated32tokens in3repetitions. All repetitions within a mode returned the same tokens. Four model profiles passed strict same-profile whole/token partition, profile/control output and full-state checks, finiteness and memory release. Cross-profile outputs/state are not required or claimed bitwise equal. Teacher-forced decode inputs match between modes.

| Prompt | Exact decode tokens/s | Online decode tokens/s | 32 generated tokens equal | Prefill KL(exact\|online) | Prefill TV | Teacher-decode KL |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| python | 13.2343 | 14.6249 | True | 0.0478712 | 0.0599912 | 2.70717e-05 |
| explain | 16.8265 | 14.4260 | False | 0.0212894 | 0.0680117 | 5.50366e-05 |

Timings are **CPU-contended, shared-GPU, fixed exact/online order**. A late-trial snapshot found load average49.44 and many unrelated CUDA compiler processes. ComfyUI remained resident at498MiB; Ninfer was inactive before and after. No unrelated job was stopped. The timings do not establish an uncontended speedup and should not be compared to prior uncontended trials.

The same-input layer3 operator audit had raw relative L2 around1.9e-7, but these full-model prefill logit comparisons have relative L2 of0.2555(Python)/0.1092(prose), KL0.04787/0.02129 and total variation0.05999/0.06801. Greedy prefill and one teacher-forced next-token winners agree on both prompts; prose diverges later in its32-token free continuation. This demonstrates amplification or downstream sensitivity, not proof of semantic quality regression. Later-layer same-input operator audits have not established that every difference has the same cause.

These are two short prompts, not a quality suite. Longer answers, task checks, long-context behavior, MTP, concurrency and matched Ninfer comparison remain open. No promotion.

Full BF16 logit files remain in the ignored local/remote trial directory; committed manifests retain per-file SHA256 and metadata. `comparison.json` preserves all computed metrics. Rerun the included comparison script against the original directory to verify hashes and recompute.
