# Initial constraints

Status: source-backed assessment, not GPU qualification.

- Model: Qwen3.8-27B, Qwen3.5 dense hybrid architecture. The exact first artifact
  recipe will be pinned before conversion. Never select by display name alone.
- Target: carrack RTX 5090, SM120a, driver 615.71.09 as inspected September 26.
- Toolchain: installed host rustc 1.98.1 / LLVM 22.1.8 enumerates SM120a/PTX8.7.
  Device nightly is not yet installed or qualified. Clock state for performance:
  not measured. No performance result exists for MeshLLM specialized kernels.
- Source: MeshLLM `4b48a298c347cea818e13d339ff069cc6172cd09`.
- Reproduction and evidence: `docs/design/assessments/issue-1393/`.

Expected and observed constraints:

1. Rust can describe the target; successful compilation and GPU execution remain
   separate gates. NVFP4 requires architecture-specific PTX and driver support.
2. BF16 full-model residency exceeds the 5090. Use per-layer reference checks and
   a quantized first resident model. Never disguise offloading as resident speed.
3. The loader uses one global symbol table. Startup-only fallback is the first
   prototype boundary; same-process multi-engine dispatch needs separate work.
4. The current loader requires 126 exports including ABI version, of which 39
   are foreign llama/ggml/mtmd functions. Recount when updating the base revision.
5. NInfer HTTP returns token-level results and timings, not per-layer logits.
   Independent oracle checks must localize numerical errors.
6. A backend `Other` marker bypasses relevant eligibility checks. Use explicit
   driver-only CUDA requirements, selected-device identity, and memory admission.
7. Same BF16 ancestry does not imply identical quantization or greedy output.
   Pin logical weights, activation/KV precision, tokenizer, and template separately.

Durable rule: distinguish source facts, emitted instructions, executed arithmetic,
model correctness, and measured performance. None substitutes for the next.
