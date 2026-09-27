# F04 experimental chunked GDN

Status: bounded matrix-form CPU reference and three NVPTX stage entrypoints are
written. This is an experimental correctness candidate only. It is not connected
to the Qwen schedule, is not selected at runtime, and has no performance claim.

## Scope and evidence

The implementation is in `reference/gdn_chunked.rs` and
`kernels/nvptx/gdn_chunked.rs`. It reuses the logical `Shape` and `Input` from
`gdn_recurrent.rs`: Q and K are FP32 `[rows,key_heads,width]`; QKV is BF16
`[rows,2*key_heads+value_heads,width]`; beta is BF16 and decay is FP32
`[rows,value_heads]`; state is FP32 `[value_heads,width,width]`. Value head `h`
uses key head `h / (value_heads / key_heads)`.

The worker added host tests for grouped heads, signed vectors, nonzero initial
state, zero and unit decay, rows 1/15/16, and a 16+1 tail split. The chunk
candidate is compared both to a separately written FP64 scalar recurrence and
the existing ordered-FP32 scalar reference. Each test fixes maximum absolute
error limits of `2e-4` for unrounded output and final state; BF16 output must
equal RNE rounding of the candidate's own FP32 output. These tests have been
written but not run: the parent retained the Cargo/build slot and owns module
registration. `rustfmt --edition 2024` completed for both Rust files.

No GPU, driver, PTX JIT, sanitizer, model, or timing trial was run. The worker
did not query a Git revision, per assignment. Parent integration should record
the exact source revision and run the CPU tests before attempting device
qualification. An independent inline Python check of the 17-row grouped-head
fixture first reported output max error `1.926e-1` while state error was
`7.021e-9`; inspection found that the check harness concatenated head-major
outputs against the scalar oracle's time-major order. After assigning outputs
by their logical `[time,head,column]` index, the same check reported output max
error `7.276e-9` and state max error `7.021e-9`. This was a harness ordering
error; it did not change repository code. There is no Ninfer code or
source-derived implementation here.

## Independently derived recurrence

For one value column, the existing scalar update is

```text
S_t = d_t * S_(t-1) + k_t * u_t
u_t = beta_t * (v_t - d_t * k_t^T * S_(t-1))
```

Unrolling state gives

```text
D(t,j) = product(d[r], r=j+1..t), with D(t,t)=1
D(t,-1) = product(d[r], r=0..t)
S_(t-1) = D(t-1,-1)*S_0 + sum(j<t, D(t-1,j)*k_j*u_j)
```

Substitution into `u_t` yields the lower-triangular system

```text
L[t,j] = beta_t * D(t,j) * dot(k_t,k_j), j<t
B[t]   = beta_t * (v_t - D(t,-1) * dot(k_t,S_0_column))
u_t    = B[t] - sum(j<t, L[t,j] * u_j)
```

Output and final state follow directly:

```text
o_t = D(t,-1)*dot(q_t,S_0_column)
      + sum(j<=t, D(t,j)*dot(q_t,k_j)*u_j)
S_last = D(last,-1)*S_0 + sum(j, D(last,j)*k_j*u_j^T)
```

Every decay product is formed directly from its bounded interval. No prefix
division is used, so a zero decay or an underflowed prefix cannot create a
division by zero or `0/0`. The scalar f64 test oracle advances `S` in time and
does not use these chunk equations.

The FP32 candidate reassociates reductions and state updates. Numerical
differences from the ordered scalar path are expected; the existing exact
recurrent tests and dispatch are unchanged. The fixed CPU test budget is a
feasibility gate for these fixtures, not a model-layer quality budget.

## NVPTX launch and pointer contracts

All three kernels are experimental standalone entrypoints. The host must check
all input values for finiteness, beta and decay in `[0,1]`, exact tensor sizes,
checked extent arithmetic, supported dimensions, and output finiteness. Every
pointer must be aligned for its element type, buffers must be disjoint as stated
below, and storage must remain live through the launch. Scratch is fixed-stride
per value head:

| Buffer | Shape | Bytes per value head |
| --- | --- | ---: |
| `coefficients` | `[16,16]` FP32 | 1,024 |
| `rhs` | `[16,128]` FP32 | 8,192 |
| `decay_products` | `[16,17]` FP32 | 1,088 |
| `updates` | `[16,128]` FP32 | 8,192 |

`gdn_chunk_prepare(k, qkv, beta, decay, initial_state, coefficients, rhs,
decay_products, rows, key_heads, value_heads, width)` reads the current chunk's
K, V, gates and initial state. Launch grid is
`[value_heads, ceil((rows*rows + 17*rows + rows*width)/256), 1]`, block
`[256,1,1]`. It writes `L` with fixed row stride 16, `B` with fixed row stride
128, and `D` with fixed row stride 17. Upper-triangle coefficients and unused
decay cells for each active row are zeroed. Its read buffers and three scratch
outputs must not overlap.

`gdn_chunk_solve(coefficients, rhs, updates, rows, value_heads, width)` reads
the first two scratch buffers and writes the fourth. Launch grid is
`[value_heads,1,1]`, block `[width,1,1]`. Each thread owns one value column and
solves rows in increasing order. Its read buffers and output must not overlap.

`gdn_chunk_finish(q, k, initial_state, decay_products, updates, out, unrounded,
final_state, rows, key_heads, value_heads, width)` reads input/state/scratch and
writes output `[rows,value_heads,width]` as BF16 RNE, unrounded output of the
same element count as FP32, and a new state `[value_heads,width,width]` as
FP32. Launch grid is
`[value_heads, ceil((rows*width + width*width)/256), 1]`, block `[256,1,1]`.
Every flat worker owns one output or final-state element. Read inputs, scratch,
and each output buffer must not overlap.

All stages require `rows` in `1..=16`, `key_heads` in `1..=64`, `value_heads`
in `1..=256` divisible by `key_heads`, and power-of-two `width` in `1..=128`.
No stage uses tensor instructions. The matrix work is parallelized across
coefficient/RHS/output/state elements, while the triangular time dependency is
solved per value column. This reduces the recurrence structure at the equation
level; no execution or speedup has been measured.

## Pending qualification

Parent-owned integration still needs to register the reference and NVPTX
modules, add the PTX site to `KNOWLEDGE/asm-inventory.md`, and provide a host
launcher that enforces these pointer, size, and launch contracts. Then run the
CPU tests, compare device output and final state to both scalar references,
cover one-row and tail chunks plus zero/unit decay, and run the applicable CUDA
sanitizers. Only after that can a same-shape GPU timing be collected; any speed
claim also requires an isolated and full-model comparison under the feature
measurement contract in `findings/ninfer-feature-parity.md`.

Parent integration on 2026-09-27: fixed one test fixture collection type, registered both modules, and passed 233 macOS crate tests, host Clippy and NVPTX compilation. GPU execution and model dispatch remain pending.
