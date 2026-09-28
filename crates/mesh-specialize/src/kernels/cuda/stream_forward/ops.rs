//! Allocation-free kernel argument packing and shared operation enqueues.
//!
//! Every launch mirrors the legacy host operation it replaces: same kernel, same
//! pointer/scalar order, same grid and block. Only the storage behind the pointers
//! (planned arena slots instead of fresh allocations) and the stream differ.

use super::{
    functions::Functions,
    graph_position::Position,
    program::{NormSlots, ProjectionSlots},
    weights::{Arithmetic, ProjectionWeights},
};
use crate::kernels::cuda::driver::{Function, graph::ActiveStream};
use anyhow::{Result, ensure};
use std::{ffi::c_void, ptr};

const MAX_ARGS: usize = 16;
pub(super) const EPSILON: f32 = 1.0e-6;

// Scalars are stored in the low bytes of 8-byte slots; CUDA reads each parameter's
// own size from its slot, which is only the value itself on little-endian hosts.
const _: () = assert!(cfg!(target_endian = "little"));

/// Kernel parameters in declaration order, held on the stack.
pub(super) struct Args {
    storage: [u64; MAX_ARGS],
    len: usize,
    overflow: bool,
}

impl Args {
    pub(super) fn new() -> Self {
        Self {
            storage: [0; MAX_ARGS],
            len: 0,
            overflow: false,
        }
    }
    fn push(mut self, value: u64) -> Self {
        if self.len < MAX_ARGS {
            self.storage[self.len] = value;
            self.len += 1;
        } else {
            self.overflow = true;
        }
        self
    }
    pub(super) fn ptr(self, address: u64) -> Self {
        self.push(address)
    }
    pub(super) fn ptrs(self, addresses: &[u64]) -> Self {
        addresses
            .iter()
            .fold(self, |args, &address| args.push(address))
    }
    pub(super) fn u32(self, value: u32) -> Self {
        self.push(u64::from(value))
    }
    pub(super) fn f32(self, value: f32) -> Self {
        self.push(u64::from(value.to_bits()))
    }
}

pub(super) fn to_u32(value: usize) -> Result<u32> {
    Ok(u32::try_from(value)?)
}

/// FP8 linear kernel choice under the default exact profile (split-K off).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fp8Kernel {
    Exact,
    Exact4,
    Verify,
    Prefill,
}

/// Mirror of `resident_fp8::Projection::run`: `(kernel, tile_rows, tile_columns, threads)`.
pub(super) fn fp8_schedule(rows: usize, channels: usize) -> (Fp8Kernel, usize, usize, u32) {
    if rows >= 16 {
        (Fp8Kernel::Prefill, 16, 8, 32)
    } else if rows >= 4 && channels >= 16_384 {
        (Fp8Kernel::Verify, 8, 16, 32)
    } else if rows >= 4 {
        (Fp8Kernel::Exact4, 4, 4, 128)
    } else {
        (Fp8Kernel::Exact, 1, 4, 128)
    }
}

/// Mirror of the baseline NVFP4 schedule: `(decode_exact, tile_rows, tile_columns, threads)`.
pub(super) fn nvfp4_schedule(rows: usize) -> (bool, usize, usize, u32) {
    if rows == 1 {
        (true, 1, 4, 128)
    } else {
        (false, 16, 8, 32)
    }
}

/// Stream enqueue context for one forward.
pub(super) struct Enqueue<'a, 's, 'm, 'ctx> {
    active: &'a ActiveStream<'s, 'ctx>,
    pub(super) kernels: &'a Functions<'m, 'ctx>,
    pub(super) position: Option<&'a Position<'m, 'ctx>>,
}

impl<'a, 's, 'm, 'ctx> Enqueue<'a, 's, 'm, 'ctx> {
    /// # Safety
    /// Every address later passed to this context's operations must lie inside a
    /// live allocation of the stream's context with the extent the operation uses
    /// (arena slots planned for at least the forward's rows, resident weights, and
    /// session state), and all of them must outlive completion of the stream. The
    /// caller synchronizes the stream before releasing or reusing any of them.
    pub(super) unsafe fn new(
        active: &'a ActiveStream<'s, 'ctx>,
        kernels: &'a Functions<'m, 'ctx>,
    ) -> Self {
        Self {
            active,
            kernels,
            position: None,
        }
    }

    pub(super) fn launch(
        &self,
        function: &Function<'_, '_>,
        grid: [u32; 3],
        block: [u32; 3],
        args: Args,
    ) -> Result<()> {
        ensure!(!args.overflow, "kernel argument list exceeds {MAX_ARGS}");
        let mut storage = args.storage;
        let mut pointers = [ptr::null_mut::<c_void>(); MAX_ARGS];
        for (pointer, value) in pointers.iter_mut().zip(storage.iter_mut()) {
            *pointer = ptr::from_mut(value).cast();
        }
        // SAFETY: `storage` outlives the driver call and holds each parameter in
        // declaration order (8-byte pointers; 4-byte scalars in little-endian low
        // bytes). Device-address validity and retention are this context's
        // construction contract; the caller mirrors the legacy kernel signature.
        unsafe {
            self.active
                .launch(function, grid, block, &mut pointers[..args.len])
        }
    }

    /// Stream-ordered device copy between disjoint ranges covered by the contract.
    pub(super) fn copy(&self, destination: u64, source: u64, bytes: usize) -> Result<()> {
        // SAFETY: Both ranges are live, disjoint allocations under the construction contract.
        unsafe { self.active.copy_device(destination, source, bytes) }
    }

    /// `embedding_norm_bf16`: gather `ids` rows from `table`, normalize with `weight`.
    pub(super) fn embedding_norm(
        &self,
        [table, ids, weight]: [u64; 3],
        out: &NormSlots,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        let args = Args::new()
            .ptrs(&[table, ids, weight, out.copy, out.out, out.raw])
            .u32(to_u32(width)?)
            .f32(EPSILON);
        self.launch(
            &self.kernels.embedding_norm,
            [to_u32(rows)?, 1, 1],
            [256, 1, 1],
            args,
        )
    }

    /// `residual_norm_bf16`: outputs `[sum, normalized, raw]`.
    pub(super) fn residual_norm(
        &self,
        [residual, branch, weight]: [u64; 3],
        outputs: [u64; 3],
        rows: usize,
        width: usize,
    ) -> Result<()> {
        let args = Args::new()
            .ptrs(&[residual, branch, weight])
            .ptrs(&outputs)
            .u32(to_u32(width)?)
            .f32(EPSILON);
        self.launch(
            &self.kernels.residual_norm,
            [to_u32(rows)?, 1, 1],
            [256, 1, 1],
            args,
        )
    }

    pub(super) fn residual_add(
        &self,
        left: u64,
        right: u64,
        output: u64,
        count: usize,
    ) -> Result<()> {
        let count = to_u32(count)?;
        let args = Args::new().ptrs(&[left, right, output]).u32(count);
        self.launch(
            &self.kernels.residual_add,
            [count.div_ceil(256), 1, 1],
            [256, 1, 1],
            args,
        )
    }

    /// `bf16_linear_decode`: outputs `[values, raw]`.
    pub(super) fn bf16_linear(
        &self,
        [input, weight]: [u64; 2],
        outputs: [u64; 2],
        rows: usize,
        [channels, width]: [usize; 2],
    ) -> Result<()> {
        let args = Args::new()
            .ptrs(&[input, weight])
            .ptrs(&outputs)
            .u32(to_u32(rows)?)
            .u32(to_u32(channels)?)
            .u32(to_u32(width)?);
        self.launch(
            &self.kernels.bf16_linear_decode,
            [to_u32(channels.div_ceil(4))?, to_u32(rows)?, 1],
            [128, 1, 1],
            args,
        )
    }

    /// `mlp_silu_product`: outputs `[values, silu, activated, raw]`.
    pub(super) fn silu_product(
        &self,
        gate: u64,
        up: u64,
        outputs: [u64; 4],
        count: usize,
    ) -> Result<()> {
        let count = to_u32(count)?;
        let args = Args::new().ptrs(&[gate, up]).ptrs(&outputs).u32(count);
        self.launch(
            &self.kernels.mlp_silu_product,
            [count.div_ceil(256), 1, 1],
            [256, 1, 1],
            args,
        )
    }

    /// Quantize `rows` BF16 input rows and run the default-profile projection.
    pub(super) fn projection(
        &self,
        weights: &ProjectionWeights,
        input: u64,
        slots: &ProjectionSlots,
        rows: usize,
    ) -> Result<()> {
        match weights.arithmetic {
            Arithmetic::Fp8 => self.fp8_projection(weights, input, slots, rows),
            Arithmetic::Nvfp4 {
                input_scale,
                factor,
            } => self.nvfp4_projection(weights, input, slots, rows, [input_scale, factor]),
        }
    }

    fn fp8_projection(
        &self,
        w: &ProjectionWeights,
        input: u64,
        s: &ProjectionSlots,
        rows: usize,
    ) -> Result<()> {
        let quantize = Args::new()
            .ptrs(&[input, s.codes, s.scales])
            .u32(to_u32(w.width)?);
        self.launch(
            &self.kernels.fp8_quantize,
            [to_u32(rows)?, 1, 1],
            [256, 1, 1],
            quantize,
        )?;
        let (kernel, tile_rows, tile_columns, threads) = fp8_schedule(rows, w.channels);
        let function = match kernel {
            Fp8Kernel::Exact => &self.kernels.fp8_linear_exact,
            Fp8Kernel::Exact4 => &self.kernels.fp8_linear_exact4,
            Fp8Kernel::Verify => &self.kernels.fp8_verify_exact,
            Fp8Kernel::Prefill => &self.kernels.fp8_prefill_exact,
        };
        let linear = Args::new()
            .ptrs(&[s.codes, w.weight, s.scales, w.scale, s.values, s.raw])
            .u32(to_u32(rows)?)
            .u32(to_u32(w.channels)?)
            .u32(to_u32(w.width)?);
        let grid = [
            to_u32(w.channels.div_ceil(tile_columns))?,
            to_u32(rows.div_ceil(tile_rows))?,
            1,
        ];
        self.launch(function, grid, [threads, 1, 1], linear)
    }

    fn nvfp4_projection(
        &self,
        w: &ProjectionWeights,
        input: u64,
        s: &ProjectionSlots,
        rows: usize,
        [input_scale, factor]: [f32; 2],
    ) -> Result<()> {
        ensure!(
            s.effective != 0,
            "NVFP4 projection lacks an effective-scale slot"
        );
        let groups = rows * w.width / 16;
        let quantize = Args::new()
            .ptrs(&[input, s.codes, s.scales, s.effective])
            .u32(to_u32(rows)?)
            .u32(to_u32(w.width)?)
            .f32(input_scale);
        self.launch(
            &self.kernels.nvfp4_quantize,
            [to_u32(groups)?, 1, 1],
            [32, 1, 1],
            quantize,
        )?;
        let (decode, tile_rows, tile_columns, threads) = nvfp4_schedule(rows);
        let function = if decode {
            &self.kernels.nvfp4_decode_exact
        } else {
            &self.kernels.nvfp4_linear
        };
        let linear = Args::new()
            .ptrs(&[s.codes, w.weight, s.scales, w.scale, s.values, s.raw])
            .u32(to_u32(rows)?)
            .u32(to_u32(w.channels)?)
            .u32(to_u32(w.width)?)
            .f32(factor);
        let grid = [
            to_u32(w.channels.div_ceil(tile_columns))?,
            to_u32(rows.div_ceil(tile_rows))?,
            1,
        ];
        self.launch(function, grid, [threads, 1, 1], linear)
    }

    /// `greedy_bf16_tiles` then `greedy_bf16_finish` into a 16-byte result.
    pub(super) fn greedy(
        &self,
        logits: u64,
        partials: u64,
        result: u64,
        vocabulary: usize,
    ) -> Result<()> {
        let vocabulary = to_u32(vocabulary)?;
        let tiles = vocabulary.div_ceil(1024);
        let tile_args = Args::new().ptrs(&[logits, partials]).u32(vocabulary);
        self.launch(
            &self.kernels.greedy_tiles,
            [tiles, 1, 1],
            [128, 1, 1],
            tile_args,
        )?;
        let finish_args = Args::new().ptrs(&[partials, result]).u32(tiles);
        self.launch(
            &self.kernels.greedy_finish,
            [1, 1, 1],
            [128, 1, 1],
            finish_args,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{Args, Fp8Kernel, MAX_ARGS, fp8_schedule, nvfp4_schedule};

    #[test]
    fn fp8_schedule_matches_the_default_exact_profile() {
        assert_eq!(fp8_schedule(1, 248_320), (Fp8Kernel::Exact, 1, 4, 128));
        assert_eq!(fp8_schedule(3, 5120), (Fp8Kernel::Exact, 1, 4, 128));
        assert_eq!(fp8_schedule(4, 5120), (Fp8Kernel::Exact4, 4, 4, 128));
        assert_eq!(fp8_schedule(15, 16_383), (Fp8Kernel::Exact4, 4, 4, 128));
        assert_eq!(fp8_schedule(4, 17_408), (Fp8Kernel::Verify, 8, 16, 32));
        assert_eq!(fp8_schedule(16, 5120), (Fp8Kernel::Prefill, 16, 8, 32));
        assert_eq!(fp8_schedule(512, 17_408), (Fp8Kernel::Prefill, 16, 8, 32));
    }

    #[test]
    fn nvfp4_schedule_matches_the_baseline_profile() {
        assert_eq!(nvfp4_schedule(1), (true, 1, 4, 128));
        assert_eq!(nvfp4_schedule(2), (false, 16, 8, 32));
        assert_eq!(nvfp4_schedule(512), (false, 16, 8, 32));
    }

    #[test]
    fn position_abi_retains_full_device_address_and_scalar_order() {
        use super::super::graph_position::past_argument;
        let address = 0x1234_5678_9abc_def0;
        let initial = || Args::new().ptr(11).ptr(22).u32(1).u32(8);
        let eager = past_argument(initial(), None, 17).u32(64);
        let graph = past_argument(initial(), Some(address), 17).u32(64);
        assert_eq!(&eager.storage[..4], &graph.storage[..4]);
        assert_eq!(eager.storage[4], 17);
        assert_eq!(graph.storage[4], address);
        assert_eq!(eager.storage[5], graph.storage[5]);
        assert_eq!(graph.len, eager.len);
        let prepare = (0..14)
            .fold(Args::new(), |args, i| args.u32(i))
            .ptr(address);
        assert_eq!(prepare.len, 15);
        assert_eq!(prepare.storage[14], address);
        assert!(!prepare.overflow);
    }

    #[test]
    fn scalar_arguments_occupy_low_bytes_and_overflow_is_tracked() {
        let args = Args::new().ptr(7).u32(0xdead_beef).f32(1.5);
        assert_eq!(args.len, 3);
        assert_eq!(
            args.storage[1].to_le_bytes()[..4],
            0xdead_beef_u32.to_le_bytes()
        );
        assert_eq!(args.storage[2], u64::from(1.5_f32.to_bits()));
        let full = (0..=MAX_ARGS).fold(Args::new(), |args, value| args.ptr(value as u64));
        assert!(full.overflow);
    }
}
