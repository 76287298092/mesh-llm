//! Fixed MLP projection launches with pre-resolved functions and borrowed weights.
//!
//! This module builds only stack arguments during enqueue. The parent driver
//! change keeps operation-name formatting on its error path. No completion lease
//! or graph resource ownership is provided here; CUDA internals are not audited
//! for allocations by this source-level contract.

use super::{
    device_view::{DeviceRead, DeviceWrite, validate_launch_access},
    driver::{Context, Function, Module, graph::Stream},
    mlp_workspace_projection::{Arithmetic, Binding},
    resident_weights::ResidentWeights,
};
use crate::kernels::nvfp4_profile;
use anyhow::{Result, ensure};
use std::{ffi::c_void, ptr};

pub(super) struct Prepared<'module, 'w, 'ctx> {
    context: &'ctx Context,
    _owner: &'w ResidentWeights<'ctx>,
    quantize: Function<'module, 'ctx>,
    linear: Function<'module, 'ctx>,
    weights: [u64; 2],
    plan: Plan,
}

#[derive(Clone, Copy)]
enum Quantization {
    Fp8,
    Nvfp4 { input_scale: f32, factor: f32 },
}

struct Plan {
    quantization: Quantization,
    quantizer: &'static str,
    linear: &'static str,
    dimensions: [u32; 3], // rows, output channels, input width
    quantize_grid: [u32; 3],
    quantize_block: [u32; 3],
    linear_grid: [u32; 3],
    linear_block: [u32; 3],
    input_bytes: usize,
    output_bytes: [usize; 5], // codes, scales, effective, values, raw
}

impl<'module, 'w, 'ctx> Prepared<'module, 'w, 'ctx> {
    pub(super) fn new(
        ctx: &'ctx Context,
        module: &'module Module<'ctx>,
        binding: &Binding<'w, 'ctx>,
        rows: usize,
    ) -> Result<Self> {
        ensure!(
            binding.owner.belongs_to(ctx) && module.belongs_to(ctx),
            "prepared projection context mismatch"
        );
        ensure!(
            binding.weights.iter().all(|&address| address != 0),
            "prepared projection has a null weight address"
        );
        let quantization = match binding.arithmetic {
            Arithmetic::Fp8 => Quantization::Fp8,
            Arithmetic::Nvfp4 {
                input_scale,
                factor,
                ..
            } => Quantization::Nvfp4 {
                input_scale,
                factor,
            },
        };
        let profile = if matches!(quantization, Quantization::Nvfp4 { .. }) {
            nvfp4_profile::current()?
        } else {
            nvfp4_profile::Profile::Baseline
        };
        let mut plan = Plan::new(
            [rows, binding.channels, binding.width],
            quantization,
            profile,
        )?;
        if rows == 1 && plan.linear == crate::kernels::nvfp4_decode_schedule::BASELINE_KERNEL {
            plan.linear = crate::kernels::nvfp4_decode_schedule::current()?.kernel();
        }
        Ok(Self {
            context: ctx,
            _owner: binding.owner,
            quantize: module.function(plan.quantizer)?,
            linear: module.function(plan.linear)?,
            weights: binding.weights,
            plan,
        })
    }

    /// Enqueue quantization and projection without completing either operation.
    ///
    /// # Safety
    /// Binding addresses must describe the verified weights held by their owner.
    /// All input/output, module and weight owners must survive completion or error
    /// draining on this stream, including failure after quantization is submitted.
    /// Prevent conflicting writes or reuse on other streams. Stack argument storage
    /// survives each driver call, but device views do not retain submitted work.
    /// If used during capture, the caller must separately retain every resource at
    /// its fixed address for all graph/executable lifetimes and replay completion.
    pub(super) unsafe fn enqueue(
        &self,
        stream: &Stream<'ctx>,
        input: &DeviceRead<'_, '_>,
        outputs: &[DeviceWrite<'_, '_>; 5],
    ) -> Result<()> {
        self.validate_views(input, outputs)?;
        let addresses = outputs.each_ref().map(DeviceWrite::pointer);
        // SAFETY: The caller retains all resources; validation covers device view
        // ranges and contexts. The driver checks stream/function context identity.
        unsafe {
            self.enqueue_quantize(stream, input.pointer(), addresses)?;
            self.enqueue_linear(stream, addresses)
        }
    }

    fn validate_views(
        &self,
        input: &DeviceRead<'_, '_>,
        outputs: &[DeviceWrite<'_, '_>; 5],
    ) -> Result<()> {
        ensure!(
            input.bytes() == self.plan.input_bytes && input.pointer().is_multiple_of(2),
            "prepared input extent/alignment mismatch"
        );
        for (output, bytes) in outputs.iter().zip(self.plan.output_bytes) {
            ensure!(
                output.bytes() >= bytes && output.pointer().is_multiple_of(4),
                "prepared output extent/alignment mismatch"
            );
        }
        validate_launch_access(self.context, &[input], &outputs.each_ref())
    }

    unsafe fn enqueue_quantize(
        &self,
        stream: &Stream<'ctx>,
        mut input: u64,
        outputs: [u64; 5],
    ) -> Result<()> {
        let [mut codes, mut scales, mut effective, _, _] = outputs;
        let [mut rows, _, mut width] = self.plan.dimensions;
        match self.plan.quantization {
            Quantization::Fp8 => {
                let mut args = [
                    argument(&mut input),
                    argument(&mut codes),
                    argument(&mut scales),
                    argument(&mut width),
                ];
                // SAFETY: Fixed FP8 pointer/width ABI; backing owners are caller-retained.
                unsafe {
                    self.quantize.launch_on_stream(
                        stream,
                        self.plan.quantize_grid,
                        self.plan.quantize_block,
                        0,
                        &mut args,
                    )
                }
            }
            Quantization::Nvfp4 {
                mut input_scale, ..
            } => {
                let mut args = [
                    argument(&mut input),
                    argument(&mut codes),
                    argument(&mut scales),
                    argument(&mut effective),
                    argument(&mut rows),
                    argument(&mut width),
                    argument(&mut input_scale),
                ];
                // SAFETY: Fixed NVFP4 four-pointer/two-u32/f32 quantization ABI.
                unsafe {
                    self.quantize.launch_on_stream(
                        stream,
                        self.plan.quantize_grid,
                        self.plan.quantize_block,
                        0,
                        &mut args,
                    )
                }
            }
        }
    }

    unsafe fn enqueue_linear(&self, stream: &Stream<'ctx>, outputs: [u64; 5]) -> Result<()> {
        let [mut codes, mut scales, _, mut values, mut raw] = outputs;
        let [mut weight, mut weight_scale] = self.weights;
        let [mut rows, mut channels, mut width] = self.plan.dimensions;
        let (mut factor, argument_count) = match self.plan.quantization {
            Quantization::Fp8 => (0.0_f32, 9),
            Quantization::Nvfp4 { factor, .. } => (factor, 10),
        };
        let mut args = [
            argument(&mut codes),
            argument(&mut weight),
            argument(&mut scales),
            argument(&mut weight_scale),
            argument(&mut values),
            argument(&mut raw),
            argument(&mut rows),
            argument(&mut channels),
            argument(&mut width),
            argument(&mut factor),
        ];
        // SAFETY: Six-pointer/mnk ABI, with trailing factor only for NVFP4.
        // All scalars stay on this stack through the call; owners survive completion.
        unsafe {
            self.linear.launch_on_stream(
                stream,
                self.plan.linear_grid,
                self.plan.linear_block,
                0,
                &mut args[..argument_count],
            )
        }
    }
}

fn argument<T>(value: &mut T) -> *mut c_void {
    ptr::from_mut(value).cast()
}

impl Plan {
    fn new(
        dimensions: [usize; 3],
        quantization: Quantization,
        profile: nvfp4_profile::Profile,
    ) -> Result<Self> {
        let [rows, channels, width] = dimensions;
        ensure!(
            (1..=512).contains(&rows)
                && (1..=32768).contains(&channels)
                && (1..=32768).contains(&width),
            "prepared MLP geometry outside admitted bounds"
        );
        let input = rows
            .checked_mul(width)
            .ok_or_else(|| anyhow::anyhow!("prepared input count overflows"))?;
        let output = rows
            .checked_mul(channels)
            .ok_or_else(|| anyhow::anyhow!("prepared output count overflows"))?;
        let (quantizer, quantize_grid, quantize_block, scratch) = match quantization {
            Quantization::Fp8 => (
                "fp8_quantize_bf16",
                [u32::try_from(rows)?, 1, 1],
                [256, 1, 1],
                [input, rows * 4, 4],
            ),
            Quantization::Nvfp4 {
                input_scale,
                factor,
            } => {
                ensure!(
                    width.is_multiple_of(16),
                    "NVFP4 width must be a multiple of 16"
                );
                ensure!(
                    input_scale.is_finite()
                        && input_scale > 0.0
                        && factor.is_finite()
                        && factor > 0.0,
                    "invalid NVFP4 prepared scale/factor"
                );
                (
                    "nvfp4_quantize_bf16",
                    [u32::try_from(input / 16)?, 1, 1],
                    [32, 1, 1],
                    [input / 2, input / 16, input / 16 * 4],
                )
            }
        };
        let schedule = schedule(dimensions, quantization, profile);
        Ok(Self {
            quantization,
            quantizer,
            linear: schedule.kernel,
            dimensions: [
                u32::try_from(rows)?,
                u32::try_from(channels)?,
                u32::try_from(width)?,
            ],
            quantize_grid,
            quantize_block,
            linear_grid: [
                u32::try_from(channels.div_ceil(schedule.tile_columns))?,
                u32::try_from(rows.div_ceil(schedule.tile_rows))?,
                1,
            ],
            linear_block: [schedule.threads, 1, 1],
            input_bytes: input
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("prepared input bytes overflow"))?,
            output_bytes: [scratch[0], scratch[1], scratch[2], output * 2, output * 4],
        })
    }
}

// Keep this FP8 dispatch in lockstep with Binding::enqueue until parent extracts
// a shared selector. NVFP4 delegates to its canonical process-fixed profile.
fn schedule(
    [rows, channels, width]: [usize; 3],
    quantization: Quantization,
    profile: nvfp4_profile::Profile,
) -> nvfp4_profile::Schedule {
    if matches!(quantization, Quantization::Nvfp4 { .. }) {
        return profile.schedule(rows, channels, width);
    }
    let (kernel, tile_rows, tile_columns, threads) = if rows >= 16 {
        ("fp8_prefill_exact", 16, 8, 32)
    } else if rows >= 4 && channels >= 16384 {
        ("fp8_verify_exact", 8, 16, 32)
    } else if rows >= 4 {
        ("fp8_linear_exact4", 4, 4, 128)
    } else {
        ("fp8_linear_exact", 1, 4, 128)
    };
    nvfp4_profile::Schedule {
        kernel,
        tile_rows,
        tile_columns,
        threads,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nvfp4_profile::Profile;

    #[test]
    fn fp8_dispatch_boundaries_and_scratch_match_workspace() {
        for (rows, channels, kernel, grid, threads) in [
            (1, 5120, "fp8_linear_exact", [1280, 1, 1], 128),
            (3, 17408, "fp8_linear_exact", [4352, 3, 1], 128),
            (4, 16383, "fp8_linear_exact4", [4096, 1, 1], 128),
            (4, 16384, "fp8_verify_exact", [1024, 1, 1], 32),
            (15, 17408, "fp8_verify_exact", [1088, 2, 1], 32),
            (16, 17408, "fp8_prefill_exact", [2176, 1, 1], 32),
        ] {
            let p =
                Plan::new([rows, channels, 5120], Quantization::Fp8, Profile::Baseline).unwrap();
            assert_eq!(p.linear, kernel);
            assert_eq!(p.linear_grid, grid);
            assert_eq!(p.linear_block, [threads, 1, 1]);
            assert_eq!(
                p.output_bytes,
                [
                    rows * 5120,
                    rows * 4,
                    4,
                    rows * channels * 2,
                    rows * channels * 4
                ]
            );
        }
    }

    #[test]
    fn nvfp4_uses_profile_schedule_and_preserves_scalar_bits() {
        let q = Quantization::Nvfp4 {
            input_scale: 836.0,
            factor: 1.0 / 5350400.0,
        };
        for profile in [
            Profile::Baseline,
            Profile::TiledPrefill,
            Profile::WidePrefill,
        ] {
            for rows in [1, 5, 16, 33, 512] {
                let p = Plan::new([rows, 17408, 5120], q, profile).unwrap();
                let expected = profile.schedule(rows, 17408, 5120);
                assert_eq!(p.linear, expected.kernel);
                assert_eq!(
                    p.linear_grid,
                    [
                        17408_usize.div_ceil(expected.tile_columns) as u32,
                        rows.div_ceil(expected.tile_rows) as u32,
                        1
                    ]
                );
                assert_eq!(p.linear_block, [expected.threads, 1, 1]);
                assert_eq!(
                    p.output_bytes,
                    [
                        rows * 2560,
                        rows * 320,
                        rows * 1280,
                        rows * 17408 * 2,
                        rows * 17408 * 4
                    ]
                );
                let Quantization::Nvfp4 {
                    input_scale,
                    factor,
                } = p.quantization
                else {
                    panic!("wrong quantization")
                };
                assert_eq!(input_scale.to_bits(), 836.0_f32.to_bits());
                assert_eq!(factor.to_bits(), (1.0_f32 / 5350400.0).to_bits());
            }
        }
    }

    #[test]
    fn rejects_invalid_geometry_and_nonfinite_scales() {
        for shape in [
            [0, 16, 16],
            [513, 16, 16],
            [1, 0, 16],
            [1, 32769, 16],
            [1, 16, usize::MAX],
        ] {
            assert!(Plan::new(shape, Quantization::Fp8, Profile::Baseline).is_err());
        }
        for input_scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let q = Quantization::Nvfp4 {
                input_scale,
                factor: 1.0,
            };
            assert!(Plan::new([1, 16, 16], q, Profile::Baseline).is_err());
        }
        let q = Quantization::Nvfp4 {
            input_scale: 1.0,
            factor: f32::NAN,
        };
        assert!(Plan::new([1, 16, 16], q, Profile::Baseline).is_err());
        let q = Quantization::Nvfp4 {
            input_scale: 1.0,
            factor: 1.0,
        };
        assert!(Plan::new([1, 16, 17], q, Profile::Baseline).is_err());
    }
}
