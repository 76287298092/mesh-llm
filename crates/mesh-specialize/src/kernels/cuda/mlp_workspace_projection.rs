//! Enqueue-only projection operations over a checked MLP workspace lease.
use super::{
    driver::{Context, Module},
    resident_weights::ResidentWeights,
    resident_workspace::WorkspaceStep,
};
use anyhow::{Result, ensure};
use std::ffi::c_void;

pub(super) enum Arithmetic {
    Fp8,
    Nvfp4 { input_scale: f32, factor: f32 },
}
pub(super) struct Binding<'w, 'ctx> {
    pub owner: &'w ResidentWeights<'ctx>,
    pub weights: [u64; 2],
    pub width: usize,
    pub channels: usize,
    pub arithmetic: Arithmetic,
}
impl Binding<'_, '_> {
    pub(super) fn regions(&self, name: &str, rows: usize) -> Vec<(String, usize)> {
        let elements = rows * self.channels;
        let (codes, scales, effective) = match self.arithmetic {
            Arithmetic::Fp8 => (rows * self.width, rows * 4, 4),
            Arithmetic::Nvfp4 { .. } => (
                rows * self.width / 2,
                rows * self.width / 16,
                rows * self.width / 16 * 4,
            ),
        };
        [
            ("codes", codes),
            ("scales", scales),
            ("effective", effective),
            ("values", elements * 2),
            ("raw", elements * 4),
        ]
        .into_iter()
        .map(|(suffix, bytes)| (format!("{name}.{suffix}"), bytes))
        .collect()
    }
    /// # Safety
    /// Input is a live BF16 [rows,width] region in ctx, disjoint from every
    /// planned output; the matching checked lease and weights outlive completion.
    pub(super) unsafe fn enqueue(
        &self,
        ctx: &Context,
        module: &Module<'_>,
        step: &WorkspaceStep<'_, '_>,
        name: &str,
        input: u64,
        rows: usize,
    ) -> Result<()> {
        ensure!(
            self.owner.belongs_to(ctx) && module.belongs_to(ctx),
            "workspace projection context mismatch"
        );
        for (region_name, bytes) in self.regions(name, rows) {
            ensure!(
                step.region(&region_name)?.bytes() >= bytes,
                "undersized MLP workspace region {region_name}"
            );
        }
        let region = |suffix: &str| -> Result<u64> {
            Ok(step.region(&format!("{name}.{suffix}"))?.pointer())
        };
        let codes = region("codes")?;
        let scales = region("scales")?;
        match self.arithmetic {
            Arithmetic::Fp8 => launch(
                module,
                "fp8_quantize_bf16",
                vec![input, codes, scales],
                vec![self.width as u32],
                None,
                [rows as u32, 1, 1],
                256,
            )?,
            Arithmetic::Nvfp4 { input_scale, .. } => launch(
                module,
                "nvfp4_quantize_bf16",
                vec![input, codes, scales, region("effective")?],
                vec![rows as u32, self.width as u32],
                Some(input_scale),
                [(rows * self.width / 16) as u32, 1, 1],
                32,
            )?,
        }
        let (kernel, tile_rows, tile_channels, threads, factor) = match self.arithmetic {
            Arithmetic::Fp8 if rows >= 16 => ("fp8_prefill_exact", 16, 8, 32, None),
            Arithmetic::Fp8 if rows >= 4 && self.channels >= 16384 => {
                ("fp8_verify_exact", 8, 16, 32, None)
            }
            Arithmetic::Fp8 if rows >= 4 => ("fp8_linear_exact4", 4, 4, 128, None),
            Arithmetic::Fp8 => ("fp8_linear_exact", 1, 4, 128, None),
            Arithmetic::Nvfp4 { factor, .. } if rows == 1 => {
                ("nvfp4_decode_exact", 1, 4, 128, Some(factor))
            }
            Arithmetic::Nvfp4 { factor, .. } => ("nvfp4_linear", 16, 8, 32, Some(factor)),
        };
        launch(
            module,
            kernel,
            vec![
                codes,
                self.weights[0],
                scales,
                self.weights[1],
                region("values")?,
                region("raw")?,
            ],
            vec![rows as u32, self.channels as u32, self.width as u32],
            factor,
            [
                self.channels.div_ceil(tile_channels) as u32,
                rows.div_ceil(tile_rows) as u32,
                1,
            ],
            threads,
        )
    }
}

/// Only called from checked chain construction/lease operations; no pointers escape the lease.
fn launch(
    module: &Module<'_>,
    kernel: &str,
    mut pointers: Vec<u64>,
    mut dimensions: Vec<u32>,
    mut scalar: Option<f32>,
    grid: [u32; 3],
    threads: u32,
) -> Result<()> {
    let mut args = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(dimensions.iter_mut().map(|d| (d as *mut u32).cast()));
    if let Some(value) = scalar.as_mut() {
        args.push((value as *mut f32).cast());
    }
    // SAFETY: Binding metadata and chain layout bound every ABI pointer and dimension;
    // ordered launches retain the exclusive workspace lease through completion/drain.
    unsafe {
        module
            .function(kernel)?
            .launch(grid, [threads, 1, 1], 0, &mut args)
    }
}
