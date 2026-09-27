use crate::artifact::schema::DType;
use anyhow::{Result, anyhow, ensure};
use std::ffi::c_void;

use super::{
    driver::{Buffer, Context, Function, Module},
    resident_fp8::Output,
    resident_weights::ResidentWeights,
};

const MIN_WIDTH: usize = 16;
const MAX_WIDTH: usize = 32_768;
const MAX_CHANNELS: usize = 32_768;
const MAX_ROWS: usize = 2_048;

/// A reference-free NVFP4 projection over verified, device-resident weights.
pub(super) struct Projection<'w, 'ctx> {
    owner: &'w ResidentWeights<'ctx>,
    weight_pointer: u64,
    scale_pointer: u64,
    width: usize,
    channels: usize,
    input_global_scale: f32,
    global_factor: f32,
}

impl<'w, 'ctx> Projection<'w, 'ctx> {
    /// Bind the projection to the checkpoint's packed weights, scales, and global multipliers.
    pub(super) fn new(
        owner: &'w ResidentWeights<'ctx>,
        prefix: &str,
        width: usize,
        channels: usize,
    ) -> Result<Self> {
        let weight_extents = validate_dimensions(width, channels)?;
        let weight_name = format!("{prefix}.weight_packed");
        let scale_name = format!("{prefix}.weight_scale");
        let weight_shape = [u64::try_from(channels)?, u64::try_from(width / 2)?];
        let scale_shape = [u64::try_from(channels)?, u64::try_from(width / 16)?];
        let weight_pointer = owner.tensor(
            &weight_name,
            DType::U8,
            &weight_shape,
            u64::try_from(weight_extents.packed_bytes)?,
        )?;
        let scale_pointer = owner.tensor(
            &scale_name,
            DType::Fp8E4m3,
            &scale_shape,
            u64::try_from(weight_extents.scale_bytes)?,
        )?;
        let input_global_scale = owner.positive_scalar(&format!("{prefix}.input_global_scale"))?;
        let weight_global_scale =
            owner.positive_scalar(&format!("{prefix}.weight_global_scale"))?;
        let global_factor = global_factor(input_global_scale, weight_global_scale)?;
        Ok(Self {
            owner,
            weight_pointer,
            scale_pointer,
            width,
            channels,
            input_global_scale,
            global_factor,
        })
    }

    /// Quantize BF16 rows on-device, run NVFP4 MMA, and return device-resident outputs.
    pub(super) fn run<'a>(
        &self,
        context: &'a Context,
        module: &Module<'_>,
        input: &Buffer<'_>,
        rows: usize,
    ) -> Result<Output<'a>> {
        ensure!(
            self.owner.belongs_to(context)
                && input.belongs_to(context)
                && module.belongs_to(context),
            "NVFP4 projection weights, input, and PTX module must belong to the same CUDA context"
        );
        let extents = run_extents(rows, self.width, self.channels)?;
        ensure!(
            input.len() == extents.input_bytes,
            "NVFP4 projection input has {} bytes, expected {}",
            input.len(),
            extents.input_bytes
        );
        let rows_u32 = u32::try_from(rows)?;
        let width_u32 = u32::try_from(self.width)?;
        let channels_u32 = u32::try_from(self.channels)?;
        let quantize = module.function("nvfp4_quantize_bf16")?;
        let linear = module.function(if rows == 1 {
            "nvfp4_decode_exact"
        } else {
            "nvfp4_linear"
        })?;
        let activation = ActivationBuffers::new(context, &extents)?;
        let output = Buffer::new(context, extents.output_bytes)?;
        let unrounded = Buffer::new(context, extents.unrounded_bytes)?;

        if let Err(error) = launch_quantizer(
            &quantize,
            input,
            &activation,
            rows_u32,
            width_u32,
            self.input_global_scale,
            extents.quantizer_grid,
        ) {
            return Err(synchronize_after_failed_launch(
                context,
                "NVFP4 input quantization",
                error,
            ));
        }
        let linear_launch = LinearLaunch {
            activation: &activation,
            weight_pointer: self.weight_pointer,
            scale_pointer: self.scale_pointer,
            output: &output,
            unrounded: &unrounded,
            dimensions: [rows_u32, channels_u32, width_u32],
            global_factor: self.global_factor,
            grid: if rows == 1 {
                [channels_u32.div_ceil(4), 1, 1]
            } else {
                extents.linear_grid
            },
        };
        if let Err(error) = launch_linear(&linear, linear_launch) {
            return Err(synchronize_after_failed_launch(
                context,
                "NVFP4 linear projection",
                error,
            ));
        }
        context.synchronize()?;
        Ok(Output {
            values: output,
            unrounded,
        })
    }
}

struct WeightExtents {
    packed_bytes: usize,
    scale_bytes: usize,
}

fn validate_dimensions(width: usize, channels: usize) -> Result<WeightExtents> {
    ensure!(
        (MIN_WIDTH..=MAX_WIDTH).contains(&width) && width.is_multiple_of(16),
        "NVFP4 projection width must be a multiple of 16 in {MIN_WIDTH}..={MAX_WIDTH}"
    );
    ensure!(
        (1..=MAX_CHANNELS).contains(&channels),
        "NVFP4 projection channels must be in 1..={MAX_CHANNELS}"
    );
    let values = checked_product(channels, width, "NVFP4 projection weights")?;
    Ok(WeightExtents {
        packed_bytes: values / 2,
        scale_bytes: values / 16,
    })
}

struct RunExtents {
    input_bytes: usize,
    packed_bytes: usize,
    scale_bytes: usize,
    effective_bytes: usize,
    output_bytes: usize,
    unrounded_bytes: usize,
    quantizer_grid: [u32; 3],
    linear_grid: [u32; 3],
}

fn run_extents(rows: usize, width: usize, channels: usize) -> Result<RunExtents> {
    ensure!(
        (1..=MAX_ROWS).contains(&rows),
        "NVFP4 projection rows must be in 1..={MAX_ROWS}"
    );
    let input_values = checked_product(rows, width, "NVFP4 projection input")?;
    let output_values = checked_product(rows, channels, "NVFP4 projection output")?;
    let groups = input_values / 16;
    Ok(RunExtents {
        input_bytes: checked_product(input_values, 2, "NVFP4 BF16 input bytes")?,
        packed_bytes: input_values / 2,
        scale_bytes: groups,
        effective_bytes: checked_product(groups, 4, "NVFP4 effective scale bytes")?,
        output_bytes: checked_product(output_values, 2, "NVFP4 BF16 output bytes")?,
        unrounded_bytes: checked_product(output_values, 4, "NVFP4 FP32 output bytes")?,
        quantizer_grid: [u32::try_from(groups)?, 1, 1],
        linear_grid: [
            u32::try_from(channels.div_ceil(8))?,
            u32::try_from(rows.div_ceil(16))?,
            1,
        ],
    })
}

fn global_factor(input_global: f32, weight_global: f32) -> Result<f32> {
    let product = input_global * weight_global;
    let factor = 1.0 / product;
    ensure!(
        [input_global, weight_global, product, factor]
            .iter()
            .all(|value| value.is_finite() && *value > 0.0),
        "NVFP4 projection global scales and reciprocal factor must be finite and positive"
    );
    Ok(factor)
}

struct ActivationBuffers<'ctx> {
    packed: Buffer<'ctx>,
    scales: Buffer<'ctx>,
    effective: Buffer<'ctx>,
}

impl<'ctx> ActivationBuffers<'ctx> {
    fn new(context: &'ctx Context, extents: &RunExtents) -> Result<Self> {
        Ok(Self {
            packed: Buffer::new(context, extents.packed_bytes)?,
            scales: Buffer::new(context, extents.scale_bytes)?,
            effective: Buffer::new(context, extents.effective_bytes)?,
        })
    }
}

fn launch_quantizer(
    function: &Function<'_, '_>,
    input: &Buffer<'_>,
    activation: &ActivationBuffers<'_>,
    rows: u32,
    width: u32,
    global_scale: f32,
    grid: [u32; 3],
) -> Result<()> {
    let mut pointers = [
        input.pointer(),
        activation.packed.pointer(),
        activation.scales.pointer(),
        activation.effective.pointer(),
    ];
    let mut dimensions = [rows, width];
    let mut global_scale = global_scale;
    let mut arguments: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|pointer| (pointer as *mut u64).cast())
        .collect();
    arguments.extend(
        dimensions
            .iter_mut()
            .map(|dimension| (dimension as *mut u32).cast()),
    );
    arguments.push((&mut global_scale as *mut f32).cast());
    // SAFETY: Exact checked buffers and scalar storage match the quantizer ABI. The input and all
    // activation allocations remain live through the linear launch and final synchronization.
    unsafe { function.launch(grid, [32, 1, 1], 0, &mut arguments) }
}

struct LinearLaunch<'a, 'ctx> {
    activation: &'a ActivationBuffers<'ctx>,
    weight_pointer: u64,
    scale_pointer: u64,
    output: &'a Buffer<'ctx>,
    unrounded: &'a Buffer<'ctx>,
    dimensions: [u32; 3],
    global_factor: f32,
    grid: [u32; 3],
}

fn launch_linear(function: &Function<'_, '_>, launch: LinearLaunch<'_, '_>) -> Result<()> {
    let mut pointers = [
        launch.activation.packed.pointer(),
        launch.weight_pointer,
        launch.activation.scales.pointer(),
        launch.scale_pointer,
        launch.output.pointer(),
        launch.unrounded.pointer(),
    ];
    let mut dimensions = launch.dimensions;
    let mut global_factor = launch.global_factor;
    let mut arguments: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|pointer| (pointer as *mut u64).cast())
        .collect();
    arguments.extend(
        dimensions
            .iter_mut()
            .map(|dimension| (dimension as *mut u32).cast()),
    );
    arguments.push((&mut global_factor as *mut f32).cast());
    // SAFETY: The verified resident weight views, generated activation buffers, and distinct
    // output allocations match the six-pointer/mnk/factor ABI and remain live through sync.
    unsafe {
        function.launch(
            launch.grid,
            if launch.dimensions[0] == 1 {
                [128, 1, 1]
            } else {
                [32, 1, 1]
            },
            0,
            &mut arguments,
        )
    }
}

fn synchronize_after_failed_launch(
    context: &Context,
    operation: &str,
    error: anyhow::Error,
) -> anyhow::Error {
    let context_message = match context.synchronize() {
        Ok(()) => format!("{operation} launch failed"),
        Err(sync_error) => {
            format!("{operation} launch failed; CUDA synchronization also failed: {sync_error:#}")
        }
    };
    error.context(context_message)
}

fn checked_product(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow!("{label} extent overflows usize"))
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_CHANNELS, MAX_ROWS, MAX_WIDTH, MIN_WIDTH, global_factor, run_extents,
        validate_dimensions,
    };

    #[test]
    fn validates_nvfp4_projection_dimensions() {
        let extents = validate_dimensions(MIN_WIDTH, 1).unwrap();
        assert_eq!(extents.packed_bytes, 8);
        assert_eq!(extents.scale_bytes, 1);
        assert!(validate_dimensions(MIN_WIDTH - 1, 1).is_err());
        assert!(validate_dimensions(MIN_WIDTH + 1, 1).is_err());
        assert!(validate_dimensions(MAX_WIDTH + 1, 1).is_err());
        assert!(validate_dimensions(MIN_WIDTH, 0).is_err());
        assert!(validate_dimensions(MIN_WIDTH, MAX_CHANNELS + 1).is_err());
        assert!(validate_dimensions(MAX_WIDTH, MAX_CHANNELS).is_ok());
    }

    #[test]
    fn validates_runtime_extents_and_launch_grids() {
        let extents = run_extents(17, 32, 13).unwrap();
        assert_eq!(extents.input_bytes, 17 * 32 * 2);
        assert_eq!(extents.packed_bytes, 17 * 16);
        assert_eq!(extents.scale_bytes, 34);
        assert_eq!(extents.effective_bytes, 34 * 4);
        assert_eq!(extents.output_bytes, 17 * 13 * 2);
        assert_eq!(extents.unrounded_bytes, 17 * 13 * 4);
        assert_eq!(extents.quantizer_grid, [34, 1, 1]);
        assert_eq!(extents.linear_grid, [2, 2, 1]);
        assert!(run_extents(0, 16, 1).is_err());
        assert!(run_extents(MAX_ROWS + 1, 16, 1).is_err());
    }

    #[test]
    fn checks_reciprocal_global_scale_profile() {
        let factor = global_factor(836.0, 6_400.0).unwrap();
        assert_eq!(factor, 1.0 / (836.0_f32 * 6_400.0_f32));
        for (input, weight) in [
            (0.0, 1.0),
            (f32::NAN, 1.0),
            (1.0, f32::INFINITY),
            (f32::MAX, f32::MAX),
            (f32::MIN_POSITIVE, f32::MIN_POSITIVE),
        ] {
            assert!(global_factor(input, weight).is_err());
        }
    }
}
