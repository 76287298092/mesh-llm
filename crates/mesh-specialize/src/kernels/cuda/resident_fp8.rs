use crate::artifact::schema::{DType, Object, ObjectKind};
use anyhow::{Result, anyhow, ensure};
use std::ffi::c_void;

use super::{
    driver::{Buffer, Context, Module},
    projections,
    resident_weights::ResidentWeights,
};

const MAX_ROWS: usize = 2048;
const MAX_WIDTH: usize = 32_768;
const MAX_CHANNELS: usize = 262_144;
const ROW_MAJOR_LAYOUT: &str = "safetensors-row-major-v1";

/// A reference-free E4M3 linear projection over verified resident weights.
pub(super) struct Projection<'w, 'ctx> {
    owner: &'w ResidentWeights<'ctx>,
    weight_name: String,
    scale_name: String,
    width: usize,
    channels: usize,
}

impl<'w, 'ctx> Projection<'w, 'ctx> {
    /// Bind a projection to exact FP8 weight and BF16 per-channel scale objects.
    pub(super) fn new(
        owner: &'w ResidentWeights<'ctx>,
        prefix: &str,
        width: usize,
        channels: usize,
    ) -> Result<Self> {
        let weight_bytes = validate_dimensions(width, channels)?;
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale");
        let weight = owner.object(&weight_name)?;
        let scales = owner.object(&scale_name)?;
        validate_metadata(
            &weight_name,
            &scale_name,
            width,
            channels,
            weight_bytes,
            weight,
            scales,
        )?;
        Ok(Self {
            owner,
            weight_name,
            scale_name,
            width,
            channels,
        })
    }

    /// Quantize resident BF16 input rows, run the exact FP8 projection, and return device outputs.
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
            "FP8 projection weights, input, and PTX module must belong to the same CUDA context"
        );
        let extents = run_extents(rows, self.width, self.channels)?;
        validate_input_length(input.len(), extents.input_bytes)?;

        let weight_pointer = self.owner.pointer(&self.weight_name)?;
        let scale_pointer = self.owner.pointer(&self.scale_name)?;
        let quantize = module.function("fp8_quantize_bf16")?;
        let (tile_rows, tile_columns, threads, kernel) = if rows >= 16 {
            (16, 8, 32, "fp8_prefill_exact")
        } else if rows >= 4 && self.channels >= 16_384 {
            (8, 16, 32, "fp8_verify_exact")
        } else if rows >= 4 {
            (4, 4, 128, "fp8_linear_exact4")
        } else {
            (1, 4, 128, "fp8_linear_exact")
        };
        let linear = module.function(kernel)?;
        let codes = Buffer::new(context, extents.code_bytes)?;
        let row_scales = Buffer::new(context, extents.row_scale_bytes)?;
        let output = Buffer::new(context, extents.output_bytes)?;
        let unrounded = Buffer::new(context, extents.unrounded_bytes)?;

        let mut pointers = [
            codes.pointer(),
            weight_pointer,
            row_scales.pointer(),
            scale_pointer,
            output.pointer(),
            unrounded.pointer(),
        ];
        let mut dimensions = [
            u32::try_from(rows)?,
            u32::try_from(self.channels)?,
            u32::try_from(self.width)?,
        ];
        let mut arguments: Vec<*mut c_void> = pointers
            .iter_mut()
            .map(|pointer| (pointer as *mut u64).cast())
            .collect();
        arguments.extend(
            dimensions
                .iter_mut()
                .map(|dimension| (dimension as *mut u32).cast()),
        );
        let grid = [
            u32::try_from(self.channels.div_ceil(tile_columns))?,
            u32::try_from(rows.div_ceil(tile_rows))?,
            1,
        ];
        if let Err(error) =
            projections::quantize(&quantize, input, &codes, &row_scales, rows, self.width)
        {
            return Err(synchronize_after_failed_launch(
                context,
                "FP8 input quantization",
                error,
            ));
        }
        // SAFETY: Metadata and run extents validate the row-major inputs, resident weights,
        // temporary FP8/scales, BF16/FP32 outputs, and u32 dimensions. All buffers live through
        // the synchronization below; the selected kernel receives its exact tile/block geometry.
        if let Err(error) = unsafe { linear.launch(grid, [threads, 1, 1], 0, &mut arguments) } {
            return Err(synchronize_after_failed_launch(
                context,
                "exact FP8 projection",
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

/// Device-resident outputs from the projection.
pub(super) struct Output<'ctx> {
    pub(super) values: Buffer<'ctx>,
    pub(super) unrounded: Buffer<'ctx>,
}

struct RunExtents {
    input_bytes: usize,
    code_bytes: usize,
    row_scale_bytes: usize,
    output_bytes: usize,
    unrounded_bytes: usize,
}

fn validate_dimensions(width: usize, channels: usize) -> Result<usize> {
    ensure!(
        (1..=MAX_WIDTH).contains(&width),
        "FP8 projection width must be in 1..={MAX_WIDTH}"
    );
    ensure!(
        (1..=MAX_CHANNELS).contains(&channels),
        "FP8 projection channels must be in 1..={MAX_CHANNELS}"
    );
    checked_product(width, channels, "FP8 projection weight")
}

fn validate_metadata(
    weight_name: &str,
    scale_name: &str,
    width: usize,
    channels: usize,
    weight_bytes: usize,
    weight: &Object,
    scales: &Object,
) -> Result<()> {
    ensure!(
        weight.name == weight_name
            && matches!(&weight.kind, ObjectKind::Tensor)
            && matches!(&weight.dtype, DType::Fp8E4m3)
            && weight.layout == ROW_MAJOR_LAYOUT
            && weight.shape.len() == 2
            && weight.shape[0] == channels as u64
            && weight.shape[1] == width as u64
            && usize::try_from(weight.length).ok() == Some(weight_bytes),
        "FP8 projection weight metadata mismatch for {weight_name}"
    );
    let scale_bytes = checked_product(channels, 2, "FP8 projection scale")?;
    ensure!(
        scales.name == scale_name
            && matches!(&scales.kind, ObjectKind::Tensor)
            && matches!(&scales.dtype, DType::Bf16)
            && scales.layout == ROW_MAJOR_LAYOUT
            && scales.shape.len() == 2
            && scales.shape[0] == channels as u64
            && scales.shape[1] == 1
            && usize::try_from(scales.length).ok() == Some(scale_bytes),
        "FP8 projection scale metadata mismatch for {scale_name}"
    );
    Ok(())
}

fn run_extents(rows: usize, width: usize, channels: usize) -> Result<RunExtents> {
    ensure!(
        (1..=MAX_ROWS).contains(&rows),
        "FP8 projection rows must be in 1..={MAX_ROWS}"
    );
    let input_elements = checked_product(rows, width, "FP8 projection input")?;
    let output_elements = checked_product(rows, channels, "FP8 projection output")?;
    Ok(RunExtents {
        input_bytes: checked_product(input_elements, 2, "BF16 projection input bytes")?,
        code_bytes: input_elements,
        row_scale_bytes: checked_product(rows, 4, "FP32 row scale bytes")?,
        output_bytes: checked_product(output_elements, 2, "BF16 projection output bytes")?,
        unrounded_bytes: checked_product(output_elements, 4, "FP32 projection output bytes")?,
    })
}

fn validate_input_length(actual: usize, expected: usize) -> Result<()> {
    ensure!(
        actual == expected,
        "FP8 projection input has {actual} bytes, expected {expected}"
    );
    Ok(())
}

fn checked_product(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow!("{label} extent overflows usize"))
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_CHANNELS, MAX_ROWS, MAX_WIDTH, ROW_MAJOR_LAYOUT, checked_product, run_extents,
        validate_dimensions, validate_input_length, validate_metadata,
    };
    use crate::artifact::schema::{DType, Object, ObjectKind};

    fn object(name: &str, dtype: DType, shape: Vec<u64>, length: u64) -> Object {
        Object {
            name: name.to_owned(),
            kind: ObjectKind::Tensor,
            dtype,
            shape,
            layout: ROW_MAJOR_LAYOUT.to_owned(),
            offset: 0,
            length,
            sha256: "0".repeat(64),
        }
    }

    #[test]
    fn validates_fp8_and_scale_metadata() {
        let weight = object("layer.weight", DType::Fp8E4m3, vec![3, 2], 6);
        let scales = object("layer.weight_scale", DType::Bf16, vec![3, 1], 6);
        validate_metadata(
            "layer.weight",
            "layer.weight_scale",
            2,
            3,
            6,
            &weight,
            &scales,
        )
        .unwrap();
        let wrong_shape = object("layer.weight", DType::Fp8E4m3, vec![2, 3], 6);
        assert!(
            validate_metadata(
                "layer.weight",
                "layer.weight_scale",
                2,
                3,
                6,
                &wrong_shape,
                &scales
            )
            .is_err()
        );
        let wrong_weight_layout = Object {
            layout: "raw-v1".to_owned(),
            ..weight.clone()
        };
        assert!(
            validate_metadata(
                "layer.weight",
                "layer.weight_scale",
                2,
                3,
                6,
                &wrong_weight_layout,
                &scales
            )
            .is_err()
        );
        let wrong_scale_layout = Object {
            layout: "raw-v1".to_owned(),
            ..scales.clone()
        };
        assert!(
            validate_metadata(
                "layer.weight",
                "layer.weight_scale",
                2,
                3,
                6,
                &weight,
                &wrong_scale_layout
            )
            .is_err()
        );
        let wrong_dtype = object("layer.weight_scale", DType::Fp8E4m3, vec![3, 1], 6);
        assert!(
            validate_metadata(
                "layer.weight",
                "layer.weight_scale",
                2,
                3,
                6,
                &weight,
                &wrong_dtype
            )
            .is_err()
        );
    }

    #[test]
    fn validates_dimension_bounds_and_products() {
        assert!(validate_dimensions(1, 1).is_ok());
        assert!(validate_dimensions(MAX_WIDTH, MAX_CHANNELS).is_ok());
        assert!(validate_dimensions(0, 1).is_err());
        assert!(validate_dimensions(MAX_WIDTH + 1, 1).is_err());
        assert!(validate_dimensions(1, 0).is_err());
        assert!(validate_dimensions(1, MAX_CHANNELS + 1).is_err());
        assert!(checked_product(usize::MAX, 2, "test").is_err());
    }

    #[test]
    fn checks_input_extent_and_checked_output_sizes() {
        assert!(validate_input_length(12, 12).is_ok());
        assert!(validate_input_length(10, 12).is_err());
        let extents = run_extents(MAX_ROWS, MAX_WIDTH, MAX_CHANNELS).unwrap();
        assert_eq!(extents.input_bytes, MAX_ROWS * MAX_WIDTH * 2);
        assert_eq!(extents.output_bytes, MAX_ROWS * MAX_CHANNELS * 2);
        assert_eq!(extents.unrounded_bytes, MAX_ROWS * MAX_CHANNELS * 4);
        assert!(run_extents(0, 1, 1).is_err());
        assert!(run_extents(MAX_ROWS + 1, 1, 1).is_err());
    }
}
