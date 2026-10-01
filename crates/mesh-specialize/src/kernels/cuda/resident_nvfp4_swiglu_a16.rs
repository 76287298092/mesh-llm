use super::{
    driver::{Buffer, Context, Module},
    resident_fp8::Output,
    resident_nvfp4::Projection,
};
use anyhow::{Result, ensure};
use std::ffi::c_void;

pub(super) struct OutputBuffers<'ctx> {
    pub(super) gate: Output<'ctx>,
    pub(super) up: Output<'ctx>,
    pub(super) activation: Buffer<'ctx>,
}

pub(super) fn run<'ctx>(
    context: &'ctx Context,
    module: &Module<'_>,
    input: &Buffer<'_>,
    gate: &Projection<'_, '_>,
    up: &Projection<'_, '_>,
) -> Result<OutputBuffers<'ctx>> {
    validate(context, module, input, gate, up)?;
    let channels = gate.channels;
    let gate_output = Buffer::new(context, checked_bytes(channels, 2, "gate output")?)?;
    let gate_raw = Buffer::new(context, checked_bytes(channels, 4, "gate raw output")?)?;
    let up_output = Buffer::new(context, checked_bytes(channels, 2, "up output")?)?;
    let up_raw = Buffer::new(context, checked_bytes(channels, 4, "up raw output")?)?;
    let activation_raw = Buffer::new(context, checked_bytes(channels, 4, "SwiGLU raw output")?)?;
    let activation = Buffer::new(context, checked_bytes(channels, 2, "SwiGLU output")?)?;
    let mut pointers = [
        input.pointer(),
        gate.weight_pointer,
        gate.scale_pointer,
        up.weight_pointer,
        up.scale_pointer,
        gate_output.pointer(),
        gate_raw.pointer(),
        up_output.pointer(),
        up_raw.pointer(),
        activation_raw.pointer(),
        activation.pointer(),
    ];
    let mut dimensions = [u32::try_from(channels)?, u32::try_from(gate.width)?];
    let mut divisors = [gate.inverse_weight_divisor, up.inverse_weight_divisor];
    let mut arguments: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|pointer| (pointer as *mut u64).cast())
        .collect();
    arguments.extend(
        dimensions
            .iter_mut()
            .map(|dimension| (dimension as *mut u32).cast()),
    );
    arguments.extend(
        divisors
            .iter_mut()
            .map(|divisor| (divisor as *mut f32).cast()),
    );
    let launched = unsafe {
        // SAFETY: Ten distinct live buffers and two validated divisors match the fixed entry ABI.
        module.function("nvfp4_swiglu_a16")?.launch(
            [u32::try_from(channels.div_ceil(8))?, 1, 1],
            [256, 1, 1],
            0,
            &mut arguments,
        )
    };
    if let Err(error) = launched {
        let drain = context.synchronize();
        return Err(error.context(format!("fused NVFP4 A16 SwiGLU failed; drain: {drain:?}")));
    }
    context.synchronize()?;
    Ok(OutputBuffers {
        gate: Output {
            values: gate_output,
            unrounded: gate_raw,
        },
        up: Output {
            values: up_output,
            unrounded: up_raw,
        },
        activation,
    })
}

fn validate(
    context: &Context,
    module: &Module<'_>,
    input: &Buffer<'_>,
    gate: &Projection<'_, '_>,
    up: &Projection<'_, '_>,
) -> Result<()> {
    ensure!(
        input.belongs_to(context)
            && module.belongs_to(context)
            && gate.owner.belongs_to(context)
            && up.owner.belongs_to(context),
        "fused NVFP4 A16 SwiGLU input/module context mismatch"
    );
    ensure!(
        gate.width == up.width
            && gate.channels == up.channels
            && (16..=32768).contains(&gate.width)
            && gate.width.is_multiple_of(16)
            && (1..=32768).contains(&gate.channels),
        "fused NVFP4 A16 SwiGLU weight geometry mismatch"
    );
    ensure!(
        input.len() == checked_bytes(gate.width, 2, "BF16 input")?,
        "fused NVFP4 A16 SwiGLU input extent mismatch"
    );
    ensure!(
        [gate.inverse_weight_divisor, up.inverse_weight_divisor]
            .into_iter()
            .all(|value| value.is_finite() && value > 0.0),
        "fused NVFP4 A16 SwiGLU divisors must be positive and finite"
    );
    Ok(())
}

fn checked_bytes(count: usize, bytes_per_value: usize, name: &str) -> Result<usize> {
    count
        .checked_mul(bytes_per_value)
        .ok_or_else(|| anyhow::anyhow!("{name} extent overflows usize"))
}
