use super::super::driver::{Buffer, Context, Module};
use super::types::Weight;
use anyhow::{Result, ensure};
use std::ffi::c_void;

pub(super) struct Outputs<'ctx> {
    pub(super) gate: Buffer<'ctx>,
    pub(super) gate_raw: Buffer<'ctx>,
    pub(super) up: Buffer<'ctx>,
    pub(super) up_raw: Buffer<'ctx>,
    pub(super) activation_raw: Buffer<'ctx>,
    pub(super) activation: Buffer<'ctx>,
}

pub(super) struct Request<'a, 'ctx> {
    pub(super) context: &'ctx Context,
    pub(super) module: &'a Module<'ctx>,
    pub(super) input: &'a Buffer<'ctx>,
    pub(super) weights: [&'a Weight<'a>; 2],
    pub(super) outputs: &'a Outputs<'ctx>,
    pub(super) dimensions: [usize; 2],
}

impl<'ctx> Outputs<'ctx> {
    pub(super) fn new(context: &'ctx Context, channels: usize) -> Result<Self> {
        let outputs = Self {
            gate: Buffer::new(context, channels * 2)?,
            gate_raw: Buffer::new(context, channels * 4)?,
            up: Buffer::new(context, channels * 2)?,
            up_raw: Buffer::new(context, channels * 4)?,
            activation_raw: Buffer::new(context, channels * 4)?,
            activation: Buffer::new(context, channels * 2)?,
        };
        outputs.gate.upload(&vec![0xff; channels * 2])?;
        outputs.gate_raw.upload(&vec![0xff; channels * 4])?;
        outputs.up.upload(&vec![0xff; channels * 2])?;
        outputs.up_raw.upload(&vec![0xff; channels * 4])?;
        outputs.activation_raw.upload(&vec![0xff; channels * 4])?;
        outputs.activation.upload(&vec![0xff; channels * 2])?;
        Ok(outputs)
    }
}

pub(super) fn run(request: Request<'_, '_>) -> Result<()> {
    let Request {
        context,
        module,
        input,
        weights,
        outputs,
        dimensions: [channels, width],
    } = request;
    ensure!(
        input.belongs_to(context) && module.belongs_to(context),
        "fused NVFP4 A16 input/module context mismatch"
    );
    let mut pointers = [
        input.pointer(),
        weights[0].address[0],
        weights[0].address[1],
        weights[1].address[0],
        weights[1].address[1],
        outputs.gate.pointer(),
        outputs.gate_raw.pointer(),
        outputs.up.pointer(),
        outputs.up_raw.pointer(),
        outputs.activation_raw.pointer(),
        outputs.activation.pointer(),
    ];
    let mut dimensions = [u32::try_from(channels)?, u32::try_from(width)?];
    let mut divisors = [1.0 / weights[0].divisor, 1.0 / weights[1].divisor];
    ensure!(
        divisors
            .iter()
            .all(|value| value.is_finite() && *value > 0.0),
        "invalid inverse NVFP4 weight divisor"
    );
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
        // SAFETY: The eleven distinct device allocations and four scalar values match the kernel ABI.
        module.function("nvfp4_swiglu_a16")?.launch(
            [u32::try_from(channels.div_ceil(8))?, 1, 1],
            [256, 1, 1],
            0,
            &mut arguments,
        )
    };
    if let Err(error) = launched {
        let drain = context.synchronize();
        return Err(error.context(format!("fused NVFP4 A16 launch failed; drain: {drain:?}")));
    }
    context.synchronize()
}
