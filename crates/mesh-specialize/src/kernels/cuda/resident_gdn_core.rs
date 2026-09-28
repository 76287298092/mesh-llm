//! Reference-free resident composition of GDN preparation, recurrence, and output normalization.

use super::{
    driver::{Buffer, Context, Module},
    resident_state::ResidentState,
    resident_weights::ResidentWeights,
};
use crate::artifact::schema::DType;
use anyhow::{Context as _, Result, anyhow, ensure};
use std::ffi::c_void;

const MAX_ROWS: usize = 2048;
const MAX_HEADS: usize = 256;
const MAX_WIDTH: usize = 256;
const MAX_QKV_CHANNELS: usize = 32_768;

pub(super) struct Input<'a, 'ctx> {
    pub(super) qkv: &'a Buffer<'ctx>,
    pub(super) a: &'a Buffer<'ctx>,
    pub(super) b: &'a Buffer<'ctx>,
    pub(super) z: &'a Buffer<'ctx>,
    pub(super) record: bool,
}

pub(super) struct GdnCore<'w, 'ctx> {
    owner: &'w ResidentWeights<'ctx>,
    a_log: u64,
    dt_bias: u64,
    f32_params: bool,
    norm_weight: u64,
    key_heads: usize,
    value_heads: usize,
    width: usize,
    qkv_channels: usize,
}

impl<'w, 'ctx> GdnCore<'w, 'ctx> {
    pub(super) fn new(
        owner: &'w ResidentWeights<'ctx>,
        prefix: &str,
        key_heads: usize,
        value_heads: usize,
        width: usize,
    ) -> Result<Self> {
        let qkv_channels = validate_shape(key_heads, value_heads, width)?;
        let norm_shape = [u64::try_from(width)?];
        let norm_bytes = checked_product(width, 2, "GDN norm weight")?;
        let (a_log, dt_bias, f32_params) = bind_parameters(owner, prefix, value_heads)?;
        let norm_weight = owner.tensor(
            &format!("{prefix}.norm.weight"),
            DType::Bf16,
            &norm_shape,
            u64::try_from(norm_bytes)?,
        )?;
        Ok(Self {
            owner,
            a_log,
            dt_bias,
            f32_params,
            norm_weight,
            key_heads,
            value_heads,
            width,
            qkv_channels,
        })
    }

    pub(super) fn run<'a>(
        &self,
        context: &'a Context,
        module: &Module<'_>,
        input: Input<'_, '_>,
        state: &mut ResidentState<'_>,
        state_name: &str,
        rows: usize,
    ) -> Result<super::resident_recovery::CoreOutput<'a>> {
        ensure!(
            !input.record || (rows <= 5 && self.key_heads <= 64 && self.width.is_power_of_two()),
            "GDN recording requires at most five rows, at most 64 key heads and power-of-two width"
        );
        let extents = run_extents(
            rows,
            self.key_heads,
            self.value_heads,
            self.width,
            self.qkv_channels,
        )?;
        validate_contexts(self.owner, context, module, state, &input)?;
        validate_inputs(&input, &extents)?;
        let state = state.pointer(state_name, extents.state_bytes)?;
        let (q, k) = prepare_qk(context, module, input.qkv, self, &extents)?;
        let (beta, decay) = prepare_gates(context, module, input.a, input.b, self, &extents)?;
        let prepared = Prepared { q, k, beta, decay };
        let (recurrent, delta) =
            run_recurrent(context, module, &input, &prepared, state, self, &extents)?;
        let output = run_gated_norm(context, module, &recurrent, input.z, self, &extents)?;
        drop(recurrent);
        let record = delta.map(|delta| super::resident_recovery::Recurrence {
            k: prepared.k,
            decay: prepared.decay,
            delta,
            rows,
            key_heads: self.key_heads,
            value_heads: self.value_heads,
            width: self.width,
        });
        Ok(super::resident_recovery::CoreOutput { output, record })
    }
}

/// Parameters remain in their verified source precision. The container is irrelevant.
pub(super) fn bind_parameters(
    owner: &ResidentWeights<'_>,
    prefix: &str,
    heads: usize,
) -> Result<(u64, u64, bool)> {
    let log_name = format!("{prefix}.A_log");
    let bias_name = format!("{prefix}.dt_bias");
    let log_dtype = &owner.object(&log_name)?.dtype;
    let bias_dtype = &owner.object(&bias_name)?.dtype;
    let f32_params = parameter_precision(log_dtype, bias_dtype, heads)?;
    let bytes = u64::try_from(checked_product(
        heads,
        if f32_params { 4 } else { 2 },
        "GDN parameter",
    )?)?;
    let shape = [u64::try_from(heads)?];
    // tensor verifies row-major layout, exact dtype/shape/bytes and arena membership.
    let a_log = owner.tensor(&log_name, log_dtype.clone(), &shape, bytes)?;
    let dt_bias = owner.tensor(&bias_name, bias_dtype.clone(), &shape, bytes)?;
    Ok((a_log, dt_bias, f32_params))
}

fn parameter_precision(a_log: &DType, dt_bias: &DType, heads: usize) -> Result<bool> {
    ensure!(a_log == dt_bias, "GDN A_log and dt_bias precisions differ");
    match a_log {
        DType::Bf16 => Ok(false),
        DType::F32 => {
            ensure!(heads == 48, "F32 GDN parameters require exact [48] shape");
            Ok(true)
        }
        _ => anyhow::bail!("unsupported GDN parameter dtype: {}", a_log.as_str()),
    }
}

#[derive(Clone, Copy)]
struct RunExtents {
    rows: usize,
    qkv_bytes: usize,
    gate_bytes: usize,
    z_bytes: usize,
    qk_elements: usize,
    gate_elements: usize,
    state_bytes: usize,
    bf16_output_bytes: usize,
    fp32_output_bytes: usize,
}

struct Prepared<'ctx> {
    q: Buffer<'ctx>,
    k: Buffer<'ctx>,
    beta: Buffer<'ctx>,
    decay: Buffer<'ctx>,
}

fn validate_shape(key_heads: usize, value_heads: usize, width: usize) -> Result<usize> {
    ensure!(
        (1..=MAX_HEADS).contains(&key_heads)
            && (1..=MAX_HEADS).contains(&value_heads)
            && value_heads.is_multiple_of(key_heads),
        "invalid GDN key/value head counts"
    );
    ensure!(
        (1..=MAX_WIDTH).contains(&width),
        "GDN width must be in 1..={MAX_WIDTH}; the qualified Q/K and gated-norm kernels use 256-thread reductions"
    );
    let qkv_heads = checked_product(key_heads, 2, "GDN QKV head count")?
        .checked_add(value_heads)
        .context("GDN QKV head count overflows usize")?;
    let qkv_channels = checked_product(qkv_heads, width, "GDN QKV channels")?;
    ensure!(
        qkv_channels <= MAX_QKV_CHANNELS,
        "GDN QKV channels exceed {MAX_QKV_CHANNELS}"
    );
    Ok(qkv_channels)
}

fn run_extents(
    rows: usize,
    key_heads: usize,
    value_heads: usize,
    width: usize,
    qkv_channels: usize,
) -> Result<RunExtents> {
    ensure!(
        (1..=MAX_ROWS).contains(&rows),
        "GDN rows must be in 1..={MAX_ROWS}"
    );
    ensure!(
        validate_shape(key_heads, value_heads, width)? == qkv_channels,
        "GDN QKV channels do not match head dimensions"
    );
    let qkv_elements = checked_product(rows, qkv_channels, "GDN QKV input")?;
    let gate_elements = checked_product(rows, value_heads, "GDN gate input")?;
    let qk_rows = checked_product(rows, key_heads, "GDN Q/K rows")?;
    let qk_elements = checked_product(qk_rows, width, "GDN Q/K output")?;
    let output_rows = checked_product(rows, value_heads, "GDN output rows")?;
    let output_elements = checked_product(output_rows, width, "GDN output")?;
    let state_elements = checked_product(value_heads, width, "GDN state rows")?;
    let state_elements = checked_product(state_elements, width, "GDN state")?;
    Ok(RunExtents {
        rows,
        qkv_bytes: checked_product(qkv_elements, 2, "GDN QKV bytes")?,
        gate_bytes: checked_product(gate_elements, 2, "GDN gate bytes")?,
        z_bytes: checked_product(output_elements, 2, "GDN Z bytes")?,
        qk_elements,
        gate_elements,
        state_bytes: checked_product(state_elements, 4, "GDN state bytes")?,
        bf16_output_bytes: checked_product(output_elements, 2, "GDN BF16 output bytes")?,
        fp32_output_bytes: checked_product(output_elements, 4, "GDN FP32 output bytes")?,
    })
}

fn validate_contexts(
    owner: &ResidentWeights<'_>,
    context: &Context,
    module: &Module<'_>,
    state: &ResidentState<'_>,
    input: &Input<'_, '_>,
) -> Result<()> {
    let pointers = [
        input.qkv.pointer(),
        input.a.pointer(),
        input.b.pointer(),
        input.z.pointer(),
    ];
    ensure!(
        owner.belongs_to(context)
            && state.belongs_to(context)
            && module.belongs_to(context)
            && input.qkv.belongs_to(context)
            && input.a.belongs_to(context)
            && input.b.belongs_to(context)
            && input.z.belongs_to(context),
        "GDN weights, state, inputs, and PTX module must belong to the same CUDA context"
    );
    ensure!(
        pointers
            .iter()
            .enumerate()
            .all(|(index, pointer)| !pointers[..index].contains(pointer)),
        "GDN QKV, A, B, and Z input buffers must be distinct"
    );
    Ok(())
}

fn validate_inputs(input: &Input<'_, '_>, extents: &RunExtents) -> Result<()> {
    validate_input_lengths(
        input.qkv.len(),
        input.a.len(),
        input.b.len(),
        input.z.len(),
        extents,
    )
}

fn validate_input_lengths(
    qkv: usize,
    a: usize,
    b: usize,
    z: usize,
    extents: &RunExtents,
) -> Result<()> {
    validate_length("GDN QKV", qkv, extents.qkv_bytes)?;
    validate_length("GDN A", a, extents.gate_bytes)?;
    validate_length("GDN B", b, extents.gate_bytes)?;
    validate_length("GDN Z", z, extents.z_bytes)
}

fn prepare_qk<'a>(
    context: &'a Context,
    module: &Module<'_>,
    qkv: &Buffer<'_>,
    core: &GdnCore<'_, '_>,
    extents: &RunExtents,
) -> Result<(Buffer<'a>, Buffer<'a>)> {
    let bytes = checked_product(extents.qk_elements, 4, "GDN Q/K bytes")?;
    let q = Buffer::new(context, bytes)?;
    let k = Buffer::new(context, bytes)?;
    let mut pointers = [qkv.pointer(), q.pointer(), k.pointer()];
    let mut dimensions = [
        u32::try_from(extents.rows)?,
        u32::try_from(core.key_heads)?,
        u32::try_from(core.value_heads)?,
        u32::try_from(core.width)?,
    ];
    let blocks = checked_product(extents.rows, core.key_heads, "GDN Q/K grid")?;
    launch_and_sync(
        context,
        module,
        &mut pointers,
        &mut dimensions,
        LaunchConfig {
            name: "gdn_qk_norm",
            grid: [u32::try_from(blocks)?, 1, 1],
            block: [256, 1, 1],
            trailing_f32: None,
            trailing_u64: None,
        },
    )?;
    Ok((q, k))
}

fn prepare_gates<'a>(
    context: &'a Context,
    module: &Module<'_>,
    a: &Buffer<'_>,
    b: &Buffer<'_>,
    core: &GdnCore<'_, '_>,
    extents: &RunExtents,
) -> Result<(Buffer<'a>, Buffer<'a>)> {
    let beta = Buffer::new(
        context,
        checked_product(extents.gate_elements, 2, "GDN beta bytes")?,
    )?;
    let gate_bytes = checked_product(extents.gate_elements, 4, "GDN gate output bytes")?;
    let g = Buffer::new(context, gate_bytes)?;
    let decay = Buffer::new(context, gate_bytes)?;
    let mut pointers = [
        a.pointer(),
        b.pointer(),
        core.a_log,
        core.dt_bias,
        beta.pointer(),
        g.pointer(),
        decay.pointer(),
    ];
    let mut dimensions = [
        u32::try_from(extents.rows)?,
        u32::try_from(core.value_heads)?,
    ];
    launch_and_sync(
        context,
        module,
        &mut pointers,
        &mut dimensions,
        LaunchConfig {
            name: if core.f32_params {
                "gdn_gates_f32_params"
            } else {
                "gdn_gates"
            },
            grid: [u32::try_from(extents.gate_elements.div_ceil(256))?, 1, 1],
            block: [256, 1, 1],
            trailing_f32: None,
            trailing_u64: None,
        },
    )?;
    drop(g);
    Ok((beta, decay))
}

fn run_recurrent<'a>(
    context: &'a Context,
    module: &Module<'_>,
    input: &Input<'_, '_>,
    prepared: &Prepared<'_>,
    state: u64,
    core: &GdnCore<'_, '_>,
    extents: &RunExtents,
) -> Result<(Buffer<'a>, Option<Buffer<'a>>)> {
    let delta = input
        .record
        .then(|| Buffer::new(context, extents.fp32_output_bytes))
        .transpose()?;
    let mut delta_pointer = delta.as_ref().map(Buffer::pointer);
    let output = Buffer::new(context, extents.bf16_output_bytes)?;
    let unrounded = Buffer::new(context, extents.fp32_output_bytes)?;
    let mut pointers = [
        prepared.q.pointer(),
        prepared.k.pointer(),
        input.qkv.pointer(),
        prepared.beta.pointer(),
        prepared.decay.pointer(),
        state,
        output.pointer(),
        unrounded.pointer(),
    ];
    let mut dimensions = [
        u32::try_from(extents.rows)?,
        u32::try_from(core.key_heads)?,
        u32::try_from(core.value_heads)?,
        u32::try_from(core.width)?,
    ];
    launch_and_sync(
        context,
        module,
        &mut pointers,
        &mut dimensions,
        LaunchConfig {
            name: if input.record {
                "gdn_recurrent_record"
            } else {
                "gdn_recurrent"
            },
            grid: [u32::try_from(core.value_heads)?, 1, 1],
            block: [u32::try_from(core.width)?, 1, 1],
            trailing_f32: None,
            trailing_u64: delta_pointer.as_mut(),
        },
    )?;
    drop(unrounded);
    Ok((output, delta))
}

fn run_gated_norm<'a>(
    context: &'a Context,
    module: &Module<'_>,
    recurrent: &Buffer<'_>,
    z: &Buffer<'_>,
    core: &GdnCore<'_, '_>,
    extents: &RunExtents,
) -> Result<Buffer<'a>> {
    let output = Buffer::new(context, extents.bf16_output_bytes)?;
    let normalized = Buffer::new(context, extents.fp32_output_bytes)?;
    let weighted = Buffer::new(context, extents.bf16_output_bytes)?;
    let silu = Buffer::new(context, extents.fp32_output_bytes)?;
    let unrounded = Buffer::new(context, extents.fp32_output_bytes)?;
    let mut pointers = [
        recurrent.pointer(),
        z.pointer(),
        core.norm_weight,
        output.pointer(),
        normalized.pointer(),
        weighted.pointer(),
        silu.pointer(),
        unrounded.pointer(),
    ];
    let groups = checked_product(extents.rows, core.value_heads, "GDN norm groups")?;
    let mut dimensions = [u32::try_from(groups)?, u32::try_from(core.width)?];
    let mut epsilon = 1.0e-6_f32;
    launch_and_sync(
        context,
        module,
        &mut pointers,
        &mut dimensions,
        LaunchConfig {
            name: "gdn_gated_rms_norm",
            grid: [u32::try_from(groups)?, 1, 1],
            block: [256, 1, 1],
            trailing_f32: Some(&mut epsilon),
            trailing_u64: None,
        },
    )?;
    drop((normalized, weighted, silu, unrounded));
    Ok(output)
}

struct LaunchConfig<'a> {
    name: &'a str,
    grid: [u32; 3],
    block: [u32; 3],
    trailing_f32: Option<&'a mut f32>,
    trailing_u64: Option<&'a mut u64>,
}

fn launch_and_sync(
    context: &Context,
    module: &Module<'_>,
    pointers: &mut [u64],
    dimensions: &mut [u32],
    mut config: LaunchConfig<'_>,
) -> Result<()> {
    let function = module.function(config.name)?;
    let mut arguments: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|pointer| (pointer as *mut u64).cast())
        .collect();
    arguments.extend(
        dimensions
            .iter_mut()
            .map(|dimension| (dimension as *mut u32).cast()),
    );
    if let Some(value) = config.trailing_f32.take() {
        arguments.push((value as *mut f32).cast());
    }
    if let Some(value) = config.trailing_u64.take() {
        arguments.push((value as *mut u64).cast());
    }
    // SAFETY: Each stage supplies the exact kernel-specific pointer/dimension order. Extents,
    // buffer contexts, and row/head bounds are validated; every stage synchronizes before any
    // scratch buffer is released or reused.
    if let Err(error) = unsafe { function.launch(config.grid, config.block, 0, &mut arguments) } {
        return Err(synchronize_after_failed_launch(context, config.name, error));
    }
    context
        .synchronize()
        .with_context(|| format!("synchronize {}", config.name))
}

fn synchronize_after_failed_launch(
    context: &Context,
    operation: &str,
    error: anyhow::Error,
) -> anyhow::Error {
    match context.synchronize() {
        Ok(()) => error.context(format!("{operation} launch failed")),
        Err(sync_error) => error.context(format!(
            "{operation} launch failed; CUDA synchronization also failed: {sync_error:#}"
        )),
    }
}

fn validate_length(name: &str, actual: usize, expected: usize) -> Result<()> {
    ensure!(
        actual == expected,
        "{name} has {actual} bytes, expected {expected}"
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
        MAX_HEADS, MAX_QKV_CHANNELS, MAX_ROWS, MAX_WIDTH, checked_product, run_extents,
        validate_input_lengths, validate_length, validate_shape,
    };

    #[test]
    fn gate_precision_defaults_to_bf16_and_requires_matched_f32_48() {
        use crate::artifact::schema::DType;
        assert!(!super::parameter_precision(&DType::Bf16, &DType::Bf16, 48).unwrap());
        assert!(!super::parameter_precision(&DType::Bf16, &DType::Bf16, 2).unwrap());
        assert!(super::parameter_precision(&DType::F32, &DType::F32, 48).unwrap());
        for (a, d) in [
            (DType::Bf16, DType::F32),
            (DType::F32, DType::Bf16),
            (DType::F16, DType::F16),
        ] {
            assert!(super::parameter_precision(&a, &d, 48).is_err());
        }
        for heads in [0, 47, 49, 256] {
            assert!(super::parameter_precision(&DType::F32, &DType::F32, heads).is_err());
        }
    }

    #[test]
    fn validates_supported_shape_and_checked_extents() {
        assert_eq!(validate_shape(1, 1, 1).unwrap(), 3);
        assert_eq!(validate_shape(16, 48, 128).unwrap(), 10_240);
        assert_eq!(validate_shape(32, 64, 256).unwrap(), MAX_QKV_CHANNELS);
        assert!(validate_shape(33, 66, 256).is_err());
        let extents = run_extents(2, 1, 2, 4, 16).unwrap();
        assert_eq!(extents.qkv_bytes, 64);
        assert_eq!(extents.gate_bytes, 8);
        assert_eq!(extents.z_bytes, 32);
        assert_eq!(extents.qk_elements, 8);
        assert_eq!(extents.state_bytes, 128);
        assert_eq!(extents.bf16_output_bytes, 32);
        assert_eq!(extents.fp32_output_bytes, 64);
    }

    #[test]
    fn rejects_invalid_rows_heads_width_and_qkv_extent() {
        assert!(run_extents(0, 1, 1, 1, 3).is_err());
        assert!(run_extents(MAX_ROWS + 1, 1, 1, 1, 3).is_err());
        assert!(validate_shape(0, 1, 1).is_err());
        assert!(validate_shape(1, 0, 1).is_err());
        assert!(validate_shape(2, 3, 1).is_err());
        assert!(validate_shape(MAX_HEADS, MAX_HEADS, MAX_WIDTH + 1).is_err());
        assert!(validate_shape(MAX_HEADS, MAX_HEADS, MAX_WIDTH).is_err());
        assert_eq!(MAX_QKV_CHANNELS, 32_768);
    }

    #[test]
    fn checks_exact_input_lengths_and_extent_overflow() {
        let extents = run_extents(2, 1, 2, 4, 16).unwrap();
        assert!(validate_input_lengths(64, 8, 6, 32, &extents).is_err());
        assert!(validate_input_lengths(64, 8, 8, 32, &extents).is_ok());
        assert!(validate_length("QKV", 10, 8).is_err());
        assert!(validate_length("QKV", 8, 8).is_ok());
        assert!(checked_product(usize::MAX, 2, "test").is_err());
    }
}
