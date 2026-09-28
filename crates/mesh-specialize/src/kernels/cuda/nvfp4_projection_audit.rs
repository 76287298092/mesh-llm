//! Same-quantized-input audit for the localized layer22 down-projection mismatch.
use super::{
    driver::{Buffer, Context, Module},
    resident_weights::ResidentWeights,
};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, ffi::c_void};
const ROW: usize = 101;
const PREFIX: &str = "tensors/model.language_model.layers.22.mlp.down_proj";
#[derive(Default)]
struct State {
    report: Option<Value>,
    token_rows: usize,
    codes: Vec<u8>,
    scales: Vec<u8>,
    exact: Vec<u8>,
    raw: Vec<u8>,
}
thread_local! { static STATE: RefCell<State> = RefCell::new(State::default()); }
pub(super) fn enabled() -> Result<bool> {
    match std::env::var("MESH_SPECIALIZE_NVFP4_AUDIT") {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(v) if v == "1" => Ok(true),
        _ => anyhow::bail!("MESH_SPECIALIZE_NVFP4_AUDIT must be absent or1"),
    }
}
pub(super) fn take() -> Option<Value> {
    STATE.with_borrow_mut(|s| std::mem::take(s).report)
}
pub(super) struct Case<'a, 'ctx> {
    pub owner: &'a ResidentWeights<'ctx>,
    pub prefix: &'a str,
    pub codes: &'a Buffer<'ctx>,
    pub scales: &'a Buffer<'ctx>,
    pub output: &'a Buffer<'ctx>,
    pub raw: &'a Buffer<'ctx>,
    pub shape: [usize; 3],
    pub input_global: f32,
    pub factor: f32,
}
pub(super) fn compare(ctx: &Context, module: &Module<'_>, c: Case<'_, '_>) -> Result<()> {
    if c.prefix != PREFIX || !enabled()? || !super::partition_stage_audit::selected(22) {
        return Ok(());
    }
    let [m, n, k] = c.shape;
    ensure!(
        n == 5120 && k == 17408,
        "unexpected localized projection geometry"
    );
    if m == 1 {
        return STATE.with_borrow_mut(|state| -> Result<()> {
            let row = state.token_rows;
            state.token_rows += 1;
            if row != ROW {
                return Ok(());
            }
            let report = state
                .report
                .as_mut()
                .context("token audit preceded batch audit")?;
            let codes = read(c.codes, 0, k / 2)?;
            let scales = read(c.scales, 0, k / 16)?;
            let output = read(c.output, 0, n * 2)?;
            let raw = read(c.raw, 0, n * 4)?;
            let inputs_equal = codes == state.codes && scales == state.scales;
            let outputs_equal = output == state.exact && raw == state.raw;
            report["token_quantized_inputs_equal"] = json!(inputs_equal);
            report["token_integer_outputs_equal"] = json!(outputs_equal);
            report["token_row"] = json!(row);
            Ok(())
        });
    }
    ensure!(m == 128, "NVFP4 audit requires128-token prefix");
    ensure!(
        STATE.with_borrow(|s| s.report.is_none()),
        "duplicate batch NVFP4 audit"
    );
    let codes = read(c.codes, ROW * k / 2, k / 2)?;
    let scales = read(c.scales, ROW * k / 16, k / 16)?;
    let native = read(c.output, ROW * n * 2, n * 2)?;
    let native_raw = read(c.raw, ROW * n * 4, n * 4)?;
    let exact = Buffer::new(ctx, n * 2)?;
    let exact_raw = Buffer::new(ctx, n * 4)?;
    launch_exact(ctx, module, &c, &exact, &exact_raw)?;
    let exact_bytes = read(&exact, 0, n * 2)?;
    let exact_raw_bytes = read(&exact_raw, 0, n * 4)?;
    let a = floats(&native_raw);
    let b = floats(&exact_raw_bytes);
    ensure!(
        a.iter().chain(&b).all(|x| x.is_finite()),
        "nonfinite audit output"
    );
    let differing = native
        .chunks_exact(2)
        .zip(exact_bytes.chunks_exact(2))
        .enumerate()
        .filter_map(|(i, (a, b))| (a != b).then_some(i))
        .collect::<Vec<_>>();
    let mut selected = differing.iter().copied().take(32).collect::<Vec<_>>();
    selected.extend([0, n / 2, n - 1]);
    let worst = (0..n)
        .max_by(|&i, &j| (a[i] - b[i]).abs().total_cmp(&(a[j] - b[j]).abs()))
        .context("empty audit")?;
    selected.push(worst);
    selected.sort_unstable();
    selected.dedup();
    let samples = oracle_samples(
        &c,
        &codes,
        &scales,
        &selected,
        OutputRows {
            native: &a,
            exact: &b,
            native_bf16: &native,
            exact_bf16: &exact_bytes,
        },
    )?;
    let oracle_exact = samples.iter().all(|s| s["integer_cpu_exact"] == true);
    let numerator: f64 = a
        .iter()
        .zip(&b)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    let denominator: f64 = b.iter().map(|&b| f64::from(b).powi(2)).sum();
    let rounded = a
        .iter()
        .zip(native.chunks_exact(2))
        .chain(b.iter().zip(exact_bytes.chunks_exact(2)))
        .all(|(&x, bytes)| crate::entry_reference::round_bf16(x).to_le_bytes() == bytes);
    ensure!(
        rounded,
        "audit BF16 output disagrees with stored raw rounding"
    );
    let report = json!({"diagnostic_only":true,"weight":PREFIX,"shape":[m,n,k],"row":ROW,"input_codes_sha256":hex::encode(Sha256::digest(&codes)),"input_scales_sha256":hex::encode(Sha256::digest(&scales)),"bf16_differing_channels":differing,"raw_relative_l2":(denominator>0.0).then(|| (numerator/denominator).sqrt()),"max_absolute_error":(a[worst]-b[worst]).abs(),"stored_rounding_exact":rounded,"independent_cpu_samples":samples,"integer_cpu_samples_exact":oracle_exact,"token_quantized_inputs_equal":false,"token_integer_outputs_equal":false,"scope":"native batch output versus integer GPU on identical quantized row; sampled logical FP64 CPU oracle; outputs never feed model"});
    STATE.with_borrow_mut(|state| {
        state.report = Some(report);
        state.codes = codes;
        state.scales = scales;
        state.exact = exact_bytes;
        state.raw = exact_raw_bytes;
    });
    Ok(())
}
fn read(buffer: &Buffer<'_>, offset: usize, length: usize) -> Result<Vec<u8>> {
    let mut bytes = vec![0; length];
    buffer.download_at(offset, &mut bytes)?;
    Ok(bytes)
}
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}
fn launch_exact(
    ctx: &Context,
    module: &Module<'_>,
    c: &Case<'_, '_>,
    out: &Buffer<'_>,
    raw: &Buffer<'_>,
) -> Result<()> {
    let [_, n, k] = c.shape;
    let mut pointers = [
        c.codes.pointer() + (ROW * k / 2) as u64,
        c.owner.pointer(&format!("{}.weight_packed", c.prefix))?,
        c.scales.pointer() + (ROW * k / 16) as u64,
        c.owner.pointer(&format!("{}.weight_scale", c.prefix))?,
        out.pointer(),
        raw.pointer(),
    ];
    let mut dims = [1u32, n as u32, k as u32];
    let mut factor = c.factor;
    let mut args = pointers
        .iter_mut()
        .map(|x| (x as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(dims.iter_mut().map(|x| (x as *mut u32).cast()));
    args.push((&mut factor as *mut f32).cast());
    // SAFETY: Resident caller validated128x5120x17408 buffers; row101 offsets are
    // inside packed/scaled allocations, aligned for the existing integer kernel.
    // Separate result buffers, weights and quantized inputs stay live through drain.
    let result = unsafe {
        module.function("nvfp4_decode_exact")?.launch(
            [n.div_ceil(4) as u32, 1, 1],
            [128, 1, 1],
            0,
            &mut args,
        )
    };
    let drained = ctx.synchronize();
    result?;
    drained
}
struct OutputRows<'a> {
    native: &'a [f32],
    exact: &'a [f32],
    native_bf16: &'a [u8],
    exact_bf16: &'a [u8],
}
fn oracle_samples(
    c: &Case<'_, '_>,
    codes: &[u8],
    scales: &[u8],
    channels: &[usize],
    outputs: OutputRows<'_>,
) -> Result<Vec<Value>> {
    let OutputRows {
        native,
        exact,
        native_bf16,
        exact_bf16,
    } = outputs;
    let k = c.shape[2];
    let mut packed = vec![0; channels.len() * k / 2];
    let mut weight_scales = vec![0; channels.len() * k / 16];
    for (i, &channel) in channels.iter().enumerate() {
        c.owner.read_range(
            &format!("{}.weight_packed", c.prefix),
            channel * k / 2,
            &mut packed[i * k / 2..(i + 1) * k / 2],
        )?;
        c.owner.read_range(
            &format!("{}.weight_scale", c.prefix),
            channel * k / 16,
            &mut weight_scales[i * k / 16..(i + 1) * k / 16],
        )?;
    }
    let reference = crate::nvfp4_linear_reference::run(
        crate::nvfp4_linear_reference::Matrix {
            packed: codes,
            scales,
            rows: 1,
            global: c.input_global,
        },
        crate::nvfp4_linear_reference::Matrix {
            packed: &packed,
            scales: &weight_scales,
            rows: channels.len(),
            global: c
                .owner
                .positive_scalar(&format!("{}.weight_global_scale", c.prefix))?,
        },
        k,
    )?;
    let mut samples = Vec::new();
    for (i, &channel) in channels.iter().enumerate() {
        let exact_bits = u16::from_le_bytes([exact_bf16[channel * 2], exact_bf16[channel * 2 + 1]]);
        let native_bits =
            u16::from_le_bytes([native_bf16[channel * 2], native_bf16[channel * 2 + 1]]);
        let agrees = exact[channel].to_bits() == reference.unrounded[i].to_bits()
            && exact_bits == reference.normalized[i];
        samples.push(json!({"channel":channel,"native_raw":native[channel],"integer_raw":exact[channel],"cpu_raw":reference.unrounded[i],"native_bf16_bits":native_bits,"integer_bf16_bits":exact_bits,"cpu_bf16_bits":reference.normalized[i],"integer_cpu_exact":agrees}));
    }
    Ok(samples)
}
