//! Same-input real-weight native/exact projection diagnostics. Never supplies model inputs.
use super::{
    driver::{Buffer, Context, Module},
    resident_weights::ResidentWeights,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{cell::RefCell, ffi::c_void};
thread_local! { static REPORTS:RefCell<Vec<Value>>=const {RefCell::new(Vec::new())}; }
pub(super) fn take_reports() -> Vec<Value> {
    REPORTS.with(|r| std::mem::take(&mut *r.borrow_mut()))
}
pub(super) struct Case<'a, 'ctx> {
    pub owner: &'a ResidentWeights<'ctx>,
    pub weight_name: &'a str,
    pub scale_name: &'a str,
    pub codes: &'a Buffer<'ctx>,
    pub row_scales: &'a Buffer<'ctx>,
    pub output: &'a Buffer<'ctx>,
    pub raw: &'a Buffer<'ctx>,
    pub shape: [usize; 3],
}
pub(super) fn compare(ctx: &Context, module: &Module<'_>, case: Case<'_, '_>) -> Result<()> {
    if !case.weight_name.contains(".layers.0.")
        || REPORTS.with(|r| r.borrow().iter().any(|v| v["weight"] == case.weight_name))
    {
        return Ok(());
    }
    let [m, n, k] = case.shape;
    let exact = Buffer::new(ctx, case.output.len())?;
    let exact_raw = Buffer::new(ctx, case.raw.len())?;
    let mut pointers = [
        case.codes.pointer(),
        case.owner.pointer(case.weight_name)?,
        case.row_scales.pointer(),
        case.owner.pointer(case.scale_name)?,
        exact.pointer(),
        exact_raw.pointer(),
    ];
    let mut dims = [u32::try_from(m)?, u32::try_from(n)?, u32::try_from(k)?];
    let mut args = pointers
        .iter_mut()
        .map(|v| (v as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(dims.iter_mut().map(|v| (v as *mut u32).cast()));
    // SAFETY: The projection caller already validated all tensor extents and context ownership.
    // These exact-control buffers match the candidate extents and remain live until synchronization.
    unsafe {
        module.function("fp8_prefill_exact")?.launch(
            [
                u32::try_from(n.div_ceil(8))?,
                u32::try_from(m.div_ceil(16))?,
                1,
            ],
            [32, 1, 1],
            0,
            &mut args,
        )?;
    }
    ctx.synchronize()?;
    let actual = words(case.output)?;
    let expected = words(&exact)?;
    let raw = floats(case.raw)?;
    let reference = floats(&exact_raw)?;
    ensure!(
        actual.len() == expected.len() && raw.len() == reference.len(),
        "audit extents disagree"
    );
    let mut worst = (0, 0.0_f64);
    let mut squared = 0.0_f64;
    let mut norm = 0.0_f64;
    for (i, (&a, &b)) in raw.iter().zip(&reference).enumerate() {
        let error = (f64::from(a) - f64::from(b)).abs();
        if error > worst.1 {
            worst = (i, error);
        }
        squared += error * error;
        norm += f64::from(b).powi(2);
    }
    let differences = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
    let mut indices = actual
        .iter()
        .zip(&expected)
        .enumerate()
        .filter_map(|(i, (a, b))| (a != b).then_some(i))
        .take(6)
        .collect::<Vec<_>>();
    if !indices.contains(&worst.0) {
        indices.push(worst.0);
    }
    let mut samples = Vec::new();
    for index in indices {
        samples.push(sample(&case, index, &raw, &reference, &actual, &expected)?);
    }
    REPORTS.with(|r|r.borrow_mut().push(json!({"weight":case.weight_name,"shape":case.shape,
        "bf16_differences":differences,"outputs":actual.len(),"raw_normalized_l2":(squared/norm.max(1e-30)).sqrt(),
        "raw_max_abs_error":worst.1,"finite":raw.iter().all(|v|v.is_finite()),"samples":samples,
        "diagnostic_only":true,"timing_claim":false})));
    Ok(())
}
fn words(buffer: &Buffer<'_>) -> Result<Vec<u16>> {
    let mut bytes = vec![0; buffer.len()];
    buffer.download(&mut bytes)?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| u16::from_le_bytes(*v))
        .collect())
}
fn floats(buffer: &Buffer<'_>) -> Result<Vec<f32>> {
    let mut bytes = vec![0; buffer.len()];
    buffer.download(&mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect())
}
fn sample(
    case: &Case<'_, '_>,
    index: usize,
    raw: &[f32],
    reference: &[f32],
    actual: &[u16],
    expected: &[u16],
) -> Result<Value> {
    let [_, n, k] = case.shape;
    let row = index / n;
    let channel = index % n;
    let mut a = vec![0; k];
    let mut w = vec![0; k];
    let mut sa = [0; 4];
    let mut sw = [0; 2];
    case.codes.download_at(row * k, &mut a)?;
    case.owner
        .read_range(case.weight_name, channel * k, &mut w)?;
    case.row_scales.download_at(row * 4, &mut sa)?;
    case.owner
        .read_range(case.scale_name, channel * 2, &mut sw)?;
    let oracle = crate::fp8_native_prefill_reference::fp8_native_prefill_reference(
        &a,
        &w,
        &[f32::from_le_bytes(sa)],
        &[u16::from_le_bytes(sw)],
        1,
        1,
        k,
    )?;
    Ok(
        json!({"row":row,"channel":channel,"native_raw":raw[index],"exact_raw":reference[index],
        "cpu_raw":oracle.unrounded[0],"native_bf16":actual[index],"exact_bf16":expected[index],"cpu_bf16":oracle.bf16[0]}),
    )
}
