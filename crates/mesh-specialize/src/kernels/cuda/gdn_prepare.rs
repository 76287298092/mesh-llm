//! Resident GDN Q/K normalization and beta/log-decay gate qualification.
use super::driver::{Buffer, Context, Function, Module};
use crate::{
    entry_reference::{bf16_to_f32, round_bf16},
    gdn_prepare_reference as reference,
    kernels::{GdnWeights, ProjectionInput},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(super) struct Input<'a, 'ctx> {
    pub(super) qkv: &'a Buffer<'ctx>,
    pub(super) qkv_words: &'a [u16],
    pub(super) a: &'a Buffer<'ctx>,
    pub(super) a_words: &'a [u16],
    pub(super) b: &'a Buffer<'ctx>,
    pub(super) b_words: &'a [u16],
}

pub(super) fn validate_connections(input: &ProjectionInput, weights: &GdnWeights) -> Result<()> {
    ensure!(
        (1..=64).contains(&weights.key_heads)
            && (1..=256).contains(&weights.value_heads)
            && weights.value_heads.is_multiple_of(weights.key_heads)
            && (1..=256).contains(&weights.width)
            && weights.width.is_power_of_two(),
        "invalid GDN shape"
    );
    ensure!(
        weights.a_projection != weights.b_projection,
        "GDN A/B must be distinct projections"
    );
    for index in [weights.a_projection, weights.b_projection] {
        ensure!(
            input
                .bf16_projections
                .get(index)
                .is_some_and(|p| p.channels == weights.value_heads),
            "GDN gate projection shape mismatch"
        );
    }
    let convolution = input
        .convolution
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("GDN needs convolution"))?;
    ensure!(
        input.projections[convolution.projection].channels
            == (2 * weights.key_heads + weights.value_heads) * weights.width,
        "GDN QKV projection shape mismatch"
    );
    ensure!(
        weights.a_log.len() == weights.value_heads * 2
            && weights.dt_bias.len() == weights.value_heads * 2,
        "GDN parameter extent mismatch"
    );
    ensure!(
        decode(&weights.a_log).iter().all(|&x| {
            let x = bf16_to_f32(x);
            x.is_finite() && (-80.0..=80.0).contains(&x)
        }),
        "GDN A_log outside qualified range"
    );
    ensure!(
        decode(&weights.dt_bias)
            .iter()
            .all(|&x| bf16_to_f32(x).is_finite()),
        "nonfinite GDN bias"
    );
    Ok(())
}

pub(super) fn check(
    context: &Context,
    module: &Module<'_>,
    input: Input<'_, '_>,
    weights: &GdnWeights,
    rows: usize,
) -> Result<Value> {
    let shape = reference::Shape {
        rows,
        key_heads: weights.key_heads,
        value_heads: weights.value_heads,
        width: weights.width,
    };
    let expected = reference::run(
        input.qkv_words,
        input.a_words,
        input.b_words,
        &decode(&weights.a_log),
        &decode(&weights.dt_bias),
        &shape,
    )?;
    let q = upload(context, &vec![0xff; expected.q.len() * 4])?;
    let k = upload(context, &vec![0xff; expected.k.len() * 4])?;
    let beta = upload(context, &vec![0xff; expected.beta.len() * 2])?;
    let g = upload(context, &vec![0xff; expected.g.len() * 4])?;
    let decay = upload(context, &vec![0xff; expected.decay.len() * 4])?;
    let a_log = upload(context, &weights.a_log)?;
    let bias = upload(context, &weights.dt_bias)?;
    let mut norm_ptrs = [input.qkv.pointer(), q.pointer(), k.pointer()];
    let mut norm_dims = [
        u32::try_from(rows)?,
        u32::try_from(shape.key_heads)?,
        u32::try_from(shape.value_heads)?,
        u32::try_from(shape.width)?,
    ];
    launch(
        &module.function("gdn_qk_norm")?,
        &mut norm_ptrs,
        &mut norm_dims,
        u32::try_from(rows * shape.key_heads)?,
    )?;
    let mut gate_ptrs = [
        input.a.pointer(),
        input.b.pointer(),
        a_log.pointer(),
        bias.pointer(),
        beta.pointer(),
        g.pointer(),
        decay.pointer(),
    ];
    let mut gate_dims = [u32::try_from(rows)?, u32::try_from(shape.value_heads)?];
    launch(
        &module.function("gdn_gates")?,
        &mut gate_ptrs,
        &mut gate_dims,
        u32::try_from(expected.beta.len().div_ceil(256))?,
    )?;
    context.synchronize()?;
    let actual_q = floats(&q, expected.q.len())?;
    let actual_k = floats(&k, expected.k.len())?;
    let actual_g = floats(&g, expected.g.len())?;
    let actual_decay = floats(&decay, expected.decay.len())?;
    let actual_beta = words(&beta, expected.beta.len())?;
    ensure!(
        actual_g.iter().all(|&x| x <= 0.0) && actual_decay.iter().all(|x| (0.0..=1.0).contains(x)),
        "GDN gate output outside domain"
    );
    let q_report = compare(&actual_q, &expected.q, 2e-6, 2e-6)?;
    let k_report = compare(&actual_k, &expected.k, 2e-6, 2e-6)?;
    let g_report = compare(&actual_g, &expected.g, 3e-6, 5e-6)?;
    let decay_report = compare(&actual_decay, &expected.decay, 3e-6, 5e-6)?;
    let mut beta_differences = 0;
    let mut beta_max_ulp = 0;
    for (&actual, &expected) in actual_beta.iter().zip(&expected.beta) {
        ensure!(
            (0.0..=1.0).contains(&bf16_to_f32(actual)),
            "invalid beta output"
        );
        beta_differences += usize::from(actual != expected);
        beta_max_ulp = beta_max_ulp.max(actual.abs_diff(expected));
    }
    ensure!(beta_max_ulp <= 1, "GDN sigmoid exceeds one BF16 ULP");
    ensure!(
        [&q_report, &k_report, &g_report, &decay_report]
            .iter()
            .all(|r| r["passed"] == true),
        "GDN preparation numerical mismatch: q={q_report}, k={k_report}, g={g_report}, decay={decay_report}"
    );
    Ok(
        json!({"all_passed":true,"shape":{"rows":rows,"key_heads":shape.key_heads,"value_heads":shape.value_heads,"width":shape.width},
        "q":q_report,"k":k_report,"g":g_report,"decay":decay_report,
        "beta":{"elements":actual_beta.len(),"reference_differences":beta_differences,"maximum_bf16_ulp":beta_max_ulp,"allowed_bf16_ulp":1},
        "qk_head_mapping":"unrepeated key heads; value head maps to key head via integer division by value/key ratio",
        "device_inputs_resident":true,"model_executable":false}),
    )
}

fn launch(
    function: &Function<'_, '_>,
    pointers: &mut [u64],
    dimensions: &mut [u32],
    grid: u32,
) -> Result<()> {
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|x| (x as *mut u64).cast())
        .collect();
    args.extend(dimensions.iter_mut().map(|x| (x as *mut u32).cast()));
    // SAFETY: The two call sites supply exactly the three-pointer/four-u32 norm
    // ABI or seven-pointer/two-u32 gate ABI. Scalar reference validation covers
    // every extent/domain; retained, disjoint allocations outlive synchronization.
    unsafe { function.launch([grid, 1, 1], [256, 1, 1], 0, &mut args) }
}

fn compare(actual: &[f32], expected: &[f32], absolute: f32, relative: f32) -> Result<Value> {
    ensure!(actual.len() == expected.len(), "GDN output extent mismatch");
    let mut maximum = 0.0_f32;
    let mut mismatches = 0;
    for (&actual, &expected) in actual.iter().zip(expected) {
        ensure!(
            actual.is_finite() && expected.is_finite(),
            "nonfinite GDN output"
        );
        let error = (actual - expected).abs();
        maximum = maximum.max(error);
        mismatches += usize::from(error > absolute + relative * expected.abs());
    }
    Ok(
        json!({"passed":mismatches==0,"elements":actual.len(),"max_abs_error":maximum,
        "mismatches":mismatches,"absolute_tolerance":absolute,"relative_tolerance":relative}),
    )
}

pub(super) fn fixtures(context: &Context, module: &Module<'_>) -> Result<Vec<Value>> {
    let shape = reference::Shape {
        rows: 7,
        key_heads: 2,
        value_heads: 6,
        width: 8,
    };
    let channels = (2 * shape.key_heads + shape.value_heads) * shape.width;
    let mut qkv: Vec<_> = (0..shape.rows * channels)
        .map(|i| round_bf16(((i * 11 % 31) as f32 - 15.0) / 8.0))
        .collect();
    qkv[..shape.width].fill(0);
    let a: Vec<_> = (0..shape.rows * shape.value_heads)
        .map(|i| round_bf16((i * 7 % 19) as f32 - 9.0))
        .collect();
    let b: Vec<_> = (0..shape.rows * shape.value_heads)
        .map(|i| round_bf16(((i * 5 % 23) as f32 - 11.0) * 2.0))
        .collect();
    let weights = GdnWeights {
        a_projection: 0,
        b_projection: 1,
        key_heads: 2,
        value_heads: 6,
        width: 8,
        a_log: (0..6)
            .flat_map(|i| round_bf16((i as f32 - 3.0) / 4.0).to_le_bytes())
            .collect(),
        dt_bias: (0..6)
            .flat_map(|i| round_bf16((i as f32 - 2.0) / 2.0).to_le_bytes())
            .collect(),
    };
    let signed = fixture_run(context, module, &qkv, &a, &b, &weights, 7)?;
    let a = [
        -1e30, -100.0, -88.0, -20.0, 0.0, 20.0, 20.125, 100.0, 1e30, 0.0,
    ]
    .map(round_bf16);
    let b = [
        1e30, 100.0, -88.0, -90.0, -92.0, -93.0, -100.0, 0.0, -1e30, 0.0,
    ]
    .map(round_bf16);
    let weights = GdnWeights {
        a_projection: 0,
        b_projection: 1,
        key_heads: 1,
        value_heads: 10,
        width: 1,
        a_log: (0..10)
            .flat_map(|i| {
                round_bf16(match i {
                    1 => -80.0,
                    9 => 80.0,
                    _ => 0.0,
                })
                .to_le_bytes()
            })
            .collect(),
        dt_bias: vec![0; 20],
    };
    let extremes = fixture_run(context, module, &[0; 12], &a, &b, &weights, 1)?;
    Ok(vec![signed, extremes])
}
fn fixture_run(
    context: &Context,
    module: &Module<'_>,
    qkv: &[u16],
    a: &[u16],
    b: &[u16],
    weights: &GdnWeights,
    rows: usize,
) -> Result<Value> {
    let bytes = |values: &[u16]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let qkv_device = upload(context, &bytes(qkv))?;
    let a_device = upload(context, &bytes(a))?;
    let b_device = upload(context, &bytes(b))?;
    check(
        context,
        module,
        Input {
            qkv: &qkv_device,
            qkv_words: qkv,
            a: &a_device,
            a_words: a,
            b: &b_device,
            b_words: b,
        },
        weights,
        rows,
    )
}
fn decode(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect()
}
fn upload<'a>(context: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}
fn words(buffer: &Buffer<'_>, count: usize) -> Result<Vec<u16>> {
    let mut bytes = vec![0; count * 2];
    buffer.download(&mut bytes)?;
    Ok(decode(&bytes))
}
fn floats(buffer: &Buffer<'_>, count: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0; count * 4];
    buffer.download(&mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compare_rejects_nonfinite_and_bad_values() {
        assert_eq!(compare(&[1.0], &[1.0], 2e-6, 2e-6).unwrap()["passed"], true);
        assert_eq!(
            compare(&[1.1], &[1.0], 2e-6, 2e-6).unwrap()["passed"],
            false
        );
        assert!(compare(&[f32::NAN], &[1.0], 2e-6, 2e-6).is_err());
    }
}
