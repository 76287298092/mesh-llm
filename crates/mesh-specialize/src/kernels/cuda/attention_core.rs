//! Resident causal attention with append-only KV state and partition checks.
use super::driver::{Buffer, Context, Module};
use crate::{
    causal_attention_reference as reference,
    entry_reference::{bf16_to_f32, round_bf16},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(super) struct Input<'a, 'ctx> {
    pub(super) q: &'a Buffer<'ctx>,
    pub(super) k: &'a Buffer<'ctx>,
    pub(super) v: &'a Buffer<'ctx>,
    pub(super) q_words: &'a [u16],
    pub(super) k_words: &'a [u16],
    pub(super) v_words: &'a [u16],
    pub(super) rows: usize,
    pub(super) query_heads: usize,
    pub(super) kv_heads: usize,
    pub(super) width: usize,
}

pub(super) fn check(context: &Context, module: &Module<'_>, input: Input<'_, '_>) -> Result<Value> {
    validate(&input)?;
    let mut partitions = vec![vec![input.rows]];
    if input.rows > 1 {
        partitions.push(vec![1, input.rows - 1]);
    }
    if input.rows > 3 {
        partitions.push(vec![2, 1, input.rows - 3]);
    }
    if input.rows > 1 {
        partitions.push(vec![1; input.rows]);
    }
    let whole = sequence(context, module, &input, &partitions[0])?;
    let mut reports = vec![whole.report];
    for partition in &partitions[1..] {
        let chunked = sequence(context, module, &input, partition)?;
        let words_exact = chunked.output == whole.output;
        let fp32_exact = chunked
            .unrounded
            .iter()
            .zip(&whole.unrounded)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        let state_exact = chunked.k == whole.k && chunked.v == whole.v;
        let mut report = chunked.report;
        report["partition_bf16_exact"] = json!(words_exact);
        report["partition_fp32_exact"] = json!(fp32_exact);
        report["partition_cache_exact"] = json!(state_exact);
        report["all_passed"] =
            json!(report["all_passed"] == true && words_exact && fp32_exact && state_exact);
        reports.push(report);
    }
    Ok(
        json!({"all_passed":reports.iter().all(|r|r["all_passed"]==true),"partitions":reports,
        "rows":input.rows,"query_heads":input.query_heads,"kv_heads":input.kv_heads,"width":input.width,"cache_capacity":input.rows+3,
        "cache_format":"BF16 token-major K/V; NaN poison after initialized prefix","device_inputs_resident":true,
        "arithmetic_profile":"FP32 QK and stable online softmax/value accumulation; BF16 output; independent f64 oracle",
        "scope":"causal attention core and persistent KV state; output gate/projection and full layer not executed"}),
    )
}

fn validate(input: &Input<'_, '_>) -> Result<()> {
    ensure!(
        (1..=2048).contains(&input.rows)
            && (1..=128).contains(&input.query_heads)
            && (1..=128).contains(&input.kv_heads)
            && (2..=256).contains(&input.width),
        "attention core shape invalid"
    );
    ensure!(
        input.query_heads.is_multiple_of(input.kv_heads),
        "attention GQA ratio invalid"
    );
    ensure!(
        input.q_words.len() == input.rows * input.query_heads * input.width
            && input.k_words.len() == input.rows * input.kv_heads * input.width
            && input.v_words.len() == input.k_words.len(),
        "attention core input extent invalid"
    );
    ensure!(
        input
            .q_words
            .iter()
            .chain(input.k_words)
            .chain(input.v_words)
            .all(|&v| bf16_to_f32(v).is_finite()),
        "attention core input nonfinite"
    );
    Ok(())
}

struct Sequence {
    output: Vec<u16>,
    unrounded: Vec<f32>,
    k: Vec<u16>,
    v: Vec<u16>,
    report: Value,
}

fn shape(input: &Input<'_, '_>, past: usize, rows: usize) -> reference::Shape {
    reference::Shape {
        rows,
        query_heads: input.query_heads,
        kv_heads: input.kv_heads,
        width: input.width,
        past,
        capacity: input.rows + 3,
        scale: 1.0 / (input.width as f32).sqrt(),
    }
}

fn sequence(
    context: &Context,
    module: &Module<'_>,
    input: &Input<'_, '_>,
    partition: &[usize],
) -> Result<Sequence> {
    ensure!(
        partition.iter().all(|&n| n > 0) && partition.iter().sum::<usize>() == input.rows,
        "invalid attention partition"
    );
    let kv_stride = input.kv_heads * input.width;
    let q_stride = input.query_heads * input.width;
    let mut expected_k = vec![0x7fc1; (input.rows + 3) * kv_stride];
    let mut expected_v = expected_k.clone();
    let k = upload_words(context, &expected_k)?;
    let v = upload_words(context, &expected_v)?;
    let output = upload(context, &vec![0xa5; input.rows * q_stride * 2])?;
    let unrounded = upload(context, &vec![0xff; input.rows * q_stride * 4])?;
    let mut expected_output = Vec::new();
    let mut expected_fp32 = Vec::new();
    let mut value_bounds = Vec::new();
    let mut boundary_reports = Vec::new();
    let mut offset = 0;
    for &rows in partition {
        let current = shape(input, offset, rows);
        let kv_range = offset * kv_stride..(offset + rows) * kv_stride;
        reference::append(
            &input.k_words[kv_range.clone()],
            &input.v_words[kv_range],
            &mut expected_k,
            &mut expected_v,
            &current,
        )?;
        let expected = reference::run(
            &input.q_words[offset * q_stride..(offset + rows) * q_stride],
            &expected_k,
            &expected_v,
            &current,
        )?;
        launch_append(module, input, &k, &v, &current)?;
        launch_attention(module, &[input.q, &k, &v, &output, &unrounded], &current)?;
        context.synchronize()?;
        let cache_exact = words(&k, expected_k.len())? == expected_k
            && words(&v, expected_v.len())? == expected_v;
        boundary_reports.push(
            json!({"prefix_tokens":offset+rows,"cache_prefix_and_poison_tail_exact":cache_exact}),
        );
        expected_output.extend(expected.output);
        expected_fp32.extend(expected.unrounded);
        value_bounds.extend(expected.value_bounds);
        offset += rows;
    }
    let actual = words(&output, input.rows * q_stride)?;
    let fp32 = floats(&unrounded, input.rows * q_stride)?;
    let comparison = compare(
        &actual,
        &fp32,
        &expected_output,
        &expected_fp32,
        &value_bounds,
    )?;
    let all_passed = comparison["passed"] == true
        && boundary_reports
            .iter()
            .all(|b| b["cache_prefix_and_poison_tail_exact"] == true);
    Ok(Sequence {
        output: actual,
        unrounded: fp32,
        k: words(&k, expected_k.len())?,
        v: words(&v, expected_v.len())?,
        report: json!({"all_passed":all_passed,"partition":partition,"comparison":comparison,"boundaries":boundary_reports}),
    })
}

fn launch_append(
    module: &Module<'_>,
    input: &Input<'_, '_>,
    k: &Buffer<'_>,
    v: &Buffer<'_>,
    shape: &reference::Shape,
) -> Result<()> {
    let offset = u64::try_from(shape.past * shape.kv_heads * shape.width * 2)?;
    let mut pointers = [
        input.k.pointer() + offset,
        input.v.pointer() + offset,
        k.pointer(),
        v.pointer(),
    ];
    let mut dimensions = [
        u32::try_from(shape.rows)?,
        u32::try_from(shape.kv_heads)?,
        u32::try_from(shape.width)?,
        u32::try_from(shape.past)?,
        u32::try_from(shape.capacity)?,
    ];
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast())
        .collect();
    args.extend(dimensions.iter_mut().map(|p| (p as *mut u32).cast()));
    // SAFETY: Reference validation establishes exact input/cache extents and
    // append bounds. Offset input pointers address this chunk inside resident
    // projections. The destination prefix/tail is disjoint and stays live.
    unsafe {
        module.function("attention_kv_append")?.launch(
            [
                u32::try_from((shape.rows * shape.kv_heads * shape.width).div_ceil(256))?,
                1,
                1,
            ],
            [256, 1, 1],
            0,
            &mut args,
        )
    }
}

fn launch_attention(
    module: &Module<'_>,
    buffers: &[&Buffer<'_>; 5],
    shape: &reference::Shape,
) -> Result<()> {
    let offset = u64::try_from(shape.past * shape.query_heads * shape.width)?;
    let mut pointers = [
        buffers[0].pointer() + offset * 2,
        buffers[1].pointer(),
        buffers[2].pointer(),
        buffers[3].pointer() + offset * 2,
        buffers[4].pointer() + offset * 4,
    ];
    let mut dimensions = [
        u32::try_from(shape.rows)?,
        u32::try_from(shape.query_heads)?,
        u32::try_from(shape.kv_heads)?,
        u32::try_from(shape.width)?,
        u32::try_from(shape.past)?,
        u32::try_from(shape.capacity)?,
    ];
    let mut scale = shape.scale;
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast())
        .collect();
    args.extend(dimensions.iter_mut().map(|p| (p as *mut u32).cast()));
    args.push((&mut scale as *mut f32).cast());
    // SAFETY: Exact shape/cache bounds were validated by the independent oracle.
    // The prior append is ordered on the same default stream. Every query/head
    // owns a distinct output region and all 256 threads enter all CTA barriers.
    unsafe {
        module.function("causal_attention_bf16")?.launch(
            [u32::try_from(shape.rows * shape.query_heads)?, 1, 1],
            [256, 1, 1],
            0,
            &mut args,
        )
    }
}

fn compare(
    actual: &[u16],
    fp32: &[f32],
    expected: &[u16],
    wide: &[f32],
    bounds: &[f32],
) -> Result<Value> {
    ensure!(
        actual.len() == expected.len()
            && fp32.len() == wide.len()
            && fp32.len() == actual.len()
            && bounds.len() == actual.len(),
        "attention comparison extent mismatch"
    );
    let mut max_error = 0.0_f32;
    let mut numerical = 0;
    let mut rounding = 0;
    let mut bf16_differences = 0;
    for (i, (&value, &reference)) in fp32.iter().zip(wide).enumerate() {
        ensure!(
            value.is_finite() && bf16_to_f32(actual[i]).is_finite(),
            "nonfinite causal attention result"
        );
        let error = (value - reference).abs();
        max_error = max_error.max(error);
        numerical += usize::from(error > 5e-6 + 3e-5 * bounds[i]);
        rounding += usize::from(actual[i] != round_bf16(value));
        bf16_differences += usize::from(actual[i] != expected[i]);
    }
    Ok(
        json!({"passed":numerical==0 && rounding==0,"elements":actual.len(),"max_abs_error":max_error,"numerical_mismatches":numerical,"rounding_mismatches":rounding,"bf16_scalar_differences":bf16_differences,"tolerance":"5e-6 + 3e-5*max(abs(causally visible V channel))"}),
    )
}

pub(super) fn fixtures(context: &Context, module: &Module<'_>) -> Result<Vec<Value>> {
    let mut reports = Vec::new();
    for (rows, q_heads, kv_heads, width, mode) in [
        (5, 4, 2, 8, 0),
        (7, 6, 2, 13, 1),
        (4, 2, 1, 256, 2),
        (1, 1, 1, 2, 0),
    ] {
        let q: Vec<_> = (0..rows * q_heads * width)
            .map(|i| {
                round_bf16(if mode == 1 {
                    0.0
                } else if mode == 2 {
                    if i % 3 == 0 { 32.0 } else { -16.0 }
                } else {
                    ((i * 17 % 23) as i32 - 11) as f32 / 8.0
                })
            })
            .collect();
        let k: Vec<_> = (0..rows * kv_heads * width)
            .map(|i| round_bf16(((i * 11 % 29) as i32 - 14) as f32 / 8.0))
            .collect();
        let v: Vec<_> = (0..k.len())
            .map(|i| round_bf16(((i * 7 % 31) as i32 - 15) as f32 / 4.0))
            .collect();
        let q_device = upload_words(context, &q)?;
        let k_device = upload_words(context, &k)?;
        let v_device = upload_words(context, &v)?;
        let mut report = check(
            context,
            module,
            Input {
                q: &q_device,
                k: &k_device,
                v: &v_device,
                q_words: &q,
                k_words: &k,
                v_words: &v,
                rows,
                query_heads: q_heads,
                kv_heads,
                width,
            },
        )?;
        report["fixture_mode"] = json!(mode);
        reports.push(report);
    }
    Ok(reports)
}

fn upload<'a>(context: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}
fn upload_words<'a>(context: &'a Context, words: &[u16]) -> Result<Buffer<'a>> {
    upload(
        context,
        &words
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn words(buffer: &Buffer<'_>, count: usize) -> Result<Vec<u16>> {
    let mut bytes = vec![0; count * 2];
    buffer.download(&mut bytes)?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect())
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
