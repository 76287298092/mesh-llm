//! Same-input real-model attention diagnostics; only the exact output feeds the model.
use super::{
    driver::{Buffer, Context, Module},
    resident_attention_core::Shape,
    resident_state::ResidentState,
};
use crate::{
    attention_online_reference as oracle,
    entry_reference::{bf16_to_f32, round_bf16},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{cell::RefCell, ffi::c_void};
thread_local! { static REPORTS:RefCell<Vec<Value>>=const {RefCell::new(Vec::new())}; }
pub(super) fn take_reports() -> Vec<Value> {
    REPORTS.with(|v| std::mem::take(&mut *v.borrow_mut()))
}
pub(super) struct Case<'a, 'ctx> {
    pub q: &'a Buffer<'ctx>,
    pub state: &'a ResidentState<'ctx>,
    pub prefix: &'a str,
    pub output: &'a Buffer<'ctx>,
    pub raw: &'a Buffer<'ctx>,
    pub shape: &'a Shape,
}
struct Host {
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    exact: Vec<f32>,
    online: Vec<f32>,
}
pub(super) fn compare(ctx: &Context, module: &Module<'_>, case: Case<'_, '_>) -> Result<()> {
    let s = case.shape;
    let kind = if s.rows >= 16 {
        "prefill"
    } else if s.rows == 1 && s.past >= 16 {
        "decode"
    } else {
        return Ok(());
    };
    if case.prefix != "layers.03" || REPORTS.with(|v| v.borrow().iter().any(|r| r["kind"] == kind))
    {
        return Ok(());
    }
    ensure!(
        s.capacity <= 2048,
        "attention audit is bounded to capacity2048"
    );
    let (output, raw) = candidate(ctx, module, &case)?;
    let exact_words = words(case.output)?;
    let online_words = words(&output)?;
    let host = Host {
        q: words(case.q)?,
        k: cache(case.state, case.prefix, "k", s)?,
        v: cache(case.state, case.prefix, "v", s)?,
        exact: floats(case.raw)?,
        online: floats(&raw)?,
    };
    let finite = host.exact.iter().chain(&host.online).all(|v| v.is_finite());
    let rounding = online_words
        .iter()
        .zip(&host.online)
        .all(|(&word, &value)| word == round_bf16(value));
    let mut samples = Vec::new();
    let mut rows = vec![0, s.rows / 2, s.rows - 1];
    if let Some(index) = host
        .exact
        .iter()
        .zip(&host.online)
        .enumerate()
        .max_by(|(_, (a, b)), (_, (c, d))| (*a - *b).abs().total_cmp(&(*c - *d).abs()))
        .map(|(index, _)| index)
    {
        rows.push(index / (s.query_heads * s.width));
    }
    rows.sort_unstable();
    rows.dedup();
    for row in rows {
        for head in 0..s.query_heads {
            samples.push(sample(&host, s, row, head)?);
        }
    }
    let (max_ratio, max_error) = compare_all(&host, s);
    let norm = host
        .exact
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>();
    let squared = host
        .exact
        .iter()
        .zip(&host.online)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum::<f64>();
    let passed =
        finite && rounding && max_ratio <= 1.0 && samples.iter().all(|v| v["all_passed"] == true);
    REPORTS.with(|v| v.borrow_mut().push(json!({"kind":kind,"layer":case.prefix,"rows":s.rows,"past":s.past,"capacity":s.capacity,"query_heads":s.query_heads,"kv_heads":s.kv_heads,"width":s.width,
        "outputs":online_words.len(),"bf16_differences":exact_words.iter().zip(&online_words).filter(|(a,b)|a!=b).count(),
        "raw_max_abs_error":max_error,"raw_relative_l2":(squared/norm.max(1e-30)).sqrt(),"all_output_exact_gpu_budget_ratio":max_ratio,
        "finite":finite,"stored_bf16_matches_raw_rounding":rounding,"samples":samples,"all_passed":passed,"diagnostic_only":true,"timing_claim":false})));
    Ok(())
}
fn candidate<'a>(
    ctx: &'a Context,
    module: &Module<'_>,
    case: &Case<'_, '_>,
) -> Result<(Buffer<'a>, Buffer<'a>)> {
    let s = case.shape;
    let output = Buffer::new(ctx, case.output.len())?;
    let raw = Buffer::new(ctx, case.raw.len())?;
    let cache_bytes = s.capacity * s.kv_heads * s.width * 2;
    let mut pointers = [
        case.q.pointer(),
        case.state
            .pointer(&format!("{}.attention.k", case.prefix), cache_bytes)?,
        case.state
            .pointer(&format!("{}.attention.v", case.prefix), cache_bytes)?,
        output.pointer(),
        raw.pointer(),
    ];
    let mut dims = [
        s.rows,
        s.query_heads,
        s.kv_heads,
        s.width,
        s.past,
        s.capacity,
    ]
    .map(|v| v as u32);
    let mut scale = 1.0 / (s.width as f32).sqrt();
    let mut args = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(dims.iter_mut().map(|d| (d as *mut u32).cast()));
    args.push((&mut scale as *mut f32).cast());
    // SAFETY: Caller validated same-context Q/cache extents for the identical ABI;
    // outputs are disjoint live allocations and the launch is drained before return.
    let launch = unsafe {
        module.function("attention_online_bf16")?.launch(
            [(s.rows * s.query_heads) as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut args,
        )
    };
    let sync = ctx.synchronize();
    launch?;
    sync?;
    Ok((output, raw))
}
fn compare_all(host: &Host, s: &Shape) -> (f32, f32) {
    let mut bounds = vec![0.0_f32; s.kv_heads * s.width];
    let mut ratio = 0.0_f32;
    let mut error = 0.0_f32;
    for position in 0..s.past + s.rows {
        for (channel, bound) in bounds.iter_mut().enumerate() {
            *bound =
                bound.max(bf16_to_f32(host.v[position * s.kv_heads * s.width + channel]).abs());
        }
        if position < s.past {
            continue;
        }
        let row = position - s.past;
        for head in 0..s.query_heads {
            for channel in 0..s.width {
                let index = (row * s.query_heads + head) * s.width + channel;
                let delta = (host.online[index] - host.exact[index]).abs();
                let bound = bounds[head / (s.query_heads / s.kv_heads) * s.width + channel];
                ratio = ratio.max(
                    delta / (oracle::COMPONENT_ABS_BUDGET + oracle::COMPONENT_REL_BUDGET * bound),
                );
                error = error.max(delta);
            }
        }
    }
    (ratio, error)
}
fn sample(host: &Host, s: &Shape, row: usize, head: usize) -> Result<Value> {
    let kv = head / (s.query_heads / s.kv_heads);
    let start = (row * s.query_heads + head) * s.width;
    let extract = |values: &[u16]| {
        (0..s.capacity)
            .flat_map(|p| {
                values[(p * s.kv_heads + kv) * s.width..(p * s.kv_heads + kv + 1) * s.width]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>()
    };
    let expected = oracle::run(
        &host.q[start..start + s.width],
        &extract(&host.k),
        &extract(&host.v),
        &oracle::Shape {
            rows: 1,
            query_heads: 1,
            kv_heads: 1,
            width: s.width,
            past: s.past + row,
            capacity: s.capacity,
            scale: 1.0 / (s.width as f32).sqrt(),
        },
    )?;
    let mut exact_ratio = 0.0_f32;
    let mut online_ratio = 0.0_f32;
    for channel in 0..s.width {
        let budget = oracle::COMPONENT_ABS_BUDGET
            + oracle::COMPONENT_REL_BUDGET * expected.value_bounds[channel];
        exact_ratio = exact_ratio
            .max((host.exact[start + channel] - expected.unrounded[channel]).abs() / budget);
        online_ratio = online_ratio
            .max((host.online[start + channel] - expected.unrounded[channel]).abs() / budget);
    }
    Ok(
        json!({"row":row,"head":head,"channels":s.width,"exact_cpu_budget_ratio":exact_ratio,"online_cpu_budget_ratio":online_ratio,"all_passed":exact_ratio<=1.0 && online_ratio<=1.0}),
    )
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
fn cache(state: &ResidentState<'_>, prefix: &str, suffix: &str, s: &Shape) -> Result<Vec<u16>> {
    let mut bytes = vec![0; s.capacity * s.kv_heads * s.width * 2];
    state.read_region(&format!("{prefix}.attention.{suffix}"), &mut bytes)?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| u16::from_le_bytes(*v))
        .collect())
}
