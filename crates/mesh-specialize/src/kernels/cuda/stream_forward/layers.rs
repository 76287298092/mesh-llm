//! Per-section enqueue sequences, in the exact order of the legacy forward and
//! of the arena template in `program.rs`.

use super::{
    ab_schedule::PairedAb,
    graph_position::past_argument,
    ops::{Args, EPSILON, Enqueue, to_u32},
    program::{NormSlots, Shapes, Slots},
    split_attention::SplitAttention,
    weights::{AttentionWeights, GdnWeights, MlpWeights, ModelWeights},
};
use crate::kernels::cuda::resident_state::ResidentState;
use anyhow::Result;

/// Per-forward position data. RoPE addresses already point at row `past`.
pub(super) struct Step {
    pub(super) rows: usize,
    pub(super) past: usize,
    pub(super) capacity: usize,
    pub(super) cos: u64,
    pub(super) sin: u64,
}

/// Embedding gather plus first norm; the residual output becomes `hidden`.
pub(super) fn entry(
    e: &Enqueue<'_, '_, '_, '_>,
    w: &ModelWeights,
    s: &Slots,
    shapes: &Shapes,
    rows: usize,
) -> Result<()> {
    let outputs = NormSlots {
        copy: s.hidden,
        out: s.entry[0],
        raw: s.entry[1],
    };
    let (table, ids) = if let Some(scale) = w.embedding_scale {
        let args = Args::new()
            .ptrs(&[w.embedding_table, scale, s.tokens, s.hidden])
            .u32(to_u32(shapes.hidden)?);
        e.launch(
            &e.kernels.fp8_embedding_gather,
            [to_u32(rows)?, 1, 1],
            [256, 1, 1],
            args,
        )?;
        // Identity IDs make table==residual safe: each thread rewrites its own
        // already-rounded BF16 source bits. No cross-row or normalized-output alias.
        // hidden/tokens/row_ids are whole-forward slots, so no planner reuse occurs.
        (s.hidden, s.row_ids)
    } else {
        (w.embedding_table, s.tokens)
    };
    e.embedding_norm([table, ids, w.first_norm], &outputs, rows, shapes.hidden)
}

/// Mirror of `resident_gdn::Layer::execute` without MLP, observers or recording.
pub(super) fn gdn(
    e: &Enqueue<'_, '_, '_, '_>,
    w: &GdnWeights,
    s: &Slots,
    shapes: &Shapes,
    state: &ResidentState<'_>,
    rows: usize,
    paired_ab: Option<&PairedAb<'_, '_>>,
) -> Result<()> {
    let g = &s.gdn;
    let k = e.kernels;
    let (h, qkv, inner) = (shapes.hidden, shapes.gdn_qkv(), shapes.gdn_inner());
    let (kh, vh, width) = (
        shapes.gdn_key_heads,
        shapes.gdn_value_heads,
        shapes.gdn_head_width,
    );
    let history = state.pointer(&w.history, qkv * 6)?;
    let recurrent = state.pointer(&w.recurrent, vh * width * width * 4)?;
    let rows_u32 = to_u32(rows)?;
    e.embedding_norm([s.hidden, s.row_ids, w.norm], &g.norm, rows, h)?;
    e.projection(&w.qkv, g.norm.out, &g.qkv, rows)?;
    e.projection(&w.z, g.norm.out, &g.z, rows)?;
    if let Some(paired) = paired_ab.filter(|_| rows == 1) {
        paired.enqueue(
            e,
            [g.norm.out, w.a, w.b, g.a[0], g.b[0], g.a[1], g.b[1]],
            rows,
        )?;
    } else {
        e.bf16_linear([g.norm.out, w.a], g.a, rows, [vh, h])?;
        e.bf16_linear([g.norm.out, w.b], g.b, rows, [vh, h])?;
    }
    let conv = Args::new()
        .ptrs(&[g.qkv.values, w.conv, history])
        .ptrs(&g.conv)
        .u32(rows_u32)
        .u32(to_u32(qkv)?);
    e.launch(
        &k.causal_conv4,
        [to_u32((rows * qkv).div_ceil(256))?, 1, 1],
        [256, 1, 1],
        conv,
    )?;
    e.copy(history, g.conv[0], qkv * 6)?;
    let dims = |args: Args| {
        Ok::<_, anyhow::Error>(
            args.u32(rows_u32)
                .u32(to_u32(kh)?)
                .u32(to_u32(vh)?)
                .u32(to_u32(width)?),
        )
    };
    e.launch(
        &k.gdn_qk_norm,
        [to_u32(rows * kh)?, 1, 1],
        [256, 1, 1],
        dims(Args::new().ptrs(&[g.conv[1], g.q, g.k]))?,
    )?;
    let gates = Args::new()
        .ptrs(&[g.a[0], g.b[0], w.a_log, w.dt_bias])
        .ptrs(&g.gates)
        .u32(rows_u32)
        .u32(to_u32(vh)?);
    e.launch(
        if w.f32_params {
            &k.gdn_gates_f32_params
        } else {
            &k.gdn_gates
        },
        [to_u32((rows * vh).div_ceil(256))?, 1, 1],
        [256, 1, 1],
        gates,
    )?;
    let recurrence = Args::new()
        .ptrs(&[g.q, g.k, g.conv[1], g.gates[0], g.gates[2]])
        .ptr(recurrent)
        .ptrs(&g.recurrent);
    e.launch(
        &k.gdn_recurrent,
        [to_u32(vh)?, 1, 1],
        [to_u32(width)?, 1, 1],
        dims(recurrence)?,
    )?;
    let groups = to_u32(rows * vh)?;
    let gated = Args::new()
        .ptrs(&[g.recurrent[0], g.z.values, w.gated_norm])
        .ptrs(&g.gated)
        .u32(groups)
        .u32(to_u32(width)?)
        .f32(EPSILON);
    e.launch(&k.gdn_gated_rms_norm, [groups, 1, 1], [256, 1, 1], gated)?;
    e.projection(&w.out, g.gated[0], &g.out, rows)?;
    debug_assert_eq!(inner, w.out.width);
    e.residual_norm(
        [s.hidden, g.out.values, w.post_norm],
        [s.post_sum, s.post_x, g.post_raw],
        rows,
        h,
    )
}

/// Mirror of `resident_attention::Layer::forward_observed` without MLP or observers.
pub(super) fn attention(
    e: &Enqueue<'_, '_, '_, '_>,
    w: &AttentionWeights,
    s: &Slots,
    shapes: &Shapes,
    state: &ResidentState<'_>,
    step: &Step,
    split: Option<&SplitAttention<'_, '_>>,
) -> Result<()> {
    let a = &s.attention;
    let k = e.kernels;
    let rows = step.rows;
    let (h, qh, kvh, aw) = (
        shapes.hidden,
        shapes.query_heads,
        shapes.kv_heads,
        shapes.attention_width,
    );
    let warp = k.causal_attention_warp.as_ref().filter(|_| rows == 1);
    if warp.is_some() {
        anyhow::ensure!(
            e.position.is_none(),
            "warp-fp64 attention has no qualified graph position variant"
        );
        crate::kernels::attention_warp_plan::validate([
            to_u32(rows)?,
            to_u32(qh)?,
            to_u32(kvh)?,
            to_u32(aw)?,
            to_u32(step.past)?,
            to_u32(step.capacity)?,
        ])?;
    }
    let cache_bytes = step.capacity * kvh * aw * 2;
    let key_state = state.pointer(&w.key_state, cache_bytes)?;
    let value_state = state.pointer(&w.value_state, cache_bytes)?;
    e.embedding_norm([s.hidden, s.row_ids, w.norm], &a.norm, rows, h)?;
    e.projection(&w.q, a.norm.out, &a.q, rows)?;
    e.projection(&w.k, a.norm.out, &a.k, rows)?;
    e.projection(&w.v, a.norm.out, &a.v, rows)?;
    for (input, weight, outputs, heads, with_gate) in [
        (a.q.values, w.q_norm, a.q_prepared, qh, 1),
        (a.k.values, w.k_norm, a.k_prepared, kvh, 0),
    ] {
        let prepare = Args::new()
            .ptrs(&[input, weight, step.cos, step.sin])
            .ptrs(&outputs)
            .u32(to_u32(rows)?)
            .u32(to_u32(heads)?)
            .u32(to_u32(aw)?)
            .u32(to_u32(shapes.rotary_dim)?)
            .u32(with_gate)
            .f32(EPSILON);
        let (function, prepare) = match e.position {
            Some(position) => (&position.prepare, prepare.ptr(position.past)),
            None => (&k.attention_qk_prepare, prepare),
        };
        e.launch(
            function,
            [to_u32(rows * heads)?, 1, 1],
            [256, 1, 1],
            prepare,
        )?;
    }
    let (past, capacity) = (to_u32(step.past)?, to_u32(step.capacity)?);
    let append = Args::new()
        .ptrs(&[a.k_prepared[0], a.v.values, key_state, value_state])
        .u32(to_u32(rows)?)
        .u32(to_u32(kvh)?)
        .u32(to_u32(aw)?);
    let append = past_argument(append, e.position.map(|p| p.past), past).u32(capacity);
    e.launch(
        e.position.map_or(&k.attention_kv_append, |p| &p.append),
        [to_u32((rows * kvh * aw).div_ceil(256))?, 1, 1],
        [256, 1, 1],
        append,
    )?;
    if let Some(split) = split.filter(|_| rows <= 8) {
        split.enqueue(
            e,
            [
                a.q_prepared[0],
                key_state,
                value_state,
                a.output[0],
                a.output[1],
            ],
            step,
        )?;
    } else {
        // SplitDecode changes M<=8; WarpFp64 changes only M=1. Other rows keep baseline.
        let scale = 1.0_f32 / (aw as f32).sqrt();
        let attend = Args::new()
            .ptrs(&[a.q_prepared[0], key_state, value_state])
            .ptrs(&a.output)
            .u32(to_u32(rows)?)
            .u32(to_u32(qh)?)
            .u32(to_u32(kvh)?)
            .u32(to_u32(aw)?);
        let attend = past_argument(attend, e.position.map(|p| p.past), past)
            .u32(capacity)
            .f32(scale);
        let (function, grid, block) = if let Some(warp) = warp {
            (
                warp,
                crate::kernels::attention_warp_plan::GRID,
                crate::kernels::attention_warp_plan::BLOCK,
            )
        } else {
            (
                e.position.map_or(&k.causal_attention, |p| &p.attention),
                [to_u32(rows * qh)?, 1, 1],
                [256, 1, 1],
            )
        };
        e.launch(function, grid, block, attend)?;
    }
    let count = to_u32(rows * shapes.query_width())?;
    let gate = Args::new()
        .ptrs(&[a.output[0], a.q_prepared[3]])
        .ptrs(&a.gated)
        .u32(count);
    e.launch(
        &k.attention_gate,
        [count.div_ceil(256), 1, 1],
        [256, 1, 1],
        gate,
    )?;
    e.projection(&w.out, a.gated[0], &a.out, rows)?;
    e.residual_norm(
        [s.hidden, a.out.values, w.post_norm],
        [s.post_sum, s.post_x, a.post_raw],
        rows,
        h,
    )
}

/// Mirror of `resident_mlp::Mlp::run` followed by the block's residual add.
pub(super) fn mlp(
    e: &Enqueue<'_, '_, '_, '_>,
    w: &MlpWeights,
    s: &Slots,
    shapes: &Shapes,
    rows: usize,
) -> Result<()> {
    let m = if w.nvfp4 { &s.mlp_nvfp4 } else { &s.mlp_fp8 };
    e.projection(&w.gate, s.post_x, &m.gate, rows)?;
    e.projection(&w.up, s.post_x, &m.up, rows)?;
    e.silu_product(
        m.gate.values,
        m.up.values,
        m.activation,
        rows * shapes.intermediate,
    )?;
    e.projection(&w.down, m.activation[0], &m.down, rows)?;
    e.residual_add(s.post_sum, m.down.values, s.hidden, rows * shapes.hidden)
}

/// Final norm of the last row (gathered through the persistent row-ID table,
/// replacing the legacy one-row device copy), FP8 head, and device greedy.
pub(super) fn head(
    e: &Enqueue<'_, '_, '_, '_>,
    w: &ModelWeights,
    s: &Slots,
    shapes: &Shapes,
    rows: usize,
) -> Result<()> {
    let last_row_id = s
        .row_ids
        .checked_add(u64::try_from((rows - 1) * 4)?)
        .ok_or_else(|| anyhow::anyhow!("row-ID address overflow"))?;
    e.embedding_norm(
        [s.hidden, last_row_id, w.final_norm],
        &s.head_norm,
        1,
        shapes.hidden,
    )?;
    e.projection(&w.head, s.head_norm.out, &s.head, 1)?;
    e.greedy(s.head.values, s.partials, s.result, shapes.vocabulary)
}
