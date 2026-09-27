//! Compact accepted-prefix recovery; target projections are never rerun here.
use super::{
    driver::{Buffer, Context, Module},
    resident_model::Session,
    resident_state::ResidentState,
};
use crate::kernels::{DecoderBlockKind, DecoderConfig};
use anyhow::{Result, ensure};
use std::ffi::c_void;

pub(super) struct CoreOutput<'ctx> {
    pub output: Buffer<'ctx>,
    pub record: Option<Recurrence<'ctx>>,
}
pub(super) struct Recurrence<'ctx> {
    pub k: Buffer<'ctx>,
    pub decay: Buffer<'ctx>,
    pub delta: Buffer<'ctx>,
    pub rows: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub width: usize,
}
pub(super) struct LayerRecord<'ctx> {
    pub recurrence: Recurrence<'ctx>,
    pub projected_qkv: Buffer<'ctx>,
    pub history_name: String,
    pub recurrent_name: String,
}
pub(super) struct Recovery<'a, 'ctx> {
    pub config: &'a DecoderConfig,
    pub records: &'a [LayerRecord<'ctx>],
    pub verified: &'a Session<'ctx>,
    pub rows: usize,
}

pub(super) fn recover(
    ctx: &Context,
    module: &Module<'_>,
    base: &mut Session<'_>,
    recovery: Recovery<'_, '_>,
) -> Result<usize> {
    ensure!(
        base.state.belongs_to(ctx)
            && recovery.verified.state.belongs_to(ctx)
            && module.belongs_to(ctx),
        "compact recovery context mismatch"
    );
    ensure!(
        (1..=5).contains(&recovery.rows),
        "invalid compact recovery prefix"
    );
    let expected_records = recovery
        .config
        .layers
        .iter()
        .filter(|layer| matches!(layer.block, DecoderBlockKind::Gdn))
        .count();
    ensure!(
        recovery.records.len() == expected_records,
        "incomplete GDN recovery records"
    );
    let transaction = base.cursor.begin(recovery.rows)?;
    let end = transaction.past() + recovery.rows;
    ensure!(
        end <= recovery.verified.cursor.past() && !recovery.verified.cursor.is_poisoned(),
        "verification session does not cover accepted prefix"
    );
    let mut records = recovery.records.iter();
    for layer in &recovery.config.layers {
        match layer.block {
            DecoderBlockKind::Gdn => {
                let record = records.next().expect("record count validated");
                ensure!(
                    record.history_name == format!("{}.gdn.history", layer.state_prefix)
                        && record.recurrent_name == format!("{}.gdn.recurrent", layer.state_prefix),
                    "GDN recovery record order differs from model"
                );
                record.apply(ctx, module, &mut base.state, recovery.rows)?;
            }
            DecoderBlockKind::Attention => {
                let shape = &recovery.config.attention_shape;
                let row_bytes = shape.kv_heads * shape.head_width * 2;
                for suffix in ["k", "v"] {
                    let name = format!("{}.attention.{suffix}", layer.state_prefix);
                    base.state.copy_state_range(
                        &name,
                        &recovery.verified.state,
                        transaction.past() * row_bytes,
                        recovery.rows * row_bytes,
                    )?;
                }
            }
        }
    }
    ctx.synchronize()?;
    Ok(transaction.commit())
}

impl LayerRecord<'_> {
    fn apply(
        &self,
        ctx: &Context,
        module: &Module<'_>,
        state: &mut ResidentState<'_>,
        rows: usize,
    ) -> Result<()> {
        let r = &self.recurrence;
        ensure!(rows <= r.rows, "accepted prefix exceeds GDN record");
        let channels = (2 * r.key_heads + r.value_heads) * r.width;
        ensure!(
            self.projected_qkv.len() == r.rows * channels * 2,
            "recorded QKV extent mismatch"
        );
        let history = Buffer::new(ctx, channels * 6)?;
        // History is time-major [3, channels]. Keep the final three rows of
        // concatenated old history and accepted raw projected QKV.
        if rows < 3 {
            state.copy_region_to(
                &self.history_name,
                rows * channels * 2,
                &history,
                0,
                (3 - rows) * channels * 2,
            )?;
        }
        let copied_rows = rows.min(3);
        history.copy_from_at(
            (3 - copied_rows) * channels * 2,
            &self.projected_qkv,
            (rows - copied_rows) * channels * 2,
            copied_rows * channels * 2,
        )?;
        let pointer = state.pointer(&self.recurrent_name, r.value_heads * r.width * r.width * 4)?;
        r.replay(ctx, module, pointer, rows)?;
        state.copy_from(&self.history_name, &history)
    }
}
impl Recurrence<'_> {
    fn replay(&self, ctx: &Context, module: &Module<'_>, state: u64, rows: usize) -> Result<()> {
        ensure!(
            self.k.belongs_to(ctx) && self.decay.belongs_to(ctx) && self.delta.belongs_to(ctx),
            "GDN record context mismatch"
        );
        ensure!(
            (1..=5).contains(&self.rows)
                && rows <= self.rows
                && (1..=64).contains(&self.key_heads)
                && (1..=256).contains(&self.value_heads)
                && self.value_heads.is_multiple_of(self.key_heads)
                && self.width.is_power_of_two()
                && self.width <= 256,
            "invalid GDN replay geometry"
        );
        ensure!(
            self.k.len() == self.rows * self.key_heads * self.width * 4
                && self.decay.len() == self.rows * self.value_heads * 4
                && self.delta.len() == self.rows * self.value_heads * self.width * 4,
            "GDN replay record extent mismatch"
        );
        let mut pointers = [
            self.k.pointer(),
            self.decay.pointer(),
            self.delta.pointer(),
            state,
        ];
        let mut dims = [
            u32::try_from(rows)?,
            u32::try_from(self.key_heads)?,
            u32::try_from(self.value_heads)?,
            u32::try_from(self.width)?,
        ];
        let mut args = pointers
            .iter_mut()
            .map(|p| (p as *mut u64).cast::<c_void>())
            .collect::<Vec<_>>();
        args.extend(dims.iter_mut().map(|d| (d as *mut u32).cast()));
        // SAFETY: Record geometry/extents and contexts are checked above; state is an
        // exact-sized disjoint owned region, and all buffers survive synchronization.
        unsafe {
            module.function("gdn_replay_state")?.launch(
                [dims[2], 1, 1],
                [dims[3], 1, 1],
                0,
                &mut args,
            )?;
        }
        ctx.synchronize()
    }
}
