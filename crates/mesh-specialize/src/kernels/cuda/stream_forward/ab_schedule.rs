//! Optional paired A/B handle and paired-only arena liveness, without new buffers.
use super::{
    ops::{Args, Enqueue, to_u32},
    plan::{ArenaPlan, BufferSpec},
    program::Shapes,
};
use crate::kernels::{
    ab_schedule::{PAIRED_KERNEL, Schedule},
    cuda::{
        driver::{Function, Module},
        resident_bf16::validate_launch_access,
    },
};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};

const OUTPUTS: [&str; 4] = ["gdn.a.values", "gdn.b.values", "gdn.a.raw", "gdn.b.raw"];
const INPUT: &str = "gdn.norm.out";

pub(super) struct PairedAb<'m, 'ctx> {
    kernel: Function<'m, 'ctx>,
    channels: usize,
    width: usize,
    merged_steps: [usize; 2],
}

impl<'m, 'ctx> PairedAb<'m, 'ctx> {
    /// Baseline and unsupported shapes return None without changing the plan or
    /// looking up a paired handle. Caller checked module/context before construction.
    pub(super) fn new(
        schedule: Schedule,
        module: &'m Module<'ctx>,
        shapes: &Shapes,
        specs: &mut [BufferSpec],
    ) -> Result<Option<Self>> {
        let Some(merged_steps) = prepare_lifetimes(schedule, shapes, specs)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            kernel: module.function(PAIRED_KERNEL)?,
            channels: shapes.gdn_value_heads,
            width: shapes.hidden,
            merged_steps,
        }))
    }

    pub(super) fn validate_plan(&self, plan: &ArenaPlan, max_rows: usize) -> Result<()> {
        validate_arena_access(plan, max_rows, self.channels, self.width)
    }

    /// Caller supplies retained, correctly sized arena/weight views in e's context.
    /// All seven spans are checked for alignment, overflow and aliasing, including
    /// normalized input against every output. Enqueue/ActiveStream check context.
    /// No allocation, lookup, upload or synchronization occurs on this path.
    pub(super) fn enqueue(
        &self,
        e: &Enqueue<'_, '_, '_, '_>,
        pointers: [u64; 7],
        rows: usize,
    ) -> Result<()> {
        ensure!(rows == 1, "paired A/B stream launch requires M=1");
        validate_launch_access(pointers, self.channels, self.width)?;
        let args = Args::new()
            .ptrs(&pointers)
            .u32(to_u32(self.channels)?)
            .u32(to_u32(self.width)?);
        e.launch(
            &self.kernel,
            [to_u32(self.channels)?, 2, 1],
            [128, 1, 1],
            args,
        )
    }

    pub(super) fn report(&self) -> Value {
        json!({"kernel":PAIRED_KERNEL,"shape":[1,self.channels,self.width],
            "grid":[self.channels,2,1],"block":[128,1,1],
            "additional_buffers":0,"static_shared_bytes":32,
            "merged_template_steps":self.merged_steps,
            "liveness":"four A/B outputs overlap both original A/B op steps; normalized input covers both",
            "other_rows":"unchanged two bf16_linear_decode launches",
            "graph_capture":"rejected until combined qualification"})
    }
}

/// The original template has consecutive A and B projection operations. Merge
/// their scheduling window by extending each output over both inclusive steps.
/// Do not renumber other operations or alter any baseline specs. Input's existing
/// lifetime must cover both steps. ArenaPlan then prevents all five live views,
/// and every other value live over this window, from overlapping.
fn prepare_lifetimes(
    schedule: Schedule,
    shapes: &Shapes,
    specs: &mut [BufferSpec],
) -> Result<Option<[usize; 2]>> {
    if schedule.select(1, shapes.gdn_value_heads, shapes.hidden) != Schedule::PairedFp64 {
        return Ok(None);
    }
    let mut indices = [0_usize; 5];
    for (i, name) in OUTPUTS.iter().chain([&INPUT]).enumerate() {
        indices[i] = specs
            .iter()
            .position(|spec| spec.name == *name)
            .with_context(|| format!("paired A/B plan is missing {name}"))?;
    }
    let [a, b] = [specs[indices[0]].first, specs[indices[1]].first];
    ensure!(
        a.checked_add(1) == Some(b) && specs[indices[2]].first == a && specs[indices[3]].first == b,
        "paired A/B expects consecutive A/B template operations with their raw outputs"
    );
    let input = &specs[indices[4]];
    ensure!(
        input.first <= a && input.last >= b,
        "normalized input must cover both paired A/B template steps"
    );
    for &index in &indices[..4] {
        specs[index].first = a;
        specs[index].last = specs[index].last.max(b);
    }
    Ok(Some([a, b]))
}

/// Validate the full max_rows slots, not just the first row used by the paired
/// kernel. The extended arena also serves unchanged multi-row baseline launches.
fn validate_arena_access(plan: &ArenaPlan, max_rows: usize, n: usize, k: usize) -> Result<()> {
    ensure!(max_rows > 0, "paired A/B arena max_rows must be positive");
    let mut spans = [(0_usize, 0_usize); 5];
    for (i, (name, width, element_bytes)) in [
        (OUTPUTS[0], n, 2),
        (OUTPUTS[1], n, 2),
        (OUTPUTS[2], n, 4),
        (OUTPUTS[3], n, 4),
        (INPUT, k, 2),
    ]
    .into_iter()
    .enumerate()
    {
        let expected = max_rows
            .checked_mul(width)
            .and_then(|v| v.checked_mul(element_bytes))
            .context("paired A/B arena extent overflow")?;
        let view = plan.get(name)?;
        ensure!(
            view.bytes == expected,
            "paired A/B slot {name} extent mismatch"
        );
        let end = view
            .offset
            .checked_add(view.bytes)
            .context("paired A/B arena address overflow")?;
        ensure!(
            end <= plan.total_bytes && view.offset.is_multiple_of(16),
            "paired A/B slot {name} is outside the arena or unaligned"
        );
        for &(start, previous_end) in &spans[..i] {
            ensure!(
                end <= start || previous_end <= view.offset,
                "paired A/B arena outputs/input overlap at {name}"
            );
        }
        spans[i] = (view.offset, end);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::program::{Slots, forward_program};
    use super::*;

    fn qwen() -> Shapes {
        Shapes {
            hidden: 5120,
            intermediate: 17408,
            vocabulary: 248_320,
            gdn_key_heads: 16,
            gdn_value_heads: 48,
            gdn_head_width: 128,
            query_heads: 24,
            kv_heads: 4,
            attention_width: 256,
            rotary_dim: 64,
        }
    }

    #[test]
    fn paired_plans_keep_all_outputs_and_input_disjoint_at_all_requested_capacities() {
        for max_rows in [1, 16, 512] {
            let shapes = qwen();
            let original = forward_program(&shapes, max_rows, false).unwrap();
            let mut paired = original.clone();
            let [a, b] = prepare_lifetimes(Schedule::PairedFp64, &shapes, &mut paired)
                .unwrap()
                .unwrap();
            assert_eq!([a, b], [7, 8]); // Original A and B projection op indices.
            for spec in &paired {
                if OUTPUTS.contains(&spec.name.as_str()) || spec.name == INPUT {
                    assert!(spec.first <= a && spec.last >= b, "{}", spec.name);
                } else {
                    assert_eq!(spec, original.iter().find(|s| s.name == spec.name).unwrap());
                }
            }
            let plan = ArenaPlan::place(&paired).unwrap();
            plan.validate(&paired).unwrap();
            validate_arena_access(&plan, max_rows, 48, 5120).unwrap();
            assert!(validate_arena_access(&plan, max_rows + 1, 48, 5120).is_err());
            let slots = Slots::resolve(&plan, 1 << 40).unwrap();
            let g = slots.gdn;
            // Distinct stand-in resident allocations outside the arena.
            validate_launch_access(
                [g.norm.out, 1 << 42, 1 << 43, g.a[0], g.b[0], g.a[1], g.b[1]],
                48,
                5120,
            )
            .unwrap();
        }
    }

    #[test]
    fn baseline_plan_is_unchanged_and_unsupported_shape_falls_back() {
        for max_rows in [1, 16, 512] {
            let mut shapes = qwen();
            let original = forward_program(&shapes, max_rows, false).unwrap();
            let mut specs = original.clone();
            assert!(
                prepare_lifetimes(Schedule::Baseline, &shapes, &mut specs)
                    .unwrap()
                    .is_none()
            );
            assert_eq!(specs, original);
            shapes.hidden = 5121;
            assert!(
                prepare_lifetimes(Schedule::PairedFp64, &shapes, &mut specs)
                    .unwrap()
                    .is_none()
            );
            assert_eq!(specs, original);
        }
    }

    #[test]
    fn rejects_short_output_slot_even_with_nonoverlapping_placement() {
        let shapes = qwen();
        let mut specs = forward_program(&shapes, 16, false).unwrap();
        prepare_lifetimes(Schedule::PairedFp64, &shapes, &mut specs).unwrap();
        specs
            .iter_mut()
            .find(|s| s.name == "gdn.a.raw")
            .unwrap()
            .bytes -= 4;
        let plan = ArenaPlan::place(&specs).unwrap();
        assert!(validate_arena_access(&plan, 16, 48, 5120).is_err());
    }

    #[test]
    fn malformed_template_rejected_before_mutation() {
        let shapes = qwen();
        let mut specs = forward_program(&shapes, 1, false).unwrap();
        specs.iter_mut().find(|s| s.name == INPUT).unwrap().last = 0;
        let before = specs.clone();
        assert!(prepare_lifetimes(Schedule::PairedFp64, &shapes, &mut specs).is_err());
        assert_eq!(specs, before);
    }
}
