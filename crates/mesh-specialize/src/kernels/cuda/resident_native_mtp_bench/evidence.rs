use super::{
    LoadedModels,
    execution::{self, Observed},
};
use crate::{engine::session::Cursor, kernels::cuda::resident_state::ResidentState};
use anyhow::Result;
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const CHUNK_BYTES: usize = 1024 * 1024;

pub(super) fn snapshot(state: &ResidentState<'_>, cursor: &Cursor) -> Result<Value> {
    let mut regions = Vec::with_capacity(state.layout().regions.len());
    let mut scratch = vec![0; CHUNK_BYTES];
    for region in &state.layout().regions {
        let length = usize::try_from(region.length)?;
        let mut digest = Sha256::new();
        for offset in (0..length).step_by(CHUNK_BYTES) {
            let bytes = &mut scratch[..(length - offset).min(CHUNK_BYTES)];
            state.read_region_at(&region.name, offset, bytes)?;
            digest.update(bytes);
        }
        regions.push(json!({"name": region.name, "bytes": region.length,
            "sha256": format!("{:x}", digest.finalize())}));
    }
    Ok(json!({"past": cursor.past(), "capacity": cursor.capacity(),
        "poisoned": cursor.is_poisoned(), "regions": regions}))
}

#[derive(Serialize)]
pub(super) struct Comparison {
    output_differences: Vec<Difference<u32>>,
    complete_fixed_outputs: bool,
    target_cursor_matches: bool,
    draft_cursor_matches_target: bool,
    target_state_differences: Vec<RegionDifference>,
    continuation_token_differences: Vec<Difference<u32>>,
    continuation_logit_differences: Vec<Difference<u16>>,
    continuation_cursor_matches: bool,
    continuation_state_differences: Vec<RegionDifference>,
    baseline_continuation: Value,
    native_continuation: Value,
}

impl Comparison {
    pub(super) fn matches(&self) -> bool {
        self.complete_fixed_outputs
            && self.output_differences.is_empty()
            && self.target_cursor_matches
            && self.draft_cursor_matches_target
            && self.target_state_differences.is_empty()
            && self.continuation_token_differences.is_empty()
            && self.continuation_logit_differences.is_empty()
            && self.continuation_cursor_matches
            && self.continuation_state_differences.is_empty()
    }
}

pub(super) fn compare(
    models: &LoadedModels<'_, '_, '_>,
    baseline: &Observed<'_>,
    native: &Observed<'_>,
) -> Result<Comparison> {
    models.context.synchronize()?;
    let target_state_differences = state_differences(&baseline.target.state, &native.target.state)?;
    let (baseline_branch, baseline_output) = execution::continuation(models, baseline)?;
    let (native_branch, native_output) = execution::continuation(models, native)?;
    Ok(Comparison {
        output_differences: differences(&baseline.tokens, &native.tokens),
        complete_fixed_outputs: baseline.tokens.len() == baseline.requested_output_tokens
            && native.tokens.len() == native.requested_output_tokens,
        target_cursor_matches: cursors_match(&baseline.target.cursor, &native.target.cursor),
        draft_cursor_matches_target: match &native.draft {
            Some(draft) => cursors_match(&native.target.cursor, &draft.cursor),
            None => false,
        },
        target_state_differences,
        continuation_token_differences: differences(
            &[baseline_output.token],
            &[native_output.token],
        ),
        continuation_logit_differences: differences(&baseline_output.logits, &native_output.logits),
        continuation_cursor_matches: cursors_match(&baseline_branch.cursor, &native_branch.cursor),
        continuation_state_differences: state_differences(
            &baseline_branch.state,
            &native_branch.state,
        )?,
        baseline_continuation: json!({"token": baseline_output.token, "logits_bf16_bits": baseline_output.logits,
            "target": snapshot(&baseline_branch.state, &baseline_branch.cursor)?}),
        native_continuation: json!({"token": native_output.token, "logits_bf16_bits": native_output.logits,
            "target": snapshot(&native_branch.state, &native_branch.cursor)?}),
    })
}

fn cursors_match(left: &Cursor, right: &Cursor) -> bool {
    left.past() == right.past()
        && left.capacity() == right.capacity()
        && left.is_poisoned() == right.is_poisoned()
        && !left.is_poisoned()
}

#[derive(Serialize)]
struct Difference<T> {
    offset: usize,
    baseline: Option<T>,
    native: Option<T>,
}

fn differences<T: Copy + Eq>(left: &[T], right: &[T]) -> Vec<Difference<T>> {
    (0..left.len().max(right.len()))
        .filter_map(|offset| {
            let baseline = left.get(offset).copied();
            let native = right.get(offset).copied();
            (baseline != native).then_some(Difference {
                offset,
                baseline,
                native,
            })
        })
        .collect()
}

#[derive(Serialize)]
struct RegionDifference {
    name: String,
    baseline_bytes: Option<u64>,
    native_bytes: Option<u64>,
    byte_differences: Vec<Difference<u8>>,
}

fn state_differences(
    left: &ResidentState<'_>,
    right: &ResidentState<'_>,
) -> Result<Vec<RegionDifference>> {
    let mut result = Vec::new();
    let mut left_chunk = vec![0; CHUNK_BYTES];
    let mut right_chunk = vec![0; CHUNK_BYTES];
    let names: std::collections::BTreeSet<_> = left
        .layout()
        .regions
        .iter()
        .chain(&right.layout().regions)
        .map(|region| region.name.as_str())
        .collect();
    for name in names {
        let left_region = left
            .layout()
            .regions
            .iter()
            .find(|region| region.name == name);
        let right_region = right
            .layout()
            .regions
            .iter()
            .find(|region| region.name == name);
        let left_length = left_region.map(|region| region.length);
        let right_length = right_region.map(|region| region.length);
        let mut byte_differences = Vec::new();
        let length = usize::try_from(left_length.unwrap_or(0).max(right_length.unwrap_or(0)))?;
        for offset in (0..length).step_by(CHUNK_BYTES) {
            let left_size = usize::try_from(left_length.unwrap_or(0))?
                .saturating_sub(offset)
                .min(CHUNK_BYTES);
            let right_size = usize::try_from(right_length.unwrap_or(0))?
                .saturating_sub(offset)
                .min(CHUNK_BYTES);
            if left_size > 0 {
                left.read_region_at(name, offset, &mut left_chunk[..left_size])?;
            }
            if right_size > 0 {
                right.read_region_at(name, offset, &mut right_chunk[..right_size])?;
            }
            byte_differences.extend(
                differences(&left_chunk[..left_size], &right_chunk[..right_size])
                    .into_iter()
                    .map(|difference| Difference {
                        offset: offset + difference.offset,
                        ..difference
                    }),
            );
        }
        if left_length != right_length || !byte_differences.is_empty() {
            result.push(RegionDifference {
                name: name.to_owned(),
                baseline_bytes: left_length,
                native_bytes: right_length,
                byte_differences,
            });
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::differences;

    #[test]
    fn exact_differences_retain_values_and_missing_outputs() {
        let baseline = [1_u32, 2, 3];
        let native = [1_u32, 4];
        let result = differences(&baseline, &native);
        assert_eq!(result.len(), 2);
        assert_eq!(
            (result[0].offset, result[0].baseline, result[0].native),
            (1, Some(2), Some(4))
        );
        assert_eq!(
            (result[1].offset, result[1].baseline, result[1].native),
            (2, Some(3), None)
        );
    }
}
