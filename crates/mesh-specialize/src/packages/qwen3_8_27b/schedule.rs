//! Fixed Qwen3.8-27B text-layer schedule and persistent-state layout.

use crate::{
    artifact::schema::{Directory, Object, ObjectKind},
    engine::layout::Layout,
};
use anyhow::{Context, Result, ensure};
use serde::Serialize;

const LAYER_COUNT: usize = 64;
const GDN_HISTORY_BYTES: u64 = 3 * 10_240 * 2;
const GDN_RECURRENT_BYTES: u64 = 48 * 128 * 128 * 4;
const KV_BYTES_PER_TOKEN: u64 = 4 * 256 * 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionKind {
    Gdn,
    Full,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MlpKind {
    Nvfp4,
    Fp8,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Layer {
    pub index: usize,
    pub attention: AttentionKind,
    pub mlp: MlpKind,
}

#[derive(Clone, Debug, Serialize)]
pub struct Schedule {
    pub layers: Vec<Layer>,
    pub context_capacity: usize,
    pub states: Layout,
}

impl Schedule {
    pub fn new(capacity: usize) -> Result<Self> {
        ensure!(
            (1..=262_144).contains(&capacity),
            "Qwen context capacity must be in 1..=262144"
        );
        let capacity_u64 = u64::try_from(capacity)?;
        let mut layers = Vec::with_capacity(LAYER_COUNT);
        let mut regions = Vec::with_capacity(LAYER_COUNT * 2);
        for index in 0..LAYER_COUNT {
            let attention = if index % 4 == 3 {
                AttentionKind::Full
            } else {
                AttentionKind::Gdn
            };
            let mlp = if index < 56 {
                MlpKind::Nvfp4
            } else {
                MlpKind::Fp8
            };
            layers.push(Layer {
                index,
                attention,
                mlp,
            });
            append_layer_state(&mut regions, index, attention, capacity_u64)?;
        }
        Ok(Self {
            layers,
            context_capacity: capacity,
            states: Layout::new(regions)?,
        })
    }
}

fn append_layer_state(
    regions: &mut Vec<(String, u64)>,
    index: usize,
    attention: AttentionKind,
    capacity: u64,
) -> Result<()> {
    match attention {
        AttentionKind::Gdn => {
            regions.push((format!("layers.{index:02}.gdn.history"), GDN_HISTORY_BYTES));
            regions.push((
                format!("layers.{index:02}.gdn.recurrent"),
                GDN_RECURRENT_BYTES,
            ));
        }
        AttentionKind::Full => {
            let cache_bytes = capacity
                .checked_mul(KV_BYTES_PER_TOKEN)
                .context("Qwen KV cache extent overflows u64")?;
            regions.push((format!("layers.{index:02}.attention.k"), cache_bytes));
            regions.push((format!("layers.{index:02}.attention.v"), cache_bytes));
        }
    }
    Ok(())
}

/// Select the compiled text weights, leaving the separate MTP head for its own phase.
pub fn text_objects(directory: &Directory) -> Result<Vec<Object>> {
    // The strict inventory selects explicit raw or native profile totals.
    let inventory = crate::packages::qwen3_8_27b::inventory::validate(directory)?;
    let mut objects: Vec<_> = directory
        .objects
        .iter()
        .filter(|object| {
            object.kind == ObjectKind::Tensor
                && (object.name.starts_with("tensors/model.language_model.")
                    || object.name.starts_with("tensors/lm_head."))
        })
        .cloned()
        .collect();
    let byte_count = objects.iter().try_fold(0_u64, |total, object| {
        total
            .checked_add(object.length)
            .context("selected text tensor byte count overflows u64")
    })?;
    ensure!(
        objects.len() == inventory.text_tensors,
        "selected text tensor count is {}; expected {}",
        objects.len(),
        inventory.text_tensors
    );
    ensure!(
        byte_count == inventory.text_bytes,
        "selected text tensor bytes are {byte_count}; expected {}",
        inventory.text_bytes
    );
    objects.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(objects)
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionKind, GDN_HISTORY_BYTES, GDN_RECURRENT_BYTES, KV_BYTES_PER_TOKEN, Layer, MlpKind,
        Schedule,
    };
    use std::collections::HashSet;

    fn expected_layer(index: usize) -> Layer {
        Layer {
            index,
            attention: if index % 4 == 3 {
                AttentionKind::Full
            } else {
                AttentionKind::Gdn
            },
            mlp: if index < 56 {
                MlpKind::Nvfp4
            } else {
                MlpKind::Fp8
            },
        }
    }

    #[test]
    fn compiled_layers_follow_fixed_attention_and_mlp_patterns() {
        let schedule = Schedule::new(1).unwrap();
        assert_eq!(schedule.layers.len(), 64);
        for (index, layer) in schedule.layers.iter().enumerate() {
            assert_eq!(layer, &expected_layer(index));
        }
        assert_eq!(
            schedule
                .layers
                .iter()
                .filter(|layer| layer.attention == AttentionKind::Full)
                .count(),
            16
        );
        assert_eq!(
            schedule
                .layers
                .iter()
                .filter(|layer| layer.attention == AttentionKind::Gdn)
                .count(),
            48
        );
        assert_eq!(
            schedule
                .layers
                .iter()
                .filter(|layer| layer.mlp == MlpKind::Nvfp4)
                .count(),
            56
        );
        assert_eq!(
            schedule
                .layers
                .iter()
                .filter(|layer| layer.mlp == MlpKind::Fp8)
                .count(),
            8
        );
    }

    #[test]
    fn state_regions_are_unique_aligned_and_exact_for_supported_capacities() {
        for capacity in [1, 131_072, 262_144] {
            let schedule = Schedule::new(capacity).unwrap();
            let mut names = HashSet::new();
            let mut gdn_history = 0;
            let mut gdn_recurrent = 0;
            let mut full_k = 0;
            let mut full_v = 0;
            for region in &schedule.states.regions {
                assert!(names.insert(region.name.as_str()));
                assert_eq!(region.offset % 256, 0, "{}", region.name);
                assert_eq!(region.length % 256, 0, "{}", region.name);
                if region.name.ends_with(".gdn.history") {
                    assert_eq!(region.length, GDN_HISTORY_BYTES);
                    gdn_history += 1;
                } else if region.name.ends_with(".gdn.recurrent") {
                    assert_eq!(region.length, GDN_RECURRENT_BYTES);
                    gdn_recurrent += 1;
                } else if region.name.ends_with(".attention.k") {
                    assert_eq!(region.length, capacity as u64 * KV_BYTES_PER_TOKEN);
                    full_k += 1;
                } else if region.name.ends_with(".attention.v") {
                    assert_eq!(region.length, capacity as u64 * KV_BYTES_PER_TOKEN);
                    full_v += 1;
                } else {
                    panic!("unexpected state region: {}", region.name);
                }
            }
            assert_eq!(
                (gdn_history, gdn_recurrent, full_k, full_v),
                (48, 48, 16, 16)
            );
            assert_eq!(
                schedule.states.bytes,
                153_944_064 + capacity as u64 * 65_536
            );
        }
    }

    #[test]
    fn context_capacity_is_bounded() {
        assert!(Schedule::new(0).is_err());
        assert!(Schedule::new(262_145).is_err());
    }
}
