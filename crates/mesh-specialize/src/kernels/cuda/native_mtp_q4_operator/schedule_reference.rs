use super::validate::ValidatedView;
use anyhow::{Context, Result};

pub(super) struct ScheduleProjection {
    pub(super) raw_f32: Vec<f32>,
    pub(super) logits_bf16: Vec<u16>,
}

pub(super) fn run(
    object_bytes: &[u8],
    input_bf16: &[u16],
    view: &ValidatedView,
) -> Result<ScheduleProjection> {
    let mut raw_f32 = Vec::with_capacity(view.source_rows.len());
    let mut logits_bf16 = Vec::with_capacity(view.source_rows.len());
    for &parent_row in &view.source_rows {
        let value = row_dot(object_bytes, input_bf16, view, parent_row)?;
        raw_f32.push(value);
        logits_bf16.push(crate::entry_reference::round_bf16(value));
    }
    Ok(ScheduleProjection {
        raw_f32,
        logits_bf16,
    })
}

fn row_dot(
    object_bytes: &[u8],
    input_bf16: &[u16],
    view: &ValidatedView,
    parent_row: u32,
) -> Result<f32> {
    let padded_k = usize::try_from(view.padded_k)?;
    let logical_k = usize::try_from(view.logical_k)?;
    let parent_row = usize::try_from(parent_row)?;
    let code_row = parent_row * (padded_k / 2);
    let groups_per_row = padded_k / 64;
    let scale_row = usize::try_from(view.scale_offset)? + parent_row * groups_per_row * 2;
    let logical_groups = logical_k.div_ceil(64);
    let mut lanes = [0.0_f32; 32];
    let mut tile_begin = 0;
    while tile_begin < logical_groups {
        let active_groups = (logical_groups - tile_begin).min(16);
        let mut group_base = 0;
        while group_base < 16 {
            for (lane, accumulator) in lanes.iter_mut().enumerate() {
                let lane_group = lane >> 3;
                let local_group = group_base + lane_group;
                if local_group < active_groups {
                    let lane_in_group = lane & 7;
                    let group = tile_begin + local_group;
                    let word_offset = code_row + group * 32 + lane_in_group * 4;
                    let packed_bytes: [u8; 4] = object_bytes
                        .get(word_offset..word_offset + 4)
                        .context("Q4 schedule-reference packed word is absent")?
                        .try_into()
                        .context("Q4 schedule-reference packed word has wrong size")?;
                    let packed = u32::from_le_bytes(packed_bytes);
                    let scale_pair = scale_row + (group & !1) * 2;
                    let bits_offset = scale_pair + (group & 1) * 2;
                    let scale_bytes: [u8; 2] = object_bytes
                        .get(bits_offset..bits_offset + 2)
                        .context("Q4 schedule-reference scale is absent")?
                        .try_into()
                        .context("Q4 schedule-reference scale has wrong size")?;
                    let scale = f32::from_bits(decode_half(u16::from_le_bytes(scale_bytes)));
                    let k_begin = group * 64 + lane_in_group * 8;
                    for code_index in 0..8 {
                        let k = k_begin + code_index;
                        if k < logical_k {
                            let nibble = ((packed >> (code_index * 4)) & 0x0f) as u8;
                            let signed = i32::from(nibble & 0x07) - i32::from(nibble & 0x08);
                            let weight = scale * signed as f32;
                            let activation = f32::from_bits(u32::from(input_bf16[k]) << 16);
                            *accumulator = weight.mul_add(activation, *accumulator);
                        }
                    }
                }
            }
            group_base += 4;
        }
        tile_begin += 16;
    }
    for offset in [16, 8, 4, 2, 1] {
        let prior = lanes;
        for lane in 0..(32 - offset) {
            lanes[lane] = prior[lane] + prior[lane + offset];
        }
    }
    Ok(lanes[0])
}

fn decode_half(bits: u16) -> u32 {
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    if exponent == 0 {
        (f32::from(bits & 0x03ff) * 2.0_f32.powi(-24)).to_bits()
    } else {
        ((exponent + 112) << 23) | (fraction << 13)
    }
}
