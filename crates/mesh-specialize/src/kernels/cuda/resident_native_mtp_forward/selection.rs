use super::{HIDDEN, INTERMEDIATE, KV_HEADS, PROPOSAL_ROWS, QUERY_HEADS, VOCABULARY};
use crate::packages::qwen3_8_27b::native_mtp_views::{NativeMtpViews, Q8MatrixView};
use anyhow::{Context as _, Result, ensure};

pub(super) const fn physical_qkv_rows() -> [std::ops::Range<usize>; 4] {
    [0..6_144, 6_144..7_168, 7_168..13_312, 13_312..14_336]
}

pub(super) const fn logical_gate_up_rows() -> [std::ops::Range<usize>; 2] {
    [0..INTERMEDIATE, INTERMEDIATE..INTERMEDIATE * 2]
}

pub(super) fn validate_views(views: &NativeMtpViews) -> Result<()> {
    validate_q8_parent(&views.fc, [5_120, 10_240], 5_120)?;
    validate_q8_parent(&views.query_gate, [12_288, 5_120], 14_336)?;
    validate_q8_parent(&views.key, [1_024, 5_120], 14_336)?;
    validate_q8_parent(&views.value, [1_024, 5_120], 14_336)?;
    validate_q8_parent(&views.attention_output, [5_120, 6_144], 5_120)?;
    validate_q8_parent(&views.mlp_gate, [17_408, 5_120], 34_816)?;
    validate_q8_parent(&views.mlp_up, [17_408, 5_120], 34_816)?;
    validate_q8_parent(&views.mlp_down, [5_120, 17_408], 5_120)?;
    ensure!(
        views.fc.source_rows.iter().copied().eq(0..5_120)
            && views
                .attention_output
                .source_rows
                .iter()
                .copied()
                .eq(0..5_120)
            && views.mlp_down.source_rows.iter().copied().eq(0..5_120),
        "native MTP projection row mapping is not identity"
    );
    let [query, key, gate, value] = physical_qkv_rows();
    ensure!(
        views.key.source_rows.iter().copied().eq(key)
            && views.value.source_rows.iter().copied().eq(value),
        "native MTP key/value row maps differ from the physical QKV parent"
    );
    ensure!(
        views.query_gate.object_id == views.key.object_id
            && views.key.object_id == views.value.object_id,
        "native MTP Q/K/gate/V views do not share their physical parent"
    );
    validate_interleaved_q_gate(&views.query_gate, query, gate)?;
    let [gate_rows, up_rows] = logical_gate_up_rows();
    ensure!(
        views.mlp_gate.source_rows.iter().copied().eq(gate_rows)
            && views.mlp_up.source_rows.iter().copied().eq(up_rows),
        "native MTP gate/up row maps differ from the physical parent halves"
    );
    ensure!(
        views.mlp_gate.object_id == views.mlp_up.object_id,
        "native MTP gate/up views do not share their physical parent"
    );
    ensure!(
        views.proposal_head.shape == [PROPOSAL_ROWS, HIDDEN]
            && views.proposal_head.padded_k == HIDDEN
            && views.proposal_head.group_size == 64
            && views
                .proposal_head
                .source_rows
                .iter()
                .copied()
                .eq(0..PROPOSAL_ROWS)
            && views.proposal_tokens.len() == PROPOSAL_ROWS
            && VOCABULARY == 248_320
            && QUERY_HEADS == 24
            && KV_HEADS == 4,
        "native proposal geometry or row mapping differs from the checked model"
    );
    let vocabulary = u32::try_from(VOCABULARY).context("native MTP vocabulary exceeds u32")?;
    for row in 0..PROPOSAL_ROWS {
        let target = views
            .proposal_tokens
            .target_id(row)
            .context("native proposal map row is missing")?;
        ensure!(
            target.value() < vocabulary,
            "native proposal target ID exceeds vocabulary"
        );
    }
    Ok(())
}

fn validate_q8_parent(
    view: &Q8MatrixView,
    logical_shape: [usize; 2],
    parent_rows: usize,
) -> Result<()> {
    let [logical_rows, k] = logical_shape;
    let code_bytes = parent_rows
        .checked_mul(k)
        .context("native Q8 code extent overflows")?;
    let scale_count = parent_rows
        .checked_mul(k / 32)
        .context("native Q8 scale count overflows")?;
    let scale_bytes = scale_count
        .checked_mul(2)
        .context("native Q8 scale extent overflows")?;
    let scale_offset = code_bytes
        .checked_add((256 - code_bytes % 256) % 256)
        .context("native Q8 scale offset overflows")?;
    ensure!(
        view.shape == logical_shape && view.padded_k == k && view.group_size == 32,
        "native Q8 logical view geometry mismatch"
    );
    ensure!(
        view.source_rows.len() == logical_rows
            && view.source_rows.iter().all(|row| *row < parent_rows),
        "native Q8 logical row map exceeds physical parent"
    );
    ensure!(
        view.codes.offset == 0
            && view.codes.bytes == u64::try_from(code_bytes)?
            && view.scale_bits.offset == u64::try_from(scale_offset)?
            && view.scale_bits.bytes == u64::try_from(scale_bytes)?
            && view.scale_count == scale_count,
        "native Q8 logical view does not retain the complete physical planes"
    );
    Ok(())
}

fn validate_interleaved_q_gate(
    view: &Q8MatrixView,
    query: std::ops::Range<usize>,
    gate: std::ops::Range<usize>,
) -> Result<()> {
    ensure!(
        view.source_rows == interleaved_q_gate_rows(query, gate),
        "native MTP interleaved Q/gate row mapping mismatch"
    );
    Ok(())
}

pub(super) fn interleaved_q_gate_rows(
    query: std::ops::Range<usize>,
    gate: std::ops::Range<usize>,
) -> Vec<usize> {
    let mut expected = Vec::with_capacity(12_288);
    for head in 0..QUERY_HEADS {
        expected.extend(query.start + head * 256..query.start + (head + 1) * 256);
        expected.extend(gate.start + head * 256..gate.start + (head + 1) * 256);
    }
    expected
}

#[cfg(test)]
mod tests {
    use super::{logical_gate_up_rows, physical_qkv_rows};

    #[test]
    fn physical_qkv_offsets_match_packed_parent_contract() {
        let [query, key, gate, value] = physical_qkv_rows();
        assert_eq!(query, 0..6_144);
        assert_eq!(key, 6_144..7_168);
        assert_eq!(gate, 7_168..13_312);
        assert_eq!(value, 13_312..14_336);
    }

    #[test]
    fn physical_gate_up_halves_match_packed_parent_contract() {
        let [gate, up] = logical_gate_up_rows();
        assert_eq!(gate, 0..17_408);
        assert_eq!(up, 17_408..34_816);
    }
}
