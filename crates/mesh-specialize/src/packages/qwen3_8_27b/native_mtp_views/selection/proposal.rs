use super::super::{ProposalTokenMap, Q4MatrixView, TargetTokenId};
use super::binding::{MatrixSelection, selected_rows};
use super::inventory::target_head_object;
use super::plane_layout::packed_planes;
use crate::artifact::ninfer::{Binding, Directory, Object};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

pub(super) const TOKEN_MAP_BYTES: u64 = 524_288;
const TARGET_VOCABULARY: i32 = 248_320;
const PROPOSAL_ROWS: usize = 131_072;

pub(super) fn q4_projection(directory: &Directory) -> Result<Q4MatrixView> {
    let selected = selected_rows(
        directory,
        MatrixSelection {
            binding: "proposal/head",
            shape: [PROPOSAL_ROWS, 5_120],
        },
    )?;
    ensure!(
        selected.object.format.as_deref() == Some("q4_g64_fp16"),
        "proposal head format is unsupported"
    );
    let (padded_k, group_size, codes, scale_bits, scale_count) = packed_planes(selected.object)?;
    ensure!(group_size == 64, "proposal-head group width mismatch");
    ensure!(
        u64::try_from(scale_count)?
            .checked_mul(2)
            .is_some_and(|expected_bytes| expected_bytes == scale_bits.bytes),
        "proposal-head FP16 scale byte count mismatch"
    );
    let mut source_rows = Vec::new();
    source_rows
        .try_reserve_exact(PROPOSAL_ROWS)
        .context("cannot reserve proposal head row map")?;
    source_rows.extend(selected.rows);
    ensure!(
        source_rows.len() == PROPOSAL_ROWS,
        "native Q4 row map extent mismatch"
    );
    ensure!(
        source_rows.iter().copied().collect::<BTreeSet<_>>().len() == source_rows.len(),
        "native proposal head row map repeats parent rows"
    );
    Ok(Q4MatrixView {
        object_id: selected.object.id.clone(),
        shape: [PROPOSAL_ROWS, 5_120],
        padded_k,
        group_size,
        codes,
        scale_bits,
        scale_count,
        source_rows,
    })
}

pub(super) fn select_token_map(directory: &Directory) -> Result<String> {
    let binding = directory
        .bindings
        .get("proposal/token_ids")
        .context("native proposal token-map binding is missing")?;
    let Binding::Object { object: object_id } = binding else {
        anyhow::bail!("native proposal token map must bind one complete object")
    };
    let object = object_by_id(directory, object_id)?;
    object.require_supported_encoding()?;
    ensure!(
        object.format.as_deref() == Some("int32")
            && object.layout.as_deref() == Some("contiguous_le_v1")
            && object.shape.as_slice() == [u64::try_from(PROPOSAL_ROWS)?]
            && object.bytes == TOKEN_MAP_BYTES,
        "native proposal token map must be signed contiguous INT32[131072]"
    );
    Ok(object.id.clone())
}

pub(super) fn parse_token_map(bytes: &[u8]) -> Result<ProposalTokenMap> {
    ensure!(
        bytes.len() == usize::try_from(TOKEN_MAP_BYTES)?,
        "native proposal token map byte length mismatch"
    );
    let mut target_ids = Vec::new();
    target_ids
        .try_reserve_exact(PROPOSAL_ROWS)
        .context("cannot reserve native proposal token IDs")?;
    let (words, remainder) = bytes.as_chunks::<4>();
    ensure!(
        remainder.is_empty(),
        "native proposal token map has a partial INT32"
    );
    for (row, word) in words.iter().enumerate() {
        let signed_id = i32::from_le_bytes(*word);
        ensure!(
            (0..TARGET_VOCABULARY).contains(&signed_id),
            "native proposal row {row} maps outside target vocabulary"
        );
        target_ids.push(TargetTokenId(u32::try_from(signed_id)?));
    }
    ensure!(
        target_ids.len() == PROPOSAL_ROWS,
        "native proposal token map row count mismatch"
    );
    Ok(ProposalTokenMap { target_ids })
}

pub(super) fn validate_distinct_target_head(
    directory: &Directory,
    proposal_head: &Q4MatrixView,
) -> Result<()> {
    let target = target_head_object(directory)?;
    ensure!(
        target.format.as_deref() == Some("fp8_e4m3fn_row_bf16")
            && target.layout.as_deref() == Some("row_scale_v1")
            && target.shape.as_slice() == [248_320, 5_120],
        "native target output head representation mismatch"
    );
    ensure!(
        target.id != proposal_head.object_id,
        "proposal shortlist must not replace the target output head"
    );
    Ok(())
}

fn object_by_id<'a>(directory: &'a Directory, object_id: &str) -> Result<&'a Object> {
    directory
        .objects
        .iter()
        .find(|object| object.id == object_id)
        .with_context(|| format!("native object {object_id} is missing"))
}
