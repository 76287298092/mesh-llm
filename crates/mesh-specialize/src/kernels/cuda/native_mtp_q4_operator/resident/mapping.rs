use super::super::super::resident_native_mtp::ResidentNativeMtp;
use crate::packages::qwen3_8_27b::native_mtp_views::NativeMtpViews;
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const TARGET_VOCABULARY: u32 = 248_320;
const TOKEN_MAP_BYTES: usize = 131_072 * 4;
const TOKEN_MAP_BYTES_U64: u64 = 524_288;

pub(super) struct ProposalMap {
    pub(super) object_id: String,
    pub(super) expected_sha256: String,
    pub(super) source_parent_sha256: String,
    pub(super) readback_sha256: String,
    pub(super) source_hash_matches: bool,
    pub(super) readback_hash_matches: bool,
    pub(super) readback_bytes_match: bool,
    pub(super) target_ids: Vec<u32>,
    minimum_target_id: u32,
    maximum_target_id: u32,
}

impl ProposalMap {
    pub(super) fn report(&self) -> Value {
        json!({
            "physical_object_id": self.object_id,
            "format": "signed-int32-le",
            "expected_signed_map_byte_sha256": self.expected_sha256,
            "source_parent_sha256": self.source_parent_sha256,
            "resident_signed_map_byte_sha256": self.readback_sha256,
            "source_hash_matches": self.source_hash_matches,
            "resident_hash_matches": self.readback_hash_matches,
            "resident_bytes_match_parsed_map": self.readback_bytes_match,
            "proposal_rows": self.target_ids.len(),
            "signed_map_byte_range": {"offset": 0, "bytes": TOKEN_MAP_BYTES},
            "target_vocabulary": TARGET_VOCABULARY,
            "minimum_target_id": self.minimum_target_id,
            "maximum_target_id": self.maximum_target_id,
        })
    }
}

pub(super) fn load(
    resident: &ResidentNativeMtp<'_>,
    views: &NativeMtpViews,
) -> Result<ProposalMap> {
    ensure!(
        views.proposal_tokens.len() == 131_072,
        "proposal token map row count mismatch"
    );
    let expected_bytes = encode_map(views)?;
    let expected_sha256 = hex::encode(Sha256::digest(&expected_bytes));
    let mut map_regions = resident
        .layout()
        .regions
        .iter()
        .filter(|region| region.length == TOKEN_MAP_BYTES_U64);
    let region = map_regions
        .next()
        .context("signed proposal map parent is absent")?;
    ensure!(
        map_regions.next().is_none(),
        "signed proposal map parent extent is ambiguous"
    );
    let object_id = region.name.clone();
    let parent = resident.parent(&object_id)?;
    ensure!(
        parent.bytes() == TOKEN_MAP_BYTES_U64,
        "signed proposal map parent extent mismatch"
    );
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(TOKEN_MAP_BYTES)
        .context("cannot reserve resident proposal-map verification")?;
    bytes.resize(TOKEN_MAP_BYTES, 0);
    parent.read_range(0, &mut bytes)?;
    let source_parent_sha256 = parent.sha256().to_owned();
    let source_hash_matches = source_parent_sha256 == expected_sha256;
    let readback_sha256 = hex::encode(Sha256::digest(&bytes));
    let readback_hash_matches = readback_sha256 == source_parent_sha256;
    let readback_bytes_match = bytes == expected_bytes;
    let (target_ids, minimum_target_id, maximum_target_id) = parse_signed_map(&expected_bytes)?;
    Ok(ProposalMap {
        object_id,
        expected_sha256,
        source_parent_sha256,
        readback_sha256,
        source_hash_matches,
        readback_hash_matches,
        readback_bytes_match,
        target_ids,
        minimum_target_id,
        maximum_target_id,
    })
}

fn encode_map(views: &NativeMtpViews) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(TOKEN_MAP_BYTES)
        .context("cannot reserve parsed proposal-map bytes")?;
    for row in 0..views.proposal_tokens.len() {
        let id = views
            .proposal_tokens
            .target_id(row)
            .context("parsed proposal map has a missing row")?
            .value();
        ensure!(
            id < TARGET_VOCABULARY,
            "parsed proposal target exceeds vocabulary"
        );
        let signed_id = i32::try_from(id).context("target token does not fit signed map")?;
        bytes.extend_from_slice(&signed_id.to_le_bytes());
    }
    ensure!(
        bytes.len() == TOKEN_MAP_BYTES,
        "encoded proposal map extent mismatch"
    );
    Ok(bytes)
}

fn parse_signed_map(bytes: &[u8]) -> Result<(Vec<u32>, u32, u32)> {
    ensure!(
        bytes.len() == TOKEN_MAP_BYTES,
        "signed proposal map byte length mismatch"
    );
    let (words, remainder) = bytes.as_chunks::<4>();
    ensure!(
        remainder.is_empty(),
        "signed proposal map has a partial INT32"
    );
    let mut target_ids = Vec::new();
    target_ids
        .try_reserve_exact(words.len())
        .context("cannot reserve signed proposal IDs")?;
    let mut minimum = TARGET_VOCABULARY;
    let mut maximum = 0;
    for word in words {
        let signed_id = i32::from_le_bytes(*word);
        ensure!(
            (0..i32::try_from(TARGET_VOCABULARY)?).contains(&signed_id),
            "signed proposal ID is out of target range"
        );
        let target_id = u32::try_from(signed_id)?;
        minimum = minimum.min(target_id);
        maximum = maximum.max(target_id);
        target_ids.push(target_id);
    }
    Ok((target_ids, minimum, maximum))
}

#[cfg(test)]
mod tests {
    use super::{TARGET_VOCABULARY, parse_signed_map};

    #[test]
    fn signed_map_keeps_target_ids_outside_the_shortlist_domain() {
        let mut bytes = vec![0; 131_072 * 4];
        bytes[..4].copy_from_slice(&200_000_i32.to_le_bytes());

        let (target_ids, minimum, maximum) = parse_signed_map(&bytes).expect("valid signed map");

        assert_eq!(target_ids[0], 200_000);
        assert!(target_ids[0] < TARGET_VOCABULARY);
        assert_eq!(minimum, 0);
        assert_eq!(maximum, 200_000);
    }

    #[test]
    fn signed_map_rejects_ids_outside_target_vocabulary() {
        let mut bytes = vec![0; 131_072 * 4];
        bytes[..4].copy_from_slice(&248_320_i32.to_le_bytes());

        let result = parse_signed_map(&bytes);

        assert!(result.is_err());
    }
}
