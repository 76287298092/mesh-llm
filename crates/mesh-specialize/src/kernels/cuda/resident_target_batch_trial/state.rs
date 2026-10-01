use super::super::resident_state::ResidentState;
use anyhow::{Context as _, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

const STATE_CHUNK_BYTES: usize = 1024 * 1024;

#[derive(Clone, Serialize)]
pub(in crate::kernels::cuda) struct RegionComparison {
    pub name: String,
    pub length: u64,
    pub identical: bool,
    pub differing_bytes: u64,
    pub first_difference: Option<u64>,
    pub left_sha256: String,
    pub right_sha256: String,
}

#[derive(Clone, Serialize)]
pub(in crate::kernels::cuda) struct StateComparison {
    pub left_aggregate_sha256: String,
    pub right_aggregate_sha256: String,
    pub regions: Vec<RegionComparison>,
    pub first_differing_region: Option<String>,
    pub equal: bool,
}

pub(in crate::kernels::cuda) fn compare_states(
    left: &ResidentState<'_>,
    right: &ResidentState<'_>,
) -> Result<StateComparison> {
    let mut left_aggregate = Sha256::new();
    let mut right_aggregate = Sha256::new();
    let mut regions = Vec::with_capacity(left.layout().regions.len());
    for left_region in &left.layout().regions {
        let right_region = right.layout().region(&left_region.name)?;
        let (comparison, left_digest, right_digest) = compare_region(
            left,
            right,
            &left_region.name,
            left_region.length,
            right_region.length,
        )?;
        hash_metadata(&mut left_aggregate, &left_region.name, left_region.length)?;
        hash_metadata(&mut right_aggregate, &left_region.name, right_region.length)?;
        left_aggregate.update(left_digest);
        right_aggregate.update(right_digest);
        regions.push(comparison);
    }
    Ok(aggregate_state(
        regions,
        left.layout() == right.layout(),
        (
            hex::encode(left_aggregate.finalize()),
            hex::encode(right_aggregate.finalize()),
        ),
    ))
}

fn compare_region(
    left: &ResidentState<'_>,
    right: &ResidentState<'_>,
    name: &str,
    left_length: u64,
    right_length: u64,
) -> Result<(RegionComparison, [u8; 32], [u8; 32])> {
    let mut left_hash = Sha256::new();
    let mut right_hash = Sha256::new();
    let length = left_length.min(right_length);
    let mut left_buffer = vec![0_u8; STATE_CHUNK_BYTES];
    let mut right_buffer = vec![0_u8; STATE_CHUNK_BYTES];
    let mut offset = 0_usize;
    let mut differing_bytes = 0_u64;
    let mut first_difference = None;
    while u64::try_from(offset)? < length {
        let remaining = usize::try_from(length - u64::try_from(offset)?)?;
        let chunk_length = remaining.min(STATE_CHUNK_BYTES);
        let left_chunk = &mut left_buffer[..chunk_length];
        let right_chunk = &mut right_buffer[..chunk_length];
        left.read_region_at(name, offset, left_chunk)?;
        right.read_region_at(name, offset, right_chunk)?;
        left_hash.update(&*left_chunk);
        right_hash.update(&*right_chunk);
        let (chunk_differences, chunk_first) = byte_differences(left_chunk, right_chunk)?;
        differing_bytes = differing_bytes
            .checked_add(chunk_differences)
            .context("state difference count overflows u64")?;
        if let Some(index) = chunk_first {
            first_difference.get_or_insert(
                u64::try_from(offset)?
                    .checked_add(index)
                    .context("state difference offset overflows u64")?,
            );
        }
        offset = offset
            .checked_add(chunk_length)
            .context("state comparison offset overflows usize")?;
    }
    let left_digest: [u8; 32] = left_hash.finalize().into();
    let right_digest: [u8; 32] = right_hash.finalize().into();
    let mut left_region_hash = Sha256::new();
    let mut right_region_hash = Sha256::new();
    hash_metadata(&mut left_region_hash, name, left_length)?;
    hash_metadata(&mut right_region_hash, name, right_length)?;
    left_region_hash.update(left_digest);
    right_region_hash.update(right_digest);
    let left_region_digest: [u8; 32] = left_region_hash.finalize().into();
    let right_region_digest: [u8; 32] = right_region_hash.finalize().into();
    let length_equal = left_length == right_length;
    if !length_equal && first_difference.is_none() {
        first_difference = Some(length);
    }
    Ok((
        RegionComparison {
            name: name.to_owned(),
            length: left_length,
            identical: length_equal && differing_bytes == 0,
            differing_bytes: differing_bytes
                .checked_add(left_length.abs_diff(right_length))
                .context("state difference count overflows u64")?,
            first_difference,
            left_sha256: hex::encode(left_region_digest),
            right_sha256: hex::encode(right_region_digest),
        },
        left_digest,
        right_digest,
    ))
}

pub(in crate::kernels::cuda) fn aggregate_state(
    regions: Vec<RegionComparison>,
    layouts_equal: bool,
    hashes: (String, String),
) -> StateComparison {
    let equal =
        layouts_equal && regions.iter().all(|region| region.identical) && hashes.0 == hashes.1;
    let first_differing_region = regions
        .iter()
        .find(|region| !region.identical)
        .map(|region| region.name.clone());
    StateComparison {
        left_aggregate_sha256: hashes.0,
        right_aggregate_sha256: hashes.1,
        regions,
        first_differing_region,
        equal,
    }
}

fn hash_metadata(hash: &mut Sha256, name: &str, length: u64) -> Result<()> {
    hash.update(u64::try_from(name.len())?.to_le_bytes());
    hash.update(name.as_bytes());
    hash.update(length.to_le_bytes());
    Ok(())
}

fn byte_differences(left: &[u8], right: &[u8]) -> Result<(u64, Option<u64>)> {
    let mut differing_bytes = 0_u64;
    let mut first_difference = None;
    for (index, (left_byte, right_byte)) in left.iter().zip(right).enumerate() {
        if left_byte != right_byte {
            differing_bytes = differing_bytes
                .checked_add(1)
                .context("state difference count overflows u64")?;
            first_difference.get_or_insert(u64::try_from(index)?);
        }
    }
    if left.len() != right.len() && first_difference.is_none() {
        first_difference = Some(u64::try_from(left.len().min(right.len()))?);
    }
    Ok((
        differing_bytes
            .checked_add(u64::try_from(left.len().abs_diff(right.len()))?)
            .context("state difference count overflows u64")?,
        first_difference,
    ))
}

#[cfg(test)]
pub(in crate::kernels::cuda) fn compare_region_bytes(
    name: &str,
    left: &[u8],
    right: &[u8],
) -> Result<RegionComparison> {
    let (differing_bytes, first_difference) = byte_differences(left, right)?;
    Ok(RegionComparison {
        name: name.to_owned(),
        length: u64::try_from(left.len())?,
        identical: left.len() == right.len() && differing_bytes == 0,
        differing_bytes,
        first_difference,
        left_sha256: hex::encode(Sha256::digest(left)),
        right_sha256: hex::encode(Sha256::digest(right)),
    })
}
