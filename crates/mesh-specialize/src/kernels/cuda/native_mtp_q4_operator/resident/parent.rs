use super::super::super::resident_native_mtp::{NativeMtpParentBinding, NativeMtpQ4Binding};
use crate::packages::qwen3_8_27b::native_source::NativeModelSource;
use anyhow::{Context as _, Result, ensure};
use sha2::{Digest, Sha256};

const READBACK_BYTES: usize = 1024 * 1024;

pub(super) fn copy_source_parent(
    source: &mut NativeModelSource,
    parent: &NativeMtpParentBinding<'_, '_>,
) -> Result<(Vec<u8>, String)> {
    let parent_bytes = usize::try_from(parent.bytes())?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(parent_bytes)
        .context("cannot reserve packed Q4 CPU reference parent")?;
    let copied = source.copy_native_mtp_parent(parent.object_id(), &mut bytes)?;
    ensure!(
        copied == parent.bytes(),
        "short packed Q4 source parent copy"
    );
    let digest = hex::encode(Sha256::digest(&bytes));
    ensure!(
        digest == parent.sha256(),
        "packed Q4 source parent hash mismatch"
    );
    Ok((bytes, digest))
}

pub(super) fn hash_resident_parent(
    binding: &NativeMtpQ4Binding<'_, '_>,
    maximum_parent_bytes: u64,
) -> Result<String> {
    ensure!(
        binding.parent().pointer()? != 0,
        "resident Q4 parent pointer is null"
    );
    ensure!(
        binding.parent().bytes() <= maximum_parent_bytes,
        "resident Q4 parent exceeds readback bound"
    );
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(READBACK_BYTES)
        .context("cannot reserve packed Q4 readback scratch")?;
    scratch.resize(READBACK_BYTES, 0);
    let mut digest = Sha256::new();
    let mut offset = 0_u64;
    while offset < binding.parent().bytes() {
        let remaining = binding.parent().bytes() - offset;
        let length = usize::try_from(remaining.min(u64::try_from(scratch.len())?))?;
        let chunk = &mut scratch[..length];
        binding.parent().read_range(offset, chunk)?;
        digest.update(&*chunk);
        offset = offset
            .checked_add(u64::try_from(length)?)
            .context("packed Q4 readback offset overflows")?;
    }
    Ok(hex::encode(digest.finalize()))
}
