use crate::packages::qwen3_8_27b::native_mtp_views::BytePlane;
use anyhow::{Context, Result, ensure};

pub(super) fn parent_range(parent_bytes: u64, plane: &BytePlane) -> Result<()> {
    let end = plane
        .offset
        .checked_add(plane.bytes)
        .context("native MTP parent-relative range overflow")?;
    ensure!(
        end <= parent_bytes,
        "native MTP plane exceeds physical parent"
    );
    Ok(())
}

pub(super) fn device_pointer(base: u64, offset: u64) -> Result<u64> {
    base.checked_add(offset)
        .context("native MTP device pointer overflow")
}

pub(super) fn saved_view<'owner, View: PartialEq + 'owner>(
    saved: impl IntoIterator<Item = &'owner View>,
    requested: &View,
) -> Result<&'owner View> {
    saved
        .into_iter()
        .find(|view| *view == requested)
        .context("native MTP view is not a saved verified selection")
}
