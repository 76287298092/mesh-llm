pub(super) mod compare;
pub(in crate::kernels) mod driver_entry;
pub(super) mod fixture;
pub(super) mod launch;
pub(in crate::kernels::cuda) mod resident;
pub(super) mod schedule_reference;
pub(super) mod validate;

pub(in crate::kernels) fn run_resident(
    artifact: &std::path::Path,
    ptx: &str,
    device: i32,
) -> anyhow::Result<serde_json::Value> {
    resident::run(artifact, ptx, device)
}

#[cfg(test)]
#[path = "native_mtp_q4_operator/tests.rs"]
mod tests;
