use super::super::{
    driver::{Context, Module},
    resident_native_mtp::ResidentNativeMtp,
};
use crate::{
    native_mtp_q4_gemv_reference, packages::qwen3_8_27b::native_source::NativeModelSource,
};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};
use std::path::Path;

mod comparison;
mod input;
pub(in crate::kernels::cuda) mod launch;
mod mapping;
mod parent;

#[cfg(test)]
#[path = "resident/tests.rs"]
mod tests;

const PROPOSAL_ROWS: usize = 131_072;
const TARGET_VOCABULARY: u32 = 248_320;
const GROUP_SIZE: usize = 64;
const LOGICAL_K: usize = 5_120;
const EXPECTED_PARENT_BYTES: usize = 356_515_840;
const HASHED_PARENT_BYTES: u64 = 356_515_840;

pub(in crate::kernels) fn run(artifact: &Path, ptx: &str, device: i32) -> Result<Value> {
    ensure!(
        ptx.contains(".target sm_120a"),
        "native Q4 resident check requires SM120a PTX"
    );
    let context = Context::new(device)?;
    let device_info = context.info();
    ensure!(
        (device_info.major, device_info.minor) == (12, 0),
        "native Q4 resident check requires SM120"
    );
    let module = Module::load(&context, ptx)?;
    let function = module.function("native_mtp_q4_head_gemv")?;
    let mut source = NativeModelSource::open(artifact)?;
    let resident = ResidentNativeMtp::load(&context, &mut source)?;
    ensure!(
        resident.belongs_to(&context),
        "native MTP arena context mismatch"
    );
    let views = resident.views();
    let view = &views.proposal_head;
    ensure!(
        std::ptr::eq(resident.context(), &context),
        "native MTP residency context identity mismatch"
    );
    validate_head(view, resident.layout().region(&view.object_id)?.length)?;
    let binding = resident.q4(view)?;

    let (source_parent, source_parent_hash) =
        parent::copy_source_parent(&mut source, binding.parent())?;
    let expected_parent_hash = binding.parent().sha256().to_owned();
    let source_parent_hash_matches = source_parent_hash == expected_parent_hash;
    let resident_parent_hash = parent::hash_resident_parent(&binding, HASHED_PARENT_BYTES)?;
    let resident_parent_hash_matches = resident_parent_hash == expected_parent_hash;
    let map = mapping::load(&resident, views)?;
    let [first_input, second_input] = input::dense_cases()?;
    let validated = super::validate::validate(&source_parent, view, &first_input.words)?;
    let mut cases = Vec::new();
    for input in [first_input, second_input] {
        let result = (|| {
            ensure!(
                source_parent_hash_matches && resident_parent_hash_matches,
                "packed Q4 parent hash gate failed"
            );
            ensure!(
                map.source_hash_matches && map.readback_hash_matches && map.readback_bytes_match,
                "signed proposal-map hash gate failed"
            );
            let fp64 = native_mtp_q4_gemv_reference::run(&source_parent, view, &input.words)?;
            let scheduled =
                super::schedule_reference::run(&source_parent, &input.words, &validated)?;
            let first = launch::run_bound(launch::BoundInput {
                context: &context,
                module: &module,
                resident: &resident,
                view,
                input_bf16: &input.words,
                validated: &validated,
                raw_poison: 0x7fc1_2345,
                bf16_poison: 0x7fc1,
            })?;
            let second = launch::run_bound(launch::BoundInput {
                context: &context,
                module: &module,
                resident: &resident,
                view,
                input_bf16: &input.words,
                validated: &validated,
                raw_poison: 0xffc2_6789,
                bf16_poison: 0xffc2,
            })?;
            comparison::evaluate(comparison::Case {
                name: input.name,
                outputs: [first, second],
                references: comparison::References {
                    fp64: &fp64,
                    scheduled: &scheduled,
                    validated: &validated,
                },
                target_ids: &map.target_ids,
            })
        })();
        cases.push(match result {
            Ok(report) => report,
            Err(error) => comparison::failed_case(input.name, &error),
        });
    }
    let hashes_passed = source_parent_hash_matches
        && resident_parent_hash_matches
        && map.source_hash_matches
        && map.readback_hash_matches
        && map.readback_bytes_match;
    Ok(json!({
        "schema_version": 1,
        "kind": "native-mtp-q4-resident-qualification-v1",
        "all_passed": hashes_passed && cases.iter().all(|case| case["all_passed"] == true),
        "device": device_info,
        "jit_log": module.jit_log(),
        "kernel_resources": function.resources()?,
        "source_verification": source.verification_report(),
        "proposal_head": {
            "physical_object_id": view.object_id,
            "shape": view.shape,
            "format": "Q4_g64_FP16",
            "layout": "row_split_k128_v1",
            "group_size": GROUP_SIZE,
            "parent_bytes": binding.parent().bytes(),
            "expected_parent_sha256": expected_parent_hash,
            "source_parent_sha256": source_parent_hash,
            "source_parent_hash_matches": source_parent_hash_matches,
            "resident_readback_sha256": resident_parent_hash,
            "resident_parent_hash_matches": resident_parent_hash_matches,
            "codes": {"offset": view.codes.offset, "bytes": view.codes.bytes},
            "scales": {"offset": view.scale_bits.offset, "bytes": view.scale_bits.bytes},
            "dense_dequantization": false,
        },
        "proposal_map": map.report(),
        "proposal_rows": PROPOSAL_ROWS,
        "target_vocabulary": TARGET_VOCABULARY,
        "fp64_max_scaled_error_limit": comparison::MAX_SCALED_ERROR,
        "cases": cases,
        "native_mtp_admitted": false,
        "model_executable": false,
        "timing_claim": false,
    }))
}

pub(in crate::kernels::cuda) fn validate_head(
    view: &crate::packages::qwen3_8_27b::native_mtp_views::Q4MatrixView,
    parent_bytes: u64,
) -> Result<()> {
    ensure!(
        view.shape == [PROPOSAL_ROWS, LOGICAL_K],
        "native proposal head shape mismatch"
    );
    ensure!(
        view.group_size == GROUP_SIZE,
        "native proposal head group size mismatch"
    );
    ensure!(
        view.padded_k == LOGICAL_K,
        "native proposal head padded K mismatch"
    );
    ensure!(
        view.source_rows.iter().copied().eq(0..PROPOSAL_ROWS),
        "native proposal head row map must be identity"
    );
    let code_bytes = PROPOSAL_ROWS
        .checked_mul(LOGICAL_K / 2)
        .context("native proposal code extent overflows")?;
    let scale_count = PROPOSAL_ROWS
        .checked_mul(LOGICAL_K / GROUP_SIZE)
        .context("native proposal scale count overflows")?;
    let scale_bytes = scale_count
        .checked_mul(2)
        .context("native proposal scale extent overflows")?;
    let scale_offset = code_bytes
        .checked_add((256 - code_bytes % 256) % 256)
        .context("native proposal scale offset overflows")?;
    ensure!(
        view.codes.offset == 0
            && view.codes.bytes == u64::try_from(code_bytes)?
            && view.scale_bits.offset == u64::try_from(scale_offset)?
            && view.scale_bits.bytes == u64::try_from(scale_bytes)?
            && view.scale_count == scale_count,
        "native proposal head packed planes mismatch"
    );
    ensure!(
        parent_bytes == u64::try_from(EXPECTED_PARENT_BYTES)?
            && scale_offset.checked_add(scale_bytes) == Some(EXPECTED_PARENT_BYTES),
        "native proposal parent extent mismatch"
    );
    Ok(())
}
