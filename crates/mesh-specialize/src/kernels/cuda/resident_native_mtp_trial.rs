use super::{
    driver::Context,
    resident_native_mtp::{NativeMtpParentBinding, ResidentNativeMtp},
};
use crate::packages::qwen3_8_27b::{native_mtp_views::BytePlane, native_source::NativeModelSource};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

const READBACK_BYTES: usize = 1024 * 1024;

pub(in crate::kernels) fn run(artifact: &Path, device: i32) -> Result<Value> {
    let started = Instant::now();
    let mut source = NativeModelSource::open(artifact)?;
    source.native_mtp_views()?;
    let source_seconds = started.elapsed().as_secs_f64();
    let context_started = Instant::now();
    let context = Context::new(device)?;
    let context_seconds = context_started.elapsed().as_secs_f64();
    let memory_before = context.memory()?;
    let load_started = Instant::now();
    let resident = ResidentNativeMtp::load(&context, &mut source)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    ensure!(
        resident.belongs_to(&context),
        "native MTP arena belongs to another context"
    );
    ensure!(
        std::ptr::eq(resident.context(), &context),
        "native MTP context identity mismatch"
    );
    let memory_resident = context.memory()?;
    ensure!(
        resident.layout().regions.len() == 14,
        "native MTP must contain 14 physical parents"
    );
    let binding_started = Instant::now();
    let bindings = check_bindings(&resident)?;
    let binding_seconds = binding_started.elapsed().as_secs_f64();
    let readback_started = Instant::now();
    let mut scratch = Vec::new();
    scratch.try_reserve_exact(READBACK_BYTES)?;
    scratch.resize(READBACK_BYTES, 0);
    let mut parents = Vec::with_capacity(resident.layout().regions.len());
    let mut physical_bytes = 0_u64;
    let mut all_passed = true;
    let mut arena_base = None;
    for region in &resident.layout().regions {
        let parent = resident.parent(&region.name)?;
        ensure!(
            parent.object_id() == region.name && parent.bytes() == region.length,
            "native MTP parent binding metadata mismatch"
        );
        let pointer = parent.pointer()?;
        let base = pointer
            .checked_sub(region.offset)
            .context("native MTP arena base underflow")?;
        let expected_base = *arena_base.get_or_insert(base);
        ensure!(
            base == expected_base,
            "native MTP parent pointer has wrong arena offset"
        );
        let parent_started = Instant::now();
        let sha256 = hash_parent(&parent, &mut scratch)?;
        let matches = sha256 == parent.sha256();
        all_passed &= matches;
        physical_bytes = physical_bytes
            .checked_add(parent.bytes())
            .context("native MTP physical byte total overflow")?;
        parents.push(json!({
            "physical_object_id": parent.object_id(), "bytes": parent.bytes(),
            "arena_offset": region.offset, "pointer": pointer,
            "expected_sha256": parent.sha256(), "sha256": sha256, "matches": matches,
            "readback_seconds": parent_started.elapsed().as_secs_f64(),
        }));
    }
    let readback_seconds = readback_started.elapsed().as_secs_f64();
    let arena_bytes = resident.layout().bytes;
    let padding_bytes = arena_bytes
        .checked_sub(physical_bytes)
        .context("native MTP physical bytes exceed arena size")?;
    drop(resident);
    let memory_after_drop = context.memory()?;
    Ok(json!({
        "schema_version": 1, "kind": "native-mtp-gpu-residency-qualification",
        "all_passed": all_passed, "gpu_residency": true, "operator_execution": false,
        "native_mtp_admitted": false, "model_executable": false,
        "dense_dequantization": false, "text_tensors_loaded": false,
        "device": context.info(), "source_verification": source.verification_report(),
        "physical_parent_count": parents.len(), "physical_parent_bytes": physical_bytes,
        "aligned_arena_bytes": arena_bytes, "alignment_padding_bytes": padding_bytes,
        "readback_chunk_bytes": READBACK_BYTES, "host_readback_scratch_bytes": scratch.len(),
        "logical_q8_views": 8, "logical_q4_views": 1,
        "bindings": bindings, "parent_hashes": parents,
        "timings_seconds": {
            "source_verification": source_seconds, "context_creation": context_seconds,
            "load": load_seconds, "binding_checks": binding_seconds,
            "readback": readback_seconds, "total": started.elapsed().as_secs_f64(),
        },
        "memory_bytes": {
            "before_load": {"free": memory_before.0, "total": memory_before.1},
            "resident": {"free": memory_resident.0, "total": memory_resident.1},
            "after_drop": {"free": memory_after_drop.0, "total": memory_after_drop.1},
        },
    }))
}

fn hash_parent(parent: &NativeMtpParentBinding<'_, '_>, scratch: &mut [u8]) -> Result<String> {
    let mut digest = Sha256::new();
    let mut offset = 0_u64;
    while offset < parent.bytes() {
        let length = usize::try_from((parent.bytes() - offset).min(u64::try_from(scratch.len())?))?;
        let chunk = &mut scratch[..length];
        parent.read_range(offset, chunk)?;
        digest.update(&*chunk);
        offset = offset
            .checked_add(u64::try_from(length)?)
            .context("native MTP readback position overflow")?;
    }
    Ok(hex::encode(digest.finalize()))
}

fn check_bindings(resident: &ResidentNativeMtp<'_>) -> Result<Vec<Value>> {
    let views = resident.views();
    let mut reports = Vec::with_capacity(9);
    for (role, view) in [
        ("fc", &views.fc),
        ("query_gate", &views.query_gate),
        ("key", &views.key),
        ("value", &views.value),
        ("attention_output", &views.attention_output),
        ("mlp_gate", &views.mlp_gate),
        ("mlp_up", &views.mlp_up),
        ("mlp_down", &views.mlp_down),
    ] {
        let binding = resident.q8(view)?;
        ensure!(
            std::ptr::eq(binding.view(), view),
            "Q8 binding did not retain saved view"
        );
        ensure!(
            binding.parent().object_id() == view.object_id,
            "Q8 binding has wrong parent"
        );
        let pointers = [binding.codes_pointer()?, binding.scale_bits_pointer()?];
        check_planes(binding.parent(), [&view.codes, &view.scale_bits], pointers)?;
        let mut foreign = view.clone();
        foreign.source_rows.reverse();
        ensure!(
            foreign != *view,
            "Q8 membership probe did not change row mapping"
        );
        ensure!(
            resident.q8(&foreign).is_err(),
            "Q8 binding accepted foreign row mapping"
        );
        reports.push(json!({
            "role": role, "format": "Q8_g32_FP16", "physical_object_id": view.object_id,
            "shape": binding.view().shape, "codes_pointer": pointers[0],
            "scale_bits_pointer": pointers[1], "saved_view_matches": true,
            "foreign_view_rejected": true, "parent_offsets_match": true,
        }));
    }
    let view = &views.proposal_head;
    let binding = resident.q4(view)?;
    ensure!(
        std::ptr::eq(binding.view(), view),
        "Q4 binding did not retain saved view"
    );
    ensure!(
        binding.parent().object_id() == view.object_id,
        "Q4 binding has wrong parent"
    );
    let pointers = [binding.codes_pointer()?, binding.scale_bits_pointer()?];
    check_planes(binding.parent(), [&view.codes, &view.scale_bits], pointers)?;
    let mut foreign = view.clone();
    foreign.source_rows.reverse();
    ensure!(
        foreign != *view,
        "Q4 membership probe did not change row mapping"
    );
    ensure!(
        resident.q4(&foreign).is_err(),
        "Q4 binding accepted foreign row mapping"
    );
    reports.push(json!({
        "role": "proposal_head", "format": "Q4_g64_FP16", "physical_object_id": view.object_id,
        "shape": binding.view().shape, "codes_pointer": pointers[0],
        "scale_bits_pointer": pointers[1], "saved_view_matches": true,
        "foreign_view_rejected": true, "parent_offsets_match": true,
    }));
    Ok(reports)
}

fn check_planes(
    parent: &NativeMtpParentBinding<'_, '_>,
    planes: [&BytePlane; 2],
    pointers: [u64; 2],
) -> Result<()> {
    for (plane, pointer) in planes.into_iter().zip(pointers) {
        let end = plane
            .offset
            .checked_add(plane.bytes)
            .context("native MTP plane end overflow")?;
        ensure!(
            end <= parent.bytes(),
            "native MTP plane exceeds parent extent"
        );
        let expected = parent
            .pointer()?
            .checked_add(plane.offset)
            .context("native MTP plane pointer overflow")?;
        ensure!(
            pointer == expected,
            "native MTP plane pointer has wrong parent offset"
        );
    }
    Ok(())
}
