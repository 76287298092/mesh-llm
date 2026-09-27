//! Bounded full-weight transfer and persistent-state qualification.

use super::{
    driver::{Context, Module},
    residency_entry,
    resident_state::ResidentState,
    resident_weights::ResidentWeights,
};
use crate::{
    artifact::{reader::VerifiedArtifact, schema::Object},
    engine::layout::Layout,
    kernels::ResidentEntryInput,
};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};

const WORKSPACE_RESERVE: u64 = 1024 * 1024 * 1024;

pub(in crate::kernels) fn run(
    ptx: &str,
    device: i32,
    artifact: &mut VerifiedArtifact,
    objects: &[Object],
    state_layout: &Layout,
    entry: &ResidentEntryInput,
) -> Result<Value> {
    ensure!(
        ptx.contains(".target sm_120a"),
        "resident trial requires SM120a PTX"
    );
    let weight_layout = Layout::new(
        objects
            .iter()
            .map(|object| (object.name.clone(), object.length)),
    )?;
    let checked_states = Layout::new(
        state_layout
            .regions
            .iter()
            .map(|region| (region.name.clone(), region.length)),
    )?;
    ensure!(&checked_states == state_layout, "noncanonical state layout");
    let required = weight_layout
        .bytes
        .checked_add(state_layout.bytes)
        .and_then(|bytes| bytes.checked_add(WORKSPACE_RESERVE))
        .context("resident admission extent overflow")?;
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "resident trial requires SM120"
    );
    let before_module = context.memory()?;
    let module = Module::load(&context, ptx)?;
    module.function("embedding_norm_bf16")?;
    let after_module = context.memory()?;
    ensure!(
        required <= u64::try_from(after_module.0)?,
        "resident weights/state/reserve require {required} bytes; only {} free",
        after_module.0
    );
    let weights = ResidentWeights::load(&context, artifact, objects)?;
    let after_weights = context.memory()?;
    let weight_checks = weights.verify()?;
    let state = ResidentState::new(&context, state_layout)?;
    let after_states = context.memory()?;
    let state_check = state.verify_zero()?;
    let entry_check = residency_entry::run(&context, &module, &weights, objects, entry)?;
    context.synchronize()?;
    let after_entry = context.memory()?;
    let weight_bytes = weights.layout().bytes;
    let state_bytes = state.layout().bytes;
    drop(state);
    drop(weights);
    context.synchronize()?;
    let after_free = context.memory()?;
    let released = after_free.0 >= after_module.0;
    let passed = weight_checks.iter().all(|check| check["matches"] == true)
        && state_check["passed"] == true
        && entry_check["passed"] == true
        && released;
    Ok(json!({
        "schema_version":1,"kind":"qwen-persistent-text-residency-trial",
        "device":info,"jit_log":module.jit_log(),"all_passed":passed,
        "weight_objects":weight_checks,"state_check":state_check,"entry_check":entry_check,
        "weight_arena_bytes":weight_bytes,"state_arena_bytes":state_bytes,
        "workspace_reserve_bytes":WORKSPACE_RESERVE,"workspace_reserve_allocated":false,
        "admission_required_bytes":required,"arena_allocations_released":released,
        "memory_before_module":memory(before_module),"memory_after_module":memory(after_module),
        "memory_after_weights":memory(after_weights),"memory_after_states":memory(after_states),
        "memory_after_entry":memory(after_entry),"memory_after_free":memory(after_free),
        "memory_note":"CUDA allocation checkpoints; not measured inference peak",
        "first_entry_executed":true,"full_model_executed":false,"timing_collected":false,
    }))
}

fn memory((free, total): (usize, usize)) -> Value {
    json!({"free_bytes":free,"total_bytes":total})
}
