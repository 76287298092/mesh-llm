use super::driver::{Buffer, Context};
use crate::{
    attention_warp_reference::{Fixture, QUERY_ELEMENTS},
    kernels::attention_staged_plan::Plan,
};
use anyhow::Result;

pub(super) const GUARD: usize = 64;

pub(super) struct Outputs<'ctx> {
    pub(super) bf16: Buffer<'ctx>,
    pub(super) raw: Buffer<'ctx>,
}

pub(super) struct Buffers<'ctx> {
    pub(super) q: Buffer<'ctx>,
    pub(super) k: Buffer<'ctx>,
    pub(super) v: Buffer<'ctx>,
    pub(super) workspace: Buffer<'ctx>,
    pub(super) legacy_workspace: Buffer<'ctx>,
    pub(super) legacy: Outputs<'ctx>,
    pub(super) candidate: Outputs<'ctx>,
}

pub(super) struct Snapshot {
    pub(super) bf16: Vec<u16>,
    pub(super) raw: Vec<f32>,
    guards: bool,
}

pub(super) fn bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

pub(super) fn download(buffer: &Buffer<'_>) -> Result<Vec<u8>> {
    let mut bytes = vec![0; buffer.len()];
    buffer.download(&mut bytes)?;
    Ok(bytes)
}

fn upload<'ctx>(context: &'ctx Context, words: &[u16]) -> Result<Buffer<'ctx>> {
    let bytes = bytes(words);
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(&bytes)?;
    Ok(buffer)
}

fn guarded(context: &Context, size: usize) -> Result<Buffer<'_>> {
    let buffer = Buffer::new(context, size + 2 * GUARD)?;
    buffer.upload(&vec![0xa5; buffer.len()])?;
    Ok(buffer)
}

fn outputs(context: &Context) -> Result<Outputs<'_>> {
    Ok(Outputs {
        bf16: guarded(context, QUERY_ELEMENTS * 2)?,
        raw: guarded(context, QUERY_ELEMENTS * 4)?,
    })
}

pub(super) fn workspace_poison(allocation_bytes: usize) -> Vec<u8> {
    let mut bytes = vec![0xa5; allocation_bytes];
    for word in bytes[GUARD..allocation_bytes - GUARD]
        .as_chunks_mut::<8>()
        .0
    {
        word.copy_from_slice(&0x7ff8_a5a5_a5a5_a5a5_u64.to_le_bytes());
    }
    bytes
}

pub(super) fn buffers<'ctx>(
    context: &'ctx Context,
    fixture: &Fixture,
    plan: Plan,
) -> Result<Buffers<'ctx>> {
    let workspace = guarded(context, plan.workspace_bytes)?;
    workspace.upload(&workspace_poison(workspace.len()))?;
    let legacy_workspace = guarded(
        context,
        Plan::new([1, 24, 4, 256, fixture.past, fixture.capacity])?.workspace_bytes,
    )?;
    Ok(Buffers {
        q: upload(context, &fixture.q)?,
        k: upload(context, &fixture.k)?,
        v: upload(context, &fixture.v)?,
        workspace,
        legacy_workspace,
        legacy: outputs(context)?,
        candidate: outputs(context)?,
    })
}

pub(super) fn guards(bytes: &[u8]) -> bool {
    bytes[..GUARD]
        .iter()
        .chain(&bytes[bytes.len() - GUARD..])
        .all(|&byte| byte == 0xa5)
}

pub(super) fn snapshot(outputs: &Outputs<'_>) -> Result<Snapshot> {
    let bf16 = download(&outputs.bf16)?;
    let raw = download(&outputs.raw)?;
    Ok(Snapshot {
        guards: guards(&bf16) && guards(&raw),
        bf16: bf16[GUARD..bf16.len() - GUARD]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|word| u16::from_le_bytes(*word))
            .collect(),
        raw: raw[GUARD..raw.len() - GUARD]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect(),
    })
}

pub(super) fn compare(actual: &Snapshot, control: &Snapshot) -> serde_json::Value {
    let raw = actual
        .raw
        .iter()
        .zip(&control.raw)
        .filter(|(actual, control)| actual.to_bits() != control.to_bits())
        .count();
    let bf16 = actual
        .bf16
        .iter()
        .zip(&control.bf16)
        .filter(|(actual, control)| actual != control)
        .count();
    let extent = actual.raw.len() == QUERY_ELEMENTS
        && control.raw.len() == QUERY_ELEMENTS
        && actual.bf16.len() == QUERY_ELEMENTS
        && control.bf16.len() == QUERY_ELEMENTS;
    serde_json::json!({"all_passed":extent&&raw==0&&bf16==0&&actual.guards&&control.guards,
        "raw_fp32_bit_differences":raw,"bf16_bit_differences":bf16,"extent_match":extent,
        "candidate_guards_intact":actual.guards,"control_guards_intact":control.guards})
}

pub(super) fn workspace_check(plan: Plan, current: &[u8], before: &[u8]) -> serde_json::Value {
    let mut stale = 0;
    let mut invalid = 0;
    for (index, (actual, old)) in current[GUARD..current.len() - GUARD]
        .as_chunks::<8>()
        .0
        .iter()
        .zip(before[GUARD..before.len() - GUARD].as_chunks::<8>().0)
        .enumerate()
    {
        if plan.initialized(index) {
            let value = f64::from_le_bytes(*actual);
            if !value.is_finite()
                || (index >= plan.head_elements
                    && index < 3 * plan.head_elements
                    && !(0.0..=1.0).contains(&value))
                || (index >= 3 * plan.head_elements
                    && index < 3 * plan.head_elements + 24
                    && !(0.0..=plan.length as f64).contains(&value))
            {
                invalid += 1;
            }
            if index >= 3 * plan.head_elements
                && index < 3 * plan.head_elements + 24
                && value == 0.0
            {
                invalid += 1;
            }
        } else if actual != old {
            stale += 1;
        }
    }
    serde_json::json!({"all_passed":guards(current)&&stale==0&&invalid==0,
        "guards_intact":guards(current),"uninitialized_suffix_writes":stale,"invalid_initialized_values":invalid})
}
