use crate::artifact::ninfer::{Binding, Directory, Object};
use crate::packages::qwen3_8_27b::native_mtp_views::{
    Bf16NormView, NativeMtpNormViews, Q4MatrixView, Q8MatrixView,
};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

const TARGET_VOCABULARY: u64 = 248_320;
const HIDDEN_SIZE: u64 = 5_120;
const PROPOSAL_ROWS: u64 = 131_072;
const MTP_OBJECTS: usize = 12;
const MTP_BYTES: u64 = 451_267_584;
const PROPOSAL_BYTES: u64 = 357_040_128;

pub(super) struct Projections {
    pub fc: Q8MatrixView,
    pub query_gate: Q8MatrixView,
    pub key: Q8MatrixView,
    pub value: Q8MatrixView,
    pub attention_output: Q8MatrixView,
    pub mlp_gate: Q8MatrixView,
    pub mlp_up: Q8MatrixView,
    pub mlp_down: Q8MatrixView,
}

pub(super) fn validate_component_config(directory: &Directory) -> Result<()> {
    let text = directory
        .components
        .get("text")
        .context("native MTP text component is missing")?;
    ensure!(
        text["config"]["vocab_size"].as_u64() == Some(TARGET_VOCABULARY),
        "native MTP target vocabulary mismatch"
    );
    ensure!(
        text["config"]["hidden_size"].as_u64() == Some(HIDDEN_SIZE),
        "native MTP hidden size mismatch"
    );
    ensure!(
        text["proposal"]["domain"].as_str() == Some("indexed")
            && text["proposal"]["rows"].as_u64() == Some(PROPOSAL_ROWS),
        "native proposal must use the indexed 131072-row domain"
    );
    Ok(())
}

pub(super) fn select_norms(directory: &Directory) -> Result<NativeMtpNormViews> {
    Ok(NativeMtpNormViews {
        embedding: norm(directory, "mtp/embedding_norm", 5_120)?,
        hidden: norm(directory, "mtp/hidden_norm", 5_120)?,
        final_norm: norm(directory, "mtp/final_norm", 5_120)?,
        input: norm(directory, "mtp/layers/0/input_norm", 5_120)?,
        post_attention: norm(directory, "mtp/layers/0/post_attention_norm", 5_120)?,
        query: norm(directory, "mtp/layers/0/attention/query_norm", 256)?,
        key: norm(directory, "mtp/layers/0/attention/key_norm", 256)?,
    })
}

fn norm(directory: &Directory, binding_name: &str, elements: usize) -> Result<Bf16NormView> {
    let binding = directory
        .bindings
        .get(binding_name)
        .with_context(|| format!("missing native MTP norm binding {binding_name}"))?;
    let Binding::Object { object: object_id } = binding else {
        anyhow::bail!("native MTP norm must bind one complete object")
    };
    let object = object_by_id(directory, object_id)?;
    object.require_supported_encoding()?;
    ensure!(
        object.format.as_deref() == Some("bf16")
            && object.layout.as_deref() == Some("contiguous_le_v1")
            && object.shape.as_slice() == [u64::try_from(elements)?]
            && object.bytes
                == u64::try_from(elements.checked_mul(2).context("norm size overflow")?)?,
        "native MTP norm representation mismatch"
    );
    Ok(Bf16NormView {
        object_id: object.id.clone(),
        elements,
        bytes: object.bytes,
    })
}

pub(super) fn validate_mtp_inventory(
    directory: &Directory,
    projections: &Projections,
    norms: &NativeMtpNormViews,
) -> Result<()> {
    const EXPECTED_BINDINGS: &[&str] = &[
        "mtp/input_projection",
        "mtp/layers/0/attention/query",
        "mtp/layers/0/attention/key",
        "mtp/layers/0/attention/gate",
        "mtp/layers/0/attention/value",
        "mtp/layers/0/attention/output",
        "mtp/layers/0/mlp/gate",
        "mtp/layers/0/mlp/up",
        "mtp/layers/0/mlp/down",
        "mtp/embedding_norm",
        "mtp/hidden_norm",
        "mtp/final_norm",
        "mtp/layers/0/input_norm",
        "mtp/layers/0/post_attention_norm",
        "mtp/layers/0/attention/query_norm",
        "mtp/layers/0/attention/key_norm",
    ];
    let binding_names = directory
        .bindings
        .keys()
        .filter(|name| name.starts_with("mtp/"))
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    ensure!(
        binding_names == EXPECTED_BINDINGS.iter().copied().collect(),
        "native MTP binding inventory mismatch"
    );
    ensure!(
        norms.iter().count() == 7,
        "native MTP norm role count mismatch"
    );
    let mut object_ids = BTreeSet::new();
    for matrix in [
        &projections.fc,
        &projections.query_gate,
        &projections.key,
        &projections.value,
        &projections.attention_output,
        &projections.mlp_gate,
        &projections.mlp_up,
        &projections.mlp_down,
    ] {
        object_ids.insert(matrix.object_id.as_str());
    }
    for norm in norms.iter() {
        object_ids.insert(norm.object_id.as_str());
    }
    ensure!(
        object_ids.len() == MTP_OBJECTS,
        "native MTP object count mismatch"
    );
    let mut bytes = 0_u64;
    for object_id in object_ids {
        bytes = bytes
            .checked_add(object_by_id(directory, object_id)?.bytes)
            .context("native MTP object byte count overflows")?;
    }
    ensure!(bytes == MTP_BYTES, "native MTP object byte total mismatch");
    Ok(())
}

pub(super) fn validate_proposal_inventory(
    directory: &Directory,
    head: &Q4MatrixView,
    token_map_object: &str,
) -> Result<()> {
    const EXPECTED_BINDINGS: &[&str] = &["proposal/head", "proposal/token_ids"];
    let binding_names = directory
        .bindings
        .keys()
        .filter(|name| name.starts_with("proposal/"))
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    ensure!(
        binding_names == EXPECTED_BINDINGS.iter().copied().collect(),
        "native proposal binding inventory mismatch"
    );
    let head_object = object_by_id(directory, &head.object_id)?;
    let token_map = object_by_id(directory, token_map_object)?;
    ensure!(
        head_object.id != token_map.id,
        "native proposal head and token map must be separate objects"
    );
    let total = head_object
        .bytes
        .checked_add(token_map.bytes)
        .context("native proposal byte count overflows")?;
    ensure!(
        total == PROPOSAL_BYTES,
        "native proposal byte total mismatch"
    );
    ensure!(
        head_object.format.as_deref() == Some("q4_g64_fp16")
            && head_object.layout.as_deref() == Some("row_split_k128_v1")
            && head_object.shape.as_slice() == [131_072, 5_120],
        "native proposal head metadata mismatch"
    );
    ensure!(
        token_map.format.as_deref() == Some("int32")
            && token_map.layout.as_deref() == Some("contiguous_le_v1")
            && token_map.shape.as_slice() == [131_072]
            && token_map.bytes == 524_288,
        "native proposal token map metadata mismatch"
    );
    Ok(())
}

pub(super) fn target_head_object(directory: &Directory) -> Result<&Object> {
    let binding = directory
        .bindings
        .get("text/output_head")
        .context("native target output-head binding is missing")?;
    let Binding::Object { object } = binding else {
        anyhow::bail!("native target output head must bind one full object")
    };
    let object = object_by_id(directory, object)?;
    object.require_supported_encoding()?;
    Ok(object)
}

fn object_by_id<'a>(directory: &'a Directory, object_id: &str) -> Result<&'a Object> {
    directory
        .objects
        .iter()
        .find(|object| object.id == object_id)
        .with_context(|| format!("native object {object_id} is missing"))
}
