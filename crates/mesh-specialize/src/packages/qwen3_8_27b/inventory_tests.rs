use anyhow::{Context, Result, ensure};
use serde_json::{Map, Value};

use crate::artifact::schema::{DType, Directory, Object, ObjectKind, SourceCheckpoint};

use super::{MODEL_ID, WEIGHTS_ID, validate};

const MODEL_HEADER: &str =
    include_str!("../../../KNOWLEDGE/evidence/checkpoint-intake-20260927/model-header.json");
const MTP_HEADER: &str =
    include_str!("../../../KNOWLEDGE/evidence/checkpoint-intake-20260927/mtp-header.json");
const SOURCE_REPOSITORY: &str = "unsloth/Qwen3.8-27B-NVFP4";
const SOURCE_REVISION: &str = "f0b7c9e722f5565102fff8481c99e4d86ae099c7";
const ZERO_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[test]
fn captured_headers_match_compiled_inventory() {
    let inventory = validate(&captured_directory()).unwrap();

    assert_eq!(
        inventory.text_tensors,
        1_620,
        "{}",
        serde_json::to_string_pretty(&inventory).unwrap()
    );
    assert_eq!(inventory.mtp_tensors, 15);
    assert_eq!(inventory.text_bytes, 21_646_480_768);
    assert_eq!(inventory.mtp_bytes, 849_398_784);
    assert_eq!(
        inventory.full_attention_layers,
        (3..64).step_by(4).collect::<Vec<_>>()
    );
    assert_eq!(inventory.linear_attention_layers.len(), 48);
    assert_eq!(inventory.nvfp4_mlp_layers, (0..56).collect::<Vec<_>>());
    assert_eq!(inventory.fp8_mlp_layers, (56..64).collect::<Vec<_>>());
}

#[test]
fn rejects_missing_extra_and_duplicate_tensors() {
    let mut missing = captured_directory();
    missing
        .objects
        .retain(|object| object.name != "tensors/model.language_model.norm.weight");
    assert_rejected(&missing);

    let mut extra = captured_directory();
    let mut unknown = extra.objects[0].clone();
    unknown.name = "tensors/model.language_model.uncompiled.weight".to_string();
    extra.objects.push(unknown);
    assert_rejected(&extra);

    let mut duplicate = captured_directory();
    duplicate.objects.push(duplicate.objects[0].clone());
    assert_rejected(&duplicate);
}

#[test]
fn rejects_wrong_tensor_shape_dtype_layout_kind_and_length() {
    let mut wrong_shape = captured_directory();
    tensor_mut(&mut wrong_shape, "tensors/model.language_model.norm.weight")
        .shape
        .push(1);
    assert_rejected(&wrong_shape);

    let mut wrong_dtype = captured_directory();
    tensor_mut(&mut wrong_dtype, "tensors/model.language_model.norm.weight").dtype = DType::Fp8E4m3;
    assert_rejected(&wrong_dtype);

    let mut wrong_layout = captured_directory();
    tensor_mut(
        &mut wrong_layout,
        "tensors/model.language_model.norm.weight",
    )
    .layout = "safetensors-row-major-v2".to_string();
    assert_rejected(&wrong_layout);

    let mut wrong_kind = captured_directory();
    tensor_mut(&mut wrong_kind, "tensors/model.language_model.norm.weight").kind =
        ObjectKind::Config;
    assert_rejected(&wrong_kind);

    let mut wrong_length = captured_directory();
    tensor_mut(
        &mut wrong_length,
        "tensors/model.language_model.norm.weight",
    )
    .length -= 2;
    assert_rejected(&wrong_length);
}

#[test]
fn rejects_wrong_model_weights_repository_and_revision() {
    let mut wrong_model = captured_directory();
    wrong_model.identity.model_id.push_str("-other");
    assert_rejected(&wrong_model);

    let mut wrong_weights = captured_directory();
    wrong_weights.identity.weights_id.push('0');
    assert_rejected(&wrong_weights);

    let mut wrong_repository = captured_directory();
    wrong_repository.source.repository.push_str("-other");
    assert_rejected(&wrong_repository);

    let mut wrong_revision = captured_directory();
    wrong_revision.source.revision.push('0');
    assert_rejected(&wrong_revision);
}

fn assert_rejected(directory: &Directory) {
    assert!(validate(directory).is_err());
}

fn tensor_mut<'a>(directory: &'a mut Directory, name: &str) -> &'a mut Object {
    directory
        .objects
        .iter_mut()
        .find(|object| object.name == name)
        .unwrap_or_else(|| panic!("missing fixture tensor {name}"))
}

fn captured_directory() -> Directory {
    let mut objects = Vec::with_capacity(1_635);
    append_header(&mut objects, MODEL_HEADER, true).unwrap();
    append_header(&mut objects, MTP_HEADER, false).unwrap();
    Directory {
        schema_version: 1,
        identity: mesh_llm_native_runtime::model_identity::ModelIdentity {
            model_id: MODEL_ID.to_string(),
            weights_id: WEIGHTS_ID.to_string(),
        },
        recipe_sha256: ZERO_SHA256.to_string(),
        source: SourceCheckpoint {
            repository: SOURCE_REPOSITORY.to_string(),
            revision: SOURCE_REVISION.to_string(),
        },
        objects,
    }
}

// These fixture adapters retain header metadata only. Offsets and hashes are placeholders;
// the tests do not open payloads or claim verified artifact integrity.
fn append_header(objects: &mut Vec<Object>, raw_header: &str, omit_vision: bool) -> Result<()> {
    let header: Map<String, Value> = serde_json::from_str(raw_header)?;
    for (name, metadata) in header {
        if name == "__metadata__" || (omit_vision && name.starts_with("model.visual.")) {
            continue;
        }
        let dtype_name = metadata
            .get("dtype")
            .and_then(Value::as_str)
            .context("fixture tensor lacks dtype")?;
        let dtype = fixture_dtype(dtype_name)?;
        let shape = metadata
            .get("shape")
            .and_then(Value::as_array)
            .context("fixture tensor lacks shape")?
            .iter()
            .map(|dimension| {
                dimension
                    .as_u64()
                    .context("fixture tensor dimension is not a u64")
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = metadata
            .get("data_offsets")
            .and_then(Value::as_array)
            .context("fixture tensor lacks data offsets")?;
        ensure!(
            offsets.len() == 2,
            "fixture tensor has invalid data offsets"
        );
        let start = offsets[0]
            .as_u64()
            .context("fixture tensor start offset is not a u64")?;
        let end = offsets[1]
            .as_u64()
            .context("fixture tensor end offset is not a u64")?;
        ensure!(end >= start, "fixture tensor data offsets are reversed");
        objects.push(Object {
            name: format!("tensors/{name}"),
            kind: ObjectKind::Tensor,
            dtype,
            shape,
            layout: "safetensors-row-major-v1".to_string(),
            offset: 0,
            length: end - start,
            sha256: ZERO_SHA256.to_string(),
        });
    }
    Ok(())
}

fn fixture_dtype(name: &str) -> Result<DType> {
    match name {
        "F32" => Ok(DType::F32),
        "BF16" => Ok(DType::Bf16),
        "F8_E4M3" => Ok(DType::Fp8E4m3),
        "U8" => Ok(DType::U8),
        other => anyhow::bail!("unexpected checkpoint fixture dtype {other}"),
    }
}
