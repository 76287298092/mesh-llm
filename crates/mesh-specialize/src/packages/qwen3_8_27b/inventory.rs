//! Fixed compiled tensor inventory for the pinned Qwen3.8-27B checkpoint.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::artifact::schema::{DType, Directory, ObjectKind};

pub const MODEL_ID: &str = "qwen3.8-27b:text:nvfp4-fp8:upstream-raw-v1";
pub const WEIGHTS_ID: &str =
    "sha256:f49713878a072f8c9043060dc0e2f3b28421301e49471bee0c13c7570e59e81e";

const SOURCE_REPOSITORY: &str = "unsloth/Qwen3.8-27B-NVFP4";
const SOURCE_REVISION: &str = "f0b7c9e722f5565102fff8481c99e4d86ae099c7";
const TENSOR_LAYOUT: &str = "safetensors-row-major-v1";

const HIDDEN: u64 = 5_120;
const VOCABULARY: u64 = 248_320;
const INTERMEDIATE: u64 = 17_408;
const LAYER_COUNT: usize = 64;
const NVFP4_MLP_LAYER_END: usize = 56;

const EXPECTED_TEXT_TENSORS: usize = 1_620;
const EXPECTED_MTP_TENSORS: usize = 15;
const EXPECTED_TEXT_BYTES: u64 = 21_646_480_768;
const EXPECTED_MTP_BYTES: u64 = 849_398_784;

#[derive(Debug, Serialize)]
pub struct Inventory {
    pub text_tensors: usize,
    pub mtp_tensors: usize,
    pub text_bytes: u64,
    pub mtp_bytes: u64,
    pub full_attention_layers: Vec<usize>,
    pub linear_attention_layers: Vec<usize>,
    pub nvfp4_mlp_layers: Vec<usize>,
    pub fp8_mlp_layers: Vec<usize>,
}

/// Validate tensor metadata against the fixed compiled Qwen model inventory.
///
/// Artifact integrity and non-tensor assets are the verified reader's responsibility.
pub fn validate(directory: &Directory) -> Result<Inventory> {
    validate_identity(directory)?;

    let native = directory.identity.model_id == super::native_views::MODEL_ID;
    let (expected_text_tensors, expected_text_bytes, expected_mtp_tensors, expected_mtp_bytes) =
        if native {
            (
                super::native_views::TEXT_TENSORS,
                super::native_views::TEXT_BYTES,
                0,
                0,
            )
        } else {
            (
                EXPECTED_TEXT_TENSORS,
                EXPECTED_TEXT_BYTES,
                EXPECTED_MTP_TENSORS,
                EXPECTED_MTP_BYTES,
            )
        };
    let mut expected = if native {
        native_tensors()?
    } else {
        compiled_tensors()?
    };
    let mut seen = HashSet::with_capacity(expected.len());
    let mut text_tensors = 0;
    let mut mtp_tensors = 0;
    let mut text_bytes = 0_u64;
    let mut mtp_bytes = 0_u64;

    for object in directory
        .objects
        .iter()
        .filter(|object| object.kind == ObjectKind::Tensor)
    {
        ensure!(
            seen.insert(object.name.as_str()),
            "duplicate tensor name: {}",
            object.name
        );
        let spec = expected
            .remove(&object.name)
            .with_context(|| format!("unexpected tensor name: {}", object.name))?;
        ensure!(
            object.dtype == spec.dtype,
            "tensor {} has dtype {:?}; expected {:?}",
            object.name,
            object.dtype,
            spec.dtype
        );
        ensure!(
            object.shape == spec.shape,
            "tensor {} has shape {:?}; expected {:?}",
            object.name,
            object.shape,
            spec.shape
        );
        ensure!(
            object.layout == TENSOR_LAYOUT,
            "tensor {} has layout {}; expected {}",
            object.name,
            object.layout,
            TENSOR_LAYOUT
        );
        ensure!(
            object.length == spec.length,
            "tensor {} has length {}; expected {}",
            object.name,
            object.length,
            spec.length
        );
        match spec.family {
            TensorFamily::Text => {
                text_tensors += 1;
                text_bytes = text_bytes
                    .checked_add(object.length)
                    .context("text tensor byte count overflows u64")?;
            }
            TensorFamily::Mtp => {
                mtp_tensors += 1;
                mtp_bytes = mtp_bytes
                    .checked_add(object.length)
                    .context("MTP tensor byte count overflows u64")?;
            }
        }
    }

    ensure!(
        expected.is_empty(),
        "missing expected tensor: {}",
        expected
            .keys()
            .next()
            .context("expected tensor set is empty")?
    );
    ensure!(
        text_tensors == expected_text_tensors,
        "compiled text tensor count is {text_tensors}; expected {expected_text_tensors}"
    );
    ensure!(
        mtp_tensors == expected_mtp_tensors,
        "compiled MTP tensor count is {mtp_tensors}; expected {expected_mtp_tensors}"
    );
    ensure!(
        text_bytes == expected_text_bytes,
        "compiled text tensor bytes are {text_bytes}; expected {expected_text_bytes}"
    );
    ensure!(
        mtp_bytes == expected_mtp_bytes,
        "compiled MTP tensor bytes are {mtp_bytes}; expected {expected_mtp_bytes}"
    );

    Ok(Inventory {
        text_tensors,
        mtp_tensors,
        text_bytes,
        mtp_bytes,
        full_attention_layers: (0..LAYER_COUNT).filter(|layer| layer % 4 == 3).collect(),
        linear_attention_layers: (0..LAYER_COUNT).filter(|layer| layer % 4 != 3).collect(),
        nvfp4_mlp_layers: (0..NVFP4_MLP_LAYER_END).collect(),
        fp8_mlp_layers: (NVFP4_MLP_LAYER_END..LAYER_COUNT).collect(),
    })
}

fn validate_identity(directory: &Directory) -> Result<()> {
    if directory.identity.model_id == super::native_views::MODEL_ID {
        ensure!(
            directory.identity.weights_id
                == format!("sha256:{}", super::native_views::SOURCE_SHA256)
                && directory.source.repository == super::native_source::SOURCE_REPOSITORY
                && directory.source.revision == super::native_views::SOURCE_SHA256,
            "native artifact identity does not match the pinned source"
        );
        return Ok(());
    }
    ensure!(
        directory.identity.model_id == MODEL_ID,
        "artifact model identity does not match compiled Qwen model"
    );
    ensure!(
        directory.identity.weights_id == WEIGHTS_ID,
        "artifact weights identity does not match compiled Qwen weights"
    );
    ensure!(
        directory.source.repository == SOURCE_REPOSITORY,
        "artifact source repository does not match pinned Qwen checkpoint"
    );
    ensure!(
        directory.source.revision == SOURCE_REVISION,
        "artifact source revision does not match pinned Qwen checkpoint"
    );
    Ok(())
}

#[derive(Clone, Copy)]
enum TensorFamily {
    Text,
    Mtp,
}

struct ExpectedTensor {
    dtype: DType,
    shape: Vec<u64>,
    length: u64,
    family: TensorFamily,
}

fn compiled_tensors() -> Result<HashMap<String, ExpectedTensor>> {
    let mut tensors = HashMap::with_capacity(EXPECTED_TEXT_TENSORS + EXPECTED_MTP_TENSORS);
    add_tensor(
        &mut tensors,
        "model.language_model.embed_tokens.weight",
        DType::Bf16,
        vec![VOCABULARY, HIDDEN],
        TensorFamily::Text,
    )?;
    add_tensor(
        &mut tensors,
        "model.language_model.norm.weight",
        DType::Bf16,
        vec![HIDDEN],
        TensorFamily::Text,
    )?;
    add_tensor(
        &mut tensors,
        "lm_head.weight",
        DType::Fp8E4m3,
        vec![VOCABULARY, HIDDEN],
        TensorFamily::Text,
    )?;
    add_tensor(
        &mut tensors,
        "lm_head.weight_scale",
        DType::Bf16,
        vec![VOCABULARY, 1],
        TensorFamily::Text,
    )?;

    for layer in 0..LAYER_COUNT {
        add_layer_norms(&mut tensors, layer)?;
        if layer % 4 == 3 {
            add_full_attention(&mut tensors, layer)?;
        } else {
            add_linear_attention(&mut tensors, layer)?;
        }
        if layer < NVFP4_MLP_LAYER_END {
            add_nvfp4_mlp(&mut tensors, layer)?;
        } else {
            add_fp8_mlp(&mut tensors, layer)?;
        }
    }

    add_mtp_tensors(&mut tensors)?;
    ensure!(
        tensors.len() == EXPECTED_TEXT_TENSORS + EXPECTED_MTP_TENSORS,
        "compiled tensor generator produced {} entries",
        tensors.len()
    );
    Ok(tensors)
}

/// Native text contract is a distinct strict inventory, not a relaxed raw gate.
fn native_tensors() -> Result<HashMap<String, ExpectedTensor>> {
    let mut tensors = compiled_tensors()?;
    tensors.retain(|name, _| {
        !name.starts_with("tensors/mtp.")
            && !name.ends_with(".self_attn.k_scale")
            && !name.ends_with(".self_attn.v_scale")
    });
    for (name, tensor) in &mut tensors {
        if name == "tensors/model.language_model.embed_tokens.weight" {
            tensor.dtype = DType::Fp8E4m3;
        } else if name.ends_with(".linear_attn.A_log") || name.ends_with(".linear_attn.dt_bias") {
            tensor.dtype = DType::F32;
        }
        tensor.length = storage_length(&tensor.dtype, &tensor.shape)?;
    }
    add_tensor(
        &mut tensors,
        "model.language_model.embed_tokens.weight_scale",
        DType::Bf16,
        vec![VOCABULARY, 1],
        TensorFamily::Text,
    )?;
    Ok(tensors)
}

fn add_layer_norms(tensors: &mut HashMap<String, ExpectedTensor>, layer: usize) -> Result<()> {
    let prefix = format!("model.language_model.layers.{layer}");
    add_tensor(
        tensors,
        &format!("{prefix}.input_layernorm.weight"),
        DType::Bf16,
        vec![HIDDEN],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.post_attention_layernorm.weight"),
        DType::Bf16,
        vec![HIDDEN],
        TensorFamily::Text,
    )
}

fn add_full_attention(tensors: &mut HashMap<String, ExpectedTensor>, layer: usize) -> Result<()> {
    let prefix = format!("model.language_model.layers.{layer}.self_attn");
    add_tensor(
        tensors,
        &format!("{prefix}.q_norm.weight"),
        DType::Bf16,
        vec![256],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.k_norm.weight"),
        DType::Bf16,
        vec![256],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.q_proj.weight"),
        DType::Fp8E4m3,
        vec![12_288, HIDDEN],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.q_proj.weight_scale"),
        DType::Bf16,
        vec![12_288, 1],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.k_proj.weight"),
        DType::Fp8E4m3,
        vec![1_024, HIDDEN],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.k_proj.weight_scale"),
        DType::Bf16,
        vec![1_024, 1],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.k_scale"),
        DType::Bf16,
        vec![1],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.v_proj.weight"),
        DType::Fp8E4m3,
        vec![1_024, HIDDEN],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.v_proj.weight_scale"),
        DType::Bf16,
        vec![1_024, 1],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.v_scale"),
        DType::Bf16,
        vec![1],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.o_proj.weight"),
        DType::Fp8E4m3,
        vec![HIDDEN, 6_144],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.o_proj.weight_scale"),
        DType::Bf16,
        vec![HIDDEN, 1],
        TensorFamily::Text,
    )
}

fn add_linear_attention(tensors: &mut HashMap<String, ExpectedTensor>, layer: usize) -> Result<()> {
    let prefix = format!("model.language_model.layers.{layer}.linear_attn");
    add_tensor(
        tensors,
        &format!("{prefix}.A_log"),
        DType::Bf16,
        vec![48],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.dt_bias"),
        DType::Bf16,
        vec![48],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.conv1d.weight"),
        DType::Bf16,
        vec![10_240, 1, 4],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.in_proj_a.weight"),
        DType::Bf16,
        vec![48, HIDDEN],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.in_proj_b.weight"),
        DType::Bf16,
        vec![48, HIDDEN],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.in_proj_qkv.weight"),
        DType::Fp8E4m3,
        vec![10_240, HIDDEN],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.in_proj_qkv.weight_scale"),
        DType::Bf16,
        vec![10_240, 1],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.in_proj_z.weight"),
        DType::Fp8E4m3,
        vec![6_144, HIDDEN],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.in_proj_z.weight_scale"),
        DType::Bf16,
        vec![6_144, 1],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.norm.weight"),
        DType::Bf16,
        vec![128],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.out_proj.weight"),
        DType::Fp8E4m3,
        vec![HIDDEN, 6_144],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.out_proj.weight_scale"),
        DType::Bf16,
        vec![HIDDEN, 1],
        TensorFamily::Text,
    )
}

fn add_nvfp4_mlp(tensors: &mut HashMap<String, ExpectedTensor>, layer: usize) -> Result<()> {
    let prefix = format!("model.language_model.layers.{layer}.mlp");
    add_nvfp4_projection(
        tensors,
        &format!("{prefix}.gate_proj"),
        INTERMEDIATE,
        HIDDEN,
    )?;
    add_nvfp4_projection(tensors, &format!("{prefix}.up_proj"), INTERMEDIATE, HIDDEN)?;
    add_nvfp4_projection(
        tensors,
        &format!("{prefix}.down_proj"),
        HIDDEN,
        INTERMEDIATE,
    )
}

fn add_nvfp4_projection(
    tensors: &mut HashMap<String, ExpectedTensor>,
    prefix: &str,
    output: u64,
    input: u64,
) -> Result<()> {
    add_tensor(
        tensors,
        &format!("{prefix}.weight_packed"),
        DType::U8,
        vec![output, input / 2],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.weight_scale"),
        DType::Fp8E4m3,
        vec![output, input / 16],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.input_global_scale"),
        DType::F32,
        vec![1],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.weight_global_scale"),
        DType::F32,
        vec![1],
        TensorFamily::Text,
    )
}

fn add_fp8_mlp(tensors: &mut HashMap<String, ExpectedTensor>, layer: usize) -> Result<()> {
    let prefix = format!("model.language_model.layers.{layer}.mlp");
    add_fp8_projection(
        tensors,
        &format!("{prefix}.gate_proj"),
        INTERMEDIATE,
        HIDDEN,
    )?;
    add_fp8_projection(tensors, &format!("{prefix}.up_proj"), INTERMEDIATE, HIDDEN)?;
    add_fp8_projection(
        tensors,
        &format!("{prefix}.down_proj"),
        HIDDEN,
        INTERMEDIATE,
    )
}

fn add_fp8_projection(
    tensors: &mut HashMap<String, ExpectedTensor>,
    prefix: &str,
    output: u64,
    input: u64,
) -> Result<()> {
    add_tensor(
        tensors,
        &format!("{prefix}.weight"),
        DType::Fp8E4m3,
        vec![output, input],
        TensorFamily::Text,
    )?;
    add_tensor(
        tensors,
        &format!("{prefix}.weight_scale"),
        DType::Bf16,
        vec![output, 1],
        TensorFamily::Text,
    )
}

fn add_mtp_tensors(tensors: &mut HashMap<String, ExpectedTensor>) -> Result<()> {
    const MTP_TENSORS: &[(&str, &[u64])] = &[
        ("mtp.fc.weight", &[5_120, 10_240]),
        ("mtp.layers.0.input_layernorm.weight", &[HIDDEN]),
        ("mtp.layers.0.mlp.down_proj.weight", &[HIDDEN, INTERMEDIATE]),
        ("mtp.layers.0.mlp.gate_proj.weight", &[INTERMEDIATE, HIDDEN]),
        ("mtp.layers.0.mlp.up_proj.weight", &[INTERMEDIATE, HIDDEN]),
        ("mtp.layers.0.post_attention_layernorm.weight", &[HIDDEN]),
        ("mtp.layers.0.self_attn.k_norm.weight", &[256]),
        ("mtp.layers.0.self_attn.k_proj.weight", &[1_024, HIDDEN]),
        ("mtp.layers.0.self_attn.o_proj.weight", &[HIDDEN, 6_144]),
        ("mtp.layers.0.self_attn.q_norm.weight", &[256]),
        ("mtp.layers.0.self_attn.q_proj.weight", &[12_288, HIDDEN]),
        ("mtp.layers.0.self_attn.v_proj.weight", &[1_024, HIDDEN]),
        ("mtp.norm.weight", &[HIDDEN]),
        ("mtp.pre_fc_norm_embedding.weight", &[HIDDEN]),
        ("mtp.pre_fc_norm_hidden.weight", &[HIDDEN]),
    ];

    for (name, shape) in MTP_TENSORS {
        add_tensor(
            tensors,
            name,
            DType::Bf16,
            shape.to_vec(),
            TensorFamily::Mtp,
        )?;
    }
    Ok(())
}

fn add_tensor(
    tensors: &mut HashMap<String, ExpectedTensor>,
    name: &str,
    dtype: DType,
    shape: Vec<u64>,
    family: TensorFamily,
) -> Result<()> {
    let key = format!("tensors/{name}");
    let length = storage_length(&dtype, &shape)?;
    let previous = tensors.insert(
        key.clone(),
        ExpectedTensor {
            dtype,
            shape,
            length,
            family,
        },
    );
    ensure!(
        previous.is_none(),
        "compiled inventory repeats tensor {key}"
    );
    Ok(())
}

fn storage_length(dtype: &DType, shape: &[u64]) -> Result<u64> {
    let elements = shape.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .context("compiled tensor element count overflows u64")
    })?;
    let bytes_per_element = match dtype {
        DType::F32 => 4,
        DType::F16 | DType::Bf16 => 2,
        DType::I8 | DType::U8 | DType::Fp8E4m3 => 1,
        DType::U4 | DType::Fp4E2m1 => {
            anyhow::bail!("compiled inventory requires byte-addressed packed tensors")
        }
    };
    elements
        .checked_mul(bytes_per_element)
        .context("compiled tensor byte length overflows u64")
}

#[cfg(test)]
#[path = "inventory_tests.rs"]
mod tests;
