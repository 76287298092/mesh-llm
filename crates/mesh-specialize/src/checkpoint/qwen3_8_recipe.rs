//! Pinned metadata for the Qwen3.8-27B upstream-raw intake recipe.

use super::safetensors::{TensorEntry, VerifiedTensorFile};
use crate::artifact::writer::{DType, ObjectKind, ObjectSource, SourceRange};
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::path::Path;

pub(super) const MODEL_ID: &str = "qwen3.8-27b:text:nvfp4-fp8:upstream-raw-v1";
pub(super) const SOURCE_REPO: &str = "unsloth/Qwen3.8-27B-NVFP4";
pub(super) const SOURCE_REV: &str = "f0b7c9e722f5565102fff8481c99e4d86ae099c7";

const MAIN_FILE: &str = "model.safetensors";
const MTP_FILE: &str = "model_mtp.safetensors";
const MAIN_TENSOR_COUNT: usize = 1_953;
const MAIN_VISUAL_COUNT: usize = 333;
const MAIN_KEPT_COUNT: usize = 1_620;
const MTP_TENSOR_COUNT: usize = 15;
const TOTAL_KEPT_COUNT: usize = MAIN_KEPT_COUNT + MTP_TENSOR_COUNT;
const VISUAL_PREFIX: &str = "model.visual.";
const LANGUAGE_PREFIX: &str = "model.language_model.";
const LM_HEAD_PREFIX: &str = "lm_head.";
const MTP_PREFIX: &str = "mtp.";
const TENSOR_LAYOUT: &str = "safetensors-row-major-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(super) struct SourcePin {
    pub(super) name: &'static str,
    pub(super) bytes: u64,
    pub(super) sha256: &'static str,
}

const MAIN_PIN: SourcePin = SourcePin {
    name: MAIN_FILE,
    bytes: 22_568_192_096,
    sha256: "c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05",
};
const MTP_PIN: SourcePin = SourcePin {
    name: MTP_FILE,
    bytes: 849_400_392,
    sha256: "1d8268aa85ace093a561e3e7b63b9d390dac1cd55a90cd55b5ec509c3c9da9fe",
};

pub(super) const SOURCE_PINS: [SourcePin; 8] = [
    MAIN_PIN,
    MTP_PIN,
    SourcePin {
        name: "config.json",
        bytes: 22_564,
        sha256: "1b3c71868d1299e52df6fc907deb202d5132b1ef0f72aae0ef6d15185dd53a5c",
    },
    SourcePin {
        name: "model.safetensors.index.json",
        bytes: 164_371,
        sha256: "429430e1b9e65b2cb98eff8cd10a06e70a09cee89c48487a3914684aeb6df57f",
    },
    SourcePin {
        name: "tokenizer.json",
        bytes: 19_989_325,
        sha256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    },
    SourcePin {
        name: "tokenizer_config.json",
        bytes: 1_047,
        sha256: "529f30018c36dca5387c99b5edf368287f386f2c32d3790aa7141956bc5119fa",
    },
    SourcePin {
        name: "generation_config.json",
        bytes: 214,
        sha256: "d0d0ed2e37cdfafef4a5067d5ea2407b05f4fb50526e47c008a5b235d50240fb",
    },
    SourcePin {
        name: "chat_template.jinja",
        bytes: 9_993,
        sha256: "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce",
    },
];

/// Build sources for the preserved language-model and MTP tensor byte ranges.
pub(super) fn tensor_sources(
    input_dir: &Path,
    main: &VerifiedTensorFile,
    mtp: &VerifiedTensorFile,
) -> Result<Vec<ObjectSource>> {
    verify_pinned_file(main, &MAIN_PIN, MAIN_TENSOR_COUNT)?;
    verify_pinned_file(mtp, &MTP_PIN, MTP_TENSOR_COUNT)?;

    let mut sources = Vec::with_capacity(TOTAL_KEPT_COUNT);
    let visual_count = append_main_tensors(input_dir, main, &mut sources)?;
    ensure!(
        visual_count == MAIN_VISUAL_COUNT,
        "pinned main checkpoint must contain exactly {MAIN_VISUAL_COUNT} excluded visual tensors"
    );
    ensure!(
        sources.len() == MAIN_KEPT_COUNT,
        "pinned main checkpoint must contain exactly {MAIN_KEPT_COUNT} retained tensors"
    );
    append_mtp_tensors(input_dir, mtp, &mut sources)?;
    ensure!(
        sources.len() == TOTAL_KEPT_COUNT,
        "pinned checkpoints must produce exactly {TOTAL_KEPT_COUNT} tensor sources"
    );
    sources.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
    Ok(sources)
}

/// Build pinned tokenizer/config objects; file reads and hashes belong to the writer.
pub(super) fn auxiliary_sources(input_dir: &Path) -> Vec<ObjectSource> {
    SOURCE_PINS
        .iter()
        .skip(2)
        .map(|pin| {
            let kind = auxiliary_kind(pin.name);
            ObjectSource {
                name: format!("assets/{}", pin.name),
                kind,
                dtype: DType::U8,
                shape: vec![pin.bytes],
                layout: "raw-v1".to_string(),
                path: input_dir.join(pin.name),
                range: None,
                expected_sha256: Some(pin.sha256.to_string()),
            }
        })
        .collect()
}

/// Serialize the fixed source, exclusion, and byte-preservation recipe.
pub(super) fn recipe_bytes() -> Result<Vec<u8>> {
    let recipe = RecipeV1 {
        version: 1,
        id: "upstream-raw-v1",
        model_id: MODEL_ID,
        source: RecipeSource {
            repository: SOURCE_REPO,
            revision: SOURCE_REV,
        },
        source_files: &SOURCE_PINS,
        byte_layout: TENSOR_LAYOUT,
        tensor_bytes: "preserved",
        excluded_prefixes: [VISUAL_PREFIX],
        mtp: "preserved-not-yet-executed",
    };
    serde_json::to_vec(&recipe).context("serialize pinned Qwen3.8 recipe")
}

fn verify_pinned_file(
    file: &VerifiedTensorFile,
    pin: &SourcePin,
    expected_tensors: usize,
) -> Result<()> {
    ensure!(
        file.file_len == pin.bytes,
        "{} length does not match its pinned size",
        pin.name
    );
    ensure!(
        file.file_sha256 == pin.sha256,
        "{} digest does not match its pinned SHA-256",
        pin.name
    );
    ensure!(
        file.tensors.len() == expected_tensors,
        "{} must contain exactly {expected_tensors} tensors",
        pin.name
    );
    Ok(())
}

fn append_main_tensors(
    input_dir: &Path,
    file: &VerifiedTensorFile,
    output: &mut Vec<ObjectSource>,
) -> Result<usize> {
    let mut visual_count = 0;
    for tensor in &file.tensors {
        if tensor.name.starts_with(VISUAL_PREFIX) {
            visual_count += 1;
            continue;
        }
        ensure!(
            tensor.name.starts_with(LANGUAGE_PREFIX) || tensor.name.starts_with(LM_HEAD_PREFIX),
            "unexpected tensor namespace in pinned main checkpoint: {}",
            tensor.name
        );
        output.push(tensor_source(input_dir.join(MAIN_FILE), tensor)?);
    }
    Ok(visual_count)
}

fn append_mtp_tensors(
    input_dir: &Path,
    file: &VerifiedTensorFile,
    output: &mut Vec<ObjectSource>,
) -> Result<()> {
    for tensor in &file.tensors {
        ensure!(
            tensor.name.starts_with(MTP_PREFIX),
            "unexpected tensor namespace in pinned MTP checkpoint: {}",
            tensor.name
        );
        output.push(tensor_source(input_dir.join(MTP_FILE), tensor)?);
    }
    Ok(())
}

fn tensor_source(path: std::path::PathBuf, tensor: &TensorEntry) -> Result<ObjectSource> {
    ensure!(
        (1..=8).contains(&tensor.shape.len()) && tensor.shape.iter().all(|size| *size != 0),
        "tensor {} must have a nonempty shape with at most eight nonzero dimensions",
        tensor.name
    );
    Ok(ObjectSource {
        name: format!("tensors/{}", tensor.name),
        kind: ObjectKind::Tensor,
        dtype: artifact_dtype(tensor.dtype)?,
        shape: tensor.shape.clone(),
        layout: TENSOR_LAYOUT.to_string(),
        path,
        range: Some(SourceRange {
            offset: tensor.offset,
            length: tensor.length,
        }),
        expected_sha256: Some(tensor.sha256.clone()),
    })
}

fn artifact_dtype(dtype: safetensors::Dtype) -> Result<DType> {
    match dtype {
        safetensors::Dtype::F32 => Ok(DType::F32),
        safetensors::Dtype::BF16 => Ok(DType::Bf16),
        safetensors::Dtype::F8_E4M3 => Ok(DType::Fp8E4m3),
        safetensors::Dtype::U8 => Ok(DType::U8),
        unsupported => anyhow::bail!("unsupported pinned tensor dtype {unsupported:?}"),
    }
}

fn auxiliary_kind(name: &str) -> ObjectKind {
    match name {
        "config.json" | "model.safetensors.index.json" | "generation_config.json" => {
            ObjectKind::Config
        }
        "tokenizer.json" | "tokenizer_config.json" | "chat_template.jinja" => ObjectKind::Tokenizer,
        _ => unreachable!("only pinned auxiliary files are passed to auxiliary_kind"),
    }
}

#[derive(Serialize)]
struct RecipeV1 {
    version: u32,
    id: &'static str,
    model_id: &'static str,
    source: RecipeSource,
    source_files: &'static [SourcePin],
    byte_layout: &'static str,
    tensor_bytes: &'static str,
    excluded_prefixes: [&'static str; 1],
    mtp: &'static str,
}

#[derive(Serialize)]
struct RecipeSource {
    repository: &'static str,
    revision: &'static str,
}

#[cfg(test)]
#[path = "qwen3_8_recipe_tests.rs"]
mod tests;
