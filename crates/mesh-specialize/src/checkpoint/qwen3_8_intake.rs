//! Lossless import of one pinned upstream quantized checkpoint.

use super::{qwen3_8_recipe as recipe, safetensors};
use crate::artifact::{
    reader::VerifiedArtifact,
    schema::{DType, ObjectKind, SourceCheckpoint},
    writer::{ObjectSource, write_artifact},
};
use anyhow::{Context, Result, ensure};
use mesh_llm_native_runtime::model_identity::ModelIdentity;
use serde::Serialize;
use std::{io::Write, path::Path};

#[derive(Debug, Serialize)]
pub struct SourceEvidence {
    pub name: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
    pub header_sha256: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct IntakeReport {
    pub schema_version: u32,
    pub kind: &'static str,
    pub identity: ModelIdentity,
    pub artifact_bytes: u64,
    pub tensor_count: usize,
    pub excluded_vision_tensors: usize,
    pub source_repository: &'static str,
    pub source_revision: &'static str,
    pub source_files: Vec<SourceEvidence>,
    pub model_artifact_verified: bool,
    pub model_executable: bool,
}

/// Verify pinned source bytes and assemble an internal artifact without quantizing
/// or executing the model. Packed NVFP4 codes remain stored as their upstream U8
/// physical tensors; interpreting the quantization belongs to the compiled model.
pub fn convert(input_directory: &Path, output: &Path) -> Result<IntakeReport> {
    ensure!(
        input_directory.is_dir(),
        "checkpoint input must be a directory"
    );
    ensure!(!output.try_exists()?, "checkpoint output already exists");
    validate_source_sizes(input_directory)?;

    let main = safetensors::verify(
        &input_directory.join(recipe::SOURCE_PINS[0].name),
        recipe::SOURCE_PINS[0].sha256,
    )
    .context("verify pinned main checkpoint")?;
    let mtp = safetensors::verify(
        &input_directory.join(recipe::SOURCE_PINS[1].name),
        recipe::SOURCE_PINS[1].sha256,
    )
    .context("verify pinned MTP checkpoint")?;
    let mut sources = recipe::tensor_sources(input_directory, &main, &mtp)?;
    let tensor_count = sources.len();
    sources.extend(recipe::auxiliary_sources(input_directory));

    let bytes = recipe::recipe_bytes()?;
    let mut recipe_file = tempfile::NamedTempFile::new().context("create intake recipe")?;
    recipe_file.write_all(&bytes)?;
    recipe_file.flush()?;
    sources.push(ObjectSource {
        name: "recipe.json".into(),
        kind: ObjectKind::Recipe,
        dtype: DType::U8,
        shape: vec![bytes.len() as u64],
        layout: "raw-v1".into(),
        path: recipe_file.path().to_path_buf(),
        range: None,
        expected_sha256: None,
    });
    let written = write_artifact(
        output,
        recipe::MODEL_ID,
        SourceCheckpoint {
            repository: recipe::SOURCE_REPO.into(),
            revision: recipe::SOURCE_REV.into(),
        },
        &sources,
    )?;
    let verified = VerifiedArtifact::open_for_identity(output, &written.directory.identity)
        .context("verify completed checkpoint artifact")?;

    let mut source_files: Vec<_> = recipe::SOURCE_PINS
        .iter()
        .map(|pin| SourceEvidence {
            name: pin.name,
            bytes: pin.bytes,
            sha256: pin.sha256,
            header_sha256: None,
        })
        .collect();
    source_files[0].header_sha256 = Some(main.header_sha256);
    source_files[1].header_sha256 = Some(mtp.header_sha256);
    Ok(IntakeReport {
        schema_version: 1,
        kind: "pinned-upstream-checkpoint-intake",
        identity: verified.identity().clone(),
        artifact_bytes: written.bytes,
        tensor_count,
        excluded_vision_tensors: 333,
        source_repository: recipe::SOURCE_REPO,
        source_revision: recipe::SOURCE_REV,
        source_files,
        model_artifact_verified: true,
        model_executable: false,
    })
}

fn validate_source_sizes(directory: &Path) -> Result<()> {
    for pin in recipe::SOURCE_PINS {
        let metadata = directory
            .join(pin.name)
            .metadata()
            .with_context(|| format!("inspect pinned source {}", pin.name))?;
        ensure!(
            metadata.is_file(),
            "pinned source {} is not a regular file",
            pin.name
        );
        ensure!(
            metadata.len() == pin.bytes,
            "pinned source {} has the wrong byte length",
            pin.name
        );
    }
    Ok(())
}
