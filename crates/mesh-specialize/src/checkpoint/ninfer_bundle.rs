//! Offline assembly of an explicitly exported, byte-preserving Ninfer bundle.
//!
//! This reads only our intermediate JSON manifest and payload.bin, never .ninfer
//! framing. Metadata is inert provenance, not an executable model specification.

use crate::artifact::{
    reader::{MAX_ARTIFACT_BYTES, VerifiedArtifact},
    schema::{ALIGNMENT, DType, MAX_OBJECTS, ObjectKind, SourceCheckpoint, align_up},
    writer::{ObjectSource, SourceRange, write_artifact},
};
use anyhow::{Context, Result, ensure};
use mesh_llm_native_runtime::model_identity::ModelIdentity;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

pub const MODEL_ID: &str = "qwen3.8-27b:ninfer-preserved-v1";
pub const STORAGE_LAYOUT: &str = "ninfer-preserved-storage-v1";
pub const SOURCE_REPOSITORY: &str = "Neroued/ninfer:artifact";
pub const SOURCE_SHA256: &str = "74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82";
pub const SOURCE_BYTES: u64 = 23_719_715_844;
pub const READER_REVISION: &str = "e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d";
pub const MAX_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
const BUFFER_BYTES: usize = 64 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    pub(crate) schema_version: u32,
    pub(crate) profile: String,
    pub(crate) source: Source,
    pub(crate) payload: Payload,
    pub(crate) objects: Vec<Storage>,
    pub(crate) model: Model,
    pub(crate) preservation: Preservation,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Source {
    pub(crate) artifact_sha256: String,
    pub(crate) artifact_bytes: u64,
    pub(crate) reader_revision: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Payload {
    pub(crate) path: String,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Storage {
    pub(crate) name: String,
    pub(crate) offset: u64,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
    pub(crate) source_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Model {
    pub(crate) components: BTreeMap<String, Value>,
    pub(crate) bindings: BTreeMap<String, Value>,
    pub(crate) uses: Vec<Value>,
    // Official reader metadata has kind-specific fields and opaque extensions.
    // Validate identity/extent/reference fields without interpreting any codec.
    pub(crate) storages: Vec<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Preservation {
    pub(crate) requantized: bool,
    pub(crate) layout_transformed: bool,
    pub(crate) all_selected_object_bytes_copied: bool,
    pub(crate) excluded_components: Vec<String>,
    pub(crate) runtime_executable: bool,
}

#[derive(Debug, Default, Serialize)]
pub struct FormatSummary {
    pub objects: usize,
    pub bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct StorageEvidence {
    pub source_id: String,
    pub mspec_name: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Serialize)]
pub struct ImportReport {
    pub schema_version: u32,
    pub kind: &'static str,
    pub identity: ModelIdentity,
    pub artifact_bytes: u64,
    pub recipe_sha256: String,
    source: Source,
    pub payload_bytes: u64,
    pub payload_sha256: String,
    pub storage_objects: usize,
    pub storage_bytes: u64,
    pub formats: BTreeMap<String, FormatSummary>,
    pub quantized_objects: usize,
    pub quantized_bytes: u64,
    pub storages: Vec<StorageEvidence>,
    pub verified_objects: usize,
    pub verified_object_bytes: u64,
    pub payload_verified: bool,
    pub full_byte_verification: bool,
    pub model_artifact_verified: bool,
    pub model_executable: bool,
    pub requantized: bool,
    pub layout_transformed: bool,
}

/// Assemble a NEW .mspec and read back every byte with bounded buffers.
///
/// Source identity is an exporter assertion, checked against exact pins here;
/// only the offline exporter reads and verifies the original source artifact.
/// On readback failure, a published artifact is retained for diagnosis, not use.
/// Success does not authorize execution of this preserved-storage profile.
pub fn convert(input_directory: &Path, output: &Path) -> Result<ImportReport> {
    ensure_new_output(output)?;
    let directory = input_directory
        .canonicalize()
        .context("resolve bundle directory")?;
    ensure!(directory.is_dir(), "bundle input must be a directory");
    let manifest_path = regular_child(&directory, "manifest.json")?;
    let bytes = read_manifest(&manifest_path)?;
    let manifest = parse_manifest(&bytes)?;
    let payload = regular_child(&directory, "payload.bin")?;
    verify_payload(&payload, &manifest)?;

    // Snapshot the exact bytes parsed above, not a second read of mutable input.
    let mut recipe = tempfile::NamedTempFile::new().context("create manifest snapshot")?;
    recipe.write_all(&bytes)?;
    recipe.flush()?;
    let recipe_sha256 = hex::encode(Sha256::digest(&bytes));
    let sources = object_sources(
        &manifest,
        &payload,
        recipe.path(),
        &recipe_sha256,
        bytes.len(),
    );
    let written = write_artifact(
        output,
        MODEL_ID,
        SourceCheckpoint {
            repository: SOURCE_REPOSITORY.into(),
            revision: SOURCE_SHA256.into(),
        },
        &sources,
    )?;
    let (verified_objects, verified_object_bytes) =
        verify_written(output, &written.directory.identity)?;
    report(
        manifest,
        written,
        recipe_sha256,
        verified_objects,
        verified_object_bytes,
    )
}

fn ensure_new_output(output: &Path) -> Result<()> {
    match fs::symlink_metadata(output) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).context("inspect bundle output path"),
        Ok(_) => anyhow::bail!("bundle output already exists"),
    }
}

fn regular_child(directory: &Path, name: &str) -> Result<PathBuf> {
    let path = directory.join(name);
    let metadata = fs::symlink_metadata(&path).with_context(|| format!("inspect bundle {name}"))?;
    ensure!(
        metadata.file_type().is_file(),
        "bundle {name} must be a regular, non-symlink file"
    );
    let resolved = path.canonicalize()?;
    ensure!(
        resolved.parent() == Some(directory),
        "bundle {name} escapes its input directory"
    );
    Ok(resolved)
}

fn read_manifest(path: &Path) -> Result<Vec<u8>> {
    let file = File::open(path).context("open bundle manifest")?;
    let metadata = file.metadata()?;
    ensure!(metadata.is_file(), "bundle manifest must be a regular file");
    ensure!(
        metadata.len() <= MAX_MANIFEST_BYTES,
        "bundle manifest exceeds 4 MiB limit"
    );
    let mut bytes = Vec::new();
    file.take(MAX_MANIFEST_BYTES + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_MANIFEST_BYTES,
        "bundle manifest exceeds 4 MiB limit"
    );
    ensure!(
        bytes.len() as u64 == metadata.len(),
        "bundle manifest length changed during read"
    );
    Ok(bytes)
}

/// Parse and validate the inert recipe without opening any payload or source file.
/// Payload and per-storage hashes remain assertions until separately byte-verified.
pub(crate) fn parse_manifest(bytes: &[u8]) -> Result<Manifest> {
    ensure!(
        bytes.len() as u64 <= MAX_MANIFEST_BYTES,
        "bundle manifest exceeds 4 MiB limit"
    );
    let manifest: Manifest = serde_json::from_slice(bytes).context("parse bundle manifest v1")?;
    validate_manifest(&manifest)?;
    Ok(manifest)
}

fn validate_manifest(manifest: &Manifest) -> Result<()> {
    ensure!(
        manifest.schema_version == 1,
        "unsupported bundle schema version"
    );
    ensure!(manifest.profile == MODEL_ID, "unsupported bundle profile");
    ensure!(
        manifest.source.artifact_sha256 == SOURCE_SHA256,
        "source artifact SHA-256 pin mismatch"
    );
    ensure!(
        manifest.source.artifact_bytes == SOURCE_BYTES,
        "source artifact byte-length pin mismatch"
    );
    ensure!(
        manifest.source.reader_revision == READER_REVISION,
        "source reader revision pin mismatch"
    );
    ensure!(
        manifest.payload.path == "payload.bin",
        "bundle payload path must be payload.bin"
    );
    ensure!(
        manifest.payload.bytes <= MAX_ARTIFACT_BYTES,
        "bundle payload exceeds prototype size limit"
    );
    validate_sha256(&manifest.payload.sha256)?;
    let preservation = &manifest.preservation;
    ensure!(
        !preservation.requantized && !preservation.layout_transformed,
        "bundle must preserve encoded storage bytes and layout"
    );
    ensure!(
        preservation.all_selected_object_bytes_copied,
        "bundle must copy every selected object byte"
    );
    ensure!(
        !preservation.runtime_executable,
        "preserved bundle must not claim runtime execution support"
    );
    ensure!(
        preservation.excluded_components == ["vision", "dflash2"],
        "unexpected excluded bundle components"
    );
    validate_ranges(manifest)?;
    validate_model(manifest)
}

fn validate_sha256(value: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "bundle SHA-256 must be 64 lowercase hexadecimal characters"
    );
    Ok(())
}

fn validate_ranges(manifest: &Manifest) -> Result<()> {
    ensure!(
        !manifest.objects.is_empty() && manifest.objects.len() < MAX_OBJECTS,
        "bundle must contain 1..{} storage objects plus a recipe",
        MAX_OBJECTS - 1
    );
    let mut ids = BTreeSet::new();
    let mut end = 0;
    for (index, object) in manifest.objects.iter().enumerate() {
        ensure!(
            object.name == format!("storage/{index:06}"),
            "bundle storage names must be unique and ordered storage/000000 onwards"
        );
        ensure!(
            !object.source_id.is_empty() && ids.insert(&object.source_id),
            "bundle source ids must be nonempty and unique"
        );
        ensure!(object.bytes > 0, "bundle object must not be empty");
        ensure!(
            object.offset == align_up(end)?,
            "bundle object ranges must be ordered, nonoverlapping and aligned to 256 bytes"
        );
        end = object
            .offset
            .checked_add(object.bytes)
            .context("bundle object range overflows u64")?;
        ensure!(
            end <= manifest.payload.bytes,
            "bundle object range exceeds payload"
        );
        validate_sha256(&object.sha256)?;
    }
    ensure!(
        end == manifest.payload.bytes,
        "bundle payload must end at the final object extent"
    );
    Ok(())
}

fn validate_model(manifest: &Manifest) -> Result<()> {
    let model = &manifest.model;
    ensure!(
        model.components.len() == 2
            && model.components.contains_key("text")
            && model.components.contains_key("mtp"),
        "bundle must contain exactly text and mtp components"
    );
    let ids: BTreeSet<_> = manifest
        .objects
        .iter()
        .map(|object| object.source_id.as_str())
        .collect();
    validate_storage_metadata(manifest)?;
    for required in [
        "text/token_embedding",
        "text/output_head",
        "proposal/head",
        "proposal/token_ids",
    ] {
        ensure!(
            model.bindings.contains_key(required),
            "missing required bundle binding {required}"
        );
    }
    let mut referenced = BTreeSet::new();
    for (name, binding) in &model.bindings {
        ensure!(
            selected_name(name) && binding.is_object(),
            "invalid selected bundle binding {name}"
        );
        ensure!(
            object_refs(binding, &ids, &mut referenced)? > 0,
            "bundle binding {name} must reference storage"
        );
    }
    for value in &model.uses {
        for field in ["parameter", "input"] {
            let name = text_field(value, field)?;
            ensure!(
                selected_name(name),
                "bundle use {field} is outside selected components"
            );
        }
        ensure!(
            model.bindings.contains_key(text_field(value, "parameter")?),
            "bundle use parameter has no selected binding"
        );
        object_refs(value, &ids, &mut referenced)?;
    }
    for component in model.components.values() {
        ensure!(component.is_object(), "bundle component must be an object");
        object_refs(component, &ids, &mut referenced)?;
        if let Some(resources) = component.get("resources") {
            for value in resources
                .as_object()
                .context("component resources must be an object")?
                .values()
            {
                resolve_ref(value, &ids, &mut referenced)?;
            }
        }
    }
    ensure!(
        referenced == ids,
        "bundle contains storage outside its selected reference closure"
    );
    Ok(())
}

fn selected_name(name: &str) -> bool {
    ["text/", "mtp/", "proposal/"]
        .iter()
        .any(|prefix| name.starts_with(prefix))
}

fn validate_storage_metadata(manifest: &Manifest) -> Result<()> {
    ensure!(
        manifest.model.storages.len() == manifest.objects.len(),
        "model storage metadata cardinality mismatch"
    );
    let by_id: BTreeMap<_, _> = manifest
        .objects
        .iter()
        .map(|object| (object.source_id.as_str(), object))
        .collect();
    let mut seen = BTreeSet::new();
    for metadata in &manifest.model.storages {
        let id = text_field(metadata, "id")?;
        ensure!(seen.insert(id), "duplicate model storage metadata id");
        let object = by_id
            .get(id)
            .context("model storage metadata has unknown source id")?;
        ensure!(
            text_field(metadata, "storage")? == object.name,
            "model storage name mismatch"
        );
        ensure!(
            metadata.get("bytes").and_then(Value::as_u64) == Some(object.bytes),
            "model storage byte-length mismatch"
        );
        ensure!(
            metadata.get("offset").is_none(),
            "exported model storage must omit original source offset"
        );
        let kind = text_field(metadata, "kind")?;
        ensure!(
            kind == "tensor" || kind == "resource",
            "unsupported source storage kind"
        );
        if kind == "tensor" {
            text_field(metadata, "format")?;
            text_field(metadata, "layout")?;
            ensure!(
                metadata
                    .get("shape")
                    .and_then(Value::as_array)
                    .is_some_and(|shape| shape.iter().all(|d| d.as_u64().is_some_and(|d| d > 0))),
                "source tensor shape must be an array of positive dimensions (or scalar [])"
            );
        } else {
            text_field(metadata, "encoding")?;
        }
    }
    Ok(())
}

fn text_field<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .with_context(|| format!("bundle metadata {field} must be a nonempty string"))
}

fn object_refs<'a>(
    value: &'a Value,
    ids: &BTreeSet<&str>,
    referenced: &mut BTreeSet<&'a str>,
) -> Result<usize> {
    let mut count = 0;
    match value {
        Value::Object(fields) => {
            for (key, child) in fields {
                if key == "object" {
                    resolve_ref(child, ids, referenced)?;
                    count += 1;
                } else {
                    count += object_refs(child, ids, referenced)?;
                }
            }
        }
        Value::Array(values) => {
            for child in values {
                count += object_refs(child, ids, referenced)?;
            }
        }
        _ => {}
    }
    Ok(count)
}

fn resolve_ref<'a>(
    value: &'a Value,
    ids: &BTreeSet<&str>,
    referenced: &mut BTreeSet<&'a str>,
) -> Result<()> {
    let id = value
        .as_str()
        .context("bundle object/resource reference must be a string")?;
    ensure!(
        ids.contains(id),
        "unknown bundle object/resource reference {id}"
    );
    referenced.insert(id);
    Ok(())
}

fn verify_payload(path: &Path, manifest: &Manifest) -> Result<()> {
    let mut file = File::open(path).context("open bundle payload")?;
    ensure!(
        file.metadata()?.is_file(),
        "bundle payload must be a regular file"
    );
    ensure!(
        file.metadata()?.len() == manifest.payload.bytes,
        "bundle payload byte-length mismatch"
    );
    let mut digest = Sha256::new();
    let mut buffer = [0; BUFFER_BYTES];
    let mut position = 0;
    for object in &manifest.objects {
        // The range validator limits each gap to one alignment block.
        let mut padding = [0; ALIGNMENT as usize];
        let gap = &mut padding[..usize::try_from(object.offset - position)?];
        file.read_exact(gap)
            .context("read bundle alignment padding")?;
        ensure!(
            gap.iter().all(|byte| *byte == 0),
            "bundle alignment padding must be zero"
        );
        digest.update(gap);
        verify_storage(&mut file, object, &mut digest, &mut buffer)?;
        position = object.offset + object.bytes; // Checked by validate_ranges.
    }
    ensure!(
        file.read(&mut buffer[..1])? == 0,
        "bundle payload has unexpected trailing bytes"
    );
    ensure!(
        file.metadata()?.len() == manifest.payload.bytes,
        "bundle payload length changed during verification"
    );
    ensure!(
        hex::encode(digest.finalize()) == manifest.payload.sha256,
        "bundle payload SHA-256 mismatch"
    );
    Ok(())
}

fn verify_storage(
    file: &mut File,
    object: &Storage,
    payload_digest: &mut Sha256,
    buffer: &mut [u8],
) -> Result<()> {
    let mut digest = Sha256::new();
    let mut remaining = object.bytes;
    while remaining != 0 {
        let count = usize::try_from(remaining.min(buffer.len() as u64))?;
        let bytes = &mut buffer[..count];
        file.read_exact(bytes)
            .with_context(|| format!("read bundle storage {}", object.name))?;
        digest.update(&*bytes);
        payload_digest.update(&*bytes);
        remaining -= count as u64;
    }
    ensure!(
        hex::encode(digest.finalize()) == object.sha256,
        "bundle object SHA-256 mismatch: {}",
        object.name
    );
    Ok(())
}

fn object_sources(
    manifest: &Manifest,
    payload: &Path,
    recipe: &Path,
    recipe_hash: &str,
    recipe_bytes: usize,
) -> Vec<ObjectSource> {
    let mut sources: Vec<_> = manifest
        .objects
        .iter()
        .map(|object| ObjectSource {
            name: object.name.clone(),
            kind: ObjectKind::Tensor,
            dtype: DType::U8,
            shape: vec![object.bytes],
            layout: STORAGE_LAYOUT.into(),
            path: payload.to_path_buf(),
            range: Some(SourceRange {
                offset: object.offset,
                length: object.bytes,
            }),
            expected_sha256: Some(object.sha256.clone()),
        })
        .collect();
    sources.push(ObjectSource {
        name: "recipe.json".into(),
        kind: ObjectKind::Recipe,
        dtype: DType::U8,
        shape: vec![recipe_bytes as u64],
        layout: "raw-v1".into(),
        path: recipe.to_path_buf(),
        range: None,
        expected_sha256: Some(recipe_hash.into()),
    });
    sources
}

fn verify_written(path: &Path, identity: &ModelIdentity) -> Result<(usize, u64)> {
    let mut artifact = VerifiedArtifact::open_for_identity(path, identity)
        .context("verify preserved artifact identity and complete payload")?;
    let names: Vec<_> = artifact
        .directory()
        .objects
        .iter()
        .map(|object| object.name.clone())
        .collect();
    let mut bytes = 0_u64;
    for name in &names {
        // The copy API rechecks hashes using its bounded buffer; never allocate
        // a storage-sized Vec, even for the largest embedding/head object.
        bytes = bytes
            .checked_add(artifact.copy_object(name, &mut io::sink())?)
            .context("verified object byte count overflows u64")?;
    }
    Ok((names.len(), bytes))
}

fn report(
    manifest: Manifest,
    written: crate::artifact::writer::WrittenArtifact,
    recipe_sha256: String,
    verified_objects: usize,
    verified_object_bytes: u64,
) -> Result<ImportReport> {
    let mut formats = BTreeMap::<String, FormatSummary>::new();
    let mut quantized_objects = 0;
    let mut quantized_bytes = 0;
    for metadata in &manifest.model.storages {
        let format = if text_field(metadata, "kind")? == "resource" {
            format!("resource/{}", text_field(metadata, "encoding")?)
        } else {
            text_field(metadata, "format")?.to_owned()
        };
        let bytes = metadata["bytes"]
            .as_u64()
            .context("missing validated storage length")?;
        let summary = formats.entry(format.clone()).or_default();
        summary.objects += 1;
        summary.bytes += bytes; // Disjoint validated extents are bounded by payload length.
        if is_quantized(&format) {
            quantized_objects += 1;
            quantized_bytes += bytes;
        }
    }
    let storage_bytes = manifest.objects.iter().map(|object| object.bytes).sum();
    let storages = manifest
        .objects
        .into_iter()
        .map(|object| StorageEvidence {
            source_id: object.source_id,
            mspec_name: object.name,
            bytes: object.bytes,
            sha256: object.sha256,
        })
        .collect::<Vec<_>>();
    Ok(ImportReport {
        schema_version: 1,
        kind: "ninfer-preserved-bundle-import",
        identity: written.directory.identity,
        artifact_bytes: written.bytes,
        recipe_sha256,
        source: manifest.source,
        payload_bytes: manifest.payload.bytes,
        payload_sha256: manifest.payload.sha256,
        storage_objects: storages.len(),
        storage_bytes,
        formats,
        quantized_objects,
        quantized_bytes,
        storages,
        verified_objects,
        verified_object_bytes,
        payload_verified: true,
        full_byte_verification: true,
        model_artifact_verified: true,
        model_executable: false,
        requantized: false,
        layout_transformed: false,
    })
}

fn is_quantized(format: &str) -> bool {
    // Reporting labels observed in the pinned export, not a runtime codec gate.
    matches!(
        format,
        "fp8_e4m3fn_row_bf16" | "nvfp4" | "q4_g64_fp16" | "q8_g32_fp16"
    )
}

#[cfg(test)]
#[path = "ninfer_bundle_tests.rs"]
mod tests;
