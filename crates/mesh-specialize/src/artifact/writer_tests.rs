use super::{
    ObjectSource, SourceCheckpoint, hash_sources, open_sources, validate_for_write, write_artifact,
    write_artifact_contents,
};
use crate::artifact::{
    header::Header,
    reader::VerifiedArtifact,
    schema::{DType, ObjectKind},
};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};
use tempfile::NamedTempFile;

fn source_checkpoint() -> SourceCheckpoint {
    SourceCheckpoint {
        repository: "test/model".to_string(),
        revision: "a".repeat(40),
    }
}

fn write_source(
    directory: &Path,
    path_name: &str,
    object_name: &str,
    kind: ObjectKind,
    shape: Vec<u64>,
    data: &[u8],
) -> ObjectSource {
    let path = directory.join(path_name);
    fs::write(&path, data).unwrap();
    ObjectSource {
        name: object_name.to_string(),
        kind,
        dtype: DType::U8,
        shape,
        layout: "raw-v1".to_string(),
        path,
    }
}

fn fixture_sources(directory: &Path, tensor: &[u8]) -> Vec<ObjectSource> {
    vec![
        write_source(
            directory,
            "tensor.input",
            "a.tensor",
            ObjectKind::Tensor,
            vec![tensor.len() as u64],
            tensor,
        ),
        write_source(
            directory,
            "recipe.input",
            "z.recipe",
            ObjectKind::Recipe,
            vec![10],
            b"recipe-v1!",
        ),
    ]
}

#[test]
fn writes_a_verified_artifact_and_copies_source_objects() {
    let temporary = tempfile::tempdir().unwrap();
    let sources = fixture_sources(temporary.path(), &[1, 2, 3, 4]);
    let output = temporary.path().join("fixture.mspec");
    let written = write_artifact(&output, "fixture:model", source_checkpoint(), &sources).unwrap();

    assert_eq!(written.bytes, fs::metadata(&output).unwrap().len());
    assert!(written.directory.identity.weights_id.starts_with("sha256:"));
    let mut artifact =
        VerifiedArtifact::open_for_identity(&output, &written.directory.identity).unwrap();
    let mut tensor = Vec::new();
    artifact.copy_object("a.tensor", &mut tensor).unwrap();
    assert_eq!(tensor, [1, 2, 3, 4]);
    let mut recipe = Vec::new();
    artifact.copy_object("z.recipe", &mut recipe).unwrap();
    assert_eq!(recipe, b"recipe-v1!");
}

#[test]
fn identity_and_bytes_are_deterministic_across_source_order() {
    let temporary = tempfile::tempdir().unwrap();
    let sources = fixture_sources(temporary.path(), &[9, 8, 7]);
    let output_a = temporary.path().join("a.mspec");
    let output_b = temporary.path().join("b.mspec");

    let first = write_artifact(&output_a, "fixture:model", source_checkpoint(), &sources).unwrap();
    let mut reversed = sources.clone();
    reversed.reverse();
    let second =
        write_artifact(&output_b, "fixture:model", source_checkpoint(), &reversed).unwrap();

    assert_eq!(first.directory.identity, second.directory.identity);
    assert_eq!(fs::read(output_a).unwrap(), fs::read(output_b).unwrap());
}

#[test]
fn refuses_to_replace_an_existing_output() {
    let temporary = tempfile::tempdir().unwrap();
    let sources = fixture_sources(temporary.path(), &[1, 2, 3]);
    let output = temporary.path().join("existing.mspec");
    fs::write(&output, b"keep these bytes").unwrap();

    assert!(write_artifact(&output, "fixture:model", source_checkpoint(), &sources).is_err());
    assert_eq!(fs::read(output).unwrap(), b"keep these bytes");
}

#[test]
fn malformed_shapes_duplicate_names_and_missing_recipe_leave_no_output() {
    let temporary = tempfile::tempdir().unwrap();
    let valid_sources = fixture_sources(temporary.path(), &[1, 2, 3]);

    let mut malformed_shape = valid_sources.clone();
    malformed_shape[0].shape = vec![4];
    assert_failure_without_output(temporary.path(), "bad-shape.mspec", &malformed_shape);

    let mut duplicate_name = valid_sources.clone();
    duplicate_name[1].name = duplicate_name[0].name.clone();
    assert_failure_without_output(temporary.path(), "duplicate-name.mspec", &duplicate_name);

    let mut missing_recipe = valid_sources;
    missing_recipe[1].kind = ObjectKind::Config;
    assert_failure_without_output(temporary.path(), "missing-recipe.mspec", &missing_recipe);
}

fn assert_failure_without_output(directory: &Path, name: &str, sources: &[ObjectSource]) {
    let output = directory.join(name);
    assert!(write_artifact(&output, "fixture:model", source_checkpoint(), sources).is_err());
    assert!(!output.exists());
}

#[test]
fn streams_objects_larger_than_the_copy_buffer() {
    let temporary = tempfile::tempdir().unwrap();
    let data: Vec<u8> = (0..(64 * 1024 + 37))
        .map(|index| (index % 251) as u8)
        .collect();
    let sources = fixture_sources(temporary.path(), &data);
    let output = temporary.path().join("large.mspec");

    let written = write_artifact(&output, "fixture:model", source_checkpoint(), &sources).unwrap();
    let mut artifact = VerifiedArtifact::open(&output).unwrap();
    let mut copied = Vec::new();
    artifact.copy_object("a.tensor", &mut copied).unwrap();
    assert_eq!(copied, data);
    assert_eq!(fs::metadata(output).unwrap().len(), written.bytes);
}

#[test]
fn rejects_same_size_and_length_mutations_between_passes() {
    let same_size = tempfile::tempdir().unwrap();
    assert_mutation_rejected(same_size.path(), "same-size.mspec", |path| {
        fs::write(path, [9, 8, 7]).unwrap();
    });

    let changed_length = tempfile::tempdir().unwrap();
    assert_mutation_rejected(changed_length.path(), "changed-length.mspec", |path| {
        OpenOptions::new()
            .append(true)
            .open(path)
            .unwrap()
            .write_all(b"extra")
            .unwrap();
    });
}

fn assert_mutation_rejected(
    directory: &Path,
    output_name: &str,
    mutate_source: impl FnOnce(&Path),
) {
    let sources = fixture_sources(directory, &[1, 2, 3]);
    let (mut retained, mut inventory, payload_len) =
        open_sources("fixture:model", source_checkpoint(), &sources).unwrap();
    validate_for_write(&inventory, payload_len).unwrap();
    hash_sources(&mut retained, &mut inventory).unwrap();
    inventory.identity.weights_id =
        crate::artifact::weights_identity(&inventory, payload_len).unwrap();
    let directory_bytes = serde_json::to_vec(&inventory).unwrap();
    let directory_digest: [u8; 32] = Sha256::digest(&directory_bytes).into();
    let header = Header::new(directory_bytes.len() as u64, directory_digest).unwrap();
    let mut temporary = NamedTempFile::new_in(directory).unwrap();

    mutate_source(&sources[0].path);
    assert!(
        write_artifact_contents(
            &mut temporary,
            &header,
            &directory_bytes,
            &mut retained,
            &inventory,
            payload_len,
        )
        .is_err()
    );
    assert!(!directory.join(output_name).exists());
}
