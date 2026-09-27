use super::{MAX_ARTIFACT_BYTES, VerifiedArtifact};
use crate::artifact::{
    header::{HEADER_LEN, Header},
    schema::{DType, Directory, Object, ObjectKind, SourceCheckpoint, align_up},
    weights_identity,
};
use mesh_llm_native_runtime::model_identity::ModelIdentity;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::Path,
};

struct ArtifactFixture {
    identity: ModelIdentity,
    payload_offset: u64,
    directory_len: u64,
    file_len: u64,
    tensor_data: Vec<u8>,
}

fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn write_fixture(path: &Path, tensor_data: &[u8], fake_weights_identity: bool) -> ArtifactFixture {
    let recipe_data = b"recipe-v1";
    let recipe_offset = align_up(tensor_data.len() as u64).unwrap();
    let payload_len = recipe_offset + recipe_data.len() as u64;
    let mut directory = Directory {
        schema_version: 1,
        identity: ModelIdentity {
            model_id: "fixture:model".to_string(),
            weights_id: "pending".to_string(),
        },
        recipe_sha256: digest(recipe_data),
        source: SourceCheckpoint {
            repository: "test/model".to_string(),
            revision: "a".repeat(40),
        },
        objects: vec![
            Object {
                name: "a.weights".to_string(),
                kind: ObjectKind::Tensor,
                dtype: DType::U8,
                shape: vec![tensor_data.len() as u64],
                layout: "row-major-v1".to_string(),
                offset: 0,
                length: tensor_data.len() as u64,
                sha256: digest(tensor_data),
            },
            Object {
                name: "z.recipe".to_string(),
                kind: ObjectKind::Recipe,
                dtype: DType::U8,
                shape: vec![recipe_data.len() as u64],
                layout: "raw-v1".to_string(),
                offset: recipe_offset,
                length: recipe_data.len() as u64,
                sha256: digest(recipe_data),
            },
        ],
    };
    let canonical_weights_id = weights_identity(&directory, payload_len).unwrap();
    directory.identity.weights_id = if fake_weights_identity {
        "fake-weights-identity".to_string()
    } else {
        canonical_weights_id
    };
    let directory_bytes = serde_json::to_vec(&directory).unwrap();
    let directory_len = directory_bytes.len() as u64;
    let directory_sha256: [u8; 32] = Sha256::digest(&directory_bytes).into();
    let header = Header::new(directory_len, directory_sha256).unwrap();
    let payload_offset = header.payload_offset;

    let mut file = File::create(path).unwrap();
    file.write_all(&header.encode().unwrap()).unwrap();
    file.write_all(&directory_bytes).unwrap();
    let directory_padding = (payload_offset - HEADER_LEN as u64 - directory_len) as usize;
    file.write_all(&vec![0; directory_padding]).unwrap();
    file.write_all(tensor_data).unwrap();
    file.write_all(&vec![0; (recipe_offset as usize) - tensor_data.len()])
        .unwrap();
    file.write_all(recipe_data).unwrap();
    file.flush().unwrap();
    let file_len = file.metadata().unwrap().len();

    ArtifactFixture {
        identity: directory.identity,
        payload_offset,
        directory_len,
        file_len,
        tensor_data: tensor_data.to_vec(),
    }
}

fn write_byte(path: &Path, offset: u64, byte: u8) {
    let mut file = OpenOptions::new().write(true).open(path).unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(&[byte]).unwrap();
    file.flush().unwrap();
}

#[test]
fn opens_verified_artifact_for_exact_identity_and_copies_objects() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fixture.mspec");
    let fixture = write_fixture(&path, &[1, 2, 3], false);
    let mut artifact = VerifiedArtifact::open_for_identity(&path, &fixture.identity).unwrap();
    assert_eq!(artifact.identity(), &fixture.identity);
    assert_eq!(artifact.directory().objects.len(), 2);

    let mut weights = Vec::new();
    assert_eq!(artifact.copy_object("a.weights", &mut weights).unwrap(), 3);
    assert_eq!(weights, fixture.tensor_data);

    let mut recipe = Vec::new();
    assert_eq!(artifact.copy_object("z.recipe", &mut recipe).unwrap(), 9);
    assert_eq!(recipe, b"recipe-v1".to_vec());
}

#[test]
fn rejects_missing_objects() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fixture.mspec");
    write_fixture(&path, &[1, 2, 3], false);
    let mut artifact = VerifiedArtifact::open(&path).unwrap();
    assert!(artifact.copy_object("missing", &mut Vec::new()).is_err());
}

#[test]
fn rejects_wrong_model_and_weights_identities() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fixture.mspec");
    let fixture = write_fixture(&path, &[1, 2, 3], false);
    let artifact = VerifiedArtifact::open(&path).unwrap();
    let wrong_model = ModelIdentity {
        model_id: "fixture:other".to_string(),
        weights_id: fixture.identity.weights_id.clone(),
    };
    let wrong_weights = ModelIdentity {
        model_id: fixture.identity.model_id.clone(),
        weights_id: "other-weights".to_string(),
    };
    assert!(artifact.require_identity(&wrong_model).is_err());
    assert!(artifact.require_identity(&wrong_weights).is_err());
    assert!(VerifiedArtifact::open_for_identity(&path, &wrong_model).is_err());
    assert!(VerifiedArtifact::open_for_identity(&path, &wrong_weights).is_err());
}

#[test]
fn rejects_directory_checksum_tampering() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fixture.mspec");
    write_fixture(&path, &[1, 2, 3], false);
    write_byte(&path, HEADER_LEN as u64, b'[');
    let error = VerifiedArtifact::open(&path).err().unwrap().to_string();
    assert!(error.contains("directory checksum mismatch"));
}

#[test]
fn rejects_object_checksum_tampering_during_open() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fixture.mspec");
    let fixture = write_fixture(&path, &[1, 2, 3], false);
    write_byte(&path, fixture.payload_offset, 9);
    let error = VerifiedArtifact::open(&path).err().unwrap().to_string();
    assert!(error.contains("object checksum mismatch"));
}

#[test]
fn rejects_fake_declared_weights_identity_even_with_recomputed_header_checksum() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fixture.mspec");
    write_fixture(&path, &[1, 2, 3], true);
    let error = VerifiedArtifact::open(&path).err().unwrap().to_string();
    assert!(error.contains("declared weight identity"));
}

#[test]
fn rejects_truncated_and_trailing_files() {
    let directory = tempfile::tempdir().unwrap();
    let truncated_path = directory.path().join("truncated.mspec");
    let truncated = write_fixture(&truncated_path, &[1, 2, 3], false);
    OpenOptions::new()
        .write(true)
        .open(&truncated_path)
        .unwrap()
        .set_len(truncated.file_len - 1)
        .unwrap();
    assert!(VerifiedArtifact::open(&truncated_path).is_err());

    let trailing_path = directory.path().join("trailing.mspec");
    write_fixture(&trailing_path, &[1, 2, 3], false);
    OpenOptions::new()
        .append(true)
        .open(&trailing_path)
        .unwrap()
        .write_all(&[0])
        .unwrap();
    assert!(VerifiedArtifact::open(&trailing_path).is_err());
}

#[test]
fn rejects_nonzero_directory_padding_and_object_gap() {
    let directory = tempfile::tempdir().unwrap();
    let directory_padding_path = directory.path().join("directory-padding.mspec");
    let fixture = write_fixture(&directory_padding_path, &[1, 2, 3], false);
    let directory_padding_offset = HEADER_LEN as u64 + fixture.directory_len;
    assert!(directory_padding_offset < fixture.payload_offset);
    write_byte(&directory_padding_path, directory_padding_offset, 1);
    let error = VerifiedArtifact::open(&directory_padding_path)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("padding must be zero"));

    let object_gap_path = directory.path().join("object-gap.mspec");
    let fixture = write_fixture(&object_gap_path, &[1, 2, 3], false);
    let object_end = fixture.payload_offset + fixture.tensor_data.len() as u64;
    let next_object = fixture.payload_offset + align_up(fixture.tensor_data.len() as u64).unwrap();
    assert!(object_end < next_object);
    write_byte(&object_gap_path, object_end, 1);
    let error = VerifiedArtifact::open(&object_gap_path)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("padding must be zero"));
}

#[test]
fn rechecks_object_digest_before_copying_from_retained_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fixture.mspec");
    let fixture = write_fixture(&path, &[1, 2, 3], false);
    let mut artifact = VerifiedArtifact::open(&path).unwrap();
    write_byte(&path, fixture.payload_offset, 9);

    let mut destination = Vec::new();
    assert!(artifact.copy_object("a.weights", &mut destination).is_err());
    assert_eq!(destination, vec![9, 2, 3]);
}

#[test]
fn rejects_file_growth_after_verification_before_copy() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fixture.mspec");
    write_fixture(&path, &[1, 2, 3], false);
    let mut artifact = VerifiedArtifact::open(&path).unwrap();
    OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(&[0])
        .unwrap();

    let mut destination = Vec::new();
    assert!(artifact.copy_object("a.weights", &mut destination).is_err());
    assert!(destination.is_empty());
}

#[test]
fn rejects_sparse_artifacts_over_the_size_limit_before_reading() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("oversized.mspec");
    File::create(&path)
        .unwrap()
        .set_len(MAX_ARTIFACT_BYTES + 1)
        .unwrap();
    assert!(VerifiedArtifact::open(&path).is_err());
}

#[test]
fn streams_large_objects_through_copy_interface() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("large.mspec");
    let tensor_data: Vec<_> = (0..150_123).map(|index| (index % 251) as u8).collect();
    write_fixture(&path, &tensor_data, false);
    let mut artifact = VerifiedArtifact::open(&path).unwrap();
    let mut copied = Vec::new();
    assert_eq!(
        artifact.copy_object("a.weights", &mut copied).unwrap(),
        tensor_data.len() as u64
    );
    assert_eq!(copied, tensor_data);
}
