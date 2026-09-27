use super::{MAX_HEADER_BYTES, verify};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

fn write_fixture(path: &Path, header_json: &str, payload: &[u8]) -> (String, Vec<u8>) {
    let mut header = header_json.as_bytes().to_vec();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(payload);
    fs::write(path, &bytes).unwrap();
    (hex::encode(Sha256::digest(&bytes)), header)
}

fn valid_header() -> &'static str {
    r#"{"scalar":{"dtype":"F32","shape":[],"data_offsets":[4,8]},"empty":{"dtype":"U8","shape":[0],"data_offsets":[8,8]},"weights":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"__metadata__":{"format":"fixture"}}"#
}

#[test]
fn verifies_mixed_scalar_and_empty_tensors_with_independent_hashes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("mixed.safetensors");
    let payload = [11_u8, 12, 13, 14, 21, 22, 23, 24];
    let (expected_file_sha, padded_header) = write_fixture(&path, valid_header(), &payload);

    let verified = verify(&path, &expected_file_sha).unwrap();
    let payload_start = 8 + u64::try_from(padded_header.len()).unwrap();
    assert_eq!(verified.file_len, fs::metadata(&path).unwrap().len());
    assert_eq!(verified.file_sha256, expected_file_sha);
    assert_eq!(
        verified.header_sha256,
        hex::encode(Sha256::digest(&padded_header))
    );
    assert_eq!(
        verified
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>(),
        ["empty", "scalar", "weights"]
    );

    let empty = &verified.tensors[0];
    assert_eq!(empty.dtype, safetensors::Dtype::U8);
    assert_eq!(empty.shape, vec![0]);
    assert_eq!(empty.offset, payload_start + 8);
    assert_eq!(empty.length, 0);
    assert_eq!(empty.sha256, hex::encode(Sha256::digest(b"")));

    let scalar = &verified.tensors[1];
    assert_eq!(scalar.dtype, safetensors::Dtype::F32);
    assert!(scalar.shape.is_empty());
    assert_eq!(scalar.offset, payload_start + 4);
    assert_eq!(scalar.length, 4);
    assert_eq!(scalar.sha256, hex::encode(Sha256::digest(&payload[4..8])));

    let weights = &verified.tensors[2];
    assert_eq!(weights.dtype, safetensors::Dtype::BF16);
    assert_eq!(weights.shape, vec![2]);
    assert_eq!(weights.offset, payload_start);
    assert_eq!(weights.length, 4);
    assert_eq!(weights.sha256, hex::encode(Sha256::digest(&payload[..4])));
}

#[test]
fn rejects_invalid_expected_digest_and_whole_file_digest() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("valid.safetensors");
    let (digest, _) = write_fixture(&path, valid_header(), &[0; 8]);

    assert!(verify(&path, &"A".repeat(64)).is_err());
    assert!(verify(&path, &"0".repeat(64)).is_err());
    assert!(verify(&path, &digest[..63]).is_err());
}

#[test]
fn rejects_duplicate_tensor_and_metadata_keys() {
    let directory = tempfile::tempdir().unwrap();
    let duplicate_tensor = directory.path().join("duplicate-tensor.safetensors");
    reject_header(
        &duplicate_tensor,
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[1],
    );

    let duplicate_metadata = directory.path().join("duplicate-metadata.safetensors");
    reject_header(
        &duplicate_metadata,
        r#"{"__metadata__":{"k":"a","k":"b"},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[1],
    );
}

#[test]
fn rejects_malformed_headers_dtypes_shapes_and_offsets() {
    let directory = tempfile::tempdir().unwrap();
    reject_header(
        &directory.path().join("malformed.safetensors"),
        "{not-json",
        &[],
    );
    reject_header(
        &directory.path().join("bad-dtype.safetensors"),
        r#"{"x":{"dtype":"NOT_A_DTYPE","shape":[1],"data_offsets":[0,1]}}"#,
        &[1],
    );
    reject_header(
        &directory.path().join("bad-rank.safetensors"),
        r#"{"x":{"dtype":"U8","shape":[1,1,1,1,1,1,1,1,1],"data_offsets":[0,1]}}"#,
        &[1],
    );
    reject_header(
        &directory.path().join("bad-offset.safetensors"),
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#,
        &[1, 2],
    );
    reject_header(
        &directory.path().join("overlap.safetensors"),
        r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}"#,
        &[1, 2, 3],
    );
    reject_header(
        &directory.path().join("trailing.safetensors"),
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[1, 2],
    );
    reject_header(
        &directory.path().join("truncated.safetensors"),
        r#"{"x":{"dtype":"U32","shape":[1],"data_offsets":[0,4]}}"#,
        &[1, 2],
    );
}

#[test]
fn rejects_header_without_object_prefix_and_oversized_header() {
    let directory = tempfile::tempdir().unwrap();
    let non_object = directory.path().join("non-object.safetensors");
    let bytes = make_bytes(b"[]", &[]);
    fs::write(&non_object, &bytes).unwrap();
    assert!(verify(&non_object, &hex::encode(Sha256::digest(&bytes))).is_err());

    let oversized = directory.path().join("oversized.safetensors");
    let length = MAX_HEADER_BYTES + 1;
    let prefix = length.to_le_bytes();
    fs::write(&oversized, prefix).unwrap();
    assert!(verify(&oversized, &hex::encode(Sha256::digest(prefix))).is_err());
}

#[test]
fn streams_payloads_larger_than_the_hash_buffer() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("large.safetensors");
    let payload: Vec<u8> = (0_usize..(64 * 1024 + 31))
        .map(|index| u8::try_from(index % 251).unwrap())
        .collect();
    let header = format!(
        r#"{{"large":{{"dtype":"U8","shape":[{}],"data_offsets":[0,{}]}}}}"#,
        payload.len(),
        payload.len()
    );
    let (expected, _) = write_fixture(&path, &header, &payload);

    let verified = verify(&path, &expected).unwrap();
    assert_eq!(verified.tensors.len(), 1);
    assert_eq!(
        verified.tensors[0].length,
        u64::try_from(payload.len()).unwrap()
    );
    assert_eq!(
        verified.tensors[0].sha256,
        hex::encode(Sha256::digest(payload))
    );
}

fn reject_header(path: &Path, header: &str, payload: &[u8]) {
    let (expected, _) = write_fixture(path, header, payload);
    assert!(
        verify(path, &expected).is_err(),
        "expected {} to be rejected",
        path.display()
    );
}

fn make_bytes(header: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(payload);
    bytes
}
