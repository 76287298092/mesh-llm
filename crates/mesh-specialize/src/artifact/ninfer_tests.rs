use super::*;
use serde_json::json;
use std::io::{self, Seek, SeekFrom, Write};
use tempfile::NamedTempFile;

fn tensor(id: &str, shape: &[u64], format: &str, layout: &str, offset: u64, bytes: u64) -> Value {
    json!({"id":id, "kind":"tensor", "shape":shape, "format":format, "layout":layout, "offset":offset, "bytes":bytes})
}

fn resource(id: &str, offset: u64, bytes: u64) -> Value {
    json!({"id":id, "kind":"resource", "encoding":"raw_bytes_v1", "offset":offset, "bytes":bytes})
}

fn document(objects: Vec<Value>, payload_bytes: u64) -> Value {
    json!({
        "components":{"text":{"config":{}}}, "objects": objects,
        "bindings":{}, "uses":[], "files":[{"path":null,"payload_bytes":payload_bytes}],
        "metadata":{"name":"synthetic"}, "provenance":{}
    })
}

fn small_document() -> Value {
    document(
        vec![tensor("weight", &[2, 4], "bf16", "contiguous_le_v1", 0, 16)],
        16,
    )
}

fn parse(value: &Value) -> Result<Directory> {
    decode_directory(&serde_json::to_vec(value)?)
}

fn raw_fixture(json: &[u8], payload: &[u8]) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(MAGIC).unwrap();
    file.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&[0x19; 16]).unwrap();
    file.write_all(json).unwrap();
    let boundary = (32 + json.len()).div_ceil(4096) * 4096;
    file.write_all(&vec![0; boundary - 32 - json.len()])
        .unwrap();
    file.write_all(payload).unwrap();
    file.flush().unwrap();
    file
}

fn fixture(value: &Value, payload: &[u8]) -> NamedTempFile {
    raw_fixture(&serde_json::to_vec(value).unwrap(), payload)
}

#[test]
fn valid_encodings_and_scalar_preserve_metadata() {
    // Expected bytes are hand-calculated, not obtained from production geometry.
    let objects = vec![
        resource("tokenizer", 0, 3),
        tensor("scalar", &[], "fp32", "contiguous_le_v1", 256, 4),
        tensor("direct", &[3, 5], "bf16", "contiguous_le_v1", 512, 30),
        tensor(
            "fp8",
            &[3, 5],
            "fp8_e4m3fn_row_bf16",
            "row_scale_v1",
            768,
            262,
        ),
        tensor(
            "nvfp4",
            &[128, 64],
            "nvfp4",
            "block_scale_k16_m128x4_v1",
            1280,
            4612,
        ),
        tensor(
            "q4",
            &[3, 129],
            "q4_g64_fp16",
            "row_split_k128_v1",
            6144,
            536,
        ),
        tensor(
            "q5",
            &[3, 129],
            "q5_g64_fp16",
            "row_split_k128_v1",
            6912,
            792,
        ),
        tensor(
            "q6",
            &[3, 129],
            "q6_g64_fp16",
            "row_split_k128_v1",
            7936,
            792,
        ),
        tensor(
            "q8",
            &[3, 129],
            "q8_g32_fp16",
            "row_split_k128_v1",
            8960,
            816,
        ),
        tensor("indices", &[3], "int32", "contiguous_le_v1", 9984, 12),
    ];
    let mut value = document(objects, 9996);
    value["components"]["text"]["resources"] = json!({"tokenizer.json":"tokenizer"});
    value["components"]["text"]["proposal"] = json!({"domain":"indexed", "rows":3});
    value["bindings"] = json!({"scalar":{"object":"scalar"}, "packed":{"object":"q8"}});
    value["uses"] = json!([{"parameter":"packed", "input":"hidden", "activation_policy":"AllowA8", "auxiliaries":{"scale":{"object":"scalar"}}}]);
    let payload: Vec<_> = (0..9996).map(|i| (i % 251) as u8).collect();
    let file = fixture(&value, &payload);
    let artifact = NinferArtifact::open(file.path()).unwrap();
    assert_eq!(artifact.artifact_id(), [0x19; 16]);
    assert_eq!(artifact.file_bytes(), artifact.payload_offset() + 9996);
    assert_eq!(artifact.payload_offset() % 4096, 0);
    assert_eq!(
        artifact.directory().objects[1].logical_elements().unwrap(),
        1
    );
    assert!(
        artifact
            .directory()
            .objects
            .iter()
            .all(Object::encoding_supported)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(artifact.directory_json()).unwrap(),
        value
    );
    assert_eq!(serde_json::to_value(artifact.directory()).unwrap(), value);
}

#[test]
fn streaming_copy_range_and_whole_file_hash() {
    let payload: Vec<_> = (0..(COPY_BYTES * 2 + 17))
        .map(|i| (i % 251) as u8)
        .collect();
    let value = document(
        vec![resource("resource", 0, payload.len() as u64)],
        payload.len() as u64,
    );
    let file = fixture(&value, &payload);
    let mut artifact = NinferArtifact::open(file.path()).unwrap();
    let mut copied = BoundedWrites::default();
    assert_eq!(
        artifact.copy_object("resource", &mut copied).unwrap(),
        payload.len() as u64
    );
    assert_eq!(copied.bytes, payload);
    assert!(copied.calls >= 3);
    let position = artifact.file.stream_position().unwrap();
    let expected = hex::encode(Sha256::digest(std::fs::read(file.path()).unwrap()));
    assert_eq!(artifact.file_sha256().unwrap(), expected);
    assert_eq!(artifact.file.stream_position().unwrap(), position);
    let mut range = Vec::new();
    assert_eq!(
        artifact
            .read_object_range("resource", 65530, 30, &mut range)
            .unwrap(),
        30
    );
    assert_eq!(range, payload[65530..65560]);
    assert_eq!(
        artifact
            .read_object_range("resource", payload.len() as u64, 0, &mut range)
            .unwrap(),
        0
    );
    assert!(
        artifact
            .read_object_range("resource", payload.len() as u64, 1, &mut range)
            .is_err()
    );
    assert!(
        artifact
            .read_object_range("resource", u64::MAX, 2, &mut range)
            .is_err()
    );
    assert!(artifact.copy_object("absent", &mut range).is_err());
}

#[derive(Default)]
struct BoundedWrites {
    bytes: Vec<u8>,
    calls: usize,
}
impl Write for BoundedWrites {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        assert!(bytes.len() <= COPY_BYTES);
        self.calls += 1;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn retained_descriptor_survives_path_replacement() {
    let mut file = fixture(&small_document(), &[1; 16]);
    let mut artifact = NinferArtifact::open(file.path()).unwrap();
    let moved = file.path().with_extension("retained");
    std::fs::rename(file.path(), &moved).unwrap();
    std::fs::write(file.path(), b"not an artifact").unwrap();
    let mut bytes = Vec::new();
    artifact.copy_object("weight", &mut bytes).unwrap();
    assert_eq!(bytes, vec![1; 16]);
    // Restore the original pathname so NamedTempFile owns cleanup of both names.
    std::fs::rename(&moved, file.path()).unwrap();
    file.flush().unwrap();
}

#[test]
fn same_length_mutation_is_not_misrepresented_as_hash_verification() {
    let mut file = fixture(&small_document(), &[1; 16]);
    let mut artifact = NinferArtifact::open(file.path()).unwrap();
    let before = artifact.file_sha256().unwrap();
    file.seek(SeekFrom::Start(artifact.payload_offset()))
        .unwrap();
    file.write_all(&[2; 16]).unwrap();
    let mut bytes = Vec::new();
    artifact.copy_object("weight", &mut bytes).unwrap();
    assert_eq!(bytes, vec![2; 16]);
    assert_ne!(artifact.file_sha256().unwrap(), before);
}

#[test]
fn changed_source_length_rejected_before_copy_or_hash() {
    for growth in [false, true] {
        let file = fixture(&small_document(), &[1; 16]);
        let mut artifact = NinferArtifact::open(file.path()).unwrap();
        let changed = if growth {
            artifact.file_bytes() + 1
        } else {
            artifact.file_bytes() - 1
        };
        file.as_file().set_len(changed).unwrap();
        let mut destination = Vec::new();
        assert!(artifact.copy_object("weight", &mut destination).is_err());
        assert!(destination.is_empty());
        assert!(artifact.file_sha256().is_err());
    }
}

#[test]
fn truncated_header_directory_and_payload_rejected() {
    for length in [0, 7, 15, 31, 34, 4095, 4100] {
        let file = fixture(&small_document(), &[0; 16]);
        file.as_file().set_len(length).unwrap();
        assert!(
            NinferArtifact::open(file.path()).is_err(),
            "length {length}"
        );
    }
    let broken_json = raw_fixture(b"{\"objects\":", &[0; 16]);
    assert!(NinferArtifact::open(broken_json.path()).is_err());
    let trailing = fixture(&small_document(), &[0; 17]);
    assert!(NinferArtifact::open(trailing.path()).is_err());
}

#[test]
fn header_magic_version_length_endianness_and_bound() {
    for magic in [*b"NINFER\0\x02", *b"NINPRT\0\x03", *b"NOTFER\0\x03"] {
        let mut file = fixture(&small_document(), &[0; 16]);
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(&magic).unwrap();
        assert!(NinferArtifact::open(file.path()).is_err());
    }
    for length in [0, MAX_DIRECTORY_BYTES + 1, u64::MAX, 1_u64 << 56] {
        let mut file = fixture(&small_document(), &[0; 16]);
        file.seek(SeekFrom::Start(8)).unwrap();
        file.write_all(&length.to_le_bytes()).unwrap();
        assert!(NinferArtifact::open(file.path()).is_err());
    }
    let mut value = small_document();
    value["files"][0]["payload_bytes"] = json!(u64::MAX);
    let file = fixture(&value, &[0; 16]);
    assert!(NinferArtifact::open(file.path()).is_err());
}

#[test]
fn multifile_and_all_non_null_entry_paths_are_rejected() {
    for path in [
        "../escape",
        "nested/part",
        "..",
        ".",
        "/absolute",
        "C:\\absolute",
        "sibling.ninfer",
        "",
    ] {
        let mut value = small_document();
        value["files"][0]["path"] = json!(path);
        assert!(parse(&value).is_err(), "entry {path}");
        value["files"][0]["path"] = Value::Null;
        value["files"]
            .as_array_mut()
            .unwrap()
            .push(json!({"path":path,"payload_bytes":8}));
        let file = fixture(&value, &[0; 16]);
        let error = NinferArtifact::open(file.path()).err().unwrap().to_string();
        assert!(error.contains("single-file"), "{error}");
    }
    let mut value = small_document();
    value["files"][0].as_object_mut().unwrap().remove("path");
    assert!(parse(&value).is_err());
}

#[test]
fn duplicate_keys_at_every_depth_rejected() {
    let original = serde_json::to_string(&small_document()).unwrap();
    for (from, to) in [
        ("\"config\":{}", "\"config\":{\"x\":1,\"x\":2}"),
        (
            "\"name\":\"synthetic\"",
            "\"name\":\"synthetic\",\"name\":\"other\"",
        ),
        ("\"bytes\":16", "\"bytes\":16,\"bytes\":16"),
        ("\"bindings\":{}", "\"bindings\":{},\"bindings\":{}"),
        ("\"config\":{}", "\"config\":{\"x\":1,\"\\u0078\":2}"),
    ] {
        assert!(original.contains(from));
        assert!(decode_directory(original.replace(from, to).as_bytes()).is_err());
    }
}

#[test]
fn duplicate_object_ids_and_use_names_rejected() {
    let mut value = document(vec![resource("same", 0, 1), resource("same", 1, 1)], 2);
    assert!(parse(&value).is_err());
    value = small_document();
    value["bindings"] = json!({"p":{"object":"weight"}});
    value["uses"] = json!([{"parameter":"p","input":"hidden"},{"parameter":"p","input":"hidden"}]);
    assert!(parse(&value).is_err());
    value["uses"][1]["input"] = json!("other_hidden");
    assert!(parse(&value).is_ok());
}

#[test]
fn strict_fields_kinds_and_numeric_types() {
    let base = small_document();
    let mutations = [
        ("kind", json!("other")),
        ("bytes", json!(16.0)),
        ("offset", json!(-1)),
        ("bytes", json!(true)),
        ("bytes", json!(0)),
        ("shape", json!([0])),
        ("shape", json!([1.5])),
        ("shape", json!([true])),
        ("encoding", json!("raw_bytes_v1")),
        ("extra", json!(1)),
    ];
    for (key, data) in mutations {
        let mut value = base.clone();
        value["objects"][0][key] = data;
        assert!(parse(&value).is_err(), "{key}: {value}");
    }
    for key in ["id", "kind", "format", "layout", "shape", "offset", "bytes"] {
        let mut value = base.clone();
        value["objects"][0].as_object_mut().unwrap().remove(key);
        assert!(parse(&value).is_err(), "missing {key}");
    }
    let mut value = document(vec![resource("r", 0, 1)], 1);
    value["objects"][0]["shape"] = json!([]);
    assert!(parse(&value).is_err());
}

#[test]
fn overflow_shapes_lengths_offsets_and_alignment_rejected() {
    for object in [
        tensor("weight", &[u64::MAX, 2], "bf16", "contiguous_le_v1", 0, 16),
        tensor("weight", &[u64::MAX], "fp32", "contiguous_le_v1", 0, 16),
        tensor(
            "weight",
            &[1, u64::MAX],
            "q4_g64_fp16",
            "row_split_k128_v1",
            0,
            16,
        ),
        tensor(
            "weight",
            &[1, u64::MAX],
            "fp8_e4m3fn_row_bf16",
            "row_scale_v1",
            0,
            16,
        ),
        tensor("weight", &[1], "bf16", "contiguous_le_v1", u64::MAX - 1, 2),
        tensor("weight", &[1], "bf16", "contiguous_le_v1", 1, 2),
        tensor("weight", &[1; 17], "bf16", "contiguous_le_v1", 0, 2),
    ] {
        assert!(parse(&document(vec![object], u64::MAX)).is_err());
    }
}

#[test]
fn object_order_overlap_and_payload_bounds_rejected() {
    for objects in [
        vec![resource("a", 1, 2), resource("b", 0, 1)],
        vec![resource("a", 0, 2), resource("b", 1, 1)],
        vec![resource("a", 1, 3)],
    ] {
        assert!(parse(&document(objects, 3)).is_err());
    }
    // Unused padding and unaligned raw resources are legal wire data.
    assert!(parse(&document(vec![resource("a", 1, 1)], 3)).is_ok());
}

#[test]
fn invalid_known_geometry_is_not_treated_as_unknown() {
    for object in [
        tensor("w", &[2, 4], "bf16", "contiguous_le_v1", 0, 15),
        tensor("w", &[2, 4], "fp32", "row_scale_v1", 0, 264),
        tensor("w", &[], "fp8_e4m3fn_row_bf16", "row_scale_v1", 0, 258),
        tensor(
            "w",
            &[127, 64],
            "nvfp4",
            "block_scale_k16_m128x4_v1",
            0,
            4612,
        ),
        tensor(
            "w",
            &[128, 63],
            "nvfp4",
            "block_scale_k16_m128x4_v1",
            0,
            4612,
        ),
        tensor("w", &[3, 129], "q5_g64_fp16", "row_split_k128_v1", 0, 791),
    ] {
        assert!(parse(&document(vec![object], 65536)).is_err());
    }
}

#[test]
fn unknown_representations_are_metadata_only() {
    for (format, layout) in [
        ("future", "row_scale_v1"),
        ("bf16", "future"),
        ("future", "future"),
    ] {
        let directory = parse(&document(vec![tensor("w", &[1], format, layout, 0, 7)], 7)).unwrap();
        assert!(!directory.objects[0].encoding_supported());
        assert!(directory.objects[0].require_supported_encoding().is_err());
    }
    let mut value = document(vec![resource("r", 0, 1)], 1);
    value["objects"][0]["encoding"] = json!("future");
    assert!(!parse(&value).unwrap().objects[0].encoding_supported());
}

#[test]
fn logical_element_spans_and_shared_fused_parent_views() {
    let mut value = document(
        vec![tensor(
            "fused",
            &[4, 8],
            "fp8_e4m3fn_row_bf16",
            "row_scale_v1",
            0,
            264,
        )],
        264,
    );
    value["bindings"] = json!({
        "first":{"parts":[{"object":"fused","range":[0,16]}]},
        "second":{"parts":[{"object":"fused","range":[16,32]}]},
        "alias":{"parts":[{"object":"fused","range":[0,16]}]},
        "whole":{"object":"fused"},
        "joined":{"parts":[{"object":"fused","range":[16,32]},{"object":"fused","range":[0,16]}]}
    });
    assert!(parse(&value).is_ok());
    for range in [
        json!([0, 264]),
        json!([4, 4]),
        json!([3, 2]),
        json!([0, 33]),
        json!([0, 1.5]),
        json!([-1, 2]),
        json!([0, 1, 2]),
    ] {
        let mut invalid = value.clone();
        invalid["bindings"]["first"]["parts"][0]["range"] = range;
        assert!(parse(&invalid).is_err());
    }
    value["bindings"]["joined"]["parts"][1]["range"] = json!([8, 24]);
    assert!(parse(&value).is_err());
}

#[test]
fn binding_schema_and_dangling_references_rejected() {
    for binding in [
        json!({"object":"absent"}),
        json!({"parts":[]}),
        json!({"object":"weight","parts":[]}),
        json!({"parts":[{"object":"weight","range":[0,1],"extra":true}]}),
        json!({"object":"weight","range":[0,1]}),
        json!({"parts":[{"object":"absent","range":[0,1]}]}),
    ] {
        let mut value = small_document();
        value["bindings"] = json!({"p":binding});
        assert!(parse(&value).is_err());
    }
    let mut value = document(vec![resource("r", 0, 1)], 1);
    value["bindings"] = json!({"p":{"object":"r"}});
    assert!(parse(&value).is_err());
}

#[test]
fn use_auxiliary_and_component_references_are_checked() {
    let mut base = small_document();
    base["bindings"] = json!({"p":{"object":"weight"}});
    base["objects"]
        .as_array_mut()
        .unwrap()
        .push(resource("resource", 16, 1));
    base["files"][0]["payload_bytes"] = json!(17);
    for usage in [
        json!({"parameter":"absent","input":"hidden"}),
        json!({"parameter":"p","input":"hidden","activation_policy":"unknown"}),
        json!({"parameter":"p","input":"hidden","activation_policy":null}),
        json!({"parameter":"p","input":"hidden","auxiliaries":{"scale":{"object":"absent"}}}),
        json!({"parameter":"p","input":"hidden","auxiliaries":{"scale":{"object":"resource"}}}),
        json!({"parameter":"p","input":"hidden","auxiliaries":{"scale":{"parts":[{"object":"weight","range":[0,9]}]}}}),
    ] {
        let mut value = base.clone();
        value["uses"] = json!([usage]);
        assert!(parse(&value).is_err());
    }
    for component in [
        json!({"config":{}, "target":"absent"}),
        json!({"config":{}, "resources":{"tokenizer":"absent"}}),
        json!({"config":{}, "resources":{"tokenizer":"weight"}}),
        json!({"config":{},"proposal":{"domain":"full","rows":1}}),
        json!({"config":{},"proposal":{"domain":"indexed","rows":0}}),
        json!({"config":[]}),
    ] {
        let mut value = base.clone();
        value["components"]["text"] = component;
        assert!(parse(&value).is_err());
    }
}

#[test]
fn json_depth_trailing_data_and_nonfinite_are_rejected() {
    assert!(decode_directory(format!("{} null", small_document()).as_bytes()).is_err());
    let nested = format!("{}0{}", "[".repeat(150), "]".repeat(150));
    assert!(decode_directory(nested.as_bytes()).is_err());
    assert!(decode_directory(b"{\"metadata\":{\"number\":NaN}}").is_err());
    assert!(decode_directory(&[0xff, 0xfe]).is_err());
}

#[test]
fn bounded_counts_are_checked_without_large_payloads() {
    let objects: Vec<_> = (0..65_537)
        .map(|index| resource(&index.to_string(), index, 1))
        .collect();
    assert!(parse(&document(objects, 65_537)).is_err());
    let mut value = small_document();
    let bindings = (0..65_537)
        .map(|i| (i.to_string(), json!({"object":"weight"})))
        .collect();
    value["bindings"] = Value::Object(bindings);
    assert!(parse(&value).is_err());
}
