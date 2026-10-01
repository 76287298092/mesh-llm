use super::*;
use serde_json::json;
use tempfile::TempDir;

struct Fixture {
    directory: TempDir,
    manifest: Value,
    objects: Vec<Vec<u8>>,
    payload: Vec<u8>,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let objects = vec![
            vec![0, 0, 128, 63],
            b"{\"vocab\":{}}".to_vec(),
            // Cross multiple streaming buffers without allocating a real model.
            vec![0xa5; BUFFER_BYTES * 2 + 7],
            vec![0x81, 0x02, 0x33, 0x44],
            vec![0x78, 0x56, 0x34, 0x12],
            vec![5, 0, 0, 0],
        ];
        let ids = [
            "auxiliary/000000",
            "resource/text/tokenizer.json",
            "weight/000001",
            "weight/000002",
            "weight/000003",
            "weight/000004",
        ];
        let formats = [
            "fp32",
            "resource",
            "fp8_e4m3fn_row_bf16",
            "q8_g32_fp16",
            "q4_g64_fp16",
            "int32",
        ];
        let mut payload = Vec::new();
        let mut entries = Vec::new();
        let mut storages = Vec::new();
        for (index, data) in objects.iter().enumerate() {
            payload.resize(align_up(payload.len() as u64).unwrap() as usize, 0);
            let name = format!("storage/{index:06}");
            entries.push(
                json!({"name": name, "offset": payload.len(), "bytes": data.len(),
                "sha256": hash(data), "source_id": ids[index]}),
            );
            let metadata = if formats[index] == "resource" {
                json!({"id": ids[index], "kind": "resource", "encoding": "raw_bytes_v1",
                    "bytes": data.len(), "storage": name})
            } else {
                json!({"id": ids[index], "kind": "tensor", "format": formats[index],
                    "layout": "opaque-pinned-layout", "shape": [], "bytes": data.len(),
                    "storage": name, "extension": {"retained": true}})
            };
            storages.push(metadata);
            payload.extend_from_slice(data);
        }
        let manifest = json!({
            "schema_version": 1, "profile": MODEL_ID,
            "source": {"artifact_sha256": SOURCE_SHA256, "artifact_bytes": SOURCE_BYTES,
                "reader_revision": READER_REVISION},
            "payload": {"path": "payload.bin", "bytes": payload.len(), "sha256": hash(&payload)},
            "objects": entries,
            "model": {
                "components": {"text": {"config": {"hidden_size": 5120},
                    "resources": {"tokenizer.json": ids[1]}, "proposal": {"domain": "indexed"}},
                    "mtp": {"config": {"architectures": ["Qwen3_5MTP"]}, "target": "text"}},
                "bindings": {"text/token_embedding": {"object": ids[2]},
                    "text/output_head": {"parts": [{"object": ids[2], "rows": [0, 1]}]},
                    "mtp/projection": {"object": ids[3]}, "proposal/head": {"object": ids[4]},
                    "proposal/token_ids": {"object": ids[5]}},
                "uses": [{"parameter": "text/output_head", "input": "mtp/final_hidden",
                    "auxiliaries": {"divisor": {"object": ids[0]}}, "activation_policy": "AllowA8"}],
                "storages": storages,
            },
            "preservation": {"requantized": false, "layout_transformed": false,
                "all_selected_object_bytes_copied": true, "excluded_components": ["vision", "dflash2"],
                "runtime_executable": false},
        });
        let fixture = Self {
            directory,
            manifest,
            objects,
            payload,
        };
        fixture.save();
        fixture
    }

    fn save(&self) {
        fs::write(self.directory.path().join("payload.bin"), &self.payload).unwrap();
        fs::write(
            self.directory.path().join("manifest.json"),
            self.manifest_bytes(),
        )
        .unwrap();
    }

    fn manifest_bytes(&self) -> Vec<u8> {
        let mut bytes = serde_json::to_vec_pretty(&self.manifest).unwrap();
        bytes.push(b'\n');
        bytes
    }

    fn output(&self) -> PathBuf {
        self.directory.path().join("out.mspec")
    }

    fn import(&self) -> Result<ImportReport> {
        convert(self.directory.path(), &self.output())
    }

    fn fails(&self, expected: &str) {
        let error = format!("{:#}", self.import().unwrap_err());
        assert!(
            error.contains(expected),
            "expected {expected:?}, got {error:?}"
        );
        assert!(
            !self.output().exists(),
            "invalid bundle must fail before publication"
        );
    }
}

fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

#[test]
fn tiny_closure_preserves_storage_manifest_provenance_and_identity() {
    let fixture = Fixture::new();
    let report = fixture.import().unwrap();
    assert_eq!(report.identity.model_id, MODEL_ID);
    assert_eq!(report.recipe_sha256, hash(&fixture.manifest_bytes()));
    assert!(
        report.full_byte_verification && report.payload_verified && report.model_artifact_verified
    );
    assert!(!report.model_executable);
    assert!(!report.requantized);
    assert!(!report.layout_transformed);
    assert_eq!(report.storage_objects, 6);
    assert_eq!(report.verified_objects, 7);
    assert_eq!(report.quantized_objects, 3);
    assert_eq!(
        report.quantized_bytes,
        fixture.objects[2..5]
            .iter()
            .map(|v| v.len() as u64)
            .sum::<u64>()
    );
    assert_eq!(
        report.verified_object_bytes,
        report.storage_bytes + fixture.manifest_bytes().len() as u64
    );
    assert_eq!(report.formats["resource/raw_bytes_v1"].objects, 1);
    let mut artifact =
        VerifiedArtifact::open_for_identity(&fixture.output(), &report.identity).unwrap();
    assert_eq!(artifact.directory().source.repository, SOURCE_REPOSITORY);
    assert_eq!(artifact.directory().source.revision, SOURCE_SHA256);
    assert!(crate::packages::qwen3_8_27b::inventory::validate(artifact.directory()).is_err());
    for (index, expected) in fixture.objects.iter().enumerate() {
        let name = format!("storage/{index:06}");
        let mut actual = Vec::new();
        artifact.copy_object(&name, &mut actual).unwrap();
        assert_eq!(&actual, expected);
        assert_eq!(report.storages[index].mspec_name, name);
        assert_eq!(report.storages[index].sha256, hash(expected));
        let entry = artifact
            .directory()
            .objects
            .iter()
            .find(|entry| entry.name == name)
            .unwrap();
        assert_eq!(entry.layout, STORAGE_LAYOUT);
        assert_eq!(entry.dtype, DType::U8);
        assert_eq!(entry.kind, ObjectKind::Tensor);
        assert_eq!(entry.shape, vec![expected.len() as u64]);
    }
    let mut recipe = Vec::new();
    artifact.copy_object("recipe.json", &mut recipe).unwrap();
    assert_eq!(recipe, fixture.manifest_bytes());
    assert!(parse_manifest(&recipe).is_ok());
}

#[test]
fn metadata_changes_identity_without_changing_payload() {
    let mut fixture = Fixture::new();
    let first = fixture.import().unwrap();
    fixture.manifest["model"]["components"]["text"]["config"]["hidden_size"] = json!(17);
    fixture.save();
    let second = convert(
        fixture.directory.path(),
        &fixture.directory.path().join("second.mspec"),
    )
    .unwrap();
    assert_ne!(first.identity.weights_id, second.identity.weights_id);
    assert_eq!(first.payload_sha256, second.payload_sha256);
}

#[test]
fn existing_artifact_is_never_replaced() {
    let fixture = Fixture::new();
    fs::write(fixture.output(), b"keep original").unwrap();
    assert!(
        fixture
            .import()
            .unwrap_err()
            .to_string()
            .contains("already exists")
    );
    assert_eq!(fs::read(fixture.output()).unwrap(), b"keep original");
}

#[test]
fn rejects_pins_versions_paths_and_preservation_claims() {
    let mutations = [
        ("/schema_version", json!(2), "schema version"),
        ("/profile", json!("other"), "profile"),
        (
            "/source/artifact_sha256",
            json!("0".repeat(64)),
            "SHA-256 pin",
        ),
        ("/source/artifact_bytes", json!(1), "byte-length pin"),
        (
            "/source/reader_revision",
            json!("0".repeat(40)),
            "revision pin",
        ),
        ("/payload/path", json!("../payload.bin"), "payload path"),
        ("/payload/path", json!("/tmp/payload.bin"), "payload path"),
        ("/preservation/requantized", json!(true), "must preserve"),
        (
            "/preservation/layout_transformed",
            json!(true),
            "must preserve",
        ),
        (
            "/preservation/runtime_executable",
            json!(true),
            "runtime execution",
        ),
        (
            "/preservation/all_selected_object_bytes_copied",
            json!(false),
            "every selected",
        ),
        (
            "/preservation/excluded_components",
            json!(["vision"]),
            "excluded",
        ),
    ];
    for (pointer, value, expected) in mutations {
        let mut fixture = Fixture::new();
        *fixture.manifest.pointer_mut(pointer).unwrap() = value;
        fixture.save();
        fixture.fails(expected);
    }
}

#[test]
fn strict_envelope_rejects_unknown_and_duplicate_fields() {
    for pointer in [
        "",
        "/source",
        "/payload",
        "/objects/0",
        "/model",
        "/preservation",
    ] {
        let mut fixture = Fixture::new();
        fixture.manifest.pointer_mut(pointer).unwrap()["extra"] = json!(true);
        fixture.save();
        fixture.fails("unknown field");
    }
    let fixture = Fixture::new();
    let text = String::from_utf8(fixture.manifest_bytes())
        .unwrap()
        .replacen(
            "\"schema_version\": 1",
            "\"schema_version\": 1, \"schema_version\": 1",
            1,
        );
    assert!(
        parse_manifest(text.as_bytes())
            .unwrap_err()
            .to_string()
            .contains("parse bundle")
    );
}

#[test]
fn rejects_missing_and_unknown_references() {
    for pointer in [
        "/model/bindings/proposal~1head/object",
        "/model/uses/0/auxiliaries/divisor/object",
        "/model/components/text/resources/tokenizer.json",
    ] {
        let mut fixture = Fixture::new();
        *fixture.manifest.pointer_mut(pointer).unwrap() = json!("not-exported");
        fixture.save();
        fixture.fails("unknown bundle object/resource reference");
    }
    let mut fixture = Fixture::new();
    fixture.manifest["model"]["bindings"]
        .as_object_mut()
        .unwrap()
        .remove("proposal/head");
    fixture.save();
    fixture.fails("missing required bundle binding");
}

#[test]
fn rejects_missing_or_extra_components_and_unreferenced_storage() {
    let mut fixture = Fixture::new();
    fixture.manifest["model"]["components"]
        .as_object_mut()
        .unwrap()
        .remove("mtp");
    fixture.save();
    fixture.fails("exactly text and mtp");
    let mut fixture = Fixture::new();
    fixture.manifest["model"]["components"]["vision"] = json!({});
    fixture.save();
    fixture.fails("exactly text and mtp");
    let mut fixture = Fixture::new();
    fixture.manifest["model"]["uses"] = json!([]); // Auxiliary object is no longer referenced.
    fixture.save();
    fixture.fails("outside its selected reference closure");
}

#[test]
fn rejects_non_string_and_empty_binding_references() {
    let mut fixture = Fixture::new();
    fixture.manifest["model"]["bindings"]["proposal/head"]["object"] = json!(17);
    fixture.save();
    fixture.fails("reference must be a string");
    let mut fixture = Fixture::new();
    fixture.manifest["model"]["bindings"]["proposal/head"] = json!({});
    fixture.save();
    fixture.fails("must reference storage");
}

#[test]
fn rejects_metadata_cardinality_names_ids_and_lengths() {
    let mutations = [
        ("/model/storages/0/bytes", json!(99), "byte-length mismatch"),
        (
            "/model/storages/0/storage",
            json!("storage/999999"),
            "storage name mismatch",
        ),
        ("/model/storages/0/id", json!("absent"), "unknown source id"),
        ("/objects/1/source_id", json!("auxiliary/000000"), "unique"),
        (
            "/objects/1/name",
            json!("storage/000000"),
            "unique and ordered",
        ),
    ];
    for (pointer, value, expected) in mutations {
        let mut fixture = Fixture::new();
        *fixture.manifest.pointer_mut(pointer).unwrap() = value;
        fixture.save();
        fixture.fails(expected);
    }
    let mut fixture = Fixture::new();
    fixture.manifest["model"]["storages"]
        .as_array_mut()
        .unwrap()
        .pop();
    fixture.save();
    fixture.fails("cardinality");
}

#[test]
fn rejects_overlap_misalignment_zero_overflow_and_out_of_range() {
    let mutations = [
        ("/objects/1/offset", json!(0), "nonoverlapping"),
        ("/objects/1/offset", json!(257), "aligned"),
        ("/objects/0/bytes", json!(0), "must not be empty"),
        ("/objects/1/bytes", json!(u64::MAX), "overflows"),
        (
            "/objects/0/bytes",
            json!(MAX_ARTIFACT_BYTES),
            "exceeds payload",
        ),
        (
            "/payload/bytes",
            json!(MAX_ARTIFACT_BYTES + 1),
            "size limit",
        ),
    ];
    for (pointer, value, expected) in mutations {
        let mut fixture = Fixture::new();
        *fixture.manifest.pointer_mut(pointer).unwrap() = value;
        fixture.save();
        fixture.fails(expected);
    }
}

#[test]
fn rejects_payload_digest_length_and_object_tamper_before_writing() {
    let mut fixture = Fixture::new();
    fixture.manifest["payload"]["sha256"] = json!("0".repeat(64));
    fixture.save();
    fixture.fails("payload SHA-256 mismatch");
    let mut fixture = Fixture::new();
    fixture.payload.pop();
    fixture.save();
    fixture.fails("payload byte-length mismatch");
    let mut fixture = Fixture::new();
    fixture.payload[0] ^= 1;
    fixture.manifest["payload"]["sha256"] = json!(hash(&fixture.payload));
    fixture.save();
    fixture.fails("object SHA-256 mismatch");
    let mut fixture = Fixture::new();
    fixture.manifest["objects"][0]["sha256"] = json!("0".repeat(64));
    fixture.save();
    fixture.fails("object SHA-256 mismatch");
}

#[test]
fn rejects_nonzero_padding_even_with_matching_whole_payload_hash() {
    let mut fixture = Fixture::new();
    fixture.payload[4] = 1;
    fixture.manifest["payload"]["sha256"] = json!(hash(&fixture.payload));
    fixture.save();
    fixture.fails("padding must be zero");
}

#[test]
fn rejects_oversized_manifest_before_allocation_or_output() {
    let fixture = Fixture::new();
    File::options()
        .write(true)
        .open(fixture.directory.path().join("manifest.json"))
        .unwrap()
        .set_len(MAX_MANIFEST_BYTES + 1)
        .unwrap();
    fixture.fails("4 MiB");
    assert!(
        parse_manifest(&vec![b' '; MAX_MANIFEST_BYTES as usize + 1])
            .unwrap_err()
            .to_string()
            .contains("4 MiB")
    );
}

#[test]
fn rejects_nonregular_source() {
    let fixture = Fixture::new();
    let path = fixture.directory.path().join("payload.bin");
    fs::remove_file(&path).unwrap();
    fs::create_dir(&path).unwrap();
    fixture.fails("regular, non-symlink");
}

#[cfg(unix)]
#[test]
fn rejects_symlink_sources_and_dangling_output() {
    use std::os::unix::fs::symlink;
    for name in ["manifest.json", "payload.bin"] {
        let fixture = Fixture::new();
        let external = tempfile::tempdir().unwrap();
        let source = fixture.directory.path().join(name);
        let target = external.path().join(name);
        fs::rename(&source, &target).unwrap();
        symlink(target, source).unwrap();
        fixture.fails("regular, non-symlink");
    }
    let fixture = Fixture::new();
    symlink(fixture.directory.path().join("absent"), fixture.output()).unwrap();
    assert!(
        fixture
            .import()
            .unwrap_err()
            .to_string()
            .contains("already exists")
    );
    assert!(
        fs::symlink_metadata(fixture.output())
            .unwrap()
            .file_type()
            .is_symlink()
    );
}
