use super::{
    MAIN_FILE, MAIN_KEPT_COUNT, MAIN_TENSOR_COUNT, MAIN_VISUAL_COUNT, MODEL_ID, MTP_FILE,
    MTP_TENSOR_COUNT, SOURCE_PINS, SOURCE_REPO, SOURCE_REV, TENSOR_LAYOUT, TOTAL_KEPT_COUNT,
    auxiliary_sources, recipe_bytes, tensor_sources,
};
use crate::artifact::writer::{DType, ObjectKind};
use crate::checkpoint::safetensors::{TensorEntry, VerifiedTensorFile};
use std::path::Path;

const FAKE_TENSOR_SHA: &str = "abababababababababababababababababababababababababababababababab";

fn entry(name: String, index: usize, dtype: safetensors::Dtype) -> TensorEntry {
    let length = match dtype {
        safetensors::Dtype::F32 => 4,
        safetensors::Dtype::BF16 => 2,
        safetensors::Dtype::F8_E4M3 | safetensors::Dtype::U8 => 1,
        _ => 1,
    };
    TensorEntry {
        name,
        dtype,
        shape: vec![1],
        offset: (index as u64) * 16,
        length,
        sha256: FAKE_TENSOR_SHA.to_string(),
    }
}

fn fixture_file(name: &str, tensors: Vec<TensorEntry>) -> VerifiedTensorFile {
    let pin = SOURCE_PINS.iter().find(|pin| pin.name == name).unwrap();
    VerifiedTensorFile {
        file_len: pin.bytes,
        file_sha256: pin.sha256.to_string(),
        header_sha256: "cd".repeat(32),
        tensors,
    }
}

fn fixture_files() -> (VerifiedTensorFile, VerifiedTensorFile) {
    let mut main = Vec::with_capacity(MAIN_TENSOR_COUNT);
    for index in 0..MAIN_VISUAL_COUNT {
        main.push(entry(
            format!("model.visual.fixture.{index:04}"),
            index,
            safetensors::Dtype::F16,
        ));
    }
    for index in 0..MAIN_KEPT_COUNT {
        let name = match index {
            0 => "lm_head.weight".to_string(),
            1 => "model.language_model.dtype.bf16".to_string(),
            2 => "model.language_model.dtype.fp8".to_string(),
            3 => "model.language_model.dtype.u8".to_string(),
            4 => "model.language_model.dtype.f32".to_string(),
            _ => format!("model.language_model.fixture.{index:04}"),
        };
        let dtype = match index % 4 {
            0 => safetensors::Dtype::F32,
            1 => safetensors::Dtype::BF16,
            2 => safetensors::Dtype::F8_E4M3,
            _ => safetensors::Dtype::U8,
        };
        main.push(entry(name, MAIN_VISUAL_COUNT + index, dtype));
    }
    let mtp = (0..MTP_TENSOR_COUNT)
        .map(|index| {
            entry(
                format!("mtp.fixture.{index:04}"),
                index,
                safetensors::Dtype::U8,
            )
        })
        .collect();
    (fixture_file(MAIN_FILE, main), fixture_file(MTP_FILE, mtp))
}

#[test]
fn source_pins_match_the_intake_hashes_and_sizes() {
    let upstream: serde_json::Value = serde_json::from_str(include_str!(
        "../../KNOWLEDGE/evidence/checkpoint-intake-20260927/upstream-pins.json"
    ))
    .unwrap();
    assert_eq!(upstream["id"], SOURCE_REPO);
    assert_eq!(upstream["sha"], SOURCE_REV);
    let files = upstream["siblings"].as_array().unwrap();
    assert_eq!(SOURCE_PINS.len(), files.len());
    for pin in SOURCE_PINS {
        let file = files
            .iter()
            .find(|file| file["rfilename"] == pin.name)
            .unwrap();
        assert_eq!(file["size"], pin.bytes);
        if let Some(lfs_hash) = file["lfs"]["sha256"].as_str() {
            assert_eq!(pin.sha256, lfs_hash);
        }
        assert_eq!(pin.sha256.len(), 64);
        assert!(
            pin.sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        );
    }
}

#[test]
fn recipe_bytes_are_deterministic_and_record_pins_and_scope() {
    let first = recipe_bytes().unwrap();
    assert_eq!(first, recipe_bytes().unwrap());
    let recipe: serde_json::Value = serde_json::from_slice(&first).unwrap();

    assert_eq!(recipe["version"], 1);
    assert_eq!(recipe["id"], "upstream-raw-v1");
    assert_eq!(recipe["model_id"], MODEL_ID);
    assert_eq!(recipe["source"]["repository"], SOURCE_REPO);
    assert_eq!(recipe["source"]["revision"], SOURCE_REV);
    assert_eq!(recipe["byte_layout"], TENSOR_LAYOUT);
    assert_eq!(recipe["tensor_bytes"], "preserved");
    assert_eq!(recipe["excluded_prefixes"][0], "model.visual.");
    assert_eq!(recipe["mtp"], "preserved-not-yet-executed");
    assert_eq!(recipe["source_files"].as_array().unwrap().len(), 8);
}

#[test]
fn tensor_sources_preserve_exact_ranges_digests_and_supported_dtypes() {
    let (main, mtp) = fixture_files();
    let root = Path::new("/checkpoint");
    let sources = tensor_sources(root, &main, &mtp).unwrap();

    assert_eq!(sources.len(), TOTAL_KEPT_COUNT);
    assert!(sources.windows(2).all(|pair| pair[0].name < pair[1].name));
    let cases = [
        (
            "tensors/lm_head.weight",
            DType::F32,
            root.join(MAIN_FILE),
            SourceRangeExpected {
                offset: 333 * 16,
                length: 4,
            },
        ),
        (
            "tensors/model.language_model.dtype.bf16",
            DType::Bf16,
            root.join(MAIN_FILE),
            SourceRangeExpected {
                offset: 334 * 16,
                length: 2,
            },
        ),
        (
            "tensors/model.language_model.dtype.fp8",
            DType::Fp8E4m3,
            root.join(MAIN_FILE),
            SourceRangeExpected {
                offset: 335 * 16,
                length: 1,
            },
        ),
        (
            "tensors/model.language_model.dtype.u8",
            DType::U8,
            root.join(MAIN_FILE),
            SourceRangeExpected {
                offset: 336 * 16,
                length: 1,
            },
        ),
        (
            "tensors/mtp.fixture.0000",
            DType::U8,
            root.join(MTP_FILE),
            SourceRangeExpected {
                offset: 0,
                length: 1,
            },
        ),
    ];
    for (name, dtype, path, range) in cases {
        let source = sources.iter().find(|source| source.name == name).unwrap();
        assert_eq!(source.kind, ObjectKind::Tensor);
        assert_eq!(source.dtype, dtype);
        assert_eq!(source.shape.as_slice(), &[1_u64]);
        assert_eq!(source.layout, TENSOR_LAYOUT);
        assert_eq!(source.path, path);
        assert_eq!(
            source
                .range
                .as_ref()
                .map(|value| (value.offset, value.length)),
            Some((range.offset, range.length))
        );
        assert_eq!(source.expected_sha256.as_deref(), Some(FAKE_TENSOR_SHA));
    }
    assert!(
        sources
            .iter()
            .all(|source| !source.name.contains("model.visual."))
    );
}

#[derive(Clone, Copy)]
struct SourceRangeExpected {
    offset: u64,
    length: u64,
}

#[test]
fn auxiliary_sources_cover_six_pinned_files_with_raw_layouts() {
    let root = Path::new("/checkpoint");
    let sources = auxiliary_sources(root);
    assert_eq!(sources.len(), 6);

    for source in &sources {
        let filename = source.path.file_name().unwrap().to_str().unwrap();
        let pin = SOURCE_PINS.iter().find(|pin| pin.name == filename).unwrap();
        assert_eq!(source.name, format!("assets/{filename}"));
        assert_eq!(source.path, root.join(filename));
        assert_eq!(source.dtype, DType::U8);
        assert_eq!(source.shape.as_slice(), &[pin.bytes]);
        assert_eq!(source.layout, "raw-v1");
        assert!(source.range.is_none());
        assert_eq!(source.expected_sha256.as_deref(), Some(pin.sha256));
        let expected_kind = match filename {
            "config.json" | "model.safetensors.index.json" | "generation_config.json" => {
                ObjectKind::Config
            }
            _ => ObjectKind::Tokenizer,
        };
        assert_eq!(source.kind, expected_kind);
    }
}

#[test]
fn rejects_mismatched_file_pins_and_tensor_counts() {
    let (main, mtp) = fixture_files();
    let root = Path::new("/checkpoint");

    let mut wrong_main_hash = main.clone();
    wrong_main_hash.file_sha256.replace_range(..1, "0");
    assert!(tensor_sources(root, &wrong_main_hash, &mtp).is_err());

    let mut wrong_mtp_size = mtp.clone();
    wrong_mtp_size.file_len -= 1;
    assert!(tensor_sources(root, &main, &wrong_mtp_size).is_err());

    let mut missing_tensor = main.clone();
    missing_tensor.tensors.pop();
    assert!(tensor_sources(root, &missing_tensor, &mtp).is_err());
}

#[test]
fn rejects_unexpected_namespaces_and_wrong_visual_count() {
    let (mut main, mtp) = fixture_files();
    main.tensors[MAIN_VISUAL_COUNT + 5].name = "unexpected.tensor".to_string();
    assert!(tensor_sources(Path::new("/checkpoint"), &main, &mtp).is_err());

    let (mut main, mtp) = fixture_files();
    main.tensors[MAIN_VISUAL_COUNT + 5].name = "model.visual.extra".to_string();
    assert!(tensor_sources(Path::new("/checkpoint"), &main, &mtp).is_err());

    let (main, mut mtp) = fixture_files();
    mtp.tensors[0].name = "not_mtp.tensor".to_string();
    assert!(tensor_sources(Path::new("/checkpoint"), &main, &mtp).is_err());
}

#[test]
fn rejects_unsupported_dtype_and_malformed_shape() {
    let (mut main, mtp) = fixture_files();
    main.tensors[MAIN_VISUAL_COUNT].dtype = safetensors::Dtype::F16;
    assert!(tensor_sources(Path::new("/checkpoint"), &main, &mtp).is_err());

    let (mut main, mtp) = fixture_files();
    main.tensors[MAIN_VISUAL_COUNT].shape.clear();
    assert!(tensor_sources(Path::new("/checkpoint"), &main, &mtp).is_err());

    let (mut main, mtp) = fixture_files();
    main.tensors[MAIN_VISUAL_COUNT].shape = vec![0];
    assert!(tensor_sources(Path::new("/checkpoint"), &main, &mtp).is_err());
}
