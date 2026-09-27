use anyhow::{Result, ensure};
use mesh_llm_native_runtime::model_identity::ModelIdentity;
use serde::{Deserialize, Serialize};

pub const MAX_OBJECTS: usize = 65_536;
pub const ALIGNMENT: u64 = 256;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Directory {
    pub schema_version: u32,
    #[serde(deserialize_with = "deserialize_identity")]
    pub identity: ModelIdentity,
    pub recipe_sha256: String,
    pub source: SourceCheckpoint,
    pub objects: Vec<Object>,
}

// Keep this versioned container strict without changing additive runtime manifests.
fn deserialize_identity<'de, D>(deserializer: D) -> std::result::Result<ModelIdentity, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Identity {
        model_id: String,
        weights_id: String,
    }
    let value = Identity::deserialize(deserializer)?;
    Ok(ModelIdentity {
        model_id: value.model_id,
        weights_id: value.weights_id,
    })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceCheckpoint {
    pub repository: String,
    pub revision: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Object {
    pub name: String,
    pub kind: ObjectKind,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub layout: String,
    /// Byte offset relative to the beginning of the package payload.
    pub offset: u64,
    pub length: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectKind {
    Tensor,
    Tokenizer,
    Config,
    Recipe,
}

impl ObjectKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Tensor => "tensor",
            Self::Tokenizer => "tokenizer",
            Self::Config => "config",
            Self::Recipe => "recipe",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    F32,
    F16,
    Bf16,
    I8,
    U8,
    U4,
    #[serde(rename = "fp8_e4m3")]
    Fp8E4m3,
    #[serde(rename = "fp4_e2m1")]
    Fp4E2m1,
}

impl DType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::U4 => "u4",
            Self::Fp8E4m3 => "fp8_e4m3",
            Self::Fp4E2m1 => "fp4_e2m1",
        }
    }
}

impl Directory {
    pub fn validate(&self, payload_len: u64) -> Result<()> {
        ensure!(
            self.schema_version == 1,
            "unsupported artifact schema version"
        );
        self.identity.validate()?;
        validate_sha256(&self.recipe_sha256, "recipe_sha256")?;
        validate_source(&self.source)?;
        ensure!(
            (1..=MAX_OBJECTS).contains(&self.objects.len()),
            "artifact must contain between 1 and {MAX_OBJECTS} objects"
        );
        validate_objects(&self.objects, &self.recipe_sha256, payload_len)
    }
}

pub fn align_up(value: u64) -> Result<u64> {
    let remainder = value % ALIGNMENT;
    if remainder == 0 {
        return Ok(value);
    }
    value
        .checked_add(ALIGNMENT - remainder)
        .ok_or_else(|| anyhow::anyhow!("aligned offset overflows u64"))
}

fn validate_source(source: &SourceCheckpoint) -> Result<()> {
    validate_bounded_text(&source.repository, "source repository")?;
    let revision = &source.revision;
    ensure!(
        (revision.len() == 40 || revision.len() == 64) && is_lower_hex(revision),
        "source revision must be 40 or 64 lowercase hexadecimal characters"
    );
    Ok(())
}

fn validate_objects(objects: &[Object], recipe_sha256: &str, payload_len: u64) -> Result<()> {
    let mut previous_name: Option<&[u8]> = None;
    let mut previous_end = 0_u64;
    let mut recipe_count = 0_usize;
    let mut recipe_object_sha256 = None;

    for object in objects {
        if let Some(previous) = previous_name {
            ensure!(
                previous < object.name.as_bytes(),
                "artifact object names must be in strict UTF-8 byte order"
            );
        }
        validate_object(object)?;
        validate_object_extent(object, previous_end, payload_len)?;
        previous_end = object
            .offset
            .checked_add(object.length)
            .ok_or_else(|| anyhow::anyhow!("artifact object end overflows u64"))?;
        previous_name = Some(object.name.as_bytes());

        if object.kind == ObjectKind::Recipe {
            recipe_count += 1;
            recipe_object_sha256 = Some(object.sha256.as_str());
        }
    }

    ensure!(
        recipe_count == 1,
        "artifact must contain exactly one recipe object"
    );
    ensure!(
        recipe_object_sha256 == Some(recipe_sha256),
        "recipe object digest must match recipe_sha256"
    );
    ensure!(
        previous_end == payload_len,
        "artifact payload length must equal the final object end"
    );
    Ok(())
}

fn validate_object(object: &Object) -> Result<()> {
    validate_bounded_text(&object.name, "object name")?;
    validate_bounded_text(&object.layout, "object layout")?;
    validate_sha256(&object.sha256, "object sha256")?;
    let elements = shape_element_count(&object.shape)?;
    let expected_length = dtype_length(&object.dtype, elements)?;
    ensure!(
        object.length == expected_length,
        "object length does not match its dtype and shape"
    );
    if object.kind != ObjectKind::Tensor {
        ensure!(
            object.dtype == DType::U8,
            "non-tensor objects must use u8 dtype"
        );
        ensure!(
            object.shape.len() == 1,
            "non-tensor objects must have a one-dimensional shape"
        );
        ensure!(
            object.layout == "raw-v1",
            "non-tensor objects must use raw-v1 layout"
        );
    }
    Ok(())
}

fn validate_object_extent(object: &Object, previous_end: u64, payload_len: u64) -> Result<()> {
    ensure!(
        object.offset == align_up(previous_end)?,
        "object offset must equal the aligned previous object end"
    );
    let end = object
        .offset
        .checked_add(object.length)
        .ok_or_else(|| anyhow::anyhow!("artifact object end overflows u64"))?;
    ensure!(end <= payload_len, "artifact object exceeds payload length");
    Ok(())
}

fn shape_element_count(shape: &[u64]) -> Result<u64> {
    ensure!(
        (1..=8).contains(&shape.len()),
        "object shape must contain between 1 and 8 dimensions"
    );
    ensure!(
        shape.iter().all(|dimension| *dimension > 0),
        "object shape dimensions must be nonzero"
    );
    shape.iter().try_fold(1_u64, |elements, dimension| {
        elements
            .checked_mul(*dimension)
            .ok_or_else(|| anyhow::anyhow!("object shape element count overflows u64"))
    })
}

fn dtype_length(dtype: &DType, elements: u64) -> Result<u64> {
    let bytes_per_element = match dtype {
        DType::F32 => Some(4),
        DType::F16 | DType::Bf16 => Some(2),
        DType::I8 | DType::U8 | DType::Fp8E4m3 => Some(1),
        DType::U4 | DType::Fp4E2m1 => None,
    };
    let Some(bytes_per_element) = bytes_per_element else {
        return Ok(elements.div_ceil(2));
    };
    elements
        .checked_mul(bytes_per_element)
        .ok_or_else(|| anyhow::anyhow!("object byte length overflows u64"))
}

fn validate_bounded_text(value: &str, label: &str) -> Result<()> {
    ensure!(!value.is_empty(), "{label} must not be empty");
    ensure!(value.len() <= 1024, "{label} must be at most 1024 bytes");
    ensure!(
        value.trim() == value,
        "{label} must not have surrounding whitespace"
    );
    ensure!(
        !value.chars().any(char::is_control),
        "{label} must not contain control characters"
    );
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    ensure!(
        value.len() == 64 && is_lower_hex(value),
        "{label} must be 64 lowercase hexadecimal characters"
    );
    Ok(())
}

fn is_lower_hex(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::{ALIGNMENT, DType, Directory, Object, ObjectKind, SourceCheckpoint, align_up};
    use mesh_llm_native_runtime::model_identity::ModelIdentity;

    fn digest(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn valid_directory() -> Directory {
        let recipe_sha256 = digest('b');
        Directory {
            schema_version: 1,
            identity: ModelIdentity {
                model_id: "model-a".to_string(),
                weights_id: "opaque-weights-a".to_string(),
            },
            recipe_sha256: recipe_sha256.clone(),
            source: SourceCheckpoint {
                repository: "https://example.invalid/model-a".to_string(),
                revision: "a".repeat(40),
            },
            objects: vec![
                Object {
                    name: "config.json".to_string(),
                    kind: ObjectKind::Config,
                    dtype: DType::U8,
                    shape: vec![3],
                    layout: "raw-v1".to_string(),
                    offset: 0,
                    length: 3,
                    sha256: digest('a'),
                },
                Object {
                    name: "recipe.json".to_string(),
                    kind: ObjectKind::Recipe,
                    dtype: DType::U8,
                    shape: vec![64],
                    layout: "raw-v1".to_string(),
                    offset: 256,
                    length: 64,
                    sha256: recipe_sha256,
                },
                Object {
                    name: "weights/layer-000".to_string(),
                    kind: ObjectKind::Tensor,
                    dtype: DType::U4,
                    shape: vec![3],
                    layout: "opaque-block-layout-v7".to_string(),
                    offset: 512,
                    length: 2,
                    sha256: digest('c'),
                },
            ],
        }
    }

    #[test]
    fn valid_multi_object_directory_accepts_odd_u4_and_opaque_tensor_layout() {
        let directory = valid_directory();
        assert!(directory.validate(514).is_ok());
        let mut fp4_directory = directory;
        fp4_directory.objects[2].dtype = DType::Fp4E2m1;
        assert!(fp4_directory.validate(514).is_ok());
    }

    #[test]
    fn dtype_tags_match_the_wire_schema() {
        let values = [
            (DType::F32, "f32"),
            (DType::F16, "f16"),
            (DType::Bf16, "bf16"),
            (DType::I8, "i8"),
            (DType::U8, "u8"),
            (DType::U4, "u4"),
            (DType::Fp8E4m3, "fp8_e4m3"),
            (DType::Fp4E2m1, "fp4_e2m1"),
        ];
        for (dtype, expected) in values {
            assert_eq!(dtype.as_str(), expected);
            assert_eq!(
                serde_json::to_string(&dtype).unwrap(),
                format!("\"{expected}\"")
            );
        }
        assert_eq!(ObjectKind::Tensor.as_str(), "tensor");
        assert_eq!(ObjectKind::Tokenizer.as_str(), "tokenizer");
        assert_eq!(ObjectKind::Config.as_str(), "config");
        assert_eq!(ObjectKind::Recipe.as_str(), "recipe");
    }

    #[test]
    fn directory_rejects_unknown_fields_at_every_metadata_level() {
        let original = serde_json::to_value(valid_directory()).unwrap();
        for pointer in ["", "/identity", "/source", "/objects/0"] {
            let mut value = original.clone();
            value
                .pointer_mut(pointer)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .insert("unsupported_field".into(), serde_json::json!(true));
            assert!(serde_json::from_value::<Directory>(value).is_err());
        }
    }

    #[test]
    fn align_up_handles_boundaries_and_overflow() {
        assert_eq!(align_up(0).unwrap(), 0);
        assert_eq!(align_up(255).unwrap(), ALIGNMENT);
        assert_eq!(align_up(256).unwrap(), 256);
        assert_eq!(align_up(257).unwrap(), 512);
        assert!(align_up(u64::MAX).is_err());
    }

    #[test]
    fn validation_rejects_schema_versions_shapes_and_overflow() {
        let mut invalid = valid_directory();
        invalid.schema_version = 2;
        assert!(invalid.validate(514).is_err());

        let mut invalid = valid_directory();
        invalid.objects[2].shape = vec![0];
        assert!(invalid.validate(514).is_err());

        let mut invalid = valid_directory();
        invalid.objects[2].shape = vec![1; 9];
        assert!(invalid.validate(514).is_err());

        let mut invalid = valid_directory();
        invalid.objects[2].shape = vec![u64::MAX, 2];
        assert!(invalid.validate(514).is_err());

        let mut invalid = valid_directory();
        invalid.objects[2].dtype = DType::F32;
        invalid.objects[2].shape = vec![u64::MAX];
        assert!(invalid.validate(514).is_err());

        let mut offset_overflow = valid_directory();
        let recipe_sha256 = offset_overflow.recipe_sha256.clone();
        let recipe_length = u64::MAX - (ALIGNMENT - 1);
        offset_overflow.objects = vec![
            Object {
                name: "a-recipe".to_string(),
                kind: ObjectKind::Recipe,
                dtype: DType::U8,
                shape: vec![recipe_length],
                layout: "raw-v1".to_string(),
                offset: 0,
                length: recipe_length,
                sha256: recipe_sha256,
            },
            Object {
                name: "b-tensor".to_string(),
                kind: ObjectKind::Tensor,
                dtype: DType::U8,
                shape: vec![256],
                layout: "opaque-layout".to_string(),
                offset: recipe_length,
                length: 256,
                sha256: digest('d'),
            },
        ];
        assert!(offset_overflow.validate(u64::MAX).is_err());
    }

    #[test]
    fn validation_rejects_unsorted_and_duplicate_names() {
        let mut unsorted = valid_directory();
        unsorted.objects.swap(0, 1);
        assert!(unsorted.validate(514).is_err());

        let mut duplicate = valid_directory();
        duplicate.objects[2].name = duplicate.objects[1].name.clone();
        assert!(duplicate.validate(514).is_err());
    }

    #[test]
    fn validation_rejects_overlap_gaps_and_trailing_payload_bytes() {
        let mut overlap = valid_directory();
        overlap.objects[1].offset = 2;
        assert!(overlap.validate(514).is_err());

        let mut gap = valid_directory();
        gap.objects[1].offset = 512;
        assert!(gap.validate(514).is_err());

        let mut nonzero_start = valid_directory();
        nonzero_start.objects[0].offset = 256;
        assert!(nonzero_start.validate(514).is_err());

        assert!(valid_directory().validate(515).is_err());
        assert!(valid_directory().validate(512).is_err());
    }

    #[test]
    fn validation_rejects_bad_digests_and_recipe_mismatch_or_count() {
        let mut bad_digest = valid_directory();
        bad_digest.objects[0].sha256 = "A".repeat(64);
        assert!(bad_digest.validate(514).is_err());

        let mut bad_recipe_hash = valid_directory();
        bad_recipe_hash.recipe_sha256 = digest('d');
        assert!(bad_recipe_hash.validate(514).is_err());

        let mut no_recipe = valid_directory();
        no_recipe.objects[1].kind = ObjectKind::Config;
        assert!(no_recipe.validate(514).is_err());

        let mut malformed_hash = valid_directory();
        malformed_hash.recipe_sha256 = "f".repeat(63);
        assert!(malformed_hash.validate(514).is_err());
    }

    #[test]
    fn non_tensor_objects_require_raw_u8_vector_layout() {
        let mut invalid = valid_directory();
        invalid.objects[0].dtype = DType::F32;
        assert!(invalid.validate(514).is_err());

        let mut invalid = valid_directory();
        invalid.objects[0].shape = vec![1, 3];
        assert!(invalid.validate(514).is_err());

        let mut invalid = valid_directory();
        invalid.objects[0].layout = "json-v2".to_string();
        assert!(invalid.validate(514).is_err());
    }

    #[test]
    fn source_and_object_text_are_bounded_and_revisions_are_lower_hex() {
        let mut invalid = valid_directory();
        invalid.source.repository = "repo\nname".to_string();
        assert!(invalid.validate(514).is_err());

        let mut invalid = valid_directory();
        invalid.source.revision = "A".repeat(40);
        assert!(invalid.validate(514).is_err());

        let mut invalid = valid_directory();
        invalid.objects[2].layout = " ".to_string();
        assert!(invalid.validate(514).is_err());

        let mut invalid = valid_directory();
        invalid.objects[2].name = "n".repeat(1025);
        assert!(invalid.validate(514).is_err());
    }
}
