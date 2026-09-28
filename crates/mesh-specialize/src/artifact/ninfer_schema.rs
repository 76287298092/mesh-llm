//! Typed v3 wire records and independent checked layout arithmetic.

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

const MAX_RECORDS: usize = 65_536;
const MAX_PARTS: usize = 131_072;
const TENSOR_ALIGNMENT: u64 = 256;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Directory {
    pub components: BTreeMap<String, Value>,
    pub objects: Vec<Object>,
    pub bindings: BTreeMap<String, Binding>,
    pub uses: Vec<Use>,
    pub files: Vec<FileRecord>,
    #[serde(default = "empty_object")]
    pub metadata: Value,
    #[serde(default = "empty_object")]
    pub provenance: Value,
}

/// Physical source record. Tensor offsets are payload-relative, not absolute.
/// Resource records have no shape/format/layout; their exposed shape is empty.
/// For tensors an empty shape is a valid scalar containing exactly one element.
#[derive(Clone, Debug)]
pub struct Object {
    pub id: String,
    pub kind: String,
    pub format: Option<String>,
    pub layout: Option<String>,
    pub shape: Vec<u64>,
    pub encoding: Option<String>,
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
enum WireObject {
    #[serde(rename = "tensor")]
    Tensor {
        id: String,
        format: String,
        layout: String,
        shape: Vec<u64>,
        offset: u64,
        bytes: u64,
    },
    #[serde(rename = "resource")]
    Resource {
        id: String,
        encoding: String,
        offset: u64,
        bytes: u64,
    },
}

impl<'de> Deserialize<'de> for Object {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Ok(match WireObject::deserialize(deserializer)? {
            WireObject::Tensor {
                id,
                format,
                layout,
                shape,
                offset,
                bytes,
            } => Self {
                id,
                kind: "tensor".to_owned(),
                format: Some(format),
                layout: Some(layout),
                shape,
                encoding: None,
                offset,
                bytes,
            },
            WireObject::Resource {
                id,
                encoding,
                offset,
                bytes,
            } => Self {
                id,
                kind: "resource".to_owned(),
                format: None,
                layout: None,
                shape: Vec::new(),
                encoding: Some(encoding),
                offset,
                bytes,
            },
        })
    }
}

impl Serialize for Object {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut value =
            json!({"id": self.id, "kind": self.kind, "offset": self.offset, "bytes": self.bytes});
        if self.kind == "tensor" {
            value["shape"] = json!(self.shape);
            value["format"] = json!(self.format);
            value["layout"] = json!(self.layout);
        } else {
            value["encoding"] = json!(self.encoding);
        }
        value.serialize(serializer)
    }
}

/// A tensor binding, never a resource reference. Parts are concatenated in
/// declaration order. A part's half-open range counts logical elements, not
/// encoded bytes (in particular not padded K or compressed code-plane bytes).
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged, deny_unknown_fields)]
pub enum Binding {
    Object { object: String },
    Parts { parts: Vec<Part> },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Part {
    pub object: String,
    pub range: [u64; 2],
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Use {
    pub parameter: String,
    pub input: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "present_policy"
    )]
    pub activation_policy: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub auxiliaries: BTreeMap<String, Binding>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FileRecord {
    // A deserialize hook makes an absent `path` an error, while allowing null.
    #[serde(deserialize_with = "required_path")]
    pub path: Option<String>,
    pub payload_bytes: u64,
}

fn required_path<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Option<String>, D::Error> {
    Option::<String>::deserialize(d)
}

fn present_policy<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<String>, D::Error> {
    String::deserialize(d).map(Some)
}

fn empty_object() -> Value {
    Value::Object(serde_json::Map::new())
}

pub(super) fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).context("NInfer integer addition overflow")
}
fn multiply(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .context("NInfer integer multiplication overflow")
}
pub(super) fn align(value: u64, alignment: u64) -> Result<u64> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        add(value, alignment - remainder)
    }
}

fn identifier(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty() && !value.contains('\0'),
        "NInfer identifier must be nonempty without NUL"
    );
    Ok(())
}

fn elements(shape: &[u64]) -> Result<u64> {
    ensure!(shape.len() <= 16, "NInfer tensor rank exceeds 16");
    shape.iter().try_fold(1, |product, &dimension| {
        ensure!(dimension > 0, "NInfer shape dimensions must be positive");
        multiply(product, dimension)
    })
}

impl Object {
    pub fn logical_elements(&self) -> Result<u64> {
        ensure!(
            self.kind == "tensor",
            "resource objects have no logical tensor elements"
        );
        elements(&self.shape)
    }

    /// Whether this reader recognizes AND validates the byte representation.
    /// This is not model/kernel qualification. Unknown formats/layouts/encodings
    /// remain inspectable metadata but MUST NOT be executed by assuming a codec.
    pub fn encoding_supported(&self) -> bool {
        self.require_supported_encoding().is_ok()
    }

    pub fn require_supported_encoding(&self) -> Result<()> {
        if self.kind == "resource" {
            ensure!(
                self.encoding.as_deref() == Some("raw_bytes_v1"),
                "unsupported NInfer resource encoding"
            );
            return Ok(());
        }
        let expected = self
            .encoded_size()?
            .context("unsupported NInfer tensor format/layout; metadata-only object")?;
        ensure!(
            self.bytes == expected,
            "NInfer object {} encoded length mismatch: expected {expected}, got {}",
            self.id,
            self.bytes
        );
        ensure!(
            self.offset.is_multiple_of(TENSOR_ALIGNMENT),
            "NInfer tensor {} is not 256-byte aligned",
            self.id
        );
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        identifier(&self.id)?;
        ensure!(self.bytes > 0, "NInfer object {} is empty", self.id);
        add(self.offset, self.bytes)?;
        if self.kind == "resource" {
            identifier(
                self.encoding
                    .as_deref()
                    .context("missing resource encoding")?,
            )?;
            return Ok(());
        }
        self.logical_elements()?;
        identifier(self.format.as_deref().context("missing tensor format")?)?;
        identifier(self.layout.as_deref().context("missing tensor layout")?)?;
        if known_layout(self.layout.as_deref().unwrap_or_default()) {
            ensure!(
                self.offset.is_multiple_of(TENSOR_ALIGNMENT),
                "NInfer tensor {} is not 256-byte aligned",
                self.id
            );
        }
        if let Some(expected) = self.encoded_size()? {
            ensure!(
                self.bytes == expected,
                "NInfer object {} encoded length mismatch: expected {expected}, got {}",
                self.id,
                self.bytes
            );
        }
        Ok(())
    }

    fn encoded_size(&self) -> Result<Option<u64>> {
        let format = self.format.as_deref().context("missing tensor format")?;
        let layout = self.layout.as_deref().context("missing tensor layout")?;
        // Future codecs can be retained for diagnostics; neither a familiar
        // layout name nor format name by itself grants executable support.
        if !known_layout(layout) || !known_format(format) {
            return Ok(None);
        }
        let count = self.logical_elements()?;
        let size = match (layout, format) {
            ("contiguous_le_v1", "bf16") => multiply(count, 2)?,
            ("contiguous_le_v1", "fp32" | "int32") => multiply(count, 4)?,
            ("row_scale_v1", "fp8_e4m3fn_row_bf16") => {
                let [n, _] = self.matrix()?;
                add(align(count, 256)?, multiply(n, 2)?)?
            }
            ("block_scale_k16_m128x4_v1", "nvfp4") => {
                let [n, k] = self.matrix()?;
                ensure!(
                    n % 128 == 0 && k % 64 == 0,
                    "NVFP4 requires N divisible by 128 and K by 64"
                );
                add(add(align(count / 2, 256)?, count / 16)?, 4)?
            }
            ("row_split_k128_v1", "q4_g64_fp16") => self.row_split_size(4, 64)?,
            ("row_split_k128_v1", "q5_g64_fp16") => self.row_split_size(5, 64)?,
            ("row_split_k128_v1", "q6_g64_fp16") => self.row_split_size(6, 64)?,
            ("row_split_k128_v1", "q8_g32_fp16") => self.row_split_size(8, 32)?,
            _ => bail!("NInfer known format {format} is incompatible with layout {layout}"),
        };
        Ok(Some(size))
    }

    fn matrix(&self) -> Result<[u64; 2]> {
        self.shape
            .as_slice()
            .try_into()
            .context("NInfer packed layout requires rank two")
    }

    fn row_split_size(&self, bits: u64, group: u64) -> Result<u64> {
        let [n, k] = self.matrix()?;
        let padded_elements = multiply(n, align(k, 128)?)?;
        let base = if bits == 8 {
            padded_elements
        } else {
            padded_elements / 2
        };
        let high = match bits {
            5 => padded_elements / 8,
            6 => padded_elements / 4,
            _ => 0,
        };
        let scales = multiply(padded_elements / group, 2)?;
        add(add(align(base, 256)?, align(high, 256)?)?, scales)
    }
}

fn known_layout(layout: &str) -> bool {
    matches!(
        layout,
        "contiguous_le_v1" | "row_scale_v1" | "block_scale_k16_m128x4_v1" | "row_split_k128_v1"
    )
}

fn known_format(format: &str) -> bool {
    matches!(
        format,
        "bf16"
            | "fp32"
            | "int32"
            | "fp8_e4m3fn_row_bf16"
            | "nvfp4"
            | "q4_g64_fp16"
            | "q5_g64_fp16"
            | "q6_g64_fp16"
            | "q8_g32_fp16"
    )
}

type ObjectIndex<'a> = BTreeMap<&'a str, &'a Object>;

impl Directory {
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            self.files.len() == 1,
            "only single-file NInfer v3 artifacts are supported; linked files are never opened"
        );
        ensure!(
            self.files[0].path.is_none(),
            "NInfer entry file path must be null; external paths are not accepted"
        );
        let payload = self.files[0].payload_bytes;
        ensure!(payload > 0, "NInfer payload must not be empty");
        ensure!(
            !self.objects.is_empty() && self.objects.len() <= MAX_RECORDS,
            "invalid NInfer object count"
        );
        ensure!(
            self.bindings.len() <= MAX_RECORDS && self.uses.len() <= MAX_RECORDS,
            "NInfer binding/use count exceeds limit"
        );
        let mut index = BTreeMap::new();
        let mut previous_end = 0;
        for object in &self.objects {
            object
                .validate()
                .with_context(|| format!("object {}", object.id))?;
            ensure!(
                index.insert(object.id.as_str(), object).is_none(),
                "duplicate NInfer object id {}",
                object.id
            );
            let end = add(object.offset, object.bytes)?;
            ensure!(
                object.offset >= previous_end && end <= payload,
                "NInfer object ranges overlap, are unordered, or exceed payload"
            );
            previous_end = end;
        }
        self.validate_components(&index)?;
        self.validate_bindings(&index)?;
        ensure!(
            self.metadata.is_object() && self.provenance.is_object(),
            "NInfer metadata/provenance must be objects"
        );
        if let Some(name) = self.metadata.get("name") {
            identifier(name.as_str().context("metadata.name must be a string")?)?;
        }
        Ok(())
    }

    fn validate_bindings(&self, index: &ObjectIndex<'_>) -> Result<()> {
        let mut total_parts = 0;
        for (name, binding) in &self.bindings {
            identifier(name)?;
            total_parts += binding
                .validate(index)
                .with_context(|| format!("binding {name}"))?;
            ensure!(
                total_parts <= MAX_PARTS,
                "NInfer total binding parts exceed limit"
            );
        }
        let mut uses = BTreeSet::new();
        for usage in &self.uses {
            identifier(&usage.parameter)?;
            identifier(&usage.input)?;
            ensure!(
                self.bindings.contains_key(&usage.parameter),
                "NInfer Use references a missing parameter"
            );
            ensure!(
                uses.insert((&usage.parameter, &usage.input)),
                "duplicate NInfer parameter/input Use"
            );
            if let Some(policy) = &usage.activation_policy {
                ensure!(
                    matches!(policy.as_str(), "A16Only" | "AllowA8" | "AllowA4"),
                    "unsupported NInfer activation policy"
                );
            }
            ensure!(
                usage.auxiliaries.len() <= MAX_RECORDS,
                "too many NInfer auxiliaries"
            );
            for (role, binding) in &usage.auxiliaries {
                identifier(role)?;
                total_parts += binding
                    .validate(index)
                    .with_context(|| format!("auxiliary {role}"))?;
                ensure!(
                    total_parts <= MAX_PARTS,
                    "NInfer total binding parts exceed limit"
                );
            }
        }
        Ok(())
    }

    fn validate_components(&self, index: &ObjectIndex<'_>) -> Result<()> {
        ensure!(
            self.components.contains_key("text") && self.components.len() <= MAX_RECORDS,
            "NInfer components must contain text and be bounded"
        );
        for (name, component) in &self.components {
            identifier(name)?;
            members(component, &["config"], &["target", "resources", "proposal"])?;
            ensure!(
                component["config"].is_object(),
                "NInfer component config must be an object"
            );
            if let Some(target) = component.get("target") {
                let target = target
                    .as_str()
                    .context("component target must be a string")?;
                identifier(target)?;
                ensure!(
                    self.components.contains_key(target),
                    "dangling NInfer component target"
                );
            }
            if let Some(resources) = component.get("resources") {
                for (role, id) in resources
                    .as_object()
                    .context("component resources must be an object")?
                {
                    identifier(role)?;
                    let id = id.as_str().context("resource reference must be a string")?;
                    identifier(id)?;
                    ensure!(
                        index
                            .get(id)
                            .is_some_and(|object| object.kind == "resource"),
                        "dangling or non-resource NInfer component resource"
                    );
                }
            }
            if let Some(proposal) = component.get("proposal") {
                ensure!(name == "text", "only text may declare a NInfer proposal");
                validate_proposal(proposal)?;
            }
        }
        Ok(())
    }
}

fn members(value: &Value, required: &[&str], optional: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .context("NInfer record must be an object")?;
    ensure!(
        required.iter().all(|key| object.contains_key(*key)),
        "missing NInfer record field"
    );
    ensure!(
        object
            .keys()
            .all(|key| required.contains(&key.as_str()) || optional.contains(&key.as_str())),
        "unknown NInfer record field"
    );
    Ok(())
}

fn validate_proposal(value: &Value) -> Result<()> {
    members(value, &["domain"], &["rows"])?;
    match value["domain"].as_str() {
        Some("full") => ensure!(value.get("rows").is_none(), "full proposal must omit rows"),
        Some("indexed") => ensure!(
            value
                .get("rows")
                .and_then(Value::as_u64)
                .is_some_and(|n| n > 0),
            "indexed proposal needs positive integer rows"
        ),
        _ => bail!("invalid NInfer proposal domain"),
    }
    Ok(())
}

impl Binding {
    fn validate(&self, index: &ObjectIndex<'_>) -> Result<usize> {
        match self {
            Self::Object { object } => {
                tensor(index, object)?;
                Ok(1)
            }
            Self::Parts { parts } => validate_parts(parts, index),
        }
    }
}

fn tensor<'a>(index: &ObjectIndex<'a>, id: &str) -> Result<&'a Object> {
    identifier(id)?;
    let object = index.get(id).context("dangling NInfer tensor binding")?;
    ensure!(
        object.kind == "tensor",
        "NInfer binding must reference a tensor, not a resource"
    );
    Ok(object)
}

fn validate_parts(parts: &[Part], index: &ObjectIndex<'_>) -> Result<usize> {
    ensure!(
        !parts.is_empty() && parts.len() <= MAX_PARTS,
        "invalid NInfer multipart count"
    );
    let mut total = 0;
    let mut intervals = Vec::with_capacity(parts.len());
    for part in parts {
        let object = tensor(index, &part.object)?;
        let [start, end] = part.range;
        ensure!(
            start < end && end <= object.logical_elements()?,
            "NInfer binding range exceeds logical elements or is empty"
        );
        total = add(total, end - start)?;
        intervals.push((part.object.as_str(), start, end));
    }
    // Aliases in DISTINCT bindings/Uses are legitimate (e.g. fused parent row
    // views). Within one concatenation, overlapping source ranges are refused.
    intervals.sort_unstable();
    for pair in intervals.windows(2) {
        ensure!(
            pair[0].0 != pair[1].0 || pair[0].2 <= pair[1].1,
            "overlapping NInfer multipart ranges"
        );
    }
    Ok(parts.len())
}
