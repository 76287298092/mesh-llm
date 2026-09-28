//! Independent, bounded single-file NInfer v3 intake.
//!
//! Wire contract: upstream artifact framing/schema/layout declarations at
//! e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d. This is an independent Rust
//! implementation, not an imported reader or a model execution implementation.
//! Unlike `.mspec`, v3 has no per-object checksum. Structural validation and the
//! header's opaque artifact ID do NOT authenticate weights. Compare `file_sha256`
//! with a trusted pinned digest before use. The retained descriptor is not an
//! immutable snapshot: same-length in-place edits are not detected by copies.

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Deserializer, de};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fmt,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

#[path = "ninfer_schema.rs"]
mod schema;
pub use schema::{Binding, Directory, FileRecord, Object, Part, Use};

#[cfg(test)]
#[path = "ninfer_tests.rs"]
mod tests;

pub const MAX_DIRECTORY_BYTES: u64 = 4 * 1024 * 1024;
const HEADER_BYTES: u64 = 32;
const COPY_BYTES: usize = 64 * 1024;
const MAGIC: &[u8; 8] = b"NINFER\0\x03";

/// A structurally validated v3 artifact, backed only by its original open file.
/// No payload is read by `open`, and no linked files are ever opened.
pub struct NinferArtifact {
    file: File,
    file_bytes: u64,
    artifact_id: [u8; 16],
    payload_offset: u64,
    directory: Directory,
    directory_json: Vec<u8>,
    object_index: BTreeMap<String, usize>,
}

impl NinferArtifact {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).context("open NInfer artifact")?;
        let metadata = file.metadata()?;
        ensure!(metadata.is_file(), "NInfer source must be a regular file");
        let file_bytes = metadata.len();
        let mut header = [0; HEADER_BYTES as usize];
        file.read_exact(&mut header)
            .context("read NInfer v3 header")?;
        ensure!(header[..7] == MAGIC[..7], "invalid NInfer entry magic");
        ensure!(
            header[7] == 3,
            "unsupported NInfer version {}; expected v3",
            header[7]
        );
        let json_bytes = u64::from_le_bytes(header[8..16].try_into()?);
        ensure!(
            (1..=MAX_DIRECTORY_BYTES).contains(&json_bytes),
            "NInfer directory exceeds 4 MiB or is empty"
        );
        let directory_end = schema::add(HEADER_BYTES, json_bytes)?;
        let payload_offset = schema::align(directory_end, 4096)?;
        ensure!(
            payload_offset <= file_bytes,
            "truncated NInfer directory or alignment region"
        );
        let mut directory_json = vec![0; usize::try_from(json_bytes)?];
        file.read_exact(&mut directory_json)
            .context("read NInfer directory")?;
        let directory = decode_directory(&directory_json)?;
        let expected = schema::add(payload_offset, directory.files[0].payload_bytes)?;
        ensure!(
            expected == file_bytes,
            "NInfer file length differs from declared payload extent"
        );
        ensure!(
            file.metadata()?.len() == file_bytes,
            "NInfer file size changed during open"
        );
        let object_index = directory
            .objects
            .iter()
            .enumerate()
            .map(|(index, object)| (object.id.clone(), index))
            .collect();
        Ok(Self {
            file,
            file_bytes,
            artifact_id: header[16..32].try_into()?,
            payload_offset,
            directory,
            directory_json,
            object_index,
        })
    }

    pub fn directory(&self) -> &Directory {
        &self.directory
    }

    /// Original validated UTF-8 JSON, including source ordering and whitespace.
    pub fn directory_json(&self) -> &[u8] {
        &self.directory_json
    }

    /// Opaque header ID, not a content hash or proof of authenticity.
    pub fn artifact_id(&self) -> [u8; 16] {
        self.artifact_id
    }
    pub fn file_bytes(&self) -> u64 {
        self.file_bytes
    }
    pub fn payload_offset(&self) -> u64 {
        self.payload_offset
    }

    /// Hash every file byte with bounded memory, restoring the current cursor.
    /// The caller must compare against its trusted pin; this does not cache a
    /// verification claim or reverify subsequent object reads.
    pub fn file_sha256(&mut self) -> Result<String> {
        self.check_length()?;
        let position = self.file.stream_position()?;
        let result = self.hash_file();
        let restored = self.file.seek(SeekFrom::Start(position));
        let digest = result?;
        restored.context("restore NInfer reader position")?;
        Ok(digest)
    }

    /// Stream exact encoded object bytes, never a decoded or gathered view.
    /// On error, discard any partial destination. There is no per-object hash.
    pub fn copy_object(&mut self, id: &str, destination: &mut impl Write) -> Result<u64> {
        let bytes = self.object(id)?.bytes;
        self.read_object_range(id, 0, bytes, destination)
    }

    /// Stream a checked BYTE range inside one physical object. Binding `range`
    /// fields instead count logical ELEMENTS and cannot be passed here directly.
    /// Length is checked before and after I/O, not a whole-file hash. Unknown
    /// encodings can be copied for inspection, not silently treated as executable.
    pub fn read_object_range(
        &mut self,
        id: &str,
        offset: u64,
        length: u64,
        destination: &mut impl Write,
    ) -> Result<u64> {
        let object = self.object(id)?;
        ensure!(
            schema::add(offset, length)? <= object.bytes,
            "NInfer object byte range is out of bounds"
        );
        let absolute = schema::add(self.payload_offset, schema::add(object.offset, offset)?)?;
        self.check_length()?;
        self.file.seek(SeekFrom::Start(absolute))?;
        let mut buffer = [0; COPY_BYTES];
        let mut remaining = length;
        while remaining != 0 {
            let count = usize::try_from(remaining.min(COPY_BYTES as u64))?;
            self.file
                .read_exact(&mut buffer[..count])
                .context("read NInfer object bytes")?;
            destination.write_all(&buffer[..count])?;
            remaining -= count as u64;
        }
        self.check_length()?;
        Ok(length)
    }

    fn object(&self, id: &str) -> Result<&Object> {
        let index = self
            .object_index
            .get(id)
            .context("NInfer object id not found")?;
        Ok(&self.directory.objects[*index])
    }

    fn check_length(&self) -> Result<()> {
        ensure!(
            self.file.metadata()?.len() == self.file_bytes,
            "NInfer source size changed after validation"
        );
        Ok(())
    }

    fn hash_file(&mut self) -> Result<String> {
        self.file.seek(SeekFrom::Start(0))?;
        let mut digest = Sha256::new();
        let mut remaining = self.file_bytes;
        let mut buffer = [0; COPY_BYTES];
        while remaining != 0 {
            let count = usize::try_from(remaining.min(COPY_BYTES as u64))?;
            self.file
                .read_exact(&mut buffer[..count])
                .context("hash NInfer file")?;
            digest.update(&buffer[..count]);
            remaining -= count as u64;
        }
        self.check_length()?;
        Ok(hex::encode(digest.finalize()))
    }
}

fn decode_directory(bytes: &[u8]) -> Result<Directory> {
    // serde_json's default recursion limit remains enabled. The enclosing byte
    // cap bounds allocations; this visitor rejects duplicate keys at EVERY depth
    // before typed deserialization can silently discard one of their values.
    let value: UniqueJson = serde_json::from_slice(bytes).context("decode unique NInfer JSON")?;
    let directory: Directory =
        serde_json::from_value(value.0).context("parse NInfer directory schema")?;
    directory.validate()?;
    Ok(directory)
}

struct UniqueJson(Value);

impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_any(UniqueVisitor)
    }
}

struct UniqueVisitor;

impl<'de> de::Visitor<'de> for UniqueVisitor {
    type Value = UniqueJson;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON without duplicate object members")
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> std::result::Result<Self::Value, E> {
        Ok(UniqueJson(Value::Bool(value)))
    }
    fn visit_i64<E: de::Error>(self, value: i64) -> std::result::Result<Self::Value, E> {
        Ok(UniqueJson(value.into()))
    }
    fn visit_u64<E: de::Error>(self, value: u64) -> std::result::Result<Self::Value, E> {
        Ok(UniqueJson(value.into()))
    }
    fn visit_f64<E: de::Error>(self, value: f64) -> std::result::Result<Self::Value, E> {
        let number = serde_json::Number::from_f64(value)
            .ok_or_else(|| E::custom("nonfinite JSON number"))?;
        Ok(UniqueJson(Value::Number(number)))
    }
    fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<Self::Value, E> {
        Ok(UniqueJson(Value::String(value.to_owned())))
    }
    fn visit_string<E: de::Error>(self, value: String) -> std::result::Result<Self::Value, E> {
        Ok(UniqueJson(Value::String(value)))
    }
    fn visit_unit<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
        Ok(UniqueJson(Value::Null))
    }
    fn visit_seq<A: de::SeqAccess<'de>>(
        self,
        mut sequence: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut items = Vec::new();
        while let Some(value) = sequence.next_element::<UniqueJson>()? {
            items.push(value.0);
        }
        Ok(UniqueJson(Value::Array(items)))
    }
    fn visit_map<A: de::MapAccess<'de>>(
        self,
        mut map: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut members = serde_json::Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if members.contains_key(&key) {
                return Err(de::Error::custom(format!("duplicate JSON key {key:?}")));
            }
            members.insert(key, map.next_value::<UniqueJson>()?.0);
        }
        Ok(UniqueJson(Value::Object(members)))
    }
}
