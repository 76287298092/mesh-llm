//! Bounded, streaming verification of generic SafeTensors files.

use anyhow::{Context, Result, ensure};
use safetensors::tensor::{Metadata, TensorInfo};
use serde::de::{self, Deserialize, Deserializer, MapAccess, Visitor};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::Read,
    path::Path,
};

const MAX_FILE_BYTES: u64 = 128 * 1024 * 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TENSORS: usize = 65_536;
const STREAM_BUFFER_BYTES: usize = 64 * 1024;
const STREAM_BUFFER_U64: u64 = 64 * 1024;

/// Verified metadata and payload digest for one tensor in a SafeTensors file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: safetensors::Dtype,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub length: u64,
    pub sha256: String,
}

/// Whole-file and header digests with verified tensor metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedTensorFile {
    pub file_len: u64,
    pub file_sha256: String,
    pub header_sha256: String,
    pub tensors: Vec<TensorEntry>,
}

struct ParsedHeader {
    metadata: Option<HashMap<String, String>>,
    tensors: Vec<(String, TensorInfo)>,
}

struct StringMap(HashMap<String, String>);

/// Verify a local regular file against its expected whole-file SHA-256.
pub fn verify(path: &Path, expected_sha256: &str) -> Result<VerifiedTensorFile> {
    validate_sha256(expected_sha256)?;
    let mut file = File::open(path).context("open SafeTensors file")?;
    let initial_metadata = file.metadata().context("read SafeTensors file metadata")?;
    ensure!(
        initial_metadata.is_file(),
        "SafeTensors source must be a regular file"
    );
    let file_len = initial_metadata.len();
    ensure!(
        file_len <= MAX_FILE_BYTES,
        "SafeTensors file exceeds the size limit"
    );

    let (prefix, header_bytes) = read_header(&mut file, file_len)?;
    let parsed = parse_header(&header_bytes)?;
    let metadata = validate_metadata(parsed)?;
    let payload_len = u64::try_from(metadata.data_len())?;
    let payload_start = 8_u64
        .checked_add(u64::try_from(header_bytes.len())?)
        .context("SafeTensors payload offset overflows u64")?;
    let expected_len = payload_start
        .checked_add(payload_len)
        .context("SafeTensors file extent overflows u64")?;
    ensure!(
        expected_len == file_len,
        "SafeTensors payload extent does not match file length"
    );

    let mut file_digest = Sha256::new();
    file_digest.update(prefix);
    file_digest.update(&header_bytes);
    let mut tensors = hash_payload(&mut file, payload_start, &metadata, &mut file_digest)?;
    let file_sha256 = hex::encode(file_digest.finalize());
    ensure!(
        file_sha256 == expected_sha256,
        "SafeTensors whole-file SHA-256 mismatch"
    );
    ensure!(
        file.metadata()
            .context("recheck SafeTensors file metadata")?
            .len()
            == file_len,
        "SafeTensors file length changed during verification"
    );
    tensors.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));

    Ok(VerifiedTensorFile {
        file_len,
        file_sha256,
        header_sha256: hex::encode(Sha256::digest(&header_bytes)),
        tensors,
    })
}

fn read_header(file: &mut File, file_len: u64) -> Result<([u8; 8], Vec<u8>)> {
    ensure!(
        file_len >= 8,
        "SafeTensors file is shorter than its header prefix"
    );
    let mut prefix = [0; 8];
    file.read_exact(&mut prefix)
        .context("read SafeTensors header length")?;
    let header_len = u64::from_le_bytes(prefix);
    ensure!(header_len != 0, "SafeTensors header must not be empty");
    ensure!(
        header_len <= MAX_HEADER_BYTES,
        "SafeTensors header exceeds the size limit"
    );
    let header_end = 8_u64
        .checked_add(header_len)
        .context("SafeTensors header extent overflows u64")?;
    ensure!(
        header_end <= file_len,
        "SafeTensors header exceeds file length"
    );
    let mut header = vec![0; usize::try_from(header_len)?];
    file.read_exact(&mut header)
        .context("read SafeTensors header")?;
    ensure!(
        header.first() == Some(&b'{'),
        "SafeTensors header must start with an object"
    );
    Ok((prefix, header))
}

fn parse_header(bytes: &[u8]) -> Result<ParsedHeader> {
    serde_json::from_slice(bytes).context("parse SafeTensors header")
}

fn validate_metadata(mut parsed: ParsedHeader) -> Result<Metadata> {
    for (name, info) in &parsed.tensors {
        validate_tensor_name(name)?;
        ensure!(info.shape.len() <= 8, "SafeTensors tensor rank exceeds 8");
    }
    parsed
        .tensors
        .sort_by_key(|(_, info)| (info.data_offsets.0, info.data_offsets.1));
    Metadata::new(parsed.metadata, parsed.tensors).context("validate SafeTensors tensor metadata")
}

fn hash_payload(
    file: &mut File,
    payload_start: u64,
    metadata: &Metadata,
    file_digest: &mut Sha256,
) -> Result<Vec<TensorEntry>> {
    let names = metadata.offset_keys();
    let mut entries = Vec::with_capacity(names.len());
    let mut previous_end = 0_u64;
    for name in names {
        let info = metadata
            .info(&name)
            .with_context(|| format!("missing parsed SafeTensors tensor {name}"))?;
        let start = u64::try_from(info.data_offsets.0)?;
        let end = u64::try_from(info.data_offsets.1)?;
        ensure!(
            start == previous_end,
            "SafeTensors tensor offsets are not contiguous"
        );
        let length = end
            .checked_sub(start)
            .context("SafeTensors tensor extent is inverted")?;
        let offset = payload_start
            .checked_add(start)
            .context("SafeTensors tensor file offset overflows u64")?;
        let digest = hash_tensor_bytes(file, length, file_digest)?;
        let shape = info
            .shape
            .iter()
            .map(|dimension| u64::try_from(*dimension))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        entries.push(TensorEntry {
            name,
            dtype: info.dtype,
            shape,
            offset,
            length,
            sha256: digest,
        });
        previous_end = end;
    }
    ensure!(
        previous_end == u64::try_from(metadata.data_len())?,
        "SafeTensors tensors do not cover the declared payload"
    );
    Ok(entries)
}

fn hash_tensor_bytes(file: &mut File, length: u64, file_digest: &mut Sha256) -> Result<String> {
    let mut tensor_digest = Sha256::new();
    let mut remaining = length;
    let mut buffer = [0; STREAM_BUFFER_BYTES];
    while remaining != 0 {
        let count = usize::try_from(remaining.min(STREAM_BUFFER_U64))?;
        let bytes = &mut buffer[..count];
        file.read_exact(bytes)
            .context("read SafeTensors tensor payload")?;
        file_digest.update(&*bytes);
        tensor_digest.update(&*bytes);
        remaining -= u64::try_from(count)?;
    }
    Ok(hex::encode(tensor_digest.finalize()))
}

fn validate_tensor_name(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty(),
        "SafeTensors tensor name must not be empty"
    );
    ensure!(
        name.len() <= 1024,
        "SafeTensors tensor name exceeds 1024 bytes"
    );
    ensure!(
        name.trim() == name,
        "SafeTensors tensor name has surrounding whitespace"
    );
    ensure!(
        !name.chars().any(char::is_control),
        "SafeTensors tensor name has a control character"
    );
    Ok(())
}

fn validate_sha256(value: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "expected SHA-256 must be 64 lowercase hexadecimal characters"
    );
    Ok(())
}

impl<'de> Deserialize<'de> for ParsedHeader {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(ParsedHeaderVisitor)
    }
}

struct ParsedHeaderVisitor;

impl<'de> Visitor<'de> for ParsedHeaderVisitor {
    type Value = ParsedHeader;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a SafeTensors header object")
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut keys = HashSet::new();
        let mut metadata = None;
        let mut tensors = Vec::new();
        while let Some(key) = map.next_key::<String>()? {
            if !keys.insert(key.clone()) {
                return Err(de::Error::custom("duplicate SafeTensors header key"));
            }
            if key == "__metadata__" {
                metadata = Some(map.next_value::<StringMap>()?.0);
            } else {
                if tensors.len() == MAX_TENSORS {
                    return Err(de::Error::custom("SafeTensors header has too many tensors"));
                }
                tensors.push((key, map.next_value::<TensorInfo>()?));
            }
        }
        Ok(ParsedHeader { metadata, tensors })
    }
}

impl<'de> Deserialize<'de> for StringMap {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(StringMapVisitor)
    }
}

struct StringMapVisitor;

impl<'de> Visitor<'de> for StringMapVisitor {
    type Value = StringMap;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a string-to-string metadata object")
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = HashMap::new();
        while let Some((key, value)) = map.next_entry::<String, String>()? {
            if values.insert(key, value).is_some() {
                return Err(de::Error::custom("duplicate SafeTensors metadata key"));
            }
        }
        Ok(StringMap(values))
    }
}

#[cfg(test)]
#[path = "safetensors_tests.rs"]
mod tests;
