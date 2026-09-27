//! CPU-only assembly for the internal `.mspec` container format.

pub use super::schema::{DType, ObjectKind, SourceCheckpoint};
use super::{
    header::{HEADER_LEN, Header},
    reader::MAX_ARTIFACT_BYTES,
    schema::{Directory, MAX_OBJECTS, Object, align_up},
};
use anyhow::{Context, Result, ensure};
use mesh_llm_native_runtime::model_identity::ModelIdentity;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};
use tempfile::NamedTempFile;

const COPY_BUFFER_BYTES: usize = 64 * 1024;
const PENDING_WEIGHTS_ID: &str =
    "sha256:0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObjectSource {
    pub name: String,
    pub kind: ObjectKind,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub layout: String,
    pub path: PathBuf,
    /// Absolute byte range in `path`; `None` selects the entire regular file.
    pub range: Option<SourceRange>,
    /// Optional expected digest of the selected bytes, not of the enclosing file.
    pub expected_sha256: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceRange {
    pub offset: u64,
    pub length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WrittenArtifact {
    pub directory: Directory,
    pub bytes: u64,
}

struct SourceFile {
    path: PathBuf,
    file_len: u64,
    offset: u64,
    length: u64,
    expected_sha256: Option<String>,
}

struct Preflight {
    payload_offset: u64,
    artifact_bytes: u64,
}

/// Assemble validated source objects into a new artifact without replacing an existing path.
pub fn write_artifact(
    output: &Path,
    model_id: &str,
    source: SourceCheckpoint,
    sources: &[ObjectSource],
) -> Result<WrittenArtifact> {
    let (inputs, mut directory, payload_len) = open_sources(model_id, source, sources)?;
    let preflight = validate_for_write(&directory, payload_len)?;
    hash_sources(&inputs, &mut directory)?;
    directory.identity.weights_id = super::weights_identity(&directory, payload_len)?;
    directory.validate(payload_len)?;

    let directory_bytes = serde_json::to_vec(&directory).context("serialize artifact directory")?;
    let directory_len = u64::try_from(directory_bytes.len())?;
    let header_digest: [u8; 32] = Sha256::digest(&directory_bytes).into();
    let header = Header::new(directory_len, header_digest)?;
    ensure!(
        header.payload_offset == preflight.payload_offset,
        "final artifact directory changed its encoded length"
    );
    ensure!(
        header
            .payload_offset
            .checked_add(payload_len)
            .context("artifact size overflows u64")?
            == preflight.artifact_bytes,
        "final artifact size differs from preflight"
    );

    let parent = output_parent(output);
    let mut temporary = NamedTempFile::new_in(parent).context("create temporary artifact")?;
    write_artifact_contents(
        &mut temporary,
        &header,
        &directory_bytes,
        &inputs,
        &directory,
        payload_len,
    )?;
    ensure!(
        temporary.as_file().metadata()?.len() == preflight.artifact_bytes,
        "written artifact size differs from preflight"
    );
    temporary
        .as_file()
        .sync_all()
        .context("sync temporary artifact")?;
    temporary
        .persist_noclobber(output)
        .map_err(|failure| failure.error)
        .context("persist artifact without replacing an existing output")?;

    Ok(WrittenArtifact {
        directory,
        bytes: preflight.artifact_bytes,
    })
}

fn open_sources(
    model_id: &str,
    source: SourceCheckpoint,
    sources: &[ObjectSource],
) -> Result<(Vec<SourceFile>, Directory, u64)> {
    ensure!(
        (1..=MAX_OBJECTS).contains(&sources.len()),
        "artifact must contain between 1 and {MAX_OBJECTS} source objects"
    );
    ensure!(
        sources
            .iter()
            .filter(|item| item.kind == ObjectKind::Recipe)
            .count()
            == 1,
        "artifact sources must contain exactly one recipe object"
    );

    let mut ordered: Vec<_> = sources.iter().collect();
    ordered.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
    for pair in ordered.windows(2) {
        ensure!(
            pair[0].name != pair[1].name,
            "artifact source names must be unique"
        );
    }

    let identity = ModelIdentity {
        model_id: model_id.to_string(),
        weights_id: PENDING_WEIGHTS_ID.to_string(),
    };
    identity.validate()?;
    let mut objects = Vec::with_capacity(ordered.len());
    let mut inputs = Vec::with_capacity(ordered.len());
    let mut previous_end = 0_u64;
    for item in ordered {
        let file = File::open(&item.path).with_context(|| "open artifact source object")?;
        let metadata = file.metadata().context("read artifact source metadata")?;
        ensure!(metadata.is_file(), "artifact source must be a regular file");
        let file_len = metadata.len();
        if let Some(expected_sha256) = &item.expected_sha256 {
            validate_expected_sha256(expected_sha256)?;
        }
        let range = item.range.clone().unwrap_or(SourceRange {
            offset: 0,
            length: file_len,
        });
        let range_end = range
            .offset
            .checked_add(range.length)
            .context("artifact source range extent overflows u64")?;
        ensure!(
            range_end <= file_len,
            "artifact source range exceeds source file length"
        );
        let offset = align_up(previous_end)?;
        previous_end = offset
            .checked_add(range.length)
            .context("artifact payload extent overflows u64")?;
        objects.push(Object {
            name: item.name.clone(),
            kind: item.kind.clone(),
            dtype: item.dtype.clone(),
            shape: item.shape.clone(),
            layout: item.layout.clone(),
            offset,
            length: range.length,
            sha256: "0".repeat(64),
        });
        inputs.push(SourceFile {
            path: item.path.clone(),
            file_len,
            offset: range.offset,
            length: range.length,
            expected_sha256: item.expected_sha256.clone(),
        });
    }

    let directory = Directory {
        schema_version: 1,
        identity,
        recipe_sha256: "0".repeat(64),
        source,
        objects,
    };
    directory.validate(previous_end)?;
    Ok((inputs, directory, previous_end))
}

fn validate_for_write(directory: &Directory, payload_len: u64) -> Result<Preflight> {
    directory.validate(payload_len)?;
    let directory_len = serialized_len(directory)?;
    ensure!(directory_len != 0, "artifact directory must not be empty");
    let header = Header::new(directory_len, [0; 32])?;
    let artifact_bytes = header
        .payload_offset
        .checked_add(payload_len)
        .context("artifact size overflows u64")?;
    ensure!(
        artifact_bytes <= MAX_ARTIFACT_BYTES,
        "artifact exceeds the prototype size limit"
    );
    Ok(Preflight {
        payload_offset: header.payload_offset,
        artifact_bytes,
    })
}

fn serialized_len(value: &impl Serialize) -> Result<u64> {
    let mut counter = CountingWriter::default();
    serde_json::to_writer(&mut counter, value).context("measure artifact directory")?;
    Ok(counter.bytes)
}

#[derive(Default)]
struct CountingWriter {
    bytes: u64,
}

impl Write for CountingWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| io::Error::other("serialized length overflows u64"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn hash_sources(sources: &[SourceFile], directory: &mut Directory) -> Result<()> {
    for (source, object) in sources.iter().zip(&mut directory.objects) {
        let sha256 = stream_source(&mut open_source(source)?, source, &mut io::sink())?;
        if let Some(expected_sha256) = &source.expected_sha256 {
            ensure!(
                sha256 == *expected_sha256,
                "artifact source slice SHA-256 does not match the expected digest for {}",
                object.name
            );
        }
        object.sha256 = sha256;
        if object.kind == ObjectKind::Recipe {
            directory.recipe_sha256.clone_from(&object.sha256);
        }
    }
    Ok(())
}

fn stream_source(file: &mut File, source: &SourceFile, output: &mut impl Write) -> Result<String> {
    verify_source_length(file, source.file_len)?;
    file.seek(SeekFrom::Start(source.offset))
        .context("seek artifact source to its selected range")?;
    let mut digest = Sha256::new();
    let mut buffer = [0; COPY_BUFFER_BYTES];
    let mut remaining = source.length;
    while remaining != 0 {
        let count = usize::try_from(remaining.min(COPY_BUFFER_BYTES as u64))?;
        let chunk = &mut buffer[..count];
        file.read_exact(chunk)
            .context("read artifact source object")?;
        output.write_all(chunk).context("write artifact payload")?;
        digest.update(&*chunk);
        remaining -= count as u64;
    }
    verify_source_length(file, source.file_len)?;
    Ok(hex::encode(digest.finalize()))
}

fn open_source(source: &SourceFile) -> Result<File> {
    let file = File::open(&source.path).context("reopen artifact source object")?;
    ensure!(
        file.metadata()?.is_file(),
        "artifact source must be a regular file"
    );
    verify_source_length(&file, source.file_len)?;
    Ok(file)
}

fn verify_source_length(file: &File, expected_len: u64) -> Result<()> {
    ensure!(
        file.metadata()
            .context("read artifact source metadata")?
            .len()
            == expected_len,
        "artifact source length changed during writing"
    );
    Ok(())
}

fn validate_expected_sha256(value: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "expected source SHA-256 must be 64 lowercase hexadecimal characters"
    );
    Ok(())
}

fn write_artifact_contents(
    output: &mut NamedTempFile,
    header: &Header,
    directory_bytes: &[u8],
    sources: &[SourceFile],
    directory: &Directory,
    payload_len: u64,
) -> Result<()> {
    let writer = output.as_file_mut();
    writer.write_all(&header.encode()?)?;
    writer.write_all(directory_bytes)?;
    let directory_end = (HEADER_LEN as u64)
        .checked_add(directory_bytes.len() as u64)
        .context("artifact directory end overflows u64")?;
    write_zero_bytes(writer, header.payload_offset - directory_end)?;

    let mut previous_end = 0_u64;
    for (source, object) in sources.iter().zip(&directory.objects) {
        write_zero_bytes(writer, object.offset - previous_end)?;
        let digest = stream_source(&mut open_source(source)?, source, writer)?;
        ensure!(
            digest == object.sha256,
            "artifact source bytes changed between hashing and assembly"
        );
        previous_end = object
            .offset
            .checked_add(object.length)
            .context("artifact payload extent overflows u64")?;
    }
    ensure!(
        previous_end == payload_len,
        "written payload length changed"
    );
    Ok(())
}

fn write_zero_bytes(output: &mut impl Write, mut length: u64) -> Result<()> {
    let zeros = [0; 256];
    while length != 0 {
        let count = usize::try_from(length.min(zeros.len() as u64))?;
        output.write_all(&zeros[..count])?;
        length -= count as u64;
    }
    Ok(())
}

fn output_parent(output: &Path) -> &Path {
    output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

#[cfg(test)]
#[path = "writer_tests.rs"]
mod tests;
