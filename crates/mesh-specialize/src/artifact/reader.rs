//! Bounded resident-file verification and streaming object reads.

use super::{
    header::{HEADER_LEN, Header},
    identity,
    schema::{Directory, Object},
};
use anyhow::{Context, Result, ensure};
use mesh_llm_native_runtime::model_identity::ModelIdentity;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom, Write},
    path::Path,
};

/// Bound for this single-device prototype, including BF16 reference artifacts.
pub const MAX_ARTIFACT_BYTES: u64 = 128 * 1024 * 1024 * 1024;
const COPY_BUFFER_BYTES: usize = 64 * 1024;

/// An open resident artifact whose complete payload passed integrity checks.
/// The file descriptor stays open. It is not an immutable filesystem snapshot;
/// every subsequent copy checks the expected object digest again.
pub struct VerifiedArtifact {
    file: File,
    file_len: u64,
    header: Header,
    directory: Directory,
}

impl VerifiedArtifact {
    /// Verify a resident file without fetching, converting, or loading a model.
    /// Support for the returned identity must still be checked by the package.
    pub fn open(path: &Path) -> Result<Self> {
        let mut file =
            File::open(path).with_context(|| format!("open mspec {}", path.display()))?;
        let metadata = file.metadata()?;
        ensure!(metadata.is_file(), "mspec source must be a regular file");
        let file_len = metadata.len();
        ensure!(
            file_len <= MAX_ARTIFACT_BYTES,
            "mspec exceeds the prototype size limit"
        );
        let (header, directory) = read_directory(&mut file, file_len)?;
        let mut buffer = vec![0; COPY_BUFFER_BYTES];
        verify_payload(&mut file, &header, &directory, &mut buffer)?;
        ensure!(
            file.metadata()?.len() == file_len,
            "mspec size changed during verification"
        );
        Ok(Self {
            file,
            file_len,
            header,
            directory,
        })
    }

    /// Verify resident bytes and require a package's exact supported identity.
    pub fn open_for_identity(path: &Path, expected: &ModelIdentity) -> Result<Self> {
        expected.validate()?;
        let artifact = Self::open(path)?;
        artifact.require_identity(expected)?;
        Ok(artifact)
    }

    pub fn identity(&self) -> &ModelIdentity {
        &self.directory.identity
    }

    pub fn directory(&self) -> &Directory {
        &self.directory
    }

    pub fn require_identity(&self, expected: &ModelIdentity) -> Result<()> {
        expected.validate()?;
        ensure!(
            self.identity() == expected,
            "mspec model or weights identity is not supported"
        );
        Ok(())
    }

    /// Copy one object from the retained descriptor, checking its digest again.
    /// On any error the destination may contain partial or invalid data and MUST
    /// be discarded. Do not execute uploaded weights before this returns success.
    pub fn copy_object(&mut self, name: &str, destination: &mut impl Write) -> Result<u64> {
        let index = self
            .directory
            .objects
            .binary_search_by(|object| object.name.as_str().cmp(name))
            .map_err(|_| anyhow::anyhow!("mspec object not found: {name}"))?;
        let object = &self.directory.objects[index];
        ensure!(
            self.file.metadata()?.len() == self.file_len,
            "mspec size changed after verification"
        );
        let absolute = self
            .header
            .payload_offset
            .checked_add(object.offset)
            .context("mspec object offset overflow")?;
        self.file.seek(SeekFrom::Start(absolute))?;
        let mut buffer = vec![0; COPY_BUFFER_BYTES];
        copy_checked(&mut self.file, object, destination, &mut buffer)?;
        Ok(object.length)
    }
}

fn read_directory(file: &mut File, file_len: u64) -> Result<(Header, Directory)> {
    let mut bytes = [0; HEADER_LEN];
    file.read_exact(&mut bytes).context("read mspec header")?;
    let header = Header::decode(&bytes, file_len)?;
    let mut directory_bytes = vec![0; usize::try_from(header.directory_len)?];
    file.read_exact(&mut directory_bytes)
        .context("read mspec directory")?;
    let digest: [u8; 32] = Sha256::digest(&directory_bytes).into();
    ensure!(
        digest == header.directory_sha256,
        "mspec directory checksum mismatch"
    );
    let directory: Directory =
        serde_json::from_slice(&directory_bytes).context("parse mspec directory")?;
    directory.validate(file_len - header.payload_offset)?;
    ensure!(
        directory.identity.weights_id == identity::calculate(&directory),
        "mspec declared weight identity does not match its content inventory"
    );
    Ok((header, directory))
}

fn verify_payload(
    file: &mut File,
    header: &Header,
    directory: &Directory,
    buffer: &mut [u8],
) -> Result<()> {
    let directory_end = (HEADER_LEN as u64)
        .checked_add(header.directory_len)
        .context("mspec directory end overflow")?;
    read_zero_padding(file, header.payload_offset - directory_end)?;
    let mut position = 0;
    for object in &directory.objects {
        read_zero_padding(file, object.offset - position)?;
        copy_checked(file, object, &mut io::sink(), buffer)?;
        position = object.offset + object.length; // schema validation checked the sum.
    }
    Ok(())
}

fn read_zero_padding(file: &mut File, length: u64) -> Result<()> {
    ensure!(
        length < super::schema::ALIGNMENT,
        "mspec padding exceeds one alignment block"
    );
    let mut padding = [0; super::schema::ALIGNMENT as usize];
    let bytes = &mut padding[..usize::try_from(length)?];
    file.read_exact(bytes).context("read mspec padding")?;
    ensure!(
        bytes.iter().all(|byte| *byte == 0),
        "mspec padding must be zero"
    );
    Ok(())
}

fn copy_checked(
    file: &mut File,
    object: &Object,
    destination: &mut impl Write,
    buffer: &mut [u8],
) -> Result<()> {
    let mut digest = Sha256::new();
    let mut remaining = object.length;
    while remaining != 0 {
        let count = usize::try_from(remaining.min(buffer.len() as u64))?;
        let bytes = &mut buffer[..count];
        file.read_exact(bytes)
            .with_context(|| format!("read mspec object {}", object.name))?;
        digest.update(&*bytes);
        destination
            .write_all(bytes)
            .with_context(|| format!("copy mspec object {}", object.name))?;
        remaining -= count as u64;
    }
    ensure!(
        hex::encode(digest.finalize()) == object.sha256,
        "mspec object checksum mismatch: {}",
        object.name
    );
    Ok(())
}

#[cfg(test)]
#[path = "reader_tests.rs"]
mod tests;
