//! Fixed-size header for mesh-specialize binary artifacts.

use anyhow::{Result, ensure};

pub const HEADER_LEN: usize = 64;
pub const MAX_DIRECTORY_BYTES: u64 = 16 * 1024 * 1024;
pub const FORMAT_VERSION: u32 = 1;

const MAGIC: &[u8; 9] = b"MESHSPEC\0";
const MAGIC_RANGE: std::ops::Range<usize> = 0..9;
const RESERVED_RANGE: std::ops::Range<usize> = 9..12;
const VERSION_RANGE: std::ops::Range<usize> = 12..16;
const DIRECTORY_LEN_RANGE: std::ops::Range<usize> = 16..24;
const PAYLOAD_OFFSET_RANGE: std::ops::Range<usize> = 24..32;
const DIRECTORY_SHA256_RANGE: std::ops::Range<usize> = 32..64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Header {
    pub directory_len: u64,
    pub payload_offset: u64,
    pub directory_sha256: [u8; 32],
}

impl Header {
    pub fn new(directory_len: u64, directory_sha256: [u8; 32]) -> Result<Self> {
        validate_directory_len(directory_len)?;
        Ok(Self {
            directory_len,
            payload_offset: expected_payload_offset(directory_len)?,
            directory_sha256,
        })
    }

    pub fn encode(&self) -> Result<[u8; HEADER_LEN]> {
        validate_directory_len(self.directory_len)?;
        let expected_offset = expected_payload_offset(self.directory_len)?;
        ensure!(
            self.payload_offset == expected_offset,
            "payload offset {} does not match aligned directory end {}",
            self.payload_offset,
            expected_offset
        );

        let mut bytes = [0; HEADER_LEN];
        bytes[MAGIC_RANGE].copy_from_slice(MAGIC);
        bytes[VERSION_RANGE].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes[DIRECTORY_LEN_RANGE].copy_from_slice(&self.directory_len.to_le_bytes());
        bytes[PAYLOAD_OFFSET_RANGE].copy_from_slice(&self.payload_offset.to_le_bytes());
        bytes[DIRECTORY_SHA256_RANGE].copy_from_slice(&self.directory_sha256);
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8; HEADER_LEN], file_len: u64) -> Result<Self> {
        ensure!(
            &bytes[MAGIC_RANGE] == MAGIC,
            "artifact header has invalid magic"
        );
        ensure!(
            bytes[RESERVED_RANGE] == [0, 0, 0],
            "artifact header reserved bytes must be zero"
        );
        let version = read_u32(bytes, VERSION_RANGE);
        ensure!(
            version == FORMAT_VERSION,
            "unsupported artifact format version {version}"
        );

        let directory_len = read_u64(bytes, DIRECTORY_LEN_RANGE);
        validate_directory_len(directory_len)?;
        let payload_offset = read_u64(bytes, PAYLOAD_OFFSET_RANGE);
        let expected_offset = expected_payload_offset(directory_len)?;
        ensure!(
            payload_offset == expected_offset,
            "payload offset {payload_offset} does not match aligned directory end {expected_offset}"
        );
        ensure!(
            file_len > payload_offset,
            "artifact file length {file_len} does not include a payload byte after offset {payload_offset}"
        );

        let mut directory_sha256 = [0; 32];
        directory_sha256.copy_from_slice(&bytes[DIRECTORY_SHA256_RANGE]);
        Ok(Self {
            directory_len,
            payload_offset,
            directory_sha256,
        })
    }
}

fn validate_directory_len(directory_len: u64) -> Result<()> {
    ensure!(
        (1..=MAX_DIRECTORY_BYTES).contains(&directory_len),
        "artifact directory length must be between 1 and {MAX_DIRECTORY_BYTES} bytes, got {directory_len}"
    );
    Ok(())
}

fn expected_payload_offset(directory_len: u64) -> Result<u64> {
    let directory_end = (HEADER_LEN as u64)
        .checked_add(directory_len)
        .ok_or_else(|| anyhow::anyhow!("artifact directory end overflows u64"))?;
    super::schema::align_up(directory_end)
}

fn read_u32(bytes: &[u8; HEADER_LEN], range: std::ops::Range<usize>) -> u32 {
    u32::from_le_bytes([
        bytes[range.start],
        bytes[range.start + 1],
        bytes[range.start + 2],
        bytes[range.start + 3],
    ])
}

fn read_u64(bytes: &[u8; HEADER_LEN], range: std::ops::Range<usize>) -> u64 {
    u64::from_le_bytes([
        bytes[range.start],
        bytes[range.start + 1],
        bytes[range.start + 2],
        bytes[range.start + 3],
        bytes[range.start + 4],
        bytes[range.start + 5],
        bytes[range.start + 6],
        bytes[range.start + 7],
    ])
}

#[cfg(test)]
mod tests {
    use super::{
        FORMAT_VERSION, HEADER_LEN, Header, MAGIC, MAX_DIRECTORY_BYTES, PAYLOAD_OFFSET_RANGE,
        RESERVED_RANGE, VERSION_RANGE,
    };

    fn sample_header() -> Header {
        Header::new(100, std::array::from_fn(|index| index as u8)).unwrap()
    }

    #[test]
    fn roundtrip_uses_specified_hand_checked_byte_offsets() {
        let header = sample_header();
        assert_eq!(header.payload_offset, 256);
        let bytes = header.encode().unwrap();
        assert_eq!(&bytes[0..9], MAGIC);
        assert_eq!(bytes[RESERVED_RANGE], [0, 0, 0]);
        assert_eq!(
            u32::from_le_bytes(bytes[VERSION_RANGE].try_into().unwrap()),
            FORMAT_VERSION
        );
        assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 100);
        assert_eq!(
            u64::from_le_bytes(bytes[PAYLOAD_OFFSET_RANGE].try_into().unwrap()),
            256
        );
        assert_eq!(&bytes[32..HEADER_LEN], &header.directory_sha256);
        assert_eq!(Header::decode(&bytes, 257).unwrap(), header);
    }

    #[test]
    fn rejects_wrong_magic_reserved_bytes_and_version() {
        let mut bytes = sample_header().encode().unwrap();
        bytes[0] ^= 1;
        assert!(Header::decode(&bytes, 257).is_err());

        let mut bytes = sample_header().encode().unwrap();
        bytes[RESERVED_RANGE.start] = 1;
        assert!(Header::decode(&bytes, 257).is_err());

        let mut bytes = sample_header().encode().unwrap();
        bytes[VERSION_RANGE.start] = 2;
        assert!(Header::decode(&bytes, 257).is_err());
    }

    #[test]
    fn rejects_zero_and_oversized_directories() {
        assert!(Header::new(0, [0; 32]).is_err());
        assert!(Header::new(MAX_DIRECTORY_BYTES + 1, [0; 32]).is_err());
        assert!(
            Header {
                directory_len: 0,
                payload_offset: 256,
                directory_sha256: [0; 32],
            }
            .encode()
            .is_err()
        );
        assert!(
            Header {
                directory_len: MAX_DIRECTORY_BYTES + 1,
                payload_offset: 0,
                directory_sha256: [0; 32],
            }
            .encode()
            .is_err()
        );

        let mut bytes = sample_header().encode().unwrap();
        bytes[16..24].copy_from_slice(&0_u64.to_le_bytes());
        assert!(Header::decode(&bytes, 257).is_err());
        bytes[16..24].copy_from_slice(&(MAX_DIRECTORY_BYTES + 1).to_le_bytes());
        assert!(Header::decode(&bytes, u64::MAX).is_err());
    }

    #[test]
    fn rejects_misaligned_or_incorrect_payload_offsets() {
        let mut header = sample_header();
        header.payload_offset = 257;
        assert!(header.encode().is_err());

        let mut bytes = sample_header().encode().unwrap();
        bytes[PAYLOAD_OFFSET_RANGE].copy_from_slice(&257_u64.to_le_bytes());
        assert!(Header::decode(&bytes, 258).is_err());

        bytes[PAYLOAD_OFFSET_RANGE].copy_from_slice(&512_u64.to_le_bytes());
        assert!(Header::decode(&bytes, 513).is_err());
    }

    #[test]
    fn requires_at_least_one_payload_byte() {
        let bytes = sample_header().encode().unwrap();
        assert!(Header::decode(&bytes, 256).is_err());
        assert!(Header::decode(&bytes, 257).is_ok());
    }
}
