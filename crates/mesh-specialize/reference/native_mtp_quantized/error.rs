use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeMtpDecodeError {
    InvalidGeometry(&'static str),
    PlaneExtent {
        plane: &'static str,
        expected: usize,
        actual: usize,
    },
    RowOutOfBounds {
        row: usize,
        rows: usize,
    },
    InvalidScale {
        group: usize,
        bits: u16,
    },
    ZeroScaleNonZeroCode {
        group: usize,
        index: usize,
    },
    InvalidQ8Code {
        index: usize,
        code: u8,
    },
    ActivationExtent {
        expected: usize,
        actual: usize,
    },
    NonFiniteActivation {
        index: usize,
    },
    NonZeroPaddingCode {
        index: usize,
        code: i8,
    },
    NonZeroPaddingScale {
        group: usize,
        bits: u16,
    },
    NonZeroPlanePadding {
        index: usize,
        byte: u8,
    },
    ProposalMapExtent {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for NativeMtpDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeometry(reason) => {
                write!(formatter, "invalid native MTP geometry: {reason}")
            }
            Self::PlaneExtent {
                plane,
                expected,
                actual,
            } => write!(
                formatter,
                "native MTP {plane} plane has {actual} bytes, expected {expected}"
            ),
            Self::RowOutOfBounds { row, rows } => {
                write!(
                    formatter,
                    "native MTP row {row} is outside {rows} selected rows"
                )
            }
            Self::InvalidScale { group, bits } => write!(
                formatter,
                "native MTP scale group {group} has invalid FP16 bits 0x{bits:04x}"
            ),
            Self::ZeroScaleNonZeroCode { group, index } => write!(
                formatter,
                "native MTP zero-scale group {group} has nonzero code at K={index}"
            ),
            Self::InvalidQ8Code { index, code } => write!(
                formatter,
                "native MTP Q8 code 0x{code:02x} at K={index} is outside [-127, 127]"
            ),
            Self::ActivationExtent { expected, actual } => write!(
                formatter,
                "native MTP activation has {actual} values, expected {expected}"
            ),
            Self::NonFiniteActivation { index } => {
                write!(formatter, "native MTP activation {index} is nonfinite")
            }
            Self::NonZeroPaddingCode { index, code } => write!(
                formatter,
                "native MTP padding code at K={index} is {code}, expected zero"
            ),
            Self::NonZeroPaddingScale { group, bits } => write!(
                formatter,
                "native MTP padding scale group {group} has bits 0x{bits:04x}, expected positive zero"
            ),
            Self::NonZeroPlanePadding { index, byte } => write!(
                formatter,
                "native MTP plane padding byte at offset {index} is 0x{byte:02x}, expected zero"
            ),
            Self::ProposalMapExtent { expected, actual } => write!(
                formatter,
                "native MTP proposal map has {actual} rows, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for NativeMtpDecodeError {}
