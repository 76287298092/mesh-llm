use super::super::BytePlane;
use crate::artifact::ninfer::Object;
use anyhow::{Context, Result, bail, ensure};

const ROW_ALIGNMENT: u64 = 128;
const PLANE_ALIGNMENT: u64 = 256;
const SCALE_ELEMENT_BYTES: u64 = 2;

type PackedPlanes = (usize, usize, BytePlane, BytePlane, usize);

pub(super) fn packed_planes(object: &Object) -> Result<PackedPlanes> {
    ensure!(
        object.layout.as_deref() == Some("row_split_k128_v1"),
        "unsupported native MTP quantized layout"
    );
    let format = object
        .format
        .as_deref()
        .context("native packed format is missing")?;
    let (bits, group_size) = match format {
        "q8_g32_fp16" => (8_u64, 32_usize),
        "q4_g64_fp16" => (4_u64, 64_usize),
        other => bail!("unsupported native MTP quantization {other}"),
    };
    let [rows, columns]: [u64; 2] = object
        .shape
        .as_slice()
        .try_into()
        .context("native packed view parent must be rank two")?;
    let padded_k = align(columns, ROW_ALIGNMENT)?;
    let padded_elements = rows
        .checked_mul(padded_k)
        .context("native packed plane element count overflows")?;
    let code_bytes = match bits {
        8 => padded_elements,
        4 => padded_elements / 2,
        _ => bail!("unsupported native MTP quantization {format}"),
    };
    let scale_offset = align(code_bytes, PLANE_ALIGNMENT)?;
    let scale_count = padded_elements / u64::try_from(group_size)?;
    let scale_bytes = scale_count
        .checked_mul(SCALE_ELEMENT_BYTES)
        .context("native packed scale plane overflows")?;
    let expected_bytes = scale_offset
        .checked_add(scale_bytes)
        .context("native packed object extent overflows")?;
    ensure!(
        object.bytes == expected_bytes,
        "native packed object byte length mismatch"
    );
    Ok((
        usize::try_from(padded_k)?,
        group_size,
        BytePlane {
            offset: 0,
            bytes: code_bytes,
        },
        BytePlane {
            offset: scale_offset,
            bytes: scale_bytes,
        },
        usize::try_from(scale_count)?,
    ))
}

fn align(value: u64, alignment: u64) -> Result<u64> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .context("native packed plane alignment overflows")
    }
}

#[cfg(test)]
mod tests {
    use super::packed_planes;
    use crate::artifact::ninfer::Object;

    fn object(format: &str, shape: [u64; 2], bytes: u64) -> Object {
        Object {
            id: "packed".into(),
            kind: "tensor".into(),
            format: Some(format.into()),
            layout: Some("row_split_k128_v1".into()),
            shape: shape.to_vec(),
            encoding: None,
            offset: 0,
            bytes,
        }
    }

    #[test]
    fn row_split_geometry_pads_k_before_counting_groups_and_planes() {
        let q8 = object("q8_g32_fp16", [3, 129], 816);
        let (padded_k, group_size, codes, scales, scale_count) =
            packed_planes(&q8).expect("valid Q8 geometry");
        assert_eq!(padded_k, 256);
        assert_eq!(group_size, 32);
        assert_eq!((codes.offset, codes.bytes), (0, 768));
        assert_eq!((scales.offset, scales.bytes), (768, 48));
        assert_eq!(scale_count, 24);

        let q4 = object("q4_g64_fp16", [3, 129], 536);
        let (padded_k, group_size, codes, scales, scale_count) =
            packed_planes(&q4).expect("valid Q4 geometry");
        assert_eq!(padded_k, 256);
        assert_eq!(group_size, 64);
        assert_eq!((codes.offset, codes.bytes), (0, 384));
        assert_eq!((scales.offset, scales.bytes), (512, 24));
        assert_eq!(scale_count, 12);
    }

    #[test]
    fn row_split_geometry_rejects_wrong_bytes_and_unknown_quantization() {
        let malformed = object("q8_g32_fp16", [3, 129], 815);
        assert!(packed_planes(&malformed).is_err());
        let unsupported = object("q5_g64_fp16", [3, 129], 792);
        assert!(packed_planes(&unsupported).is_err());
    }

    #[test]
    fn row_split_scale_plane_count_matches_schema_byte_formula() {
        let q8 = object("q8_g32_fp16", [3, 129], 816);
        let (_, _, _, q8_scales, q8_count) = packed_planes(&q8).expect("Q8 planes");
        assert!(
            u64::try_from(q8_count)
                .ok()
                .and_then(|count| count.checked_mul(2))
                .is_some_and(|bytes| bytes == q8_scales.bytes)
        );
        assert_eq!(q8_count, 24);
        let q4 = object("q4_g64_fp16", [3, 129], 536);
        let (_, _, _, q4_scales, q4_count) = packed_planes(&q4).expect("Q4 planes");
        assert!(
            u64::try_from(q4_count)
                .ok()
                .and_then(|count| count.checked_mul(2))
                .is_some_and(|bytes| bytes == q4_scales.bytes)
        );
        assert_eq!(q4_count, 12);
    }
}
