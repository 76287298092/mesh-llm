use super::{
    binding::{encoded_table, validate_shape},
    output_extents, token_bytes, validate_tokens,
};
use crate::artifact::schema::DType;

#[test]
fn representation_dispatch_is_explicit_and_bounded() {
    assert!(!encoded_table(&DType::Bf16, 3, 2).unwrap());
    assert!(encoded_table(&DType::Fp8E4m3, 248_320, 5120).unwrap());
    assert!(encoded_table(&DType::Fp8E4m3, 248_320, 5119).is_err());
    assert!(encoded_table(&DType::Fp8E4m3, 248_319, 5120).is_err());
    assert!(encoded_table(&DType::F32, 248_320, 5120).is_err());
}

#[test]
fn validates_minimum_and_maximum_embedding_shapes() {
    assert_eq!(validate_shape(1, 1, 1e-6).unwrap(), (2, 2));
    assert_eq!(
        validate_shape(1_048_576, 32768, 1e-6).unwrap(),
        (68_719_476_736, 65_536)
    );
    assert_eq!(output_extents(1, 1).unwrap(), (1, 2, 4, 4));
    assert_eq!(
        output_extents(2048, 32768).unwrap(),
        (67_108_864, 134_217_728, 268_435_456, 8192)
    );
}

#[test]
fn rejects_invalid_embedding_dimensions_and_epsilon() {
    for (vocabulary, width) in [(0, 1), (1_048_577, 1), (1, 0), (1, 32769)] {
        assert!(validate_shape(vocabulary, width, 1e-6).is_err());
    }
    for epsilon in [0.0, f32::INFINITY, f32::NAN] {
        assert!(validate_shape(3, 2, epsilon).is_err());
    }
    assert!(output_extents(0, 2).is_err());
}

#[test]
fn validates_first_and_last_tokens_and_rejects_out_of_range_ids() {
    assert!(validate_tokens(3, &[0, 2]).is_ok());
    assert!(validate_tokens(3, &[3]).is_err());
    assert!(validate_tokens(3, &[]).is_err());
    assert!(validate_tokens(3, &vec![0; 2049]).is_err());
}

#[test]
fn serializes_token_ids_in_little_endian_order() {
    assert_eq!(token_bytes(&[0x0102_0304], 4).unwrap(), [4, 3, 2, 1]);
    assert!(token_bytes(&[1], 8).is_err());
}
