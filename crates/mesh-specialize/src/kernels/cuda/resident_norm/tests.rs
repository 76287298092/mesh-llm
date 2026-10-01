use super::extents::{
    MAX_ELEMENTS, matrix_extents, validate_add_extents, validate_matrix_buffer,
    validate_matrix_pair,
};

#[test]
fn validates_norm_input_at_minimum_and_maximum_extents() {
    assert_eq!(matrix_extents(1, 1).unwrap(), (1, 2, 4));
    assert_eq!(
        validate_matrix_buffer(134_217_728, 2048, 32768).unwrap(),
        (MAX_ELEMENTS, 134_217_728, 268_435_456)
    );
}

#[test]
fn rejects_invalid_norm_shapes_and_input_byte_lengths() {
    assert!(matrix_extents(0, 1).is_err());
    assert!(matrix_extents(2049, 1).is_err());
    assert!(matrix_extents(1, 0).is_err());
    assert!(matrix_extents(1, 32769).is_err());
    assert!(validate_matrix_buffer(4, 1, 1).is_err());
    assert!(validate_matrix_pair(2, 4, 1, 1).is_err());
}

#[test]
fn validates_residual_add_even_extents_and_count_bounds() {
    assert_eq!(validate_add_extents(2, 2).unwrap(), (1, 2));
    assert_eq!(
        validate_add_extents(134_217_728, 134_217_728).unwrap(),
        (MAX_ELEMENTS, 134_217_728)
    );
    assert!(validate_add_extents(0, 0).is_err());
    assert!(validate_add_extents(2, 4).is_err());
    assert!(validate_add_extents(3, 3).is_err());
    assert!(validate_add_extents(134_217_730, 134_217_730).is_err());
}
