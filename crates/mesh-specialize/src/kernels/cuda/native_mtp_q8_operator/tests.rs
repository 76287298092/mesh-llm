use super::{fixture, validate};

#[test]
fn host_boundary_accepts_valid_parent_mapped_view() {
    let (object, view, input) = fixture::fixture_for_test(160).expect("fixture construction");
    let validated = validate::validate(&object, &view, &input).expect("valid mapped view");
    assert_eq!(validated.source_rows, [2, 0]);
    assert_eq!(validated.scale_offset, 768);
    assert_eq!(validated.padded_k, 256);
}

#[test]
fn host_boundary_accepts_all_synthetic_widths_with_parent_mapping() {
    for width in [128, 160, 256, 5120, 10240, 17408] {
        let (object, view, input) = fixture::fixture_for_test(width).expect("fixture construction");
        let validated = validate::validate(&object, &view, &input).expect("valid mapped view");

        assert_eq!(validated.logical_k, u32::try_from(width).expect("fixture width fits u32"));
        assert_eq!(validated.source_rows, [2, 0]);
    }
}

#[test]
fn host_boundary_rejects_invalid_q8_code_before_launch() {
    let (mut object, view, input) = fixture::fixture_for_test(128).expect("fixture construction");
    object[2 * 128] = 0x80;
    assert!(validate::validate(&object, &view, &input).is_err());
}

#[test]
fn host_boundary_rejects_nonzero_logical_tail_code() {
    let (mut object, view, input) = fixture::fixture_for_test(160).expect("fixture construction");
    object[2 * 256 + 160] = 1;
    assert!(validate::validate(&object, &view, &input).is_err());
}
