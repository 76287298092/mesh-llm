use super::{compare, fixture, schedule_reference, validate};
use crate::native_mtp_q4_gemv_reference;

#[test]
fn host_boundary_accepts_q4_layout_boundaries_and_k5120() {
    for width in [128, 160, 256, 5120] {
        let (object, view, input) = fixture::fixture_for_test(width).expect("fixture construction");
        let validated = validate::validate(&object, &view, &input).expect("valid Q4 view");

        assert_eq!(
            validated.logical_k,
            u32::try_from(width).expect("K fits u32")
        );
        assert_eq!(validated.source_rows, [2, 0]);
        assert_eq!(
            validated.padded_k,
            u32::try_from(width.div_ceil(128) * 128).expect("padded K fits u32")
        );
    }
}

#[test]
fn host_boundary_rejects_invalid_scale_and_nonzero_padded_nibble() {
    let (mut object, view, input) = fixture::fixture_for_test(160).expect("fixture construction");
    let scale_offset = usize::try_from(view.scale_bits.offset).expect("scale offset fits usize");
    object[scale_offset + 16..scale_offset + 18].copy_from_slice(&0x8000_u16.to_le_bytes());
    assert!(validate::validate(&object, &view, &input).is_err());

    let (mut object, view, input) = fixture::fixture_for_test(160).expect("fixture construction");
    let parent_row = view.source_rows[0];
    let tail_index = parent_row * (view.padded_k / 2) + 160 / 2;
    object[tail_index] = 0x01;
    assert!(validate::validate(&object, &view, &input).is_err());
}

#[test]
fn argmax_keeps_the_first_bf16_tie_and_remaps_that_proposal_row() {
    let logits = [0x3f80_u16, 0x3f80, 0x3f00];
    let proposal_tokens = [91_337_u32, 17, 5];
    let winner = compare::first_argmax(&logits).expect("finite proposal logits");

    assert_eq!(winner, 0);
    assert_eq!(proposal_tokens[winner], 91_337);
}

#[test]
fn independent_oracle_returns_fp64_dot_and_rne_bf16_logit() {
    let (object, view, input) = fixture::fixture_for_test(128).expect("fixture construction");
    let output = native_mtp_q4_gemv_reference::run(&object, &view, &input)
        .expect("independent Q4 projection");

    assert_eq!(output.raw_f64, [3.0 - 7.0 * 2.0_f64.powi(-24), 6.0]);
    assert_eq!(output.logits_bf16, [0x4040, 0x40c0]);
}

#[test]
fn independent_oracle_covers_k160_group_and_tail_addresses() {
    let (object, view, input) = fixture::fixture_for_test(160).expect("fixture construction");
    let output = native_mtp_q4_gemv_reference::run(&object, &view, &input)
        .expect("independent K160 Q4 projection");

    assert_eq!(output.raw_f64, [1.0 - 7.0 * 2.0_f64.powi(-24), 5.0]);
    assert_eq!(output.logits_bf16, [0x3f80, 0x40a0]);
}

#[test]
fn r4w1_schedule_reference_covers_partial_row_group_and_reordered_parent_rows() {
    let (object, mut view, input) = fixture::fixture_for_test(128).expect("fixture construction");
    view.shape[0] = 3;
    view.source_rows.push(1);
    let validated = validate::validate(&object, &view, &input).expect("valid selected Q4 rows");

    let output =
        schedule_reference::run(&object, &input, &validated).expect("R4W1 schedule reference");

    assert_eq!(
        output.raw_f32[0].to_bits(),
        (3.0_f32 - 2.0_f32.powi(-21)).to_bits()
    );
    assert_eq!(output.raw_f32[1].to_bits(), 6.0_f32.to_bits());
    assert_eq!(output.raw_f32[2].to_bits(), 0.0_f32.to_bits());
    assert_eq!(output.logits_bf16, [0x4040, 0x40c0, 0]);
}

#[test]
fn r4w1_schedule_reference_handles_partial_k_group_and_vector_tail() {
    let (object, view, input) = fixture::fixture_for_test(160).expect("fixture construction");
    let validated = validate::validate(&object, &view, &input).expect("valid logical K tail");

    let output = schedule_reference::run(&object, &input, &validated)
        .expect("R4W1 schedule reference with K tail");

    assert_eq!(
        output.raw_f32[0].to_bits(),
        (1.0_f32 - 2.0_f32.powi(-21)).to_bits()
    );
    assert_eq!(output.raw_f32[1].to_bits(), 5.0_f32.to_bits());
    assert_eq!(output.logits_bf16, [0x3f80, 0x40a0]);
}

#[test]
fn r4w1_schedule_reference_reads_logical_k_remainder_after_last_full_vector() {
    let (mut object, view, input) = fixture::fixture_for_test(163).expect("fixture construction");
    let parent_row = view.source_rows[0];
    let last_code = parent_row * (view.padded_k / 2) + 162 / 2;
    object[last_code] |= 0x01;
    let validated = validate::validate(&object, &view, &input).expect("valid three-element tail");

    let output = schedule_reference::run(&object, &input, &validated)
        .expect("R4W1 schedule reference at vector tail");

    assert_eq!(
        output.raw_f32[0].to_bits(),
        (2.0_f32 - 7.0 * 2.0_f32.powi(-24)).to_bits()
    );
}
