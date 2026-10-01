use super::super::super::schedule_reference::ScheduleProjection;
use super::super::super::validate::ValidatedView;
use super::super::comparison::{Case, References, evaluate, failed_case, first_argmax};
use super::super::launch::BoundOutput;
use crate::native_mtp_q4_gemv_reference::Q4ProjectionReference;
use serde_json::json;

struct Fixture {
    fp64: Q4ProjectionReference,
    scheduled: ScheduleProjection,
    validated: ValidatedView,
    outputs: [BoundOutput; 2],
    target_ids: Vec<u32>,
}

impl Fixture {
    fn new(raw: &[f32], logits: &[u16], target_ids: &[u32]) -> Self {
        let raw_f32 = raw.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        let count = raw.len();
        Self {
            fp64: Q4ProjectionReference {
                raw_f64: raw.iter().map(|value| f64::from(*value)).collect(),
                logits_bf16: logits.to_vec(),
            },
            scheduled: ScheduleProjection {
                raw_f32: raw.to_vec(),
                logits_bf16: logits.to_vec(),
            },
            validated: ValidatedView {
                selected_rows: u32::try_from(count).expect("test row count fits u32"),
                logical_k: 5_120,
                padded_k: 5_120,
                scale_offset: 0,
                source_rows: (0..count)
                    .map(|row| u32::try_from(row).expect("test parent row fits u32"))
                    .collect(),
            },
            outputs: [
                BoundOutput {
                    raw_f32_bits: raw_f32.clone(),
                    logits_bf16: logits.to_vec(),
                },
                BoundOutput {
                    raw_f32_bits: raw_f32,
                    logits_bf16: logits.to_vec(),
                },
            ],
            target_ids: target_ids.to_vec(),
        }
    }

    fn evaluate(self, name: &'static str) -> anyhow::Result<serde_json::Value> {
        let Self {
            fp64,
            scheduled,
            validated,
            outputs,
            target_ids,
        } = self;
        evaluate(Case {
            name,
            outputs,
            references: References {
                fp64: &fp64,
                scheduled: &scheduled,
                validated: &validated,
            },
            target_ids: &target_ids,
        })
    }
}

#[test]
fn detects_interior_corruption_even_when_the_winner_is_unchanged() {
    let mut given = Fixture::new(&[1.0, 0.25, -0.5], &[0x3f80, 0x3e80, 0xbf00], &[8, 9, 10]);
    given.outputs[0].raw_f32_bits[1] = 0x3f80_0000;
    given.outputs[0].logits_bf16[1] = 0x3f80;

    let when = given
        .evaluate("interior-corruption")
        .expect("comparison report");

    assert_eq!(when["all_passed"], json!(false));
    assert_eq!(when["failures_first_16"][0]["proposal_row"], json!(1));
    assert_eq!(
        when["selection"]["expected_schedule_proposal_row"],
        json!(0)
    );
}

#[test]
fn rejects_repeats_that_are_identically_wrong() {
    let mut given = Fixture::new(&[1.0], &[0x3f80], &[200_000]);
    for output in &mut given.outputs {
        output.raw_f32_bits[0] = 0x4000_0000;
        output.logits_bf16[0] = 0x4000;
    }

    let when = given
        .evaluate("identically-wrong-repeats")
        .expect("comparison report");

    assert_eq!(when["all_passed"], json!(false));
    assert_eq!(when["repeat_raw_bit_mismatches"], json!(0));
    assert_eq!(when["schedule_raw_f32_bit_mismatches"], json!(2));
}

#[test]
fn rejects_nonfinite_raw_output() {
    let mut given = Fixture::new(&[1.0], &[0x3f80], &[200_000]);
    given.outputs[1].raw_f32_bits[0] = f32::NAN.to_bits();

    let when = given
        .evaluate("nonfinite-output")
        .expect("comparison report");

    assert_eq!(when["all_passed"], json!(false));
    assert_eq!(when["nonfinite_outputs"], json!(1));
}

#[test]
fn rejects_nonfinite_bf16_output() {
    let mut given = Fixture::new(&[1.0], &[0x3f80], &[200_000]);
    given.outputs[0].logits_bf16[0] = 0x7f80;

    let when = given
        .evaluate("nonfinite-bf16-output")
        .expect("comparison report");

    assert_eq!(when["all_passed"], json!(false));
    assert_eq!(
        when["selection"]["expected_schedule_proposal_row"],
        json!(0)
    );
    assert_eq!(when["selection"]["first_proposal_row"], json!(null));
}

#[test]
fn rejects_output_extent_mismatch() {
    let mut given = Fixture::new(&[1.0, 2.0], &[0x3f80, 0x4000], &[8, 9]);
    given.outputs[1].logits_bf16.pop();

    let when = given.evaluate("wrong-output-extent");

    let error = when.expect_err("extent mismatch should fail comparison");
    let report = failed_case("wrong-output-extent", &error);
    assert_eq!(report["all_passed"], json!(false));
    assert_eq!(report["case"], json!("wrong-output-extent"));
    assert!(
        report["error"]
            .as_str()
            .is_some_and(|message| message.contains("extent"))
    );
}

#[test]
fn rejects_repeat_mismatch_against_the_schedule() {
    let mut given = Fixture::new(&[1.0], &[0x3f80], &[200_000]);
    given.outputs[1].raw_f32_bits[0] = 0x3f80_0001;

    let when = given
        .evaluate("repeat-mismatch")
        .expect("comparison report");

    assert_eq!(when["all_passed"], json!(false));
    assert_eq!(when["repeat_raw_bit_mismatches"], json!(1));
}

#[test]
fn first_tie_winner_remaps_to_target_vocab_outside_shortlist() {
    let given = Fixture::new(&[1.0, 1.0], &[0x3f80, 0x3f80], &[200_000, 17]);

    let when = given
        .evaluate("first-tie-remap")
        .expect("comparison report");

    assert_eq!(first_argmax(&[0x3f80, 0x3f80]), Some(0));
    assert_eq!(
        when["selection"]["expected_schedule_proposal_row"],
        json!(0)
    );
    assert_eq!(when["selection"]["expected_target_token"], json!(200_000));
    assert_eq!(when["selection"]["signed_map_remap_matches"], json!(true));
}
