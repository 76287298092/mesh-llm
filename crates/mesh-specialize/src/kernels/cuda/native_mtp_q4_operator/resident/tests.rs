use super::validate_head;
use crate::packages::qwen3_8_27b::native_mtp_views::{BytePlane, Q4MatrixView};

fn proposal_head() -> Q4MatrixView {
    Q4MatrixView {
        object_id: "proposal/head".into(),
        shape: [131_072, 5_120],
        padded_k: 5_120,
        group_size: 64,
        codes: BytePlane {
            offset: 0,
            bytes: 335_544_320,
        },
        scale_bits: BytePlane {
            offset: 335_544_320,
            bytes: 20_971_520,
        },
        scale_count: 10_485_760,
        source_rows: (0..131_072).collect(),
    }
}

#[test]
fn full_proposal_parent_extent_matches_packed_planes() {
    let given = proposal_head();

    let when = validate_head(&given, 356_515_840);

    assert!(when.is_ok());
}

#[test]
fn proposal_parent_rejects_wrong_physical_extent() {
    let given = proposal_head();

    let when = validate_head(&given, 356_515_839);

    assert!(when.is_err());
}

#[test]
fn proposal_parent_rejects_nonidentity_row_order() {
    let mut given = proposal_head();
    given.source_rows[131_071] = 0;

    let when = validate_head(&given, 356_515_840);

    assert!(when.is_err());
}
