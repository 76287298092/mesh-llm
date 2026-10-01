use super::{checks, execution, selection};
use crate::{
    engine::session::Cursor,
    kernels::DecoderConfig,
    packages::qwen3_8_27b::native_mtp_views::{
        Bf16NormView, BytePlane, NativeMtpNormViews, NativeMtpViews, Q4MatrixView, Q8MatrixView,
    },
};

fn q8(
    object_id: &str,
    shape: [usize; 2],
    parent_rows: usize,
    source_rows: Vec<usize>,
) -> Q8MatrixView {
    let k = shape[1];
    let code_bytes = parent_rows * k;
    let scale_count = parent_rows * k / 32;
    let alignment_padding = if object_id == "fc" {
        0
    } else {
        (256 - code_bytes % 256) % 256
    };
    let scale_offset = code_bytes + alignment_padding;
    Q8MatrixView {
        object_id: object_id.into(),
        shape,
        padded_k: k,
        group_size: 32,
        codes: BytePlane {
            offset: 0,
            bytes: u64::try_from(code_bytes).expect("codes fit u64"),
        },
        scale_bits: BytePlane {
            offset: u64::try_from(scale_offset).expect("scale offset fits u64"),
            bytes: u64::try_from(scale_count * 2).expect("scale bytes fit u64"),
        },
        scale_count,
        source_rows,
    }
}

fn views() -> NativeMtpViews {
    let [query_rows, _, gate_rows, _] = selection::physical_qkv_rows();
    let qkv = q8(
        "qkv",
        [12_288, 5_120],
        14_336,
        selection::interleaved_q_gate_rows(query_rows, gate_rows),
    );
    let mlp = q8("mlp", [17_408, 5_120], 34_816, (0..17_408).collect());
    let norm = |name: &str, elements| Bf16NormView {
        object_id: name.into(),
        elements,
        bytes: u64::try_from(elements * 2).expect("norm bytes fit u64"),
    };
    NativeMtpViews {
        fc: q8("fc", [5_120, 10_240], 5_120, (0..5_120).collect()),
        query_gate: qkv,
        key: q8("qkv", [1_024, 5_120], 14_336, (6_144..7_168).collect()),
        value: q8("qkv", [1_024, 5_120], 14_336, (13_312..14_336).collect()),
        attention_output: q8(
            "attention-output",
            [5_120, 6_144],
            5_120,
            (0..5_120).collect(),
        ),
        mlp_gate: mlp.clone(),
        mlp_up: q8("mlp", [17_408, 5_120], 34_816, (17_408..34_816).collect()),
        mlp_down: q8("mlp-down", [5_120, 17_408], 5_120, (0..5_120).collect()),
        norms: NativeMtpNormViews {
            embedding: norm("embedding", 5_120),
            hidden: norm("hidden", 5_120),
            final_norm: norm("final", 5_120),
            input: norm("input", 5_120),
            post_attention: norm("post", 5_120),
            query: norm("query", 256),
            key: norm("key", 256),
        },
        proposal_head: Q4MatrixView {
            object_id: "q4".into(),
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
        },
        proposal_tokens: crate::packages::qwen3_8_27b::native_mtp_views::parse_proposal_token_map(
            &(0..131_072_i32)
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .expect("signed shortlist target map"),
    }
}

#[test]
fn saved_native_views_validate_complete_physical_qkv_and_gate_up_maps() {
    let given = views();

    let when = selection::validate_views(&given);

    assert!(when.is_ok());
}

#[test]
fn saved_native_views_reject_wrong_key_parent_slice() {
    let mut given = views();
    given.key.source_rows[0] = 7_168;

    let when = selection::validate_views(&given);

    assert!(when.is_err());
}

#[test]
fn saved_native_views_reject_swapped_gate_up_half_rows() {
    let mut given = views();
    given.mlp_up.source_rows.swap(0, 1);

    let when = selection::validate_views(&given);

    assert!(when.is_err());
}

#[test]
fn q_gate_logical_rows_interleave_each_physical_head_pair() {
    let [query, _, gate, _] = selection::physical_qkv_rows();
    let mapped = selection::interleaved_q_gate_rows(query, gate);

    assert_eq!(&mapped[..3], [0, 1, 2]);
    assert_eq!(mapped[255], 255);
    assert_eq!(mapped[256], 7_168);
    assert_eq!(mapped[511], 7_423);
    assert_eq!(mapped[512], 256);
    assert_eq!(mapped[768], 7_424);
}

#[test]
fn saved_native_views_reject_wrong_qkv_gate_interleave_row() {
    let mut given = views();
    given.query_gate.source_rows[256] = 256;

    let when = selection::validate_views(&given);

    assert!(when.is_err());
}

#[test]
fn shortlist_first_tie_maps_the_winning_row_to_target_id() {
    let mut logits = vec![0_u16; 131_072];
    logits[11] = 0x3f80;
    logits[21] = 0x3f80;
    let map = (0..131_072_i32)
        .map(|row| if row == 11 { 200_000 } else { row })
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    let tokens = crate::packages::qwen3_8_27b::native_mtp_views::parse_proposal_token_map(&map)
        .expect("signed proposal token mapping");

    let row = execution::first_argmax(&logits).expect("finite shortlist");

    assert_eq!(row, 11);
    assert_eq!(
        tokens.target_id(row).map(|token| token.value()),
        Some(200_000)
    );
    assert_eq!(tokens.len(), 131_072);
}

#[test]
fn forward_preflight_rejects_capacity_before_cursor_transaction() {
    let given = cursor_at_capacity();

    let when = checks::validate_session_capacity(&given, 1);

    assert!(when.is_err());
    assert_eq!(given.past(), 2);
    assert!(!given.is_poisoned());
}

#[test]
fn forward_preflight_accepts_final_row_at_capacity_boundary() {
    let mut given = Cursor::new(2).expect("bounded cursor");
    given.begin(1).expect("first row").commit();

    let when = checks::validate_session_capacity(&given, 1);

    assert!(when.is_ok());
    assert_eq!(given.begin(1).expect("exact capacity row").commit(), 2);
    assert_eq!(given.past(), 2);
}

fn cursor_at_capacity() -> Cursor {
    let mut cursor = Cursor::new(2).expect("bounded cursor");
    cursor.begin(2).expect("fill capacity").commit();
    cursor
}

#[test]
fn native_model_config_requires_fixed_attention_geometry() {
    let mut config = DecoderConfig {
        layers: Vec::new(),
        gdn_shape: crate::kernels::GdnShape {
            hidden: 5_120,
            intermediate: 17_408,
            key_heads: 16,
            value_heads: 48,
            head_width: 128,
        },
        attention_shape: crate::kernels::ResidentAttentionShape {
            hidden: 5_120,
            intermediate: 17_408,
            query_heads: 24,
            kv_heads: 4,
            head_width: 256,
            rotary_dim: 64,
            rope_theta: 1e7,
        },
        embedding_table: String::new(),
        first_norm: String::new(),
        final_norm: String::new(),
        head_prefix: String::new(),
        hidden: 5_120,
        vocabulary: 248_320,
        capacity: 32,
        state_layout: crate::engine::layout::Layout::new([("test.state".to_owned(), 2)])
            .expect("valid layout"),
    };
    assert!(checks::validate_model_config(&config).is_ok());
    config.attention_shape.head_width = 128;

    let when = checks::validate_model_config(&config);

    assert!(when.is_err());
}
