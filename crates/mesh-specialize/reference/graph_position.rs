//! Host-only position/address checks using independent logical attention oracles.
//! These tests do not execute PTX or certify graph replay; qwen-stream-check in
//! graph mode must compare actual device results and every persistent state byte.

use crate::{attention_prepare_reference as prepare, causal_attention_reference as attention};

#[test]
fn base_rope_addressing_matches_independently_generated_single_positions() {
    let capacity = 33;
    let rotary = 4;
    let positions: Vec<u32> = (0..capacity).collect();
    let (cos, sin) = prepare::text_rope_tables(&positions, rotary, 10_000.0).unwrap();
    for past in [0, 1, 7, 16, capacity - 1] {
        for with_gate in [false, true] {
            let shape = prepare::Shape {
                rows: 1,
                heads: 2,
                width: 8,
                rotary_dim: rotary,
                with_gate,
            };
            let input: Vec<u16> = (0..16 * (1 + usize::from(with_gate)))
                .map(|i| crate::entry_reference::round_bf16((i as f32 - 9.0) / 16.0))
                .collect();
            let weight = vec![0; 8];
            // BF16 addressing, not byte addressing: kernel pointer.add multiplies by two.
            let begin = past as usize * rotary / 2;
            let end = begin + rotary / 2;
            let based = prepare::run(
                &input,
                &weight,
                &cos[begin..end],
                &sin[begin..end],
                &shape,
                1.0e-6,
            )
            .unwrap();
            let (single_cos, single_sin) =
                prepare::text_rope_tables(&[past], rotary, 10_000.0).unwrap();
            let compact =
                prepare::run(&input, &weight, &single_cos, &single_sin, &shape, 1.0e-6).unwrap();
            assert_eq!(based, compact);
            assert_eq!(begin * 2, past as usize * rotary);
        }
    }
}

#[test]
fn append_offsets_and_causal_lengths_cover_nonzero_and_last_positions() {
    let capacity = 33;
    let row_width = 2 * 4;
    for past in [0, 1, 7, 16, capacity - 1] {
        let shape = attention::Shape {
            rows: 1,
            query_heads: 4,
            kv_heads: 2,
            width: 4,
            past,
            capacity,
            scale: 0.5,
        };
        let poison = 0x7fc0;
        let mut keys = vec![poison; capacity * row_width];
        let mut values = keys.clone();
        let prefix = past * row_width;
        keys[..prefix].fill(0x3f00);
        values[..prefix].fill(0x3f80);
        let old_keys = keys.clone();
        let old_values = values.clone();
        let appended = vec![0x4000; row_width];
        attention::append(&appended, &appended, &mut keys, &mut values, &shape).unwrap();
        assert_eq!(&keys[..prefix], &old_keys[..prefix]);
        assert_eq!(&values[..prefix], &old_values[..prefix]);
        assert_eq!(&keys[prefix..prefix + row_width], &appended);
        assert_eq!(&values[prefix..prefix + row_width], &appended);
        assert!(keys[prefix + row_width..].iter().all(|&x| x == poison));
        assert!(values[prefix + row_width..].iter().all(|&x| x == poison));
        let q = vec![0x3f80; 4 * 4];
        let full = attention::run(&q, &keys, &values, &shape).unwrap();
        let length = past + 1;
        let compact_shape = attention::Shape {
            capacity: length,
            ..shape
        };
        let compact = attention::run(
            &q,
            &keys[..length * row_width],
            &values[..length * row_width],
            &compact_shape,
        )
        .unwrap();
        assert_eq!(full, compact);
    }
}

#[test]
fn position_entries_delegate_to_exact_bodies_without_new_arithmetic() {
    let adapters = include_str!("../kernels/nvptx/graph_position.rs");
    for symbol in ["prepare_body(", "append_body(", "attention_body::<false>("] {
        assert!(adapters.contains(symbol));
    }
    assert!(!adapters.contains("attention_body::<true>"));
    assert!(!adapters.contains("asm!"));
    assert!(!adapters.contains("extern \"C\""));
    assert!(adapters.contains("*past as usize * (rotary_dim / 2) as usize"));
}
