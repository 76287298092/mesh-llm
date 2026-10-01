use super::checks::{device_pointer, parent_range, saved_view};
use crate::packages::qwen3_8_27b::native_mtp_views::{
    Bf16NormView, BytePlane, Q4MatrixView, Q8MatrixView,
};

fn q8() -> Q8MatrixView {
    Q8MatrixView {
        object_id: "parent".into(),
        shape: [2, 128],
        padded_k: 128,
        group_size: 32,
        codes: BytePlane {
            offset: 0,
            bytes: 256,
        },
        scale_bits: BytePlane {
            offset: 256,
            bytes: 16,
        },
        scale_count: 8,
        source_rows: vec![0, 1],
    }
}

#[test]
fn parent_range_accepts_plane_when_it_ends_at_parent_boundary() {
    let plane = BytePlane {
        offset: 256,
        bytes: 16,
    };

    let result = parent_range(272, &plane);

    assert!(result.is_ok());
}

#[test]
fn parent_range_rejects_plane_when_it_overruns_parent() {
    let plane = BytePlane {
        offset: 256,
        bytes: 17,
    };

    let result = parent_range(272, &plane);

    assert!(result.is_err());
}

#[test]
fn parent_range_rejects_plane_when_addition_overflows() {
    let plane = BytePlane {
        offset: u64::MAX,
        bytes: 1,
    };

    let result = parent_range(u64::MAX, &plane);

    assert!(result.is_err());
}

#[test]
fn parent_range_rejects_empty_read_when_offset_is_outside_parent() {
    let plane = BytePlane {
        offset: 273,
        bytes: 0,
    };

    let result = parent_range(272, &plane);

    assert!(result.is_err());
}

#[test]
fn device_pointer_rejects_offset_when_addition_overflows() {
    let base = u64::MAX;

    let result = device_pointer(base, 1);

    assert!(result.is_err());
}

#[test]
fn q8_binding_uses_saved_metadata_when_request_is_equal_clone() {
    let saved = q8();
    let requested = saved.clone();

    let result = saved_view([&saved], &requested).unwrap();

    assert!(std::ptr::eq(result, &saved));
}

#[test]
fn q8_binding_rejects_foreign_view_when_same_parent_has_different_rows() {
    let saved = q8();
    let mut requested = saved.clone();
    requested.source_rows.reverse();

    let result = saved_view([&saved], &requested);

    assert!(result.is_err());
}

#[test]
fn q4_binding_rejects_foreign_view_when_plane_is_changed() {
    let saved = Q4MatrixView {
        object_id: "head".into(),
        shape: [2, 128],
        padded_k: 128,
        group_size: 64,
        codes: BytePlane {
            offset: 0,
            bytes: 128,
        },
        scale_bits: BytePlane {
            offset: 256,
            bytes: 8,
        },
        scale_count: 4,
        source_rows: vec![0, 1],
    };
    let mut requested = saved.clone();
    requested.scale_bits.offset = 128;

    let result = saved_view([&saved], &requested);

    assert!(result.is_err());
}

#[test]
fn native_norm_binding_resolves_equal_clone_to_saved_view() {
    let saved = Bf16NormView {
        object_id: "mtp-embedding-norm".into(),
        elements: 5120,
        bytes: 10_240,
    };
    let requested = saved.clone();

    let bound = saved_view([&saved], &requested).unwrap();

    assert!(std::ptr::eq(bound, &saved));
}

#[test]
fn native_norm_binding_rejects_view_with_changed_saved_extent() {
    let saved = Bf16NormView {
        object_id: "mtp-embedding-norm".into(),
        elements: 5120,
        bytes: 10_240,
    };
    let mut requested = saved.clone();
    requested.bytes = 10_238;

    let result = saved_view([&saved], &requested);

    assert!(result.is_err());
}
