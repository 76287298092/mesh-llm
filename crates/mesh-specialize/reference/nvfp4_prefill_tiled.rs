//! Logical oracle for the experimental schedule, independent of GPU fragments.
use crate::{
    nvfp4_linear_reference::{self, Matrix},
    projection_reference::LinearReference,
};
use anyhow::{Result, ensure};

/// Restrict the independent decoded-product oracle to candidate admission bounds.
pub fn run(input: Matrix<'_>, weights: Matrix<'_>, width: usize) -> Result<LinearReference> {
    ensure!(
        (1..=512).contains(&input.rows),
        "prefill M must be in 1..=512"
    );
    ensure!(
        (8..=32768).contains(&weights.rows) && weights.rows.is_multiple_of(8),
        "prefill N must be a multiple of 8 in 8..=32768"
    );
    ensure!(
        (64..=32768).contains(&width) && width.is_multiple_of(64),
        "prefill K must be a multiple of 64 in 64..=32768"
    );
    nvfp4_linear_reference::run(input, weights, width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_column_mapping_across_cta_boundaries_and_partial_tiles() {
        let (m, n, k) = (33, 40, 128);
        let mut a = vec![0; m * k / 2];
        let mut w = vec![0; n * k / 2];
        // A row r selects K=r. W row c has a signed alternating value.
        for row in 0..m {
            a[row * k / 2 + row / 2] = 2 << ((row % 2) * 4);
        }
        for column in 0..n {
            for index in 0..k {
                let code = [2, 4, 6][index % 3] | if column % 2 == 0 { 0 } else { 8 };
                w[column * k / 2 + index / 2] |= code << ((index % 2) * 4);
            }
        }
        let sa = vec![0x38; m * k / 16];
        let sw = vec![0x38; n * k / 16];
        let result = run(
            Matrix {
                packed: &a,
                scales: &sa,
                rows: m,
                global: 1.0,
            },
            Matrix {
                packed: &w,
                scales: &sw,
                rows: n,
                global: 1.0,
            },
            k,
        )
        .unwrap();
        for row in 0..m {
            for column in 0..n {
                assert_eq!(
                    result.unrounded[row * n + column],
                    [1.0, 2.0, 4.0][row % 3] * if column % 2 == 0 { 1.0 } else { -1.0 }
                );
            }
        }
    }

    #[test]
    fn cancellation_scale_groups_and_three_k_tiles_have_hand_sum() {
        let mut a = vec![0; 96];
        let mut w = vec![0; 8 * 96];
        // Products at K=0,16,32,64,128: 1,-2,4,-1,1. Three tiles reuse slot zero.
        for (index, code) in [(0, 2), (16, 10), (32, 2), (64, 10), (128, 2)] {
            a[index / 2] = code;
            for row in 0..8 {
                w[row * 96 + index / 2] = 2;
            }
        }
        let sa = [0x38, 0x40, 0x48, 0, 0x38, 0, 0, 0, 0x38, 0, 0, 0];
        let sw = vec![0x38; 96];
        let result = run(
            Matrix {
                packed: &a,
                scales: &sa,
                rows: 1,
                global: 2.0,
            },
            Matrix {
                packed: &w,
                scales: &sw,
                rows: 8,
                global: 1.0,
            },
            192,
        )
        .unwrap();
        assert_eq!(result.unrounded, [1.5; 8]);
        assert_eq!(result.absolute_sums, [4.5; 8]);
    }

    #[test]
    fn rejects_shapes_outside_candidate_before_scalar_dispatch() {
        let empty = Matrix {
            packed: &[],
            scales: &[],
            rows: 1,
            global: 1.0,
        };
        for (m, n, k) in [(513, 8, 64), (1, 7, 64), (1, 8, 80), (0, 8, 64)] {
            assert!(run(Matrix { rows: m, ..empty }, Matrix { rows: n, ..empty }, k).is_err());
        }
    }
}
