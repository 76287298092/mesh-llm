//! Independent logical 8x8 matrix layouts for shared-memory load qualification.

pub(super) fn input(seed: u16) -> Vec<u32> {
    (0..128_u16)
        .map(|i| {
            let low = (2 * i).wrapping_mul(257).wrapping_add(seed);
            let high = (2 * i + 1).wrapping_mul(257).wrapping_add(seed);
            u32::from(low) | (u32::from(high) << 16)
        })
        .collect()
}

pub(super) fn expected(input: &[u32], mode: u32) -> Result<Vec<u32>, String> {
    if input.len() != 128 || mode >= 8 {
        return Err("shared-load fixture requires 128 words and mode 0..7".into());
    }
    let matrices = if mode & 2 == 0 { 2 } else { 4 };
    let transpose = mode & 4 != 0;
    let half = |matrix: usize, row: usize, column: usize| {
        let index = matrix * 64 + row * 8 + column;
        (input[index / 2] >> (16 * (index % 2))) & 0xffff
    };
    let mut result = vec![0; 128];
    for lane in 0..32 {
        let row = lane / 4;
        let column = (lane % 4) * 2;
        for matrix in 0..matrices {
            result[lane * 4 + matrix] = if transpose {
                half(matrix, column, row) | (half(matrix, column + 1, row) << 16)
            } else {
                half(matrix, row, column) | (half(matrix, row, column + 1) << 16)
            };
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_selects_columns_and_x2_zeroes_unused_words() {
        let input = input(1);
        let normal = expected(&input, 0).unwrap();
        let transposed = expected(&input, 4).unwrap();
        assert_eq!(normal[0], 1 | (258 << 16));
        assert_eq!(transposed[0], 1 | (2057 << 16));
        assert_eq!(&normal[2..4], &[0, 0]);
        assert_eq!(expected(&input, 2).unwrap()[3], 49345 | (49602 << 16));
        assert_eq!(expected(&input, 3).unwrap(), expected(&input, 2).unwrap());
        assert!(expected(&input[..127], 0).is_err());
        assert!(expected(&input, 8).is_err());
    }
}
