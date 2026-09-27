//! Logical test matrices generated independently of CUDA fragment packing.

use crate::reference::{decode_e2m1, decode_ue4m3, matmul};

pub(super) struct Fixture {
    pub name: String,
    pub a: Vec<u8>,
    pub b: Vec<u8>,
    pub sa: Vec<u8>,
    pub sb: Vec<u8>,
    pub expected: Vec<f32>,
}

pub(super) fn fixtures() -> Result<Vec<Fixture>, String> {
    let mut result = vec![fixture(
        "ones",
        vec![2; 1024],
        vec![2; 512],
        vec![0x38; 64],
        vec![0x38; 32],
    )?];
    for seed in [1393_u64, 5090, 27] {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 32) as u8
        };
        let a = (0..1024).map(|_| next() & 15).collect();
        let b = (0..512).map(|_| next() & 15).collect();
        let scales = [0x28, 0x30, 0x38, 0x40];
        let sa = (0..64).map(|_| scales[(next() & 3) as usize]).collect();
        let sb = (0..32).map(|_| scales[(next() & 3) as usize]).collect();
        result.push(fixture(&format!("signed-{seed}"), a, b, sa, sb)?);
    }
    Ok(result)
}

fn fixture(
    name: &str,
    a: Vec<u8>,
    b: Vec<u8>,
    sa: Vec<u8>,
    sb: Vec<u8>,
) -> Result<Fixture, String> {
    let mut dense_a = Vec::with_capacity(1024);
    for row in 0..16 {
        for k in 0..64 {
            dense_a.push(decode_e2m1(a[row * 64 + k])? * decode_ue4m3(sa[row * 4 + k / 16])?);
        }
    }
    let mut dense_b = Vec::with_capacity(512);
    for k in 0..64 {
        for col in 0..8 {
            dense_b.push(decode_e2m1(b[k * 8 + col])? * decode_ue4m3(sb[(k / 16) * 8 + col])?);
        }
    }
    let expected = matmul(&dense_a, &dense_b, 16, 8, 64)?;
    Ok(Fixture {
        name: name.into(),
        a,
        b,
        sa,
        sb,
        expected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_reference_has_known_ones_and_signed_nonuniform_cases() {
        let cases = fixtures().unwrap();
        assert_eq!(cases.len(), 4);
        assert_eq!(cases[0].name, "ones");
        assert_eq!(cases[0].expected, vec![64.0; 128]);
        for case in &cases[1..] {
            assert_eq!(case.expected.len(), 128);
            assert!(case.expected.iter().any(|x| *x < 0.0));
            assert!(case.expected.iter().any(|x| *x > 0.0));
            assert_eq!(
                case.a
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                16
            );
            assert_eq!(case.b.len(), 512);
            assert_eq!(case.sa.len(), 64);
            assert_eq!(case.sb.len(), 32);
        }
    }
}
