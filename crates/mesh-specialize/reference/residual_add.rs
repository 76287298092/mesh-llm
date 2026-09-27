//! Independent BF16 residual addition and rounding boundary.
use crate::entry_reference::{bf16_to_f32, round_bf16};
use anyhow::{Result, ensure};
pub fn run(left: &[u16], right: &[u16]) -> Result<Vec<u16>> {
    ensure!(
        (1..=67108864).contains(&left.len()) && left.len() == right.len(),
        "residual add extent mismatch"
    );
    left.iter()
        .zip(right)
        .map(|(&l, &r)| {
            let (l, r) = (bf16_to_f32(l), bf16_to_f32(r));
            ensure!(
                l.is_finite() && r.is_finite(),
                "residual add nonfinite input"
            );
            let output = round_bf16(l + r);
            ensure!(bf16_to_f32(output).is_finite(), "residual add overflow");
            Ok(output)
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_add_checks_rounding_signed_zero_and_overflow() {
        assert_eq!(
            run(
                &[round_bf16(1.0), 0x8000],
                &[round_bf16(1.0 / 256.0), 0x8000]
            )
            .unwrap(),
            [round_bf16(1.0), 0x8000]
        );
        assert!(run(&[0x7f7f], &[0x7f7f]).is_err());
        assert!(run(&[0x7f80], &[0]).is_err());
        assert!(run(&[0], &[]).is_err());
    }
}
