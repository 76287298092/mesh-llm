//! Deterministic greedy selection at the model's BF16 logit boundary.
use anyhow::{Result, ensure};
pub fn greedy(logits: &[u16]) -> Result<u32> {
    ensure!(
        !logits.is_empty() && logits.len() <= 262144,
        "invalid logit extent"
    );
    let mut selected = 0;
    let mut best = f32::NEG_INFINITY;
    for (index, &bits) in logits.iter().enumerate() {
        let value = f32::from_bits(u32::from(bits) << 16);
        ensure!(value.is_finite(), "nonfinite logit at {index}");
        if value > best {
            best = value;
            selected = index;
        }
    }
    Ok(u32::try_from(selected)?)
}
#[cfg(test)]
mod tests {
    use super::greedy;
    #[test]
    fn chooses_first_tie_and_handles_negative_logits() {
        assert_eq!(greedy(&[0xbf80, 0xc000, 0xbf00]).unwrap(), 2);
        assert_eq!(greedy(&[0x3f80, 0x4000, 0x4000]).unwrap(), 1);
        assert_eq!(greedy(&[0x8000, 0]).unwrap(), 0);
        assert!(greedy(&[]).is_err());
        assert!(greedy(&[0x7fc0]).is_err());
    }
}
