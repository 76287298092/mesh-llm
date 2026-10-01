//! Mathematical FP64 activation/product oracle, not Ninfer expf equivalence.
//! Only the final product rounds to BF16. No FP32 or BF16 activation boundary.

#[derive(Clone, Copy, Debug)]
pub enum Operation {
    AttentionGate,
    SiluMul,
}

#[derive(Clone, Copy, Debug)]
pub struct Input {
    pub gate: u16,
    pub factor: u16,
}

pub fn decode(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}

pub const fn finite(bits: u16) -> bool {
    bits & 0x7f80 != 0x7f80
}

pub fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

impl Operation {
    pub const fn entry(self) -> &'static str {
        match self {
            Self::AttentionGate => "native_mtp_attention_gate",
            Self::SiluMul => "native_mtp_silu_mul",
        }
    }

    pub fn activation(self, gate: u16) -> f64 {
        let value = decode(gate);
        match self {
            Self::AttentionGate => sigmoid(value),
            Self::SiluMul => value * sigmoid(value),
        }
    }

    pub fn expected(self, input: Input) -> u16 {
        round(self.activation(input.gate) * decode(input.factor))
    }
}

/// Direct FP64 to BF16 RNE, including subnormals and signed zero.
/// Neighbor search avoids an intermediate FP32 double-rounding boundary.
pub fn round(value: f64) -> u16 {
    let sign = if value.is_sign_negative() { 0x8000 } else { 0 };
    let magnitude = value.abs();
    if magnitude.is_nan() {
        return sign | 0x7fc0;
    }
    if magnitude >= decode(0x7f7f) + 2.0_f64.powi(119) {
        return sign | 0x7f80;
    }
    let mut lower = 0_u16;
    let mut upper = 0x7f7f_u16;
    while lower < upper {
        let middle = lower + (upper - lower).div_ceil(2);
        if decode(middle) <= magnitude {
            lower = middle;
        } else {
            upper = middle - 1;
        }
    }
    if lower == 0x7f7f {
        return sign | lower;
    }
    let next = lower + 1;
    let midpoint = (decode(lower) + decode(next)) * 0.5;
    let rounded =
        if magnitude > midpoint || (magnitude.total_cmp(&midpoint).is_eq() && lower & 1 != 0) {
            next
        } else {
            lower
        };
    sign | rounded
}

/// Finite signed zeros, midpoint-sensitive factors, moderate/saturated gates,
/// and a bounded raw-BF16 sweep. Factors avoid BF16 product overflow.
pub fn inputs() -> Vec<Input> {
    let factors = [0x0000, 0x8000, 0x3f81, 0xbf41, 0x42c8, 0x3eab];
    let special = [
        0x0000, 0x8000, 0x3c00, 0xbc00, 0x3f80, 0xbf80, 0x4000, 0xc000, 0x4180, 0xc180, 0x42b4,
        0xc2b4, 0x42c8, 0xc2c8, 0x60ad, 0xe0ad,
    ];
    let gates = special.into_iter().chain(
        (0x3b00_u16..=0x42c8)
            .step_by(7)
            .flat_map(|gate| [gate, gate | 0x8000]),
    );
    gates
        .flat_map(|gate| factors.map(|factor| Input { gate, factor }))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_ties_to_even_without_fp32_double_rounding() {
        let midpoint = (decode(0x3f80) + decode(0x3f81)) * 0.5;
        let above = f64::from_bits(midpoint.to_bits() + 1);
        assert_eq!(round(midpoint), 0x3f80);
        assert_eq!(round(above), 0x3f81);
        assert_eq!(round((decode(0x3f81) + decode(0x3f82)) * 0.5), 0x3f82);
        assert_eq!(round(decode(1) * 0.5), 0);
        assert_eq!(round(decode(3) * 0.5), 2);
    }

    #[test]
    fn preserves_signed_zero_when_gate_or_factor_is_zero() {
        assert_eq!(
            Operation::AttentionGate.expected(Input {
                gate: 0,
                factor: 0x8000
            }),
            0x8000
        );
        assert_eq!(
            Operation::SiluMul.expected(Input {
                gate: 0x8000,
                factor: 0x3f80
            }),
            0x8000
        );
        assert_eq!(
            Operation::SiluMul.expected(Input {
                gate: 0x8000,
                factor: 0xbf80
            }),
            0
        );
    }

    #[test]
    fn corpus_distinguishes_premature_activation_rounding_for_both_operations() {
        let cases = inputs();
        for operation in [Operation::AttentionGate, Operation::SiluMul] {
            assert!(cases.iter().copied().any(|input| {
                operation.expected(input)
                    != round(decode(round(operation.activation(input.gate))) * decode(input.factor))
            }));
            assert!(
                cases
                    .iter()
                    .copied()
                    .all(|input| finite(operation.expected(input)))
            );
        }
        assert!(cases.len() < 4096);
    }
}
