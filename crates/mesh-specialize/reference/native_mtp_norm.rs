//! Independent Rust-schedule oracle, not a Ninfer BF16-pair/rsqrtf oracle.

use anyhow::{Result, ensure};

#[derive(Clone, Copy)]
pub enum Gain {
    Offset,
    Plain,
}

pub struct Request<'a> {
    pub input: &'a [u16],
    pub weight: &'a [u16],
    pub epsilon: f32,
}

pub struct Reference {
    pub scheduled: Vec<u16>,
    pub ideal: Vec<f64>,
}

pub fn decode(word: u16) -> f32 {
    f32::from_bits(u32::from(word) << 16)
}

pub fn encode(value: f32) -> u16 {
    if value.is_nan() {
        return 0x7fc0;
    }
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    let bytes = rounded.to_le_bytes();
    u16::from_le_bytes([bytes[2], bytes[3]])
}

fn add_rn(left: f32, right: f32) -> f32 {
    f32::from_bits((left + right).to_bits())
}

fn mul_rn(left: f32, right: f32) -> f32 {
    f32::from_bits((left * right).to_bits())
}

fn div_rn(left: f32, right: f32) -> f32 {
    f32::from_bits((left / right).to_bits())
}

fn scheduled_sum(row: &[u16]) -> f32 {
    let mut partials = [0.0_f32; 256];
    for (thread, partial) in partials.iter_mut().enumerate() {
        for &word in row.iter().skip(thread).step_by(256) {
            let value = decode(word);
            *partial = add_rn(*partial, mul_rn(value, value));
        }
    }
    let mut stride = 128;
    while stride > 0 {
        let (left, right) = partials.split_at_mut(stride);
        for (destination, &source) in left.iter_mut().zip(right.iter()) {
            *destination = add_rn(*destination, source);
        }
        stride /= 2;
    }
    partials[0]
}

pub fn evaluate(request: &Request<'_>, gain: Gain) -> Result<Reference> {
    let width = request.weight.len();
    ensure!(matches!(width, 256 | 5120), "unsupported native norm width");
    ensure!(
        !request.input.is_empty() && request.input.len().is_multiple_of(width)
            && request.input.len() <= 30_720,
        "native norm reference input extent outside trial bound"
    );
    ensure!(request.epsilon.is_finite() && request.epsilon > 0.0, "invalid epsilon");
    ensure!(
        request.input.iter().chain(request.weight).all(|&word| decode(word).is_finite()),
        "nonfinite native norm reference input or weight"
    );
    let width_value = f32::from(u16::try_from(width)?);
    let mut result = Reference {
        scheduled: Vec::with_capacity(request.input.len()),
        ideal: Vec::with_capacity(request.input.len()),
    };
    for row in request.input.chunks_exact(width) {
        let mean = div_rn(scheduled_sum(row), width_value);
        let denominator = add_rn(mean, request.epsilon);
        let root = f32::from_bits(denominator.sqrt().to_bits());
        let inverse = div_rn(1.0, root);
        let ideal_sum: f64 = row.iter().map(|&word| f64::from(decode(word)).powi(2)).sum();
        let ideal_inverse = 1.0 / (ideal_sum / f64::from(width_value)
            + f64::from(request.epsilon)).sqrt();
        for (&word, &weight_word) in row.iter().zip(request.weight) {
            let value = decode(word);
            let weight = decode(weight_word);
            let (scheduled_gain, ideal_gain) = match gain {
                Gain::Offset => (add_rn(1.0, weight), 1.0 + f64::from(weight)),
                Gain::Plain => (weight, f64::from(weight)),
            };
            let normalized = mul_rn(value, inverse);
            result.scheduled.push(encode(mul_rn(normalized, scheduled_gain)));
            result.ideal.push(f64::from(value) * ideal_inverse * ideal_gain);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_gain_when_variance_and_epsilon_sum_to_four() {
        let mut input = vec![0x3f80; 256];
        input[1] = 0xbf80;
        let mut weight = vec![0; 256];
        weight[1] = 0xbf80;
        weight[2] = 0x3f80;
        let request = Request { input: &input, weight: &weight, epsilon: 3.0 };

        let result = evaluate(&request, Gain::Offset).unwrap();

        assert_eq!(&result.scheduled[..4], &[0x3f00, 0x8000, 0x3f80, 0x3f00]);
        assert_eq!(&result.ideal[..4], &[0.5, -0.0, 1.0, 0.5]);
    }

    #[test]
    fn scalar_strided_reduction_when_rounding_discards_small_partials() {
        let mut row = vec![0x3f80; 5120];
        row[0] = 0x4580;

        let sum = scheduled_sum(&row);

        assert_eq!(sum.to_bits(), 16_782_316.0_f32.to_bits());
    }

    #[test]
    fn plain_gamma_mutation_when_weight_is_zero() {
        let input = vec![0x3f80; 256];
        let weight = vec![0; 256];
        let request = Request { input: &input, weight: &weight, epsilon: 3.0 };

        let offset = evaluate(&request, Gain::Offset).unwrap();
        let plain = evaluate(&request, Gain::Plain).unwrap();

        assert!(offset.scheduled.iter().all(|&word| word == 0x3f00));
        assert!(plain.scheduled.iter().all(|&word| word == 0));
    }

    #[test]
    fn bf16_rounding_when_halfway_between_even_and_odd_words() {
        let values = [f32::from_bits(0x3f808000), f32::from_bits(0x3f818000)];

        let words = values.map(encode);

        assert_eq!(words, [0x3f80, 0x3f82]);
    }
}
