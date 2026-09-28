//! Exact-order schedule checks, separate from the existing independent FP64 oracle.
//! These checks prove a host operation-order mapping, not CUDA code generation.
use crate::entry_reference::{bf16_to_f32, round_bf16};

pub const QUERY_ELEMENTS: usize = 24 * 256;
pub const KV_ROW: usize = 4 * 256;

pub struct Fixture {
    pub past: usize,
    pub capacity: usize,
    pub q: Vec<u16>,
    pub k: Vec<u16>,
    pub v: Vec<u16>,
}

#[derive(Clone, Copy, Debug)]
pub enum Pattern {
    Hashed,
    Uniform,
    Cancellation,
    SignedZero,
    WideExponent,
}

fn hashed(index: usize, seed: u32) -> u32 {
    let mut x = (index as u32).wrapping_add(seed);
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb_352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

fn value(index: usize, seed: u32) -> u16 {
    round_bf16(((hashed(index, seed) % 1025) as f32 - 512.0) / 512.0)
}

fn wide(index: usize, seed: u32) -> u16 {
    let x = hashed(index, seed);
    // Finite BF16 normals over exponents -120..120, nontrivial mantissas/signs.
    ((x & 1) as u16) << 15 | (((x >> 1) % 241 + 7) as u16) << 7 | ((x >> 10) as u16 & 127)
}

pub fn fixture(past: usize, pattern: Pattern) -> Fixture {
    let capacity = past + 20;
    let q = (0..QUERY_ELEMENTS)
        .map(|i| match pattern {
            Pattern::Uniform | Pattern::Cancellation => 0,
            Pattern::SignedZero => ((i & 1) as u16) << 15,
            Pattern::WideExponent => wide(i, 17),
            Pattern::Hashed => value(i, 17),
        })
        .collect();
    let mut k = vec![0x7fc1; capacity * KV_ROW];
    let mut v = vec![0xffc2; capacity * KV_ROW];
    for i in 0..(past + 1) * KV_ROW {
        k[i] = match pattern {
            Pattern::WideExponent => wide(i, 311),
            Pattern::SignedZero => (((i / 7) & 1) as u16) << 15,
            _ => value(i, 311),
        };
        v[i] = match pattern {
            Pattern::SignedZero => (((i / 3) & 1) as u16) << 15,
            Pattern::Cancellation => {
                let sign = if (i / KV_ROW).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
                round_bf16(sign * (1.0 + (i % 256) as f32 / 256.0))
            }
            _ => value(i, 971),
        };
    }
    Fixture {
        past,
        capacity,
        q,
        k,
        v,
    }
}

/// Model the control's 256-slot shared tree without reusing the warp mapping.
pub fn control_dot(q: &[u16; 256], k: &[u16; 256]) -> f64 {
    let mut p = std::array::from_fn::<_, 256, _>(|i| {
        f64::from(bf16_to_f32(q[i])) * f64::from(bf16_to_f32(k[i]))
    });
    let mut stride = 128;
    while stride != 0 {
        for lane in 0..stride {
            p[lane] += p[lane + stride];
        }
        stride /= 2;
    }
    p[0]
}

/// Model eight Q/K registers per lane and simultaneous full-warp DOWN shuffles.
pub fn warp_dot(q: &[u16; 256], k: &[u16; 256]) -> f64 {
    let mut lanes = std::array::from_fn::<_, 32, _>(|lane| {
        let p = std::array::from_fn::<_, 8, _>(|c| {
            let i = lane + 32 * c;
            f64::from(bf16_to_f32(q[i])) * f64::from(bf16_to_f32(k[i]))
        });
        ((p[0] + p[4]) + (p[2] + p[6])) + ((p[1] + p[5]) + (p[3] + p[7]))
    });
    for offset in [16, 8, 4, 2, 1] {
        let previous = lanes;
        for lane in 0..32 {
            let other = if lane + offset < 32 {
                lane + offset
            } else {
                lane
            };
            lanes[lane] = previous[lane] + previous[other];
        }
    }
    lanes[0]
}

#[cfg(test)]
use crate::kernels::exponential as schedule_exp;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_exponents_cancellation_and_signed_zero_preserve_every_dot_bit() {
        for seed in 0..128 {
            let q = std::array::from_fn(|i| wide(i, seed));
            let k = std::array::from_fn(|i| wide(i, seed + 311));
            assert_eq!(control_dot(&q, &k).to_bits(), warp_dot(&q, &k).to_bits());
        }
        let q = [0x3f80; 256];
        let k = std::array::from_fn(|i| match i % 4 {
            0 => 0x7b00,
            1 => 0xfb00,
            2 => 0x8000,
            _ => 0x0000,
        });
        assert_eq!(control_dot(&q, &k).to_bits(), warp_dot(&q, &k).to_bits());
    }

    fn scheduled(f: &Fixture, dot: fn(&[u16; 256], &[u16; 256]) -> f64) -> Vec<u32> {
        let mut output = Vec::new();
        for head in 0..24 {
            let q = f.q[head * 256..(head + 1) * 256].try_into().unwrap();
            let mut maximum = f64::NEG_INFINITY;
            let mut normalizer = 0.0_f64;
            let mut accumulators = [0.0_f64; 256];
            for token in 0..=f.past {
                let start = token * KV_ROW + (head / 6) * 256;
                let k = f.k[start..start + 256].try_into().unwrap();
                let score = dot(q, k) * 0.0625;
                let next = if score > maximum { score } else { maximum };
                let alpha = if normalizer == 0.0 {
                    0.0
                } else {
                    schedule_exp::exp_nonpositive(maximum - next)
                };
                let beta = schedule_exp::exp_nonpositive(score - next);
                normalizer = normalizer * alpha + beta;
                maximum = next;
                for (c, a) in accumulators.iter_mut().enumerate() {
                    *a = *a * alpha + beta * f64::from(bf16_to_f32(f.v[start + c]));
                }
            }
            output.extend(accumulators.map(|a| ((a / normalizer) as f32).to_bits()));
        }
        output
    }

    #[test]
    fn full_online_recurrence_retains_bits_under_dot_schedule_change() {
        for pattern in [
            Pattern::Hashed,
            Pattern::Uniform,
            Pattern::Cancellation,
            Pattern::SignedZero,
            Pattern::WideExponent,
        ] {
            let f = fixture(32, pattern);
            assert_eq!(scheduled(&f, control_dot), scheduled(&f, warp_dot));
        }
    }

    #[test]
    fn shards_cover_each_output_once_and_keep_control_abi() {
        let mut owners = vec![0_u32; QUERY_ELEMENTS];
        for block in 0..96 {
            for lane in 0..32 {
                for c in 0..2 {
                    let index = (block / 4) * 256 + (block % 4) * 64 + lane + 32 * c;
                    owners[index] += 1;
                }
            }
        }
        assert!(owners.iter().all(|&count| count == 1));
        let source = include_str!("../kernels/nvptx/attention_warp_fp64.rs");
        let signature = source
            .split("fn causal_attention_warp_fp64(")
            .nth(1)
            .unwrap()
            .split(") {")
            .next()
            .unwrap();
        let compact: String = signature.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(
            compact,
            concat!(
                "q:*constu16,cache_k:*constu16,cache_v:*constu16,output:*mutu16,unrounded:*mutf32,",
                "rows:u32,query_heads:u32,kv_heads:u32,width:u32,past:u32,capacity:u32,scale:f32,"
            )
        );
    }

    #[test]
    fn device_source_keeps_fixed_geometry_and_control_operations() {
        let source = include_str!("../kernels/nvptx/attention_warp_fp64.rs");
        for marker in [
            "rows != 1",
            "query_heads != 24",
            "kv_heads != 4",
            "width != 256",
            "block >= 96",
            "threads != 32",
            "for token in 0..=past",
            "lane + 32 * c",
            "(block % 4) * 64 + lane",
            "shfl.sync.down.b32",
            "shfl.sync.idx.b32",
            "super::exponential::exp_nonpositive",
            "multiply_rn",
            "add_rn",
            "divide_rn",
        ] {
            assert!(source.contains(marker), "missing {marker}");
        }
        for forbidden in ["bar.sync", ".shared", "fma.", "exp2", "ex2."] {
            assert!(!source.contains(forbidden), "unexpected {forbidden}");
        }
    }
}
