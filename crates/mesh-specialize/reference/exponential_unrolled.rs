//! Inputs and host operation-order checks for the isolated exponential experiment.
//! GPU outputs must match the unchanged device helper; host agreement is supplemental.
use core::f64::consts::LOG2_E;

/// Raw bits preserve both zeros and NaN signs/payloads through upload and reporting.
pub fn input_bits() -> Vec<u64> {
    let mut bits = Vec::new();
    // Exactly represented 1/1024 spacing over the requested dense interval.
    for step in 0..=131_072 {
        bits.push((-128.0 + f64::from(step) / 1024.0).to_bits());
    }
    // Exercise both sides of the cutoff and the gradual-underflow region.
    for step in 0..=16_384 {
        bits.push((-745.125 + f64::from(step) / 65_536.0).to_bits());
    }
    for x in [-745.0_f64, -744.0, -709.0, -708.5, -128.0, -1.0] {
        neighbors(&mut bits, x);
    }
    // Cast boundaries satisfy x*LOG2_E - 0.5 approximately equal to a negative integer.
    // Adjacent representable values cover rounding around the computed boundary.
    for n in 1..=1075 {
        neighbors(&mut bits, (0.5 - f64::from(n)) / LOG2_E);
    }
    bits.extend([
        0x0000_0000_0000_0000,
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x8000_0000_0000_0001,
        0x0010_0000_0000_0000,
        0x8010_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0000,
        0xfff8_0000_0000_0000,
        0x7ff8_1234_5678_9abc,
        0xfff8_1234_5678_9abc,
        0x7ff0_0000_0000_0001,
        0xfff0_0000_0000_0001,
        0x7ff7_ffff_ffff_ffff,
        0xfff7_ffff_ffff_ffff,
        1.0_f64.to_bits(),
        f64::MAX.to_bits(),
        (-f64::MAX).to_bits(),
    ]);
    bits
}

fn neighbors(output: &mut Vec<u64>, value: f64) {
    let bits = value.to_bits();
    for delta in -4_i64..=4 {
        output.push(bits.wrapping_add_signed(delta));
    }
}

#[cfg(test)]
#[path = "../kernels/nvptx/exponential_unrolled.rs"]
mod candidate;
#[cfg(test)]
use crate::kernels::exponential as control;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_host_result_bit_matches_original_helper() {
        for bits in input_bits() {
            let value = f64::from_bits(bits);
            assert_eq!(
                candidate::exp_nonpositive(value).to_bits(),
                control::exp_nonpositive(value).to_bits(),
                "input bits {bits:016x}"
            );
        }
    }

    #[test]
    fn coefficient_literals_match_all_original_binary64_divisions() {
        let source = include_str!("../kernels/nvptx/exponential_unrolled.rs");
        let encodings: Vec<u64> = source
            .lines()
            .filter_map(|line| {
                let hex = line.split("f64::from_bits(0x").nth(1)?.split(')').next()?;
                Some(u64::from_str_radix(&hex.replace('_', ""), 16).unwrap())
            })
            .collect();
        let divisors = [
            20_922_789_888_000.0_f64,
            1_307_674_368_000.0,
            87_178_291_200.0,
            6_227_020_800.0,
            479_001_600.0,
            39_916_800.0,
            3_628_800.0,
            362_880.0,
            40_320.0,
            5_040.0,
            720.0,
            120.0,
            24.0,
            6.0,
            2.0,
            1.0,
            1.0,
        ];
        assert_eq!(encodings, divisors.map(|d| (1.0 / d).to_bits()));
        assert_eq!(source.matches("p = add_rn(").count(), 17);
        assert!(source.contains("let mut p = 0.0_f64;"));
        assert!(!source.contains("for coefficient"));
        assert!(!source.contains(".mul_add("));
    }

    #[test]
    fn controls_remain_false_specializations_and_new_entry_retains_abi() {
        let control = include_str!("../kernels/nvptx/causal_attention.rs");
        let candidate = include_str!("../kernels/nvptx/attention_unrolled_fp64.rs");
        let graph = include_str!("../kernels/nvptx/graph_position.rs");
        assert!(control.contains("attention_body::<false>("));
        assert!(graph.contains("attention_body::<false>("));
        assert!(candidate.contains("attention_body::<true>("));
        let signature = candidate
            .split("fn causal_attention_unrolled_fp64(")
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
}
