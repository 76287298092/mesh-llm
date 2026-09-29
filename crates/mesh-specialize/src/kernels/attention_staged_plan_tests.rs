use super::*;

#[test]
fn workspace_and_grid_are_linear_and_capacity_strided() {
    let plan = Plan::new([1, 24, 4, 256, 8191, 131072]).unwrap();
    assert_eq!(plan.grids, [[8192, 24, 1], [24, 1, 1], [24, 4, 1]]);
    assert_eq!(BLOCKS, [[32, 1, 1], [32, 1, 1], [64, 1, 1]]);
    assert_eq!(plan.workspace_bytes, 72 * 1024 * 1024 + 24 * 8);
    assert_eq!(
        plan.offsets(),
        [0, 24 * 1024 * 1024, 48 * 1024 * 1024, 72 * 1024 * 1024]
    );
    let max = Plan::new([1, 24, 4, 256, 262143, 262144]).unwrap();
    assert_eq!(max.grids[0], [262144, 24, 1]);
}

#[test]
fn prefix_schedule_admits_running_max_plane_and_parallel_exponentials() {
    assert_eq!(
        CoefficientSchedule::parse(None).unwrap(),
        CoefficientSchedule::SerialV1
    );
    assert_eq!(
        CoefficientSchedule::parse(Some("prefix-parallel-v2")).unwrap(),
        CoefficientSchedule::PrefixParallelV2
    );
    assert!(CoefficientSchedule::parse(Some("online")).is_err());
    let plan = Plan::new_with_schedule(
        [1, 24, 4, 256, 8190, 8191],
        CoefficientSchedule::PrefixParallelV2,
    )
    .unwrap();
    let serial = Plan::new([1, 24, 4, 256, 8190, 8191]).unwrap();
    assert_eq!(
        plan.workspace_bytes,
        serial.workspace_bytes + plan.head_elements * 8
    );
    assert_eq!(
        plan.workspace_bytes,
        (4 * plan.head_elements + HEADS) * size_of::<f64>()
    );
    assert_eq!(
        plan.prefix_grids().unwrap(),
        [
            [8191, 24, 1],
            [24, 1, 1],
            [64, 24, 1],
            [24, 1, 1],
            [24, 4, 1]
        ]
    );
    assert_eq!(
        PREFIX_SCHEDULE_KERNELS,
        [
            KERNELS[0],
            PREFIX_KERNELS[0],
            PREFIX_KERNELS[1],
            PREFIX_KERNELS[2],
            KERNELS[2],
        ]
    );
    assert_eq!(
        PREFIX_BLOCKS,
        [[32, 1, 1], [128, 1, 1], [32, 1, 1], [64, 1, 1]]
    );
    let max_plane = plan.prefix_offset();
    assert!(plan.initialized(max_plane + 23 * plan.capacity + 8190));
    assert!(!plan.initialized(max_plane + 23 * plan.capacity + 8191));
    assert!(plan.initialized(plan.workspace_bytes / size_of::<f64>() - 1));
    assert!(!plan.initialized(plan.workspace_bytes / size_of::<f64>()));
    for length in [1, 106, 129, 512, 8191] {
        let plan = Plan::new_with_schedule(
            [1, 24, 4, 256, length - 1, 8191],
            CoefficientSchedule::PrefixParallelV2,
        )
        .unwrap();
        assert_eq!(plan.length, length);
        assert_eq!(
            plan.prefix_grids().unwrap()[2][0],
            u32::try_from(length.div_ceil(PREFIX_SCAN_TILE)).unwrap()
        );
    }
}

#[test]
fn parallel_exponent_coefficients_match_v1_bits_and_ordered_normalizers() {
    for length in [1, 106, 129, 512, 8191] {
        let scores = (0..length)
            .map(|key| match key % 13 {
                0 => -0.0,
                1 => 0.0,
                2 => -745.0,
                3 => -32.0,
                _ => -f64::from(u32::try_from((key * 7919) % 16_381).unwrap()) / 64.0,
            })
            .collect::<Vec<_>>();
        let mut old_maximum = f64::NEG_INFINITY;
        let mut old_normalizer = 0.0_f64;
        let mut old_alpha = Vec::with_capacity(length);
        let mut old_beta = Vec::with_capacity(length);
        let mut running_maxima = Vec::with_capacity(length);
        for &score in &scores {
            let next_maximum = if score > old_maximum {
                score
            } else {
                old_maximum
            };
            let alpha = if old_normalizer == 0.0 {
                0.0
            } else {
                crate::kernels::exponential::exp_nonpositive(old_maximum - next_maximum)
            };
            let beta = crate::kernels::exponential::exp_nonpositive(score - next_maximum);
            old_normalizer = old_normalizer * alpha + beta;
            old_maximum = next_maximum;
            running_maxima.push(next_maximum);
            old_alpha.push(alpha.to_bits());
            old_beta.push(beta.to_bits());
        }

        let new_alpha = (0..length)
            .map(|key| {
                let alpha = if key == 0 {
                    0.0
                } else {
                    crate::kernels::exponential::exp_nonpositive(
                        running_maxima[key - 1] - running_maxima[key],
                    )
                };
                alpha.to_bits()
            })
            .collect::<Vec<_>>();
        let new_beta = scores
            .iter()
            .zip(&running_maxima)
            .map(|(&score, &maximum)| {
                crate::kernels::exponential::exp_nonpositive(score - maximum).to_bits()
            })
            .collect::<Vec<_>>();
        let mut new_normalizer = 0.0_f64;
        let mut corrected_alpha = Vec::with_capacity(length);
        for (&alpha, &beta) in new_alpha.iter().zip(&new_beta) {
            let alpha = if new_normalizer == 0.0 {
                0.0
            } else {
                f64::from_bits(alpha)
            };
            let beta = f64::from_bits(beta);
            new_normalizer = new_normalizer * alpha + beta;
            corrected_alpha.push(alpha.to_bits());
        }

        assert_eq!(corrected_alpha, old_alpha, "alpha length {length}");
        assert_eq!(new_beta, old_beta, "beta length {length}");
        assert_eq!(
            new_normalizer.to_bits(),
            old_normalizer.to_bits(),
            "length {length}"
        );
    }
}

#[test]
fn rejects_other_rows_shapes_capacity_and_overflows() {
    for dimensions in [
        [5, 24, 4, 256, 0, 5],
        [1, 12, 4, 256, 0, 1],
        [1, 24, 8, 256, 0, 1],
        [1, 24, 4, 128, 0, 1],
        [1, 24, 4, 256, 1, 1],
        [1, 24, 4, 256, 0, 0],
        [1, 24, 4, 256, 0, 262145],
        [1, 24, 4, 256, usize::MAX, 131072],
    ] {
        assert!(Plan::new(dimensions).is_err());
    }
}

#[test]
fn initialized_prefix_is_separate_in_every_head_and_array() {
    let plan = Plan::new([1, 24, 4, 256, 1, 9]).unwrap();
    for array in 0..3 {
        for head in 0..24 {
            for key in 0..9 {
                assert_eq!(
                    plan.initialized(array * plan.head_elements + head * 9 + key),
                    key < 2
                );
            }
        }
    }
    assert!(plan.initialized(3 * plan.head_elements + 23));
    assert!(!plan.initialized(3 * plan.head_elements + 24));
}

#[test]
fn addresses_must_be_aligned_bounded_and_disjoint() {
    let plan = Plan::new([1, 24, 4, 256, 0, 1]).unwrap();
    let good = [0x10000, 0x20000, 0x30000, 0x40000, 0x50000, 0x60000];
    assert!(validate_addresses(plan, good).is_ok());
    for index in 0..6 {
        let mut bad = good;
        bad[index] += 1;
        assert!(validate_addresses(plan, bad).is_err());
    }
    let mut overlap = good;
    overlap[5] = good[1];
    assert!(validate_addresses(plan, overlap).is_err());
    let mut overflow = good;
    overflow[5] = u64::MAX - 7;
    assert!(validate_addresses(plan, overflow).is_err());
}

#[test]
fn source_keeps_exact_tree_serial_recurrences_and_capacity_planes() {
    let source = include_str!("../../kernels/nvptx/attention_staged_fp64.rs");
    let compact: String = source.chars().filter(|c| !c.is_whitespace()).collect();
    for required in [
        "warp_dot(local_tree(products))",
        "lane!=0",
        "for key in 0..length",
        "normalizer=add_rn(multiply_rn(normalizer,alpha),beta)",
        "accumulator=add_rn(multiply_rn(accumulator,alpha),multiply_rn(beta,value))",
        "3*plane+head as usize",
        "let plane=24*capacity as usize",
    ] {
        let required: String = required.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(compact.contains(&required), "missing {required}");
    }
    for forbidden in ["bar.sync", ".shared", "fma.", "ex2."] {
        assert!(!source.contains(forbidden));
    }
}

#[test]
fn sanitizer_defaults_disable_expensive_trial_work() {
    let options = TrialOptions::parse(None, None).unwrap();
    assert!(!options.long_cases && !options.timing);
    let full = TrialOptions::parse(Some("full"), Some("on")).unwrap();
    assert!(full.long_cases && full.timing);
    assert!(TrialOptions::parse(Some("all"), None).is_err());
    assert!(TrialOptions::parse(None, Some("yes")).is_err());
}

#[test]
fn trial_pasts_cover_requested_attention_lengths() {
    assert_eq!(SHORT_TRIAL_PASTS.map(|past| past + 1), [1, 2, 33, 106, 128]);
    assert_eq!(
        FULL_TRIAL_PASTS.map(|past| past + 1),
        [1, 2, 33, 106, 128, 512, 8191]
    );
}
