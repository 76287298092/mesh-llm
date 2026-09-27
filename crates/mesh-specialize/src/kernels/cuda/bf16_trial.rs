//! Independent cancellation and tile-tail qualification for the BF16 projection.
use super::{
    driver::{Buffer, Context, Module},
    projections,
};
use crate::{
    entry_reference::{bf16_to_f32, round_bf16},
    projection_reference,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub(super) fn run(ctx: &Context, module: &Module<'_>) -> Result<Value> {
    let mut cases = Vec::new();
    for (rows, channels, width, cancellation) in
        [(1, 1, 1, false), (17, 9, 513, false), (2, 9, 5120, true)]
    {
        let (input, weights) = fixture(rows, channels, width, cancellation);
        let expected = projection_reference::linear_bf16(&input, &weights, rows, width)?;
        let input = upload(ctx, &input)?;
        let weights = upload(ctx, &weights)?;
        let output = projections::run_linear(
            ctx,
            &module.function("bf16_linear")?,
            &[&input, &weights],
            [rows, channels, width],
        )?;
        let (actual, unrounded) = output.read(rows * channels)?;
        ensure!(
            unrounded.iter().all(|v| v.is_finite()),
            "nonfinite BF16 fixture projection"
        );
        let differences = actual
            .iter()
            .zip(&expected.normalized)
            .filter(|(a, b)| a != b)
            .count();
        let max_error = unrounded
            .iter()
            .zip(&expected.unrounded)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
            .fold(0.0_f64, f64::max);
        let rounding = actual
            .iter()
            .zip(&unrounded)
            .filter(|(a, b)| **a != round_bf16(**b))
            .count();
        cases.push(json!({"shape":[rows,channels,width],"cancellation":cancellation,"bf16_differences":differences,"rounding_mismatches":rounding,"max_fp32_abs_error":max_error,"all_passed":differences==0&&rounding==0}));
    }
    Ok(
        json!({"all_passed":cases.iter().all(|c|c["all_passed"]==true),"cases":cases,"resources":module.function("bf16_linear")?.resources()?}),
    )
}

fn fixture(rows: usize, channels: usize, width: usize, cancellation: bool) -> (Vec<u16>, Vec<u16>) {
    if !cancellation {
        let input = (0..rows * width)
            .map(|i| round_bf16(((i * 13 % 31) as f32 - 15.0) / 8.0))
            .collect();
        let weights = (0..channels * width)
            .map(|i| round_bf16(((i * 11 % 43) as f32 - 21.0) / 16.0))
            .collect();
        return (input, weights);
    }
    let mut input = vec![0x3f80; rows * width];
    for word in &mut input[width..] {
        *word = 0xbf80;
    }
    let mut weights = vec![0; channels * width];
    for (channel, row) in weights.chunks_exact_mut(width).enumerate() {
        row[0] = round_bf16(16_777_216.0);
        row[1] = round_bf16(1.0);
        row[2] = round_bf16(-16_777_216.0);
        row[3] = round_bf16((channel as f32 - 4.0) / 256.0);
        row[width - 1] = round_bf16(1.0 / 1_048_576.0);
    }
    (input, weights)
}

fn upload<'a>(ctx: &'a Context, words: &[u16]) -> Result<Buffer<'a>> {
    ensure!(
        words.iter().all(|&w| bf16_to_f32(w).is_finite()),
        "nonfinite BF16 fixture input"
    );
    let bytes = words
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let result = Buffer::new(ctx, bytes.len())?;
    result.upload(&bytes)?;
    Ok(result)
}
