use super::driver::{Buffer, Context, Module};
use crate::{
    causal_conv4_reference,
    entry_reference::{bf16_to_f32, round_bf16},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub(super) fn run(ctx: &Context, module: &Module<'_>) -> Result<Value> {
    let output = Buffer::new(ctx, 65536 * 4)?;
    let mut pointer = output.pointer();
    let mut args = [(&mut pointer as *mut u64).cast()];
    // SAFETY: A full BF16-domain grid writes one f32 per code to this exact-sized buffer.
    unsafe {
        module
            .function("silu_bf16_probe")?
            .launch([256, 1, 1], [256, 1, 1], 0, &mut args)?;
    }
    ctx.synchronize()?;
    let mut raw = vec![0; output.len()];
    output.download(&mut raw)?;
    let mut checked = 0;
    let mut differences = Vec::new();
    let mut max_scaled_error = 0.0_f64;
    for (index, bytes) in raw.as_chunks::<4>().0.iter().enumerate() {
        let input = bf16_to_f32(index as u16);
        if !input.is_finite() {
            continue;
        }
        let actual = f32::from_le_bytes(*bytes);
        let expected = causal_conv4_reference::silu(input);
        ensure!(
            actual.is_finite(),
            "nonfinite GPU SiLU for BF16 code {index}"
        );
        let error = (f64::from(actual) - f64::from(expected)).abs()
            / f64::from(expected).abs().max(f64::from(f32::MIN_POSITIVE));
        max_scaled_error = max_scaled_error.max(error);
        if round_bf16(actual) != round_bf16(expected) {
            differences.push(index);
        }
        checked += 1;
    }
    Ok(
        json!({"all_passed":differences.is_empty()&&max_scaled_error<=2e-7,"finite_inputs":checked,
        "bf16_differences":differences.len(),"first_mismatching_codes":differences.iter().take(16).collect::<Vec<_>>(),"maximum_scaled_fp32_error":max_scaled_error}),
    )
}
