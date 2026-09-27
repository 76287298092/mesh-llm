//! Stateful causal convolution qualification with device-resident input and history.
use super::driver::{Buffer, Context, Function, Module};
use crate::{
    causal_conv4_reference as reference,
    entry_reference::{bf16_to_f32, round_bf16},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(super) fn check(
    context: &Context,
    module: &Module<'_>,
    input: &Buffer<'_>,
    input_words: &[u16],
    weights: &[u8],
    shape: [usize; 2],
) -> Result<Value> {
    let history = vec![0; shape[1] * 3];
    check_history(
        context,
        module,
        input,
        input_words,
        weights,
        &history,
        shape,
    )
}

fn check_history(
    context: &Context,
    module: &Module<'_>,
    input: &Buffer<'_>,
    input_words: &[u16],
    weights: &[u8],
    history: &[u16],
    shape: [usize; 2],
) -> Result<Value> {
    let [rows, channels] = shape;
    ensure!(
        weights.len().is_multiple_of(2),
        "odd convolution weight byte count"
    );
    let weight_words: Vec<_> = weights
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    reference::run(input_words, &weight_words, history, rows, channels)?;
    let weights = upload(context, weights)?;
    let mut partitions = vec![vec![rows]];
    if rows > 1 {
        partitions.push(vec![1, rows - 1]);
    }
    if rows > 3 {
        partitions.push(vec![2, 1, rows - 3]);
    }
    if rows > 1 {
        partitions.push(vec![1; rows]);
    }
    let trial = Trial {
        context,
        function: module.function("causal_conv4_bf16")?,
        input,
        input_words,
        weights: &weights,
        weight_words: &weight_words,
        history,
        channels,
    };
    let whole = trial.sequence(&partitions[0])?;
    let mut reports = vec![whole.report];
    for partition in &partitions[1..] {
        let chunked = trial.sequence(partition)?;
        ensure!(
            chunked.output == whole.output,
            "convolution outputs depend on chunk partition"
        );
        ensure!(
            chunked.history == whole.history,
            "convolution final history depends on chunk partition"
        );
        reports.push(chunked.report);
    }
    Ok(
        json!({"all_passed":true,"shape_tc":shape,"elements":rows*channels,
        "partitions":reports,"chunk_outputs_exact":true,"chunk_final_history_exact":true,
        "history_words":3*channels,"history_order":"oldest-to-newest, time-major, raw pre-convolution input",
        "profile":"FP32 ordered multiply/add, BF16 convolution rounding, FP32 SiLU, BF16 output",
        "reference_input":"downloaded independently-qualified projection BF16; GPU input is never replaced",
        "state_transport":"out-of-place device history with ping-pong buffers; no host history replacement"}),
    )
}

struct Sequence {
    output: Vec<u16>,
    history: Vec<u16>,
    report: Value,
}

struct Trial<'a, 'context, 'module> {
    context: &'context Context,
    function: Function<'module, 'context>,
    input: &'a Buffer<'context>,
    input_words: &'a [u16],
    weights: &'a Buffer<'context>,
    weight_words: &'a [u16],
    history: &'a [u16],
    channels: usize,
}

impl Trial<'_, '_, '_> {
    fn sequence(&self, partition: &[usize]) -> Result<Sequence> {
        let channels = self.channels;
        ensure!(partition.iter().all(|&n| n > 0), "empty convolution chunk");
        ensure!(
            partition.iter().sum::<usize>() * channels == self.input_words.len(),
            "invalid convolution partition"
        );
        let history_bytes: Vec<_> = self.history.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut state = upload(self.context, &history_bytes)?;
        let mut next = upload(self.context, &vec![0xa5; history_bytes.len()])?;
        let mut expected_state = self.history.to_vec();
        let mut output = Vec::with_capacity(self.input_words.len());
        let mut reports = Vec::new();
        let mut offset = 0;
        for &rows in partition {
            let count = rows * channels;
            let expected = reference::run(
                &self.input_words[offset..offset + count],
                self.weight_words,
                &expected_state,
                rows,
                channels,
            )?;
            let actual = self.launch(
                self.input.pointer() + (offset * 2) as u64,
                &state,
                &next,
                rows,
            )?;
            let report = compare(&actual.0, &actual.1, &actual.2, &expected)?;
            ensure!(
                report["passed"] == true,
                "causal convolution numerical check failed: {report}"
            );
            let actual_history = words(&next, 3 * channels)?;
            ensure!(
                actual_history == expected.next_history,
                "causal convolution history mismatch"
            );
            expected_state = expected.next_history;
            output.extend(actual.0);
            reports.push(report);
            std::mem::swap(&mut state, &mut next);
            offset += count;
        }
        Ok(Sequence {
            output,
            history: words(&state, 3 * channels)?,
            report: json!({"chunks":partition,"checks":reports,"history_exact_at_every_boundary":true}),
        })
    }

    fn launch(
        &self,
        input: u64,
        state: &Buffer<'_>,
        next: &Buffer<'_>,
        rows: usize,
    ) -> Result<(Vec<u16>, Vec<f32>, Vec<f32>)> {
        let count = rows * self.channels;
        let output = upload(self.context, &vec![0xa5; count * 2])?;
        let conv = upload(self.context, &vec![0xff; count * 4])?;
        let silu = upload(self.context, &vec![0xff; count * 4])?;
        let mut pointers = [
            input,
            self.weights.pointer(),
            state.pointer(),
            next.pointer(),
            output.pointer(),
            conv.pointer(),
            silu.pointer(),
        ];
        let mut dimensions = [u32::try_from(rows)?, u32::try_from(self.channels)?];
        let mut args: Vec<*mut c_void> = pointers
            .iter_mut()
            .map(|p| (p as *mut u64).cast())
            .collect();
        args.extend(dimensions.iter_mut().map(|p| (p as *mut u32).cast()));
        // SAFETY: Seven disjoint, aligned allocations and two u32 dimensions match
        // causal_conv4_bf16. The retained input allocation covers this validated
        // offset/chunk, weights cover channels*4, states cover channels*3, and all
        // output extents cover rows*channels. Synchronization precedes reuse/drop.
        unsafe {
            self.function.launch(
                [u32::try_from(count.div_ceil(256))?, 1, 1],
                [256, 1, 1],
                0,
                &mut args,
            )?;
        }
        self.context.synchronize()?;
        Ok((
            words(&output, count)?,
            floats(&conv, count)?,
            floats(&silu, count)?,
        ))
    }
}

fn compare(
    output: &[u16],
    conv: &[f32],
    silu: &[f32],
    expected: &reference::ConvReference,
) -> Result<Value> {
    ensure!(
        output.len() == conv.len()
            && conv.len() == silu.len()
            && silu.len() == expected.output.len(),
        "convolution result extent mismatch"
    );
    let mut conv_error = 0.0_f32;
    let mut silu_error = 0.0_f32;
    let mut full_silu_error = 0.0_f32;
    let mut numerical = 0;
    let mut rounding = 0;
    let mut bf16_differences = 0;
    for i in 0..output.len() {
        let rounded_conv = bf16_to_f32(round_bf16(conv[i]));
        ensure!(
            conv[i].is_finite()
                && rounded_conv.is_finite()
                && silu[i].is_finite()
                && bf16_to_f32(output[i]).is_finite(),
            "nonfinite causal convolution output"
        );
        let ce = (conv[i] - expected.convolution[i]).abs();
        let activation_reference = reference::silu(rounded_conv);
        let se = (silu[i] - activation_reference).abs();
        conv_error = conv_error.max(ce);
        silu_error = silu_error.max(se);
        full_silu_error = full_silu_error.max((silu[i] - expected.activated[i]).abs());
        numerical += usize::from(f64::from(ce) > 1e-6 + 2e-6 * expected.absolute_sums[i]);
        numerical += usize::from(se > 2e-6 + 2e-6 * activation_reference.abs());
        rounding += usize::from(output[i] != round_bf16(silu[i]));
        bf16_differences += usize::from(output[i] != expected.output[i]);
    }
    Ok(
        json!({"elements":output.len(),"passed":numerical==0 && rounding==0,
        "convolution_max_abs_error":conv_error,"silu_max_abs_error":silu_error,
        "full_scalar_activation_max_abs_error":full_silu_error,
        "numerical_mismatches":numerical,"rounding_mismatches":rounding,
        "bf16_reference_differences":bf16_differences,
        "convolution_tolerance":"1e-6 + 2e-6 * sum(abs(products))",
        "silu_tolerance":"2e-6 + 2e-6 * abs(f64 SiLU of rounded GPU convolution)"}),
    )
}

pub(super) fn fixtures(context: &Context, module: &Module<'_>) -> Result<Vec<Value>> {
    let channels = 13;
    let input: Vec<_> = (0..7 * channels)
        .map(|i| round_bf16(((i * 11 % 31) as f32 - 15.0) / 8.0))
        .collect();
    let history: Vec<_> = (0..3 * channels)
        .map(|i| round_bf16(((i * 5 % 17) as f32 - 8.0) / 4.0))
        .collect();
    let weights: Vec<_> = (0..4 * channels)
        .flat_map(|i| round_bf16(((i * 3 % 11) as f32 - 5.0) / 8.0).to_le_bytes())
        .collect();
    let bytes: Vec<_> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let device = upload(context, &bytes)?;
    let signed = check_history(
        context,
        module,
        &device,
        &input,
        &weights,
        &history,
        [7, channels],
    )?;
    let input = [-1e30, -100.0, -88.0, -20.0, -0.0, 0.0, 20.0, 100.0, 1e30].map(round_bf16);
    let weights: Vec<_> = (0..input.len())
        .flat_map(|_| [0_u16, 0, 0, 0x3f80])
        .flat_map(u16::to_le_bytes)
        .collect();
    let bytes: Vec<_> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let device = upload(context, &bytes)?;
    let extremes = check(context, module, &device, &input, &weights, [1, input.len()])?;
    Ok(vec![signed, extremes])
}

fn upload<'a>(context: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}
fn words(buffer: &Buffer<'_>, count: usize) -> Result<Vec<u16>> {
    let mut bytes = vec![0; count * 2];
    buffer.download(&mut bytes)?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect())
}
fn floats(buffer: &Buffer<'_>, count: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0; count * 4];
    buffer.download(&mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn comparison_rejects_wrong_silu_rounding_nonfinite_and_convolution() {
        let expected = reference::run(&[0x3f80], &[0, 0, 0, 0x3f80], &[0, 0, 0], 1, 1).unwrap();
        let activation = expected.activated[0];
        assert_eq!(
            compare(&[round_bf16(activation)], &[1.0], &[activation], &expected).unwrap()["passed"],
            true
        );
        assert_eq!(
            compare(&[round_bf16(activation)], &[1.0], &[0.5], &expected).unwrap()["passed"],
            false
        );
        assert_eq!(
            compare(&[0], &[1.0], &[activation], &expected).unwrap()["passed"],
            false
        );
        assert_eq!(
            compare(&[round_bf16(activation)], &[2.0], &[activation], &expected).unwrap()["passed"],
            false
        );
        assert!(compare(&[0x7fc0], &[f32::NAN], &[activation], &expected).is_err());
    }
}
