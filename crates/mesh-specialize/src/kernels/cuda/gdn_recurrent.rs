//! Resident recurrent matrix updates, independently checked at each chunk boundary.
use super::driver::{Buffer, Context, Function, Module};
use crate::{entry_reference::round_bf16, gdn_recurrent_reference as reference};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(super) struct Input<'a, 'ctx> {
    pub(super) q: &'a Buffer<'ctx>,
    pub(super) k: &'a Buffer<'ctx>,
    pub(super) qkv: &'a Buffer<'ctx>,
    pub(super) beta: &'a Buffer<'ctx>,
    pub(super) decay: &'a Buffer<'ctx>,
    pub(super) host: reference::Input<'a>,
}

pub(super) fn check<'a>(
    context: &'a Context,
    module: &Module<'a>,
    input: Input<'_, 'a>,
    shape: &reference::Shape,
) -> Result<Value> {
    check_state(
        context,
        module,
        input,
        shape,
        &vec![0.0; shape.value_heads * shape.width * shape.width],
    )
}

fn check_state<'a>(
    context: &'a Context,
    module: &Module<'a>,
    input: Input<'_, 'a>,
    shape: &reference::Shape,
    initial: &[f32],
) -> Result<Value> {
    // Validate all dimensions, exact extents and domains before any device launch.
    let wide = reference::run(&input.host, initial, shape, reference::Reduction::WideF64)?;
    let mut partitions = vec![vec![shape.rows]];
    if shape.rows > 1 {
        partitions.push(vec![1, shape.rows - 1]);
    }
    if shape.rows > 3 {
        partitions.push(vec![2, 1, shape.rows - 3]);
    }
    if shape.rows > 1 {
        partitions.push(vec![1; shape.rows]);
    }
    let trial = Trial {
        context,
        function: module.function("gdn_recurrent")?,
        input,
        shape,
        initial,
    };
    let whole = trial.sequence(&partitions[0])?;
    let mut reports = vec![whole.report];
    for partition in &partitions[1..] {
        let chunked = trial.sequence(partition)?;
        ensure!(
            chunked.output == whole.output,
            "GDN output depends on partition"
        );
        exact(
            &chunked.unrounded,
            &whole.unrounded,
            "partition FP32 output",
        )?;
        exact(&chunked.state, &whole.state, "partition final state")?;
        reports.push(chunked.report);
    }
    Ok(json!({
        "all_passed":true,
        "shape":{"rows":shape.rows,"key_heads":shape.key_heads,"value_heads":shape.value_heads,"width":shape.width},
        "elements":whole.output.len(),"state_elements":whole.state.len(),
        "partitions":reports,"chunk_outputs_exact":true,"chunk_final_state_exact":true,
        "reference":"independent logical scalar, increasing-key ordered FP32 multiply/add; BF16 RNE output",
        "wide_reduction_diagnostic":{"acceptance_gate":false,
            "output_max_abs_error":max_error(&whole.unrounded,&wide.unrounded)?,
            "state_max_abs_error":max_error(&whole.state,&wide.state)?,
            "bf16_output_differences":whole.output.iter().zip(&wide.output).filter(|(a,b)|a!=b).count()},
        "state_transport":"in-place resident FP32 state, one thread owns each value column; no host replacement between chunks",
        "qk_head_mapping":"value head divided by value/key head ratio",
        "device_inputs_resident":true,"model_executable":false
    }))
}

struct Sequence {
    output: Vec<u16>,
    unrounded: Vec<f32>,
    state: Vec<f32>,
    report: Value,
}

struct Trial<'a, 'ctx, 'module> {
    context: &'ctx Context,
    function: Function<'module, 'ctx>,
    input: Input<'a, 'ctx>,
    shape: &'a reference::Shape,
    initial: &'a [f32],
}

impl Trial<'_, '_, '_> {
    fn sequence(&self, partition: &[usize]) -> Result<Sequence> {
        ensure!(
            partition.iter().all(|&n| n > 0) && partition.iter().sum::<usize>() == self.shape.rows,
            "invalid GDN recurrence partition"
        );
        let state = upload(self.context, &float_bytes(self.initial))?;
        let count = self.shape.rows * self.shape.value_heads * self.shape.width;
        let output = upload(self.context, &vec![0xa5; count * 2])?;
        let unrounded = upload(self.context, &vec![0xff; count * 4])?;
        let mut expected_state = self.initial.to_vec();
        let mut expected_output = Vec::with_capacity(count);
        let mut expected_unrounded = Vec::with_capacity(count);
        let mut offset = 0;
        for &rows in partition {
            let expected = reference::run(
                &self.host_chunk(offset, rows),
                &expected_state,
                &reference::Shape {
                    rows,
                    ..*self.shape
                },
                reference::Reduction::OrderedF32,
            )?;
            self.launch(offset, rows, &state, &output, &unrounded)?;
            exact(
                &floats(&state, expected_state.len())?,
                &expected.state,
                "chunk state vs scalar",
            )?;
            expected_state = expected.state;
            expected_output.extend(expected.output);
            expected_unrounded.extend(expected.unrounded);
            offset += rows;
        }
        let actual_output = words(&output, count)?;
        let actual_unrounded = floats(&unrounded, count)?;
        ensure!(
            actual_output == expected_output,
            "GDN BF16 output differs from ordered scalar reference"
        );
        exact(
            &actual_unrounded,
            &expected_unrounded,
            "FP32 output vs scalar",
        )?;
        Ok(Sequence {
            output: actual_output,
            unrounded: actual_unrounded,
            state: floats(&state, expected_state.len())?,
            report: json!({"chunks":partition,"output_bf16_exact":true,"output_fp32_exact":true,
                "state_exact_at_every_boundary":true,"state_elements_per_boundary":expected_state.len()}),
        })
    }

    fn host_chunk(&self, offset: usize, rows: usize) -> reference::Input<'_> {
        let keys = self.shape.key_heads * self.shape.width;
        let channels = (2 * self.shape.key_heads + self.shape.value_heads) * self.shape.width;
        let heads = self.shape.value_heads;
        let end = offset + rows;
        reference::Input {
            q: &self.input.host.q[offset * keys..end * keys],
            k: &self.input.host.k[offset * keys..end * keys],
            qkv: &self.input.host.qkv[offset * channels..end * channels],
            beta: &self.input.host.beta[offset * heads..end * heads],
            decay: &self.input.host.decay[offset * heads..end * heads],
        }
    }

    fn launch(
        &self,
        offset: usize,
        rows: usize,
        state: &Buffer<'_>,
        output: &Buffer<'_>,
        unrounded: &Buffer<'_>,
    ) -> Result<()> {
        let keys = offset * self.shape.key_heads * self.shape.width;
        let gates = offset * self.shape.value_heads;
        let channels =
            offset * (2 * self.shape.key_heads + self.shape.value_heads) * self.shape.width;
        let values = offset * self.shape.value_heads * self.shape.width;
        let mut pointers = [
            self.input.q.pointer() + (keys * 4) as u64,
            self.input.k.pointer() + (keys * 4) as u64,
            self.input.qkv.pointer() + (channels * 2) as u64,
            self.input.beta.pointer() + (gates * 2) as u64,
            self.input.decay.pointer() + (gates * 4) as u64,
            state.pointer(),
            output.pointer() + (values * 2) as u64,
            unrounded.pointer() + (values * 4) as u64,
        ];
        let mut dims = [
            u32::try_from(rows)?,
            u32::try_from(self.shape.key_heads)?,
            u32::try_from(self.shape.value_heads)?,
            u32::try_from(self.shape.width)?,
        ];
        let mut args: Vec<*mut c_void> = pointers
            .iter_mut()
            .map(|x| (x as *mut u64).cast())
            .collect();
        args.extend(dims.iter_mut().map(|x| (x as *mut u32).cast()));
        // SAFETY: Eight disjoint allocations plus four u32 dimensions match the
        // kernel ABI. Validated partition offsets cover every row. Each head/column
        // has exactly one owner, and every launch synchronizes before state reuse.
        unsafe {
            self.function
                .launch([dims[2], 1, 1], [dims[3], 1, 1], 0, &mut args)?;
        }
        self.context.synchronize()
    }
}

fn exact(actual: &[f32], expected: &[f32], label: &str) -> Result<()> {
    ensure!(actual.len() == expected.len(), "{label}: extent mismatch");
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        ensure!(
            a.is_finite() && e.is_finite() && a.to_bits() == e.to_bits(),
            "{label}: index {index}: {a:?} vs {e:?}"
        );
    }
    Ok(())
}
fn max_error(actual: &[f32], expected: &[f32]) -> Result<f64> {
    ensure!(actual.len() == expected.len(), "diagnostic extent mismatch");
    ensure!(
        actual.iter().chain(expected).all(|v| v.is_finite()),
        "nonfinite diagnostic value"
    );
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(&a, &e)| (f64::from(a) - f64::from(e)).abs())
        .fold(0.0, f64::max))
}
fn upload<'a>(context: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}
fn float_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn word_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
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

pub(super) fn fixtures(context: &Context, module: &Module<'_>) -> Result<Vec<Value>> {
    let mut reports = Vec::new();
    for shape in [
        reference::Shape {
            rows: 7,
            key_heads: 2,
            value_heads: 6,
            width: 8,
        },
        reference::Shape {
            rows: 3,
            key_heads: 1,
            value_heads: 2,
            width: 1,
        },
        reference::Shape {
            rows: 3,
            key_heads: 1,
            value_heads: 1,
            width: 256,
        },
    ] {
        reports.extend(fixture_shape(context, module, &shape)?);
    }
    Ok(reports)
}

fn fixture_shape(
    context: &Context,
    module: &Module<'_>,
    shape: &reference::Shape,
) -> Result<Vec<Value>> {
    let channels = (2 * shape.key_heads + shape.value_heads) * shape.width;
    let q: Vec<_> = (0..shape.rows * shape.key_heads * shape.width)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) / 64.0)
        .collect();
    let k: Vec<_> = (0..q.len())
        .map(|i| ((i * 11 % 19) as f32 - 9.0) / 32.0)
        .collect();
    let qkv: Vec<_> = (0..shape.rows * channels)
        .map(|i| round_bf16(((i * 3 % 29) as f32 - 14.0) / 8.0))
        .collect();
    let beta: Vec<_> = (0..shape.rows * shape.value_heads)
        .map(|i| round_bf16((i % 5) as f32 / 4.0))
        .collect();
    let decay: Vec<_> = (0..beta.len()).map(|i| (i % 7) as f32 / 6.0).collect();
    let state: Vec<_> = (0..shape.value_heads * shape.width * shape.width)
        .map(|i| ((i * 13 % 31) as f32 - 15.0) / 16.0)
        .collect();
    let host = reference::Input {
        q: &q,
        k: &k,
        qkv: &qkv,
        beta: &beta,
        decay: &decay,
    };
    let q_device = upload(context, &float_bytes(&q))?;
    let k_device = upload(context, &float_bytes(&k))?;
    let qkv_device = upload(context, &word_bytes(&qkv))?;
    let beta_device = upload(context, &word_bytes(&beta))?;
    let decay_device = upload(context, &float_bytes(&decay))?;
    let resident = || Input {
        q: &q_device,
        k: &k_device,
        qkv: &qkv_device,
        beta: &beta_device,
        decay: &decay_device,
        host: reference::Input {
            q: host.q,
            k: host.k,
            qkv: host.qkv,
            beta: host.beta,
            decay: host.decay,
        },
    };
    let nonzero = check_state(context, module, resident(), shape, &state)?;
    let reset = check(context, module, resident(), shape)?;
    Ok(vec![nonzero, reset])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_comparison_rejects_nonfinite_changed_bits_and_extent() {
        assert!(exact(&[1.0], &[1.0], "test").is_ok());
        assert!(exact(&[1.0], &[1.00001], "test").is_err());
        assert!(exact(&[0.0], &[-0.0], "test").is_err());
        assert!(exact(&[f32::NAN], &[f32::NAN], "test").is_err());
        assert!(exact(&[], &[0.0], "test").is_err());
    }
}
