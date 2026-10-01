//! Real-weight comparison of existing MLP execution and two workspace schedules.
use super::{
    driver::{Buffer, Context, Module},
    resident_mlp::{Mlp, ResultBuffers},
    resident_mlp_workspace::Chain,
    resident_projection::Quantization,
    resident_weights::ResidentWeights,
};
use crate::{
    artifact::{model_source::ModelArtifact, schema::Object},
    entry_reference::round_bf16,
    kernels::MlpWorkspaceCase,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{collections::BTreeMap, time::Instant};

pub(crate) fn run(
    ptx: &str,
    device: i32,
    artifact: &mut ModelArtifact,
    objects: &[Object],
    cases: &[MlpWorkspaceCase],
) -> Result<Value> {
    let ctx = Context::new(device)?;
    let info = ctx.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "workspace trial requires SM120"
    );
    let module = Module::load(&ctx, ptx)?;
    let weights = ResidentWeights::load(&ctx, artifact, objects)?;
    let mut reports = Vec::new();
    for case in cases {
        for rows in [1, 5, 128, 512] {
            reports.push(check(&ctx, &module, &weights, case, rows)?);
        }
    }
    Ok(
        json!({"kind":"real-weight-mlp-workspace-trial","device":info,"all_passed":reports.iter().all(|v|v["all_passed"]==true),"cases":reports,
        "scope":"real checkpoint weights and deterministic signed BF16 input fixtures; not model throughput or quality"}),
    )
}
fn check(
    ctx: &Context,
    module: &Module<'_>,
    weights: &ResidentWeights<'_>,
    case: &MlpWorkspaceCase,
    rows: usize,
) -> Result<Value> {
    ensure!(
        crate::kernels::nvfp4_mlp_schedule::current()?
            == crate::kernels::nvfp4_mlp_schedule::Schedule::Baseline,
        "MLP workspace checks require the baseline NVFP4 MLP schedule"
    );
    let kind = if case.fp8 {
        Quantization::Fp8
    } else {
        Quantization::Nvfp4
    };
    let control = Mlp::new(weights, &case.prefix, case.width, case.channels, kind)?;
    let bytes = (0..rows * case.width)
        .flat_map(|i| round_bf16(((i * 13 + i / 19) % 127) as f32 / 64.0 - 0.984375).to_le_bytes())
        .collect::<Vec<_>>();
    let input = Buffer::new(ctx, bytes.len())?;
    input.upload(&bytes)?;
    let expected = snapshot(control.run(ctx, module, &input, rows)?)?;
    let mut chain = Chain::new(
        ctx,
        weights,
        &case.prefix,
        [rows, case.width, case.channels],
        kind,
    )?;
    let addresses = chain.addresses()?;
    let mut variants = Vec::new();
    for operator_waits in [true, false] {
        chain.run(module, &input, operator_waits)?;
        let actual = chain.snapshot()?.into_iter().collect::<BTreeMap<_, _>>();
        let exact = actual == expected;
        let mut timings = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            chain.run(module, &input, operator_waits)?;
            timings.push(start.elapsed().as_secs_f64());
        }
        ensure!(
            chain.snapshot()?.into_iter().collect::<BTreeMap<_, _>>() == expected,
            "workspace reuse changed output"
        );
        ensure!(
            chain.addresses()? == addresses,
            "workspace addresses changed across reuse"
        );
        variants.push(
            json!({"operator_waits":operator_waits,"all_passed":exact,"wall_seconds":timings}),
        );
    }
    let prepared_times = chain.prepared_trial(module, &input, false)?;
    let prepared_exact = chain.snapshot()?.into_iter().collect::<BTreeMap<_, _>>() == expected;
    ensure!(prepared_exact, "prepared MLP changed output");
    ensure!(
        chain.addresses()? == addresses,
        "prepared MLP changed workspace addresses"
    );
    let mut control_times = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        let output = control.run(ctx, module, &input, rows)?;
        drop(output);
        control_times.push(start.elapsed().as_secs_f64());
    }
    let abort_checked = rows == 1;
    if abort_checked {
        let mut aborted = Chain::new(
            ctx,
            weights,
            &case.prefix,
            [rows, case.width, case.channels],
            kind,
        )?;
        let failure = aborted
            .abort_after_gate(module, &input)
            .err()
            .ok_or_else(|| anyhow::anyhow!("injected gate failure unexpectedly succeeded"))?;
        ensure!(
            failure.to_string().contains("injected"),
            "abort did not reach queued gate"
        );
        ensure!(
            aborted.run(module, &input, false).is_err(),
            "aborted workspace was reused"
        );
    }
    if abort_checked {
        let mut aborted = Chain::new(
            ctx,
            weights,
            &case.prefix,
            [rows, case.width, case.channels],
            kind,
        )?;
        let error = aborted
            .prepared_trial(module, &input, true)
            .err()
            .ok_or_else(|| anyhow::anyhow!("prepared abort unexpectedly succeeded"))?;
        ensure!(
            error.to_string().contains("injected"),
            "prepared abort did not reach gate"
        );
        ensure!(
            aborted.prepared_trial(module, &input, false).is_err(),
            "prepared aborted workspace was reused"
        );
    }
    Ok(
        json!({"prefix":case.prefix,"rows":rows,"width":case.width,"channels":case.channels,"fp8":case.fp8,
        "stable_addresses":true,"abort_after_gate_checked":abort_checked,
        "prepared_exact":prepared_exact,"prepared_warmup_repetitions":1,"prepared_submit_and_completion_seconds":prepared_times,
        "prepared_abort_after_gate_checked":abort_checked,"workspace_bytes":chain.bytes(),"control_wall_seconds":control_times,"variants":variants,
        "all_passed":variants.iter().all(|v|v["all_passed"]==true),"compared_buffers":expected.len(),
        "input_fixture":"deterministic signed BF16, not recorded model activations","timing_order":"workspace-waits,workspace-one-wait,prepared-stream,control; screening only; prepared excludes preparation/view binding"}),
    )
}
fn snapshot(output: ResultBuffers<'_>) -> Result<BTreeMap<String, Vec<u8>>> {
    let entries = [
        ("gate.values", &output.gate.values),
        ("gate.raw", &output.gate.unrounded),
        ("up.values", &output.up.values),
        ("up.raw", &output.up.unrounded),
        ("activation.values", &output.activation),
        ("down.values", &output.down.values),
        ("down.raw", &output.down.unrounded),
    ];
    entries
        .into_iter()
        .map(|(name, buffer)| {
            let mut bytes = vec![0; buffer.len()];
            buffer.download(&mut bytes)?;
            let finite =
                if name.ends_with(".raw") {
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|v| f32::from_le_bytes(*v).is_finite())
                } else {
                    bytes.as_chunks::<2>().0.iter().all(|v| {
                        f32::from_bits(u32::from(u16::from_le_bytes(*v)) << 16).is_finite()
                    })
                };
            ensure!(finite, "nonfinite control MLP values in {name}");
            Ok((name.to_owned(), bytes))
        })
        .collect()
}
