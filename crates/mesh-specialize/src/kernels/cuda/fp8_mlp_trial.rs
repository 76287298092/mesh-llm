//! Independent validation around the reference-free persistent FP8 MLP path.

use super::{
    driver::{Buffer, Context, Module},
    resident_mlp::Fp8Mlp,
    resident_weights::ResidentWeights,
};
use crate::{
    artifact::{reader::VerifiedArtifact, schema::Object},
    engine::layout::Layout,
    entry_reference::{bf16_to_f32, round_bf16},
    kernels::Fp8MlpCase,
};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};

pub(in crate::kernels) fn run(
    ptx: &str,
    device: i32,
    artifact: &mut VerifiedArtifact,
    objects: &[Object],
    cases: &[Fp8MlpCase],
) -> Result<Value> {
    ensure!(
        ptx.contains(".target sm_120a"),
        "FP8 MLP requires SM120a PTX"
    );
    ensure!(
        (1..=16).contains(&cases.len()),
        "invalid FP8 MLP trial case count"
    );
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "FP8 MLP trial requires SM120"
    );
    let module = Module::load(&context, ptx)?;
    for name in ["fp8_quantize_bf16", "fp8_linear_wide", "mlp_silu_product"] {
        module.function(name)?;
    }
    let before = context.memory()?;
    let layout = Layout::new(
        objects
            .iter()
            .map(|object| (object.name.clone(), object.length)),
    )?;
    let needed = layout
        .bytes
        .checked_add(1024 * 1024 * 1024)
        .context("FP8 MLP admission overflow")?;
    ensure!(
        needed <= u64::try_from(before.0)?,
        "insufficient GPU memory for resident text weights plus workspace reserve"
    );
    let weights = ResidentWeights::load(&context, artifact, objects)?;
    context_rejection(&context, &module, &weights, &cases[0], ptx, device)?;
    let resident = context.memory()?;
    let mut reports = Vec::new();
    for case in cases {
        reports.push(run_case(&context, &module, &weights, case)?);
    }
    drop(weights);
    context.synchronize()?;
    let after = context.memory()?;
    Ok(json!({
        "schema_version":1,"kind":"qwen-resident-fp8-mlp-trial","device":info,
        "all_passed":reports.iter().all(|report|report["all_passed"]==true)&&after.0>=before.0,
        "cases":reports,"resident_weight_objects":objects.len(),"weight_arena_bytes":layout.bytes,
        "memory_before_weights":{"free_bytes":before.0,"total_bytes":before.1},
        "memory_with_weights":{"free_bytes":resident.0,"total_bytes":resident.1},
        "memory_after_free":{"free_bytes":after.0,"total_bytes":after.1},
        "arena_allocations_released":after.0>=before.0,
        "reference_free_execution":true,"device_intermediates_resident":true,
        "foreign_context_rejections":5,
        "linear_resources":module.function("fp8_linear_wide")?.resources()?,
        "timing_collected":false,"full_model_executed":false,
    }))
}

fn context_rejection(
    context: &Context,
    module: &Module<'_>,
    weights: &ResidentWeights<'_>,
    case: &Fp8MlpCase,
    ptx: &str,
    device: i32,
) -> Result<()> {
    let foreign = Context::new(device)?;
    let foreign_module = Module::load(&foreign, ptx)?;
    let input = Buffer::new(context, case.width * 2)?;
    let foreign_input = Buffer::new(&foreign, case.width * 2)?;
    let projection = super::resident_fp8::Projection::new(
        weights,
        &format!("{}.gate_proj", case.prefix),
        case.width,
        case.channels,
    )?;
    let checks = [
        rejected(projection.run(context, module, &foreign_input, 1)),
        rejected(projection.run(context, &foreign_module, &input, 1)),
        rejected(projection.run(&foreign, &foreign_module, &foreign_input, 1)),
        rejected(super::resident_activation::run(
            context,
            module,
            &foreign_input,
            &input,
            case.width,
        )),
        rejected(super::resident_activation::run(
            context,
            &foreign_module,
            &input,
            &input,
            case.width,
        )),
    ];
    ensure!(
        checks.into_iter().all(|value| value),
        "resident execution accepted a foreign CUDA context"
    );
    Ok(())
}

fn rejected<T>(result: Result<T>) -> bool {
    result
        .err()
        .is_some_and(|error| error.to_string().contains("CUDA context"))
}

fn run_case(
    context: &Context,
    module: &Module<'_>,
    weights: &ResidentWeights<'_>,
    case: &Fp8MlpCase,
) -> Result<Value> {
    let mlp = Fp8Mlp::new(weights, &case.prefix, case.width, case.channels)?;
    let bytes: Vec<_> = case
        .input
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let input = Buffer::new(context, bytes.len())?;
    input.upload(&bytes)?;
    let result = mlp.run(context, module, &input, case.rows)?;
    let checks = [
        (
            "gate",
            compare(
                &result.gate.values,
                Some(&result.gate.unrounded),
                &case.reference.gate,
                case.channels,
            )?,
        ),
        (
            "up",
            compare(
                &result.up.values,
                Some(&result.up.unrounded),
                &case.reference.up,
                case.channels,
            )?,
        ),
        (
            "activation",
            compare(
                &result.activation,
                None,
                &case.reference.activation,
                case.channels,
            )?,
        ),
        (
            "down",
            compare(
                &result.down.values,
                Some(&result.down.unrounded),
                &case.reference.down,
                case.width,
            )?,
        ),
    ];
    let memory = context.memory()?;
    let mut report = json!({"prefix":case.prefix,"rows":case.rows,"width":case.width,"channels":case.channels,
        "all_passed":checks.iter().all(|(_,check)|check["all_passed"]==true),
        "memory_with_outputs":{"free_bytes":memory.0,"total_bytes":memory.1}});
    for (name, check) in checks {
        report[name] = check;
    }
    Ok(report)
}

fn compare(
    buffer: &Buffer<'_>,
    unrounded: Option<&Buffer<'_>>,
    expected: &[u16],
    width: usize,
) -> Result<Value> {
    let mut bytes = vec![0; buffer.len()];
    ensure!(
        bytes.len() == expected.len() * 2,
        "MLP comparison extent mismatch"
    );
    buffer.download(&mut bytes)?;
    let actual: Vec<_> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| u16::from_le_bytes(*bytes))
        .collect();
    let decode = |words: &[u16]| {
        words
            .iter()
            .map(|&word| bf16_to_f32(word))
            .collect::<Vec<_>>()
    };
    let mut report = crate::layer_comparison_reference::compare_partitioned(
        &decode(&actual),
        &decode(expected),
        width,
    )?;
    report["bf16_differences"] = json!(actual.iter().zip(expected).filter(|(a, b)| a != b).count());
    if let Some(unrounded) = unrounded {
        ensure!(
            unrounded.len() == expected.len() * 4,
            "MLP unrounded extent mismatch"
        );
        let mut floats = vec![0; unrounded.len()];
        unrounded.download(&mut floats)?;
        let failures = floats
            .as_chunks::<4>()
            .0
            .iter()
            .zip(&actual)
            .filter(|(bytes, word)| {
                let value = f32::from_le_bytes(**bytes);
                !value.is_finite() || round_bf16(value) != **word
            })
            .count();
        report["rounding_mismatches"] = json!(failures);
        report["all_passed"] = json!(report["all_passed"] == true && failures == 0);
    }
    Ok(report)
}
