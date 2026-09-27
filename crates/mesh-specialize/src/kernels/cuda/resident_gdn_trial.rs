//! Compare persistent GDN execution with an independent whole-layer reference.
use super::{
    driver::{Buffer, Context, Module},
    resident_embedding::Embedding,
    resident_gdn::Layer,
    resident_mlp::Quantization,
    resident_state::ResidentState,
    resident_weights::ResidentWeights,
};
use crate::{
    artifact::{reader::VerifiedArtifact, schema::Object},
    engine::layout::Layout,
    entry_reference::bf16_to_f32,
    kernels::{ResidentGdnCase, ResidentGdnConfig},
};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};

pub(in crate::kernels) fn run(
    ptx: &str,
    device: i32,
    artifact: &mut VerifiedArtifact,
    objects: &[Object],
    config: &ResidentGdnConfig,
    cases: &[ResidentGdnCase],
) -> Result<Value> {
    ensure!(
        ptx.contains(".target sm_120a"),
        "resident GDN requires SM120a PTX"
    );
    ensure!((1..=16).contains(&cases.len()), "invalid GDN case count");
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "resident GDN requires SM120"
    );
    copy_fixture(&context, device)?;
    let module = Module::load(&context, ptx)?;
    for name in [
        "embedding_norm_bf16",
        "fp8_quantize_bf16",
        "fp8_linear_wide",
        "bf16_linear",
        "causal_conv4_bf16",
        "gdn_qk_norm",
        "gdn_gates",
        "gdn_recurrent",
        "gdn_gated_rms_norm",
        "residual_norm_bf16",
        "nvfp4_quantize_bf16",
        "nvfp4_linear",
        "mlp_silu_product",
        "residual_add_bf16",
    ] {
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
        .checked_add(config.state_layout.bytes)
        .and_then(|bytes| bytes.checked_add(1024 * 1024 * 1024))
        .context("GDN admission overflow")?;
    ensure!(
        needed <= u64::try_from(before.0)?,
        "insufficient memory for resident GDN trial"
    );
    let weights = ResidentWeights::load(&context, artifact, objects)?;
    let layer = Layer::new(
        &weights,
        &config.prefix,
        &config.state_prefix,
        &config.shape,
        Quantization::Nvfp4,
    )?;
    let embedding = Embedding::new(
        &weights,
        &config.table_name,
        &format!("{}.input_layernorm.weight", config.prefix),
        [config.vocabulary, config.shape.hidden],
        1e-6,
    )?;
    let mut reports = Vec::new();
    for case in cases {
        let trial = Trial {
            context: &context,
            module: &module,
            layer: &layer,
            embedding: &embedding,
            config,
        };
        reports.push(trial.check(case)?);
    }
    drop(embedding);
    drop(layer);
    drop(weights);
    context.synchronize()?;
    let after = context.memory()?;
    Ok(
        json!({"schema_version":1,"kind":"qwen-reference-free-resident-gdn-trial","device":info,
        "all_passed":reports.iter().all(|case|case["all_passed"]==true)&&after.0>=before.0,"cases":reports,
        "resident_weight_objects":objects.len(),"weight_arena_bytes":layout.bytes,"state_arena_bytes":config.state_layout.bytes,
        "device_copy_fixture_passed":true,"reference_free_execution":true,
        "memory_before":{"free_bytes":before.0,"total_bytes":before.1},"memory_after_free":{"free_bytes":after.0,"total_bytes":after.1},
        "arena_allocations_released":after.0>=before.0,"full_model_executed":false,"timing_collected":false}),
    )
}

struct Trial<'a, 'w, 'ctx> {
    context: &'ctx Context,
    module: &'a Module<'ctx>,
    layer: &'a Layer<'w, 'ctx>,
    embedding: &'a Embedding<'w, 'ctx>,
    config: &'a ResidentGdnConfig,
}
struct Sequence {
    output: Vec<u16>,
    history: Vec<u16>,
    state: Vec<f32>,
    memory: Value,
}
impl Trial<'_, '_, '_> {
    fn check(&self, case: &ResidentGdnCase) -> Result<Value> {
        ensure!(!case.tokens.is_empty(), "empty GDN trial case");
        let whole = self.sequence(&case.tokens, &[case.tokens.len()])?;
        let hidden = compare_words(
            &whole.output,
            &case.reference.output,
            self.config.shape.hidden,
        )?;
        let channels = (2 * self.config.shape.key_heads + self.config.shape.value_heads)
            * self.config.shape.head_width;
        let history = compare_words(
            &whole.history,
            &case.reference.convolution_history,
            channels,
        )?;
        let recurrent = crate::layer_comparison_reference::compare_partitioned(
            &whole.state,
            &case.reference.recurrent_state,
            self.config.shape.head_width * self.config.shape.head_width,
        )?;
        let mut partitions = vec![vec![1; case.tokens.len()]];
        if case.tokens.len() > 1 {
            partitions.push(vec![1, case.tokens.len() - 1]);
        }
        let mut checks = Vec::new();
        for partition in partitions {
            let sequence = self.sequence(&case.tokens, &partition)?;
            let exact = sequence.output == whole.output
                && sequence.history == whole.history
                && sequence
                    .state
                    .iter()
                    .zip(&whole.state)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                && sequence.state.len() == whole.state.len();
            checks.push(json!({"partition":partition,"output_history_state_bit_exact":exact}));
        }
        Ok(
            json!({"tokens":case.tokens,"hidden":hidden,"history":history,"recurrent":recurrent,"partitions":checks,
            "memory_with_sequence":whole.memory,
            "all_passed":hidden["all_passed"]==true&&history["all_passed"]==true&&recurrent["all_passed"]==true&&checks.iter().all(|check|check["output_history_state_bit_exact"]==true)}),
        )
    }

    fn sequence(&self, tokens: &[u32], partition: &[usize]) -> Result<Sequence> {
        ensure!(
            partition.iter().all(|&rows| rows > 0)
                && partition.iter().sum::<usize>() == tokens.len(),
            "invalid resident GDN partition"
        );
        let mut state = ResidentState::new(self.context, &self.config.state_layout)?;
        let mut output = Vec::new();
        let mut offset = 0;
        for &rows in partition {
            let entry =
                self.embedding
                    .run(self.context, self.module, &tokens[offset..offset + rows])?;
            let result =
                self.layer
                    .forward(self.context, self.module, &entry.residual, &mut state, rows)?;
            let mut bytes = vec![0; result.len()];
            result.download(&mut bytes)?;
            output.extend(
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|bytes| u16::from_le_bytes(*bytes)),
            );
            offset += rows;
        }
        let shape = &self.config.shape;
        let mut history = vec![0; (2 * shape.key_heads + shape.value_heads) * shape.head_width * 6];
        state.read_region(
            &format!("{}.gdn.history", self.config.state_prefix),
            &mut history,
        )?;
        let mut recurrent = vec![0; shape.value_heads * shape.head_width * shape.head_width * 4];
        state.read_region(
            &format!("{}.gdn.recurrent", self.config.state_prefix),
            &mut recurrent,
        )?;
        let memory = self.context.memory()?;
        Ok(Sequence {
            output,
            history: history
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes(*b))
                .collect(),
            state: recurrent
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            memory: json!({"free_bytes":memory.0,"total_bytes":memory.1}),
        })
    }
}

fn compare_words(actual: &[u16], expected: &[u16], width: usize) -> Result<Value> {
    let decode = |values: &[u16]| {
        values
            .iter()
            .map(|&value| bf16_to_f32(value))
            .collect::<Vec<_>>()
    };
    let mut report = crate::layer_comparison_reference::compare_partitioned(
        &decode(actual),
        &decode(expected),
        width,
    )?;
    report["bf16_differences"] = json!(actual.iter().zip(expected).filter(|(a, b)| a != b).count());
    Ok(report)
}

fn copy_fixture(ctx: &Context, device: i32) -> Result<()> {
    let source = Buffer::new(ctx, 16)?;
    source.upload(&(0..16).collect::<Vec<u8>>())?;
    let target = Buffer::new(ctx, 32)?;
    target.upload(&[0; 32])?;
    target.copy_from_at(7, &source, 3, 5)?;
    let mut actual = [0; 32];
    target.download(&mut actual)?;
    let mut expected = [0; 32];
    expected[7..12].copy_from_slice(&[3, 4, 5, 6, 7]);
    ensure!(actual == expected, "device-copy offset fixture failed");
    ensure!(
        target.copy_from_at(31, &source, 0, 2).is_err()
            && target.copy_from_at(0, &source, 15, 2).is_err()
            && source.copy_from_at(0, &source, 1, 1).is_err(),
        "device-copy range/alias rejection failed"
    );
    let foreign = Context::new(device)?;
    let foreign_buffer = Buffer::new(&foreign, 16)?;
    ensure!(
        target.copy_from_at(0, &foreign_buffer, 0, 1).is_err(),
        "device copy accepted foreign CUDA context"
    );
    Ok(())
}
