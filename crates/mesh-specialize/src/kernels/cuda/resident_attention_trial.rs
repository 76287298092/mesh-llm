//! Independent complete-block and persistent K/V qualification.
use super::{
    driver::{Buffer, Context, Module},
    resident_attention::{Layer, Step},
    resident_embedding::Embedding,
    resident_projection::Quantization,
    resident_state::ResidentState,
    resident_weights::ResidentWeights,
};
use crate::{
    artifact::{reader::VerifiedArtifact, schema::Object},
    engine::layout::Layout,
    entry_reference::bf16_to_f32,
    kernels::{ResidentAttentionCase, ResidentAttentionConfig},
};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};

pub(in crate::kernels) fn run(
    ptx: &str,
    device: i32,
    artifact: &mut VerifiedArtifact,
    objects: &[Object],
    config: &ResidentAttentionConfig,
    cases: &[ResidentAttentionCase],
) -> Result<Value> {
    ensure!(
        ptx.contains(".target sm_120a"),
        "resident attention requires SM120a PTX"
    );
    ensure!(
        (1..=16).contains(&cases.len()),
        "invalid resident attention case count"
    );
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "resident attention requires SM120"
    );
    let module = Module::load(&context, ptx)?;
    for name in [
        "embedding_norm_bf16",
        "fp8_quantize_bf16",
        "fp8_linear_wide",
        "attention_qk_prepare",
        "attention_kv_append",
        "causal_attention_bf16",
        "attention_gate_bf16",
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
        .context("attention admission overflow")?;
    ensure!(
        needed <= u64::try_from(before.0)?,
        "insufficient memory for resident attention trial"
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
    let trial = Trial {
        context: &context,
        module: &module,
        layer: &layer,
        embedding: &embedding,
        config,
    };
    let reports = cases
        .iter()
        .map(|case| trial.check(case))
        .collect::<Result<Vec<_>>>()?;
    drop(layer);
    drop(weights);
    context.synchronize()?;
    let after = context.memory()?;
    Ok(
        json!({"schema_version":1,"kind":"qwen-reference-free-resident-attention-trial","device":info,
        "all_passed":reports.iter().all(|case|case["all_passed"]==true)&&after.0>=before.0,"cases":reports,
        "resident_weight_objects":objects.len(),"weight_arena_bytes":layout.bytes,"state_arena_bytes":config.state_layout.bytes,
        "reference_free_execution":true,"memory_before":{"free_bytes":before.0,"total_bytes":before.1},
        "memory_after_free":{"free_bytes":after.0,"total_bytes":after.1},"arena_allocations_released":after.0>=before.0,
        "full_model_executed":false,"timing_collected":false}),
    )
}

struct Trial<'a, 'w, 'ctx> {
    context: &'ctx Context,
    module: &'a Module<'ctx>,
    layer: &'a Layer<'w, 'ctx>,
    embedding: &'a Embedding<'w, 'ctx>,
    config: &'a ResidentAttentionConfig,
}
struct Sequence {
    output: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    boundaries: Vec<Value>,
    memory: Value,
}
impl Trial<'_, '_, '_> {
    fn check(&self, case: &ResidentAttentionCase) -> Result<Value> {
        ensure!(
            !case.tokens.is_empty() && case.tokens.len() <= self.config.capacity,
            "invalid attention trial tokens"
        );
        let whole = self.sequence(case, &[case.tokens.len()])?;
        let hidden = compare_words(
            &whole.output,
            &case.reference.output,
            self.config.shape.hidden,
        )?;
        let mut partitions = vec![vec![1; case.tokens.len()]];
        if case.tokens.len() > 1 {
            partitions.push(vec![1, case.tokens.len() - 1]);
        }
        if case.tokens.len() > 3 {
            partitions.push(vec![2, 1, case.tokens.len() - 3]);
        }
        let mut checks = Vec::new();
        for partition in partitions {
            let actual = self.sequence(case, &partition)?;
            let exact = actual.output == whole.output && actual.k == whole.k && actual.v == whole.v;
            checks.push(json!({"partition":partition,"output_and_cache_bit_exact":exact,"boundaries":actual.boundaries}));
        }
        Ok(
            json!({"tokens":case.tokens,"hidden":hidden,"whole_boundaries":whole.boundaries,"partitions":checks,
            "memory_with_sequence":whole.memory,"capacity_overrun_rejected":true,
            "all_passed":hidden["all_passed"]==true&&checks.iter().all(|check|check["output_and_cache_bit_exact"]==true)}),
        )
    }

    fn sequence(&self, case: &ResidentAttentionCase, partition: &[usize]) -> Result<Sequence> {
        ensure!(
            partition.iter().all(|&rows| rows > 0)
                && partition.iter().sum::<usize>() == case.tokens.len(),
            "invalid attention partition"
        );
        let mut state = ResidentState::new(self.context, &self.config.state_layout)?;
        let mut output = Vec::new();
        let mut past = 0;
        let mut boundaries = Vec::new();
        for &rows in partition {
            let entry =
                self.embedding
                    .run(self.context, self.module, &case.tokens[past..past + rows])?;
            let result = self.layer.forward(
                self.context,
                self.module,
                &entry.residual,
                &mut state,
                &Step {
                    rows,
                    past,
                    capacity: self.config.capacity,
                },
            )?;
            output.extend(words(&result)?);
            past += rows;
            let (k, v) = self.cache(&state)?;
            let initialized = past * self.config.shape.kv_heads * self.config.shape.head_width;
            ensure!(
                case.reference.k_cache.len() >= initialized
                    && case.reference.v_cache.len() >= initialized,
                "attention reference cache extent mismatch"
            );
            let prefix = k[..initialized] == case.reference.k_cache[..initialized]
                && v[..initialized] == case.reference.v_cache[..initialized];
            let tail = k[initialized..]
                .iter()
                .chain(&v[initialized..])
                .all(|&word| word == 0);
            ensure!(
                prefix && tail,
                "resident attention cache prefix/tail mismatch at {past} tokens"
            );
            boundaries.push(json!({"prefix_tokens":past,"cache_reference_exact":prefix,"unused_tail_zero":tail}));
        }
        let (k, v) = self.cache(&state)?;
        let dummy = Buffer::new(self.context, self.config.shape.hidden * 2)?;
        ensure!(
            self.layer
                .forward(
                    self.context,
                    self.module,
                    &dummy,
                    &mut state,
                    &Step {
                        rows: 1,
                        past: self.config.capacity,
                        capacity: self.config.capacity
                    }
                )
                .is_err(),
            "attention accepted context overflow"
        );
        let after = self.cache(&state)?;
        ensure!(
            after.0 == k && after.1 == v,
            "rejected attention request mutated cache"
        );
        let memory = self.context.memory()?;
        Ok(Sequence {
            output,
            k,
            v,
            boundaries,
            memory: json!({"free_bytes":memory.0,"total_bytes":memory.1}),
        })
    }

    fn cache(&self, state: &ResidentState<'_>) -> Result<(Vec<u16>, Vec<u16>)> {
        let count =
            self.config.capacity * self.config.shape.kv_heads * self.config.shape.head_width;
        let read = |suffix: &str| -> Result<Vec<u16>> {
            let mut bytes = vec![0; count * 2];
            state.read_region(
                &format!("{}.attention.{suffix}", self.config.state_prefix),
                &mut bytes,
            )?;
            Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes(*b))
                .collect())
        };
        Ok((read("k")?, read("v")?))
    }
}

fn words(buffer: &Buffer<'_>) -> Result<Vec<u16>> {
    let mut bytes = vec![0; buffer.len()];
    buffer.download(&mut bytes)?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect())
}
fn compare_words(actual: &[u16], expected: &[u16], width: usize) -> Result<Value> {
    let decode = |values: &[u16]| values.iter().map(|&v| bf16_to_f32(v)).collect::<Vec<_>>();
    let mut report = crate::layer_comparison_reference::compare_partitioned(
        &decode(actual),
        &decode(expected),
        width,
    )?;
    report["bf16_differences"] = json!(actual.iter().zip(expected).filter(|(a, b)| a != b).count());
    Ok(report)
}
