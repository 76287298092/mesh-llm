//! Full-attention input-stage trial, separate from full model execution.
use super::{
    attention_core, attention_prepare,
    driver::{Buffer, Context, Module},
    embedding_norm, projections,
};
use crate::{
    attention_prepare_reference as reference, entry_reference,
    kernels::{AttentionInput, EmbeddingNormInput},
    projection_reference,
};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};

pub(in crate::kernels) fn run(ptx: &str, device: i32, input: &AttentionInput) -> Result<Value> {
    ensure!(
        ptx.contains(".target sm_120a"),
        "attention PTX must target SM120a"
    );
    validate(input)?;
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "attention trial requires SM120"
    );
    let before = context.memory()?;
    let module = Module::load(&context, ptx)?;
    let fixtures = attention_prepare::fixtures(&context, &module)?;
    let core_fixtures = attention_core::fixtures(&context, &module)?;
    let table = upload(&context, &input.entry.table)?;
    let norm = upload(&context, &input.entry.weight)?;
    let mut cases = Vec::new();
    for (tokens, positions) in input.entry.batches.iter().zip(&input.positions) {
        cases.push(run_case(
            &context, &module, input, tokens, positions, &table, &norm,
        )?);
    }
    drop(table);
    drop(norm);
    let after = context.memory()?;
    Ok(
        json!({"schema_version":2,"kind":"qwen-causal-attention-trial","all_passed":fixtures.iter().chain(&core_fixtures).chain(&cases).all(|v|v["all_passed"]==true),
        "device":info,"cases":cases,"fixtures":fixtures,"causal_attention_fixtures":core_fixtures,"causal_attention_resources":module.function("causal_attention_bf16")?.resources()?,"kv_append_resources":module.function("attention_kv_append")?.resources()?,"prepare_resources":module.function("attention_qk_prepare")?.resources()?,
        "memory_before":{"free_bytes":before.0,"total_bytes":before.1},"memory_after":{"free_bytes":after.0,"total_bytes":after.1},
        "causal_attention_core_executed":true,"full_attention_executed":false,"full_model_executed":false,"timing_collected":false,
        "input_scope":"embedding rows used as synthetic hidden input to layer 3; layers 0..2 are not executed",
        "rope_scope":"text positions with equal T/H/W; explicit CPU-generated BF16 coefficient profile, not full multimodal RoPE or tested usable context"}),
    )
}

fn validate(input: &AttentionInput) -> Result<()> {
    ensure!(
        (1..=16).contains(&input.entry.batches.len())
            && input.positions.len() == input.entry.batches.len(),
        "attention batch count invalid"
    );
    ensure!(
        (1..=128).contains(&input.query_heads)
            && (1..=128).contains(&input.kv_heads)
            && input.query_heads.is_multiple_of(input.kv_heads),
        "attention head shape invalid"
    );
    ensure!(
        (2..=1024).contains(&input.head_width)
            && input.rotary_dim >= 2
            && input.rotary_dim <= input.head_width
            && input.rotary_dim.is_multiple_of(2),
        "attention head/rotary width invalid"
    );
    ensure!(
        input.q_norm.len() == input.head_width * 2 && input.k_norm.len() == input.head_width * 2,
        "attention norm extent invalid"
    );
    ensure!(
        decode(&input.q_norm)
            .iter()
            .chain(&decode(&input.k_norm))
            .all(|&v| entry_reference::bf16_to_f32(v).is_finite()),
        "attention norm nonfinite"
    );
    for (tokens, positions) in input.entry.batches.iter().zip(&input.positions) {
        ensure!(
            tokens.len() == positions.len(),
            "attention position count mismatch"
        );
        entry_reference::embedding_norm(
            &input.entry.table,
            tokens,
            &input.entry.weight,
            input.entry.width,
            input.entry.epsilon,
        )?;
        reference::text_rope_tables(positions, input.rotary_dim, input.rope_theta)?;
    }
    let widths = [
        input.query_heads * input.head_width * 2,
        input.kv_heads * input.head_width,
        input.kv_heads * input.head_width,
    ];
    for (p, channels) in input.projections.iter().zip(widths) {
        ensure!(
            p.channels == channels
                && p.weights.len() == channels * input.entry.width
                && p.scales.len() == channels * 2,
            "attention projection extent invalid"
        );
        ensure!(
            p.weights.iter().all(|v| v & 127 != 127),
            "attention FP8 weight nonfinite"
        );
        ensure!(
            decode(&p.scales).iter().all(|&v| {
                let x = entry_reference::bf16_to_f32(v);
                x.is_finite() && x > 0.0
            }),
            "attention FP8 scale invalid"
        );
    }
    Ok(())
}

struct Entry<'a> {
    codes: Buffer<'a>,
    scales: Buffer<'a>,
    quantized: projection_reference::QuantizedRows,
    report: Value,
}

fn entry<'a>(
    context: &'a Context,
    module: &Module<'_>,
    input: &EmbeddingNormInput,
    tokens: &[u32],
    table: &Buffer<'_>,
    norm: &Buffer<'_>,
) -> Result<Entry<'a>> {
    let expected = entry_reference::embedding_norm(
        &input.table,
        tokens,
        &input.weight,
        input.width,
        input.epsilon,
    )?;
    let ids = upload(
        context,
        &tokens
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let count = tokens.len() * input.width;
    let residual = upload(context, &vec![0xa5; count * 2])?;
    let normalized = upload(context, &vec![0xa5; count * 2])?;
    let unrounded = upload(context, &vec![0xff; count * 4])?;
    embedding_norm::launch(
        &module.function("embedding_norm_bf16")?,
        &[table, &ids, norm, &residual, &normalized, &unrounded],
        tokens.len(),
        input.width,
        input.epsilon,
    )?;
    context.synchronize()?;
    let norm_words = words(&normalized, count)?;
    let report = embedding_norm::compare(
        tokens,
        &words(&residual, count)?,
        &norm_words,
        &floats(&unrounded, count)?,
        &expected,
    )?;
    let quantized = projection_reference::quantize(&norm_words, tokens.len(), input.width)?;
    let codes = upload(context, &vec![127; count])?;
    let scales = upload(context, &vec![0xff; tokens.len() * 4])?;
    projections::quantize(
        &module.function("fp8_quantize_bf16")?,
        &normalized,
        &codes,
        &scales,
        tokens.len(),
        input.width,
    )?;
    context.synchronize()?;
    let mut actual_codes = vec![0; count];
    codes.download(&mut actual_codes)?;
    ensure!(
        actual_codes == quantized.codes && floats(&scales, tokens.len())? == quantized.scales,
        "attention input FP8 quantization mismatch"
    );
    Ok(Entry {
        codes,
        scales,
        quantized,
        report,
    })
}

fn run_case(
    context: &Context,
    module: &Module<'_>,
    input: &AttentionInput,
    tokens: &[u32],
    positions: &[u32],
    table: &Buffer<'_>,
    norm: &Buffer<'_>,
) -> Result<Value> {
    let entry = entry(context, module, &input.entry, tokens, table, norm)?;
    let (cos, sin) = reference::text_rope_tables(positions, input.rotary_dim, input.rope_theta)?;
    let mut reports = Vec::new();
    let mut q = None;
    let mut k = None;
    let mut v = None;
    for (index, p) in input.projections.iter().enumerate() {
        let weights = upload(context, &p.weights)?;
        let scales = upload(context, &p.scales)?;
        let result = projections::run_linear(
            context,
            &module.function("fp8_linear")?,
            &[&entry.codes, &weights, &entry.scales, &scales],
            [tokens.len(), p.channels, input.entry.width],
        )?;
        let (words, unrounded) = result.read(tokens.len() * p.channels)?;
        let expected = projection_reference::linear(
            &entry.quantized,
            &p.weights,
            &decode(&p.scales),
            input.entry.width,
        )?;
        let projection = projections::compare(&words, &unrounded, &expected)?;
        let preparation = if index < 2 {
            let shape = reference::Shape {
                rows: tokens.len(),
                heads: if index == 0 {
                    input.query_heads
                } else {
                    input.kv_heads
                },
                width: input.head_width,
                rotary_dim: input.rotary_dim,
                with_gate: index == 0,
            };
            let checked = attention_prepare::check(
                context,
                module,
                attention_prepare::Input {
                    device: &result.bf16,
                    words: &words,
                    weight: &decode(if index == 0 {
                        &input.q_norm
                    } else {
                        &input.k_norm
                    }),
                    cos: &cos,
                    sin: &sin,
                    shape: &shape,
                },
            )?;
            let value = (checked.device, checked.words);
            if index == 0 {
                q = Some(value);
            } else {
                k = Some(value);
            }
            Some(checked.report)
        } else {
            v = Some((result.bf16, words));
            None
        };
        let passed = projection["passed"] == true
            && preparation.as_ref().is_none_or(|v| v["all_passed"] == true);
        reports.push(json!({"projection_name":p.name,"all_passed":passed,"projection":projection,"preparation":preparation,"shape_mnk":[tokens.len(),p.channels,input.entry.width]}));
    }
    let (q, q_words) = q.context("missing prepared Q")?;
    let (k, k_words) = k.context("missing prepared K")?;
    let (v, v_words) = v.context("missing projected V")?;
    let core = attention_core::check(
        context,
        module,
        attention_core::Input {
            q: &q,
            k: &k,
            v: &v,
            q_words: &q_words,
            k_words: &k_words,
            v_words: &v_words,
            rows: tokens.len(),
            query_heads: input.query_heads,
            kv_heads: input.kv_heads,
            width: input.head_width,
        },
    )?;
    Ok(
        json!({"all_passed":entry.report["passed"]==true && reports.iter().all(|r|r["all_passed"]==true) && core["all_passed"]==true,"tokens":tokens,"positions":positions,"entry":entry.report,"input_activation_quantization_exact":true,"projections":reports,"causal_attention":core}),
    )
}

fn upload<'a>(context: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}
fn decode(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect()
}
fn words(buffer: &Buffer<'_>, count: usize) -> Result<Vec<u16>> {
    let mut bytes = vec![0; count * 2];
    buffer.download(&mut bytes)?;
    Ok(decode(&bytes))
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
