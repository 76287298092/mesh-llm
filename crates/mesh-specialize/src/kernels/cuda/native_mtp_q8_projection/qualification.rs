use super::super::{
    driver::{Buffer, Context, FunctionResources, Module},
    resident_native_mtp::ResidentNativeMtp,
};
use super::{
    DensePattern, ProjectionKind, RealParentQualificationRequest, ResidentProjectionRequest,
    compare, fixture, project_resident_q8, reference,
};
use crate::packages::qwen3_8_27b::{
    native_mtp_views::{NativeMtpViews, Q8MatrixView},
    native_source::NativeModelSource,
};
use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const RESIDENT_HASH_CHUNK_BYTES: usize = 1024 * 1024;

struct Executor<'a, 'ctx> {
    context: &'a Context,
    module: &'a Module<'ctx>,
    resident: &'a ResidentNativeMtp<'ctx>,
    carrier: &'a Q8MatrixView,
    object: &'a [u8],
    kind: ProjectionKind,
}

struct QualificationEvidence<'e, 'a, 'ctx> {
    expected: &'e reference::ProjectionReference,
    outputs: &'e [Vec<u16>; 2],
    rows: usize,
    k: usize,
    tokens: usize,
    split_warps: usize,
    input_columns_distinct: bool,
    output_columns_distinct: bool,
    resources: Option<FunctionResources>,
    parent_bytes: usize,
    source_hash: &'e str,
    resident_hash: &'e str,
    source_copy_hash: &'e str,
    executor: &'e Executor<'a, 'ctx>,
    pattern: DensePattern,
}

pub(super) fn run(request: RealParentQualificationRequest<'_>) -> Result<Value> {
    ensure!(
        matches!(request.tokens, 1 | 5),
        "native MTP Q8 qualification requires T1 or T5"
    );
    let mut source = NativeModelSource::open(request.artifact)?;
    let views = source.native_mtp_views()?.clone();
    let carrier = source_parent_view(&views, request.kind).clone();
    let source_hash = source_parent_hash(&mut source, &carrier.object_id)?;
    let context = Context::new(request.device)?;
    let module = Module::load(&context, request.ptx)?;
    let resident = ResidentNativeMtp::load(&context, &mut source)?;
    ensure!(
        resident.belongs_to(&context),
        "resident MTP parent has the wrong CUDA device"
    );
    let resident_carrier = resident_view(resident.views(), request.kind)?;
    ensure!(
        resident_carrier == &carrier,
        "resident saved view differs from source view"
    );
    let binding = resident.q8(resident_carrier)?;
    ensure!(
        binding.parent().object_id() == carrier.object_id,
        "projection parent object identity changed"
    );
    ensure!(
        binding.parent().sha256() == source_hash,
        "resident Q8 parent hash differs from verified source"
    );
    let parent_bytes = parent_extent(&carrier)?;
    ensure!(
        binding.parent().bytes() == u64::try_from(parent_bytes)?,
        "resident physical parent byte extent mismatch"
    );
    ensure!(
        binding.codes_pointer()? % 16 == 0 && binding.scale_bits_pointer()? % 16 == 0,
        "resident packed Q8 planes are not aligned"
    );
    let resident_hash = hash_resident_parent(binding.parent())?;
    ensure!(
        resident_hash == source_hash,
        "resident parent readback hash differs from source"
    );
    let mut object = Vec::new();
    object
        .try_reserve_exact(parent_bytes)
        .context("Q8 parent oracle allocation failed")?;
    let copied = source.copy_native_mtp_parent(&carrier.object_id, &mut object)?;
    ensure!(
        copied == u64::try_from(parent_bytes)?,
        "short packed Q8 physical parent copy"
    );
    let actual_hash = hex::encode(Sha256::digest(&object));
    ensure!(
        actual_hash == source_hash,
        "Q8 physical parent source hash mismatch"
    );
    let executor = Executor {
        context: &context,
        module: &module,
        resident: &resident,
        carrier: resident_carrier,
        object: &object,
        kind: request.kind,
    };
    let [_, k] = request.kind.dimensions();
    let rows = request.kind.parent_rows();
    let input = request.pattern.input(k, request.tokens)?;
    ensure!(
        input
            .iter()
            .all(|&word| fixture::input_value(word).abs() <= 2.0),
        "Q8 activation fixture exceeded the BF16 half-unit bound"
    );
    let identity = fixture::identity_view(executor.carrier, request.kind)?;
    let split_warps = request
        .kind
        .split_warps(request.tokens)
        .context("unsupported native MTP Q8 projection split count")?;
    let expected = reference::run(reference::ProjectionReferenceRequest {
        parent: executor.object,
        view: &identity,
        input_bf16: &input,
        split_warps,
    })?;
    let input_columns_distinct = columns_are_distinct(&input, k);
    ensure!(
        request.tokens != 5 || input_columns_distinct,
        "T5 activation columns are not pairwise distinct"
    );
    let input_buffer = device_input(executor.context, &input)?;
    let mut outputs = [Vec::new(), Vec::new()];
    let resources = Some(
        executor
            .module
            .function(
                request
                    .kind
                    .entry(request.tokens)
                    .context("unsupported Q8 projection entry")?,
            )?
            .resources()?,
    );
    for (repeat, output) in outputs.iter_mut().enumerate() {
        let output_buffer = project_resident_q8(ResidentProjectionRequest {
            context: executor.context,
            module: executor.module,
            resident: executor.resident,
            carrier_view: executor.carrier,
            input: &input_buffer,
            kind: executor.kind,
            tokens: request.tokens,
            initialization: DensePattern::initialization(repeat),
        })?;
        let mut bytes = vec![0_u8; rows * request.tokens * 2];
        output_buffer.download(&mut bytes)?;
        output.extend(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|word| u16::from_le_bytes(*word)),
        );
    }
    let output_columns_distinct =
        request.tokens != 5 || columns_are_distinct(&expected.output_bf16, rows);
    qualification_report(QualificationEvidence {
        expected: &expected,
        outputs: &outputs,
        rows,
        k,
        tokens: request.tokens,
        split_warps,
        input_columns_distinct,
        output_columns_distinct,
        resources,
        parent_bytes,
        source_hash: &source_hash,
        resident_hash: &resident_hash,
        source_copy_hash: &actual_hash,
        executor: &executor,
        pattern: request.pattern,
    })
}

fn source_parent_hash(source: &mut NativeModelSource, object_id: &str) -> Result<String> {
    Ok(source
        .native_mtp_parents()?
        .find(|parent| parent.object_id() == object_id)
        .context("Q8 physical parent is missing from the verified source")?
        .sha256()
        .to_owned())
}

fn qualification_report(evidence: QualificationEvidence<'_, '_, '_>) -> Result<Value> {
    let report = compare::compare(
        evidence.expected,
        evidence.outputs,
        evidence.rows,
        evidence.tokens,
    )?;
    let all_passed = report["all_passed"] == true
        && (evidence.tokens != 5 || evidence.input_columns_distinct)
        && (evidence.tokens != 5 || evidence.output_columns_distinct);
    Ok(json!({
        "schema_version": 1,
        "kind": "native-mtp-q8-resident-projection-qualification-v1",
        "all_passed": all_passed,
        "device": evidence.executor.context.info(),
        "jit_log": evidence.executor.module.jit_log(),
        "projection": format!("{:?}", evidence.executor.kind),
        "shape": [evidence.rows, evidence.k],
        "tokens": evidence.tokens,
        "split_warps": evidence.split_warps,
        "input_pattern": evidence.pattern.name(),
        "physical_parent": {
            "object_id": evidence.executor.carrier.object_id,
            "bytes": evidence.parent_bytes,
            "expected_sha256": evidence.source_hash,
            "source_copy_sha256": evidence.source_copy_hash,
            "resident_readback_sha256": evidence.resident_hash,
            "source_hash_matches": evidence.source_copy_hash == evidence.source_hash,
            "resident_hash_matches": evidence.resident_hash == evidence.source_hash,
            "saved_binding_identity": true,
            "resident_device_identity": true,
            "weight_reupload": false,
            "dequantization": false,
            "query_gate_logical_rows_used": false,
        },
        "kernel_resources": evidence.resources,
        "output_columns_pairwise_distinct": evidence.output_columns_distinct,
        "input_columns_pairwise_distinct": evidence.input_columns_distinct,
        "five_distinct_columns": (evidence.tokens == 5).then_some(evidence.output_columns_distinct),
        "five_distinct_input_columns": (evidence.tokens == 5).then_some(evidence.input_columns_distinct),
        "reference": {
            "f64_diagnostic_only": true,
            "max_scheduled_f32_f64_delta": max_f64_delta(evidence.expected),
            "max_mathematical_error_bound": max_bound(evidence.expected),
        },
        "result": report,
        "native_mtp_admitted": false,
        "model_executable": false,
        "timing_claim": false,
    }))
}

fn source_parent_view(views: &NativeMtpViews, kind: ProjectionKind) -> &Q8MatrixView {
    match kind {
        ProjectionKind::QueryKeyValue => &views.query_gate,
        ProjectionKind::MlpGateUp => &views.mlp_gate,
        ProjectionKind::AttentionOutput => &views.attention_output,
        ProjectionKind::MlpDown => &views.mlp_down,
    }
}

fn resident_view(views: &NativeMtpViews, kind: ProjectionKind) -> Result<&Q8MatrixView> {
    Ok(match kind {
        ProjectionKind::QueryKeyValue => &views.query_gate,
        ProjectionKind::MlpGateUp => &views.mlp_gate,
        ProjectionKind::AttentionOutput => &views.attention_output,
        ProjectionKind::MlpDown => &views.mlp_down,
    })
}

fn parent_extent(view: &Q8MatrixView) -> Result<usize> {
    usize::try_from(
        view.scale_bits
            .offset
            .checked_add(view.scale_bits.bytes)
            .context("Q8 parent extent overflow")?,
    )
    .context("Q8 parent extent exceeds usize")
}

fn device_input<'ctx>(context: &'ctx Context, words: &[u16]) -> Result<Buffer<'ctx>> {
    let bytes = words
        .len()
        .checked_mul(2)
        .context("Q8 input bytes overflow")?;
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(bytes)
        .context("Q8 input allocation failed")?;
    for word in words {
        encoded.extend_from_slice(&word.to_le_bytes());
    }
    let input = Buffer::new(context, bytes)?;
    input.upload(&encoded)?;
    Ok(input)
}

fn columns_are_distinct(words: &[u16], rows: usize) -> bool {
    if rows == 0 || !words.len().is_multiple_of(rows) {
        return false;
    }
    words.chunks_exact(rows).enumerate().all(|(index, column)| {
        words
            .chunks_exact(rows)
            .skip(index + 1)
            .all(|other| column != other)
    })
}

fn max_f64_delta(reference: &reference::ProjectionReference) -> f64 {
    reference
        .scheduled_f32
        .iter()
        .zip(&reference.mathematical_f64)
        .map(|(&scheduled, &mathematical)| (f64::from(scheduled) - mathematical).abs())
        .fold(0.0_f64, f64::max)
}

fn max_bound(reference: &reference::ProjectionReference) -> f64 {
    reference
        .mathematical_error_bound
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
}

fn hash_resident_parent(
    parent: &super::super::resident_native_mtp::NativeMtpParentBinding<'_, '_>,
) -> Result<String> {
    let scratch_bytes = usize::try_from(parent.bytes())?.min(RESIDENT_HASH_CHUNK_BYTES);
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(scratch_bytes)
        .context("resident Q8 hash scratch allocation failed")?;
    scratch.resize(scratch_bytes, 0);
    let mut digest = Sha256::new();
    let mut offset = 0_u64;
    while offset < parent.bytes() {
        let length = usize::try_from((parent.bytes() - offset).min(u64::try_from(scratch.len())?))?;
        let chunk = &mut scratch[..length];
        parent.read_range(offset, chunk)?;
        digest.update(&*chunk);
        offset = offset
            .checked_add(u64::try_from(length)?)
            .context("resident Q8 hash offset overflow")?;
    }
    Ok(hex::encode(digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::columns_are_distinct;

    #[test]
    fn all_five_columns_when_identical_values_are_repeated_returns_false() {
        assert!(!columns_are_distinct(&[1, 2, 1, 2, 1, 2, 1, 2, 1, 2], 2));
    }

    #[test]
    fn all_five_columns_when_values_differ_returns_true() {
        assert!(columns_are_distinct(&[1, 2, 2, 3, 3, 4, 4, 5, 5, 6], 2));
    }
}
