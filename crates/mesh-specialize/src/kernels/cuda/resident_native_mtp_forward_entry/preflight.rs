use super::Request;
use crate::{
    artifact::model_source::ModelArtifact,
    kernels::cuda::driver::Module,
    packages::qwen3_8_27b::{decoder, schedule},
};
use anyhow::{Result, ensure};

pub(super) fn validate(request: &Request<'_>) -> Result<()> {
    request.fixture.validate()?;
    ensure!(
        request.fixture.continuation.len() >= 2,
        "shifted native continuation requires two explicit continuation tokens"
    );
    ensure!(request.device >= 0, "device ordinal must be nonnegative");
    ensure!(
        request.ptx.contains(".target sm_120a"),
        "native forward trial requires SM120a PTX"
    );
    ensure!(
        matches!(request.artifact, ModelArtifact::Ninfer(_)),
        "native forward trial requires a verified NInfer source"
    );
    ensure!(
        request.objects == schedule::text_objects(request.artifact.directory())?,
        "native forward trial requires the complete canonical text object schedule"
    );
    let config = request.config;
    ensure!(
        config.capacity == request.fixture.capacity
            && config.hidden == 5_120
            && config.vocabulary == 248_320
            && config.layers.len() == 64,
        "native forward trial config must retain fixture capacity and canonical target geometry"
    );
    let expected = decoder::config(config.capacity)?;
    ensure!(
        config.state_layout == expected.state_layout,
        "target state layout differs from capacity"
    );
    ensure!(
        config.embedding_table == expected.embedding_table
            && config.first_norm == expected.first_norm
            && config.final_norm == expected.final_norm
            && config.head_prefix == expected.head_prefix,
        "target tensor roles differ from the canonical decoder"
    );
    for (layer, canonical) in config.layers.iter().zip(&expected.layers) {
        ensure!(
            layer.prefix == canonical.prefix
                && layer.state_prefix == canonical.state_prefix
                && std::mem::discriminant(&layer.block) == std::mem::discriminant(&canonical.block)
                && std::mem::discriminant(&layer.mlp) == std::mem::discriminant(&canonical.mlp),
            "target layer schedule differs from the canonical decoder"
        );
    }
    let shape = &config.attention_shape;
    ensure!(
        shape.hidden == 5_120
            && shape.intermediate == 17_408
            && shape.query_heads == 24
            && shape.kv_heads == 4
            && shape.head_width == 256
            && shape.rotary_dim == 64
            && shape.rope_theta.is_finite()
            && shape.rope_theta > 0.0,
        "native forward trial attention geometry is invalid"
    );
    let gdn = &config.gdn_shape;
    ensure!(
        gdn.hidden == 5_120
            && gdn.intermediate == 17_408
            && gdn.key_heads == 16
            && gdn.value_heads == 48
            && gdn.head_width == 128,
        "target GDN geometry differs from the canonical decoder"
    );
    crate::kernels::attention_profile::current()?;
    crate::kernels::nvfp4_mlp_schedule::current()?;
    Ok(())
}

pub(super) fn functions(module: &Module<'_>) -> Result<()> {
    for name in [
        "embedding_norm_bf16",
        "fp8_embedding_gather",
        "residual_norm_bf16",
        "residual_add_bf16",
        "attention_qk_prepare",
        "attention_kv_append",
        "native_mtp_q8_sliced_k_fc_c4",
        "native_mtp_q8_sliced_k_fc_c8",
        "native_mtp_q8_projection_qkv_c4",
        "native_mtp_q8_projection_qkv_c8",
        "native_mtp_q8_projection_mlp_c4",
        "native_mtp_q8_projection_mlp_c8",
        "native_mtp_q8_projection_attention_output_c4",
        "native_mtp_q8_projection_attention_output_c8",
        "native_mtp_q8_projection_mlp_down_c4",
        "native_mtp_q8_projection_mlp_down_c8",
        "native_mtp_q4_head_gemv",
        "native_mtp_attention_gate",
        "native_mtp_silu_mul",
        "fp8_quantize_bf16",
        "fp8_linear_exact",
        "fp8_linear_exact4",
        "fp8_prefill_exact",
        "fp8_verify_exact",
        "bf16_linear_decode",
        "causal_conv4_bf16",
        "gdn_qk_norm",
        "gdn_gates_f32_params",
        "gdn_recurrent",
        "gdn_gated_rms_norm",
        "nvfp4_quantize_bf16",
        "nvfp4_linear",
        "nvfp4_decode_exact",
        "mlp_silu_product",
        "attention_gate_bf16",
    ] {
        module.function(name)?;
    }
    let profile = crate::kernels::attention_profile::current()?;
    for rows in 1..=5 {
        module.function(profile.kernel_for_rows(rows))?;
    }
    if profile.uses_split(1) {
        for name in [
            "attention_split_decode_bf16",
            "attention_split_bf16",
            "attention_split_reduce_bf16",
        ] {
            module.function(name)?;
        }
    }
    if profile.uses_staged(1) {
        use crate::kernels::attention_staged_plan::{
            CoefficientSchedule, KERNELS, PREFIX_SCHEDULE_KERNELS,
        };
        let names: &[&str] = match CoefficientSchedule::current()? {
            CoefficientSchedule::SerialV1 => &KERNELS,
            CoefficientSchedule::PrefixParallelV2 => &PREFIX_SCHEDULE_KERNELS,
        };
        for name in names {
            module.function(name)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::packages::qwen3_8_27b::target_batch_trial::Fixture;

    #[test]
    fn fixture_rejects_when_unused_continuation_exceeds_capacity() {
        let fixture = Fixture {
            prefix: vec![10, 11],
            target_tokens: [20, 21, 22, 23, 24],
            continuation: vec![30, 31, 32],
            capacity: 9,
        };
        let result = fixture.validate();
        assert!(result.is_err());
    }
}
