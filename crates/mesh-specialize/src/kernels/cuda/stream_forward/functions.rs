//! Kernel handles resolved once per executor, never per launch.

use super::super::driver::{Function, Module};
use crate::kernels::{
    fp8_decode_schedule::{self, Decisions, Schedule, Selection},
    fp8_profile::Profile,
};
use anyhow::{Context as _, Result};

/// Every kernel the default exact profile can launch in a whole forward.
pub(super) struct Functions<'m, 'ctx> {
    pub(super) fp8_embedding_gather: Function<'m, 'ctx>,
    pub(super) embedding_norm: Function<'m, 'ctx>,
    pub(super) residual_norm: Function<'m, 'ctx>,
    pub(super) residual_add: Function<'m, 'ctx>,
    pub(super) fp8_quantize: Function<'m, 'ctx>,
    pub(super) fp8_linear_exact: Function<'m, 'ctx>,
    fp8_linear_exact_vector16: Option<Function<'m, 'ctx>>,
    fp8_decode_schedule: Schedule,
    fp8_decode_decisions: Decisions,
    pub(super) fp8_linear_exact4: Function<'m, 'ctx>,
    pub(super) fp8_verify_exact: Function<'m, 'ctx>,
    pub(super) fp8_prefill_exact: Function<'m, 'ctx>,
    pub(super) nvfp4_quantize: Function<'m, 'ctx>,
    pub(super) nvfp4_decode_exact: Function<'m, 'ctx>,
    pub(super) nvfp4_linear: Function<'m, 'ctx>,
    pub(super) bf16_linear_decode: Function<'m, 'ctx>,
    pub(super) causal_conv4: Function<'m, 'ctx>,
    pub(super) gdn_qk_norm: Function<'m, 'ctx>,
    pub(super) gdn_gates_f32_params: Function<'m, 'ctx>,
    pub(super) gdn_gates: Function<'m, 'ctx>,
    pub(super) gdn_recurrent: Function<'m, 'ctx>,
    pub(super) gdn_gated_rms_norm: Function<'m, 'ctx>,
    pub(super) attention_qk_prepare: Function<'m, 'ctx>,
    pub(super) attention_kv_append: Function<'m, 'ctx>,
    pub(super) causal_attention: Function<'m, 'ctx>,
    pub(super) causal_attention_warp: Option<Function<'m, 'ctx>>,
    pub(super) attention_gate: Function<'m, 'ctx>,
    pub(super) mlp_silu_product: Function<'m, 'ctx>,
    pub(super) greedy_tiles: Function<'m, 'ctx>,
    pub(super) greedy_finish: Function<'m, 'ctx>,
}

pub(super) const KERNEL_NAMES: [&str; 26] = [
    "fp8_embedding_gather",
    "embedding_norm_bf16",
    "residual_norm_bf16",
    "residual_add_bf16",
    "fp8_quantize_bf16",
    "fp8_linear_exact",
    "fp8_linear_exact4",
    "fp8_verify_exact",
    "fp8_prefill_exact",
    "nvfp4_quantize_bf16",
    "nvfp4_decode_exact",
    "nvfp4_linear",
    "bf16_linear_decode",
    "causal_conv4_bf16",
    "gdn_qk_norm",
    "gdn_gates",
    "gdn_gates_f32_params",
    "gdn_recurrent",
    "gdn_gated_rms_norm",
    "attention_qk_prepare",
    "attention_kv_append",
    "causal_attention_bf16",
    "attention_gate_bf16",
    "mlp_silu_product",
    "greedy_bf16_tiles",
    "greedy_bf16_finish",
];

impl<'m, 'ctx> Functions<'m, 'ctx> {
    /// StreamForward admits only the exact arithmetic profile. All pointer and
    /// finite-code contracts are inherited from the prevalidated stream bindings.
    pub(super) fn fp8_decode(
        &self,
        rows: usize,
        width: usize,
        pointers: [u64; 2],
    ) -> Result<&Function<'m, 'ctx>> {
        let selection = self
            .fp8_decode_schedule
            .select(Profile::Exact, rows, width, pointers);
        self.fp8_decode_decisions.record(selection);
        if selection == Selection::Vector16 {
            self.fp8_linear_exact_vector16
                .as_ref()
                .context("vector16 handle was not resolved")
        } else {
            Ok(&self.fp8_linear_exact)
        }
    }

    pub(super) fn fp8_decode_report(&self) -> serde_json::Value {
        let mut report = self.fp8_decode_decisions.report(self.fp8_decode_schedule);
        report["vector16_handle_resolved"] = self.fp8_linear_exact_vector16.is_some().into();
        report
    }

    pub(super) fn new(module: &'m Module<'ctx>) -> Result<Self> {
        let get = |name: &str| {
            module
                .function(name)
                .with_context(|| format!("resolve stream forward kernel {name}"))
        };
        let fp8_decode_schedule = fp8_decode_schedule::current()?;
        let fp8_linear_exact_vector16 = (fp8_decode_schedule == Schedule::Vector16)
            .then(|| get(fp8_decode_schedule::VECTOR16_KERNEL))
            .transpose()?;
        let causal_attention_warp = crate::kernels::attention_profile::current()?
            .uses_warp(1)
            .then(|| get(crate::kernels::attention_warp_plan::KERNEL))
            .transpose()?;
        Ok(Self {
            causal_attention_warp,
            fp8_linear_exact_vector16,
            fp8_decode_schedule,
            fp8_decode_decisions: Decisions::default(),
            fp8_embedding_gather: get("fp8_embedding_gather")?,
            embedding_norm: get("embedding_norm_bf16")?,
            residual_norm: get("residual_norm_bf16")?,
            residual_add: get("residual_add_bf16")?,
            fp8_quantize: get("fp8_quantize_bf16")?,
            fp8_linear_exact: get("fp8_linear_exact")?,
            fp8_linear_exact4: get("fp8_linear_exact4")?,
            fp8_verify_exact: get("fp8_verify_exact")?,
            fp8_prefill_exact: get("fp8_prefill_exact")?,
            nvfp4_quantize: get("nvfp4_quantize_bf16")?,
            nvfp4_decode_exact: get("nvfp4_decode_exact")?,
            nvfp4_linear: get("nvfp4_linear")?,
            bf16_linear_decode: get("bf16_linear_decode")?,
            causal_conv4: get("causal_conv4_bf16")?,
            gdn_qk_norm: get("gdn_qk_norm")?,
            gdn_gates_f32_params: get("gdn_gates_f32_params")?,
            gdn_gates: get("gdn_gates")?,
            gdn_recurrent: get("gdn_recurrent")?,
            gdn_gated_rms_norm: get("gdn_gated_rms_norm")?,
            attention_qk_prepare: get("attention_qk_prepare")?,
            attention_kv_append: get("attention_kv_append")?,
            causal_attention: get("causal_attention_bf16")?,
            attention_gate: get("attention_gate_bf16")?,
            mlp_silu_product: get("mlp_silu_product")?,
            greedy_tiles: get("greedy_bf16_tiles")?,
            greedy_finish: get("greedy_bf16_finish")?,
        })
    }
}
