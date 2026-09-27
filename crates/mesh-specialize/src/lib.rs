//! Internal specialized-runtime experiments. No serving or Skippy ABI yet.

pub mod artifact;
pub mod checkpoint;
pub mod engine;
pub mod kernels;
pub mod packages;

#[path = "../reference/decoder_attention.rs"]
pub mod decoder_attention_reference;
#[path = "../reference/decoder_gdn.rs"]
pub mod decoder_gdn_reference;
#[path = "../reference/decoder_mlp.rs"]
pub mod decoder_mlp_reference;
#[path = "../reference/decoder_ops.rs"]
pub mod decoder_ops_reference;

#[path = "../reference/embedding_norm.rs"]
pub mod entry_reference;

#[path = "../reference/resident_entry.rs"]
pub mod resident_entry_reference;

#[path = "../reference/fp8_mlp.rs"]
pub mod fp8_mlp_reference;

#[path = "../reference/causal_conv4.rs"]
pub mod causal_conv4_reference;
#[path = "../reference/gated_rms_norm.rs"]
pub mod gated_norm_reference;
#[path = "../reference/gdn_prepare.rs"]
pub mod gdn_prepare_reference;
#[path = "../reference/gdn_recurrent.rs"]
pub mod gdn_recurrent_reference;
#[path = "../reference/projections.rs"]
pub mod projection_reference;
#[path = "../reference/residual_norm.rs"]
pub mod residual_norm_reference;

#[path = "../reference/arithmetic.rs"]
pub mod reference;

#[path = "../reference/nvfp4_quantize.rs"]
pub mod nvfp4_quantize_reference;

#[path = "../reference/nvfp4_linear.rs"]
pub mod nvfp4_linear_reference;

#[path = "../reference/mlp_activation.rs"]
pub mod mlp_activation_reference;

#[path = "../reference/residual_add.rs"]
pub mod residual_add_reference;

#[path = "../reference/layer_comparison.rs"]
pub mod layer_comparison_reference;
#[path = "../reference/qwen_gdn_layer.rs"]
pub mod qwen_gdn_layer_reference;

#[path = "../reference/attention_prepare.rs"]
pub mod attention_prepare_reference;

#[path = "../reference/causal_attention.rs"]
pub mod causal_attention_reference;

#[path = "../reference/attention_gate.rs"]
pub mod attention_gate_reference;
#[path = "../reference/qwen_attention_layer.rs"]
pub mod qwen_attention_layer_reference;

#[path = "../reference/mtp.rs"]
pub mod mtp_reference;

#[path = "../reference/fp8_a16_decode.rs"]
pub mod fp8_a16_decode_reference;

#[path = "../reference/fp8_native_prefill.rs"]
pub mod fp8_native_prefill_reference;

#[path = "../reference/fp8_swiglu_exact.rs"]
pub mod fp8_swiglu_exact_reference;

#[path = "../reference/gdn_chunked.rs"]
pub mod gdn_chunked_reference;

#[path = "../reference/attention_online.rs"]
pub mod attention_online_reference;

#[path = "../reference/kv_fp8.rs"]
pub mod kv_fp8_reference;

#[path = "../reference/gdn_replay.rs"]
pub mod gdn_replay_reference;

#[path = "../reference/greedy_bf16.rs"]
pub mod greedy_bf16_reference;
