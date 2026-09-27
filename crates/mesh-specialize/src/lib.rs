//! Internal specialized-runtime experiments. No serving or Skippy ABI yet.

pub mod artifact;
pub mod checkpoint;
pub mod kernels;
pub mod packages;

#[path = "../reference/embedding_norm.rs"]
pub mod entry_reference;

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
