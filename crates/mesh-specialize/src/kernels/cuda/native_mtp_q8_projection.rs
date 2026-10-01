// SPDX-License-Identifier: Apache-2.0
// Derived from NInfer contributors' Q8 sliced-K schedule at
// e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d, src/ops/linear/q8/.
// Modified: resident Rust CUDA-driver projection and qualification helpers.
mod compare;
mod fixture;
pub(in crate::kernels::cuda) mod launch;
mod qualification;

#[cfg(test)]
mod tests;

#[path = "../../../reference/native_mtp_q8_projection.rs"]
pub(super) mod reference;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::kernels) enum DensePattern {
    AlternatingUnit,
    SignedMix,
}

use super::{
    driver::{Buffer, Context, Module},
    resident_native_mtp::ResidentNativeMtp,
};
use crate::packages::qwen3_8_27b::native_mtp_views::Q8MatrixView;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::kernels) enum ProjectionKind {
    QueryKeyValue,
    MlpGateUp,
    AttentionOutput,
    MlpDown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum OutputInitialization {
    Poison(u16),
}

pub(super) struct ResidentProjectionRequest<'a, 'ctx> {
    pub(super) context: &'a Context,
    pub(super) module: &'a Module<'ctx>,
    pub(super) resident: &'a ResidentNativeMtp<'ctx>,
    pub(super) carrier_view: &'a Q8MatrixView,
    pub(super) input: &'a Buffer<'ctx>,
    pub(super) kind: ProjectionKind,
    pub(super) tokens: usize,
    pub(super) initialization: OutputInitialization,
}

pub(in crate::kernels::cuda) struct DeviceProjectionRequest<'a, 'ctx> {
    pub(in crate::kernels::cuda) context: &'a Context,
    pub(in crate::kernels::cuda) module: &'a Module<'ctx>,
    pub(in crate::kernels::cuda) resident: &'a ResidentNativeMtp<'ctx>,
    pub(in crate::kernels::cuda) carrier_view: &'a Q8MatrixView,
    pub(in crate::kernels::cuda) input: &'a Buffer<'ctx>,
    pub(in crate::kernels::cuda) kind: ProjectionKind,
    pub(in crate::kernels::cuda) tokens: usize,
}

pub(in crate::kernels) struct RealParentQualificationRequest<'a> {
    pub(in crate::kernels) artifact: &'a std::path::Path,
    pub(in crate::kernels) ptx: &'a str,
    pub(in crate::kernels) device: i32,
    pub(in crate::kernels) kind: ProjectionKind,
    pub(in crate::kernels) tokens: usize,
    pub(in crate::kernels) pattern: DensePattern,
}

impl ProjectionKind {
    pub(super) const fn dimensions(self) -> [usize; 2] {
        match self {
            Self::QueryKeyValue => [14_336, 5_120],
            Self::MlpGateUp => [34_816, 5_120],
            Self::AttentionOutput => [5_120, 6_144],
            Self::MlpDown => [5_120, 17_408],
        }
    }

    pub(super) const fn entry(self, tokens: usize) -> Option<&'static str> {
        match (self, tokens) {
            (Self::QueryKeyValue, 1) => Some("native_mtp_q8_projection_qkv_c4"),
            (Self::QueryKeyValue, 5) => Some("native_mtp_q8_projection_qkv_c8"),
            (Self::MlpGateUp, 1) => Some("native_mtp_q8_projection_mlp_c4"),
            (Self::MlpGateUp, 5) => Some("native_mtp_q8_projection_mlp_c8"),
            (Self::AttentionOutput, 1) => Some("native_mtp_q8_projection_attention_output_c4"),
            (Self::AttentionOutput, 5) => Some("native_mtp_q8_projection_attention_output_c8"),
            (Self::MlpDown, 1) => Some("native_mtp_q8_projection_mlp_down_c4"),
            (Self::MlpDown, 5) => Some("native_mtp_q8_projection_mlp_down_c8"),
            _ => None,
        }
    }

    pub(super) const fn split_warps(self, tokens: usize) -> Option<usize> {
        match (self, tokens) {
            (Self::QueryKeyValue, 1) => Some(8),
            (Self::QueryKeyValue, 5) => Some(4),
            (Self::MlpGateUp, 1 | 5) => Some(4),
            (Self::AttentionOutput | Self::MlpDown, 1 | 5) => Some(8),
            _ => None,
        }
    }

    pub(super) const fn parent_rows(self) -> usize {
        match self {
            Self::QueryKeyValue => 14_336,
            Self::MlpGateUp => 34_816,
            Self::AttentionOutput | Self::MlpDown => 5_120,
        }
    }

    pub(super) const fn physical_k(self) -> usize {
        self.dimensions()[1]
    }

    fn validate_carrier(
        self,
        resident: &ResidentNativeMtp<'_>,
        view: &Q8MatrixView,
    ) -> anyhow::Result<()> {
        let views = resident.views();
        let valid = match self {
            Self::QueryKeyValue => [&views.query_gate, &views.key, &views.value].contains(&view),
            Self::MlpGateUp => views.mlp_gate == *view || views.mlp_up == *view,
            Self::AttentionOutput => views.attention_output == *view,
            Self::MlpDown => views.mlp_down == *view,
        };
        anyhow::ensure!(
            valid,
            "Q8 projection carrier is not a saved view for this parent"
        );
        Ok(())
    }
}

pub(super) fn project_resident_q8<'a, 'ctx>(
    request: ResidentProjectionRequest<'a, 'ctx>,
) -> anyhow::Result<Buffer<'a>> {
    launch::project(request)
}

pub(in crate::kernels::cuda) fn project_resident_q8_device<'a, 'ctx>(
    request: DeviceProjectionRequest<'a, 'ctx>,
) -> anyhow::Result<Buffer<'a>> {
    launch::project_device(request)
}

pub(in crate::kernels) fn qualify_real_parent(
    request: RealParentQualificationRequest<'_>,
) -> anyhow::Result<serde_json::Value> {
    qualification::run(request)
}
