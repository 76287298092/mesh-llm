//! Checked metadata views for native Q8 MTP and Q4 proposal tensors.

#[cfg(test)]
#[path = "../../../reference/native_mtp_quantized.rs"]
mod cpu_reference;
#[cfg(test)]
#[path = "native_mtp_views/head_tests.rs"]
mod head_tests;
mod selection;
#[cfg(test)]
mod test_fixtures;
#[cfg(test)]
mod tests;

use crate::artifact::ninfer::NinferArtifact;
use anyhow::{Context, Result};

/// A byte plane inside one physical NInfer tensor object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytePlane {
    /// Byte offset relative to the physical object's beginning.
    pub offset: u64,
    /// Number of bytes in this complete parent plane.
    pub bytes: u64,
}

/// A selected Q8_g32_FP16 matrix, retaining the parent's packed planes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Q8MatrixView {
    /// Physical NInfer object containing both planes.
    pub object_id: String,
    /// Selected logical `[N, K]` shape.
    pub shape: [usize; 2],
    /// K padded to the row-split layout's 128-column boundary.
    pub padded_k: usize,
    /// Quantization group width, 32 for Q8_g32_FP16.
    pub group_size: usize,
    /// Packed signed-code plane in the parent object.
    pub codes: BytePlane,
    /// Raw FP16 group-scale bits in the parent object.
    pub scale_bits: BytePlane,
    /// Number of FP16 scale values in the group-scale plane.
    pub scale_count: usize,
    /// Parent row for each selected output row, including checked permutations.
    pub source_rows: Vec<usize>,
}

/// A selected Q4_g64_FP16 proposal-head matrix, retaining packed parent planes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Q4MatrixView {
    /// Physical NInfer object containing both planes.
    pub object_id: String,
    /// Selected logical `[N, K]` shape.
    pub shape: [usize; 2],
    /// K padded to the row-split layout's 128-column boundary.
    pub padded_k: usize,
    /// Quantization group width, 64 for Q4_g64_FP16.
    pub group_size: usize,
    /// Packed-code plane in the parent object, even K in the low nibble and odd K in the high.
    pub codes: BytePlane,
    /// Raw FP16 group-scale bits in the parent object.
    pub scale_bits: BytePlane,
    /// Number of FP16 scale values in the group-scale plane.
    pub scale_count: usize,
    /// Parent row for each selected output row.
    pub source_rows: Vec<usize>,
}

/// A contiguous little-endian BF16 norm tensor needed by the native MTP block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bf16NormView {
    /// Physical NInfer object containing the unchanged norm words.
    pub object_id: String,
    /// Number of BF16 elements.
    pub elements: usize,
    /// Exact source byte length.
    pub bytes: u64,
}

/// Native MTP norms separated by consumer role.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeMtpNormViews {
    pub embedding: Bf16NormView,
    pub hidden: Bf16NormView,
    pub final_norm: Bf16NormView,
    pub input: Bf16NormView,
    pub post_attention: Bf16NormView,
    pub query: Bf16NormView,
    pub key: Bf16NormView,
}

impl NativeMtpNormViews {
    pub(super) fn iter(&self) -> impl Iterator<Item = &Bf16NormView> {
        [
            &self.embedding,
            &self.hidden,
            &self.final_norm,
            &self.input,
            &self.post_attention,
            &self.query,
            &self.key,
        ]
        .into_iter()
    }
}

/// A target-vocabulary token ID whose signed source value has been validated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetTokenId(u32);

impl TargetTokenId {
    /// Return the validated target-vocabulary index.
    pub const fn value(self) -> u32 {
        self.0
    }
}

/// The proposal-row to target-vocabulary mapping after signed-ID validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProposalTokenMap {
    target_ids: Vec<TargetTokenId>,
}

impl ProposalTokenMap {
    /// Return the target token for a shortlist row, if the row exists.
    pub fn target_id(&self, proposal_row: usize) -> Option<TargetTokenId> {
        self.target_ids.get(proposal_row).copied()
    }

    /// Return the number of proposal rows in the mapping.
    pub fn len(&self) -> usize {
        self.target_ids.len()
    }

    /// Report whether the mapping contains no proposal rows.
    pub fn is_empty(&self) -> bool {
        self.target_ids.is_empty()
    }
}

/// Checked host views for the native MTP block and shortlist proposal head.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeMtpViews {
    /// Input projection `[5120, 10240]`.
    pub fc: Q8MatrixView,
    /// Interleaved query/gate projection `[12288, 5120]`.
    pub query_gate: Q8MatrixView,
    /// Key projection `[1024, 5120]`.
    pub key: Q8MatrixView,
    /// Value projection `[1024, 5120]`.
    pub value: Q8MatrixView,
    /// Attention output projection `[5120, 6144]`.
    pub attention_output: Q8MatrixView,
    /// MLP gate projection `[17408, 5120]`.
    pub mlp_gate: Q8MatrixView,
    /// MLP up projection `[17408, 5120]`.
    pub mlp_up: Q8MatrixView,
    /// MLP down projection `[5120, 17408]`.
    pub mlp_down: Q8MatrixView,
    /// BF16 norm objects selected by named MTP bindings.
    pub norms: NativeMtpNormViews,
    /// Proposal-only Q4 head `[131072, 5120]`.
    pub proposal_head: Q4MatrixView,
    /// Validated mapping from each proposal row to a target vocabulary ID.
    pub proposal_tokens: ProposalTokenMap,
}

impl NativeMtpViews {
    /// Resolve and validate native MTP/proposal metadata and the signed token map.
    ///
    /// Packed code and FP16 scale bytes remain in the retained NInfer source.
    /// This function does not admit or execute native MTP.
    ///
    /// # Errors
    /// Returns an error if a binding, tensor extent, packed format, or signed token
    /// ID fails the checked native MTP metadata contract.
    pub fn resolve(artifact: &mut NinferArtifact) -> Result<Self> {
        let selected = selection::plan(artifact.directory())?;
        let map_byte_count = selection::token_map_bytes();
        let map_bytes = usize::try_from(map_byte_count)?;
        let mut raw_map = Vec::new();
        raw_map
            .try_reserve_exact(map_bytes)
            .context("cannot reserve native proposal token map")?;
        let copied = artifact.read_object_range(
            &selected.token_map_object,
            0,
            map_byte_count,
            &mut raw_map,
        )?;
        anyhow::ensure!(copied == map_byte_count, "short native token-map read");
        let proposal_tokens = parse_proposal_token_map(&raw_map)?;
        Ok(Self {
            fc: selected.fc,
            query_gate: selected.query_gate,
            key: selected.key,
            value: selected.value,
            attention_output: selected.attention_output,
            mlp_gate: selected.mlp_gate,
            mlp_up: selected.mlp_up,
            mlp_down: selected.mlp_down,
            norms: selected.norms,
            proposal_head: selected.proposal_head,
            proposal_tokens,
        })
    }
}

pub(crate) fn parse_proposal_token_map(bytes: &[u8]) -> anyhow::Result<ProposalTokenMap> {
    selection::parse_token_map(bytes)
}
