use super::{
    driver::{Buffer, Context},
    resident_weights::BufferRegionSink,
};
use crate::{
    engine::layout::{Layout, Region},
    packages::qwen3_8_27b::{
        native_mtp_views::{Bf16NormView, BytePlane, NativeMtpViews, Q4MatrixView, Q8MatrixView},
        native_source::NativeModelSource,
    },
};
use anyhow::{Context as _, Result, ensure};
use std::collections::BTreeMap;

mod checks;
mod norm_binding;
#[cfg(test)]
mod tests;

pub(in crate::kernels::cuda) use norm_binding::NativeMtpNormBinding;

const MAX_READBACK_BYTES: usize = 1024 * 1024;

pub(super) struct ResidentNativeMtp<'ctx> {
    arena: Buffer<'ctx>,
    layout: Layout,
    hashes: BTreeMap<String, String>,
    views: NativeMtpViews,
}

pub(super) struct NativeMtpParentBinding<'owner, 'ctx> {
    arena: &'owner Buffer<'ctx>,
    region: &'owner Region,
    sha256: &'owner str,
}

pub(super) struct NativeMtpQ8Binding<'owner, 'ctx> {
    parent: NativeMtpParentBinding<'owner, 'ctx>,
    view: &'owner Q8MatrixView,
}

pub(super) struct NativeMtpQ4Binding<'owner, 'ctx> {
    parent: NativeMtpParentBinding<'owner, 'ctx>,
    view: &'owner Q4MatrixView,
}

impl<'ctx> ResidentNativeMtp<'ctx> {
    pub(super) fn load(context: &'ctx Context, source: &mut NativeModelSource) -> Result<Self> {
        let views = source.native_mtp_views()?.clone();
        let parents: Vec<_> = source.native_mtp_parents()?.cloned().collect();
        let layout = Layout::new(
            parents
                .iter()
                .map(|parent| (parent.object_id().to_owned(), parent.bytes())),
        )?;
        let arena = Buffer::new(context, usize::try_from(layout.bytes)?)?;
        let mut hashes = BTreeMap::new();
        for parent in &parents {
            let region = layout.region(parent.object_id())?;
            let mut sink = BufferRegionSink::new(
                &arena,
                usize::try_from(region.offset)?,
                usize::try_from(region.length)?,
            );
            let copied = source.copy_native_mtp_parent(parent.object_id(), &mut sink)?;
            ensure!(
                copied == region.length,
                "short resident native MTP parent copy"
            );
            sink.finish(parent.object_id())?;
            hashes.insert(parent.object_id().to_owned(), parent.sha256().to_owned());
        }
        Ok(Self {
            arena,
            layout,
            hashes,
            views,
        })
    }

    pub(super) fn layout(&self) -> &Layout {
        &self.layout
    }

    pub(super) fn views(&self) -> &NativeMtpViews {
        &self.views
    }

    pub(super) fn context(&self) -> &'ctx Context {
        self.arena.context()
    }

    pub(super) fn belongs_to(&self, context: &Context) -> bool {
        self.arena.belongs_to(context)
    }

    pub(super) fn parent(&self, object_id: &str) -> Result<NativeMtpParentBinding<'_, 'ctx>> {
        Ok(NativeMtpParentBinding {
            arena: &self.arena,
            region: self.layout.region(object_id)?,
            sha256: self
                .hashes
                .get(object_id)
                .context("native MTP parent hash missing")?,
        })
    }

    pub(super) fn q8(&self, view: &Q8MatrixView) -> Result<NativeMtpQ8Binding<'_, 'ctx>> {
        let saved = checks::saved_view(
            [
                &self.views.fc,
                &self.views.query_gate,
                &self.views.key,
                &self.views.value,
                &self.views.attention_output,
                &self.views.mlp_gate,
                &self.views.mlp_up,
                &self.views.mlp_down,
            ],
            view,
        )?;
        let parent = self.parent(&saved.object_id)?;
        parent.plane_pointer(&saved.codes)?;
        parent.plane_pointer(&saved.scale_bits)?;
        Ok(NativeMtpQ8Binding {
            parent,
            view: saved,
        })
    }

    pub(super) fn q4(&self, view: &Q4MatrixView) -> Result<NativeMtpQ4Binding<'_, 'ctx>> {
        let saved = checks::saved_view([&self.views.proposal_head], view)?;
        let parent = self.parent(&saved.object_id)?;
        parent.plane_pointer(&saved.codes)?;
        parent.plane_pointer(&saved.scale_bits)?;
        Ok(NativeMtpQ4Binding {
            parent,
            view: saved,
        })
    }

    /// Bind a saved BF16 norm view. The result borrows this resident owner.
    /// Equal clones resolve to the saved view; changed metadata is rejected.
    pub(super) fn norm(&self, view: &Bf16NormView) -> Result<NativeMtpNormBinding<'_, 'ctx>> {
        let saved = checks::saved_view(
            [
                &self.views.norms.embedding,
                &self.views.norms.hidden,
                &self.views.norms.final_norm,
                &self.views.norms.input,
                &self.views.norms.post_attention,
                &self.views.norms.query,
                &self.views.norms.key,
            ],
            view,
        )?;
        norm_binding::bind(self.parent(&saved.object_id)?, saved)
    }

    pub(super) fn embedding_norm(&self) -> Result<NativeMtpNormBinding<'_, 'ctx>> {
        self.norm(&self.views.norms.embedding)
    }
}

impl NativeMtpParentBinding<'_, '_> {
    pub(super) fn object_id(&self) -> &str {
        &self.region.name
    }

    pub(super) fn bytes(&self) -> u64 {
        self.region.length
    }

    pub(super) const fn sha256(&self) -> &str {
        self.sha256
    }

    pub(super) fn pointer(&self) -> Result<u64> {
        checks::device_pointer(self.arena.pointer(), self.region.offset)
    }

    fn plane_pointer(&self, plane: &BytePlane) -> Result<u64> {
        checks::parent_range(self.region.length, plane)?;
        checks::device_pointer(self.pointer()?, plane.offset)
    }

    pub(super) fn read_range(&self, offset: u64, destination: &mut [u8]) -> Result<()> {
        ensure!(
            destination.len() <= MAX_READBACK_BYTES,
            "native MTP readback exceeds chunk limit"
        );
        let plane = BytePlane {
            offset,
            bytes: u64::try_from(destination.len())?,
        };
        checks::parent_range(self.region.length, &plane)?;
        let arena_offset = self
            .region
            .offset
            .checked_add(offset)
            .context("native MTP readback arena offset overflow")?;
        self.arena
            .download_at(usize::try_from(arena_offset)?, destination)
    }
}

impl<'owner, 'ctx> NativeMtpQ8Binding<'owner, 'ctx> {
    pub(super) const fn view(&self) -> &'owner Q8MatrixView {
        self.view
    }

    pub(super) const fn parent(&self) -> &NativeMtpParentBinding<'owner, 'ctx> {
        &self.parent
    }

    pub(super) fn codes_pointer(&self) -> Result<u64> {
        self.parent.plane_pointer(&self.view.codes)
    }

    pub(super) fn scale_bits_pointer(&self) -> Result<u64> {
        self.parent.plane_pointer(&self.view.scale_bits)
    }
}

impl<'owner, 'ctx> NativeMtpQ4Binding<'owner, 'ctx> {
    pub(super) const fn view(&self) -> &'owner Q4MatrixView {
        self.view
    }

    pub(super) const fn parent(&self) -> &NativeMtpParentBinding<'owner, 'ctx> {
        &self.parent
    }

    pub(super) fn codes_pointer(&self) -> Result<u64> {
        self.parent.plane_pointer(&self.view.codes)
    }

    pub(super) fn scale_bits_pointer(&self) -> Result<u64> {
        self.parent.plane_pointer(&self.view.scale_bits)
    }
}
