use super::{NativeMtpParentBinding, checks};
use crate::kernels::cuda::driver::Context;
use crate::packages::qwen3_8_27b::native_mtp_views::{Bf16NormView, BytePlane};
use anyhow::{Context as _, Result, ensure};

/// Checked BF16 gamma view borrowing the verified native-MTP parent arena.
pub(in crate::kernels::cuda) struct NativeMtpNormBinding<'owner, 'ctx> {
    parent: NativeMtpParentBinding<'owner, 'ctx>,
    view: &'owner Bf16NormView,
}

pub(super) fn bind<'owner, 'ctx>(
    parent: NativeMtpParentBinding<'owner, 'ctx>,
    view: &'owner Bf16NormView,
) -> Result<NativeMtpNormBinding<'owner, 'ctx>> {
    validate_parent(view, parent.object_id(), parent.bytes())?;
    parent.plane_pointer(&BytePlane {
        offset: 0,
        bytes: view.bytes,
    })?;
    Ok(NativeMtpNormBinding { parent, view })
}

fn validate_parent(view: &Bf16NormView, object_id: &str, parent_bytes: u64) -> Result<()> {
    ensure!(
        view.object_id == object_id,
        "native MTP norm parent identity mismatch"
    );
    let expected_bytes = u64::try_from(view.elements)?
        .checked_mul(2)
        .context("native MTP norm BF16 extent overflows u64")?;
    ensure!(
        view.bytes == expected_bytes,
        "native MTP norm BF16 extent mismatch"
    );
    ensure!(
        parent_bytes == view.bytes,
        "native MTP norm extent differs from its physical parent"
    );
    checks::parent_range(
        parent_bytes,
        &BytePlane {
            offset: 0,
            bytes: view.bytes,
        },
    )
}

impl NativeMtpNormBinding<'_, '_> {
    pub(in crate::kernels::cuda) fn elements(&self) -> usize {
        self.view.elements
    }

    pub(in crate::kernels::cuda) fn pointer(&self) -> Result<u64> {
        self.parent.plane_pointer(&BytePlane {
            offset: 0,
            bytes: self.view.bytes,
        })
    }

    pub(in crate::kernels::cuda) fn belongs_to(&self, context: &Context) -> bool {
        self.parent.arena.belongs_to(context)
    }
}

#[cfg(test)]
mod tests {
    use super::validate_parent;
    use crate::packages::qwen3_8_27b::native_mtp_views::Bf16NormView;

    fn view() -> Bf16NormView {
        Bf16NormView {
            object_id: "norm-parent".into(),
            elements: 4,
            bytes: 8,
        }
    }

    #[test]
    fn accepts_exact_norm_parent_extent() {
        let saved = view();

        let result = validate_parent(&saved, "norm-parent", 8);

        assert!(result.is_ok());
    }

    #[test]
    fn rejects_norm_with_different_parent_identity() {
        let saved = view();

        let result = validate_parent(&saved, "other-parent", 8);

        assert!(result.is_err());
    }

    #[test]
    fn rejects_parent_extent_mismatch_and_bf16_size_overflow() {
        let saved = view();
        assert!(validate_parent(&saved, "norm-parent", 10).is_err());

        let mut oversized = saved;
        oversized.elements = usize::MAX;

        assert!(validate_parent(&oversized, "norm-parent", u64::MAX).is_err());
    }

    #[test]
    fn rejects_saved_view_byte_length_that_disagrees_with_bf16_elements() {
        let mut saved = view();
        saved.bytes = 6;

        let result = validate_parent(&saved, "norm-parent", 6);

        assert!(result.is_err());
    }
}
