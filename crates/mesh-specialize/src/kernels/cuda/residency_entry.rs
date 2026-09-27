//! Validate the first model operation through views into persistent weights.

use super::{
    driver::{Buffer, Context, Module},
    embedding_norm,
    resident_weights::ResidentWeights,
};
use crate::{
    artifact::schema::{DType, Object},
    kernels::ResidentEntryInput,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(super) fn run(
    context: &Context,
    module: &Module<'_>,
    weights: &ResidentWeights<'_>,
    objects: &[Object],
    input: &ResidentEntryInput,
) -> Result<Value> {
    validate(objects, input)?;
    let token_bytes: Vec<_> = input
        .tokens
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect();
    let tokens = Buffer::new(context, token_bytes.len())?;
    tokens.upload(&token_bytes)?;
    let count = input.width * input.tokens.len();
    let residual = Buffer::new(context, count * 2)?;
    let normalized = Buffer::new(context, count * 2)?;
    let unrounded = Buffer::new(context, count * 4)?;
    residual.upload(&vec![0xa5; count * 2])?;
    normalized.upload(&vec![0xa5; count * 2])?;
    unrounded.upload(&vec![0xff; count * 4])?;
    let function = module.function("embedding_norm_bf16")?;
    let mut pointers = [
        weights.pointer(&input.table_name)?,
        tokens.pointer(),
        weights.pointer(&input.norm_name)?,
        residual.pointer(),
        normalized.pointer(),
        unrounded.pointer(),
    ];
    let mut width = u32::try_from(input.width)?;
    let mut epsilon = input.epsilon;
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast())
        .collect();
    args.push((&mut width as *mut u32).cast());
    args.push((&mut epsilon as *mut f32).cast());
    // SAFETY: The six pointers match embedding_norm_bf16. Inventory metadata and
    // token bounds validate the resident views; all output allocations contain
    // rows * width elements. Owners stay live until synchronization below.
    unsafe {
        function.launch(
            [u32::try_from(input.tokens.len())?, 1, 1],
            [256, 1, 1],
            0,
            &mut args,
        )?;
    }
    context.synchronize()?;
    let allocated = context.memory()?;
    let mut residual_bytes = vec![0; count * 2];
    let mut normalized_bytes = vec![0; count * 2];
    let mut unrounded_bytes = vec![0; count * 4];
    residual.download(&mut residual_bytes)?;
    normalized.download(&mut normalized_bytes)?;
    unrounded.download(&mut unrounded_bytes)?;
    let residual: Vec<_> = residual_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    let normalized: Vec<_> = normalized_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    let unrounded: Vec<_> = unrounded_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let mut report = embedding_norm::compare(
        &input.tokens,
        &residual,
        &normalized,
        &unrounded,
        &input.reference,
    )?;
    report["resources"] = json!(function.resources()?);
    report["resident_weight_views"] = json!(true);
    report["memory_with_entry_outputs"] =
        json!({"free_bytes":allocated.0,"total_bytes":allocated.1});
    Ok(report)
}

fn validate(objects: &[Object], input: &ResidentEntryInput) -> Result<()> {
    ensure!(
        (1..=32768).contains(&input.width),
        "invalid resident entry width"
    );
    ensure!(
        (1..=2048).contains(&input.tokens.len()),
        "invalid resident token count"
    );
    ensure!(
        input.epsilon.is_finite() && input.epsilon > 0.0,
        "invalid entry epsilon"
    );
    let table = objects
        .iter()
        .find(|object| object.name == input.table_name)
        .ok_or_else(|| anyhow::anyhow!("resident embedding table is missing"))?;
    let norm = objects
        .iter()
        .find(|object| object.name == input.norm_name)
        .ok_or_else(|| anyhow::anyhow!("resident entry norm is missing"))?;
    let width = u64::try_from(input.width)?;
    ensure!(
        table.dtype == DType::Bf16 && table.shape.len() == 2 && table.shape[1] == width,
        "resident embedding shape or dtype mismatch"
    );
    let bytes = table.shape[0]
        .checked_mul(width)
        .and_then(|n| n.checked_mul(2));
    ensure!(
        bytes == Some(table.length),
        "resident embedding byte extent mismatch"
    );
    ensure!(
        input
            .tokens
            .iter()
            .all(|&token| u64::from(token) < table.shape[0]),
        "resident token out of bounds"
    );
    ensure!(
        norm.dtype == DType::Bf16 && norm.shape == [width] && norm.length == width * 2,
        "resident norm extent mismatch"
    );
    let count = input.width * input.tokens.len();
    ensure!(
        input.reference.residual.len() == count
            && input.reference.normalized.len() == count
            && input.reference.unrounded.len() == count,
        "resident entry reference extent mismatch"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate;
    use crate::{
        artifact::schema::{DType, Object, ObjectKind},
        entry_reference::EntryReference,
        kernels::ResidentEntryInput,
    };

    fn object(name: &str, dtype: DType, shape: Vec<u64>, length: u64) -> Object {
        Object {
            name: name.to_owned(),
            kind: ObjectKind::Tensor,
            dtype,
            shape,
            layout: "row_major".to_owned(),
            offset: 0,
            length,
            sha256: "0".repeat(64),
        }
    }

    fn objects() -> Vec<Object> {
        vec![
            object("table", DType::Bf16, vec![3, 2], 12),
            object("norm", DType::Bf16, vec![2], 4),
        ]
    }

    fn input() -> ResidentEntryInput {
        ResidentEntryInput {
            table_name: "table".to_owned(),
            norm_name: "norm".to_owned(),
            tokens: vec![0, 2],
            width: 2,
            epsilon: 1e-6,
            reference: EntryReference {
                residual: vec![0; 4],
                normalized: vec![0; 4],
                unrounded: vec![0.0; 4],
            },
        }
    }

    #[test]
    fn accepts_first_and_last_vocabulary_tokens() {
        assert!(validate(&objects(), &input()).is_ok());
    }

    #[test]
    fn rejects_out_of_range_tokens_and_table_metadata_mismatches() {
        let mut out_of_range = input();
        out_of_range.tokens = vec![3, 0];
        assert!(validate(&objects(), &out_of_range).is_err());

        let baseline = objects();
        let mut wrong_dtype = baseline.clone();
        wrong_dtype[0].dtype = DType::F32;
        assert!(validate(&wrong_dtype, &input()).is_err());

        let mut wrong_shape = baseline.clone();
        wrong_shape[0].shape = vec![3, 3];
        assert!(validate(&wrong_shape, &input()).is_err());

        let mut wrong_byte_length = baseline;
        wrong_byte_length[0].length = 10;
        assert!(validate(&wrong_byte_length, &input()).is_err());

        let mut wrong_norm_dtype = objects();
        wrong_norm_dtype[1].dtype = DType::F32;
        assert!(validate(&wrong_norm_dtype, &input()).is_err());

        let mut wrong_norm_shape = objects();
        wrong_norm_shape[1].shape = vec![1, 2];
        assert!(validate(&wrong_norm_shape, &input()).is_err());

        let mut wrong_norm_byte_length = objects();
        wrong_norm_byte_length[1].length = 2;
        assert!(validate(&wrong_norm_byte_length, &input()).is_err());
    }

    #[test]
    fn rejects_overflowing_table_shape_product() {
        let mut invalid_objects = objects();
        invalid_objects[0].shape = vec![u64::MAX, 2];
        invalid_objects[0].length = 12;
        assert!(validate(&invalid_objects, &input()).is_err());
    }

    #[test]
    fn rejects_invalid_width_token_count_and_epsilon() {
        for width in [0, 32769] {
            let mut invalid = input();
            invalid.width = width;
            assert!(validate(&objects(), &invalid).is_err());
        }

        let mut no_tokens = input();
        no_tokens.tokens.clear();
        assert!(validate(&objects(), &no_tokens).is_err());

        let mut too_many_tokens = input();
        too_many_tokens.tokens = vec![0; 2049];
        assert!(validate(&objects(), &too_many_tokens).is_err());

        for epsilon in [0.0, f32::INFINITY, f32::NAN] {
            let mut invalid = input();
            invalid.epsilon = epsilon;
            assert!(validate(&objects(), &invalid).is_err());
        }
    }

    #[test]
    fn rejects_mismatched_reference_extents() {
        let mut invalid = input();
        invalid.reference.normalized.pop();
        assert!(validate(&objects(), &invalid).is_err());
    }
}
