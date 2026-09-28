//! Resident encoded embedding lookup with the existing BF16 normalization boundary.

use super::{
    driver::{Buffer, Context, Module},
    resident_norm::Normalized,
    resident_weights::ResidentWeights,
};
use crate::artifact::schema::DType;
use anyhow::{Context as _, Result, ensure};
use std::ffi::c_void;

pub(super) struct Embedding<'w, 'ctx> {
    owner: &'w ResidentWeights<'ctx>,
    table: u64,
    scale: Option<u64>,
    norm: u64,
    vocabulary: usize,
    width: usize,
    epsilon: f32,
}

impl<'w, 'ctx> Embedding<'w, 'ctx> {
    pub(super) fn new(
        owner: &'w ResidentWeights<'ctx>,
        table_name: &str,
        norm_name: &str,
        shape: [usize; 2],
        epsilon: f32,
    ) -> Result<Self> {
        let [vocabulary, width] = shape;
        let (_, norm_bytes) = validate_shape(vocabulary, width, epsilon)?;
        let width_u64 = u64::try_from(width).context("embedding width does not fit u64")?;
        let (table, scale) = bind_table(owner, table_name, vocabulary, width)?;
        let norm = owner.tensor(norm_name, DType::Bf16, &[width_u64], norm_bytes)?;
        Ok(Self {
            owner,
            table,
            scale,
            norm,
            vocabulary,
            width,
            epsilon,
        })
    }

    /// Lookup token rows and run the resident fused BF16 embedding/norm kernel.
    pub(super) fn run<'a>(
        &self,
        context: &'a Context,
        module: &Module<'_>,
        tokens: &[u32],
    ) -> Result<Normalized<'a>> {
        ensure!(
            self.owner.belongs_to(context),
            "resident embedding belongs to another context"
        );
        ensure!(
            module.belongs_to(context),
            "resident embedding module belongs to another context"
        );
        validate_tokens(self.vocabulary, tokens)?;
        let (_count, bf16_bytes, fp32_bytes, id_bytes) = output_extents(tokens.len(), self.width)?;
        let ids_host = token_bytes(tokens, id_bytes)?;
        let ids = Buffer::new(context, id_bytes)?;
        ids.upload(&ids_host)?;
        let residual = Buffer::new(context, bf16_bytes)?;
        let normalized = Buffer::new(context, bf16_bytes)?;
        let unrounded = Buffer::new(context, fp32_bytes)?;

        let gathered = self.gather(context, module, &ids, tokens.len(), bf16_bytes)?;
        let (table, row_ids) = gathered
            .as_ref()
            .map_or((self.table, ids.pointer()), |(table, ids)| {
                (table.pointer(), ids.pointer())
            });
        let mut pointers = [
            table,
            row_ids,
            self.norm,
            residual.pointer(),
            normalized.pointer(),
            unrounded.pointer(),
        ];
        let mut width = u32::try_from(self.width).context("embedding width does not fit u32")?;
        let mut epsilon = self.epsilon;
        let mut args: Vec<*mut c_void> = pointers
            .iter_mut()
            .map(|pointer| (pointer as *mut u64).cast())
            .collect();
        args.push((&mut width as *mut u32).cast());
        args.push((&mut epsilon as *mut f32).cast());
        // SAFETY: Arguments follow embedding_norm_bf16's six-pointer ABI. Tokens
        // are in range, all allocations have exact extents, and resident weights,
        // module, and temporaries belong to this context and live through sync.
        unsafe {
            module.function("embedding_norm_bf16")?.launch(
                [u32::try_from(tokens.len())?, 1, 1],
                [256, 1, 1],
                0,
                &mut args,
            )?;
        }
        context.synchronize()?;
        drop(ids);
        drop(unrounded);
        Ok(Normalized {
            residual,
            normalized,
        })
    }

    /// Dequantize only requested rows, retaining temporaries through the norm launch.
    fn gather<'a>(
        &self,
        context: &'a Context,
        module: &Module<'_>,
        tokens: &Buffer<'_>,
        rows: usize,
        bytes: usize,
    ) -> Result<Option<(Buffer<'a>, Buffer<'a>)>> {
        let Some(scale) = self.scale else {
            return Ok(None);
        };
        let output = Buffer::new(context, bytes)?;
        let ids = Buffer::new(context, rows * 4)?;
        let sequential: Vec<u32> = (0..u32::try_from(rows)?).collect();
        ids.upload(&token_bytes(&sequential, rows * 4)?)?;
        let mut pointers = [self.table, scale, tokens.pointer(), output.pointer()];
        let mut width = u32::try_from(self.width)?;
        let mut args: Vec<*mut c_void> = pointers
            .iter_mut()
            .map(|p| (p as *mut u64).cast())
            .collect();
        args.push((&mut width as *mut u32).cast());
        // SAFETY: Constructor verified the encoded table/scale extents; run checked
        // token IDs and contexts. Output is rows*width BF16, not a vocabulary table.
        unsafe {
            module.function("fp8_embedding_gather")?.launch(
                [u32::try_from(rows)?, 1, 1],
                [256, 1, 1],
                0,
                &mut args,
            )?;
        }
        context.synchronize()?;
        Ok(Some((output, ids)))
    }
}

/// Bind normalized logical views, independently of their source container.
/// Both execution paths use this exact dtype/layout/shape/extent validation.
pub(super) fn bind_table(
    owner: &ResidentWeights<'_>,
    name: &str,
    vocabulary: usize,
    width: usize,
) -> Result<(u64, Option<u64>)> {
    let (bf16_bytes, _) = validate_shape(vocabulary, width, 1e-6)?;
    let dtype = &owner.object(name)?.dtype;
    let encoded = encoded_table(dtype, vocabulary, width)?;
    let shape = [u64::try_from(vocabulary)?, u64::try_from(width)?];
    let table = owner.tensor(
        name,
        dtype.clone(),
        &shape,
        if encoded { bf16_bytes / 2 } else { bf16_bytes },
    )?;
    let scale = if encoded {
        let prefix = name
            .strip_suffix(".weight")
            .context("embedding name must end in .weight")?;
        Some(owner.tensor(
            &format!("{prefix}.weight_scale"),
            DType::Bf16,
            &[shape[0], 1],
            shape[0] * 2,
        )?)
    } else {
        None
    };
    Ok((table, scale))
}

fn encoded_table(dtype: &DType, vocabulary: usize, width: usize) -> Result<bool> {
    match dtype {
        DType::Bf16 => Ok(false),
        DType::Fp8E4m3 => {
            ensure!(
                [vocabulary, width] == [248_320, 5120],
                "encoded embedding requires exact [248320, 5120] shape"
            );
            Ok(true)
        }
        _ => anyhow::bail!("unsupported embedding dtype: {}", dtype.as_str()),
    }
}

fn validate_shape(vocabulary: usize, width: usize, epsilon: f32) -> Result<(u64, u64)> {
    ensure!(
        (1..=1_048_576).contains(&vocabulary),
        "embedding vocabulary is out of range"
    );
    ensure!(
        (1..=32768).contains(&width),
        "embedding width is out of range"
    );
    ensure!(
        epsilon.is_finite() && epsilon > 0.0,
        "embedding epsilon must be positive and finite"
    );
    let vocabulary = u64::try_from(vocabulary).context("embedding vocabulary does not fit u64")?;
    let width = u64::try_from(width).context("embedding width does not fit u64")?;
    let table_bytes = vocabulary
        .checked_mul(width)
        .and_then(|elements| elements.checked_mul(2))
        .context("embedding table byte extent overflows u64")?;
    let norm_bytes = width
        .checked_mul(2)
        .context("embedding norm byte extent overflows u64")?;
    Ok((table_bytes, norm_bytes))
}

fn validate_tokens(vocabulary: usize, tokens: &[u32]) -> Result<()> {
    ensure!(
        (1..=2048).contains(&tokens.len()),
        "embedding token count is out of range"
    );
    let vocabulary = u64::try_from(vocabulary).context("embedding vocabulary does not fit u64")?;
    ensure!(
        tokens.iter().all(|&token| u64::from(token) < vocabulary),
        "embedding token is out of vocabulary range"
    );
    Ok(())
}

fn output_extents(rows: usize, width: usize) -> Result<(usize, usize, usize, usize)> {
    ensure!(
        (1..=2048).contains(&rows),
        "embedding token count is out of range"
    );
    ensure!(
        (1..=32768).contains(&width),
        "embedding width is out of range"
    );
    let count = rows
        .checked_mul(width)
        .context("embedding output element count overflows usize")?;
    let bf16_bytes = count
        .checked_mul(2)
        .context("embedding BF16 output extent overflows usize")?;
    let fp32_bytes = count
        .checked_mul(4)
        .context("embedding FP32 output extent overflows usize")?;
    let id_bytes = rows
        .checked_mul(4)
        .context("embedding token byte extent overflows usize")?;
    Ok((count, bf16_bytes, fp32_bytes, id_bytes))
}

fn token_bytes(tokens: &[u32], expected_bytes: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(expected_bytes);
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    ensure!(
        bytes.len() == expected_bytes,
        "embedding token byte extent mismatch"
    );
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::{output_extents, token_bytes, validate_shape, validate_tokens};

    #[test]
    fn representation_dispatch_is_explicit_and_bounded() {
        use crate::artifact::schema::DType;
        assert!(!super::encoded_table(&DType::Bf16, 3, 2).unwrap());
        assert!(super::encoded_table(&DType::Fp8E4m3, 248_320, 5120).unwrap());
        assert!(super::encoded_table(&DType::Fp8E4m3, 248_320, 5119).is_err());
        assert!(super::encoded_table(&DType::Fp8E4m3, 248_319, 5120).is_err());
        assert!(super::encoded_table(&DType::F32, 248_320, 5120).is_err());
    }

    #[test]
    fn validates_minimum_and_maximum_embedding_shapes() {
        assert_eq!(validate_shape(1, 1, 1e-6).unwrap(), (2, 2));
        assert_eq!(
            validate_shape(1_048_576, 32768, 1e-6).unwrap(),
            (68_719_476_736, 65_536)
        );
        assert_eq!(output_extents(1, 1).unwrap(), (1, 2, 4, 4));
        assert_eq!(
            output_extents(2048, 32768).unwrap(),
            (67_108_864, 134_217_728, 268_435_456, 8192)
        );
    }

    #[test]
    fn rejects_invalid_embedding_dimensions_and_epsilon() {
        for (vocabulary, width) in [(0, 1), (1_048_577, 1), (1, 0), (1, 32769)] {
            assert!(validate_shape(vocabulary, width, 1e-6).is_err());
        }
        for epsilon in [0.0, f32::INFINITY, f32::NAN] {
            assert!(validate_shape(3, 2, epsilon).is_err());
        }
        assert!(output_extents(0, 2).is_err());
    }

    #[test]
    fn validates_first_and_last_tokens_and_rejects_out_of_range_ids() {
        assert!(validate_tokens(3, &[0, 2]).is_ok());
        assert!(validate_tokens(3, &[3]).is_err());
        assert!(validate_tokens(3, &[]).is_err());
        assert!(validate_tokens(3, &vec![0; 2049]).is_err());
    }

    #[test]
    fn serializes_token_ids_in_little_endian_order() {
        assert_eq!(token_bytes(&[0x0102_0304], 4).unwrap(), [4, 3, 2, 1]);
        assert!(token_bytes(&[1], 8).is_err());
    }
}
