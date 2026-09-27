//! Resident BF16 embedding lookup and input normalization.

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
        let (table_bytes, norm_bytes) = validate_shape(vocabulary, width, epsilon)?;
        let vocabulary_u64 =
            u64::try_from(vocabulary).context("embedding vocabulary does not fit u64")?;
        let width_u64 = u64::try_from(width).context("embedding width does not fit u64")?;
        let table = owner.tensor(
            table_name,
            DType::Bf16,
            &[vocabulary_u64, width_u64],
            table_bytes,
        )?;
        let norm = owner.tensor(norm_name, DType::Bf16, &[width_u64], norm_bytes)?;
        Ok(Self {
            owner,
            table,
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

        let mut pointers = [
            self.table,
            ids.pointer(),
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
