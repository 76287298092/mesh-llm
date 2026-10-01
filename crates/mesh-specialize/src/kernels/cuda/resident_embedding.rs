//! Resident encoded embedding lookup with the existing BF16 normalization boundary.

mod binding;
#[cfg(test)]
mod tests;

use self::binding::validate_shape;
use super::{
    driver::{Buffer, Context, Module},
    resident_native_mtp::ResidentNativeMtp,
    resident_norm::{Norm, Normalized},
    resident_weights::ResidentWeights,
};
use anyhow::{Context as _, Result, ensure};
use std::ffi::c_void;

pub(super) use binding::bind_table;

pub(super) struct Embedding<'w, 'ctx> {
    owner: &'w ResidentWeights<'ctx>,
    table: u64,
    scale: Option<u64>,
    norm: Norm<'w, 'ctx>,
    vocabulary: usize,
    width: usize,
    epsilon: f32,
}

pub(super) struct NativeMtpEmbeddingConfig<'view> {
    pub(super) table_name: &'view str,
    pub(super) shape: [usize; 2],
    pub(super) epsilon: f32,
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
        validate_shape(vocabulary, width, epsilon)?;
        let (table, scale) = bind_table(owner, table_name, vocabulary, width)?;
        let norm = Norm::new(owner, norm_name, width, epsilon)?;
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

    /// Use the canonical target table and saved native-MTP embedding gamma.
    /// Both owners must outlive this embedding; FP8 rows are gathered to BF16 first.
    pub(super) fn from_native_mtp(
        owner: &'w ResidentWeights<'ctx>,
        native_mtp: &'w ResidentNativeMtp<'ctx>,
        config: NativeMtpEmbeddingConfig<'_>,
    ) -> Result<Self> {
        let binding = native_mtp.embedding_norm()?;
        let [vocabulary, width] = config.shape;
        validate_shape(vocabulary, width, config.epsilon)?;
        ensure!(
            owner.belongs_to(native_mtp.context()),
            "canonical embedding and native MTP norms belong to different contexts"
        );
        let (table, scale) = bind_table(owner, config.table_name, vocabulary, width)?;
        let norm = Norm::from_native_mtp(binding, width, config.epsilon)?;
        Ok(Self {
            owner,
            table,
            scale,
            norm,
            vocabulary,
            width,
            epsilon: config.epsilon,
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
            self.owner.belongs_to(context) && self.norm.belongs_to(context),
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
            self.norm.weight_pointer(),
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

    pub(super) fn belongs_to(&self, context: &Context) -> bool {
        self.owner.belongs_to(context) && self.norm.belongs_to(context)
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
