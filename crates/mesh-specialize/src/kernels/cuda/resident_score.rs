//! Teacher-forced row scoring: the model's final norm and FP8 head over chunks of
//! hidden rows, then per-row log-softmax statistics on device.
use super::{
    driver::{Buffer, Context, Module},
    resident_fp8,
    resident_head::Head,
};
use crate::{
    engine::teacher_scoring::{Record, TOP_K},
    kernels::fp8_profile::{self, Profile},
    row_logprob_topk_reference as reference,
};
use anyhow::{Context as _, Result, bail, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(super) const KERNEL: &str = "row_logprob_topk_bf16";
const THREADS: u32 = 256;
/// 128 rows keep BF16 plus FP32 head outputs near 191 MB per chunk.
const EXACT_CHUNK_ROWS: usize = 128;
const REFERENCE_ROWS: usize = 4;

/// Head rows per chunk, chosen so every chunk runs the same head kernel family
/// as the one-row generation head under the active FP8 profile: A16 GEMV
/// profiles project one row, the sliced A16 head at most eight, native-prefill
/// profiles stay below the 16-row native threshold, exact uses 128.
pub(super) fn chunk_rows(profile: Profile) -> usize {
    match profile {
        Profile::A16Decode | Profile::A16HeadGemv => 1,
        Profile::A16Head => 8,
        Profile::NativePrefill
        | Profile::NativePrefillAudit
        | Profile::NativePrefillShort
        | Profile::NativePrefillShortAudit => 15,
        Profile::Exact => EXACT_CHUNK_ROWS,
    }
}

pub(super) struct Scorer<'m, 'w, 'ctx> {
    head: &'m Head<'w, 'ctx>,
    width: usize,
    vocabulary: usize,
    chunk_rows: usize,
}

impl<'m, 'w, 'ctx> Scorer<'m, 'w, 'ctx> {
    pub(super) fn new(head: &'m Head<'w, 'ctx>, width: usize, vocabulary: usize) -> Result<Self> {
        ensure!(
            (1..=32_768).contains(&width) && (TOP_K..=262_144).contains(&vocabulary),
            "scoring width or vocabulary is out of range"
        );
        Ok(Self {
            head,
            width,
            vocabulary,
            chunk_rows: chunk_rows(fp8_profile::current()?),
        })
    }

    pub(super) fn chunk_rows(&self) -> usize {
        self.chunk_rows
    }

    /// Score local hidden rows `[first_row, first_row + targets.len())` of a
    /// `rows x width` BF16 buffer; row `i` predicts `targets[i - first_row]`.
    pub(super) fn score(
        &self,
        context: &Context,
        module: &Module<'_>,
        hidden: &Buffer<'_>,
        rows: usize,
        first_row: usize,
        targets: &[u32],
    ) -> Result<Vec<Record>> {
        self.validate(hidden, rows, first_row, targets)?;
        let mut records = Vec::with_capacity(targets.len());
        for (index, chunk) in targets.chunks(self.chunk_rows).enumerate() {
            let start = first_row + index * self.chunk_rows;
            let logits = self.project(context, module, hidden, start, chunk.len())?;
            records.extend(self.reduce(context, module, &logits, chunk)?);
        }
        Ok(records)
    }

    /// Independent evidence for the first chunk of one window: device records
    /// against the FP64 oracle on the first rows, and chunked head logits
    /// against the one-row generation head for the chunk's first and last rows.
    pub(super) fn check(
        &self,
        context: &Context,
        module: &Module<'_>,
        hidden: &Buffer<'_>,
        rows: usize,
        first_row: usize,
        targets: &[u32],
    ) -> Result<Value> {
        self.validate(hidden, rows, first_row, targets)?;
        let count = targets.len().min(self.chunk_rows);
        let logits = self.project(context, module, hidden, first_row, count)?;
        let records = self.reduce(context, module, &logits, &targets[..count])?;
        let mut worst = 0.0_f64;
        let mut reference_error = None;
        for (row, record) in records.iter().take(REFERENCE_ROWS).enumerate() {
            let values = self.download_row(&logits, row)?;
            let oracle = reference::score_row(&values, record.target).map_err(anyhow::Error::msg)?;
            match reference::matches_device(
                &oracle,
                record.target_logprob,
                record.logsumexp,
                &record.top_ids,
                &record.top_logprobs,
            ) {
                Ok(error) => worst = worst.max(error),
                Err(error) => {
                    reference_error = Some(format!("row {row}: {error}"));
                    break;
                }
            }
        }
        let mut head_rows = Vec::new();
        for row in [0, count - 1] {
            if head_rows.iter().any(|entry: &Value| entry["row"] == row) {
                continue;
            }
            let single = self.single_row_head(context, module, hidden, first_row + row)?;
            let chunked = self.download_row(&logits, row)?;
            let differing = single.iter().zip(&chunked).filter(|(a, b)| a != b).count();
            head_rows.push(json!({"row": row, "bf16_differences": differing}));
        }
        let head_exact = head_rows.iter().all(|entry| entry["bf16_differences"] == 0);
        Ok(json!({
            "passed": reference_error.is_none() && head_exact,
            "reference_rows": records.len().min(REFERENCE_ROWS),
            "reference_max_abs_error": worst,
            "reference_error": reference_error,
            "reference": "reference/row_logprob_topk.rs FP64 host exp/ln, exact top-64 ids",
            "chunk_rows": count,
            "head_rows_vs_one_row_head": head_rows,
            "head_bit_exact": head_exact,
        }))
    }

    fn validate(
        &self,
        hidden: &Buffer<'_>,
        rows: usize,
        first_row: usize,
        targets: &[u32],
    ) -> Result<()> {
        ensure!(
            (1..=2048).contains(&rows) && !targets.is_empty(),
            "scoring rows are out of range"
        );
        ensure!(
            hidden.len() == rows * self.width * 2,
            "scoring hidden extent mismatch"
        );
        ensure!(
            first_row
                .checked_add(targets.len())
                .is_some_and(|end| end <= rows),
            "scored rows exceed the hidden rows"
        );
        ensure!(
            targets.iter().all(|&t| (t as usize) < self.vocabulary),
            "scoring target outside vocabulary"
        );
        Ok(())
    }

    /// Final norm and head over `count` rows starting at `start`; BF16 logits.
    fn project<'a>(
        &self,
        context: &'a Context,
        module: &Module<'_>,
        hidden: &Buffer<'_>,
        start: usize,
        count: usize,
    ) -> Result<Buffer<'a>> {
        let row_bytes = self.width * 2;
        let chunk = Buffer::new(context, count * row_bytes)?;
        chunk.copy_from_at(0, hidden, start * row_bytes, count * row_bytes)?;
        let resident_fp8::Output { values, unrounded } =
            self.head.run_all(context, module, &chunk, count)?;
        drop(unrounded);
        ensure!(
            values.len() == count * self.vocabulary * 2,
            "scoring head logit extent mismatch"
        );
        Ok(values)
    }

    /// The generation head (`Head::run`, one final row) for one hidden row.
    fn single_row_head(
        &self,
        context: &Context,
        module: &Module<'_>,
        hidden: &Buffer<'_>,
        row: usize,
    ) -> Result<Vec<u16>> {
        let row_bytes = self.width * 2;
        let single = Buffer::new(context, row_bytes)?;
        single.copy_from_at(0, hidden, row * row_bytes, row_bytes)?;
        let output = self.head.run(context, module, &single, 1)?;
        self.download_row(&output.values, 0)
    }

    fn download_row(&self, logits: &Buffer<'_>, row: usize) -> Result<Vec<u16>> {
        let mut raw = vec![0_u8; self.vocabulary * 2];
        logits.download_at(row * raw.len(), &mut raw)?;
        Ok(raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|word| u16::from_le_bytes(*word))
            .collect())
    }

    fn reduce(
        &self,
        context: &Context,
        module: &Module<'_>,
        logits: &Buffer<'_>,
        targets: &[u32],
    ) -> Result<Vec<Record>> {
        let rows = targets.len();
        ensure!(
            logits.len() == rows * self.vocabulary * 2,
            "scoring logit extent mismatch"
        );
        let target_ids = Buffer::new(context, rows * 4)?;
        target_ids.upload(&words(targets.iter().map(|t| t.to_le_bytes())))?;
        let buffers = [rows * 4, rows * 4, rows * TOP_K * 4, rows * TOP_K * 4, rows * 4]
            .map(|bytes| Buffer::new(context, bytes));
        let [target_logprobs, logsumexps, top_ids, top_logprobs, status] = buffers;
        let outputs = [target_logprobs?, logsumexps?, top_ids?, top_logprobs?, status?];
        let mut pointers = [logits.pointer(), target_ids.pointer()]
            .into_iter()
            .chain(outputs.iter().map(Buffer::pointer))
            .collect::<Vec<u64>>();
        let mut dimensions = [u32::try_from(self.vocabulary)?, u32::try_from(self.vocabulary)?];
        let mut arguments: Vec<*mut c_void> = pointers
            .iter_mut()
            .map(|pointer| (pointer as *mut u64).cast())
            .collect();
        arguments.extend(dimensions.iter_mut().map(|value| (value as *mut u32).cast()));
        let function = module.function(KERNEL)?;
        // SAFETY: `rows` contiguous BF16 rows with stride equal to the validated
        // vocabulary (>= 64), `rows` targets and exactly sized outputs stay live
        // through the synchronization below; grid and block match the contract.
        let launched = unsafe {
            function.launch([u32::try_from(rows)?, 1, 1], [THREADS, 1, 1], 0, &mut arguments)
        };
        let synchronized = context.synchronize();
        launched.context("row log-probability launch failed")?;
        synchronized?;
        decode_records(targets, &outputs)
    }
}

fn words(items: impl Iterator<Item = [u8; 4]>) -> Vec<u8> {
    items.flatten().collect()
}

fn download_words(buffer: &Buffer<'_>) -> Result<Vec<[u8; 4]>> {
    let mut raw = vec![0_u8; buffer.len()];
    buffer.download(&mut raw)?;
    Ok(raw.as_chunks::<4>().0.to_vec())
}

fn decode_records(targets: &[u32], outputs: &[Buffer<'_>; 5]) -> Result<Vec<Record>> {
    let [target_logprobs, logsumexps, top_ids, top_logprobs, status] =
        outputs.each_ref().map(download_words);
    let (target_logprobs, logsumexps) = (target_logprobs?, logsumexps?);
    let (top_ids, top_logprobs, status) = (top_ids?, top_logprobs?, status?);
    let mut records = Vec::with_capacity(targets.len());
    for (row, &target) in targets.iter().enumerate() {
        let code = u32::from_le_bytes(status[row]);
        if code != 0 {
            bail!("scoring row {row} status {code:#x} (1 nonfinite, 2 target, 4 target nonfinite, 8 short)");
        }
        let record = Record {
            target,
            target_logprob: f32::from_le_bytes(target_logprobs[row]),
            logsumexp: f32::from_le_bytes(logsumexps[row]),
            top_ids: std::array::from_fn(|i| u32::from_le_bytes(top_ids[row * TOP_K + i])),
            top_logprobs: std::array::from_fn(|i| {
                f32::from_le_bytes(top_logprobs[row * TOP_K + i])
            }),
        };
        ensure!(
            record.target_logprob.is_finite()
                && record.target_logprob <= 0.0
                && record.logsumexp.is_finite(),
            "scoring row {row} produced an invalid target log-probability"
        );
        records.push(record);
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::{EXACT_CHUNK_ROWS, chunk_rows};
    use crate::kernels::fp8_profile::Profile;

    #[test]
    fn chunks_keep_the_generation_head_kernel_family() {
        assert_eq!(chunk_rows(Profile::Exact), EXACT_CHUNK_ROWS);
        assert_eq!(chunk_rows(Profile::A16Decode), 1);
        assert_eq!(chunk_rows(Profile::A16HeadGemv), 1);
        assert_eq!(chunk_rows(Profile::A16Head), 8);
        assert!(chunk_rows(Profile::NativePrefill) < 16);
        assert!(chunk_rows(Profile::NativePrefillShort) < 16);
    }

    // 128 rows x 248,320 x (2 + 4) bytes stays under 256 MB.
    const _: () = assert!(EXACT_CHUNK_ROWS * 248_320 * 6 <= 256 * 1024 * 1024);
}
