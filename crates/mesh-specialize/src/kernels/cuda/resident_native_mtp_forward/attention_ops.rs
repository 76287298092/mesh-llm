use super::{HEAD_WIDTH, KV_HEADS, Model, QUERY_HEADS};
use crate::kernels::cuda::{
    driver::{Buffer, Context},
    resident_attention_prepare::Tables,
};
use anyhow::{Context as _, Result, ensure};

const QKV_ROWS: usize = 14_336;
const QUERY_START: usize = 0;
const KEY_START: usize = 6_144;
const GATE_START: usize = 7_168;
const VALUE_START: usize = 13_312;
const QUERY_BYTES: usize = QUERY_HEADS * HEAD_WIDTH * 4;
const KEY_VALUE_BYTES: usize = KV_HEADS * HEAD_WIDTH * 2;

pub(super) struct QueryKeyValue<'a> {
    pub(super) query: super::super::resident_attention_prepare::Output<'a>,
    pub(super) key: super::super::resident_attention_prepare::Output<'a>,
    pub(super) value: Buffer<'a>,
}

pub(super) fn prepare<'a>(
    context: &'a Context,
    module: &crate::kernels::cuda::driver::Module<'_>,
    model: &Model<'_, '_>,
    qkv: &Buffer<'_>,
    rows: usize,
    past: usize,
) -> Result<QueryKeyValue<'a>> {
    let query_gate = interleave_query_gate(context, qkv, rows)?;
    let key = select_rows(context, qkv, rows, KEY_START, KEY_VALUE_BYTES)?;
    let tables = model.rope.tables(past, rows)?;
    let cosine = upload_words(context, &tables.cos)?;
    let sine = upload_words(context, &tables.sin)?;
    let prepared_query = model.query_prepare.run(
        context,
        module,
        &query_gate,
        Tables {
            cos: &cosine,
            sin: &sine,
        },
        rows,
    )?;
    let prepared_key = model.key_prepare.run(
        context,
        module,
        &key,
        Tables {
            cos: &cosine,
            sin: &sine,
        },
        rows,
    )?;
    Ok(QueryKeyValue {
        query: prepared_query,
        key: prepared_key,
        value: select_rows(context, qkv, rows, VALUE_START, KEY_VALUE_BYTES)?,
    })
}

fn interleave_query_gate<'a>(
    context: &'a Context,
    qkv: &Buffer<'_>,
    rows: usize,
) -> Result<Buffer<'a>> {
    let row_bytes = QKV_ROWS
        .checked_mul(2)
        .context("native QKV row extent overflow")?;
    let input_bytes = rows
        .checked_mul(row_bytes)
        .context("native QKV extent overflow")?;
    ensure!(
        qkv.belongs_to(context) && qkv.len() == input_bytes,
        "native QKV input context or extent mismatch"
    );
    let output_bytes = rows
        .checked_mul(QUERY_BYTES)
        .context("native Q/gate extent overflow")?;
    let output = Buffer::new(context, output_bytes)?;
    for row in 0..rows {
        let source_row = row
            .checked_mul(row_bytes)
            .context("native QKV row offset overflow")?;
        let destination_row = row
            .checked_mul(QUERY_BYTES)
            .context("native Q/gate row offset overflow")?;
        for head in 0..QUERY_HEADS {
            let query_row = source_row
                .checked_add((QUERY_START + head * HEAD_WIDTH) * 2)
                .context("native query offset overflow")?;
            let gate_row = source_row
                .checked_add((GATE_START + head * HEAD_WIDTH) * 2)
                .context("native gate offset overflow")?;
            let output_head = destination_row
                .checked_add(head * HEAD_WIDTH * 4)
                .context("native interleaved head offset overflow")?;
            output.copy_from_at(output_head, qkv, query_row, HEAD_WIDTH * 2)?;
            output.copy_from_at(output_head + HEAD_WIDTH * 2, qkv, gate_row, HEAD_WIDTH * 2)?;
        }
    }
    Ok(output)
}

fn select_rows<'a>(
    context: &'a Context,
    input: &Buffer<'_>,
    rows: usize,
    first_row: usize,
    row_bytes: usize,
) -> Result<Buffer<'a>> {
    let input_row_bytes = QKV_ROWS
        .checked_mul(2)
        .context("native QKV row extent overflow")?;
    let input_bytes = rows
        .checked_mul(input_row_bytes)
        .context("native QKV extent overflow")?;
    let output_bytes = rows
        .checked_mul(row_bytes)
        .context("native projection slice extent overflow")?;
    ensure!(
        input.belongs_to(context) && input.len() == input_bytes,
        "native projection slice input context or extent mismatch"
    );
    ensure!(
        row_bytes.is_multiple_of(2)
            && first_row
                .checked_mul(2)
                .and_then(|start| start.checked_add(row_bytes))
                .is_some_and(|end| end <= input_row_bytes),
        "native projection row slice exceeds QKV output"
    );
    let output = Buffer::new(context, output_bytes)?;
    for row in 0..rows {
        let source_offset = row
            .checked_mul(input_row_bytes)
            .and_then(|start| start.checked_add(first_row.checked_mul(2)?))
            .context("native projection slice source offset overflow")?;
        let destination_offset = row
            .checked_mul(row_bytes)
            .context("native projection slice destination offset overflow")?;
        output.copy_from_at(destination_offset, input, source_offset, row_bytes)?;
    }
    Ok(output)
}

fn upload_words<'a>(context: &'a Context, words: &[u16]) -> Result<Buffer<'a>> {
    let bytes = words
        .len()
        .checked_mul(2)
        .context("native RoPE table extent overflow")?;
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(bytes)
        .context("native RoPE table allocation failed")?;
    for word in words {
        encoded.extend_from_slice(&word.to_le_bytes());
    }
    let output = Buffer::new(context, bytes)?;
    output.upload(&encoded)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{GATE_START, KEY_START, QKV_ROWS, QUERY_BYTES, VALUE_START};
    #[test]
    fn physical_qkv_ranges_stay_distinct_and_ordered() {
        assert_eq!(KEY_START, 6_144);
        assert_eq!(GATE_START, 7_168);
        assert_eq!(VALUE_START, 13_312);
        assert_eq!(VALUE_START + 1_024, QKV_ROWS);
        assert_eq!(QUERY_BYTES, 24_576);
    }
}
