use super::{MAX_PROMPT_ROWS, copy_hidden_rows, hidden_bytes, validate_hidden_extent};
use crate::kernels::cuda::{
    driver::{Buffer, Context, Module},
    resident_mtp, resident_native_mtp_forward,
};
use anyhow::{Context as _, Result, ensure};

pub(super) enum Draft<'m, 'w, 'ctx> {
    Legacy(&'m resident_mtp::Model<'w, 'ctx>),
    #[expect(
        dead_code,
        reason = "native MTP speculation driver is not wired until native forward admission"
    )]
    Native(&'m resident_native_mtp_forward::Model<'w, 'ctx>),
}

pub(super) struct Output<'ctx> {
    pub(super) hidden: Buffer<'ctx>,
    pub(super) token: u32,
    pub(super) past: usize,
}

const NATIVE_WIDTH: usize = 5_120;
const NATIVE_MAX_ROWS: usize = 5;

impl Draft<'_, '_, '_> {
    pub(super) fn target_hidden<'ctx>(
        &self,
        context: &'ctx Context,
        module: &Module<'_>,
        raw: &Buffer<'_>,
        rows: usize,
    ) -> Result<Buffer<'ctx>> {
        match self {
            Self::Legacy(model) => model.target_hidden(context, module, raw, rows),
            Self::Native(model) => {
                if native_direct(rows)? {
                    return model.target_hidden(context, module, raw, rows);
                }
                validate_hidden_extent(raw.len(), rows, NATIVE_WIDTH)?;
                let normalized = Buffer::new(context, hidden_bytes(rows, NATIVE_WIDTH)?)?;
                let row_bytes = hidden_bytes(1, NATIVE_WIDTH)?;
                for row in 0..rows {
                    let input = copy_hidden_rows(context, raw, rows, row, 1, NATIVE_WIDTH)?;
                    let output = model.target_hidden(context, module, &input, 1)?;
                    normalized.copy_from_at(row * row_bytes, &output, 0, row_bytes)?;
                }
                Ok(normalized)
            }
        }
    }

    pub(super) fn forward<'ctx>(
        &self,
        context: &'ctx Context,
        module: &Module<'_>,
        tokens: &[u32],
        hidden: &Buffer<'_>,
        session: &mut resident_mtp::Session<'ctx>,
    ) -> Result<Output<'ctx>> {
        match self {
            Self::Legacy(model) => {
                let output = model.forward(context, module, tokens, hidden, session)?;
                Ok(Output {
                    hidden: output.hidden,
                    token: output.token,
                    past: output.past,
                })
            }
            Self::Native(model) => {
                if native_direct(tokens.len())? {
                    let output = model.forward(context, module, tokens, hidden, session)?;
                    return Ok(Output {
                        hidden: output.hidden,
                        token: output.token,
                        past: output.past,
                    });
                }
                validate_hidden_extent(hidden.len(), tokens.len(), NATIVE_WIDTH)?;
                let expected_past = session
                    .cursor
                    .past()
                    .checked_add(tokens.len())
                    .context("native draft prefill cursor overflows usize")?;
                ensure!(
                    expected_past <= session.cursor.capacity(),
                    "native draft prefill exceeds capacity"
                );
                let output_hidden =
                    Buffer::new(context, hidden_bytes(tokens.len(), NATIVE_WIDTH)?)?;
                let row_bytes = hidden_bytes(1, NATIVE_WIDTH)?;
                let serial_cursor = session.cursor.fork()?;
                let transaction = session.cursor.begin(tokens.len())?;
                let mut serial_session = resident_native_mtp_forward::Session {
                    state: session.state.fork(context)?,
                    cursor: serial_cursor,
                };
                let mut last_token = None;
                for (row, &token) in tokens.iter().enumerate() {
                    let input =
                        copy_hidden_rows(context, hidden, tokens.len(), row, 1, NATIVE_WIDTH)?;
                    let output =
                        model.forward(context, module, &[token], &input, &mut serial_session)?;
                    validate_hidden_extent(output.hidden.len(), 1, NATIVE_WIDTH)?;
                    output_hidden.copy_from_at(row * row_bytes, &output.hidden, 0, row_bytes)?;
                    last_token = Some(output.token);
                }
                ensure!(
                    serial_session.cursor.past() == expected_past,
                    "native draft serial prefill cursor differs"
                );
                let token = last_token.context("native draft prefill produced no token")?;
                session.state = serial_session.state;
                let past = transaction.commit();
                Ok(Output {
                    hidden: output_hidden,
                    token,
                    past,
                })
            }
        }
    }
}

fn native_direct(rows: usize) -> Result<bool> {
    ensure!(
        (1..=MAX_PROMPT_ROWS).contains(&rows),
        "native draft rows are out of range"
    );
    Ok(rows <= NATIVE_MAX_ROWS)
}

#[cfg(test)]
mod tests {
    use super::native_direct;

    #[test]
    fn native_prefill_uses_whole_steps_when_rows_exceed_forward_limit() {
        let rows = [1, 2, 3, 4, 5, 6, 512];
        let routes: Vec<bool> = rows
            .into_iter()
            .map(|rows| native_direct(rows).unwrap())
            .collect();
        assert_eq!(routes, [true, true, true, true, true, false, false]);
    }

    #[test]
    fn native_prefill_rejects_empty_and_oversized_inputs() {
        let results = [0, 513, usize::MAX].map(native_direct);
        assert!(results.into_iter().all(|result| result.is_err()));
    }
}
