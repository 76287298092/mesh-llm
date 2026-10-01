use crate::kernels::{
    DecoderConfig,
    cuda::{
        driver::{Buffer, Context, Module},
        resident_model::{LogitsSelection, Model as TargetModel, Session as TargetSession},
        resident_native_mtp_forward::{Model as NativeModel, Session},
    },
};
use crate::packages::qwen3_8_27b::target_batch_trial::Fixture;
use anyhow::{Result, ensure};
use serde_json::{Value, json};

const ROW_BYTES: usize = 5_120 * 2;

pub(super) struct Engines<'a, 'w, 'ctx> {
    pub context: &'ctx Context,
    pub module: &'a Module<'ctx>,
    pub target: &'a TargetModel<'w, 'ctx>,
    pub draft: &'a NativeModel<'w, 'ctx>,
    pub config: &'a DecoderConfig,
}

pub(super) struct Prepared<'ctx> {
    pub base: Session<'ctx>,
    pub tokens: [u32; 5],
    pub hidden: Buffer<'ctx>,
    pub continuation_hidden: Buffer<'ctx>,
}

impl<'ctx> Engines<'_, '_, 'ctx> {
    pub(super) fn prepare(&self, fixture: &Fixture) -> Result<Prepared<'ctx>> {
        let mut target_session = TargetSession::new(self.context, self.config)?;
        let mut base = Session::new(self.context, self.config)?;
        let mut shifted = fixture.prefix[1..].to_vec();
        shifted.push(fixture.target_tokens[0]);
        for (prompt, teacher) in fixture.prefix.chunks(5).zip(shifted.chunks(5)) {
            let hidden = self.target_rows(prompt, &mut target_session)?;
            for (row, &token) in teacher.iter().enumerate() {
                let input = Buffer::new(self.context, ROW_BYTES)?;
                input.copy_from_at(0, &hidden, row * ROW_BYTES, ROW_BYTES)?;
                self.draft
                    .forward(self.context, self.module, &[token], &input, &mut base)?;
            }
        }
        ensure!(
            base.cursor.past() == fixture.prefix.len(),
            "native serial base cursor differs from prefix"
        );
        let hidden = self.target_rows(&fixture.target_tokens, &mut target_session)?;
        let continuation_hidden =
            self.target_rows(&fixture.continuation[..1], &mut target_session)?;
        Ok(Prepared {
            base,
            tokens: shifted_verification(fixture),
            hidden,
            continuation_hidden,
        })
    }

    fn target_rows(
        &self,
        tokens: &[u32],
        session: &mut TargetSession<'ctx>,
    ) -> Result<Buffer<'ctx>> {
        let output = self.target.forward_detailed(
            self.context,
            self.module,
            tokens,
            session,
            LogitsSelection::All,
            None,
        )?;
        ensure!(
            output.hidden.len() == tokens.len() * ROW_BYTES,
            "canonical detailed target hidden omitted rows"
        );
        self.draft
            .target_hidden(self.context, self.module, &output.hidden, tokens.len())
    }
}

pub(super) fn shifted_verification(fixture: &Fixture) -> [u32; 5] {
    let [_, second, third, fourth, fifth] = fixture.target_tokens;
    [second, third, fourth, fifth, fixture.continuation[0]]
}

pub(super) fn alignment_report(fixture: &Fixture) -> Value {
    json!({
        "prefix": fixture.prefix,
        "teacher_forced_target_inputs": fixture.target_tokens,
        "supplied_continuation": fixture.continuation,
        "capacity": fixture.capacity,
        "base_pairs": "target hidden at prefix[i] with prefix[i+1], final prefix hidden with target_tokens[0]",
        "verification_pairs": "target hidden at target_tokens[i] with target_tokens[i+1], final target_tokens hidden with continuation[0]",
        "continuation_token_index": 1,
        "continuation_hidden_token_index": 0,
        "unused_continuation_start_index": 2,
        "unused_continuation_qualified": false,
        "shorter_and_recursive_continuation_scope": "same fixed pair is reused by the one-token harness after every case; only T5 is contiguous teacher forcing, others are repeat/state probes",
        "target_vocabulary": 248_320,
        "proposal_rows": 131_072,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verification_shifts_when_fixture_tokens_are_explicit() {
        let fixture = Fixture {
            prefix: vec![10, 11],
            target_tokens: [20, 21, 22, 23, 24],
            continuation: vec![30, 31, 32],
            capacity: 10,
        };
        let tokens = shifted_verification(&fixture);
        assert_eq!(tokens, [21, 22, 23, 24, 30]);
    }
}
