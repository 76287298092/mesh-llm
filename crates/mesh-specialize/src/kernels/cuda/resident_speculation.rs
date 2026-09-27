//! Greedy speculative decoding with isolated target verification and rollback.

use super::{
    driver::{Buffer, Context, Module},
    resident_model::{LogitsSelection, Model as TargetModel, Session},
    resident_mtp::{self, Model as DraftModel},
};
use crate::kernels::DecoderConfig;
use anyhow::{Context as _, Result, ensure};
use serde::Serialize;
use std::time::Instant;

const MIN_PROMPT_ROWS: usize = 1;
const MAX_PROMPT_ROWS: usize = 512;
const MIN_OUTPUT_TOKENS: usize = 2;
const MAX_OUTPUT_TOKENS: usize = 128;
const MIN_DRAFT_DEPTH: usize = 1;
const MAX_DRAFT_DEPTH: usize = 4;

pub(super) struct Request<'a> {
    pub(super) tokens: &'a [u32],
    pub(super) output_tokens: usize,
    pub(super) depth: usize,
    pub(super) forced_first_round: Option<&'a [u32]>,
}

pub(super) struct Run<'ctx> {
    pub(super) compact_recovery: bool,
    pub(super) first_round_accepted: Option<usize>,
    pub(super) tokens: Vec<u32>,
    pub(super) target_session: Session<'ctx>,
    pub(super) rounds: usize,
    pub(super) all_accepted_rounds: usize,
    pub(super) drafted: usize,
    pub(super) accepted: usize,
    pub(super) verify_rows: usize,
    pub(super) replay_rows: usize,
    pub(super) prefill_seconds: f64,
    pub(super) decode_seconds: f64,
    pub(super) phase_seconds: PhaseSeconds,
}

#[derive(Default, Serialize)]
pub(super) struct PhaseSeconds {
    pub(super) draft: f64,
    pub(super) verification: f64,
    pub(super) replay: f64,
    pub(super) teacher: f64,
}

struct Engines<'m, 'ctx, 'tw, 'dw> {
    context: &'ctx Context,
    module: &'m Module<'ctx>,
    target: &'m TargetModel<'tw, 'ctx>,
    draft: &'m DraftModel<'dw, 'ctx>,
    config: &'m DecoderConfig,
    compact_recovery: bool,
}

struct BaseState<'ctx> {
    target: Session<'ctx>,
    draft: resident_mtp::Session<'ctx>,
    pending: u32,
    cache: DraftCache<'ctx>,
}

struct DraftCache<'ctx> {
    token: u32,
    last_hidden: Buffer<'ctx>,
    past: usize,
}

struct RoundResult {
    emitted: Vec<u32>,
    drafted: usize,
    accepted: usize,
    verify_rows: usize,
    replay_rows: usize,
    phase_seconds: PhaseSeconds,
}

pub(super) fn run<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    target: &TargetModel<'_, 'ctx>,
    draft: &DraftModel<'_, 'ctx>,
    config: &DecoderConfig,
    request: &Request<'_>,
) -> Result<Run<'ctx>> {
    let compact_recovery = compact_recovery_enabled()?;
    let expected_final_past = validate_request(config, request)?;
    let vocab = u32::try_from(config.vocabulary).context("decoder vocabulary does not fit u32")?;
    ensure!(
        request.tokens.iter().all(|&token| token < vocab),
        "speculation prompt token is outside vocabulary"
    );
    if let Some(tokens) = request.forced_first_round {
        ensure!(
            tokens.iter().all(|&token| token < vocab),
            "forced draft token is outside vocabulary"
        );
    }

    let engines = Engines {
        compact_recovery,
        context,
        module,
        target,
        draft,
        config,
    };
    let (mut state, prefill_seconds) = engines.prefill(request.tokens)?;
    let mut output_tokens = vec![state.pending];
    let mut forced_first_round = request.forced_first_round;
    let mut first_round_accepted = None;
    let mut rounds = 0_usize;
    let mut all_accepted_rounds = 0_usize;
    let mut drafted = 0_usize;
    let mut accepted = 0_usize;
    let mut verify_rows = 0_usize;
    let mut replay_rows = 0_usize;
    let mut phase_seconds = PhaseSeconds::default();

    let decode_start = Instant::now();
    while output_tokens.len() < request.output_tokens {
        let remaining = request.output_tokens - output_tokens.len();
        // Reserve one target token after the proposals so the output cap is exact.
        let proposal_count = request.depth.min(remaining.saturating_sub(1));
        let forced = if proposal_count == 0 {
            None
        } else {
            forced_first_round.take()
        };
        let round = engines.round(&mut state, proposal_count, forced)?;
        first_round_accepted.get_or_insert(round.accepted);
        ensure!(
            !round.emitted.is_empty() && round.emitted.len() <= remaining,
            "speculative round made invalid output progress"
        );
        if round.drafted > 0 && round.accepted == round.drafted {
            checked_add_counter(
                &mut all_accepted_rounds,
                1,
                "all-accepted speculation round",
            )?;
        }
        output_tokens.extend(round.emitted);
        checked_add_counter(&mut rounds, 1, "speculation round")?;
        checked_add_counter(&mut drafted, round.drafted, "drafted token")?;
        checked_add_counter(&mut accepted, round.accepted, "accepted token")?;
        checked_add_counter(&mut verify_rows, round.verify_rows, "verification row")?;
        checked_add_counter(&mut replay_rows, round.replay_rows, "replay row")?;
        phase_seconds.draft += round.phase_seconds.draft;
        phase_seconds.verification += round.phase_seconds.verification;
        phase_seconds.replay += round.phase_seconds.replay;
        phase_seconds.teacher += round.phase_seconds.teacher;
        let expected_past = request
            .tokens
            .len()
            .checked_add(output_tokens.len() - 1)
            .context("speculation output cursor overflows usize")?;
        ensure!(
            state.target.cursor.past() == expected_past && state.cache.past == expected_past,
            "target and draft cursors diverged after a speculation round"
        );
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    ensure!(
        output_tokens.len() == request.output_tokens,
        "speculation returned the wrong number of tokens"
    );
    ensure!(
        state.target.cursor.past() == expected_final_past
            && state.cache.past == expected_final_past,
        "final target and draft cursors do not match the emitted prefix"
    );

    Ok(Run {
        compact_recovery,
        first_round_accepted,
        tokens: output_tokens,
        target_session: state.target,
        rounds,
        all_accepted_rounds,
        drafted,
        accepted,
        verify_rows,
        replay_rows,
        prefill_seconds,
        decode_seconds,
        phase_seconds,
    })
}

impl<'m, 'ctx, 'tw, 'dw> Engines<'m, 'ctx, 'tw, 'dw> {
    fn prefill(&self, prompt: &[u32]) -> Result<(BaseState<'ctx>, f64)> {
        let mut target_session = Session::new(self.context, self.config)?;
        let mut draft_session = resident_mtp::Session::new(self.context, self.config)?;
        let start = Instant::now();
        let target_output = self.target.forward_detailed(
            self.context,
            self.module,
            prompt,
            &mut target_session,
            LogitsSelection::Last,
            None,
        )?;
        ensure!(
            target_output.tokens.len() == 1 && target_output.past == prompt.len(),
            "target prefill returned an invalid greedy prefix"
        );
        let pending = target_output.tokens[0];
        ensure!(
            target_session.cursor.past() == prompt.len(),
            "target prefill cursor does not match the prompt"
        );

        let target_hidden = self.draft.target_hidden(
            self.context,
            self.module,
            &target_output.hidden,
            prompt.len(),
        )?;
        let shifted = shifted_prompt(prompt, pending);
        let draft_output = self.draft.forward(
            self.context,
            self.module,
            &shifted,
            &target_hidden,
            &mut draft_session,
        )?;
        ensure!(
            draft_output.past == target_output.past,
            "MTP prefill cursor does not match target prefill"
        );
        ensure!(
            draft_session.cursor.past() == target_session.cursor.past(),
            "MTP and target session cursors differ after prefill"
        );
        validate_hidden_extent(draft_output.hidden.len(), prompt.len(), self.config.hidden)?;
        let last_hidden = copy_hidden_rows(
            self.context,
            &draft_output.hidden,
            prompt.len(),
            prompt.len() - 1,
            1,
            self.config.hidden,
        )?;
        let cache = DraftCache {
            token: draft_output.token,
            last_hidden,
            past: draft_output.past,
        };
        ensure!(
            cache.past == target_session.cursor.past(),
            "target and draft cursors differ after prefill"
        );
        Ok((
            BaseState {
                target: target_session,
                draft: draft_session,
                pending,
                cache,
            },
            start.elapsed().as_secs_f64(),
        ))
    }

    fn proposals(
        &self,
        state: &BaseState<'ctx>,
        count: usize,
        forced_first: Option<&[u32]>,
    ) -> Result<Vec<u32>> {
        ensure!(
            state.target.cursor.past() == state.cache.past
                && state.draft.cursor.past() == state.cache.past,
            "target and draft cursors differ before drafting"
        );
        if count == 0 {
            return Ok(Vec::new());
        }
        if let Some(tokens) = forced_first {
            ensure!(
                tokens.len() == count,
                "forced draft prefix length differs from first round"
            );
            return Ok(tokens.to_vec());
        }
        let mut proposals = Vec::with_capacity(count);
        proposals.push(state.cache.token);
        if count == 1 {
            return Ok(proposals);
        }

        let mut draft_branch = state.draft.fork(self.context)?;
        let mut previous_hidden = None;
        for index in 1..count {
            let hidden = previous_hidden.as_ref().unwrap_or(&state.cache.last_hidden);
            let previous_token = proposals[index - 1];
            let output = self.draft.forward(
                self.context,
                self.module,
                &[previous_token],
                hidden,
                &mut draft_branch,
            )?;
            let expected_past = state
                .cache
                .past
                .checked_add(index)
                .context("draft branch cursor overflows usize")?;
            ensure!(
                output.past == expected_past && draft_branch.cursor.past() == expected_past,
                "draft branch cursor advanced by an unexpected amount"
            );
            validate_hidden_extent(output.hidden.len(), 1, self.config.hidden)?;
            proposals.push(output.token);
            previous_hidden = Some(output.hidden);
        }
        Ok(proposals)
    }

    fn round(
        &self,
        state: &mut BaseState<'ctx>,
        proposal_count: usize,
        forced_first: Option<&[u32]>,
    ) -> Result<RoundResult> {
        let base_past = state.target.cursor.past();
        ensure!(
            base_past == state.cache.past && state.draft.cursor.past() == state.cache.past,
            "target and draft cursors differ before verification"
        );
        let draft_start = Instant::now();
        let proposals = self.proposals(state, proposal_count, forced_first)?;
        let draft_seconds = draft_start.elapsed().as_secs_f64();
        let mut verify_inputs = Vec::with_capacity(proposals.len() + 1);
        verify_inputs.push(state.pending);
        verify_inputs.extend_from_slice(&proposals);

        let verification_start = Instant::now();
        let mut verification_session = state.target.fork(self.context)?;
        let verified = if self.compact_recovery {
            self.target.forward_recorded(
                self.context,
                self.module,
                &verify_inputs,
                &mut verification_session,
            )?
        } else {
            self.target.forward_detailed(
                self.context,
                self.module,
                &verify_inputs,
                &mut verification_session,
                LogitsSelection::All,
                None,
            )?
        };
        let verification_seconds = verification_start.elapsed().as_secs_f64();
        let verify_past = base_past
            .checked_add(verify_inputs.len())
            .context("target verification cursor overflows usize")?;
        ensure!(
            verified.past == verify_past,
            "target verification cursor advanced unexpectedly"
        );
        let accepted = longest_accepted_prefix(&proposals, &verified.tokens)?;
        let correction = verified.tokens[accepted];
        let mut emitted = Vec::with_capacity(accepted + 1);
        emitted.extend_from_slice(&proposals[..accepted]);
        emitted.push(correction);

        let (replay_rows, replay_seconds) = if accepted == proposals.len() {
            state.target = verification_session;
            (0, 0.0)
        } else if self.compact_recovery {
            let replay_start = Instant::now();
            let past = super::resident_recovery::recover(
                self.context,
                self.module,
                &mut state.target,
                super::resident_recovery::Recovery {
                    config: self.config,
                    records: &verified.recovery,
                    verified: &verification_session,
                    rows: accepted + 1,
                },
            )?;
            ensure!(
                past == base_past + accepted + 1,
                "compact recovery cursor differs"
            );
            (accepted + 1, replay_start.elapsed().as_secs_f64())
        } else {
            drop(verification_session);
            let mut replay_inputs = Vec::with_capacity(accepted + 1);
            replay_inputs.push(state.pending);
            replay_inputs.extend_from_slice(&proposals[..accepted]);
            let replay_start = Instant::now();
            let replay = self.target.forward(
                self.context,
                self.module,
                &replay_inputs,
                &mut state.target,
                None,
            )?;
            ensure!(
                replay.token == correction,
                "target replay correction differs from verification"
            );
            let replay_past = base_past
                .checked_add(replay_inputs.len())
                .context("target replay cursor overflows usize")?;
            ensure!(
                replay.past == replay_past && state.target.cursor.past() == replay_past,
                "target replay cursor advanced unexpectedly"
            );
            (replay_inputs.len(), replay_start.elapsed().as_secs_f64())
        };

        let teacher_start = Instant::now();
        let teacher_rows = accepted + 1;
        let raw_prefix = copy_hidden_rows(
            self.context,
            &verified.hidden,
            verify_inputs.len(),
            0,
            teacher_rows,
            self.config.hidden,
        )?;
        let target_hidden =
            self.draft
                .target_hidden(self.context, self.module, &raw_prefix, teacher_rows)?;
        let teacher_tokens = teacher_tokens(&proposals[..accepted], correction);
        let draft_output = self.draft.forward(
            self.context,
            self.module,
            &teacher_tokens,
            &target_hidden,
            &mut state.draft,
        )?;
        let expected_past = base_past
            .checked_add(teacher_rows)
            .context("accepted prefix cursor overflows usize")?;
        ensure!(
            draft_output.past == expected_past && state.target.cursor.past() == expected_past,
            "target and draft cursors differ after correction replay"
        );
        ensure!(
            state.draft.cursor.past() == expected_past,
            "MTP session cursor differs after correction replay"
        );
        validate_hidden_extent(draft_output.hidden.len(), teacher_rows, self.config.hidden)?;
        let last_hidden = copy_hidden_rows(
            self.context,
            &draft_output.hidden,
            teacher_rows,
            teacher_rows - 1,
            1,
            self.config.hidden,
        )?;
        state.pending = correction;
        state.cache = DraftCache {
            token: draft_output.token,
            last_hidden,
            past: draft_output.past,
        };
        let teacher_seconds = teacher_start.elapsed().as_secs_f64();
        Ok(RoundResult {
            emitted,
            drafted: proposals.len(),
            accepted,
            verify_rows: verify_inputs.len(),
            replay_rows,
            phase_seconds: PhaseSeconds {
                draft: draft_seconds,
                verification: verification_seconds,
                replay: replay_seconds,
                teacher: teacher_seconds,
            },
        })
    }
}

pub(super) fn compact_recovery_enabled() -> Result<bool> {
    match std::env::var("MESH_SPECIALIZE_MTP_RECOVERY").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("full-forward") => Ok(false),
        Ok("compact") => Ok(true),
        _ => anyhow::bail!("MESH_SPECIALIZE_MTP_RECOVERY must be full-forward or compact"),
    }
}

fn validate_request(config: &DecoderConfig, request: &Request<'_>) -> Result<usize> {
    validate_bounds(
        request.tokens.len(),
        request.output_tokens,
        request.depth,
        config.capacity,
    )
}

fn validate_bounds(
    prompt_rows: usize,
    output_tokens: usize,
    depth: usize,
    capacity: usize,
) -> Result<usize> {
    ensure!(
        (MIN_PROMPT_ROWS..=MAX_PROMPT_ROWS).contains(&prompt_rows),
        "speculation prompt rows are out of range"
    );
    ensure!(
        (MIN_OUTPUT_TOKENS..=MAX_OUTPUT_TOKENS).contains(&output_tokens),
        "speculation output token count is out of range"
    );
    ensure!(
        (MIN_DRAFT_DEPTH..=MAX_DRAFT_DEPTH).contains(&depth),
        "speculation draft depth is out of range"
    );
    let expected_final_past = prompt_rows
        .checked_add(output_tokens - 1)
        .context("speculation final cursor overflows usize")?;
    ensure!(
        expected_final_past <= capacity,
        "speculation output exceeds target context capacity"
    );
    Ok(expected_final_past)
}

fn longest_accepted_prefix(proposals: &[u32], target_tokens: &[u32]) -> Result<usize> {
    let expected_rows = proposals
        .len()
        .checked_add(1)
        .context("speculation verification row count overflows usize")?;
    ensure!(
        target_tokens.len() == expected_rows,
        "target verification token extent mismatch"
    );
    Ok(proposals
        .iter()
        .zip(target_tokens)
        .take_while(|(draft, target)| draft == target)
        .count())
}

fn shifted_prompt(prompt: &[u32], first_target_token: u32) -> Vec<u32> {
    let mut shifted = Vec::with_capacity(prompt.len());
    shifted.extend_from_slice(&prompt[1..]);
    shifted.push(first_target_token);
    shifted
}

fn teacher_tokens(accepted: &[u32], correction: u32) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(accepted.len() + 1);
    tokens.extend_from_slice(accepted);
    tokens.push(correction);
    tokens
}

fn hidden_bytes(rows: usize, width: usize) -> Result<usize> {
    rows.checked_mul(width)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| anyhow::anyhow!("speculation hidden extent overflows usize"))
}

fn validate_hidden_extent(actual_bytes: usize, rows: usize, width: usize) -> Result<()> {
    let expected = hidden_bytes(rows, width)?;
    ensure!(
        actual_bytes == expected,
        "speculation hidden buffer extent mismatch"
    );
    Ok(())
}

fn copy_hidden_rows<'ctx>(
    context: &'ctx Context,
    source: &Buffer<'_>,
    source_rows: usize,
    first_row: usize,
    rows: usize,
    width: usize,
) -> Result<Buffer<'ctx>> {
    validate_hidden_extent(source.len(), source_rows, width)?;
    ensure!(rows > 0, "speculation hidden copy must include a row");
    let end_row = first_row
        .checked_add(rows)
        .context("speculation hidden row range overflows usize")?;
    ensure!(
        end_row <= source_rows,
        "speculation hidden row range is out of bounds"
    );
    let row_bytes = hidden_bytes(1, width)?;
    let source_offset = first_row
        .checked_mul(row_bytes)
        .context("speculation hidden byte offset overflows usize")?;
    let bytes = hidden_bytes(rows, width)?;
    let output = Buffer::new(context, bytes)?;
    output.copy_from_at(0, source, source_offset, bytes)?;
    Ok(output)
}

fn checked_add_counter(counter: &mut usize, amount: usize, label: &str) -> Result<()> {
    *counter = counter
        .checked_add(amount)
        .ok_or_else(|| anyhow::anyhow!("{label} counter overflows usize"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_DRAFT_DEPTH, MAX_OUTPUT_TOKENS, MAX_PROMPT_ROWS, longest_accepted_prefix,
        shifted_prompt, teacher_tokens, validate_bounds,
    };

    #[test]
    fn acceptance_handles_forced_rejection_full_acceptance_and_mixed_prefixes() {
        assert_eq!(longest_accepted_prefix(&[9, 2], &[4, 8, 7]).unwrap(), 0);
        assert_eq!(longest_accepted_prefix(&[4, 8], &[4, 8, 7]).unwrap(), 2);
        assert_eq!(
            longest_accepted_prefix(&[4, 2, 7], &[4, 8, 7, 1]).unwrap(),
            1
        );
        assert!(longest_accepted_prefix(&[1], &[1]).is_err());
    }

    #[test]
    fn prompt_shift_and_teacher_forcing_keep_next_token_alignment() {
        assert_eq!(shifted_prompt(&[10, 11, 12], 13), [11, 12, 13]);
        assert_eq!(shifted_prompt(&[10], 13), [13]);
        assert_eq!(teacher_tokens(&[20, 21], 22), [20, 21, 22]);
    }

    #[test]
    fn speculation_bounds_include_the_unconsumed_final_output_token() {
        assert_eq!(validate_bounds(512, 128, 4, 639).unwrap(), 639);
        assert!(validate_bounds(0, 2, 1, 10).is_err());
        assert!(validate_bounds(1, 1, 1, 10).is_err());
        assert!(validate_bounds(MAX_PROMPT_ROWS + 1, 2, 1, 1000).is_err());
        assert!(validate_bounds(1, MAX_OUTPUT_TOKENS + 1, 1, 1000).is_err());
        assert!(validate_bounds(1, 2, MAX_DRAFT_DEPTH + 1, 1000).is_err());
        assert!(validate_bounds(512, 128, 4, 638).is_err());
    }
}
