use super::{Case, report::CapacityReport};
use crate::engine::session::Cursor;
use anyhow::{Result, ensure};

pub(super) fn cases() -> Vec<Case> {
    let mut cases = (0..=4).map(|accepted| Case {
        depth: 4, output_tokens: 6, requested_acceptance: Some(accepted),
    }).collect::<Vec<_>>();
    for depth in [1, 4] {
        for output_tokens in [8, 2, 3, 4, 5] {
            cases.push(Case { depth, output_tokens, requested_acceptance: None });
        }
    }
    cases
}

pub(super) fn forced(target: &[u32], accepted: usize, vocabulary: usize) -> Result<Vec<u32>> {
    ensure!(target.len() == 4 && accepted <= 4, "invalid depth-four forced fixture extent");
    ensure!(vocabulary >= 2 && target.iter().all(|token| usize::try_from(*token)
        .is_ok_and(|id| id < vocabulary)), "invalid target-derived fixture token");
    let mut proposals = target.to_vec();
    if accepted < 4 {
        let token = proposals[accepted];
        proposals[accepted] = if token == 0 { 1 } else { token - 1 };
    }
    Ok(proposals)
}

pub(super) fn capacity_check(capacity: usize) -> Result<CapacityReport> {
    let mut cursor = Cursor::new(capacity)?;
    let mut remaining = capacity;
    while remaining > 0 {
        let rows = remaining.min(2048);
        cursor.begin(rows)?.commit();
        remaining -= rows;
    }
    let before = super::report::CursorReport::from(&cursor);
    let rejected = match cursor.begin(1) {
        Err(error) => Some(format!("{error:#}")),
        Ok(transaction) => { transaction.commit(); None }
    };
    let after = super::report::CursorReport::from(&cursor);
    Ok(CapacityReport {
        passed: rejected.is_some() && before == after && !after.poisoned,
        exercised: "public Cursor::begin(1) at full capacity; no run_native or device-state rejection invocation",
        before, after, rejection: rejected,
    })
}
