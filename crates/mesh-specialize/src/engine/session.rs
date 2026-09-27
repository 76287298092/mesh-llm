//! Transactional sequence cursor for a stateful decoder session.

use anyhow::{Context, Result, ensure};

const MAX_CAPACITY: usize = 262_144;
const MAX_ROWS: usize = 2_048;

/// Tracks committed rows and prevents reuse after a partially failed decoder step.
#[derive(Debug)]
pub struct Cursor {
    past: usize,
    capacity: usize,
    poisoned: bool,
}

impl Cursor {
    /// Create a cursor with a fixed, nonzero context capacity.
    pub fn new(capacity: usize) -> Result<Self> {
        ensure!(
            (1..=MAX_CAPACITY).contains(&capacity),
            "session capacity must be in 1..={MAX_CAPACITY}"
        );
        Ok(Self {
            past: 0,
            capacity,
            poisoned: false,
        })
    }

    pub fn past(&self) -> usize {
        self.past
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Copy the committed cursor position into an independent, usable cursor.
    pub fn fork(&self) -> Result<Self> {
        ensure!(!self.poisoned, "session cursor is poisoned");
        Ok(Self {
            past: self.past,
            capacity: self.capacity,
            poisoned: false,
        })
    }

    /// Begin a bounded decoder transaction.
    ///
    /// The caller must keep this transaction alive across every decoder layer and
    /// final logits calculation, and commit it only after the entire step succeeds.
    /// Dropping it early poisons the cursor because device state may be partially updated.
    pub fn begin(&mut self, rows: usize) -> Result<Transaction<'_>> {
        let end = validate_request(self.past, self.capacity, self.poisoned, rows)?;
        Ok(Transaction {
            past: self.past,
            rows,
            end,
            cursor: self,
            committed: false,
        })
    }
}

/// Exclusive lease on a cursor while one complete decoder step is in flight.
pub struct Transaction<'a> {
    cursor: &'a mut Cursor,
    past: usize,
    rows: usize,
    end: usize,
    committed: bool,
}

impl Transaction<'_> {
    /// The committed prefix length at the start of this transaction.
    pub fn past(&self) -> usize {
        self.past
    }

    /// Number of rows reserved by this transaction.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Fixed capacity of the session.
    pub fn capacity(&self) -> usize {
        self.cursor.capacity
    }

    /// Commit the new prefix length after all decoder layers and final logits succeed.
    pub fn commit(mut self) -> usize {
        self.cursor.past = self.end;
        self.committed = true;
        self.end
    }
}

impl Drop for Transaction<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.cursor.poisoned = true;
        }
    }
}

fn validate_request(past: usize, capacity: usize, poisoned: bool, rows: usize) -> Result<usize> {
    ensure!(!poisoned, "session cursor is poisoned");
    ensure!(
        (1..=MAX_ROWS).contains(&rows),
        "rows must be in 1..={MAX_ROWS}"
    );
    let end = past
        .checked_add(rows)
        .context("session cursor position overflows usize")?;
    ensure!(end <= capacity, "session request exceeds context capacity");
    Ok(end)
}

#[cfg(test)]
mod tests {
    use super::{Cursor, MAX_CAPACITY, MAX_ROWS, validate_request};

    #[test]
    fn successful_transactions_advance_only_when_committed() {
        let mut cursor = Cursor::new(20).unwrap();
        {
            let transaction = cursor.begin(17).unwrap();
            assert_eq!(transaction.past(), 0);
            assert_eq!(transaction.rows(), 17);
            assert_eq!(transaction.capacity(), 20);
            assert_eq!(transaction.commit(), 17);
        }
        assert_eq!(cursor.past(), 17);
        assert_eq!(cursor.begin(1).unwrap().commit(), 18);
        assert_eq!(cursor.past(), 18);
        assert!(!cursor.is_poisoned());
    }

    #[test]
    fn dropping_an_uncommitted_transaction_poisons_the_cursor() {
        let mut cursor = Cursor::new(32).unwrap();
        drop(cursor.begin(4).unwrap());
        assert_eq!(cursor.past(), 0);
        assert!(cursor.is_poisoned());
        assert!(cursor.begin(1).is_err());
    }

    #[test]
    fn rejected_requests_leave_the_cursor_unchanged_and_usable() {
        let mut cursor = Cursor::new(2_048).unwrap();
        for rows in [0, MAX_ROWS + 1, 2_049] {
            assert!(cursor.begin(rows).is_err());
            assert_eq!(cursor.past(), 0);
            assert!(!cursor.is_poisoned());
        }
        assert!(validate_request(usize::MAX, usize::MAX, false, 1).is_err());
        assert_eq!(cursor.past(), 0);
        assert!(!cursor.is_poisoned());

        let mut short = Cursor::new(2).unwrap();
        assert!(short.begin(3).is_err());
        assert_eq!(short.past(), 0);
        assert!(!short.is_poisoned());
    }

    #[test]
    fn full_capacity_commit_rejects_more_rows_without_poisoning() {
        let mut cursor = Cursor::new(3).unwrap();
        assert_eq!(cursor.begin(3).unwrap().commit(), 3);
        assert!(cursor.begin(1).is_err());
        assert_eq!(cursor.past(), 3);
        assert!(!cursor.is_poisoned());
    }

    #[test]
    fn forked_cursors_advance_independently() {
        let mut original = Cursor::new(20).unwrap();
        original.begin(4).unwrap().commit();

        let mut fork = original.fork().unwrap();
        assert_eq!(fork.past(), 4);
        assert_eq!(fork.capacity(), 20);
        assert_eq!(fork.begin(3).unwrap().commit(), 7);
        assert_eq!(fork.past(), 7);
        assert_eq!(original.past(), 4);
        assert!(!original.is_poisoned());
        assert!(!fork.is_poisoned());
    }

    #[test]
    fn poisoned_cursor_cannot_be_forked() {
        let mut cursor = Cursor::new(20).unwrap();
        drop(cursor.begin(1).unwrap());
        assert!(cursor.is_poisoned());
        assert!(cursor.fork().is_err());
    }

    #[test]
    fn enforces_capacity_bounds() {
        assert!(Cursor::new(0).is_err());
        assert_eq!(Cursor::new(MAX_CAPACITY).unwrap().capacity(), MAX_CAPACITY);
        assert!(Cursor::new(MAX_CAPACITY + 1).is_err());
    }
}
