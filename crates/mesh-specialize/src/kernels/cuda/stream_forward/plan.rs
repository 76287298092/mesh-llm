//! Pure-host arena planning with liveness-based reuse.
//!
//! A [`Program`] records an ordered list of operations and the named buffers each
//! one reads and writes. Every buffer gets an inclusive step interval from its
//! first write to its last read. [`ArenaPlan::place`] assigns aligned offsets so
//! that buffers with intersecting intervals never share bytes. Buffers that an
//! operation reads and buffers it writes are live at the same step, so a kernel's
//! inputs and outputs never alias. Nothing here touches CUDA.

use anyhow::{Context as _, Result, bail, ensure};
use std::collections::HashMap;

/// Alignment of every arena offset and of the arena extent.
pub(super) const ALIGNMENT: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BufferSpec {
    pub(super) name: String,
    pub(super) bytes: usize,
    pub(super) first: usize,
    pub(super) last: usize,
}

impl BufferSpec {
    fn conflicts(&self, other: &Self) -> bool {
        self.first <= other.last && other.first <= self.last
    }
}

/// An ordered record of operations over named buffers.
#[derive(Default)]
pub(super) struct Program {
    specs: Vec<BufferSpec>,
    index: HashMap<String, usize>,
    step: usize,
    pinned: Vec<usize>,
    whole: Vec<usize>,
}

impl Program {
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Declare a buffer that is live for the entire program (inputs uploaded
    /// before the first operation, persistent tables, carried activations).
    pub(super) fn whole(&mut self, name: &str, bytes: usize) -> Result<()> {
        ensure!(
            !self.index.contains_key(name),
            "arena buffer `{name}` declared twice"
        );
        ensure!(bytes > 0, "arena buffer `{name}` has zero bytes");
        let index = self.specs.len();
        self.specs.push(BufferSpec {
            name: name.to_owned(),
            bytes,
            first: 0,
            last: 0,
        });
        self.index.insert(name.to_owned(), index);
        self.whole.push(index);
        Ok(())
    }

    /// Record one operation. Reads must name buffers that are already written
    /// or whole-program. A repeated write must repeat the same extent.
    pub(super) fn op(&mut self, reads: &[&str], writes: &[(&str, usize)]) -> Result<()> {
        self.step = self.step.checked_add(1).context("program step overflow")?;
        let step = self.step;
        for name in reads {
            let index = *self
                .index
                .get(*name)
                .with_context(|| format!("arena buffer `{name}` is read before any write"))?;
            let spec = &mut self.specs[index];
            spec.last = spec.last.max(step);
        }
        for &(name, bytes) in writes {
            ensure!(bytes > 0, "arena buffer `{name}` has zero bytes");
            if let Some(&index) = self.index.get(name) {
                let spec = &mut self.specs[index];
                ensure!(
                    spec.bytes == bytes,
                    "arena buffer `{name}` rewritten with {bytes} bytes, planned {}",
                    spec.bytes
                );
                spec.last = spec.last.max(step);
            } else {
                self.index.insert(name.to_owned(), self.specs.len());
                self.specs.push(BufferSpec {
                    name: name.to_owned(),
                    bytes,
                    first: step,
                    last: step,
                });
            }
        }
        Ok(())
    }

    /// Keep a buffer live through the end of the program (host readback).
    pub(super) fn keep(&mut self, name: &str) -> Result<()> {
        let index = *self
            .index
            .get(name)
            .with_context(|| format!("cannot keep unknown arena buffer `{name}`"))?;
        self.pinned.push(index);
        Ok(())
    }

    /// Finish the program and return buffer lifetimes.
    pub(super) fn finish(mut self) -> Vec<BufferSpec> {
        let end = self.step.max(1);
        for index in self.whole {
            self.specs[index].first = 0;
            self.specs[index].last = end;
        }
        for index in self.pinned {
            self.specs[index].last = end;
        }
        self.specs
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Placement {
    pub(super) offset: usize,
    pub(super) bytes: usize,
}

/// Offsets for every planned buffer within one arena allocation.
#[derive(Debug)]
pub(super) struct ArenaPlan {
    pub(super) total_bytes: usize,
    pub(super) peak_live_bytes: usize,
    placements: HashMap<String, Placement>,
}

impl ArenaPlan {
    /// Greedy-by-size placement: largest buffers first, each at the lowest
    /// aligned offset that avoids every placed buffer with a conflicting lifetime.
    pub(super) fn place(specs: &[BufferSpec]) -> Result<Self> {
        ensure!(!specs.is_empty(), "arena plan has no buffers");
        let mut order: Vec<usize> = (0..specs.len()).collect();
        order.sort_by(|&left, &right| {
            specs[right]
                .bytes
                .cmp(&specs[left].bytes)
                .then(specs[left].first.cmp(&specs[right].first))
                .then(specs[left].name.cmp(&specs[right].name))
        });
        let mut offsets = vec![None; specs.len()];
        for &index in &order {
            let spec = &specs[index];
            let mut busy: Vec<(usize, usize)> = order
                .iter()
                .filter_map(|&other| {
                    let offset = offsets[other]?;
                    specs[other]
                        .conflicts(spec)
                        .then(|| (offset, offset + specs[other].bytes))
                })
                .collect();
            busy.sort_unstable();
            offsets[index] = Some(first_fit(&busy, spec.bytes)?);
        }
        let mut placements = HashMap::with_capacity(specs.len());
        let mut end = 0_usize;
        for (spec, offset) in specs.iter().zip(offsets) {
            let offset = offset.context("unplaced arena buffer")?;
            end = end.max(offset.checked_add(spec.bytes).context("arena overflow")?);
            ensure!(
                placements
                    .insert(
                        spec.name.clone(),
                        Placement {
                            offset,
                            bytes: spec.bytes
                        }
                    )
                    .is_none(),
                "arena buffer `{}` planned twice",
                spec.name
            );
        }
        let plan = Self {
            total_bytes: align_up(end)?,
            peak_live_bytes: peak_live_bytes(specs),
            placements,
        };
        plan.validate(specs)?;
        Ok(plan)
    }

    pub(super) fn get(&self, name: &str) -> Result<&Placement> {
        self.placements
            .get(name)
            .with_context(|| format!("arena buffer `{name}` is not planned"))
    }

    /// Offset of a planned buffer, or `None` when the program never used it.
    pub(super) fn offset(&self, name: &str) -> Option<usize> {
        self.placements.get(name).map(|placement| placement.offset)
    }

    /// Check alignment, bounds, and that no two simultaneously live buffers overlap.
    pub(super) fn validate(&self, specs: &[BufferSpec]) -> Result<()> {
        ensure!(
            self.total_bytes.is_multiple_of(ALIGNMENT),
            "arena extent is unaligned"
        );
        for spec in specs {
            let placement = self.get(&spec.name)?;
            ensure!(
                placement.offset.is_multiple_of(ALIGNMENT)
                    && placement.bytes == spec.bytes
                    && placement.offset + placement.bytes <= self.total_bytes,
                "arena buffer `{}` is misplaced",
                spec.name
            );
        }
        for (position, left) in specs.iter().enumerate() {
            let a = self.get(&left.name)?;
            for right in &specs[position + 1..] {
                if !left.conflicts(right) {
                    continue;
                }
                let b = self.get(&right.name)?;
                if a.offset < b.offset + b.bytes && b.offset < a.offset + a.bytes {
                    bail!(
                        "live arena buffers `{}` and `{}` overlap",
                        left.name,
                        right.name
                    );
                }
            }
        }
        Ok(())
    }
}

fn first_fit(busy: &[(usize, usize)], bytes: usize) -> Result<usize> {
    let mut candidate = 0_usize;
    for &(start, end) in busy {
        let candidate_end = candidate.checked_add(bytes).context("arena overflow")?;
        if candidate_end <= start {
            break;
        }
        candidate = candidate.max(align_up(end)?);
    }
    Ok(candidate)
}

pub(super) fn align_up(value: usize) -> Result<usize> {
    value
        .checked_next_multiple_of(ALIGNMENT)
        .context("arena alignment overflow")
}

/// Largest sum of simultaneously live buffer sizes: a lower bound for any placement.
pub(super) fn peak_live_bytes(specs: &[BufferSpec]) -> usize {
    let end = specs.iter().map(|spec| spec.last).max().unwrap_or(0);
    (0..=end)
        .map(|step| {
            specs
                .iter()
                .filter(|spec| spec.first <= step && step <= spec.last)
                .map(|spec| spec.bytes)
                .sum::<usize>()
        })
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{ALIGNMENT, ArenaPlan, BufferSpec, Program, first_fit, peak_live_bytes};

    fn spec(name: &str, bytes: usize, first: usize, last: usize) -> BufferSpec {
        BufferSpec {
            name: name.to_owned(),
            bytes,
            first,
            last,
        }
    }

    #[test]
    fn disjoint_lifetimes_share_storage() {
        let specs = [spec("a", 1000, 1, 2), spec("b", 1000, 3, 4)];
        let plan = ArenaPlan::place(&specs).unwrap();
        assert_eq!(plan.get("a").unwrap().offset, plan.get("b").unwrap().offset);
        assert_eq!(plan.total_bytes, 1024);
    }

    #[test]
    fn touching_lifetimes_never_alias() {
        // An operation's input (last use at step 2) and output (first write at 2).
        let specs = [spec("input", 300, 1, 2), spec("output", 300, 2, 3)];
        let plan = ArenaPlan::place(&specs).unwrap();
        let input = plan.get("input").unwrap();
        let output = plan.get("output").unwrap();
        assert!(input.offset + input.bytes <= output.offset || output.offset >= 512);
        assert_eq!(plan.total_bytes, 1024);
    }

    #[test]
    fn offsets_are_aligned_and_validation_rejects_overlap() {
        let specs = [
            spec("x", 1, 1, 5),
            spec("y", 257, 2, 3),
            spec("z", 3, 3, 4),
        ];
        let plan = ArenaPlan::place(&specs).unwrap();
        for name in ["x", "y", "z"] {
            assert!(plan.get(name).unwrap().offset.is_multiple_of(ALIGNMENT));
        }
        plan.validate(&specs).unwrap();
        let shifted = [spec("x", 1, 1, 5), spec("y", 257, 2, 3), spec("z", 3, 3, 4)];
        let mut forged = ArenaPlan::place(&shifted).unwrap();
        let target = forged.get("x").unwrap().offset;
        forged.placements.get_mut("y").unwrap().offset = target;
        assert!(forged.validate(&shifted).is_err());
    }

    #[test]
    fn first_fit_uses_gaps_and_skips_small_ones() {
        assert_eq!(first_fit(&[], 10).unwrap(), 0);
        assert_eq!(first_fit(&[(256, 512)], 256).unwrap(), 0);
        assert_eq!(first_fit(&[(0, 100), (1024, 2048)], 512).unwrap(), 256);
        assert_eq!(first_fit(&[(0, 100), (300, 2048)], 512).unwrap(), 2048);
    }

    #[test]
    fn program_tracks_lifetimes_whole_and_kept_buffers() {
        let mut program = Program::new();
        program.whole("tokens", 16).unwrap();
        program.op(&["tokens"], &[("h", 64)]).unwrap();
        program.op(&["h"], &[("t", 64), ("dead", 32)]).unwrap();
        program.op(&["t"], &[("out", 8)]).unwrap();
        program.keep("h").unwrap();
        program.op(&["out"], &[("last", 8)]).unwrap();
        let specs = program.finish();
        let find = |name: &str| specs.iter().find(|s| s.name == name).unwrap().clone();
        assert_eq!((find("tokens").first, find("tokens").last), (0, 4));
        assert_eq!((find("h").first, find("h").last), (1, 4));
        assert_eq!((find("t").first, find("t").last), (2, 3));
        assert_eq!((find("dead").first, find("dead").last), (2, 2));
        let plan = ArenaPlan::place(&specs).unwrap();
        assert!(plan.total_bytes >= plan.peak_live_bytes);
        assert!(plan.offset("missing").is_none());
    }

    #[test]
    fn program_rejects_read_before_write_and_extent_changes() {
        let mut program = Program::new();
        assert!(program.op(&["unknown"], &[]).is_err());
        program.op(&[], &[("a", 8)]).unwrap();
        assert!(program.op(&[], &[("a", 16)]).is_err());
        assert!(program.op(&[], &[("zero", 0)]).is_err());
        assert!(program.whole("a", 8).is_err());
        assert!(program.keep("unknown").is_err());
    }

    #[test]
    fn peak_live_is_a_lower_bound() {
        let specs = [
            spec("a", 100, 0, 3),
            spec("b", 50, 1, 2),
            spec("c", 70, 2, 4),
        ];
        assert_eq!(peak_live_bytes(&specs), 220);
        let plan = ArenaPlan::place(&specs).unwrap();
        assert!(plan.total_bytes >= 220);
    }
}
