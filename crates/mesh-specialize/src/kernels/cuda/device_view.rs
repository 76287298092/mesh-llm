//! Checked borrowed device byte views, not asynchronous execution leases.
//!
//! `Buffer` permits shared-reference mutation, so a read view does not establish
//! global device immutability. Unsafe enqueue callers must retain all owners and
//! modules through completion or error draining and prevent conflicting work.
//! Captured graphs need separate ownership covering capture and every replay.

use super::driver::{Buffer, Context};
use anyhow::{Result, anyhow, ensure};
use std::marker::PhantomData;

#[derive(Clone, Copy, Debug)]
struct CheckedRange {
    start: u64,
    end: u64,
    bytes: usize,
}

impl CheckedRange {
    fn new(base: u64, allocation_bytes: usize, request: ByteRange) -> Result<Self> {
        ensure!(base != 0, "device allocation has a null address");
        ensure!(request.bytes != 0, "device view must not be empty");
        ensure!(
            request.alignment.is_power_of_two(),
            "invalid device view alignment"
        );
        let relative_end = request
            .offset
            .checked_add(request.bytes)
            .ok_or_else(|| anyhow!("device view offset plus length overflows"))?;
        ensure!(
            relative_end <= allocation_bytes,
            "device view exceeds its parent range"
        );
        let offset = u64::try_from(request.offset)?;
        let length = u64::try_from(request.bytes)?;
        let start = base
            .checked_add(offset)
            .ok_or_else(|| anyhow!("device view start address overflows"))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| anyhow!("device view end address overflows"))?;
        let alignment = u64::try_from(request.alignment)?;
        ensure!(
            start.is_multiple_of(alignment),
            "misaligned device view address"
        );
        Ok(Self {
            start,
            end,
            bytes: request.bytes,
        })
    }

    fn subrange(self, request: ByteRange) -> Result<Self> {
        Self::new(self.start, self.bytes, request)
    }

    fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

/// Allocation-relative byte request. Every field is validated before use.
#[derive(Clone, Copy, Debug)]
pub(super) struct ByteRange {
    pub offset: usize,
    pub bytes: usize,
    pub alignment: usize,
}

impl ByteRange {
    fn whole(bytes: usize) -> Self {
        Self {
            offset: 0,
            bytes,
            alignment: 1,
        }
    }
}

/// Read access for one checked launch; shared Buffer APIs can still mutate memory.
pub(super) struct DeviceRead<'owner, 'ctx> {
    context: &'ctx Context,
    range: CheckedRange,
    _owner: PhantomData<&'owner Buffer<'ctx>>,
}

/// Borrowed write access. Deliberately neither Clone nor Copy.
pub(super) struct DeviceWrite<'owner, 'ctx> {
    context: &'ctx Context,
    range: CheckedRange,
    _owner: PhantomData<&'owner mut Buffer<'ctx>>,
}

impl<'owner, 'ctx> DeviceRead<'owner, 'ctx> {
    pub(super) fn from_buffer(buffer: &'owner Buffer<'ctx>) -> Result<Self> {
        Ok(Self {
            context: buffer.context(),
            range: CheckedRange::new(
                buffer.pointer(),
                buffer.len(),
                ByteRange::whole(buffer.len()),
            )?,
            _owner: PhantomData,
        })
    }

    pub(super) fn subrange(&self, request: ByteRange) -> Result<DeviceRead<'_, 'ctx>> {
        Ok(DeviceRead {
            context: self.context,
            range: self.range.subrange(request)?,
            _owner: PhantomData,
        })
    }

    pub(super) fn context(&self) -> &'ctx Context {
        self.context
    }
    pub(super) fn bytes(&self) -> usize {
        self.range.bytes
    }
    /// Extracting an address does not extend the borrow or retain GPU resources.
    pub(super) fn pointer(&self) -> u64 {
        self.range.start
    }
}

impl<'owner, 'ctx> DeviceWrite<'owner, 'ctx> {
    pub(super) fn as_read(&self) -> DeviceRead<'_, 'ctx> {
        DeviceRead {
            context: self.context,
            range: self.range,
            _owner: PhantomData,
        }
    }

    pub(super) fn context(&self) -> &'ctx Context {
        self.context
    }
    pub(super) fn bytes(&self) -> usize {
        self.range.bytes
    }
    /// The caller must keep the owner alive until submitted work completes.
    pub(super) fn pointer(&self) -> u64 {
        self.range.start
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Access {
    Read,
    Write,
}

/// Borrow disjoint writable regions together, validating all requests first.
/// Consumers can reborrow each produced region through `as_read`.
/// The workspace adapter must also validate requests against their named regions.
pub(super) fn partition<'owner, 'ctx, const N: usize>(
    buffer: &'owner mut Buffer<'ctx>,
    requests: [ByteRange; N],
) -> Result<[DeviceWrite<'owner, 'ctx>; N]> {
    let ranges = partition_ranges(buffer.pointer(), buffer.len(), &requests)?;
    let context = buffer.context();
    Ok(std::array::from_fn(|index| DeviceWrite {
        context,
        range: ranges[index],
        _owner: PhantomData,
    }))
}

fn partition_ranges<const N: usize>(
    base: u64,
    bytes: usize,
    requests: &[ByteRange; N],
) -> Result<[CheckedRange; N]> {
    ensure!(N != 0, "device partition must not be empty");
    let ranges = requests
        .iter()
        .map(|request| CheckedRange::new(base, bytes, *request))
        .collect::<Result<Vec<_>>>()?;
    validate_disjoint(&ranges)?;
    ranges
        .try_into()
        .map_err(|_| anyhow!("device partition extent mismatch"))
}

fn validate_disjoint(ranges: &[CheckedRange]) -> Result<()> {
    for (index, range) in ranges.iter().enumerate() {
        for other in &ranges[index + 1..] {
            ensure!(
                !range.overlaps(*other),
                "overlapping device write or partition ranges"
            );
        }
    }
    Ok(())
}

fn validate_access_pair(
    left: CheckedRange,
    left_access: Access,
    right: CheckedRange,
    right_access: Access,
) -> Result<()> {
    if matches!((left_access, right_access), (Access::Read, Access::Read)) {
        return Ok(());
    }
    ensure!(
        !left.overlaps(right),
        "overlapping device launch access ranges"
    );
    Ok(())
}

fn same_identity<T>(left: &T, right: &T) -> bool {
    std::ptr::eq(left, right)
}

/// Validate one launch's contexts and access intervals. No in-place exceptions.
/// This does not order streams or establish completion/graph lifetime safety.
pub(super) fn validate_launch_access(
    context: &Context,
    reads: &[&DeviceRead<'_, '_>],
    writes: &[&DeviceWrite<'_, '_>],
) -> Result<()> {
    ensure!(
        reads
            .iter()
            .all(|view| same_identity(context, view.context())),
        "device read context mismatch"
    );
    ensure!(
        writes
            .iter()
            .all(|view| same_identity(context, view.context())),
        "device write context mismatch"
    );
    for read in reads {
        for write in writes {
            validate_access_pair(read.range, Access::Read, write.range, Access::Write)?;
        }
    }
    for (index, write) in writes.iter().enumerate() {
        for other in &writes[index + 1..] {
            validate_access_pair(write.range, Access::Write, other.range, Access::Write)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(offset: usize, bytes: usize, alignment: usize) -> ByteRange {
        ByteRange {
            offset,
            bytes,
            alignment,
        }
    }

    fn range(offset: usize, bytes: usize) -> CheckedRange {
        CheckedRange::new(256, 64, request(offset, bytes, 1)).unwrap()
    }

    #[test]
    fn whole_tail_and_nested_ranges_preserve_exact_extents() {
        let whole = range(0, 64);
        assert_eq!((whole.start, whole.end, whole.bytes), (256, 320, 64));
        assert_eq!(range(63, 1).end, 320);
        let sub = whole.subrange(request(16, 16, 4)).unwrap();
        let nested = sub.subrange(request(4, 4, 4)).unwrap();
        assert_eq!((nested.start, nested.bytes), (276, 4));
        assert!(sub.subrange(request(15, 2, 1)).is_err());
    }

    #[test]
    fn rejects_empty_bounds_and_invalid_alignment() {
        for spec in [
            request(0, 0, 1),
            request(65, 1, 1),
            request(63, 2, 1),
            request(0, 2, 0),
            request(0, 2, 3),
            request(1, 2, 2),
            request(2, 4, 4),
        ] {
            assert!(CheckedRange::new(256, 64, spec).is_err());
        }
        assert!(CheckedRange::new(0, 64, request(0, 4, 4)).is_err());
        assert!(CheckedRange::new(257, 64, request(0, 4, 4)).is_err());
        assert!(CheckedRange::new(257, 64, request(3, 4, 4)).is_ok());
        assert!(CheckedRange::new(256, 64, request(2, 2, 2)).is_ok());
    }

    #[test]
    fn rejects_offset_start_and_exclusive_end_overflow() {
        assert!(CheckedRange::new(256, usize::MAX, request(usize::MAX, 1, 1)).is_err());
        assert!(CheckedRange::new(u64::MAX, 4, request(1, 1, 1)).is_err());
        assert!(CheckedRange::new(u64::MAX - 1, 4, request(0, 2, 1)).is_err());
    }

    #[test]
    fn half_open_ranges_allow_adjacency_but_detect_every_overlap() {
        let a = range(0, 16);
        assert!(!a.overlaps(range(16, 16)));
        for b in [range(0, 16), range(4, 4), range(8, 16)] {
            assert!(a.overlaps(b));
            assert!(b.overlaps(a));
            assert!(validate_disjoint(&[a, b]).is_err());
        }
        assert!(validate_disjoint(&[a, range(16, 16)]).is_ok());
    }

    #[test]
    fn partition_validates_all_requests_before_returning() {
        let write = |offset, bytes| request(offset, bytes, 4);
        assert!(partition_ranges(256, 64, &[write(0, 16), write(16, 16)]).is_ok());
        assert!(partition_ranges(256, 64, &[write(0, 16), write(0, 16)]).is_err());
        assert!(partition_ranges(256, 64, &[write(0, 16), write(60, 8)]).is_err());
        assert!(partition_ranges::<0>(256, 64, &[]).is_err());
    }

    #[test]
    fn launch_access_allows_shared_reads_but_rejects_writes() {
        let a = range(0, 16);
        for b in [range(0, 16), range(4, 4), range(8, 16)] {
            assert!(validate_access_pair(a, Access::Read, b, Access::Read).is_ok());
            assert!(validate_access_pair(a, Access::Read, b, Access::Write).is_err());
            assert!(validate_access_pair(a, Access::Write, b, Access::Read).is_err());
            assert!(validate_access_pair(a, Access::Write, b, Access::Write).is_err());
        }
        assert!(validate_access_pair(a, Access::Read, range(16, 16), Access::Write).is_ok());
        assert!(validate_access_pair(a, Access::Write, range(16, 16), Access::Write).is_ok());
    }

    #[test]
    fn identity_compares_objects_not_equal_values() {
        let first = Box::new(7_u32);
        let second = Box::new(7_u32);
        assert!(same_identity(first.as_ref(), first.as_ref()));
        assert!(!same_identity(first.as_ref(), second.as_ref()));
    }
}
