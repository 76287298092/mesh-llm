//! Checked, device-independent placement for named allocations.

use anyhow::{Context, Result, ensure};
use serde::Serialize;

const ALIGNMENT: u64 = 256;
const MAX_REGIONS: usize = 65_536;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Region {
    pub name: String,
    pub offset: u64,
    pub length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Layout {
    pub regions: Vec<Region>,
    pub bytes: u64,
}

impl Layout {
    /// Build a canonical layout with every region start and the total size aligned to 256 bytes.
    pub fn new(entries: impl IntoIterator<Item = (String, u64)>) -> Result<Self> {
        let mut pending = Vec::new();
        for (name, length) in entries {
            ensure!(
                pending.len() < MAX_REGIONS,
                "layout has more than {MAX_REGIONS} regions"
            );
            ensure!(!name.is_empty(), "layout region name is empty");
            ensure!(length > 0, "layout region `{name}` has zero length");
            usize::try_from(length).context("layout region length does not fit usize")?;
            pending.push((name, length));
        }
        ensure!(!pending.is_empty(), "layout has no regions");

        pending.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        ensure!(
            pending.windows(2).all(|pair| pair[0].0 != pair[1].0),
            "layout has duplicate region names"
        );

        let mut regions = Vec::with_capacity(pending.len());
        let mut cursor = 0_u64;
        for (name, length) in pending {
            let offset = align_up(cursor)?;
            let end = offset
                .checked_add(length)
                .context("layout region end overflows u64")?;
            usize::try_from(offset).context("layout region offset does not fit usize")?;
            usize::try_from(end).context("layout region end does not fit usize")?;
            regions.push(Region {
                name,
                offset,
                length,
            });
            cursor = end;
        }
        let bytes = align_up(cursor)?;
        usize::try_from(bytes).context("layout size does not fit usize")?;
        Ok(Self { regions, bytes })
    }

    /// Find a region by name in the canonical sorted region list.
    pub fn region(&self, name: &str) -> Result<&Region> {
        self.regions
            .binary_search_by(|region| region.name.as_str().cmp(name))
            .map(|index| &self.regions[index])
            .map_err(|_| anyhow::anyhow!("layout region `{name}` was not found"))
    }
}

fn align_up(value: u64) -> Result<u64> {
    value
        .checked_add(ALIGNMENT - 1)
        .map(|rounded| rounded & !(ALIGNMENT - 1))
        .ok_or_else(|| anyhow::anyhow!("layout alignment overflows u64"))
}

#[cfg(test)]
mod tests {
    use super::{ALIGNMENT, Layout, MAX_REGIONS};

    fn entries(items: &[(&str, u64)]) -> Vec<(String, u64)> {
        items
            .iter()
            .map(|(name, length)| ((*name).to_owned(), *length))
            .collect()
    }

    #[test]
    fn aligns_regions_and_final_size_after_sorting_names() {
        let layout = Layout::new(entries(&[("gamma", 257), ("alpha", 1), ("beta", 256)])).unwrap();
        assert_eq!(
            layout.regions,
            [
                super::Region {
                    name: "alpha".to_owned(),
                    offset: 0,
                    length: 1,
                },
                super::Region {
                    name: "beta".to_owned(),
                    offset: 256,
                    length: 256,
                },
                super::Region {
                    name: "gamma".to_owned(),
                    offset: 512,
                    length: 257,
                },
            ]
        );
        assert_eq!(layout.bytes, 1024);
        assert!(
            layout
                .regions
                .iter()
                .all(|region| region.offset % ALIGNMENT == 0)
        );
        assert!(
            layout
                .regions
                .windows(2)
                .all(|pair| pair[0].offset + pair[0].length <= pair[1].offset)
        );
        assert_eq!(layout.bytes % ALIGNMENT, 0);
    }

    #[test]
    fn input_order_does_not_change_the_layout() {
        let forward = Layout::new(entries(&[("alpha", 1), ("beta", 257)])).unwrap();
        let reverse = Layout::new(entries(&[("beta", 257), ("alpha", 1)])).unwrap();
        assert_eq!(forward, reverse);
    }

    #[test]
    fn lookup_uses_canonical_names_and_reports_missing_regions() {
        let layout = Layout::new(entries(&[("z", 1), ("a", 256)])).unwrap();
        assert_eq!(layout.region("a").unwrap().offset, 0);
        assert_eq!(layout.region("z").unwrap().offset, 256);
        assert!(layout.region("missing").is_err());
    }

    #[test]
    fn rejects_invalid_entries_and_overflow() {
        assert!(Layout::new(Vec::<(String, u64)>::new()).is_err());
        assert!(Layout::new(entries(&[("", 1)])).is_err());
        assert!(Layout::new(entries(&[("zero", 0)])).is_err());
        assert!(Layout::new(entries(&[("same", 1), ("same", 2)])).is_err());
        assert!(Layout::new(entries(&[("a", u64::MAX), ("b", 1)])).is_err());
    }

    #[test]
    fn rejects_more_than_the_region_limit() {
        let too_many = (0..=MAX_REGIONS).map(|index| (index.to_string(), 1));
        assert!(Layout::new(too_many).is_err());
    }
}
