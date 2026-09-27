//! Bounded ownership policy for reusable decoder-prefix checkpoints.
//!
//! This module owns opaque payloads and matches exact request token prefixes. It
//! cannot determine whether a payload represents a healthy or complete session;
//! the caller must enforce the checkpoint contract before insertion.

use anyhow::{Context, Result, ensure};

/// Exact model and execution identity required to reuse a checkpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixIdentity {
    weights_content_id: String,
    arithmetic_profile: String,
    kv_format: String,
    geometry_id: String,
}

impl PrefixIdentity {
    /// Create an identity after checking that every component is present.
    pub fn new(
        weights_content_id: impl Into<String>,
        arithmetic_profile: impl Into<String>,
        kv_format: impl Into<String>,
        geometry_id: impl Into<String>,
    ) -> Result<Self> {
        let weights_content_id = weights_content_id.into();
        let arithmetic_profile = arithmetic_profile.into();
        let kv_format = kv_format.into();
        let geometry_id = geometry_id.into();

        for (name, value) in [
            ("weights_content_id", weights_content_id.as_str()),
            ("arithmetic_profile", arithmetic_profile.as_str()),
            ("kv_format", kv_format.as_str()),
            ("geometry_id", geometry_id.as_str()),
        ] {
            ensure!(
                !value.trim().is_empty(),
                "prefix identity `{name}` is empty"
            );
        }

        Ok(Self {
            weights_content_id,
            arithmetic_profile,
            kv_format,
            geometry_id,
        })
    }

    pub fn weights_content_id(&self) -> &str {
        &self.weights_content_id
    }

    pub fn arithmetic_profile(&self) -> &str {
        &self.arithmetic_profile
    }

    pub fn kv_format(&self) -> &str {
        &self.kv_format
    }

    pub fn geometry_id(&self) -> &str {
        &self.geometry_id
    }
}

/// Cumulative cache operation counters. Values saturate instead of wrapping.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrefixCacheCounters {
    pub hits: u64,
    pub misses: u64,
    pub insertions: u64,
    pub replacements: u64,
    pub evictions: u64,
    pub rejected_inserts: u64,
}

/// A borrowed checkpoint and the number of request tokens it covers.
#[derive(Debug)]
pub struct PrefixMatch<'a, T> {
    /// Immutable access only: the caller must fork state before continuing decode.
    pub payload: &'a T,
    pub prefix_len: usize,
}

/// A least-recently-used cache that owns opaque checkpoint payloads.
///
/// Entries stay ordered from least recently used to most recently used. This
/// deterministic index policy avoids a wrapping recency stamp.
pub struct PrefixCache<T> {
    max_entries: usize,
    max_bytes: usize,
    bytes: usize,
    entries: Vec<Entry<T>>,
    counters: PrefixCacheCounters,
}

struct Entry<T> {
    identity: PrefixIdentity,
    consumed_prefix: Vec<u32>,
    accounted_bytes: usize,
    payload: T,
}

impl<T> PrefixCache<T> {
    /// Create a cache with nonzero entry-count and accounted-byte limits.
    pub fn new(max_entries: usize, max_bytes: usize) -> Result<Self> {
        ensure!(max_entries > 0, "prefix cache max_entries must be nonzero");
        ensure!(max_bytes > 0, "prefix cache max_bytes must be nonzero");
        Ok(Self {
            max_entries,
            max_bytes,
            bytes: 0,
            entries: Vec::new(),
            counters: PrefixCacheCounters::default(),
        })
    }

    /// Insert an owned checkpoint for all tokens in `consumed_prefix`.
    ///
    /// A duplicate identity and exact token prefix replaces the old payload.
    /// Least-recently-used entries are dropped until both configured limits fit.
    /// `accounted_bytes` is caller-supplied and does not include cache metadata.
    pub fn insert(
        &mut self,
        identity: PrefixIdentity,
        consumed_prefix: Vec<u32>,
        accounted_bytes: usize,
        payload: T,
    ) -> Result<()> {
        if consumed_prefix.is_empty() {
            return self.reject_insert("prefix cache token prefix is empty");
        }
        if accounted_bytes == 0 {
            return self.reject_insert("prefix cache accounted bytes must be nonzero");
        }
        if accounted_bytes > self.max_bytes {
            return self.reject_insert("prefix checkpoint exceeds the cache byte limit");
        }

        let replacement_index = self.entries.iter().position(|entry| {
            entry.identity == identity && entry.consumed_prefix == consumed_prefix
        });
        let is_replacement = replacement_index.is_some();
        if let Some(index) = replacement_index {
            let replaced_bytes = self.entries[index].accounted_bytes;
            let bytes_without_replaced = self
                .bytes
                .checked_sub(replaced_bytes)
                .context("prefix cache byte accounting underflow while replacing")?;
            self.entries.remove(index);
            self.bytes = bytes_without_replaced;
        }

        let available_bytes = self.max_bytes - accounted_bytes;
        while self.entries.len() >= self.max_entries || self.bytes > available_bytes {
            self.evict_lru()?;
        }

        let total_bytes = self
            .bytes
            .checked_add(accounted_bytes)
            .context("prefix cache byte accounting overflows usize")?;
        ensure!(
            total_bytes <= self.max_bytes,
            "prefix cache byte limit was exceeded after eviction"
        );
        self.entries.push(Entry {
            identity,
            consumed_prefix,
            accounted_bytes,
            payload,
        });
        self.bytes = total_bytes;

        if is_replacement {
            increment(&mut self.counters.replacements);
        } else {
            increment(&mut self.counters.insertions);
        }
        Ok(())
    }

    /// Return the longest exact cached prefix of the request and refresh its LRU position.
    ///
    /// The borrowed payload remains owned by this cache. A miss changes no entry
    /// recency, and no entry from a different identity is considered.
    pub fn lookup(
        &mut self,
        identity: &PrefixIdentity,
        request_tokens: &[u32],
    ) -> Option<PrefixMatch<'_, T>> {
        let mut best_match: Option<(usize, usize)> = None;
        for (index, entry) in self.entries.iter().enumerate() {
            if &entry.identity == identity && request_tokens.starts_with(&entry.consumed_prefix) {
                let prefix_len = entry.consumed_prefix.len();
                if best_match.is_none_or(|(_, best_len)| prefix_len > best_len) {
                    best_match = Some((index, prefix_len));
                }
            }
        }

        let Some((index, prefix_len)) = best_match else {
            increment(&mut self.counters.misses);
            return None;
        };

        let matched = self.entries.remove(index);
        self.entries.push(matched);
        increment(&mut self.counters.hits);
        let entry = self.entries.last()?;
        Some(PrefixMatch {
            payload: &entry.payload,
            prefix_len,
        })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total caller-accounted payload bytes currently owned by the cache.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Snapshot cumulative counters; clearing entries does not reset them.
    pub fn counters(&self) -> PrefixCacheCounters {
        self.counters
    }

    /// Drop all cached payloads and reset used bytes while retaining counters.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }

    fn evict_lru(&mut self) -> Result<()> {
        let oldest = self
            .entries
            .first()
            .context("prefix cache cannot evict from an empty cache")?;
        let remaining_bytes = self
            .bytes
            .checked_sub(oldest.accounted_bytes)
            .context("prefix cache byte accounting underflow while evicting")?;
        self.entries.remove(0);
        self.bytes = remaining_bytes;
        increment(&mut self.counters.evictions);
        Ok(())
    }

    fn reject_insert(&mut self, message: &'static str) -> Result<()> {
        increment(&mut self.counters.rejected_inserts);
        anyhow::bail!("{message}");
    }
}

fn increment(counter: &mut u64) {
    *counter = counter.saturating_add(1);
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::{PrefixCache, PrefixIdentity};

    fn identity() -> PrefixIdentity {
        identity_with("weights-a", "exact-v1", "bf16-kv", "qwen-27b-131k")
    }

    fn identity_with(
        weights_content_id: &str,
        arithmetic_profile: &str,
        kv_format: &str,
        geometry_id: &str,
    ) -> PrefixIdentity {
        PrefixIdentity::new(
            weights_content_id,
            arithmetic_profile,
            kv_format,
            geometry_id,
        )
        .unwrap()
    }

    #[test]
    fn rejects_zero_limits_and_empty_identity_components() {
        assert!(PrefixCache::<()>::new(0, 1).is_err());
        assert!(PrefixCache::<()>::new(1, 0).is_err());

        assert!(PrefixIdentity::new("", "exact-v1", "bf16-kv", "geometry").is_err());
        assert!(PrefixIdentity::new("weights", " ", "bf16-kv", "geometry").is_err());
        assert!(PrefixIdentity::new("weights", "exact-v1", "", "geometry").is_err());
        assert!(PrefixIdentity::new("weights", "exact-v1", "bf16-kv", "\t").is_err());
    }

    #[test]
    fn rejects_empty_zero_sized_and_oversized_checkpoints() {
        let mut cache = PrefixCache::<&str>::new(2, 4).unwrap();
        assert!(cache.insert(identity(), Vec::new(), 1, "empty").is_err());
        assert!(cache.insert(identity(), vec![1], 0, "zero").is_err());
        assert!(cache.insert(identity(), vec![1], 5, "oversized").is_err());
        assert!(cache.is_empty());
        assert_eq!(cache.bytes(), 0);
        assert_eq!(cache.counters().rejected_inserts, 3);
    }

    #[test]
    fn rejects_a_mismatch_in_each_identity_component() {
        let mut cache = PrefixCache::new(4, 32).unwrap();
        let base = identity();
        cache.insert(base, vec![10, 20], 4, "checkpoint").unwrap();

        let mismatches = [
            identity_with("weights-b", "exact-v1", "bf16-kv", "qwen-27b-131k"),
            identity_with("weights-a", "exact-v2", "bf16-kv", "qwen-27b-131k"),
            identity_with("weights-a", "exact-v1", "fp8-kv", "qwen-27b-131k"),
            identity_with("weights-a", "exact-v1", "bf16-kv", "qwen-27b-64k"),
        ];
        for mismatch in mismatches {
            assert!(cache.lookup(&mismatch, &[10, 20, 30]).is_none());
        }
        assert_eq!(cache.counters().misses, 4);
    }

    #[test]
    fn requires_an_exact_complete_token_prefix_and_selects_the_longest_match() {
        let mut cache = PrefixCache::new(4, 32).unwrap();
        let key = identity();
        cache.insert(key.clone(), vec![7, 19], 3, "short").unwrap();
        cache
            .insert(key.clone(), vec![7, 19, 23], 4, "long")
            .unwrap();

        assert!(cache.lookup(&key, &[7, 18, 23]).is_none());
        assert!(cache.lookup(&key, &[7]).is_none());
        let matched = cache.lookup(&key, &[7, 19, 23, 31]).unwrap();
        assert_eq!(matched.prefix_len, 3);
        assert_eq!(*matched.payload, "long");
    }

    #[test]
    fn lookup_updates_only_the_matched_entry_lru_position() {
        let mut cache = PrefixCache::new(2, 32).unwrap();
        let key = identity();
        cache.insert(key.clone(), vec![1], 2, "one").unwrap();
        cache.insert(key.clone(), vec![2], 2, "two").unwrap();

        assert_eq!(*cache.lookup(&key, &[1, 8]).unwrap().payload, "one");
        assert!(cache.lookup(&key, &[9]).is_none());
        cache.insert(key.clone(), vec![3], 2, "three").unwrap();

        assert!(cache.lookup(&key, &[2]).is_none());
        assert!(cache.lookup(&key, &[1]).is_some());
        assert!(cache.lookup(&key, &[3]).is_some());
        assert_eq!(cache.counters().evictions, 1);
    }

    #[test]
    fn byte_limit_evicts_as_many_oldest_entries_as_needed() {
        let mut cache = PrefixCache::new(4, 5).unwrap();
        let key = identity();
        cache.insert(key.clone(), vec![1], 3, "three").unwrap();
        cache.insert(key.clone(), vec![2], 2, "two").unwrap();
        cache.insert(key.clone(), vec![3], 4, "four").unwrap();

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.bytes(), 4);
        assert!(cache.lookup(&key, &[1]).is_none());
        assert!(cache.lookup(&key, &[2]).is_none());
        assert_eq!(*cache.lookup(&key, &[3]).unwrap().payload, "four");
        assert_eq!(cache.counters().evictions, 2);
    }

    #[test]
    fn replacement_updates_accounting_without_duplicate_keys() {
        let mut cache = PrefixCache::new(2, 8).unwrap();
        let key = identity();
        cache.insert(key.clone(), vec![7], 3, "old").unwrap();
        cache.insert(key.clone(), vec![8], 2, "other").unwrap();
        cache.insert(key.clone(), vec![7], 5, "new").unwrap();
        assert!(cache.insert(key.clone(), vec![7], 9, "too large").is_err());

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.bytes(), 7);
        assert_eq!(*cache.lookup(&key, &[7, 9]).unwrap().payload, "new");
        assert_eq!(cache.counters().insertions, 2);
        assert_eq!(cache.counters().replacements, 1);
        assert_eq!(cache.counters().rejected_inserts, 1);
    }

    struct DropSpy(Arc<AtomicUsize>);

    impl Drop for DropSpy {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn cache_owns_payload_until_replacement_or_clear() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut cache = PrefixCache::new(1, 8).unwrap();
        cache
            .insert(identity(), vec![1], 8, DropSpy(Arc::clone(&drops)))
            .unwrap();

        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(cache.len(), 1);
        cache
            .insert(identity(), vec![1], 8, DropSpy(Arc::clone(&drops)))
            .unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.bytes(), 0);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert_eq!(cache.counters().insertions, 1);
        assert_eq!(cache.counters().replacements, 1);
    }
}
