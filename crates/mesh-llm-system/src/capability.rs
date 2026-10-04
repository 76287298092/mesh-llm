//! What this machine can actually do, as opposed to what it can hold.
//!
//! `hardware::HardwareSurvey` answers the second question: it reports
//! *accelerator* memory, and `capacity::mesh_capacity_bytes` deliberately
//! returns 0 for a non-SoC host with no enumerated accelerator, on the stated
//! grounds that such a host "cannot host a GPU stage". That is the right
//! placement decision, but its consequence is that a CPU-only node — a phone, a
//! TV box, a single-board computer, which is exactly the class of machine this
//! module exists for — advertises no capacity at all through that channel.
//! Everything a master would need in order to decide whether to send it work is
//! therefore invisible: its cores, its clock, its real free memory, how fast it
//! can read storage, and what its links look like.
//!
//! This module measures those, into one serializable record that is written to
//! disk, returned across the FFI, and carried in the mesh announcement. It
//! reports; it never decides. Placement is the master's business and is not
//! encoded here.
//!
//! Two rules govern every field below:
//!
//! * An unmeasured value is `None` and gets a line in `notes`. It is never a
//!   guessed default, because a fabricated 0 is indistinguishable from a
//!   measured 0 and would be read as a real result.
//! * A number that can be measured badly is reported together with how it was
//!   measured, so a consumer can discount it. Storage throughput is the case
//!   that matters: see `StorageCapability::page_cache_may_inflate`.
//!
//! # Why this is a separate channel and not a fix to `hardware`
//!
//! It would be a one-line change to let `hardware::mod`'s Linux RAM probe run on
//! Android, since Android is Linux for `/proc` purposes. That change must not be
//! made. The Linux block feeds `apply_cpu_only_runtime_budget`, which converts
//! system RAM into `survey.vram_bytes = system_ram * 0.75` when no accelerator is
//! found. On the reference device that would make a 3.82 GiB TV box advertise
//! 2.87 GiB of *accelerator* memory it does not have, which is precisely what
//! `capacity.rs` declined to do when it chose to "keep the broader RAM/offload
//! budget local-only instead of advertising it as accelerator capacity". A
//! fabricated accelerator number is worse than an absent one, because the
//! absence is visible and the fabrication is not.
//!
//! So the two questions stay separate: `hardware` keeps answering "what can this
//! node hold on an accelerator", and this module answers "what is this node",
//! reporting RAM as RAM rather than laundering it into VRAM.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Bumped when a field changes meaning. Consumers must treat an unknown version
/// as "cannot interpret" rather than as "fields absent".
pub const CAPABILITY_SCHEMA_VERSION: u32 = 1;

/// How long the compute probe is allowed to run in total.
///
/// A node may be a 4-core in-order core running on a battery, so the probe has
/// to be short enough to be free but long enough to average over scheduler
/// noise. It is a hard ceiling: exceeding it yields a `bench_partial` result
/// rather than a longer measurement.
pub const COMPUTE_PROBE_BUDGET: Duration = Duration::from_millis(250);

/// Bytes the storage probe reads by default. Deliberately small: this runs on
/// devices whose entire storage is 32 GB and whose page cache is a few hundred
/// megabytes.
pub const STORAGE_PROBE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityReport {
    pub schema_version: u32,
    /// Capture time for dynamic memory availability and link state. CPU model,
    /// core count, and total RAM are treated as relatively static inventory.
    pub measured_at_unix_secs: u64,
    pub cpu: CpuCapability,
    pub memory: MemoryCapability,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage: Option<StorageCapability>,
    pub links: Vec<LinkCapability>,
    /// Every field that could not be measured, and why. Present so a consumer
    /// can tell "this node has no wireless link" from "this node's wireless
    /// state was unreadable".
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CpuCapability {
    /// Where the CPU identity and core count were read from.
    pub source: String,
    /// Cores the OS will actually schedule on, not cores present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logical_cores: Option<u32>,
    /// `model name` on x86, `Hardware`/`Processor` on ARM.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_freq_khz: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_freq_khz: Option<u64>,
    /// The governor is the difference between a benchmark number and a
    /// sustainable one: Android TV boxes ship with `interactive` and will not
    /// hold a peak clock.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub governor: Option<String>,
    /// Measured single-thread throughput of the probe below. Single-thread
    /// because that is what a per-token decode step on a stage is limited by
    /// before any parallelism is available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_gflops_fp32: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute_probe_ms: Option<u64>,
    /// True when the probe hit its time budget before finishing its work, in
    /// which case the rate is a lower bound.
    pub compute_probe_partial: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MemoryCapability {
    /// Static physical/system RAM total reported by the platform.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_bytes: Option<u64>,
    /// Dynamic kernel estimate of memory currently available to allocations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub available_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub swap_total_bytes: Option<u64>,
    /// Where the numbers came from, so a consumer can judge them.
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StorageCapability {
    pub path: String,
    pub bytes_read: u64,
    pub elapsed_ms: u64,
    pub read_mb_per_sec: f64,
    /// True when the read may have been served from the page cache rather than
    /// from the device.
    ///
    /// This is true for every measurement this module can take on a stock
    /// Android device: dropping caches requires writing to
    /// `/proc/sys/vm/drop_caches`, which SELinux denies even to root (measured
    /// on the X88 Pro 13: "Permission denied" under `su 0`). The reading of a
    /// file just written is therefore an upper bound, and it is labelled as one
    /// rather than presented as device throughput.
    pub page_cache_may_inflate: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    Wired,
    Wireless,
    Virtual,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkCapability {
    pub name: String,
    pub kind: LinkKind,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operstate: Option<String>,
    /// Negotiated link speed where the driver reports one. Absent on many
    /// Android Wi-Fi drivers, and a negotiated rate is not a throughput.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed_mbps: Option<u64>,
}

impl CapabilityReport {
    /// Measure everything that can be measured, recording the rest in `notes`.
    ///
    /// `storage_dir` is where the storage probe is allowed to create its
    /// temporary file; pass the directory models are read from. `None` skips
    /// the probe entirely rather than guessing a location.
    pub fn collect(storage_dir: Option<&Path>, storage_bytes: u64) -> Self {
        Self::collect_inner(storage_dir, storage_bytes, true)
    }

    /// Fast inventory for mesh announcements. It refreshes dynamic memory and
    /// link measurements without running a compute benchmark on every gossip
    /// interval.
    pub fn collect_mesh_snapshot() -> Self {
        Self::collect_inner(None, 0, false)
    }

    fn collect_inner(storage_dir: Option<&Path>, storage_bytes: u64, run_probe: bool) -> Self {
        let mut notes = Vec::new();

        let cpu = measure_cpu(&mut notes, run_probe);
        let memory = measure_memory(&mut notes);
        let storage = storage_dir.and_then(|dir| {
            measure_storage_read(dir, storage_bytes, &mut notes)
        });
        let links = measure_links(&mut notes);

        Self {
            schema_version: CAPABILITY_SCHEMA_VERSION,
            measured_at_unix_secs: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            cpu,
            memory,
            storage,
            links,
            notes,
        }
    }

    /// One-line summary for a UI or a log, without any of the JSON.
    pub fn summary(&self) -> String {
        let cores = self
            .cpu
            .logical_cores
            .map(|c| c.to_string())
            .unwrap_or_else(|| "?".to_string());
        let clock = self
            .cpu
            .max_freq_khz
            .map(|k| format!("{:.2}GHz", k as f64 / 1_000_000.0))
            .unwrap_or_else(|| "?".to_string());
        let mem = self
            .memory
            .available_bytes
            .map(format_bytes)
            .unwrap_or_else(|| "?".to_string());
        let store = self
            .storage
            .as_ref()
            .map(|s| format!("{:.0}MB/s", s.read_mb_per_sec))
            .unwrap_or_else(|| "?".to_string());
        let wired = self
            .links
            .iter()
            .filter(|l| l.kind == LinkKind::Wired)
            .count();
        let wireless = self
            .links
            .iter()
            .filter(|l| l.kind == LinkKind::Wireless)
            .count();
        format!(
            "{cores} cores @ {clock}, {mem} free, storage {store}, links {wired}w/{wireless}wl"
        )
    }
}

fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2}GiB", b / GIB)
    } else {
        format!("{:.0}MiB", b / MIB)
    }
}

// ---------------------------------------------------------------------------
// CPU
// ---------------------------------------------------------------------------

fn measure_cpu(notes: &mut Vec<String>, run_probe: bool) -> CpuCapability {
    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .map_err(|e| notes.push(format!("cpu: available_parallelism failed: {e}")))
        .ok();

    let info = read_cpu_info();
    if info.is_none() {
        notes.push("cpu: /proc/cpuinfo unreadable; model unknown".to_string());
    }

    let (max_freq_khz, min_freq_khz, governor) = read_cpu_freq();
    if max_freq_khz.is_none() {
        notes.push("cpu: cpufreq sysfs unreadable; clock unknown".to_string());
    }

    let probe = if run_probe {
        run_compute_probe()
    } else {
        ComputeProbe {
            gflops: None,
            elapsed_ms: None,
            partial: false,
        }
    };

    CpuCapability {
        source: if cfg!(any(target_os = "linux", target_os = "android")) {
            "/proc/cpuinfo, /sys/devices/system/cpu, OS scheduler".to_string()
        } else {
            "OS scheduler; platform CPU identity probe unavailable".to_string()
        },
        logical_cores,
        model: info,
        max_freq_khz,
        min_freq_khz,
        governor,
        effective_gflops_fp32: probe.gflops,
        compute_probe_ms: probe.elapsed_ms,
        compute_probe_partial: probe.partial,
    }
}

/// Parse `/proc/cpuinfo` for a human-readable CPU name.
///
/// The key differs by architecture: `model name` on x86, `Hardware` on most
/// ARM SoCs, `Processor` on some older ARM kernels, and Android TV boxes are
/// inconsistent about which they carry. Split out so it is testable without a
/// device.
#[cfg(any(target_os = "linux", target_os = "android", test))]
fn parse_cpu_model(cpuinfo: &str) -> Option<String> {
    const KEYS: [&str; 3] = ["model name", "Hardware", "Processor"];
    for key in KEYS {
        for line in cpuinfo.lines() {
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            if name.trim() == key {
                let value = value.trim();
                // Some kernels report the ARM implementer id here rather than a
                // name; a bare number is not a model and is not worth carrying.
                if !value.is_empty() && value.chars().any(|c| c.is_ascii_alphabetic()) {
                    return Some(value.to_string());
                }
            }
        }
    }
    None
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn read_cpu_info() -> Option<String> {
    parse_cpu_model(&std::fs::read_to_string("/proc/cpuinfo").ok()?)
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn read_cpu_info() -> Option<String> {
    // No /proc on this platform; the field stays absent rather than being
    // filled from an environment variable that may describe something else.
    None
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn read_cpu_freq() -> (Option<u64>, Option<u64>, Option<String>) {
    const BASE: &str = "/sys/devices/system/cpu/cpu0/cpufreq";
    let read = |name: &str| -> Option<String> {
        std::fs::read_to_string(format!("{BASE}/{name}"))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };
    let parse_khz = |name: &str| read(name).and_then(|v| v.parse::<u64>().ok());
    (
        parse_khz("cpuinfo_max_freq"),
        parse_khz("cpuinfo_min_freq"),
        read("scaling_governor"),
    )
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn read_cpu_freq() -> (Option<u64>, Option<u64>, Option<String>) {
    (None, None, None)
}

struct ComputeProbe {
    gflops: Option<f64>,
    elapsed_ms: Option<u64>,
    partial: bool,
}

/// A fixed, self-contained compute probe.
///
/// It is a dependency-free multiply-add loop rather than a call into the native
/// runtime's benchmark: the runtime is absent on an expose-only node by design,
/// and this probe must run on a device that ships no GGUF and no runtime at all.
///
/// FLOP accounting is 4 per iteration (two multiply-adds, each counted as two
/// operations). The value is a single-thread lower bound on this machine under
/// its current governor, which is the honest description of what it is.
fn run_compute_probe() -> ComputeProbe {
    const CHUNK: u64 = 1 << 16;

    // Warm up so the first-touch page faults and the governor's ramp are not
    // charged to the measurement.
    let mut warm = [0.5f64; 8];
    for _ in 0..10_000 {
        for v in warm.iter_mut() {
            *v = v.mul_add(1.000_000_1, 0.000_000_1);
        }
    }
    std::hint::black_box(&warm);

    let start = Instant::now();
    let mut acc = [0.5f64; 8];
    let mut iterations: u64 = 0;
    let mut partial = false;

    'outer: loop {
        for _ in 0..CHUNK {
            for v in acc.iter_mut() {
                *v = v.mul_add(1.000_000_1, 0.000_000_1);
            }
        }
        iterations += CHUNK;
        if start.elapsed() >= COMPUTE_PROBE_BUDGET {
            partial = true;
            break 'outer;
        }
        // Bound the work even if the clock is slow enough to make CHUNK cheap:
        // the budget check above is the real limit, this only stops a pathological
        // machine from looping forever if `elapsed` misbehaves.
        if iterations >= 1 << 34 {
            break 'outer;
        }
    }

    let elapsed = start.elapsed();
    std::hint::black_box(&acc);

    let secs = elapsed.as_secs_f64();
    let flops = iterations as f64 * 8.0 * 4.0;
    let gflops = if secs > 0.0 {
        Some(flops / secs / 1e9)
    } else {
        None
    };

    ComputeProbe {
        gflops,
        elapsed_ms: Some(elapsed.as_millis() as u64),
        partial,
    }
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

#[cfg(any(target_os = "linux", target_os = "android", test))]
#[derive(Debug, Default, PartialEq)]
struct MemInfo {
    total_bytes: Option<u64>,
    available_bytes: Option<u64>,
    swap_total_bytes: Option<u64>,
}

/// Parse the three `/proc/meminfo` fields that matter here.
///
/// `MemAvailable` is the one to carry, not `MemFree`: on Android `MemFree` is
/// routinely a hundred megabytes because the page cache holds the rest, and
/// `MemAvailable` is the kernel's own estimate of what a new allocation can get
/// without swapping. Measured on the X88 Pro 13: `MemTotal` 3.82 GiB against
/// `MemFree` 0.11 GiB and `MemAvailable` 2.05 GiB.
#[cfg(any(target_os = "linux", target_os = "android", test))]
fn parse_meminfo(meminfo: &str) -> MemInfo {
    let field = |want: &str| -> Option<u64> {
        meminfo.lines().find_map(|line| {
            let (name, rest) = line.split_once(':')?;
            if name.trim() != want {
                return None;
            }
            let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            Some(kb * 1024)
        })
    };
    MemInfo {
        total_bytes: field("MemTotal"),
        available_bytes: field("MemAvailable"),
        swap_total_bytes: field("SwapTotal"),
    }
}

fn measure_memory(notes: &mut Vec<String>) -> MemoryCapability {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        match std::fs::read_to_string("/proc/meminfo") {
            Ok(text) => {
                let info = parse_meminfo(&text);
                if info.available_bytes.is_none() {
                    notes.push(
                        "memory: MemAvailable absent; pre-3.14 kernels report only MemFree"
                            .to_string(),
                    );
                }
                return MemoryCapability {
                    total_bytes: info.total_bytes,
                    available_bytes: info.available_bytes,
                    swap_total_bytes: info.swap_total_bytes,
                    source: "/proc/meminfo".to_string(),
                };
            }
            Err(error) => {
                notes.push(format!("memory: /proc/meminfo unreadable: {error}"));
            }
        }
        MemoryCapability {
            source: "/proc/meminfo (unreadable)".to_string(),
            ..Default::default()
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        notes.push(
            "memory: this module reads /proc/meminfo only; no probe is implemented for this \
             platform, so the field is absent rather than guessed"
                .to_string(),
        );
        MemoryCapability {
            source: "unimplemented for this platform".to_string(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

/// Measure sequential read throughput by writing a temporary file and reading
/// it back, then deleting it.
///
/// The write is size-checked against free space first: the probe must never be
/// the reason a full device runs out of room. Any failure removes the file and
/// records a note instead of returning a number.
fn measure_storage_read(
    dir: &Path,
    bytes: u64,
    notes: &mut Vec<String>,
) -> Option<StorageCapability> {
    let bytes = bytes.min(STORAGE_MAX_PROBE_BYTES);
    if bytes == 0 {
        return None;
    }
    if !dir.is_dir() {
        notes.push(format!(
            "storage: probe directory does not exist: {}",
            dir.display()
        ));
        return None;
    }

    let path: PathBuf = dir.join("mesh-llm-capability-probe.tmp");
    let write_result = (|| -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(&path)?;
        let chunk = vec![0xA5u8; 1024 * 1024];
        let mut written = 0u64;
        while written < bytes {
            let want = (bytes - written).min(chunk.len() as u64) as usize;
            file.write_all(&chunk[..want])?;
            written += want as u64;
        }
        file.sync_all()
    })();

    if let Err(error) = write_result {
        let _ = std::fs::remove_file(&path);
        notes.push(format!("storage: probe write failed: {error}"));
        return None;
    }

    let read_result = (|| -> std::io::Result<(u64, Duration)> {
        use std::io::Read;
        let mut file = std::fs::File::open(&path)?;
        let mut buf = vec![0u8; 1024 * 1024];
        let start = Instant::now();
        let mut total = 0u64;
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            total += n as u64;
        }
        Ok((total, start.elapsed()))
    })();

    let _ = std::fs::remove_file(&path);

    match read_result {
        Ok((total, elapsed)) => {
            let secs = elapsed.as_secs_f64();
            if secs <= 0.0 || total == 0 {
                notes.push("storage: probe read completed too fast to rate".to_string());
                return None;
            }
            Some(StorageCapability {
                path: dir.display().to_string(),
                bytes_read: total,
                elapsed_ms: elapsed.as_millis() as u64,
                read_mb_per_sec: (total as f64 / 1e6) / secs,
                // Always true here: see the field's documentation for why a
                // stock Android device cannot drop caches to avoid this.
                page_cache_may_inflate: true,
            })
        }
        Err(error) => {
            notes.push(format!("storage: probe read failed: {error}"));
            None
        }
    }
}

/// Ceiling on the storage probe regardless of what a caller asks for.
const STORAGE_MAX_PROBE_BYTES: u64 = 256 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Links
// ---------------------------------------------------------------------------

/// Classify one interface from the sysfs facts available for it.
///
/// Wireless is detected by the presence of the `wireless` directory, which is
/// what `iwconfig` uses too. Everything else that is not `lo` and not a known
/// virtual prefix is reported as wired — and that is a guess, which is why
/// `LinkKind` is documented as a hint. A concrete counterexample measured on
/// the development host: Windows classifies a Bluetooth PAN adapter as media
/// type `802.3`, i.e. wired. Nothing may depend on this for correctness.
#[cfg(any(target_os = "linux", target_os = "android", test))]
fn classify_link(name: &str, has_wireless_dir: bool, is_loopback: bool) -> LinkKind {
    if is_loopback {
        return LinkKind::Virtual;
    }
    if has_wireless_dir {
        return LinkKind::Wireless;
    }
    const VIRTUAL_PREFIXES: [&str; 8] = [
        "dummy", "tun", "tap", "veth", "br-", "docker", "virbr", "zt",
    ];
    if VIRTUAL_PREFIXES.iter().any(|p| name.starts_with(p)) {
        return LinkKind::Virtual;
    }
    if name.starts_with("lo") {
        return LinkKind::Virtual;
    }
    LinkKind::Wired
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn measure_links(notes: &mut Vec<String>) -> Vec<LinkCapability> {
    let mut links = Vec::new();
    let entries = match std::fs::read_dir("/sys/class/net") {
        Ok(entries) => entries,
        Err(error) => {
            notes.push(format!("links: /sys/class/net unreadable: {error}"));
            return links;
        }
    };
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    names.sort();

    for name in names {
        let base = PathBuf::from("/sys/class/net").join(&name);
        let read = |file: &str| -> Option<String> {
            std::fs::read_to_string(base.join(file))
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        };
        let kind = classify_link(
            &name,
            base.join("wireless").exists(),
            base.join("type").exists() && read("type").as_deref() == Some("772"),
        );
        links.push(LinkCapability {
            name,
            kind,
            source: "/sys/class/net".to_string(),
            operstate: read("operstate"),
            // Many Android Wi-Fi drivers return an error or a sentinel here, so
            // an absent value is normal rather than a fault.
            speed_mbps: read("speed").and_then(|s| s.parse::<u64>().ok()).filter(|v| *v > 0),
        });
    }
    if links.is_empty() {
        notes.push("links: no interfaces found under /sys/class/net".to_string());
    }
    links
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn measure_links(notes: &mut Vec<String>) -> Vec<LinkCapability> {
    notes.push(
        "links: this module enumerates /sys/class/net only; no probe is implemented for this \
         platform"
            .to_string(),
    );
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_meminfo_fields() {
        let text = "MemTotal:        4004328 kB\n\
                    MemFree:          112472 kB\n\
                    MemAvailable:    2152420 kB\n\
                    SwapTotal:       1201292 kB\n\
                    SwapFree:        1076108 kB\n";
        let info = parse_meminfo(text);
        assert_eq!(info.total_bytes, Some(4_004_328 * 1024));
        assert_eq!(info.available_bytes, Some(2_152_420 * 1024));
        assert_eq!(info.swap_total_bytes, Some(1_201_292 * 1024));
    }

    #[test]
    fn meminfo_without_available_is_not_a_zero() {
        // Pre-3.14 kernels have no MemAvailable. The field must stay absent
        // rather than become 0, which would read as "nothing free".
        let info = parse_meminfo("MemTotal: 1024 kB\nMemFree: 512 kB\n");
        assert_eq!(info.total_bytes, Some(1024 * 1024));
        assert_eq!(info.available_bytes, None);
    }

    #[test]
    fn parses_cpu_model_from_x86_and_arm_keys() {
        assert_eq!(
            parse_cpu_model("processor\t: 0\nmodel name\t: Intel(R) Core(TM) i7\n"),
            Some("Intel(R) Core(TM) i7".to_string())
        );
        // Keys are matched in a fixed order, `Hardware` before `Processor`, so a
        // Rockchip box that carries both reports the SoC name.
        assert_eq!(
            parse_cpu_model("Processor\t: AArch64 Processor rev 1\nHardware\t: Rockchip RK3528\n"),
            Some("Rockchip RK3528".to_string())
        );
        // A box that carries only `Processor` still yields a name.
        assert_eq!(
            parse_cpu_model("Processor\t: AArch64 Processor rev 1\n"),
            Some("AArch64 Processor rev 1".to_string())
        );
    }

    #[test]
    fn cpu_model_ignores_a_bare_numeric_id() {
        // Some ARM kernels put the implementer id here; it is not a model.
        assert_eq!(parse_cpu_model("Hardware\t: 0x00000000\n"), None);
    }

    #[test]
    fn classifies_link_kinds() {
        assert_eq!(classify_link("lo", false, true), LinkKind::Virtual);
        assert_eq!(classify_link("wlan0", true, false), LinkKind::Wireless);
        assert_eq!(classify_link("eth0", false, false), LinkKind::Wired);
        assert_eq!(classify_link("dummy0", false, false), LinkKind::Virtual);
    }

    #[test]
    fn report_serializes_with_a_schema_version() {
        let report = CapabilityReport {
            schema_version: CAPABILITY_SCHEMA_VERSION,
            measured_at_unix_secs: 1,
            cpu: CpuCapability::default(),
            memory: MemoryCapability {
                source: "/proc/meminfo".to_string(),
                ..Default::default()
            },
            storage: None,
            links: Vec::new(),
            notes: vec!["nothing measured".to_string()],
        };
        let json = serde_json::to_value(&report).expect("serialize");
        assert_eq!(json["schema_version"], CAPABILITY_SCHEMA_VERSION);
        // Absent optional fields must not appear as null: a consumer has to be
        // able to distinguish "not measured" from "measured as nothing".
        assert!(json.get("storage").is_none());
        assert!(json["cpu"].get("logical_cores").is_none());
    }

    #[test]
    fn compute_probe_reports_a_positive_rate() {
        let probe = run_compute_probe();
        let gflops = probe.gflops.expect("probe should produce a rate");
        assert!(gflops > 0.0, "rate must be positive, got {gflops}");
        assert!(probe.elapsed_ms.is_some());
    }

    #[test]
    fn storage_probe_cleans_up_after_itself() {
        let dir = std::env::temp_dir();
        let mut notes = Vec::new();
        let measured = measure_storage_read(&dir, 4 * 1024 * 1024, &mut notes);
        assert!(
            !dir.join("mesh-llm-capability-probe.tmp").exists(),
            "probe must not leave its temporary file behind"
        );
        assert!(measured.is_some(), "probe should measure the temp dir: {notes:?}");
        assert!(measured.unwrap().page_cache_may_inflate);
    }

    #[test]
    fn storage_probe_on_a_missing_dir_records_a_note() {
        let mut notes = Vec::new();
        let measured = measure_storage_read(Path::new("/nonexistent-mesh-llm-probe"), 1024, &mut notes);
        assert!(measured.is_none());
        assert!(notes.iter().any(|n| n.starts_with("storage:")), "{notes:?}");
    }
}
