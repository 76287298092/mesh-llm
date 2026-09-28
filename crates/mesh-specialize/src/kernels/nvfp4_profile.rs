//! Process-fixed NVFP4 prefill scheduling experiments. Default arithmetic is retained.
use anyhow::{Result, anyhow, bail};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Baseline,
    TiledPrefill,
}

pub struct Schedule {
    pub kernel: &'static str,
    pub tile_rows: usize,
    pub tile_columns: usize,
    pub threads: u32,
}

impl Profile {
    pub fn name(self) -> &'static str {
        match self {
            Self::Baseline => "nvfp4-native-prefill-integer-decode-v1",
            Self::TiledPrefill => "nvfp4-tiled32-prefill-integer-decode-v1",
        }
    }

    pub fn schedule(self, rows: usize, columns: usize, width: usize) -> Schedule {
        let (kernel, tile_rows, tile_columns, threads) = if rows == 1 {
            ("nvfp4_decode_exact", 1, 4, 128)
        } else if self == Self::TiledPrefill
            && (16..=512).contains(&rows)
            && (8..=32768).contains(&columns)
            && columns.is_multiple_of(8)
            && (64..=32768).contains(&width)
            && width.is_multiple_of(64)
        {
            ("nvfp4_prefill_tiled", 32, 32, 256)
        } else {
            ("nvfp4_linear", 16, 8, 32)
        };
        Schedule {
            kernel,
            tile_rows,
            tile_columns,
            threads,
        }
    }
}

fn parse(value: Option<&str>) -> Result<Profile> {
    match value {
        None | Some("baseline") => Ok(Profile::Baseline),
        Some("tiled-prefill") => Ok(Profile::TiledPrefill),
        _ => bail!("MESH_SPECIALIZE_NVFP4_PROFILE must be baseline or tiled-prefill"),
    }
}

pub fn current() -> Result<Profile> {
    static PROFILE: OnceLock<Result<Profile, String>> = OnceLock::new();
    match PROFILE.get_or_init(|| match std::env::var("MESH_SPECIALIZE_NVFP4_PROFILE") {
        Ok(value) => parse(Some(&value)).map_err(|e| e.to_string()),
        Err(std::env::VarError::NotPresent) => parse(None).map_err(|e| e.to_string()),
        Err(e) => Err(e.to_string()),
    }) {
        Ok(profile) => Ok(*profile),
        Err(error) => Err(anyhow!(error.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_and_small_batches_retain_prior_dispatch() {
        assert_eq!(parse(None).unwrap(), Profile::Baseline);
        assert!(parse(Some("tiled")).is_err());
        assert_eq!(parse(Some("tiled-prefill")).unwrap(), Profile::TiledPrefill);
        for profile in [Profile::Baseline, Profile::TiledPrefill] {
            assert_eq!(
                profile.schedule(1, 5120, 17408).kernel,
                "nvfp4_decode_exact"
            );
            assert_eq!(profile.schedule(5, 5120, 17408).kernel, "nvfp4_linear");
        }
    }

    #[test]
    fn tiled_dispatch_is_bounded_and_falls_back_outside_admission() {
        for shape in [[16, 8, 64], [33, 40, 192], [512, 17408, 5120]] {
            let s = Profile::TiledPrefill.schedule(shape[0], shape[1], shape[2]);
            assert_eq!(
                (s.kernel, s.tile_rows, s.tile_columns, s.threads),
                ("nvfp4_prefill_tiled", 32, 32, 256)
            );
        }
        for shape in [[513, 8, 64], [16, 9, 64], [16, 8, 80], [16, 32776, 64]] {
            assert_eq!(
                Profile::TiledPrefill
                    .schedule(shape[0], shape[1], shape[2])
                    .kernel,
                "nvfp4_linear"
            );
        }
    }
}
