//! Explicit operator checks, including direct verified native-artifact weights.
use super::probe;
use crate::command::DynResult;
use std::path::Path;

pub(super) fn synthetic(args: &[String]) -> DynResult<()> {
    probe::run_probe(args, mesh_specialize::kernels::nvfp4_prmt_trial)
}

pub(super) fn real(args: &[String]) -> DynResult<()> {
    let [artifact_flag, artifact, rest @ ..] = args else {
        return Err("usage: xtask specialize nvfp4-prmt-real-check --artifact PATH --ptx PATH --device ORDINAL --output NEW_FILE".into());
    };
    if artifact_flag != "--artifact" {
        return Err(
            "nvfp4-prmt-real-check requires --artifact first, then --ptx, --device, --output"
                .into(),
        );
    }
    // Reuse the established exclusive-create report writer, PTX hashing, device
    // parsing, saved error report, and nonzero exit on an unsuccessful check.
    probe::run_probe(rest, |ptx, device| {
        mesh_specialize::kernels::nvfp4_prmt_real_trial(Path::new(artifact), ptx, device)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn invalid_arguments_are_rejected_before_file_or_gpu_access() {
        assert!(synthetic(&[]).is_err());
        assert!(real(&[]).is_err());
        assert!(real(&args(&["--ptx", "unused"])).is_err());
        assert!(
            real(&args(&[
                "--artifact",
                "unused",
                "--device",
                "0",
                "--ptx",
                "unused",
                "--output",
                "unused",
            ]))
            .is_err()
        );
    }
}
