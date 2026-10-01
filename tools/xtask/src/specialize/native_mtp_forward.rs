use crate::command::{DynResult, print_json};
use mesh_specialize::packages::qwen3_8_27b::native_mtp_forward_trial::{self, Fixture};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

const USAGE: &str = "usage: xtask specialize native-mtp-forward-check --artifact PATH --fixture PATH --ptx PATH --device ORDINAL --output NEW_FILE";

struct Request<'a> {
    artifact: &'a Path,
    fixture: &'a Path,
    ptx: &'a Path,
    device: i32,
    output: &'a Path,
}

impl<'a> Request<'a> {
    fn parse(args: &'a [String]) -> DynResult<Self> {
        let [
            artifact_flag,
            artifact,
            fixture_flag,
            fixture,
            ptx_flag,
            ptx,
            device_flag,
            device,
            output_flag,
            output,
        ] = args
        else {
            return Err(USAGE.into());
        };
        if [
            artifact_flag.as_str(),
            fixture_flag.as_str(),
            ptx_flag.as_str(),
            device_flag.as_str(),
            output_flag.as_str(),
        ] != ["--artifact", "--fixture", "--ptx", "--device", "--output"]
        {
            return Err(USAGE.into());
        }
        let device: i32 = device.parse()?;
        if device < 0
            || [artifact, fixture, ptx, output]
                .iter()
                .any(|path| path.is_empty())
        {
            return Err(
                "native forward request requires nonempty paths and a nonnegative device".into(),
            );
        }
        Ok(Self {
            artifact: Path::new(artifact),
            fixture: Path::new(fixture),
            ptx: Path::new(ptx),
            device,
            output: Path::new(output),
        })
    }
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let request = Request::parse(args)?;
    let passed = super::native_mtp_report::retain(
        request.output,
        json!({
            "artifact_path": request.artifact, "fixture_path": request.fixture,
            "ptx_path": request.ptx, "device": request.device,
        }),
        || {
            let raw = fs::read(request.fixture)?;
            let fixture = Fixture::parse(&raw)?;
            let ptx = fs::read_to_string(request.ptx)?;
            if !ptx.contains(".target sm_120a") || ptx.contains('\0') {
                return Err("native forward request requires SM120a PTX without NUL bytes".into());
            }
            let mut report = native_mtp_forward_trial::run(native_mtp_forward_trial::Request {
                artifact: request.artifact,
                ptx: &ptx,
                device: request.device,
                fixture: &fixture,
            })?;
            report["fixture_sha256"] = json!(hex::encode(Sha256::digest(&raw)));
            report["ptx_sha256"] = json!(hex::encode(Sha256::digest(ptx.as_bytes())));
            Ok(report)
        },
    )?;
    print_json(&json!({"output": request.output, "all_passed": passed}))?;
    if !passed {
        return Err("native forward check failed; inspect saved report".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_parses_when_flags_are_ordered() {
        let args: Vec<_> = [
            "--artifact",
            "model",
            "--fixture",
            "fixture",
            "--ptx",
            "ptx",
            "--device",
            "2",
            "--output",
            "report",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let request = Request::parse(&args).unwrap();
        assert_eq!(request.artifact, Path::new("model"));
        assert_eq!(request.fixture, Path::new("fixture"));
        assert_eq!(request.ptx, Path::new("ptx"));
        assert_eq!(request.device, 2);
        assert_eq!(request.output, Path::new("report"));
    }

    #[test]
    fn request_rejects_when_flags_device_paths_or_shape_are_invalid() {
        let valid: Vec<_> = [
            "--artifact",
            "model",
            "--fixture",
            "fixture",
            "--ptx",
            "ptx",
            "--device",
            "0",
            "--output",
            "report",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        for (index, value) in [
            (0, "--fixture"),
            (2, "--artifact"),
            (4, "--wrong"),
            (6, "--wrong"),
            (8, "--wrong"),
            (7, "-1"),
            (7, "x"),
            (7, "2147483648"),
            (1, ""),
            (3, ""),
            (5, ""),
            (9, ""),
        ] {
            let mut args = valid.clone();
            args[index] = value.into();
            assert!(Request::parse(&args).is_err());
        }
        assert!(Request::parse(&valid[..8]).is_err());
        let mut extra = valid;
        extra.push("extra".into());
        assert!(Request::parse(&extra).is_err());
    }
}
