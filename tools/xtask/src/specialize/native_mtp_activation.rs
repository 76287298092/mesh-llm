use crate::command::{DynResult, print_json};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

const USAGE: &str = "usage: xtask specialize native-mtp-activation-check --ptx PATH --device ORDINAL --output NEW_FILE";

struct Request<'a> {
    ptx: &'a Path,
    device: i32,
    output: &'a Path,
}

impl<'a> Request<'a> {
    fn parse(args: &'a [String]) -> DynResult<Self> {
        let [ptx_flag, ptx, device_flag, device, output_flag, output] = args else {
            return Err(USAGE.into());
        };
        if [
            ptx_flag.as_str(),
            device_flag.as_str(),
            output_flag.as_str(),
        ] != ["--ptx", "--device", "--output"]
        {
            return Err(USAGE.into());
        }
        let device: i32 = device.parse()?;
        if device < 0 || ptx.is_empty() || output.is_empty() {
            return Err(
                "activation request requires nonempty paths and a nonnegative device".into(),
            );
        }
        Ok(Self {
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
            "ptx_path": request.ptx, "device": request.device,
        }),
        || {
            let ptx = fs::read_to_string(request.ptx)?;
            if !ptx.contains(".target sm_120a") || ptx.contains('\0') {
                return Err("activation request requires SM120a PTX without NUL bytes".into());
            }
            let mut report =
                mesh_specialize::kernels::native_mtp_activation_check(&ptx, request.device)?;
            report["ptx_sha256"] = json!(hex::encode(Sha256::digest(ptx.as_bytes())));
            Ok(report)
        },
    )?;
    print_json(&json!({"output": request.output, "all_passed": passed}))?;
    if !passed {
        return Err("activation check failed; inspect saved report".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_parses_when_flags_are_ordered() {
        let args: Vec<_> = ["--ptx", "ptx", "--device", "3", "--output", "report"]
            .into_iter()
            .map(String::from)
            .collect();
        let request = Request::parse(&args).unwrap();
        assert_eq!(request.ptx, Path::new("ptx"));
        assert_eq!(request.device, 3);
        assert_eq!(request.output, Path::new("report"));
    }

    #[test]
    fn request_rejects_when_flags_device_paths_or_shape_are_invalid() {
        let valid: Vec<_> = ["--ptx", "ptx", "--device", "0", "--output", "report"]
            .into_iter()
            .map(String::from)
            .collect();
        for (index, value) in [
            (0, "--output"),
            (2, "--wrong"),
            (4, "--ptx"),
            (3, "-1"),
            (3, "x"),
            (3, "2147483648"),
            (1, ""),
            (5, ""),
        ] {
            let mut args = valid.clone();
            args[index] = value.into();
            assert!(Request::parse(&args).is_err());
        }
        assert!(Request::parse(&valid[..4]).is_err());
        let mut extra = valid;
        extra.push("extra".into());
        assert!(Request::parse(&extra).is_err());
    }
}
