use crate::command::{DynResult, print_json};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::Instant,
};

const USAGE: &str = "usage: xtask specialize native-mtp-q8-fc-resident-check --artifact PATH --ptx PATH --device ORDINAL --output NEW_FILE";

struct Args<'a> {
    artifact: &'a str,
    ptx: &'a str,
    device: i32,
    output: &'a str,
}

fn parse_args(args: &[String]) -> DynResult<Args<'_>> {
    let [
        artifact_flag,
        artifact,
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
    if artifact_flag != "--artifact"
        || ptx_flag != "--ptx"
        || device_flag != "--device"
        || output_flag != "--output"
    {
        return Err(USAGE.into());
    }
    let device = device
        .parse::<i32>()
        .map_err(|_| "--device must be a nonnegative integer ordinal")?;
    if device < 0 {
        return Err("--device must be a nonnegative integer ordinal".into());
    }
    Ok(Args {
        artifact,
        ptx,
        device,
        output,
    })
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    run_with_trial(args, |artifact, ptx, device| {
        mesh_specialize::kernels::native_mtp_q8_fc_resident_trial(artifact, ptx, device)
            .map_err(Into::into)
    })
}

fn run_with_trial(
    args: &[String],
    trial: impl FnOnce(&Path, &str, i32) -> Result<Value, Box<dyn std::error::Error + Send + Sync>>,
) -> DynResult<()> {
    let args = parse_args(args)?;
    let ptx = fs::read_to_string(args.ptx)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(Path::new(args.output))?;
    let started = Instant::now();
    let (mut report, failure) = match trial(Path::new(args.artifact), &ptx, args.device) {
        Ok(report) => (report, None),
        Err(error) => {
            let message = format!("{error:#}");
            (
                json!({
                    "kind": "native-mtp-q8-fc-resident-qualification",
                    "all_passed": false,
                    "error": message,
                }),
                Some(message),
            )
        }
    };
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    report["artifact_path"] = json!(args.artifact);
    report["ptx_path"] = json!(args.ptx);
    report["ptx_sha256"] = json!(hex::encode(Sha256::digest(ptx.as_bytes())));
    report["device_ordinal"] = json!(args.device);
    let all_passed = report["all_passed"] == true;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    print_json(&json!({
        "output": args.output,
        "all_passed": all_passed,
    }))?;
    if !all_passed {
        let message = failure.unwrap_or_else(|| {
            "native MTP Q8 FC resident qualification failed; inspect saved report".into()
        });
        return Err(message.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{parse_args, run_with_trial};
    use crate::command::unique_temp_dir;
    use serde_json::{Value, json};
    use std::{fs, path::Path};

    fn args(output: &Path) -> Vec<String> {
        vec![
            "--artifact".into(),
            "artifact.ninfer".into(),
            "--ptx".into(),
            "kernel.ptx".into(),
            "--device".into(),
            "0".into(),
            "--output".into(),
            output.display().to_string(),
        ]
    }

    #[test]
    fn parser_rejects_missing_extra_reordered_and_invalid_device_arguments() {
        let output = Path::new("unused.json");
        let valid = args(output);
        assert!(parse_args(&valid).is_ok());
        assert!(parse_args(&valid[..7]).is_err());
        let mut extra = valid.clone();
        extra.push("unexpected".into());
        assert!(parse_args(&extra).is_err());
        let mut reordered = valid.clone();
        reordered.swap(0, 2);
        assert!(parse_args(&reordered).is_err());
        let mut invalid_device = valid.clone();
        invalid_device[5] = "ordinal".into();
        assert!(parse_args(&invalid_device).is_err());
        invalid_device[5] = "-1".into();
        assert!(parse_args(&invalid_device).is_err());
    }

    #[test]
    fn malformed_arguments_do_not_create_the_report_path() -> crate::command::DynResult<()> {
        let directory = unique_temp_dir("native-mtp-fc-resident-args");
        fs::create_dir(&directory)?;
        let output = directory.join("report.json");
        let mut malformed = args(&output);
        malformed[0] = "--ptx".into();

        let result = run_with_trial(&malformed, |_, _, _| Ok(json!({"all_passed": true})));
        assert!(result.is_err());
        assert!(!output.exists());
        fs::remove_dir(directory)?;
        Ok(())
    }

    #[test]
    fn returned_failed_report_is_saved_and_returns_error() -> crate::command::DynResult<()> {
        let directory = unique_temp_dir("native-mtp-fc-resident-failed-report");
        fs::create_dir(&directory)?;
        let ptx_path = directory.join("kernel.ptx");
        let output = directory.join("report.json");
        fs::write(&ptx_path, ".version 8.7\n")?;
        let mut command_args = args(&output);
        command_args[3] = ptx_path.display().to_string();
        let result = run_with_trial(&command_args, |_, _, _| {
            Ok(json!({"all_passed": false, "cases": [{"error": "mismatch"}]}))
        });

        assert!(result.is_err());
        let report: Value = serde_json::from_slice(&fs::read(&output)?)?;
        assert_eq!(report["all_passed"], false);
        assert_eq!(report["cases"][0]["error"], "mismatch");
        fs::remove_file(output)?;
        fs::remove_file(ptx_path)?;
        fs::remove_dir(directory)?;
        Ok(())
    }

    #[test]
    fn report_without_true_all_passed_is_not_success() -> crate::command::DynResult<()> {
        let directory = unique_temp_dir("native-mtp-fc-resident-missing-status");
        fs::create_dir(&directory)?;
        let ptx_path = directory.join("kernel.ptx");
        let output = directory.join("report.json");
        fs::write(&ptx_path, ".version 8.7\n")?;
        let mut command_args = args(&output);
        command_args[3] = ptx_path.display().to_string();
        let result = run_with_trial(&command_args, |_, _, _| Ok(json!({"cases": []})));

        assert!(result.is_err());
        let report: Value = serde_json::from_slice(&fs::read(&output)?)?;
        assert!(report.get("all_passed").is_none());
        fs::remove_file(output)?;
        fs::remove_file(ptx_path)?;
        fs::remove_dir(directory)?;
        Ok(())
    }

    #[test]
    fn report_with_true_all_passed_succeeds() -> crate::command::DynResult<()> {
        let directory = unique_temp_dir("native-mtp-fc-resident-passed-report");
        fs::create_dir(&directory)?;
        let ptx_path = directory.join("kernel.ptx");
        let output = directory.join("report.json");
        fs::write(&ptx_path, ".version 8.7\n")?;
        let mut command_args = args(&output);
        command_args[3] = ptx_path.display().to_string();
        let result = run_with_trial(&command_args, |_, _, _| Ok(json!({"all_passed": true})));

        assert!(result.is_ok());
        fs::remove_file(output)?;
        fs::remove_file(ptx_path)?;
        fs::remove_dir(directory)?;
        Ok(())
    }

    #[test]
    fn trial_error_is_saved_and_returns_error() -> crate::command::DynResult<()> {
        let directory = unique_temp_dir("native-mtp-fc-resident-trial-error");
        fs::create_dir(&directory)?;
        let ptx_path = directory.join("kernel.ptx");
        let output = directory.join("report.json");
        fs::write(&ptx_path, ".version 8.7\n")?;
        let mut command_args = args(&output);
        command_args[3] = ptx_path.display().to_string();
        let result = run_with_trial(&command_args, |_, _, _| {
            Err(std::io::Error::other("resident fixture failed").into())
        });

        assert!(result.is_err());
        let report: Value = serde_json::from_slice(&fs::read(&output)?)?;
        assert_eq!(report["all_passed"], false);
        assert_eq!(report["error"], "resident fixture failed");
        fs::remove_file(output)?;
        fs::remove_file(ptx_path)?;
        fs::remove_dir(directory)?;
        Ok(())
    }
}
