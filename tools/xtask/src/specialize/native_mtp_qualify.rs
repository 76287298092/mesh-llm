use crate::command::{DynResult, print_json};
use mesh_specialize::packages::qwen3_8_27b::native_source::mtp_qualification;
use serde_json::json;
use std::{fs::OpenOptions, io::Write, path::Path, time::Instant};

const USAGE: &str = "usage: xtask specialize native-mtp-qualify --artifact PATH --output NEW_FILE";

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [artifact_flag, artifact, output_flag, output] = args else {
        return Err(USAGE.into());
    };
    if artifact_flag != "--artifact" || output_flag != "--output" {
        return Err(USAGE.into());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let started = Instant::now();
    let (mut report, failure) = match mtp_qualification::qualify(Path::new(artifact)) {
        Ok(report) => (report, None),
        Err(error) => {
            let message = format!("{error:#}");
            (
                json!({
                    "schema_version": 1,
                    "kind": "native-mtp-physical-parent-qualification",
                    "all_passed": false,
                    "native_mtp_admitted": false,
                    "model_executable": false,
                    "error": message,
                }),
                Some(message),
            )
        }
    };
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    report["artifact_path"] = json!(artifact);
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    print_json(&json!({"output": output, "all_passed": failure.is_none()}))?;
    if let Some(message) = failure {
        return Err(message.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualification_rejects_missing_arguments() {
        assert!(run(&[]).is_err());
    }

    #[test]
    fn qualification_rejects_unknown_flags_before_opening_files() {
        let args = ["--input", "missing.ninfer", "--output", "unused.json"].map(str::to_owned);
        assert!(run(&args).unwrap_err().to_string().contains("usage:"));
    }
}
