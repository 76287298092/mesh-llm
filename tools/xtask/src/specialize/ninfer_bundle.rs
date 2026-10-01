//! CPU-only intermediate bundle assembly; deliberately no runtime invocation.

use crate::command::{DynResult, print_json};
use serde_json::json;
use std::{fs::OpenOptions, io::Write, path::Path, time::Instant};

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [
        input_flag,
        input,
        output_flag,
        output,
        report_flag,
        report_path,
    ] = args
    else {
        return Err(usage().into());
    };
    if input_flag != "--input-directory" || output_flag != "--output" || report_flag != "--report" {
        return Err(usage().into());
    }
    if output == report_path {
        return Err("artifact and report paths must differ".into());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(report_path)?;
    let started = Instant::now();
    let result =
        mesh_specialize::checkpoint::ninfer_bundle::convert(Path::new(input), Path::new(output));
    let (report, failure) = match result {
        Ok(evidence) => (
            json!({
                "all_passed": true,
                "elapsed_seconds": started.elapsed().as_secs_f64(),
                "artifact_path": output,
                "model_executable": false,
                "evidence": evidence,
            }),
            None,
        ),
        Err(error) => {
            let message = format!("{error:#}");
            (
                json!({
                    "all_passed": false,
                    "elapsed_seconds": started.elapsed().as_secs_f64(),
                    "artifact_path": output,
                    "model_executable": false,
                    "error": message,
                }),
                Some(message),
            )
        }
    };
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    print_json(
        &json!({"report": report_path, "all_passed": report["all_passed"], "model_executable": false}),
    )?;
    if let Some(message) = failure {
        return Err(message.into());
    }
    Ok(())
}

fn usage() -> &'static str {
    "usage: xtask specialize ninfer-bundle-import --input-directory PATH --output NEW_FILE --report NEW_FILE"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_arguments_do_not_create_output() {
        assert!(run(&[]).is_err());
        let args = [
            "--input-directory",
            "in",
            "--output",
            "same",
            "--report",
            "same",
        ]
        .map(str::to_owned);
        assert!(run(&args).unwrap_err().to_string().contains("must differ"));
    }

    #[test]
    fn report_overwrite_is_refused() {
        let mut unique = [0; 16];
        getrandom::fill(&mut unique).unwrap();
        let directory = std::env::temp_dir().join(format!("ninfer-cli-{}", hex::encode(unique)));
        std::fs::create_dir(&directory).unwrap();
        let report = directory.join("report.json");
        let artifact = directory.join("out.mspec");
        std::fs::write(&report, b"keep original report").unwrap();
        let args = vec![
            "--input-directory".into(),
            directory.display().to_string(),
            "--output".into(),
            artifact.display().to_string(),
            "--report".into(),
            report.display().to_string(),
        ];
        assert!(run(&args).is_err());
        assert_eq!(std::fs::read(report).unwrap(), b"keep original report");
        assert!(!artifact.exists());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
