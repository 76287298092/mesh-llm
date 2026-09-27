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
        mesh_specialize::checkpoint::qwen3_8_intake::convert(Path::new(input), Path::new(output));
    let (report, failure) = match result {
        Ok(evidence) => (
            json!({
                "all_passed": true,
                "elapsed_seconds": started.elapsed().as_secs_f64(),
                "artifact_path": output,
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
                    "error": message,
                    "model_executable": false,
                }),
                Some(message),
            )
        }
    };
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    print_json(&json!({"report": report_path, "all_passed": report["all_passed"]}))?;
    if let Some(message) = failure {
        return Err(message.into());
    }
    Ok(())
}

fn usage() -> &'static str {
    "usage: xtask specialize checkpoint-import --input-directory PATH --output NEW_FILE --report NEW_FILE"
}
