use crate::command::{DynResult, print_json};
use mesh_specialize::kernels::{NativeMtpQ8ProjectionResidentRequest, Q8ProjectionCaseRange};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::Instant,
};

const USAGE: &str = "usage: xtask specialize native-mtp-q8-projection-resident-check --artifact PATH --ptx PATH --device ORDINAL --output NEW_FILE [--case-range START END]";

struct Args<'a> {
    artifact: &'a str,
    ptx: &'a str,
    device: i32,
    output: &'a str,
    case_range: Q8ProjectionCaseRange,
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
    ] = args.get(..8).ok_or(USAGE)?
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
    let case_range = match args.get(8..) {
        Some([]) => Q8ProjectionCaseRange::all(),
        Some([range_flag, start, end]) if range_flag == "--case-range" => {
            let start = start
                .parse::<usize>()
                .map_err(|_| "--case-range START must be an integer in 0..16")?;
            let end = end
                .parse::<usize>()
                .map_err(|_| "--case-range END must be an integer in 1..=16")?;
            Q8ProjectionCaseRange::new(start, end)?
        }
        _ => return Err(USAGE.into()),
    };
    Ok(Args {
        artifact,
        ptx,
        device,
        output,
        case_range,
    })
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    run_with_trial(args, |request| {
        mesh_specialize::kernels::native_mtp_q8_projection_resident_trial_range(request)
            .map_err(Into::into)
    })
}

fn run_with_trial(
    args: &[String],
    trial: impl FnOnce(
        NativeMtpQ8ProjectionResidentRequest<'_>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>>,
) -> DynResult<()> {
    let args = parse_args(args)?;
    let ptx = fs::read_to_string(args.ptx)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(Path::new(args.output))?;
    let started = Instant::now();
    let request = NativeMtpQ8ProjectionResidentRequest {
        artifact: Path::new(args.artifact),
        ptx: &ptx,
        device: args.device,
        case_range: args.case_range,
    };
    let (mut report, failure) = match trial(request) {
        Ok(report) => (report, None),
        Err(error) => {
            let message = format!("{error:#}");
            (
                json!({
                    "kind": "native-mtp-q8-resident-projection-matrix-qualification-v1",
                    "all_passed": false,
                    "error": message,
                }),
                Some(message),
            )
        }
    };
    let report_object = report
        .as_object_mut()
        .ok_or("qualification report must be a JSON object")?;
    report_object.insert(
        "elapsed_seconds".into(),
        json!(started.elapsed().as_secs_f64()),
    );
    report_object.insert("artifact_path".into(), json!(args.artifact));
    report_object.insert("ptx_path".into(), json!(args.ptx));
    report_object.insert(
        "ptx_sha256".into(),
        json!(hex::encode(Sha256::digest(ptx.as_bytes()))),
    );
    report_object.insert("device_ordinal".into(), json!(args.device));
    report_object.insert(
        "selected_case_range".into(),
        json!({
            "start": args.case_range.start(),
            "end": args.case_range.end(),
        }),
    );
    report_object.insert(
        "selected_case_indices".into(),
        json!(args.case_range.indices().collect::<Vec<_>>()),
    );
    report_object.insert("native_mtp_admitted".into(), json!(false));
    report_object.insert("model_executable".into(), json!(false));
    report_object.insert("full_model_executed".into(), json!(false));
    report_object.insert("timing_claim".into(), json!(false));
    report_object.insert("timing_collected".into(), json!(false));
    report_object.insert("model_prefill_tokens_per_second".into(), Value::Null);
    report_object.insert("model_decode_tokens_per_second".into(), Value::Null);
    let all_passed = report_object.get("all_passed") == Some(&Value::Bool(true));
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    print_json(&json!({"output": args.output, "all_passed": all_passed}))?;
    if !all_passed {
        let message = failure.unwrap_or_else(|| {
            "native MTP Q8 resident projection qualification failed; inspect saved report".into()
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

    fn args(output: &Path, ptx: &Path) -> Vec<String> {
        vec![
            "--artifact".into(),
            "model.ninfer".into(),
            "--ptx".into(),
            ptx.display().to_string(),
            "--device".into(),
            "0".into(),
            "--output".into(),
            output.display().to_string(),
        ]
    }

    #[test]
    fn parser_rejects_missing_extra_reordered_and_invalid_device_arguments() {
        let valid = args(Path::new("unused.json"), Path::new("kernel.ptx"));
        let parsed = parse_args(&valid).expect("default arguments are valid");
        assert_eq!(parsed.case_range.start(), 0);
        assert_eq!(parsed.case_range.end(), 16);
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

        let mut ranged = valid.clone();
        ranged.extend(["--case-range".into(), "5".into(), "11".into()]);
        let parsed = parse_args(&ranged).expect("trailing case range is valid");
        assert_eq!(
            parsed.case_range.indices().collect::<Vec<_>>(),
            (5..11).collect::<Vec<_>>()
        );
        for (start, end) in [("0", "0"), ("4", "3"), ("0", "17"), ("16", "17")] {
            let mut invalid_range = valid.clone();
            invalid_range.extend(["--case-range".into(), start.into(), end.into()]);
            assert!(parse_args(&invalid_range).is_err());
        }
        let mut reordered_range = ranged;
        reordered_range.swap(8, 0);
        assert!(parse_args(&reordered_range).is_err());
    }

    #[test]
    fn malformed_arguments_do_not_create_a_report() -> crate::command::DynResult<()> {
        let directory = unique_temp_dir("native-mtp-q8-projection-args");
        fs::create_dir(&directory)?;
        let output = directory.join("report.json");

        let no_args: Vec<String> = Vec::new();
        let result = run_with_trial(&no_args, |_| Ok(json!({"all_passed": true})));

        assert!(result.is_err());
        assert!(!output.exists());
        fs::remove_dir(directory)?;
        Ok(())
    }

    #[test]
    fn invalid_case_range_does_not_create_a_report_or_start_trial() -> crate::command::DynResult<()>
    {
        let directory = unique_temp_dir("native-mtp-q8-projection-range-args");
        fs::create_dir(&directory)?;
        let output = directory.join("report.json");
        let ptx = directory.join("kernel.ptx");
        fs::write(&ptx, ".version 8.7\n")?;
        let mut command_args = args(&output, &ptx);
        command_args.extend(["--case-range".into(), "16".into(), "16".into()]);
        let trial_started = std::cell::Cell::new(false);

        let result = run_with_trial(&command_args, |_| {
            trial_started.set(true);
            Ok(json!({"all_passed": true}))
        });

        assert!(result.is_err());
        assert!(!trial_started.get());
        assert!(!output.exists());
        fs::remove_file(ptx)?;
        fs::remove_dir(directory)?;
        Ok(())
    }

    #[test]
    fn selected_case_range_is_recorded_with_original_indices_and_failure()
    -> crate::command::DynResult<()> {
        let directory = unique_temp_dir("native-mtp-q8-projection-range");
        fs::create_dir(&directory)?;
        let output = directory.join("report.json");
        let ptx = directory.join("kernel.ptx");
        fs::write(&ptx, ".version 8.7\n")?;
        let mut command_args = args(&output, &ptx);
        command_args.extend(["--case-range".into(), "5".into(), "11".into()]);

        let result = run_with_trial(&command_args, |request| {
            let indices = request.case_range.indices().collect::<Vec<_>>();
            assert_eq!(indices, (5..11).collect::<Vec<_>>());
            let cases = indices
                .iter()
                .map(|case_index| {
                    let passed = *case_index != 8;
                    json!({
                        "case_index": case_index,
                        "all_passed": passed,
                        "error": (!passed).then_some("case 8 failed"),
                    })
                })
                .collect::<Vec<_>>();
            Ok(json!({"all_passed": false, "cases": cases}))
        });

        assert!(result.is_err());
        let report: Value = serde_json::from_slice(&fs::read(&output)?)?;
        assert_eq!(report["selected_case_range"]["start"], 5);
        assert_eq!(report["selected_case_range"]["end"], 11);
        assert_eq!(report["selected_case_indices"], json!([5, 6, 7, 8, 9, 10]));
        assert_eq!(
            report["cases"].as_array().map(|cases| cases
                .iter()
                .map(|case| case["case_index"].as_u64())
                .collect::<Vec<_>>()),
            Some(vec![Some(5), Some(6), Some(7), Some(8), Some(9), Some(10)])
        );
        assert_eq!(report["cases"][3]["all_passed"], false);
        assert!(report["cases"][3]["error"].is_string());
        fs::remove_file(output)?;
        fs::remove_file(ptx)?;
        fs::remove_dir(directory)?;
        Ok(())
    }

    #[test]
    fn existing_report_path_is_preserved_before_trial_starts() -> crate::command::DynResult<()> {
        let directory = unique_temp_dir("native-mtp-q8-projection-existing");
        fs::create_dir(&directory)?;
        let output = directory.join("report.json");
        let ptx = directory.join("kernel.ptx");
        fs::write(&ptx, ".version 8.7\n")?;
        fs::write(&output, "prior evidence\n")?;
        let command_args = args(&output, &ptx);
        let trial_started = std::cell::Cell::new(false);

        let result = run_with_trial(&command_args, |_| {
            trial_started.set(true);
            Ok(json!({"all_passed": true}))
        });

        assert!(result.is_err());
        assert!(!trial_started.get());
        assert_eq!(fs::read_to_string(&output)?, "prior evidence\n");
        fs::remove_file(output)?;
        fs::remove_file(ptx)?;
        fs::remove_dir(directory)?;
        Ok(())
    }

    #[test]
    fn failed_cases_are_all_saved_and_return_failure() -> crate::command::DynResult<()> {
        let directory = unique_temp_dir("native-mtp-q8-projection-failed");
        fs::create_dir(&directory)?;
        let output = directory.join("report.json");
        let ptx = directory.join("kernel.ptx");
        fs::write(&ptx, ".version 8.7\n")?;
        let command_args = args(&output, &ptx);
        let cases = (0..16)
            .map(|case_index| {
                let passed = case_index != 3 && case_index != 11;
                json!({
                    "case_index": case_index,
                    "all_passed": passed,
                    "error": (!passed).then_some(format!("case {case_index} failed")),
                })
            })
            .collect::<Vec<_>>();

        let result = run_with_trial(&command_args, |_| {
            Ok(json!({
                "all_passed": false,
                "cases": cases,
                "native_mtp_admitted": true,
                "model_executable": true,
                "full_model_executed": true,
                "timing_claim": true,
                "timing_collected": true,
                "model_prefill_tokens_per_second": 99.0,
                "model_decode_tokens_per_second": 99.0,
            }))
        });

        assert!(result.is_err());
        let report: Value = serde_json::from_slice(&fs::read(&output)?)?;
        assert_eq!(report["cases"].as_array().map(Vec::len), Some(16));
        assert_eq!(report["selected_case_range"]["start"], 0);
        assert_eq!(report["selected_case_range"]["end"], 16);
        assert_eq!(report["selected_case_indices"][15], 15);
        assert_eq!(report["cases"][3]["error"], "case 3 failed");
        assert_eq!(report["cases"][11]["error"], "case 11 failed");
        assert_eq!(report["native_mtp_admitted"], false);
        assert_eq!(report["model_executable"], false);
        assert_eq!(report["full_model_executed"], false);
        assert_eq!(report["timing_claim"], false);
        assert_eq!(report["timing_collected"], false);
        assert_eq!(report["model_prefill_tokens_per_second"], Value::Null);
        assert_eq!(report["model_decode_tokens_per_second"], Value::Null);
        fs::remove_file(output)?;
        fs::remove_file(ptx)?;
        fs::remove_dir(directory)?;
        Ok(())
    }

    #[test]
    fn trial_errors_are_saved_before_the_command_returns_failure() -> crate::command::DynResult<()>
    {
        let directory = unique_temp_dir("native-mtp-q8-projection-error");
        fs::create_dir(&directory)?;
        let output = directory.join("report.json");
        let ptx = directory.join("kernel.ptx");
        fs::write(&ptx, ".version 8.7\n")?;
        let command_args = args(&output, &ptx);

        let result = run_with_trial(&command_args, |_| {
            Err(std::io::Error::other("projection matrix failed").into())
        });

        assert!(result.is_err());
        let report: Value = serde_json::from_slice(&fs::read(&output)?)?;
        assert_eq!(report["all_passed"], false);
        assert_eq!(report["error"], "projection matrix failed");
        fs::remove_file(output)?;
        fs::remove_file(ptx)?;
        fs::remove_dir(directory)?;
        Ok(())
    }

    #[test]
    fn explicit_true_all_passed_report_succeeds() -> crate::command::DynResult<()> {
        let directory = unique_temp_dir("native-mtp-q8-projection-passed");
        fs::create_dir(&directory)?;
        let output = directory.join("report.json");
        let ptx = directory.join("kernel.ptx");
        fs::write(&ptx, ".version 8.7\n")?;
        let command_args = args(&output, &ptx);

        let result = run_with_trial(&command_args, |_| Ok(json!({"all_passed": true})));

        assert!(result.is_ok());
        let report: Value = serde_json::from_slice(&fs::read(&output)?)?;
        assert_eq!(report["all_passed"], true);
        fs::remove_file(output)?;
        fs::remove_file(ptx)?;
        fs::remove_dir(directory)?;
        Ok(())
    }
}
