use crate::command::{DynResult, print_json};
use mesh_specialize::packages::qwen3_8_27b::target_batch_trial::{self, Fixture, SelectedRows};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::Path,
};

const USAGE: &str = "usage: xtask specialize target-batch-decode-check --artifact PATH --fixture PATH --ptx PATH --device ORDINAL --output NEW_FILE [--rows N (1..=5)]";

struct Request<'a> {
    artifact: &'a Path,
    fixture: &'a Path,
    ptx: &'a Path,
    device: i32,
    output: &'a Path,
    selected_rows: Option<SelectedRows>,
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
            trailing @ ..,
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
        let selected_rows = match trailing {
            [] => None,
            [flag, rows] if flag == "--rows" => {
                Some(SelectedRows::try_from(rows.parse::<usize>()?)?)
            }
            _ => return Err(USAGE.into()),
        };
        let device: i32 = device.parse()?;
        if device < 0
            || [artifact, fixture, ptx, output]
                .iter()
                .any(|path| path.is_empty())
        {
            return Err(
                "target batch request requires nonempty paths and a nonnegative device".into(),
            );
        }
        Ok(Self {
            artifact: Path::new(artifact),
            fixture: Path::new(fixture),
            ptx: Path::new(ptx),
            device,
            output: Path::new(output),
            selected_rows,
        })
    }
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let request = Request::parse(args)?;
    let raw = fs::read(request.fixture)?;
    let fixture = Fixture::parse(&raw)?;
    let ptx = fs::read_to_string(request.ptx)?;
    if !ptx.contains(".target sm_120a") || ptx.contains('\0') {
        return Err("target batch request requires SM120a PTX without NUL bytes".into());
    }
    let metadata = json!({
        "artifact_path": request.artifact,
        "fixture_path": request.fixture,
        "ptx_path": request.ptx,
        "artifact_sha256": null,
        "fixture_sha256": hex::encode(Sha256::digest(&raw)),
        "ptx_sha256": hex::encode(Sha256::digest(ptx.as_bytes())),
        "device": request.device,
        "selected_rows": request.selected_rows,
    });
    let passed = retain(request.output, metadata, |metadata| {
        metadata["artifact_sha256"] = json!(file_sha256(request.artifact)?);
        Ok(target_batch_trial::run(target_batch_trial::Request {
            artifact: request.artifact,
            ptx: &ptx,
            device: request.device,
            fixture: &fixture,
            selected_rows: request.selected_rows,
        })?)
    })?;
    print_json(&json!({"output": request.output, "all_passed": passed,
        "selected_rows": request.selected_rows}))?;
    if !passed {
        return Err("target batch decode check failed; inspect saved report".into());
    }
    Ok(())
}

fn retain(
    output: &Path,
    mut metadata: Value,
    trial: impl FnOnce(&mut Value) -> DynResult<Value>,
) -> DynResult<bool> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let result = trial(&mut metadata);
    let mut report = match result {
        Ok(Value::Object(report)) => Value::Object(report),
        Ok(value) => json!({"all_passed": false, "trial_report": value,
            "error": "trial returned a non-object report"}),
        Err(error) => json!({"all_passed": false, "error": format!("{error:#}")}),
    };
    let passed = report.get("all_passed").and_then(Value::as_bool) == Some(true);
    report["all_passed"] = json!(passed);
    report["native_mtp_admitted"] = json!(false);
    report["timing_claim"] = json!(false);
    report["selected_rows"] = metadata["selected_rows"].clone();
    report["request"] = metadata;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(passed)
}

fn file_sha256(path: &Path) -> DynResult<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hex::encode(hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn trailing_rows_selects_one_case_and_rejects_invalid_selection_before_output() {
        let scratch = Scratch::new();
        let output = scratch.0.join("report.json");
        let mut args: Vec<String> = [
            "--artifact",
            "unused",
            "--fixture",
            "unused",
            "--ptx",
            "unused",
            "--device",
            "0",
            "--output",
            output.to_str().unwrap(),
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(Request::parse(&args).unwrap().selected_rows, None);
        args.extend(["--rows".to_owned(), "3".to_owned()]);
        assert_eq!(
            Request::parse(&args).unwrap().selected_rows.unwrap().get(),
            3
        );
        for rows in ["0", "6", "-1", "x", "184467440737095516160"] {
            args[11] = rows.to_owned();
            assert!(Request::parse(&args).is_err());
            assert!(run(&args).is_err());
            assert!(!output.exists());
        }
        args[11] = "1".to_owned();
        args[10] = "--wrong".to_owned();
        assert!(Request::parse(&args).is_err());
        assert!(Request::parse(&args[..11]).is_err());
        args[10] = "--rows".to_owned();
        args.push("extra".to_owned());
        assert!(Request::parse(&args).is_err());
    }

    #[test]
    fn selected_failure_preserves_phase_evidence_and_selection() {
        let scratch = Scratch::new();
        let output = scratch.0.join("report.json");
        let case = json!({"rows": 3, "passed": false,
            "verification": {"passed": false, "logits": {"differing_words": 1,
                "first_mismatch_index": 248319, "first_left_word": 10, "first_right_word": 11}},
            "continuation": {"passed": false, "batch_error": "retained failure"}});
        let passed = retain(&output, json!({"selected_rows": 3}), |_| {
            Ok(json!({
                "all_passed": false, "cases": [case.clone()],
                "native_mtp_admitted": true, "timing_claim": true,
            }))
        })
        .unwrap();
        let saved: Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
        assert!(!passed);
        assert_eq!(saved["selected_rows"], 3);
        assert_eq!(saved["request"]["selected_rows"], 3);
        assert_eq!(saved["cases"], json!([case]));
        assert_eq!(saved["native_mtp_admitted"], false);
        assert_eq!(saved["timing_claim"], false);
    }

    struct Scratch(std::path::PathBuf);
    impl Scratch {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "target-batch-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn request_rejects_bad_flags_device_and_shape() {
        let valid: Vec<String> = [
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
        assert!(Request::parse(&valid).is_ok());
        for (index, value) in [(0, "--wrong"), (7, "-1"), (7, "x"), (9, "")] {
            let mut args = valid.clone();
            args[index] = value.to_owned();
            assert!(Request::parse(&args).is_err());
        }
        assert!(Request::parse(&valid[..8]).is_err());
    }

    #[test]
    fn existing_output_prevents_trial_and_preserves_bytes() {
        let scratch = Scratch::new();
        let output = scratch.0.join("report.json");
        fs::write(&output, b"existing").unwrap();
        let result = retain(&output, json!({}), |_| panic!("trial must not run"));
        assert!(result.is_err());
        assert_eq!(fs::read(output).unwrap(), b"existing");
    }

    #[test]
    fn malformed_fixture_does_not_create_output() {
        let scratch = Scratch::new();
        let fixture = scratch.0.join("fixture.json");
        let output = scratch.0.join("report.json");
        fs::write(&fixture, b"{}").unwrap();
        let args = vec![
            "--artifact".into(),
            "unused".into(),
            "--fixture".into(),
            fixture.to_str().unwrap().into(),
            "--ptx".into(),
            "unused".into(),
            "--device".into(),
            "0".into(),
            "--output".into(),
            output.to_str().unwrap().into(),
        ];
        let result = run(&args);
        assert!(result.is_err());
        assert!(!output.exists());
    }

    #[test]
    fn failures_remain_saved_when_trial_fails_or_returns_nonboolean() {
        for value in [json!(false), json!("true"), json!(1), Value::Null] {
            let scratch = Scratch::new();
            let output = scratch.0.join("report.json");
            let passed = retain(&output, json!({"device": 0}), |_| {
                Ok(json!({
                    "all_passed": value, "cases": [{"rows": 1, "passed": false}],
                    "native_mtp_admitted": true, "timing_claim": true,
                }))
            })
            .unwrap();
            let saved: Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
            assert!(!passed);
            assert_eq!(saved["all_passed"], false);
            assert_eq!(saved["cases"][0]["rows"], 1);
            assert_eq!(saved["native_mtp_admitted"], false);
            assert_eq!(saved["timing_claim"], false);
        }
    }

    #[test]
    fn trial_error_is_retained_as_failure() {
        let scratch = Scratch::new();
        let output = scratch.0.join("report.json");
        let passed = retain(&output, json!({}), |_| Err("load failed".into())).unwrap();
        let saved: Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
        assert!(!passed);
        assert_eq!(saved["all_passed"], false);
        assert_eq!(saved["error"], "load failed");
    }

    #[test]
    fn missing_status_or_nonobject_report_fails_closed() {
        for value in [json!({"cases": []}), json!([true]), json!(true)] {
            let scratch = Scratch::new();
            let output = scratch.0.join("report.json");
            let passed = retain(&output, json!({}), |_| Ok(value)).unwrap();
            let saved: Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
            assert!(!passed);
            assert_eq!(saved["all_passed"], false);
        }
    }

    #[test]
    fn explicit_true_succeeds_after_report_is_saved() {
        let scratch = Scratch::new();
        let output = scratch.0.join("report.json");
        let passed = retain(&output, json!({"device": 2}), |_| {
            Ok(json!({
                "all_passed": true, "native_mtp_admitted": true, "timing_claim": true,
            }))
        })
        .unwrap();
        let saved: Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
        assert!(passed);
        assert_eq!(saved["all_passed"], true);
        assert_eq!(saved["native_mtp_admitted"], false);
        assert_eq!(saved["timing_claim"], false);
        assert_eq!(saved["request"]["device"], 2);
    }
}
