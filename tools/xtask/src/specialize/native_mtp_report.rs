use crate::command::DynResult;
use serde_json::{Value, json};
use std::{fs::OpenOptions, io::Write, path::Path};

pub(super) fn retain(
    output: &Path,
    metadata: Value,
    trial: impl FnOnce() -> DynResult<Value>,
) -> DynResult<bool> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let mut report = match trial() {
        Ok(Value::Object(report)) => Value::Object(report),
        Ok(value) => json!({"all_passed": false, "trial_report": value,
            "error": "trial returned a non-object report"}),
        Err(error) => json!({"all_passed": false, "error": format!("{error:#}")}),
    };
    let passed = report.get("all_passed").and_then(Value::as_bool) == Some(true);
    report["all_passed"] = json!(passed);
    for claim in [
        "source_arithmetic_qualified",
        "native_mtp_admitted",
        "model_executable",
        "timing_claim",
    ] {
        report[claim] = json!(false);
    }
    report["request"] = metadata;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(passed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "native-mtp-report-{}-{}",
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
    fn existing_output_prevents_trial_when_reserved() {
        let scratch = Scratch::new();
        let output = scratch.0.join("report.json");
        fs::write(&output, b"existing").unwrap();
        let result = retain(&output, json!({}), || panic!("trial must not run"));
        assert!(result.is_err());
        assert_eq!(fs::read(output).unwrap(), b"existing");
    }

    #[test]
    fn report_is_saved_when_trial_errors() {
        let scratch = Scratch::new();
        let output = scratch.0.join("report.json");
        let passed = retain(&output, json!({"device": 2}), || Err("load failed".into())).unwrap();
        let saved: Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
        assert!(!passed);
        assert_eq!(saved["error"], "load failed");
        assert_eq!(saved["request"]["device"], 2);
    }

    #[test]
    fn status_fails_closed_when_not_explicit_boolean_true() {
        for report in [
            json!({}),
            json!({"all_passed": false}),
            json!({"all_passed": "true"}),
            json!({"all_passed": 1}),
            json!({"all_passed": null}),
            json!([true]),
        ] {
            let scratch = Scratch::new();
            let output = scratch.0.join("report.json");
            let passed = retain(&output, json!({}), || Ok(report)).unwrap();
            let saved: Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
            assert!(!passed);
            assert_eq!(saved["all_passed"], false);
        }
    }

    #[test]
    fn evidence_survives_when_trial_passes_without_admission() {
        let scratch = Scratch::new();
        let output = scratch.0.join("report.json");
        let passed = retain(&output, json!({}), || {
            Ok(json!({"all_passed": true,
            "cases": [{"exact_mismatches": 0}], "source_arithmetic_qualified": true,
            "native_mtp_admitted": true, "model_executable": true, "timing_claim": true}))
        })
        .unwrap();
        let saved: Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
        assert!(passed);
        assert_eq!(saved["cases"][0]["exact_mismatches"], 0);
        for claim in [
            "source_arithmetic_qualified",
            "native_mtp_admitted",
            "model_executable",
            "timing_claim",
        ] {
            assert_eq!(saved[claim], false);
        }
    }
}
