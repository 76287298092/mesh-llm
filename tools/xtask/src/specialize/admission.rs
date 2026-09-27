use crate::command::DynResult;
use mesh_llm_native_runtime::cuda_admission::CudaSelectedDevice;
use mesh_llm_native_runtime::model_identity::ModelIdentity;
use mesh_llm_native_runtime::{
    CandidateEvaluation, HostRuntimeProfile, NativeRuntimeArtifact, NativeRuntimeManifest,
    RuntimeSelection,
    model_selection::{ModelRuntimeRequest, evaluate_native_runtime_artifact_for_request},
};
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

const MAX_FIXTURE_BYTES: u64 = 64 * 1024;
const REPORT_NOTE: &str = "Synthetic policy fixture only; no model files were loaded.";

#[derive(Debug)]
struct RunArgs {
    fixture: PathBuf,
    device: i32,
    expected_compatible: bool,
    output: PathBuf,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: u32,
    runtime: NativeRuntimeArtifact,
    requested: ModelIdentity,
}

#[derive(Serialize)]
struct AdmissionReport<'a> {
    schema_version: u32,
    kind: &'static str,
    fixture: &'a Fixture,
    device: Option<&'a CudaSelectedDevice>,
    host_profile: &'a HostRuntimeProfile,
    evaluation: Option<&'a CandidateEvaluation>,
    expected_compatible: bool,
    all_passed: bool,
    model_artifact_verified: bool,
    note: &'static str,
    error: Option<String>,
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let args = parse_args(args)?;
    let fixture = load_fixture(&args.fixture)?;
    let mut report_file = create_new_output(&args.output)?;
    let host_profile = mesh_llm_hardware_profile::host_runtime_profile();

    let raw_device = match mesh_specialize::kernels::device_probe(args.device) {
        Ok(value) => value,
        Err(error) => {
            let message = format!("selected CUDA device probe failed: {error:#}");
            write_error_report(
                &mut report_file,
                &fixture,
                &host_profile,
                args.expected_compatible,
                message.clone(),
            )?;
            return Err(message.into());
        }
    };
    let device: CudaSelectedDevice = match serde_json::from_value(raw_device) {
        Ok(device) => device,
        Err(error) => {
            let message = format!("could not decode selected CUDA device evidence: {error}");
            write_error_report(
                &mut report_file,
                &fixture,
                &host_profile,
                args.expected_compatible,
                message.clone(),
            )?;
            return Err(message.into());
        }
    };
    let request = ModelRuntimeRequest {
        identity: &fixture.requested,
        cuda_device: Some(&device),
    };
    let Some(mesh_version) = fixture.runtime.mesh_version.as_deref() else {
        let message = "fixture runtime must declare mesh_version".to_string();
        write_error_report(
            &mut report_file,
            &fixture,
            &host_profile,
            args.expected_compatible,
            message.clone(),
        )?;
        return Err(message.into());
    };
    let evaluation = evaluate_native_runtime_artifact_for_request(
        &fixture.runtime,
        &host_profile,
        mesh_version,
        Some(&fixture.runtime.skippy_abi),
        &RuntimeSelection::Recommended,
        &request,
    );
    let all_passed = evaluation.compatible == args.expected_compatible;
    let report = AdmissionReport {
        schema_version: 1,
        kind: "driver-admission-policy-probe",
        fixture: &fixture,
        device: Some(&device),
        host_profile: &host_profile,
        evaluation: Some(&evaluation),
        expected_compatible: args.expected_compatible,
        all_passed,
        model_artifact_verified: false,
        note: REPORT_NOTE,
        error: None,
    };
    write_report(&mut report_file, &report)?;
    if !all_passed {
        return Err("CUDA admission result did not match --expect".into());
    }
    Ok(())
}

fn parse_args(args: &[String]) -> DynResult<RunArgs> {
    let [
        fixture_flag,
        fixture,
        device_flag,
        device,
        expect_flag,
        expectation,
        output_flag,
        output,
    ] = args
    else {
        return Err(usage_error().into());
    };
    if fixture_flag != "--fixture"
        || device_flag != "--device"
        || expect_flag != "--expect"
        || output_flag != "--output"
    {
        return Err(usage_error().into());
    }
    let device = device
        .parse::<i32>()
        .map_err(|_| "--device must be a nonnegative integer ordinal")?;
    if device < 0 {
        return Err("--device must be a nonnegative integer ordinal".into());
    }
    let expected_compatible = match expectation.as_str() {
        "compatible" => true,
        "rejected" => false,
        _ => return Err("--expect must be compatible or rejected".into()),
    };
    Ok(RunArgs {
        fixture: PathBuf::from(fixture),
        device,
        expected_compatible,
        output: PathBuf::from(output),
    })
}

fn usage_error() -> String {
    "usage: xtask specialize admission-probe --fixture PATH --device ORD --expect compatible|rejected --output NEW_FILE".to_string()
}

fn load_fixture(path: &Path) -> DynResult<Fixture> {
    let file = File::open(path)?;
    if file.metadata()?.len() > MAX_FIXTURE_BYTES {
        return Err("admission fixture exceeds 64 KiB".into());
    }
    let mut contents = Vec::new();
    file.take(MAX_FIXTURE_BYTES + 1)
        .read_to_end(&mut contents)?;
    if u64::try_from(contents.len()).map_or(true, |length| length > MAX_FIXTURE_BYTES) {
        return Err("admission fixture exceeds 64 KiB".into());
    }
    let fixture: Fixture = serde_json::from_slice(&contents)?;
    validate_fixture(&fixture)?;
    Ok(fixture)
}

fn validate_fixture(fixture: &Fixture) -> DynResult<()> {
    if fixture.schema_version != 1 {
        return Err("unsupported fixture schema_version; expected 1".into());
    }
    NativeRuntimeManifest {
        runtime: fixture.runtime.clone(),
    }
    .validate()?;
    fixture.requested.validate()?;
    if fixture.runtime.mesh_version.is_none() {
        return Err("fixture runtime must declare mesh_version".into());
    }
    Ok(())
}

fn create_new_output(path: &Path) -> DynResult<File> {
    Ok(OpenOptions::new().write(true).create_new(true).open(path)?)
}

fn write_error_report(
    file: &mut File,
    fixture: &Fixture,
    host_profile: &HostRuntimeProfile,
    expected_compatible: bool,
    error: String,
) -> DynResult<()> {
    let report = AdmissionReport {
        schema_version: 1,
        kind: "driver-admission-policy-probe",
        fixture,
        device: None,
        host_profile,
        evaluation: None,
        expected_compatible,
        all_passed: false,
        model_artifact_verified: false,
        note: REPORT_NOTE,
        error: Some(error),
    };
    write_report(file, &report)
}

fn write_report(file: &mut File, report: &AdmissionReport<'_>) -> DynResult<()> {
    serde_json::to_writer_pretty(&mut *file, report)?;
    file.write_all(b"\n")?;
    file.flush()?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_args;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn parser_requires_the_declared_argument_order() {
        let reordered = args(&[
            "--device",
            "0",
            "--fixture",
            "fixture.json",
            "--expect",
            "compatible",
            "--output",
            "report.json",
        ]);
        assert!(parse_args(&reordered).is_err());
    }

    #[test]
    fn parser_rejects_unknown_expectation_and_negative_device() {
        let invalid_expectation = args(&[
            "--fixture",
            "fixture.json",
            "--device",
            "0",
            "--expect",
            "maybe",
            "--output",
            "report.json",
        ]);
        assert!(parse_args(&invalid_expectation).is_err());

        let negative_device = args(&[
            "--fixture",
            "fixture.json",
            "--device",
            "-1",
            "--expect",
            "rejected",
            "--output",
            "report.json",
        ]);
        assert!(parse_args(&negative_device).is_err());
    }
}
