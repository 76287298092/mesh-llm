use crate::command::{DynResult, print_json};
use mesh_specialize::{
    artifact::model_source::ModelArtifact,
    kernels::{StreamCheckRequest, stream_forward_check},
    packages::qwen3_8_27b::{decoder, inventory, schedule},
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::Instant,
};

const USAGE: &str = "usage: xtask specialize qwen-stream-check --artifact PATH --tokens COMMA_IDS --ptx PATH --device ORDINAL --output NEW_FILE --decode-steps N";

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [
        artifact_flag,
        artifact,
        tokens_flag,
        tokens,
        ptx_flag,
        ptx_path,
        device_flag,
        device,
        output_flag,
        output,
        steps_flag,
        steps,
    ] = args
    else {
        return Err(USAGE.into());
    };
    if [
        artifact_flag.as_str(),
        tokens_flag.as_str(),
        ptx_flag.as_str(),
        device_flag.as_str(),
        output_flag.as_str(),
        steps_flag.as_str(),
    ] != [
        "--artifact",
        "--tokens",
        "--ptx",
        "--device",
        "--output",
        "--decode-steps",
    ] {
        return Err(format!("stream check flags must follow documented order; {USAGE}").into());
    }
    let tokens = tokens
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    let decode_steps: usize = steps.parse()?;
    let device = device.parse()?;
    let ptx = fs::read_to_string(ptx_path)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let start = Instant::now();
    let mut report = check(Path::new(artifact), &ptx, device, &tokens, decode_steps)
        .unwrap_or_else(|error| json!({"all_passed":false,"error":format!("{error:#}")}));
    report["harness_elapsed_seconds"] = json!(start.elapsed().as_secs_f64());
    report["artifact_path"] = json!(artifact);
    report["ptx_sha256"] = json!(hex::encode(Sha256::digest(ptx.as_bytes())));
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    print_json(&json!({"output":output,"all_passed":report["all_passed"]}))?;
    if report["all_passed"] != true {
        return Err("stream check failed; inspect saved report".into());
    }
    Ok(())
}

fn check(
    path: &Path,
    ptx: &str,
    device: i32,
    tokens: &[u32],
    decode_steps: usize,
) -> DynResult<serde_json::Value> {
    if !(1..=512).contains(&tokens.len()) || decode_steps > 512 {
        return Err(
            "stream check requires 1..=512 prompt tokens and at most 512 decode steps".into(),
        );
    }
    let mut artifact = ModelArtifact::open(path)?;
    inventory::validate(artifact.directory())?;
    let objects = schedule::text_objects(artifact.directory())?;
    let config = decoder::config(tokens.len() + decode_steps)?;
    let request = StreamCheckRequest {
        tokens,
        decode_steps,
    };
    let mut report = stream_forward_check(ptx, device, &mut artifact, &objects, &config, &request)?;
    report["identity"] = json!(artifact.identity());
    report["model_source"] = artifact.verification_report();
    Ok(report)
}
