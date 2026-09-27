use crate::command::{DynResult, print_json};
use mesh_specialize::{kernels::ModelBenchRequest, packages::qwen3_8_27b::model_benchmark};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::Instant,
};

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [
        artifact_flag,
        artifact,
        tokens_flag,
        tokens,
        output_tokens_flag,
        output_tokens,
        repetitions_flag,
        repetitions,
        ptx_flag,
        ptx_path,
        device_flag,
        device,
        output_flag,
        output,
    ] = args
    else {
        return Err("usage: xtask specialize qwen-model-bench --artifact PATH --tokens COMMA_IDS --output-tokens 2..16 --repetitions 1..3 --ptx PATH --device ORDINAL --output NEW_FILE".into());
    };
    if [
        artifact_flag.as_str(),
        tokens_flag.as_str(),
        output_tokens_flag.as_str(),
        repetitions_flag.as_str(),
        ptx_flag.as_str(),
        device_flag.as_str(),
        output_flag.as_str(),
    ] != [
        "--artifact",
        "--tokens",
        "--output-tokens",
        "--repetitions",
        "--ptx",
        "--device",
        "--output",
    ] {
        return Err("benchmark flags must follow documented order".into());
    }
    let tokens = tokens
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    let request = ModelBenchRequest {
        tokens: &tokens,
        output_tokens: output_tokens.parse()?,
        repetitions: repetitions.parse()?,
    };
    let device = device.parse()?;
    let ptx = fs::read_to_string(ptx_path)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let started = Instant::now();
    let result = model_benchmark::run(Path::new(artifact), &ptx, device, &request);
    let failed = result.is_err();
    let mut report =
        result.unwrap_or_else(|error| json!({"completed":false,"error":format!("{error:#}")}));
    report["harness_elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    report["artifact_path"] = json!(artifact);
    report["ptx_sha256"] = json!(hex::encode(Sha256::digest(ptx.as_bytes())));
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    print_json(&json!({"output":output,"completed":!failed}))?;
    if failed {
        return Err("model benchmark failed; inspect saved report".into());
    }
    Ok(())
}
