use crate::command::{DynResult, print_json};
use mesh_specialize::packages::qwen3_8_27b::model_profile;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::Instant,
};

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let (args, teacher_token) = match args {
        [base @ .., flag, token] if flag == "--teacher-token" => {
            (base, Some(token.parse::<u32>()?))
        }
        _ => (args, None),
    };
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
    ] = args
    else {
        return Err("usage: xtask specialize qwen-model-profile --artifact PATH --tokens COMMA_IDS --ptx PATH --device ORDINAL --output NEW_FILE [--teacher-token ID]".into());
    };
    if [
        artifact_flag.as_str(),
        tokens_flag.as_str(),
        ptx_flag.as_str(),
        device_flag.as_str(),
        output_flag.as_str(),
    ] != ["--artifact", "--tokens", "--ptx", "--device", "--output"]
    {
        return Err("profile flags must follow documented order".into());
    }
    let tokens = tokens
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    let device = device.parse()?;
    let ptx = fs::read_to_string(ptx_path)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let start = Instant::now();
    let mut report = model_profile::run(Path::new(artifact), &ptx, device, &tokens, teacher_token)
        .unwrap_or_else(|error| json!({"all_passed":false,"error":format!("{error:#}")}));
    report["harness_elapsed_seconds"] = json!(start.elapsed().as_secs_f64());
    report["artifact_path"] = json!(artifact);
    report["ptx_sha256"] = json!(hex::encode(Sha256::digest(ptx.as_bytes())));
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    print_json(&json!({"output":output,"all_passed":report["all_passed"]}))?;
    if report["all_passed"] != true {
        return Err("model profile failed; inspect saved report".into());
    }
    Ok(())
}
