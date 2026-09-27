use crate::command::{DynResult, print_json};
use mesh_specialize::packages::qwen3_8_27b::{self, model_reference::ModelReference};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::Path,
    time::Instant,
};

pub(super) fn reference(args: &[String]) -> DynResult<()> {
    let [
        artifact_flag,
        artifact,
        tokens_flag,
        tokens,
        output_flag,
        output,
    ] = args
    else {
        return Err("usage: xtask specialize qwen-model-reference --artifact PATH --tokens COMMA_IDS --output NEW_FILE".into());
    };
    if artifact_flag != "--artifact" || tokens_flag != "--tokens" || output_flag != "--output" {
        return Err("expected --artifact, --tokens, --output in order".into());
    }
    let tokens = tokens
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    if !(1..=17).contains(&tokens.len()) {
        return Err("reference requires 1..17 tokens".into());
    }
    let mut file = create(output)?;
    let result =
        qwen3_8_27b::model_reference::run(Path::new(artifact), &tokens, |layer, seconds| {
            print_json(&json!({"reference_layer":layer,"seconds":seconds})).map_err(|error| {
                std::io::Error::other(format!("reference progress output: {error}")).into()
            })
        });
    let reference = match result {
        Ok(reference) => reference,
        Err(error) => {
            save(
                &mut file,
                &json!({"all_passed":false,"error":format!("{error:#}")}),
            )?;
            return Err(format!("CPU reference failed: {error:#}").into());
        }
    };
    let token = mesh_specialize::engine::sampling::greedy(&reference.logits)?;
    serde_json::to_writer(&mut file, &reference)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    print_json(
        &json!({"output":output,"layers":reference.layer_outputs.len(),"reference_token":token,"elapsed_seconds":reference.elapsed_seconds}),
    )
}

pub(super) fn check(args: &[String]) -> DynResult<()> {
    let [
        artifact_flag,
        artifact,
        reference_flag,
        reference_path,
        ptx_flag,
        ptx,
        device_flag,
        device,
        output_flag,
        output,
    ] = args
    else {
        return Err("usage: xtask specialize qwen-model-check --artifact PATH --reference PATH --ptx PATH --device ORDINAL --output NEW_FILE".into());
    };
    if artifact_flag != "--artifact"
        || reference_flag != "--reference"
        || ptx_flag != "--ptx"
        || device_flag != "--device"
        || output_flag != "--output"
    {
        return Err("expected --artifact, --reference, --ptx, --device, --output in order".into());
    }
    let mut raw = Vec::new();
    File::open(reference_path)?
        .take(64 * 1024 * 1024 + 1)
        .read_to_end(&mut raw)?;
    if raw.len() > 64 * 1024 * 1024 {
        return Err("reference exceeds 64 MiB".into());
    }
    let reference: ModelReference = serde_json::from_slice(&raw)?;
    let device = device.parse()?;
    let ptx = fs::read_to_string(ptx)?;
    let mut file = create(output)?;
    let started = Instant::now();
    let result = qwen3_8_27b::model_reference::trial(Path::new(artifact), &reference, &ptx, device);
    let mut report = match result {
        Ok(report) => report,
        Err(error) => {
            json!({"all_passed":false,"error":format!("{error:#}"),"full_model_executed":false})
        }
    };
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    report["artifact_path"] = json!(artifact);
    report["reference_path"] = json!(reference_path);
    report["reference_sha256"] = json!(hex::encode(Sha256::digest(&raw)));
    report["ptx_sha256"] = json!(hex::encode(Sha256::digest(ptx.as_bytes())));
    save(&mut file, &report)?;
    print_json(
        &json!({"output":output,"all_passed":report["all_passed"],"full_model_executed":report["full_model_executed"]}),
    )?;
    if report["all_passed"] != true {
        return Err("model comparison failed; inspect saved report".into());
    }
    Ok(())
}

fn create(path: &str) -> DynResult<File> {
    Ok(OpenOptions::new().write(true).create_new(true).open(path)?)
}
fn save(file: &mut File, report: &Value) -> DynResult<()> {
    serde_json::to_writer_pretty(&mut *file, report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}
