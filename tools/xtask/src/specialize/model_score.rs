use crate::command::{DynResult, print_json};
use mesh_specialize::{
    kernels::{ModelScoreRequest, ScoreStream},
    packages::qwen3_8_27b::model_score,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

const USAGE: &str = "usage: xtask specialize qwen-model-score --artifact PATH --streams STREAMS_JSON --context C --stride S --ptx PATH --device ORDINAL --output NEW_DIRECTORY";

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [
        artifact_flag,
        artifact,
        streams_flag,
        streams_path,
        context_flag,
        context,
        stride_flag,
        stride,
        ptx_flag,
        ptx_path,
        device_flag,
        device,
        output_flag,
        output,
    ] = args
    else {
        return Err(USAGE.into());
    };
    if [
        artifact_flag.as_str(),
        streams_flag.as_str(),
        context_flag.as_str(),
        stride_flag.as_str(),
        ptx_flag.as_str(),
        device_flag.as_str(),
        output_flag.as_str(),
    ] != [
        "--artifact",
        "--streams",
        "--context",
        "--stride",
        "--ptx",
        "--device",
        "--output",
    ] {
        return Err("score flags must follow documented order".into());
    }
    let context: usize = context.parse()?;
    let stride: usize = stride.parse()?;
    if !(2..=model_score::MAX_CONTEXT).contains(&context) {
        return Err(format!(
            "--context {context} is unsupported: scoring is limited to 2..={} until chunked prefill exists",
            model_score::MAX_CONTEXT
        )
        .into());
    }
    let (corpus_id, streams) = read_streams(Path::new(streams_path))?;
    let device = device.parse()?;
    let ptx = fs::read_to_string(ptx_path)?;
    let output = PathBuf::from(output);
    fs::create_dir(&output)
        .map_err(|error| format!("refusing to reuse {}: {error}", output.display()))?;
    let mut files = streams
        .iter()
        .map(|stream| {
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output.join(record_file(&stream.id)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let request = ModelScoreRequest {
        corpus_id: &corpus_id,
        streams: &streams,
        context,
        stride,
    };
    let start = Instant::now();
    let mut sink = |index: usize, bytes: &[u8]| files[index].write_all(bytes);
    let mut report = model_score::run(Path::new(artifact), &ptx, device, &request, &mut sink)
        .unwrap_or_else(|error| json!({"all_passed": false, "error": format!("{error:#}")}));
    for file in &files {
        file.sync_all()?;
    }
    annotate(&mut report, &streams, artifact, &ptx, start)?;
    let manifest = output.join("manifest.json");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&manifest)?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    print_json(&json!({"output": output, "all_passed": report["all_passed"]}))?;
    if report["all_passed"] != true {
        return Err("model scoring failed; inspect saved manifest".into());
    }
    Ok(())
}

fn record_file(id: &str) -> String {
    format!("{id}.scores.bin")
}

fn read_streams(path: &Path) -> DynResult<(String, Vec<ScoreStream>)> {
    let value: Value = serde_json::from_slice(&fs::read(path)?)?;
    let corpus_id = value["corpus_id"]
        .as_str()
        .ok_or("streams JSON needs a string corpus_id")?
        .to_owned();
    let entries = value["streams"]
        .as_array()
        .ok_or("streams JSON needs a streams array")?;
    let mut streams = Vec::with_capacity(entries.len());
    for entry in entries {
        let id = entry["id"].as_str().ok_or("stream needs a string id")?;
        if id.is_empty()
            || !id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
            || id.starts_with('.')
        {
            return Err(
                format!("stream id {id:?} must be [A-Za-z0-9._-] and not start with '.'").into(),
            );
        }
        let domain = entry["domain"]
            .as_str()
            .ok_or("stream needs a string domain")?;
        let tokens = entry["tokens"]
            .as_array()
            .ok_or("stream needs a tokens array")?
            .iter()
            .map(|token| {
                token
                    .as_u64()
                    .and_then(|token| u32::try_from(token).ok())
                    .ok_or("stream token must be a u32")
            })
            .collect::<Result<Vec<_>, _>>()?;
        streams.push(ScoreStream {
            id: id.to_owned(),
            domain: domain.to_owned(),
            tokens,
        });
    }
    Ok((corpus_id, streams))
}

fn annotate(
    report: &mut Value,
    streams: &[ScoreStream],
    artifact: &str,
    ptx: &str,
    start: Instant,
) -> DynResult<()> {
    if let Some(entries) = report["streams"].as_array_mut() {
        for (entry, stream) in entries.iter_mut().zip(streams) {
            entry["records"] = json!(record_file(&stream.id));
        }
    }
    report["wall_seconds"] = json!(start.elapsed().as_secs_f64());
    report["artifact_path"] = json!(artifact);
    report["artifact_sha256"] = json!(file_sha256(Path::new(artifact))?);
    report["ptx_sha256"] = json!(hex::encode(Sha256::digest(ptx.as_bytes())));
    Ok(())
}

fn file_sha256(path: &Path) -> DynResult<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}
