//! Opt-in chunked prefill measurement; never changes qwen-model-bench defaults.

use crate::command::{DynResult, print_json};
use mesh_specialize::{
    engine::prefill_chunks::Plan,
    kernels::{ChunkedBenchRequest, qwen_chunked_benchmark},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
    time::Instant,
};

const USAGE: &str = "usage: xtask specialize qwen-chunked-bench --artifact PATH --tokens-file PATH --chunk-size 1..512 --output-tokens 2..512 --repetitions 1..5 --ptx PATH --device ORDINAL --output NEW_FILE";
const MAX_TOKEN_FILE_BYTES: u64 = 1024 * 1024;
const PROFILE_ENV: [&str; 9] = [
    "MESH_SPECIALIZE_EXECUTION",
    "MESH_SPECIALIZE_FP8_PROFILE",
    "MESH_SPECIALIZE_NVFP4_PROFILE",
    "MESH_SPECIALIZE_ATTENTION_PROFILE",
    "MESH_SPECIALIZE_MLP_WORKSPACE",
    "MESH_SPECIALIZE_FP8_SPLIT_K",
    "MESH_SPECIALIZE_NVFP4_AUDIT",
    "MESH_SPECIALIZE_GPU_GREEDY",
    "CUDA_VISIBLE_DEVICES",
];

struct Args<'a> {
    artifact: &'a str,
    tokens_file: &'a str,
    chunk_size: &'a str,
    output_tokens: &'a str,
    repetitions: &'a str,
    ptx: &'a str,
    device: &'a str,
    output: &'a str,
}

fn parse(args: &[String]) -> DynResult<Args<'_>> {
    let [
        af,
        artifact,
        tf,
        tokens_file,
        cf,
        chunk_size,
        of,
        output_tokens,
        rf,
        repetitions,
        pf,
        ptx,
        df,
        device,
        ff,
        output,
    ] = args
    else {
        return Err(USAGE.into());
    };
    if [af, tf, cf, of, rf, pf, df, ff].map(String::as_str)
        != [
            "--artifact",
            "--tokens-file",
            "--chunk-size",
            "--output-tokens",
            "--repetitions",
            "--ptx",
            "--device",
            "--output",
        ]
    {
        return Err(format!("flags must follow documented order; {USAGE}").into());
    }
    Ok(Args {
        artifact,
        tokens_file,
        chunk_size,
        output_tokens,
        repetitions,
        ptx,
        device,
        output,
    })
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let parsed = parse(args)?;
    // Reserve the output before input parsing or any GPU access. Existing paths,
    // including symlinks, are refused by create_new; never overwrite prior evidence.
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(parsed.output)?;
    let mut report = initial_report(&parsed, args);
    save(&mut file, &report)?;
    let start = Instant::now();
    let result = execute(&parsed, &mut report);
    report["harness_elapsed_seconds"] = json!(start.elapsed().as_secs_f64());
    let failed = result.is_err();
    if let Err(error) = result {
        report["completed"] = json!(false);
        report["status"] = json!("failed");
        report["error"] = json!(format!("{error:#}"));
        if report["partition_check"]["status"] == "running" {
            report["partition_check"]["status"] = json!("failed");
        }
    } else {
        report["status"] = json!("completed");
    }
    save(&mut file, &report)?;
    print_json(&json!({
        "output": parsed.output, "completed": !failed,
        "partition_check_status": report["partition_check"]["status"],
        "partition_check_passed": report["partition_check"]["passed"],
    }))?;
    if failed {
        return Err("chunked benchmark failed; inspect saved report".into());
    }
    // An exact partition mismatch remains prominently reported but is not an
    // execution failure. Consumers must inspect partition_check.passed separately.
    Ok(())
}

fn initial_report(args: &Args<'_>, argv: &[String]) -> Value {
    let environment: serde_json::Map<String, Value> = PROFILE_ENV.iter().map(|&name| {
        let value = match std::env::var_os(name) {
            None => Value::Null,
            Some(value) => match value.into_string() {
                Ok(value) => json!({"value": value}),
                Err(value) => json!({"non_unicode_encoded_bytes_hex": hex::encode(value.as_encoded_bytes())}),
            },
        };
        (name.to_owned(), value)
    }).collect();
    json!({
        "schema_version": 1, "kind": "qwen-chunked-benchmark",
        "status": "started", "completed": false, "phase": "input_validation",
        "argv": argv, "artifact_path": args.artifact, "ptx_path": args.ptx,
        "tokens_file": args.tokens_file, "environment": environment,
        "harness_version": env!("CARGO_PKG_VERSION"),
        "mode": "ordinary", "execution": "stream", "concurrency": 1,
        "prefix_reuse": false, "eos_stopping": false,
        "scope": "fixed-output raw token IDs, no EOS stopping, no tokenization, no serving or quality qualification",
        "timing_note": "host wall intervals include token upload, sequential forward, stream synchronization, greedy selection and readback; no diagnostic logits or state hashing; session initialization and warmup excluded",
        "prefill_note": "one session across all chunks; every chunk computes a last-row vocabulary head and greedy selection, wasted and discarded on non-final chunks; only final prefill selection is output token 0",
        "allocation_note": "one weight arena and one model arena reused across warmup and timed repetitions; a fresh zeroed session per repetition; diagnostic streams and sessions are separate",
        "memory_note": "checked admission plus 512 MiB margin, not a reserved budget or peak measurement; concurrent external GPU allocations can invalidate the free-memory snapshot",
        "partition_check": {"status": "not_run", "passed": null, "reason": "execution has not reached diagnostics"},
    })
}

fn execute(args: &Args<'_>, report: &mut Value) -> DynResult<()> {
    let chunk_size: usize = args.chunk_size.parse()?;
    let output_tokens: usize = args.output_tokens.parse()?;
    let repetitions: usize = args.repetitions.parse()?;
    let device: i32 = args.device.parse()?;
    if device < 0 {
        return Err("device ordinal must be nonnegative".into());
    }
    if !(1..=5).contains(&repetitions) {
        return Err("repetitions must be 1..=5".into());
    }
    let mut raw = Vec::new();
    File::open(args.tokens_file)?
        .take(MAX_TOKEN_FILE_BYTES + 1)
        .read_to_end(&mut raw)?;
    if raw.len() as u64 > MAX_TOKEN_FILE_BYTES {
        return Err("tokens file exceeds 1 MiB bound".into());
    }
    report["tokens_file_sha256"] = json!(hex::encode(Sha256::digest(&raw)));
    let tokens: Vec<u32> = serde_json::from_slice(&raw)?;
    report["prompt_token_ids"] = json!(tokens);
    report["prompt_tokens"] = json!(tokens.len());
    report["input_tokens_sha256"] = json!(token_hash(&tokens));
    report["input_hash_encoding"] = json!("concatenated little-endian u32 token IDs");
    let plan = Plan::new(tokens.len(), chunk_size, output_tokens)?;
    report["plan"] = json!(plan);
    report["requested_repetitions"] = json!(repetitions);
    report["output_tokens"] = json!(output_tokens);
    report["device_ordinal"] = json!(device);
    let ptx = fs::read_to_string(args.ptx)?;
    report["ptx_sha256"] = json!(hex::encode(Sha256::digest(ptx.as_bytes())));
    let executable = std::env::current_exe()?;
    report["executable_path"] = json!(executable);
    report["executable_sha256"] = json!(file_hash(&executable)?);
    report["source_revision"] =
        json!("not captured; record parent qualification revision alongside executable_sha256");
    let request = ChunkedBenchRequest {
        tokens: &tokens,
        chunk_size,
        output_tokens,
        repetitions,
    };
    qwen_chunked_benchmark(Path::new(args.artifact), &ptx, device, &request, report)?;
    Ok(())
}

fn token_hash(tokens: &[u32]) -> String {
    let mut hash = Sha256::new();
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    hex::encode(hash.finalize())
}

fn file_hash(path: &Path) -> DynResult<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hex::encode(hash.finalize()))
}

fn save(file: &mut File, report: &Value) -> DynResult<()> {
    let encoded = serde_json::to_vec_pretty(report)?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&encoded)?;
    file.write_all(b"\n")?;
    file.set_len(
        u64::try_from(encoded.len())?
            .checked_add(1)
            .ok_or("report length overflow")?,
    )?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{parse, token_hash};

    fn args() -> Vec<String> {
        "--artifact model --tokens-file ids.json --chunk-size 128 --output-tokens 2 --repetitions 1 --ptx model.ptx --device 0 --output new.json".split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn only_accepts_exact_ordered_flags() {
        assert_eq!(parse(&args()).unwrap().chunk_size, "128");
        let mut wrong = args();
        wrong.swap(0, 2);
        assert!(parse(&wrong).is_err());
        assert!(parse(&wrong[..14]).is_err());
        let mut extra = args();
        extra.push("--eos".into());
        assert!(parse(&extra).is_err());
    }

    #[test]
    fn token_digest_has_defined_encoding_and_order() {
        use sha2::{Digest, Sha256};
        assert_eq!(
            token_hash(&[1, 256]),
            hex::encode(Sha256::digest([1, 0, 0, 0, 0, 1, 0, 0]))
        );
        assert_ne!(token_hash(&[1, 256]), token_hash(&[256, 1]));
        assert_ne!(token_hash(&[1]), token_hash(&[1, 0]));
    }
}
