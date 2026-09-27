use crate::command::{DynResult, print_json};
use mesh_specialize::{
    kernels::SpeculationRequest,
    packages::qwen3_8_27b::{model_reference::ModelReference, mtp},
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::Instant,
};

pub(super) fn reference(args: &[String]) -> DynResult<()> {
    let [a, artifact, r, reference, o, output] = args else {
        return Err(
            "usage: qwen-mtp-reference --artifact PATH --target-reference PATH --output NEW_FILE"
                .into(),
        );
    };
    if [a.as_str(), r.as_str(), o.as_str()] != ["--artifact", "--target-reference", "--output"] {
        return Err("invalid MTP reference flags".into());
    }
    let target: ModelReference = serde_json::from_slice(&fs::read(reference)?)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let result = mtp::reference(Path::new(artifact), &target)?;
    serde_json::to_writer(&mut file, &result)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    print_json(&json!({"output":output,"completed":true}))
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [
        a,
        artifact,
        r,
        reference,
        t,
        tokens,
        n,
        count,
        d,
        depth,
        k,
        reps,
        p,
        ptx,
        g,
        device,
        o,
        output,
    ] = args
    else {
        return Err("usage: qwen-mtp-check --artifact PATH --reference PATH --tokens COMMA_IDS --output-tokens 3..128 --depth 1..4 --repetitions 1..3 --ptx PATH --device ORDINAL --output NEW_FILE".into());
    };
    if [
        a.as_str(),
        r.as_str(),
        t.as_str(),
        n.as_str(),
        d.as_str(),
        k.as_str(),
        p.as_str(),
        g.as_str(),
        o.as_str(),
    ] != [
        "--artifact",
        "--reference",
        "--tokens",
        "--output-tokens",
        "--depth",
        "--repetitions",
        "--ptx",
        "--device",
        "--output",
    ] {
        return Err("invalid MTP check flags".into());
    }
    let reference: mtp::Reference = serde_json::from_slice(&fs::read(reference)?)?;
    let tokens = tokens
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    let request = SpeculationRequest {
        tokens: &tokens,
        output_tokens: count.parse()?,
        depth: depth.parse()?,
        repetitions: reps.parse()?,
    };
    let ptx = fs::read_to_string(ptx)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let start = Instant::now();
    let result = mtp::trial(
        Path::new(artifact),
        &ptx,
        device.parse()?,
        &reference,
        &request,
    );
    let failed = result.is_err();
    let mut report =
        result.unwrap_or_else(|e| json!({"all_passed":false,"error":format!("{e:#}")}));
    report["harness_elapsed_seconds"] = json!(start.elapsed().as_secs_f64());
    report["ptx_sha256"] = json!(hex::encode(Sha256::digest(ptx.as_bytes())));
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    print_json(&json!({"output":output,"completed":!failed}))?;
    if failed {
        return Err("MTP check failed; inspect saved report".into());
    }
    Ok(())
}
