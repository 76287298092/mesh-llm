use crate::command::{DynResult, print_json};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [a, artifact, p, ptx, d, device, o, output] = args else {
        return Err("usage: mlp-workspace-check --artifact PATH --ptx PATH --device ORDINAL --output NEW_FILE".into());
    };
    if [a.as_str(), p.as_str(), d.as_str(), o.as_str()]
        != ["--artifact", "--ptx", "--device", "--output"]
    {
        return Err("unexpected workspace flags".into());
    }
    let ptx = fs::read_to_string(ptx)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let result = mesh_specialize::packages::qwen3_8_27b::model_benchmark::mlp_workspace(
        Path::new(artifact),
        &ptx,
        device.parse()?,
    );
    let mut report =
        result.unwrap_or_else(|e| json!({"all_passed":false,"error":format!("{e:#}")}));
    report["ptx_sha256"] = json!(hex::encode(Sha256::digest(ptx.as_bytes())));
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    print_json(&json!({"output":output,"all_passed":report["all_passed"]}))?;
    if report["all_passed"] != true {
        return Err("workspace qualification failed; inspect saved report".into());
    }
    Ok(())
}
