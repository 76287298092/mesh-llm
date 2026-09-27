use crate::command::{DynResult, print_json};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
};

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [ptx_flag, ptx_path, device_flag, device, output_flag, output] = args else {
        return Err(
            "usage: xtask specialize nvfp4-probe --ptx PATH --device ORDINAL --output NEW_FILE"
                .into(),
        );
    };
    if ptx_flag != "--ptx" || device_flag != "--device" || output_flag != "--output" {
        return Err("expected --ptx, --device, and --output in that order".into());
    }
    let ptx = fs::read_to_string(ptx_path)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let result = mesh_specialize::kernels::nvfp4_probe(&ptx, device.parse()?);
    let (mut report, error) = match result {
        Ok(report) => (report, None),
        Err(error) => (
            json!({"all_passed": false, "error": error.to_string()}),
            Some(error.to_string()),
        ),
    };
    report["ptx_sha256"] = json!(hex::encode(Sha256::digest(ptx.as_bytes())));
    report["ptx_path"] = json!(ptx_path);
    file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    print_json(&json!({"output": output, "all_passed": report["all_passed"]}))?;
    if report["all_passed"] != true {
        return Err(error
            .unwrap_or_else(|| "NVFP4 probe numerical mismatch".into())
            .into());
    }
    Ok(())
}
