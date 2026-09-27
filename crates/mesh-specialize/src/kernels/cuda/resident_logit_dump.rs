//! Optional diagnostic exports for cross-process, same-input arithmetic comparisons.
use anyhow::{Context, Result};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
};

pub(super) fn write(
    tokens: &[u32],
    teacher: Option<u32>,
    logits: [(&str, &[u16]); 4],
) -> Result<Option<Value>> {
    let Some(directory) = std::env::var_os("MESH_SPECIALIZE_LOGIT_DUMP_DIR") else {
        return Ok(None);
    };
    let directory = PathBuf::from(directory);
    fs::create_dir(&directory)
        .with_context(|| format!("create fresh logit dump {}", directory.display()))?;
    let mut files = Vec::new();
    for (name, words) in logits {
        let bytes = words
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let filename = format!("{name}.bf16le");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join(&filename))?;
        file.write_all(&bytes)?;
        files.push(json!({"file":filename,"elements":words.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))}));
    }
    let metadata = json!({"schema_version":1,"arithmetic_profile":crate::kernels::fp8_profile::current()?.name(),
        "prefix_token_ids":tokens,"teacher_token":teacher,"encoding":"BF16 little-endian u16",
        "files":files,"scope":"diagnostic exports; compare matching input IDs and weights across profiles"});
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("manifest.json"))?;
    file.write_all(serde_json::to_string_pretty(&metadata)?.as_bytes())?;
    Ok(Some(json!({"directory":directory,"manifest":metadata})))
}
