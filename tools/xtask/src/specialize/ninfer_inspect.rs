//! Read-only native artifact qualification; no CUDA, conversion or model execution.
use crate::command::{DynResult, print_json};
use mesh_specialize::artifact::{
    model_source::ModelArtifact, ninfer::NinferArtifact, schema::ObjectKind,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::OpenOptions,
    io::{self, Write},
    path::Path,
    time::Instant,
};

const PIN: &str = "74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82";
const USAGE: &str = "usage: xtask specialize ninfer-inspect --artifact PATH --output NEW_FILE [--hash-objects|--canonical]";

pub(super) fn run(args: &[String]) -> DynResult<()> {
    if !(args.len() == 4 || args.len() == 5)
        || args[0] != "--artifact"
        || args[2] != "--output"
        || (args.len() == 5 && !matches!(args[4].as_str(), "--hash-objects" | "--canonical"))
    {
        return Err(USAGE.into());
    }
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])?;
    let started = Instant::now();
    let result = if args.get(4).is_some_and(|flag| flag == "--canonical") {
        canonical(Path::new(&args[1]))
    } else {
        inspect(Path::new(&args[1]), args.len() == 5)
    };
    let (mut report, failed) = match result {
        Ok(value) => (value, false),
        Err(error) => (json!({"all_passed":false,"error":error.to_string()}), true),
    };
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    serde_json::to_writer_pretty(&mut output, &report)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    print_json(&json!({"output":args[3],"all_passed":report["all_passed"]}))?;
    if failed {
        return Err("native artifact inspection failed; see report".into());
    }
    Ok(())
}

fn inspect(path: &Path, hash_objects: bool) -> DynResult<serde_json::Value> {
    let mut artifact = NinferArtifact::open(path)?;
    let first = artifact.file_sha256()?;
    let directory = artifact.directory().clone();
    let mut objects = Vec::new();
    if hash_objects {
        for object in &directory.objects {
            let mut digest = HashSink::default();
            let copied = artifact.copy_object(&object.id, &mut digest)?;
            if copied != object.bytes || digest.bytes != object.bytes {
                return Err("object byte count differs".into());
            }
            objects.push(
                json!({"id":object.id,"bytes":copied,"sha256":hex::encode(digest.hash.finalize())}),
            );
        }
    }
    let last = artifact.file_sha256()?;
    if first != last {
        return Err("native artifact changed during inspection".into());
    }
    Ok(
        json!({"schema_version":1,"kind":"independent-rust-ninfer-v3-inspection",
        "all_passed":true,"model_executable":false,"source_sha256":first,
        "benchmark_pin_matches":first==PIN,"file_bytes":artifact.file_bytes(),
        "payload_offset":artifact.payload_offset(),"artifact_id":hex::encode(artifact.artifact_id()),
        "directory":directory,"object_hashes":objects,"hashed_objects":hash_objects}),
    )
}

fn canonical(path: &Path) -> DynResult<serde_json::Value> {
    let source = ModelArtifact::open(path)?;
    if !matches!(&source, ModelArtifact::Ninfer(_)) {
        return Err("canonical native inspection requires a .ninfer source".into());
    }
    let objects = source
        .directory()
        .objects
        .iter()
        .filter(|object| object.kind == ObjectKind::Tensor)
        .collect::<Vec<_>>();
    Ok(
        json!({"schema_version":1,"kind":"native-canonical-view-hashes",
        "all_passed":true,"model_execution_performed":false,"identity":source.identity(),
        "source":source.verification_report(),"objects":objects}),
    )
}

#[derive(Default)]
struct HashSink {
    hash: Sha256,
    bytes: u64,
}
impl Write for HashSink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| io::Error::other("hash count overflow"))?;
        self.hash.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
