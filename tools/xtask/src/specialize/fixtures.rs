use std::fs::OpenOptions;
use std::io::Write;

use serde_json::{Value, json};

use crate::command::DynResult;

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [flag, output] = args else {
        return Err("usage: xtask specialize baseline-plan --output NEW_FILE".into());
    };
    if flag != "--output" {
        return Err("expected --output".into());
    }
    let plan = plan();
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    file.write_all(&serde_json::to_vec_pretty(&plan)?)?;
    file.write_all(b"\n")?;
    crate::command::print_json(&json!({"plan": output, "requests": 9}))
}

fn plan() -> Value {
    let mut requests = vec![request("warmup", 8, 64)];
    for (size, rows) in [("short", 24), ("medium", 256), ("long", 1536)] {
        for sample in 1..=2 {
            requests.push(request(&format!("{size}-{sample}"), rows, 512));
        }
    }
    let mut warm = requests.last().expect("long case exists").clone();
    warm["id"] = json!("long-2-warm");
    requests.push(warm);
    requests.push(request("short-final", 24, 512));
    json!({
        "schema_version": 1,
        "base_url": "http://127.0.0.1:1235/v1",
        "api_key_env": "NINFER_API_KEY",
        "timeout_seconds": 180,
        "max_total_seconds": 900,
        "requests": requests
    })
}

fn request(id: &str, rows: usize, max_tokens: u32) -> Value {
    // The distinct prefix precedes the repeated table to avoid accidental reuse
    // across the cold cases. Actual cache counts remain the source of truth.
    let mut prompt = format!("Experiment case {id}. Read the following synthetic inventory.\n");
    for row in 0..rows {
        prompt.push_str(&format!(
            "Item {row:05}: warehouse north; blue bolts {}; green washers {}; inspection passed.\n",
            11 + row % 83,
            3 + row % 47
        ));
    }
    prompt.push_str(
        "Explain how to audit this inventory. Give a detailed numbered checklist, \
         including counting, sampling, discrepancies, and documentation. Continue \
         until you have explained at least forty practical checks.",
    );
    json!({
        "id": id,
        "body": {
            "model": "qwen3.8-27b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 1393,
            "max_tokens": max_tokens,
            "stream": true,
            "stream_options": {"include_usage": true},
            "chat_template_kwargs": {"enable_thinking": false}
        }
    })
}
