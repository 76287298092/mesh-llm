use super::observations::{StreamObservations, TrialObservation};
use crate::command::{DynResult, write_json_file};
use reqwest::blocking::{Client, Response};
use reqwest::redirect::Policy;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAX_REQUESTS: usize = 12;
const MAX_BODY_BYTES: usize = 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 32 * 1024 * 1024;
const MAX_EVENT_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
struct RunArgs {
    plan: PathBuf,
    output: PathBuf,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema_version: u32,
    base_url: String,
    api_key_env: String,
    timeout_seconds: u64,
    max_total_seconds: u64,
    requests: Vec<PlanRequest>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PlanRequest {
    id: String,
    body: Value,
}

#[derive(Serialize)]
struct TrialRecord<'a> {
    id: &'a str,
    request_body: &'a Value,
    start_unix_ms: u128,
    elapsed_ms: f64,
    observation: Option<TrialObservation>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct RunSummary {
    completed_ids: Vec<String>,
    elapsed_ms: f64,
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let args = parse_args(args)?;
    let mut plan = load_plan(&args.plan)?;
    validate_plan(&mut plan)?;
    let api_key = read_api_key(&plan.api_key_env)?;
    let client = build_client()?;

    fs::create_dir(&args.output)?;
    write_new_json_file(&args.output.join("plan.json"), &plan)?;

    let run_started = Instant::now();
    let mut completed_ids = Vec::with_capacity(plan.requests.len());
    for request in &plan.requests {
        let started_at_unix_ms = unix_time_millis()?;
        let request_started = Instant::now();
        let result = execute_request(
            &client,
            &plan,
            request,
            &api_key,
            run_started,
            request_started,
        );
        let elapsed_ms = request_started.elapsed().as_secs_f64() * 1_000.0;
        let (observation, error) = match result {
            Ok(observation) => (Some(observation), None),
            Err(error) => (None, Some(error)),
        };
        let record = TrialRecord {
            id: &request.id,
            request_body: &request.body,
            start_unix_ms: started_at_unix_ms,
            elapsed_ms,
            observation,
            error: error.clone(),
        };
        write_new_json_file(
            &args.output.join(format!("trial-{}.json", request.id)),
            &record,
        )?;

        if let Some(error) = error {
            return Err(format!("request {} failed: {error}", request.id).into());
        }
        completed_ids.push(request.id.clone());
    }

    let summary = RunSummary {
        completed_ids,
        elapsed_ms: run_started.elapsed().as_secs_f64() * 1_000.0,
    };
    write_json_file(&args.output.join("summary.json"), &summary)?;
    crate::command::print_json(&summary)?;
    Ok(())
}

fn parse_args(args: &[String]) -> DynResult<RunArgs> {
    let mut plan = None;
    let mut output = None;
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = args.get(index + 1).filter(|value| !value.starts_with("--"));
        let value = value.ok_or_else(|| format!("missing value for {flag}"))?;
        match flag {
            "--plan" => set_once(&mut plan, PathBuf::from(value), flag)?,
            "--output" => set_once(&mut output, PathBuf::from(value), flag)?,
            _ => return Err(format!("unknown argument: {flag}").into()),
        }
        index += 2;
    }
    let plan = plan.ok_or_else(|| "missing required argument: --plan".to_string())?;
    let output = output.ok_or_else(|| "missing required argument: --output".to_string())?;
    Ok(RunArgs { plan, output })
}

fn set_once<T>(slot: &mut Option<T>, value: T, flag: &str) -> DynResult<()> {
    if slot.is_some() {
        return Err(format!("duplicate argument: {flag}").into());
    }
    *slot = Some(value);
    Ok(())
}

fn load_plan(path: &Path) -> DynResult<Plan> {
    let contents = fs::read(path)?;
    Ok(serde_json::from_slice(&contents)?)
}

fn validate_plan(plan: &mut Plan) -> DynResult<()> {
    if plan.schema_version != 1 {
        return Err("unsupported plan schema_version; expected 1".into());
    }
    validate_base_url(&plan.base_url)?;
    validate_env_name(&plan.api_key_env)?;
    if !(1..=180).contains(&plan.timeout_seconds) {
        return Err("timeout_seconds must be between 1 and 180".into());
    }
    if !(1..=1800).contains(&plan.max_total_seconds) {
        return Err("max_total_seconds must be between 1 and 1800".into());
    }
    if !(1..=MAX_REQUESTS).contains(&plan.requests.len()) {
        return Err(format!("requests must contain between 1 and {MAX_REQUESTS} entries").into());
    }

    let mut ids = std::collections::BTreeSet::new();
    for request in &mut plan.requests {
        validate_request_id(&request.id)?;
        if !ids.insert(request.id.clone()) {
            return Err(format!("duplicate request id: {}", request.id).into());
        }
        normalize_request_body(&mut request.body)?;
    }
    Ok(())
}

fn validate_base_url(base_url: &str) -> DynResult<()> {
    let authority = base_url
        .strip_prefix("http://127.0.0.1:")
        .or_else(|| base_url.strip_prefix("http://localhost:"))
        .ok_or_else(|| {
            "base_url must use http://127.0.0.1:PORT/v1 or http://localhost:PORT/v1".to_string()
        })?;
    let port = authority
        .strip_suffix("/v1")
        .ok_or_else(|| "base_url must end with exactly /v1".to_string())?;
    if port.is_empty() || !port.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err("base_url must include a valid loopback port".into());
    }
    let port = port
        .parse::<u16>()
        .map_err(|_| "base_url must include a valid loopback port")?;
    if port == 0 {
        return Err("base_url port must be between 1 and 65535".into());
    }
    Ok(())
}

fn validate_env_name(name: &str) -> DynResult<()> {
    let mut bytes = name.bytes();
    let Some(first) = bytes.next() else {
        return Err("api_key_env must be an environment variable name".into());
    };
    if !(first == b'_' || first.is_ascii_alphabetic())
        || !bytes.all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
    {
        return Err("api_key_env must be an environment variable name".into());
    }
    Ok(())
}

fn validate_request_id(id: &str) -> DynResult<()> {
    if id.is_empty()
        || id.len() > 80
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    {
        return Err("request id must be 1-80 ASCII letters, digits, '_' or '-'".into());
    }
    Ok(())
}

fn normalize_request_body(body: &mut Value) -> DynResult<()> {
    let Value::Object(object) = body else {
        return Err("each request body must be a JSON object".into());
    };
    let model = object.get("model").and_then(Value::as_str);
    if model.is_none_or(|model| model.trim().is_empty()) {
        return Err("each request body must include a nonempty model".into());
    }
    validate_token_limits(object)?;

    object.insert("stream".to_string(), Value::Bool(true));
    let stream_options = object
        .entry("stream_options".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    if !stream_options.is_object() {
        *stream_options = Value::Object(Map::new());
    }
    stream_options
        .as_object_mut()
        .expect("stream_options was normalized to an object")
        .insert("include_usage".to_string(), Value::Bool(true));

    if serde_json::to_vec(body)?.len() > MAX_BODY_BYTES {
        return Err("serialized request body exceeds 1 MiB".into());
    }
    Ok(())
}

fn validate_token_limits(object: &Map<String, Value>) -> DynResult<()> {
    let max_tokens = token_limit(object, "max_tokens")?;
    let max_completion_tokens = token_limit(object, "max_completion_tokens")?;
    if max_tokens.is_none() && max_completion_tokens.is_none() {
        return Err("each request body must set max_tokens or max_completion_tokens".into());
    }
    if matches!((max_tokens, max_completion_tokens), (Some(left), Some(right)) if left != right) {
        return Err("max_tokens and max_completion_tokens must not conflict".into());
    }
    Ok(())
}

fn token_limit(object: &Map<String, Value>, key: &str) -> DynResult<Option<u64>> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    let value = value
        .as_u64()
        .filter(|value| (1..=512).contains(value))
        .ok_or_else(|| format!("{key} must be an integer between 1 and 512"))?;
    Ok(Some(value))
}

fn read_api_key(name: &str) -> DynResult<String> {
    match std::env::var(name) {
        Ok(value) if !value.is_empty() => Ok(value),
        _ => Err(format!("API key environment variable {name} is missing or empty").into()),
    }
}

fn build_client() -> DynResult<Client> {
    Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .no_proxy()
        .redirect(Policy::none())
        .build()
        .map_err(|_| "failed to initialize HTTP client".into())
}

fn execute_request(
    client: &Client,
    plan: &Plan,
    request: &PlanRequest,
    api_key: &str,
    run_started: Instant,
    request_started: Instant,
) -> Result<TrialObservation, String> {
    let overall = Duration::from_secs(plan.max_total_seconds);
    let remaining = overall
        .checked_sub(run_started.elapsed())
        .filter(|remaining| !remaining.is_zero())
        .ok_or_else(|| "overall time budget expired before request".to_string())?;
    let timeout = remaining.min(Duration::from_secs(plan.timeout_seconds));
    let url = format!("{}/chat/completions", plan.base_url);
    let response = client
        .post(url)
        .bearer_auth(api_key)
        .json(&request.body)
        .timeout(timeout)
        .send()
        .map_err(|error| {
            if error.is_timeout() {
                "HTTP request timed out".to_string()
            } else {
                "HTTP request transport failed".to_string()
            }
        })?;
    if !response.status().is_success() {
        return Err(format!(
            "server returned HTTP status {}",
            response.status().as_u16()
        ));
    }
    read_observation(response, request_started)
}

fn read_observation(
    response: Response,
    request_started: Instant,
) -> Result<TrialObservation, String> {
    let mut observations = StreamObservations::default();
    let mut reader = BufReader::new(response);
    let saw_done = read_sse_events(&mut reader, |data| {
        let elapsed_ms = request_started.elapsed().as_secs_f64() * 1_000.0;
        observations.push_data(data, elapsed_ms)
    })?;
    if !saw_done {
        return Err("SSE stream ended before [DONE]".to_string());
    }
    let total_ms = request_started.elapsed().as_secs_f64() * 1_000.0;
    observations.finish(total_ms)
}

fn read_sse_events<R, F>(reader: &mut R, mut on_data: F) -> Result<bool, String>
where
    R: BufRead,
    F: FnMut(&str) -> Result<(), String>,
{
    let mut response_bytes = 0usize;
    let mut event_bytes = 0usize;
    let mut data_lines = Vec::new();
    loop {
        let mut bytes = Vec::new();
        if !read_bounded_line(reader, &mut bytes, &mut response_bytes)? {
            return Ok(false);
        }
        event_bytes = event_bytes
            .checked_add(bytes.len())
            .ok_or_else(|| "SSE event exceeds size limit".to_string())?;
        if event_bytes > MAX_EVENT_BYTES {
            return Err("SSE event exceeds 1 MiB".to_string());
        }
        let line = std::str::from_utf8(&bytes)
            .map_err(|_| "SSE response contains invalid UTF-8".to_string())?;
        let line = line.strip_suffix('\n').unwrap_or(line);
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() {
            if !data_lines.is_empty() {
                let data = data_lines.join("\n");
                data_lines.clear();
                let is_done = data == "[DONE]";
                on_data(&data)?;
                if is_done {
                    return Ok(true);
                }
            }
            event_bytes = 0;
        } else if line.starts_with(':') {
            continue;
        } else if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.strip_prefix(' ').unwrap_or(data).to_string());
        } else if line == "data" {
            data_lines.push(String::new());
        }
    }
}

fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    line: &mut Vec<u8>,
    response_bytes: &mut usize,
) -> Result<bool, String> {
    loop {
        let available = reader
            .fill_buf()
            .map_err(|_| "failed to read SSE response".to_string())?;
        if available.is_empty() {
            return Ok(!line.is_empty());
        }
        let count = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |position| position + 1);
        let next_response_bytes = response_bytes
            .checked_add(count)
            .ok_or_else(|| "SSE response exceeds size limit".to_string())?;
        if next_response_bytes > MAX_RESPONSE_BYTES {
            return Err("SSE response exceeds 32 MiB".to_string());
        }
        if line.len().saturating_add(count) > MAX_EVENT_BYTES {
            return Err("SSE line exceeds 1 MiB".to_string());
        }
        line.extend_from_slice(&available[..count]);
        reader.consume(count);
        *response_bytes = next_response_bytes;
        if line.last() == Some(&b'\n') {
            return Ok(true);
        }
    }
}

fn unix_time_millis() -> DynResult<u128> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())
}

fn write_new_json_file<T: Serialize>(path: &Path, value: &T) -> DynResult<()> {
    let contents = serde_json::to_vec_pretty(value)?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&contents)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parses_only_the_two_required_arguments() {
        let args = strings(["--output", "out", "--plan", "plan.json"]);
        let parsed = parse_args(&args).expect("valid arguments");
        assert_eq!(parsed.plan, PathBuf::from("plan.json"));
        assert_eq!(parsed.output, PathBuf::from("out"));
        assert!(parse_args(&strings(["--plan", "p", "--plan", "q", "--output", "o"])).is_err());
        assert!(parse_args(&strings(["--plan", "p", "--output", "o", "--extra", "x"])).is_err());
    }

    #[test]
    fn validates_and_normalizes_bounded_request_plans() {
        let mut plan = sample_plan();
        plan.requests[0].body["stream"] = Value::Bool(false);
        plan.requests[0].body["stream_options"] = Value::String("ignored".to_string());
        validate_plan(&mut plan).expect("valid plan");
        assert_eq!(plan.requests[0].body["stream"], Value::Bool(true));
        assert_eq!(
            plan.requests[0].body["stream_options"]["include_usage"],
            Value::Bool(true)
        );

        let mut conflicting = sample_plan();
        conflicting.requests[0].body["max_completion_tokens"] = Value::from(2);
        assert!(validate_plan(&mut conflicting).is_err());
        let mut unsafe_url = sample_plan();
        unsafe_url.base_url = "http://user:pass@localhost:8000/v1".to_string();
        assert!(validate_plan(&mut unsafe_url).is_err());
    }

    #[test]
    fn frames_sse_data_and_stops_at_done() {
        let input = b": keepalive\r\ndata: {\r\ndata: \"n\":1}\r\n\r\ndata: [DONE]\r\n\r\ndata: ignored\n\n";
        let mut reader = Cursor::new(input);
        let mut records = Vec::new();
        let saw_done = read_sse_events(&mut reader, |data| {
            records.push(data.to_string());
            Ok(())
        })
        .expect("SSE framing");
        assert!(saw_done);
        assert_eq!(records, ["{\n\"n\":1}", "[DONE]"]);
    }

    #[test]
    fn does_not_dispatch_unterminated_sse_events() {
        let mut reader = Cursor::new(b"data: {\"n\":1}\n");
        let mut records = Vec::new();
        let saw_done = read_sse_events(&mut reader, |data| {
            records.push(data.to_string());
            Ok(())
        })
        .expect("SSE framing");
        assert!(!saw_done);
        assert!(records.is_empty());
    }

    #[test]
    fn feeds_framed_events_into_stream_observations() {
        let terminal = serde_json::json!({
            "choices": [{"delta": {"content": "ready"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            "timings": {
                "prompt_ms": 5.0,
                "predicted_ms": 2.0,
                "prompt_per_second": 600.0,
                "predicted_per_second": 500.0,
                "cache_n": 1,
                "prompt_n": 2,
                "predicted_n": 1
            }
        });
        let input = format!("data: {terminal}\n\ndata: [DONE]\n\n");
        let mut reader = Cursor::new(input.into_bytes());
        let mut observations = StreamObservations::default();
        let mut elapsed_ms = 1.0;
        let saw_done = read_sse_events(&mut reader, |data| {
            observations.push_data(data, elapsed_ms)?;
            elapsed_ms += 1.0;
            Ok(())
        })
        .expect("SSE framing and observation parsing");
        assert!(saw_done);
        let observation = observations
            .finish(elapsed_ms)
            .expect("complete observation");
        assert_eq!(observation.first_visible_ms, Some(1.0));
        assert_eq!(observation.usage["completion_tokens"], 1);
    }

    #[test]
    fn enforces_sse_response_and_event_bounds() {
        let too_large_line = vec![b'a'; MAX_EVENT_BYTES + 1];
        let mut reader = Cursor::new(too_large_line);
        assert!(read_sse_events(&mut reader, |_| Ok(())).is_err());

        let mut bounded_event = vec![b':'];
        bounded_event.resize(MAX_EVENT_BYTES - 2, b'x');
        bounded_event.extend_from_slice(b"\n\n");
        let mut response = Vec::with_capacity(MAX_RESPONSE_BYTES + MAX_EVENT_BYTES);
        for _ in 0..=MAX_RESPONSE_BYTES / MAX_EVENT_BYTES {
            response.extend_from_slice(&bounded_event);
        }
        let mut reader = Cursor::new(response);
        assert!(read_sse_events(&mut reader, |_| Ok(())).is_err());
    }

    fn sample_plan() -> Plan {
        Plan {
            schema_version: 1,
            base_url: "http://127.0.0.1:8000/v1".to_string(),
            api_key_env: "NINFER_API_KEY".to_string(),
            timeout_seconds: 30,
            max_total_seconds: 600,
            requests: vec![PlanRequest {
                id: "short-1".to_string(),
                body: serde_json::json!({"model":"model-id","max_tokens":1}),
            }],
        }
    }

    fn strings<const N: usize>(values: [&str; N]) -> Vec<String> {
        values.into_iter().map(str::to_owned).collect()
    }
}
