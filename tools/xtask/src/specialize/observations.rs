use serde::Serialize;
use serde_json::Value;

#[derive(Default)]
pub(super) struct StreamObservations {
    records: Vec<StreamRecord>,
    first_visible_ms: Option<f64>,
    last_elapsed_ms: Option<f64>,
    done: bool,
}

impl StreamObservations {
    pub(super) fn push_data(&mut self, data: &str, elapsed_ms: f64) -> Result<(), String> {
        if self.done {
            return Err("stream record received after [DONE]".to_string());
        }
        validate_elapsed(elapsed_ms)?;
        validate_monotonic(self.last_elapsed_ms, elapsed_ms)?;

        if data.trim() == "[DONE]" {
            self.done = true;
            self.last_elapsed_ms = Some(elapsed_ms);
            return Ok(());
        }

        let value: Value = serde_json::from_str(data)
            .map_err(|_| "stream event contains malformed JSON".to_string())?;
        if is_error_object(&value) {
            return Err("stream returned an error object".to_string());
        }

        let visible = has_visible_delta(&value);
        self.records.push(StreamRecord {
            elapsed_ms,
            data: value,
        });
        self.last_elapsed_ms = Some(elapsed_ms);
        if self.first_visible_ms.is_none() && visible {
            self.first_visible_ms = Some(elapsed_ms);
        }
        Ok(())
    }

    pub(super) fn finish(self, total_ms: f64) -> Result<TrialObservation, String> {
        validate_elapsed(total_ms)
            .map_err(|_| "total time must be finite and nonnegative".to_string())?;
        if !self.done {
            return Err("stream ended without [DONE]".to_string());
        }
        validate_total_time(&self.records, self.last_elapsed_ms, total_ms)?;

        let usage = terminal_field(&self.records, "usage")?;
        let timings = terminal_field(&self.records, "timings")?;
        let (prompt_tokens, completion_tokens) = validate_usage(&usage)?;
        validate_timings(&timings, prompt_tokens, completion_tokens)?;
        let finish_reason = terminal_finish_reason(&self.records);

        Ok(TrialObservation {
            first_visible_ms: self.first_visible_ms,
            total_ms,
            usage,
            timings,
            finish_reason,
            records: self.records,
        })
    }
}

#[derive(Serialize)]
pub(super) struct StreamRecord {
    pub(super) elapsed_ms: f64,
    pub(super) data: Value,
}

#[derive(Serialize)]
pub(super) struct TrialObservation {
    pub(super) first_visible_ms: Option<f64>,
    pub(super) total_ms: f64,
    pub(super) usage: Value,
    pub(super) timings: Value,
    pub(super) finish_reason: Option<String>,
    pub(super) records: Vec<StreamRecord>,
}

fn validate_elapsed(elapsed_ms: f64) -> Result<(), String> {
    if !elapsed_ms.is_finite() || elapsed_ms < 0.0 {
        return Err("elapsed time must be finite and nonnegative".to_string());
    }
    Ok(())
}

fn validate_monotonic(previous_ms: Option<f64>, elapsed_ms: f64) -> Result<(), String> {
    if previous_ms.is_some_and(|previous| elapsed_ms < previous) {
        return Err("stream elapsed times must be monotonic".to_string());
    }
    Ok(())
}

fn validate_total_time(
    records: &[StreamRecord],
    last_elapsed_ms: Option<f64>,
    total_ms: f64,
) -> Result<(), String> {
    if last_elapsed_ms.is_some_and(|elapsed_ms| elapsed_ms > total_ms)
        || records.iter().any(|record| record.elapsed_ms > total_ms)
    {
        return Err("stream elapsed time exceeds total time".to_string());
    }
    Ok(())
}

fn is_error_object(value: &Value) -> bool {
    value
        .as_object()
        .and_then(|object| object.get("error"))
        .is_some_and(|error| !error.is_null())
}

fn has_visible_delta(event: &Value) -> bool {
    event
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(|choices| choices.iter().any(choice_has_visible_delta))
}

fn choice_has_visible_delta(choice: &Value) -> bool {
    let Some(delta) = choice.get("delta").and_then(Value::as_object) else {
        return false;
    };

    ["content", "reasoning_content", "reasoning"]
        .iter()
        .any(|field| delta.get(*field).is_some_and(has_nonempty_value))
        || delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .is_some_and(|calls| calls.iter().any(tool_call_has_visible_piece))
}

fn has_nonempty_value(value: &Value) -> bool {
    match value {
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => items.iter().any(has_nonempty_value),
        Value::Object(fields) => fields
            .iter()
            .any(|(name, value)| name != "type" && has_nonempty_value(value)),
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}

fn tool_call_has_visible_piece(call: &Value) -> bool {
    let has_piece = |value: &Value| value.as_str().is_some_and(|text| !text.is_empty());
    if call.get("name").is_some_and(&has_piece) || call.get("arguments").is_some_and(&has_piece) {
        return true;
    }

    call.get("function").is_some_and(|function| {
        function.get("name").is_some_and(&has_piece)
            || function.get("arguments").is_some_and(&has_piece)
    })
}

fn terminal_field(records: &[StreamRecord], field: &str) -> Result<Value, String> {
    records
        .iter()
        .rev()
        .find_map(|record| record.data.get(field))
        .cloned()
        .ok_or_else(|| format!("stream is missing terminal {field}"))
}

fn terminal_finish_reason(records: &[StreamRecord]) -> Option<String> {
    records.iter().rev().find_map(|record| {
        record
            .data
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| {
                choices
                    .iter()
                    .rev()
                    .find_map(|choice| choice.get("finish_reason").and_then(Value::as_str))
            })
            .map(str::to_owned)
    })
}

fn validate_usage(usage: &Value) -> Result<(u64, u64), String> {
    let prompt_tokens = required_integer(usage, "prompt_tokens", "usage")?;
    let completion_tokens = required_integer(usage, "completion_tokens", "usage")?;
    Ok((prompt_tokens, completion_tokens))
}

fn validate_timings(
    timings: &Value,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> Result<(), String> {
    for field in [
        "prompt_ms",
        "predicted_ms",
        "prompt_per_second",
        "predicted_per_second",
    ] {
        let value = timings
            .get(field)
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("terminal timings is missing numeric {field}"))?;
        if !value.is_finite() || value < 0.0 {
            return Err(format!(
                "terminal timings {field} must be finite and nonnegative"
            ));
        }
    }

    let cache_n = required_integer(timings, "cache_n", "timings")?;
    let prompt_n = required_integer(timings, "prompt_n", "timings")?;
    let predicted_n = required_integer(timings, "predicted_n", "timings")?;
    let counted_prompt_tokens = cache_n
        .checked_add(prompt_n)
        .ok_or_else(|| "terminal timing input token count overflowed".to_string())?;
    if counted_prompt_tokens != prompt_tokens {
        return Err("terminal usage and timing input token counts disagree".to_string());
    }
    if predicted_n != completion_tokens {
        return Err("terminal usage and timing output token counts disagree".to_string());
    }
    Ok(())
}

fn required_integer(value: &Value, field: &str, parent: &str) -> Result<u64, String> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("terminal {parent} is missing nonnegative integer {field}"))
}

#[cfg(test)]
mod tests {
    use super::{StreamObservations, TrialObservation};
    use serde_json::{Value, json};

    fn valid_terminal(
        prompt_tokens: u64,
        completion_tokens: u64,
        cache_n: u64,
        prompt_n: u64,
        predicted_n: u64,
    ) -> Value {
        json!({
            "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            },
            "timings": {
                "prompt_ms": 1.0,
                "predicted_ms": 2.0,
                "prompt_per_second": 10.0,
                "predicted_per_second": 5.0,
                "cache_n": cache_n,
                "prompt_n": prompt_n,
                "predicted_n": predicted_n
            }
        })
    }

    fn push_json(
        observations: &mut StreamObservations,
        event: Value,
        elapsed_ms: f64,
    ) -> Result<(), String> {
        observations.push_data(&event.to_string(), elapsed_ms)
    }

    fn finish_single_event(event: Value) -> Result<TrialObservation, String> {
        let mut observations = StreamObservations::default();
        push_json(&mut observations, event, 1.0)?;
        observations.push_data("[DONE]", 2.0)?;
        observations.finish(2.0)
    }

    #[test]
    fn role_only_does_not_count_but_reasoning_does() {
        let mut observations = StreamObservations::default();
        push_json(
            &mut observations,
            json!({"choices": [{"delta": {"role": "assistant"}}]}),
            1.0,
        )
        .unwrap();
        push_json(
            &mut observations,
            json!({"choices": [{"delta": {"content": "", "reasoning": ""}}]}),
            1.5,
        )
        .unwrap();
        push_json(
            &mut observations,
            json!({"choices": [{"delta": {"reasoning_content": "thinking"}}]}),
            2.0,
        )
        .unwrap();
        push_json(&mut observations, valid_terminal(3, 1, 0, 3, 1), 3.0).unwrap();
        observations.push_data("[DONE]", 4.0).unwrap();

        let observation = observations.finish(5.0).unwrap();
        assert_eq!(observation.first_visible_ms, Some(2.0));
        assert_eq!(observation.finish_reason.as_deref(), Some("stop"));
        assert_eq!(observation.records.len(), 4);
        assert_eq!(observation.records[0].elapsed_ms, 1.0);
    }

    #[test]
    fn one_completion_token_can_have_visible_content() {
        let mut observations = StreamObservations::default();
        push_json(
            &mut observations,
            json!({"choices": [{"delta": {"content": "x"}}]}),
            1.0,
        )
        .unwrap();
        push_json(&mut observations, valid_terminal(2, 1, 0, 2, 1), 2.0).unwrap();
        observations.push_data("[DONE]", 3.0).unwrap();

        let observation = observations.finish(3.0).unwrap();
        assert_eq!(observation.first_visible_ms, Some(1.0));
        assert_eq!(observation.usage["completion_tokens"], 1);
    }

    #[test]
    fn zero_token_completion_allows_no_visible_chunk() {
        let observation = finish_single_event(valid_terminal(0, 0, 0, 0, 0)).unwrap();

        assert_eq!(observation.first_visible_ms, None);
        assert_eq!(observation.usage["completion_tokens"], 0);
    }

    #[test]
    fn buffered_completion_can_have_no_visible_chunk() {
        let observation = finish_single_event(valid_terminal(2, 1, 0, 2, 1)).unwrap();

        assert_eq!(observation.first_visible_ms, None);
        assert_eq!(observation.usage["completion_tokens"], 1);
    }

    #[test]
    fn cached_input_tokens_are_included_in_prompt_total() {
        let observation = finish_single_event(valid_terminal(10, 0, 7, 3, 0)).unwrap();

        assert_eq!(observation.timings["cache_n"], 7);
        assert_eq!(observation.timings["prompt_n"], 3);
    }

    #[test]
    fn tool_name_and_argument_pieces_count_as_visible() {
        let mut observations = StreamObservations::default();
        push_json(
            &mut observations,
            json!({"choices": [{"delta": {"tool_calls": [{
                "index": 0,
                "function": {"name": "search", "arguments": "{"}
            }]}}]}),
            1.0,
        )
        .unwrap();
        push_json(&mut observations, valid_terminal(1, 1, 0, 1, 1), 2.0).unwrap();
        observations.push_data("[DONE]", 3.0).unwrap();

        assert_eq!(
            observations.finish(3.0).unwrap().first_visible_ms,
            Some(1.0)
        );
    }

    #[test]
    fn malformed_and_error_events_are_rejected() {
        let mut observations = StreamObservations::default();
        assert!(observations.push_data("not json", 1.0).is_err());
        assert!(
            observations
                .push_data(r#"{"error":{"message":"failed"}}"#, 1.0)
                .is_err()
        );
    }

    #[test]
    fn truncated_stream_and_missing_terminal_fields_are_rejected() {
        let observations = StreamObservations::default();
        assert!(observations.finish(0.0).is_err());

        let observation = finish_single_event(json!({"choices": []}));
        assert!(observation.is_err());
    }

    #[test]
    fn usage_and_timing_count_disagreement_is_rejected() {
        assert!(finish_single_event(valid_terminal(4, 1, 2, 1, 1)).is_err());
        assert!(finish_single_event(valid_terminal(3, 1, 0, 3, 0)).is_err());
        assert!(finish_single_event(valid_terminal(3, 1, 0, 3, u64::MAX)).is_err());
        assert!(finish_single_event(valid_terminal(0, 0, u64::MAX, 1, 0)).is_err());
    }

    #[test]
    fn records_after_done_are_rejected() {
        let mut observations = StreamObservations::default();
        observations.push_data("[DONE]", 1.0).unwrap();

        assert!(observations.push_data("{}", 2.0).is_err());
        assert!(observations.push_data("[DONE]", 2.0).is_err());
    }

    #[test]
    fn elapsed_times_must_be_finite_monotonic_and_within_total() {
        let mut invalid = StreamObservations::default();
        assert!(invalid.push_data("{}", f64::NAN).is_err());
        assert!(invalid.push_data("{}", -1.0).is_err());

        let mut non_monotonic = StreamObservations::default();
        push_json(&mut non_monotonic, json!({}), 2.0).unwrap();
        assert!(push_json(&mut non_monotonic, json!({}), 1.0).is_err());

        let mut too_long = StreamObservations::default();
        push_json(&mut too_long, json!({}), 2.0).unwrap();
        too_long.push_data("[DONE]", 3.0).unwrap();
        assert!(too_long.finish(2.5).is_err());
        assert!(StreamObservations::default().finish(f64::INFINITY).is_err());
    }

    #[test]
    fn timing_fields_must_be_finite_and_nonnegative() {
        let mut event = valid_terminal(0, 0, 0, 0, 0);
        event["timings"]["prompt_ms"] = json!(-0.1);
        assert!(finish_single_event(event).is_err());

        let mut event = valid_terminal(0, 0, 0, 0, 0);
        event["timings"]["predicted_per_second"] = Value::Null;
        assert!(finish_single_event(event).is_err());
    }
}
