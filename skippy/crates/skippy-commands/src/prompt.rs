//! Interactive client for a running OpenAI-compatible Skippy endpoint.

use std::{
    io::{self, BufRead, BufReader, IsTerminal},
    path::PathBuf,
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use reqwest::blocking::Client;
use rustyline::{DefaultEditor, error::ReadlineError};
use serde_json::{Value, json};

use crate::console;

pub struct PromptCommand {
    pub endpoint: String,
    pub model: Option<String>,
    pub max_new_tokens: u32,
    pub raw: bool,
    pub no_think: bool,
    pub history_path: Option<PathBuf>,
}

pub fn run(args: PromptCommand) -> Result<()> {
    if args.max_new_tokens == 0 {
        bail!("--max-new-tokens must be greater than zero");
    }
    let endpoint = args.endpoint.trim_end_matches('/');
    let client = Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .context("build prompt HTTP client")?;
    let model = match args.model.as_deref() {
        Some(model) if !model.trim().is_empty() => model.to_string(),
        Some(_) => bail!("--model cannot be empty"),
        None => first_model(&client, endpoint)?,
    };
    let mut editor = if io::stdin().is_terminal() {
        let mut editor = DefaultEditor::new().context("initialize prompt history")?;
        if let Some(path) = args.history_path.as_ref()
            && path.exists()
        {
            editor.load_history(path).context("load prompt history")?;
        }
        Some(editor)
    } else {
        None
    };
    let stdin = io::stdin();
    let mut stdin = editor.is_none().then(|| stdin.lock());
    let mut messages = Vec::new();
    let mut entered = Vec::new();
    console::write_status(&format!(
        "Connected to {endpoint} as {model}. Enter :quit to exit, :reset to clear chat, or :history."
    ))?;
    loop {
        let Some(line) = read_line(editor.as_mut(), stdin.as_mut())? else {
            break;
        };
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        match input {
            ":quit" | ":q" | ":exit" => break,
            ":reset" => {
                messages.clear();
                console::write_status("Chat reset.")?;
                continue;
            }
            ":history" => {
                for (index, prompt) in entered.iter().enumerate() {
                    console::write_line(&format!("{}: {prompt}", index + 1))?;
                }
                continue;
            }
            _ => {}
        }
        if let Some(editor) = editor.as_mut() {
            editor.add_history_entry(input)?;
        }
        entered.push(input.to_string());
        let assistant = if args.raw {
            let body = request_body(&model, input, &[], &args);
            stream_completion(&client, endpoint, "completions", body, true)?
        } else {
            messages.push(json!({"role": "user", "content": input}));
            let body = request_body(&model, input, &messages, &args);
            match stream_completion(&client, endpoint, "chat/completions", body, false) {
                Ok(assistant) => assistant,
                Err(error) => {
                    messages.pop();
                    return Err(error);
                }
            }
        };
        if !args.raw {
            messages.push(json!({"role": "assistant", "content": assistant}));
        }
    }
    if let (Some(editor), Some(path)) = (editor.as_mut(), args.history_path.as_ref()) {
        editor.save_history(path).context("save prompt history")?;
    }
    Ok(())
}

fn read_line(
    editor: Option<&mut DefaultEditor>,
    stdin: Option<&mut io::StdinLock<'_>>,
) -> Result<Option<String>> {
    if let Some(editor) = editor {
        return match editor.readline("> ") {
            Ok(line) => Ok(Some(line)),
            Err(ReadlineError::Eof) => Ok(None),
            Err(ReadlineError::Interrupted) => {
                console::write_status("^C")?;
                Ok(Some(String::new()))
            }
            Err(error) => Err(error).context("read prompt"),
        };
    }
    let mut line = String::new();
    let stdin = stdin.context("standard input is unavailable")?;
    if stdin.read_line(&mut line).context("read prompt")? == 0 {
        Ok(None)
    } else {
        Ok(Some(line))
    }
}

fn first_model(client: &Client, endpoint: &str) -> Result<String> {
    let response = client
        .get(format!("{endpoint}/models"))
        .send()
        .context("list models")?
        .error_for_status()
        .context("list models")?;
    let body: Value = response.json().context("parse model list")?;
    body["data"][0]["id"]
        .as_str()
        .map(str::to_string)
        .context("/models returned no model ID; pass --model")
}

fn request_body(model: &str, input: &str, messages: &[Value], args: &PromptCommand) -> Value {
    let mut body = if args.raw {
        json!({"model": model, "prompt": input, "stream": true, "max_tokens": args.max_new_tokens})
    } else {
        json!({"model": model, "messages": messages, "stream": true, "max_tokens": args.max_new_tokens})
    };
    if args.no_think {
        body["reasoning_effort"] = json!("none");
    }
    body
}

fn stream_completion(
    client: &Client,
    endpoint: &str,
    route: &str,
    body: Value,
    raw: bool,
) -> Result<String> {
    let response = client
        .post(format!("{endpoint}/{route}"))
        .json(&body)
        .send()
        .with_context(|| format!("request {route}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let detail = response.text().unwrap_or_default();
        bail!("{route} returned {status}: {detail}");
    }
    print_stream(BufReader::new(response), raw)
}

fn print_stream(reader: impl BufRead, raw: bool) -> Result<String> {
    let mut output = String::new();
    let mut done = false;
    for line in reader.lines() {
        let line = line.context("read completion stream")?;
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            done = true;
            break;
        }
        let chunk: Value = serde_json::from_str(data).context("parse completion event")?;
        if let Some(error) = chunk.get("error") {
            bail!("completion stream failed: {error}");
        }
        let text = if raw {
            chunk["choices"][0]["text"].as_str()
        } else {
            chunk["choices"][0]["delta"]["content"].as_str()
        };
        if let Some(text) = text {
            console::write_text(text).context("write completion")?;
            output.push_str(text);
        }
    }
    ensure!(done, "completion stream ended before [DONE]");
    console::write_line("")?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::print_stream;

    #[test]
    fn chat_stream_requires_done_after_content() {
        let complete =
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: [DONE]\n";
        assert_eq!(print_stream(Cursor::new(complete), false).unwrap(), "hello");
        let truncated = b"data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n";
        assert!(
            print_stream(Cursor::new(truncated), false)
                .unwrap_err()
                .to_string()
                .contains("before [DONE]")
        );
    }

    #[test]
    fn raw_stream_reads_completion_text() {
        let body = b"data: {\"choices\":[{\"text\":\"answer\"}]}\n\ndata: [DONE]\n";
        assert_eq!(print_stream(Cursor::new(body), true).unwrap(), "answer");
    }
}
