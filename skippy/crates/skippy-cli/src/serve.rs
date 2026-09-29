//! One public serving entry point with explicit internal stage transports.

use std::{
    future::Future,
    io::IsTerminal,
    net::SocketAddr,
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use skippy_commands::console::{self, OutputMode};
use tokio::{sync::oneshot, task::JoinHandle};

use crate::{
    cli::{
        OpenAiGuardrailsCliMode, ServeArgs, ServeBinaryArgs, ServeCommandArgs, ServeOpenAiArgs,
        StageTransport,
    },
    conversion, shutdown_signal,
};

pub async fn run(mut args: ServeCommandArgs) -> Result<()> {
    validate(&args)?;
    if let Some(model) = args.model.take() {
        let local = Path::new(&model).is_file() || Path::new(&model).is_dir();
        let path = if local {
            model.clone().into()
        } else {
            let cache = skippy_config::paths::model_cache_dir(None)?;
            skippy_commands::models::download_model(&cache, &model, None, None)
                .await?
                .primary_path
        };
        args.public.model_path = Some(path);
        if !local {
            args.public.model_id.get_or_insert(model);
        }
    }
    match args.stage_transport {
        Some(StageTransport::Http) => serve_http_stage(args).await,
        Some(StageTransport::Binary) => serve_binary_stage(args).await,
        None => serve_public(args).await,
    }
}

pub(crate) fn validate(args: &ServeCommandArgs) -> Result<()> {
    if args.worker_only && args.prompt {
        bail!("--prompt requires a public inference API");
    }
    if args.worker_only && args.stage_transport.is_none() {
        bail!("--worker-only requires --stage-transport");
    }
    if args.prompt && (!std::io::stdin().is_terminal() || console::mode() != OutputMode::Human) {
        bail!("--prompt requires an interactive terminal and human output");
    }
    if console::mode() == OutputMode::Json {
        bail!("serving produces a stream of events; use --output jsonl or --output human");
    }
    if args.public.config.is_none() && args.public.model_path.is_none() && args.model.is_none() {
        bail!("provide --model, --model-path, or --config");
    }
    if args.stage_transport == Some(StageTransport::Http) && !args.worker_only {
        bail!("HTTP stage transport requires --worker-only; use binary for a public stage-0 API");
    }
    Ok(())
}

async fn serve_public(args: ServeCommandArgs) -> Result<()> {
    let bind_addr = args.public.bind_addr;
    let startup_timeout = Duration::from_secs(args.public.startup_timeout_secs.max(1));
    let options = conversion::local_openai_options(args.public)?;
    let model_id = options
        .model_id
        .clone()
        .unwrap_or_else(|| options.config.model_id.clone());
    let (stop, stopped) = oneshot::channel();
    let shutdown = shutdown_signal()?;
    let server = skippy_api::serving::serve_local_openai_with_shutdown(options, async move {
        tokio::select! { _ = shutdown => {}, _ = stopped => {} }
    });
    serve_with_readiness(
        server,
        bind_addr,
        model_id,
        args.prompt,
        stop,
        startup_timeout,
    )
    .await
}

async fn serve_http_stage(args: ServeCommandArgs) -> Result<()> {
    let startup_timeout = Duration::from_secs(args.public.startup_timeout_secs.max(1));
    let config = args.public.config.context("--config is required")?;
    let options = conversion::stage_http_options(ServeArgs {
        config,
        topology: args.public.topology,
        bind_addr: None,
        metrics_otlp_grpc: args.public.metrics_otlp_grpc,
        telemetry_queue_capacity: args.public.telemetry_queue_capacity,
        telemetry_level: args.public.telemetry_level,
    })?;
    let bind_addr = options.bind_addr;
    let model_id = options.config.model_id.clone();
    let server = skippy_serving::http::serve_stage_http_with_shutdown(options, shutdown_signal()?);
    serve_worker_with_readiness(server, bind_addr, model_id, "http", startup_timeout).await
}

async fn serve_binary_stage(mut args: ServeCommandArgs) -> Result<()> {
    if args.worker_only && args.stage.openai_bind_addr.is_some() {
        bail!("--openai-bind-addr conflicts with --worker-only");
    }
    if args.public.openai_guardrails != OpenAiGuardrailsCliMode::Metrics {
        bail!("--openai-guardrails is not supported by the binary stage frontend");
    }
    apply_public_frontend_tuning(&args.public, &mut args.stage);
    let startup_timeout = Duration::from_secs(args.public.startup_timeout_secs.max(1));
    args.stage.config = args.public.config.context("--config is required")?;
    args.stage.topology = args.public.topology;
    args.stage.metrics_otlp_grpc = args.public.metrics_otlp_grpc;
    args.stage.telemetry_queue_capacity = args.public.telemetry_queue_capacity;
    args.stage.telemetry_level = args.public.telemetry_level;
    args.stage.worker_only = args.worker_only;
    args.stage.api_bind_addr = Some(args.public.bind_addr);
    let options = conversion::binary_stage_options(args.stage)?;
    let Some(openai) = options.openai.as_ref() else {
        if args.prompt {
            bail!("--prompt requires stage 0 to expose the public inference API");
        }
        if !args.worker_only {
            bail!("this stage has no public API; add --worker-only or serve stage 0");
        }
        let bind_addr = options.bind_addr;
        let model_id = options.config.model_id.clone();
        let server = skippy_serving::binary_transport::serve_binary_stage_with_shutdown(
            options,
            shutdown_signal()?,
        );
        return serve_worker_with_readiness(server, bind_addr, model_id, "binary", startup_timeout)
            .await;
    };
    let bind_addr = openai.bind_addr;
    let model_id = openai
        .model_id
        .clone()
        .unwrap_or_else(|| options.config.model_id.clone());
    let (stop, stopped) = oneshot::channel();
    let shutdown = shutdown_signal()?;
    let server =
        skippy_serving::binary_transport::serve_binary_stage_with_shutdown(options, async move {
            tokio::select! { _ = shutdown => {}, _ = stopped => {} }
        });
    serve_with_readiness(
        server,
        bind_addr,
        model_id,
        args.prompt,
        stop,
        startup_timeout,
    )
    .await
}

fn apply_public_frontend_tuning(public: &ServeOpenAiArgs, stage: &mut ServeBinaryArgs) {
    if let Some(value) = &public.model_id {
        stage.openai_model_id = Some(value.clone());
    }
    if public.default_max_tokens != 16 {
        stage.openai_default_max_tokens = public.default_max_tokens;
    }
    if let Some(value) = public.generation_concurrency {
        stage.openai_generation_concurrency = Some(value);
    }
    if public.adaptive_generation_concurrency {
        stage.openai_adaptive_generation_concurrency = true;
    }
    if let Some(value) = public.adaptive_generation_min_concurrency {
        stage.openai_adaptive_generation_min_concurrency = Some(value);
    }
    if let Some(value) = public.generation_queue_capacity {
        stage.openai_generation_queue_capacity = Some(value);
    }
    if public.generation_admission_timeout_secs
        != skippy_serving::frontend::DEFAULT_GENERATION_ADMISSION_TIMEOUT_SECS
    {
        stage.openai_generation_admission_timeout_secs = public.generation_admission_timeout_secs;
    }
    if public.prefill_chunk_size != 256 {
        stage.openai_prefill_chunk_size = public.prefill_chunk_size;
    }
    if public.prefill_chunk_policy != "adaptive-ramp" {
        stage.openai_prefill_chunk_policy = public.prefill_chunk_policy.clone();
    }
    if let Some(value) = &public.prefill_chunk_schedule {
        stage.openai_prefill_chunk_schedule = Some(value.clone());
    }
    if public.prefill_adaptive_start != 128 {
        stage.openai_prefill_adaptive_start = public.prefill_adaptive_start;
    }
    if public.prefill_adaptive_step != 128 {
        stage.openai_prefill_adaptive_step = public.prefill_adaptive_step;
    }
    if public.prefill_adaptive_max != 384 {
        stage.openai_prefill_adaptive_max = public.prefill_adaptive_max;
    }
    if public.prefill_adaptive_target_ms != 100.0 {
        stage.openai_prefill_adaptive_target_ms = public.prefill_adaptive_target_ms;
    }
    if let Some(value) = &public.speculative_config {
        stage.openai_speculative_config = Some(value.clone());
    }
}

async fn serve_worker_with_readiness(
    server: impl Future<Output = Result<()>> + Send + 'static,
    bind_addr: SocketAddr,
    model_id: String,
    transport: &str,
    startup_timeout: Duration,
) -> Result<()> {
    if bind_addr.port() == 0 {
        bail!("stage listener must use a fixed port so readiness can be reported");
    }
    let mut server = tokio::spawn(server);
    console::status(&format!("🧠 Loading {transport} stage for {model_id}"))?;
    let deadline = Instant::now() + startup_timeout;
    let probe = SocketAddr::new(
        if bind_addr.is_ipv4() {
            "127.0.0.1".parse()?
        } else {
            "::1".parse()?
        },
        bind_addr.port(),
    );
    loop {
        tokio::select! {
            outcome = &mut server => {
                outcome.context("join serving task")??;
                bail!("stage exited before its listener became ready");
            }
            connection = tokio::net::TcpStream::connect(probe) => {
                if connection.is_ok() && !server.is_finished() { break; }
            }
        }
        if Instant::now() >= deadline {
            bail!(
                "stage did not become ready at {bind_addr} within {} seconds",
                startup_timeout.as_secs()
            );
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    console::event(
        "ready",
        &serde_json::json!({
            "model_id": model_id, "transport": transport, "bind_addr": bind_addr,
        }),
    )?;
    if console::mode() == OutputMode::Human {
        console::status(&format!("✅ {transport} stage ready at {bind_addr}"))?;
    }
    server.await.context("join serving task")?
}

async fn serve_with_readiness(
    server: impl Future<Output = Result<()>> + Send + 'static,
    bind_addr: SocketAddr,
    model_id: String,
    prompt: bool,
    stop: oneshot::Sender<()>,
    startup_timeout: Duration,
) -> Result<()> {
    if bind_addr.port() == 0 {
        bail!("--bind-addr must use a fixed port so readiness can be reported");
    }
    let host = if bind_addr.is_ipv4() {
        "127.0.0.1"
    } else {
        "[::1]"
    };
    let api_base = format!("http://{host}:{}/v1", bind_addr.port());
    let mut server = tokio::spawn(server);
    console::status("🧠 Loading model and starting the API")?;
    wait_for_ready(&api_base, &model_id, &mut server, startup_timeout).await?;
    console::event(
        "ready",
        &serde_json::json!({"model_id":model_id,"api_base":api_base}),
    )?;
    if console::mode() == OutputMode::Human {
        console::status(&format!("✅ {model_id} ready at {api_base}"))?;
    }
    if !prompt {
        return server.await.context("join serving task")?;
    }
    let prompt_args = skippy_commands::prompt::PromptCommand {
        endpoint: api_base,
        model: Some(model_id),
        max_new_tokens: 128,
        raw: false,
        no_think: false,
        history_path: None,
    };
    let prompt_task =
        tokio::task::spawn_blocking(move || skippy_commands::prompt::run(prompt_args));
    let result = prompt_task.await.context("join interactive prompt")?;
    let _ = stop.send(());
    server.await.context("join serving task")??;
    result
}

async fn wait_for_ready(
    api_base: &str,
    model_id: &str,
    server: &mut JoinHandle<Result<()>>,
    startup_timeout: Duration,
) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()?;
    let deadline = Instant::now() + startup_timeout;
    loop {
        tokio::select! {
            outcome = &mut *server => {
                outcome.context("join serving task")??;
                bail!("server exited before the API became ready");
            }
            response = client.get(format!("{api_base}/models")).send() => {
                if let Ok(response) = response && response.status().is_success() {
                    let body: serde_json::Value = response.json().await.context("read ready model list")?;
                    let listed = body["data"].as_array().is_some_and(|items| items.iter().any(|item| item["id"] == model_id));
                    if listed && !server.is_finished() { return Ok(()); }
                }
            }
        }
        if Instant::now() >= deadline {
            bail!(
                "API did not become ready within {} seconds at {api_base}",
                startup_timeout.as_secs()
            );
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Command};
    use clap::Parser;

    #[test]
    fn common_frontend_flags_reach_binary_stage_zero() {
        let cli = Cli::try_parse_from([
            "skippy",
            "serve",
            "--config",
            "stage-0.json",
            "--stage-transport",
            "binary",
            "--model-id",
            "served-model",
            "--default-max-tokens",
            "64",
            "--generation-concurrency",
            "2",
            "--prefill-chunk-size",
            "512",
        ])
        .unwrap();
        let Command::Serve(mut args) = cli.command else {
            panic!("expected serve");
        };
        apply_public_frontend_tuning(&args.public, &mut args.stage);
        assert_eq!(args.stage.openai_model_id.as_deref(), Some("served-model"));
        assert_eq!(args.stage.openai_default_max_tokens, 64);
        assert_eq!(args.stage.openai_generation_concurrency, Some(2));
        assert_eq!(args.stage.openai_prefill_chunk_size, 512);
    }
}
