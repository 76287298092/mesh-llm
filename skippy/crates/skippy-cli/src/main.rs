mod cli;
mod conversion;
mod local_model;
mod runtime;
mod serve;

#[cfg(unix)]
use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use cli::{Cli, Command, OutputFormat};
use std::io::IsTerminal;

#[tokio::main]
async fn main() -> std::process::ExitCode {
    match run_main().await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            let _ = skippy_commands::console::failure(&error);
            std::process::ExitCode::FAILURE
        }
    }
}

async fn run_main() -> Result<()> {
    let Some(cli) = parse_cli()? else {
        return Ok(());
    };
    let output = match (&cli.command, cli.output) {
        (Command::Serve(_), OutputFormat::Auto) if !std::io::stdout().is_terminal() => {
            OutputFormat::Jsonl
        }
        (Command::Prompt(_), OutputFormat::Auto) => OutputFormat::Human,
        _ => cli.output,
    };
    skippy_commands::console::install(output.into());
    if matches!(&cli.command, Command::Prompt(_))
        && skippy_commands::console::mode() != skippy_commands::console::OutputMode::Human
    {
        anyhow::bail!("interactive prompt requires human output; agents should call the API");
    }
    if let Command::Serve(args) = &cli.command {
        serve::validate(args)?;
    }
    #[cfg(feature = "dynamic-native-runtime")]
    let automatic_runtime = cli.native_runtime.bundle_dirs.is_empty()
        && cli.native_runtime.release.is_none()
        && cli.native_runtime.selection.is_none();
    let include_adjacent = !matches!(
        &cli.command,
        Command::Runtime {
            command: cli::RuntimeCommand::Install { .. }
        }
    );
    let native_options = runtime::resolve_options(cli.native_runtime, include_adjacent)?;
    #[cfg(feature = "dynamic-native-runtime")]
    if matches!(&cli.command, Command::Serve(_) | Command::PlanSplit(_)) {
        runtime::prepare_native_runtime(&native_options, automatic_runtime).await?;
    }
    match cli.command {
        Command::Doctor => runtime::doctor(&native_options),
        Command::Prompt(args) => {
            tokio::task::spawn_blocking(move || {
                skippy_commands::prompt::run(skippy_commands::prompt::PromptCommand {
                    endpoint: args.endpoint,
                    model: args.model,
                    max_new_tokens: args.max_new_tokens,
                    raw: args.raw,
                    no_think: args.no_think,
                    history_path: args.history_path,
                })
            })
            .await?
        }
        Command::Serve(args) => serve::run(*args).await,
        Command::Models { cache_dir, command } => {
            skippy_commands::models::run(cache_dir, command.into()).await
        }
        Command::PlanSplit(args) => skippy_commands::split::run(args.into()),
        Command::Runtime { command } => {
            skippy_commands::runtime::run(
                command.into(),
                &runtime::command_options(&native_options),
            )
            .await
        }
        Command::ExampleConfig => {
            skippy_commands::console::write_json(&skippy_config::example_config())
        }
    }
}

fn parse_cli() -> Result<Option<Cli>> {
    match Cli::try_parse() {
        Ok(cli) => Ok(Some(cli)),
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            error.print()?;
            Ok(None)
        }
        Err(error) => {
            if requested_jsonl_output() {
                skippy_commands::console::install(skippy_commands::console::OutputMode::Jsonl);
            }
            anyhow::bail!(error.to_string())
        }
    }
}

fn requested_jsonl_output() -> bool {
    let args = std::env::args_os().collect::<Vec<_>>();
    args.windows(2)
        .any(|pair| pair[0] == "--output" && pair[1] == "jsonl")
        || args.iter().any(|arg| arg == "--output=jsonl")
}

fn shutdown_signal() -> Result<impl std::future::Future<Output = ()> + Send + 'static> {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .context("install SIGTERM handler")?;
        let mut interrupt =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
                .context("install SIGINT handler")?;
        Ok(async move {
            tokio::select! {
                _ = interrupt.recv() => {}
                _ = terminate.recv() => {}
            }
        })
    }
    #[cfg(not(unix))]
    {
        Ok(async {
            if let Err(error) = tokio::signal::ctrl_c().await {
                let _ = skippy_events::diagnostics::emit(
                    skippy_events::diagnostics::ServingDiagnostic::Warning {
                        message: format!("interrupt handler failed: {error}"),
                        context: None,
                    },
                );
            }
        })
    }
}
