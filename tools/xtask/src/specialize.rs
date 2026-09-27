mod admission;
mod baseline;
mod checkpoint;
mod entry;
mod fixtures;
mod observations;
mod probe;

use crate::command::DynResult;

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    match args {
        [command, rest @ ..] if command == "qwen-projection-check" => entry::projections(rest),
        [command, rest @ ..] if command == "qwen-entry-check" => entry::run(rest),
        [command, rest @ ..] if command == "checkpoint-import" => checkpoint::run(rest),
        [command, rest @ ..] if command == "admission-probe" => admission::run(rest),
        [command, rest @ ..] if command == "baseline" => baseline::run(rest),
        [command, rest @ ..] if command == "baseline-plan" => fixtures::run(rest),
        [command, rest @ ..] if command == "nvfp4-probe" => probe::run(rest),
        [command, rest @ ..] if command == "instruction-probe" => probe::instructions(rest),
        [command, rest @ ..] if command == "workload-probe" => probe::workloads(rest),
        [command, rest @ ..] if command == "workload-check" => probe::workload_check(rest),
        _ => Err("usage: xtask specialize baseline --plan PATH --output NEW_DIRECTORY".into()),
    }
}
