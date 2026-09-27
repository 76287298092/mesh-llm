mod baseline;
mod fixtures;
mod observations;
mod probe;

use crate::command::DynResult;

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    match args {
        [command, rest @ ..] if command == "baseline" => baseline::run(rest),
        [command, rest @ ..] if command == "baseline-plan" => fixtures::run(rest),
        [command, rest @ ..] if command == "nvfp4-probe" => probe::run(rest),
        _ => Err("usage: xtask specialize baseline --plan PATH --output NEW_DIRECTORY".into()),
    }
}
