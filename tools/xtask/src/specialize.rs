mod baseline;
mod fixtures;
mod observations;

use crate::command::DynResult;

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    match args {
        [command, rest @ ..] if command == "baseline" => baseline::run(rest),
        [command, rest @ ..] if command == "baseline-plan" => fixtures::run(rest),
        _ => Err("usage: xtask specialize baseline --plan PATH --output NEW_DIRECTORY".into()),
    }
}
