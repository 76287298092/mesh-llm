mod admission;
mod baseline;
mod checkpoint;
mod entry;
mod fixtures;
mod model;
mod model_bench;
mod model_profile;
mod mtp;
mod observations;
mod probe;

use crate::command::DynResult;

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    match args {
        [command, rest @ ..] if command == "qwen-model-profile" => model_profile::run(rest),
        [command, rest @ ..] if command == "qwen-mtp-reference" => mtp::reference(rest),
        [command, rest @ ..] if command == "qwen-mtp-check" => mtp::run(rest),
        [command, rest @ ..] if command == "qwen-model-bench" => model_bench::run(rest),
        [command, rest @ ..] if command == "qwen-model-trace" => model::trace(rest),
        [command, rest @ ..] if command == "qwen-model-reference" => model::reference(rest),
        [command, rest @ ..] if command == "qwen-model-check" => model::check(rest),
        [command, rest @ ..] if command == "qwen-resident-gdn-check" => entry::resident_gdn(rest),
        [command, rest @ ..] if command == "qwen-resident-attention-check" => {
            entry::resident_attention(rest)
        }
        [command, rest @ ..] if command == "qwen-fp8-mlp-check" => entry::fp8_mlp(rest),
        [command, rest @ ..] if command == "qwen-residency-check" => entry::residency(rest),
        [command, rest @ ..] if command == "qwen-attention-check" => entry::attention(rest),
        [command, rest @ ..] if command == "qwen-projection-check" => entry::projections(rest),
        [command, rest @ ..] if command == "qwen-entry-check" => entry::run(rest),
        [command, rest @ ..] if command == "checkpoint-import" => checkpoint::run(rest),
        [command, rest @ ..] if command == "admission-probe" => admission::run(rest),
        [command, rest @ ..] if command == "baseline" => baseline::run(rest),
        [command, rest @ ..] if command == "baseline-plan" => fixtures::run(rest),
        [command, rest @ ..] if command == "nvfp4-probe" => probe::run(rest),
        [command, rest @ ..] if command == "feature-projection-check" => {
            probe::feature_projection(rest)
        }
        [command, rest @ ..] if command == "instruction-probe" => probe::instructions(rest),
        [command, rest @ ..] if command == "workload-probe" => probe::workloads(rest),
        [command, rest @ ..] if command == "workload-check" => probe::workload_check(rest),
        _ => Err("usage: xtask specialize baseline --plan PATH --output NEW_DIRECTORY".into()),
    }
}
