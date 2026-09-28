mod admission;
mod baseline;
mod checkpoint;
mod chunked_bench;
mod entry;
mod fixtures;
mod mlp_workspace;
mod model;
mod model_bench;
mod model_profile;
mod model_score;
mod mtp;
mod ninfer_inspect;
mod nvfp4_prmt;
mod observations;
mod probe;
mod stream_check;

use crate::command::DynResult;

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    match args {
        [command, rest @ ..] if command == "ninfer-inspect" => ninfer_inspect::run(rest),
        [command, rest @ ..] if command == "qwen-chunked-bench" => chunked_bench::run(rest),
        [command, rest @ ..] if command == "qwen-model-profile" => model_profile::run(rest),
        [command, rest @ ..] if command == "qwen-model-score" => model_score::run(rest),
        [command, rest @ ..] if command == "qwen-stream-check" => stream_check::run(rest),
        [command, rest @ ..] if command == "qwen-mtp-reference" => mtp::reference(rest),
        [command, rest @ ..] if command == "qwen-mtp-check" => mtp::run(rest),
        [command, rest @ ..] if command == "mlp-workspace-check" => mlp_workspace::run(rest),
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
        [command, rest @ ..] if command == "nvfp4-prmt-check" => nvfp4_prmt::synthetic(rest),
        [command, rest @ ..] if command == "nvfp4-prmt-real-check" => nvfp4_prmt::real(rest),
        [command, rest @ ..] if command == "attention-v2-check" => probe::attention_v2(rest),
        [command, rest @ ..] if command == "attention-warp-check" => probe::attention_warp(rest),
        [command, rest @ ..] if command == "attention-staged-check" => {
            probe::attention_staged(rest)
        }
        [command, rest @ ..] if command == "attention-unrolled-check" => {
            probe::attention_unrolled(rest)
        }
        [command, rest @ ..] if command == "exponential-unrolled-check" => {
            probe::exponential_unrolled(rest)
        }
        [command, rest @ ..] if command == "feature-attention-check" => {
            probe::feature_attention(rest)
        }
        [command, rest @ ..] if command == "feature-gdn-replay-check" => {
            probe::feature_gdn_replay(rest)
        }
        [command, rest @ ..] if command == "feature-graph-check" => probe::feature_graph(rest),
        [command, rest @ ..] if command == "feature-fusion-check" => probe::feature_fusion(rest),
        [command, rest @ ..] if command == "greedy-check" => probe::greedy(rest),
        [command, rest @ ..] if command == "a16-head-check" => probe::a16_head(rest),
        [command, rest @ ..] if command == "bf16-ab-decode-check" => probe::bf16_ab_decode(rest),
        [command, rest @ ..] if command == "native-parameter-check" => {
            probe::native_parameter(rest)
        }
        [command, rest @ ..] if command == "nvfp4-pipeline-check" => probe::nvfp4_pipeline(rest),
        [command, rest @ ..] if command == "feature-projection-check" => {
            probe::feature_projection(rest)
        }
        [command, rest @ ..] if command == "fp8-exact-check" => probe::fp8_exact(rest),
        [command, rest @ ..] if command == "instruction-probe" => probe::instructions(rest),
        [command, rest @ ..] if command == "workload-probe" => probe::workloads(rest),
        [command, rest @ ..] if command == "workload-check" => probe::workload_check(rest),
        _ => Err("usage: xtask specialize baseline --plan PATH --output NEW_DIRECTORY".into()),
    }
}
