use super::{
    driver::{Context, Module},
    resident_model::Model,
    resident_target_batch_trial,
    resident_weights::ResidentWeights,
};
use crate::kernels::TargetBatchLoadRequest;
use anyhow::{Result, ensure};

pub(in crate::kernels) fn run(request: TargetBatchLoadRequest<'_>) -> Result<serde_json::Value> {
    ensure!(
        request.ptx.contains(".target sm_120a"),
        "target batch trial requires SM120a PTX"
    );
    let context = Context::new(request.device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "target batch trial requires SM120"
    );
    let module = Module::load(&context, request.ptx)?;
    let weights = ResidentWeights::load(&context, request.artifact, request.objects)?;
    let model = Model::new(&weights, request.config)?;
    resident_target_batch_trial::run(resident_target_batch_trial::Request {
        model: &model,
        context: &context,
        module: &module,
        config: request.config,
        prefix: &request.fixture.prefix,
        target_tokens: &request.fixture.target_tokens,
        continuation: &request.fixture.continuation,
        selected_rows: request.selected_rows,
    })
}
