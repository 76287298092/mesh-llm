use super::DecoderConfig;
use crate::{
    artifact::{model_source::ModelArtifact, schema::Object},
    packages::qwen3_8_27b::target_batch_trial::Fixture,
};
use anyhow::Result;
use serde_json::Value;

pub struct NativeMtpForwardLoadRequest<'a> {
    pub artifact: &'a mut ModelArtifact,
    pub objects: &'a [Object],
    pub config: &'a DecoderConfig,
    pub ptx: &'a str,
    pub device: i32,
    pub fixture: &'a Fixture,
}

pub fn native_mtp_forward_check(request: NativeMtpForwardLoadRequest<'_>) -> Result<Value> {
    #[cfg(target_os = "linux")]
    {
        use super::cuda::resident_native_mtp_forward_entry::{self, Request};
        resident_native_mtp_forward_entry::run(Request {
            artifact: request.artifact,
            objects: request.objects,
            config: request.config,
            ptx: request.ptx,
            device: request.device,
            fixture: request.fixture,
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = request;
        anyhow::bail!("Native MTP forward trial requires Linux")
    }
}

pub fn native_mtp_activation_check(ptx: &str, device: i32) -> Result<Value> {
    #[cfg(target_os = "linux")]
    {
        super::cuda::native_mtp_activation_entry::run(ptx, device)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptx, device);
        anyhow::bail!("Native MTP activation trial requires Linux")
    }
}
