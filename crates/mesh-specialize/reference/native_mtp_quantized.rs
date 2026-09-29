//! Independent exact CPU decode of Ninfer grouped Q8/Q4 weight rows.

#[cfg(test)]
#[path = "native_mtp_quantized/error.rs"]
mod error;
#[cfg(test)]
#[path = "native_mtp_quantized/head.rs"]
mod head;
#[cfg(test)]
#[path = "native_mtp_quantized/row.rs"]
mod row;
#[cfg(test)]
#[path = "native_mtp_quantized/views.rs"]
mod views;

#[cfg(test)]
pub use head::{MtpHeadStep, q4_g64_fp16_mtp_head_step};
#[cfg(test)]
pub use views::decode_q8_view_row;
