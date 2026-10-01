use super::super::driver::{Buffer, Context, Module};
use super::{
    compare,
    types::{Case, Weight},
};
use crate::entry_reference::round_bf16;
use anyhow::Result;
use serde_json::{Value, json};

pub(super) fn run_synthetic(context: &Context, module: &Module<'_>) -> Result<Value> {
    let mut cases = Vec::new();
    for width in [16, 80, 5120, 17408] {
        let channels = 8;
        let input: Vec<u16> = (0..width)
            .map(|index| round_bf16(((index * 29 % 97) as f32 - 48.0) / 32.0))
            .collect();
        let gate_packed = pack(channels, width, |row, column| {
            ((row * 7 + column * 3 + column / 16) % 16) as u8
        });
        let up_packed = pack(channels, width, |row, column| {
            ((row * 11 + column * 5 + column / 16 + 8) % 16) as u8
        });
        let gate_scales = scales(channels, width, 7);
        let up_scales = scales(channels, width, 13);
        let gate_codes = upload(context, &gate_packed)?;
        let gate_scale_bytes = upload(context, &gate_scales)?;
        let up_codes = upload(context, &up_packed)?;
        let up_scale_bytes = upload(context, &up_scales)?;
        let gate = Weight {
            address: [gate_codes.pointer(), gate_scale_bytes.pointer()],
            packed: &gate_packed,
            scales: &gate_scales,
            divisor: 1.25,
        };
        let up = Weight {
            address: [up_codes.pointer(), up_scale_bytes.pointer()],
            packed: &up_packed,
            scales: &up_scales,
            divisor: 2.5,
        };
        cases.push(compare::run(
            context,
            module,
            Case {
                source: "asymmetric-signed-synthetic-packed-planes",
                input: &input,
                width,
                gate,
                up,
            },
        )?);
    }
    let all_passed = cases.iter().all(|case| case["all_passed"] == true);
    Ok(json!({
        "kind": "nvfp4-a16-swiglu-synthetic-operator-check-v1",
        "kernel_resources": module.function("nvfp4_swiglu_a16")?.resources()?,
        "arithmetic_profile": "direct BF16 input, FP32 FMA accumulation, independent gate/up group scales, raw SiLU-product to BF16",
        "all_passed": all_passed,
        "cases": cases,
        "scope": "synthetic operator comparison only; no model quality or timing claim",
    }))
}

fn pack(rows: usize, width: usize, code: impl Fn(usize, usize) -> u8) -> Vec<u8> {
    (0..rows * width / 2)
        .map(|index| {
            let row = index / (width / 2);
            let first = index % (width / 2) * 2;
            code(row, first) | (code(row, first + 1) << 4)
        })
        .collect()
}

fn scales(rows: usize, width: usize, salt: usize) -> Vec<u8> {
    (0..rows * width / 16)
        .map(|index| [0x30, 0x38, 0x40, 0x42][(index * salt + index / 5) % 4])
        .collect()
}

fn upload<'ctx>(context: &'ctx Context, bytes: &[u8]) -> Result<Buffer<'ctx>> {
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}
