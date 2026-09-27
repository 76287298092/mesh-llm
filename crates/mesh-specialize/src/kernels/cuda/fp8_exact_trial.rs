//! Exact FP8 dot qualification against independent logical FP64 arithmetic.
use super::driver::{Buffer, Context, Module};
use crate::{
    entry_reference::round_bf16,
    projection_reference::{self, QuantizedRows},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(super) fn run(ctx: &Context, module: &Module<'_>) -> Result<Value> {
    let mut cases = Vec::new();
    for (shape, kind) in [
        ([254, 254, 1], 0),
        ([2, 9, 513], 1),
        ([5, 9, 513], 1),
        ([17, 19, 513], 1),
        ([1, 5, 32768], 2),
        ([1, 5, 5120], 3),
    ] {
        let [m, n, k] = shape;
        let finite = (0..=255_u16)
            .map(|v| v as u8)
            .filter(|v| v & 127 != 127)
            .collect::<Vec<_>>();
        let mut a = (0..m * k)
            .map(|i| finite[(i * 37 + i / 7) % 254])
            .collect::<Vec<_>>();
        let mut w = (0..n * k)
            .map(|i| finite[(i * 53 + i / 11) % 254])
            .collect::<Vec<_>>();
        if kind == 0 {
            a = finite.clone();
            w = finite;
        }
        if kind == 2 {
            a.fill(126);
            for (i, v) in w.iter_mut().enumerate() {
                *v = if (i / k) % 2 == 0 { 126 } else { 254 };
            }
        }
        if kind == 3 {
            a.fill(56);
            for row in w.chunks_exact_mut(k) {
                row.fill(0);
                row[0] = 126;
                row[1] = 1;
                row[2] = 254;
                row[k - 1] = 1;
            }
        }
        let sa = (0..m)
            .map(|i| {
                if kind == 0 {
                    1.0
                } else {
                    (i % 7 + 1) as f32 / 16.0
                }
            })
            .collect::<Vec<_>>();
        let sw = (0..n)
            .map(|i| {
                round_bf16(if kind == 0 {
                    1.0
                } else {
                    (i % 5 + 1) as f32 / 8.0
                })
            })
            .collect::<Vec<_>>();
        let expected = projection_reference::linear(
            &QuantizedRows {
                codes: a.clone(),
                scales: sa.clone(),
            },
            &w,
            &sw,
            k,
        )?;
        for tile_rows in [1, 4, 16] {
            let (actual, unrounded) = execute(ctx, module, [m, n, k, tile_rows], &a, &w, &sa, &sw)?;
            let bf16_differences = actual
                .iter()
                .zip(&expected.normalized)
                .filter(|(a, b)| a != b)
                .count();
            let fp32_differences = unrounded
                .iter()
                .zip(&expected.unrounded)
                .filter(|(a, b)| a != b)
                .count();
            let finite = unrounded.iter().all(|v| v.is_finite());
            cases.push(json!({"shape":shape,"tile_rows":tile_rows,"fixture":kind,"outputs":m*n,"bf16_differences":bf16_differences,"fp32_differences":fp32_differences,"all_passed":finite&&bf16_differences==0&&fp32_differences==0}));
        }
    }
    Ok(
        json!({"all_passed":cases.iter().all(|c|c["all_passed"]==true),"cases":cases,"resources":module.function("fp8_linear_exact")?.resources()?,"tiled_resources":module.function("fp8_linear_exact4")?.resources()?,"prefill_resources":module.function("fp8_prefill_exact")?.resources()?}),
    )
}

fn upload<'a>(ctx: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let b = Buffer::new(ctx, bytes.len())?;
    b.upload(bytes)?;
    Ok(b)
}

fn execute(
    ctx: &Context,
    module: &Module<'_>,
    shape: [usize; 4],
    a: &[u8],
    w: &[u8],
    sa: &[f32],
    sw: &[u16],
) -> Result<(Vec<u16>, Vec<f32>)> {
    let [m, n, k, tile_rows] = shape;
    ensure!(
        a.len() == m * k && w.len() == n * k && sa.len() == m && sw.len() == n,
        "fixture extents"
    );
    let a = upload(ctx, a)?;
    let w = upload(ctx, w)?;
    let sa = upload(
        ctx,
        &sa.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    )?;
    let sw = upload(
        ctx,
        &sw.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    )?;
    let out = upload(ctx, &vec![0xa5; m * n * 2])?;
    let raw = upload(ctx, &vec![0xff; m * n * 4])?;
    let mut pointers = [
        a.pointer(),
        w.pointer(),
        sa.pointer(),
        sw.pointer(),
        out.pointer(),
        raw.pointer(),
    ];
    let mut dims = [u32::try_from(m)?, u32::try_from(n)?, u32::try_from(k)?];
    let mut args = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast())
        .collect::<Vec<*mut c_void>>();
    args.extend(dims.iter_mut().map(|p| (p as *mut u32).cast()));
    // SAFETY: The checked finite fixtures have complete row-major extents and disjoint
    // typed allocations; each variant receives its required tile/block geometry.
    // All buffers live through synchronization.
    unsafe {
        module
            .function(if tile_rows == 16 {
                "fp8_prefill_exact"
            } else if tile_rows == 4 {
                "fp8_linear_exact4"
            } else {
                "fp8_linear_exact"
            })?
            .launch(
                [
                    u32::try_from(n.div_ceil(if tile_rows == 16 { 8 } else { 4 }))?,
                    u32::try_from(m.div_ceil(tile_rows))?,
                    1,
                ],
                [if tile_rows == 16 { 32 } else { 128 }, 1, 1],
                0,
                &mut args,
            )?;
    }
    ctx.synchronize()?;
    let mut bytes = vec![0; m * n * 2];
    out.download(&mut bytes)?;
    let mut floats = vec![0; m * n * 4];
    raw.download(&mut floats)?;
    Ok((
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| u16::from_le_bytes(*v))
            .collect(),
        floats
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect(),
    ))
}
