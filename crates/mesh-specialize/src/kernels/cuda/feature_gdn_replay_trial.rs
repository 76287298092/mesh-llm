//! Exact synthetic qualification of compact GDN records and every accepted prefix.
use super::driver::{Buffer, Context, Module};
use crate::{
    entry_reference::round_bf16,
    gdn_recurrent_reference::{self as recurrence, Input, Reduction, Shape},
    gdn_replay_reference,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(crate) fn run(ptx: &str, device: i32) -> Result<Value> {
    let ctx = Context::new(device)?;
    let info = ctx.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "GDN replay probe requires SM120"
    );
    let module = Module::load(&ctx, ptx)?;
    let cases = [1, 2, 128]
        .into_iter()
        .map(|width| check_case(&ctx, &module, width))
        .collect::<Result<Vec<_>>>()?;
    Ok(
        json!({"all_passed":true,"kind":"gdn-compact-replay-probe","device":info,"cases":cases,
        "record_resources":module.function("gdn_recurrent_record")?.resources()?,
        "replay_resources":module.function("gdn_replay_state")?.resources()?,
        "scope":"synthetic exact recurrence evidence only; full MTP recovery remains unqualified"}),
    )
}

struct Fixture {
    shape: Shape,
    q: Vec<f32>,
    k: Vec<f32>,
    qkv: Vec<u16>,
    beta: Vec<u16>,
    decay: Vec<f32>,
    state: Vec<f32>,
}
impl Fixture {
    fn new(width: usize) -> Self {
        let shape = Shape {
            rows: 5,
            key_heads: 2,
            value_heads: 4,
            width,
        };
        let values = |count, factor: usize| {
            (0..count)
                .map(|i| ((i * factor % 37) as f32 - 18.0) / 137.0)
                .collect::<Vec<_>>()
        };
        Self {
            q: values(5 * 2 * width, 7),
            k: values(5 * 2 * width, 11),
            qkv: values(5 * 8 * width, 13)
                .into_iter()
                .map(round_bf16)
                .collect(),
            beta: (0..20)
                .map(|i| round_bf16([0.0, 0.13, 0.83, 1.0][i % 4]))
                .collect(),
            decay: (0..20)
                .map(|i| [0.0, 0.1234567, 0.9876543, 1.0][i % 4])
                .collect(),
            state: values(4 * width * width, 17),
            shape,
        }
    }
    fn input(&self, rows: usize) -> Input<'_> {
        let w = self.shape.width;
        Input {
            q: &self.q[..rows * 2 * w],
            k: &self.k[..rows * 2 * w],
            qkv: &self.qkv[..rows * 8 * w],
            beta: &self.beta[..rows * 4],
            decay: &self.decay[..rows * 4],
        }
    }
}
struct DeviceCase<'a> {
    q: Buffer<'a>,
    k: Buffer<'a>,
    qkv: Buffer<'a>,
    beta: Buffer<'a>,
    decay: Buffer<'a>,
    state: Buffer<'a>,
    out: Buffer<'a>,
    raw: Buffer<'a>,
    delta: Buffer<'a>,
}
impl<'a> DeviceCase<'a> {
    fn new(ctx: &'a Context, fixture: &Fixture) -> Result<Self> {
        let count = fixture.shape.rows * fixture.shape.value_heads * fixture.shape.width;
        Ok(Self {
            q: upload(ctx, &floats(&fixture.q))?,
            k: upload(ctx, &floats(&fixture.k))?,
            qkv: upload(ctx, &words(&fixture.qkv))?,
            beta: upload(ctx, &words(&fixture.beta))?,
            decay: upload(ctx, &floats(&fixture.decay))?,
            state: upload(ctx, &floats(&fixture.state))?,
            out: upload(ctx, &vec![0xa5; count * 2])?,
            raw: upload(ctx, &vec![0xa5; count * 4])?,
            delta: upload(ctx, &vec![0xa5; count * 4])?,
        })
    }
}
fn check_case(ctx: &Context, module: &Module<'_>, width: usize) -> Result<Value> {
    let fixture = Fixture::new(width);
    let expected = gdn_replay_reference::record(&fixture.input(5), &fixture.state, &fixture.shape)?;
    let device = DeviceCase::new(ctx, &fixture)?;
    record(ctx, module, &device, &fixture.shape)?;
    equal(&device.out, &words(&expected.output), "record output")?;
    equal(
        &device.raw,
        &floats(&expected.unrounded),
        "record raw output",
    )?;
    equal(&device.delta, &floats(&expected.delta), "record deltas")?;
    equal(&device.state, &floats(&expected.state), "record state")?;
    for rows in 0..=5 {
        let expected = if rows == 0 {
            fixture.state.clone()
        } else {
            let shape = Shape {
                rows,
                ..fixture.shape.clone()
            };
            recurrence::run(
                &fixture.input(rows),
                &fixture.state,
                &shape,
                Reduction::OrderedF32,
            )?
            .state
        };
        device.state.upload(&floats(&fixture.state))?;
        if rows > 0 {
            replay(
                ctx,
                module,
                &device,
                &Shape {
                    rows,
                    ..fixture.shape.clone()
                },
            )?;
        }
        equal(&device.state, &floats(&expected), "accepted-prefix state")?;
    }
    Ok(
        json!({"width":width,"key_heads":2,"value_heads":4,"record_rows":5,"prefixes_checked":6,"state_values_per_prefix":fixture.state.len(),"all_passed":true}),
    )
}
fn record(ctx: &Context, module: &Module<'_>, d: &DeviceCase<'_>, shape: &Shape) -> Result<()> {
    let mut pointers = [
        d.q.pointer(),
        d.k.pointer(),
        d.qkv.pointer(),
        d.beta.pointer(),
        d.decay.pointer(),
        d.state.pointer(),
        d.out.pointer(),
        d.raw.pointer(),
    ];
    let mut dims = [
        u32::try_from(shape.rows)?,
        u32::try_from(shape.key_heads)?,
        u32::try_from(shape.value_heads)?,
        u32::try_from(shape.width)?,
    ];
    let mut delta = d.delta.pointer();
    let mut args = pointers
        .iter_mut()
        .map(|v| (v as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(dims.iter_mut().map(|v| (v as *mut u32).cast()));
    args.push((&mut delta as *mut u64).cast());
    // SAFETY: The fixed fixture and independent reference validate all extents; each owned
    // allocation implements the documented ABI and lives through context synchronization.
    unsafe {
        module.function("gdn_recurrent_record")?.launch(
            [dims[2], 1, 1],
            [dims[3], 1, 1],
            0,
            &mut args,
        )?;
    }
    ctx.synchronize()
}
fn replay(ctx: &Context, module: &Module<'_>, d: &DeviceCase<'_>, shape: &Shape) -> Result<()> {
    let mut pointers = [
        d.k.pointer(),
        d.decay.pointer(),
        d.delta.pointer(),
        d.state.pointer(),
    ];
    let mut dims = [
        u32::try_from(shape.rows)?,
        u32::try_from(shape.key_heads)?,
        u32::try_from(shape.value_heads)?,
        u32::try_from(shape.width)?,
    ];
    let mut args = pointers
        .iter_mut()
        .map(|v| (v as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.extend(dims.iter_mut().map(|v| (v as *mut u32).cast()));
    // SAFETY: Prefix rows never exceed the recorded fixture rows, state was restored to its
    // exact original bytes, and all device buffers remain live through synchronization.
    unsafe {
        module.function("gdn_replay_state")?.launch(
            [dims[2], 1, 1],
            [dims[3], 1, 1],
            0,
            &mut args,
        )?;
    }
    ctx.synchronize()
}
fn upload<'a>(ctx: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let b = Buffer::new(ctx, bytes.len())?;
    b.upload(bytes)?;
    Ok(b)
}
fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn words(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn equal(buffer: &Buffer<'_>, expected: &[u8], label: &str) -> Result<()> {
    let mut actual = vec![0; buffer.len()];
    buffer.download(&mut actual)?;
    ensure!(
        actual == expected,
        "GDN replay {label} differs from independent reference"
    );
    Ok(())
}
