//! Independent CPU selection and GPU boundary/reuse qualification.
use super::{
    driver::{Buffer, Context, Module},
    resident_greedy::Selector,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub(crate) fn run(ptx: &str, device: i32) -> Result<Value> {
    let ctx = Context::new(device)?;
    ensure!(
        (ctx.info().major, ctx.info().minor) == (12, 0),
        "greedy trial requires SM120"
    );
    let module = Module::load(&ctx, ptx)?;
    let mut cases = Vec::new();
    for n in [1, 2, 31, 32, 127, 128, 1023, 1024, 1025, 248320, 262144] {
        let mut values = vec![0xbf80; n];
        values[n - 1] = 0x8001;
        cases.push(check(&ctx, &module, "negative-last", &values)?);
        values.fill(0x8000);
        values[n - 1] = 0;
        cases.push(check(&ctx, &module, "signed-zero-first", &values)?);
        values.fill(0x0001);
        values[n - 1] = 0x0002;
        cases.push(check(&ctx, &module, "positive-subnormal-last", &values)?);
    }
    let finite = (0..=u16::MAX)
        .filter(|v| v & 0x7f80 != 0x7f80)
        .collect::<Vec<_>>();
    cases.push(check(&ctx, &module, "all-finite-codes", &finite)?);
    cases.push(check(
        &ctx,
        &module,
        "all-finite-reversed",
        &finite.iter().rev().copied().collect::<Vec<_>>(),
    )?);
    for invalid in [0x7f80, 0xff80, 0x7f81, 0xff81, 0x7fc0, 0xffff] {
        cases.push(check(&ctx, &module, "single-nonfinite", &[invalid])?);
        cases.push(check(&ctx, &module, "all-nonfinite", &vec![invalid; 1025])?);
        for position in [0, 127, 1023, 1024, 248319] {
            let mut values = vec![0x3f80; 248320];
            values[position] = invalid;
            values[248319] = invalid;
            cases.push(check(&ctx, &module, "nonfinite-position", &values)?);
        }
    }
    ensure!(
        Selector::new(&ctx, 0).is_err() && Selector::new(&ctx, 262145).is_err(),
        "greedy accepted invalid extent"
    );
    Ok(
        json!({"kind":"exact-bf16-greedy-qualification","all_passed":true,"device":ctx.info(),"cases":cases,
        "resources":{"tiles":module.function("greedy_bf16_tiles")?.resources()?,"finish":module.function("greedy_bf16_finish")?.resources()?},
        "scope":"independent CPU finite ordering, nonfinite rejection and scratch reuse; no model performance claim"}),
    )
}
fn check(ctx: &Context, module: &Module<'_>, name: &str, values: &[u16]) -> Result<Value> {
    let expected = crate::engine::sampling::greedy(values);
    let oracle = crate::greedy_bf16_reference::greedy(values);
    ensure!(
        expected.as_ref().ok().copied() == oracle.as_ref().ok().map(|v| v.token),
        "independent CPU selectors disagree"
    );
    let bad = values
        .iter()
        .position(|b| !f32::from_bits(u32::from(*b) << 16).is_finite());
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let input = Buffer::new(ctx, bytes.len())?;
    input.upload(&bytes)?;
    let mut selector = Selector::new(ctx, values.len())?;
    for _ in 0..3 {
        let actual = selector.inspect(module, &input)?;
        match &expected {
            Ok(token) => {
                ensure!(
                    actual == [*token, 0, u32::MAX, u32::from(values[*token as usize])],
                    "GPU greedy differs from CPU for {name}"
                );
                let selected = selector.select(module, &input)?;
                ensure!(
                    selected.token == *token && selected.bits == values[*token as usize],
                    "selection wrapper mismatch"
                );
            }
            Err(_) => {
                ensure!(
                    actual[1] == 1 && actual[2] == bad.expect("invalid oracle input") as u32,
                    "GPU did not reject first nonfinite for {name}"
                );
                ensure!(
                    selector.select(module, &input).is_err(),
                    "GPU wrapper accepted invalid logits"
                );
            }
        }
    }
    Ok(
        json!({"name":name,"vocabulary":values.len(),"all_passed":true,"expected_token":expected.ok(),"first_nonfinite":bad,"reuses":3}),
    )
}
