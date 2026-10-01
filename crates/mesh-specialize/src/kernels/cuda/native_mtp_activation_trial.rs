//! Experimental exact mathematical gate; never source-arithmetic admission.

mod compare;
#[path = "../../../reference/native_mtp_activation.rs"]
mod oracle;

use super::driver::{Buffer, Context, Module};
use anyhow::{Context as _, Result, ensure};
use oracle::{Input, Operation};
use serde_json::{Value, json};
use std::ffi::c_void;

const POISONS: [u16; 2] = [0x7fc1, 0xffc2];

pub(super) fn run(context: &Context, module: &Module<'_>) -> Result<Value> {
    ensure!(
        module.belongs_to(context),
        "activation trial module/context mismatch"
    );
    let inputs = oracle::inputs();
    let mut cases = Vec::with_capacity(2);
    for operation in [Operation::AttentionGate, Operation::SiluMul] {
        let result = qualify(context, module, (operation, &inputs));
        cases.push(match result {
            Ok(report) => report,
            Err(error) => json!({
                "entry":operation.entry(), "completed":false, "all_passed":false,
                "error":format!("{error:#}"),
            }),
        });
    }
    Ok(json!({
        "all_passed":cases.iter().all(|case| case["all_passed"] == true),
        "gate":"experimental_single_product_rounding_mathematical_bf16_exact",
        "oracle":"stable FP64 sigmoid/SiLU, FP64 product, direct BF16 RNE",
        "gpu_exp_implementation":"ex2.approx.ftz.f32; not Ninfer expf verified",
        "source_arithmetic_qualified":false, "native_mtp_admitted":false,
        "model_executable":false, "timing_claim":false,
        "output_poisons":POISONS, "repeats":2, "tolerance":0,
        "cases":cases,
    }))
}

fn qualify(context: &Context, module: &Module<'_>, case: (Operation, &[Input])) -> Result<Value> {
    let (operation, inputs) = case;
    let expected: Vec<_> = inputs
        .iter()
        .copied()
        .map(|input| operation.expected(input))
        .collect();
    ensure!(
        expected.iter().copied().all(oracle::finite),
        "mathematical corpus overflows BF16"
    );
    let gates: Vec<_> = inputs.iter().map(|input| input.gate).collect();
    let factors: Vec<_> = inputs.iter().map(|input| input.factor).collect();
    let gate = upload(context, &gates)?;
    let factor = upload(context, &factors)?;
    let output = Buffer::new(context, inputs.len() * 2)?;
    let function = module.function(operation.entry())?;
    let mut repeats = Vec::with_capacity(2);
    let mut outputs = Vec::with_capacity(2);
    for (repeat, poison) in POISONS.into_iter().enumerate() {
        if let Err(error) = output.upload(&bytes(&vec![poison; inputs.len()])) {
            repeats.push(json!({"repeat":repeat,"poison":poison,"completed":false,
                "all_passed":false,"error":format!("{error:#}")}));
            return Ok(json!({"entry":operation.entry(),"all_passed":false,
                "completed":false,"repeats":repeats}));
        }
        let mut left = match operation {
            Operation::AttentionGate => factor.pointer(),
            Operation::SiluMul => gate.pointer(),
        };
        let mut right = match operation {
            Operation::AttentionGate => gate.pointer(),
            Operation::SiluMul => factor.pointer(),
        };
        let mut destination = output.pointer();
        let mut count =
            u32::try_from(inputs.len()).context("activation trial count exceeds u32")?;
        let mut arguments = [
            std::ptr::from_mut(&mut left).cast::<c_void>(),
            std::ptr::from_mut(&mut right).cast::<c_void>(),
            std::ptr::from_mut(&mut destination).cast::<c_void>(),
            std::ptr::from_mut(&mut count).cast::<c_void>(),
        ];
        // SAFETY: Exact three-u64/one-u32 ABI, disjoint initialized BF16 arrays
        // covering count, same context, 256-thread CTA. All allocations remain
        // live through the unconditional drain, including a failed launch.
        let launch =
            unsafe { function.launch([count.div_ceil(256), 1, 1], [256, 1, 1], 0, &mut arguments) };
        let drain = context.synchronize();
        let completion = match launch {
            Ok(()) => drain.context("activation trial synchronized drain failed"),
            Err(error) => Err(error.context(format!("activation launch failed; drain: {drain:?}"))),
        };
        if let Err(error) = completion {
            repeats.push(json!({"repeat":repeat,"poison":poison,"completed":false,
                "all_passed":false,"error":format!("{error:#}")}));
            return Ok(json!({"entry":operation.entry(),"all_passed":false,
                "completed":false,"repeats":repeats}));
        }
        let mut raw = vec![0_u8; output.len()];
        if let Err(error) = output.download(&mut raw) {
            repeats.push(json!({"repeat":repeat,"poison":poison,"completed":false,
                "all_passed":false,"error":format!("{error:#}")}));
            return Ok(json!({"entry":operation.entry(),"all_passed":false,
                "completed":false,"repeats":repeats}));
        }
        let actual = compare::words(&raw).context("invalid BF16 output bytes")?;
        let check = compare::compare(&expected, &actual);
        repeats.push(json!({"repeat":repeat,"poison":poison,"completed":true,
            "comparison":comparison(&check, inputs),"all_passed":check.passed()}));
        outputs.push(actual);
    }
    let repeat_check = compare::compare(&outputs[0], &outputs[1]);
    Ok(json!({
        "entry":operation.entry(), "completed":true, "elements":inputs.len(),
        "all_passed":repeats.iter().all(|repeat| repeat["all_passed"] == true) && repeat_check.passed(),
        "repeats":repeats, "repeat_mismatches":repeat_check.exact_mismatches,
        "repeat_comparison":comparison(&repeat_check, inputs),
    }))
}

fn comparison(check: &compare::Comparison, inputs: &[Input]) -> Value {
    let failures: Vec<_> = check
        .failures
        .iter()
        .map(|failure| {
            let input = inputs.get(failure.index);
            json!({"index":failure.index, "expected_word":failure.expected,
            "actual_word":failure.actual, "gate_word":input.map(|value| value.gate),
            "factor_word":input.map(|value| value.factor)})
        })
        .collect();
    json!({"all_passed":check.passed(),"expected_elements":check.expected_elements,
        "actual_elements":check.actual_elements,"compared_elements":check.compared_elements,
        "finite":check.nonfinite == 0,"finite_outputs":check.finite,
        "nonfinite_outputs":check.nonfinite,"exact_mismatches":check.exact_mismatches,
        "first_16_failures":failures})
}

fn bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn upload<'ctx>(context: &'ctx Context, words: &[u16]) -> Result<Buffer<'ctx>> {
    let raw = bytes(words);
    let buffer = Buffer::new(context, raw.len())?;
    buffer.upload(&raw)?;
    Ok(buffer)
}
