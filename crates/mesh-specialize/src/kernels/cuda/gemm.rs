use super::{
    driver::{Buffer, Context, Event, Function, Module},
    upload_words,
};
use crate::kernels::{gemm_fixtures::Fixture, nvfp4_layout};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(super) fn cases(context: &Context, module: &Module<'_>) -> Result<Vec<Value>> {
    let function = module.function("nvfp4_gemm_packed")?;
    crate::kernels::gemm_fixtures::fixtures()
        .map_err(anyhow::Error::msg)?
        .iter()
        .map(|case| run_case(context, &function, case))
        .collect()
}

fn run_case(context: &Context, function: &Function<'_, '_>, case: &Fixture) -> Result<Value> {
    let a = upload_words(context, &case.a)?;
    let b = upload_words(context, &case.b)?;
    let sa = upload_words(context, &case.scale_a)?;
    let sb = upload_words(context, &case.scale_b)?;
    let count = case.m_tiles as usize * case.n_tiles as usize * 128;
    let output = upload_words(context, &vec![f32::NAN.to_bits(); count])?;
    let allocated = context.memory()?;
    let mut pointers = [
        a.pointer(),
        b.pointer(),
        sa.pointer(),
        sb.pointer(),
        output.pointer(),
    ];
    let mut sizes = [case.m_tiles, case.n_tiles, case.k_tiles];
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast())
        .collect();
    args.extend(sizes.iter_mut().map(|s| (s as *mut u32).cast()));
    launch(function, case, &mut args)?;
    context.synchronize()?;
    let actual = unpack(&output, case)?;
    let comparison = compare(&actual, case)?;
    let timing = timings(context, function, case, &mut args)?;
    Ok(json!({"kernel":"nvfp4_gemm_packed", "fixture":case.name,
        "m":case.m,"n":case.n,"k":case.k,"grid":[case.n_tiles,case.m_tiles,1],
        "padded_dimensions":[case.m_tiles*16,case.n_tiles*8,case.k_tiles*64],
        "passed":comparison["mismatches"]==0,"comparison":comparison,
        "resources":function.resources()?, "timing":timing,
        "allocation_payload_bytes":4*(case.a.len()+case.b.len()+case.scale_a.len()+case.scale_b.len()+count),
        "free_bytes_with_allocations":allocated.0,
        "oracle":"independent decoded f64 logical products; repeating row/column classes; all outputs checked"}))
}

fn unpack(output: &Buffer<'_>, case: &Fixture) -> Result<Vec<f32>> {
    let rows = case.m_tiles as usize * 16;
    let columns = case.n_tiles as usize * 8;
    let mut bytes = vec![0; rows * columns * 4];
    output.download(&mut bytes)?;
    let packed: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect();
    let mut actual = vec![f32::NAN; rows * columns];
    for (index, tile) in packed.chunks_exact(128).enumerate() {
        let logical = nvfp4_layout::unpack_output(tile).map_err(anyhow::Error::msg)?;
        let tile_m = index / case.n_tiles as usize;
        let tile_n = index % case.n_tiles as usize;
        for row in 0..16 {
            let start = (tile_m * 16 + row) * columns + tile_n * 8;
            actual[start..start + 8].copy_from_slice(&logical[row * 8..row * 8 + 8]);
        }
    }
    Ok(actual)
}

fn compare(actual: &[f32], case: &Fixture) -> Result<Value> {
    let columns = case.n_tiles as usize * 8;
    ensure!(
        case.expected.len() == case.m * case.n,
        "GEMM oracle length mismatch"
    );
    let mut mismatches = 0;
    let mut max_abs_error = 0.0_f32;
    let mut first_mismatches = Vec::new();
    for (index, &a) in actual.iter().enumerate() {
        let row = index / columns;
        let column = index % columns;
        let expected = if row < case.m && column < case.n {
            case.expected[row * case.n + column]
        } else {
            0.0
        };
        let error = (a - expected).abs();
        max_abs_error = max_abs_error.max(error);
        // E2M1 values and these power-of-two scales give exactly representable
        // sums for the bounded K values. This checks padded rows/columns too.
        if !a.is_finite() || error != 0.0 {
            mismatches += 1;
            if first_mismatches.len() < 8 {
                first_mismatches
                    .push(json!({"row":row,"column":column,"actual":a,"expected":expected}));
            }
        }
    }
    Ok(
        json!({"mismatches":mismatches,"max_abs_error":max_abs_error,
        "elements_including_padding":actual.len(),"exact_required":true,
        "first_mismatches":first_mismatches,"first_actual":&actual[..actual.len().min(8)],
        "first_expected":&case.expected[..case.expected.len().min(8)]}),
    )
}

fn timings(
    context: &Context,
    function: &Function<'_, '_>,
    case: &Fixture,
    args: &mut [*mut c_void],
) -> Result<Value> {
    for _ in 0..10 {
        launch(function, case, args)?;
    }
    context.synchronize()?;
    let start = Event::new(context)?;
    let end = Event::new(context)?;
    let mut batch_ms = Vec::new();
    for _ in 0..5 {
        start.record()?;
        for _ in 0..100 {
            launch(function, case, args)?;
        }
        end.record()?;
        end.synchronize()?;
        batch_ms.push(end.elapsed_since(&start)?);
    }
    let per_launch: Vec<_> = batch_ms.iter().map(|ms| ms * 10.0).collect();
    Ok(
        json!({"warmup_launches":10,"launches_per_batch":100,"batch_event_ms":batch_ms,
        "event_us_per_launch":per_launch,
        "note":"resident packed inputs; default-stream event batches include host submission gaps; not model throughput"}),
    )
}

fn launch(function: &Function<'_, '_>, case: &Fixture, args: &mut [*mut c_void]) -> Result<()> {
    // SAFETY: run_case owns exact packed buffers and the eight typed arguments
    // through synchronization. Each block is one full warp for one output tile.
    unsafe { function.launch([case.n_tiles, case.m_tiles, 1], [32, 1, 1], 0, args) }
}
