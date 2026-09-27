use crate::{
    cublas::Blas,
    driver::{Buffer, Context},
    gemm_fixtures, rms_norm_fixtures,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

pub(super) fn run(args: &[String]) -> Result<()> {
    let [
        ptx_key,
        ptx,
        library_key,
        library,
        device_key,
        device,
        output_key,
        output,
    ] = args
    else {
        anyhow::bail!(
            "usage: mesh-specialize-validate --ptx PATH --library PATH --device ORDINAL --output NEW_FILE"
        );
    };
    ensure!(
        ptx_key == "--ptx"
            && library_key == "--library"
            && device_key == "--device"
            && output_key == "--output",
        "unexpected argument order"
    );
    let ptx = fs::read_to_string(ptx)?;
    let device = device.parse()?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let report = match compare_library(&ptx, Path::new(library), device) {
        Ok(report) => report,
        Err(error) => json!({"all_passed":false,"error":format!("{error:#}")}),
    };
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    ensure!(
        report["all_passed"] == true,
        "library validation failed; inspect saved report"
    );
    Ok(())
}

fn compare_library(ptx: &str, library: &Path, device: i32) -> Result<Value> {
    // First run the Rust kernels in their own context and destroy it. The cuBLAS
    // comparison then uses a fresh context and independent GPU operations.
    let rust = mesh_specialize::kernels::workload_check(ptx, device)?;
    let context = Context::new(device)?;
    let blas = Blas::new(&context, library)?;
    let before = context.memory()?;
    let mut cases = rms_cases(&context, &blas)?;
    cases.extend(gemm_cases(&context, &blas)?);
    context.synchronize()?;
    let after = context.memory()?;
    Ok(
        json!({"schema_version":1,"kind":"independent-cuda-library-correctness",
        "all_passed":rust["all_passed"]==true && cases.iter().all(|c|c["passed"]==true),
        "device":context.info(),"library_path":library,"library_version":blas.version()?,
        "math_mode":"CUBLAS_PEDANTIC_MATH","rust":rust,"library_cases":cases,
        "free_bytes_before_cases":before.0,"free_bytes_after_cases":after.0,
        "comparison":"Both independent GPU implementations are checked against the same logical f64 oracle. GEMM is exact; RMSNorm uses absolute-plus-relative tolerance on each implementation.",
        "performance_comparison":false}),
    )
}

fn gemm_cases(context: &Context, blas: &Blas<'_>) -> Result<Vec<Value>> {
    let mut results = Vec::new();
    for case in gemm_fixtures::fixtures().map_err(anyhow::Error::msg)? {
        let (dense_a, dense_b) =
            gemm_fixtures::dense_inputs(case.m, case.n, case.k).map_err(anyhow::Error::msg)?;
        let a = upload(context, &dense_a)?;
        let b = upload(context, &dense_b)?;
        let out = upload(context, &vec![f32::NAN; case.m * case.n])?;
        // SAFETY: buffers hold the entire row-major operands and disjoint output.
        // All dimensions fit i32 and allocations remain alive until synchronization.
        unsafe {
            blas.gemm(
                case.m.try_into()?,
                case.n.try_into()?,
                case.k.try_into()?,
                a.pointer(),
                b.pointer(),
                out.pointer(),
            )?;
        }
        context.synchronize()?;
        let actual = download(&out, case.m * case.n)?;
        let comparison = compare(&actual, &case.expected, 0.0, 0.0)?;
        results.push(json!({"fixture":case.name,"operation":"cublasSgemm_v2",
            "m":case.m,"n":case.n,"k":case.k,"passed":comparison["mismatches"]==0,
            "comparison":comparison,"dense_operand_bytes":4*(dense_a.len()+dense_b.len()),
            "rust_packed_operand_words":case.a.len()+case.b.len()+case.scale_a.len()+case.scale_b.len(),
            "rust_tile_counts":[case.m_tiles,case.n_tiles,case.k_tiles]}));
    }
    Ok(results)
}

fn rms_cases(context: &Context, blas: &Blas<'_>) -> Result<Vec<Value>> {
    let mut results = Vec::new();
    for case in rms_norm_fixtures::fixtures().map_err(anyhow::Error::msg)? {
        let input = upload(context, &case.input)?;
        let weight = upload(context, &case.weight)?;
        let out = upload(context, &vec![f32::NAN; case.expected.len()])?;
        let width: i32 = case.width.try_into()?;
        // SAFETY: input/output hold rows*width floats; weights hold width floats.
        // The three allocations are disjoint and remain live through synchronization.
        unsafe {
            blas.weight(
                case.rows.try_into()?,
                width,
                input.pointer(),
                weight.pointer(),
                out.pointer(),
            )?;
        }
        for row in 0..case.rows {
            let offset = u64::try_from(row * case.width * 4)?;
            // SAFETY: the row contains width floats within the live input allocation.
            let norm = unsafe { blas.norm(width, input.pointer() + offset)? };
            ensure!(norm.is_finite(), "nonfinite cuBLAS norm");
            let factor = (1.0
                / (f64::from(norm).powi(2) / case.width as f64 + f64::from(case.epsilon)).sqrt())
                as f32;
            // SAFETY: each row has width writable floats; prior weight multiplication
            // and this scale operation use the same default stream in this context.
            unsafe {
                blas.scale(width, factor, out.pointer() + offset)?;
            }
        }
        context.synchronize()?;
        let actual = download(&out, case.expected.len())?;
        let comparison = compare(&actual, &case.expected, 2e-6, 2e-6)?;
        results.push(
            json!({"fixture":case.name,"operation":"cublasSnrm2+Sdgmm+Sscal",
            "rows":case.rows,"width":case.width,"epsilon":case.epsilon,
            "passed":comparison["mismatches"]==0,"comparison":comparison,
            "normalization_scalar":"host f64 reciprocal sqrt from cuBLAS f32 norm"}),
        );
    }
    Ok(results)
}

fn upload<'a>(context: &'a Context, values: &[f32]) -> Result<Buffer<'a>> {
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(&bytes)?;
    Ok(buffer)
}

fn download(buffer: &Buffer<'_>, count: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0; count * 4];
    buffer.download(&mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect())
}

fn compare(actual: &[f32], expected: &[f32], absolute: f32, relative: f32) -> Result<Value> {
    ensure!(
        actual.len() == expected.len(),
        "oracle output shape mismatch"
    );
    let mut mismatches = 0;
    let mut max_abs_error = 0.0_f32;
    for (&a, &e) in actual.iter().zip(expected) {
        ensure!(a.is_finite() && e.is_finite(), "nonfinite reference output");
        let error = (a - e).abs();
        max_abs_error = max_abs_error.max(error);
        mismatches += usize::from(error > absolute + relative * e.abs());
    }
    Ok(
        json!({"elements":actual.len(),"mismatches":mismatches,"max_abs_error":max_abs_error,
        "absolute_tolerance":absolute,"relative_tolerance":relative}),
    )
}
