use crate::{artifact::schema::DType, kernels::cuda::resident_weights::ResidentWeights};
use anyhow::{Context as _, Result, ensure};

pub(in crate::kernels::cuda) fn bind_table(
    owner: &ResidentWeights<'_>,
    name: &str,
    vocabulary: usize,
    width: usize,
) -> Result<(u64, Option<u64>)> {
    let (bf16_bytes, _) = validate_shape(vocabulary, width, 1e-6)?;
    let dtype = &owner.object(name)?.dtype;
    let encoded = encoded_table(dtype, vocabulary, width)?;
    let shape = [u64::try_from(vocabulary)?, u64::try_from(width)?];
    let table = owner.tensor(
        name,
        dtype.clone(),
        &shape,
        if encoded { bf16_bytes / 2 } else { bf16_bytes },
    )?;
    let scale = if encoded {
        let prefix = name
            .strip_suffix(".weight")
            .context("embedding name must end in .weight")?;
        Some(owner.tensor(
            &format!("{prefix}.weight_scale"),
            DType::Bf16,
            &[shape[0], 1],
            shape[0] * 2,
        )?)
    } else {
        None
    };
    Ok((table, scale))
}

pub(super) fn encoded_table(dtype: &DType, vocabulary: usize, width: usize) -> Result<bool> {
    match dtype {
        DType::Bf16 => Ok(false),
        DType::Fp8E4m3 => {
            ensure!(
                [vocabulary, width] == [248_320, 5120],
                "encoded embedding requires exact [248320, 5120] shape"
            );
            Ok(true)
        }
        _ => anyhow::bail!("unsupported embedding dtype: {}", dtype.as_str()),
    }
}

pub(super) fn validate_shape(vocabulary: usize, width: usize, epsilon: f32) -> Result<(u64, u64)> {
    ensure!(
        (1..=1_048_576).contains(&vocabulary),
        "embedding vocabulary is out of range"
    );
    ensure!(
        (1..=32768).contains(&width),
        "embedding width is out of range"
    );
    ensure!(
        epsilon.is_finite() && epsilon > 0.0,
        "embedding epsilon must be positive and finite"
    );
    let vocabulary = u64::try_from(vocabulary).context("embedding vocabulary does not fit u64")?;
    let width = u64::try_from(width).context("embedding width does not fit u64")?;
    let table_bytes = vocabulary
        .checked_mul(width)
        .and_then(|elements| elements.checked_mul(2))
        .context("embedding table byte extent overflows u64")?;
    let norm_bytes = width
        .checked_mul(2)
        .context("embedding norm byte extent overflows u64")?;
    Ok((table_bytes, norm_bytes))
}
