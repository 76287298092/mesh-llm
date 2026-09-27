//! Independent full-batch scalar composition of one GDN decoder block.

use anyhow::{Context, Result, ensure};

use crate::{
    causal_conv4_reference, decoder_mlp_reference, decoder_ops_reference, gated_norm_reference,
    gdn_prepare_reference, gdn_recurrent_reference, kernels, residual_add_reference,
    residual_norm_reference,
};

const MAX_ROWS: usize = 2048;
const MAX_HIDDEN: usize = 32_768;
const MAX_HEADS: usize = 256;
const MAX_KEY_HEADS: usize = 64;
const MAX_HEAD_WIDTH: usize = 256;
const EPSILON: f32 = 1.0e-6;

pub struct Weights {
    pub input_norm: Vec<u16>,
    pub post_norm: Vec<u16>,
    pub qkv: kernels::Fp8Projection,
    pub z: kernels::Fp8Projection,
    pub a: kernels::Bf16Projection,
    pub b: kernels::Bf16Projection,
    pub convolution: Vec<u16>,
    pub a_log: Vec<u16>,
    pub dt_bias: Vec<u16>,
    pub gated_norm: Vec<u16>,
    pub out: kernels::Fp8Projection,
    pub mlp: decoder_mlp_reference::Weights,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Dimensions {
    input_elements: usize,
    qkv_channels: usize,
    inner: usize,
}

/// Run a complete GDN block from BF16 hidden rows, starting with zero state.
pub fn run(
    hidden: &[u16],
    rows: usize,
    shape: &kernels::GdnShape,
    weights: &Weights,
) -> Result<Vec<u16>> {
    let dimensions = validate(hidden, rows, shape, weights)?;
    let normalized =
        decoder_ops_reference::normalize(hidden, &weights.input_norm, rows, shape.hidden)?;
    let qkv = decoder_ops_reference::fp8(&normalized, &weights.qkv, rows, shape.hidden)?;
    let z = decoder_ops_reference::fp8(&normalized, &weights.z, rows, shape.hidden)?;
    let a = decoder_ops_reference::bf16(&normalized, &weights.a, rows, shape.hidden)?;
    let b = decoder_ops_reference::bf16(&normalized, &weights.b, rows, shape.hidden)?;

    let convolution = causal_conv4_reference::run(
        &qkv,
        &weights.convolution,
        &vec![0; checked_product(dimensions.qkv_channels, 3, "GDN history")?],
        rows,
        dimensions.qkv_channels,
    )?;
    let gdn_shape = gdn_prepare_reference::Shape {
        rows,
        key_heads: shape.key_heads,
        value_heads: shape.value_heads,
        width: shape.head_width,
    };
    let prepared = gdn_prepare_reference::run(
        &convolution.output,
        &a,
        &b,
        &weights.a_log,
        &weights.dt_bias,
        &gdn_shape,
    )?;
    let recurrent_state = checked_product(
        checked_product(shape.value_heads, shape.head_width, "GDN state heads")?,
        shape.head_width,
        "GDN recurrent state",
    )?;
    let recurrent = gdn_recurrent_reference::run(
        &gdn_recurrent_reference::Input {
            q: &prepared.q,
            k: &prepared.k,
            qkv: &convolution.output,
            beta: &prepared.beta,
            decay: &prepared.decay,
        },
        &vec![0.0; recurrent_state],
        &gdn_recurrent_reference::Shape {
            rows,
            key_heads: shape.key_heads,
            value_heads: shape.value_heads,
            width: shape.head_width,
        },
        gdn_recurrent_reference::Reduction::OrderedF32,
    )?;
    let gated = gated_norm_reference::run(
        &recurrent.output,
        &z,
        &weights.gated_norm,
        checked_product(rows, shape.value_heads, "GDN output groups")?,
        shape.head_width,
        EPSILON,
    )?;
    let output_projection =
        decoder_ops_reference::fp8(&gated.output, &weights.out, rows, dimensions.inner)?;
    let post_attention = residual_norm_reference::run(
        hidden,
        &output_projection,
        &weights.post_norm,
        rows,
        shape.hidden,
        EPSILON,
    )?;
    let mlp =
        decoder_mlp_reference::run(&post_attention.normalized, rows, shape.hidden, &weights.mlp)?;
    residual_add_reference::run(&post_attention.residual, &mlp)
}

fn validate(
    hidden: &[u16],
    rows: usize,
    shape: &kernels::GdnShape,
    weights: &Weights,
) -> Result<Dimensions> {
    let dimensions = dimensions(rows, shape)?;
    ensure!(
        hidden.len() == dimensions.input_elements,
        "GDN hidden extent mismatch"
    );
    ensure!(
        hidden
            .iter()
            .all(|&value| crate::entry_reference::bf16_to_f32(value).is_finite()),
        "GDN hidden input contains a nonfinite BF16 value"
    );
    validate_weights(shape, dimensions, weights)?;
    Ok(dimensions)
}

fn dimensions(rows: usize, shape: &kernels::GdnShape) -> Result<Dimensions> {
    ensure!(
        (1..=MAX_ROWS).contains(&rows),
        "GDN rows must be in 1..={MAX_ROWS}"
    );
    ensure!(
        (16..=MAX_HIDDEN).contains(&shape.hidden)
            && (16..=MAX_HIDDEN).contains(&shape.intermediate),
        "GDN hidden and intermediate widths must be in 16..={MAX_HIDDEN}"
    );
    ensure!(
        (1..=MAX_KEY_HEADS).contains(&shape.key_heads)
            && (1..=MAX_HEADS).contains(&shape.value_heads)
            && shape.value_heads.is_multiple_of(shape.key_heads),
        "invalid GDN key/value head counts"
    );
    ensure!(
        (1..=MAX_HEAD_WIDTH).contains(&shape.head_width) && shape.head_width.is_power_of_two(),
        "GDN head width must be a power of two in 1..={MAX_HEAD_WIDTH}"
    );
    let qkv_heads = checked_product(shape.key_heads, 2, "GDN QKV heads")?
        .checked_add(shape.value_heads)
        .context("GDN QKV head count overflows usize")?;
    let qkv_channels = checked_product(qkv_heads, shape.head_width, "GDN QKV channels")?;
    let inner = checked_product(shape.value_heads, shape.head_width, "GDN inner width")?;
    Ok(Dimensions {
        input_elements: checked_product(rows, shape.hidden, "GDN hidden rows")?,
        qkv_channels,
        inner,
    })
}

fn validate_weights(
    shape: &kernels::GdnShape,
    dimensions: Dimensions,
    weights: &Weights,
) -> Result<()> {
    validate_words("GDN input norm", &weights.input_norm, shape.hidden)?;
    validate_words("GDN post norm", &weights.post_norm, shape.hidden)?;
    validate_words(
        "GDN convolution",
        &weights.convolution,
        checked_product(dimensions.qkv_channels, 4, "GDN convolution")?,
    )?;
    validate_words("GDN A_log", &weights.a_log, shape.value_heads)?;
    ensure!(
        weights
            .a_log
            .iter()
            .all(|&bits| (-80.0..=80.0).contains(&crate::entry_reference::bf16_to_f32(bits))),
        "GDN A_log is outside the qualified finite range"
    );
    validate_words("GDN dt_bias", &weights.dt_bias, shape.value_heads)?;
    validate_words("GDN gated norm", &weights.gated_norm, shape.head_width)?;
    validate_fp8(&weights.qkv, dimensions.qkv_channels, shape.hidden, "QKV")?;
    validate_fp8(&weights.z, dimensions.inner, shape.hidden, "Z")?;
    validate_bf16_projection(&weights.a, shape.value_heads, shape.hidden, "A")?;
    validate_bf16_projection(&weights.b, shape.value_heads, shape.hidden, "B")?;
    validate_fp8(&weights.out, shape.hidden, dimensions.inner, "GDN output")?;
    validate_mlp(&weights.mlp, shape.hidden, shape.intermediate)
}

fn validate_words(name: &str, values: &[u16], expected: usize) -> Result<()> {
    ensure!(values.len() == expected, "{name} has the wrong BF16 extent");
    ensure!(
        values
            .iter()
            .all(|&bits| crate::entry_reference::bf16_to_f32(bits).is_finite()),
        "{name} contains a nonfinite BF16 value"
    );
    Ok(())
}

fn validate_fp8(
    projection: &kernels::Fp8Projection,
    channels: usize,
    width: usize,
    name: &str,
) -> Result<()> {
    let weight_count = checked_product(channels, width, name)?;
    let scale_bytes = checked_product(channels, 2, "FP8 scales")?;
    ensure!(
        projection.channels == channels
            && projection.weights.len() == weight_count
            && projection.scales.len() == scale_bytes,
        "GDN {name} projection extent mismatch"
    );
    ensure!(
        projection.weights.iter().all(|code| code & 0x7f != 0x7f),
        "GDN {name} projection contains a nonfinite E4M3 code"
    );
    let (scales, remainder) = projection.scales.as_chunks::<2>();
    ensure!(
        remainder.is_empty(),
        "GDN {name} scale bytes are not BF16 aligned"
    );
    ensure!(
        scales.iter().all(|bytes| {
            let scale = crate::entry_reference::bf16_to_f32(u16::from_le_bytes(*bytes));
            scale.is_finite() && scale > 0.0
        }),
        "GDN {name} projection has an invalid scale"
    );
    Ok(())
}

fn validate_bf16_projection(
    projection: &kernels::Bf16Projection,
    channels: usize,
    width: usize,
    name: &str,
) -> Result<()> {
    let values = checked_product(channels, width, name)?;
    validate_bytes(name, &projection.weights, checked_product(values, 2, name)?)?;
    ensure!(
        projection.channels == channels,
        "GDN {name} channel mismatch"
    );
    Ok(())
}

fn validate_bytes(name: &str, bytes: &[u8], expected: usize) -> Result<()> {
    ensure!(bytes.len() == expected, "{name} byte extent mismatch");
    let (words, remainder) = bytes.as_chunks::<2>();
    ensure!(
        remainder.is_empty(),
        "{name} byte extent is not BF16 aligned"
    );
    ensure!(
        words.iter().all(|word| {
            crate::entry_reference::bf16_to_f32(u16::from_le_bytes(*word)).is_finite()
        }),
        "{name} contains a nonfinite BF16 value"
    );
    Ok(())
}

fn validate_mlp(
    weights: &decoder_mlp_reference::Weights,
    hidden: usize,
    intermediate: usize,
) -> Result<()> {
    match weights {
        decoder_mlp_reference::Weights::Nvfp4(mlp) => {
            ensure!(
                intermediate.is_multiple_of(16),
                "GDN NVFP4 MLP intermediate width must be a multiple of 16"
            );
            ensure!(
                mlp.gate.channels == intermediate
                    && mlp.up.channels == intermediate
                    && mlp.down.channels == hidden,
                "GDN NVFP4 MLP channels do not match the block shape"
            );
            validate_nvfp4(&mlp.gate, hidden, "gate")?;
            validate_nvfp4(&mlp.up, hidden, "up")?;
            validate_nvfp4(&mlp.down, intermediate, "down")
        }
        decoder_mlp_reference::Weights::Fp8(mlp) => {
            ensure!(
                mlp.gate.channels == intermediate
                    && mlp.up.channels == intermediate
                    && mlp.down.channels == hidden,
                "GDN FP8 MLP channels do not match the block shape"
            );
            validate_fp8_mlp_projection(&mlp.gate, hidden, "gate")?;
            validate_fp8_mlp_projection(&mlp.up, hidden, "up")?;
            validate_fp8_mlp_projection(&mlp.down, intermediate, "down")
        }
    }
}

fn validate_nvfp4(projection: &kernels::Nvfp4Projection, width: usize, name: &str) -> Result<()> {
    ensure!(
        (16..=MAX_HIDDEN).contains(&projection.channels) && width.is_multiple_of(16),
        "GDN NVFP4 MLP {name} shape is invalid"
    );
    let elements = checked_product(projection.channels, width, "NVFP4 MLP matrix")?;
    ensure!(
        projection.packed.len() == elements / 2 && projection.scales.len() == elements / 16,
        "GDN NVFP4 MLP {name} storage extent mismatch"
    );
    ensure!(
        projection.scales.iter().all(|&code| code <= 126),
        "GDN NVFP4 MLP {name} has a nonfinite E4M3 scale code"
    );
    ensure!(
        projection.input_global.is_finite()
            && projection.input_global > 0.0
            && projection.weight_global.is_finite()
            && projection.weight_global > 0.0,
        "GDN NVFP4 MLP {name} global scale is invalid"
    );
    Ok(())
}

fn validate_fp8_mlp_projection(
    projection: &crate::fp8_mlp_reference::Projection,
    width: usize,
    name: &str,
) -> Result<()> {
    let elements = checked_product(projection.channels, width, "FP8 MLP matrix")?;
    ensure!(
        projection.weights.len() == elements && projection.scales.len() == projection.channels,
        "GDN FP8 MLP {name} storage extent mismatch"
    );
    ensure!(
        projection.weights.iter().all(|code| code & 0x7f != 0x7f)
            && projection.scales.iter().all(|&scale| {
                let value = crate::entry_reference::bf16_to_f32(scale);
                value.is_finite() && value > 0.0
            }),
        "GDN FP8 MLP {name} contains an invalid weight or scale"
    );
    Ok(())
}

fn checked_product(left: usize, right: usize, name: &str) -> Result<usize> {
    left.checked_mul(right)
        .with_context(|| format!("{name} extent overflows usize"))
}

#[cfg(test)]
mod tests {
    use super::{MAX_HIDDEN, MAX_ROWS, Weights, checked_product, dimensions, run};
    use crate::{
        decoder_mlp_reference,
        kernels::{Bf16Projection, Fp8Projection, GdnShape, Nvfp4Mlp, Nvfp4Projection},
    };

    fn shape() -> GdnShape {
        GdnShape {
            hidden: 5120,
            key_heads: 16,
            value_heads: 48,
            head_width: 128,
            intermediate: 17_408,
        }
    }

    #[test]
    fn reports_pinned_and_largest_supported_shape_extents() {
        let dims = dimensions(17, &shape()).unwrap();
        assert_eq!(dims.input_elements, 87_040);
        assert_eq!(dims.qkv_channels, 10_240);
        assert_eq!(dims.inner, 6_144);
        let mut largest = shape();
        largest.key_heads = 64;
        largest.value_heads = 256;
        largest.head_width = 256;
        assert_eq!(dimensions(1, &largest).unwrap().qkv_channels, 98_304);
    }

    #[test]
    fn rejects_invalid_rows_shapes_ratios_and_extents() {
        assert!(dimensions(0, &shape()).is_err());
        assert!(dimensions(MAX_ROWS + 1, &shape()).is_err());

        let mut invalid = shape();
        invalid.hidden = 15;
        assert!(dimensions(1, &invalid).is_err());
        let mut invalid = shape();
        invalid.intermediate = MAX_HIDDEN + 1;
        assert!(dimensions(1, &invalid).is_err());
        let mut invalid = shape();
        invalid.value_heads = 17;
        assert!(dimensions(1, &invalid).is_err());
        let mut invalid = shape();
        invalid.head_width = 257;
        assert!(dimensions(1, &invalid).is_err());
        let mut invalid = shape();
        invalid.key_heads = 65;
        assert!(dimensions(1, &invalid).is_err());
    }

    fn bf16_bytes(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn fp8_projection(channels: usize, width: usize) -> Fp8Projection {
        Fp8Projection {
            name: "fixture".to_owned(),
            weights: vec![0; channels * width],
            scales: bf16_bytes(&vec![0x3f80; channels]),
            channels,
        }
    }

    fn nvfp4_projection(channels: usize, width: usize) -> Nvfp4Projection {
        Nvfp4Projection {
            name: "fixture".to_owned(),
            packed: vec![0; channels * width / 2],
            scales: vec![0; channels * width / 16],
            input_global: 1.0,
            weight_global: 1.0,
            channels,
        }
    }

    fn zero_branch_weights() -> Weights {
        let hidden = 16;
        let inner = 1;
        Weights {
            input_norm: vec![0; hidden],
            post_norm: vec![0; hidden],
            qkv: fp8_projection(3, hidden),
            z: fp8_projection(inner, hidden),
            a: Bf16Projection {
                name: "a".to_owned(),
                weights: bf16_bytes(&vec![0; hidden]),
                channels: 1,
            },
            b: Bf16Projection {
                name: "b".to_owned(),
                weights: bf16_bytes(&vec![0; hidden]),
                channels: 1,
            },
            convolution: vec![0; 3 * 4],
            a_log: vec![0],
            dt_bias: vec![0],
            gated_norm: vec![0],
            out: fp8_projection(hidden, inner),
            mlp: decoder_mlp_reference::Weights::Nvfp4(Nvfp4Mlp {
                gate: nvfp4_projection(hidden, hidden),
                up: nvfp4_projection(hidden, hidden),
                down: nvfp4_projection(hidden, hidden),
            }),
        }
    }

    #[test]
    fn zero_branch_preserves_the_input_residual() {
        let hidden = [0x3f80; 16];
        let output = run(&hidden, 1, &shape_for_zero_branch(), &zero_branch_weights()).unwrap();
        assert_eq!(output, hidden);
    }

    #[test]
    fn rejects_projection_channel_links_before_running_components() {
        let mut weights = zero_branch_weights();
        weights.qkv.channels = 4;
        assert!(run(&[0x3f80; 16], 1, &shape_for_zero_branch(), &weights).is_err());
        assert!(
            run(
                &[0x3f80; 15],
                1,
                &shape_for_zero_branch(),
                &zero_branch_weights()
            )
            .is_err()
        );
        assert!(checked_product(usize::MAX, 2, "test").is_err());
    }

    fn shape_for_zero_branch() -> GdnShape {
        GdnShape {
            hidden: 16,
            key_heads: 1,
            value_heads: 1,
            head_width: 1,
            intermediate: 16,
        }
    }
}
