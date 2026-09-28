use super::*;
use serde_json::json;

fn observed_directory() -> Value {
    let text = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/KNOWLEDGE/evidence/reassess-20260928/ninfer-identity/ninfer-artifact-inspect.json"
    ));
    let v: Value = serde_json::from_str(text).unwrap();
    json!({"components":v["components"],"objects":v["object_records"],
        "bindings":v["binding_records"],"uses":v["use_records"]})
}

#[test]
fn actual_metadata_maps_every_text_binding_without_narrowing() {
    let views = plan(&observed_directory()).unwrap();
    assert_eq!(views.len(), TEXT_TENSORS);
    assert_eq!(views.iter().map(|v| v.bytes).sum::<u64>(), TEXT_BYTES);
    let embedding = views
        .iter()
        .find(|v| v.name.ends_with("embed_tokens.weight"))
        .unwrap();
    assert_eq!(embedding.dtype, DType::Fp8E4m3);
    for v in &views {
        if v.name.ends_with(".A_log") || v.name.ends_with(".dt_bias") {
            assert_eq!(v.dtype, DType::F32);
            assert_eq!(v.bytes, 192);
        }
        assert!(!v.name.ends_with(".k_scale") && !v.name.ends_with(".v_scale"));
    }
}

#[test]
fn rejects_missing_bindings_partial_rows_and_wrong_geometry() {
    let mut v = observed_directory();
    v["bindings"]
        .as_object_mut()
        .unwrap()
        .remove("text/layers/0/gdn/a_log");
    assert!(plan(&v).is_err());
    let mut v = observed_directory();
    v["bindings"]["text/layers/3/attention/query"]["parts"][0]["range"][0] = json!(1);
    assert!(plan(&v).is_err());
    let mut v = observed_directory();
    v["components"]["text"]["config"]["num_attention_heads"] = json!(23);
    assert!(plan(&v).is_err());
}

#[test]
fn rejects_wrong_parent_extent_and_missing_activation_divisor() {
    let mut v = observed_directory();
    let obj = v["objects"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|o| o["id"] == "weight/000002")
        .unwrap();
    obj["bytes"] = json!(1);
    assert!(plan(&v).is_err());
    let mut v = observed_directory();
    let u = v["uses"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|u| u["parameter"] == "text/layers/0/mlp/gate")
        .unwrap();
    u.as_object_mut().unwrap().remove("auxiliaries");
    assert!(plan(&v).is_err());
}

#[test]
fn nvfp4_scale_transform_round_trips_boundaries() {
    for (rows, width) in [(128, 64), (256, 128), (384, 512)] {
        let bytes = (0..rows * width / 16)
            .map(|i| ((i * 73) ^ (i / 251)) as u8)
            .collect::<Vec<_>>();
        let transform = Transform::NvScales { rows, width };
        let out = transforms::forward(&transform, &bytes).unwrap();
        transforms::verify_inverse(&transform, &bytes, &out).unwrap();
        for row in [0, 31, 32, 127, rows - 1] {
            for g in [0, 3, width / 16 - 1] {
                let j = ((row / 128) * (width / 64) + g / 4) * 512
                    + (row % 32) * 16
                    + ((row % 128) / 32) * 4
                    + g % 4;
                assert_eq!(out[row * (width / 16) + g], bytes[j]);
            }
        }
        let mut bad = out;
        bad[0] ^= 1;
        assert!(transforms::verify_inverse(&transform, &bytes, &bad).is_err());
    }
}

#[test]
fn convolution_transpose_preserves_signed_zero_bits() {
    let bytes = [0, 0, 0, 128, 1, 0, 2, 128, 3, 0, 4, 128, 5, 0, 6, 128];
    let kind = Transform::Conv { channels: 2 };
    let out = transforms::forward(&kind, &bytes).unwrap();
    assert_eq!(&out[..8], &[0, 0, 1, 0, 3, 0, 5, 0]);
    assert_eq!(&out[8..], &[0, 128, 2, 128, 4, 128, 6, 128]);
    transforms::verify_inverse(&kind, &bytes, &out).unwrap();
}

#[test]
fn interleave_selects_both_code_and_scale_rows_without_overlaps() {
    let bytes = (0..16).collect::<Vec<u8>>();
    let kind = Transform::Rows {
        row_bytes: 2,
        order: vec![0, 1, 4, 5, 2, 3, 6, 7],
    };
    let out = transforms::forward(&kind, &bytes).unwrap();
    assert_eq!(&out[..8], &[0, 1, 2, 3, 8, 9, 10, 11]);
    transforms::verify_inverse(&kind, &bytes, &out).unwrap();
    assert!(
        transforms::forward(
            &Transform::Rows {
                row_bytes: 2,
                order: vec![0, 0]
            },
            &bytes
        )
        .is_err()
    );
    assert!(
        transforms::forward(
            &Transform::Rows {
                row_bytes: 2,
                order: vec![8]
            },
            &bytes
        )
        .is_err()
    );
}
