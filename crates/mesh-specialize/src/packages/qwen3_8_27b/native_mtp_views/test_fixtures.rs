use crate::artifact::ninfer::{Binding, Directory, Object, Part};
use serde_json::json;
use std::collections::BTreeMap;

pub(super) struct TensorSpec<'a> {
    id: &'a str,
    shape: &'a [u64],
    format: &'a str,
    layout: &'a str,
    offset: u64,
    bytes: u64,
}

pub(super) fn tensor(spec: TensorSpec<'_>) -> Object {
    Object {
        id: spec.id.into(),
        kind: "tensor".into(),
        format: Some(spec.format.into()),
        layout: Some(spec.layout.into()),
        shape: spec.shape.to_vec(),
        encoding: None,
        offset: spec.offset,
        bytes: spec.bytes,
    }
}

pub(super) fn synthetic_directory() -> Directory {
    let components = components();
    let mut fixture = TensorFixture::new();
    add_fused_attention(&mut fixture);
    add_projection_objects(&mut fixture);
    add_norm_objects(&mut fixture);
    add_proposal_objects(&mut fixture);
    fixture.directory(components)
}

fn components() -> BTreeMap<String, serde_json::Value> {
    BTreeMap::from([
        (
            "text".into(),
            json!({
                "config": {"hidden_size":5120,"vocab_size":248320},
                "proposal":{"domain":"indexed","rows":131072}
            }),
        ),
        ("mtp".into(), json!({"config":{}})),
    ])
}

struct TensorFixture {
    objects: Vec<Object>,
    bindings: BTreeMap<String, Binding>,
    offset: u64,
}

struct Q8FixtureSpec<'a> {
    id: &'a str,
    binding: &'a str,
    shape: [u64; 2],
    bytes: u64,
}

struct NormFixtureSpec<'a> {
    name: &'a str,
    elements: usize,
}

impl TensorFixture {
    fn new() -> Self {
        Self {
            objects: Vec::new(),
            bindings: BTreeMap::new(),
            offset: 0,
        }
    }

    fn add_tensor(&mut self, object: Object) {
        self.offset = object
            .offset
            .checked_add(object.bytes)
            .expect("fixture tensor extent");
        self.objects.push(object);
    }

    fn add_q8(&mut self, spec: Q8FixtureSpec<'_>) {
        let Q8FixtureSpec {
            id,
            binding,
            shape,
            bytes,
        } = spec;
        self.add_tensor(tensor(TensorSpec {
            id,
            shape: &shape,
            format: "q8_g32_fp16",
            layout: "row_split_k128_v1",
            offset: self.offset,
            bytes,
        }));
        self.bindings
            .insert(binding.into(), Binding::Object { object: id.into() });
    }

    fn add_norm(&mut self, spec: NormFixtureSpec<'_>) {
        let NormFixtureSpec { name, elements } = spec;
        let id = format!("norm-{name}");
        let bytes = u64::try_from(elements * 2).expect("fixture norm size");
        self.add_tensor(tensor(TensorSpec {
            id: &id,
            shape: &[u64::try_from(elements).expect("fixture norm size")],
            format: "bf16",
            layout: "contiguous_le_v1",
            offset: self.offset,
            bytes,
        }));
        self.bindings
            .insert(name.into(), Binding::Object { object: id });
    }

    fn directory(self, components: BTreeMap<String, serde_json::Value>) -> Directory {
        Directory {
            components,
            objects: self.objects,
            bindings: self.bindings,
            uses: Vec::new(),
            files: Vec::new(),
            metadata: json!({}),
            provenance: json!({}),
        }
    }
}

fn add_fused_attention(fixture: &mut TensorFixture) {
    const PARENT_BYTES: u64 = 77_987_840;
    let qkv_offset = fixture.offset;
    fixture.add_tensor(tensor(TensorSpec {
        id: "qkv",
        shape: &[14_336, 5_120],
        format: "q8_g32_fp16",
        layout: "row_split_k128_v1",
        offset: qkv_offset,
        bytes: PARENT_BYTES,
    }));
    for (name, range) in [
        ("mtp/layers/0/attention/query", [0, 31_457_280]),
        ("mtp/layers/0/attention/key", [31_457_280, 36_700_160]),
        ("mtp/layers/0/attention/gate", [36_700_160, 68_157_440]),
        ("mtp/layers/0/attention/value", [68_157_440, 73_400_320]),
    ] {
        fixture.bindings.insert(
            name.into(),
            Binding::Parts {
                parts: vec![Part {
                    object: "qkv".into(),
                    range,
                }],
            },
        );
    }
}

fn add_projection_objects(fixture: &mut TensorFixture) {
    fixture.add_q8(Q8FixtureSpec {
        id: "fc",
        binding: "mtp/input_projection",
        shape: [5_120, 10_240],
        bytes: 55_705_600,
    });
    fixture.add_q8(Q8FixtureSpec {
        id: "attention-output",
        binding: "mtp/layers/0/attention/output",
        shape: [5_120, 6_144],
        bytes: 33_423_360,
    });
    let mlp_offset = fixture.offset;
    fixture.add_tensor(tensor(TensorSpec {
        id: "mlp",
        shape: &[34_816, 5_120],
        format: "q8_g32_fp16",
        layout: "row_split_k128_v1",
        offset: mlp_offset,
        bytes: 189_399_040,
    }));
    for (name, range) in [
        ("mtp/layers/0/mlp/gate", [0, 89_128_960]),
        ("mtp/layers/0/mlp/up", [89_128_960, 178_257_920]),
    ] {
        fixture.bindings.insert(
            name.into(),
            Binding::Parts {
                parts: vec![Part {
                    object: "mlp".into(),
                    range,
                }],
            },
        );
    }
    fixture.add_q8(Q8FixtureSpec {
        id: "mlp-down",
        binding: "mtp/layers/0/mlp/down",
        shape: [5_120, 17_408],
        bytes: 94_699_520,
    });
}

fn add_norm_objects(fixture: &mut TensorFixture) {
    for spec in [
        NormFixtureSpec {
            name: "mtp/embedding_norm",
            elements: 5_120,
        },
        NormFixtureSpec {
            name: "mtp/hidden_norm",
            elements: 5_120,
        },
        NormFixtureSpec {
            name: "mtp/final_norm",
            elements: 5_120,
        },
        NormFixtureSpec {
            name: "mtp/layers/0/input_norm",
            elements: 5_120,
        },
        NormFixtureSpec {
            name: "mtp/layers/0/post_attention_norm",
            elements: 5_120,
        },
        NormFixtureSpec {
            name: "mtp/layers/0/attention/query_norm",
            elements: 256,
        },
        NormFixtureSpec {
            name: "mtp/layers/0/attention/key_norm",
            elements: 256,
        },
    ] {
        fixture.add_norm(spec);
    }
}

fn add_proposal_objects(fixture: &mut TensorFixture) {
    let proposal_offset = fixture.offset;
    fixture.add_tensor(tensor(TensorSpec {
        id: "proposal-head",
        shape: &[131_072, 5_120],
        format: "q4_g64_fp16",
        layout: "row_split_k128_v1",
        offset: proposal_offset,
        bytes: 356_515_840,
    }));
    fixture.bindings.insert(
        "proposal/head".into(),
        Binding::Object {
            object: "proposal-head".into(),
        },
    );
    fixture.add_tensor(tensor(TensorSpec {
        id: "token-map",
        shape: &[131_072],
        format: "int32",
        layout: "contiguous_le_v1",
        offset: 23_718_755_584,
        bytes: 524_288,
    }));
    fixture.bindings.insert(
        "proposal/token_ids".into(),
        Binding::Object {
            object: "token-map".into(),
        },
    );
    fixture.add_tensor(tensor(TensorSpec {
        id: "target-head",
        shape: &[248_320, 5_120],
        format: "fp8_e4m3fn_row_bf16",
        layout: "row_scale_v1",
        offset: 23_719_279_872,
        bytes: 1_271_895_040,
    }));
    fixture.bindings.insert(
        "text/output_head".into(),
        Binding::Object {
            object: "target-head".into(),
        },
    );
}

pub(super) fn exact_map(value: i32) -> Vec<u8> {
    let mut bytes = vec![0; 524_288];
    let (words, remainder) = bytes.as_chunks_mut::<4>();
    assert!(
        remainder.is_empty(),
        "fixture token map size is a multiple of four"
    );
    for word in words {
        word.copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub(super) fn observed_directory() -> Directory {
    let raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/KNOWLEDGE/evidence/reassess-20260928/ninfer-identity/ninfer-artifact-inspect.json"
    ));
    let value: serde_json::Value = serde_json::from_str(raw).expect("inspector evidence is JSON");
    serde_json::from_value(json!({
        "components":value["components"],
        "objects":value["object_records"],
        "bindings":value["binding_records"],
        "uses":value["use_records"],
        "files":[{"path":null,"payload_bytes":value["payload_bytes"]}],
        "metadata":{},
        "provenance":{}
    }))
    .expect("inspected NInfer records match the reader schema")
}
