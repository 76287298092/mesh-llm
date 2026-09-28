//! Checked canonical views of the pinned native artifact. No numeric conversion.
//! File framing and source integrity belong to the independent artifact reader.

mod mapping;
#[cfg(test)]
mod tests;
mod transforms;

use crate::artifact::schema::{DType, Object, ObjectKind};
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const MODEL_ID: &str = "qwen3.8-27b:text:ninfer-v3-control-v1";
pub const SOURCE_SHA256: &str = "74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82";
pub const SOURCE_BYTES: u64 = 23_719_715_844;
pub const TEXT_TENSORS: usize = 1_589;
pub const TEXT_BYTES: u64 = 20_375_588_160;
pub const LAYOUT: &str = "safetensors-row-major-v1"; // Historical row-major consumer ABI, not provenance.

#[derive(Clone, Debug)]
pub(crate) enum Transform {
    Copy,
    Rows { row_bytes: usize, order: Vec<usize> },
    Conv { channels: usize },
    NvScales { rows: usize, width: usize },
}

#[derive(Clone, Debug)]
pub(crate) struct View {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub storage: String,
    pub offset: u64,
    pub source_bytes: u64,
    pub bytes: u64,
    pub transform: Transform,
}

impl View {
    pub fn object(&self, offset: u64, sha256: String) -> Object {
        Object {
            name: self.name.clone(),
            kind: ObjectKind::Tensor,
            dtype: self.dtype.clone(),
            shape: self.shape.clone(),
            layout: LAYOUT.into(),
            offset,
            length: self.bytes,
            sha256,
        }
    }

    pub fn materialize(&self, source: &[u8]) -> Result<Vec<u8>> {
        ensure!(
            source.len() as u64 == self.source_bytes,
            "native view source extent mismatch"
        );
        let bytes = transforms::forward(&self.transform, source)?;
        ensure!(
            bytes.len() as u64 == self.bytes,
            "native view output extent mismatch"
        );
        // All nontrivial permutations are verified against their inverse, byte for byte.
        transforms::verify_inverse(&self.transform, source, &bytes)?;
        Ok(bytes)
    }
}

#[derive(Clone, Debug)]
struct Storage {
    id: String,
    format: String,
    layout: String,
    shape: Vec<usize>,
    bytes: u64,
}

#[derive(Clone, Debug)]
struct Span {
    object: Storage,
    first: usize,
    end: usize,
}

struct Builder<'a> {
    bindings: &'a serde_json::Map<String, Value>,
    uses: &'a [Value],
    objects: BTreeMap<String, Storage>,
    used: BTreeSet<String>,
    views: Vec<View>,
}

pub(crate) fn plan(directory: &Value) -> Result<Vec<View>> {
    let components = directory["components"]
        .as_object()
        .context("native components missing")?;
    let config = &components
        .get("text")
        .context("native text component missing")?["config"];
    mapping::validate_config(config)?;
    let bindings = directory["bindings"]
        .as_object()
        .context("native bindings missing")?;
    let uses = directory["uses"]
        .as_array()
        .context("native uses missing")?;
    let mut objects = BTreeMap::new();
    for item in directory["objects"]
        .as_array()
        .context("native objects missing")?
    {
        if item["kind"] != "tensor" {
            continue;
        }
        let id = text(item, "id")?.to_owned();
        let shape = item["shape"]
            .as_array()
            .context("native shape missing")?
            .iter()
            .map(|v| {
                usize::try_from(v.as_u64().context("native dimension must be unsigned")?)
                    .map_err(Into::into)
            })
            .collect::<Result<Vec<_>>>()?;
        let object = Storage {
            id: id.clone(),
            format: text(item, "format")?.into(),
            layout: text(item, "layout")?.into(),
            shape,
            bytes: number(item, "bytes")?,
        };
        ensure!(
            objects.insert(id, object).is_none(),
            "duplicate native storage"
        );
    }
    let mut builder = Builder {
        bindings,
        uses,
        objects,
        used: BTreeSet::new(),
        views: Vec::new(),
    };
    mapping::build(&mut builder)?;
    let text_names = bindings
        .keys()
        .filter(|n| n.starts_with("text/"))
        .cloned()
        .collect::<BTreeSet<_>>();
    ensure!(
        builder.used == text_names,
        "native text binding coverage mismatch"
    );
    builder.views.sort_by(|a, b| a.name.cmp(&b.name));
    ensure!(
        builder.views.windows(2).all(|w| w[0].name != w[1].name),
        "duplicate canonical view"
    );
    ensure!(
        builder.views.len() == TEXT_TENSORS,
        "native canonical tensor count mismatch"
    );
    let bytes = builder.views.iter().try_fold(0_u64, |sum, v| {
        sum.checked_add(v.bytes).context("native total overflow")
    })?;
    ensure!(bytes == TEXT_BYTES, "native canonical byte count mismatch");
    Ok(builder.views)
}

impl Builder<'_> {
    fn span(&mut self, name: &str) -> Result<Span> {
        let binding = self
            .bindings
            .get(name)
            .with_context(|| format!("missing native binding {name}"))?
            .clone();
        self.used.insert(name.into());
        self.binding_span(&binding)
    }

    fn binding_span(&self, binding: &Value) -> Result<Span> {
        let (id, range) = if let Some(id) = binding.get("object") {
            ensure!(
                binding.as_object().is_some_and(|v| v.len() == 1),
                "ambiguous direct binding"
            );
            (id.as_str().context("invalid direct object")?, None)
        } else {
            let parts = binding["parts"]
                .as_array()
                .context("missing binding parts")?;
            ensure!(
                parts.len() == 1,
                "this model requires one parent per source binding"
            );
            (text(&parts[0], "object")?, Some(&parts[0]["range"]))
        };
        let object = self
            .objects
            .get(id)
            .context("binding references absent tensor")?
            .clone();
        let elements = product(&object.shape)?;
        let (first, end) = if let Some(range) = range {
            let range = range.as_array().context("invalid element range")?;
            ensure!(range.len() == 2, "invalid range arity");
            (
                usize::try_from(range[0].as_u64().context("invalid range start")?)?,
                usize::try_from(range[1].as_u64().context("invalid range end")?)?,
            )
        } else {
            (0, elements)
        };
        ensure!(
            first < end && end <= elements,
            "native element range out of bounds"
        );
        Ok(Span { object, first, end })
    }

    fn push(
        &mut self,
        name: &str,
        dtype: DType,
        shape: Vec<u64>,
        storage: &Storage,
        source: std::ops::Range<u64>,
        transform: Transform,
    ) -> Result<()> {
        ensure!(
            source.start < source.end && source.end <= storage.bytes,
            "native byte range out of bounds"
        );
        let word = match dtype {
            DType::Bf16 => 2,
            DType::F32 => 4,
            DType::Fp8E4m3 | DType::U8 => 1,
            _ => anyhow::bail!("unsupported canonical dtype"),
        };
        let elements = shape.iter().try_fold(1_u64, |p, &d| {
            p.checked_mul(d).context("view shape overflow")
        })?;
        let bytes = elements.checked_mul(word).context("view byte overflow")?;
        self.views.push(View {
            name: format!("tensors/{name}"),
            dtype,
            shape,
            storage: storage.id.clone(),
            offset: source.start,
            source_bytes: source.end - source.start,
            bytes,
            transform,
        });
        Ok(())
    }

    fn direct(&mut self, source: &str, name: &str, dtype: DType, shape: &[u64]) -> Result<()> {
        let span = self.span(source)?;
        let (format, word) = match dtype {
            DType::Bf16 => ("bf16", 2),
            DType::F32 => ("fp32", 4),
            _ => anyhow::bail!("invalid direct dtype"),
        };
        ensure!(
            span.object.format == format && span.object.layout == "contiguous_le_v1",
            "direct format mismatch for {source}"
        );
        let expected = shape.iter().try_fold(1_u64, |p, &d| {
            p.checked_mul(d).context("direct shape overflow")
        })?;
        ensure!(
            (span.end - span.first) as u64 == expected,
            "direct shape mismatch for {source}"
        );
        ensure!(
            span.object.bytes == product(&span.object.shape)? as u64 * word,
            "direct storage extent mismatch"
        );
        self.push(
            name,
            dtype,
            shape.to_vec(),
            &span.object,
            span.first as u64 * word..span.end as u64 * word,
            Transform::Copy,
        )
    }

    fn fp8(
        &mut self,
        sources: &[String],
        name: &str,
        n: usize,
        k: usize,
        interleave: bool,
    ) -> Result<()> {
        let spans = sources
            .iter()
            .map(|s| self.span(s))
            .collect::<Result<Vec<_>>>()?;
        let first = spans.first().context("empty FP8 sources")?;
        let parent = &first.object;
        ensure!(
            parent.format == "fp8_e4m3fn_row_bf16" && parent.layout == "row_scale_v1",
            "FP8 storage format mismatch"
        );
        ensure!(
            parent.shape.len() == 2 && parent.shape[1] == k,
            "FP8 parent geometry mismatch"
        );
        let mut rows = Vec::new();
        for span in &spans {
            ensure!(
                span.object.id == parent.id && span.first % k == 0 && span.end % k == 0,
                "FP8 views must cover complete rows of one parent"
            );
            rows.extend(span.first / k..span.end / k);
        }
        ensure!(rows.len() == n, "FP8 row count mismatch");
        if interleave {
            ensure!(
                spans.len() == 2 && n == 12288 && spans[0].end - spans[0].first == 6144 * k,
                "invalid Q/gate geometry"
            );
            let flat = rows.clone();
            for h in 0..24 {
                for c in 0..256 {
                    rows[h * 512 + c] = flat[h * 256 + c];
                    rows[h * 512 + 256 + c] = flat[6144 + h * 256 + c];
                }
            }
        }
        let codes = product(&parent.shape)? as u64;
        let scales = aligned(codes)?;
        let expected_bytes = scales
            .checked_add(
                (parent.shape[0] as u64)
                    .checked_mul(2)
                    .context("FP8 scale size overflow")?,
            )
            .context("FP8 parent size overflow")?;
        ensure!(parent.bytes == expected_bytes, "FP8 parent size mismatch");
        self.row_view(
            &format!("{name}.weight"),
            DType::Fp8E4m3,
            vec![n as u64, k as u64],
            parent,
            (0, k),
            &rows,
        )?;
        self.row_view(
            &format!("{name}.weight_scale"),
            DType::Bf16,
            vec![n as u64, 1],
            parent,
            (scales, 2),
            &rows,
        )
    }

    fn row_view(
        &mut self,
        name: &str,
        dtype: DType,
        shape: Vec<u64>,
        parent: &Storage,
        plane: (u64, usize),
        rows: &[usize],
    ) -> Result<()> {
        let (base, row_bytes) = plane;
        ensure!(
            rows.iter().copied().collect::<BTreeSet<_>>().len() == rows.len(),
            "duplicate native row selection"
        );
        let lo = *rows.iter().min().context("no rows")?;
        let hi = rows.iter().max().context("no rows")? + 1;
        ensure!(hi <= parent.shape[0], "row selection outside parent");
        let contiguous = rows.iter().enumerate().all(|(i, &r)| r == lo + i);
        let transform = if contiguous {
            Transform::Copy
        } else {
            Transform::Rows {
                row_bytes,
                order: rows.iter().map(|r| r - lo).collect(),
            }
        };
        self.push(
            name,
            dtype,
            shape,
            parent,
            base + (lo * row_bytes) as u64..base + (hi * row_bytes) as u64,
            transform,
        )
    }

    fn nvfp4(&mut self, source: &str, name: &str, n: usize, k: usize, input: &str) -> Result<()> {
        let span = self.span(source)?;
        let p = &span.object;
        ensure!(
            p.format == "nvfp4" && p.layout == "block_scale_k16_m128x4_v1",
            "NVFP4 representation mismatch"
        );
        ensure!(
            p.shape.len() == 2
                && p.shape[1] == k
                && p.shape[0].is_multiple_of(128)
                && k.is_multiple_of(64),
            "NVFP4 parent geometry mismatch"
        );
        ensure!(
            span.first % k == 0 && span.end % k == 0 && span.end - span.first == n * k,
            "NVFP4 child geometry mismatch"
        );
        let first = span.first / k;
        ensure!(
            first.is_multiple_of(128) && n.is_multiple_of(128),
            "NVFP4 child must cover complete scale tiles"
        );
        let codes = product(&p.shape)? as u64 / 2;
        let scales = aligned(codes)?;
        let divisor = scales + (product(&p.shape)? / 16) as u64;
        ensure!(p.bytes == divisor + 4, "NVFP4 parent size mismatch");
        self.push(
            &format!("{name}.weight_packed"),
            DType::U8,
            vec![n as u64, (k / 2) as u64],
            p,
            (span.first / 2) as u64..(span.end / 2) as u64,
            Transform::Copy,
        )?;
        self.push(
            &format!("{name}.weight_scale"),
            DType::Fp8E4m3,
            vec![n as u64, (k / 16) as u64],
            p,
            scales + (span.first / 16) as u64..scales + (span.end / 16) as u64,
            Transform::NvScales { rows: n, width: k },
        )?;
        self.push(
            &format!("{name}.weight_global_scale"),
            DType::F32,
            vec![1],
            p,
            divisor..divisor + 4,
            Transform::Copy,
        )?;
        let uses = self
            .uses
            .iter()
            .filter(|u| u["parameter"] == source && u["input"] == input)
            .collect::<Vec<_>>();
        ensure!(
            uses.len() == 1 && uses[0]["activation_policy"] == "AllowA4",
            "missing/ambiguous NVFP4 use"
        );
        let aux = self.binding_span(&uses[0]["auxiliaries"]["activation_input_divisor"])?;
        ensure!(
            aux.object.format == "fp32"
                && aux.object.layout == "contiguous_le_v1"
                && aux.object.bytes == 4
                && aux.first == 0
                && aux.end == 1,
            "invalid activation divisor"
        );
        self.push(
            &format!("{name}.input_global_scale"),
            DType::F32,
            vec![1],
            &aux.object,
            0..4,
            Transform::Copy,
        )
    }
}

fn number(value: &Value, key: &str) -> Result<u64> {
    value[key]
        .as_u64()
        .with_context(|| format!("missing unsigned {key}"))
}
fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value[key]
        .as_str()
        .with_context(|| format!("missing string {key}"))
}
fn product(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1_usize, |p, &d| {
        p.checked_mul(d).context("native shape overflow")
    })
}
fn aligned(x: u64) -> Result<u64> {
    x.checked_add(255)
        .map(|v| v / 256 * 256)
        .context("native alignment overflow")
}
