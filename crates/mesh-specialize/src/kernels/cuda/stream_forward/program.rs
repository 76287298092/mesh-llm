//! The whole-forward buffer template and the arena slot map derived from it.
//!
//! The template lists, in execution order, one GDN block, one attention block,
//! one FP8 MLP, one NVFP4 MLP, the head, and greedy selection. A real forward
//! executes a block section followed by exactly one MLP section per layer, so
//! every buffer that is simultaneously live in a real forward is also
//! simultaneously live in this template: layer-local temporaries never cross a
//! section, and the only cross-section buffers (`hidden`, `post.*`, inputs, and
//! outputs) have template lifetimes that cover every section they span. The
//! executor in `layers.rs` must issue operations in this order and touch only
//! the named buffers declared here.

use super::plan::{ArenaPlan, BufferSpec, Program};
use crate::kernels::DecoderConfig;
use anyhow::{Context as _, Result, ensure};

pub(super) const MAX_ROWS: usize = 512;

/// Model extents that determine every arena buffer size.
#[derive(Clone, Copy, Debug)]
pub(super) struct Shapes {
    pub(super) hidden: usize,
    pub(super) intermediate: usize,
    pub(super) vocabulary: usize,
    pub(super) gdn_key_heads: usize,
    pub(super) gdn_value_heads: usize,
    pub(super) gdn_head_width: usize,
    pub(super) query_heads: usize,
    pub(super) kv_heads: usize,
    pub(super) attention_width: usize,
    pub(super) rotary_dim: usize,
}

impl Shapes {
    pub(super) fn from_config(config: &DecoderConfig) -> Result<Self> {
        let gdn = &config.gdn_shape;
        let attention = &config.attention_shape;
        ensure!(
            gdn.hidden == config.hidden
                && attention.hidden == config.hidden
                && gdn.intermediate == attention.intermediate,
            "stream forward requires one hidden and intermediate width"
        );
        Ok(Self {
            hidden: config.hidden,
            intermediate: gdn.intermediate,
            vocabulary: config.vocabulary,
            gdn_key_heads: gdn.key_heads,
            gdn_value_heads: gdn.value_heads,
            gdn_head_width: gdn.head_width,
            query_heads: attention.query_heads,
            kv_heads: attention.kv_heads,
            attention_width: attention.head_width,
            rotary_dim: attention.rotary_dim,
        })
    }

    pub(super) fn gdn_qkv(&self) -> usize {
        (2 * self.gdn_key_heads + self.gdn_value_heads) * self.gdn_head_width
    }
    pub(super) fn gdn_inner(&self) -> usize {
        self.gdn_value_heads * self.gdn_head_width
    }
    pub(super) fn query_width(&self) -> usize {
        self.query_heads * self.attention_width
    }
    pub(super) fn kv_width(&self) -> usize {
        self.kv_heads * self.attention_width
    }
    pub(super) fn greedy_tiles(&self) -> usize {
        self.vocabulary.div_ceil(1024)
    }
}

/// Build buffer lifetimes for a forward of `rows` rows.
pub(super) fn forward_program(shapes: &Shapes, rows: usize) -> Result<Vec<BufferSpec>> {
    ensure!(
        (1..=MAX_ROWS).contains(&rows),
        "stream forward rows must be in 1..={MAX_ROWS}"
    );
    let h = shapes.hidden;
    let mut p = Program::new();
    p.whole("tokens", rows * 4)?;
    p.whole("row_ids", rows * 4)?;
    p.whole("hidden", rows * h * 2)?;
    p.op(
        &["tokens"],
        &[
            ("hidden", rows * h * 2),
            ("entry.normalized", rows * h * 2),
            ("entry.raw", rows * h * 4),
        ],
    )?;
    gdn_section(&mut p, shapes, rows)?;
    attention_section(&mut p, shapes, rows)?;
    mlp_section(&mut p, shapes, rows, "mlp8", false)?;
    mlp_section(&mut p, shapes, rows, "mlp4", true)?;
    norm(&mut p, "head.norm", "hidden", 1, h)?;
    projection(
        &mut p,
        "head",
        "head.norm.out",
        [1, h, shapes.vocabulary],
        false,
    )?;
    p.op(
        &["head.values"],
        &[("greedy.partials", shapes.greedy_tiles() * 16)],
    )?;
    p.op(&["greedy.partials"], &[("greedy.result", 16)])?;
    p.keep("head.values")?;
    p.keep("greedy.result")?;
    Ok(p.finish())
}

fn norm(p: &mut Program, name: &str, input: &str, rows: usize, width: usize) -> Result<()> {
    let n = |suffix: &str| format!("{name}.{suffix}");
    p.op(
        &[input, "row_ids"],
        &[
            (&n("copy"), rows * width * 2),
            (&n("out"), rows * width * 2),
            (&n("raw"), rows * width * 4),
        ],
    )
}

/// Quantize then project; `shape` is `[rows, width, channels]`.
fn projection(
    p: &mut Program,
    name: &str,
    input: &str,
    shape: [usize; 3],
    nvfp4: bool,
) -> Result<()> {
    let [rows, width, channels] = shape;
    let n = |suffix: &str| format!("{name}.{suffix}");
    if nvfp4 {
        p.op(
            &[input],
            &[
                (&n("codes"), rows * width / 2),
                (&n("scales"), rows * width / 16),
                (&n("effective"), rows * width / 16 * 4),
            ],
        )?;
    } else {
        p.op(
            &[input],
            &[(&n("codes"), rows * width), (&n("scales"), rows * 4)],
        )?;
    }
    p.op(
        &[&n("codes"), &n("scales")],
        &[
            (&n("values"), rows * channels * 2),
            (&n("raw"), rows * channels * 4),
        ],
    )
}

fn bf16_projection(p: &mut Program, name: &str, input: &str, rows: usize, channels: usize) -> Result<()> {
    p.op(
        &[input],
        &[
            (&format!("{name}.values"), rows * channels * 2),
            (&format!("{name}.raw"), rows * channels * 4),
        ],
    )
}

fn gdn_section(p: &mut Program, s: &Shapes, m: usize) -> Result<()> {
    let (h, qkv, inner, vh) = (s.hidden, s.gdn_qkv(), s.gdn_inner(), s.gdn_value_heads);
    let qk = m * s.gdn_key_heads * s.gdn_head_width * 4;
    norm(p, "gdn.norm", "hidden", m, h)?;
    projection(p, "gdn.qkv", "gdn.norm.out", [m, h, qkv], false)?;
    projection(p, "gdn.z", "gdn.norm.out", [m, h, inner], false)?;
    bf16_projection(p, "gdn.a", "gdn.norm.out", m, vh)?;
    bf16_projection(p, "gdn.b", "gdn.norm.out", m, vh)?;
    p.op(
        &["gdn.qkv.values"],
        &[
            ("gdn.conv.next", qkv * 6),
            ("gdn.conv.values", m * qkv * 2),
            ("gdn.conv.raw", m * qkv * 4),
            ("gdn.conv.silu", m * qkv * 4),
        ],
    )?;
    p.op(&["gdn.conv.next"], &[])?; // stream copy into persistent history
    p.op(&["gdn.conv.values"], &[("gdn.q", qk), ("gdn.k", qk)])?;
    p.op(
        &["gdn.a.values", "gdn.b.values"],
        &[
            ("gdn.beta", m * vh * 2),
            ("gdn.g", m * vh * 4),
            ("gdn.decay", m * vh * 4),
        ],
    )?;
    p.op(
        &["gdn.q", "gdn.k", "gdn.conv.values", "gdn.beta", "gdn.decay"],
        &[("gdn.rec.values", m * inner * 2), ("gdn.rec.raw", m * inner * 4)],
    )?;
    p.op(
        &["gdn.rec.values", "gdn.z.values"],
        &[
            ("gdn.gated.values", m * inner * 2),
            ("gdn.gated.normalized", m * inner * 4),
            ("gdn.gated.weighted", m * inner * 2),
            ("gdn.gated.silu", m * inner * 4),
            ("gdn.gated.raw", m * inner * 4),
        ],
    )?;
    projection(p, "gdn.out", "gdn.gated.values", [m, inner, h], false)?;
    p.op(
        &["hidden", "gdn.out.values"],
        &[
            ("post.sum", m * h * 2),
            ("post.x", m * h * 2),
            ("gdn.post.raw", m * h * 4),
        ],
    )
}

fn attention_section(p: &mut Program, s: &Shapes, m: usize) -> Result<()> {
    let (h, qw, kvw) = (s.hidden, s.query_width(), s.kv_width());
    norm(p, "attn.norm", "hidden", m, h)?;
    projection(p, "attn.q", "attn.norm.out", [m, h, 2 * qw], false)?;
    projection(p, "attn.k", "attn.norm.out", [m, h, kvw], false)?;
    projection(p, "attn.v", "attn.norm.out", [m, h, kvw], false)?;
    for (name, input, width) in [("attn.qp", "attn.q.values", qw), ("attn.kp", "attn.k.values", kvw)] {
        let n = |suffix: &str| format!("{name}.{suffix}");
        p.op(
            &[input],
            &[
                (&n("values"), m * width * 2),
                (&n("normalized"), m * width * 2),
                (&n("raw"), m * width * 4),
                (&n("gate"), m * width * 2),
            ],
        )?;
    }
    p.op(&["attn.kp.values", "attn.v.values"], &[])?; // K/V append into state
    p.op(
        &["attn.qp.values"],
        &[("attn.o.values", m * qw * 2), ("attn.o.raw", m * qw * 4)],
    )?;
    p.op(
        &["attn.o.values", "attn.qp.gate"],
        &[
            ("attn.gated.values", m * qw * 2),
            ("attn.gated.sigmoid", m * qw * 4),
            ("attn.gated.activated", m * qw * 2),
            ("attn.gated.raw", m * qw * 4),
        ],
    )?;
    projection(p, "attn.out", "attn.gated.values", [m, qw, h], false)?;
    p.op(
        &["hidden", "attn.out.values"],
        &[
            ("post.sum", m * h * 2),
            ("post.x", m * h * 2),
            ("attn.post.raw", m * h * 4),
        ],
    )
}

fn mlp_section(p: &mut Program, s: &Shapes, m: usize, name: &str, nvfp4: bool) -> Result<()> {
    let (h, i) = (s.hidden, s.intermediate);
    let n = |suffix: &str| format!("{name}.{suffix}");
    projection(p, &n("gate"), "post.x", [m, h, i], nvfp4)?;
    projection(p, &n("up"), "post.x", [m, h, i], nvfp4)?;
    p.op(
        &[&n("gate.values"), &n("up.values")],
        &[
            (&n("act.values"), m * i * 2),
            (&n("act.silu"), m * i * 4),
            (&n("act.activated"), m * i * 2),
            (&n("act.raw"), m * i * 4),
        ],
    )?;
    projection(p, &n("down"), &n("act.values"), [m, i, h], nvfp4)?;
    p.op(&["post.sum", &n("down.values")], &[("hidden", m * h * 2)])
}

/// Device addresses for one projection's quantized input and outputs.
/// `effective` is zero for FP8, which has no effective-scale output.
#[derive(Clone, Copy, Debug)]
pub(super) struct ProjectionSlots {
    pub(super) codes: u64,
    pub(super) scales: u64,
    pub(super) effective: u64,
    pub(super) values: u64,
    pub(super) raw: u64,
}

/// `embedding_norm_bf16` outputs: residual copy, normalized, FP32 diagnostic.
#[derive(Clone, Copy, Debug)]
pub(super) struct NormSlots {
    pub(super) copy: u64,
    pub(super) out: u64,
    pub(super) raw: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct GdnSlots {
    pub(super) norm: NormSlots,
    pub(super) qkv: ProjectionSlots,
    pub(super) z: ProjectionSlots,
    /// BF16 A and B outputs as `[values, raw]`.
    pub(super) a: [u64; 2],
    pub(super) b: [u64; 2],
    /// Convolution outputs in kernel order: next history, values, conv, SiLU.
    pub(super) conv: [u64; 4],
    pub(super) q: u64,
    pub(super) k: u64,
    /// Gate outputs in kernel order: beta, g, decay.
    pub(super) gates: [u64; 3],
    /// Recurrence outputs: values, raw.
    pub(super) recurrent: [u64; 2],
    /// Gated-norm outputs in kernel order: values, normalized, weighted, SiLU, raw.
    pub(super) gated: [u64; 5],
    pub(super) out: ProjectionSlots,
    pub(super) post_raw: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct AttentionSlots {
    pub(super) norm: NormSlots,
    pub(super) q: ProjectionSlots,
    pub(super) k: ProjectionSlots,
    pub(super) v: ProjectionSlots,
    /// Preparation outputs in kernel order: values, normalized, raw, gate.
    pub(super) q_prepared: [u64; 4],
    pub(super) k_prepared: [u64; 4],
    /// Attention outputs: values, raw.
    pub(super) output: [u64; 2],
    /// Gate outputs in kernel order: values, sigmoid, activated, raw.
    pub(super) gated: [u64; 4],
    pub(super) out: ProjectionSlots,
    pub(super) post_raw: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct MlpSlots {
    pub(super) gate: ProjectionSlots,
    pub(super) up: ProjectionSlots,
    /// SiLU-product outputs in kernel order: values, SiLU, activated, raw.
    pub(super) activation: [u64; 4],
    pub(super) down: ProjectionSlots,
}

/// Every device address the executor uses, resolved once at construction.
#[derive(Clone, Copy, Debug)]
pub(super) struct Slots {
    pub(super) tokens: u64,
    pub(super) row_ids: u64,
    pub(super) hidden: u64,
    /// Entry outputs discarded after the embedding: normalized, raw.
    pub(super) entry: [u64; 2],
    pub(super) post_sum: u64,
    pub(super) post_x: u64,
    pub(super) gdn: GdnSlots,
    pub(super) attention: AttentionSlots,
    pub(super) mlp_fp8: MlpSlots,
    pub(super) mlp_nvfp4: MlpSlots,
    pub(super) head_norm: NormSlots,
    pub(super) head: ProjectionSlots,
    pub(super) partials: u64,
    pub(super) result: u64,
    /// Arena byte offsets for host readback.
    pub(super) logits_offset: usize,
    pub(super) result_offset: usize,
    pub(super) row_ids_offset: usize,
}

struct Resolver<'a> {
    plan: &'a ArenaPlan,
    base: u64,
}

impl Resolver<'_> {
    fn at(&self, name: &str) -> Result<u64> {
        let offset = u64::try_from(self.plan.get(name)?.offset)?;
        self.base
            .checked_add(offset)
            .with_context(|| format!("arena address overflow for `{name}`"))
    }
    fn optional(&self, name: &str) -> Result<u64> {
        match self.plan.offset(name) {
            Some(_) => self.at(name),
            None => Ok(0),
        }
    }
    fn many<const N: usize>(&self, prefix: &str, suffixes: [&str; N]) -> Result<[u64; N]> {
        let mut addresses = [0; N];
        for (address, suffix) in addresses.iter_mut().zip(suffixes) {
            *address = self.at(&format!("{prefix}.{suffix}"))?;
        }
        Ok(addresses)
    }
    fn projection(&self, name: &str) -> Result<ProjectionSlots> {
        Ok(ProjectionSlots {
            codes: self.at(&format!("{name}.codes"))?,
            scales: self.at(&format!("{name}.scales"))?,
            effective: self.optional(&format!("{name}.effective"))?,
            values: self.at(&format!("{name}.values"))?,
            raw: self.at(&format!("{name}.raw"))?,
        })
    }
    fn norm(&self, name: &str) -> Result<NormSlots> {
        let [copy, out, raw] = self.many(name, ["copy", "out", "raw"])?;
        Ok(NormSlots { copy, out, raw })
    }
    fn gdn(&self) -> Result<GdnSlots> {
        Ok(GdnSlots {
            norm: self.norm("gdn.norm")?,
            qkv: self.projection("gdn.qkv")?,
            z: self.projection("gdn.z")?,
            a: self.many("gdn.a", ["values", "raw"])?,
            b: self.many("gdn.b", ["values", "raw"])?,
            conv: self.many("gdn.conv", ["next", "values", "raw", "silu"])?,
            q: self.at("gdn.q")?,
            k: self.at("gdn.k")?,
            gates: [self.at("gdn.beta")?, self.at("gdn.g")?, self.at("gdn.decay")?],
            recurrent: self.many("gdn.rec", ["values", "raw"])?,
            gated: self.many(
                "gdn.gated",
                ["values", "normalized", "weighted", "silu", "raw"],
            )?,
            out: self.projection("gdn.out")?,
            post_raw: self.at("gdn.post.raw")?,
        })
    }
    fn attention(&self) -> Result<AttentionSlots> {
        let prepared = ["values", "normalized", "raw", "gate"];
        Ok(AttentionSlots {
            norm: self.norm("attn.norm")?,
            q: self.projection("attn.q")?,
            k: self.projection("attn.k")?,
            v: self.projection("attn.v")?,
            q_prepared: self.many("attn.qp", prepared)?,
            k_prepared: self.many("attn.kp", prepared)?,
            output: self.many("attn.o", ["values", "raw"])?,
            gated: self.many("attn.gated", ["values", "sigmoid", "activated", "raw"])?,
            out: self.projection("attn.out")?,
            post_raw: self.at("attn.post.raw")?,
        })
    }
    fn mlp(&self, name: &str) -> Result<MlpSlots> {
        Ok(MlpSlots {
            gate: self.projection(&format!("{name}.gate"))?,
            up: self.projection(&format!("{name}.up"))?,
            activation: self.many(
                &format!("{name}.act"),
                ["values", "silu", "activated", "raw"],
            )?,
            down: self.projection(&format!("{name}.down"))?,
        })
    }
}

impl Slots {
    /// Resolve every executor buffer against an arena based at `base`.
    pub(super) fn resolve(plan: &ArenaPlan, base: u64) -> Result<Self> {
        let r = Resolver { plan, base };
        Ok(Self {
            tokens: r.at("tokens")?,
            row_ids: r.at("row_ids")?,
            hidden: r.at("hidden")?,
            entry: r.many("entry", ["normalized", "raw"])?,
            post_sum: r.at("post.sum")?,
            post_x: r.at("post.x")?,
            gdn: r.gdn()?,
            attention: r.attention()?,
            mlp_fp8: r.mlp("mlp8")?,
            mlp_nvfp4: r.mlp("mlp4")?,
            head_norm: r.norm("head.norm")?,
            head: r.projection("head")?,
            partials: r.at("greedy.partials")?,
            result: r.at("greedy.result")?,
            logits_offset: plan.get("head.values")?.offset,
            result_offset: plan.get("greedy.result")?.offset,
            row_ids_offset: plan.get("row_ids")?.offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_ROWS, Shapes, Slots, forward_program};
    use super::super::plan::{ArenaPlan, BufferSpec, peak_live_bytes};

    fn qwen() -> Shapes {
        Shapes {
            hidden: 5120,
            intermediate: 17408,
            vocabulary: 248_320,
            gdn_key_heads: 16,
            gdn_value_heads: 48,
            gdn_head_width: 128,
            query_heads: 24,
            kv_heads: 4,
            attention_width: 256,
            rotary_dim: 64,
        }
    }

    fn find<'a>(specs: &'a [BufferSpec], name: &str) -> &'a BufferSpec {
        specs.iter().find(|spec| spec.name == name).unwrap()
    }

    #[test]
    fn qwen_program_places_and_resolves_every_slot() {
        for rows in [1, 2, 4, 15, 16, 17, 128, MAX_ROWS] {
            let specs = forward_program(&qwen(), rows).unwrap();
            let plan = ArenaPlan::place(&specs).unwrap();
            plan.validate(&specs).unwrap();
            assert!(plan.total_bytes >= plan.peak_live_bytes);
            assert!(plan.total_bytes <= specs.iter().map(|s| s.bytes + 256).sum());
            let slots = Slots::resolve(&plan, 1 << 40).unwrap();
            assert_eq!(slots.gdn.qkv.effective, 0);
            assert_ne!(slots.mlp_nvfp4.gate.effective, 0);
            let tokens = plan.get("tokens").unwrap().offset as u64;
            assert_eq!(slots.tokens, (1 << 40) + tokens);
        }
    }

    #[test]
    fn carried_buffers_are_never_reused_during_the_forward() {
        let specs = forward_program(&qwen(), 16).unwrap();
        let end = specs.iter().map(|s| s.last).max().unwrap();
        for name in ["tokens", "row_ids", "hidden"] {
            let spec = find(&specs, name);
            assert_eq!((spec.first, spec.last), (0, end), "{name}");
        }
        for name in ["head.values", "greedy.result"] {
            assert_eq!(find(&specs, name).last, end, "{name}");
        }
        let post = find(&specs, "post.sum");
        let gdn_first = find(&specs, "gdn.norm.out").first;
        let nvfp4_add = find(&specs, "mlp4.down.values").last;
        assert!(post.first < find(&specs, "attn.norm.out").first);
        assert!(post.first > gdn_first && post.last == nvfp4_add);
    }

    #[test]
    fn slot_extents_grow_with_rows_so_max_rows_plans_cover_smaller_forwards() {
        let small = forward_program(&qwen(), 1).unwrap();
        let large = forward_program(&qwen(), MAX_ROWS).unwrap();
        assert_eq!(small.len(), large.len());
        for spec in &small {
            let other = find(&large, &spec.name);
            assert!(other.bytes >= spec.bytes, "{}", spec.name);
            assert_eq!((other.first, other.last), (spec.first, spec.last));
        }
    }

    #[test]
    fn peak_live_bound_matches_the_widest_step() {
        let specs = forward_program(&qwen(), MAX_ROWS).unwrap();
        // The MLP SiLU product at 512 rows keeps gate, up, and four outputs live.
        let activation = 512 * 17408 * (2 + 2 + 2 + 4 + 2 + 4);
        assert!(peak_live_bytes(&specs) >= activation);
        assert!(forward_program(&qwen(), 0).is_err());
        assert!(forward_program(&qwen(), MAX_ROWS + 1).is_err());
    }
}
