//! Resident Q/K normalization, gate split, and partial RoPE wrapper.

use super::{
    driver::{Buffer, Context, Module},
    resident_weights::ResidentWeights,
};
use crate::artifact::schema::DType;
use anyhow::{Context as _, Result, anyhow, ensure};
use std::ffi::c_void;

const EPSILON: f32 = 1e-6;

pub(super) struct Preparation<'w, 'ctx> {
    owner: &'w ResidentWeights<'ctx>,
    weight: u64,
    heads: usize,
    width: usize,
    rotary_dim: usize,
    with_gate: bool,
}

pub(super) struct Tables<'a, 'ctx> {
    pub(super) cos: &'a Buffer<'ctx>,
    pub(super) sin: &'a Buffer<'ctx>,
}

pub(super) struct Output<'ctx> {
    pub(super) values: Buffer<'ctx>,
    /// Valid only when the preparation was created with `with_gate = true`.
    pub(super) gate: Buffer<'ctx>,
}

struct Extents {
    input_bytes: usize,
    table_bytes: usize,
    output_bytes: usize,
    diagnostic_bytes: usize,
    grid_x: u32,
}

impl<'w, 'ctx> Preparation<'w, 'ctx> {
    pub(super) fn new(
        owner: &'w ResidentWeights<'ctx>,
        name: &str,
        heads: usize,
        width: usize,
        rotary_dim: usize,
        with_gate: bool,
    ) -> Result<Self> {
        validate_shape(1, heads, width, rotary_dim)?;
        let width_u64 = u64::try_from(width).context("attention width does not fit u64")?;
        let weight_bytes = width_u64
            .checked_mul(2)
            .context("attention norm weight extent overflows u64")?;
        let weight = owner.tensor(name, DType::Bf16, &[width_u64], weight_bytes)?;
        Ok(Self {
            owner,
            weight,
            heads,
            width,
            rotary_dim,
            with_gate,
        })
    }

    /// Prepare resident row-major Q or K values with resident RoPE tables.
    pub(super) fn run<'a>(
        &self,
        context: &'a Context,
        module: &Module<'_>,
        input: &Buffer<'_>,
        tables: Tables<'_, '_>,
        rows: usize,
    ) -> Result<Output<'a>> {
        ensure!(
            self.owner.belongs_to(context),
            "attention weight belongs to another context"
        );
        ensure!(
            module.belongs_to(context),
            "attention module belongs to another context"
        );
        ensure!(
            input.belongs_to(context),
            "attention input belongs to another context"
        );
        ensure!(
            tables.cos.belongs_to(context),
            "attention cosine table belongs to another context"
        );
        ensure!(
            tables.sin.belongs_to(context),
            "attention sine table belongs to another context"
        );
        let extents = validate_buffers(
            [rows, self.heads, self.width, self.rotary_dim],
            self.with_gate,
            [input.len(), tables.cos.len(), tables.sin.len()],
        )?;

        let values = Buffer::new(context, extents.output_bytes)?;
        let normalized = Buffer::new(context, extents.output_bytes)?;
        let unrounded = Buffer::new(context, extents.diagnostic_bytes)?;
        let gate = Buffer::new(context, extents.output_bytes)?;
        let mut pointers = [
            input.pointer(),
            self.weight,
            tables.cos.pointer(),
            tables.sin.pointer(),
            values.pointer(),
            normalized.pointer(),
            unrounded.pointer(),
            gate.pointer(),
        ];
        let mut dimensions = [
            u32::try_from(rows).context("attention row count does not fit u32")?,
            u32::try_from(self.heads).context("attention head count does not fit u32")?,
            u32::try_from(self.width).context("attention width does not fit u32")?,
            u32::try_from(self.rotary_dim).context("rotary width does not fit u32")?,
            u32::from(self.with_gate),
        ];
        let mut epsilon = EPSILON;
        let mut args: Vec<*mut c_void> = pointers
            .iter_mut()
            .map(|pointer| (pointer as *mut u64).cast())
            .collect();
        args.extend(
            dimensions
                .iter_mut()
                .map(|dimension| (dimension as *mut u32).cast()),
        );
        args.push((&mut epsilon as *mut f32).cast());

        let function = module.function("attention_qk_prepare")?;
        // SAFETY: Arguments follow the eight-pointer attention_qk_prepare ABI.
        // Exact dimensions, extents, context ownership, and output storage were
        // validated above; all allocations stay live through synchronization.
        let launch = unsafe { function.launch([extents.grid_x, 1, 1], [256, 1, 1], 0, &mut args) };
        if let Err(launch_error) = launch {
            return match context.synchronize() {
                Ok(()) => Err(launch_error),
                Err(sync_error) => Err(anyhow!(
                    "attention preparation launch failed: {launch_error}; synchronization also failed: {sync_error}"
                )),
            };
        }
        context.synchronize()?;
        drop(normalized);
        drop(unrounded);
        Ok(Output { values, gate })
    }
}

fn validate_shape(rows: usize, heads: usize, width: usize, rotary_dim: usize) -> Result<()> {
    ensure!(
        (1..=2048).contains(&rows),
        "attention row count is out of range"
    );
    ensure!(
        (1..=128).contains(&heads),
        "attention head count is out of range"
    );
    ensure!(
        (2..=256).contains(&width),
        "attention width is out of range"
    );
    ensure!(
        (2..=width).contains(&rotary_dim) && rotary_dim.is_multiple_of(2),
        "rotary width must be even and within the attention width"
    );
    Ok(())
}

fn extents(
    rows: usize,
    heads: usize,
    width: usize,
    rotary_dim: usize,
    with_gate: bool,
) -> Result<Extents> {
    validate_shape(rows, heads, width, rotary_dim)?;
    let head_count = rows
        .checked_mul(heads)
        .context("attention head count overflows usize")?;
    let output_elements = head_count
        .checked_mul(width)
        .context("attention output element count overflows usize")?;
    let input_factor = if with_gate { 2 } else { 1 };
    let input_bytes = output_elements
        .checked_mul(input_factor)
        .and_then(|elements| elements.checked_mul(2))
        .context("attention input extent overflows usize")?;
    let table_bytes = rows
        .checked_mul(rotary_dim / 2)
        .and_then(|elements| elements.checked_mul(2))
        .context("attention RoPE table extent overflows usize")?;
    let output_bytes = output_elements
        .checked_mul(2)
        .context("attention output extent overflows usize")?;
    let diagnostic_bytes = output_elements
        .checked_mul(4)
        .context("attention diagnostic extent overflows usize")?;
    let grid_x = u32::try_from(head_count).context("attention grid does not fit u32")?;
    Ok(Extents {
        input_bytes,
        table_bytes,
        output_bytes,
        diagnostic_bytes,
        grid_x,
    })
}

fn validate_buffers(shape: [usize; 4], with_gate: bool, buffers: [usize; 3]) -> Result<Extents> {
    let [rows, heads, width, rotary_dim] = shape;
    let [input_bytes, cos_bytes, sin_bytes] = buffers;
    let extents = extents(rows, heads, width, rotary_dim, with_gate)?;
    ensure!(
        input_bytes == extents.input_bytes,
        "attention input extent mismatch"
    );
    ensure!(
        cos_bytes == extents.table_bytes && sin_bytes == extents.table_bytes,
        "attention RoPE table extent mismatch"
    );
    Ok(extents)
}

#[cfg(test)]
mod tests {
    use super::{extents, validate_buffers};

    #[test]
    fn computes_q_and_k_resident_extents() {
        let q = extents(17, 2, 256, 64, true).unwrap();
        assert_eq!(q.input_bytes, 34_816);
        assert_eq!(q.table_bytes, 1_088);
        assert_eq!(q.output_bytes, 17_408);
        assert_eq!(q.diagnostic_bytes, 34_816);
        assert_eq!(q.grid_x, 34);

        let k = extents(1, 4, 8, 4, false).unwrap();
        assert_eq!(k.input_bytes, 64);
        assert_eq!(k.table_bytes, 4);
        assert_eq!(k.output_bytes, 64);
        assert_eq!(k.grid_x, 4);

        let maximum = extents(2048, 128, 256, 256, true).unwrap();
        assert_eq!(maximum.input_bytes, 268_435_456);
        assert_eq!(maximum.table_bytes, 524_288);
        assert_eq!(maximum.output_bytes, 134_217_728);
        assert_eq!(maximum.diagnostic_bytes, 268_435_456);
        assert_eq!(maximum.grid_x, 262_144);
    }

    #[test]
    fn rejects_invalid_dimensions_and_buffer_extents() {
        for (rows, heads, width, rotary) in [
            (0, 1, 8, 4),
            (2049, 1, 8, 4),
            (1, 0, 8, 4),
            (1, 129, 8, 4),
            (1, 1, 1, 1),
            (1, 1, 257, 2),
            (1, 1, 8, 3),
            (1, 1, 8, 10),
        ] {
            assert!(extents(rows, heads, width, rotary, false).is_err());
        }
        assert!(validate_buffers([1, 1, 8, 4], true, [32, 8, 6]).is_err());
        assert!(validate_buffers([1, 1, 8, 4], false, [16, 8, 8]).is_err());
    }
}
