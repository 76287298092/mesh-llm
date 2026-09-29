use anyhow::{Context as _, Result, anyhow, ensure};
use std::ffi::c_void;

use super::{
    driver::{Buffer, Context, Function, Module},
    resident_state::ResidentState,
};

const MAX_ROWS: usize = 2_048;
const MAX_HEADS: usize = 128;
const MAX_WIDTH: usize = 256;
const MAX_CAPACITY: usize = 262_144;
const THREADS: u32 = 256;

pub(super) struct Shape {
    pub(super) rows: usize,
    pub(super) query_heads: usize,
    pub(super) kv_heads: usize,
    pub(super) width: usize,
    pub(super) past: usize,
    pub(super) capacity: usize,
}

pub(super) struct Input<'a, 'ctx> {
    pub(super) q: &'a Buffer<'ctx>,
    pub(super) k: &'a Buffer<'ctx>,
    pub(super) v: &'a Buffer<'ctx>,
}

struct Extents {
    query_bytes: usize,
    key_value_input_bytes: usize,
    cache_bytes: usize,
    output_bytes: usize,
    unrounded_bytes: usize,
    append_grid: [u32; 3],
    attention_grid: [u32; 3],
    append_dimensions: [u32; 5],
    attention_dimensions: [u32; 6],
    scale: f32,
}

/// Append this chunk to persistent K/V state and run causal attention for its queries.
///
/// The caller owns the prefix cursor and advances it only after the enclosing block succeeds.
pub(super) fn run<'a>(
    context: &'a Context,
    module: &Module<'_>,
    input: Input<'_, '_>,
    state: &mut ResidentState<'_>,
    state_prefix: &str,
    shape: &Shape,
) -> Result<Buffer<'a>> {
    ensure!(
        input.q.belongs_to(context)
            && input.k.belongs_to(context)
            && input.v.belongs_to(context)
            && state.belongs_to(context)
            && module.belongs_to(context),
        "attention inputs, state, and PTX module must belong to the same CUDA context"
    );
    let extents = validate_shape(shape)?;
    validate_input_lengths(input.q.len(), input.k.len(), input.v.len(), &extents)?;

    let key_state_name = format!("{state_prefix}.attention.k");
    let value_state_name = format!("{state_prefix}.attention.v");
    let key_state = state.pointer(&key_state_name, extents.cache_bytes)?;
    let value_state = state.pointer(&value_state_name, extents.cache_bytes)?;
    let append = module.function("attention_kv_append")?;
    let profile = crate::kernels::attention_profile::current()?;
    let warp = profile.uses_warp(shape.rows);
    if warp {
        crate::kernels::attention_warp_plan::validate(extents.attention_dimensions)?;
    }
    // Admit geometry and prepare every allocation/handle before mutating KV state.
    // Keep this owner, raw output and BF16 output alive through the final drain.
    let split = profile
        .uses_split(shape.rows)
        .then(|| super::resident_attention_split::Prepared::new(context, module, shape))
        .transpose()?;
    let staged = profile
        .uses_staged(shape.rows)
        .then(|| {
            crate::kernels::attention_staged_plan::CoefficientSchedule::current().and_then(
                |schedule| {
                    super::resident_attention_staged::Prepared::new(
                        context, module, shape, schedule,
                    )
                },
            )
        })
        .transpose()?;
    let attention = module.function(profile.kernel_for_rows(shape.rows))?;
    let output = Buffer::new(context, extents.output_bytes)?;
    let unrounded = Buffer::new(context, extents.unrounded_bytes)?;

    let append_launch = AppendLaunch {
        pointers: [input.k.pointer(), input.v.pointer(), key_state, value_state],
        dimensions: extents.append_dimensions,
        grid: extents.append_grid,
    };
    let attention_launch = AttentionLaunch {
        pointers: [
            input.q.pointer(),
            key_state,
            value_state,
            output.pointer(),
            unrounded.pointer(),
        ],
        dimensions: extents.attention_dimensions,
        scale: extents.scale,
        grid: if warp {
            crate::kernels::attention_warp_plan::GRID
        } else {
            extents.attention_grid
        },
        block: if warp {
            crate::kernels::attention_warp_plan::BLOCK
        } else {
            [THREADS, 1, 1]
        },
    };
    if let Err(error) = launch_append(&append, append_launch) {
        return Err(synchronize_after_failed_launch(
            context,
            "attention KV append",
            error,
        ));
    }
    let launched = if let Some(staged) = &staged {
        // SAFETY: Same-context initialized inputs and output extents were checked
        // above. Staged scratch and every output remain live through the common
        // success/error drain, including failure after only one stage was queued.
        unsafe { staged.launch(attention_launch.pointers) }
    } else {
        match &split {
            // SAFETY: Append precedes this launch; all owners survive the drain below.
            Some(split) => unsafe { split.launch(attention_launch.pointers) },
            None => launch_attention(&attention, attention_launch),
        }
    };
    if let Err(error) = launched {
        return Err(synchronize_after_failed_launch(
            context,
            "causal attention",
            error,
        ));
    }
    context.synchronize()?;
    if profile.is_audit() {
        super::attention_audit::compare(
            context,
            module,
            super::attention_audit::Case {
                q: input.q,
                state,
                prefix: state_prefix,
                output: &output,
                raw: &unrounded,
                shape,
            },
        )?;
    }
    Ok(output)
}

fn validate_shape(shape: &Shape) -> Result<Extents> {
    ensure!(
        (1..=MAX_ROWS).contains(&shape.rows)
            && (1..=MAX_HEADS).contains(&shape.query_heads)
            && (1..=MAX_HEADS).contains(&shape.kv_heads)
            && (2..=MAX_WIDTH).contains(&shape.width),
        "attention core rows, heads, or width are outside supported bounds"
    );
    ensure!(
        shape.query_heads.is_multiple_of(shape.kv_heads),
        "attention query-head count must be divisible by KV-head count"
    );
    ensure!(
        (1..=MAX_CAPACITY).contains(&shape.capacity),
        "attention cache capacity must be in 1..={MAX_CAPACITY}"
    );
    let initialized_end = shape
        .past
        .checked_add(shape.rows)
        .context("attention prefix plus row count overflows usize")?;
    ensure!(
        initialized_end <= shape.capacity,
        "attention append range exceeds cache capacity"
    );

    let query_values = checked_product(
        checked_product(shape.rows, shape.query_heads, "attention query heads")?,
        shape.width,
        "attention query values",
    )?;
    let key_value_values = checked_product(
        checked_product(shape.rows, shape.kv_heads, "attention KV heads")?,
        shape.width,
        "attention KV input values",
    )?;
    let cache_values = checked_product(
        checked_product(shape.capacity, shape.kv_heads, "attention cache heads")?,
        shape.width,
        "attention cache values",
    )?;
    let append_blocks = key_value_values.div_ceil(THREADS as usize);
    let attention_blocks = checked_product(shape.rows, shape.query_heads, "attention CTAs")?;
    Ok(Extents {
        query_bytes: checked_product(query_values, 2, "attention BF16 query bytes")?,
        key_value_input_bytes: checked_product(key_value_values, 2, "attention BF16 KV bytes")?,
        cache_bytes: checked_product(cache_values, 2, "attention BF16 cache bytes")?,
        output_bytes: checked_product(query_values, 2, "attention BF16 output bytes")?,
        unrounded_bytes: checked_product(query_values, 4, "attention FP32 output bytes")?,
        append_grid: [u32::try_from(append_blocks)?, 1, 1],
        attention_grid: [u32::try_from(attention_blocks)?, 1, 1],
        append_dimensions: [
            u32::try_from(shape.rows)?,
            u32::try_from(shape.kv_heads)?,
            u32::try_from(shape.width)?,
            u32::try_from(shape.past)?,
            u32::try_from(shape.capacity)?,
        ],
        attention_dimensions: [
            u32::try_from(shape.rows)?,
            u32::try_from(shape.query_heads)?,
            u32::try_from(shape.kv_heads)?,
            u32::try_from(shape.width)?,
            u32::try_from(shape.past)?,
            u32::try_from(shape.capacity)?,
        ],
        scale: 1.0_f32 / (shape.width as f32).sqrt(),
    })
}

fn validate_input_lengths(
    query_bytes: usize,
    key_bytes: usize,
    value_bytes: usize,
    extents: &Extents,
) -> Result<()> {
    ensure!(
        query_bytes == extents.query_bytes
            && key_bytes == extents.key_value_input_bytes
            && value_bytes == extents.key_value_input_bytes,
        "attention query/K/V buffer lengths do not match the requested chunk shape"
    );
    Ok(())
}

struct AppendLaunch {
    pointers: [u64; 4],
    dimensions: [u32; 5],
    grid: [u32; 3],
}

fn launch_append(function: &Function<'_, '_>, launch: AppendLaunch) -> Result<()> {
    let mut pointers = launch.pointers;
    let mut dimensions = launch.dimensions;
    let mut arguments: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|pointer| (pointer as *mut u64).cast())
        .collect();
    arguments.extend(
        dimensions
            .iter_mut()
            .map(|dimension| (dimension as *mut u32).cast()),
    );
    // SAFETY: Checked source and state extents match the four-pointer/five-u32 ABI. Input pointers
    // name the current chunk directly; state and input buffers remain live through final sync.
    unsafe { function.launch(launch.grid, [THREADS, 1, 1], 0, &mut arguments) }
}

struct AttentionLaunch {
    pointers: [u64; 5],
    dimensions: [u32; 6],
    scale: f32,
    grid: [u32; 3],
    block: [u32; 3],
}

fn launch_attention(function: &Function<'_, '_>, launch: AttentionLaunch) -> Result<()> {
    let mut pointers = launch.pointers;
    let mut dimensions = launch.dimensions;
    let mut scale = launch.scale;
    let mut arguments: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|pointer| (pointer as *mut u64).cast())
        .collect();
    arguments.extend(
        dimensions
            .iter_mut()
            .map(|dimension| (dimension as *mut u32).cast()),
    );
    arguments.push((&mut scale as *mut f32).cast());
    // SAFETY: Checked query/cache/output extents match the five-pointer/six-u32/scale ABI. The
    // output allocations and resident cache live until the synchronization after this launch.
    // Warp geometry is admitted before KV mutation; other profiles keep the control geometry.
    unsafe { function.launch(launch.grid, launch.block, 0, &mut arguments) }
}

fn synchronize_after_failed_launch(
    context: &Context,
    operation: &str,
    error: anyhow::Error,
) -> anyhow::Error {
    let context_message = match context.synchronize() {
        Ok(()) => format!("{operation} launch failed"),
        Err(sync_error) => {
            format!("{operation} launch failed; CUDA synchronization also failed: {sync_error:#}")
        }
    };
    error.context(context_message)
}

fn checked_product(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow!("{label} extent overflows usize"))
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_CAPACITY, MAX_HEADS, MAX_ROWS, MAX_WIDTH, Shape, checked_product,
        validate_input_lengths, validate_shape,
    };

    fn shape() -> Shape {
        Shape {
            rows: 17,
            query_heads: 24,
            kv_heads: 4,
            width: 256,
            past: 11,
            capacity: 64,
        }
    }

    #[test]
    fn validates_gqa_bounds_extents_and_launches() {
        let extents = validate_shape(&shape()).unwrap();
        assert_eq!(extents.query_bytes, 17 * 24 * 256 * 2);
        assert_eq!(extents.key_value_input_bytes, 17 * 4 * 256 * 2);
        assert_eq!(extents.cache_bytes, 64 * 4 * 256 * 2);
        assert_eq!(extents.output_bytes, extents.query_bytes);
        assert_eq!(extents.unrounded_bytes, 17 * 24 * 256 * 4);
        assert_eq!(extents.append_grid, [68, 1, 1]);
        assert_eq!(extents.attention_grid, [408, 1, 1]);
        assert_eq!(extents.append_dimensions, [17, 4, 256, 11, 64]);
        assert_eq!(extents.attention_dimensions, [17, 24, 4, 256, 11, 64]);
        assert_eq!(extents.scale, 1.0_f32 / 16.0);
    }

    #[test]
    fn rejects_invalid_heads_dimensions_and_capacity() {
        let mut invalid = shape();
        invalid.query_heads = MAX_HEADS + 1;
        assert!(validate_shape(&invalid).is_err());
        invalid = shape();
        invalid.kv_heads = 0;
        assert!(validate_shape(&invalid).is_err());
        invalid = shape();
        invalid.query_heads = 10;
        assert!(validate_shape(&invalid).is_err());
        invalid = shape();
        invalid.width = MAX_WIDTH + 1;
        assert!(validate_shape(&invalid).is_err());
        invalid = shape();
        invalid.rows = MAX_ROWS + 1;
        assert!(validate_shape(&invalid).is_err());
        invalid = shape();
        invalid.capacity = MAX_CAPACITY + 1;
        assert!(validate_shape(&invalid).is_err());
        invalid = shape();
        invalid.capacity = 0;
        assert!(validate_shape(&invalid).is_err());
    }

    #[test]
    fn rejects_cache_overrun_and_prefix_overflow() {
        let mut invalid = shape();
        invalid.past = invalid.capacity;
        assert!(validate_shape(&invalid).is_err());
        invalid = shape();
        invalid.past = usize::MAX;
        assert!(validate_shape(&invalid).is_err());
    }

    #[test]
    fn validates_exact_chunk_buffer_lengths() {
        let extents = validate_shape(&shape()).unwrap();
        assert!(
            validate_input_lengths(
                extents.query_bytes,
                extents.key_value_input_bytes,
                extents.key_value_input_bytes,
                &extents,
            )
            .is_ok()
        );
        assert!(
            validate_input_lengths(
                extents.query_bytes - 2,
                extents.key_value_input_bytes,
                extents.key_value_input_bytes,
                &extents,
            )
            .is_err()
        );
        assert!(
            validate_input_lengths(
                extents.query_bytes,
                extents.key_value_input_bytes,
                extents.key_value_input_bytes - 2,
                &extents,
            )
            .is_err()
        );
    }

    #[test]
    fn extent_products_use_checked_arithmetic() {
        assert!(checked_product(usize::MAX, 2, "test extent").is_err());
    }
}
