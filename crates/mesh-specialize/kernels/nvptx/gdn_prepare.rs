use core::arch::asm;

const QK_THREADS: u32 = 256;
const EPSILON: f32 = 1.0e-6_f32;
const SMALL_LOG1P_EXP: f32 = 0.0625_f32;
const SOFTPLUS_LINEAR_THRESHOLD: f32 = 20.0_f32;

#[inline(always)]
fn block_and_thread() -> (u32, u32) {
    let block: u32;
    let thread: u32;
    // SAFETY: Reads the calling thread's 1D block and thread coordinates without memory effects.
    unsafe {
        asm!(
            "mov.u32 {block}, %ctaid.x;",
            "mov.u32 {thread}, %tid.x;",
            block = out(reg32) block,
            thread = out(reg32) thread,
            options(nomem, nostack),
        )
    };
    (block, thread)
}

#[inline(always)]
fn qk_shared_bases() -> (u32, u32) {
    let q_base: u32;
    let k_base: u32;
    // SAFETY: Declares two CTA-local 256-element FP32 partial arrays.
    unsafe {
        asm!(
            ".shared .align 4 .b8 gdn_q_partials[1024];",
            ".shared .align 4 .b8 gdn_k_partials[1024];",
            "mov.u32 {q_base}, gdn_q_partials;",
            "mov.u32 {k_base}, gdn_k_partials;",
            q_base = out(reg32) q_base,
            k_base = out(reg32) k_base,
            options(nostack),
        )
    };
    (q_base, k_base)
}

#[inline(always)]
fn store_shared_f32(base: u32, index: u32, value: f32) {
    // SAFETY: The caller uses indices 0..256 in a 1024-byte shared array.
    let address = base + index * 4;
    unsafe {
        asm!(
            "st.shared.f32 [{address}], {value};",
            address = in(reg32) address,
            value = in(reg32) value,
            options(nostack),
        )
    };
}

#[inline(always)]
fn load_shared_f32(base: u32, index: u32) -> f32 {
    let value: f32;
    // SAFETY: The caller uses indices 0..256 in a 1024-byte shared array.
    let address = base + index * 4;
    unsafe {
        asm!(
            "ld.shared.f32 {value}, [{address}];",
            value = out(reg32) value,
            address = in(reg32) address,
            options(nostack),
        )
    };
    value
}

#[inline(always)]
fn block_barrier() {
    // SAFETY: Every thread in the block reaches each reduction barrier uniformly.
    unsafe { asm!("bar.sync 0;", options(nostack)) };
}

#[inline(always)]
fn fp32_multiply_rn(left: f32, right: f32) -> f32 {
    let product: f32;
    // SAFETY: This scalar FP32 operation has no memory or stack effects.
    unsafe {
        asm!(
            "mul.rn.f32 {product}, {left}, {right};",
            product = out(reg32) product,
            left = in(reg32) left,
            right = in(reg32) right,
            options(nomem, nostack),
        )
    };
    product
}

#[inline(always)]
fn fp32_add_rn(left: f32, right: f32) -> f32 {
    let sum: f32;
    // SAFETY: This scalar FP32 operation has no memory or stack effects.
    unsafe {
        asm!(
            "add.rn.f32 {sum}, {left}, {right};",
            sum = out(reg32) sum,
            left = in(reg32) left,
            right = in(reg32) right,
            options(nomem, nostack),
        )
    };
    sum
}

#[inline(always)]
fn fp32_divide_rn(numerator: f32, denominator: f32) -> f32 {
    let quotient: f32;
    // SAFETY: This scalar FP32 operation has no memory or stack effects.
    unsafe {
        asm!(
            "div.rn.f32 {quotient}, {numerator}, {denominator};",
            quotient = out(reg32) quotient,
            numerator = in(reg32) numerator,
            denominator = in(reg32) denominator,
            options(nomem, nostack),
        )
    };
    quotient
}

#[inline(always)]
fn fp32_sqrt_rn(value: f32) -> f32 {
    let root: f32;
    // SAFETY: This scalar FP32 operation has no memory or stack effects.
    unsafe {
        asm!(
            "sqrt.rn.f32 {root}, {value};",
            root = out(reg32) root,
            value = in(reg32) value,
            options(nomem, nostack),
        )
    };
    root
}

#[inline(always)]
fn fp32_exp2_approx(exponent: f32) -> f32 {
    let result: f32;
    // SAFETY: This scalar approximate exponential has no memory or stack effects.
    unsafe {
        asm!(
            "ex2.approx.f32 {result}, {exponent};",
            result = out(reg32) result,
            exponent = in(reg32) exponent,
            options(nomem, nostack),
        )
    };
    result
}

#[inline(always)]
fn fp32_log2_approx(value: f32) -> f32 {
    let result: f32;
    // SAFETY: This scalar approximate logarithm has no memory or stack effects.
    unsafe {
        asm!(
            "lg2.approx.f32 {result}, {value};",
            result = out(reg32) result,
            value = in(reg32) value,
            options(nomem, nostack),
        )
    };
    result
}

#[inline(always)]
fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline(always)]
fn encode_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0x7f80_0000 && mantissa != 0 {
        return 0x7fc0;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[inline(always)]
fn exp_from_argument(argument: f32) -> f32 {
    let exponent = fp32_multiply_rn(argument, core::f32::consts::LOG2_E);
    fp32_exp2_approx(exponent)
}

#[inline(always)]
fn exp_negative_absolute(value: f32) -> f32 {
    let absolute = f32::from_bits(value.to_bits() & 0x7fff_ffff);
    let negative_absolute = f32::from_bits(absolute.to_bits() | 0x8000_0000);
    exp_from_argument(negative_absolute)
}

#[inline(always)]
fn stable_sigmoid(value: f32) -> f32 {
    let exponential = exp_negative_absolute(value);
    let denominator = fp32_add_rn(1.0_f32, exponential);
    if value >= 0.0_f32 {
        fp32_divide_rn(1.0_f32, denominator)
    } else {
        fp32_divide_rn(exponential, denominator)
    }
}

#[inline(always)]
fn log1p_exp(exponential: f32) -> f32 {
    if exponential <= SMALL_LOG1P_EXP {
        let mut polynomial = fp32_multiply_rn(0.2_f32, exponential);
        polynomial = fp32_add_rn(polynomial, -0.25_f32);
        polynomial = fp32_multiply_rn(polynomial, exponential);
        polynomial = fp32_add_rn(polynomial, 0.333_333_34_f32);
        polynomial = fp32_multiply_rn(polynomial, exponential);
        polynomial = fp32_add_rn(polynomial, -0.5_f32);
        polynomial = fp32_multiply_rn(polynomial, exponential);
        polynomial = fp32_add_rn(polynomial, 1.0_f32);
        fp32_multiply_rn(polynomial, exponential)
    } else {
        let one_plus_exponential = fp32_add_rn(1.0_f32, exponential);
        fp32_multiply_rn(
            fp32_log2_approx(one_plus_exponential),
            core::f32::consts::LN_2,
        )
    }
}

#[inline(always)]
fn stable_softplus(value: f32) -> f32 {
    if value > SOFTPLUS_LINEAR_THRESHOLD {
        return value;
    }
    let positive = if value > 0.0_f32 { value } else { 0.0_f32 };
    let log1p = log1p_exp(exp_negative_absolute(value));
    fp32_add_rn(positive, log1p)
}

/// Normalize Q and K head vectors from time-major BF16 QKV rows.
///
/// Input rows contain contiguous Q heads, then K heads, then V heads, each with
/// `width` BF16 values. Q and K outputs contain the unrepeated `[rows, key_heads,
/// width]` vectors in FP32. Each vector uses its own sum of squares plus epsilon;
/// Q receives the additional `sqrt(width)` divisor specified by the GDN schedule.
///
/// # Safety
/// Launch `grid = [rows * key_heads, 1, 1]` and `block = [256, 1, 1]`. `rows` must
/// be 1..=2048, `key_heads` 1..=64, `value_heads` 1..=256 and divisible by
/// `key_heads`, and `width` must be a power of two in 1..=256. `qkv` must cover
/// `rows * (2 * key_heads + value_heads) * width` readable BF16 values; `q` and `k`
/// must each cover `rows * key_heads * width` writable `f32` values. All pointer
/// extents and index arithmetic must fit `usize`; pointers must be correctly aligned,
/// mutually disjoint, and live until kernel completion. The host must validate all
/// QKV values are finite.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn gdn_qk_norm(
    qkv: *const u16,
    q: *mut f32,
    k: *mut f32,
    _rows: u32,
    key_heads: u32,
    value_heads: u32,
    width: u32,
) {
    let (q_shared, k_shared) = qk_shared_bases();
    let (block, thread) = block_and_thread();
    let key_heads_usize = key_heads as usize;
    let value_heads_usize = value_heads as usize;
    let width_usize = width as usize;
    let block_usize = block as usize;
    let row = block_usize / key_heads_usize;
    let head = block_usize % key_heads_usize;
    let input_row_width = (2 * key_heads_usize + value_heads_usize) * width_usize;
    let q_start = row * input_row_width + head * width_usize;
    let k_start = row * input_row_width + (key_heads_usize + head) * width_usize;

    let mut q_value = 0.0_f32;
    let mut k_value = 0.0_f32;
    if thread < width {
        let input_index = thread as usize;
        // SAFETY: Grid coordinates and dimensions select valid Q and K vectors.
        unsafe {
            q_value = decode_bf16(*qkv.add(q_start + input_index));
            k_value = decode_bf16(*qkv.add(k_start + input_index));
        }
    }
    let q_partial = fp32_multiply_rn(q_value, q_value);
    let k_partial = fp32_multiply_rn(k_value, k_value);
    store_shared_f32(q_shared, thread, q_partial);
    store_shared_f32(k_shared, thread, k_partial);
    block_barrier();

    let mut stride = QK_THREADS / 2;
    while stride > 0 {
        if thread < stride {
            let q_left = load_shared_f32(q_shared, thread);
            let q_right = load_shared_f32(q_shared, thread + stride);
            let k_left = load_shared_f32(k_shared, thread);
            let k_right = load_shared_f32(k_shared, thread + stride);
            store_shared_f32(q_shared, thread, fp32_add_rn(q_left, q_right));
            store_shared_f32(k_shared, thread, fp32_add_rn(k_left, k_right));
        }
        block_barrier();
        stride /= 2;
    }

    let q_sum = load_shared_f32(q_shared, 0);
    let k_sum = load_shared_f32(k_shared, 0);
    let q_root = fp32_sqrt_rn(fp32_add_rn(q_sum, EPSILON));
    let k_root = fp32_sqrt_rn(fp32_add_rn(k_sum, EPSILON));
    let q_inverse = fp32_divide_rn(1.0_f32, q_root);
    let k_inverse = fp32_divide_rn(1.0_f32, k_root);
    let width_root = fp32_sqrt_rn(width as f32);

    if thread < width {
        let output_index = block_usize * width_usize + thread as usize;
        let normalized_q = fp32_divide_rn(fp32_multiply_rn(q_value, q_inverse), width_root);
        let normalized_k = fp32_multiply_rn(k_value, k_inverse);
        // SAFETY: Each block owns one Q/K head and each active thread one width element.
        unsafe {
            q.add(output_index).write(normalized_q);
            k.add(output_index).write(normalized_k);
        }
    }
}

/// Compute stable GDN beta, softplus gate, and exponential decay values.
///
/// `a`, `b`, and the outputs are time-major `[rows, heads]`; `a_log` and `dt_bias`
/// contain one BF16 parameter per head. `beta` is the BF16-rounded stable sigmoid of
/// `b`. The gate is `g = -exp(a_log) * softplus(a + dt_bias)`, and `decay = exp(g)`.
///
/// # Safety
/// Launch `grid = [ceil(rows * heads / 256), 1, 1]` and `block = [256, 1, 1]`, with
/// nonzero `rows` and `heads`; the flattened element count and grid size must fit the
/// device address space, `u32`, and hardware limits. `a` and `b` must each cover
/// `rows * heads` readable BF16 values, `a_log` and `dt_bias` each `heads` readable
/// BF16 values, `beta` `rows * heads` writable BF16 values, and `g` and `decay` each
/// `rows * heads` writable `f32` values. All pointer extents and index arithmetic
/// must fit `usize`; pointers must be correctly aligned, mutually disjoint, and live
/// until kernel completion. The host must validate all inputs are finite and each
/// `a_log` lies in [-80, 80], and reject non-finite or overflowing outputs.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn gdn_gates(
    a: *const u16,
    b: *const u16,
    a_log: *const u16,
    dt_bias: *const u16,
    beta: *mut u16,
    g: *mut f32,
    decay: *mut f32,
    rows: u32,
    heads: u32,
) {
    let (block, thread) = block_and_thread();
    let heads_usize = heads as usize;
    let element_count = rows as usize * heads_usize;
    let flat_index = block as usize * QK_THREADS as usize + thread as usize;
    if flat_index >= element_count {
        return;
    }

    let head = flat_index % heads_usize;
    // SAFETY: The host contract provides all time-major elements and per-head parameters.
    let (a_value, b_value, a_log_value, dt_bias_value) = unsafe {
        (
            decode_bf16(*a.add(flat_index)),
            decode_bf16(*b.add(flat_index)),
            decode_bf16(*a_log.add(head)),
            decode_bf16(*dt_bias.add(head)),
        )
    };
    let beta_value = encode_bf16_rne(stable_sigmoid(b_value));
    let softplus_input = fp32_add_rn(a_value, dt_bias_value);
    let softplus = stable_softplus(softplus_input);
    let negative_exp_a_log = f32::from_bits(exp_from_argument(a_log_value).to_bits() ^ 0x8000_0000);
    let gate = fp32_multiply_rn(negative_exp_a_log, softplus);
    let decay_value = exp_from_argument(gate);
    // SAFETY: Each flattened output thread writes one distinct beta/g/decay element.
    unsafe {
        beta.add(flat_index).write(beta_value);
        g.add(flat_index).write(gate);
        decay.add(flat_index).write(decay_value);
    }
}
