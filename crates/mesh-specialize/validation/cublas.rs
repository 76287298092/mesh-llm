//! Dynamically loaded cuBLAS operations for standalone validation binaries.

use crate::driver::Context;
use anyhow::{Result, anyhow, bail};
use libloading::Library;
use std::marker::PhantomData;
use std::path::Path;
use std::ptr;
use std::rc::Rc;

type Status = i32;
type Handle = *mut std::ffi::c_void;

const SUCCESS: Status = 0;
const MATH_PEDANTIC: i32 = 2;
const POINTER_MODE_HOST: i32 = 0;
const OP_N: i32 = 0;
const SIDE_LEFT: i32 = 0;

type CreateFn = unsafe extern "C" fn(*mut Handle) -> Status;
type DestroyFn = unsafe extern "C" fn(Handle) -> Status;
type GetVersionFn = unsafe extern "C" fn(Handle, *mut i32) -> Status;
type SetMathModeFn = unsafe extern "C" fn(Handle, i32) -> Status;
type SetPointerModeFn = unsafe extern "C" fn(Handle, i32) -> Status;
type SgemmFn = unsafe extern "C" fn(
    Handle,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f32,
    *const f32,
    i32,
    *const f32,
    i32,
    *const f32,
    *mut f32,
    i32,
) -> Status;
type Snrm2Fn = unsafe extern "C" fn(Handle, i32, *const f32, i32, *mut f32) -> Status;
type SscalFn = unsafe extern "C" fn(Handle, i32, *const f32, *mut f32, i32) -> Status;
type SdgmmFn = unsafe extern "C" fn(
    Handle,
    i32,
    i32,
    i32,
    *const f32,
    i32,
    *const f32,
    i32,
    *mut f32,
    i32,
) -> Status;

struct Api {
    create: CreateFn,
    destroy: DestroyFn,
    get_version: GetVersionFn,
    set_math_mode: SetMathModeFn,
    set_pointer_mode: SetPointerModeFn,
    sgemm: SgemmFn,
    snrm2: Snrm2Fn,
    sscal: SscalFn,
    sdgmm: SdgmmFn,
    _library: Library,
}

impl Api {
    fn load(path: &Path) -> Result<Self> {
        // SAFETY: The Api retains the library as long as any copied function pointer can be used.
        let library = unsafe { Library::new(path) }
            .map_err(|error| anyhow!("failed to load cuBLAS {}: {error}", path.display()))?;
        Ok(Self {
            create: load_symbol(&library, b"cublasCreate_v2\0")?,
            destroy: load_symbol(&library, b"cublasDestroy_v2\0")?,
            get_version: load_symbol(&library, b"cublasGetVersion_v2\0")?,
            set_math_mode: load_symbol(&library, b"cublasSetMathMode\0")?,
            set_pointer_mode: load_symbol(&library, b"cublasSetPointerMode_v2\0")?,
            sgemm: load_symbol(&library, b"cublasSgemm_v2\0")?,
            snrm2: load_symbol(&library, b"cublasSnrm2_v2\0")?,
            sscal: load_symbol(&library, b"cublasSscal_v2\0")?,
            sdgmm: load_symbol(&library, b"cublasSdgmm\0")?,
            _library: library,
        })
    }
}

fn load_symbol<T: Copy>(library: &Library, name: &'static [u8]) -> Result<T> {
    // SAFETY: name is a NUL-terminated cuBLAS symbol and each call uses its declared ABI type;
    // the containing Api retains library for the lifetime of the copied pointer.
    let symbol = unsafe { library.get::<T>(name) }.map_err(|error| {
        let symbol_name = String::from_utf8_lossy(&name[..name.len().saturating_sub(1)]);
        anyhow!("failed to resolve cuBLAS symbol {symbol_name}: {error}")
    })?;
    Ok(*symbol)
}

fn check_status(status: Status, operation: &str) -> Result<()> {
    if status == SUCCESS {
        Ok(())
    } else {
        Err(anyhow!(
            "{operation} failed with cuBLAS status code {status}"
        ))
    }
}

fn cleanup_warning(operation: &str, status: Status) {
    if status != SUCCESS {
        tracing::warn!(
            operation,
            status_code = status,
            "cuBLAS cleanup operation failed"
        );
    }
}

/// A cuBLAS handle bound to and borrowed from one CUDA context.
pub(super) struct Blas<'ctx> {
    context: &'ctx Context,
    handle: Handle,
    api: Api,
    _thread_bound: PhantomData<Rc<()>>,
}

impl<'ctx> Blas<'ctx> {
    /// Load cuBLAS from the requested library path, create a handle, and set host modes.
    pub(super) fn new(context: &'ctx Context, library_path: &Path) -> Result<Self> {
        let api = Api::load(library_path)?;
        let _current_context = context.activate()?;
        let mut handle = ptr::null_mut();
        // SAFETY: handle is writable and the borrowed CUDA context is current for creation.
        let status = unsafe { (api.create)(&mut handle) };
        if status != SUCCESS {
            return Err(anyhow!(
                "cublasCreate_v2 failed with cuBLAS status code {status}"
            ));
        }
        if handle.is_null() {
            bail!("cublasCreate_v2 succeeded but returned a null handle");
        }

        let blas = Self {
            context,
            handle,
            api,
            _thread_bound: PhantomData,
        };
        if let Err(error) = blas.set_modes() {
            drop(blas);
            return Err(error);
        }
        Ok(blas)
    }

    /// Return the version integer reported by the loaded cuBLAS library.
    pub(super) fn version(&self) -> Result<i32> {
        let _current_context = self.context.activate()?;
        let mut version = 0;
        // SAFETY: The handle belongs to this current context and version is writable output.
        check_status(
            unsafe { (self.api.get_version)(self.handle, &mut version) },
            "cublasGetVersion_v2",
        )?;
        Ok(version)
    }

    /// Compute row-major M×K times K×N, writing a row-major M×N result.
    ///
    /// # Safety
    /// a, b, and out must be aligned non-null device pointers accessible by the borrowed context.
    /// a must cover m*k floats, b k*n floats, and out m*n writable floats; output must not overlap
    /// either input. Keep all allocations live until the operation finishes.
    pub(super) unsafe fn gemm(
        &self,
        m: i32,
        n: i32,
        k: i32,
        a: u64,
        b: u64,
        out: u64,
    ) -> Result<()> {
        if m <= 0 || n <= 0 || k <= 0 {
            bail!("cuBLAS GEMM dimensions must be positive, got m={m}, n={n}, k={k}");
        }
        ensure_device_pointers(&[a, b, out])?;
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        let _current_context = self.context.activate()?;
        // SAFETY: The caller guarantees extents, alignment, non-aliasing, and allocation lifetime.
        check_status(
            unsafe {
                (self.api.sgemm)(
                    self.handle,
                    OP_N,
                    OP_N,
                    n,
                    m,
                    k,
                    &alpha,
                    device_const(b),
                    n,
                    device_const(a),
                    k,
                    &beta,
                    device_mut(out),
                    n,
                )
            },
            "cublasSgemm_v2",
        )
    }

    /// Compute the Euclidean norm of count device floats and return it to the host.
    ///
    /// # Safety
    /// input must be an aligned non-null device pointer to at least count readable floats
    /// accessible by this context, and its allocation must remain live until the operation finishes.
    pub(super) unsafe fn norm(&self, count: i32, input: u64) -> Result<f32> {
        if count <= 0 {
            bail!("cuBLAS norm count must be positive, got {count}");
        }
        ensure_device_pointers(&[input])?;
        let mut result = 0.0_f32;
        let _current_context = self.context.activate()?;
        // SAFETY: The caller guarantees input extent, alignment, context access, and lifetime;
        // increment one reads contiguous floats and host pointer mode writes to result.
        check_status(
            unsafe { (self.api.snrm2)(self.handle, count, device_const(input), 1, &mut result) },
            "cublasSnrm2_v2",
        )?;
        Ok(result)
    }

    /// Multiply each row-major matrix row by the corresponding element of weights.
    ///
    /// # Safety
    /// input and out must be aligned non-null device pointers to rows*width floats, and weights to
    /// width floats, all accessible by this context. out must be writable and disjoint from both
    /// inputs. Keep allocations live until the operation finishes.
    pub(super) unsafe fn weight(
        &self,
        rows: i32,
        width: i32,
        input: u64,
        weights: u64,
        out: u64,
    ) -> Result<()> {
        if rows <= 0 || width <= 0 {
            bail!("cuBLAS weight dimensions must be positive, got rows={rows}, width={width}");
        }
        ensure_device_pointers(&[input, weights, out])?;
        let _current_context = self.context.activate()?;
        // SAFETY: The caller guarantees extents, alignment, disjoint output, and allocation lifetime.
        check_status(
            unsafe {
                (self.api.sdgmm)(
                    self.handle,
                    SIDE_LEFT,
                    width,
                    rows,
                    device_const(input),
                    width,
                    device_const(weights),
                    1,
                    device_mut(out),
                    width,
                )
            },
            "cublasSdgmm",
        )
    }

    /// Scale count contiguous device floats by factor in place.
    ///
    /// # Safety
    /// output must be an aligned non-null device pointer to at least count writable floats
    /// accessible by this context, and its allocation must remain live until the operation finishes.
    pub(super) unsafe fn scale(&self, count: i32, factor: f32, output: u64) -> Result<()> {
        if count <= 0 {
            bail!("cuBLAS scale count must be positive, got {count}");
        }
        if !factor.is_finite() {
            bail!("cuBLAS scale factor must be finite, got {factor}");
        }
        ensure_device_pointers(&[output])?;
        let _current_context = self.context.activate()?;
        // SAFETY: The caller guarantees output extent, alignment, context access, and lifetime;
        // host pointer mode makes factor the scalar value read by cuBLAS.
        check_status(
            unsafe { (self.api.sscal)(self.handle, count, &factor, device_mut(output), 1) },
            "cublasSscal_v2",
        )
    }

    fn set_modes(&self) -> Result<()> {
        let _current_context = self.context.activate()?;
        // SAFETY: The handle is live and the borrowed CUDA context is current.
        check_status(
            unsafe { (self.api.set_math_mode)(self.handle, MATH_PEDANTIC) },
            "cublasSetMathMode",
        )?;
        // SAFETY: The handle is live and the borrowed CUDA context is current.
        check_status(
            unsafe { (self.api.set_pointer_mode)(self.handle, POINTER_MODE_HOST) },
            "cublasSetPointerMode_v2",
        )
    }
}

impl Drop for Blas<'_> {
    fn drop(&mut self) {
        match self.context.activate() {
            Ok(_current_context) => {
                // SAFETY: This wrapper uniquely owns the handle; its context and loaded library live.
                cleanup_warning("cublasDestroy_v2", unsafe {
                    (self.api.destroy)(self.handle)
                });
            }
            Err(error) => {
                tracing::warn!(error = %error, "failed to activate CUDA context before destroying cuBLAS handle")
            }
        }
    }
}

fn ensure_device_pointers(pointers: &[u64]) -> Result<()> {
    if pointers.contains(&0) {
        bail!("cuBLAS operation received a null device pointer");
    }
    Ok(())
}

fn device_const(pointer: u64) -> *const f32 {
    ptr::without_provenance(pointer as usize)
}

fn device_mut(pointer: u64) -> *mut f32 {
    ptr::without_provenance_mut(pointer as usize)
}
