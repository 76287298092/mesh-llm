//! CUDA Driver API ownership for the specialized runtime.

use anyhow::{Result, anyhow, bail};
use libloading::Library;
use serde::Serialize;
use std::ffi::{CString, c_char, c_int, c_uint, c_void};
use std::marker::PhantomData;
use std::ptr;
use std::rc::Rc;
type CuResult = c_int;
type CuDevice = c_int;
type CuDevicePtr = u64;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuEvent = *mut c_void;
type CuStream = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;
const DEVICE_NAME_BUFFER_BYTES: usize = 256;
const JIT_LOG_BUFFER_BYTES: usize = 16 * 1024;
const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 0;
const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: c_int = 1;
const CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: c_int = 3;
const CU_FUNC_ATTRIBUTE_NUM_REGS: c_int = 4;
const CU_JIT_INFO_LOG_BUFFER: c_int = 3;
const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_int = 4;
const CU_JIT_ERROR_LOG_BUFFER: c_int = 5;
const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;
const CU_JIT_LOG_VERBOSE: c_int = 12;

type CuInitFn = unsafe extern "C" fn(c_uint) -> CuResult;
type CuDriverGetVersionFn = unsafe extern "C" fn(*mut c_int) -> CuResult;
type CuDeviceGetFn = unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult;
type CuDeviceGetNameFn = unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult;
#[repr(C)]
struct CuUuid {
    bytes: [u8; 16],
}
type CuDeviceGetUuidV2Fn = unsafe extern "C" fn(*mut CuUuid, CuDevice) -> CuResult;
type CuDeviceComputeCapabilityFn =
    unsafe extern "C" fn(*mut c_int, *mut c_int, CuDevice) -> CuResult;
type CuDeviceTotalMemV2Fn = unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult;
type CuCtxCreateV2Fn = unsafe extern "C" fn(*mut CuContext, c_uint, CuDevice) -> CuResult;
type CuCtxDestroyV2Fn = unsafe extern "C" fn(CuContext) -> CuResult;
type CuCtxPushCurrentV2Fn = unsafe extern "C" fn(CuContext) -> CuResult;
type CuCtxPopCurrentV2Fn = unsafe extern "C" fn(*mut CuContext) -> CuResult;
type CuCtxSynchronizeFn = unsafe extern "C" fn() -> CuResult;
type CuMemGetInfoV2Fn = unsafe extern "C" fn(*mut usize, *mut usize) -> CuResult;
type CuMemAllocV2Fn = unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult;
type CuMemFreeV2Fn = unsafe extern "C" fn(CuDevicePtr) -> CuResult;
type CuMemcpyHtoDV2Fn = unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult;
type CuMemcpyDtoHV2Fn = unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult;
type CuModuleLoadDataExFn = unsafe extern "C" fn(
    *mut CuModule,
    *const c_void,
    c_uint,
    *mut c_int,
    *mut *mut c_void,
) -> CuResult;
type CuModuleUnloadFn = unsafe extern "C" fn(CuModule) -> CuResult;
type CuModuleGetFunctionFn =
    unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
type CuLaunchKernelFn = unsafe extern "C" fn(
    CuFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    CuStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;
type CuFuncGetAttributeFn = unsafe extern "C" fn(*mut c_int, c_int, CuFunction) -> CuResult;
type CuEventCreateFn = unsafe extern "C" fn(*mut CuEvent, c_uint) -> CuResult;
type CuEventRecordFn = unsafe extern "C" fn(CuEvent, CuStream) -> CuResult;
type CuEventSynchronizeFn = unsafe extern "C" fn(CuEvent) -> CuResult;
type CuEventElapsedTimeFn = unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> CuResult;
type CuEventDestroyV2Fn = unsafe extern "C" fn(CuEvent) -> CuResult;
struct Api {
    cu_init: CuInitFn,
    cu_driver_get_version: CuDriverGetVersionFn,
    cu_device_get: CuDeviceGetFn,
    cu_device_get_name: CuDeviceGetNameFn,
    cu_device_get_uuid_v2: CuDeviceGetUuidV2Fn,
    cu_device_compute_capability: CuDeviceComputeCapabilityFn,
    cu_device_total_mem_v2: CuDeviceTotalMemV2Fn,
    cu_ctx_create_v2: CuCtxCreateV2Fn,
    cu_ctx_destroy_v2: CuCtxDestroyV2Fn,
    cu_ctx_push_current_v2: CuCtxPushCurrentV2Fn,
    cu_ctx_pop_current_v2: CuCtxPopCurrentV2Fn,
    cu_ctx_synchronize: CuCtxSynchronizeFn,
    cu_mem_get_info_v2: CuMemGetInfoV2Fn,
    cu_mem_alloc_v2: CuMemAllocV2Fn,
    cu_mem_free_v2: CuMemFreeV2Fn,
    cu_memcpy_htod_v2: CuMemcpyHtoDV2Fn,
    cu_memcpy_dtoh_v2: CuMemcpyDtoHV2Fn,
    cu_module_load_data_ex: CuModuleLoadDataExFn,
    cu_module_unload: CuModuleUnloadFn,
    cu_module_get_function: CuModuleGetFunctionFn,
    cu_launch_kernel: CuLaunchKernelFn,
    cu_func_get_attribute: CuFuncGetAttributeFn,
    cu_event_create: CuEventCreateFn,
    cu_event_record: CuEventRecordFn,
    cu_event_synchronize: CuEventSynchronizeFn,
    cu_event_elapsed_time: CuEventElapsedTimeFn,
    cu_event_destroy_v2: CuEventDestroyV2Fn,
    // Keep the library after the copied pointers so it outlives every possible API use.
    _library: Library,
}

impl Api {
    #[cfg(target_os = "linux")]
    fn load() -> Result<Self> {
        // SAFETY: The loaded library is retained in `Api`; each resolved symbol below is assigned
        // only to the CUDA Driver ABI function type declared for that symbol.
        let library = unsafe { Library::new("libcuda.so.1") }
            .map_err(|error| anyhow!("failed to load CUDA driver libcuda.so.1: {error}"))?;
        Ok(Self {
            cu_init: load_symbol(&library, b"cuInit\0")?,
            cu_driver_get_version: load_symbol(&library, b"cuDriverGetVersion\0")?,
            cu_device_get: load_symbol(&library, b"cuDeviceGet\0")?,
            cu_device_get_name: load_symbol(&library, b"cuDeviceGetName\0")?,
            cu_device_get_uuid_v2: load_symbol(&library, b"cuDeviceGetUuid_v2\0")?,
            cu_device_compute_capability: load_symbol(&library, b"cuDeviceComputeCapability\0")?,
            cu_device_total_mem_v2: load_symbol(&library, b"cuDeviceTotalMem_v2\0")?,
            cu_ctx_create_v2: load_symbol(&library, b"cuCtxCreate_v2\0")?,
            cu_ctx_destroy_v2: load_symbol(&library, b"cuCtxDestroy_v2\0")?,
            cu_ctx_push_current_v2: load_symbol(&library, b"cuCtxPushCurrent_v2\0")?,
            cu_ctx_pop_current_v2: load_symbol(&library, b"cuCtxPopCurrent_v2\0")?,
            cu_ctx_synchronize: load_symbol(&library, b"cuCtxSynchronize\0")?,
            cu_mem_get_info_v2: load_symbol(&library, b"cuMemGetInfo_v2\0")?,
            cu_mem_alloc_v2: load_symbol(&library, b"cuMemAlloc_v2\0")?,
            cu_mem_free_v2: load_symbol(&library, b"cuMemFree_v2\0")?,
            cu_memcpy_htod_v2: load_symbol(&library, b"cuMemcpyHtoD_v2\0")?,
            cu_memcpy_dtoh_v2: load_symbol(&library, b"cuMemcpyDtoH_v2\0")?,
            cu_module_load_data_ex: load_symbol(&library, b"cuModuleLoadDataEx\0")?,
            cu_module_unload: load_symbol(&library, b"cuModuleUnload\0")?,
            cu_module_get_function: load_symbol(&library, b"cuModuleGetFunction\0")?,
            cu_launch_kernel: load_symbol(&library, b"cuLaunchKernel\0")?,
            cu_func_get_attribute: load_symbol(&library, b"cuFuncGetAttribute\0")?,
            cu_event_create: load_symbol(&library, b"cuEventCreate\0")?,
            cu_event_record: load_symbol(&library, b"cuEventRecord\0")?,
            cu_event_synchronize: load_symbol(&library, b"cuEventSynchronize\0")?,
            cu_event_elapsed_time: load_symbol(&library, b"cuEventElapsedTime\0")?,
            cu_event_destroy_v2: load_symbol(&library, b"cuEventDestroy_v2\0")?,
            _library: library,
        })
    }
}

fn load_symbol<T: Copy>(library: &Library, name: &'static [u8]) -> Result<T> {
    // SAFETY: Call sites use NUL-terminated CUDA symbol names and the matching ABI function type;
    // the returned pointer remains valid because `Api` retains `library`.
    let symbol = unsafe { library.get::<T>(name) }.map_err(|error| {
        let symbol_name = String::from_utf8_lossy(&name[..name.len().saturating_sub(1)]);
        anyhow!("failed to resolve CUDA driver symbol {symbol_name}: {error}")
    })?;
    Ok(*symbol)
}
fn check_cuda(result: CuResult, operation: &str) -> Result<()> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(anyhow!("{operation} failed with CUDA result code {result}"))
    }
}
fn report_cleanup_error(operation: &str, result: CuResult) {
    if result != CUDA_SUCCESS {
        tracing::warn!(
            operation,
            result_code = result,
            "CUDA cleanup operation failed"
        );
    }
}
/// Temporarily make one context current and restore the prior context on drop.
pub(super) struct CurrentContextGuard<'api> {
    api: &'api Api,
    context: CuContext,
    active: bool,
    _thread_bound: PhantomData<Rc<()>>,
}
impl<'api> CurrentContextGuard<'api> {
    fn push(api: &'api Api, context: CuContext) -> Result<Self> {
        // SAFETY: The context is live for the guard's lifetime and CUDA pushes it on this thread.
        check_cuda(
            unsafe { (api.cu_ctx_push_current_v2)(context) },
            "cuCtxPushCurrent_v2",
        )?;
        Ok(Self {
            api,
            context,
            active: true,
            _thread_bound: PhantomData,
        })
    }
    fn adopt_current(api: &'api Api, context: CuContext) -> Self {
        Self {
            api,
            context,
            active: true,
            _thread_bound: PhantomData,
        }
    }
    fn pop(mut self) -> Result<()> {
        self.pop_inner()
    }
    fn pop_inner(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        let mut popped = ptr::null_mut();
        // SAFETY: `popped` is writable and this guard owns the most recent context-stack push.
        let result = unsafe { (self.api.cu_ctx_pop_current_v2)(&mut popped) };
        self.active = false;
        check_cuda(result, "cuCtxPopCurrent_v2")?;
        if popped != self.context {
            if !popped.is_null() {
                // SAFETY: Restore the unexpected context returned by pop to preserve the caller's
                // prior stack entry while reporting that our own context was not on top.
                let restore_result = unsafe { (self.api.cu_ctx_push_current_v2)(popped) };
                report_cleanup_error("cuCtxPushCurrent_v2 after unexpected pop", restore_result);
            }
            bail!(
                "cuCtxPopCurrent_v2 returned context {popped:p}, expected {expected:p}",
                expected = self.context
            );
        }
        Ok(())
    }
}
impl Drop for CurrentContextGuard<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.pop_inner() {
            tracing::warn!(error = %error, "failed to restore prior CUDA context");
        }
    }
}
/// Selected CUDA device properties collected when creating a context.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(super) struct DeviceInfo {
    pub(super) ordinal: c_int,
    pub(super) uuid: String,
    pub(super) name: String,
    pub(super) major: c_int,
    pub(super) minor: c_int,
    pub(super) total_bytes: usize,
    pub(super) driver_version: c_int,
}

fn query_device_info(
    api: &Api,
    device: CuDevice,
    ordinal: c_int,
    driver_version: c_int,
) -> Result<DeviceInfo> {
    let mut name_buffer = [0 as c_char; DEVICE_NAME_BUFFER_BYTES];
    // SAFETY: name_buffer provides the writable capacity passed to the driver.
    check_cuda(
        unsafe {
            (api.cu_device_get_name)(
                name_buffer.as_mut_ptr(),
                DEVICE_NAME_BUFFER_BYTES as c_int,
                device,
            )
        },
        "cuDeviceGetName",
    )?;
    let name = decode_c_string_buffer(&name_buffer);
    if name.is_empty() {
        bail!("cuDeviceGetName returned an empty name for CUDA device {ordinal}");
    }
    let mut uuid = CuUuid { bytes: [0; 16] };
    // SAFETY: uuid is a writable CUuuid output and device came from cuDeviceGet.
    check_cuda(
        unsafe { (api.cu_device_get_uuid_v2)(&mut uuid, device) },
        "cuDeviceGetUuid_v2",
    )?;
    let mut major = 0;
    let mut minor = 0;
    // SAFETY: Both integer outputs are writable and device came from cuDeviceGet.
    check_cuda(
        unsafe { (api.cu_device_compute_capability)(&mut major, &mut minor, device) },
        "cuDeviceComputeCapability",
    )?;
    let mut total_bytes = 0;
    // SAFETY: total_bytes is a valid writable size_t output location.
    check_cuda(
        unsafe { (api.cu_device_total_mem_v2)(&mut total_bytes, device) },
        "cuDeviceTotalMem_v2",
    )?;
    Ok(DeviceInfo {
        ordinal,
        uuid: format_uuid(&uuid.bytes),
        name,
        major,
        minor,
        total_bytes,
        driver_version,
    })
}

fn format_uuid(bytes: &[u8; 16]) -> String {
    format!(
        "GPU-{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15],
    )
}
/// An owned CUDA context and the API table used to create it.
pub(super) struct Context {
    api: Api,
    raw: CuContext,
    info: DeviceInfo,
    // CUDA contexts are current-thread state; never move or share this wrapper across threads.
    _thread_bound: PhantomData<Rc<()>>,
}

impl Context {
    /// Load the Linux CUDA driver, select `device_ordinal`, and create its current context.
    pub(super) fn new(device_ordinal: c_int) -> Result<Self> {
        #[cfg(not(target_os = "linux"))]
        {
            let _ = device_ordinal;
            bail!("CUDA driver contexts are supported only on Linux")
        }
        #[cfg(target_os = "linux")]
        {
            if device_ordinal < 0 {
                bail!("CUDA device ordinal must be nonnegative, got {device_ordinal}");
            }
            let api = Api::load()?;
            // SAFETY: `cuInit` takes only a flags value and the resolved pointer has the declared ABI.
            check_cuda(unsafe { (api.cu_init)(0) }, "cuInit")?;
            let mut driver_version = 0;
            // SAFETY: `driver_version` is a valid writable output location for the driver call.
            check_cuda(
                unsafe { (api.cu_driver_get_version)(&mut driver_version) },
                "cuDriverGetVersion",
            )?;
            let mut device = -1;
            // SAFETY: `device` is writable and the ordinal was checked above.
            check_cuda(
                unsafe { (api.cu_device_get)(&mut device, device_ordinal) },
                "cuDeviceGet",
            )?;
            let info = query_device_info(&api, device, device_ordinal, driver_version)?;
            let mut raw = ptr::null_mut();
            // SAFETY: `raw` is writable; `device` is a valid driver device handle.
            check_cuda(
                unsafe { (api.cu_ctx_create_v2)(&mut raw, 0, device) },
                "cuCtxCreate_v2",
            )?;
            if raw.is_null() {
                bail!("cuCtxCreate_v2 succeeded but returned a null context");
            }
            if let Err(error) = CurrentContextGuard::adopt_current(&api, raw).pop() {
                // SAFETY: `raw` is the context just created above; the library remains alive.
                let destroy_result = unsafe { (api.cu_ctx_destroy_v2)(raw) };
                report_cleanup_error(
                    "cuCtxDestroy_v2 after context restore failure",
                    destroy_result,
                );
                return Err(anyhow!(
                    "failed to restore the previous current context after cuCtxCreate_v2: {error}"
                ));
            }
            Ok(Self {
                api,
                raw,
                info,
                _thread_bound: PhantomData,
            })
        }
    }
    /// Return a copy of the selected device's immutable properties.
    pub(super) fn info(&self) -> DeviceInfo {
        self.info.clone()
    }
    /// Return currently free and total bytes visible to the current CUDA context.
    pub(super) fn memory(&self) -> Result<(usize, usize)> {
        let _current_context = self.activate()?;
        let mut free_bytes = 0;
        let mut total_bytes = 0;
        // SAFETY: Both outputs are valid writable size_t locations; this context is still alive.
        check_cuda(
            unsafe { (self.api.cu_mem_get_info_v2)(&mut free_bytes, &mut total_bytes) },
            "cuMemGetInfo_v2",
        )?;
        Ok((free_bytes, total_bytes))
    }
    /// Wait for work previously submitted to this context to finish.
    pub(super) fn synchronize(&self) -> Result<()> {
        let _current_context = self.activate()?;
        // SAFETY: The API call has no pointer arguments and this context remains alive.
        check_cuda(
            unsafe { (self.api.cu_ctx_synchronize)() },
            "cuCtxSynchronize",
        )
    }
    pub(super) fn activate(&self) -> Result<CurrentContextGuard<'_>> {
        CurrentContextGuard::push(&self.api, self.raw)
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: This wrapper uniquely owns `raw`, is thread-bound, and `api` still owns libcuda.
        let result = unsafe { (self.api.cu_ctx_destroy_v2)(self.raw) };
        report_cleanup_error("cuCtxDestroy_v2", result);
    }
}
/// A device allocation whose lifetime is bounded by its owning context.
pub(super) struct Buffer<'ctx> {
    context: &'ctx Context,
    pointer: CuDevicePtr,
    bytes: usize,
    _thread_bound: PhantomData<Rc<()>>,
}
impl<'ctx> Buffer<'ctx> {
    /// Allocate `bytes` on the context's selected device.
    pub(super) fn new(context: &'ctx Context, bytes: usize) -> Result<Self> {
        if bytes == 0 {
            bail!("CUDA buffer allocation size must be greater than zero");
        }
        let _current_context = context.activate()?;
        let mut pointer = 0;
        // SAFETY: `pointer` is writable and `bytes` is nonzero; the context remains alive.
        check_cuda(
            unsafe { (context.api.cu_mem_alloc_v2)(&mut pointer, bytes) },
            "cuMemAlloc_v2",
        )?;
        if pointer == 0 {
            bail!("cuMemAlloc_v2 succeeded but returned a null device pointer");
        }
        Ok(Self {
            context,
            pointer,
            bytes,
            _thread_bound: PhantomData,
        })
    }
    /// Return the CUDA device pointer as an integer for kernel argument construction.
    pub(super) fn pointer(&self) -> CuDevicePtr {
        self.pointer
    }
    /// Return the allocation size in bytes.
    pub(super) fn len(&self) -> usize {
        self.bytes
    }
    /// Copy host bytes into the beginning of this device allocation.
    pub(super) fn upload(&self, source: &[u8]) -> Result<()> {
        self.upload_at(0, source)
    }
    /// Copy host bytes into this allocation beginning at byte `offset`.
    pub(super) fn upload_at(&self, offset: usize, source: &[u8]) -> Result<()> {
        check_transfer_range(self.bytes, offset, source.len(), "upload")?;
        if source.is_empty() {
            return Ok(());
        }
        let destination = checked_device_pointer_offset(self.pointer, offset, "upload")?;
        let _current_context = self.context.activate()?;
        // SAFETY: The source is live through this synchronous copy, and the validated
        // range and checked device pointer cover every transferred byte.
        check_cuda(
            unsafe {
                (self.context.api.cu_memcpy_htod_v2)(
                    destination,
                    source.as_ptr().cast::<c_void>(),
                    source.len(),
                )
            },
            "cuMemcpyHtoD_v2",
        )
    }
    /// Copy bytes from the beginning of this device allocation into `destination`.
    pub(super) fn download(&self, destination: &mut [u8]) -> Result<()> {
        self.download_at(0, destination)
    }
    /// Copy bytes from this allocation beginning at byte `offset` into `destination`.
    pub(super) fn download_at(&self, offset: usize, destination: &mut [u8]) -> Result<()> {
        check_transfer_range(self.bytes, offset, destination.len(), "download")?;
        if destination.is_empty() {
            return Ok(());
        }
        let source = checked_device_pointer_offset(self.pointer, offset, "download")?;
        let _current_context = self.context.activate()?;
        // SAFETY: The destination is writable through this synchronous copy, and
        // the validated range and checked device pointer cover every transferred byte.
        check_cuda(
            unsafe {
                (self.context.api.cu_memcpy_dtoh_v2)(
                    destination.as_mut_ptr().cast::<c_void>(),
                    source,
                    destination.len(),
                )
            },
            "cuMemcpyDtoH_v2",
        )
    }
}
impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        match self.context.activate() {
            Ok(_current_context) => {
                // SAFETY: This wrapper uniquely owns the allocation and its context is current.
                let result = unsafe { (self.context.api.cu_mem_free_v2)(self.pointer) };
                report_cleanup_error("cuMemFree_v2", result);
            }
            Err(error) => {
                tracing::warn!(error = %error, "failed to activate CUDA context before freeing buffer");
            }
        }
    }
}

fn check_transfer_range(
    allocation_bytes: usize,
    offset: usize,
    transfer_bytes: usize,
    operation: &str,
) -> Result<usize> {
    let end = offset.checked_add(transfer_bytes).ok_or_else(|| {
        anyhow!("CUDA {operation} offset {offset} plus length {transfer_bytes} overflows usize")
    })?;
    if offset > allocation_bytes || end > allocation_bytes {
        return Err(anyhow!(
            "CUDA {operation} range offset {offset} length {transfer_bytes} exceeds allocation size {allocation_bytes}"
        ));
    }
    Ok(end)
}

fn checked_device_pointer_offset(
    pointer: CuDevicePtr,
    offset: usize,
    operation: &str,
) -> Result<CuDevicePtr> {
    let offset = u64::try_from(offset)
        .map_err(|_| anyhow!("CUDA {operation} offset exceeds the device pointer range"))?;
    pointer
        .checked_add(offset)
        .ok_or_else(|| anyhow!("CUDA {operation} device pointer plus offset overflows u64"))
}
/// A loaded PTX module that unloads before its borrowed context can be dropped.
pub(super) struct Module<'ctx> {
    context: &'ctx Context,
    raw: CuModule,
    jit_log: String,
    _thread_bound: PhantomData<Rc<()>>,
}
impl<'ctx> Module<'ctx> {
    /// JIT-load PTX with bounded information and error logs.
    pub(super) fn load(context: &'ctx Context, ptx: &str) -> Result<Self> {
        Self::load_with_register_limit(context, ptx, None)
    }

    /// Specify the JIT register limit for an explicit register-budget experiment.
    /// This is a compiler limit; callers must inspect the actual function resource
    /// count before relying on it for `setmaxnreg` preconditions.
    pub(super) fn load_with_register_limit(
        context: &'ctx Context,
        ptx: &str,
        register_limit: Option<u32>,
    ) -> Result<Self> {
        if register_limit.is_some_and(|limit| !(24..=256).contains(&limit) || limit % 8 != 0) {
            bail!("register limit must be a multiple of eight from 24 through 256");
        }
        let ptx =
            CString::new(ptx).map_err(|error| anyhow!("PTX contains an interior NUL: {error}"))?;
        let mut info_buffer = vec![0 as c_char; JIT_LOG_BUFFER_BYTES];
        let mut error_buffer = vec![0 as c_char; JIT_LOG_BUFFER_BYTES];
        let mut options = vec![
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_LOG_VERBOSE,
        ];
        let mut option_values = vec![
            info_buffer.as_mut_ptr().cast::<c_void>(),
            ptr::without_provenance_mut::<c_void>(JIT_LOG_BUFFER_BYTES),
            error_buffer.as_mut_ptr().cast::<c_void>(),
            ptr::without_provenance_mut::<c_void>(JIT_LOG_BUFFER_BYTES),
            ptr::without_provenance_mut::<c_void>(1),
        ];
        if let Some(limit) = register_limit {
            options.push(0); // CU_JIT_MAX_REGISTERS
            option_values.push(ptr::without_provenance_mut::<c_void>(limit as usize));
        }
        let mut raw = ptr::null_mut();
        let _current_context = context.activate()?;
        // SAFETY: PTX and the two log buffers stay live for this call. CUDA's JIT size and verbose
        // values are scalar integers encoded as pointer-sized option values; the context is alive.
        let result = unsafe {
            (context.api.cu_module_load_data_ex)(
                &mut raw,
                ptx.as_ptr().cast::<c_void>(),
                options.len() as c_uint,
                options.as_mut_ptr(),
                option_values.as_mut_ptr(),
            )
        };
        let info_log = decode_c_string_buffer(&info_buffer);
        let error_log = decode_c_string_buffer(&error_buffer);
        let jit_log = format_jit_logs(&info_log, &error_log);
        if result != CUDA_SUCCESS {
            bail!("cuModuleLoadDataEx failed with CUDA result code {result}; JIT logs: {jit_log}");
        }
        if raw.is_null() {
            bail!("cuModuleLoadDataEx succeeded but returned a null module; JIT logs: {jit_log}");
        }
        Ok(Self {
            context,
            raw,
            jit_log,
            _thread_bound: PhantomData,
        })
    }
    /// Return the bounded, labeled JIT information and error logs.
    pub(super) fn jit_log(&self) -> &str {
        &self.jit_log
    }
    /// Resolve a named kernel function from this module.
    pub(super) fn function<'module>(&'module self, name: &str) -> Result<Function<'module, 'ctx>> {
        let name = CString::new(name)
            .map_err(|error| anyhow!("CUDA function name contains an interior NUL: {error}"))?;
        let _current_context = self.context.activate()?;
        let mut raw = ptr::null_mut();
        // SAFETY: `raw` is writable, `name` is NUL-terminated, and the borrowed module is alive.
        check_cuda(
            unsafe { (self.context.api.cu_module_get_function)(&mut raw, self.raw, name.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        if raw.is_null() {
            bail!("cuModuleGetFunction succeeded but returned a null function");
        }
        Ok(Function {
            module: self,
            raw,
            _thread_bound: PhantomData,
        })
    }
}
impl Drop for Module<'_> {
    fn drop(&mut self) {
        match self.context.activate() {
            Ok(_current_context) => {
                // SAFETY: This wrapper uniquely owns the module and its context is current.
                let result = unsafe { (self.context.api.cu_module_unload)(self.raw) };
                report_cleanup_error("cuModuleUnload", result);
            }
            Err(error) => {
                tracing::warn!(error = %error, "failed to activate CUDA context before unloading module");
            }
        }
    }
}
/// Resource values reported by CUDA for a compiled function.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub(super) struct FunctionResources {
    pub(super) registers: c_int,
    pub(super) static_shared_bytes: c_int,
    pub(super) local_bytes: c_int,
    pub(super) max_threads_per_block: c_int,
}
/// A function handle that cannot outlive the module from which it was resolved.
pub(super) struct Function<'module, 'ctx> {
    module: &'module Module<'ctx>,
    raw: CuFunction,
    _thread_bound: PhantomData<Rc<()>>,
}
impl Function<'_, '_> {
    /// Query register, shared-memory, local-memory, and block-thread attributes.
    pub(super) fn resources(&self) -> Result<FunctionResources> {
        Ok(FunctionResources {
            registers: self.attribute(CU_FUNC_ATTRIBUTE_NUM_REGS, "register count")?,
            static_shared_bytes: self.attribute(
                CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                "static shared-memory size",
            )?,
            local_bytes: self.attribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, "local-memory size")?,
            max_threads_per_block: self.attribute(
                CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                "maximum threads per block",
            )?,
        })
    }
    /// Launch this function on CUDA's default stream.
    ///
    /// # Safety
    /// Each element of `args` must point to live, correctly aligned host storage containing one
    /// argument value of the exact type and size expected by the kernel. Those host argument
    /// values are read during this call. Any device pointers encoded in them must remain allocated
    /// until the asynchronous kernel launch has completed. The supplied dimensions must also be
    /// valid for this function and device.
    pub(super) unsafe fn launch(
        &self,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        if grid.contains(&0) {
            bail!("CUDA launch grid dimensions must all be nonzero");
        }
        if block.contains(&0) {
            bail!("CUDA launch block dimensions must all be nonzero");
        }
        let _current_context = self.module.context.activate()?;
        let kernel_params = if args.is_empty() {
            ptr::null_mut()
        } else {
            args.as_mut_ptr()
        };
        // SAFETY: The caller upholds the documented kernel argument and lifetime contract; the
        // function's module and context remain borrowed and valid for this call.
        check_cuda(
            unsafe {
                (self.module.context.api.cu_launch_kernel)(
                    self.raw,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    shared_bytes,
                    ptr::null_mut(),
                    kernel_params,
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
    fn attribute(&self, attribute: c_int, label: &str) -> Result<c_int> {
        let _current_context = self.module.context.activate()?;
        let mut value = 0;
        // SAFETY: `value` is a writable output and `raw` is a live function from the borrowed module.
        check_cuda(
            unsafe {
                (self.module.context.api.cu_func_get_attribute)(&mut value, attribute, self.raw)
            },
            &format!("cuFuncGetAttribute ({label})"),
        )?;
        Ok(value)
    }
}
/// A CUDA event that belongs to and borrows its context.
pub(super) struct Event<'ctx> {
    context: &'ctx Context,
    raw: CuEvent,
    _thread_bound: PhantomData<Rc<()>>,
}
impl<'ctx> Event<'ctx> {
    /// Create an event suitable for recording on the default stream.
    pub(super) fn new(context: &'ctx Context) -> Result<Self> {
        let _current_context = context.activate()?;
        let mut raw = ptr::null_mut();
        // SAFETY: `raw` is writable; flags zero requests a standard CUDA event.
        check_cuda(
            unsafe { (context.api.cu_event_create)(&mut raw, 0) },
            "cuEventCreate",
        )?;
        if raw.is_null() {
            bail!("cuEventCreate succeeded but returned a null event");
        }
        Ok(Self {
            context,
            raw,
            _thread_bound: PhantomData,
        })
    }
    /// Record this event on CUDA's default stream.
    pub(super) fn record(&self) -> Result<()> {
        let _current_context = self.context.activate()?;
        // SAFETY: The event and its context are live; a null stream selects the default stream.
        check_cuda(
            unsafe { (self.context.api.cu_event_record)(self.raw, ptr::null_mut()) },
            "cuEventRecord",
        )
    }
    /// Wait until work recorded before this event has completed.
    pub(super) fn synchronize(&self) -> Result<()> {
        let _current_context = self.context.activate()?;
        // SAFETY: The event is live and belongs to the borrowed context.
        check_cuda(
            unsafe { (self.context.api.cu_event_synchronize)(self.raw) },
            "cuEventSynchronize",
        )
    }
    /// Return elapsed milliseconds from `start` to this event.
    pub(super) fn elapsed_since(&self, start: &Event<'ctx>) -> Result<f32> {
        if !ptr::eq(self.context, start.context) {
            bail!("CUDA event elapsed time requires events from the same context");
        }
        let _current_context = self.context.activate()?;
        let mut milliseconds = 0.0;
        // SAFETY: Both events are live in the same context; `milliseconds` is writable output.
        check_cuda(
            unsafe {
                (self.context.api.cu_event_elapsed_time)(&mut milliseconds, start.raw, self.raw)
            },
            "cuEventElapsedTime",
        )?;
        Ok(milliseconds)
    }
}
impl Drop for Event<'_> {
    fn drop(&mut self) {
        match self.context.activate() {
            Ok(_current_context) => {
                // SAFETY: This wrapper uniquely owns the event and its context is current.
                let result = unsafe { (self.context.api.cu_event_destroy_v2)(self.raw) };
                report_cleanup_error("cuEventDestroy_v2", result);
            }
            Err(error) => {
                tracing::warn!(error = %error, "failed to activate CUDA context before destroying event");
            }
        }
    }
}

fn decode_c_string_buffer(buffer: &[c_char]) -> String {
    let length = buffer
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(buffer.len());
    let bytes: Vec<u8> = buffer[..length]
        .iter()
        .map(|byte| byte.to_ne_bytes()[0])
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}
fn format_jit_logs(info_log: &str, error_log: &str) -> String {
    match (info_log.is_empty(), error_log.is_empty()) {
        (true, true) => String::new(),
        (false, true) => format!("info: {info_log}"),
        (true, false) => format!("error: {error_log}"),
        (false, false) => format!("info: {info_log}\nerror: {error_log}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        check_transfer_range, checked_device_pointer_offset, decode_c_string_buffer,
        format_jit_logs, format_uuid,
    };
    use std::ffi::c_char;
    #[test]
    fn transfer_range_accepts_exact_end_and_rejects_overrun() {
        assert_eq!(check_transfer_range(8, 4, 4, "upload").unwrap(), 8);
        assert!(
            check_transfer_range(8, 7, 2, "upload")
                .unwrap_err()
                .to_string()
                .contains("range offset 7 length 2 exceeds allocation size 8")
        );
    }
    #[test]
    fn transfer_range_rejects_usize_overflow() {
        assert!(
            check_transfer_range(usize::MAX, usize::MAX, 1, "download")
                .unwrap_err()
                .to_string()
                .contains("overflows usize")
        );
    }
    #[test]
    fn empty_transfer_is_allowed_at_end_but_not_beyond_it() {
        assert_eq!(check_transfer_range(8, 8, 0, "upload").unwrap(), 8);
        assert!(check_transfer_range(8, 9, 0, "upload").is_err());
        assert_eq!(check_transfer_range(8, 3, 0, "download").unwrap(), 3);
    }
    #[test]
    fn device_pointer_offset_is_checked() {
        assert_eq!(checked_device_pointer_offset(32, 7, "upload").unwrap(), 39);
        assert!(checked_device_pointer_offset(u64::MAX, 1, "upload").is_err());
    }
    #[test]
    fn jit_log_buffers_are_bounded_cleaned_and_labeled() {
        let terminated = [b'o' as c_char, b'k' as c_char, 0, b'x' as c_char];
        let full = [b'a' as c_char, b'b' as c_char];
        assert_eq!(decode_c_string_buffer(&terminated), "ok");
        assert_eq!(decode_c_string_buffer(&full), "ab");
        assert_eq!(
            format_jit_logs("compiled", "warning"),
            "info: compiled\nerror: warning"
        );
        assert_eq!(format_jit_logs("", ""), "");
    }

    #[test]
    fn cuda_uuid_uses_lowercase_gpu_uuid_format() {
        let bytes = std::array::from_fn(|index| index as u8);
        assert_eq!(
            format_uuid(&bytes),
            "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f"
        );
        assert_eq!(
            format_uuid(&[0; 16]),
            "GPU-00000000-0000-0000-0000-000000000000"
        );
    }
}
