//! CUDA stream and graph ownership for bounded fixed-shape replay probes.
//!
//! The wrappers borrow a thread-bound [`super::Context`], which keeps both the CUDA
//! context and its dynamically loaded driver library alive. Captured kernel nodes
//! still contain raw module and device addresses, so Rust cannot infer their true
//! retention period. Capture finalization, instantiation, and replay are unsafe and
//! require the caller to retain every referenced module and allocation.

use super::{Context, CuResult, Function, check_cuda, load_symbol, report_cleanup_error};
use anyhow::{Result, anyhow, bail, ensure};
use std::cell::{Cell, RefCell};
use std::ffi::{c_int, c_uint, c_ulonglong, c_void};
use std::marker::PhantomData;
use std::ptr;
use std::rc::Rc;

type CuStream = *mut c_void;
type CuGraph = *mut c_void;
type CuGraphExec = *mut c_void;

const CU_STREAM_NON_BLOCKING: c_uint = 1;
const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_int = 1;

type CuStreamCreateFn = unsafe extern "C" fn(*mut CuStream, c_uint) -> CuResult;
type CuStreamDestroyV2Fn = unsafe extern "C" fn(CuStream) -> CuResult;
type CuStreamSynchronizeFn = unsafe extern "C" fn(CuStream) -> CuResult;
type CuStreamBeginCaptureFn = unsafe extern "C" fn(CuStream, c_int) -> CuResult;
type CuStreamEndCaptureFn = unsafe extern "C" fn(CuStream, *mut CuGraph) -> CuResult;
type CuGraphInstantiateWithFlagsFn =
    unsafe extern "C" fn(*mut CuGraphExec, CuGraph, c_ulonglong) -> CuResult;
type CuGraphLaunchFn = unsafe extern "C" fn(CuGraphExec, CuStream) -> CuResult;
type CuGraphDestroyFn = unsafe extern "C" fn(CuGraph) -> CuResult;
type CuGraphExecDestroyFn = unsafe extern "C" fn(CuGraphExec) -> CuResult;

#[derive(Clone, Copy)]
struct GraphApi {
    stream_create: CuStreamCreateFn,
    stream_destroy_v2: CuStreamDestroyV2Fn,
    stream_synchronize: CuStreamSynchronizeFn,
    stream_begin_capture: CuStreamBeginCaptureFn,
    stream_end_capture: CuStreamEndCaptureFn,
    graph_instantiate_with_flags: CuGraphInstantiateWithFlagsFn,
    graph_launch: CuGraphLaunchFn,
    graph_destroy: CuGraphDestroyFn,
    graph_exec_destroy: CuGraphExecDestroyFn,
}

impl GraphApi {
    fn load(context: &Context) -> Result<Self> {
        let library = &context.api._library;
        Ok(Self {
            stream_create: load_symbol(library, b"cuStreamCreate\0")?,
            stream_destroy_v2: load_symbol(library, b"cuStreamDestroy_v2\0")?,
            stream_synchronize: load_symbol(library, b"cuStreamSynchronize\0")?,
            stream_begin_capture: load_symbol(library, b"cuStreamBeginCapture\0")?,
            stream_end_capture: load_symbol(library, b"cuStreamEndCapture\0")?,
            graph_instantiate_with_flags: load_symbol(library, b"cuGraphInstantiateWithFlags\0")?,
            graph_launch: load_symbol(library, b"cuGraphLaunch\0")?,
            graph_destroy: load_symbol(library, b"cuGraphDestroy\0")?,
            graph_exec_destroy: load_symbol(library, b"cuGraphExecDestroy\0")?,
        })
    }
}

thread_local! {
    /// CUDA THREAD_LOCAL capture restricts unsafe calls made by this host thread.
    /// Keep a mirror so wrappers can reject synchronization or unrelated launches.
    static ACTIVE_CAPTURE_CONTEXTS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

fn context_key(context: &Context) -> usize {
    ptr::from_ref(context) as usize
}

fn any_capture_active() -> bool {
    ACTIVE_CAPTURE_CONTEXTS.with(|contexts| {
        contexts
            .try_borrow()
            .map_or(true, |contexts| !contexts.is_empty())
    })
}

fn capture_active_for(context: &Context) -> bool {
    let key = context_key(context);
    ACTIVE_CAPTURE_CONTEXTS.with(|contexts| {
        contexts
            .try_borrow()
            .map_or(true, |contexts| contexts.contains(&key))
    })
}

fn register_capture(context: &Context) -> Result<()> {
    ACTIVE_CAPTURE_CONTEXTS.with(|contexts| {
        let mut contexts = contexts
            .try_borrow_mut()
            .map_err(|_| anyhow!("CUDA graph capture state is already borrowed"))?;
        ensure!(
            contexts.is_empty(),
            "nested CUDA graph capture on one host thread is not supported"
        );
        contexts.push(context_key(context));
        Ok(())
    })
}

fn unregister_capture(context: &Context) {
    let key = context_key(context);
    let result = ACTIVE_CAPTURE_CONTEXTS.try_with(|contexts| {
        if let Ok(mut contexts) = contexts.try_borrow_mut()
            && let Some(index) = contexts.iter().position(|active| *active == key)
        {
            contexts.remove(index);
        }
    });
    if result.is_err() {
        tracing::warn!("failed to clear thread-local CUDA graph capture state");
    }
}

fn ensure_no_capture(operation: &str) -> Result<()> {
    ensure!(
        !any_capture_active(),
        "cannot {operation} while CUDA graph capture is active on this host thread"
    );
    Ok(())
}

/// Reject host-side operations that cannot safely occur during a graph capture.
///
/// The parent driver may use this before synchronous copies, allocation, context
/// synchronization, event instrumentation, or other host-side work.
pub(in crate::kernels::cuda) fn ensure_driver_operation_allowed(operation: &str) -> Result<()> {
    ensure_no_capture(operation)
}

/// Guard used by the existing default-stream launch path before profiling or launch.
///
/// The parent driver module should call this at the start of `Function::launch`; the
/// stream launch method below never touches the launch profiler or default stream.
pub(in crate::kernels::cuda) fn ensure_default_stream_launch_allowed(
    _context: &Context,
) -> Result<()> {
    ensure_driver_operation_allowed("launch on the default stream")
}

/// An owned nonblocking CUDA stream tied to its context and creating thread.
pub(in crate::kernels::cuda) struct Stream<'ctx> {
    context: &'ctx Context,
    api: GraphApi,
    raw: CuStream,
    capturing: Cell<bool>,
    _thread_bound: PhantomData<Rc<()>>,
}

impl<'ctx> Stream<'ctx> {
    /// Create a nonblocking stream in `context`.
    pub(in crate::kernels::cuda) fn new(context: &'ctx Context) -> Result<Self> {
        ensure_no_capture("create a CUDA stream")?;
        let api = GraphApi::load(context)?;
        let _current_context = context.activate()?;
        let mut raw = ptr::null_mut();
        // SAFETY: `raw` is writable and the selected context is current. The flag requests
        // CU_STREAM_NON_BLOCKING, so this stream does not synchronize with the legacy stream.
        let result = unsafe { (api.stream_create)(&mut raw, CU_STREAM_NON_BLOCKING) };
        if result != super::CUDA_SUCCESS {
            if !raw.is_null() {
                // SAFETY: CUDA returned a handle despite the failed create result; the context
                // guard is still active and the handle is released before returning the error.
                let cleanup = unsafe { (api.stream_destroy_v2)(raw) };
                report_cleanup_error("cuStreamDestroy_v2 after create failure", cleanup);
            }
            check_cuda(result, "cuStreamCreate")?;
        }
        if raw.is_null() {
            bail!("cuStreamCreate succeeded but returned a null stream");
        }
        Ok(Self {
            context,
            api,
            raw,
            capturing: Cell::new(false),
            _thread_bound: PhantomData,
        })
    }

    /// Return whether this stream currently owns an active capture guard.
    pub(in crate::kernels::cuda) fn is_capturing(&self) -> bool {
        self.capturing.get()
    }

    /// Begin thread-local graph capture on this stream.
    pub(in crate::kernels::cuda) fn begin_capture(&self) -> Result<StreamCapture<'_, 'ctx>> {
        ensure!(!self.capturing.get(), "CUDA stream is already capturing");
        ensure_no_capture("begin another CUDA graph capture")?;
        register_capture(self.context)?;
        let _current_context = match self.context.activate() {
            Ok(guard) => guard,
            Err(error) => {
                unregister_capture(self.context);
                return Err(error);
            }
        };
        // SAFETY: The stream is owned by this wrapper, has the same live context, and no capture
        // is active on this host thread. THREAD_LOCAL confines capture restrictions to this thread.
        let result = unsafe {
            (self.api.stream_begin_capture)(self.raw, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        };
        if result != super::CUDA_SUCCESS {
            unregister_capture(self.context);
            check_cuda(result, "cuStreamBeginCapture")?;
        }
        self.capturing.set(true);
        Ok(StreamCapture {
            stream: self,
            active: true,
            _thread_bound: PhantomData,
        })
    }

    /// End and discard a capture whose guard could not clean it up during drop.
    pub(in crate::kernels::cuda) fn abort_capture(&mut self) -> Result<()> {
        if !self.capturing.get() {
            return Ok(());
        }
        abort_capture_inner(self.context, self.api, self.raw, &self.capturing)
    }

    /// Wait for all operations in this stream to complete.
    pub(in crate::kernels::cuda) fn synchronize(&self) -> Result<()> {
        ensure!(
            !self.capturing.get(),
            "cannot synchronize a CUDA stream during graph capture"
        );
        ensure_no_capture("synchronize a CUDA stream")?;
        let _current_context = self.context.activate()?;
        // SAFETY: The stream belongs to the active context and is not being captured.
        check_cuda(
            unsafe { (self.api.stream_synchronize)(self.raw) },
            "cuStreamSynchronize",
        )
    }

    fn belongs_to(&self, context: &Context) -> bool {
        ptr::eq(self.context, context)
    }
}

impl Drop for Stream<'_> {
    fn drop(&mut self) {
        if self.capturing.get() {
            if let Err(error) =
                abort_capture_inner(self.context, self.api, self.raw, &self.capturing)
            {
                tracing::warn!(error = %error, "failed to abort CUDA capture before destroying stream");
            }
        }
        match self.context.activate() {
            Ok(_current_context) => {
                // SAFETY: This wrapper uniquely owns the stream and its context is current.
                let result = unsafe { (self.api.stream_destroy_v2)(self.raw) };
                report_cleanup_error("cuStreamDestroy_v2", result);
            }
            Err(error) => {
                tracing::warn!(error = %error, "failed to activate CUDA context before destroying stream");
            }
        }
    }
}

/// A live stream capture. Dropping the guard ends capture and destroys its graph.
pub(in crate::kernels::cuda) struct StreamCapture<'stream, 'ctx> {
    stream: &'stream Stream<'ctx>,
    active: bool,
    _thread_bound: PhantomData<Rc<()>>,
}

impl<'ctx> StreamCapture<'_, 'ctx> {
    /// Finish capture and return its graph.
    ///
    /// # Safety
    /// CUDA copies raw kernel function and device-address values into graph nodes. Every module
    /// and allocation referenced by captured launches must remain alive at a stable address while
    /// the returned graph or any executable instantiated from it exists, and until all replayed
    /// work has completed. The caller must also keep host argument storage live through each
    /// `cuLaunchKernel` call and avoid allocations, host synchronization, default-stream launches,
    /// and profiler events during capture.
    pub(in crate::kernels::cuda) unsafe fn finish(mut self) -> Result<Graph<'ctx>> {
        let _current_context = self.stream.context.activate()?;
        let mut raw = ptr::null_mut();
        // SAFETY: This guard began capture on the same stream and remains on the creating thread.
        let result = unsafe { (self.stream.api.stream_end_capture)(self.stream.raw, &mut raw) };
        self.release_capture_state();
        if result != super::CUDA_SUCCESS {
            if !raw.is_null() {
                // SAFETY: The partial graph was returned by this context's EndCapture call.
                let cleanup = unsafe { (self.stream.api.graph_destroy)(raw) };
                report_cleanup_error("cuGraphDestroy after capture failure", cleanup);
            }
            check_cuda(result, "cuStreamEndCapture")?;
        }
        if raw.is_null() {
            bail!("cuStreamEndCapture succeeded but returned a null graph");
        }
        Ok(Graph {
            context: self.stream.context,
            api: self.stream.api,
            raw,
            _thread_bound: PhantomData,
        })
    }

    /// Abort capture and destroy any partial graph returned by CUDA.
    pub(in crate::kernels::cuda) fn abort(mut self) -> Result<()> {
        self.abort_inner()
    }

    fn abort_inner(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        let result = abort_capture_inner(
            self.stream.context,
            self.stream.api,
            self.stream.raw,
            &self.stream.capturing,
        );
        if !self.stream.capturing.get() {
            self.active = false;
        }
        result
    }

    fn release_capture_state(&mut self) {
        if self.active {
            self.active = false;
            self.stream.capturing.set(false);
            unregister_capture(self.stream.context);
        }
    }
}

impl Drop for StreamCapture<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.abort_inner() {
            tracing::warn!(error = %error, "failed to clean up dropped CUDA graph capture");
        }
    }
}

fn abort_capture_inner(
    context: &Context,
    api: GraphApi,
    stream: CuStream,
    capturing: &Cell<bool>,
) -> Result<()> {
    let _current_context = context.activate()?;
    let mut graph = ptr::null_mut();
    // SAFETY: The stream is live, capture began on it on this thread, and the context is current.
    let result = unsafe { (api.stream_end_capture)(stream, &mut graph) };
    if !graph.is_null() {
        // SAFETY: EndCapture transferred ownership of this partial graph to the caller.
        let cleanup = unsafe { (api.graph_destroy)(graph) };
        report_cleanup_error("cuGraphDestroy after capture abort", cleanup);
    }
    capturing.set(false);
    unregister_capture(context);
    check_cuda(result, "cuStreamEndCapture (abort)")
}

/// An owned captured CUDA graph tied to the context and creating thread.
pub(in crate::kernels::cuda) struct Graph<'ctx> {
    context: &'ctx Context,
    api: GraphApi,
    raw: CuGraph,
    _thread_bound: PhantomData<Rc<()>>,
}

impl<'ctx> Graph<'ctx> {
    /// Instantiate this graph with CUDA's default graph flags.
    ///
    /// # Safety
    /// Every module and device allocation referenced by captured nodes must remain live at the
    /// same address until this graph and every executable instantiated from it are destroyed and
    /// all in-flight graph work has completed. CUDA does not retain the Rust [`Function`] or
    /// buffer owners whose raw values were copied into kernel nodes.
    pub(in crate::kernels::cuda) unsafe fn instantiate(&self) -> Result<GraphExec<'ctx>> {
        ensure_no_capture("instantiate a CUDA graph")?;
        let _current_context = self.context.activate()?;
        let mut raw = ptr::null_mut();
        // SAFETY: `self.raw` is a live graph in the current context; zero selects default flags.
        let result = unsafe {
            (self.api.graph_instantiate_with_flags)(&mut raw, self.raw, 0 as c_ulonglong)
        };
        if result != super::CUDA_SUCCESS {
            if !raw.is_null() {
                // SAFETY: CUDA returned a partial executable handle in this context.
                let cleanup = unsafe { (self.api.graph_exec_destroy)(raw) };
                report_cleanup_error("cuGraphExecDestroy after instantiate failure", cleanup);
            }
            check_cuda(result, "cuGraphInstantiateWithFlags")?;
        }
        if raw.is_null() {
            bail!("cuGraphInstantiateWithFlags succeeded but returned a null executable graph");
        }
        Ok(GraphExec {
            context: self.context,
            api: self.api,
            raw,
            _thread_bound: PhantomData,
        })
    }
}

impl Drop for Graph<'_> {
    fn drop(&mut self) {
        match self.context.activate() {
            Ok(_current_context) => {
                // SAFETY: This wrapper uniquely owns the graph and its context is current.
                let result = unsafe { (self.api.graph_destroy)(self.raw) };
                report_cleanup_error("cuGraphDestroy", result);
            }
            Err(error) => {
                tracing::warn!(error = %error, "failed to activate CUDA context before destroying graph");
            }
        }
    }
}

/// An executable CUDA graph tied to the context and creating thread.
pub(in crate::kernels::cuda) struct GraphExec<'ctx> {
    context: &'ctx Context,
    api: GraphApi,
    raw: CuGraphExec,
    _thread_bound: PhantomData<Rc<()>>,
}

impl GraphExec<'_> {
    /// Enqueue this executable graph on a noncapturing stream in the same context.
    ///
    /// # Safety
    /// Every module and allocation referenced by captured nodes must remain live at the same
    /// address through this launch and until the stream has completed all work. The caller must
    /// synchronize before releasing those resources; CUDA serializes launches of one executable
    /// handle.
    pub(in crate::kernels::cuda) unsafe fn launch(&self, stream: &Stream<'_>) -> Result<()> {
        ensure!(
            ptr::eq(self.context, stream.context),
            "CUDA graph launch requires a stream from the same context"
        );
        ensure!(
            !stream.capturing.get(),
            "cannot launch an executable graph into a stream during capture"
        );
        ensure_no_capture("launch a CUDA graph")?;
        let _current_context = self.context.activate()?;
        // SAFETY: The graph, stream, and context are live; the caller upholds the captured-resource
        // and in-flight completion requirements documented for this method.
        check_cuda(
            unsafe { (self.api.graph_launch)(self.raw, stream.raw) },
            "cuGraphLaunch",
        )
    }
}

impl Drop for GraphExec<'_> {
    fn drop(&mut self) {
        match self.context.activate() {
            Ok(_current_context) => {
                // SAFETY: This wrapper uniquely owns the executable graph and its context is current.
                let result = unsafe { (self.api.graph_exec_destroy)(self.raw) };
                report_cleanup_error("cuGraphExecDestroy", result);
            }
            Err(error) => {
                tracing::warn!(error = %error, "failed to activate CUDA context before destroying executable graph");
            }
        }
    }
}

impl Function<'_, '_> {
    /// Enqueue this function on an owned stream without adding profiler events or synchronizing.
    ///
    /// # Safety
    /// Each element of `args` must point to live, correctly aligned host storage containing one
    /// argument value of the exact type and size expected by the kernel. The host argument storage
    /// is read during this call. Device pointers encoded in those values must remain allocated
    /// until the stream work completes. If `stream` is capturing, every referenced module and
    /// allocation must remain live at its stable address through all graph and executable lifetimes
    /// and every replay. The supplied dimensions must be valid for the function and device.
    pub(in crate::kernels::cuda) unsafe fn launch_on_stream(
        &self,
        stream: &Stream<'_>,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        validate_launch_dimensions(grid, block)?;
        let context = self.module.context;
        ensure!(
            stream.belongs_to(context),
            "CUDA kernel launch requires a stream from the function's context"
        );
        if stream.capturing.get() {
            ensure!(
                capture_active_for(context),
                "CUDA stream capture state is inconsistent on this host thread"
            );
        } else {
            ensure_no_capture("launch a kernel outside the active capture stream")?;
        }
        let _current_context = context.activate()?;
        let kernel_params = if args.is_empty() {
            ptr::null_mut()
        } else {
            args.as_mut_ptr()
        };
        let operation = format!("cuLaunchKernel ({})", self.name.to_str()?);
        // SAFETY: The caller upholds the argument, buffer-retention, and dimension contracts; the
        // function, stream, and context remain live for this driver call.
        check_cuda(
            unsafe {
                (context.api.cu_launch_kernel)(
                    self.raw,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    shared_bytes,
                    stream.raw,
                    kernel_params,
                    ptr::null_mut(),
                )
            },
            &operation,
        )
    }
}

fn validate_launch_dimensions(grid: [u32; 3], block: [u32; 3]) -> Result<()> {
    if grid.contains(&0) {
        bail!("CUDA launch grid dimensions must all be nonzero");
    }
    if block.contains(&0) {
        bail!("CUDA launch block dimensions must all be nonzero");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_launch_dimensions;

    #[test]
    fn stream_launch_rejects_zero_grid_or_block_dimensions() {
        assert!(validate_launch_dimensions([1, 1, 1], [32, 1, 1]).is_ok());
        assert!(validate_launch_dimensions([0, 1, 1], [32, 1, 1]).is_err());
        assert!(validate_launch_dimensions([1, 1, 1], [32, 0, 1]).is_err());
    }
}
