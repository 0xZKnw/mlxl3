//! Owned MLX arrays through a small C ABI. There is no Python runtime.
use crate::contracts;
use anyhow::{Context, Result, bail, ensure};
use std::{
    ffi::{CStr, CString, c_char, c_void},
    fs::File,
    marker::PhantomData,
    os::fd::AsRawFd,
    ptr::NonNull,
    rc::Rc,
    sync::OnceLock,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Dtype {
    Bool = 0,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
    BFloat16,
    Complex64,
}
impl Dtype {
    pub fn item_size(self) -> usize {
        match self {
            Self::Bool | Self::UInt8 | Self::Int8 => 1,
            Self::UInt16 | Self::Int16 | Self::Float16 | Self::BFloat16 => 2,
            Self::UInt32 | Self::Int32 | Self::Float32 => 4,
            _ => 8,
        }
    }
    fn from_code(code: i32) -> Result<Self> {
        Ok(match code {
            0 => Self::Bool,
            1 => Self::UInt8,
            2 => Self::UInt16,
            3 => Self::UInt32,
            4 => Self::UInt64,
            5 => Self::Int8,
            6 => Self::Int16,
            7 => Self::Int32,
            8 => Self::Int64,
            9 => Self::Float16,
            10 => Self::Float32,
            11 => Self::Float64,
            12 => Self::BFloat16,
            13 => Self::Complex64,
            _ => bail!("unsupported MLX dtype {code}"),
        })
    }
}

unsafe extern "C" {
    fn mlxl3_start_capture(path: *const c_char) -> i32;
    fn mlxl3_stop_capture() -> i32;
    fn mlxl3_set_cache_limit(bytes: u64) -> i32;
    fn mlxl3_pack_embedding(p: *mut c_void, out: *mut *mut c_void) -> i32;
    fn mlxl3_array_retained_bytes(p: *mut c_void, out: *mut u64) -> i32;
    fn mlxl3_array_compact_copy(p: *mut c_void, out: *mut *mut c_void) -> i32;
    fn mlxl3_array_affine4(
        x: *mut c_void,
        w: *mut c_void,
        scales: *mut c_void,
        biases: *mut c_void,
        indices: *mut c_void,
        out: *mut *mut c_void,
    ) -> i32;
    fn mlxl3_mlx_error() -> *const c_char;
    fn mlxl3_mlx_init(metallib: *const c_char) -> i32;
    fn mlxl3_memory_stats(out: *mut u64, reset_peak: bool) -> i32;
    fn mlxl3_synchronize() -> i32;
    fn mlxl3_clear_cache() -> i32;
    fn mlxl3_array_free(p: *mut c_void);
    fn mlxl3_array_clone(p: *mut c_void, out: *mut *mut c_void) -> i32;
    fn mlxl3_array_metadata(
        p: *mut c_void,
        rank: *mut usize,
        dims: *mut *const i32,
        dtype: *mut i32,
    ) -> i32;
    fn mlxl3_array_from_bytes(
        data: *const u8,
        count: usize,
        dims: *const i32,
        rank: usize,
        dtype: i32,
        out: *mut *mut c_void,
    ) -> i32;
    fn mlxl3_array_from_file(
        fd: i32,
        offset: u64,
        dims: *const i32,
        rank: usize,
        dtype: i32,
        out: *mut *mut c_void,
    ) -> i32;
    fn mlxl3_arrays_from_files(
        fds: *const i32,
        offsets: *const u64,
        dims: *const i32,
        ranks: *const usize,
        dtypes: *const i32,
        count: usize,
        workers: usize,
        out: *mut *mut c_void,
    ) -> i32;
    fn mlxl3_array_zeros(dims: *const i32, rank: usize, dtype: i32, out: *mut *mut c_void) -> i32;
    fn mlxl3_array_eval(p: *mut c_void) -> i32;
    fn mlxl3_arrays_async_eval(inputs: *const *mut c_void, count: usize) -> i32;
    fn mlxl3_array_copy_bytes(p: *mut c_void, out: *mut u8, count: usize) -> i32;
    fn mlxl3_array_unary(
        p: *mut c_void,
        operation: i32,
        args: *const i32,
        nargs: usize,
        scalar: f32,
        flag: i32,
        out: *mut *mut c_void,
    ) -> i32;
    fn mlxl3_array_binary(
        lhs: *mut c_void,
        rhs: *mut c_void,
        operation: i32,
        arg: i32,
        scalar: f32,
        out: *mut *mut c_void,
    ) -> i32;
    fn mlxl3_array_concatenate(
        inputs: *const *mut c_void,
        count: usize,
        axis: i32,
        out: *mut *mut c_void,
    ) -> i32;
    fn mlxl3_array_sdpa(
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        scale: f32,
        causal: i32,
        out: *mut *mut c_void,
    ) -> i32;
    fn mlxl3_array_sdpa_mask(
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        mask: *mut c_void,
        scale: f32,
        out: *mut *mut c_void,
    ) -> i32;
    fn mlxl3_array_rope_freqs(
        input: *mut c_void,
        freqs: *mut c_void,
        dims: i32,
        offset: i32,
        out: *mut *mut c_void,
    ) -> i32;
    fn mlxl3_metal_kernel(
        name: *const c_char,
        input_names: *const *const c_char,
        output_names: *const *const c_char,
        header: *const c_char,
        source: *const c_char,
        inputs: *const *mut c_void,
        ninputs: usize,
        output_dims: *const i32,
        output_ranks: *const usize,
        output_types: *const i32,
        noutputs: usize,
        grid: *const i32,
        threadgroup: *const i32,
        outputs: *mut *mut c_void,
    ) -> i32;
}

fn checked(status: i32) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    // The C bridge owns this thread-local NUL-terminated buffer until its next call.
    let message = unsafe { CStr::from_ptr(mlxl3_mlx_error()) }.to_string_lossy();
    bail!("MLX: {message}")
}
fn initialize() -> Result<()> {
    thread_local! { static INITIALIZED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
    if INITIALIZED.get() {
        return Ok(());
    }
    let root = std::env::var_os("MLXL3_MLX_ROOT").map(std::path::PathBuf::from);
    let bundled = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join("mlx.metallib")));
    let library = root
        .map(|root| root.join("lib/mlx.metallib"))
        .filter(|path| path.is_file())
        .or_else(|| bundled.filter(|path| path.is_file()))
        .unwrap_or_else(|| {
            std::path::Path::new(env!("MLXL3_MLX_BUILD_ROOT")).join("lib/mlx.metallib")
        });
    ensure!(
        library.is_file(),
        "MLX metallib missing: {}",
        library.display()
    );
    let path = CString::new(library.to_str().context("MLX path must be UTF-8")?)?;
    let cache_limit = match std::env::var("MLXL3_ALLOCATOR_CACHE_MIB") {
        Ok(value) => Some(
            value
                .parse::<usize>()
                .ok()
                .and_then(crate::memory_policy::mib_budget)
                .context("MLXL3_ALLOCATOR_CACHE_MIB must be an integer in 0..4096")?,
        ),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    checked(unsafe { mlxl3_mlx_init(path.as_ptr()) })?;
    if let Some(bytes) = cache_limit {
        checked(unsafe { mlxl3_set_cache_limit(bytes as u64) })?;
    }
    INITIALIZED.set(true);
    Ok(())
}
fn bytes_for(shape: &[i32], dtype: Dtype) -> Result<usize> {
    contracts::array_bytes(shape, dtype.item_size())
        .context("negative tensor dimension or tensor byte size overflow")
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct MemoryStats {
    pub mlx_active_bytes: u64,
    pub mlx_cache_bytes: u64,
    pub mlx_peak_bytes: u64,
    pub process_footprint_bytes: Option<u64>,
    pub process_lifetime_peak_bytes: Option<u64>,
}

pub fn memory_stats(reset_peak: bool) -> Result<MemoryStats> {
    initialize()?;
    let mut values = [0u64; 5];
    checked(unsafe { mlxl3_memory_stats(values.as_mut_ptr(), reset_peak) })?;
    Ok(MemoryStats {
        mlx_active_bytes: values[0],
        mlx_cache_bytes: values[1],
        mlx_peak_bytes: values[2],
        process_footprint_bytes: (values[3] != 0).then_some(values[3]),
        process_lifetime_peak_bytes: (values[4] != 0).then_some(values[4]),
    })
}

/// Release unused allocator buffers; live arrays and their graphs are retained.
pub fn clear_cache() -> Result<()> {
    initialize()?;
    checked(unsafe { mlxl3_clear_cache() })
}

/// Finish submitted GPU work before reporting a settled load/unload state.
pub fn synchronize() -> Result<()> {
    initialize()?;
    checked(unsafe { mlxl3_synchronize() })
}

pub struct MetalCapture {
    active: bool,
}
impl MetalCapture {
    pub fn start(path: &std::path::Path) -> Result<Self> {
        ensure!(!path.exists(), "refusing to overwrite Metal capture");
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(std::path::Path::new("."));
        ensure!(
            parent.is_dir() && path.extension().is_some_and(|e| e == "gputrace"),
            "capture requires an existing parent and .gputrace path"
        );
        let path = CString::new(path.to_str().context("capture path must be UTF-8")?)?;
        initialize()?;
        checked(unsafe { mlxl3_start_capture(path.as_ptr()) })?;
        Ok(Self { active: true })
    }
    pub fn finish(mut self) -> Result<()> {
        let synced = synchronize();
        let stopped = checked(unsafe { mlxl3_stop_capture() });
        if stopped.is_ok() {
            self.active = false;
        }
        synced?;
        stopped
    }
}
impl Drop for MetalCapture {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if let Err(error) = synchronize() {
            eprintln!("Metal capture synchronization failed: {error}");
        }
        if let Err(error) = checked(unsafe { mlxl3_stop_capture() }) {
            eprintln!("Metal capture cleanup failed: {error}");
        }
    }
}

/// One explicitly selected decode request per process, after user warmup.
pub fn capture_for_request(request: &str) -> Result<Option<MetalCapture>> {
    use std::sync::atomic::{AtomicBool, Ordering};
    static CAPTURED: AtomicBool = AtomicBool::new(false);
    let path = std::env::var_os("MLXL3_METAL_CAPTURE_PATH");
    let target = std::env::var("MLXL3_METAL_CAPTURE_REQUEST").ok();
    if path.is_none() && target.is_none() {
        return Ok(None);
    }
    let path = path.context("MLXL3_METAL_CAPTURE_PATH is required")?;
    let target = target
        .filter(|s| !s.is_empty())
        .context("MLXL3_METAL_CAPTURE_REQUEST must name a request")?;
    if !crate::memory_policy::capture_matches(
        target.as_bytes(),
        request.as_bytes(),
        CAPTURED.load(Ordering::Relaxed),
    ) {
        return Ok(None);
    }
    if CAPTURED
        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return Ok(None);
    }
    match MetalCapture::start(std::path::Path::new(&path)) {
        Ok(capture) => Ok(Some(capture)),
        Err(error) => {
            CAPTURED.store(false, Ordering::Relaxed);
            Err(error)
        }
    }
}

/// Call only at a request boundary, after dropping unused model/session state.
/// Live weights and saved prompt snapshots retain their buffers.
pub fn release_idle_cache() -> Result<()> {
    release_idle_cache_with_policy(false)
}

/// Memory saving releases even small unused buffers, at the cost of reuse.
pub fn release_idle_cache_with_policy(memory_saver: bool) -> Result<()> {
    synchronize()?;
    if contracts::release_idle_cache(memory_stats(false)?.mlx_cache_bytes, memory_saver) {
        clear_cache()?;
    }
    Ok(())
}

pub fn is_m5_gpu() -> Result<bool> {
    initialize()?;
    // Allow testing the portable path, never force TensorOps on an older GPU.
    if std::env::var_os("MLXL3_DISABLE_TENSOR_OPS").is_some() {
        return Ok(false);
    }
    static IS_M5: OnceLock<bool> = OnceLock::new();
    if let Some(&is_m5) = IS_M5.get() {
        return Ok(is_m5);
    }
    // device_info is declared in MLX headers but not exported by its dylib.
    let device = metal::Device::system_default().context("no Metal device")?;
    let is_m5 = device.name() == "Apple M5" || device.name().starts_with("Apple M5 ");
    Ok(*IS_M5.get_or_init(|| is_m5))
}

#[derive(Debug)]
pub struct Array {
    handle: NonNull<c_void>,
    shape: Vec<i32>,
    dtype: Dtype,
    // MLX graph handles stay on the creating thread; no unverified Send/Sync.
    _thread: PhantomData<Rc<()>>,
}
impl Drop for Array {
    fn drop(&mut self) {
        unsafe { mlxl3_array_free(self.handle.as_ptr()) }
    }
}
impl Clone for Array {
    fn clone(&self) -> Self {
        self.try_clone().expect("could not clone MLX array handle")
    }
}
impl Array {
    fn owned(ptr: *mut c_void) -> Result<Self> {
        let handle = NonNull::new(ptr).context("MLX returned a null array")?;
        let mut rank = 0;
        let mut dims = std::ptr::null();
        let mut code = 0;
        if let Err(error) =
            checked(unsafe { mlxl3_array_metadata(ptr, &mut rank, &mut dims, &mut code) })
        {
            unsafe { mlxl3_array_free(ptr) };
            return Err(error);
        }
        let dtype = match Dtype::from_code(code) {
            Ok(x) => x,
            Err(e) => {
                unsafe { mlxl3_array_free(ptr) };
                return Err(e);
            }
        };
        let shape = if rank == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(dims, rank) }.to_vec()
        };
        Ok(Self {
            handle,
            shape,
            dtype,
            _thread: PhantomData,
        })
    }
    fn output(f: impl FnOnce(*mut *mut c_void) -> i32) -> Result<Self> {
        let mut out = std::ptr::null_mut();
        checked(f(&mut out))?;
        Self::owned(out)
    }
    pub fn from_bytes(data: &[u8], shape: &[i32], dtype: Dtype) -> Result<Self> {
        ensure!(
            bytes_for(shape, dtype)? == data.len(),
            "tensor byte count does not match shape and dtype"
        );
        if dtype == Dtype::Bool {
            ensure!(data.iter().all(|&x| x <= 1), "invalid boolean byte");
        }
        initialize()?;
        Self::output(|out| unsafe {
            mlxl3_array_from_bytes(
                data.as_ptr(),
                data.len(),
                shape.as_ptr(),
                shape.len(),
                dtype as i32,
                out,
            )
        })
    }
    pub fn from_file(file: &File, offset: u64, shape: &[i32], dtype: Dtype) -> Result<Self> {
        bytes_for(shape, dtype)?;
        initialize()?;
        Self::output(|out| unsafe {
            mlxl3_array_from_file(
                file.as_raw_fd(),
                offset,
                shape.as_ptr(),
                shape.len(),
                dtype as i32,
                out,
            )
        })
    }
    pub fn from_files(
        files: &[&File],
        offsets: &[u64],
        shapes: &[Vec<i32>],
        dtypes: &[Dtype],
        workers: usize,
    ) -> Result<Vec<Self>> {
        let count = files.len();
        ensure!(
            count == offsets.len() && count == shapes.len() && count == dtypes.len(),
            "batched tensor metadata length mismatch"
        );
        if count == 0 {
            return Ok(Vec::new());
        }
        let fds = files
            .iter()
            .map(|file| file.as_raw_fd())
            .collect::<Vec<_>>();
        let ranks = shapes.iter().map(Vec::len).collect::<Vec<_>>();
        let dims = shapes.iter().flatten().copied().collect::<Vec<_>>();
        let types = dtypes.iter().map(|&dtype| dtype as i32).collect::<Vec<_>>();
        for (shape, &dtype) in shapes.iter().zip(dtypes) {
            bytes_for(shape, dtype)?;
            ensure!(
                dtype != Dtype::Bool,
                "batched boolean tensors are unsupported"
            );
        }
        initialize()?;
        let mut pointers = vec![std::ptr::null_mut(); count];
        checked(unsafe {
            mlxl3_arrays_from_files(
                fds.as_ptr(),
                offsets.as_ptr(),
                dims.as_ptr(),
                ranks.as_ptr(),
                types.as_ptr(),
                count,
                workers,
                pointers.as_mut_ptr(),
            )
        })?;
        let mut arrays = Vec::with_capacity(count);
        for (index, pointer) in pointers.iter().copied().enumerate() {
            match Self::owned(pointer) {
                Ok(array) => arrays.push(array),
                Err(error) => {
                    for &remaining in &pointers[index + 1..] {
                        unsafe { mlxl3_array_free(remaining) };
                    }
                    return Err(error);
                }
            }
        }
        Ok(arrays)
    }
    pub fn from_f16_bits(data: &[u16], shape: &[i32]) -> Result<Self> {
        Self::from_words(data, shape, Dtype::Float16)
    }
    pub fn from_u16(data: &[u16], shape: &[i32]) -> Result<Self> {
        Self::from_words(data, shape, Dtype::UInt16)
    }
    pub fn from_u32(data: &[u32], shape: &[i32]) -> Result<Self> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data))
        };
        Self::from_bytes(bytes, shape, Dtype::UInt32)
    }
    fn from_words(data: &[u16], shape: &[i32], dtype: Dtype) -> Result<Self> {
        // u16 has no padding or invalid bit patterns; copying preserves FP16 payload bits.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data))
        };
        Self::from_bytes(bytes, shape, dtype)
    }
    pub fn from_f32(data: &[f32], shape: &[i32]) -> Result<Self> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data))
        };
        Self::from_bytes(bytes, shape, Dtype::Float32)
    }
    pub fn from_i32(data: &[i32], shape: &[i32]) -> Result<Self> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data))
        };
        Self::from_bytes(bytes, shape, Dtype::Int32)
    }
    pub fn zeros(shape: &[i32]) -> Result<Self> {
        Self::zeros_dtype(shape, Dtype::Float32)
    }
    pub fn zeros_dtype(shape: &[i32], dtype: Dtype) -> Result<Self> {
        bytes_for(shape, dtype)?;
        initialize()?;
        Self::output(|out| unsafe {
            mlxl3_array_zeros(shape.as_ptr(), shape.len(), dtype as i32, out)
        })
    }
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
    pub fn byte_len(&self) -> Result<usize> {
        bytes_for(&self.shape, self.dtype)
    }
    /// Full backing allocation, including storage outside a sliced view.
    /// Evaluates this array; only use at cache/request boundaries.
    pub fn retained_bytes(&self) -> Result<usize> {
        let mut bytes = 0;
        checked(unsafe { mlxl3_array_retained_bytes(self.handle.as_ptr(), &mut bytes) })?;
        usize::try_from(bytes).context("retained array size overflow")
    }
    /// Detach wasteful views only at a saved prefill checkpoint, never per token.
    pub fn compact_for_cache(&self) -> Result<Self> {
        if std::env::var("MLXL3_CACHE_COMPACTION").as_deref() == Ok("0")
            || !crate::memory_policy::compact_wasteful_view(
                self.byte_len()?,
                self.retained_bytes()?,
            )
        {
            return self.try_clone();
        }
        Self::output(|out| unsafe { mlxl3_array_compact_copy(self.handle.as_ptr(), out) })
    }
    pub(crate) fn pack_embedding(&self) -> Result<Vec<Self>> {
        let mut pointers = [std::ptr::null_mut(); 3];
        checked(unsafe { mlxl3_pack_embedding(self.handle.as_ptr(), pointers.as_mut_ptr()) })?;
        let converted: Vec<_> = pointers.into_iter().map(Self::owned).collect();
        converted.into_iter().collect()
    }
    pub fn eval(&self) -> Result<()> {
        checked(unsafe { mlxl3_array_eval(self.handle.as_ptr()) })
    }
    /// Submit graphs together; reading an output still waits for completion.
    pub fn async_eval_all(inputs: &[&Self]) -> Result<()> {
        let pointers: Vec<_> = inputs.iter().map(|value| value.handle.as_ptr()).collect();
        checked(unsafe { mlxl3_arrays_async_eval(pointers.as_ptr(), pointers.len()) })
    }
    pub fn try_clone(&self) -> Result<Self> {
        Self::output(|out| unsafe { mlxl3_array_clone(self.handle.as_ptr(), out) })
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = vec![0; bytes_for(&self.shape, self.dtype)?];
        checked(unsafe {
            mlxl3_array_copy_bytes(self.handle.as_ptr(), bytes.as_mut_ptr(), bytes.len())
        })?;
        Ok(bytes)
    }
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        let bytes = if self.dtype == Dtype::Float32 {
            self.to_bytes()?
        } else {
            self.astype(Dtype::Float32)?.to_bytes()?
        };
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_ne_bytes(*x))
            .collect())
    }
    pub fn to_u32(&self) -> Result<Vec<u32>> {
        ensure!(self.dtype == Dtype::UInt32, "tensor is not UInt32");
        Ok(self
            .to_bytes()?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| u32::from_ne_bytes(*x))
            .collect())
    }
    pub fn to_f16_bits(&self) -> Result<Vec<u16>> {
        let bytes = if self.dtype == Dtype::Float16 {
            self.to_bytes()?
        } else {
            self.astype(Dtype::Float16)?.to_bytes()?
        };
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|x| u16::from_ne_bytes(*x))
            .collect())
    }
    fn unary(&self, op: i32, args: &[i32], scalar: f32, flag: i32) -> Result<Self> {
        Self::output(|out| unsafe {
            mlxl3_array_unary(
                self.handle.as_ptr(),
                op,
                args.as_ptr(),
                args.len(),
                scalar,
                flag,
                out,
            )
        })
    }
    fn binary(&self, other: &Self, op: i32, arg: i32, scalar: f32) -> Result<Self> {
        Self::output(|out| unsafe {
            mlxl3_array_binary(
                self.handle.as_ptr(),
                other.handle.as_ptr(),
                op,
                arg,
                scalar,
                out,
            )
        })
    }
    pub fn reshape(&self, shape: &[i32]) -> Result<Self> {
        self.unary(0, shape, 0., 0)
    }
    pub fn transpose(&self, axes: &[i32]) -> Result<Self> {
        self.unary(1, axes, 0., 0)
    }
    pub fn slice(&self, axis: i32, start: i32, end: i32) -> Result<Self> {
        self.unary(2, &[axis, start, end], 0., 0)
    }
    pub fn astype(&self, dtype: Dtype) -> Result<Self> {
        self.unary(3, &[], 0., dtype as i32)
    }
    pub fn astype_f16(&self) -> Result<Self> {
        self.astype(Dtype::Float16)
    }
    pub fn hadamard_transform(&self, scale: Option<f32>) -> Result<Self> {
        self.unary(4, &[], scale.unwrap_or(0.), i32::from(scale.is_some()))
    }
    pub fn sigmoid(&self) -> Result<Self> {
        self.unary(5, &[], 0., 0)
    }
    pub fn sum(&self, axis: i32, keepdims: bool) -> Result<Self> {
        self.unary(6, &[axis], 0., i32::from(keepdims))
    }
    pub fn view(&self, dtype: Dtype) -> Result<Self> {
        self.unary(7, &[], 0., dtype as i32)
    }
    pub fn rope(&self, dims: i32, base: f32, offset: i32) -> Result<Self> {
        self.unary(8, &[dims, offset], base, 0)
    }
    pub fn rope_with_freqs(&self, dims: i32, offset: i32, freqs: &Self) -> Result<Self> {
        ensure!(
            dims > 0 && dims % 2 == 0 && freqs.shape() == [dims / 2],
            "invalid RoPE frequency dimensions"
        );
        Self::output(|out| unsafe {
            mlxl3_array_rope_freqs(
                self.handle.as_ptr(),
                freqs.handle.as_ptr(),
                dims,
                offset,
                out,
            )
        })
    }
    pub fn log_probs(&self) -> Result<Self> {
        self.unary(9, &[], 0., 0)
    }
    /// The chat bridge's greedy contract: normalize before argmax so BF16/FP16
    /// rounding and tie-breaking match ordinary generation and DFlash verify.
    pub fn chat_greedy_ids(&self) -> Result<Vec<u32>> {
        self.log_probs()?.argmax()?.to_u32()
    }
    pub fn softmax_precise(&self) -> Result<Self> {
        self.unary(10, &[], 0., 0)
    }
    pub fn softmax(&self) -> Result<Self> {
        self.unary(17, &[], 0., 0)
    }
    pub fn negative(&self) -> Result<Self> {
        self.unary(11, &[], 0., 0)
    }
    pub fn exp(&self) -> Result<Self> {
        self.unary(12, &[], 0., 0)
    }
    pub fn rms_norm_without_weight(&self, eps: f32) -> Result<Self> {
        self.unary(13, &[], eps, 0)
    }
    pub fn silu(&self) -> Result<Self> {
        self.unary(14, &[], 0., 0)
    }
    pub fn tanh(&self) -> Result<Self> {
        self.unary(15, &[], 0., 0)
    }
    pub fn scalar_mul(&self, scalar: f32) -> Result<Self> {
        ensure!(scalar.is_finite(), "array scalar must be finite");
        self.unary(16, &[], scalar, 0)
    }
    pub fn softcap(&self, scalar: f32) -> Result<Self> {
        ensure!(scalar.is_finite() && scalar > 0.0, "invalid softcap");
        self.unary(18, &[], scalar, 0)
    }
    pub fn argmax(&self) -> Result<Self> {
        self.unary(19, &[], 0., 0)
    }
    pub fn broadcast_to(&self, shape: &[i32]) -> Result<Self> {
        ensure!(shape.iter().all(|&x| x >= 0), "invalid broadcast shape");
        self.unary(20, shape, 0., 0)
    }
    pub fn argsort(&self) -> Result<Self> {
        self.unary(21, &[], 0., 0)
    }
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.binary(other, 0, 0, 0.)
    }
    pub fn mul(&self, other: &Self) -> Result<Self> {
        self.binary(other, 1, 0, 0.)
    }
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        self.binary(other, 2, 0, 0.)
    }

    pub(crate) fn affine4(
        &self,
        weight: &Self,
        scales: &Self,
        biases: &Self,
        indices: Option<&Self>,
    ) -> Result<Self> {
        Self::output(|out| unsafe {
            mlxl3_array_affine4(
                self.handle.as_ptr(),
                weight.handle.as_ptr(),
                scales.handle.as_ptr(),
                biases.handle.as_ptr(),
                indices.map_or(std::ptr::null_mut(), |a| a.handle.as_ptr()),
                out,
            )
        })
    }
    pub fn take(&self, indices: &Self, axis: i32) -> Result<Self> {
        self.binary(indices, 3, axis, 0.)
    }
    pub fn rms_norm(&self, weight: &Self, eps: f32) -> Result<Self> {
        self.binary(weight, 4, 0, eps)
    }
    pub fn conv1d(&self, weight: &Self, groups: i32) -> Result<Self> {
        self.binary(weight, 5, groups, 0.)
    }
    pub(crate) fn add_rms_norm(
        &self,
        residual: &Self,
        weight: &Self,
        eps: f32,
    ) -> Result<(Self, Self)> {
        let fallback = || {
            let h = self.add(residual)?;
            let normalized = h.rms_norm(weight, eps)?;
            Ok((h, normalized))
        };
        let Some((&width, leading)) = self.shape().split_last() else {
            return fallback();
        };
        let rows = leading.iter().try_fold(1i32, |n, &dim| n.checked_mul(dim));
        let Some((threads, grid)) =
            rows.and_then(|rows| contracts::mtp_add_norm_launch(rows, width))
        else {
            return fallback();
        };
        if self.shape() != residual.shape()
            || weight.shape() != [width]
            || self.dtype() != Dtype::Float16
            || residual.dtype() != Dtype::Float16
            || weight.dtype() != Dtype::Float16
            || !eps.is_finite()
            || eps <= 0.
        {
            return fallback();
        }
        let epsilon = Array::from_f32(&[eps], &[1])?;
        let mut result = metal_kernel(
            &format!("mlxl3_mtp_add_norm_{width}"),
            &["x", "residual", "weight", "eps"],
            &["h", "normed"],
            &format!("using namespace metal;\n#define AXIS {width}u\n#define THREADS {threads}u\n"),
            include_str!("../shaders/mtp_add_rms.metal"),
            &[self, residual, weight, &epsilon],
            &[self.shape().to_vec(), self.shape().to_vec()],
            &[Dtype::Float16, Dtype::Float16],
            [grid, 1, 1],
            [threads, 1, 1],
        )?;
        let normed = result.pop().context("missing fused RMSNorm output")?;
        Ok((
            result.pop().context("missing fused residual output")?,
            normed,
        ))
    }

    pub fn swiglu(&self, up: &Self) -> Result<Self> {
        self.binary(up, 6, 0, 0.)
    }
    pub fn logaddexp(&self, other: &Self) -> Result<Self> {
        self.binary(other, 7, 0, 0.)
    }
    pub fn precise_swiglu(&self, value: &Self) -> Result<Self> {
        self.binary(value, 8, 0, 0.)
    }
    pub fn div(&self, other: &Self) -> Result<Self> {
        self.binary(other, 9, 0, 0.)
    }
    pub fn geglu(&self, up: &Self) -> Result<Self> {
        self.binary(up, 10, 0, 0.)
    }
    pub fn concatenate(inputs: &[&Self], axis: i32) -> Result<Self> {
        ensure!(!inputs.is_empty(), "concatenate requires an input");
        let pointers: Vec<_> = inputs.iter().map(|x| x.handle.as_ptr()).collect();
        Self::output(|out| unsafe {
            mlxl3_array_concatenate(pointers.as_ptr(), pointers.len(), axis, out)
        })
    }
    pub fn sdpa(q: &Self, k: &Self, v: &Self, scale: f32, causal: bool) -> Result<Self> {
        Self::output(|out| unsafe {
            mlxl3_array_sdpa(
                q.handle.as_ptr(),
                k.handle.as_ptr(),
                v.handle.as_ptr(),
                scale,
                i32::from(causal),
                out,
            )
        })
    }
    pub fn sdpa_mask(q: &Self, k: &Self, v: &Self, scale: f32, mask: &Self) -> Result<Self> {
        Self::output(|out| unsafe {
            mlxl3_array_sdpa_mask(
                q.handle.as_ptr(),
                k.handle.as_ptr(),
                v.handle.as_ptr(),
                mask.handle.as_ptr(),
                scale,
                out,
            )
        })
    }
}
pub fn concatenate(inputs: &[&Array], axis: i32) -> Result<Array> {
    Array::concatenate(inputs, axis)
}

#[allow(clippy::too_many_arguments)]
pub fn metal_kernel(
    name: &str,
    input_names: &[&str],
    output_names: &[&str],
    header: &str,
    source: &str,
    inputs: &[&Array],
    output_shapes: &[Vec<i32>],
    output_dtypes: &[Dtype],
    grid: [i32; 3],
    threadgroup: [i32; 3],
) -> Result<Vec<Array>> {
    ensure!(
        input_names.len() == inputs.len(),
        "kernel input name count mismatch"
    );
    ensure!(
        !output_names.is_empty()
            && output_names.len() == output_shapes.len()
            && output_names.len() == output_dtypes.len(),
        "kernel output metadata count mismatch"
    );
    ensure!(
        grid.into_iter().chain(threadgroup).all(|x| x > 0),
        "kernel launch dimensions must be positive"
    );
    for (s, dt) in output_shapes.iter().zip(output_dtypes) {
        bytes_for(s, *dt)?;
    }
    let name = CString::new(name)?;
    let header = CString::new(header)?;
    let source = CString::new(source)?;
    let input_strings = input_names
        .iter()
        .map(|x| CString::new(*x))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let output_strings = output_names
        .iter()
        .map(|x| CString::new(*x))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let in_names: Vec<_> = input_strings.iter().map(|x| x.as_ptr()).collect();
    let out_names: Vec<_> = output_strings.iter().map(|x| x.as_ptr()).collect();
    let in_ptrs: Vec<_> = inputs.iter().map(|x| x.handle.as_ptr()).collect();
    let dims: Vec<i32> = output_shapes.iter().flatten().copied().collect();
    let ranks: Vec<usize> = output_shapes.iter().map(Vec::len).collect();
    let dtypes: Vec<i32> = output_dtypes.iter().map(|x| *x as i32).collect();
    let mut pointers = vec![std::ptr::null_mut(); output_shapes.len()];
    initialize()?;
    checked(unsafe {
        mlxl3_metal_kernel(
            name.as_ptr(),
            in_names.as_ptr(),
            out_names.as_ptr(),
            header.as_ptr(),
            source.as_ptr(),
            in_ptrs.as_ptr(),
            in_ptrs.len(),
            dims.as_ptr(),
            ranks.as_ptr(),
            dtypes.as_ptr(),
            pointers.len(),
            grid.as_ptr(),
            threadgroup.as_ptr(),
            pointers.as_mut_ptr(),
        )
    })?;
    // Take ownership of every returned handle even if one metadata read fails.
    let converted: Vec<_> = pointers.into_iter().map(Array::owned).collect();
    converted.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "physical Apple GPU; run alone, compaction enabled"]
    fn compact_cache_views_preserve_all_bits_and_release_parent_storage() -> Result<()> {
        for dtype in [
            Dtype::Float16,
            Dtype::BFloat16,
            Dtype::Float32,
            Dtype::UInt32,
        ] {
            let pattern = (0..65_536u32)
                .flat_map(|v| {
                    if dtype.item_size() == 2 {
                        (v as u16).to_ne_bytes().to_vec()
                    } else {
                        (v.rotate_left(13) ^ 0xdead_beef).to_ne_bytes().to_vec()
                    }
                })
                .collect::<Vec<_>>();
            for strided in [false, true] {
                let view = if strided {
                    let mut bytes = Vec::with_capacity(pattern.len() * 32);
                    for element in pattern.chunks_exact(dtype.item_size()) {
                        for _ in 0..32 {
                            bytes.extend_from_slice(element);
                        }
                    }
                    Array::from_bytes(&bytes, &[65_536, 32], dtype)?.slice(1, 31, 32)?
                } else {
                    Array::from_bytes(&pattern.repeat(32), &[32, 65_536], dtype)?
                        .slice(0, 19, 20)?
                };
                let before = view.retained_bytes()?;
                let compact = view.compact_for_cache()?;
                assert_eq!(compact.dtype(), dtype);
                assert_eq!(compact.shape(), view.shape());
                assert_eq!(
                    compact.to_bytes()?,
                    pattern,
                    "dtype={dtype:?}, strided={strided}"
                );
                assert!(compact.retained_bytes()? <= view.byte_len()? * 2);
                assert!(compact.retained_bytes()? < before);
                drop(view);
                assert_eq!(compact.to_bytes()?, pattern);
            }
        }
        let dense = Array::from_f32(&[0., -0., 1., -1.], &[4])?;
        let duplicate = dense.compact_for_cache()?;
        assert_eq!(dense.to_bytes()?, duplicate.to_bytes()?);
        assert_eq!(dense.retained_bytes()?, duplicate.retained_bytes()?);
        let mut out = std::ptr::null_mut();
        assert!(
            checked(unsafe { mlxl3_array_compact_copy(std::ptr::null_mut(), &mut out) }).is_err()
        );
        assert!(
            checked(unsafe {
                mlxl3_array_compact_copy(dense.handle.as_ptr(), std::ptr::null_mut())
            })
            .is_err()
        );
        Ok(())
    }
    #[test]
    #[ignore = "physical Apple GPU; run alone with MTL_CAPTURE_ENABLED=1"]
    fn capture_guard_closes_success_and_drop_without_overwriting() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let input = Array::from_f32(&[1., 7., -3., 2.], &[4])?;
        input.add(&input)?.eval()?;
        input.mul(&input)?.eval()?;
        for index in 0..3 {
            let path = directory.path().join(format!("capture-{index}.gputrace"));
            let capture = MetalCapture::start(&path)?;
            assert_eq!(input.add(&input)?.to_f32()?, [2., 14., -6., 4.]);
            if index == 1 {
                drop(capture);
            } else {
                capture.finish()?;
            }
            assert!(
                path.is_dir() && std::fs::read_dir(&path)?.next().is_some(),
                "empty capture artifact"
            );
            assert!(MetalCapture::start(&path).is_err());
        }
        assert!(MetalCapture::start(&directory.path().join("wrong.txt")).is_err());
        assert!(MetalCapture::start(&directory.path().join("missing/trace.gputrace")).is_err());
        assert!(checked(unsafe { mlxl3_start_capture(std::ptr::null()) }).is_err());
        Ok(())
    }
    #[test]
    #[ignore = "physical Apple GPU; run alone with MLXL3_ALLOCATOR_CACHE_MIB=0 or 1"]
    fn allocator_cache_limit_preserves_live_arrays() -> Result<()> {
        let limit = std::env::var("MLXL3_ALLOCATOR_CACHE_MIB")?.parse::<usize>()?;
        ensure!(limit <= 1, "run this check with cache limit0 or1");
        let input = Array::from_f16_bits(&vec![0x3c00; 512 * 512], &[512, 512])?;
        let live = input.try_clone()?;
        for _ in 0..3 {
            input.add(&input)?.eval()?;
        }
        synchronize()?;
        assert!(memory_stats(false)?.mlx_cache_bytes <= (limit * 1_048_576) as u64);
        assert!(live.to_f16_bits()?.iter().all(|&word| word == 0x3c00));
        Ok(())
    }
    #[test]
    #[ignore = "physical Apple GPU; run alone"]
    fn retained_bytes_include_storage_outside_view() -> Result<()> {
        let view = {
            let parent = Array::from_f16_bits(&vec![0x3c00; 512 * 512], &[512, 512])?;
            parent.slice(0, 511, 512)?.slice(1, 0, 1)?
        };
        assert_eq!(view.byte_len()?, 2);
        assert!(view.retained_bytes()? >= 512 * 512 * 2);
        assert_eq!(view.to_f16_bits()?, [0x3c00]);
        Ok(())
    }

    #[test]
    #[ignore = "requires physical Apple GPU; run alone"]
    fn async_group_retains_parents_and_rejects_null() -> Result<()> {
        let (first, second) = {
            let input = Array::from_u32(&[1, 7, 19, u32::MAX], &[2, 2])?;
            let first = input.transpose(&[1, 0])?;
            let second = input.reshape(&[4])?;
            Array::async_eval_all(&[&first, &second])?;
            (first, second)
        };
        assert_eq!(first.shape(), &[2, 2]);
        assert_eq!(first.to_u32()?, [1, 19, 7, u32::MAX]);
        assert_eq!(second.shape(), &[4]);
        assert_eq!(second.to_u32()?, [1, 7, 19, u32::MAX]);
        Array::async_eval_all(&[])?;
        let empty = Array::from_u32(&[], &[0])?;
        Array::async_eval_all(&[&empty])?;
        assert!(empty.to_u32()?.is_empty());
        assert!(checked(unsafe { mlxl3_arrays_async_eval(std::ptr::null(), 1) }).is_err());
        let invalid = [first.handle.as_ptr(), std::ptr::null_mut()];
        assert!(checked(unsafe { mlxl3_arrays_async_eval(invalid.as_ptr(), 2) }).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple GPU; run alone, optional MLXL3_NORM_BENCH=1 after parity"]
    fn fused_mtp_norm_matches_stock_outputs_and_fallbacks() -> Result<()> {
        use half::f16;
        let evaluate = |pair: (Array, Array)| -> Result<()> {
            pair.1.eval()?;
            pair.0.eval()
        };
        let mut checked = 0usize;
        for width in [64, 2048, 5120] {
            for rows in [1, 2, 3, 4, 5, 24] {
                for seed in [0, 1, 17, 123] {
                    let x = (0..rows * width)
                        .map(|i| {
                            if seed == 0 {
                                0
                            } else {
                                let bits = (i as u32).wrapping_mul(1777).wrapping_add(seed) as u16;
                                // Finite FP16 extremes, subnormals, both signs;
                                // keep the addition finite too, with the sign/mantissa.
                                (bits & 0x83ff) | (bits & 0x7400)
                            }
                        })
                        .collect::<Vec<_>>();
                    let residual = x
                        .iter()
                        .enumerate()
                        .map(|(i, &bits)| {
                            if seed == 17 {
                                bits ^ 0x8000
                            } else {
                                f16::from_f32(((i % 251) as f32 - 125.) / 128.).to_bits()
                            }
                        })
                        .collect::<Vec<_>>();
                    let weight = (0..width)
                        .map(|i| f16::from_f32(0.5 + (i % 17) as f32 / 16.).to_bits())
                        .collect::<Vec<_>>();
                    let x = Array::from_f16_bits(&x, &[1, rows, width])?;
                    let residual = Array::from_f16_bits(&residual, &[1, rows, width])?;
                    let weight = Array::from_f16_bits(&weight, &[width])?;
                    let expected = x.add(&residual)?;
                    let normalized = expected.rms_norm(&weight, 1e-6)?;
                    let (actual, actual_norm) = x.add_rms_norm(&residual, &weight, 1e-6)?;
                    for (a, b) in [(&actual, &expected), (&actual_norm, &normalized)] {
                        let a = a.to_f16_bits()?;
                        let b = b.to_f16_bits()?;
                        assert_eq!(a.len(), (rows * width) as usize);
                        assert!(a.iter().chain(&b).all(|&x| f16::from_bits(x).is_finite()));
                        let differences = a.iter().zip(&b).filter(|(a, b)| a != b).count();
                        assert_eq!(
                            differences, 0,
                            "fused norm rows={rows} width={width} seed={seed}"
                        );
                        checked += a.len();
                    }
                    // Unsupported FP32 follows the original MLX expression.
                    let x32 = x.astype(Dtype::Float32)?;
                    let r32 = residual.astype(Dtype::Float32)?;
                    let w32 = weight.astype(Dtype::Float32)?;
                    let (h32, n32) = x32.add_rms_norm(&r32, &w32, 1e-6)?;
                    let expected32 = x32.add(&r32)?;
                    assert_eq!(h32.to_bytes()?, expected32.to_bytes()?);
                    assert_eq!(
                        n32.to_bytes()?,
                        expected32.rms_norm(&w32, 1e-6)?.to_bytes()?
                    );
                }
            }
        }
        eprintln!(
            "Fused norm: {checked} finite FP16 output words exact, FP32/unsupported shape fallbacks exact"
        );
        if std::env::var("MLXL3_NORM_BENCH").as_deref() == Ok("1") {
            for width in [2048, 5120] {
                for rows in 1..=4 {
                    let x = Array::from_f32(
                        &(0..rows * width)
                            .map(|i| ((i % 251) as f32 - 125.) / 128.)
                            .collect::<Vec<_>>(),
                        &[1, rows, width],
                    )?
                    .astype(Dtype::Float16)?;
                    let residual = x.scalar_mul(0.75)?;
                    let weight = Array::from_f32(&vec![1.; width as usize], &[width])?
                        .astype(Dtype::Float16)?;
                    let operation = |fused| -> Result<(Array, Array)> {
                        if fused {
                            x.add_rms_norm(&residual, &weight, 1e-6)
                        } else {
                            let h = x.add(&residual)?;
                            let n = h.rms_norm(&weight, 1e-6)?;
                            Ok((h, n))
                        }
                    };
                    for _ in 0..10 {
                        evaluate(operation(false)?)?;
                        evaluate(operation(true)?)?;
                    }
                    let mut timings = [Vec::new(), Vec::new()];
                    for repeat in 0..60 {
                        for fused in if repeat % 2 == 0 {
                            [false, true]
                        } else {
                            [true, false]
                        } {
                            let start = std::time::Instant::now();
                            evaluate(operation(fused)?)?;
                            timings[usize::from(fused)].push(start.elapsed().as_secs_f64() * 1000.);
                        }
                    }
                    for series in &mut timings {
                        series.sort_by(f64::total_cmp);
                    }
                    eprintln!(
                        "Fused norm width={width} rows={rows}: stock_ms={} fused_ms={} stock_samples={:?} fused_samples={:?}",
                        timings[0][30], timings[1][30], timings[0], timings[1]
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires physical Apple GPU; run alone (global allocator cache)"]
    fn clear_cache_preserves_live_arrays_and_releases_unused_buffers() -> Result<()> {
        let live = Array::from_f32(&[1.0, 2.0, 3.0], &[3])?;
        live.eval()?;
        {
            // zeros() can remain a scalar broadcast without a 32 MiB buffer.
            let temporary = Array::from_f32(&vec![3.25; 8 * 1024 * 1024], &[8 * 1024 * 1024])?;
            temporary.eval()?;
        }
        let before = memory_stats(false)?;
        assert!(before.mlx_cache_bytes >= 32 * 1024 * 1024);
        release_idle_cache_with_policy(true)?;
        let after = memory_stats(false)?;
        assert_eq!(after.mlx_active_bytes, before.mlx_active_bytes);
        assert_eq!(after.mlx_cache_bytes, 0);
        assert_eq!(live.to_f32()?, [1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    #[ignore = "requires Apple GPU and a temporary 2 GiB allocation"]
    fn checkpoint_reads_larger_than_darwin_syscall_limit() -> Result<()> {
        use std::os::unix::fs::FileExt;

        let file = tempfile::tempfile()?;
        let elements = i32::MAX / 4 + 2;
        let offset = 3u64;
        file.set_len(offset + elements as u64 * 4)?;
        let markers = [
            0,
            64 * 1024 * 1024 / 4 - 1,
            64 * 1024 * 1024 / 4,
            elements - 1,
        ];
        for (number, &index) in markers.iter().enumerate() {
            file.write_all_at(
                &(number as u32 + 1).to_ne_bytes(),
                offset + index as u64 * 4,
            )?;
        }
        for batched in [false, true] {
            let array = if batched {
                Array::from_files(&[&file], &[offset], &[vec![elements]], &[Dtype::UInt32], 1)?
                    .remove(0)
            } else {
                Array::from_file(&file, offset, &[elements], Dtype::UInt32)?
            };
            for (number, &index) in markers.iter().enumerate() {
                assert_eq!(
                    array.slice(0, index, index + 1)?.to_u32()?,
                    vec![number as u32 + 1]
                );
            }
            assert_eq!(array.slice(0, 1, 2)?.to_u32()?, vec![0]);
        }
        assert!(Array::from_file(&file, u64::MAX, &[1], Dtype::UInt32).is_err());
        assert!(
            Array::from_file(&file, offset + elements as u64 * 4 - 1, &[1], Dtype::UInt32).is_err()
        );
        Ok(())
    }

    #[test]
    fn validates_before_entering_mlx() {
        assert!(Array::from_bytes(&[0], &[1], Dtype::Float32).is_err());
        assert!(Array::from_bytes(&[], &[-1], Dtype::Float32).is_err());
        assert!(Array::from_bytes(&[2], &[1], Dtype::Bool).is_err());
        assert!(bytes_for(&[i32::MAX; 4], Dtype::Float64).is_err());
        assert!(Array::concatenate(&[], 0).is_err());
    }
    #[test]
    #[ignore = "requires Apple GPU; run explicitly outside sandbox"]
    fn native_array_and_kernel_smoke() -> Result<()> {
        use std::io::Write;

        let mut file = tempfile::tempfile()?;
        file.write_all(&[9, 9, 1, 0, 0, 0, 2, 0, 0, 0])?;
        assert_eq!(
            Array::from_file(&file, 2, &[2], Dtype::UInt32)?.to_u32()?,
            vec![1, 2]
        );
        let arrays = Array::from_files(
            &[&file, &file],
            &[2, 6],
            &[vec![1], vec![1]],
            &[Dtype::UInt32, Dtype::UInt32],
            2,
        )?;
        assert_eq!(arrays[0].to_u32()?, vec![1]);
        assert_eq!(arrays[1].to_u32()?, vec![2]);
        let a = Array::from_f32(&[1., 2., 3., 4.], &[2, 2])?;
        let b = a.transpose(&[1, 0])?;
        assert_eq!(b.to_f32()?, vec![1., 3., 2., 4.]);
        assert_eq!(a.matmul(&b)?.to_f32()?, vec![5., 11., 11., 25.]);
        let gate = Array::from_f32(&[0., 1.], &[2])?;
        let value = Array::from_f32(&[2., 3.], &[2])?;
        let geglu = gate.geglu(&value)?.to_f32()?;
        assert_eq!(geglu[0], 0.);
        assert!((geglu[1] - 2.5236).abs() < 0.001);
        assert!((gate.scalar_mul(2.)?.tanh()?.to_f32()?[1] - 0.964).abs() < 0.001);
        assert_eq!(a.argmax()?.to_u32()?, vec![1, 1]);
        assert_eq!(a.chat_greedy_ids()?, vec![1, 1]);
        let tied = Array::from_f32(&[4., 4., -1., -2., 1., 1.], &[2, 3])?;
        for dtype in [Dtype::Float16, Dtype::BFloat16, Dtype::Float32] {
            assert_eq!(tied.astype(dtype)?.chat_greedy_ids()?, vec![0, 1]);
        }
        let memory = memory_stats(true)?;
        assert!(memory.mlx_peak_bytes >= memory.mlx_active_bytes);
        assert!(memory.process_footprint_bytes.unwrap_or(0) > 0);
        assert!(memory.process_lifetime_peak_bytes >= memory.process_footprint_bytes);
        assert_eq!(
            Array::from_u32(&[2, 0, 1], &[3])?.argsort()?.to_u32()?,
            vec![1, 2, 0]
        );
        assert_eq!(
            Array::from_f32(&[1., 2.], &[1, 2])?
                .broadcast_to(&[2, 2])?
                .to_f32()?,
            vec![1., 2., 1., 2.]
        );
        let rope_input = Array::from_f32(&[1., 2., 3., 4.], &[1, 1, 1, 4])?;
        let frequencies = Array::from_f32(&[1., 100.], &[2])?;
        let regular = rope_input.rope(4, 10_000., 3)?.to_f32()?;
        let explicit = rope_input.rope_with_freqs(4, 3, &frequencies)?.to_f32()?;
        assert!(
            regular
                .iter()
                .zip(explicit)
                .all(|(left, right)| (left - right).abs() < 1e-6)
        );
        let half = Array::from_f16_bits(&[0x3c00, 0x8000, 0x7e01, 0x7c00], &[4])?;
        assert_eq!(
            half.clone().to_f16_bits()?,
            vec![0x3c00, 0x8000, 0x7e01, 0x7c00]
        );
        let result = metal_kernel(
            "mlxl3_smoke",
            &["inp"],
            &["out"],
            "",
            "uint i = thread_position_in_grid.x; out[i] = inp[i] * 2.0f;",
            &[&a],
            &[vec![2, 2]],
            &[Dtype::Float32],
            [4, 1, 1],
            [4, 1, 1],
        )?;
        assert_eq!(result[0].to_f32()?, vec![2., 4., 6., 8.]);
        assert!(a.reshape(&[3]).is_err());
        assert_eq!(a.to_f32()?, vec![1., 2., 3., 4.]);
        Ok(())
    }
}

#[cfg(kani)]
mod verification {
    use super::*;
    #[kani::proof]
    fn async_group_preserves_values() {
        let values: [u32; 2] = kani::any();
        let input = Array::from_u32(&values, &[2]).unwrap();
        let other = input.reshape(&[1, 2]).unwrap();
        Array::async_eval_all(&[&input, &other]).unwrap();
        assert_eq!(input.to_u32().unwrap(), values);
        assert_eq!(other.to_u32().unwrap(), values);
    }
}
