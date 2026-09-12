//! Owned MLX arrays through a small C ABI. There is no Python runtime.
use anyhow::{Context, Result, bail, ensure};
use std::{
    ffi::{CStr, CString, c_char, c_void},
    marker::PhantomData,
    ptr::NonNull,
    rc::Rc,
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
    fn mlxl3_mlx_error() -> *const c_char;
    fn mlxl3_mlx_init(metallib: *const c_char) -> i32;
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
    fn mlxl3_array_zeros(dims: *const i32, rank: usize, dtype: i32, out: *mut *mut c_void) -> i32;
    fn mlxl3_array_eval(p: *mut c_void) -> i32;
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
    let root = std::env::var_os("MLXL3_MLX_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| env!("MLXL3_MLX_BUILD_ROOT").into());
    let library = root.join("lib/mlx.metallib");
    ensure!(
        library.is_file(),
        "MLX metallib missing: {}",
        library.display()
    );
    let path = CString::new(library.to_str().context("MLX path must be UTF-8")?)?;
    checked(unsafe { mlxl3_mlx_init(path.as_ptr()) })?;
    INITIALIZED.set(true);
    Ok(())
}
fn bytes_for(shape: &[i32], dtype: Dtype) -> Result<usize> {
    shape.iter().try_fold(dtype.item_size(), |size, &dim| {
        let dim = usize::try_from(dim).context("negative tensor dimension")?;
        size.checked_mul(dim).context("tensor byte size overflow")
    })
}
pub fn is_m5_gpu() -> Result<bool> {
    initialize()?;
    // device_info is declared in MLX headers but not exported by its dylib.
    let device = metal::Device::system_default().context("no Metal device")?;
    Ok(device.name() == "Apple M5" || device.name().starts_with("Apple M5 "))
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
    pub fn from_f16_bits(data: &[u16], shape: &[i32]) -> Result<Self> {
        Self::from_words(data, shape, Dtype::Float16)
    }
    pub fn from_u16(data: &[u16], shape: &[i32]) -> Result<Self> {
        Self::from_words(data, shape, Dtype::UInt16)
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
    pub fn eval(&self) -> Result<()> {
        checked(unsafe { mlxl3_array_eval(self.handle.as_ptr()) })
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
    pub fn log_probs(&self) -> Result<Self> {
        self.unary(9, &[], 0., 0)
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
    pub fn take(&self, indices: &Self, axis: i32) -> Result<Self> {
        self.binary(indices, 3, axis, 0.)
    }
    pub fn rms_norm(&self, weight: &Self, eps: f32) -> Result<Self> {
        self.binary(weight, 4, 0, eps)
    }
    pub fn conv1d(&self, weight: &Self, groups: i32) -> Result<Self> {
        self.binary(weight, 5, groups, 0.)
    }
    pub fn swiglu(&self, up: &Self) -> Result<Self> {
        self.binary(up, 6, 0, 0.)
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
        let a = Array::from_f32(&[1., 2., 3., 4.], &[2, 2])?;
        let b = a.transpose(&[1, 0])?;
        assert_eq!(b.to_f32()?, vec![1., 3., 2., 4.]);
        assert_eq!(a.matmul(&b)?.to_f32()?, vec![5., 11., 11., 25.]);
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
