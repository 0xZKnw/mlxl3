//! Direct Metal ownership. No MLX/Python runtime; synchronous reference API.
use crate::codec::{self, Codebook};
use anyhow::{Context, Result, ensure};
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

pub struct Metal {
    device: Device,
    queue: CommandQueue,
    pack_pipeline: ComputePipelineState,
    decode_pipeline: ComputePipelineState,
    codebook_pipeline: ComputePipelineState,
    qmv_pipeline: ComputePipelineState,
    inverse: Buffer,
}

impl Metal {
    pub fn new() -> Result<Self> {
        objc::rc::autoreleasepool(|| {
            let device = Device::system_default().context("no Metal device")?;
            let options = CompileOptions::new();
            options.set_fast_math_enabled(false);
            let library = device
                .new_library_with_source(include_str!("../shaders/exl3.metal"), &options)
                .map_err(|e| anyhow::anyhow!("Metal compilation failed: {e}"))?;
            let pipeline = |name| -> Result<_> {
                let function = library
                    .get_function(name, None)
                    .map_err(anyhow::Error::msg)?;
                device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(anyhow::Error::msg)
            };
            let pack_pipeline = pipeline("exl3_pack")?;
            let decode_pipeline = pipeline("exl3_decode")?;
            let codebook_pipeline = pipeline("exl3_codebook")?;
            let qmv_pipeline = pipeline("exl3_qmv")?;
            ensure!(
                qmv_pipeline.thread_execution_width() == 32
                    && qmv_pipeline.max_total_threads_per_threadgroup() >= 128,
                "EXL3 reference QMV requires 32-wide SIMD and 128 threads"
            );
            let perm: Vec<u16> = codec::permutation_inverse()
                .into_iter()
                .map(|i| i as u16)
                .collect();
            let inverse = Self::upload(&device, &perm);
            let queue = device.new_command_queue();
            Ok(Self {
                device,
                queue,
                pack_pipeline,
                decode_pipeline,
                codebook_pipeline,
                qmv_pipeline,
                inverse,
            })
        })
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    fn upload(device: &Device, data: &[u16]) -> Buffer {
        // Metal copies the bytes before returning. The Rust slice can then drop.
        device.new_buffer_with_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn output(&self, count: usize) -> Buffer {
        self.device
            .new_buffer((count * 2) as u64, MTLResourceOptions::StorageModeShared)
    }

    fn dispatch(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        params: [u32; 5],
        groups: usize,
        threads: usize,
    ) -> Result<()> {
        objc::rc::autoreleasepool(|| {
            let command = self.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            for (i, buffer) in buffers.iter().enumerate() {
                encoder.set_buffer(i as u64, Some(buffer), 0);
            }
            encoder.set_bytes(
                buffers.len() as u64,
                std::mem::size_of_val(&params) as u64,
                params.as_ptr().cast(),
            );
            encoder.dispatch_thread_groups(
                MTLSize::new(groups as u64, 1, 1),
                MTLSize::new(threads as u64, 1, 1),
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            ensure!(
                command.status() == MTLCommandBufferStatus::Completed,
                "Metal command failed ({:?})",
                command.status()
            );
            Ok(())
        })
    }

    fn read(buffer: &Buffer, count: usize) -> Vec<u16> {
        // Called only after a successful wait. Shared storage is CPU-coherent,
        // buffer length was allocated from this count, and u16 is POD/aligned.
        unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u16>(), count).to_vec() }
    }

    pub fn pack(&self, encoded: &[u16], k: usize) -> Result<Vec<u16>> {
        codec::check_k(k)?;
        ensure!(
            encoded.len().is_multiple_of(256),
            "expected whole 256-state tiles"
        );
        if encoded.is_empty() {
            return Ok(Vec::new());
        }
        let count = encoded.len() / 256 * 16 * k;
        let n = u32::try_from(count).context("Metal pack index overflow")?;
        ensure!(
            encoded.len() <= u32::MAX as usize,
            "Metal input index overflow"
        );
        let input = Self::upload(&self.device, encoded);
        let output = self.output(count);
        self.dispatch(
            &self.pack_pipeline,
            &[&input, &output],
            [n, k as u32, 0, 0, 0],
            count.div_ceil(256),
            256,
        )?;
        Ok(Self::read(&output, count))
    }

    pub fn decode(&self, packed: &[u16], k: usize, cb: Codebook) -> Result<(Vec<u16>, Vec<u16>)> {
        codec::check_k(k)?;
        ensure!(
            packed.len().is_multiple_of(16 * k),
            "expected whole packed tiles"
        );
        if packed.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let count = packed.len() / (16 * k) * 256;
        let n = u32::try_from(count).context("Metal decode index overflow")?;
        let input = Self::upload(&self.device, packed);
        let states = self.output(count);
        let values = self.output(count);
        self.dispatch(
            &self.decode_pipeline,
            &[&input, &states, &values],
            [n, k as u32, cb as u32, 0, 0],
            count.div_ceil(256),
            256,
        )?;
        Ok((Self::read(&states, count), Self::read(&values, count)))
    }

    pub fn codebook(&self, cb: Codebook) -> Result<Vec<u16>> {
        let out = self.output(65536);
        self.dispatch(
            &self.codebook_pipeline,
            &[&out],
            [65536, 0, cb as u32, 0, 0],
            256,
            256,
        )?;
        Ok(Self::read(&out, 65536))
    }

    /// Half-bit inputs and outputs; operates in the rotated EXL3 basis.
    pub fn qmv_inner(
        &self,
        x: &[u16],
        packed: &[u16],
        cols: usize,
        k: usize,
        cb: Codebook,
    ) -> Result<Vec<u16>> {
        codec::check_k(k)?;
        let rows = x.len();
        ensure!(
            rows > 0 && cols > 0 && rows.is_multiple_of(16) && cols.is_multiple_of(16),
            "expected positive 16-aligned dimensions"
        );
        let tiles = (rows / 16)
            .checked_mul(cols / 16)
            .context("QMV shape overflow")?;
        let size = tiles.checked_mul(16 * k).context("QMV size overflow")?;
        ensure!(
            packed.len() == size && size <= u32::MAX as usize,
            "invalid trellis size or Metal index overflow"
        );
        let rows = u32::try_from(rows)?;
        let n = u32::try_from(cols)?;
        let x = Self::upload(&self.device, x);
        let packed = Self::upload(&self.device, packed);
        let y = self.output(cols);
        self.dispatch(
            &self.qmv_pipeline,
            &[&x, &packed, &self.inverse, &y],
            [n, k as u32, cb as u32, rows, n],
            cols,
            128,
        )?;
        Ok(Self::read(&y, cols))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires physical Apple GPU; run cargo test -- --ignored"]
    fn metal_codec_matches_cpu() -> Result<()> {
        let gpu = Metal::new()?;
        for cb in [Codebook::Default, Codebook::Mcg, Codebook::Mul1] {
            let actual = gpu.codebook(cb)?;
            for word in 0..=u16::MAX {
                ensure!(
                    actual[word as usize] == codec::decode_codeword(word, cb).to_bits(),
                    "codebook mismatch {cb:?}/{word}"
                );
            }
            for k in 1..=8 {
                let input: Vec<_> = (0..1024).map(|i| (i * 701 + 33) as u16).collect();
                let packed = gpu.pack(&input, k)?;
                assert_eq!(packed, codec::pack(&input, k)?);
                let (states, values) = gpu.decode(&packed, k, cb)?;
                assert_eq!(states, codec::unpack(&packed, k)?);
                assert_eq!(
                    values,
                    states
                        .iter()
                        .map(|s| codec::decode_codeword(*s, cb).to_bits())
                        .collect::<Vec<_>>()
                );
                assert!(
                    gpu.qmv_inner(&[0; 32], &packed[..4 * 16 * k], 32, k, cb)?
                        .iter()
                        .all(|&bits| bits == 0)
                );
            }
        }
        Ok(())
    }
}
