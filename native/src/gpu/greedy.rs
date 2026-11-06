//! Canonical Half normalization stays on the device; only the selected index escapes.
use super::{MatrixView, Metal, buffer, upload};
use anyhow::{Context, Result, ensure};
use metal::{Buffer, CommandBufferRef, MTLResourceOptions as Storage, MTLSize};

impl Metal {
    pub(super) fn device_greedy_available(&self) -> bool {
        self.device_greedy_pipeline
            .as_ref()
            .is_some_and(|pipeline| {
                pipeline.thread_execution_width() == 32
                    && pipeline.max_total_threads_per_threadgroup() >= 1024
                    && self.logsumexp.thread_execution_width() == 32
                    && self.logsumexp.max_total_threads_per_threadgroup() >= 1024
            })
    }
    pub(super) fn encode_device_greedy(
        &self,
        cmd: &CommandBufferRef,
        logits: &Buffer,
    ) -> Result<Buffer> {
        self.encode_normalized_greedy(cmd, logits, false)
    }

    pub(super) fn encode_normalized_greedy(
        &self,
        cmd: &CommandBufferRef,
        logits: &Buffer,
        strict_raw_logits: bool,
    ) -> Result<Buffer> {
        ensure!(
            self.device_greedy_available()
                && self.owns_command_queue(cmd)
                && logits.device().registry_id() == self.device.registry_id()
                && logits.length() == 248_320 * 2,
            "ineligible device greedy input"
        );
        let lse = buffer(&self.device, 2, Storage::StorageModePrivate)?;
        let selected = buffer(&self.device, 4, Storage::StorageModeShared)?;
        Self::encode(
            cmd,
            &self.logsumexp,
            &[logits, &lse],
            [248_320i32],
            MTLSize::new(1, 1, 1),
            1024,
        );
        Self::encode(
            cmd,
            self.device_greedy_pipeline
                .as_ref()
                .context("missing device greedy pipeline")?,
            &[logits, &lse, &selected],
            [248_320u32, u32::from(strict_raw_logits)],
            MTLSize::new(1, 1, 1),
            1024,
        );
        Ok(selected)
    }
    pub(super) fn read_device_greedy(&self, selected: &Buffer) -> Result<u32> {
        ensure!(
            selected.length() == 4,
            "invalid device greedy readback extent"
        );
        // Called only after the owning command completed; the Shared output is four aligned bytes.
        let index = unsafe { *selected.contents().cast::<u32>() };
        #[cfg(test)]
        let index = if self.device_greedy_reject_result.get() {
            u32::MAX
        } else {
            index
        };
        ensure!(
            crate::gpu_greedy_policy::valid_result(index, 248_320),
            "invalid normalized device greedy distribution"
        );
        self.device_greedy_calls
            .set(self.device_greedy_calls.get().saturating_add(1));
        Ok(index)
    }
    pub(super) fn device_greedy_from_logits(&self, logits: &[u16]) -> Result<u32> {
        ensure!(logits.len() == 248_320, "invalid Flash vocabulary width");
        ensure!(
            self.device_greedy_available(),
            "device greedy is disabled or unsupported"
        );
        let logits = upload(&self.device, logits)?;
        let selected = self.command(|cmd| self.encode_device_greedy(cmd, &logits))?;
        self.read_device_greedy(&selected)
    }
    pub(super) fn device_greedy_head(
        &self,
        head: &super::FlashProjection<'_>,
        mixed: &MatrixView,
        capture: bool,
    ) -> Result<(u32, Option<Vec<u16>>)> {
        ensure!(!capture || cfg!(test), "logits capture is test-only");
        let (selected, logits) = self.command(|cmd| {
            let storage = if capture {
                Storage::StorageModeShared
            } else {
                Storage::StorageModePrivate
            };
            let logits = head.encode_row(cmd, mixed, storage)?;
            Ok((
                self.encode_device_greedy(cmd, &logits.buffer)?,
                logits.buffer,
            ))
        })?;
        let index = self.read_device_greedy(&selected)?;
        let captured = capture.then(|| Self::read(&logits, 248_320));
        self.device_greedy_heads
            .set(self.device_greedy_heads.get().saturating_add(1));
        Ok((index, captured))
    }
}
