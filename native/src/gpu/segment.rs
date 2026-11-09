//! Commands and their resources move together through encode, submit and retirement.
use super::*;
use metal::CommandBuffer;

pub(super) struct GpuSegment<'a, T> {
    gpu: &'a Metal,
    command: CommandBuffer,
    resources: T,
}
pub(super) struct SubmittedSegment<'a, T> {
    gpu: &'a Metal,
    command: CommandBuffer,
    resources: Option<T>,
    retired: bool,
}

impl Metal {
    pub(super) fn segment<T>(
        &self,
        encode: impl FnOnce(&CommandBufferRef) -> Result<T>,
    ) -> Result<GpuSegment<'_, T>> {
        ensure!(
            crate::frame_arena::segment_admitted(
                self.segment_pending.get(),
                self.segment_failed.get()
            ),
            "GPU segment queue is full or failed"
        );
        let command = self.queue.new_command_buffer().to_owned();
        let resources = encode(&command)?;
        Ok(GpuSegment {
            gpu: self,
            command,
            resources,
        })
    }
}
impl<'a, T> GpuSegment<'a, T> {
    pub(super) fn submit(self) -> Result<SubmittedSegment<'a, T>> {
        ensure!(
            crate::frame_arena::segment_admitted(
                self.gpu.segment_pending.get(),
                self.gpu.segment_failed.get()
            ),
            "GPU segment queue is full or failed"
        );
        self.gpu
            .segment_pending
            .set(self.gpu.segment_pending.get() + 1);
        self.command.commit();
        #[cfg(test)]
        self.gpu
            .segment_submissions
            .set(self.gpu.segment_submissions.get() + 1);
        Ok(SubmittedSegment {
            gpu: self.gpu,
            command: self.command,
            resources: Some(self.resources),
            retired: false,
        })
    }
}
impl<T> SubmittedSegment<'_, T> {
    fn drain(&mut self) -> Result<()> {
        if !self.retired {
            self.gpu.wait_command(&self.command);
            self.retired = true;
            self.gpu
                .segment_pending
                .set(self.gpu.segment_pending.get() - 1);
            if self.command.status() != MTLCommandBufferStatus::Completed {
                self.gpu.segment_failed.set(true);
            }
        }
        ensure!(
            !self.gpu.segment_failed.get(),
            "GPU segment completion failed"
        );
        Ok(())
    }
    pub(super) fn retire(mut self) -> Result<T> {
        self.drain()?;
        self.resources
            .take()
            .context("GPU segment resources already retired")
    }
}
impl<T> Drop for SubmittedSegment<'_, T> {
    fn drop(&mut self) {
        let _ = self.drain();
    }
}

#[cfg(test)]
#[test]
#[ignore = "physical Metal segment lifetime and bounded admission"]
fn physical_segments_retire_on_drop_and_reject_failed_or_full_queues() -> Result<()> {
    let gpu = Metal::new()?;
    let make = || {
        gpu.segment(|cmd| {
            let output = buffer(&gpu.device, 128, Storage::StorageModeShared)?;
            let blit = cmd.new_blit_command_encoder();
            blit.fill_buffer(&output, metal::NSRange::new(0, 128), 71);
            blit.end_encoding();
            Ok(output)
        })
    };
    let first = make()?.submit()?;
    let second = make()?.submit()?;
    ensure!(gpu.segment_pending.get() == 2 && make().is_err());
    drop(first);
    ensure!(gpu.segment_pending.get() == 1);
    let output = second.retire()?;
    ensure!(gpu.segment_pending.get() == 0);
    let bytes = unsafe { std::slice::from_raw_parts(output.contents().cast::<u8>(), 128) };
    ensure!(bytes.iter().all(|&byte| byte == 71));
    ensure!(
        gpu.segment::<()>(|_| anyhow::bail!("injected encode failure"))
            .is_err()
    );
    ensure!(gpu.segment_pending.get() == 0);
    gpu.segment_failed.set(true);
    ensure!(make().is_err() && gpu.command(|_| Ok(())).is_err());
    Ok(())
}
