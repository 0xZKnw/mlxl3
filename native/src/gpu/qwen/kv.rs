//! Append-only attention backing. Snapshots expose only their initialized prefix.
use super::{Context, FloatType, MatrixView, Metal, Result, Storage, ensure};
use crate::qwen35_config::native_qwen_kv_plan;
use metal::CommandBufferRef;
use std::{cell::Cell, rc::Rc};

const WIDTH: u32 = 512;
const ROW_BYTES: u64 = WIDTH as u64 * 2;

struct Slab {
    buffers: [MatrixView; 2],
    capacity: u32,
    // A reserved suffix can outlive a dropped block or failed command. Always
    // fork before reusing it so every previously published prefix is immutable.
    reserved: Cell<u32>,
}

#[derive(Clone)]
pub(super) struct DeviceKv {
    slab: Rc<Slab>,
    len: u32,
}

impl DeviceKv {
    pub(super) fn append(
        gpu: &Metal,
        cmd: &CommandBufferRef,
        prior: Option<&Self>,
        [key, value]: [&MatrixView; 2],
        offset: u32,
        limit: u32,
    ) -> Result<Self> {
        ensure!(
            prior.map_or(0, |kv| kv.len) == offset,
            "Qwen device KV position mismatch"
        );
        ensure!(
            [key, value].iter().all(|v| {
                v.dtype == FloatType::F16
                    && v.layout.shape == [1, WIDTH]
                    && v.layout.offset == 0
                    && v.layout.strides[1] == 1
                    && v.buffer.length() == ROW_BYTES
                    && v.buffer.device().registry_id() == gpu.device.registry_id()
            }),
            "invalid Qwen device KV input"
        );
        let (reserved, allocated) =
            prior.map_or((0, 0), |kv| (kv.slab.reserved.get(), kv.slab.capacity));
        let (capacity, at, fork) = native_qwen_kv_plan(offset, reserved, allocated, limit)
            .context("invalid Qwen device KV capacity")?;
        let slab = match prior {
            Some(kv) if !fork => kv.slab.clone(),
            _ => {
                let allocate = || {
                    MatrixView::output(
                        gpu,
                        FloatType::F16,
                        capacity,
                        WIDTH,
                        Storage::StorageModeShared,
                    )
                };
                Rc::new(Slab {
                    buffers: [allocate()?, allocate()?],
                    capacity,
                    reserved: Cell::new(0),
                })
            }
        };
        let len = offset + 1;
        // Reserve before encoding. No result becomes visible until the caller
        // completes the command and publishes the returned snapshot.
        slab.reserved.set(len);
        let blit = cmd.new_blit_command_encoder();
        if let Some(kv) = prior
            && fork
            && at != 0
        {
            for (old, new) in kv.slab.buffers.iter().zip(&slab.buffers) {
                blit.copy_from_buffer(&old.buffer, 0, &new.buffer, 0, at);
            }
        }
        for (src, dst) in [key, value].into_iter().zip(&slab.buffers) {
            blit.copy_from_buffer(&src.buffer, 0, &dst.buffer, at, ROW_BYTES);
        }
        blit.end_encoding();
        Ok(Self { slab, len })
    }

    pub(super) fn views(&self) -> [MatrixView; 2] {
        self.slab.buffers.each_ref().map(|buffer| {
            let mut prefix = buffer.clone();
            prefix.layout.shape[0] = self.len;
            prefix
        })
    }

    pub(super) fn retained_bytes(&self) -> usize {
        self.slab.capacity as usize * WIDTH as usize * 4
    }

    pub(super) fn words(&self) -> [Vec<u16>; 2] {
        let count = self.len as usize * WIDTH as usize;
        self.slab
            .buffers
            .each_ref()
            .map(|v| Metal::read(&v.buffer, count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a physical Metal device"]
    fn qwen_device_kv_preserves_prefixes_across_grow_fork_and_abort() -> Result<()> {
        let gpu = Metal::new()?;
        let make_row = |tag: u32, kind: u32| {
            let values = (0..WIDTH)
                .map(|c| (tag * 67 + kind * 123 + c % 37) as u16)
                .collect::<Vec<_>>();
            super::super::matrix(&gpu, &values, 1, WIDTH)
        };
        let append = |prior: Option<&DeviceKv>, offset, tag| {
            let key = make_row(tag, 0)?;
            let value = make_row(tag, 1)?;
            gpu.command(|cmd| DeviceKv::append(&gpu, cmd, prior, [&key, &value], offset, 137))
        };
        let verify = |kv: &DeviceKv, tags: &[u32]| {
            for (kind, row) in kv.words().iter().enumerate() {
                let expected = tags
                    .iter()
                    .flat_map(|tag| {
                        (0..WIDTH).map(move |c| (*tag * 67 + kind as u32 * 123 + c % 37) as u16)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(*row, expected);
            }
            for view in kv.views() {
                assert_eq!(view.shape(), [tags.len() as u32, WIDTH]);
            }
        };
        let first = append(None, 0, 1)?;
        let second = append(Some(&first), 1, 2)?;
        assert!(Rc::ptr_eq(&first.slab, &second.slab));
        let mut last = second.clone();
        let mut tags = vec![1, 2];
        for offset in 2..137 {
            last = append(Some(&last), offset, offset + 1)?;
            tags.push(offset + 1);
            if [63, 64, 127, 128].contains(&offset) {
                verify(&last, &tags);
            }
        }
        verify(&last, &tags);
        assert_eq!(last.slab.capacity, 137);
        assert!(append(Some(&last), 137, 130).is_err());
        assert!(append(Some(&first), 2, 130).is_err());
        let branch = append(Some(&first), 1, 200)?;
        assert!(!Rc::ptr_eq(&first.slab, &branch.slab));
        verify(&first, &[1]);
        verify(&second, &[1, 2]);
        verify(&branch, &[1, 200]);

        let key = make_row(210, 0)?;
        let value = make_row(210, 1)?;
        let aborted = gpu.command(|cmd| -> Result<()> {
            DeviceKv::append(&gpu, cmd, Some(&branch), [&key, &value], 2, 137)?;
            anyhow::bail!("abort after reservation, before submission")
        });
        assert!(aborted.is_err());
        let resumed = append(Some(&branch), 2, 220)?;
        assert!(!Rc::ptr_eq(&branch.slab, &resumed.slab));
        verify(&branch, &[1, 200]);
        verify(&resumed, &[1, 200, 220]);
        verify(&last, &tags);
        Ok(())
    }
}
