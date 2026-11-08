//! Ordered B2/B4 transitions with each canonical prefix retained.
use super::*;

impl FlashGdn<'_> {
    pub(super) fn encode_temporal(
        &self,
        cmd: &CommandBufferRef,
        inputs: &[MatrixView],
        qkv: &MatrixView,
        z: &MatrixView,
    ) -> Result<Vec<PendingGdn>> {
        let rows = u32::try_from(inputs.len())?;
        let key = crate::frame_arena::temporal_key(rows).context("invalid temporal GDN rows")?;
        let gpu = self.gpu;
        ensure!(
            qkv.shape() == [rows, 10240]
                && z.shape() == [rows, 6144]
                && qkv.layout.offset == 0
                && z.layout.offset == 0,
            "invalid packed temporal GDN projections"
        );
        let pipeline = &gpu
            .gdn
            .temporal
            .as_ref()
            .context("temporal GDN pipeline missing")?[key];
        ensure!(
            pipeline.thread_execution_width() == 32
                && pipeline.max_total_threads_per_threadgroup() >= 512,
            "unsupported temporal GDN threadgroup"
        );
        let storage = Storage::StorageModePrivate;
        let a = MatrixView::output(gpu, FloatType::F16, rows, 48, storage)?;
        let b = MatrixView::output(gpu, FloatType::F16, rows, 48, storage)?;
        for (row, input) in inputs.iter().enumerate() {
            ensure!(
                input.shape() == [1, 2560]
                    && input.dtype == FloatType::F16
                    && input.layout.offset == 0,
                "invalid temporal GDN input"
            );
            let offset = row as u64 * 96;
            if gpu.gdn_joint_ab.get() {
                Metal::encode_bound(
                    cmd,
                    gpu.gdn.joint_ab.as_ref().context("joint GDN a/b missing")?,
                    [
                        (&input.buffer, 0),
                        (&self.a, 0),
                        (&self.b, 0),
                        (&a.buffer, offset),
                        (&b.buffer, offset),
                    ]
                    .into_iter(),
                    [48u32],
                    MTLSize::new(24, 1, 1),
                    256,
                );
                #[cfg(test)]
                gpu.gdn_joint_ab_uses.set(gpu.gdn_joint_ab_uses.get() + 1);
            } else {
                for (weight, output) in [(&self.a, &a), (&self.b, &b)] {
                    Metal::encode_bound(
                        cmd,
                        &gpu.gdn.dense,
                        [(&input.buffer, 0), (weight, 0), (&output.buffer, offset)].into_iter(),
                        [48u32],
                        MTLSize::new(12, 1, 1),
                        256,
                    );
                }
            }
        }
        let previous = if let Some(frame) = &self.state.frame {
            frame.clone()
        } else {
            let zero = self.lease_frame(storage)?;
            let blit = cmd.new_blit_command_encoder();
            for view in [&zero.history, &zero.recurrent] {
                blit.fill_buffer(&view.buffer, NSRange::new(0, view.buffer.length()), 0);
            }
            blit.end_encoding();
            zero
        };
        let frames = (0..rows)
            .map(|_| self.lease_frame(storage))
            .collect::<Result<Vec<_>>>()?;
        let gated = MatrixView::output(gpu, FloatType::F16, rows, 6144, storage)?;
        let mut bindings = vec![
            &qkv.buffer,
            &z.buffer,
            &a.buffer,
            &b.buffer,
            &previous.history.buffer,
            &self.conv,
            &self.a_log,
            &self.dt_bias,
            &previous.recurrent.buffer,
            &frames[0].history.buffer,
            &frames[0].recurrent.buffer,
            &self.gain,
            &gated.buffer,
        ];
        for row in 1..4 {
            let frame = &frames[row.min(frames.len() - 1)];
            bindings.extend([&frame.history.buffer, &frame.recurrent.buffer]);
        }
        Metal::encode(
            cmd,
            pipeline,
            &bindings,
            [self.eps.to_bits(), rows],
            MTLSize::new(48, 1, 1),
            512,
        );
        let output = self.out.encode_rows(cmd, &gated, storage)?;
        let mut pending = Vec::with_capacity(rows as usize);
        for row in 0..rows as usize {
            pending.push(PendingGdn {
                #[cfg(test)]
                test_gated: gated.row(row as u32)?,
                command: cmd.to_owned(),
                next: frames[row].clone(),
                _previous: if row == 0 {
                    previous.clone()
                } else {
                    frames[row - 1].clone()
                },
                values: output.row(row as u32)?,
                steps: None,
            });
        }
        #[cfg(test)]
        TEMPORAL_BATCHES.with(|count| count.set(count.get() + 1));
        Ok(pending)
    }
}
