use super::*;
use crate::tensor_layout::MatrixLayout;
use metal::foreign_types::ForeignType;

fn row(gpu: &Metal, fp32: bool, step: usize) -> Result<MatrixView> {
    let x: Vec<_> = (0..20481)
        .map(|i| (((i * 17 + step * 79) % 257) as f32 - 128.) / 231.)
        .collect();
    let storage = if fp32 {
        gpu.f32_storage(&x)?
    } else {
        gpu.f16_storage(
            &x.iter()
                .map(|&x| f16::from_f32(x).to_bits())
                .collect::<Vec<_>>(),
        )?
    };
    storage.matrix(MatrixLayout {
        shape: [1, 10240],
        offset: 20479,
        strides: [20480, -2],
    })
}
fn output(read: &FlashHcRead) -> Result<[Vec<u32>; 2]> {
    let result = [
        read.mixed.read_output()?,
        read.gate.as_ref().context("gate required")?.read_output()?,
    ];
    ensure!(result[0].len() == 2560 && result[1].len() == 4);
    for words in &result {
        ensure!(words.iter().all(|&v| match read.mixed.dtype {
            FloatType::F16 => f16::from_bits(v as u16).is_finite(),
            FloatType::F32 => f32::from_bits(v).is_finite(),
        }));
    }
    Ok(result)
}
fn requests() -> usize {
    super::super::BUFFER_REQUESTS.with(std::cell::Cell::get)
}

#[test]
#[ignore = "physical HC workspace reuse, exact outputs, busy/drop/late error and recovery; exclusive GPU"]
fn physical_workspace_reuses_only_completed_private_temporaries() -> Result<()> {
    let checkpoint =
        crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = Metal::new()?;
    let mut cases = 0usize;
    let mut fresh_counts = Vec::new();
    for fp32 in [false, true] {
        let fused = crate::contracts::flash_hc_kernel_eligible(gpu.hc_up_mix.get(), false, fp32);
        let with_up = !fused;
        let fresh_requests = if fused { 6 } else { 7 };
        fresh_counts.push(fresh_requests);
        let mut hc = FlashHc::load(
            &gpu,
            &checkpoint,
            "model.language_model.layers.0.attn_hyper_connection",
            1e-6,
            true,
        )?;
        let mut retained = None;
        for step in 0..3 {
            let input = row(&gpu, fp32, step)?;
            gpu.hc_workspace_enabled.set(false);
            let expected = output(&hc.read(&input, false)?)?;
            gpu.hc_workspace_enabled.set(true);
            let before = requests();
            let uses = gpu.hc_workspace_uses.get();
            let actual = hc.read(&input, false)?;
            ensure!(requests() - before == if step == 0 { fresh_requests } else { 3 });
            ensure!(gpu.hc_workspace_uses.get() - uses == 1);
            ensure!(output(&actual)? == expected);
            if let Some((old, words)) = retained.take() {
                ensure!(output(&old)? == words);
            }
            retained = Some((actual, expected));
            if step == 0 {
                let poison = gpu.command(|cmd| {
                    let (lease, views) =
                        workspace::Cache::acquire(&hc.workspace, &gpu, cmd, input.dtype, with_up)?
                            .context("poison lease required")?;
                    let blit = cmd.new_blit_command_encoder();
                    for v in [&views.normalized, &views.down, &views.active]
                        .into_iter()
                        .chain(views.up.as_ref())
                    {
                        blit.fill_buffer(&v.buffer, metal::NSRange::new(0, v.buffer.length()), 255);
                    }
                    blit.end_encoding();
                    Ok(lease)
                })?;
                drop(poison);
            }
            cases += 1;
        }
        let input = row(&gpu, fp32, 2)?;
        let expected = output(&hc.read(&input, false)?)?;
        let cmd = gpu.queue.new_command_buffer().to_owned();
        let (held, views) =
            workspace::Cache::acquire(&hc.workspace, &gpu, &cmd, input.dtype, with_up)?
                .context("expected lease")?;
        let ids = [&views.normalized, &views.down, &views.active]
            .into_iter().chain(views.up.as_ref())
            .map(|v| v.buffer.as_ptr() as usize).collect::<Vec<_>>();
        ensure!(ids.len() == if with_up { 4 } else { 3 });
        ensure!(
            ids.iter()
                .enumerate()
                .all(|(i, id)| *id != 0 && !ids[..i].contains(id))
        );
        ensure!(hc.release_workspace().is_err());
        let before = requests();
        let uses = gpu.hc_workspace_uses.get();
        ensure!(output(&hc.read(&input, false)?)? == expected);
        ensure!(requests() - before == fresh_requests && gpu.hc_workspace_uses.get() == uses);
        drop(held); // NotEnqueued: invalidate, never make the same backing available.
        let before = requests();
        ensure!(output(&hc.read(&input, false)?)? == expected);
        ensure!(requests() - before == fresh_requests);
        let fresh_cmd = gpu.queue.new_command_buffer().to_owned();
        let (fresh_lease, fresh_views) =
            workspace::Cache::acquire(&hc.workspace, &gpu, &fresh_cmd, input.dtype, with_up)?
                .context("fresh lease required")?;
        for v in [
            &fresh_views.normalized,
            &fresh_views.down,
            &fresh_views.active,
        ]
        .into_iter()
        .chain(fresh_views.up.as_ref())
        {
            ensure!(
                !ids.contains(&(v.buffer.as_ptr() as usize)),
                "abandoned backing was reused"
            );
        }
        drop(fresh_lease);
        let cmd = gpu.queue.new_command_buffer().to_owned();
        let pending = hc.encode_read(&cmd, &input, false)?;
        let old_mixed = pending.mixed.clone();
        drop(pending);
        let before = requests();
        ensure!(output(&hc.read(&input, false)?)? == expected);
        ensure!(requests() - before == fresh_requests);
        cmd.commit();
        cmd.wait_until_completed();
        ensure!(cmd.status() == metal::MTLCommandBufferStatus::Completed);
        ensure!(old_mixed.read_output()? == expected[0]);
        let commits = gpu.helper_commits.get();
        let failure: Result<()> = gpu.command(|cmd| {
            let _read = hc.encode_read(cmd, &input, false)?;
            anyhow::bail!("injected downstream failure after all HC dispatches");
        });
        ensure!(failure.is_err() && gpu.helper_commits.get() == commits);
        let before = requests();
        ensure!(output(&hc.read(&input, false)?)? == expected);
        ensure!(requests() - before == fresh_requests);
        // Error after acquiring a populated cache but before commit must invalidate it.
        let saved = hc.gain.clone();
        hc.gain = super::super::buffer(&gpu.device, 4, Storage::StorageModePrivate)?;
        let commits = gpu.helper_commits.get();
        ensure!(hc.read(&input, false).is_err());
        ensure!(gpu.helper_commits.get() == commits);
        hc.gain = saved;
        let before = requests();
        ensure!(output(&hc.read(&input, false)?)? == expected);
        ensure!(requests() - before == fresh_requests);
        let other = row(&gpu, !fp32, 1)?;
        let before = requests();
        let uses = gpu.hc_workspace_uses.get();
        hc.read(&other, false)?;
        let other_fused = crate::contracts::flash_hc_kernel_eligible(gpu.hc_up_mix.get(), false, !fp32);
        ensure!(requests() - before == if other_fused { 6 } else { 7 });
        ensure!(gpu.hc_workspace_uses.get() == uses);
        let before = requests();
        ensure!(output(&hc.read(&input, true)?)? == expected);
        ensure!(requests() - before == 7);
        let before = requests();
        ensure!(output(&hc.read(&input, false)?)? == expected);
        ensure!(requests() - before == 3);
        hc.release_workspace()?;
        let before = requests();
        ensure!(output(&hc.read(&input, false)?)? == expected);
        ensure!(requests() - before == fresh_requests);
    }
    ensure!(cases == 6);
    eprintln!(
        "I125_HC_WORKSPACE_COMPONENT {}",
        serde_json::json!({"parity":true,"cases":cases,"finite_words":cases*2564,"fresh_requests_by_dtype_f16_f32":fresh_counts,"reused_requests":3,"busy_fallback":true,"dtype_fallback":true,"trace_fallback":true,"uncommitted_drop_invalidates":true,"late_error_recovery":true,"old_outputs_retained":true,"poisoned_scratch":true,"abandoned_noalias":true,"postencode_error":true})
    );
    Ok(())
}
