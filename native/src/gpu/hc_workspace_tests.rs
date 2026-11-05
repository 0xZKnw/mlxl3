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
            .into_iter()
            .chain(views.up.as_ref())
            .map(|v| v.buffer.as_ptr() as usize)
            .collect::<Vec<_>>();
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
        let other_fused =
            crate::contracts::flash_hc_kernel_eligible(gpu.hc_up_mix.get(), false, !fp32);
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

#[test]
#[ignore = "physical HC workspace OFF/ON four rows, independent initial archive and full states; run alone"]
fn physical_workspace_preserves_model_states_logits_hidden_and_prefix() -> Result<()> {
    use super::super::{
        FlashNext,
        ssd_direct_tests::{check_initial, check_tensors},
    };
    let checkpoint =
        crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = Metal::new()?;
    let initial_cross_layer_enabled = gpu.cross_layer.get();
    ensure!(
        initial_cross_layer_enabled,
        "requires accepted L1 on qualified M5"
    );
    let initial_workspace_enabled = gpu.hc_workspace_enabled.get();
    ensure!(
        initial_workspace_enabled,
        "requires explicit workspace1 on qualified M5"
    );
    let mut model = FlashNext::load(&gpu, &checkpoint, 4096)?;
    gpu.hc_workspace_enabled.set(false);
    let empty = model.snapshot();
    let (first, states) = model.forward_trace(248044)?;
    check_initial(&first, &states)?;
    check_tensors(&states)?;
    drop(states);
    model.restore(&empty)?;
    let finite = |v: &[u16]| v.len() == 248320 && v.iter().all(|&b| f16::from_bits(b).is_finite());
    let mut steps = Vec::new();
    let mut state_words = 0usize;
    for token in [248044, 1, 42, 7] {
        let prefix = model.snapshot();
        gpu.hc_workspace_enabled.set(false);
        let uses = gpu.hc_workspace_uses.get();
        let triples = gpu.cross_layer_triples.get();
        let expected = model.forward(token)?;
        let control_cross_layer_triples = gpu.cross_layer_triples.get() - triples;
        ensure!(control_cross_layer_triples == 34);
        ensure!(gpu.hc_workspace_uses.get() == uses && finite(&expected));
        let hidden = model
            .last_hidden()
            .context("missing control hidden")?
            .export_bytes(&gpu)?;
        ensure!(
            hidden.len() == 40960
                && hidden
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|v| f32::from_le_bytes(*v).is_finite())
        );
        let triples = gpu.cross_layer_triples.get();
        let (expected_next, expected_states) = model.forward_trace(11)?;
        ensure!(gpu.cross_layer_triples.get() == triples);
        check_tensors(&expected_states)?;
        ensure!(finite(&expected_next));
        model.restore(&prefix)?;
        gpu.hc_workspace_enabled.set(true);
        let uses = gpu.hc_workspace_uses.get();
        let triples = gpu.cross_layer_triples.get();
        let actual = model.forward(token)?;
        let cross_layer_triples = gpu.cross_layer_triples.get() - triples;
        ensure!(cross_layer_triples == 34);
        let actual_uses = gpu.hc_workspace_uses.get() - uses;
        ensure!(actual_uses == 97 && actual == expected && finite(&actual));
        ensure!(model.offset() == prefix.position() + 1);
        ensure!(
            model
                .last_hidden()
                .context("missing candidate hidden")?
                .export_bytes(&gpu)?
                == hidden
        );
        let candidate = model.snapshot();
        // Read the candidate state through the unchanged workspace-OFF continuation.
        gpu.hc_workspace_enabled.set(false);
        let uses = gpu.hc_workspace_uses.get();
        let triples = gpu.cross_layer_triples.get();
        let (next, states) = model.forward_trace(11)?;
        let continuation_cross_layer_triples = gpu.cross_layer_triples.get() - triples;
        ensure!(continuation_cross_layer_triples == 0);
        ensure!(gpu.hc_workspace_uses.get() == uses && next == expected_next && finite(&next));
        state_words += check_tensors(&states)?;
        ensure!(states.len() == expected_states.len());
        for (actual, expected) in states.iter().zip(&expected_states) {
            ensure!(
                actual.name == expected.name
                    && actual.shape == expected.shape
                    && actual.data == expected.data,
                "workspace changed {}",
                actual.name
            );
        }
        model.restore(&candidate)?;
        steps.push(serde_json::json!({"token":token,"control_cross_layer_triples":control_cross_layer_triples,"cross_layer_triples":cross_layer_triples,"continuation_cross_layer_triples":continuation_cross_layer_triples,"workspace_uses":actual_uses,"control_workspace_uses":0,"continuation_workspace_uses":0,"snapshot_position":candidate.position()}));
    }
    ensure!(steps.len() == 4 && state_words == 115_212_288 && model.offset() == 4);
    ensure!(gpu.readers()?.in_flight() == 0);
    eprintln!(
        "I125_HC_WORKSPACE_MODEL {}",
        serde_json::json!({"parity":true,"initial_workspace_enabled":initial_workspace_enabled,"steps":steps,"actual_workspace_uses":388,"compared_state_words":state_words,"compared_logits":1986560,"compared_hidden_words":40960,"initial_cross_layer_enabled":initial_cross_layer_enabled,"initial_archive_states":84,"context":4096,"retained_prefix":true,"explicit_legacy_continuation":true})
    );
    Ok(())
}
