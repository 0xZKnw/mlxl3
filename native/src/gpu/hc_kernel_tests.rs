use super::super::{
    FlashNext,
    ssd_direct_tests::{check_initial, check_tensors},
};
use super::*;
use crate::{checkpoint, tensor_layout::MatrixLayout};

fn set(gpu: &Metal, up: bool, down: bool) {
    gpu.hc_up_mix.set(up);
    gpu.hc_down_inject.set(down);
}
fn finite(words: &[u32], dtype: FloatType, count: usize) -> bool {
    words.len() == count
        && words.iter().all(|&v| match dtype {
            FloatType::F16 => u16::try_from(v).is_ok_and(|v| f16::from_bits(v).is_finite()),
            FloatType::F32 => f32::from_bits(v).is_finite(),
        })
}
fn values(read: &FlashHcRead) -> Result<(Vec<u32>, Option<Vec<u32>>)> {
    let mixed = read.mixed.read_output()?;
    ensure!(finite(&mixed, read.mixed.dtype, 2560));
    let gate = read
        .gate
        .as_ref()
        .map(|v| -> Result<_> {
            let words = v.read_output()?;
            ensure!(finite(&words, v.dtype, 4));
            Ok(words)
        })
        .transpose()?;
    Ok((mixed, gate))
}
fn finite_logits(values: &[u16]) -> bool {
    values.len() == 248320 && values.iter().all(|&v| f16::from_bits(v).is_finite())
}
fn ready() -> Result<Metal> {
    let gpu = Metal::new()?;
    ensure!(
        gpu.hc_up_mix.get() && gpu.hc_down_inject.get(),
        "both HC flags must be 1 before Metal::new on M5"
    );
    Ok(gpu)
}

#[test]
#[ignore = "physical exact HC kernels four variants, half/trace/strided fallbacks and dropped encode; run alone"]
fn physical_hc_kernel_variants_preserve_components_and_fallbacks() -> Result<()> {
    let c = checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = ready()?;
    let mut cases = Vec::new();
    for (prefix, combine) in [
        ("model.language_model.layers.0.attn_hyper_connection", true),
        ("model.language_model.layers.0.mlp_hyper_connection", true),
        ("model.language_model.hyper_connection_mixer", false),
    ] {
        let hc = FlashHc::load(&gpu, &c, prefix, 1e-6, combine)?;
        for half in [false, true] {
            for strided in [false, true] {
                let stride = if strided { 2 } else { 1 };
                let offset = i64::from(strided);
                let data = (0..10240 * stride + usize::from(strided))
                    .map(|i| ((i * 17 % 113) as f32 - 56.) / 79.)
                    .collect::<Vec<_>>();
                let storage = if half {
                    gpu.f16_storage(
                        &data
                            .iter()
                            .map(|&v| f16::from_f32(v).to_bits())
                            .collect::<Vec<_>>(),
                    )?
                } else {
                    gpu.f32_storage(&data)?
                };
                let input = storage.matrix(MatrixLayout {
                    shape: [1, 10240],
                    strides: [10240 * stride as i64, stride as i64],
                    offset,
                })?;
                set(&gpu, false, false);
                let control = values(&hc.read(&input, false)?)?;
                ensure!(control.1.is_some() == combine);
                let control_trace = hc.read(&input, true)?;
                let control_trace =
                    serde_json::to_value(control_trace.trace.context("missing HC trace")?)?;
                for (up, down) in [(false, false), (true, false), (false, true), (true, true)] {
                    set(&gpu, up, down);
                    let before = (gpu.hc_up_mix_uses.get(), gpu.hc_down_inject_uses.get());
                    let actual = hc.read(&input, false)?;
                    ensure!(values(&actual)? == control);
                    let counts = (
                        gpu.hc_up_mix_uses.get() - before.0,
                        gpu.hc_down_inject_uses.get() - before.1,
                    );
                    ensure!(
                        counts
                            == (
                                usize::from(up && !half),
                                usize::from(down && !half && combine)
                            ),
                        "unexpected active HC kernel counts"
                    );
                    let trace = hc.read(&input, true)?;
                    ensure!(values(&trace)? == control);
                    let trace = trace.trace.context("missing legacy trace")?;
                    for (words, count) in [
                        (&trace.normalized, 10240),
                        (&trace.down, 320),
                        (&trace.active, 320),
                        (&trace.up, 10240),
                    ] {
                        ensure!(finite(words, input.dtype, count));
                    }
                    ensure!(serde_json::to_value(trace)? == control_trace);
                    ensure!(
                        (
                            gpu.hc_up_mix_uses.get() - before.0,
                            gpu.hc_down_inject_uses.get() - before.1
                        ) == counts
                    );
                    cases.push(serde_json::json!({"prefix":prefix,"half":half,"strided":strided,"up":up,"down":down,"uses":counts}));
                }
                let commits = gpu.helper_commits.get();
                let malformed = storage.matrix(MatrixLayout {
                    shape: [1, 4],
                    strides: [4, 1],
                    offset: 0,
                })?;
                ensure!(hc.read(&malformed, false).is_err() && gpu.helper_commits.get() == commits);
                // Encode successfully, then fail before commit: retained resources may be dropped
                // with the unsubmitted command. Repeating the same real read must still succeed.
                let failed: Result<()> = gpu.command(|cmd| {
                    let _pending = hc.encode_read(cmd, &input, false)?;
                    bail!("intentional failure before commit")
                });
                ensure!(failed.is_err() && gpu.helper_commits.get() == commits);
                ensure!(values(&hc.read(&input, false)?)? == control);
            }
        }
    }
    ensure!(cases.len() == 48);
    eprintln!(
        "NATIVE_HC_KERNEL_COMPONENT {}",
        serde_json::json!({"cases":cases,"case_count":48,"trace_fallback":true,"half_fallback":true,"before_commit_failure_recovered":true,"parity":true})
    );
    Ok(())
}

#[test]
#[ignore = "physical HC four kernel variants, untraced actual97/96 counts, context4096 full states and independent archives; run alone"]
fn physical_hc_kernel_variants_preserve_model_states_and_actual_paths() -> Result<()> {
    let c = checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = ready()?;
    let workspace_enabled = gpu.hc_workspace_enabled.get();
    set(&gpu, false, false);
    let mut model = FlashNext::load(&gpu, &c, 4096)?;
    ensure!(model.prefetcher.is_none());
    let empty = model.snapshot();
    let (initial, states) = model.forward_trace(248044)?;
    check_initial(&initial, &states)?;
    check_tensors(&states)?;
    drop(states);
    model.restore(&empty)?;
    let mut cases = Vec::new();
    let mut compared = 0usize;
    for token in [248044, 1, 42, 7] {
        let saved = model.snapshot();
        set(&gpu, false, false);
        ensure!(finite_logits(&model.forward(19)?));
        model.restore(&saved)?;
        let expected = model.forward(token)?;
        ensure!(finite_logits(&expected));
        let expected_hidden = model
            .last_hidden()
            .context("missing control hidden")?
            .export_bytes(&gpu)?;
        ensure!(
            expected_hidden.len() == 40960
                && expected_hidden
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|&v| f32::from_le_bytes(v).is_finite())
        );
        let control_state = model.snapshot();
        let (expected_next, expected_states) = model.forward_trace(11)?;
        check_tensors(&expected_states)?;
        for (up, down) in [(false, false), (true, false), (false, true), (true, true)] {
            model.restore(&saved)?;
            set(&gpu, up, down);
            let before = (gpu.hc_up_mix_uses.get(), gpu.hc_down_inject_uses.get());
            let workspace_before = gpu.hc_workspace_uses.get();
            let actual = model.forward(token)?;
            let workspace_uses = gpu.hc_workspace_uses.get() - workspace_before;
            ensure!(workspace_uses == if workspace_enabled { 97 } else { 0 });
            let counts = (
                gpu.hc_up_mix_uses.get() - before.0,
                gpu.hc_down_inject_uses.get() - before.1,
            );
            ensure!(
                counts == (usize::from(up) * 97, usize::from(down) * 96),
                "full model did not execute all HC variants"
            );
            ensure!(
                actual == expected
                    && finite_logits(&actual)
                    && model
                        .last_hidden()
                        .context("missing candidate hidden")?
                        .export_bytes(&gpu)?
                        == expected_hidden
            );
            set(&gpu, false, false);
            let (next, states) = model.forward_trace(11)?;
            ensure!(next == expected_next && finite_logits(&next) && states.len() == 84);
            compared += check_tensors(&states)?;
            for (actual, expected) in states.iter().zip(&expected_states) {
                ensure!(
                    actual.name == expected.name
                        && actual.shape == expected.shape
                        && actual.data == expected.data,
                    "HC kernel changed state {}",
                    actual.name
                );
            }
            cases.push(serde_json::json!({"token":token,"up":up,"down":down,"up_uses":counts.0,"down_uses":counts.1,"workspace_uses":workspace_uses,"parity":true}));
        }
        model.restore(&control_state)?;
    }
    ensure!(
        cases.len() == 16
            && compared == 460_849_152
            && model.offset() == 4
            && gpu.readers()?.in_flight() == 0
    );
    eprintln!(
        "NATIVE_HC_KERNEL_MODEL {}",
        serde_json::json!({"cases":cases,"case_count":16,"workspace_enabled":workspace_enabled,"initial_archive_states":84,"compared_state_words":compared,"compared_logits":7946240,"context":4096,"explicit_legacy_continuation":true,"parity":true})
    );
    Ok(())
}
