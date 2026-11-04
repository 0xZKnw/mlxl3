use super::*;
use super::super::{
    FlashHc, FlashNext,
    ssd_direct_tests::{check_initial, check_tensors},
};
use crate::{checkpoint, tensor_layout::MatrixLayout};
use half::f16;

fn finite_logits(x: &[u16]) -> bool {
    x.len() == 248320 && x.iter().all(|&x| f16::from_bits(x).is_finite())
}
fn finite_hidden(x: &[u8]) -> bool {
    x.len() == 40960 && x.as_chunks::<4>().0.iter().all(|x| f32::from_le_bytes(*x).is_finite())
}

fn component_fallbacks(gpu: &Metal, c: &Checkpoint) -> Result<()> {
    let hc = FlashHc::load(gpu, c, "model.language_model.layers.0.attn_hyper_connection", 1e-6, true)?;
    let mut gdn = FlashGdn::load(gpu, c, "model.language_model.layers.0.linear_attn", 1e-6)?;
    for half_streams in [false, true] {
        let row: Vec<_> = (0..10240).map(|i| ((i % 113) as f32 - 56.) / 79.).collect();
        let storage = if half_streams {
            gpu.f16_storage(&row.iter().map(|&x| f16::from_f32(x).to_bits()).collect::<Vec<_>>())?
        } else { gpu.f32_storage(&row)? };
        let streams = storage.matrix(MatrixLayout {shape:[1,10240],strides:[10240,1],offset:0})?;
        let saved = gdn.snapshot();
        gpu.gdn_half_elision.set(false);
        let (read, expected) = gdn.forward_after_hc(&hc, &streams, false)?;
        ensure!(expected.values.dtype == FloatType::F32);
        let expected_written = hc.write(&streams, &expected.values, read.gate.as_ref().context("missing gate")?)?.export_bytes(gpu)?;
        gdn.restore(&saved)?;
        gpu.gdn_half_elision.set(true);
        let before = gpu.gdn_half_elisions.get();
        let (read, actual) = gdn.forward_after_hc(&hc, &streams, false)?;
        ensure!(actual.values.dtype == if half_streams {FloatType::F32} else {FloatType::F16});
        ensure!(gpu.gdn_half_elisions.get()-before == usize::from(!half_streams));
        let written = hc.write(&streams, &actual.values, read.gate.as_ref().context("missing gate")?)?.export_bytes(gpu)?;
        ensure!(written == expected_written && finite_hidden(&written));
        gdn.restore(&saved)?;
        let before = gpu.gdn_half_elisions.get();
        let public = gdn.forward(&read.mixed, false)?;
        ensure!(public.values.dtype == FloatType::F32 && gpu.gdn_half_elisions.get()==before);
        gdn.restore(&saved)?;
        let traced = gdn.forward(&read.mixed, true)?;
        ensure!(traced.values.dtype == FloatType::F32 && gpu.gdn_half_elisions.get()==before);
        ensure!(traced.trace.as_ref().is_some_and(|x| x.len()==13));
        gdn.restore(&saved)?;
    }
    gpu.gdn_half_elision.set(false);
    Ok(())
}

#[test]
#[ignore = "physical private GDN epilogues, actual counters, public/F16/trace fallback, context4096 full independent states; run alone"]
fn physical_private_gdn_epilogues_preserve_model_states_and_actual_paths() -> Result<()> {
    let c = checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = Metal::new()?;
    ensure!(gpu.gdn_half_elision.get() && gpu.cross_layer.get());
    component_fallbacks(&gpu, &c)?;
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
    for token in [248044,1,42,7] {
        let saved = model.snapshot();
        gpu.gdn_half_elision.set(false);
        ensure!(finite_logits(&model.forward(19)?));
        model.restore(&saved)?;
        let triples = gpu.cross_layer_triples.get();
        let expected = model.forward(token)?;
        ensure!(gpu.cross_layer_triples.get()-triples==34 && finite_logits(&expected));
        let hidden = model.last_hidden().context("missing control hidden")?.export_bytes(&gpu)?;
        ensure!(finite_hidden(&hidden));
        let control_state = model.snapshot();
        let (expected_next, expected_states) = model.forward_trace(11)?;
        ensure!(finite_logits(&expected_next));
        check_tensors(&expected_states)?;
        for half in [false,true] {
            model.restore(&saved)?;
            gpu.gdn_half_elision.set(half);
            let count = gpu.gdn_half_elisions.get();
            let triples = gpu.cross_layer_triples.get();
            let actual = model.forward(token)?;
            let half_uses = gpu.gdn_half_elisions.get()-count;
            ensure!(half_uses == usize::from(half)*36 && gpu.cross_layer_triples.get()-triples==34);
            ensure!(actual==expected && finite_logits(&actual));
            ensure!(model.last_hidden().context("missing candidate hidden")?.export_bytes(&gpu)?==hidden);
            gpu.gdn_half_elision.set(false);
            let (next, states) = model.forward_trace(11)?;
            ensure!(next==expected_next && finite_logits(&next) && states.len()==84);
            compared += check_tensors(&states)?;
            for (a,b) in states.iter().zip(&expected_states) {
                ensure!(a.name==b.name && a.shape==b.shape && a.data==b.data,"GDN epilogue state {}",a.name);
            }
            cases.push(serde_json::json!({"token":token,"half":half,"half_uses":half_uses,"cross_layer_triples":34,"parity":true}));
        }
        model.restore(&control_state)?;
    }
    ensure!(cases.len()==8 && compared==230424576 && model.offset()==4 && gpu.readers()?.in_flight()==0);
    eprintln!("NATIVE_GDN_EPILOGUE_MODEL {}",serde_json::json!({"parity":true,"cases":cases,"case_count":8,"compared_state_words":compared,"compared_logits":3973120,"compared_hidden_words":81920,"initial_archive_states":84,"context":4096,"public_f32_fallback":true,"trace_fallback":true,"half_stream_fallback":true,"explicit_legacy_continuation":true}));
    Ok(())
}
