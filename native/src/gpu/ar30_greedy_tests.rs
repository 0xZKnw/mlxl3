//! Independent CPU normalization/selection versus the real device epilogue.
use super::*;
use half::f16;

fn cpu_normalized_first(logits: &[u16]) -> Option<u32> {
    if logits.len() != 248_320 {
        return None;
    }
    let input = logits
        .iter()
        .map(|&bits| f64::from(f16::from_bits(bits).to_f32()))
        .collect::<Vec<_>>();
    if input.iter().any(|v| v.is_nan() || *v == f64::INFINITY) {
        return None;
    }
    let max = input
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .reduce(f64::max)?;
    let lse = f16::from_f64(max + input.iter().map(|v| (*v - max).exp()).sum::<f64>().ln());
    if !lse.is_finite() {
        return None;
    }
    // Sequential host comparison is independent of the candidate's SIMD reduction.
    let normalized = input
        .iter()
        .map(|&v| f16::from_f32(v as f32 - lse.to_f32()).to_f32())
        .collect::<Vec<_>>();
    crate::speculative::greedy_log_prob_id(&normalized)
}

#[test]
fn cpu_greedy_oracle_detects_raw_argmax_and_preserves_rounding_created_ties() {
    let mut input = vec![0u16; 248_320];
    input[1] = f16::from_f32(0.001).to_bits();
    assert_eq!(cpu_normalized_first(&input), Some(0));
    assert!(f16::from_bits(input[1]) > f16::from_bits(input[0]));
    assert_eq!(cpu_normalized_first(&[]), None);
    input[0] = f16::NAN.to_bits();
    assert_eq!(cpu_normalized_first(&input), None);
}

#[test]
#[ignore = "physical Apple M5 normalized GPU greedy; MLXL3_NATIVE_DEVICE_GREEDY=1, run alone"]
fn physical_ar30_device_greedy_matches_independent_half_normalized_oracle() -> Result<()> {
    let gpu = Metal::new()?;
    ensure!(
        gpu.device_greedy_available(),
        "qualified device greedy option required"
    );
    let row = |x: f32| vec![f16::from_f32(x).to_bits(); 248_320];
    let mut cases = vec![row(0.), row(-65504.), row(65504.)];
    let mut tie = row(0.);
    tie[1] = f16::from_f32(0.001).to_bits();
    cases.push(tie);
    let mut subnormal = row(0.);
    subnormal[1] = 1;
    cases.push(subnormal);
    let mut signed = row(-32.);
    signed[0] = 0x8000;
    signed[1] = 0;
    cases.push(signed);
    let mut boundaries = row(-32.);
    for i in [31, 32, 1023, 1024] {
        boundaries[i] = 0;
    }
    cases.push(boundaries);
    let mut masked = row(f32::NEG_INFINITY);
    masked[248_319] = f16::from_f32(1.).to_bits();
    cases.push(masked);
    cases.push(row(f32::NEG_INFINITY));
    let mut nan = row(0.);
    nan[12345] = f16::NAN.to_bits();
    cases.push(nan);
    let mut positive_inf = row(0.);
    positive_inf[248_319] = f16::INFINITY.to_bits();
    cases.push(positive_inf);
    let mut count = 0u64;
    for input in &cases {
        let expected = cpu_normalized_first(input);
        let stock = gpu
            .flash_log_probs(input)
            .ok()
            .and_then(|p| crate::speculative::greedy_log_prob_id(&p));
        ensure!(
            stock == expected,
            "independent CPU normalizer does not agree on adversarial fixture"
        );
        ensure!(
            gpu.device_greedy_from_logits(input).ok() == expected,
            "device normalized selector differs"
        );
        count += u64::from(expected.is_some());
    }
    ensure!(gpu.device_greedy_calls.get() == count && count == 8);
    ensure!(
        gpu.device_greedy_from_logits(&[]).is_err() && gpu.device_greedy_from_logits(&[0]).is_err()
    );
    ensure!(gpu.device_greedy_calls.get() == count);
    println!(
        "AR30_DEVICE_GREEDY {}",
        serde_json::json!({"parity":true,"cases":cases.len(),"input_half_words":cases.len()*248_320,"valid_selections":count,"invalid_distributions":3,"raw_argmax_mutation_detected":true,"readback_bytes":count*4,"wrong_extent_rejected":true})
    );
    Ok(())
}
