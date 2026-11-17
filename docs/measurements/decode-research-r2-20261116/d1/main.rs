use half::f16;
use mlxl3_native::sampling::{History, Sampler};
use std::hint::black_box;
use std::time::Instant;

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
fn gauss(s: &mut u64) -> f64 {
    let u1 = ((xorshift(s) >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
    let u2 = (xorshift(s) >> 11) as f64 / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
/// Log probabilities as the engine sees them: F16 log-softmax widened to f32.
fn log_probs(vocab: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut logits: Vec<f64> = (0..vocab).map(|_| 2.3 * gauss(&mut s)).collect();
    for (rank, boost) in [18.0, 16.5, 15.0].into_iter().enumerate() {
        let i = (xorshift(&mut s) as usize) % vocab;
        logits[i] = boost - rank as f64 * 0.1;
    }
    let max = logits.iter().cloned().fold(f64::MIN, f64::max);
    let lse = max + logits.iter().map(|l| (l - max).exp()).sum::<f64>().ln();
    logits.iter().map(|l| f16::from_f64(l - lse).to_f32()).collect()
}
fn stats(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2], v[v.len() / 10], v[v.len() * 9 / 10])
}
fn main() {
    let sampler = Sampler::new(0.2, 80, 1.0, 1.05, 0.0).unwrap();
    let greedy_like = Sampler::new(0.2, 80, 1.0, 1.0, 0.0).unwrap();
    for vocab in [128_000usize, 248_320] {
        let rows: Vec<Vec<f32>> = (0..8).map(|i| log_probs(vocab, 0x9E37_79B9 + i)).collect();
        let bytes: Vec<Vec<u8>> = rows
            .iter()
            .map(|r| r.iter().flat_map(|x| x.to_ne_bytes()).collect())
            .collect();
        let history = History::from_tokens(&(0..20u32).map(|i| i * 977 % vocab as u32).collect::<Vec<_>>());
        let mut random = 0x1234_5678_9ABC_DEFu64;
        let (warm, iters) = (200, 2000);
        let mut convert = Vec::new();
        let mut dist = Vec::new();
        let mut dist_nopen = Vec::new();
        let mut total = Vec::new();
        let mut sink = 0u64;
        for i in 0..warm + iters {
            let b = &bytes[i % bytes.len()];
            let t0 = Instant::now();
            // Same conversion as Array::to_f32 (bytes -> Vec<f32>).
            let values: Vec<f32> = b.as_chunks::<4>().0.iter().map(|x| f32::from_ne_bytes(*x)).collect();
            let t1 = Instant::now();
            let token = sampler.distribution(black_box(values), &history, &[]).unwrap().sample(&mut random);
            let t2 = Instant::now();
            sink += u64::from(token);
            let values: Vec<f32> = rows[i % rows.len()].clone();
            let t3 = Instant::now();
            let token = greedy_like.distribution(black_box(values), &history, &[]).unwrap().sample(&mut random);
            let t4 = Instant::now();
            sink += u64::from(token);
            if i >= warm {
                convert.push((t1 - t0).as_secs_f64() * 1e6);
                dist.push((t2 - t1).as_secs_f64() * 1e6);
                total.push((t2 - t0).as_secs_f64() * 1e6);
                dist_nopen.push((t4 - t3).as_secs_f64() * 1e6);
            }
        }
        let f = |name: &str, v: Vec<f64>| {
            let (m, lo, hi) = stats(v);
            println!("vocab={vocab} {name}: median {m:.1} us  p10 {lo:.1}  p90 {hi:.1}");
        };
        f("bytes->Vec<f32>", convert);
        f("distribution+sample (T0.2 k80 rep1.05)", dist);
        f("distribution+sample (T0.2 k80 rep1.0)", dist_nopen);
        f("convert+distribution+sample", total);
        println!("sink {sink}");
    }
}
