//! LFM3-Omni audio input: 16 kHz log-mel features, the convolutional downsampler,
//! the LFM2 encoder and the projector into the language model.
// Reference: LiquidAI/d1-Omni-2B (revision 51918b53), `feature_extraction_lfm3_omni.py`,
// `modeling_lfm3_omni.py` (Lfm3OmniDownsampler, Lfm3OmniEncoder, Lfm3OmniProjector) and
// `media.py` (`read_audio`, `resampled`). Apache-2.0 model code.
use anyhow::{Context, Result, bail, ensure};

pub const SAMPLE_RATE: u32 = 16_000;
pub const MEL_BINS: usize = 128;
const N_FFT: usize = 400;
const HOP: usize = 160;
const BINS: usize = N_FFT / 2 + 1;
const LOG_FLOOR: f64 = -12.0;
/// The longest clip a request may carry, in samples at 16 kHz (five minutes).
pub const MAX_SAMPLES: usize = 5 * 60 * SAMPLE_RATE as usize;
/// One encoder token per eight spectrogram frames.
pub const STRIDE: usize = 8;

/// A mono waveform at 16 kHz, as the feature extractor takes it.
#[derive(Clone, Debug, PartialEq)]
pub struct Clip {
    pub samples: Vec<f32>,
}

impl Clip {
    /// A 16-bit PCM WAV file: channels averaged, then resampled to 16 kHz.
    pub fn from_wav(bytes: &[u8]) -> Result<Self> {
        let (channels, rate) = read_wav(bytes)?;
        let length = channels[0].len();
        let mono: Vec<f32> = (0..length)
            .map(|i| {
                let sum: f32 = channels.iter().map(|c| c[i]).sum();
                sum / channels.len() as f32
            })
            .collect();
        let samples = if rate == SAMPLE_RATE {
            mono
        } else {
            resample(&mono, rate, SAMPLE_RATE)?
        };
        ensure!(!samples.is_empty(), "the audio clip is empty");
        ensure!(
            samples.len() <= MAX_SAMPLES,
            "an audio clip is at most five minutes long"
        );
        ensure!(
            samples.iter().all(|s| s.is_finite()),
            "non-finite audio sample"
        );
        Ok(Self { samples })
    }
    /// Spectrogram frames once the clip is right-padded to whole frames.
    pub fn frames(&self) -> usize {
        frame_count(self.samples.len())
    }
    /// Placeholder tokens the clip takes in a prompt.
    pub fn token_count(&self) -> usize {
        self.frames().div_ceil(STRIDE)
    }
}

pub fn frame_count(samples: usize) -> usize {
    1 + (padded_length(samples) - N_FFT) / HOP
}

fn padded_length(samples: usize) -> usize {
    if samples < N_FFT {
        N_FFT
    } else {
        samples + (HOP - (samples - N_FFT) % HOP) % HOP
    }
}

fn le_u16(bytes: &[u8], at: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        bytes
            .get(at..at + 2)
            .context("truncated WAV header")?
            .try_into()?,
    ))
}
fn le_u32(bytes: &[u8], at: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        bytes
            .get(at..at + 4)
            .context("truncated WAV header")?
            .try_into()?,
    ))
}

/// The channels of a 16-bit PCM WAV file, as floats in [-1, 1), and its sample rate.
pub fn read_wav(bytes: &[u8]) -> Result<(Vec<Vec<f32>>, u32)> {
    ensure!(
        bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "audio must be a WAV file (RIFF/WAVE)"
    );
    let mut at = 12;
    let mut format = None;
    let mut data = None;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = le_u32(bytes, at + 4)? as usize;
        let body = at + 8;
        let end = body.checked_add(size).context("WAV chunk overflow")?;
        if id == b"fmt " {
            ensure!(
                size >= 16 && end <= bytes.len(),
                "truncated WAV format chunk"
            );
            let tag = le_u16(bytes, body)?;
            let channels = le_u16(bytes, body + 2)?;
            let rate = le_u32(bytes, body + 4)?;
            let block = le_u16(bytes, body + 12)?;
            let bits = le_u16(bytes, body + 14)?;
            // WAVE_FORMAT_EXTENSIBLE carries its sub-format GUID; PCM's starts with 0x0001.
            let pcm = tag == 1 || (tag == 0xFFFE && size >= 26 && le_u16(bytes, body + 24)? == 1);
            ensure!(pcm && bits == 16, "16-bit PCM WAV only");
            ensure!(
                (1..=8).contains(&channels) && block == 2 * channels,
                "invalid WAV channel layout"
            );
            ensure!(
                (1_000..=384_000).contains(&rate),
                "unsupported WAV sample rate {rate}"
            );
            format = Some((channels as usize, rate));
        } else if id == b"data" {
            // A streamed WAV may record a size past the end: read what is there.
            data = Some((body, end.min(bytes.len())));
        }
        if data.is_some() && format.is_some() {
            break;
        }
        at = end + (size & 1);
    }
    let (channels, rate) = format.context("WAV file has no format chunk")?;
    let (start, end) = data.context("WAV file has no data chunk")?;
    let frames = (end - start) / (2 * channels);
    ensure!(frames > 0, "the audio clip is empty");
    ensure!(
        frames as u64 * u64::from(SAMPLE_RATE) <= MAX_SAMPLES as u64 * u64::from(rate),
        "an audio clip is at most five minutes long"
    );
    let mut out = vec![Vec::with_capacity(frames); channels];
    for frame in 0..frames {
        for (channel, samples) in out.iter_mut().enumerate() {
            let at = start + 2 * (frame * channels + channel);
            samples.push(f32::from(i16::from_le_bytes([bytes[at], bytes[at + 1]])) / 32768.);
        }
    }
    Ok((out, rate))
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// A mono waveform at `target`, resampled as torchaudio's `functional.resample` does by
/// default (Hann-windowed sinc over 6 zero crossings, rolloff 0.99), in float32 like the reference.
pub fn resample(wave: &[f32], rate: u32, target: u32) -> Result<Vec<f32>> {
    ensure!(rate > 0 && target > 0, "invalid sample rate");
    if rate == target {
        return Ok(wave.to_vec());
    }
    let divisor = gcd(rate, target);
    let (old, new) = ((rate / divisor) as usize, (target / divisor) as usize);
    ensure!(
        old <= 4_096 && new <= 4_096,
        "sample rates {rate} and {target} share too small a divisor to resample"
    );
    let zeros = 6f32;
    let cut = old.min(new) as f32 * 0.99;
    let width = (zeros * old as f32 / cut).ceil() as usize;
    let taps = 2 * width + old;
    let mut kernel = vec![0f32; new * taps];
    for j in 0..new {
        for l in 0..taps {
            let base = -(j as f32) / new as f32;
            let t = ((base + (l as f32 - width as f32) / old as f32) * cut).clamp(-zeros, zeros);
            let window = (t * std::f32::consts::PI / zeros / 2.).cos().powi(2);
            let t = t * std::f32::consts::PI;
            let sinc = if t == 0. { 1. } else { t.sin() / t };
            kernel[j * taps + l] = sinc * (window * (cut / old as f32));
        }
    }
    let padded_len = wave.len() + 2 * width + old;
    let at = |i: usize| -> f32 {
        if i < width || i >= width + wave.len() {
            0.
        } else {
            wave[i - width]
        }
    };
    let steps = (padded_len - taps) / old + 1;
    let length = (new * wave.len()).div_ceil(old);
    let mut out = Vec::with_capacity(steps * new);
    for m in 0..steps {
        for j in 0..new {
            let mut sum = 0f32;
            for l in 0..taps {
                sum += at(m * old + l) * kernel[j * taps + l];
            }
            out.push(sum);
        }
    }
    out.truncate(length);
    Ok(out)
}

fn hertz_to_mel(freq: f64) -> f64 {
    let logstep = 27.0 / 6.4f64.ln();
    if freq >= 1000. {
        15. + (freq / 1000.).ln() * logstep
    } else {
        3. * freq / 200.
    }
}
fn mel_to_hertz(mel: f64) -> f64 {
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= 15. {
        1000. * (logstep * (mel - 15.)).exp()
    } else {
        200. * mel / 3.
    }
}

/// transformers' `mel_filter_bank` with Slaney scale and norm: `[BINS][MEL_BINS]`.
fn mel_filters() -> Vec<f64> {
    let (low, high) = (hertz_to_mel(0.), hertz_to_mel(f64::from(SAMPLE_RATE) / 2.));
    let points: Vec<f64> = (0..MEL_BINS + 2)
        .map(|i| mel_to_hertz(low + (high - low) * i as f64 / (MEL_BINS + 1) as f64))
        .collect();
    let nyquist = f64::from(SAMPLE_RATE / 2);
    let mut filters = vec![0f64; BINS * MEL_BINS];
    for bin in 0..BINS {
        let freq = nyquist * bin as f64 / (BINS - 1) as f64;
        for m in 0..MEL_BINS {
            let down = -(points[m] - freq) / (points[m + 1] - points[m]);
            let up = (points[m + 2] - freq) / (points[m + 2] - points[m + 1]);
            let enorm = 2. / (points[m + 2] - points[m]);
            filters[bin * MEL_BINS + m] = down.min(up).max(0.) * enorm;
        }
    }
    filters
}

/// The normalized log-mel spectrogram `[MEL_BINS][frames]` of a 16 kHz clip:
/// periodic Hann window, no centering, power, Slaney mel, log10 floored at -12, (x + 4) / 4.
pub fn log_mel(clip: &Clip) -> Vec<f32> {
    let frames = clip.frames();
    let window: Vec<f64> = (0..N_FFT)
        .map(|n| 0.5 - 0.5 * (2. * std::f64::consts::PI * n as f64 / N_FFT as f64).cos())
        .collect();
    let (mut cos, mut sin) = (vec![0f64; BINS * N_FFT], vec![0f64; BINS * N_FFT]);
    for k in 0..BINS {
        for n in 0..N_FFT {
            let angle = 2. * std::f64::consts::PI * ((k * n) % N_FFT) as f64 / N_FFT as f64;
            cos[k * N_FFT + n] = angle.cos();
            sin[k * N_FFT + n] = angle.sin();
        }
    }
    let filters = mel_filters();
    let mut out = vec![0f32; MEL_BINS * frames];
    let mut segment = vec![0f64; N_FFT];
    let mut power = vec![0f64; BINS];
    for frame in 0..frames {
        for (n, value) in segment.iter_mut().enumerate() {
            let sample = clip.samples.get(frame * HOP + n).copied().unwrap_or(0.);
            *value = f64::from(sample) * window[n];
        }
        for (k, p) in power.iter_mut().enumerate() {
            let (mut re, mut im) = (0f64, 0f64);
            let (c, s) = (
                &cos[k * N_FFT..(k + 1) * N_FFT],
                &sin[k * N_FFT..(k + 1) * N_FFT],
            );
            for n in 0..N_FFT {
                re += segment[n] * c[n];
                im -= segment[n] * s[n];
            }
            *p = re * re + im * im;
        }
        for m in 0..MEL_BINS {
            let energy: f64 = (0..BINS)
                .map(|k| filters[k * MEL_BINS + m] * power[k])
                .sum();
            let log = if energy > 0. {
                energy.log10().max(LOG_FLOOR)
            } else {
                LOG_FLOOR
            };
            out[m * frames + frame] = ((log + 4.) / 4.) as f32;
        }
    }
    out
}

/// A media item's bytes from a `data:` URL, with its MIME type.
#[cfg(feature = "chat")]
pub fn data_url(url: &str) -> Result<(String, Vec<u8>)> {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    let rest = url.strip_prefix("data:").context("expected a data: URL")?;
    let (header, encoded) = rest.split_once(',').context("invalid data: URL")?;
    let Some(mime) = header.strip_suffix(";base64") else {
        bail!("data: URLs must be base64")
    };
    ensure!(
        encoded.len() <= 64 * 1024 * 1024,
        "media item exceeds 48 MiB"
    );
    Ok((
        mime.to_owned(),
        STANDARD.decode(encoded).context("invalid base64 media")?,
    ))
}

#[cfg(feature = "mlx")]
mod runtime {
    use super::*;
    use crate::{
        array::{Array, Dtype},
        checkpoint::Checkpoint,
        lfm2::half_weight,
    };

    struct Dense {
        weight: Array,
        bias: Option<Array>,
    }
    impl Dense {
        fn load(
            checkpoint: &Checkpoint,
            prefix: &str,
            output: i32,
            input: i32,
            bias: bool,
        ) -> Result<Self> {
            let raw = half_weight(checkpoint, &format!("{prefix}.weight"), None)?;
            ensure!(
                raw.shape().iter().product::<i32>() == output * input && raw.shape()[0] == output,
                "{prefix}: unexpected weight shape {:?}",
                raw.shape()
            );
            Ok(Self {
                weight: raw
                    .reshape(&[output, input])?
                    .transpose(&[1, 0])?
                    .astype(Dtype::Float32)?,
                bias: bias
                    .then(|| -> Result<Array> {
                        half_weight(checkpoint, &format!("{prefix}.bias"), Some(&[output]))?
                            .astype(Dtype::Float32)
                    })
                    .transpose()?,
            })
        }
        fn forward(&self, x: &Array) -> Result<Array> {
            let y = x.matmul(&self.weight)?;
            match &self.bias {
                Some(bias) => y.add(bias),
                None => Ok(y),
            }
        }
    }

    fn vector(checkpoint: &Checkpoint, name: &str, size: i32) -> Result<Array> {
        half_weight(checkpoint, name, Some(&[size]))?.astype(Dtype::Float32)
    }

    /// `[out, in/groups, 3, 3]` (PyTorch) as MLX's `[out, 3, 3, in/groups]`.
    fn kernel(checkpoint: &Checkpoint, name: &str, out: i32, input: i32) -> Result<Array> {
        half_weight(checkpoint, name, Some(&[out, input, 3, 3]))?
            .transpose(&[0, 2, 3, 1])?
            .astype(Dtype::Float32)
    }

    enum Operator {
        Conv {
            input: Dense,
            output: Dense,
            weight: Array,
        },
        Attention {
            q: Dense,
            k: Dense,
            v: Dense,
            o: Dense,
            q_norm: Array,
            k_norm: Array,
        },
    }

    struct Layer {
        operator_norm: Array,
        ffn_norm: Array,
        operator: Operator,
        w1: Dense,
        w3: Dense,
        w2: Dense,
    }

    pub struct AudioEncoder {
        conv_in: (Array, Array),
        conv_mid: (Array, Array),
        lin_mid: Dense,
        conv_end: (Array, Array),
        lin_end: Dense,
        lin_out: Dense,
        layers: Vec<Layer>,
        norm: Array,
        projector_norm: (Array, Array),
        linear_1: Dense,
        linear_2: Dense,
        hidden: i32,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        window: i32,
        theta: f32,
        eps: f32,
    }

    impl AudioEncoder {
        /// The encoder of an `lfm3_omni` config, its weights read from `checkpoint`.
        pub fn load(
            checkpoint: &Checkpoint,
            config: &serde_json::Value,
            text_hidden: i32,
        ) -> Result<Self> {
            let encoder = &config["audio_encoder_config"];
            let lfm = &encoder["text_config"];
            let number = |v: &serde_json::Value, key: &str| -> Result<i32> {
                Ok(i32::try_from(
                    v[key]
                        .as_i64()
                        .with_context(|| format!("missing audio {key}"))?,
                )?)
            };
            let hidden = number(encoder, "hidden_size")?;
            let channels = number(encoder, "conv_channels")?;
            let mels = number(encoder, "num_mel_bins")?;
            ensure!(
                mels as usize == MEL_BINS && channels > 0 && hidden > 0,
                "unsupported audio encoder"
            );
            ensure!(
                number(lfm, "hidden_size")? == hidden,
                "audio encoder width mismatch"
            );
            let heads = number(lfm, "num_attention_heads")?;
            let kv_heads = number(lfm, "num_key_value_heads")?;
            ensure!(
                heads > 0 && kv_heads > 0 && hidden % heads == 0 && heads % kv_heads == 0,
                "invalid audio heads"
            );
            let head_dim = hidden / heads;
            ensure!(
                number(lfm, "conv_L_cache")? == 3,
                "unsupported audio convolution length"
            );
            ensure!(
                lfm["conv_bias"] == false,
                "unsupported audio convolution bias"
            );
            let types = lfm["layer_types"]
                .as_array()
                .context("missing audio layer types")?;
            ensure!(
                types.len() as i64 == lfm["num_hidden_layers"].as_i64().unwrap_or(-1),
                "audio layer count mismatch"
            );
            let theta = lfm["rope_parameters"]["rope_theta"]
                .as_f64()
                .context("missing audio RoPE theta")? as f32;
            ensure!(
                lfm["rope_parameters"]["rope_type"] == "default",
                "unsupported audio RoPE"
            );
            let eps = lfm["norm_eps"].as_f64().context("missing audio norm eps")? as f32;
            let window = number(encoder, "sliding_window")?;
            ensure!(
                window > 0 && theta.is_finite() && theta > 0. && eps > 0.,
                "invalid audio encoder settings"
            );
            let root = "model.audio_encoder";
            let ff = checkpoint
                .tensors
                .get(&format!("{root}.lfm.layers.0.feed_forward.w1.weight"))
                .context("missing audio feed-forward weights")?
                .shape[0] as i32;
            let mut layers = Vec::with_capacity(types.len());
            for (index, kind) in types.iter().enumerate() {
                let prefix = format!("{root}.lfm.layers.{index}");
                let operator = match kind.as_str() {
                    Some("conv") => {
                        let weight = half_weight(
                            checkpoint,
                            &format!("{prefix}.conv.conv.weight"),
                            Some(&[hidden, 1, 3]),
                        )?
                        .transpose(&[0, 2, 1])?
                        .astype(Dtype::Float32)?;
                        Operator::Conv {
                            input: Dense::load(
                                checkpoint,
                                &format!("{prefix}.conv.in_proj"),
                                3 * hidden,
                                hidden,
                                false,
                            )?,
                            output: Dense::load(
                                checkpoint,
                                &format!("{prefix}.conv.out_proj"),
                                hidden,
                                hidden,
                                false,
                            )?,
                            weight,
                        }
                    }
                    Some("full_attention") => {
                        let a = format!("{prefix}.self_attn");
                        Operator::Attention {
                            q: Dense::load(
                                checkpoint,
                                &format!("{a}.q_proj"),
                                hidden,
                                hidden,
                                false,
                            )?,
                            k: Dense::load(
                                checkpoint,
                                &format!("{a}.k_proj"),
                                kv_heads * head_dim,
                                hidden,
                                false,
                            )?,
                            v: Dense::load(
                                checkpoint,
                                &format!("{a}.v_proj"),
                                kv_heads * head_dim,
                                hidden,
                                false,
                            )?,
                            o: Dense::load(
                                checkpoint,
                                &format!("{a}.out_proj"),
                                hidden,
                                hidden,
                                false,
                            )?,
                            q_norm: vector(
                                checkpoint,
                                &format!("{a}.q_layernorm.weight"),
                                head_dim,
                            )?,
                            k_norm: vector(
                                checkpoint,
                                &format!("{a}.k_layernorm.weight"),
                                head_dim,
                            )?,
                        }
                    }
                    other => bail!("unsupported audio layer type {other:?}"),
                };
                layers.push(Layer {
                    operator_norm: vector(
                        checkpoint,
                        &format!("{prefix}.operator_norm.weight"),
                        hidden,
                    )?,
                    ffn_norm: vector(checkpoint, &format!("{prefix}.ffn_norm.weight"), hidden)?,
                    operator,
                    w1: Dense::load(
                        checkpoint,
                        &format!("{prefix}.feed_forward.w1"),
                        ff,
                        hidden,
                        false,
                    )?,
                    w3: Dense::load(
                        checkpoint,
                        &format!("{prefix}.feed_forward.w3"),
                        ff,
                        hidden,
                        false,
                    )?,
                    w2: Dense::load(
                        checkpoint,
                        &format!("{prefix}.feed_forward.w2"),
                        hidden,
                        ff,
                        false,
                    )?,
                });
            }
            let d = format!("{root}.downsample");
            let flat = channels * mels / 8;
            let projector = "model.audio_projector";
            Ok(Self {
                conv_in: (
                    kernel(checkpoint, &format!("{d}.conv_in.weight"), channels, 1)?,
                    vector(checkpoint, &format!("{d}.conv_in.bias"), channels)?,
                ),
                conv_mid: (
                    kernel(checkpoint, &format!("{d}.conv_mid.weight"), channels, 1)?,
                    vector(checkpoint, &format!("{d}.conv_mid.bias"), channels)?,
                ),
                lin_mid: Dense::load(
                    checkpoint,
                    &format!("{d}.lin_mid"),
                    channels,
                    channels,
                    true,
                )?,
                conv_end: (
                    kernel(checkpoint, &format!("{d}.conv_end.weight"), channels, 1)?,
                    vector(checkpoint, &format!("{d}.conv_end.bias"), channels)?,
                ),
                lin_end: Dense::load(
                    checkpoint,
                    &format!("{d}.lin_end"),
                    channels,
                    channels,
                    true,
                )?,
                lin_out: Dense::load(checkpoint, &format!("{d}.lin_out"), hidden, flat, true)?,
                layers,
                norm: vector(
                    checkpoint,
                    &format!("{root}.lfm.embedding_norm.weight"),
                    hidden,
                )?,
                projector_norm: (
                    vector(
                        checkpoint,
                        &format!("{projector}.layer_norm.weight"),
                        hidden,
                    )?,
                    vector(checkpoint, &format!("{projector}.layer_norm.bias"), hidden)?,
                ),
                linear_1: Dense::load(
                    checkpoint,
                    &format!("{projector}.linear_1"),
                    text_hidden,
                    hidden,
                    true,
                )?,
                linear_2: Dense::load(
                    checkpoint,
                    &format!("{projector}.linear_2"),
                    text_hidden,
                    text_hidden,
                    true,
                )?,
                hidden,
                heads,
                kv_heads,
                head_dim,
                window,
                theta,
                eps,
            })
        }

        /// The encoder output after its final norm, `[1, tokens, hidden]`, from a log-mel
        /// spectrogram `[MEL_BINS][frames]` (one clip, no padding).
        pub fn encode(&self, mel: &[f32], frames: usize) -> Result<Array> {
            ensure!(
                frames > 0 && mel.len() == MEL_BINS * frames,
                "invalid log-mel spectrogram"
            );
            let f = frames as i32;
            let x = Array::from_f32(mel, &[1, MEL_BINS as i32, f, 1])?;
            let conv = |x: &Array, (w, b): &(Array, Array), groups: i32| -> Result<Array> {
                x.conv2d_stride2(w, groups)?.add(b)
            };
            let channels = self.conv_in.0.shape()[0];
            let x = conv(&x, &self.conv_in, 1)?.silu()?;
            let x = self
                .lin_mid
                .forward(&conv(&x, &self.conv_mid, channels)?)?
                .silu()?;
            let x = self
                .lin_end
                .forward(&conv(&x, &self.conv_end, channels)?)?
                .silu()?;
            // NHWC [1, mel/8, tokens, channels] -> [1, tokens, channels * mel/8], channel-major like `flatten(1, 2)`.
            let (height, tokens) = (x.shape()[1], x.shape()[2]);
            ensure!(
                tokens as usize == frames.div_ceil(STRIDE),
                "unexpected downsampled length"
            );
            let x = x
                .transpose(&[0, 2, 3, 1])?
                .reshape(&[1, tokens, channels * height])?;
            let mut hidden = self.lin_out.forward(&x)?;
            let mask = crate::lfm3::sliding_mask(tokens, tokens, self.window)?;
            for layer in &self.layers {
                let normalized = hidden.rms_norm(&layer.operator_norm, self.eps)?;
                let mixed = match &layer.operator {
                    Operator::Conv {
                        input,
                        output,
                        weight,
                    } => {
                        let bcx = input.forward(&normalized)?;
                        let h = self.hidden;
                        let b = bcx.slice(2, 0, h)?;
                        let c = bcx.slice(2, h, 2 * h)?;
                        let bx = b.mul(&bcx.slice(2, 2 * h, 3 * h)?)?;
                        let padded = Array::concatenate(
                            &[&Array::zeros_dtype(&[1, 2, h], Dtype::Float32)?, &bx],
                            1,
                        )?;
                        output.forward(&c.mul(&padded.conv1d(weight, h)?)?)?
                    }
                    Operator::Attention {
                        q,
                        k,
                        v,
                        o,
                        q_norm,
                        k_norm,
                    } => {
                        let q = q
                            .forward(&normalized)?
                            .reshape(&[1, tokens, self.heads, self.head_dim])?
                            .rms_norm(q_norm, self.eps)?
                            .transpose(&[0, 2, 1, 3])?
                            .rope(self.head_dim, self.theta, 0)?;
                        let k = k
                            .forward(&normalized)?
                            .reshape(&[1, tokens, self.kv_heads, self.head_dim])?
                            .rms_norm(k_norm, self.eps)?
                            .transpose(&[0, 2, 1, 3])?
                            .rope(self.head_dim, self.theta, 0)?;
                        let v = v
                            .forward(&normalized)?
                            .reshape(&[1, tokens, self.kv_heads, self.head_dim])?
                            .transpose(&[0, 2, 1, 3])?;
                        let out =
                            Array::sdpa_mask(&q, &k, &v, (self.head_dim as f32).powf(-0.5), &mask)?
                                .transpose(&[0, 2, 1, 3])?
                                .reshape(&[1, tokens, self.hidden])?;
                        o.forward(&out)?
                    }
                };
                let h = hidden.add(&mixed)?;
                let normalized = h.rms_norm(&layer.ffn_norm, self.eps)?;
                hidden = h.add(
                    &layer.w2.forward(
                        &layer
                            .w1
                            .forward(&normalized)?
                            .swiglu(&layer.w3.forward(&normalized)?)?,
                    )?,
                )?;
            }
            hidden.rms_norm(&self.norm, self.eps)
        }

        /// The clip's features in the language model's space, `[tokens, text_hidden]` (F16).
        pub fn features(&self, clip: &Clip) -> Result<Array> {
            let encoded = self.encode(&log_mel(clip), clip.frames())?;
            let (weight, bias) = &self.projector_norm;
            let x = encoded
                .layer_norm_without_weight(1e-5)?
                .mul(weight)?
                .add(bias)?;
            let x = self.linear_2.forward(&self.linear_1.forward(&x)?.gelu()?)?;
            let tokens = x.shape()[1];
            let x = x.reshape(&[tokens, x.shape()[2]])?.astype(Dtype::Float16)?;
            x.eval()?;
            Ok(x)
        }

        /// Projected features in float32 for parity checks, `[tokens, text_hidden]`.
        pub fn features_f32(&self, clip: &Clip) -> Result<(Array, Array)> {
            let encoded = self.encode(&log_mel(clip), clip.frames())?;
            let (weight, bias) = &self.projector_norm;
            let x = encoded
                .layer_norm_without_weight(1e-5)?
                .mul(weight)?
                .add(bias)?;
            let x = self.linear_2.forward(&self.linear_1.forward(&x)?.gelu()?)?;
            Ok((encoded, x))
        }
    }
}
#[cfg(feature = "mlx")]
pub use runtime::AudioEncoder;

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(channels: u16, rate: u32, samples: &[i16]) -> Vec<u8> {
        let data = samples.len() * 2;
        let mut out = Vec::new();
        out.extend(b"RIFF");
        out.extend(((36 + data) as u32).to_le_bytes());
        out.extend(b"WAVEfmt ");
        out.extend(16u32.to_le_bytes());
        out.extend(1u16.to_le_bytes());
        out.extend(channels.to_le_bytes());
        out.extend(rate.to_le_bytes());
        out.extend((rate * 2 * u32::from(channels)).to_le_bytes());
        out.extend((2 * channels).to_le_bytes());
        out.extend(16u16.to_le_bytes());
        out.extend(b"LIST");
        out.extend(4u32.to_le_bytes());
        out.extend(b"INFO");
        out.extend(b"data");
        out.extend((data as u32).to_le_bytes());
        for s in samples {
            out.extend(s.to_le_bytes());
        }
        out
    }

    #[test]
    fn frames_follow_the_feature_extractor() {
        assert_eq!(frame_count(1), 1);
        assert_eq!(frame_count(400), 1);
        assert_eq!(frame_count(401), 2);
        assert_eq!(frame_count(560), 2);
        assert_eq!(frame_count(561), 3);
        assert_eq!(frame_count(16_000), 99);
        for n in [1usize, 399, 400, 401, 559, 560, 561, 16_000, 119_164] {
            let padded = padded_length(n);
            assert!(
                padded >= n.max(N_FFT)
                    && (padded - N_FFT).is_multiple_of(HOP)
                    && padded - n.max(N_FFT) < HOP
            );
        }
        let clip = Clip {
            samples: vec![0.; 16_000],
        };
        assert_eq!(clip.token_count(), 13);
    }

    #[test]
    fn wav_reads_pcm_and_skips_chunks() {
        let (channels, rate) = read_wav(&wav(2, 22_050, &[0, 16_384, -32_768, 32_767])).unwrap();
        assert_eq!(rate, 22_050);
        assert_eq!(channels, vec![vec![0., -1.], vec![0.5, 32_767. / 32_768.]]);
        let clip = Clip::from_wav(&wav(1, 16_000, &[100; 800])).unwrap();
        assert_eq!(clip.samples.len(), 800);
        let mut bad = wav(1, 16_000, &[1, 2]);
        bad[20] = 3; // IEEE float tag
        assert!(read_wav(&bad).is_err());
        assert!(read_wav(b"RIFF\0\0\0\0WAVE").is_err());
        assert!(read_wav(b"not a wav").is_err());
        assert!(Clip::from_wav(&wav(1, 16_000, &[])).is_err());
    }

    #[test]
    fn resampling_keeps_length_and_low_tones() {
        let tone: Vec<f32> = (0..22_050)
            .map(|i| (2. * std::f32::consts::PI * 300. * i as f32 / 22_050.).sin())
            .collect();
        let out = resample(&tone, 22_050, 16_000).unwrap();
        assert_eq!(out.len(), 16_000);
        // a 300 Hz tone survives: compare with the analytic tone away from the edges
        let error = (2_000..14_000)
            .map(|i| (out[i] - (2. * std::f32::consts::PI * 300. * i as f32 / 16_000.).sin()).abs())
            .fold(0f32, f32::max);
        assert!(error < 0.01, "{error}");
        assert_eq!(resample(&tone, 16_000, 16_000).unwrap(), tone);
    }

    #[test]
    fn mel_filters_are_slaney_triangles() {
        let filters = mel_filters();
        for m in 0..MEL_BINS {
            let column: Vec<f64> = (0..BINS).map(|k| filters[k * MEL_BINS + m]).collect();
            assert!(column.iter().all(|v| *v >= 0. && v.is_finite()));
            assert!(column.iter().any(|v| *v > 0.), "empty filter {m}");
        }
        assert_eq!(mel_to_hertz(hertz_to_mel(440.)).round(), 440.);
        assert_eq!(mel_to_hertz(hertz_to_mel(4_000.)).round(), 4_000.);
        let silence = log_mel(&Clip {
            samples: vec![0.; 1_600],
        });
        assert!(silence.iter().all(|v| *v == -2.));
    }
}
