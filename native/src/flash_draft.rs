//! Test prototype for the pinned Flash Next drafts; no Desktop activation.
//! Equations: SGLang 3c560089 DFlash2 and e00930c5 DSpark (Apache-2.0).
use super::*;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Dflash2,
    Dspark,
}

impl Kind {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "dflash2" => Ok(Self::Dflash2),
            "dspark" => Ok(Self::Dspark),
            _ => anyhow::bail!("unknown Flash draft kind"),
        }
    }
    fn captures(self) -> [usize; 5] {
        match self {
            Self::Dflash2 => [5, 15, 25, 35, 45],
            Self::Dspark => [1, 12, 23, 34, 45],
        }
    }
    fn mask(self) -> u32 {
        if self == Self::Dflash2 {
            248077
        } else {
            248070
        }
    }
    fn rope(self) -> f32 {
        if self == Self::Dflash2 {
            10_000.0
        } else {
            10_000_000.0
        }
    }
    fn intermediate(self) -> i32 {
        if self == Self::Dflash2 { 7680 } else { 9728 }
    }
    fn rows(self, proposals: usize) -> Result<i32> {
        ensure!((1..=7).contains(&proposals), "invalid draft proposal count");
        Ok(i32::try_from(proposals)? + i32::from(self == Self::Dflash2))
    }
}

struct Draft {
    kind: Kind,
    weights: BTreeMap<String, Array>,
    cache: Vec<Option<(Array, Array)>>,
    offset: i32,
    policy: PrefixPolicy,
    record_rounds: bool,
}

#[derive(Default)]
enum PrefixPolicy {
    #[default]
    Fixed,
    Forced(usize),
    Adaptive(PrefixPlan),
}

struct PrefixPlan {
    temperatures: [f64; 3],
    costs: [f64; 4],
    calibration_prompt_ids: Vec<String>,
}

impl PrefixPlan {
    fn parse(data: &Value, identity: &Value) -> Result<Self> {
        ensure!(
            data["status"] == "calibrated_greedy_profile"
                && data["version"] == 1
                && data["query_width"] == 3
                && &data["identity"] == identity,
            "foreign or incomplete DSpark profile"
        );
        let temperatures: [f64; 3] = serde_json::from_value(data["temperatures"].clone())?;
        let costs: [f64; 4] = serde_json::from_value(data["round_seconds"].clone())?;
        let calibration_prompt_ids: Vec<String> =
            serde_json::from_value(data["calibration_prompt_ids"].clone())?;
        ensure!(
            temperatures
                .iter()
                .all(|t| t.is_finite() && (0.25..=8.).contains(t))
                && flash_support::draft_prefix(&[0.5; 3], &costs).is_some()
                && calibration_prompt_ids.len() >= 2
                && calibration_prompt_ids
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == calibration_prompt_ids.len()
                && calibration_prompt_ids
                    .iter()
                    .all(|s| s.len() == 16 && s.bytes().all(|c| c.is_ascii_hexdigit())),
            "invalid DSpark calibration"
        );
        ensure!(
            data["calibration_blocks"].as_u64().is_some_and(|n| n >= 32),
            "insufficient DSpark calibration blocks"
        );
        let observations: [u64; 4] = serde_json::from_value(data["cost_observations"].clone())?;
        ensure!(
            observations.iter().all(|&n| n >= 16),
            "insufficient DSpark cost observations"
        );
        Ok(Self {
            temperatures,
            costs,
            calibration_prompt_ids,
        })
    }
}

impl PrefixPolicy {
    fn choose(&self, confidence: &[f32], count: usize) -> Result<usize> {
        match self {
            Self::Fixed => Ok(count),
            Self::Forced(prefix) => {
                ensure!(*prefix <= 3, "invalid forced prefix");
                Ok((*prefix).min(count))
            }
            Self::Adaptive(plan) => {
                ensure!(
                    count <= 3 && confidence.len() == count,
                    "incomplete DSpark confidence"
                );
                let calibrated = confidence
                    .iter()
                    .zip(plan.temperatures)
                    .map(|(&p, t)| -> Result<f64> {
                        ensure!(
                            p.is_finite() && (0.0..=1.0).contains(&p),
                            "invalid DSpark probability"
                        );
                        let p = f64::from(p).clamp(1e-6, 1. - 1e-6);
                        let z = (p.ln() - (-p).ln_1p()) / t;
                        let tail = (-z.abs()).exp();
                        Ok(if z >= 0. {
                            1. / (1. + tail)
                        } else {
                            tail / (1. + tail)
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                flash_support::draft_prefix(&calibrated, &plan.costs[..=count])
                    .context("invalid DSpark prefix cost model")
            }
        }
    }
}

fn prompt_id(tokens: &[u32]) -> String {
    let hash = tokens
        .iter()
        .flat_map(|t| t.to_le_bytes())
        .fold(0xcbf29ce484222325u64, |hash, b| {
            (hash ^ u64::from(b)).wrapping_mul(0x100000001b3)
        });
    format!("{hash:016x}")
}

fn profile_identity(target: &Path, draft: &Path) -> Result<Value> {
    fn stamp(path: &Path) -> Result<Value> {
        let path = path.canonicalize()?;
        let metadata = path.metadata()?;
        Ok(
            serde_json::json!({"path":path,"bytes":metadata.len(),"modified_ns":metadata.modified()?.duration_since(std::time::UNIX_EPOCH)?.as_nanos().to_string()}),
        )
    }
    fn files(root: &Path) -> Result<Vec<Value>> {
        let mut paths = std::fs::read_dir(root)?
            .map(|p| p.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        paths.sort();
        paths
            .into_iter()
            .filter(|p| {
                p.extension()
                    .is_some_and(|x| x == "json" || x == "safetensors" || x == "bin")
            })
            .map(|p| stamp(&p))
            .collect()
    }
    let mut options = BTreeMap::new();
    for key in [
        "MLXL3_SSD_CACHE_GIB",
        "MLXL3_ALLOCATOR_CACHE_MIB",
        "MLXL3_SSD_READERS",
        "MLXL3_SSD_OVERLAP",
        "MLXL3_SSD_LOOKAHEAD_DEPTH",
        "MLXL3_FLASH_THINKING",
    ] {
        options.insert(key, std::env::var(key).ok());
    }
    Ok(
        serde_json::json!({"target":files(target)?,"draft":files(draft)?,"binary":stamp(&std::env::current_exe()?)?,"options":options,"query_width":3,"context":4096,"greedy":true}),
    )
}

fn configure_policy(draft: &mut Draft, target_path: &Path) -> Result<()> {
    let profile = std::env::var("MLXL3_FLASH_PREFIX_PROFILE").ok();
    let forced = std::env::var("MLXL3_FLASH_FORCE_PREFIX").ok();
    ensure!(
        profile.is_none() || forced.is_none(),
        "choose profile or forced prefix"
    );
    if let Some(path) = profile {
        use std::io::Read;
        ensure!(draft.kind == Kind::Dspark, "prefix profile requires DSpark");
        let mut payload = Vec::new();
        File::open(path)?.take(65537).read_to_end(&mut payload)?;
        ensure!(payload.len() <= 65536, "oversized DSpark profile");
        let data: Value = serde_json::from_slice(&payload)?;
        let path = std::env::var("MLXL3_FLASH_DRAFT_MODEL")?;
        let identity = profile_identity(target_path, Path::new(&path))?;
        draft.policy = PrefixPolicy::Adaptive(PrefixPlan::parse(&data, &identity)?);
    } else if let Some(value) = forced {
        let prefix = value.parse::<usize>()?;
        ensure!(prefix <= 3, "forced prefix must be 0..3");
        draft.policy = PrefixPolicy::Forced(prefix);
    }
    Ok(())
}

fn residual_norm(x: &Array, residual: &Array, weight: &Array) -> Result<(Array, Array)> {
    ensure!(
        x.shape().len() == 2
            && (1..=8).contains(&x.shape()[0])
            && x.shape()[1] == H
            && residual.shape() == x.shape()
            && weight.shape() == [H]
            && [x, residual, weight]
                .iter()
                .all(|a| a.dtype() == Dtype::BFloat16),
        "invalid Flash draft residual norm"
    );
    let sum = x
        .astype(Dtype::Float32)?
        .add(&residual.astype(Dtype::Float32)?)?;
    let inverse = sum
        .mul(&sum)?
        .sum(-1, true)?
        .scalar_mul(1. / H as f32)?
        .add(&Array::from_f32(&[1e-6], &[])?)?
        .rsqrt()?;
    let retained = sum.astype(Dtype::BFloat16)?;
    // SGLang normalizes the rounded residual with the unrounded sum's variance.
    let normalized = retained
        .astype(Dtype::Float32)?
        .mul(&weight.astype(Dtype::Float32)?)?
        .mul(&inverse)?
        .astype(Dtype::BFloat16)?;
    Ok((normalized, retained))
}

#[test]
#[ignore = "physical Apple GPU; independent BF16 residual-norm counterexample; run alone"]
fn flash_residual_norm_matches_independent_counterexample() -> Result<()> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../docs/measurements/flash-next-ssd-impl/opt-43-mtp-route-head/drafts-speed/residual-norm-counterexample.json"
    ))?;
    let case = &fixture["results"][0];
    let values = |name: &str| -> Result<Vec<f32>> {
        let pattern = case[name]
            .as_array()
            .context("missing residual norm fixture")?;
        ensure!(pattern.len() == 16, "incomplete norm fixture");
        (0..H as usize)
            .map(|i| {
                Ok(f32::from_bits(u32::try_from(
                    pattern[i % 16].as_u64().context("invalid fixture bits")?,
                )?))
            })
            .collect()
    };
    let x = Array::from_f32(&values("x_bits")?, &[1, H])?.astype(Dtype::BFloat16)?;
    let residual = Array::from_f32(&values("residual_bits")?, &[1, H])?.astype(Dtype::BFloat16)?;
    let weight = Array::from_f32(&vec![1.; H as usize], &[H])?.astype(Dtype::BFloat16)?;
    let (normalized, retained) = residual_norm(&x, &residual, &weight)?;
    let actual = normalized.astype(Dtype::Float32)?.to_f32()?;
    ensure!(
        actual == values("guide_bits")?,
        "residual norm differs from independent fixture"
    );
    ensure!(
        retained.to_bytes()? == x.add(&residual)?.to_bytes()?,
        "retained residual differs"
    );
    let old = x
        .add(&residual)?
        .rms_norm(&weight, 1e-6)?
        .astype(Dtype::Float32)?
        .to_f32()?;
    ensure!(
        actual.iter().zip(&old).filter(|(a, b)| a != b).count() == 320,
        "old rounding mutation escaped"
    );
    for shape in [[0, H], [9, H], [1, H - 1]] {
        let invalid = Array::zeros_dtype(&shape, Dtype::BFloat16)?;
        ensure!(residual_norm(&invalid, &invalid, &weight).is_err());
    }
    ensure!(residual_norm(&x.astype(Dtype::Float16)?, &residual, &weight).is_err());
    println!(
        "FLASH_DRAFT_NORM {}",
        serde_json::json!({"status":"passed","count":actual.len(),"different_from_old":320,"memory":array::memory_stats(false)?})
    );
    Ok(())
}

impl Draft {
    fn reset(&mut self) {
        self.cache = (0..5).map(|_| None).collect();
        self.offset = 0;
    }
    fn load(path: &Path, kind: Kind) -> Result<Self> {
        let tensors = crate::checkpoint::read_header(&path.join("model.safetensors"))
            .context("reading raw BF16 Flash draft header")?;
        let config: Value = serde_json::from_reader(File::open(path.join("config.json"))?)?;
        let architecture = if kind == Kind::Dflash2 {
            "DFlash2DraftModel"
        } else {
            "DSparkDraftModel"
        };
        ensure!(
            config["architectures"] == serde_json::json!([architecture])
                && config["hidden_size"] == H
                && config["num_hidden_layers"] == 5
                && config["intermediate_size"] == kind.intermediate()
                && config["num_attention_heads"] == 32
                && config["num_key_value_heads"] == 8
                && config["head_dim"] == 128
                && config["vocab_size"] == 248320
                && config["rms_norm_eps"] == 1e-6
                && config["attention_bias"] == false
                && config["dflash_config"]["target_layer_ids"]
                    == serde_json::json!(kind.captures())
                && config["dflash_config"]["mask_token_id"] == kind.mask(),
            "unsupported Flash draft configuration"
        );
        if kind == Kind::Dflash2 {
            ensure!(
                config["rope_parameters"]["rope_theta"] == 10000
                    && config["sliding_window"] == 2048
                    && config["use_sliding_window"] == true
                    && config["dflash_config"]["conv_group_size"] == 16
                    && config["dflash_config"]["conv_kernel_size"] == 2
                    && config["dflash_config"]["selector_rank"] == 256
                    && config["dflash_config"]["selector_top_k"] == 16,
                "unsupported DFlash2 geometry"
            );
        } else {
            ensure!(
                config["rope_theta"] == 10_000_000
                    && config["sliding_window"].is_null()
                    && config["use_sliding_window"] == false
                    && config["dflash_config"]["markov_head_type"] == "vanilla"
                    && config["dflash_config"]["markov_rank"] == 256
                    && config["dflash_config"]["confidence_head_with_markov"] == true
                    && config["sample_from_anchor"].as_bool().unwrap_or(true),
                "unsupported DSpark geometry"
            );
        }
        let mut shapes = BTreeMap::new();
        shapes.insert("fc.weight".to_owned(), vec![H, 5 * H]);
        for name in ["hidden_norm.weight", "norm.weight"] {
            shapes.insert(name.to_owned(), vec![H]);
        }
        for layer in 0..5 {
            let p = format!("layers.{layer}");
            for name in ["input_layernorm", "post_attention_layernorm"] {
                shapes.insert(format!("{p}.{name}.weight"), vec![H]);
            }
            for name in ["q_norm", "k_norm"] {
                shapes.insert(format!("{p}.self_attn.{name}.weight"), vec![128]);
            }
            for (name, output, input) in [
                ("q_proj", 4096, H),
                ("k_proj", 1024, H),
                ("v_proj", 1024, H),
                ("o_proj", H, 4096),
            ] {
                shapes.insert(format!("{p}.self_attn.{name}.weight"), vec![output, input]);
            }
            for name in ["gate_proj", "up_proj"] {
                shapes.insert(
                    format!("{p}.mlp.{name}.weight"),
                    vec![kind.intermediate(), H],
                );
            }
            shapes.insert(
                format!("{p}.mlp.down_proj.weight"),
                vec![H, kind.intermediate()],
            );
            if kind == Kind::Dflash2 {
                for name in ["attention_conv", "mlp_conv"] {
                    shapes.insert(format!("{p}.{name}.base_kernel"), vec![2, 2, H]);
                    shapes.insert(format!("{p}.{name}.kernel_projection.weight"), vec![640, H]);
                }
            }
        }
        if kind == Kind::Dflash2 {
            shapes.insert(
                "candidate_selector.hidden_projection.weight".to_owned(),
                vec![256, H],
            );
            for name in ["predecessor_codebook", "successor_codebook"] {
                shapes.insert(format!("candidate_selector.{name}"), vec![248320, 256]);
            }
        } else {
            for name in ["markov_w1", "markov_w2"] {
                shapes.insert(format!("markov_head.{name}.weight"), vec![248320, 256]);
            }
            shapes.insert("confidence_head.proj.weight".to_owned(), vec![1, 2816]);
            shapes.insert("confidence_head.proj.bias".to_owned(), vec![1]);
        }
        ensure!(
            tensors.len() == shapes.len(),
            "unexpected or missing draft tensors"
        );
        let mut weights = BTreeMap::new();
        for (name, shape) in shapes {
            let info = tensors
                .get(&name)
                .with_context(|| format!("missing {name}"))?;
            ensure!(
                info.dtype == "BF16"
                    && info.shape == shape.iter().map(|&n| n as usize).collect::<Vec<_>>(),
                "invalid draft tensor {name}"
            );
            let (file, offset) = info.source()?;
            let weight = Array::from_file(file, offset, &shape, Dtype::BFloat16)?;
            weight.eval()?;
            weights.insert(name, weight);
        }
        Ok(Self {
            kind,
            weights,
            cache: (0..5).map(|_| None).collect(),
            offset: 0,
            policy: PrefixPolicy::Fixed,
            record_rounds: false,
        })
    }
    fn weight(&self, name: &str) -> &Array {
        &self.weights[name]
    }
    fn linear(&self, x: &Array, prefix: &str) -> Result<Array> {
        x.matmul(
            &self
                .weight(&format!("{prefix}.weight"))
                .transpose(&[1, 0])?,
        )
    }
    fn norm(&self, x: &Array, prefix: &str) -> Result<Array> {
        x.rms_norm(self.weight(&format!("{prefix}.weight")), 1e-6)
    }
    fn append(&mut self, captured: &Array, start: i32) -> Result<()> {
        ensure!(
            captured.shape().len() == 2
                && captured.shape()[0] > 0
                && captured.shape()[1] == 5 * H
                && captured.dtype() == Dtype::BFloat16
                && start == self.offset
                && start.checked_add(captured.shape()[0]).is_some(),
            "invalid draft capture append"
        );
        let rows = captured.shape()[0];
        let context = self.norm(&self.linear(captured, "fc")?, "hidden_norm")?;
        let mut cache = Vec::with_capacity(5);
        for layer in 0..5 {
            let p = format!("layers.{layer}.self_attn");
            let keys = self
                .norm(
                    &self
                        .linear(&context, &format!("{p}.k_proj"))?
                        .reshape(&[1, rows, 8, 128])?,
                    &format!("{p}.k_norm"),
                )?
                .transpose(&[0, 2, 1, 3])?
                .rope(128, self.kind.rope(), start)?;
            let values = self
                .linear(&context, &format!("{p}.v_proj"))?
                .reshape(&[1, rows, 8, 128])?
                .transpose(&[0, 2, 1, 3])?;
            let (keys, values) = match &self.cache[layer] {
                Some((old_k, old_v)) => (
                    Array::concatenate(&[old_k, &keys], 2)?,
                    Array::concatenate(&[old_v, &values], 2)?,
                ),
                None => (keys, values),
            };
            let length = keys.shape()[2];
            let first = if self.kind == Kind::Dflash2 {
                (length - 2048).max(0)
            } else {
                0
            };
            let keys = keys.slice(2, first, length)?;
            let values = values.slice(2, first, length)?;
            keys.eval()?;
            values.eval()?;
            cache.push(Some((keys, values)));
        }
        self.cache = cache;
        self.offset = start + rows;
        Ok(())
    }
    fn convolution(&self, x: &Array, dynamic: &Array, prefix: &str, side: i32) -> Result<Array> {
        let rows = x.shape()[0];
        let delta = dynamic
            .reshape(&[rows, 2, 2, 160, 1])?
            .slice(1, side, side + 1)?
            .broadcast_to(&[rows, 1, 2, 160, 16])?
            .reshape(&[rows, 2, H])?;
        let base = self
            .weight(&format!("{prefix}.base_kernel"))
            .slice(0, side, side + 1)?;
        let coefficient = delta.add(&base)?;
        let current = x.mul(&coefficient.slice(1, 0, 1)?.reshape(&[rows, H])?)?;
        let shifted = Array::concatenate(
            &[
                &Array::zeros_dtype(&[1, H], Dtype::BFloat16)?,
                &x.slice(0, 0, rows - 1)?,
            ],
            0,
        )?;
        current.add(&shifted.mul(&coefficient.slice(1, 1, 2)?.reshape(&[rows, H])?)?)
    }
    fn sublayer_input(&self, x: &Array, prefix: &str) -> Result<(Array, Option<Array>)> {
        if self.kind == Kind::Dflash2 {
            let dynamic = self.linear(x, &format!("{prefix}.kernel_projection"))?;
            Ok((self.convolution(x, &dynamic, prefix, 0)?, Some(dynamic)))
        } else {
            Ok((x.try_clone()?, None))
        }
    }
    fn sublayer_output(&self, x: &Array, dynamic: Option<&Array>, prefix: &str) -> Result<Array> {
        if let Some(dynamic) = dynamic {
            self.convolution(x, dynamic, prefix, 1)
        } else {
            x.try_clone()
        }
    }
    fn attention(&self, x: &Array, layer: usize, position: i32) -> Result<Array> {
        let rows = x.shape()[0];
        let p = format!("layers.{layer}.self_attn");
        let q = self
            .norm(
                &self
                    .linear(x, &format!("{p}.q_proj"))?
                    .reshape(&[1, rows, 32, 128])?,
                &format!("{p}.q_norm"),
            )?
            .transpose(&[0, 2, 1, 3])?
            .rope(128, self.kind.rope(), position)?;
        let k = self
            .norm(
                &self
                    .linear(x, &format!("{p}.k_proj"))?
                    .reshape(&[1, rows, 8, 128])?,
                &format!("{p}.k_norm"),
            )?
            .transpose(&[0, 2, 1, 3])?
            .rope(128, self.kind.rope(), position)?;
        let v = self
            .linear(x, &format!("{p}.v_proj"))?
            .reshape(&[1, rows, 8, 128])?
            .transpose(&[0, 2, 1, 3])?;
        let (keys, values, past) = match &self.cache[layer] {
            Some((old_k, old_v)) => (
                Array::concatenate(&[old_k, &k], 2)?,
                Array::concatenate(&[old_v, &v], 2)?,
                old_k.shape()[2],
            ),
            None => (k, v, 0),
        };
        let output = if self.kind == Kind::Dflash2 && past > 0 {
            let total = past + rows;
            let mut mask = vec![f32::NEG_INFINITY; (rows * total) as usize];
            for row in 0..rows {
                for col in (past + row + 1 - 2048).max(0)..past {
                    mask[(row * total + col) as usize] = 0.;
                }
                for col in past..total {
                    mask[(row * total + col) as usize] = 0.;
                }
            }
            Array::sdpa_mask(
                &q,
                &keys,
                &values,
                128f32.sqrt().recip(),
                &Array::from_f32(&mask, &[1, 1, rows, total])?.astype(Dtype::BFloat16)?,
            )?
        } else {
            Array::sdpa(&q, &keys, &values, 128f32.sqrt().recip(), false)?
        };
        self.linear(
            &output
                .transpose(&[0, 2, 1, 3])?
                .reshape(&[rows, 4096])?
                .astype(Dtype::BFloat16)?,
            &format!("{p}.o_proj"),
        )
    }
    fn forward(&self, input: &Array, proposals: usize) -> Result<(Array, Vec<Array>)> {
        let rows = self.kind.rows(proposals)?;
        ensure!(
            input.shape() == [rows, H] && input.dtype() == Dtype::BFloat16 && self.offset > 0,
            "invalid Flash draft input"
        );
        let mut hidden = input.try_clone()?;
        let mut normalized = self.norm(input, "layers.0.input_layernorm")?;
        let mut trace = Vec::new();
        for layer in 0..5 {
            let p = format!("layers.{layer}");
            let (x, dynamic) = self.sublayer_input(&normalized, &format!("{p}.attention_conv"))?;
            let attention = self.sublayer_output(
                &self.attention(&x, layer, self.offset)?,
                dynamic.as_ref(),
                &format!("{p}.attention_conv"),
            )?;
            let (normalized_mlp, residual) = residual_norm(
                &attention,
                &hidden,
                self.weight(&format!("{p}.post_attention_layernorm.weight")),
            )?;
            let (x, dynamic) = self.sublayer_input(&normalized_mlp, &format!("{p}.mlp_conv"))?;
            let gate = self.linear(&x, &format!("{p}.mlp.gate_proj"))?;
            let up = self.linear(&x, &format!("{p}.mlp.up_proj"))?;
            let mlp = self.linear(&gate.swiglu(&up)?, &format!("{p}.mlp.down_proj"))?;
            let mlp = self.sublayer_output(&mlp, dynamic.as_ref(), &format!("{p}.mlp_conv"))?;
            let next_norm = if layer == 4 {
                "norm.weight".to_owned()
            } else {
                format!("layers.{}.input_layernorm.weight", layer + 1)
            };
            (normalized, hidden) = residual_norm(&mlp, &residual, self.weight(&next_norm))?;
            trace.push(hidden.try_clone()?);
        }
        Ok((normalized, trace))
    }

    fn select(
        &self,
        head: &Projection,
        hidden: &Array,
        proposals: usize,
        anchor: u32,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let rows = self.kind.rows(proposals)?;
        ensure!(
            hidden.shape() == [rows, H] && hidden.dtype() == Dtype::BFloat16 && anchor < 248320,
            "invalid draft selection input"
        );
        let first = i32::from(self.kind == Kind::Dflash2);
        let selected = hidden.slice(0, first, rows)?;
        let logits = head
            .forward(&selected.astype(Dtype::Float16)?)?
            .astype(Dtype::BFloat16)?;
        self.select_logits(&logits, &selected, anchor)
    }

    fn select_logits(
        &self,
        logits: &Array,
        hidden: &Array,
        anchor: u32,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        ensure!(logits.shape().len() == 2, "invalid selector rank");
        let rows = logits.shape()[0];
        ensure!(
            logits.shape() == [rows, 248320]
                && hidden.shape() == [rows, H]
                && (1..=7).contains(&rows)
                && logits.dtype() == Dtype::BFloat16
                && hidden.dtype() == Dtype::BFloat16
                && anchor < 248320,
            "invalid draft selector tensors"
        );
        let mut previous = Array::from_u32(&[anchor], &[1])?;
        let mut output = Vec::new();
        let mut confidence = Vec::new();
        let validity;
        if self.kind == Kind::Dflash2 {
            let (ids, unary, valid) = top16(logits)?;
            validity = Some(valid);
            let projected = self.linear(hidden, "candidate_selector.hidden_projection")?;
            let predecessors = Array::concatenate(
                &[
                    &previous.reshape(&[1, 1])?.broadcast_to(&[1, 16])?,
                    &ids.slice(0, 0, rows - 1)?,
                ],
                0,
            )?;
            let pred = self
                .weight("candidate_selector.predecessor_codebook")
                .take(&predecessors.reshape(&[rows * 16])?, 0)?
                .reshape(&[rows, 16, 256])?;
            let keys = self
                .weight("candidate_selector.successor_codebook")
                .take(&ids.reshape(&[rows * 16])?, 0)?
                .reshape(&[rows, 16, 256])?;
            let edges = pred
                .mul(&projected.reshape(&[rows, 1, 256])?)?
                .matmul(&keys.transpose(&[0, 2, 1])?)?
                .astype(Dtype::Float32)?;
            let maps = edges.add(&unary.reshape(&[rows, 1, 16])?)?.argmax()?;
            let mut rank = Array::from_u32(&[0], &[1])?;
            for row in 0..rows {
                rank = maps
                    .slice(0, row, row + 1)?
                    .reshape(&[16])?
                    .take(&rank, 0)?;
                output.push(ids.slice(0, row, row + 1)?.reshape(&[16])?.take(&rank, 0)?);
            }
        } else {
            validity = None;
            let w = self
                .weight("confidence_head.proj.weight")
                .astype(Dtype::Float32)?
                .transpose(&[1, 0])?;
            let b = self
                .weight("confidence_head.proj.bias")
                .astype(Dtype::Float32)?;
            for row in 0..rows {
                let latent = self
                    .weight("markov_head.markov_w1.weight")
                    .take(&previous, 0)?;
                let bias = latent.matmul(
                    &self
                        .weight("markov_head.markov_w2.weight")
                        .transpose(&[1, 0])?,
                )?;
                let corrected = logits.slice(0, row, row + 1)?.add(&bias)?;
                previous = corrected.argmax()?;
                let features = Array::concatenate(&[&hidden.slice(0, row, row + 1)?, &latent], 1)?
                    .astype(Dtype::Float32)?;
                confidence.push(features.matmul(&w)?.add(&b)?.sigmoid()?.reshape(&[1])?);
                output.push(previous.try_clone()?);
            }
        }
        let tokens = Array::concatenate(&output.iter().collect::<Vec<_>>(), 0)?;
        let output = if let Some(valid) = validity {
            let mut checked = Array::concatenate(&[&tokens, &valid], 0)?.to_u32()?;
            ensure!(
                checked.len() == 2 * rows as usize
                    && checked[rows as usize..].iter().all(|&v| v == 1),
                "invalid top16 candidates"
            );
            checked.truncate(rows as usize);
            checked
        } else {
            tokens.to_u32()?
        };
        ensure!(
            output.len() == rows as usize && output.iter().all(|&t| t < 248320),
            "invalid selected tokens"
        );
        let confidence = if confidence.is_empty() {
            Vec::new()
        } else {
            let values = Array::concatenate(&confidence.iter().collect::<Vec<_>>(), 0)?.to_f32()?;
            ensure!(
                values.len() == rows as usize
                    && values
                        .iter()
                        .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
                "invalid confidence"
            );
            values
        };
        Ok((output, confidence))
    }
}

fn top16(logits: &Array) -> Result<(Array, Array, Array)> {
    ensure!(
        logits.shape().len() == 2
            && (1..=7).contains(&logits.shape()[0])
            && logits.shape()[1] == 248320
            && logits.dtype() == Dtype::BFloat16,
        "invalid top16 input"
    );
    let rows = logits.shape()[0];
    let header = format!(
        "#define DFLASH_STAGE 1\n#define DFLASH_VOCABULARY 248320u\n{}",
        include_str!("../shaders/dflash_select.h")
    );
    let mut partial = array::metal_kernel(
        "mlxl3_flash_draft_top16_v1",
        &["logits"],
        &["partial_ids", "partial_values"],
        &header,
        include_str!("../shaders/dflash_select.metal"),
        &[logits],
        &[vec![rows * 8, 16], vec![rows * 8, 16]],
        &[Dtype::UInt32, Dtype::Float32],
        [rows * 8 * 256, 1, 1],
        [256, 1, 1],
    )?;
    let values = partial.pop().context("missing partial values")?;
    let ids = partial.pop().context("missing partial IDs")?;
    let mut merged = array::metal_kernel(
        "mlxl3_flash_draft_top16_merge_v1",
        &["partial_ids", "partial_values"],
        &["candidates", "unary", "valid_rows"],
        include_str!("../shaders/dflash_select.h"),
        include_str!("../shaders/flash_draft_top16.metal"),
        &[&ids, &values],
        &[vec![rows, 16], vec![rows, 16], vec![rows]],
        &[Dtype::UInt32, Dtype::Float32, Dtype::UInt32],
        [rows * 32, 1, 1],
        [32, 1, 1],
    )?;
    let valid = merged.pop().context("missing top16 validity")?;
    let unary = merged.pop().context("missing top16 unary")?;
    let candidates = merged.pop().context("missing top16 candidates")?;
    Ok((candidates, unary, valid))
}

fn mean_streams(streams: &Array) -> Result<Array> {
    ensure!(
        streams.shape() == [1, WIDTH] && streams.dtype() == Dtype::Float32,
        "invalid captured HC streams"
    );
    streams
        .reshape(&[1, HC, H])?
        .sum(1, false)?
        .scalar_mul(0.25)?
        .astype(Dtype::BFloat16)
}

struct BlockParts {
    logits: Array,
    captured: Array,
    prefixes: Vec<Vec<LayerState>>,
    hidden: Vec<Array>,
}

/// The mutable borrow prevents forwarding before commit; dropping resets state.
struct Verification<'a> {
    target: &'a mut FlashNext,
    base: i32,
    parts: BlockParts,
    committed: bool,
}

impl Verification<'_> {
    fn commit(mut self, retained: usize) -> Result<()> {
        let total = self.parts.hidden.len();
        let offset = crate::flash_support::draft_commit_offset(
            self.base,
            self.target.offset,
            total,
            retained,
        )
        .context("invalid or stale Flash draft commit")?;
        ensure!(
            self.parts.prefixes.len() == total - 1,
            "invalid verification tape"
        );
        if retained < total {
            let hidden = self.parts.hidden[retained - 1].try_clone()?;
            let states = std::mem::take(&mut self.parts.prefixes[retained - 1]);
            ensure!(
                states.len() == self.target.layers.len(),
                "incomplete Flash draft snapshot"
            );
            for (layer, state) in self.target.layers.iter_mut().zip(states) {
                layer.restore(state)?;
            }
            self.target.last_hidden = Some(hidden);
        }
        self.target.offset = offset;
        self.committed = true;
        Ok(())
    }
}

impl Drop for Verification<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.target.reset();
        }
    }
}

impl FlashNext {
    fn forward_captured(&mut self, token: u32, kind: Kind) -> Result<(Array, Array)> {
        ensure!(
            self.verification.is_none() && self.offset < self.context && token < 248320,
            "invalid captured Flash forward"
        );
        let result = (|| -> Result<_> {
            let mut streams = self.streams(token)?;
            let mut captures = Vec::with_capacity(5);
            for i in 0..self.layers.len() {
                let (current, rest) = self.layers.split_at_mut(i + 1);
                let next = (self.lookahead_depth > 0)
                    .then(|| rest.first_mut().map(|l| &mut l.moe))
                    .flatten();
                streams =
                    current[i].forward_lookahead(&streams, token, next, self.lookahead_depth)?;
                if kind.captures().contains(&i) {
                    captures.push(mean_streams(&streams)?);
                }
            }
            ensure!(captures.len() == 5, "incomplete Flash captures");
            let captured = Array::concatenate(&captures.iter().collect::<Vec<_>>(), 1)?;
            let logits = self.logits(&streams)?;
            captured.eval()?;
            self.drain()?;
            self.last_hidden = Some(streams);
            self.offset += 1;
            Ok((logits, captured))
        })();
        if result.is_err() {
            self.reset();
        }
        result
    }

    fn draft_input(&self, anchor: u32, kind: Kind, proposals: usize) -> Result<Array> {
        ensure!(anchor < 248320, "invalid Flash draft anchor");
        let rows = kind.rows(proposals)?;
        let (file, base) = self.embed.source()?;
        let mut embeddings = Vec::new();
        for row in 0..rows {
            let token = if row == 0 { anchor } else { kind.mask() };
            let offset = flash_support::row_offset(base, u64::from(token), 248320, 5120)
                .context("draft embedding overflow")?;
            embeddings.push(Array::from_file(file, offset, &[1, H], Dtype::BFloat16)?);
        }
        Array::concatenate(&embeddings.iter().collect::<Vec<_>>(), 0)
    }

    fn verify_draft(&mut self, tokens: &[u32], kind: Kind) -> Result<Verification<'_>> {
        ensure!(
            self.verification.is_none()
                && (1..=8).contains(&tokens.len())
                && tokens.len() <= usize::try_from(self.context - self.offset)?
                && tokens.iter().all(|&t| t < 248320),
            "invalid Flash draft verification"
        );
        let base = self.offset;
        let result = (|| -> Result<BlockParts> {
            let mut streams = tokens
                .iter()
                .map(|&t| self.streams(t))
                .collect::<Result<Vec<_>>>()?;
            let mut prefixes: Vec<Vec<LayerState>> = (0..tokens.len() - 1)
                .map(|_| Vec::with_capacity(48))
                .collect();
            let mut captures = Vec::with_capacity(5);
            for (index, layer) in self.layers.iter_mut().enumerate() {
                let mut prepared = Vec::with_capacity(tokens.len());
                for (row, (&token, stream)) in tokens.iter().zip(&streams).enumerate() {
                    prepared.push(layer.prepare(stream, token)?);
                    if row + 1 < tokens.len() {
                        prefixes[row].push(layer.snapshot()?);
                    }
                }
                let inputs =
                    Array::concatenate(&prepared.iter().map(|p| &p.1).collect::<Vec<_>>(), 0)?;
                let values = layer.moe.forward_draft_rows(&inputs)?;
                streams = prepared
                    .iter()
                    .enumerate()
                    .map(|(row, p)| {
                        Hc::write(&p.0, &values.slice(0, row as i32, row as i32 + 1)?, &p.2)
                    })
                    .collect::<Result<Vec<_>>>()?;
                if kind.captures().contains(&index) {
                    let rows = streams
                        .iter()
                        .map(mean_streams)
                        .collect::<Result<Vec<_>>>()?;
                    captures.push(Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0)?);
                }
            }
            ensure!(captures.len() == 5, "incomplete verification captures");
            let heads = streams
                .iter()
                .map(|x| -> Result<Array> {
                    let (mixed, _) = self.mixer.read(x)?;
                    self.head.forward(&mixed.astype(Dtype::Float16)?)
                })
                .collect::<Result<Vec<_>>>()?;
            let logits = Array::concatenate(&heads.iter().collect::<Vec<_>>(), 0)?;
            let captured = Array::concatenate(&captures.iter().collect::<Vec<_>>(), 1)?;
            logits.eval()?;
            captured.eval()?;
            self.drain()?;
            self.last_hidden = Some(streams.last().context("missing last stream")?.try_clone()?);
            self.offset += tokens.len() as i32;
            Ok(BlockParts {
                logits,
                captured,
                prefixes,
                hidden: streams,
            })
        })();
        match result {
            Ok(parts) => Ok(Verification {
                target: self,
                base,
                parts,
                committed: false,
            }),
            Err(error) => {
                self.reset();
                Err(error.context("Flash draft verification reset after failure"))
            }
        }
    }
}

struct Generated {
    tokens: Vec<u32>,
    seconds: f64,
    draft_seconds: f64,
    verification_seconds: f64,
    accepted: usize,
    proposed: usize,
    blocks: usize,
    widths: [usize; 8],
    trace: Vec<Value>,
    decode_io: [u64; 3],
}

fn generate(
    target: &mut FlashNext,
    draft: &mut Draft,
    prompt: &[u32],
    budget: usize,
    width: usize,
) -> Result<Generated> {
    ensure!(
        !prompt.is_empty()
            && (1..=128).contains(&budget)
            && (1..=7).contains(&width)
            && target.offset() == 0
            && draft.offset == 0
            && prompt.iter().all(|&t| t < 248320)
            && prompt
                .len()
                .checked_add(budget - 1)
                .is_some_and(|n| n <= target.context as usize),
        "invalid draft generation budget or state"
    );
    if let PrefixPolicy::Adaptive(plan) = &draft.policy {
        ensure!(
            draft.kind == Kind::Dspark && width == 3,
            "adaptive profile requires DSpark width3"
        );
        ensure!(
            !plan.calibration_prompt_ids.contains(&prompt_id(prompt)),
            "validation prompt overlaps calibration data"
        );
    }
    let operation = (|| -> Result<Generated> {
        let mut last = None;
        for &token in prompt {
            let (logits, captured) = target.forward_captured(token, draft.kind)?;
            draft.append(&captured, target.offset() - 1)?;
            last = Some(logits);
        }
        let anchor = last.context("empty captured prefill")?.chat_greedy_ids()?;
        ensure!(anchor.len() == 1, "invalid prefill token");
        let mut result = Generated {
            tokens: anchor,
            seconds: 0.,
            draft_seconds: 0.,
            verification_seconds: 0.,
            accepted: 0,
            proposed: 0,
            blocks: 0,
            widths: [0; 8],
            trace: Vec::new(),
            decode_io: [0; 3],
        };
        let decode_io_start = target.io_stats();
        let mut pending = VecDeque::new();
        let started = std::time::Instant::now();
        while result.tokens.len() < budget {
            if let Some(next) = pending.pop_front() {
                result.tokens.push(next);
                continue;
            }
            let remaining = budget - result.tokens.len();
            let context = usize::try_from(target.context - target.offset)?;
            ensure!(context > 0, "Flash draft context exhausted");
            let count = width.min(context - 1).min(remaining - 1);
            let anchor = *result.tokens.last().context("missing anchor")?;
            if count == 0 {
                result
                    .tokens
                    .push(target.forward(anchor)?.chat_greedy_ids()?[0]);
                continue;
            }
            let before = std::time::Instant::now();
            let round_started = before;
            let io_before = target.io_stats();
            let input = target.draft_input(anchor, draft.kind, count)?;
            let (hidden, _) = draft.forward(&input, count)?;
            let (mut proposals, confidence) = draft.select(&target.head, &hidden, count, anchor)?;
            ensure!(proposals.len() == count, "incomplete proposal block");
            let selected = draft.policy.choose(&confidence, count)?;
            ensure!(
                selected <= count && selected <= 7,
                "invalid selected prefix"
            );
            proposals.truncate(selected);
            result.draft_seconds += before.elapsed().as_secs_f64();
            let verify = std::iter::once(anchor)
                .chain(proposals.iter().copied())
                .collect::<Vec<_>>();
            let before = std::time::Instant::now();
            let start = target.offset();
            let verification = target.verify_draft(&verify, draft.kind)?;
            let ids = verification.parts.logits.chat_greedy_ids()?;
            let acceptance = crate::speculative::greedy_accept(&proposals, &ids)
                .context("invalid draft acceptance")?;
            let retained = acceptance.accepted_draft_tokens + 1;
            let captured = verification.parts.captured.slice(0, 0, retained as i32)?;
            verification.commit(retained)?;
            draft.append(&captured, start)?;
            result.verification_seconds += before.elapsed().as_secs_f64();
            pending.extend(
                proposals[..acceptance.accepted_draft_tokens]
                    .iter()
                    .copied(),
            );
            pending.push_back(acceptance.target_token);
            result.blocks += 1;
            result.accepted += acceptance.accepted_draft_tokens;
            result.proposed += selected;
            result.widths[selected] += 1;
            if draft.record_rounds {
                let io_after = target.io_stats();
                result.trace.push(serde_json::json!({"drafted":count,"selected":selected,"confidence":confidence,
                    "accepted":acceptance.accepted_draft_tokens,"round_seconds":round_started.elapsed().as_secs_f64(),
                    "decode_io_hits_misses_bytes":[io_after.0-io_before.0,io_after.1-io_before.1,io_after.2-io_before.2]}));
            }
        }
        result.seconds = if budget == 1 {
            0.
        } else {
            started.elapsed().as_secs_f64()
        };
        let decode_io_end = target.io_stats();
        result.decode_io = [
            decode_io_end.0 - decode_io_start.0,
            decode_io_end.1 - decode_io_start.1,
            decode_io_end.2 - decode_io_start.2,
        ];
        ensure!(
            result.tokens.len() == budget
                && pending.is_empty()
                && result.seconds.is_finite()
                && (budget == 1 || result.seconds > 0.),
            "incomplete draft generation: tokens={}, budget={budget}, pending={}, seconds={}",
            result.tokens.len(),
            pending.len(),
            result.seconds
        );
        Ok(result)
    })();
    if operation.is_err() {
        target.reset();
        draft.reset();
    }
    operation
}

fn oracle_array(root: &Path, data: &Value, name: &str) -> Result<Array> {
    let entry = &data["arrays"][name];
    let shape = entry["shape"]
        .as_array()
        .context("missing oracle shape")?
        .iter()
        .map(|v| {
            i32::try_from(v.as_u64().context("invalid oracle dimension")?)
                .map_err(anyhow::Error::from)
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        !shape.is_empty() && shape.iter().all(|&d| d > 0),
        "empty oracle shape"
    );
    let count = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
        .context("oracle count overflow")?;
    ensure!(
        entry["count"].as_u64() == Some(count as u64),
        "invalid oracle count"
    );
    let bytes = std::fs::read(root.join(entry["file"].as_str().context("missing oracle file")?))?;
    ensure!(
        bytes.len() == count.checked_mul(4).context("oracle bytes overflow")?,
        "incomplete oracle file"
    );
    let values = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&v| f32::from_le_bytes(v))
        .collect::<Vec<_>>();
    ensure!(values.iter().all(|v| v.is_finite()), "nonfinite oracle");
    Array::from_f32(&values, &shape)
}

fn compare_oracle(
    root: &Path,
    data: &Value,
    name: &str,
    actual: &Array,
    tolerance: f64,
) -> Result<Value> {
    let expected = oracle_array(root, data, name)?;
    ensure!(
        expected.shape() == actual.shape(),
        "oracle shape differs at {name}"
    );
    let a = actual.astype(Dtype::Float32)?.to_f32()?;
    let b = expected.to_f32()?;
    ensure!(
        !a.is_empty() && a.len() == b.len() && a.iter().all(|v| v.is_finite()),
        "invalid native output at {name}"
    );
    let energy = b.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
    let error = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum::<f64>();
    let relative = (error / energy.max(1e-30)).sqrt();
    ensure!(
        relative <= tolerance,
        "draft oracle differs at {name}: {relative} > {tolerance}"
    );
    Ok(
        serde_json::json!({"name":name,"count":a.len(),"relative_rmse":relative,"tolerance":tolerance,"byte_equal":actual.astype(Dtype::Float32)?.to_bytes()?==expected.to_bytes()?}),
    )
}

#[test]
fn draft_shapes_keep_anchor_conventions_and_reject_unknown_modes() -> Result<()> {
    for count in 1..=7 {
        assert_eq!(Kind::Dflash2.rows(count)?, count as i32 + 1);
        assert_eq!(Kind::Dspark.rows(count)?, count as i32);
    }
    for count in [0, 8, usize::MAX] {
        assert!(Kind::Dflash2.rows(count).is_err());
        assert!(Kind::Dspark.rows(count).is_err());
    }
    for name in ["", "Qwen38Dense", "DFlash2"] {
        assert!(Kind::parse(name).is_err());
    }
    assert_eq!(Kind::parse("dflash2")?.captures(), [5, 15, 25, 35, 45]);
    assert_eq!(Kind::parse("dspark")?.captures(), [1, 12, 23, 34, 45]);
    Ok(())
}

#[test]
fn raw_draft_reader_does_not_require_exl3_metadata_and_rejects_foreign_config() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let header = serde_json::to_vec(
        &serde_json::json!({"weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}),
    )?;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend([0, 0]);
    std::fs::write(directory.path().join("model.safetensors"), bytes)?;
    std::fs::write(
        directory.path().join("config.json"),
        b"{\"architectures\":[\"Foreign\"]}",
    )?;
    ensure!(!directory.path().join("quantization_config.json").exists());
    for kind in [Kind::Dflash2, Kind::Dspark] {
        let error = Draft::load(directory.path(), kind)
            .err()
            .context("foreign draft accepted")?;
        ensure!(
            format!("{error:#}").contains("unsupported Flash draft configuration"),
            "raw BF16 reader failed before configuration validation: {error:#}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU, pinned Flash draft and independent NumPy fixture; run alone"]
fn flash_draft_matches_numpy_backbone_cache_and_selector() -> Result<()> {
    let kind = Kind::parse(&std::env::var("MLXL3_FLASH_DRAFT_KIND")?)?;
    let path = std::env::var("MLXL3_FLASH_DRAFT_MODEL")?;
    let root = std::env::var("MLXL3_FLASH_DRAFT_ORACLE")?;
    let root = Path::new(&root);
    let data: Value = serde_json::from_reader(File::open(root.join("oracle.json"))?)?;
    let count = usize::try_from(
        data["proposals"]
            .as_u64()
            .context("missing proposal count")?,
    )?;
    ensure!(
        data["status"] == "complete_cpu_oracle"
            && data["kind"] == std::env::var("MLXL3_FLASH_DRAFT_KIND")?
            && data["residual_norm"] == "sglang-3c560089-fp32-variance-bf16-residual",
        "incomplete or foreign oracle"
    );
    let mut draft = Draft::load(Path::new(&path), kind)?;
    let other = if kind == Kind::Dflash2 {
        Kind::Dspark
    } else {
        Kind::Dflash2
    };
    ensure!(
        Draft::load(Path::new(&path), other).is_err(),
        "foreign model accepted"
    );
    let captured = oracle_array(root, &data, "captured")?.astype(Dtype::BFloat16)?;
    draft.append(&captured, 0)?;
    let mut checks = Vec::new();
    for (layer, pair) in draft.cache.iter().enumerate() {
        let (keys, values) = pair.as_ref().context("missing cache")?;
        checks.push(compare_oracle(
            root,
            &data,
            &format!("keys-{layer}"),
            keys,
            0.01,
        )?);
        checks.push(compare_oracle(
            root,
            &data,
            &format!("values-{layer}"),
            values,
            0.01,
        )?);
    }
    let cache_before = draft
        .cache
        .iter()
        .map(|p| -> Result<_> {
            let (k, v) = p.as_ref().context("cache")?;
            Ok((k.to_bytes()?, v.to_bytes()?))
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        draft.append(&captured, 0).is_err() && draft.offset == captured.shape()[0],
        "invalid append changed offset"
    );
    for (pair, expected) in draft.cache.iter().zip(cache_before) {
        let (k, v) = pair.as_ref().context("cache")?;
        ensure!(
            (k.to_bytes()?, v.to_bytes()?) == expected,
            "invalid append changed cache"
        );
    }
    let input = oracle_array(root, &data, "input")?.astype(Dtype::BFloat16)?;
    let (hidden, trace) = draft.forward(&input, count)?;
    for (i, value) in trace.iter().enumerate() {
        checks.push(compare_oracle(
            root,
            &data,
            &format!("layer-{i}"),
            value,
            0.02,
        )?);
    }
    checks.push(compare_oracle(root, &data, "hidden", &hidden, 0.02)?);
    ensure!(
        draft
            .forward(&input.astype(Dtype::Float32)?, count)
            .is_err(),
        "draft accepted FP32 input"
    );
    let rows = kind.rows(count)?;
    for shape in [[0, H], [rows + 1, H], [rows, H - 1]] {
        ensure!(
            draft
                .forward(&Array::zeros_dtype(&shape, Dtype::BFloat16)?, count)
                .is_err(),
            "draft accepted invalid shape"
        );
    }
    let logits = oracle_array(root, &data, "selector_logits")?.astype(Dtype::BFloat16)?;
    let selector_hidden = oracle_array(root, &data, "selector_hidden")?.astype(Dtype::BFloat16)?;
    let (tokens, confidence) = draft.select_logits(&logits, &selector_hidden, 7)?;
    ensure!(
        serde_json::json!(tokens) == data["selector_tokens"],
        "selector disagrees with independent oracle"
    );
    if kind == Kind::Dspark {
        let expected = data["selector_confidence"]
            .as_array()
            .context("missing confidence")?;
        ensure!(
            confidence.len() == expected.len() && confidence.len() == count,
            "confidence count mismatch"
        );
        for (&value, expected) in confidence.iter().zip(expected) {
            ensure!(
                (f64::from(value) - expected.as_f64().context("invalid confidence")?).abs() <= 1e-4,
                "confidence differs"
            );
        }
    } else {
        ensure!(confidence.is_empty());
    }
    let coupled_root = std::env::var("MLXL3_FLASH_DRAFT_HEAD_REFERENCE")?;
    let coupled_root = Path::new(&coupled_root);
    let coupled: Value = serde_json::from_reader(File::open(coupled_root.join("reference.json"))?)?;
    ensure!(
        coupled["status"] == "complete"
            && coupled["draft_head"]["kind"] == data["kind"]
            && coupled["draft_head"]["revision"] == data["revision"]
            && coupled["draft_head"]["hidden_sha256"] == data["arrays"]["hidden"]["sha256"],
        "foreign or incomplete coupled head oracle"
    );
    let checkpoint = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let head = Projection::load(&checkpoint, "lm_head", H, 248320, false)?;
    let cpu_hidden = oracle_array(root, &data, "hidden")?.astype(Dtype::BFloat16)?;
    let first = i32::from(kind == Kind::Dflash2);
    let logits = head
        .forward(&cpu_hidden.slice(0, first, rows)?.astype(Dtype::Float16)?)?
        .astype(Dtype::BFloat16)?;
    let head_oracle =
        serde_json::json!({"arrays":{"coupled_logits":coupled["draft_head"]["logits"]}});
    let head_check = compare_oracle(coupled_root, &head_oracle, "coupled_logits", &logits, 0.004)?;
    let (cpu_hidden_tokens, cpu_hidden_confidence) = draft.select(&head, &cpu_hidden, count, 7)?;
    let (actual_tokens, actual_confidence) = draft.select(&head, &hidden, count, 7)?;
    ensure!(
        serde_json::json!(cpu_hidden_tokens) == coupled["draft_head"]["tokens"],
        "selector differs on real independent backbone input: actual={cpu_hidden_tokens:?}, expected={}",
        coupled["draft_head"]["tokens"]
    );
    ensure!(
        serde_json::json!(actual_tokens) == coupled["draft_head"]["tokens"],
        "coupled backbone/head/proposals differ from reference: actual={actual_tokens:?}, expected={}",
        coupled["draft_head"]["tokens"]
    );
    let expected_confidence = coupled["draft_head"]["confidence"]
        .as_array()
        .context("missing real confidence reference")?;
    ensure!(expected_confidence.len() == cpu_hidden_confidence.len());
    for (&actual, expected) in cpu_hidden_confidence.iter().zip(expected_confidence) {
        ensure!(
            (f64::from(actual)
                - expected
                    .as_f64()
                    .context("invalid real confidence reference")?)
            .abs()
                <= 1e-4,
            "confidence differs on real independent input"
        );
    }
    let mut zero_reference = None;
    for count in [7, 1, 2, 3] {
        let zeros = Array::zeros_dtype(&[count, H], Dtype::BFloat16)?;
        let logits = Array::zeros_dtype(&[count, 248320], Dtype::BFloat16)?;
        let (tokens, confidence) = draft.select_logits(&logits, &zeros, 7)?;
        ensure!(tokens.len() == count as usize);
        if kind == Kind::Dflash2 {
            ensure!(
                tokens == vec![0; count as usize],
                "top16 ties must prefer lowest vocabulary IDs"
            );
        }
        if let Some((all_tokens, all_confidence)) = &zero_reference {
            let all_tokens: &Vec<u32> = all_tokens;
            let all_confidence: &Vec<f32> = all_confidence;
            ensure!(tokens == all_tokens[..count as usize]);
            if kind == Kind::Dspark {
                ensure!(confidence == all_confidence[..count as usize]);
            }
        } else {
            zero_reference = Some((tokens, confidence));
        }
    }
    if kind == Kind::Dflash2 {
        let invalid = Array::from_f32(&vec![f32::NEG_INFINITY; 248320], &[1, 248320])?
            .astype(Dtype::BFloat16)?;
        ensure!(
            draft
                .select_logits(&invalid, &Array::zeros_dtype(&[1, H], Dtype::BFloat16)?, 7)
                .is_err(),
            "invalid top16 rows must be rejected before use"
        );
    }
    ensure!(
        draft
            .select_logits(&Array::from_f32(&[1.], &[])?, &selector_hidden, 7)
            .is_err(),
        "scalar selector accepted"
    );
    ensure!(
        draft
            .select_logits(&logits, &selector_hidden, 248320)
            .is_err(),
        "out-of-vocabulary predecessor accepted"
    );
    println!(
        "FLASH_DRAFT_NUMPY {}",
        serde_json::json!({"status":"passed","kind":format!("{kind:?}"),"checks":checks,"selector_tokens":tokens,"confidence":confidence,
            "coupled":{"head":head_check,"cpu_hidden_tokens":cpu_hidden_tokens,"tokens":actual_tokens,"confidence":actual_confidence},"memory":array::memory_stats(false)?})
    );
    Ok(())
}

fn disk_state(target: &FlashNext, root: &Path, label: &str, save: bool) -> Result<usize> {
    let values = super::tests::state_bytes(target)?;
    ensure!(!values.is_empty(), "empty state comparison");
    for (index, value) in values.iter().enumerate() {
        let path = root.join(format!("{label}-{index}.bin"));
        if save {
            std::fs::write(path, value)?;
        } else {
            ensure!(
                *value == std::fs::read(path)?,
                "state differs at {label}/{index}"
            );
        }
    }
    Ok(values.len())
}

#[test]
#[ignore = "physical Apple GPU and independent Flash draft capture fixture; full prefix tape; run alone"]
fn flash_draft_captures_and_all_prefix_commits_match_serial() -> Result<()> {
    let checkpoint = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let root = std::env::var("MLXL3_FLASH_DRAFT_ORACLE")?;
    let root = Path::new(&root);
    let oracle: Value = serde_json::from_reader(File::open(root.join("oracle.json"))?)?;
    let kind = Kind::parse(oracle["kind"].as_str().context("missing kind")?)?;
    ensure!(
        oracle["status"] == "complete_cpu_oracle",
        "incomplete capture oracle"
    );
    let captures = oracle_array(root, &oracle, "captured")?;
    let tokens = oracle["tokens"]
        .as_array()
        .context("missing oracle tokens")?
        .iter()
        .map(|t| u32::try_from(t.as_u64().context("bad token")?).map_err(anyhow::Error::from))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        tokens.len() >= 5 && captures.shape() == [tokens.len() as i32, 12800],
        "vacuous capture fixture"
    );
    let mut target = FlashNext::from_checkpoint(&checkpoint)?;
    let mut logits = Vec::new();
    for &token in &tokens {
        logits.push(target.forward(token)?.to_bytes()?);
    }
    target.reset();
    let mut actual = Vec::new();
    for (i, &token) in tokens.iter().enumerate() {
        let (value, capture) = target.forward_captured(token, kind)?;
        ensure!(
            value.to_bytes()? == logits[i],
            "captured forward changed target logits"
        );
        actual.push(capture);
    }
    let actual = Array::concatenate(&actual.iter().collect::<Vec<_>>(), 0)?;
    let capture_check = compare_oracle(root, &oracle, "captured", &actual, 0.004)?;
    let serial = [EOS, 78768, 11, 7, 5, 19, 23, 13, 17, 31, 42];
    let state = tempfile::tempdir()?;
    target.reset();
    let mut expected_logits = Vec::new();
    let mut expected_captures = Vec::new();
    for (i, &token) in serial.iter().enumerate() {
        let (value, captured) = target.forward_captured(token, kind)?;
        let values = value.astype(Dtype::Float32)?.to_f32()?;
        ensure!(values.len() == 248320 && values.iter().all(|v| v.is_finite()));
        expected_logits.push(value.to_bytes()?);
        expected_captures.push(captured.to_bytes()?);
        disk_state(&target, state.path(), &format!("s{i}"), true)?;
    }
    let mut cases = 0;
    for width in [1, 2, 4, 8] {
        for retained in 1..=width {
            target.reset();
            target.forward(serial[0])?;
            target.forward(serial[1])?;
            let verification = target.verify_draft(&serial[2..2 + width], kind)?;
            let expected = expected_logits[2..2 + width].concat();
            ensure!(
                verification.parts.logits.shape() == [width as i32, 248320]
                    && verification.parts.logits.to_bytes()? == expected,
                "wide target logits differ"
            );
            ensure!(
                verification.parts.captured.shape() == [width as i32, 12800]
                    && verification.parts.captured.to_bytes()?
                        == expected_captures[2..2 + width].concat(),
                "wide target captures differ"
            );
            verification.commit(retained)?;
            disk_state(&target, state.path(), &format!("s{}", 1 + retained), false)?;
            let next = target.forward(serial[2 + retained])?;
            ensure!(
                next.to_bytes()? == expected_logits[2 + retained],
                "forward after prefix differs"
            );
            disk_state(&target, state.path(), &format!("s{}", 2 + retained), false)?;
            cases += 1;
        }
    }
    target.reset();
    target.forward(EOS)?;
    let before = super::tests::state_bytes(&target)?;
    for invalid in [vec![], vec![248320], vec![7; 9]] {
        ensure!(target.verify_draft(&invalid, kind).is_err());
    }
    ensure!(
        super::tests::state_bytes(&target)? == before,
        "invalid verification changed state"
    );
    drop(target.verify_draft(&[7, 11, 13, 17], kind)?);
    ensure!(
        target.offset() == 0 && target.last_hidden.is_none(),
        "abandoned verification did not reset"
    );
    for retained in [0, 5] {
        target.forward(EOS)?;
        ensure!(
            target
                .verify_draft(&[7, 11, 13, 17], kind)?
                .commit(retained)
                .is_err()
        );
        ensure!(
            target.offset() == 0 && target.last_hidden.is_none(),
            "invalid commit did not reset"
        );
    }
    ensure!(cases == 15);
    println!(
        "FLASH_DRAFT_PREFIX {}",
        serde_json::json!({"status":"passed","kind":format!("{kind:?}"),"prefix_cases":cases,"captures":capture_check,"widths":[1,2,4,8],"target_logits_and_states":"byte_exact","memory":array::memory_stats(false)?})
    );
    Ok(())
}

fn thinking_mode() -> Result<bool> {
    match std::env::var("MLXL3_FLASH_THINKING").as_deref() {
        Ok("on") | Err(std::env::VarError::NotPresent) => Ok(true),
        Ok("off") => Ok(false),
        _ => anyhow::bail!("MLXL3_FLASH_THINKING must be on or off"),
    }
}

fn prompt(target_path: &Path) -> Result<Vec<u32>> {
    prompt_with_thinking(target_path, thinking_mode()?)
}

fn prompt_with_thinking(target_path: &Path, thinking: bool) -> Result<Vec<u32>> {
    let tokenizer = crate::tokenizer::ChatTokenizer::load(target_path)?;
    let text=std::env::var("MLXL3_FLASH_PROMPT").unwrap_or_else(|_|"Explique en fran\u{e7}ais, avec un exemple concret, comment un cache LFU \u{e9}vite les lectures SSD inutiles pendant l\u{2019}inf\u{e9}rence d\u{2019}un mod\u{e8}le MoE.".into());
    tokenizer.encode(&tokenizer.render_values_with_thinking(
        &serde_json::json!([{"role":"user","content":text}]),
        None,
        thinking,
    )?)
}

#[test]
#[ignore = "installed pinned Flash tokenizer; no GPU; run with model fixture"]
fn flash_real_prompt_modes_match_template_and_preserve_default() -> Result<()> {
    let root = std::env::var("MLXL3_FLASH_MODEL")?;
    let tokenizer = crate::tokenizer::ChatTokenizer::load(Path::new(&root))?;
    let messages = serde_json::json!([{"role":"user","content":"hello"}]);
    let default = tokenizer.render_values(&messages, None)?;
    let on = tokenizer.render_values_with_thinking(&messages, None, true)?;
    let off = tokenizer.render_values_with_thinking(&messages, None, false)?;
    ensure!(default == on && on.ends_with("<think>\n"));
    ensure!(
        off.ends_with("<think>\n\n</think>\n\n")
            && !off.contains("Reasoning effort is set to xhigh")
    );
    ensure!(tokenizer.encode(&on)? != tokenizer.encode(&off)?);
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU, target and pinned draft; genuine greedy generation; run alone"]
fn flash_draft_greedy_generation_matches_tokens_and_states() -> Result<()> {
    let target_path = std::env::var("MLXL3_FLASH_MODEL")?;
    let target_path = Path::new(&target_path);
    let checkpoint = crate::checkpoint::inspect(target_path)?;
    let kind = Kind::parse(&std::env::var("MLXL3_FLASH_DRAFT_KIND")?)?;
    let path = std::env::var("MLXL3_FLASH_DRAFT_MODEL")?;
    let width = std::env::var("MLXL3_FLASH_DRAFT_WIDTH")
        .unwrap_or_else(|_| "3".into())
        .parse::<usize>()?;
    let mut target = FlashNext::from_checkpoint(&checkpoint)?;
    target.context = 4096;
    let mut draft = Draft::load(Path::new(&path), kind)?;
    configure_policy(&mut draft, target_path)?;
    let tokens = prompt(target_path)?;
    let directory = tempfile::tempdir()?;
    let mut results = Vec::new();
    for budget in [1, 2, 3, 17, 64] {
        target.reset();
        draft.reset();
        let mut expected = vec![target.forward_tokens(&tokens)?.chat_greedy_ids()?[0]];
        while expected.len() < budget {
            expected.push(
                target
                    .forward(*expected.last().context("anchor")?)?
                    .chat_greedy_ids()?[0],
            );
        }
        disk_state(&target, directory.path(), "final", true)?;
        target.reset();
        ensure!(generate(&mut target, &mut draft, &tokens, 0, width).is_err());
        ensure!(generate(&mut target, &mut draft, &tokens, budget, 0).is_err());
        ensure!(target.offset() == 0 && draft.offset == 0);
        let actual = generate(&mut target, &mut draft, &tokens, budget, width)?;
        ensure!(
            budget != 1 || actual.seconds == 0.,
            "empty decode must report zero seconds"
        );
        ensure!(
            actual.tokens == expected,
            "draft generation differs, budget={budget}"
        );
        disk_state(&target, directory.path(), "final", false)?;
        results.push(serde_json::json!({"budget":budget,"accepted":actual.accepted,"proposed":actual.proposed,"blocks":actual.blocks,"target_offset":target.offset(),"decode_seconds":actual.seconds,"draft_seconds":actual.draft_seconds,"verify_seconds":actual.verification_seconds}));
    }
    println!(
        "FLASH_DRAFT_GREEDY {}",
        serde_json::json!({"status":"passed_quality_not_speed","kind":format!("{kind:?}"),"width":width,"results":results,"memory":array::memory_stats(false)?})
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; prototype throughput only after numerical qualification; run alone"]
fn flash_draft_speed_arm() -> Result<()> {
    let target_path = std::env::var("MLXL3_FLASH_MODEL")?;
    let target_path = Path::new(&target_path);
    let checkpoint = crate::checkpoint::inspect(target_path)?;
    let mode = std::env::var("MLXL3_FLASH_DRAFT_MODE")?;
    ensure!(
        matches!(mode.as_str(), "off" | "mtp" | "dflash2" | "dspark"),
        "unknown speed mode"
    );
    let width = std::env::var("MLXL3_FLASH_DRAFT_WIDTH")
        .unwrap_or_else(|_| "3".into())
        .parse::<usize>()?;
    ensure!((1..=7).contains(&width));
    let mut target = FlashNext::from_checkpoint(&checkpoint)?;
    target.context = 4096;
    let mut head = if mode == "mtp" {
        Some(FlashMtpHead::load(&checkpoint, &target)?)
    } else {
        None
    };
    let mut draft = if matches!(mode.as_str(), "dflash2" | "dspark") {
        Some(Draft::load(
            Path::new(&std::env::var("MLXL3_FLASH_DRAFT_MODEL")?),
            Kind::parse(&mode)?,
        )?)
    } else {
        None
    };
    if let Some(draft) = &mut draft {
        configure_policy(draft, target_path)?;
    }
    let tokens = prompt(target_path)?;
    let mut records = Vec::new();
    for repetition in 0..3 {
        target.reset();
        let io_before = target.io_stats();
        let (generated, seconds, proposed, accepted, blocks, profile) = if let Some(draft) =
            &mut draft
        {
            draft.reset();
            let result = generate(&mut target, draft, &tokens, 128, width)?;
            (
                result.tokens,
                result.seconds,
                result.proposed,
                result.accepted,
                result.blocks,
                serde_json::json!({"draft_seconds":result.draft_seconds,"verify_and_cache_seconds":result.verification_seconds,"prefix_histogram":result.widths,"decode_io_hits_misses_bytes":result.decode_io}),
            )
        } else {
            if let Some(head) = &mut head {
                head.reset();
            }
            let mut logits = None;
            for &token in &tokens {
                if let Some(head) = &mut head
                    && let Some(hidden) = &target.last_hidden
                {
                    head.extend_cache(&target, hidden, token)?;
                }
                logits = Some(target.forward(token)?);
            }
            let mut generated = vec![logits.context("empty prompt")?.chat_greedy_ids()?[0]];
            let mut session = FlashMtpSession::default();
            let started = std::time::Instant::now();
            while generated.len() < 128 {
                let anchor = *generated.last().context("missing anchor")?;
                let next = if let Some(head) = &mut head {
                    session.advance(&mut target, head, anchor, 4096, 128 - generated.len())?
                } else {
                    target.forward(anchor)?.chat_greedy_ids()?[0]
                };
                generated.push(next);
            }
            (
                generated,
                started.elapsed().as_secs_f64(),
                session.proposed,
                session.accepted,
                session.blocks,
                serde_json::json!({"mtp_profile_seconds":session.profile_seconds}),
            )
        };
        ensure!(
            generated.len() == 128
                && seconds.is_finite()
                && seconds > 0.
                && generated.iter().all(|&t| t < 248320)
        );
        let memory = array::memory_stats(false)?;
        let io_after = target.io_stats();
        let io = [
            io_after
                .0
                .checked_sub(io_before.0)
                .context("hit counter moved backwards")?,
            io_after
                .1
                .checked_sub(io_before.1)
                .context("miss counter moved backwards")?,
            io_after
                .2
                .checked_sub(io_before.2)
                .context("I/O counter moved backwards")?,
        ];
        ensure!(
            memory
                .process_lifetime_peak_bytes
                .is_some_and(|n| n > 0 && n <= 10 << 30),
            "prototype memory limit exceeded"
        );
        let record = serde_json::json!({"repetition":repetition,"warmup":repetition==0,"mode":mode,
            "width":width,"generated":generated,"decode_tokens":127,"decode_seconds":seconds,"decode_tps":127./seconds,
            "proposed":proposed,"accepted":accepted,"blocks":blocks,"profile":profile,"memory":memory,
            "target_io_including_prefill_hits_misses_bytes":io,"prompt_tokens":tokens.len(),"context_limit":4096});
        println!("FLASH_DRAFT_SPEED_SAMPLE {record}");
        records.push(record);
    }
    println!(
        "FLASH_DRAFT_SPEED {}",
        serde_json::json!({"status":"completed_prototype_arm","mode":mode,"samples":records,
        "scope":"optimized Rust unit binary with test instrumentation; not a delivered bridge speed gain"})
    );
    Ok(())
}

#[test]
fn prefix_profiles_reject_foreign_incomplete_and_invalid_calibration() -> Result<()> {
    let identity = serde_json::json!({"fixture":1});
    let good = serde_json::json!({"status":"calibrated_greedy_profile","version":1,"query_width":3,"identity":identity,
        "temperatures":[1.,1.,1.],"round_seconds":[1.,1.,1.,1.],"cost_observations":[16,16,16,16],
        "calibration_blocks":32,"calibration_prompt_ids":["0000000000000000","1111111111111111"]});
    let plan = PrefixPlan::parse(&good, &identity)?;
    assert_eq!(PrefixPolicy::Adaptive(plan).choose(&[0.75; 3], 3)?, 3);
    assert_eq!(PrefixPolicy::Forced(0).choose(&[], 3)?, 0);
    assert!(PrefixPolicy::Forced(4).choose(&[], 3).is_err());
    for (key, value) in [
        ("status", serde_json::json!("incomplete")),
        ("identity", serde_json::json!({"fixture":2})),
        ("query_width", serde_json::json!(7)),
        ("temperatures", serde_json::json!([0., 1., 1.])),
        ("round_seconds", serde_json::json!([1., 0., 1., 1.])),
        ("calibration_blocks", serde_json::json!(31)),
        ("cost_observations", serde_json::json!([16, 16, 0, 16])),
        (
            "calibration_prompt_ids",
            serde_json::json!(["0000000000000000", "0000000000000000"]),
        ),
    ] {
        let mut bad = good.clone();
        bad[key] = value;
        assert!(
            PrefixPlan::parse(&bad, &identity).is_err(),
            "accepted invalid {key}"
        );
    }
    let plan = PrefixPlan::parse(&good, &identity)?;
    let policy = PrefixPolicy::Adaptive(plan);
    assert!(policy.choose(&[f32::NAN], 1).is_err());
    assert!(policy.choose(&[0.5], 2).is_err());
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU and pinned DSpark; all prefix policies against serial target; run alone"]
fn flash_prefix_policies_preserve_real_tokens_and_states() -> Result<()> {
    let root = std::env::var("MLXL3_FLASH_MODEL")?;
    let checkpoint = crate::checkpoint::inspect(Path::new(&root))?;
    let mut target = FlashNext::from_checkpoint(&checkpoint)?;
    target.context = 4096;
    let mut draft = Draft::load(
        Path::new(&std::env::var("MLXL3_FLASH_DRAFT_MODEL")?),
        Kind::Dspark,
    )?;
    let tokens = prompt(Path::new(&root))?;
    let mut expected = vec![target.forward_tokens(&tokens)?.chat_greedy_ids()?[0]];
    while expected.len() < 17 {
        expected.push(
            target
                .forward(*expected.last().context("empty reference")?)?
                .chat_greedy_ids()?[0],
        );
    }
    let directory = tempfile::tempdir()?;
    disk_state(&target, directory.path(), "prefix-policy", true)?;
    for case in 0..6 {
        target.reset();
        draft.reset();
        draft.policy = if case < 4 {
            PrefixPolicy::Forced(case)
        } else {
            let costs = if case == 4 {
                [0.001, 100., 100., 100.]
            } else {
                [100., 100., 100., 0.001]
            };
            PrefixPolicy::Adaptive(PrefixPlan {
                temperatures: [1.; 3],
                costs,
                calibration_prompt_ids: vec!["calibration-a".into(), "calibration-b".into()],
            })
        };
        let generated = generate(&mut target, &mut draft, &tokens, 17, 3)?;
        ensure!(
            generated.tokens == expected,
            "prefix policy changed greedy output"
        );
        disk_state(&target, directory.path(), "prefix-policy", false)?;
        ensure!(generated.widths.iter().sum::<usize>() == generated.blocks);
        if case == 0 || case == 4 {
            ensure!(generated.widths[0] == generated.blocks);
        }
    }
    println!(
        "FLASH_DRAFT_POLICIES {}",
        serde_json::json!({"status":"passed","cases":6,"tokens_per_case":17,"states":"byte_exact","memory":array::memory_stats(false)?})
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; profile DSpark B1..B4 using separate calibration prompts; run alone"]
fn flash_draft_calibration_arm() -> Result<()> {
    let root = std::env::var("MLXL3_FLASH_MODEL")?;
    let root = Path::new(&root);
    let draft_path = std::env::var("MLXL3_FLASH_DRAFT_MODEL")?;
    let prefix = std::env::var("MLXL3_FLASH_FORCE_PREFIX")?.parse::<usize>()?;
    ensure!(prefix <= 3 && std::env::var_os("MLXL3_FLASH_PREFIX_PROFILE").is_none());
    let checkpoint = crate::checkpoint::inspect(root)?;
    let mut target = FlashNext::from_checkpoint(&checkpoint)?;
    target.context = 4096;
    let mut draft = Draft::load(Path::new(&draft_path), Kind::Dspark)?;
    draft.policy = PrefixPolicy::Forced(prefix);
    let tokenizer = crate::tokenizer::ChatTokenizer::load(root)?;
    let directory = tempfile::tempdir()?;
    let mut runs = Vec::new();
    for (name, text) in [
        (
            "CAL_ARITH",
            "Calcule 23 fois 17 et explique ta méthode en français.",
        ),
        (
            "CAL_CODE",
            "Écris une fonction Python qui compte les occurrences de chaque mot, puis explique son fonctionnement.",
        ),
    ] {
        let messages = serde_json::json!([{"role":"user","content":text}]);
        let tokens = tokenizer.encode(&tokenizer.render_values_with_thinking(
            &messages,
            None,
            thinking_mode()?,
        )?)?;
        target.reset();
        let mut expected = vec![target.forward_tokens(&tokens)?.chat_greedy_ids()?[0]];
        while expected.len() < 64 {
            expected.push(
                target
                    .forward(*expected.last().context("empty calibration reference")?)?
                    .chat_greedy_ids()?[0],
            );
        }
        disk_state(&target, directory.path(), name, true)?;
        for repetition in 0..3 {
            target.reset();
            draft.reset();
            draft.record_rounds = repetition > 0;
            let budget = if repetition == 0 { 16 } else { 64 };
            let generated = generate(&mut target, &mut draft, &tokens, budget, 3)?;
            ensure!(
                generated.tokens == expected[..budget],
                "calibration generation diverged"
            );
            if repetition > 0 {
                disk_state(&target, directory.path(), name, false)?;
            }
            runs.push(serde_json::json!({"name":name,"prompt_id":prompt_id(&tokens),"warmup":repetition==0,"repetition":repetition,
                "tokens":generated.tokens,"trace":generated.trace,"states_checked":repetition>0}));
        }
    }
    println!(
        "FLASH_DRAFT_CALIBRATION {}",
        serde_json::json!({"status":"complete_calibration_arm","forced_prefix":prefix,
        "identity":profile_identity(root,Path::new(&draft_path))?,"runs":runs,"memory":array::memory_stats(false)?})
    );
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(100)]
fn actual_flash_draft_generation_checks_empty_decode_and_committed_state() {
    let checkpoint =
        crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL").unwrap()))
            .unwrap();
    let mut target = FlashNext::from_checkpoint(&checkpoint).unwrap();
    let mut draft = Draft::load(
        Path::new(&std::env::var("MLXL3_FLASH_DRAFT_MODEL").unwrap()),
        Kind::Dflash2,
    )
    .unwrap();
    let prompt = kani::any::<[u32; 2]>();
    let budget = if kani::any::<bool>() { 1 } else { 2 };
    if let Ok(result) = generate(&mut target, &mut draft, &prompt, budget, 3) {
        assert_eq!(result.tokens.len(), budget);
        assert!(result.seconds.is_finite());
        assert!(if budget == 1 {
            result.seconds == 0.
        } else {
            result.seconds > 0.
        });
        assert_eq!(target.offset(), (prompt.len() + budget - 1) as i32);
        kani::cover!(budget == 1);
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(100)]
fn actual_raw_draft_loader_checks_types_and_geometry() {
    let path = std::env::var("MLXL3_FLASH_DRAFT_MODEL").unwrap();
    let kind = if kani::any::<bool>() {
        Kind::Dflash2
    } else {
        Kind::Dspark
    };
    if let Ok(draft) = Draft::load(Path::new(&path), kind) {
        assert_eq!(
            draft.weights.len(),
            if kind == Kind::Dflash2 { 81 } else { 62 }
        );
        assert!(draft.weights.values().all(|w| w.dtype() == Dtype::BFloat16));
        assert_eq!(draft.offset, 0);
        kani::cover!(kind == Kind::Dflash2);
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(100)]
fn actual_flash_draft_norm_and_selector_preserve_shapes() {
    let values = kani::any::<[f32; 4]>();
    let data = (0..H as usize).map(|i| values[i % 4]).collect::<Vec<_>>();
    let x = Array::from_f32(&data, &[1, H])
        .unwrap()
        .astype(Dtype::BFloat16)
        .unwrap();
    let weight = Array::from_f32(&vec![1.; H as usize], &[H])
        .unwrap()
        .astype(Dtype::BFloat16)
        .unwrap();
    if let Ok((normalized, retained)) = residual_norm(&x, &x, &weight) {
        assert_eq!(normalized.shape(), [1, H]);
        assert_eq!(retained.shape(), [1, H]);
        assert_eq!(normalized.dtype(), Dtype::BFloat16);
        kani::cover!(values[0] == 0.);
    }
    let draft = Draft::load(
        Path::new(&std::env::var("MLXL3_FLASH_DRAFT_MODEL").unwrap()),
        Kind::Dflash2,
    )
    .unwrap();
    let logits = Array::zeros_dtype(&[1, 248320], Dtype::BFloat16).unwrap();
    if let Ok((tokens, confidence)) = draft.select_logits(&logits, &x, 7) {
        assert_eq!(tokens.len(), 1);
        assert!(tokens[0] < 248320);
        assert!(confidence.is_empty());
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(100)]
fn actual_flash_draft_verification_checks_tokens_shapes_and_commit() {
    let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL").unwrap()))
        .unwrap();
    let mut target = FlashNext::from_checkpoint(&c).unwrap();
    let tokens = kani::any::<[u32; 4]>();
    let retained = kani::any::<usize>();
    kani::assume((1..=4).contains(&retained));
    let committed = match target.verify_draft(&tokens, Kind::Dflash2) {
        Ok(verification) => {
            assert!(tokens.iter().all(|&t| t < 248320));
            assert_eq!(verification.parts.logits.shape(), [4, 248320]);
            verification.commit(retained).unwrap();
            true
        }
        Err(_) => false,
    };
    if committed {
        assert_eq!(target.offset(), retained as i32);
        kani::cover!(retained == 1);
        kani::cover!(retained == 4);
    }
}
