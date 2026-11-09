//! Qwen Gated DeltaNet over owned Metal buffers and the shared EXL3 projection.
use super::{FlashProjection, MatrixView, Metal, checkpoint_float_weight, view::FloatType};
use crate::{checkpoint::Checkpoint, tensor_layout::MatrixLayout};
use anyhow::{Context, Result, ensure};
use metal::{
    Buffer, CommandBufferRef, CompileOptions, ComputePipelineState, MTLResourceOptions as Storage,
    MTLSize, NSRange,
};
use std::rc::Rc;

pub(super) struct Pipelines {
    dense: ComputePipelineState,
    prepare: ComputePipelineState,
    step: ComputePipelineState,
    norm_gate: ComputePipelineState,
}

#[derive(Clone)]
pub struct QwenGdnState {
    owner: Rc<()>,
    history: MatrixView,
    recurrent: MatrixView,
}

#[derive(serde::Serialize)]
pub struct QwenGdnTrace {
    pub conv_state: Vec<u16>,
    pub state: Vec<f32>,
}

pub struct QwenGdn<'a> {
    gpu: &'a Metal,
    pipelines: Rc<Pipelines>,
    qkv: FlashProjection<'a>,
    z: FlashProjection<'a>,
    output: FlashProjection<'a>,
    a: Buffer,
    b: Buffer,
    conv: Buffer,
    a_log: Buffer,
    dt_bias: Buffer,
    norm: Buffer,
    hidden: u32,
    key_heads: u32,
    value_heads: u32,
    conv_length: u32,
    eps: f32,
    owner: Rc<()>,
    state: Option<QwenGdnState>,
}

impl<'a> QwenGdn<'a> {
    pub fn from_checkpoint(gpu: &'a Metal, c: &Checkpoint, prefix: &str) -> Result<Self> {
        let root: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(c.path.join("config.json"))?)?;
        let config = &root["text_config"];
        let number = |name| {
            config[name]
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .with_context(|| format!("missing or invalid Qwen {name}"))
        };
        ensure!(
            number("linear_key_head_dim")? == 128 && number("linear_value_head_dim")? == 128,
            "native Qwen GDN requires128-wide heads"
        );
        Self::load(
            gpu,
            c,
            prefix,
            number("hidden_size")?,
            number("linear_num_key_heads")?,
            number("linear_num_value_heads")?,
            number("linear_conv_kernel_dim")?,
            config["rms_norm_eps"]
                .as_f64()
                .context("missing Qwen norm epsilon")? as f32,
        )
    }

    pub fn hidden_size(&self) -> u32 {
        self.hidden
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load(
        gpu: &'a Metal,
        c: &Checkpoint,
        prefix: &str,
        hidden: u32,
        key_heads: u32,
        value_heads: u32,
        conv_length: u32,
        eps: f32,
    ) -> Result<Self> {
        ensure!(
            matches!(c.model_type.as_str(), "qwen3_5" | "qwen3_5_moe"),
            "requires a Qwen3.5 checkpoint"
        );
        ensure!(eps.is_finite() && eps > 0., "invalid Qwen GDN epsilon");
        let [_, values, channels, _, _] =
            crate::contracts::native_qwen_gdn_extents(hidden, key_heads, value_heads, conv_length)
                .context("invalid native Qwen GDN geometry")?;
        let projection = |name: &str, dims| -> Result<FlashProjection<'a>> {
            let p = FlashProjection::load_exl3(gpu, c, &format!("{prefix}.{name}"))?;
            ensure!(p.dimensions() == dims, "invalid Qwen projection {name}");
            Ok(p)
        };
        let weight = |name: &str, dims: &[usize], dtype| {
            checkpoint_float_weight(gpu, c, &format!("{prefix}.{name}"), dims, dtype)
        };
        Ok(Self {
            gpu,
            pipelines: Pipelines::load(gpu, hidden, key_heads, value_heads, conv_length)?,
            qkv: projection("in_proj_qkv", [hidden, channels])?,
            z: projection("in_proj_z", [hidden, values])?,
            output: projection("out_proj", [values, hidden])?,
            a: weight(
                "in_proj_a.weight",
                &[value_heads as usize, hidden as usize],
                FloatType::F16,
            )?,
            b: weight(
                "in_proj_b.weight",
                &[value_heads as usize, hidden as usize],
                FloatType::F16,
            )?,
            conv: weight(
                "conv1d.weight",
                &[channels as usize, 1, conv_length as usize],
                FloatType::F16,
            )?,
            a_log: weight("A_log", &[value_heads as usize], FloatType::F32)?,
            dt_bias: weight("dt_bias", &[value_heads as usize], FloatType::F16)?,
            norm: weight("norm.weight", &[128], FloatType::F16)?,
            hidden,
            key_heads,
            value_heads,
            conv_length,
            eps,
            owner: Rc::new(()),
            state: None,
        })
    }
    pub fn reset(&mut self) {
        self.state = None;
    }
    pub fn snapshot(&self) -> Option<QwenGdnState> {
        self.state.clone()
    }
    pub fn restore(&mut self, state: Option<QwenGdnState>) -> Result<()> {
        ensure!(
            state
                .as_ref()
                .is_none_or(|s| Rc::ptr_eq(&self.owner, &s.owner)),
            "Qwen state belongs to another layer"
        );
        self.state = state;
        Ok(())
    }
    pub fn states(&self) -> Result<Option<QwenGdnTrace>> {
        self.state
            .as_ref()
            .map(|s| {
                let conv_state = s
                    .history
                    .export_bytes(self.gpu)?
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .copied()
                    .map(u16::from_le_bytes)
                    .collect();
                let state = s
                    .recurrent
                    .export_bytes(self.gpu)?
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .copied()
                    .map(f32::from_le_bytes)
                    .collect();
                Ok(QwenGdnTrace { conv_state, state })
            })
            .transpose()
    }
    pub fn forward(&mut self, x: &MatrixView) -> Result<MatrixView> {
        let (next, output) = self
            .gpu
            .command(|cmd| self.encode_step(cmd, x, self.state.as_ref()))?;
        self.state = Some(next);
        Ok(output)
    }

    pub(super) fn encode_step(
        &self,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        previous: Option<&QwenGdnState>,
    ) -> Result<(QwenGdnState, MatrixView)> {
        ensure!(
            previous.is_none_or(|s| Rc::ptr_eq(&self.owner, &s.owner)),
            "Qwen state belongs to another layer"
        );
        ensure!(
            x.shape() == [1, self.hidden]
                && x.dtype == FloatType::F16
                && x.layout
                    == (MatrixLayout {
                        shape: [1, self.hidden],
                        strides: [i64::from(self.hidden), 1],
                        offset: 0
                    })
                && x.buffer.device().registry_id() == self.gpu.device.registry_id(),
            "invalid native Qwen input"
        );
        let gpu = self.gpu;
        let (value_groups, _) = crate::contracts::packed_gdn_grid(self.value_heads, 128)
            .context("invalid Qwen state grid")?;
        let channels = (self.key_heads * 2 + self.value_heads) * 128;
        let values = self.value_heads * 128;
        let row =
            |kind, width| MatrixView::output(gpu, kind, 1, width, Storage::StorageModePrivate);
        let frame = || -> Result<QwenGdnState> {
            Ok(QwenGdnState {
                owner: self.owner.clone(),
                history: row(FloatType::F16, (self.conv_length - 1) * channels)?,
                recurrent: row(FloatType::F32, values * 128)?,
            })
        };
        let previous = if let Some(s) = previous {
            s.clone()
        } else {
            let s = frame()?;
            let blit = cmd.new_blit_command_encoder();
            for a in [&s.history, &s.recurrent] {
                blit.fill_buffer(&a.buffer, NSRange::new(0, a.buffer.length()), 0);
            }
            blit.end_encoding();
            s
        };
        let next = frame()?;
        let qkv = self.qkv.encode_row(cmd, x, Storage::StorageModePrivate)?;
        let z = self.z.encode_row(cmd, x, Storage::StorageModePrivate)?;
        let dense = |weight| -> Result<MatrixView> {
            let out = row(FloatType::F16, self.value_heads)?;
            Metal::encode(
                cmd,
                &self.pipelines.dense,
                &[&x.buffer, weight, &out.buffer],
                [self.value_heads],
                MTLSize::new(u64::from(self.value_heads / 4), 1, 1),
                256,
            );
            Ok(out)
        };
        let a = dense(&self.a)?;
        let b = dense(&self.b)?;
        let q = row(FloatType::F16, self.key_heads * 128)?;
        let k = row(FloatType::F16, self.key_heads * 128)?;
        let v = row(FloatType::F16, values)?;
        Metal::encode(
            cmd,
            &self.pipelines.prepare,
            &[
                &qkv.buffer,
                &previous.history.buffer,
                &self.conv,
                &q.buffer,
                &k.buffer,
                &v.buffer,
                &next.history.buffer,
            ],
            [0u32],
            MTLSize::new(1, u64::from(self.value_heads.max(self.key_heads)), 1),
            32,
        );
        let delta = row(FloatType::F16, values)?;
        Metal::encode(
            cmd,
            &self.pipelines.step,
            &[
                &q.buffer,
                &k.buffer,
                &v.buffer,
                &a.buffer,
                &b.buffer,
                &self.a_log,
                &self.dt_bias,
                &previous.recurrent.buffer,
                &delta.buffer,
                &next.recurrent.buffer,
            ],
            [128u32],
            MTLSize::new(1, u64::from(value_groups), u64::from(self.value_heads)),
            64,
        );
        let gated = row(FloatType::F16, values)?;
        Metal::encode(
            cmd,
            &self.pipelines.norm_gate,
            &[&delta.buffer, &z.buffer, &self.norm, &gated.buffer],
            [self.eps.to_bits()],
            MTLSize::new(u64::from(self.value_heads), 1, 1),
            32,
        );
        Ok((
            next,
            self.output
                .encode_row(cmd, &gated, Storage::StorageModeShared)?,
        ))
    }
}

#[cfg(test)]
#[test]
#[ignore = "physical Qwen GDN: retained branches, reset, foreign state and invalid input"]
fn retained_states_replay_after_branch_and_rejected_input() -> Result<()> {
    use std::path::Path;
    let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_QWEN_NATIVE_MODEL")?))?;
    let gpu = Metal::new()?;
    let prefix = "model.language_model.layers.0.linear_attn";
    let mut layer = QwenGdn::from_checkpoint(&gpu, &c, prefix)?;
    let input = |step: usize, rows: u32| -> Result<MatrixView> {
        let bits = (0..rows as usize * layer.hidden_size() as usize)
            .map(|i| {
                half::f16::from_f32(((i * 17 + step * 83) % 257) as f32 / 237. - 128. / 237.)
                    .to_bits()
            })
            .collect::<Vec<_>>();
        gpu.f16_storage(&bits)?.matrix(MatrixLayout {
            shape: [rows, layer.hidden_size()],
            strides: [i64::from(layer.hidden_size()), 1],
            offset: 0,
        })
    };
    let a = input(0, 1)?;
    let b = input(1, 1)?;
    let alternate = input(3, 1)?;
    let invalid = input(0, 2)?;
    for eps in [0., -1., f32::NAN, f32::INFINITY] {
        let error = QwenGdn::load(&gpu, &c, prefix, layer.hidden_size(), 16, 32, 4, eps)
            .err()
            .context("invalid epsilon accepted")?;
        ensure!(error.to_string().contains("epsilon"));
    }
    let error = QwenGdn::load(&gpu, &c, prefix, layer.hidden_size() + 1, 16, 32, 4, 1e-6)
        .err()
        .context("unaligned hidden accepted")?;
    ensure!(error.to_string().contains("geometry"));

    ensure!(layer.states()?.is_none());
    let initial = layer.forward(&a)?.read_output()?;
    let before = layer.snapshot();
    let next = layer.forward(&b)?.read_output()?;
    let next_state = layer.states()?.unwrap();
    layer.restore(before.clone())?;
    layer.forward(&alternate)?;
    layer.restore(before.clone())?;
    let previous = layer.states()?.unwrap();
    ensure!(layer.forward(&invalid).is_err());
    let after = layer.states()?.unwrap();
    ensure!(
        previous.conv_state == after.conv_state
            && previous
                .state
                .iter()
                .map(|v| v.to_bits())
                .eq(after.state.iter().map(|v| v.to_bits()))
    );
    let mut foreign = QwenGdn::from_checkpoint(&gpu, &c, prefix)?;
    foreign.forward(&a)?;
    ensure!(layer.restore(foreign.snapshot()).is_err());
    ensure!(layer.forward(&b)?.read_output()? == next);
    let replay = layer.states()?.unwrap();
    ensure!(
        next_state.conv_state == replay.conv_state
            && next_state
                .state
                .iter()
                .map(|v| v.to_bits())
                .eq(replay.state.iter().map(|v| v.to_bits()))
    );
    layer.reset();
    ensure!(layer.states()?.is_none());
    ensure!(layer.forward(&a)?.read_output()? == initial);
    Ok(())
}

impl Pipelines {
    fn load(
        gpu: &Metal,
        hidden: u32,
        key_heads: u32,
        value_heads: u32,
        conv_length: u32,
    ) -> Result<Rc<Self>> {
        let key = [hidden, key_heads, value_heads, conv_length];
        if let Some(pipelines) = gpu.qwen_gdn_pipelines.borrow().get(&key) {
            return Ok(Rc::clone(pipelines));
        }
        let [keys, _, channels, _, _] =
            crate::contracts::native_qwen_gdn_extents(hidden, key_heads, value_heads, conv_length)
                .context("invalid native Qwen GDN geometry")?;
        let source = format!(
            "#define HIDDEN {hidden}\n#define KEY_HEADS {key_heads}\n#define VALUE_HEADS {value_heads}\n#define KEY_DIM 128\n#define VALUE_DIM 128\n#define KEYS {keys}\n#define CONV_DIMS {channels}\n#define CONV_LENGTH {conv_length}\n#define CONV_HISTORY {}\n",
            conv_length - 1
        ) + &include_str!("../../shaders/qwen_gdn.metal")
            .replace(
                "#include \"flash_hc_gemv.metal\"",
                include_str!("../../shaders/flash_hc_gemv.metal"),
            )
            .replace(
                "#include \"gated_delta_prepare.metal\"",
                include_str!("../../shaders/gated_delta_prepare.metal"),
            )
            .replace(
                "#include \"gated_delta_packed.metal\"",
                include_str!("../../shaders/gated_delta_packed.metal"),
            );
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = gpu
            .device
            .new_library_with_source(&source, &options)
            .map_err(anyhow::Error::msg)?;
        let pipeline = |name| -> Result<ComputePipelineState> {
            let function = library
                .get_function(name, None)
                .map_err(anyhow::Error::msg)?;
            let p = gpu
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(anyhow::Error::msg)?;
            ensure!(
                p.thread_execution_width() == 32 && p.max_total_threads_per_threadgroup() >= 256,
                "unsupported native Qwen SIMD width"
            );
            Ok(p)
        };
        let pipelines = Rc::new(Self {
            dense: pipeline("qwen_dense")?,
            prepare: pipeline("qwen_prepare")?,
            step: pipeline("qwen_step")?,
            norm_gate: pipeline("qwen_norm_gate")?,
        });
        // Publish only after all compilations succeed. Owner fixes device,
        // embedded source and fast-math=false; geometry is the only variable.
        gpu.qwen_gdn_pipelines
            .borrow_mut()
            .insert(key, Rc::clone(&pipelines));
        Ok(pipelines)
    }
}

#[cfg(test)]
#[test]
#[ignore = "physical Metal: Qwen GDN pipeline reuse, geometry and device separation"]
fn shared_gdn_pipelines_preserve_geometry_and_owner() -> Result<()> {
    let gpu = Metal::new()?;
    let first = Pipelines::load(&gpu, 2048, 16, 32, 4)?;
    for _ in 0..30 {
        assert!(Rc::ptr_eq(&first, &Pipelines::load(&gpu, 2048, 16, 32, 4)?));
    }
    assert_eq!(gpu.qwen_gdn_pipelines.borrow().len(), 1);
    let other = Pipelines::load(&gpu, 5120, 16, 48, 4)?;
    assert!(!Rc::ptr_eq(&first, &other));
    assert_eq!(gpu.qwen_gdn_pipelines.borrow().len(), 2);
    assert!(Pipelines::load(&gpu, 0, 16, 32, 4).is_err());
    assert_eq!(gpu.qwen_gdn_pipelines.borrow().len(), 2);
    let another = Metal::new()?;
    assert!(!Rc::ptr_eq(
        &first,
        &Pipelines::load(&another, 2048, 16, 32, 4)?
    ));
    Ok(())
}
