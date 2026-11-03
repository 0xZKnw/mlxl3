use super::*;
use std::path::Path;

#[test]
#[ignore = "physical Flash GDN state and full residual-layer oracle"]
fn five_gdn_states_and_residual_layers() -> Result<()> {
    let model = std::env::var_os("MLXL3_GDN_MODEL").context("missing Flash checkpoint")?;
    let frozen =
        PathBuf::from(std::env::var_os("MLXL3_GDN_FROZEN").context("missing frozen GDN probe")?);
    let root = PathBuf::from(
        std::env::var_os("MLXL3_GDN_ORACLE_DIR").context("missing temporary output")?,
    );
    let c = crate::checkpoint::inspect(Path::new(&model))?;
    let spec: Value = serde_json::from_slice(&std::fs::read(frozen.join("reference.json"))?)?;
    ensure!(spec["status"] == "complete" && spec["mode"] == "eval");
    let steps = spec["attention"]["2"]
        .as_array()
        .context("missing GDN probe")?;
    ensure!(steps.len() == 5);
    let mut layer = Gdn::load(&c, "model.language_model.layers.2.linear_attn", 1e-6)?;
    for (i, step) in steps.iter().enumerate() {
        let bytes =
            std::fs::read(frozen.join(step["input"]["file"].as_str().context("missing input")?))?;
        ensure!(bytes.len() == H as usize * 4);
        let x = Array::from_f32(
            &bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&x| f32::from_le_bytes(x))
                .collect::<Vec<_>>(),
            &[1, H],
        )?;
        layer.forward(&x)?.eval()?;
        let state = layer.state.as_ref().context("missing recurrent state")?;
        let history = layer.history.as_ref().context("missing conv history")?;
        ensure!(state.shape() == [1, 48, 128, 128] && state.dtype() == Dtype::Float32);
        ensure!(history.shape() == [1, 3, 10240] && history.dtype() == Dtype::Float16);
        std::fs::write(root.join(format!("{i}-state.bin")), state.to_bytes()?)?;
        std::fs::write(root.join(format!("{i}-history.bin")), history.to_bytes()?)?;
    }
    for half in [false, true] {
        let mut block = Layer::load(
            &c,
            "model.language_model.layers.0",
            true,
            false,
            1e-6,
            1e7,
            10,
            crate::ssd_moe::ResolvedPolicy::resolve(true, false)?,
        )?;
        for step in 0..3 {
            let index = if step == 2 { 0 } else { step };
            let input: Vec<_> = (0..WIDTH)
                .map(|i| (((i * 17 + index * 83) % 257) - 128) as f32 / 237.)
                .collect();
            let x = Array::from_f32(&input, &[1, WIDTH])?;
            let x = if half { x.astype(Dtype::Float16)? } else { x };
            let y = block.forward(&x, EOS)?;
            ensure!(y.shape() == [1, WIDTH] && y.dtype() == Dtype::Float32);
            std::fs::write(root.join(format!("layer-{half}-{step}.bin")), y.to_bytes()?)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "physical PLE history and full Flash layer oracle"]
fn ple_history_and_complete_layer_on_twelve_steps() -> Result<()> {
    let model = std::env::var_os("MLXL3_PLE_MODEL").context("missing Flash model")?;
    let frozen =
        PathBuf::from(std::env::var_os("MLXL3_PLE_FROZEN").context("missing PLE capture")?);
    let root = PathBuf::from(std::env::var_os("MLXL3_PLE_ORACLE_DIR").context("missing output")?);
    let c = crate::checkpoint::inspect(Path::new(&model))?;
    let spec: Value = serde_json::from_reader(File::open(frozen.join("reference.json"))?)?;
    ensure!(spec["status"] == "complete" && spec["mode"] == "eval");
    let cases = spec["attention"]["1"]
        .as_array()
        .context("missing PLE probe")?;
    ensure!(cases.len() == 5);
    let order = [0, 1, 2, 3, 4, 0, 0, 4, 1, 2, 3, 4];
    for half in [false, true] {
        let mut layer = Layer::load(
            &c,
            "model.language_model.layers.1",
            true,
            true,
            1e-6,
            1e7,
            10,
            crate::ssd_moe::ResolvedPolicy::resolve(true, false)?,
        )?;
        for (i, &index) in order.iter().enumerate() {
            let case = &cases[index];
            let bytes = std::fs::read(
                frozen.join(
                    case["input"]["file"]
                        .as_str()
                        .context("missing PLE input")?,
                ),
            )?;
            ensure!(bytes.len() == 10240 * 4);
            let x = Array::from_f32(
                &bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&x| f32::from_le_bytes(x))
                    .collect::<Vec<_>>(),
                &[1, WIDTH],
            )?;
            let x = if half { x.astype(Dtype::Float16)? } else { x };
            let token = u32::try_from(case["token"].as_u64().context("missing token")?)?;
            let out = layer.forward(&x, token)?;
            ensure!(out.dtype() == Dtype::Float32 && out.shape() == [1, WIDTH]);
            let ple = layer.ple.as_ref().context("PLE not executed")?;
            let history = ple.history.as_ref().context("missing history")?;
            ensure!(history.dtype() == Dtype::Float32 && history.shape() == [1, 9, WIDTH]);
            std::fs::write(root.join(format!("layer-{half}-{i}.bin")), out.to_bytes()?)?;
            std::fs::write(
                root.join(format!("history-{half}-{i}.bin")),
                history.to_bytes()?,
            )?;
            let (_, gated) = ple
                .stages
                .iter()
                .find(|(name, _)| name == "gated")
                .context("missing PLE branch")?;
            let (_, conv) = ple
                .stages
                .iter()
                .find(|(name, _)| name == "convolution")
                .context("missing PLE convolution")?;
            let value = x.add(&gated.add(conv)?)?;
            std::fs::write(root.join(format!("ple-{half}-{i}.bin")), value.to_bytes()?)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "physical Flash dense-QSA single-query and complete-layer oracle"]
fn dense_qsa_across_simd_stripes_and_residual() -> Result<()> {
    let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_QSA_MODEL")?))?;
    let root = PathBuf::from(std::env::var("MLXL3_QSA_ORACLE_DIR")?);
    for half in [false, true] {
        let mut attn = Qsa::load(&c, "model.language_model.layers.15.self_attn", 1e-6, 1e7)?;
        let mut layer = Layer::load(
            &c,
            "model.language_model.layers.3",
            false,
            false,
            1e-6,
            1e7,
            10,
            crate::ssd_moe::ResolvedPolicy::resolve(true, false)?,
        )?;
        for (full, width, count) in [(false, H, 512), (true, WIDTH, 4)] {
            let mut file = File::create(root.join(format!("rows-{half}-{full}.bin")))?;
            for t in 0..count {
                let row = (0..width)
                    .map(|i| ((i * 17 + t * 83) % 257 - 128) as f32 / 237.)
                    .collect::<Vec<_>>();
                let x = Array::from_f32(&row, &[1, width])?;
                let x = if half { x.astype(Dtype::Float16)? } else { x };
                let y = if full {
                    layer.forward(&x, EOS)?
                } else {
                    attn.forward(&x)?
                };
                ensure!(
                    y.shape() == [1, width] && y.dtype() == x.dtype(),
                    "unexpected QSA output shape or dtype"
                );
                use std::io::Write;
                file.write_all(&y.to_bytes()?)?;
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "physical long Flash QSA independent MLX oracle, including exact sparse positions"]
fn qsa_index_boundaries_and_cache() -> Result<()> {
    let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_QSA_MODEL")?))?;
    let dir = PathBuf::from(std::env::var("MLXL3_QSA_ORACLE_DIR")?);
    let mut qsa = Qsa::load(&c, "model.language_model.layers.15.self_attn", 1e-6, 1e7)?;
    for t in 0..4096 {
        let n = t + 1;
        let sampled = crate::flash_support::qsa_probe_checkpoint(n as u32);
        qsa.sparse_trace = sampled.then(BTreeMap::new);
        let values = crate::flash_support::qsa_probe_row(t as u32);
        let input = Array::from_f32(&values, &[1, H])?;
        let out = qsa.forward(&input)?;
        out.eval()?;
        if sampled {
            let write = |name: &str, data: &Array| -> Result<()> {
                std::fs::write(dir.join(format!("{n}-{name}.bin")), data.to_bytes()?)?;
                Ok(())
            };
            write("out", &out)?;
            for (name, a) in qsa.sparse_trace.take().context("missing trace")? {
                write(name, &a)?;
            }
            if let Some(tail) = &qsa.state.index_tail {
                write("index_tail", tail)?;
            }
            if n % 4 == 0 {
                let blocks = qsa.state.blocks.as_ref().context("missing index blocks")?;
                write("index_block", &blocks.slice(2, n / 4 - 1, n / 4)?)?;
            }
            if n == 4096 {
                for (name, v) in [("keys", &qsa.state.keys), ("values", &qsa.state.values)] {
                    write(
                        name,
                        &v.as_ref().context("missing KV")?.transpose(&[0, 2, 1, 3])?,
                    )?;
                }
                write(
                    "blocks",
                    qsa.state.blocks.as_ref().context("missing blocks")?,
                )?;
            }
        }
    }
    Ok(())
}
