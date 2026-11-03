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
