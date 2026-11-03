use super::*;
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Case {
    prefix: String,
    mlp: bool,
    fp32: bool,
    words: Vec<u32>,
}

#[test]
#[ignore = "physical Flash HC and MLP differential oracle"]
fn flash_hc_mlp_checkpoint_rows() -> Result<()> {
    let root = PathBuf::from(std::env::var_os("MLXL3_HC_FIXTURES").context("missing HC fixtures")?);
    let model = std::env::var_os("MLXL3_HC_MODEL").context("missing checkpoint")?;
    let c = crate::checkpoint::inspect(Path::new(&model))?;
    let config: Value = serde_json::from_reader(File::open(c.path.join("config.json"))?)?;
    let eps = config["text_config"]["rms_norm_eps"]
        .as_f64()
        .context("missing epsilon")? as f32;
    let cases: Vec<Case> = serde_json::from_reader(File::open(root.join("cases.json"))?)?;
    ensure!(!cases.is_empty());
    for (index, case) in cases.into_iter().enumerate() {
        ensure!(case.words.len() == WIDTH as usize);
        let x = if case.fp32 {
            Array::from_f32(
                &case
                    .words
                    .into_iter()
                    .map(f32::from_bits)
                    .collect::<Vec<_>>(),
                &[1, WIDTH],
            )?
        } else {
            Array::from_f16_bits(
                &case
                    .words
                    .into_iter()
                    .map(u16::try_from)
                    .collect::<Result<Vec<_>, _>>()?,
                &[1, WIDTH],
            )?
        };
        let words = |x: &Array| -> Result<Vec<u32>> {
            ensure!(
                x.dtype()
                    == if case.fp32 {
                        Dtype::Float32
                    } else {
                        Dtype::Float16
                    }
            );
            Ok(if case.fp32 {
                x.to_f32()?.into_iter().map(f32::to_bits).collect()
            } else {
                x.to_f16_bits()?.into_iter().map(u32::from).collect()
            })
        };
        let p = if case.mlp {
            format!("{}.mlp_hyper_connection", case.prefix)
        } else {
            case.prefix.clone()
        };
        let hc = Hc::load(&c, &p, eps, !p.ends_with(".hyper_connection_mixer"))?;
        let norm = hc.norm.forward(&x.reshape(&[1, 1, WIDTH])?)?;
        let down = hc.down.forward_flash_hc(&norm)?;
        let active = down.qwen4_hc_activate()?;
        let up = hc.up.forward_flash_hc(&active)?;
        let (mixed, gate) = hc.read(&x)?;
        let mut output = None;
        let mut indices = None;
        if case.mlp {
            let mut moe = SsdMoe::load(&c, &format!("{}.mlp", case.prefix), 10, false)?;
            indices = Some(moe.route(&mixed)?.0.to_u32()?);
            output = Some(words(&Hc::write(
                &x,
                &moe.forward(&mixed)?,
                gate.as_ref().context("missing gate")?,
            )?)?);
        }
        let record = serde_json::json!({"mixed":words(&mixed)?,"gate":gate.as_ref().map(words).transpose()?,"output":output,"indices":indices,
            "trace":{"normalized":words(&norm)?,"down":words(&down)?,"active":words(&active)?,"up":words(&up)?}});
        serde_json::to_writer(
            std::io::BufWriter::new(File::create(root.join(format!("row-{index}.json")))?),
            &record,
        )?;
    }
    Ok(())
}
