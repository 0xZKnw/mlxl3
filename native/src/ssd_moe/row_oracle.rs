use super::*;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Case {
    prefix: String,
    fp32: bool,
    slots: usize,
    rows: Vec<Vec<u32>>,
}

#[test]
#[ignore = "physical stock Flash checkpoint MoE differential oracle"]
fn flash_moe_row_stock_oracle() -> Result<()> {
    let root =
        PathBuf::from(std::env::var_os("MLXL3_MOE_FIXTURES").context("missing imposed MoE rows")?);
    let model =
        PathBuf::from(std::env::var_os("MLXL3_MOE_CHECKPOINT").context("missing Flash model")?);
    let checkpoint = crate::checkpoint::inspect(&model)?;
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(root.join("cases.json"))?)?;
    ensure!(!cases.is_empty());
    let mut answers = Vec::new();
    for case in cases {
        ensure!(!case.rows.is_empty());
        let mut layer = SsdMoe::load(&checkpoint, &case.prefix, case.slots, false)?;
        let mut rows = Vec::new();
        for words in case.rows {
            ensure!(words.len() == 2560);
            let x = if case.fp32 {
                Array::from_f32(
                    &words.into_iter().map(f32::from_bits).collect::<Vec<_>>(),
                    &[1, H],
                )?
            } else {
                Array::from_f16_bits(
                    &words
                        .into_iter()
                        .map(u16::try_from)
                        .collect::<Result<Vec<_>, _>>()?,
                    &[1, H],
                )?
            };
            let (indices, _) = layer.route(&x)?;
            let y = layer.forward(&x)?;
            ensure!(y.shape() == [1, H] && y.dtype() == x.dtype());
            let words: Vec<u32> = if case.fp32 {
                y.to_f32()?.into_iter().map(f32::to_bits).collect()
            } else {
                y.to_f16_bits()?.into_iter().map(u32::from).collect()
            };
            let scalar = layer.multiplier.forward(&x)?;
            let sig = scalar.sigmoid()?;
            let raw = |a: &Array| -> Result<u32> {
                Ok(if case.fp32 {
                    a.to_f32()?[0].to_bits()
                } else {
                    u32::from(a.to_f16_bits()?[0])
                })
            };
            let switch = [raw(&sig)?, raw(&scalar)?];
            rows.push(
                serde_json::json!({"indices":indices.to_u32()?,"values":words,"switch":switch}),
            );
        }
        answers.push(rows);
    }
    std::fs::write(root.join("expected.json"), serde_json::to_vec(&answers)?)?;
    Ok(())
}
