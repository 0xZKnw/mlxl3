//! Q27D-2026-11-17-03: multi-row QMV kernel lab on real Qwen27B projections.
//! Test-only; production shaders and paths are untouched.
use super::*;
use std::time::Instant;

const MAPPED: &str = include_str!("../../shaders/_qmv_mapped_tile_kernel.metal");
const TILE: &str = include_str!("../../shaders/_qmv_tile_kernel.metal");

#[derive(Clone, Copy)]
struct Variant {
    mb: i32,
    nt: i32,
    lane: bool,
    /// Text edit applied to the shader source; "" is the stock source.
    edit: &'static str,
}

fn rows_input(m: i32, width: i32, seed: u32) -> Result<Array> {
    let bits = (0..(m * width) as u32)
        .map(|i| {
            let mixed = i.wrapping_mul(2_654_435_761).wrapping_add(seed) >> 8;
            half::f16::from_f32((mixed % 2001) as f32 / 1000.0 - 1.0).to_bits()
        })
        .collect::<Vec<_>>();
    let value = Array::from_f16_bits(&bits, &[m, width])?;
    value.eval()?;
    Ok(value)
}

fn lane_indices(width: i32) -> Result<Array> {
    let values = (0..width)
        .map(|index| crate::contracts::exl3_lane_source_index(index as u32) as i32)
        .collect::<Vec<_>>();
    Array::from_i32(&values, &[width])
}

/// Ablations replace one cost centre with a cheap stand-in; their outputs are wrong by design.
fn edited(source: &str, edit: &str) -> Result<String> {
    let replace = |text: String, from: &str, to: &str| -> Result<String> {
        ensure!(text.contains(from), "ablation anchor missing: {from}");
        Ok(text.replace(from, to))
    };
    let mut text = source.to_owned();
    for part in edit.split('+').filter(|part| !part.is_empty()) {
        text = match part {
            "nodecode" => {
                let mut value = text;
                for index in 0..4 {
                    value = replace(
                        value,
                        &format!("float(mlxl3_decode_codeword(cw{index}, CB))"),
                        &format!("float(cw{index} & 0xffu)"),
                    )?;
                }
                value
            }
            "nox" => {
                let value = replace(
                    text,
                    "x_quad = x_quads[lane & 3u];",
                    "x_quad = half4(1.0h);",
                )?;
                replace(
                    value,
                    "x_pair = x_pairs[lane & 7u];",
                    "x_pair = half2(1.0h);",
                )?
            }
            "noload" => {
                let from = if text.contains("ulong merged = (ulong(words[word0[group]]) << 32) |\n                                       ulong(words[word1[group]]);") {
                    "ulong merged = (ulong(words[word0[group]]) << 32) |\n                                       ulong(words[word1[group]]);"
                } else {
                    "ulong merged = (ulong(words[word0[group]]) << 32) |\n                                               ulong(words[word1[group]]);"
                };
                replace(
                    text,
                    from,
                    "ulong merged = ulong((tile_k + group) * 2654435761u + lane * 40503u) << 13;",
                )?
            }
            // Candidate layout (not an ablation): xhat stored as F32, no half->float per load.
            "xf32" => {
                let mut value = text;
                for (from, to) in [
                    ("half4 x_quad(0);", "float4 x_quad(0);"),
                    (
                        "half4 x_quad = half4(0.0h);",
                        "float4 x_quad = float4(0.0f);",
                    ),
                    (
                        "const device half4* x_quads",
                        "const device float4* x_quads",
                    ),
                    (
                        "reinterpret_cast<const device half4*>(",
                        "reinterpret_cast<const device float4*>(",
                    ),
                ] {
                    if value.contains(from) {
                        value = value.replace(from, to);
                    }
                }
                ensure!(
                    value.contains("const device float4* x_quads"),
                    "xf32 anchors missing"
                );
                value
            }
            "mul1fast" => text,
            // Non-factorizable load ablation: the stand-in depends on the output tile too.
            "noload2" => {
                let mut value = text;
                let stand_in = "ulong merged = ulong((tile_k * 2654435761u) ^ (lane * 40503u) ^ (group * 977u) ^ (output_tile * 7919u)) << 13;";
                let mut hits = 0;
                for from in [
                    "ulong merged = (ulong(words[word0[group]]) << 32) |\n                                       ulong(words[word1[group]]);",
                    "ulong merged = (ulong(words[word0[group]]) << 32) |\n                                               ulong(words[word1[group]]);",
                    "ulong merged = (ulong(words[word0[group]]) << 32) |\n                                       ulong(words[word1[group]]);",
                ] {
                    if value.contains(from) {
                        value = value.replace(from, stand_in);
                        hits += 1;
                    }
                }
                ensure!(hits > 0, "noload2 anchor missing");
                value
            }
            // Candidate (tile kernel only): stage the loads of two input tiles per iteration.
            "preload" => {
                ensure!(
                    text.contains("#if MLXL3_AR20_PRELOAD"),
                    "preload anchor missing"
                );
                format!("#define MLXL3_AR20_PRELOAD 1\n{text}")
            }
            other => anyhow::bail!("unknown lab edit {other}"),
        };
    }
    Ok(text)
}

/// Codebook header, optionally with the one-multiply Mul1 byte sum (candidate, exact by identity).
fn lab_header(cb: Codebook, edit: &str) -> Result<String> {
    let header = codebook_header(cb);
    if !edit.split('+').any(|part| part == "mul1fast") {
        return Ok(header);
    }
    const FROM: &str = "uint sum=0x6400u+(pairs&0xffffu)+(pairs>>16);";
    ensure!(
        cb == Codebook::Mul1 && header.contains(FROM),
        "mul1fast needs the Mul1 header"
    );
    Ok(header.replace(FROM, "uint sum=((pairs|0x6400u)*0x10001u)>>16;"))
}

/// Edits whose output is wrong by construction (timing attribution only).
fn ablation(edit: &str) -> bool {
    edit.split('+')
        .any(|part| matches!(part, "nodecode" | "nox" | "noload" | "noload2"))
}

fn name(kind: &str, cols: i32, m: i32, v: Variant) -> String {
    format!(
        "q27lab_{kind}_{cols}_{m}_{}_{}_{}_{}",
        v.mb,
        v.nt,
        u8::from(v.lane),
        v.edit.replace('+', "_")
    )
}

fn grouped_partials(g: &Exl3Group, xhat: &Array, m: i32, v: Variant) -> Result<Array> {
    let groups = i32::try_from(g.widths.len())?;
    let tiles = g.cols / 16;
    ensure!(tiles % v.nt == 0, "NT must divide the output tiles");
    let splits = split_count(g.rows / 16, tiles);
    let header = lab_header(g.cb, v.edit)?
        + &format!("\n#define MLXL3_QMV_LANE_ORDER {}\n", u8::from(v.lane))
        + &format!(
            "\n#define MLXL3_QMV_NT {nt}u\n#define MLXL3_QMV_MB {mb}u\n#define MLXL3_MATRIX_ROWS {m}u\n#define MLXL3_QMV_SG {sg}u\n#define MLXL3_K_BITS {k}u\n#define MLXL3_K3_WINDOW_DECODE 0\n#define MLXL3_BATCH_ROWS 1\n#define GROUPS {groups}\n#define K {k}\n#define CB {cb}\n#define PACKED_U32 {words}\n#define INPUT_DIMS {rows}\n#define TILES_K {kt}\n#define TILES_N {tiles}\n#define N_SPLITS {splits}\n#define LOCAL_OUTPUT_DIMS {cols}\n#define IDENTITY_MAP 1\n#define EXPERT_MAP 0\n#define OUTPUT_TILES {tiles}\n#define ROUTING_REPEAT 1\n#define PROJECTION_STRIDE_TILES 0\n",
            nt = v.nt,
            mb = v.mb,
            sg = g.simdgroups,
            k = g.k,
            cb = g.cb as u32,
            words = g.k * 8,
            rows = g.rows,
            kt = g.rows / 16,
            cols = g.cols,
        );
    let words = g.trellis.reshape(&[-1])?.view(Dtype::UInt32)?;
    Ok(array::metal_kernel(
        &name("grouped", g.cols, m, v),
        &["xhat", "trellis", "tile_map", "tile_sub"],
        &["yhat"],
        &header,
        &edited(MAPPED, v.edit)?,
        &[xhat, &words, &g.identity, &g.tile_sub],
        &[vec![m, splits, g.cols]],
        &[Dtype::Float32],
        [
            (tiles / v.nt) * g.simdgroups * 32,
            (m + v.mb - 1) / v.mb,
            splits,
        ],
        [g.simdgroups * 32, 1, 1],
    )?
    .remove(0))
}

fn tile_partials(l: &Exl3Linear, xhat: &Array, m: i32, v: Variant) -> Result<Array> {
    let tiles = l.cols / 16;
    ensure!(tiles % v.nt == 0, "NT must divide the output tiles");
    let splits = l.splits(l.rows / 16, tiles);
    let header = lab_header(l.cb, v.edit)?
        + "\n#define MLXL3_REFERENCE_MTP_BATCH 0\n"
        + &format!("\n#define MLXL3_QMV_LANE_ORDER {}\n", u8::from(v.lane))
        + &format!(
            "\n#define MLXL3_QMV_NT {nt}u\n#define MLXL3_QMV_MB {mb}u\n#define MLXL3_MATRIX_ROWS {m}u\n#define MLXL3_QMV_SG {sg}u\n#define MLXL3_K_BITS {k}u\n#define MLXL3_FUSE_OUTPUT 0\n#define K {k}\n#define CB {cb}\n#define PACKED_U32 {words}\n#define INPUT_DIMS {input}\n#define TILES_K {kt}\n#define TILES_N {tiles}\n#define N_SPLITS {splits}\n#define OUTPUT_DIMS {output}\n",
            nt = v.nt,
            mb = v.mb,
            sg = l.simdgroups,
            k = l.k,
            cb = l.cb as u32,
            words = l.k * 8,
            input = l.rows,
            kt = l.rows / 16,
            output = l.cols,
        );
    let words = l.trellis.reshape(&[-1])?.view(Dtype::UInt32)?;
    Ok(array::metal_kernel(
        &name("tile", l.cols, m, v),
        &["xhat", "trellis", "svh"],
        &["yhat"],
        &header,
        &edited(TILE, v.edit)?,
        &[xhat, &words, &l.svh],
        &[vec![m, splits, l.cols]],
        &[Dtype::Float32],
        [
            (tiles / v.nt) * l.simdgroups * 32,
            (m + v.mb - 1) / v.mb,
            splits,
        ],
        [l.simdgroups * 32, 1, 1],
    )?
    .remove(0))
}

fn finish(partials: &Array, svh: &Array, m: i32, cols: i32) -> Result<Vec<u16>> {
    partials
        .sum(1, false)?
        .astype(Dtype::Float16)?
        .reshape(&[m, cols / 128, 128])?
        .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
        .reshape(&[m, cols])?
        .mul(&svh.astype(Dtype::Float16)?)?
        .to_f16_bits()
}

fn time_calls(mut call: impl FnMut() -> Result<Array>) -> Result<f64> {
    let mut samples = Vec::new();
    for round in 0..7 {
        array::synchronize()?;
        let start = Instant::now();
        let outputs = (0..8).map(|_| call()).collect::<Result<Vec<_>>>()?;
        Array::async_eval_all(&outputs.iter().collect::<Vec<_>>())?;
        array::synchronize()?;
        if round >= 2 {
            samples.push(start.elapsed().as_secs_f64() * 1e3 / 8.0);
        }
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[samples.len() / 2])
}

fn time_calls_n(calls: usize, mut call: impl FnMut() -> Result<Array>) -> Result<f64> {
    let mut samples = Vec::new();
    for round in 0..7 {
        array::synchronize()?;
        let start = Instant::now();
        let outputs = (0..calls).map(|_| call()).collect::<Result<Vec<_>>>()?;
        Array::async_eval_all(&outputs.iter().collect::<Vec<_>>())?;
        array::synchronize()?;
        if round >= 2 {
            samples.push(start.elapsed().as_secs_f64() * 1e3 / calls as f64);
        }
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[samples.len() / 2])
}

fn variants(m: i32) -> Vec<Variant> {
    let mut out = Vec::new();
    for mb in [1, 2, 3, 4, 6, 8] {
        if mb > m || (mb == 1 && m > 1) {
            continue;
        }
        for nt in [1, 2, 4] {
            if mb * nt > 16 {
                continue;
            }
            for lane in [false, true] {
                out.push(Variant {
                    mb,
                    nt,
                    lane,
                    edit: "",
                });
            }
        }
    }
    out
}

#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_multirow_kernel_lab() -> Result<()> {
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    let layer = std::env::var("MLXL3_Q27_LAB_LAYER").map_or(Ok(10), |v| v.parse::<usize>())?;
    let prefix = format!("model.language_model.layers.{layer}.mlp");
    let group = Exl3Group::new(vec![
        Exl3Linear::from_checkpoint(&checkpoint, &format!("{prefix}.gate_proj"))?,
        Exl3Linear::from_checkpoint(&checkpoint, &format!("{prefix}.up_proj"))?,
    ])?;
    let down = Exl3Linear::from_checkpoint(&checkpoint, &format!("{prefix}.down_proj"))?;
    ensure!(
        group.rows == 5120 && group.cols == 34816 && down.rows == 17408 && down.cols == 5120,
        "requires dense Qwen27B MLP projections"
    );
    let mut report = serde_json::json!({
        "status": "running", "layer": layer,
        "grouped": {"k": group.k, "sg": group.simdgroups, "cols": group.cols},
        "tile": {"k": down.k, "sg": down.simdgroups, "cols": down.cols},
        "rows": [],
    });
    let save = |report: &serde_json::Value| -> Result<()> {
        std::fs::write(&output, serde_json::to_vec_pretty(report)?)?;
        Ok(())
    };
    save(&report)?;
    let lanes = [lane_indices(5120)?, lane_indices(17408)?];
    let widths = std::env::var("MLXL3_Q27_LAB_ROWS").unwrap_or_else(|_| "1,2,3,4,8".into());
    let edits: &[&str] = &[
        "nodecode",
        "nox",
        "noload",
        "nodecode+nox",
        "nodecode+nox+noload",
    ];
    for kind in ["grouped", "tile"] {
        let (input, cols, svh) = if kind == "grouped" {
            (group.rows, group.cols, &group.svh)
        } else {
            (down.rows, down.cols, &down.svh)
        };
        let lane_index = &lanes[usize::from(kind == "tile")];
        let mut stock_m1 = None;
        for m in widths
            .split(',')
            .map(|v| v.parse::<i32>())
            .collect::<std::result::Result<Vec<_>, _>>()?
        {
            let x = rows_input(m, input, 7 + m as u32)?;
            // Stock prologue for both layouts; the lane layout only permutes each tile.
            let (plain, groups) = if kind == "grouped" {
                let groups = group.widths.len() as i32;
                (
                    x.reshape(&[m, 1, input])?
                        .mul(
                            &group
                                .suh
                                .astype(Dtype::Float16)?
                                .reshape(&[1, groups, input])?,
                        )?
                        .reshape(&[m * groups, input / 128, 128])?
                        .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?,
                    groups,
                )
            } else {
                (
                    x.mul(&down.suh.astype(Dtype::Float16)?)?
                        .reshape(&[m, input / 128, 128])?
                        .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?,
                    1,
                )
            };
            let plain = plain.reshape(&[-1])?;
            let permuted = plain
                .reshape(&[m, groups, input])?
                .take(lane_index, 2)?
                .reshape(&[-1])?;
            plain.eval()?;
            permuted.eval()?;
            // Shipped M1 oracle, one row at a time, final F16 bits.
            let mut oracle = Vec::new();
            for row in 0..m {
                let one = x.slice(0, row, row + 1)?;
                if kind == "grouped" {
                    for part in group.forward(&one)? {
                        oracle.extend(part.to_f16_bits()?);
                    }
                } else {
                    oracle.extend(down.forward(&one)?.to_f16_bits()?);
                }
            }
            ensure!(
                oracle.len() == (m * cols) as usize
                    && oracle.iter().all(|&v| half::f16::from_bits(v).is_finite()),
                "incomplete or non-finite oracle"
            );
            let run = |v: Variant| -> Result<Array> {
                let xhat = if v.lane { &permuted } else { &plain };
                if kind == "grouped" {
                    grouped_partials(&group, xhat, m, v)
                } else {
                    tile_partials(&down, xhat, m, v)
                }
            };
            let mut list = variants(m);
            if m == 1 || m == 4 {
                let base = if m == 1 {
                    Variant {
                        mb: 1,
                        nt: 4,
                        lane: true,
                        edit: "",
                    }
                } else {
                    Variant {
                        mb: 2,
                        nt: 2,
                        lane: true,
                        edit: "",
                    }
                };
                list.extend(edits.iter().map(|&edit| Variant { edit, ..base }));
            }
            for v in list {
                let exact = if v.edit.is_empty() {
                    let actual = finish(&run(v)?, svh, m, cols)?;
                    Some(actual.len() == oracle.len() && actual == oracle)
                } else {
                    None
                };
                let elapsed = if exact == Some(false) {
                    None
                } else {
                    Some(time_calls(|| run(v))?)
                };
                if m == 1 && v.mb == 1 && v.nt == 4 && v.lane && v.edit.is_empty() {
                    stock_m1 = elapsed;
                }
                let row = serde_json::json!({
                    "kind": kind, "m": m, "mb": v.mb, "nt": v.nt, "lane": v.lane, "edit": v.edit,
                    "exact": exact, "ms_per_call": elapsed,
                    "relative_to_m1": elapsed.zip(stock_m1).map(|(a, b)| a / b),
                });
                println!("{row}");
                report["rows"].as_array_mut().context("rows")?.push(row);
                save(&report)?;
            }
        }
    }
    report["status"] = "complete".into();
    save(&report)
}

enum Module {
    Group(Exl3Group),
    Single(Exl3Linear),
}

impl Module {
    fn dims(&self) -> (i32, i32, usize, i32, &Array) {
        match self {
            Self::Group(g) => (g.rows, g.cols, g.k, g.simdgroups, &g.svh),
            Self::Single(l) => (l.rows, l.cols, l.k, l.simdgroups, &l.svh),
        }
    }

    /// Stock transformed input of `m` rows, flattened, without lane permutation.
    fn xhat(&self, x: &Array, m: i32) -> Result<(Array, i32)> {
        let (input, ..) = self.dims();
        let (value, groups) = match self {
            Self::Group(g) => {
                let groups = g.widths.len() as i32;
                (
                    x.reshape(&[m, 1, input])?
                        .mul(&g.suh.astype(Dtype::Float16)?.reshape(&[1, groups, input])?)?
                        .reshape(&[m * groups, input / 128, 128])?,
                    groups,
                )
            }
            Self::Single(l) => (
                x.mul(&l.suh.astype(Dtype::Float16)?)?
                    .reshape(&[m, input / 128, 128])?,
                1,
            ),
        };
        Ok((
            value
                .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
                .reshape(&[-1])?,
            groups,
        ))
    }

    fn partials(&self, xhat: &Array, m: i32, v: Variant) -> Result<Array> {
        match self {
            Self::Group(g) => grouped_partials(g, xhat, m, v),
            Self::Single(l) => tile_partials(l, xhat, m, v),
        }
    }

    fn oracle(&self, x: &Array, m: i32) -> Result<Vec<u16>> {
        let mut out = Vec::new();
        for row in 0..m {
            let one = x.slice(0, row, row + 1)?;
            match self {
                Self::Group(g) => {
                    for part in g.forward(&one)? {
                        out.extend(part.to_f16_bits()?);
                    }
                }
                Self::Single(l) => out.extend(l.forward(&one)?.to_f16_bits()?),
            }
        }
        Ok(out)
    }

    /// Layout the shipped engine uses for `m` rows of this module.
    fn stock(&self, m: i32) -> Result<Variant> {
        let (input, cols, k, _, _) = self.dims();
        let tiles = cols / 16;
        let by_tiles = if tiles >= 1024 {
            if tiles % 2 == 0 { 2 } else { 1 }
        } else if tiles % 4 == 0 {
            4
        } else if tiles % 2 == 0 {
            2
        } else {
            1
        };
        let (grouped, mul1, has_lane) = match self {
            Self::Group(g) => (true, g.cb == Codebook::Mul1, g.lane_indices.is_some()),
            Self::Single(l) => (false, l.cb == Codebook::Mul1, l.lane_indices.is_some()),
        };
        let lane = has_lane && matches!(m, 1 | 2);
        if m == 1 {
            let nt = if crate::contracts::dense_decode_nt4(input, cols, k, mul1, true) {
                4
            } else {
                by_tiles
            };
            return Ok(Variant {
                mb: 1,
                nt,
                lane,
                edit: "",
            });
        }
        let (mb, _) = if grouped {
            crate::contracts::qmv_grouped_batch_layout(m, input, cols, k, mul1, true, false)
        } else {
            crate::contracts::qmv_batch_layout(m, false, tiles >= 1024)
        }
        .context("stock layout")?;
        let mut nt = if !grouped && m >= 8 && tiles % 4 == 0 {
            4
        } else {
            by_tiles
        };
        if mb == 2 {
            nt = nt.min(2);
        }
        if mb == 3 {
            nt = 1;
        }
        Ok(Variant {
            mb,
            nt,
            lane,
            edit: "",
        })
    }
}

/// Every distinct decode geometry of the dense text model, first occurrence each.
fn inventory(checkpoint: &Checkpoint) -> Result<Vec<(String, Module)>> {
    let geometry = |prefix: &str| -> Option<(Vec<usize>, bool)> {
        let info = checkpoint.tensors.get(&format!("{prefix}.trellis"))?;
        Some((
            info.shape.clone(),
            checkpoint.tensors.contains_key(&format!("{prefix}.mul1")),
        ))
    };
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    let mut add = |label: &str, prefixes: &[String]| -> Result<()> {
        let Some(shapes) = prefixes
            .iter()
            .map(|p| geometry(p))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(());
        };
        let linears = || {
            prefixes
                .iter()
                .map(|p| Exl3Linear::from_checkpoint(checkpoint, p))
                .collect::<Result<Vec<_>>>()
        };
        if prefixes.len() > 1 {
            let same = shapes.iter().all(|s| {
                s.0[0] == shapes[0].0[0] && s.0[2] == shapes[0].0[2] && s.1 == shapes[0].1
            });
            if !same || shapes[0].0[2] == 7 * 16 {
                for (index, prefix) in prefixes.iter().enumerate() {
                    if seen.insert((false, shapes[index].clone())) {
                        out.push((
                            format!("{label}[{index}]"),
                            Module::Single(Exl3Linear::from_checkpoint(checkpoint, prefix)?),
                        ));
                    }
                }
                return Ok(());
            }
            let key = (
                true,
                (
                    shapes.iter().flat_map(|s| s.0.clone()).collect::<Vec<_>>(),
                    shapes[0].1,
                ),
            );
            if seen.insert(key) {
                out.push((label.to_owned(), Module::Group(Exl3Group::new(linears()?)?)));
            }
        } else if seen.insert((false, shapes[0].clone())) {
            out.push((label.to_owned(), Module::Single(linears()?.remove(0))));
        }
        Ok(())
    };
    for layer in 0..64 {
        let p = format!("model.language_model.layers.{layer}");
        add(
            "mlp_inputs",
            &[format!("{p}.mlp.gate_proj"), format!("{p}.mlp.up_proj")],
        )?;
        add("mlp_down", &[format!("{p}.mlp.down_proj")])?;
        add(
            "gdn_inputs",
            &[
                format!("{p}.linear_attn.in_proj_qkv"),
                format!("{p}.linear_attn.in_proj_z"),
            ],
        )?;
        add("gdn_out", &[format!("{p}.linear_attn.out_proj")])?;
        add("gdn_b", &[format!("{p}.linear_attn.in_proj_b")])?;
        add("gdn_a", &[format!("{p}.linear_attn.in_proj_a")])?;
        add(
            "attn_qkv",
            &[
                format!("{p}.self_attn.q_proj"),
                format!("{p}.self_attn.k_proj"),
                format!("{p}.self_attn.v_proj"),
            ],
        )?;
        add("attn_out", &[format!("{p}.self_attn.o_proj")])?;
    }
    add("head", &["lm_head".to_owned()])?;
    Ok(out)
}

#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_geometry_layout_lab() -> Result<()> {
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    let modules = inventory(&checkpoint)?;
    ensure!(!modules.is_empty(), "no decode geometry found");
    let mut report = serde_json::json!({"status": "running", "rows": []});
    let save = |report: &serde_json::Value| -> Result<()> {
        std::fs::write(&output, serde_json::to_vec_pretty(report)?)?;
        Ok(())
    };
    save(&report)?;
    for (label, module) in &modules {
        let (input, cols, k, sg, svh) = module.dims();
        if k == 7 || (cols < 1024 && label.starts_with("gdn_")) {
            continue;
        }
        let lane_index = lane_indices(input)?;
        let mut m1 = None;
        for m in 1..=4i32 {
            let x = rows_input(m, input, 31 + m as u32)?;
            let (plain, groups) = module.xhat(&x, m)?;
            let permuted = plain
                .reshape(&[m, groups, input])?
                .take(&lane_index, 2)?
                .reshape(&[-1])?;
            plain.eval()?;
            permuted.eval()?;
            let oracle = module.oracle(&x, m)?;
            ensure!(
                oracle.len() == (m * cols) as usize
                    && oracle.iter().all(|&v| half::f16::from_bits(v).is_finite()),
                "incomplete or non-finite oracle"
            );
            let stock = module.stock(m)?;
            let mut list = vec![(stock, true)];
            for nt in [1, 2, 4] {
                if (cols / 16) % nt != 0 || m * nt > 4 {
                    continue;
                }
                for lane in [false, true] {
                    let v = Variant {
                        mb: m,
                        nt,
                        lane,
                        edit: "",
                    };
                    if (v.mb, v.nt, v.lane) != (stock.mb, stock.nt, stock.lane) {
                        list.push((v, false));
                    }
                }
            }
            // Interleave two timing passes so slow drift does not favour list order.
            let mut timings = vec![Vec::new(); list.len()];
            let mut exact = vec![false; list.len()];
            for pass in 0..2 {
                for (index, (v, _)) in list.iter().enumerate() {
                    let xhat = if v.lane { &permuted } else { &plain };
                    if pass == 0 {
                        let actual = finish(&module.partials(xhat, m, *v)?, svh, m, cols)?;
                        exact[index] = actual == oracle;
                    }
                    if exact[index] {
                        timings[index].push(time_calls(|| module.partials(xhat, m, *v))?);
                    }
                }
            }
            for (index, (v, is_stock)) in list.iter().enumerate() {
                let elapsed =
                    (timings[index].len() == 2).then(|| timings[index].iter().sum::<f64>() / 2.0);
                if m == 1 && *is_stock {
                    m1 = elapsed;
                }
                let row = serde_json::json!({
                    "label": label, "input": input, "cols": cols, "k": k, "sg": sg,
                    "grouped": matches!(module, Module::Group(_)),
                    "m": m, "mb": v.mb, "nt": v.nt, "lane": v.lane, "stock": is_stock,
                    "exact": exact[index], "ms_per_call": elapsed, "passes": timings[index],
                    "relative_to_stock_m1": elapsed.zip(m1).map(|(a, b)| a / b),
                });
                println!("{row}");
                report["rows"].as_array_mut().context("rows")?.push(row);
            }
            save(&report)?;
        }
    }
    report["status"] = "complete".into();
    save(&report)
}

/// Q27D-2026-11-17-05: input-side variants of the shared layouts (F32 xhat, no x loads bound).
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_x_layout_lab() -> Result<()> {
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    let prefix = "model.language_model.layers.10";
    let modules = [
        (
            "mlp_inputs",
            Module::Group(Exl3Group::new(vec![
                Exl3Linear::from_checkpoint(&checkpoint, &format!("{prefix}.mlp.gate_proj"))?,
                Exl3Linear::from_checkpoint(&checkpoint, &format!("{prefix}.mlp.up_proj"))?,
            ])?),
        ),
        (
            "mlp_down",
            Module::Single(Exl3Linear::from_checkpoint(
                &checkpoint,
                &format!("{prefix}.mlp.down_proj"),
            )?),
        ),
        (
            "gdn_inputs",
            Module::Group(Exl3Group::new(vec![
                Exl3Linear::from_checkpoint(
                    &checkpoint,
                    &format!("{prefix}.linear_attn.in_proj_qkv"),
                )?,
                Exl3Linear::from_checkpoint(
                    &checkpoint,
                    &format!("{prefix}.linear_attn.in_proj_z"),
                )?,
            ])?),
        ),
        (
            "head",
            Module::Single(Exl3Linear::from_checkpoint(&checkpoint, "lm_head")?),
        ),
    ];
    let mut report = serde_json::json!({"status": "running", "rows": []});
    let save = |report: &serde_json::Value| -> Result<()> {
        std::fs::write(&output, serde_json::to_vec_pretty(report)?)?;
        Ok(())
    };
    save(&report)?;
    for (label, module) in &modules {
        let (input, cols, k, sg, svh) = module.dims();
        let lane_index = lane_indices(input)?;
        for m in 1..=4i32 {
            let x = rows_input(m, input, 57 + m as u32)?;
            let (plain, groups) = module.xhat(&x, m)?;
            let permuted = plain
                .reshape(&[m, groups, input])?
                .take(&lane_index, 2)?
                .reshape(&[-1])?;
            let wide = permuted.astype(Dtype::Float32)?;
            permuted.eval()?;
            wide.eval()?;
            let oracle = module.oracle(&x, m)?;
            ensure!(oracle.len() == (m * cols) as usize, "incomplete oracle");
            let nt = match m {
                1 => module.stock(1)?.nt,
                2 => 2,
                _ => 1,
            };
            let mut list = [
                "",
                "xf32",
                "mul1fast",
                "xf32+mul1fast",
                "nox",
                "noload2",
                "nodecode+noload2",
            ]
            .map(|edit| Variant {
                mb: m,
                nt,
                lane: true,
                edit,
            })
            .to_vec();
            if matches!(module, Module::Single(_)) && m <= 2 {
                // D06 / AR20 preload: never tried on the MLX tile kernel.
                list.push(Variant {
                    mb: m,
                    nt,
                    lane: true,
                    edit: "preload",
                });
                if m == 1 && nt != 2 {
                    list.push(Variant {
                        mb: 1,
                        nt: 2,
                        lane: true,
                        edit: "",
                    });
                    list.push(Variant {
                        mb: 1,
                        nt: 2,
                        lane: true,
                        edit: "preload",
                    });
                }
            }
            let mut timings = vec![Vec::new(); list.len()];
            let mut exact = vec![None; list.len()];
            for pass in 0..3 {
                for (index, v) in list.iter().enumerate() {
                    let xhat = if v.edit.split('+').any(|e| e == "xf32") {
                        &wide
                    } else {
                        &permuted
                    };
                    if pass == 0 && !ablation(v.edit) {
                        exact[index] =
                            Some(finish(&module.partials(xhat, m, *v)?, svh, m, cols)? == oracle);
                    }
                    if exact[index] != Some(false) {
                        timings[index].push(time_calls(|| module.partials(xhat, m, *v))?);
                    }
                }
            }
            for (index, v) in list.iter().enumerate() {
                let mean = (!timings[index].is_empty())
                    .then(|| timings[index].iter().sum::<f64>() / timings[index].len() as f64);
                let base = timings[0].iter().sum::<f64>() / timings[0].len().max(1) as f64;
                let row = serde_json::json!({
                    "label": label, "input": input, "cols": cols, "k": k, "sg": sg,
                    "m": m, "mb": v.mb, "nt": v.nt, "edit": v.edit, "exact": exact[index],
                    "ms_per_call": mean, "passes": timings[index],
                    "relative_to_stock_layout": mean.map(|value| value / base),
                });
                println!("{row}");
                report["rows"].as_array_mut().context("rows")?.push(row);
            }
            save(&report)?;
        }
    }
    report["status"] = "complete".into();
    save(&report)
}

/// The one-multiply Mul1 byte sum equals the shipped three-operation sum for every codeword.
#[test]
fn mul1_fast_byte_sum_matches_for_all_codewords() {
    let mut checked = 0u32;
    for x in 0u32..=0xffff {
        let bits = x.wrapping_mul(0x83DC_D12D);
        let pairs = (bits & 0x00ff_00ff).wrapping_add((bits >> 8) & 0x00ff_00ff);
        let stock = 0x6400u32
            .wrapping_add(pairs & 0xffff)
            .wrapping_add(pairs >> 16);
        let fast = (pairs | 0x6400).wrapping_mul(0x1_0001) >> 16;
        assert_eq!(stock, fast, "codeword {x:#06x}");
        assert!(stock <= 0xffff);
        checked += 1;
    }
    assert_eq!(checked, 65_536);
}

impl Module {
    /// TensorOps (M5 neural accelerator) projection of `m` rows: final F16 values.
    fn tensor(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Single(l) => qmm_tensor(
                x,
                &l.trellis,
                &l.suh,
                &l.svh,
                l.rows,
                l.cols,
                l.k,
                l.cb,
                l.cols / 16,
                0,
            ),
            Self::Group(g) => {
                let mut outputs = Vec::with_capacity(g.widths.len());
                let (mut tiles, mut scales) = (0, 0);
                for (index, &width) in g.widths.iter().enumerate() {
                    outputs.push(qmm_tensor(
                        x,
                        &g.trellis,
                        &g.suh
                            .slice(0, index as i32, index as i32 + 1)?
                            .reshape(&[g.rows])?,
                        &g.svh.slice(0, scales, scales + width)?,
                        g.rows,
                        width,
                        g.k,
                        g.cb,
                        g.cols / 16,
                        tiles,
                    )?);
                    tiles += width / 16;
                    scales += width;
                }
                Array::concatenate(&outputs.iter().collect::<Vec<_>>(), 1)
            }
        }
    }

    /// Q27D-17 lane TensorOps projection of `m` rows.
    fn tensor_lane(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Single(l) => qmm_tensor_lane(
                x,
                &l.trellis,
                &l.suh,
                &l.svh,
                l.lane_indices.as_ref().context("no lane indices")?,
                l.rows,
                l.cols,
                l.k,
                l.cb,
                l.cols / 16,
                0,
            ),
            Self::Group(g) => {
                let indices = g.lane_indices.as_ref().context("no lane indices")?;
                let mut outputs = Vec::with_capacity(g.widths.len());
                let (mut tiles, mut scales) = (0, 0);
                for (index, &width) in g.widths.iter().enumerate() {
                    outputs.push(qmm_tensor_lane(
                        x,
                        &g.trellis,
                        &g.suh
                            .slice(0, index as i32, index as i32 + 1)?
                            .reshape(&[g.rows])?,
                        &g.svh.slice(0, scales, scales + width)?,
                        indices,
                        g.rows,
                        width,
                        g.k,
                        g.cb,
                        g.cols / 16,
                        tiles,
                    )?);
                    tiles += width / 16;
                    scales += width;
                }
                Array::concatenate(&outputs.iter().collect::<Vec<_>>(), 1)
            }
        }
    }

    /// Shipped projection of `m` rows (QMV batch layouts), concatenated like `tensor`.
    fn shipped(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Single(l) => l.forward(x),
            Self::Group(g) => Array::concatenate(&g.forward(x)?.iter().collect::<Vec<_>>(), 1),
        }
    }
}

/// Q27D-15: TensorOps against shipped QMV at 1..16 rows: time, numeric gap, row independence.
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_tensor_rows_lab() -> Result<()> {
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    ensure!(array::supports_exl3_tensor_ops()?, "requires M5 TensorOps");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    let prefix = "model.language_model.layers.10";
    let modules = [
        (
            "mlp_inputs",
            Module::Group(Exl3Group::new(vec![
                Exl3Linear::from_checkpoint(&checkpoint, &format!("{prefix}.mlp.gate_proj"))?,
                Exl3Linear::from_checkpoint(&checkpoint, &format!("{prefix}.mlp.up_proj"))?,
            ])?),
        ),
        (
            "mlp_down",
            Module::Single(Exl3Linear::from_checkpoint(
                &checkpoint,
                &format!("{prefix}.mlp.down_proj"),
            )?),
        ),
        (
            "gdn_inputs",
            Module::Group(Exl3Group::new(vec![
                Exl3Linear::from_checkpoint(
                    &checkpoint,
                    &format!("{prefix}.linear_attn.in_proj_qkv"),
                )?,
                Exl3Linear::from_checkpoint(
                    &checkpoint,
                    &format!("{prefix}.linear_attn.in_proj_z"),
                )?,
            ])?),
        ),
        (
            "gdn_out",
            Module::Single(Exl3Linear::from_checkpoint(
                &checkpoint,
                &format!("{prefix}.linear_attn.out_proj"),
            )?),
        ),
        (
            "head",
            Module::Single(Exl3Linear::from_checkpoint(&checkpoint, "lm_head")?),
        ),
    ];
    let mut report = serde_json::json!({"status": "running", "rows": []});
    let save = |report: &serde_json::Value| -> Result<()> {
        std::fs::write(&output, serde_json::to_vec_pretty(report)?)?;
        Ok(())
    };
    save(&report)?;
    for (label, module) in &modules {
        let (input, cols, ..) = module.dims();
        let mut shipped_m1 = None;
        for m in [1i32, 2, 3, 4, 8, 16] {
            let x = rows_input(m, input, 91 + m as u32)?;
            let oracle = module.oracle(&x, m)?;
            let lane = std::env::var("MLXL3_Q27_LAB_TENSOR").as_deref() == Ok("lane");
            let run_tensor = |x: &Array| {
                if lane {
                    module.tensor_lane(x)
                } else {
                    module.tensor(x)
                }
            };
            let tensor = run_tensor(&x)?.to_f16_bits()?;
            ensure!(
                tensor.len() == oracle.len() && tensor.len() == (m * cols) as usize,
                "short output"
            );
            ensure!(
                tensor.iter().all(|&v| half::f16::from_bits(v).is_finite()),
                "non-finite tensor output"
            );
            // Gap to the shipped single-row oracle.
            let (mut differing, mut max_abs, mut max_ref) = (0usize, 0f32, 0f32);
            for (&a, &b) in tensor.iter().zip(&oracle) {
                if a != b {
                    differing += 1;
                }
                let (a, b) = (
                    half::f16::from_bits(a).to_f32(),
                    half::f16::from_bits(b).to_f32(),
                );
                max_abs = max_abs.max((a - b).abs());
                max_ref = max_ref.max(b.abs());
            }
            // Row independence: row r of the M-row call equals the single-row tensor call.
            let mut independent = true;
            for row in 0..m {
                let alone = run_tensor(&x.slice(0, row, row + 1)?)?.to_f16_bits()?;
                let start = (row * cols) as usize;
                independent &= alone[..] == tensor[start..start + cols as usize];
            }
            let mut tensor_ms = Vec::new();
            let mut shipped_ms = Vec::new();
            for _ in 0..2 {
                tensor_ms.push(time_calls(|| run_tensor(&x))?);
                shipped_ms.push(time_calls(|| module.shipped(&x))?);
            }
            let (t, q) = (
                tensor_ms.iter().sum::<f64>() / 2.0,
                shipped_ms.iter().sum::<f64>() / 2.0,
            );
            if m == 1 {
                shipped_m1 = Some(q);
            }
            let row = serde_json::json!({
                "label": label, "input": input, "cols": cols, "m": m,
                "tensor_ms": t, "shipped_ms": q, "tensor_over_shipped": t / q,
                "tensor_over_shipped_m1": shipped_m1.map(|base| t / base),
                "shipped_over_shipped_m1": shipped_m1.map(|base| q / base),
                "differing_words": differing, "words": oracle.len(), "max_abs": max_abs,
                "max_reference_abs": max_ref, "rows_independent": independent,
            });
            println!("{row}");
            report["rows"].as_array_mut().context("rows")?.push(row);
            save(&report)?;
        }
    }
    report["status"] = "complete".into();
    save(&report)
}

/// Q27D-17: which (input row, output column) of a 16-deep x 32-wide right operand each lane
/// holds in the M5 `matmul2d` cooperative tensor (BM 16, BN 32, BK 16, half/half/float).
#[test]
#[ignore = "M5 TensorOps; writes MLXL3_Q27_KERNEL_LAB"]
fn q27_cooperative_right_layout() -> Result<()> {
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    ensure!(array::supports_exl3_tensor_ops()?, "requires M5 TensorOps");
    let header = "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\nusing namespace mpp;\n";
    let source = r#"
uint lane = thread_index_in_simdgroup;
constexpr auto descriptor = tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, false,
    tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
tensor_ops::matmul2d<descriptor, execution_simdgroup> operation;
auto right = operation.get_right_input_cooperative_tensor<half, half, float>();
for (ushort index = 0; index < 32; ++index) {
    int value = -1;
    if (index < right.get_capacity() && right.is_valid_element(index)) {
        auto c = right.get_multidimensional_index(index);
        value = int(c[0]) * 1000 + int(c[1]);
    }
    layout[lane * 32u + index] = value;
}
if (lane == 0u) layout[1024] = int(right.get_capacity());
"#;
    let dummy = Array::from_i32(&[0], &[1])?;
    let out = array::metal_kernel(
        "q27lab_cooperative_right_layout",
        &["dummy"],
        &["layout"],
        header,
        source,
        &[&dummy],
        &[vec![1025]],
        &[Dtype::Int32],
        [32, 1, 1],
        [32, 1, 1],
    )?
    .remove(0);
    let values = out.astype(Dtype::Float32)?.to_f32()?;
    let capacity = values[1024] as usize;
    let mut lanes = Vec::new();
    for lane in 0..32 {
        let cells = (0..capacity)
            .filter_map(|index| {
                let v = values[lane * 32 + index] as i32;
                (v >= 0).then_some((v / 1000, v % 1000))
            })
            .collect::<Vec<_>>();
        lanes.push(serde_json::json!({"lane": lane, "column_row": cells}));
    }
    let report = serde_json::json!({"capacity": capacity, "lanes": lanes,
        "note": "pairs are (coordinate[0] = output column, coordinate[1] = input row) as used by _qmm_tensor_kernel"});
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}

/// Q27D-17: lane layout of the LEFT operand (M 32 x K 16) for the transposed product
/// (weights on the left, activations on the right), BM 32, BN 16, BK 16.
#[test]
#[ignore = "M5 TensorOps; writes MLXL3_Q27_KERNEL_LAB"]
fn q27_cooperative_left_layout() -> Result<()> {
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    ensure!(array::supports_exl3_tensor_ops()?, "requires M5 TensorOps");
    let (lab_m, lab_k) = (
        std::env::var("MLXL3_Q27_LAB_M").map_or(Ok(32), |v| v.parse::<u32>())?,
        std::env::var("MLXL3_Q27_LAB_K").map_or(Ok(16), |v| v.parse::<u32>())?,
    );
    let header = format!(
        "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\nusing namespace mpp;\n#define LAB_M {lab_m}\n#define LAB_K {lab_k}\n"
    );
    let header = header.as_str();
    let source = r#"
uint lane = thread_index_in_simdgroup;
constexpr auto descriptor = tensor_ops::matmul2d_descriptor(LAB_M, 16, LAB_K, false, true, false,
    tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
tensor_ops::matmul2d<descriptor, execution_simdgroup> operation;
auto left = operation.get_left_input_cooperative_tensor<half, half, float>();
for (ushort index = 0; index < 64; ++index) {
    int value = -1;
    if (index < left.get_capacity() && left.is_valid_element(index)) {
        auto c = left.get_multidimensional_index(index);
        value = int(c[0]) * 1000 + int(c[1]);
    }
    layout[lane * 64u + index] = value;
}
if (lane == 0u) layout[2048] = int(left.get_capacity());
"#;
    let dummy = Array::from_i32(&[0], &[1])?;
    let out = array::metal_kernel(
        &format!("q27lab_cooperative_left_layout_{lab_m}_{lab_k}"),
        &["dummy"],
        &["layout"],
        header,
        source,
        &[&dummy],
        &[vec![2049]],
        &[Dtype::Int32],
        [32, 1, 1],
        [32, 1, 1],
    )?
    .remove(0);
    let values = out.astype(Dtype::Float32)?.to_f32()?;
    let capacity = values[2048] as usize;
    let lanes = (0..32)
        .map(|lane| {
            let cells = (0..capacity)
                .filter_map(|index| {
                    let v = values[lane * 64 + index] as i32;
                    (v >= 0).then_some((v / 1000, v % 1000))
                })
                .collect::<Vec<_>>();
            serde_json::json!({"lane": lane, "c0_c1": cells})
        })
        .collect::<Vec<_>>();
    let report = serde_json::json!({"capacity": capacity, "lanes": lanes,
        "note": "pairs are (coordinate[0], coordinate[1]) of the M x K left operand"});
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

/// Q27D-17 tuning: K split target of the lane TensorOps kernel at 1, 4 and 8 rows.
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_tensor_lane_split_lab() -> Result<()> {
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    let prefix = "model.language_model.layers.10";
    let modules = [
        (
            "mlp_inputs",
            Module::Group(Exl3Group::new(vec![
                Exl3Linear::from_checkpoint(&checkpoint, &format!("{prefix}.mlp.gate_proj"))?,
                Exl3Linear::from_checkpoint(&checkpoint, &format!("{prefix}.mlp.up_proj"))?,
            ])?),
        ),
        (
            "mlp_down",
            Module::Single(Exl3Linear::from_checkpoint(
                &checkpoint,
                &format!("{prefix}.mlp.down_proj"),
            )?),
        ),
        (
            "gdn_inputs",
            Module::Group(Exl3Group::new(vec![
                Exl3Linear::from_checkpoint(
                    &checkpoint,
                    &format!("{prefix}.linear_attn.in_proj_qkv"),
                )?,
                Exl3Linear::from_checkpoint(
                    &checkpoint,
                    &format!("{prefix}.linear_attn.in_proj_z"),
                )?,
            ])?),
        ),
        (
            "gdn_out",
            Module::Single(Exl3Linear::from_checkpoint(
                &checkpoint,
                &format!("{prefix}.linear_attn.out_proj"),
            )?),
        ),
        (
            "head",
            Module::Single(Exl3Linear::from_checkpoint(&checkpoint, "lm_head")?),
        ),
    ];
    let mut report = serde_json::json!({"status": "running", "rows": []});
    for (label, module) in &modules {
        let (input, cols, ..) = module.dims();
        for m in [1i32, 4, 8] {
            let x = rows_input(m, input, 17 + m as u32)?;
            let oracle = module.oracle(&x, m)?;
            let shipped = time_calls(|| module.shipped(&x))?;
            let shapes = [(32, 32), (32, 64), (64, 32)];
            let tokens_lab = std::env::var("MLXL3_Q27_LAB_TOKENS").is_ok();
            let targets: Vec<((i32, i32), (i32, i32))> = if tokens_lab {
                vec![((32, 32), (512, 8)), ((32, 32), (512, 8))]
            } else if std::env::var("MLXL3_Q27_LAB_SHAPES").is_ok() {
                shapes
                    .iter()
                    .flat_map(|&shape| [(shape, (1024, 8)), (shape, (512, 8))])
                    .collect()
            } else {
                [(1024, 8), (2048, 16), (4096, 16), (8192, 32), (16384, 64)]
                    .map(|t| ((32, 16), t))
                    .to_vec()
            };
            for (variant, (shape, target)) in targets.into_iter().enumerate() {
                let token_tile = if tokens_lab && variant == 1 { 8 } else { 16 };
                let (elapsed, finite, near) = with_tensor_lane_tokens(token_tile, || {
                    with_tensor_lane_shape(shape, || {
                        with_tensor_lane_target(target, || -> Result<_> {
                            let bits = module.tensor_lane(&x)?.to_f16_bits()?;
                            let finite = bits.iter().all(|&v| half::f16::from_bits(v).is_finite());
                            let near = bits.iter().zip(&oracle).all(|(&a, &b)| {
                                (half::f16::from_bits(a).to_f32()
                                    - half::f16::from_bits(b).to_f32())
                                .abs()
                                    <= 0.02
                            });
                            Ok((time_calls(|| module.tensor_lane(&x))?, finite, near))
                        })
                    })
                })?;
                let row = serde_json::json!({"label": label, "cols": cols, "m": m,
                    "shape": format!("{}x{}t{}", shape.0, shape.1, token_tile), "target": target.0, "limit": target.1, "lane_ms": elapsed, "shipped_ms": shipped,
                    "lane_over_shipped": elapsed / shipped, "finite": finite, "within_0_02": near});
                println!("{row}");
                report["rows"].as_array_mut().context("rows")?.push(row);
            }
        }
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    report["status"] = "complete".into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

/// Q27D-24: where the lane TensorOps kernel spends its time (ablations, not candidates).
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_tensor_lane_ablation_lab() -> Result<()> {
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    let prefix = "model.language_model.layers.10";
    let modules = [
        (
            "mlp_inputs",
            Module::Group(Exl3Group::new(vec![
                Exl3Linear::from_checkpoint(&checkpoint, &format!("{prefix}.mlp.gate_proj"))?,
                Exl3Linear::from_checkpoint(&checkpoint, &format!("{prefix}.mlp.up_proj"))?,
            ])?),
        ),
        (
            "mlp_down",
            Module::Single(Exl3Linear::from_checkpoint(
                &checkpoint,
                &format!("{prefix}.mlp.down_proj"),
            )?),
        ),
        (
            "head",
            Module::Single(Exl3Linear::from_checkpoint(&checkpoint, "lm_head")?),
        ),
    ];
    let mut report = serde_json::json!({"status": "running", "rows": []});
    for (label, module) in &modules {
        for m in [1i32, 8] {
            let (input, ..) = module.dims();
            let x = rows_input(m, input, 5 + m as u32)?;
            let shipped = time_calls(|| module.shipped(&x))?;
            for edit in [
                "",
                "nodecode",
                "noload",
                "norightload",
                "norun",
                "nodecode+noload",
                "nodecode+noload+norightload",
            ] {
                let elapsed = if edit.is_empty() {
                    time_calls(|| module.tensor_lane(&x))?
                } else {
                    with_tensor_lane_edit(edit, || time_calls(|| module.tensor_lane(&x)))?
                };
                let row = serde_json::json!({"label": label, "m": m, "edit": if edit.is_empty() { "stock" } else { edit },
                    "lane_ms": elapsed, "shipped_ms": shipped, "lane_over_shipped": elapsed / shipped});
                println!("{row}");
                report["rows"].as_array_mut().context("rows")?.push(row);
            }
        }
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    report["status"] = "complete".into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

/// Q27D-27: raw throughput of `matmul2d` on the M5 (no decode, no weight loads): cost per
/// operation by shape, and whether one accumulator chain per simdgroup serializes the unit.
#[test]
#[ignore = "M5 TensorOps; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_tensor_op_throughput_lab() -> Result<()> {
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    ensure!(array::supports_exl3_tensor_ops()?, "requires M5 TensorOps");
    let source = r#"
uint lane = thread_index_in_simdgroup;
uint group = threadgroup_position_in_grid.x;
constexpr auto descriptor = tensor_ops::matmul2d_descriptor(
    OP_M, OP_N, OP_K, false, true, false,
    tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
tensor_ops::matmul2d<descriptor, execution_simdgroup> operation;
using left_type = tensor_ops::matmul2d<descriptor, execution_simdgroup>
    ::cooperative_tensor_left_input_t<half, half, float>;
auto left0 = operation.get_left_input_cooperative_tensor<half, half, float>();
auto left1 = operation.get_left_input_cooperative_tensor<half, half, float>();
for (ushort i = 0; i < left0.get_capacity(); ++i) {
    if (left0.is_valid_element(i)) {
        left0[i] = half(float(lane * 7u + i) * 1e-3f);
        left1[i] = half(float(lane * 5u + i) * 1e-3f);
    }
}
auto acc0 = operation.get_destination_cooperative_tensor<
    left_type, tensor<device half, dextents<int, 2>, tensor_inline>, float>();
auto acc1 = operation.get_destination_cooperative_tensor<
    left_type, tensor<device half, dextents<int, 2>, tensor_inline>, float>();
for (ushort i = 0; i < acc0.get_capacity(); ++i) {
    if (acc0.is_valid_element(i)) { acc0[i] = 0.0f; acc1[i] = 0.0f; }
}
device half* activation = const_cast<device half*>(x);
for (uint it = 0u; it < OP_ITER; it += OP_ACC) {
    auto right = tensor(activation + (it % 64u) * OP_K, dextents<int, 2>{int(OP_K), int(OP_N)},
        array<int, 2>{1, int(64u * OP_K + OP_K)});
#if OP_ACC == 1
    operation.run(left0, right, acc0);
#else
    operation.run(left0, right, acc0);
    operation.run(left1, right, acc1);
#endif
}
float total = 0.0f;
for (ushort i = 0; i < acc0.get_capacity(); ++i) {
    if (acc0.is_valid_element(i)) total += acc0[i] + acc1[i];
}
out[group * 32u + lane] = total;
"#;
    let groups = 4096i32;
    let iterations = 256u32;
    let mut report = serde_json::json!({"status": "running", "groups": groups, "iterations": iterations, "rows": []});
    let shapes: [(u32, u32, u32); 12] = [
        (32, 16, 32),
        (32, 8, 32),
        (32, 32, 32),
        (16, 16, 32),
        (16, 8, 32),
        (8, 16, 32),
        (16, 32, 32),
        (8, 32, 32),
        (64, 16, 32),
        (32, 16, 64),
        (64, 8, 32),
        (32, 16, 16),
    ];
    for (m, n, k) in shapes {
        // x holds 65 tiles of N x K (stride 65 K) so every read is in bounds.
        let x = Array::from_f32(
            &(0..(n * (65 * k)))
                .map(|i| ((i % 251) as f32) * 1e-3)
                .collect::<Vec<_>>(),
            &[i32::try_from(n * 65 * k)?],
        )?
        .astype(Dtype::Float16)?;
        for accs in [1u32, 2] {
            let header = format!(
                "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\nusing namespace mpp;\n#define OP_M {m}u\n#define OP_N {n}u\n#define OP_K {k}u\n#define OP_ITER {iterations}u\n#define OP_ACC {accs}\n"
            );
            let name = format!("q27lab_op_throughput_m{m}_n{n}_k{k}_a{accs}");
            let call = || -> Result<Array> {
                Ok(array::metal_kernel(
                    &name,
                    &["x"],
                    &["out"],
                    &header,
                    source,
                    &[&x],
                    &[vec![groups * 32]],
                    &[Dtype::Float32],
                    [groups * 32, 1, 1],
                    [32, 1, 1],
                )?
                .remove(0))
            };
            let values = call()?.to_f32()?;
            let finite = values.iter().all(|v| v.is_finite()) && values.iter().any(|&v| v != 0.0);
            let elapsed = time_calls(call)?;
            let ops = f64::from(groups) * f64::from(iterations);
            let tflops = 2.0 * ops * f64::from(m * n * k) / (elapsed * 1e-3) / 1e12;
            let left_g_per_s = ops * f64::from(m * k) / (elapsed * 1e-3) / 1e9;
            let row = serde_json::json!({"m": m, "n": n, "k": k, "accumulators": accs, "ms": elapsed,
                "ns_per_op_per_core": elapsed * 1e6 / ops * 10.0, "tflops": tflops,
                "left_elements_g_per_s": left_g_per_s, "finite_nonzero": finite});
            println!("{row}");
            report["rows"].as_array_mut().context("rows")?.push(row);
        }
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    report["status"] = "complete".into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

/// Q27D-28: double-buffered left operand in the lane TensorOps kernel; outputs must be
/// bit-identical to the single-buffer kernel (same operations in the same order).
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_tensor_lane_buffers_lab() -> Result<()> {
    use super::with_tensor_lane_buffers;
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    let prefix = "model.language_model.layers.10";
    let modules = [
        (
            "mlp_inputs",
            Module::Group(Exl3Group::new(vec![
                Exl3Linear::from_checkpoint(&checkpoint, &format!("{prefix}.mlp.gate_proj"))?,
                Exl3Linear::from_checkpoint(&checkpoint, &format!("{prefix}.mlp.up_proj"))?,
            ])?),
        ),
        (
            "mlp_down",
            Module::Single(Exl3Linear::from_checkpoint(
                &checkpoint,
                &format!("{prefix}.mlp.down_proj"),
            )?),
        ),
        (
            "gdn_inputs",
            Module::Group(Exl3Group::new(vec![
                Exl3Linear::from_checkpoint(
                    &checkpoint,
                    &format!("{prefix}.linear_attn.in_proj_qkv"),
                )?,
                Exl3Linear::from_checkpoint(
                    &checkpoint,
                    &format!("{prefix}.linear_attn.in_proj_z"),
                )?,
            ])?),
        ),
        (
            "gdn_out",
            Module::Single(Exl3Linear::from_checkpoint(
                &checkpoint,
                &format!("{prefix}.linear_attn.out_proj"),
            )?),
        ),
        (
            "head",
            Module::Single(Exl3Linear::from_checkpoint(&checkpoint, "lm_head")?),
        ),
    ];
    let mut report = serde_json::json!({"status": "running", "rows": []});
    let mut all_identical = true;
    for (label, module) in &modules {
        let (input, cols, ..) = module.dims();
        for m in [1i32, 4, 8, 16] {
            let x = rows_input(m, input, 29 + m as u32)?;
            let shipped = time_calls(|| module.shipped(&x))?;
            let one = with_tensor_lane_buffers(1, || module.tensor_lane(&x))?.to_f16_bits()?;
            let two = with_tensor_lane_buffers(2, || module.tensor_lane(&x))?.to_f16_bits()?;
            let finite = one.iter().all(|&v| half::f16::from_bits(v).is_finite());
            let identical = one == two && one.len() == (m * cols) as usize;
            all_identical &= identical && finite;
            // Alternated order: 1, 2, 2, 1.
            let mut times = [Vec::new(), Vec::new()];
            for buffers in [1, 2, 2, 1] {
                let t =
                    with_tensor_lane_buffers(buffers, || time_calls(|| module.tensor_lane(&x)))?;
                times[(buffers - 1) as usize].push(t);
            }
            let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
            let row = serde_json::json!({"label": label, "cols": cols, "m": m, "shipped_ms": shipped,
                "single_ms": times[0], "double_ms": times[1],
                "double_over_single": mean(&times[1]) / mean(&times[0]),
                "single_over_shipped": mean(&times[0]) / shipped,
                "bit_identical": identical, "finite": finite, "values": one.len()});
            println!("{row}");
            report["rows"].as_array_mut().context("rows")?.push(row);
        }
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    report["status"] = if all_identical {
        "complete"
    } else {
        "divergent"
    }
    .into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    ensure!(all_identical, "double-buffered lane kernel diverged");
    Ok(())
}

/// Q27D-30: cost of the MLX prologue (F16, signs, Hadamard, lane order, padding) and epilogue
/// (split sum, F16, Hadamard, signs) around each lane TensorOps call at 8 rows.
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_tensor_lane_glue_lab() -> Result<()> {
    use super::{lane_epilogue, lane_prologue};
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    // (name, prefix, calls per verify): 48 GDN layers, 16 attention layers, 64 MLPs, one head.
    let linears = [
        (
            "gdn_qkv",
            "model.language_model.layers.10.linear_attn.in_proj_qkv",
            48,
        ),
        (
            "gdn_z",
            "model.language_model.layers.10.linear_attn.in_proj_z",
            48,
        ),
        (
            "gdn_out",
            "model.language_model.layers.10.linear_attn.out_proj",
            48,
        ),
        (
            "attn_q",
            "model.language_model.layers.11.self_attn.q_proj",
            16,
        ),
        (
            "attn_k",
            "model.language_model.layers.11.self_attn.k_proj",
            16,
        ),
        (
            "attn_v",
            "model.language_model.layers.11.self_attn.v_proj",
            16,
        ),
        (
            "attn_o",
            "model.language_model.layers.11.self_attn.o_proj",
            16,
        ),
        (
            "mlp_gate",
            "model.language_model.layers.10.mlp.gate_proj",
            64,
        ),
        ("mlp_up", "model.language_model.layers.10.mlp.up_proj", 64),
        (
            "mlp_down",
            "model.language_model.layers.10.mlp.down_proj",
            64,
        ),
        ("head", "lm_head", 1),
    ];
    let (m, padded) = (8i32, 16i32);
    // Q27D-30 revision: 64 calls per submission amortize the submission and synchronization.
    let calls_per_round =
        std::env::var("MLXL3_Q27_LAB_CALLS").map_or(Ok(64), |v| v.parse::<usize>())?;
    let mut report = serde_json::json!({"status": "running", "rows": []});
    let (mut full_total, mut glue_total) = (0.0, 0.0);
    for (name, prefix, calls) in linears {
        let linear = match Exl3Linear::from_checkpoint(&checkpoint, prefix) {
            Ok(linear) => linear,
            Err(error) => {
                println!("skip {name}: {error:#}");
                continue;
            }
        };
        let Some(indices) = linear.lane_indices.as_ref() else {
            println!("skip {name}: no lane indices");
            continue;
        };
        let x = rows_input(m, linear.rows, 41)?;
        let groups = linear.cols / 32;
        let splits = crate::contracts::tensor_lane_splits(groups, linear.rows / 32);
        let partials = Array::from_f32(
            &(0..(splits * padded * linear.cols) as u32)
                .map(|i| ((i.wrapping_mul(2654435761) >> 12) % 4001) as f32 / 2000.0 - 1.0)
                .collect::<Vec<_>>(),
            &[splits, padded, linear.cols],
        )?;
        let full = time_calls_n(calls_per_round, || {
            qmm_tensor_lane(
                &x,
                &linear.trellis,
                &linear.suh,
                &linear.svh,
                indices,
                linear.rows,
                linear.cols,
                linear.k,
                linear.cb,
                linear.cols / 16,
                0,
            )
        })?;
        let prologue = time_calls_n(calls_per_round, || {
            lane_prologue(&x, &linear.suh, indices, m, padded, linear.rows)
        })?;
        let epilogue = time_calls_n(calls_per_round, || {
            lane_epilogue(&partials, &linear.svh, splits, m, padded, linear.cols)?.astype(x.dtype())
        })?;
        full_total += full * f64::from(calls);
        glue_total += (prologue + epilogue) * f64::from(calls);
        let row = serde_json::json!({"name": name, "input": linear.rows, "output": linear.cols,
            "splits": splits, "calls_per_verify": calls, "full_ms": full, "prologue_ms": prologue,
            "epilogue_ms": epilogue, "glue_share": (prologue + epilogue) / full, "x_dtype": format!("{:?}", x.dtype())});
        println!("{row}");
        report["rows"].as_array_mut().context("rows")?.push(row);
    }
    report["verify_projection_ms"] = full_total.into();
    report["verify_glue_ms"] = glue_total.into();
    report["calls_per_round"] = calls_per_round.into();
    report["status"] = "complete".into();
    println!("{report}");
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

/// Q27D-31: fused lane prologue/epilogue must reproduce the MLX chains bit for bit; then time.
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_tensor_lane_fused_lab() -> Result<()> {
    use super::with_tensor_lane_fused;
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    let gdn = "model.language_model.layers.10";
    let attn = "model.language_model.layers.11";
    let single = |name: &str| -> Result<Module> {
        Ok(Module::Single(Exl3Linear::from_checkpoint(
            &checkpoint,
            name,
        )?))
    };
    let group = |names: &[String]| -> Result<Module> {
        Ok(Module::Group(Exl3Group::new(
            names
                .iter()
                .map(|n| Exl3Linear::from_checkpoint(&checkpoint, n))
                .collect::<Result<Vec<_>>>()?,
        )?))
    };
    let modules = [
        (
            "gdn_inputs",
            group(&[
                format!("{gdn}.linear_attn.in_proj_qkv"),
                format!("{gdn}.linear_attn.in_proj_z"),
            ])?,
        ),
        ("gdn_out", single(&format!("{gdn}.linear_attn.out_proj"))?),
        ("attn_q", single(&format!("{attn}.self_attn.q_proj"))?),
        ("attn_k", single(&format!("{attn}.self_attn.k_proj"))?),
        ("attn_v", single(&format!("{attn}.self_attn.v_proj"))?),
        ("attn_o", single(&format!("{attn}.self_attn.o_proj"))?),
        (
            "mlp_inputs",
            group(&[format!("{gdn}.mlp.gate_proj"), format!("{gdn}.mlp.up_proj")])?,
        ),
        ("mlp_down", single(&format!("{gdn}.mlp.down_proj"))?),
        ("head", single("lm_head")?),
    ];
    let mut report = serde_json::json!({"status": "running", "rows": []});
    let mut identical_all = true;
    let mut compared = 0usize;
    for (label, module) in &modules {
        let (input, cols, ..) = module.dims();
        for m in [3i32, 4, 8, 16] {
            for dtype in [Dtype::Float16, Dtype::BFloat16] {
                let x = rows_input(m, input, 53 + m as u32)?.astype(dtype)?;
                x.eval()?;
                let stock = with_tensor_lane_fused(false, || module.tensor_lane(&x))?;
                let fused = with_tensor_lane_fused(true, || module.tensor_lane(&x))?;
                let (a, b) = (
                    stock.astype(Dtype::Float32)?.to_f32()?,
                    fused.astype(Dtype::Float32)?.to_f32()?,
                );
                let bits_equal = stock.dtype() == fused.dtype()
                    && stock.shape() == fused.shape()
                    && a.len() == (m * cols) as usize
                    && a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits());
                let finite = a.iter().all(|v| v.is_finite());
                let differing = a
                    .iter()
                    .zip(&b)
                    .filter(|(p, q)| p.to_bits() != q.to_bits())
                    .count();
                identical_all &= bits_equal && finite;
                compared += a.len();
                let timing = dtype == Dtype::Float16 && m == 8;
                let (stock_ms, fused_ms) = if timing {
                    (
                        with_tensor_lane_fused(false, || {
                            time_calls_n(64, || module.tensor_lane(&x))
                        })?,
                        with_tensor_lane_fused(true, || {
                            time_calls_n(64, || module.tensor_lane(&x))
                        })?,
                    )
                } else {
                    (f64::NAN, f64::NAN)
                };
                let row = serde_json::json!({"label": label, "m": m, "dtype": format!("{dtype:?}"),
                    "values": a.len(), "bit_identical": bits_equal, "differing": differing, "finite": finite,
                    "stock_ms": if timing { serde_json::json!(stock_ms) } else { serde_json::Value::Null },
                    "fused_ms": if timing { serde_json::json!(fused_ms) } else { serde_json::Value::Null }});
                println!("{row}");
                report["rows"].as_array_mut().context("rows")?.push(row);
            }
        }
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    report["values_compared"] = compared.into();
    report["status"] = if identical_all {
        "complete"
    } else {
        "divergent"
    }
    .into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    ensure!(
        identical_all,
        "fused lane glue diverged from the MLX chains"
    );
    Ok(())
}

/// Q27D-33: fused prologue/epilogue in the exact QMV paths (`Exl3Linear::forward`,
/// `Exl3Group::forward`) must reproduce the MLX chains bit for bit at 1..8 rows; then time.
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_qmv_fused_glue_lab() -> Result<()> {
    use super::with_qmv_fused_glue;
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    let gdn = "model.language_model.layers.10";
    let attn = "model.language_model.layers.11";
    let single = |name: &str| -> Result<Module> {
        Ok(Module::Single(Exl3Linear::from_checkpoint(
            &checkpoint,
            name,
        )?))
    };
    let group = |names: &[String]| -> Result<Module> {
        Ok(Module::Group(Exl3Group::new(
            names
                .iter()
                .map(|n| Exl3Linear::from_checkpoint(&checkpoint, n))
                .collect::<Result<Vec<_>>>()?,
        )?))
    };
    let modules = [
        (
            "gdn_inputs",
            group(&[
                format!("{gdn}.linear_attn.in_proj_qkv"),
                format!("{gdn}.linear_attn.in_proj_z"),
            ])?,
        ),
        ("gdn_out", single(&format!("{gdn}.linear_attn.out_proj"))?),
        ("attn_q", single(&format!("{attn}.self_attn.q_proj"))?),
        ("attn_k", single(&format!("{attn}.self_attn.k_proj"))?),
        ("attn_o", single(&format!("{attn}.self_attn.o_proj"))?),
        (
            "mlp_inputs",
            group(&[format!("{gdn}.mlp.gate_proj"), format!("{gdn}.mlp.up_proj")])?,
        ),
        ("mlp_down", single(&format!("{gdn}.mlp.down_proj"))?),
        ("head", single("lm_head")?),
    ];
    let forward = |module: &Module, x: &Array| -> Result<Array> {
        match module {
            Module::Single(l) => l.forward(x),
            Module::Group(g) => {
                let outputs = g.forward(x)?;
                Array::concatenate(&outputs.iter().collect::<Vec<_>>(), 1)
            }
        }
    };
    let mut report = serde_json::json!({"status": "running", "rows": []});
    let (mut identical_all, mut compared) = (true, 0usize);
    for (label, module) in &modules {
        let (input, cols, ..) = module.dims();
        for m in [1i32, 2, 3, 4, 5, 8] {
            for dtype in [Dtype::Float16, Dtype::BFloat16] {
                let x = rows_input(m, input, 71 + m as u32)?.astype(dtype)?;
                x.eval()?;
                let stock = with_qmv_fused_glue(false, || forward(module, &x))?;
                let fused = with_qmv_fused_glue(true, || forward(module, &x))?;
                let (a, b) = (
                    stock.astype(Dtype::Float32)?.to_f32()?,
                    fused.astype(Dtype::Float32)?.to_f32()?,
                );
                let bits_equal = stock.dtype() == fused.dtype()
                    && stock.shape() == fused.shape()
                    && a.len() == (m * cols) as usize
                    && a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits());
                let finite = a.iter().all(|v| v.is_finite());
                let differing = a
                    .iter()
                    .zip(&b)
                    .filter(|(p, q)| p.to_bits() != q.to_bits())
                    .count();
                identical_all &= bits_equal && finite;
                compared += a.len();
                let timing = dtype == Dtype::Float16 && matches!(m, 1 | 2);
                let times = if timing {
                    Some((
                        with_qmv_fused_glue(false, || time_calls_n(64, || forward(module, &x)))?,
                        with_qmv_fused_glue(true, || time_calls_n(64, || forward(module, &x)))?,
                    ))
                } else {
                    None
                };
                let row = serde_json::json!({"label": label, "m": m, "dtype": format!("{dtype:?}"),
                    "values": a.len(), "bit_identical": bits_equal, "differing": differing, "finite": finite,
                    "stock_ms": times.map(|t| t.0), "fused_ms": times.map(|t| t.1)});
                println!("{row}");
                report["rows"].as_array_mut().context("rows")?.push(row);
            }
        }
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    report["values_compared"] = compared.into();
    report["status"] = if identical_all {
        "complete"
    } else {
        "divergent"
    }
    .into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    ensure!(identical_all, "fused QMV glue diverged from the MLX chains");
    Ok(())
}

/// Q27P-01: prefill projections of the 27B at 256 rows (the production chunk) on the M5
/// TensorOps kernel: time, effective TFLOPS and decoded weights per second (diagnostic).
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_prefill_projection_lab() -> Result<()> {
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    let gdn = "model.language_model.layers.10";
    let attn = "model.language_model.layers.11";
    let single = |name: &str| -> Result<Module> {
        Ok(Module::Single(Exl3Linear::from_checkpoint(
            &checkpoint,
            name,
        )?))
    };
    let group = |names: &[String]| -> Result<Module> {
        Ok(Module::Group(Exl3Group::new(
            names
                .iter()
                .map(|n| Exl3Linear::from_checkpoint(&checkpoint, n))
                .collect::<Result<Vec<_>>>()?,
        )?))
    };
    // (label, module, calls per 256-token chunk of the 64-layer model)
    let modules = [
        (
            "gdn_inputs",
            group(&[
                format!("{gdn}.linear_attn.in_proj_qkv"),
                format!("{gdn}.linear_attn.in_proj_z"),
            ])?,
            48,
        ),
        (
            "gdn_out",
            single(&format!("{gdn}.linear_attn.out_proj"))?,
            48,
        ),
        ("attn_q", single(&format!("{attn}.self_attn.q_proj"))?, 16),
        ("attn_k", single(&format!("{attn}.self_attn.k_proj"))?, 16),
        ("attn_v", single(&format!("{attn}.self_attn.v_proj"))?, 16),
        ("attn_o", single(&format!("{attn}.self_attn.o_proj"))?, 16),
        (
            "mlp_inputs",
            group(&[format!("{gdn}.mlp.gate_proj"), format!("{gdn}.mlp.up_proj")])?,
            64,
        ),
        ("mlp_down", single(&format!("{gdn}.mlp.down_proj"))?, 64),
    ];
    let mut report = serde_json::json!({"status": "running", "rows": []});
    let mut chunk_ms = 0.0;
    for (label, module, calls) in &modules {
        let (input, cols, ..) = module.dims();
        for m in [256i32, 512] {
            let x = rows_input(m, input, 91 + m as u32)?;
            let out = module.tensor(&x)?;
            let values = out.astype(Dtype::Float32)?.to_f32()?;
            let finite =
                values.len() == (m * cols) as usize && values.iter().all(|v| v.is_finite());
            let ms = time_calls_n(8, || module.tensor(&x))?;
            let macs = f64::from(m) * f64::from(input) * f64::from(cols);
            if m == 256 {
                chunk_ms += ms * f64::from(*calls);
            }
            let row = serde_json::json!({"label": label, "m": m, "input": input, "output": cols, "ms": ms,
                "tflops": 2.0 * macs / (ms * 1e-3) / 1e12,
                "weights_g_per_s": f64::from(input) * f64::from(cols) / (ms * 1e-3) / 1e9,
                "calls_per_chunk": calls, "finite": finite});
            println!("{row}");
            report["rows"].as_array_mut().context("rows")?.push(row);
        }
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    report["projections_ms_per_256_chunk"] = chunk_ms.into();
    report["status"] = "complete".into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}

/// Q27P-02: prefill kernel with one decoded weight tile shared by 2/4/8 row blocks must match
/// the stock prefill kernel bit for bit; then time at 256 rows.
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_prefill_shared_lab() -> Result<()> {
    use super::with_dense_prefill_shared;
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    let gdn = "model.language_model.layers.10";
    let attn = "model.language_model.layers.11";
    let single = |name: &str| -> Result<Module> {
        Ok(Module::Single(Exl3Linear::from_checkpoint(
            &checkpoint,
            name,
        )?))
    };
    let group = |names: &[String]| -> Result<Module> {
        Ok(Module::Group(Exl3Group::new(
            names
                .iter()
                .map(|n| Exl3Linear::from_checkpoint(&checkpoint, n))
                .collect::<Result<Vec<_>>>()?,
        )?))
    };
    let modules = [
        (
            "gdn_inputs",
            group(&[
                format!("{gdn}.linear_attn.in_proj_qkv"),
                format!("{gdn}.linear_attn.in_proj_z"),
            ])?,
        ),
        ("gdn_out", single(&format!("{gdn}.linear_attn.out_proj"))?),
        ("attn_q", single(&format!("{attn}.self_attn.q_proj"))?),
        ("attn_k", single(&format!("{attn}.self_attn.k_proj"))?),
        ("attn_o", single(&format!("{attn}.self_attn.o_proj"))?),
        (
            "mlp_inputs",
            group(&[format!("{gdn}.mlp.gate_proj"), format!("{gdn}.mlp.up_proj")])?,
        ),
        ("mlp_down", single(&format!("{gdn}.mlp.down_proj"))?),
        ("head", single("lm_head")?),
    ];
    let mut report = serde_json::json!({"status": "running", "rows": []});
    let (mut identical_all, mut compared) = (true, 0usize);
    for (label, module) in &modules {
        let (input, cols, ..) = module.dims();
        let sizes: &[i32] = if *label == "head" {
            &[24, 73]
        } else {
            &[24, 73, 128, 200, 256, 512]
        };
        for &m in sizes {
            let x = rows_input(m, input, 131 + m as u32)?;
            let stock_bits = with_dense_prefill_shared(0, || module.tensor(&x))?.to_f16_bits()?;
            ensure!(
                stock_bits.len() == (m * cols) as usize,
                "short stock output"
            );
            let finite = stock_bits
                .iter()
                .all(|&v| half::f16::from_bits(v).is_finite());
            let mut row = serde_json::json!({"label": label, "m": m, "values": stock_bits.len(), "finite": finite});
            for g in [2, 4, 8] {
                let bits = with_dense_prefill_shared(g, || module.tensor(&x))?.to_f16_bits()?;
                let differing = bits.iter().zip(&stock_bits).filter(|(a, b)| a != b).count()
                    + bits.len().abs_diff(stock_bits.len());
                identical_all &= differing == 0 && finite;
                compared += bits.len();
                row[format!("g{g}_differing")] = differing.into();
            }
            if m == 256 {
                let mut times = serde_json::Map::new();
                for g in [0, 2, 4, 8, 0] {
                    let t = with_dense_prefill_shared(g, || time_calls_n(8, || module.tensor(&x)))?;
                    times
                        .entry(format!("g{g}_ms"))
                        .or_insert(serde_json::json!([]))
                        .as_array_mut()
                        .context("times")?
                        .push(t.into());
                }
                row["times"] = serde_json::Value::Object(times);
            }
            println!("{row}");
            report["rows"].as_array_mut().context("rows")?.push(row);
        }
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    report["values_compared"] = compared.into();
    report["status"] = if identical_all {
        "complete"
    } else {
        "divergent"
    }
    .into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    ensure!(
        identical_all,
        "shared prefill kernel diverged from the stock kernel"
    );
    Ok(())
}

/// Q27P-03: 32-deep shared prefill kernel (not bit-exact): deviation from the stock kernel and
/// time at 256 rows, against the bit-exact shared kernel at depth 16.
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_prefill_depth_lab() -> Result<()> {
    use super::{with_dense_prefill_depth, with_dense_prefill_shared};
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    let gdn = "model.language_model.layers.10";
    let attn = "model.language_model.layers.11";
    let single = |name: &str| -> Result<Module> {
        Ok(Module::Single(Exl3Linear::from_checkpoint(
            &checkpoint,
            name,
        )?))
    };
    let group = |names: &[String]| -> Result<Module> {
        Ok(Module::Group(Exl3Group::new(
            names
                .iter()
                .map(|n| Exl3Linear::from_checkpoint(&checkpoint, n))
                .collect::<Result<Vec<_>>>()?,
        )?))
    };
    let modules = [
        (
            "gdn_inputs",
            group(&[
                format!("{gdn}.linear_attn.in_proj_qkv"),
                format!("{gdn}.linear_attn.in_proj_z"),
            ])?,
            48,
        ),
        (
            "gdn_out",
            single(&format!("{gdn}.linear_attn.out_proj"))?,
            48,
        ),
        ("attn_q", single(&format!("{attn}.self_attn.q_proj"))?, 16),
        ("attn_k", single(&format!("{attn}.self_attn.k_proj"))?, 32),
        ("attn_o", single(&format!("{attn}.self_attn.o_proj"))?, 16),
        (
            "mlp_inputs",
            group(&[format!("{gdn}.mlp.gate_proj"), format!("{gdn}.mlp.up_proj")])?,
            64,
        ),
        ("mlp_down", single(&format!("{gdn}.mlp.down_proj"))?, 64),
    ];
    let mut report = serde_json::json!({"status": "running", "rows": []});
    let (mut exact_ms, mut deep_ms) = (0.0, 0.0);
    for (label, module, calls) in &modules {
        let (input, cols, ..) = module.dims();
        for m in [73i32, 256] {
            let x = rows_input(m, input, 151 + m as u32)?;
            let stock = with_dense_prefill_shared(0, || module.tensor(&x))?
                .astype(Dtype::Float32)?
                .to_f32()?;
            let deep = with_dense_prefill_shared(4, || {
                with_dense_prefill_depth(32, || module.tensor(&x))
            })?
            .astype(Dtype::Float32)?
            .to_f32()?;
            ensure!(
                stock.len() == (m * cols) as usize && deep.len() == stock.len(),
                "short outputs"
            );
            let finite = deep.iter().all(|v| v.is_finite());
            let differing = stock
                .iter()
                .zip(&deep)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let max_abs = stock
                .iter()
                .zip(&deep)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let scale = stock.iter().map(|v| v.abs()).fold(0f32, f32::max);
            let mut row = serde_json::json!({"label": label, "m": m, "values": stock.len(), "finite": finite,
                "differing_fraction": differing as f64 / stock.len() as f64, "max_abs": max_abs, "max_stock": scale});
            if m == 256 {
                let mut samples = [Vec::new(), Vec::new()];
                for depth in [16, 32, 32, 16] {
                    let t = with_dense_prefill_shared(4, || {
                        with_dense_prefill_depth(depth, || time_calls_n(8, || module.tensor(&x)))
                    })?;
                    samples[usize::from(depth == 32)].push(t);
                }
                let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
                exact_ms += mean(&samples[0]) * f64::from(*calls);
                deep_ms += mean(&samples[1]) * f64::from(*calls);
                row["depth16_ms"] = samples[0].clone().into();
                row["depth32_ms"] = samples[1].clone().into();
            }
            println!("{row}");
            report["rows"].as_array_mut().context("rows")?.push(row);
        }
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    report["chunk_ms_depth16"] = exact_ms.into();
    report["chunk_ms_depth32"] = deep_ms.into();
    report["status"] = "complete".into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}

/// Q27P-04: tile shapes of the shared prefill kernel; outputs must match the stock kernel bit
/// for bit, then time at 256 rows.
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_prefill_tile_lab() -> Result<()> {
    use super::{with_dense_prefill_shared, with_prefill_tile};
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    let gdn = "model.language_model.layers.10";
    let attn = "model.language_model.layers.11";
    let single = |name: &str| -> Result<Module> {
        Ok(Module::Single(Exl3Linear::from_checkpoint(
            &checkpoint,
            name,
        )?))
    };
    let group = |names: &[String]| -> Result<Module> {
        Ok(Module::Group(Exl3Group::new(
            names
                .iter()
                .map(|n| Exl3Linear::from_checkpoint(&checkpoint, n))
                .collect::<Result<Vec<_>>>()?,
        )?))
    };
    let modules = [
        (
            "gdn_inputs",
            group(&[
                format!("{gdn}.linear_attn.in_proj_qkv"),
                format!("{gdn}.linear_attn.in_proj_z"),
            ])?,
            48,
        ),
        (
            "gdn_out",
            single(&format!("{gdn}.linear_attn.out_proj"))?,
            48,
        ),
        ("attn_q", single(&format!("{attn}.self_attn.q_proj"))?, 16),
        ("attn_k", single(&format!("{attn}.self_attn.k_proj"))?, 32),
        ("attn_o", single(&format!("{attn}.self_attn.o_proj"))?, 16),
        (
            "mlp_inputs",
            group(&[format!("{gdn}.mlp.gate_proj"), format!("{gdn}.mlp.up_proj")])?,
            64,
        ),
        ("mlp_down", single(&format!("{gdn}.mlp.down_proj"))?, 64),
    ];
    let shapes = [(64, 32), (32, 64), (64, 64), (32, 32)];
    let mut report = serde_json::json!({"status": "running", "rows": []});
    let mut chunk = [0f64; 4];
    let mut identical_all = true;
    for (label, module, calls) in &modules {
        let (input, ..) = module.dims();
        for m in [73i32, 256] {
            let x = rows_input(m, input, 171 + m as u32)?;
            let stock = with_dense_prefill_shared(0, || module.tensor(&x))?.to_f16_bits()?;
            let mut row = serde_json::json!({"label": label, "m": m, "values": stock.len()});
            for (index, &shape) in shapes.iter().enumerate() {
                let bits = with_dense_prefill_shared(4, || {
                    with_prefill_tile(shape, || module.tensor(&x))
                })?
                .to_f16_bits()?;
                let differing = bits.iter().zip(&stock).filter(|(a, b)| a != b).count()
                    + bits.len().abs_diff(stock.len());
                identical_all &= differing == 0;
                row[format!("{}x{}_differing", shape.0, shape.1)] = differing.into();
                if m == 256 {
                    let a = with_dense_prefill_shared(4, || {
                        with_prefill_tile(shape, || time_calls_n(8, || module.tensor(&x)))
                    })?;
                    let b = with_dense_prefill_shared(4, || {
                        with_prefill_tile(shape, || time_calls_n(8, || module.tensor(&x)))
                    })?;
                    let ms = (a + b) / 2.0;
                    chunk[index] += ms * f64::from(*calls);
                    row[format!("{}x{}_ms", shape.0, shape.1)] = ms.into();
                }
            }
            println!("{row}");
            report["rows"].as_array_mut().context("rows")?.push(row);
        }
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    for (index, shape) in shapes.iter().enumerate() {
        report[format!("chunk_ms_{}x{}", shape.0, shape.1)] = chunk[index].into();
    }
    report["status"] = if identical_all {
        "complete"
    } else {
        "divergent"
    }
    .into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}

/// Q27P-05: light tiles for the shared prefill kernel (16-row blocks keep 16 FP32 accumulators
/// per lane instead of 64), with 8 or 16 row blocks per decoded tile and 16- or 32-deep
/// operations. Outputs must match the stock kernel bit for bit; then time at 256 rows.
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_prefill_light_lab() -> Result<()> {
    use super::{with_dense_prefill_depth, with_dense_prefill_shared, with_prefill_tile};
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    let gdn = "model.language_model.layers.10";
    let attn = "model.language_model.layers.11";
    let single = |name: &str| -> Result<Module> {
        Ok(Module::Single(Exl3Linear::from_checkpoint(
            &checkpoint,
            name,
        )?))
    };
    let group = |names: &[String]| -> Result<Module> {
        Ok(Module::Group(Exl3Group::new(
            names
                .iter()
                .map(|n| Exl3Linear::from_checkpoint(&checkpoint, n))
                .collect::<Result<Vec<_>>>()?,
        )?))
    };
    let modules = [
        (
            "gdn_inputs",
            group(&[
                format!("{gdn}.linear_attn.in_proj_qkv"),
                format!("{gdn}.linear_attn.in_proj_z"),
            ])?,
            48,
        ),
        (
            "gdn_out",
            single(&format!("{gdn}.linear_attn.out_proj"))?,
            48,
        ),
        ("attn_q", single(&format!("{attn}.self_attn.q_proj"))?, 16),
        ("attn_k", single(&format!("{attn}.self_attn.k_proj"))?, 32),
        ("attn_o", single(&format!("{attn}.self_attn.o_proj"))?, 16),
        (
            "mlp_inputs",
            group(&[format!("{gdn}.mlp.gate_proj"), format!("{gdn}.mlp.up_proj")])?,
            64,
        ),
        ("mlp_down", single(&format!("{gdn}.mlp.down_proj"))?, 64),
    ];
    // (rows per threadgroup share, depth, (BM, BN)); the first entry is the current default.
    let configs: [(i32, i32, (i32, i32)); 7] = [
        (4, 16, (64, 32)),
        (16, 16, (16, 32)),
        (16, 32, (16, 32)),
        (8, 32, (16, 32)),
        (8, 32, (32, 32)),
        (8, 16, (32, 32)),
        (16, 32, (16, 64)),
    ];
    let label_of = |c: &(i32, i32, (i32, i32))| format!("g{}_d{}_{}x{}", c.0, c.1, c.2.0, c.2.1);
    let mut report = serde_json::json!({"status": "running", "rows": []});
    let mut chunk = vec![0f64; configs.len()];
    let mut identical_all = true;
    for (label, module, calls) in &modules {
        let (input, ..) = module.dims();
        for m in [24i32, 73, 256] {
            let x = rows_input(m, input, 191 + m as u32)?;
            let stock = with_dense_prefill_shared(0, || module.tensor(&x))?.to_f16_bits()?;
            let mut row = serde_json::json!({"label": label, "m": m, "values": stock.len()});
            for (index, config) in configs.iter().enumerate() {
                let run = || {
                    with_dense_prefill_shared(config.0, || {
                        with_dense_prefill_depth(config.1, || {
                            with_prefill_tile(config.2, || module.tensor(&x))
                        })
                    })
                };
                let bits = run()?.to_f16_bits()?;
                let differing = bits.iter().zip(&stock).filter(|(a, b)| a != b).count()
                    + bits.len().abs_diff(stock.len());
                identical_all &= differing == 0;
                row[format!("{}_differing", label_of(config))] = differing.into();
                if m == 256 {
                    let time = || {
                        with_dense_prefill_shared(config.0, || {
                            with_dense_prefill_depth(config.1, || {
                                with_prefill_tile(config.2, || {
                                    time_calls_n(8, || module.tensor(&x))
                                })
                            })
                        })
                    };
                    let ms = (time()? + time()?) / 2.0;
                    chunk[index] += ms * f64::from(*calls);
                    row[format!("{}_ms", label_of(config))] = ms.into();
                }
            }
            println!("{row}");
            report["rows"].as_array_mut().context("rows")?.push(row);
        }
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    for (index, config) in configs.iter().enumerate() {
        report[format!("chunk_ms_{}", label_of(config))] = chunk[index].into();
    }
    report["status"] = if identical_all {
        "complete"
    } else {
        "divergent"
    }
    .into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}

/// Q27P-06: raw `matmul2d` throughput by orientation over a projection-like K sweep (no decode,
/// no weight loads): prefill orientation (activations left from memory, weights right in
/// registers) against the transposed one (weights left in registers, activations right).
#[test]
#[ignore = "M5 TensorOps; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_tensor_op_orientation_lab() -> Result<()> {
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    ensure!(array::supports_exl3_tensor_ops()?, "requires M5 TensorOps");
    let prefill = r#"
uint lane = thread_index_in_simdgroup;
uint group = threadgroup_position_in_grid.x;
device half* activation = const_cast<device half*>(x);
constexpr auto descriptor = tensor_ops::matmul2d_descriptor(
    OP_M, OP_N, OP_K, false, false, false,
    tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
tensor_ops::matmul2d<descriptor, execution_simdgroup> operation;
auto right = operation.get_right_input_cooperative_tensor<half, half, float>();
for (ushort i = 0; i < right.get_capacity(); ++i) {
    if (right.is_valid_element(i)) right[i] = half(float(lane * 7u + i) * 1e-3f);
}
using right_type = tensor_ops::matmul2d<descriptor, execution_simdgroup>
    ::cooperative_tensor_right_input_t<half, half, float>;
auto acc = operation.get_destination_cooperative_tensor<
    tensor<device half, dextents<int, 2>, tensor_inline>, right_type, float>();
for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) acc[i] = 0.0f;
}
for (uint depth = 0u; depth < OP_INPUT; depth += OP_K) {
    auto left = tensor(activation + depth, dextents<int, 2>{int(OP_K), int(OP_M)},
        array<int, 2>{1, int(OP_INPUT)});
    operation.run(left, right, acc);
}
float total = 0.0f;
for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) total += acc[i];
}
out[group * 32u + lane] = total;
"#;
    let transposed = r#"
uint lane = thread_index_in_simdgroup;
uint group = threadgroup_position_in_grid.x;
device half* activation = const_cast<device half*>(x);
constexpr auto descriptor = tensor_ops::matmul2d_descriptor(
    OP_M, OP_N, OP_K, false, true, false,
    tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
tensor_ops::matmul2d<descriptor, execution_simdgroup> operation;
using left_type = tensor_ops::matmul2d<descriptor, execution_simdgroup>
    ::cooperative_tensor_left_input_t<half, half, float>;
auto left = operation.get_left_input_cooperative_tensor<half, half, float>();
for (ushort i = 0; i < left.get_capacity(); ++i) {
    if (left.is_valid_element(i)) left[i] = half(float(lane * 7u + i) * 1e-3f);
}
auto acc = operation.get_destination_cooperative_tensor<
    left_type, tensor<device half, dextents<int, 2>, tensor_inline>, float>();
for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) acc[i] = 0.0f;
}
for (uint depth = 0u; depth < OP_INPUT; depth += OP_K) {
    auto right = tensor(activation + depth, dextents<int, 2>{int(OP_K), int(OP_N)},
        array<int, 2>{1, int(OP_INPUT)});
    operation.run(left, right, acc);
}
float total = 0.0f;
for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) total += acc[i];
}
out[group * 32u + lane] = total;
"#;
    let (groups, input) = (2048i32, 5120u32);
    let mut report =
        serde_json::json!({"status": "running", "groups": groups, "input": input, "rows": []});
    // (orientation, M, N, K): prefill = (rows, columns, depth); transposed = (columns, rows, depth).
    let cases: [(&str, u32, u32, u32); 11] = [
        ("prefill", 64, 32, 16),
        ("prefill", 64, 32, 32),
        ("prefill", 32, 32, 32),
        ("prefill", 32, 32, 16),
        ("prefill", 16, 32, 32),
        ("prefill", 32, 64, 32),
        ("transposed", 32, 16, 32),
        ("transposed", 32, 32, 32),
        ("transposed", 32, 64, 32),
        ("transposed", 64, 32, 32),
        ("transposed", 64, 64, 32),
    ];
    for (orientation, m, n, k) in cases {
        let rows = if orientation == "prefill" { m } else { n };
        let x = Array::from_f32(
            &(0..rows * input)
                .map(|i| ((i % 251) as f32) * 1e-3)
                .collect::<Vec<_>>(),
            &[i32::try_from(rows * input)?],
        )?
        .astype(Dtype::Float16)?;
        x.eval()?;
        let header = format!(
            "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\nusing namespace mpp;\n#define OP_M {m}u\n#define OP_N {n}u\n#define OP_K {k}u\n#define OP_INPUT {input}u\n"
        );
        let name = format!("q27lab_op_orientation_{orientation}_m{m}_n{n}_k{k}");
        let source = if orientation == "prefill" {
            prefill
        } else {
            transposed
        };
        let call = || -> Result<Array> {
            Ok(array::metal_kernel(
                &name,
                &["x"],
                &["out"],
                &header,
                source,
                &[&x],
                &[vec![groups * 32]],
                &[Dtype::Float32],
                [groups * 32, 1, 1],
                [32, 1, 1],
            )?
            .remove(0))
        };
        let values = call()?.to_f32()?;
        let finite = values.iter().all(|v| v.is_finite()) && values.iter().any(|&v| v != 0.0);
        let elapsed = (time_calls(call)? + time_calls(call)?) / 2.0;
        let macs = f64::from(groups) * f64::from(input) * f64::from(m * n);
        let row = serde_json::json!({"orientation": orientation, "m": m, "n": n, "k": k, "ms": elapsed,
            "tflops": 2.0 * macs / (elapsed * 1e-3) / 1e12, "finite_nonzero": finite});
        println!("{row}");
        report["rows"].as_array_mut().context("rows")?.push(row);
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    report["status"] = "complete".into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

/// Q27P-07: window-decode prefill kernel with the weight tile read by the unit from threadgroup
/// memory; outputs must match the stock prefill kernel bit for bit, then time at 256 rows.
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_prefill_window_lab() -> Result<()> {
    use super::{
        with_dense_prefill_shared, with_dense_prefill_step_tiles, with_dense_prefill_window,
    };
    // Q27P-07b: with MLXL3_Q27_LAB_STEP_TILES the three window columns are 2, 4 and 8 depth
    // tiles per barrier at group 4 instead of groups 2, 4 and 8 at one tile.
    let steps_lab = std::env::var("MLXL3_Q27_LAB_STEP_TILES").is_ok();
    let variant = |slot: i32| if steps_lab { (4, slot) } else { (slot, 1) };
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    let gdn = "model.language_model.layers.10";
    let attn = "model.language_model.layers.11";
    let single = |name: &str| -> Result<Module> {
        Ok(Module::Single(Exl3Linear::from_checkpoint(
            &checkpoint,
            name,
        )?))
    };
    let group = |names: &[String]| -> Result<Module> {
        Ok(Module::Group(Exl3Group::new(
            names
                .iter()
                .map(|n| Exl3Linear::from_checkpoint(&checkpoint, n))
                .collect::<Result<Vec<_>>>()?,
        )?))
    };
    let modules = [
        (
            "gdn_inputs",
            group(&[
                format!("{gdn}.linear_attn.in_proj_qkv"),
                format!("{gdn}.linear_attn.in_proj_z"),
            ])?,
            48,
        ),
        (
            "gdn_out",
            single(&format!("{gdn}.linear_attn.out_proj"))?,
            48,
        ),
        ("attn_q", single(&format!("{attn}.self_attn.q_proj"))?, 16),
        ("attn_k", single(&format!("{attn}.self_attn.k_proj"))?, 32),
        ("attn_o", single(&format!("{attn}.self_attn.o_proj"))?, 16),
        (
            "mlp_inputs",
            group(&[format!("{gdn}.mlp.gate_proj"), format!("{gdn}.mlp.up_proj")])?,
            64,
        ),
        ("mlp_down", single(&format!("{gdn}.mlp.down_proj"))?, 64),
        ("head", single("lm_head")?, 0),
    ];
    let mut report = serde_json::json!({"status": "running", "rows": []});
    let (mut identical_all, mut compared) = (true, 0usize);
    let mut chunk = [0f64; 4];
    for (label, module, calls) in &modules {
        let (input, cols, ..) = module.dims();
        let sizes: &[i32] = if *label == "head" {
            &[24, 73]
        } else {
            &[24, 73, 128, 200, 256, 512]
        };
        for &m in sizes {
            let x = rows_input(m, input, 211 + m as u32)?;
            let stock = with_dense_prefill_shared(0, || module.tensor(&x))?.to_f16_bits()?;
            ensure!(stock.len() == (m * cols) as usize, "short stock output");
            let finite = stock.iter().all(|&v| half::f16::from_bits(v).is_finite());
            let mut row = serde_json::json!({"label": label, "m": m, "values": stock.len(), "finite": finite});
            for g in [2, 4, 8] {
                let (rows, tiles) = variant(g);
                let bits = with_dense_prefill_shared(rows, || {
                    with_dense_prefill_step_tiles(tiles, || {
                        with_dense_prefill_window(true, || module.tensor(&x))
                    })
                })?
                .to_f16_bits()?;
                let differing = bits.iter().zip(&stock).filter(|(a, b)| a != b).count()
                    + bits.len().abs_diff(stock.len());
                identical_all &= differing == 0 && finite;
                compared += bits.len();
                row[format!("w{g}_differing")] = differing.into();
            }
            if m == 256 {
                // shared g4 (current default), then window g2 / g4 / g8, two alternated series.
                let mut times = [0f64; 4];
                for _ in 0..2 {
                    times[0] += with_dense_prefill_shared(4, || {
                        with_dense_prefill_window(false, || time_calls_n(8, || module.tensor(&x)))
                    })? / 2.0;
                    for (slot, g) in [(1usize, 2), (2, 4), (3, 8)] {
                        let (rows, tiles) = variant(g);
                        times[slot] += with_dense_prefill_shared(rows, || {
                            with_dense_prefill_step_tiles(tiles, || {
                                with_dense_prefill_window(true, || {
                                    time_calls_n(8, || module.tensor(&x))
                                })
                            })
                        })? / 2.0;
                    }
                }
                for (slot, t) in times.iter().enumerate() {
                    chunk[slot] += t * f64::from(*calls);
                }
                row["shared4_ms"] = times[0].into();
                row["window2_ms"] = times[1].into();
                row["window4_ms"] = times[2].into();
                row["window8_ms"] = times[3].into();
                row["window4_tflops"] = (2.0 * f64::from(m) * f64::from(input) * f64::from(cols)
                    / (times[2] * 1e-3)
                    / 1e12)
                    .into();
            }
            println!("{row}");
            report["rows"].as_array_mut().context("rows")?.push(row);
        }
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    report["chunk_ms"] = serde_json::json!({"shared4": chunk[0], "window2": chunk[1], "window4": chunk[2], "window8": chunk[3]});
    report["step_tiles_lab"] = steps_lab.into();
    report["values_compared"] = compared.into();
    report["status"] = if identical_all {
        "complete"
    } else {
        "divergent"
    }
    .into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    ensure!(
        identical_all,
        "window prefill kernel diverged from the stock kernel"
    );
    Ok(())
}

/// Q27P-08: share of the MLX prologue and epilogue chains in a 256-row prefill projection.
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_prefill_glue_lab() -> Result<()> {
    use super::with_dense_prefill_shared;
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    let gdn = "model.language_model.layers.10";
    let attn = "model.language_model.layers.11";
    let single = |name: &str| -> Result<Module> {
        Ok(Module::Single(Exl3Linear::from_checkpoint(
            &checkpoint,
            name,
        )?))
    };
    let group = |names: &[String]| -> Result<Module> {
        Ok(Module::Group(Exl3Group::new(
            names
                .iter()
                .map(|n| Exl3Linear::from_checkpoint(&checkpoint, n))
                .collect::<Result<Vec<_>>>()?,
        )?))
    };
    let modules = [
        (
            "gdn_inputs",
            group(&[
                format!("{gdn}.linear_attn.in_proj_qkv"),
                format!("{gdn}.linear_attn.in_proj_z"),
            ])?,
            48,
        ),
        (
            "gdn_out",
            single(&format!("{gdn}.linear_attn.out_proj"))?,
            48,
        ),
        ("attn_q", single(&format!("{attn}.self_attn.q_proj"))?, 16),
        ("attn_k", single(&format!("{attn}.self_attn.k_proj"))?, 32),
        ("attn_o", single(&format!("{attn}.self_attn.o_proj"))?, 16),
        (
            "mlp_inputs",
            group(&[format!("{gdn}.mlp.gate_proj"), format!("{gdn}.mlp.up_proj")])?,
            64,
        ),
        ("mlp_down", single(&format!("{gdn}.mlp.down_proj"))?, 64),
    ];
    let m = 256i32;
    let scale = Some(1.0 / 128.0f32.sqrt());
    let mut report = serde_json::json!({"status": "running", "rows": []});
    let (mut full_chunk, mut glue_chunk) = (0.0, 0.0);
    for (label, module, calls) in &modules {
        let (input, cols, ..) = module.dims();
        let x = rows_input(m, input, 231)?;
        // (suh, svh, width) per member, as `qmm_tensor` receives them.
        let members: Vec<(Array, Array, i32)> = match module {
            Module::Single(l) => vec![(l.suh.try_clone()?, l.svh.try_clone()?, l.cols)],
            Module::Group(g) => {
                let mut out = Vec::new();
                let mut cursor = 0;
                for (index, &width) in g.widths.iter().enumerate() {
                    out.push((
                        g.suh
                            .slice(0, index as i32, index as i32 + 1)?
                            .reshape(&[g.rows])?,
                        g.svh.slice(0, cursor, cursor + width)?,
                        width,
                    ));
                    cursor += width;
                }
                out
            }
        };
        let raws = members
            .iter()
            .map(|(_, _, width)| rows_input(m, *width, 233))
            .collect::<Result<Vec<_>>>()?;
        let full = with_dense_prefill_shared(4, || time_calls_n(8, || module.tensor(&x)))?;
        let prologue = time_calls_n(8, || {
            let parts = members
                .iter()
                .map(|(suh, _, _)| {
                    x.astype(Dtype::Float16)?
                        .reshape(&[m, input])?
                        .mul(&suh.astype(Dtype::Float16)?)?
                        .reshape(&[m, input / 128, 128])?
                        .hadamard_transform(scale)?
                        .reshape(&[m, input])
                })
                .collect::<Result<Vec<_>>>()?;
            Array::concatenate(&parts.iter().collect::<Vec<_>>(), 1)
        })?;
        let epilogue = time_calls_n(8, || {
            let parts = members
                .iter()
                .zip(&raws)
                .map(|((_, svh, width), raw)| {
                    raw.slice(0, 0, m)?
                        .reshape(&[m, width / 128, 128])?
                        .hadamard_transform(scale)?
                        .reshape(&[m, *width])?
                        .mul(&svh.astype(Dtype::Float16)?)?
                        .astype(x.dtype())
                })
                .collect::<Result<Vec<_>>>()?;
            Array::concatenate(&parts.iter().collect::<Vec<_>>(), 1)
        })?;
        full_chunk += full * f64::from(*calls);
        glue_chunk += (prologue + epilogue) * f64::from(*calls);
        let kernel = full - prologue - epilogue;
        let row = serde_json::json!({"label": label, "input": input, "output": cols, "full_ms": full,
            "prologue_ms": prologue, "epilogue_ms": epilogue, "glue_share": (prologue + epilogue) / full,
            "kernel_tflops_estimate": 2.0 * f64::from(m) * f64::from(input) * f64::from(cols) / (kernel * 1e-3) / 1e12});
        println!("{row}");
        report["rows"].as_array_mut().context("rows")?.push(row);
    }
    report["chunk_full_ms"] = full_chunk.into();
    report["chunk_glue_ms"] = glue_chunk.into();
    report["status"] = "complete".into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}

/// Q27P-09: which structural element of the prefill kernel costs throughput (no decode):
/// simdgroups per threadgroup, distinct activation rows, a barrier per step, and a right
/// operand in threadgroup memory instead of cooperative registers.
#[test]
#[ignore = "M5 TensorOps; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_tensor_op_structure_lab() -> Result<()> {
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    ensure!(array::supports_exl3_tensor_ops()?, "requires M5 TensorOps");
    let source = r#"
uint lane = thread_index_in_simdgroup;
uint simdgroup = simdgroup_index_in_threadgroup;
uint group = threadgroup_position_in_grid.x;
uint row_block = (group * OP_SG + simdgroup) % (OP_ROWS / OP_M);
device half* activation = const_cast<device half*>(x);
constexpr auto descriptor = tensor_ops::matmul2d_descriptor(
    OP_M, OP_N, OP_K, false, false, false,
    tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
tensor_ops::matmul2d<descriptor, execution_simdgroup> operation;
using left_type = tensor<device half, dextents<int, 2>, tensor_inline>;
#if OP_TG_RIGHT
threadgroup half tile[OP_K * OP_N];
for (uint e = thread_index_in_threadgroup; e < OP_K * OP_N; e += 32u * OP_SG) {
    tile[e] = half(float(e % 97u) * 1e-3f);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
using right_type = tensor<threadgroup half, dextents<int, 2>, tensor_inline>;
auto right = tensor(tile, dextents<int, 2>{int(OP_N), int(OP_K)}, array<int, 2>{1, int(OP_N)});
#else
auto right = operation.get_right_input_cooperative_tensor<half, half, float>();
for (ushort i = 0; i < right.get_capacity(); ++i) {
    if (right.is_valid_element(i)) right[i] = half(float(lane * 7u + i) * 1e-3f);
}
using right_type = tensor_ops::matmul2d<descriptor, execution_simdgroup>
    ::cooperative_tensor_right_input_t<half, half, float>;
#endif
auto acc = operation.get_destination_cooperative_tensor<left_type, right_type, float>();
for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) acc[i] = 0.0f;
}
for (uint depth = 0u; depth < OP_INPUT; depth += OP_K) {
#if OP_BARRIER
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    auto left = tensor(activation + ulong(row_block * OP_M) * OP_INPUT + depth,
        dextents<int, 2>{int(OP_K), int(OP_M)}, array<int, 2>{1, int(OP_INPUT)});
    operation.run(left, right, acc);
}
float total = 0.0f;
for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) total += acc[i];
}
out[(group * OP_SG + simdgroup) * 32u + lane] = total;
"#;
    let (m, n, k, input, simdgroups) = (64u32, 32u32, 16u32, 5120u32, 2048i32);
    let mut report = serde_json::json!({"status": "running", "rows": []});
    // (simdgroups per threadgroup, distinct rows, barrier, threadgroup right)
    let cases: [(i32, u32, u8, u8); 9] = [
        (1, 64, 0, 0),
        (1, 256, 0, 0),
        (1, 1024, 0, 0),
        (4, 64, 0, 0),
        (4, 256, 0, 0),
        (4, 256, 1, 0),
        (4, 256, 0, 1),
        (4, 256, 1, 1),
        (1, 256, 0, 1),
    ];
    for (sg, rows, barrier, tg_right) in cases {
        let x = Array::from_f32(
            &(0..rows * input)
                .map(|i| ((i % 251) as f32) * 1e-3)
                .collect::<Vec<_>>(),
            &[i32::try_from(rows * input)?],
        )?
        .astype(Dtype::Float16)?;
        x.eval()?;
        let header = format!(
            "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\nusing namespace mpp;\n#define OP_M {m}u\n#define OP_N {n}u\n#define OP_K {k}u\n#define OP_INPUT {input}u\n#define OP_SG {sg}u\n#define OP_ROWS {rows}u\n#define OP_BARRIER {barrier}\n#define OP_TG_RIGHT {tg_right}\n"
        );
        let name = format!("q27lab_op_structure_sg{sg}_r{rows}_b{barrier}_t{tg_right}");
        let call = || -> Result<Array> {
            Ok(array::metal_kernel(
                &name,
                &["x"],
                &["out"],
                &header,
                source,
                &[&x],
                &[vec![simdgroups * 32]],
                &[Dtype::Float32],
                [simdgroups * 32, 1, 1],
                [32 * sg, 1, 1],
            )?
            .remove(0))
        };
        let values = call()?.to_f32()?;
        let finite = values.iter().all(|v| v.is_finite()) && values.iter().any(|&v| v != 0.0);
        let elapsed = (time_calls(call)? + time_calls(call)?) / 2.0;
        let macs = f64::from(simdgroups) * f64::from(input) * f64::from(m * n);
        let row = serde_json::json!({"simdgroups_per_group": sg, "rows": rows, "barrier": barrier, "threadgroup_right": tg_right,
            "ms": elapsed, "tflops": 2.0 * macs / (elapsed * 1e-3) / 1e12, "finite_nonzero": finite});
        println!("{row}");
        report["rows"].as_array_mut().context("rows")?.push(row);
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    report["status"] = "complete".into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

/// Q27P-11: fused prologue/epilogue on the prefill TensorOps path must reproduce the MLX
/// chains bit for bit (same shared kernel on both sides); then time at 256 rows.
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_prefill_fused_glue_lab() -> Result<()> {
    use super::{with_dense_prefill_fused, with_dense_prefill_shared};
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    let gdn = "model.language_model.layers.10";
    let attn = "model.language_model.layers.11";
    let single = |name: &str| -> Result<Module> {
        Ok(Module::Single(Exl3Linear::from_checkpoint(
            &checkpoint,
            name,
        )?))
    };
    let group = |names: &[String]| -> Result<Module> {
        Ok(Module::Group(Exl3Group::new(
            names
                .iter()
                .map(|n| Exl3Linear::from_checkpoint(&checkpoint, n))
                .collect::<Result<Vec<_>>>()?,
        )?))
    };
    let modules = [
        (
            "gdn_inputs",
            group(&[
                format!("{gdn}.linear_attn.in_proj_qkv"),
                format!("{gdn}.linear_attn.in_proj_z"),
            ])?,
            48,
        ),
        (
            "gdn_out",
            single(&format!("{gdn}.linear_attn.out_proj"))?,
            48,
        ),
        ("attn_q", single(&format!("{attn}.self_attn.q_proj"))?, 16),
        ("attn_k", single(&format!("{attn}.self_attn.k_proj"))?, 32),
        ("attn_o", single(&format!("{attn}.self_attn.o_proj"))?, 16),
        (
            "mlp_inputs",
            group(&[format!("{gdn}.mlp.gate_proj"), format!("{gdn}.mlp.up_proj")])?,
            64,
        ),
        ("mlp_down", single(&format!("{gdn}.mlp.down_proj"))?, 64),
        ("head", single("lm_head")?, 0),
    ];
    let mut report = serde_json::json!({"status": "running", "rows": []});
    let (mut identical_all, mut compared) = (true, 0usize);
    let mut chunk = [0f64; 2];
    for (label, module, calls) in &modules {
        let (input, cols, ..) = module.dims();
        let sizes: &[i32] = if *label == "head" {
            &[24, 73]
        } else {
            &[24, 73, 128, 200, 256, 512]
        };
        for &m in sizes {
            let mut row = serde_json::json!({"label": label, "m": m});
            for dtype in [Dtype::Float16, Dtype::BFloat16] {
                let x = rows_input(m, input, 271 + m as u32)?.astype(dtype)?;
                x.eval()?;
                let run = |fused: bool| {
                    with_dense_prefill_shared(4, || {
                        with_dense_prefill_fused(fused, || module.tensor(&x))
                    })
                };
                let (stock, fused) = (run(false)?, run(true)?);
                let (a, b) = (
                    stock.astype(Dtype::Float32)?.to_f32()?,
                    fused.astype(Dtype::Float32)?.to_f32()?,
                );
                let differing = a
                    .iter()
                    .zip(&b)
                    .filter(|(p, q)| p.to_bits() != q.to_bits())
                    .count()
                    + a.len().abs_diff(b.len());
                let ok = stock.dtype() == fused.dtype()
                    && stock.shape() == fused.shape()
                    && a.len() == (m * cols) as usize
                    && a.iter().all(|v| v.is_finite())
                    && differing == 0;
                identical_all &= ok;
                compared += a.len();
                row[format!("{dtype:?}_differing")] = differing.into();
                if m == 256 && dtype == Dtype::Float16 {
                    let mut times = [0f64; 2];
                    for _ in 0..2 {
                        times[0] += with_dense_prefill_shared(4, || {
                            with_dense_prefill_fused(false, || {
                                time_calls_n(8, || module.tensor(&x))
                            })
                        })? / 2.0;
                        times[1] += with_dense_prefill_shared(4, || {
                            with_dense_prefill_fused(true, || time_calls_n(8, || module.tensor(&x)))
                        })? / 2.0;
                    }
                    chunk[0] += times[0] * f64::from(*calls);
                    chunk[1] += times[1] * f64::from(*calls);
                    row["mlx_ms"] = times[0].into();
                    row["fused_ms"] = times[1].into();
                }
            }
            println!("{row}");
            report["rows"].as_array_mut().context("rows")?.push(row);
        }
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    report["chunk_ms"] = serde_json::json!({"mlx": chunk[0], "fused": chunk[1]});
    report["values_compared"] = compared.into();
    report["status"] = if identical_all {
        "complete"
    } else {
        "divergent"
    }
    .into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    ensure!(
        identical_all,
        "fused prefill glue diverged from the MLX chains"
    );
    Ok(())
}

/// Q27P-12: what the window prefill kernel costs without its per-step decode (ablation with
/// wrong outputs), fused glue on both sides, against the estimate at the unit's ceiling.
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_KERNEL_LAB; run alone"]
fn q27_prefill_decode_ablation_lab() -> Result<()> {
    use super::{
        with_dense_prefill_fused, with_dense_prefill_shared, with_dense_prefill_step_tiles,
        with_dense_prefill_window, with_prefill_skip_decode,
    };
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_KERNEL_LAB")?);
    ensure!(!output.exists(), "preserve previous kernel lab reports");
    let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_TEST_MODEL",
    )?))?;
    let gdn = "model.language_model.layers.10";
    let modules = [
        (
            "mlp_inputs",
            Module::Group(Exl3Group::new(vec![
                Exl3Linear::from_checkpoint(&checkpoint, &format!("{gdn}.mlp.gate_proj"))?,
                Exl3Linear::from_checkpoint(&checkpoint, &format!("{gdn}.mlp.up_proj"))?,
            ])?),
        ),
        (
            "mlp_down",
            Module::Single(Exl3Linear::from_checkpoint(
                &checkpoint,
                &format!("{gdn}.mlp.down_proj"),
            )?),
        ),
        (
            "gdn_out",
            Module::Single(Exl3Linear::from_checkpoint(
                &checkpoint,
                &format!("{gdn}.linear_attn.out_proj"),
            )?),
        ),
    ];
    let mut report = serde_json::json!({"status": "running", "rows": []});
    for (label, module) in &modules {
        let (input, cols, ..) = module.dims();
        let x = rows_input(256, input, 291)?;
        let base = |tiles: i32, skip: bool| -> Result<f64> {
            with_dense_prefill_fused(true, || {
                with_dense_prefill_shared(4, || {
                    with_dense_prefill_window(true, || {
                        with_dense_prefill_step_tiles(tiles, || {
                            if skip {
                                with_prefill_skip_decode(|| time_calls_n(8, || module.tensor(&x)))
                            } else {
                                time_calls_n(8, || module.tensor(&x))
                            }
                        })
                    })
                })
            })
        };
        let mut row = serde_json::json!({"label": label,
            "ceiling_ms_at_15_5_tflops": 2.0 * 256.0 * f64::from(input) * f64::from(cols) / 15.5e12 * 1e3});
        for tiles in [1, 4] {
            let (full, skipped) = (
                (base(tiles, false)? + base(tiles, false)?) / 2.0,
                (base(tiles, true)? + base(tiles, true)?) / 2.0,
            );
            row[format!("t{tiles}_ms")] = full.into();
            row[format!("t{tiles}_nodecode_ms")] = skipped.into();
        }
        println!("{row}");
        report["rows"].as_array_mut().context("rows")?.push(row);
    }
    report["status"] = "complete".into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
