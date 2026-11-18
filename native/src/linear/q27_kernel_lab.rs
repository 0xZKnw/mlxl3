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
