use super::{apply, kernel_requests_for_test, last_rows_for_test, reset_kernel_requests_for_test};
use crate::array::{Array, Dtype};
use anyhow::{Context, Result, ensure};
use half::f16;

const WIDTH: usize = 6144;
const ROW_SHAPE: [i32; 4] = [1, 1, 48, 128];

// This follows the uncompiled production graph, including the per-row F16 boundary.
pub(super) fn stock(normed: &[Array], z: &Array) -> Result<Array> {
    ensure!(!normed.is_empty() && normed.len() <= 8);
    let rows = normed.len() as i32;
    let mut gated = Vec::with_capacity(normed.len());
    for (row, value) in normed.iter().enumerate() {
        let gate = if rows == 1 {
            z.try_clone()?
        } else {
            z.slice(1, row as i32, row as i32 + 1)?
        };
        gated.push(
            value
                .astype(Dtype::Float32)?
                .mul(&gate.astype(Dtype::Float32)?.sigmoid()?)?
                .astype(Dtype::Float16)?,
        );
    }
    if rows == 1 {
        gated
            .pop()
            .expect("one gated row")
            .reshape(&[1, WIDTH as i32])
    } else {
        Array::concatenate(&gated.iter().collect::<Vec<_>>(), 1)?.reshape(&[rows, WIDTH as i32])
    }
}

fn snapshot(normed: &[Array], z: &Array) -> Result<Vec<Vec<u8>>> {
    normed
        .iter()
        .chain(std::iter::once(z))
        .map(Array::to_bytes)
        .collect()
}

fn check(normed: &[Array], z: &Array, finite: bool) -> Result<Vec<u8>> {
    let rows = normed.len();
    let before = snapshot(normed, z)?;
    reset_kernel_requests_for_test();
    let readbacks = Array::host_readbacks_for_test();
    let actual = apply(normed, z)?;
    ensure!(Array::host_readbacks_for_test() == readbacks);
    ensure!(
        kernel_requests_for_test() == 1 && last_rows_for_test() == rows,
        "GDN epilogue kernel request path was not exercised once"
    );
    let reference = stock(normed, z)?;
    for value in [&actual, &reference] {
        ensure!(
            value.shape() == [rows as i32, WIDTH as i32] && value.dtype() == Dtype::Float16,
            "GDN epilogue changed output metadata"
        );
        let values = value.to_f32()?;
        ensure!(
            values.len() == rows * WIDTH,
            "empty or incomplete GDN output"
        );
        if finite {
            ensure!(values.iter().all(|v| v.is_finite()), "nonfinite GDN output");
        }
    }
    let bytes = actual.to_bytes()?;
    let expected = reference.to_bytes()?;
    ensure!(bytes.len() == rows * WIDTH * 2 && expected.len() == bytes.len());
    if finite {
        if bytes != expected {
            let index = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .zip(expected.as_chunks::<2>().0.iter())
                .position(|(a, e)| a != e)
                .context("missing first GDN mismatch")?;
            let row = index / WIDTH;
            let column = index % WIDTH;
            let norm = f32::from_ne_bytes(
                before[row][column * 4..column * 4 + 4]
                    .try_into()
                    .context("invalid normalized diagnostic bytes")?,
            );
            let gate_bits =
                u16::from_ne_bytes([before[rows][index * 2], before[rows][index * 2 + 1]]);
            let actual_bits = u16::from_ne_bytes([bytes[index * 2], bytes[index * 2 + 1]]);
            let expected_bits = u16::from_ne_bytes([expected[index * 2], expected[index * 2 + 1]]);
            let gate = if rows == 1 {
                z.try_clone()?
            } else {
                z.slice(1, row as i32, row as i32 + 1)?
            };
            // This extra stock readback is diagnostic-only, after parity has already failed.
            let probabilities = gate.astype(Dtype::Float32)?.sigmoid()?.to_f32()?;
            ensure!(
                probabilities.len() == WIDTH,
                "incomplete sigmoid diagnostic"
            );
            let probability = probabilities[column];
            eprintln!(
                "GDN_EPILOGUE_FIRST_MISMATCH rows={rows} index={index} row={row} column={column} norm_f32_bits=0x{:08x} norm_f32={norm:?} gate_f16_bits=0x{gate_bits:04x} gate_f32={:?} actual_f16_bits=0x{actual_bits:04x} actual_f32={:?} stock_f16_bits=0x{expected_bits:04x} stock_f32={:?} stock_sigmoid_f32_bits=0x{:08x} stock_sigmoid_f32={probability:?}",
                norm.to_bits(),
                f16::from_bits(gate_bits).to_f32(),
                f16::from_bits(actual_bits).to_f32(),
                f16::from_bits(expected_bits).to_f32(),
                probability.to_bits(),
            );
        }
        ensure!(
            bytes == expected,
            "GDN epilogue differs from stock MLX oracle"
        );
    } else {
        for (actual, expected) in bytes
            .as_chunks::<2>()
            .0
            .iter()
            .zip(expected.as_chunks::<2>().0.iter())
        {
            let a = u16::from_ne_bytes([actual[0], actual[1]]);
            let e = u16::from_ne_bytes([expected[0], expected[1]]);
            ensure!(
                a == e || (f16::from_bits(a).is_nan() && f16::from_bits(e).is_nan()),
                "GDN epilogue differs from nonfinite stock oracle"
            );
        }
    }
    ensure!(
        snapshot(normed, z)? == before,
        "GDN epilogue mutated an input alias"
    );
    Ok(bytes)
}

fn normalized(values: &[f32]) -> Result<Vec<Array>> {
    ensure!(!values.is_empty() && values.len().is_multiple_of(WIDTH));
    values
        .as_chunks::<WIDTH>()
        .0
        .iter()
        .map(|row| Array::from_f32(row, &ROW_SHAPE))
        .collect()
}

fn half_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|v| f16::from_f32(v).to_bits().to_ne_bytes())
        .collect()
}

#[test]
#[ignore = "physical Apple GPU; first T17 correctness smoke B1/B2, run alone"]
fn smoke_b1_b2_fp32_norms_match_stock_and_independent_zero_gate_literals() -> Result<()> {
    let anchors = [
        1.0,
        -1.0,
        0.0,
        -0.0,
        f32::from_bits(0x3f80_0fff),
        f32::from_bits(0x3f80_1000),
        f32::from_bits(0x3f80_1001),
        1.0 / 3.0,
        -1.0003,
        3.0003,
        65504.0,
        -65504.0,
        f32::from_bits(0x3900_0000),
        -f32::from_bits(0x3900_0000),
        0.75,
        -2.5,
    ];
    ensure!(
        anchors.iter().any(|&v| f16::from_f32(v).to_f32() != v),
        "smoke must exercise genuinely FP32 normalized values"
    );
    let gate_bits = [
        0x0000, 0x8000, 0x3c00, 0xbc00, 0x4000, 0xc000, 0x3400, 0xb400, 0x7bff, 0xfbff, 0x0001,
        0x8001, 0x3bff, 0xbbff, 0x3c01, 0xbc01,
    ];
    let mut cases = 0;
    let mut values_checked = 0;
    for rows in [1usize, 2] {
        let values = (0..rows * WIDTH)
            .map(|i| anchors[(i + i / WIDTH * 3) % anchors.len()])
            .collect::<Vec<_>>();
        let normed = normalized(&values)?;
        let zeros = (0..rows * WIDTH)
            .map(|i| if i % 2 == 0 { 0x0000 } else { 0x8000 })
            .collect::<Vec<_>>();
        let z = Array::from_f16_bits(&zeros, &[1, rows as i32, 48, 128])?;
        let bytes = check(&normed, &z, true)?;
        ensure!(
            bytes == half_bytes(values.iter().map(|&v| v * 0.5)),
            "GDN epilogue differs from independent zero-gate literal oracle"
        );
        cases += 1;
        values_checked += rows * WIDTH;

        let gates = (0..rows * WIDTH)
            .map(|i| gate_bits[(i * 7 + i / WIDTH) % gate_bits.len()])
            .collect::<Vec<_>>();
        let z = Array::from_f16_bits(&gates, &[1, rows as i32, 48, 128])?;
        let bytes = check(&normed, &z, true)?;
        ensure!(
            bytes != half_bytes(values),
            "smoke fixture cannot detect omission of the sigmoid gate"
        );
        cases += 1;
        values_checked += rows * WIDTH;
    }
    ensure!(cases == 4 && values_checked == 36864);
    println!(
        "GDN_EPILOGUE_SMOKE cases={cases} values={values_checked} rows=1,2 norm=F32 gate=F16 parity=byte_exact"
    );
    Ok(())
}

fn finite_norm_anchors() -> [f32; 16] {
    [
        1.0,
        -1.0,
        0.0,
        -0.0,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(0x3380_0000),
        -f32::from_bits(0x3380_0000),
        65504.0,
        -65504.0,
        f32::from_bits(0x3f80_1000),
        f32::from_bits(0x3f80_0fff),
        f32::from_bits(0x3f80_1001),
        -f32::from_bits(0x3f80_1000),
    ]
}

#[test]
#[ignore = "physical Apple GPU; exhaustive finite F16 gates and FP32 product boundaries, run alone"]
fn exhaustive_finite_half_gates_and_fp32_products_match_stock() -> Result<()> {
    let patterns = (0..=u16::MAX)
        .filter(|&bits| f16::from_bits(bits).is_finite())
        .collect::<Vec<_>>();
    ensure!(patterns.len() == 63488);
    let anchors = finite_norm_anchors();
    let mut cases = 0;
    let mut gate_norm_pairs = 0;
    let mut output_values = 0;
    for rows in [1usize, 2, 4, 8] {
        let n = rows * WIDTH;
        let mut seen = vec![false; usize::from(u16::MAX) + 1];
        for phase in 0..anchors.len() {
            for (batch, chunk) in patterns.chunks(n).enumerate() {
                let gates = (0..n).map(|i| chunk[i % chunk.len()]).collect::<Vec<_>>();
                let values = (0..n)
                    .map(|i| anchors[(batch * n + i + phase) % anchors.len()])
                    .collect::<Vec<_>>();
                let normed = normalized(&values)?;
                let z = Array::from_f16_bits(&gates, &[1, rows as i32, 48, 128])?;
                check(&normed, &z, true).with_context(|| {
                    format!(
                        "GDN exhaustive case rows={rows} phase={phase} batch={batch} real_gate_count={} values={n}",
                        chunk.len()
                    )
                })?;
                for &bits in chunk {
                    seen[usize::from(bits)] = true;
                }
                gate_norm_pairs += chunk.len();
                output_values += n;
                cases += 1;
            }
        }
        ensure!(seen.iter().filter(|&&v| v).count() == 63488);
        for (bits, &visited) in seen.iter().enumerate() {
            ensure!(visited == f16::from_bits(bits as u16).is_finite());
        }
    }
    ensure!(cases == 352 && gate_norm_pairs == 4_063_232 && output_values == 5_013_504);
    println!(
        "GDN_EPILOGUE_EXHAUSTIVE cases={cases} finite_gate_patterns=63488 norm_anchors=16 gate_norm_pairs={gate_norm_pairs} values={output_values} rows=1,2,4,8 parity=byte_exact_stock_ftz"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; all served rows, strided poison padding and immutable aliases, run alone"]
fn rows_strides_and_aliases_preserve_stock_and_sources() -> Result<()> {
    let mut cases = 0;
    for rows in 1usize..=8 {
        for strides in 0..4 {
            for aliases in [false, true] {
                let mut normed: Vec<Array> = Vec::with_capacity(rows);
                let mut owners = Vec::new();
                for row in 0..rows {
                    if aliases && row > 0 {
                        normed.push(normed[0].try_clone()?);
                        continue;
                    }
                    let columns = if strides & 1 == 0 { 128 } else { 256 };
                    let values = (0..48 * columns)
                        .map(|i| {
                            if i % columns >= 128 {
                                f32::NAN
                            } else {
                                ((i * 37 + row * 101) % 4093) as f32 / 257.0 - 7.0
                            }
                        })
                        .collect::<Vec<_>>();
                    let owner = Array::from_f32(&values, &[1, 1, 48, columns as i32])?;
                    normed.push(if strides & 1 == 0 {
                        owner.try_clone()?
                    } else {
                        owner.slice(3, 0, 128)?
                    });
                    owners.push(owner);
                }
                let columns = if strides & 2 == 0 { 128 } else { 256 };
                let gates = (0..rows * 48 * columns)
                    .map(|i| {
                        if i % columns >= 128 {
                            [0x7e11, 0x7c00, 0xfc00][i % 3]
                        } else {
                            f16::from_f32(((i * 13 % 2039) as f32 - 1019.0) / 127.0).to_bits()
                        }
                    })
                    .collect::<Vec<_>>();
                let owner = Array::from_f16_bits(&gates, &[1, rows as i32, 48, columns as i32])?;
                let z = if strides & 2 == 0 {
                    owner.try_clone()?
                } else {
                    owner.slice(3, 0, 128)?
                };
                owners.push(owner);
                let before = owners
                    .iter()
                    .map(Array::to_bytes)
                    .collect::<Result<Vec<_>>>()?;
                check(&normed, &z, true)?;
                ensure!(
                    owners
                        .iter()
                        .map(Array::to_bytes)
                        .collect::<Result<Vec<_>>>()?
                        == before,
                    "candidate mutated underlying storage or poison padding"
                );
                cases += 1;
            }
        }
    }
    ensure!(cases == 64);
    println!(
        "GDN_EPILOGUE_LAYOUTS cases={cases} rows=1..8 stride_masks=4 aliases=2 poison_padding=true parity=byte_exact source_mutation=false"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; infinite gates and nonfinite product classes, run alone"]
fn infinite_gates_and_nonfinite_products_match_stock() -> Result<()> {
    let mut finite_cases = 0;
    let mut nonfinite_cases = 0;
    for rows in [1usize, 2, 4, 8] {
        let values = (0..rows * WIDTH)
            .map(|i| [0.0, -0.0, 1.0, -1.0, 2.5, -2.5, 65504.0, -65504.0][i % 8])
            .collect::<Vec<_>>();
        let gates = (0..rows * WIDTH)
            .map(|i| if i % 2 == 0 { 0x7c00 } else { 0xfc00 })
            .collect::<Vec<_>>();
        let z = Array::from_f16_bits(&gates, &[1, rows as i32, 48, 128])?;
        let bytes = check(&normalized(&values)?, &z, true)?;
        ensure!(
            bytes
                == half_bytes(
                    values
                        .iter()
                        .enumerate()
                        .map(|(i, &value)| value * if i % 2 == 0 { 1.0 } else { 0.0 })
                ),
            "infinite sigmoid gate differs from independent literal limits"
        );
        finite_cases += 1;

        let values = (0..rows * WIDTH)
            .map(|i| {
                [
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::NAN,
                    f32::MAX,
                    -f32::MAX,
                ][i % 5]
            })
            .collect::<Vec<_>>();
        check(&normalized(&values)?, &z, false)?;
        nonfinite_cases += 1;
    }
    ensure!(finite_cases == 4 && nonfinite_cases == 4);
    println!(
        "GDN_EPILOGUE_NONFINITE finite_cases={finite_cases} nonfinite_cases={nonfinite_cases} finite_parity=byte_exact nan_parity=class_only other_parity=byte_exact"
    );
    Ok(())
}

fn reject(normed: &[Array], z: &Array) -> Result<()> {
    let before = snapshot(normed, z)?;
    reset_kernel_requests_for_test();
    let readbacks = Array::host_readbacks_for_test();
    ensure!(
        apply(normed, z).is_err(),
        "invalid GDN metadata was accepted"
    );
    ensure!(
        kernel_requests_for_test() == 0
            && last_rows_for_test() == 0
            && Array::host_readbacks_for_test() == readbacks,
        "invalid GDN metadata requested a kernel or readback"
    );
    ensure!(snapshot(normed, z)? == before, "rejected input was mutated");
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; invalid metadata must reject before a kernel graph request, run alone"]
fn invalid_metadata_rejects_before_kernel_request() -> Result<()> {
    let good = Array::zeros_dtype(&ROW_SHAPE, Dtype::Float32)?;
    let z = Array::zeros_dtype(&ROW_SHAPE, Dtype::Float16)?;
    let mut cases = 0;
    for rows in [0usize, 9] {
        let normed = (0..rows)
            .map(|_| good.try_clone())
            .collect::<Result<Vec<_>>>()?;
        reject(&normed, &z)?;
        cases += 1;
    }
    for shape in [
        vec![],
        vec![6144],
        vec![1, 6144],
        vec![1, 48, 128],
        vec![1, 1, 1, 48, 128],
        vec![2, 1, 48, 128],
        vec![1, 2, 48, 128],
        vec![1, 1, 47, 128],
        vec![1, 1, 49, 128],
        vec![1, 1, 48, 127],
        vec![1, 1, 48, 129],
        vec![1, 1, 1, 6144],
        vec![1, 1, 48, 0],
    ] {
        reject(&[Array::zeros_dtype(&shape, Dtype::Float32)?], &z)?;
        cases += 1;
    }
    for shape in [
        vec![],
        vec![6144],
        vec![1, 6144],
        vec![1, 48, 128],
        vec![1, 1, 1, 48, 128],
        vec![2, 1, 48, 128],
        vec![1, 0, 48, 128],
        vec![1, 2, 48, 128],
        vec![1, 9, 48, 128],
        vec![1, 1, 47, 128],
        vec![1, 1, 49, 128],
        vec![1, 1, 48, 127],
        vec![1, 1, 48, 129],
        vec![1, 1, 1, 6144],
        vec![1, 1, 48, 0],
    ] {
        reject(
            &[good.try_clone()?],
            &Array::zeros_dtype(&shape, Dtype::Float16)?,
        )?;
        cases += 1;
    }
    for dtype in [Dtype::Bool, Dtype::UInt32, Dtype::BFloat16, Dtype::Float16] {
        reject(&[Array::zeros_dtype(&ROW_SHAPE, dtype)?], &z)?;
        cases += 1;
    }
    for dtype in [Dtype::Bool, Dtype::UInt32, Dtype::BFloat16, Dtype::Float32] {
        reject(
            &[good.try_clone()?],
            &Array::zeros_dtype(&ROW_SHAPE, dtype)?,
        )?;
        cases += 1;
    }
    let z2 = Array::zeros_dtype(&[1, 2, 48, 128], Dtype::Float16)?;
    for index in 0..2 {
        for (shape, dtype) in [
            (&ROW_SHAPE[..], Dtype::Float16),
            (&[1, 1, 1, 6144][..], Dtype::Float32),
        ] {
            let mut normed = vec![good.try_clone()?, good.try_clone()?];
            normed[index] = Array::zeros_dtype(shape, dtype)?;
            reject(&normed, &z2)?;
            cases += 1;
        }
    }
    ensure!(cases == 42);
    println!(
        "GDN_EPILOGUE_INVALID cases={cases} kernel_graph_requests=0 host_readbacks=0 source_mutation=false"
    );
    Ok(())
}

fn diagnostic_uniform_output(value: &Array, dtype: Dtype) -> Result<(Vec<u8>, u32)> {
    ensure!(value.shape() == [1, WIDTH as i32] && value.dtype() == dtype);
    let values = value.to_f32()?;
    ensure!(
        values.len() == WIDTH && values.iter().all(|v| v.is_finite()),
        "empty or nonfinite diagnostic output"
    );
    let bytes = value.to_bytes()?;
    let item_size = dtype.item_size();
    ensure!(bytes.len() == WIDTH * item_size);
    ensure!(
        bytes
            .chunks_exact(item_size)
            .all(|item| item == &bytes[..item_size]),
        "diagnostic fixture did not produce uniform output"
    );
    let bits = match dtype {
        Dtype::Float32 => u32::from_ne_bytes(bytes[..4].try_into()?),
        Dtype::Float16 => u32::from(u16::from_ne_bytes(bytes[..2].try_into()?)),
        _ => anyhow::bail!("unsupported diagnostic output dtype"),
    };
    Ok((bytes, bits))
}

#[test]
#[ignore = "physical Apple GPU; DIAG-02 original counterexample and test-only FP32 intermediates, run alone"]
fn diagnostic_counterexample_exp_volatile_and_stock_probability_boundaries() -> Result<()> {
    const NORM_BITS: u32 = 0x3f80_1001;
    const GATE_BITS: u16 = 0xc8dc;
    let normed = vec![Array::from_f32(
        &vec![f32::from_bits(NORM_BITS); WIDTH],
        &ROW_SHAPE,
    )?];
    let z = Array::from_f16_bits(&vec![GATE_BITS; WIDTH], &ROW_SHAPE)?;
    let before = snapshot(&normed, &z)?;

    let stock_probability = z.astype(Dtype::Float32)?.sigmoid()?;
    let (probability_bytes, probability_bits) = diagnostic_uniform_output(
        &stock_probability.reshape(&[1, WIDTH as i32])?,
        Dtype::Float32,
    )?;
    let stock_product = normed[0].mul(&stock_probability)?;
    let (product_bytes, product_bits) =
        diagnostic_uniform_output(&stock_product.reshape(&[1, WIDTH as i32])?, Dtype::Float32)?;
    let stock_result = stock_product
        .astype(Dtype::Float16)?
        .reshape(&[1, WIDTH as i32])?;
    let (stock_bytes, stock_bits) = diagnostic_uniform_output(&stock_result, Dtype::Float16)?;
    ensure!(
        stock_bits == 0x03f2,
        "DIAG-02 stock no longer reproduces the recorded counterexample"
    );

    reset_kernel_requests_for_test();
    let readbacks = Array::host_readbacks_for_test();
    let current = apply(&normed, &z)?;
    ensure!(
        kernel_requests_for_test() == 1
            && last_rows_for_test() == 1
            && Array::host_readbacks_for_test() == readbacks
    );
    let (current_bytes, current_bits) = diagnostic_uniform_output(&current, Dtype::Float16)?;
    ensure!(
        current_bits == 0x03f2 && current_bytes == stock_bytes,
        "FIX-01 current apply differs from stock at the recorded counterexample"
    );
    println!(
        "GDN_DIAG02_BASELINE values={WIDTH} norm_f32_bits=0x{NORM_BITS:08x} gate_f16_bits=0x{GATE_BITS:04x} stock_probability_bits=0x{probability_bits:08x} stock_probability={:?} stock_product_bits=0x{product_bits:08x} stock_product={:?} stock_out_bits=0x{stock_bits:04x} current_apply_out_bits=0x{current_bits:04x} recorded_stock_p_bits=0x387c4076 recorded_midpoint_product_bits=0x387c6000 tie_even_out=0x03f2",
        f32::from_bits(probability_bits),
        f32::from_bits(product_bits),
    );

    let production_shader = include_str!("../../shaders/flash_gdn_epilogue.metal");
    ensure!(
        production_shader.matches("metal::precise::exp(").count() == 1
            && !production_shader.contains("metal::exp("),
        "FIX-01 diagnostic requires exactly one precise exp in the real shader"
    );
    let shader = production_shader.replacen("metal::precise::exp(", "metal::exp(", 1);
    const STORE: &str = "output[index] = half(product);";
    const STORES: &str = "output[index] = half(product);\np_out[index] = probability;\nproduct_out[index] = product;";
    const GATE_BLOCK: &str = "float gate = float(z[index]);\nfloat y = 1.0f / (1.0f + metal::exp(metal::abs(gate)));\n// The stock sigmoid and product each materialize FP32 before the next operation.\nvolatile float probability = gate < 0.0f ? y : 1.0f - y;";
    ensure!(
        shader.matches(STORE).count() == 1
            && shader.matches("metal::exp(").count() == 1
            && shader.matches("volatile float ").count() == 2
            && shader.matches(GATE_BLOCK).count() == 1,
        "DIAG-02 real shader changed; diagnostic derivation requires review"
    );
    let different = |actual: &[u8], expected: &[u8], size: usize| {
        actual
            .chunks_exact(size)
            .zip(expected.chunks_exact(size))
            .filter(|(a, e)| a != e)
            .count()
    };
    let mut variants = 0;
    for (name, precise, remove_volatile, stock_input) in [
        ("original", false, false, false),
        ("precise_exp", true, false, false),
        ("without_volatile", false, true, false),
        ("precise_without_volatile", true, true, false),
        ("stock_probability_input", false, false, true),
    ] {
        let mut source = shader.replace(STORE, STORES);
        if precise {
            source = source.replace("metal::exp(", "metal::precise::exp(");
        }
        if remove_volatile {
            source = source.replace("volatile float ", "float ");
        }
        let (names, inputs) = if stock_input {
            source = source.replace(
                GATE_BLOCK,
                "volatile float probability = stock_probability[index];",
            );
            ensure!(!source.contains("metal::exp("));
            (
                ["normed0", "stock_probability"],
                [&normed[0], &stock_probability],
            )
        } else {
            (["normed0", "z"], [&normed[0], &z])
        };
        let outputs = crate::array::metal_kernel(
            &format!("mlxl3_gdn_epilogue_diag02_{name}"),
            &names,
            &["p_out", "product_out", "output"],
            "#define ROWS 1u\n#define WIDTH 6144u\n",
            &source,
            &inputs,
            &[
                vec![1, WIDTH as i32],
                vec![1, WIDTH as i32],
                vec![1, WIDTH as i32],
            ],
            &[Dtype::Float32, Dtype::Float32, Dtype::Float16],
            [WIDTH as i32, 1, 1],
            [128, 1, 1],
        )?;
        ensure!(outputs.len() == 3, "missing diagnostic outputs");
        let (p_bytes, p_bits) = diagnostic_uniform_output(&outputs[0], Dtype::Float32)?;
        let (x_bytes, x_bits) = diagnostic_uniform_output(&outputs[1], Dtype::Float32)?;
        let (out_bytes, out_bits) = diagnostic_uniform_output(&outputs[2], Dtype::Float16)?;
        let probability_mismatches = different(&p_bytes, &probability_bytes, 4);
        let product_mismatches = different(&x_bytes, &product_bytes, 4);
        let output_mismatches = different(&out_bytes, &stock_bytes, 2);
        println!(
            "GDN_DIAG02_VARIANT name={name} values={WIDTH} probability_f32_bits=0x{p_bits:08x} probability_f32={:?} product_f32_bits=0x{x_bits:08x} product_f32={:?} out_f16_bits=0x{out_bits:04x} out_f32={:?} probability_mismatches={probability_mismatches} product_mismatches={product_mismatches} output_mismatches={output_mismatches} output_matches_stock={}",
            f32::from_bits(p_bits),
            f32::from_bits(x_bits),
            f16::from_bits(out_bits as u16).to_f32(),
            output_mismatches == 0,
        );
        if name == "original" {
            ensure!(
                out_bits == 0x03f1,
                "DIAG-02 historical original variant no longer reproduces 03f1"
            );
        } else if name == "precise_exp" {
            ensure!(
                out_bytes == current_bytes,
                "DIAG-02 sensor changed precise output: three-output shader differs from current apply"
            );
        }
        variants += 1;
    }
    ensure!(variants == 5 && snapshot(&normed, &z)? == before);
    println!(
        "GDN_DIAG02_COMPLETE variants={variants} values_per_output={WIDTH} outputs_per_variant=3 status=diagnostic_only_no_candidate_parity_or_gain_claim"
    );
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
fn actual_gdn_epilogue_smoke_entry() {
    smoke_b1_b2_fp32_norms_match_stock_and_independent_zero_gate_literals().unwrap();
}
