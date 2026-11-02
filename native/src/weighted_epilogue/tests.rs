use super::{apply, kernel_requests_for_test, reset_kernel_requests_for_test};
use crate::array::{Array, Dtype};
use anyhow::{Result, ensure};
use half::f16;

const FLOATS: [Dtype; 2] = [Dtype::Float16, Dtype::Float32];

struct Fixture {
    hot: Vec<u16>,
    cold: Vec<u16>,
    order: Vec<u32>,
    canonical: Vec<f32>,
    weights: Vec<f32>,
    width: usize,
}

fn poison(index: usize) -> u16 {
    [0x7e11, 0x7c00, 0xfc00][index % 3]
}

fn fixture(
    rows: usize,
    width: usize,
    is_hot: &[bool],
    padded_hot: bool,
    duplicate_order: bool,
) -> Fixture {
    assert_eq!(rows, is_hot.len());
    let truth = (0..rows)
        .flat_map(|row| {
            (0..width).map(move |column| {
                f16::from_f32(((row * 29 + column * 7 + 3) % 257) as f32 / 64. - 2.).to_f32()
            })
        })
        .collect::<Vec<_>>();
    let mut hot = Vec::new();
    let mut cold = Vec::new();
    let mut hot_position = vec![0; rows];
    let mut cold_position = vec![0; rows];
    for (row, &row_is_hot) in is_hot.iter().enumerate() {
        if row_is_hot || padded_hot {
            hot_position[row] = hot.len() / width;
            hot.extend((0..width).map(|column| {
                if row_is_hot {
                    f16::from_f32(truth[row * width + column]).to_bits()
                } else {
                    poison(row + column)
                }
            }));
        }
        if !row_is_hot {
            cold_position[row] = cold.len() / width;
            cold.extend(
                truth[row * width..(row + 1) * width]
                    .iter()
                    .map(|&value| f16::from_f32(value).to_bits()),
            );
        }
    }
    if hot.is_empty() {
        hot.extend((0..width).map(poison));
    }
    if cold.is_empty() {
        cold.extend((0..width).map(poison));
    }
    let hot_rows = hot.len() / width;
    let mut order = Vec::with_capacity(rows);
    let mut canonical = Vec::with_capacity(rows * width);
    for destination in 0..rows {
        let origin = if duplicate_order {
            if destination % 5 == 0 {
                0
            } else {
                rows - 1 - destination
            }
        } else {
            destination
        };
        order.push(if is_hot[origin] {
            hot_position[origin] as u32
        } else {
            (hot_rows + cold_position[origin]) as u32
        });
        canonical.extend_from_slice(&truth[origin * width..(origin + 1) * width]);
    }
    let weights = (0..rows)
        .map(|row| [1. / 3., -0.2, 0.5, -1.25, 0., 0.125, 0.7][row % 7])
        .collect();
    Fixture {
        hot,
        cold,
        order,
        canonical,
        weights,
        width,
    }
}

fn source(values: &[u16], width: usize, strided: bool) -> Result<Array> {
    ensure!(width > 0);
    let rows = values.len() / width;
    ensure!(rows > 0 && values.len() == rows * width);
    if strided {
        let mut transposed = vec![0; values.len()];
        for row in 0..rows {
            for column in 0..width {
                transposed[column * rows + row] = values[row * width + column];
            }
        }
        Array::from_f16_bits(&transposed, &[width as i32, rows as i32])?.transpose(&[1, 0])
    } else {
        Array::from_f16_bits(values, &[rows as i32, width as i32])
    }
}

fn weights(values: &[f32], dtype: Dtype, shape: &[i32], strided: bool) -> Result<Array> {
    let packed;
    let (values, shape) = if strided {
        packed = values
            .iter()
            .flat_map(|&value| [value, f32::NAN])
            .collect::<Vec<_>>();
        (packed.as_slice(), vec![values.len() as i32, 2])
    } else {
        (values, shape.to_vec())
    };
    let array = match dtype {
        Dtype::Float16 => Array::from_f16_bits(
            &values
                .iter()
                .map(|&value| f16::from_f32(value).to_bits())
                .collect::<Vec<_>>(),
            &shape,
        )?,
        Dtype::Float32 => Array::from_f32(values, &shape)?,
        _ => anyhow::bail!("test weight dtype must be floating"),
    };
    if strided {
        array.slice(1, 0, 1)
    } else {
        Ok(array)
    }
}

fn product_dtype(activation: Dtype, weight: Dtype) -> Dtype {
    if activation == Dtype::Float16 && weight == Dtype::Float16 {
        Dtype::Float16
    } else {
        Dtype::Float32
    }
}

fn literal_bytes(values: &[f32], dtype: Dtype) -> Vec<u8> {
    match dtype {
        Dtype::Float16 => values
            .iter()
            .flat_map(|&value| f16::from_f32(value).to_bits().to_ne_bytes())
            .collect(),
        Dtype::Float32 => values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect(),
        _ => unreachable!("only float outputs are tested"),
    }
}

fn cpu_expected(fixture: &Fixture, activation: Dtype, weight_dtype: Dtype) -> Vec<u8> {
    let product = fixture
        .canonical
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let weight = fixture.weights[index / fixture.width];
            let weight = if weight_dtype == Dtype::Float16 {
                f16::from_f32(weight).to_f32()
            } else {
                weight
            };
            f16::from_f32(value).to_f32() * weight
        })
        .collect::<Vec<_>>();
    literal_bytes(&product, product_dtype(activation, weight_dtype))
}

fn check(
    fixture: &Fixture,
    activation: Dtype,
    weight_dtype: Dtype,
    strides: usize,
    weight_shape: &[i32],
) -> Result<Array> {
    let rows = fixture.order.len();
    let hot = source(&fixture.hot, fixture.width, strides & 1 != 0)?;
    let cold = source(&fixture.cold, fixture.width, strides & 2 != 0)?;
    let weights = weights(
        &fixture.weights,
        weight_dtype,
        weight_shape,
        strides & 4 != 0,
    )?;
    let before = [hot.to_bytes()?, cold.to_bytes()?, weights.to_bytes()?];
    reset_kernel_requests_for_test();
    let readbacks = Array::host_readbacks_for_test();
    let actual = apply(&hot, &cold, &fixture.order, &weights, activation)?;
    ensure!(Array::host_readbacks_for_test() == readbacks);
    ensure!(
        kernel_requests_for_test() == 1,
        "candidate path was not requested"
    );
    let expected_dtype = product_dtype(activation, weight_dtype);
    ensure!(
        actual.shape() == [rows as i32, fixture.width as i32] && actual.dtype() == expected_dtype,
        "weighted matrix shape/promotion differs"
    );
    let indices = Array::from_u32(&fixture.order, &[rows as i32])?;
    let stock = Array::concatenate(&[&hot, &cold], 0)?
        .take(&indices, 0)?
        .astype(activation)?
        .mul(&weights.reshape(&[rows as i32, 1])?)?;
    ensure!(
        stock.dtype() == expected_dtype,
        "MLX promotion contract changed"
    );
    let bytes = actual.to_bytes()?;
    ensure!(
        bytes.len() == rows * fixture.width * expected_dtype.item_size()
            && bytes == cpu_expected(fixture, activation, weight_dtype)
            && bytes == stock.to_bytes()?,
        "weighted result differs from CPU or stock MLX oracle"
    );
    let values = actual.to_f32()?;
    ensure!(values.len() == rows * fixture.width && values.iter().all(|v| v.is_finite()));
    if rows.is_multiple_of(10) {
        let shape = [rows as i32 / 10, 10, fixture.width as i32];
        let reduced = actual.reshape(&shape)?.sum(1, false)?;
        let expected = stock.reshape(&shape)?.sum(1, false)?;
        ensure!(
            reduced.shape() == [rows as i32 / 10, fixture.width as i32]
                && reduced.dtype() == expected.dtype()
                && reduced.to_bytes()? == expected.to_bytes()?,
            "unchanged MLX routed sum differs"
        );
        let values = reduced.to_f32()?;
        ensure!(values.len() == rows / 10 * fixture.width && values.iter().all(|v| v.is_finite()));
    }
    ensure!(
        before == [hot.to_bytes()?, cold.to_bytes()?, weights.to_bytes()?],
        "candidate or stock oracle mutated an input alias"
    );
    Ok(actual)
}

#[test]
#[ignore = "physical Apple GPU; exhaustive Hot/Cold partitions, run alone"]
fn all_partitions_match_stock_cpu_promotion_and_unchanged_routed_sum() -> Result<()> {
    let mut cases = 0;
    for partition in 0..1024usize {
        let is_hot = (0..10)
            .map(|row| partition & (1 << row) != 0)
            .collect::<Vec<_>>();
        let fixture = fixture(10, 17, &is_hot, true, false);
        for activation in FLOATS {
            for weight_dtype in FLOATS {
                drop(check(&fixture, activation, weight_dtype, 0, &[10, 1])?);
                cases += 1;
            }
        }
    }
    ensure!(cases == 4096);
    println!(
        "WEIGHTED_EPILOGUE_PARTITIONS partitions=1024 cases={cases} rows=10 width=17 parity=byte_exact"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; production shapes, strides and bounds, run alone"]
fn production_shapes_strides_duplicates_and_route_width_boundaries_match_oracles() -> Result<()> {
    let mut production = 0;
    for tokens in 1..=8 {
        let rows = tokens * 10;
        let is_hot = (0..rows).map(|row| row % 3 == 1).collect::<Vec<_>>();
        for padded in [false, true] {
            let fixture = fixture(rows, 2560, &is_hot, padded, true);
            for strides in 0..8 {
                for activation in FLOATS {
                    for weight_dtype in FLOATS {
                        drop(check(
                            &fixture,
                            activation,
                            weight_dtype,
                            strides,
                            &[tokens as i32, 10],
                        )?);
                        production += 1;
                    }
                }
            }
        }
    }
    let mut routes = 0;
    for rows in 1..=80 {
        let is_hot = (0..rows).map(|row| row % 2 == 0).collect::<Vec<_>>();
        for padded in [false, true] {
            let fixture = fixture(rows, 17, &is_hot, padded, true);
            for activation in FLOATS {
                for weight_dtype in FLOATS {
                    let shape = vec![rows as i32];
                    drop(check(&fixture, activation, weight_dtype, rows % 8, &shape)?);
                    routes += 1;
                }
            }
        }
    }
    let mut widths = 0;
    for width in [1, 15, 16, 17, 31, 32, 33, 255, 256, 257, 2559, 2560, 2561] {
        let is_hot = (0..17).map(|row| row % 3 == 0).collect::<Vec<_>>();
        for padded in [false, true] {
            let fixture = fixture(17, width, &is_hot, padded, true);
            for activation in FLOATS {
                for weight_dtype in FLOATS {
                    drop(check(&fixture, activation, weight_dtype, 7, &[1, 17, 1])?);
                    widths += 1;
                }
            }
        }
    }
    let mut single_rows = 0;
    for hot in [false, true] {
        for padded in [false, true] {
            let fixture = fixture(1, 17, &[hot], padded, true);
            for shape in [vec![1], vec![1, 1]] {
                for activation in FLOATS {
                    for weight_dtype in FLOATS {
                        drop(check(&fixture, activation, weight_dtype, 0, &shape)?);
                        single_rows += 1;
                    }
                }
            }
        }
    }
    ensure!(production == 512 && routes == 640 && widths == 104 && single_rows == 32);
    println!(
        "WEIGHTED_EPILOGUE_SHAPES production={production} route_boundaries={routes} width_boundaries={widths} single_rows={single_rows} cases=1288 parity=byte_exact"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; independent literal products and signed zero, run alone"]
fn literal_products_preserve_signed_zero_duplicates_and_all_weight_shapes() -> Result<()> {
    let fixture = Fixture {
        hot: [2., -4., 0.5]
            .map(|value| f16::from_f32(value).to_bits())
            .into_iter()
            .chain([poison(0), poison(1), poison(2)])
            .collect(),
        cold: [-8., 16., -2., 1., -2., 4.]
            .map(|value| f16::from_f32(value).to_bits())
            .to_vec(),
        order: vec![0, 3, 0, 2],
        canonical: vec![2., -4., 0.5, 1., -2., 4., 2., -4., 0.5, -8., 16., -2.],
        weights: vec![0.5, -0.25, 2., 0.],
        width: 3,
    };
    let literal = [1., -2., 0.25, -0.25, 0.5, -1., 4., -8., 1., -0.0, 0.0, -0.0];
    let mut cases = 0;
    for shape in [vec![4], vec![4, 1], vec![1, 4], vec![2, 2]] {
        for activation in FLOATS {
            for weight_dtype in FLOATS {
                let actual = check(&fixture, activation, weight_dtype, 0, &shape)?;
                ensure!(
                    actual.to_bytes()?
                        == literal_bytes(&literal, product_dtype(activation, weight_dtype)),
                    "power-of-two literal oracle or signed zero differs"
                );
                cases += 1;
            }
        }
    }
    ensure!(cases == 16);
    println!(
        "WEIGHTED_EPILOGUE_LITERAL cases={cases} signed_zero=true unused_hot_poison=true parity=byte_exact"
    );
    Ok(())
}

fn reject(
    hot: &Array,
    cold: &Array,
    order: &[u32],
    weights: &Array,
    activation: Dtype,
) -> Result<()> {
    reset_kernel_requests_for_test();
    let readbacks = Array::host_readbacks_for_test();
    ensure!(apply(hot, cold, order, weights, activation).is_err());
    ensure!(
        kernel_requests_for_test() == 0 && Array::host_readbacks_for_test() == readbacks,
        "invalid input reached a kernel graph request or host readback"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU arrays; invalid inputs must fail before requesting a kernel"]
fn invalid_inputs_reject_before_kernel_request_without_mutating_sources() -> Result<()> {
    let hot = Array::from_f16_bits(&[0x3c00; 2 * 17], &[2, 17])?;
    let cold = Array::from_f16_bits(&[0x4000; 3 * 17], &[3, 17])?;
    let weights = weights(&[0.25; 10], Dtype::Float32, &[10, 1], false)?;
    let order = [0, 3, 1, 4, 2, 0, 3, 1, 4, 2];
    let before = [hot.to_bytes()?, cold.to_bytes()?, weights.to_bytes()?];
    let mut cases = 0;
    for activation in [
        Dtype::Bool,
        Dtype::UInt8,
        Dtype::UInt16,
        Dtype::UInt32,
        Dtype::UInt64,
        Dtype::Int8,
        Dtype::Int16,
        Dtype::Int32,
        Dtype::Int64,
        Dtype::Float64,
        Dtype::BFloat16,
        Dtype::Complex64,
    ] {
        reject(&hot, &cold, &order, &weights, activation)?;
        cases += 1;
    }
    for dtype in [Dtype::Float32, Dtype::BFloat16, Dtype::UInt16] {
        reject(
            &Array::zeros_dtype(&[2, 17], dtype)?,
            &cold,
            &order,
            &weights,
            Dtype::Float16,
        )?;
        reject(
            &hot,
            &Array::zeros_dtype(&[3, 17], dtype)?,
            &order,
            &weights,
            Dtype::Float16,
        )?;
        cases += 2;
    }
    for dtype in [Dtype::BFloat16, Dtype::UInt32] {
        reject(
            &hot,
            &cold,
            &order,
            &Array::zeros_dtype(&[10, 1], dtype)?,
            Dtype::Float16,
        )?;
        cases += 1;
    }
    for shape in [vec![34], vec![1, 2, 17]] {
        reject(
            &Array::zeros_dtype(&shape, Dtype::Float16)?,
            &cold,
            &order,
            &weights,
            Dtype::Float16,
        )?;
        cases += 1;
    }
    for shape in [vec![51], vec![1, 3, 17]] {
        reject(
            &hot,
            &Array::zeros_dtype(&shape, Dtype::Float16)?,
            &order,
            &weights,
            Dtype::Float16,
        )?;
        cases += 1;
    }
    reject(
        &hot,
        &Array::zeros_dtype(&[3, 18], Dtype::Float16)?,
        &order,
        &weights,
        Dtype::Float16,
    )?;
    reject(
        &Array::zeros_dtype(&[2, 0], Dtype::Float16)?,
        &Array::zeros_dtype(&[3, 0], Dtype::Float16)?,
        &order,
        &weights,
        Dtype::Float16,
    )?;
    cases += 2;
    for rows in [0, 81] {
        reject(
            &Array::zeros_dtype(&[rows, 17], Dtype::Float16)?,
            &cold,
            &order,
            &weights,
            Dtype::Float16,
        )?;
        reject(
            &hot,
            &Array::zeros_dtype(&[rows, 17], Dtype::Float16)?,
            &order,
            &weights,
            Dtype::Float16,
        )?;
        cases += 2;
    }
    for rows in [0usize, 81] {
        reject(
            &hot,
            &cold,
            &vec![0; rows],
            &Array::zeros_dtype(&[rows as i32], Dtype::Float32)?,
            Dtype::Float16,
        )?;
        cases += 1;
    }
    for bad in [5, u32::MAX] {
        reject(&hot, &cold, &[bad; 10], &weights, Dtype::Float16)?;
        cases += 1;
    }
    for shape in [vec![0], vec![9], vec![11], vec![]] {
        reject(
            &hot,
            &cold,
            &order,
            &Array::zeros_dtype(&shape, Dtype::Float32)?,
            Dtype::Float16,
        )?;
        cases += 1;
    }
    reject(
        &hot,
        &cold,
        &[0],
        &Array::from_f32(&[0.5], &[])?,
        Dtype::Float32,
    )?;
    cases += 1;
    ensure!(cases == 39);
    ensure!(before == [hot.to_bytes()?, cold.to_bytes()?, weights.to_bytes()?]);
    println!(
        "WEIGHTED_EPILOGUE_INVALID cases={cases} kernel_graph_requests=0 host_readbacks=0 source_mutation=false"
    );
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
fn actual_weighted_epilogue_literal_entry() {
    literal_products_preserve_signed_zero_duplicates_and_all_weight_shapes().unwrap();
}

#[test]
#[ignore = "physical Apple GPU; finite FP16 boundaries and tiny weights, run alone"]
fn finite_boundary_values_and_tiny_weights_match_stock() -> Result<()> {
    const ROWS: usize = 10;
    const WIDTH: usize = 17;
    let mut hot_bits = vec![0; ROWS * WIDTH];
    let mut cold_bits = vec![0; ROWS * WIDTH];
    let mut selected_bits = Vec::with_capacity(ROWS * WIDTH);
    for row in 0..ROWS {
        for column in 0..WIDTH {
            // Only one max-finite pair per column keeps the unchanged sum finite.
            let magnitude = match row / 2 {
                0 => [0x7bff, 0x7bfe][column % 2],
                1 => [0x0001, 0x03ff, 0x03fe][column % 3],
                2 => [0x0400, 0x0401, 0x0402][column % 3],
                3 => [0x0000, 0x3bff, 0x3c00, 0x3c01][column % 4],
                _ => [0x3c01, 0x3555, 0x3556, 0x3bff][column % 4],
            };
            let bits = magnitude | if row % 2 == 0 { 0 } else { 0x8000 };
            let index = row * WIDTH + column;
            hot_bits[index] = if row % 2 == 0 { bits } else { poison(index) };
            cold_bits[index] = if row % 2 == 1 {
                bits
            } else {
                poison(index + 1)
            };
            selected_bits.push(bits);
        }
    }
    for edge in [
        0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x7bff, 0xfbff, 0x3bff,
        0xbbff, 0x3c01, 0xbc01,
    ] {
        ensure!(
            selected_bits.contains(&edge),
            "finite boundary was not selected"
        );
    }
    let hot = source(&hot_bits, WIDTH, false)?;
    let cold = source(&cold_bits, WIDTH, false)?;
    let order = (0..ROWS)
        .map(|row| (if row % 2 == 0 { row } else { ROWS + row }) as u32)
        .collect::<Vec<_>>();
    let indices = Array::from_u32(&order, &[ROWS as i32])?;
    let sources_before = [hot.to_bytes()?, cold.to_bytes()?];
    let anchors = [
        1.,
        -1.,
        0.75,
        f32::from_bits(0x3f3f_ffff),
        f32::from_bits(0x3f40_0001),
        0.5,
        0.,
        -0.,
        f16::from_bits(0x0001).to_f32(),
        -f16::from_bits(0x0001).to_f32(),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::from_bits(0x007f_ffff),
        -f32::from_bits(0x007f_ffff),
    ];
    ensure!(anchors.iter().all(|w| w.is_finite() && w.abs() <= 1.));
    let mut cases = 0;
    for anchor in anchors {
        for activation in FLOATS {
            for weight_dtype in FLOATS {
                let weights = weights(&[anchor; ROWS], weight_dtype, &[ROWS as i32, 1], false)?;
                let weights_before = weights.to_bytes()?;
                reset_kernel_requests_for_test();
                let readbacks = Array::host_readbacks_for_test();
                let actual = apply(&hot, &cold, &order, &weights, activation)?;
                ensure!(
                    kernel_requests_for_test() == 1
                        && Array::host_readbacks_for_test() == readbacks
                );
                // MLX's GPU oracle, rather than CPU IEEE arithmetic, defines tiny-value FTZ.
                let stock = Array::concatenate(&[&hot, &cold], 0)?
                    .take(&indices, 0)?
                    .astype(activation)?
                    .mul(&weights)?;
                let expected_dtype = product_dtype(activation, weight_dtype);
                let actual_bytes = actual.to_bytes()?;
                ensure!(
                    actual.shape() == [ROWS as i32, WIDTH as i32]
                        && actual.dtype() == expected_dtype
                        && stock.dtype() == expected_dtype
                        && actual_bytes.len() == ROWS * WIDTH * expected_dtype.item_size()
                        && actual_bytes == stock.to_bytes()?,
                    "finite boundary or tiny-weight stock oracle differs"
                );
                let values = actual.to_f32()?;
                ensure!(values.len() == ROWS * WIDTH && values.iter().all(|v| v.is_finite()));
                let shape = [1, ROWS as i32, WIDTH as i32];
                let reduced = actual.reshape(&shape)?.sum(1, false)?;
                let reference = stock.reshape(&shape)?.sum(1, false)?;
                ensure!(
                    reduced.shape() == [1, WIDTH as i32]
                        && reduced.dtype() == reference.dtype()
                        && reduced.to_bytes()? == reference.to_bytes()?,
                    "finite boundary unchanged stock sum differs"
                );
                let values = reduced.to_f32()?;
                ensure!(values.len() == WIDTH && values.iter().all(|v| v.is_finite()));
                ensure!(weights.to_bytes()? == weights_before);
                cases += 1;
            }
        }
    }
    ensure!(cases == 64);
    ensure!(sources_before == [hot.to_bytes()?, cold.to_bytes()?]);
    println!(
        "WEIGHTED_EPILOGUE_FINITE_BOUNDARIES cases={cases} fp16_edges=true fp32_tiny_weights=true stock_gpu_oracle=byte_exact"
    );
    Ok(())
}
