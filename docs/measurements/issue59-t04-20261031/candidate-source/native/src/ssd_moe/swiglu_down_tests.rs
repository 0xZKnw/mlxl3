//! T04 oracles adapted from PR61's two boundary tests; no timing or promotion.
use super::*;
use crate::array::{MetalCapture, swiglu_scaled_requests_for_test, synchronize};
use crate::linear::{ExpertSlots, expert_slots, expert_slots_swiglu_down};
use half::f16;

const WIDTH: i32 = 640;
const ROWS: [i32; 7] = [1, 2, 4, 8, 10, 20, 80];

fn exact(actual: &Array, expected: &Array, shape: &[i32], dtype: Dtype) -> Result<usize> {
    ensure!(actual.shape() == shape && expected.shape() == shape);
    ensure!(actual.dtype() == dtype && expected.dtype() == dtype);
    let count = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))
        .context("invalid oracle cardinality")?;
    let a = actual.to_f32()?;
    let e = expected.to_f32()?;
    ensure!(count > 0 && a.len() == count && e.len() == count);
    ensure!(
        a.iter().chain(&e).all(|v| v.is_finite()),
        "nonfinite boundary oracle"
    );
    let a = actual.to_bytes()?;
    let e = expected.to_bytes()?;
    ensure!(a.len() == count * dtype.item_size() && e.len() == a.len());
    ensure!(a == e, "T04 candidate differs in complete output bytes");
    Ok(count)
}

fn matrix(values: &[f32], rows: i32, width: i32, dtype: Dtype, strided: bool) -> Result<Array> {
    ensure!(values.len() == rows as usize * width as usize);
    if strided {
        let transposed = (0..width as usize)
            .flat_map(|c| (0..rows as usize).map(move |r| values[r * width as usize + c]))
            .collect::<Vec<_>>();
        Array::from_f32(&transposed, &[width, rows])?
            .astype(dtype)?
            .transpose(&[1, 0])
    } else {
        Array::from_f32(values, &[rows, width])?.astype(dtype)
    }
}

fn stock(gate: &Array, up: &Array, scale: &Array) -> Result<Array> {
    gate.swiglu(up)?.astype(Dtype::Float16)?.mul(scale)
}

fn h128(value: &Array, rows: i32) -> Result<Array> {
    value
        .reshape(&[rows, 5, 128])?
        .hadamard_transform(Some(1. / 128f32.sqrt()))?
        .reshape(&[rows, WIDTH])
}

fn rejected_before_ffi(run: impl FnOnce() -> Result<Array>) -> Result<()> {
    let calls = swiglu_scaled_requests_for_test();
    let reads = Array::host_readbacks_for_test();
    ensure!(run().is_err(), "invalid boundary metadata accepted");
    ensure!(
        swiglu_scaled_requests_for_test() == calls,
        "invalid input entered boundary FFI"
    );
    ensure!(
        Array::host_readbacks_for_test() == reads,
        "invalid input read back data"
    );
    Ok(())
}

fn fixture(rows: i32, dtype: Dtype, strided: bool, boundaries: bool) -> Result<[Array; 3]> {
    let n = rows as usize * WIDTH as usize;
    let mut gate = (0..n)
        .map(|i| ((i * 37 % 8191) as f32 - 4095.) / 512.)
        .collect::<Vec<_>>();
    let mut up = (0..n)
        .map(|i| ((i * 101 % 8191) as f32 - 4095.) / 1024.)
        .collect::<Vec<_>>();
    let mut scale = (0..n)
        .map(|i| ((i * 73 % 8191) as f32 - 4095.) / 2048.)
        .collect::<Vec<_>>();
    if dtype == Dtype::Float32 {
        for (i, value) in gate.iter_mut().enumerate() {
            *value += (i % 3) as f32 / 65536.;
        }
    }
    if boundaries {
        let patterns = [
            0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x3c00, 0x3c01, 0x3555,
            0xb555, 0x4c00, 0xcc00, 0x7bff, 0xfbff,
        ];
        for i in 0..n {
            gate[i] = f16::from_bits(patterns[i % patterns.len()]).to_f32();
            up[i] = if i % 2 == 0 { 1. / 1024. } else { -1. / 512. };
            scale[i] = if i % 3 == 0 { -0.125 } else { 0.25 };
        }
        // Analytical rounding probes: stock GPU is authoritative, including FTZ.
        gate[0] = f16::from_bits(1).to_f32();
        up[0] = 2.;
        scale[0] = 1.;
        gate[1] = 16.;
        up[1] = f16::from_bits(1).to_f32();
        scale[1] = 1. / 32.;
    }
    Ok([
        matrix(&gate, rows, WIDTH, dtype, strided)?,
        matrix(&up, rows, WIDTH, dtype, strided)?,
        matrix(&scale, rows, WIDTH, Dtype::Float16, strided)?,
    ])
}

#[test]
#[ignore = "physical Apple GPU; short T04 F16/F32 smoke before the full matrices"]
fn t04_smoke_matches_stock_and_independent_literals() -> Result<()> {
    let mut cases = 0;
    let mut values = 0;
    for rows in [1, 2] {
        for dtype in [Dtype::Float16, Dtype::Float32] {
            for zero_gate in [false, true] {
                let n = rows as usize * WIDTH as usize;
                let gate =
                    Array::from_f32(&vec![if zero_gate { 0. } else { 16. }; n], &[rows, WIDTH])?
                        .astype(dtype)?;
                let up = Array::from_f32(&vec![0.125; n], &[rows, WIDTH])?.astype(dtype)?;
                let scale =
                    Array::from_f32(&vec![0.25; n], &[rows, WIDTH])?.astype(Dtype::Float16)?;
                let before = [gate.to_bytes()?, up.to_bytes()?, scale.to_bytes()?];
                let reference = stock(&gate, &up, &scale)?;
                let calls = swiglu_scaled_requests_for_test();
                let reads = Array::host_readbacks_for_test();
                let actual = gate.swiglu_scaled(&up, &scale)?;
                ensure!(swiglu_scaled_requests_for_test() == calls + 1);
                ensure!(Array::host_readbacks_for_test() == reads);
                values += exact(&actual, &reference, &[rows, WIDTH], Dtype::Float16)?;
                let literal =
                    Array::from_f32(&vec![if zero_gate { 0. } else { 0.5 }; n], &[rows, WIDTH])?
                        .astype(Dtype::Float16)?;
                exact(&actual, &literal, &[rows, WIDTH], Dtype::Float16)?;
                ensure!(before == [gate.to_bytes()?, up.to_bytes()?, scale.to_bytes()?]);
                cases += 1;
            }
        }
    }
    ensure!(cases == 8 && values == 7680);
    println!(
        "T04_SMOKE cases={cases} values={values} rows=1,2 gate_dtype=F16,F32 parity=byte_exact literal_anchors=true"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; PR61 T04 half roundings, strides and unchanged native H128"]
fn t04_swiglu_down_preserves_both_half_roundings_and_h128() -> Result<()> {
    let mut cases = 0;
    let mut values = 0;
    let mut mutation_detected = [false; 2];
    for rows in ROWS {
        for (d, dtype) in [Dtype::Float16, Dtype::Float32].into_iter().enumerate() {
            for strided in [false, true] {
                for boundaries in [false, true] {
                    let [gate, up, scale] = fixture(rows, dtype, strided, boundaries)?;
                    if dtype == Dtype::Float32 && !boundaries {
                        ensure!(
                            gate.to_f32()?
                                .iter()
                                .any(|&v| f16::from_f32(v).to_f32() != v),
                            "F32 fixture was accidentally rounded to F16"
                        );
                    }
                    let before = [gate.to_bytes()?, up.to_bytes()?, scale.to_bytes()?];
                    let reference = stock(&gate, &up, &scale)?;
                    let calls = swiglu_scaled_requests_for_test();
                    let reads = Array::host_readbacks_for_test();
                    let actual = gate.swiglu_scaled(&up, &scale)?;
                    ensure!(swiglu_scaled_requests_for_test() == calls + 1);
                    ensure!(Array::host_readbacks_for_test() == reads);
                    values += exact(&actual, &reference, &[rows, WIDTH], Dtype::Float16)
                        .with_context(|| format!("rows={rows},dtype={dtype:?},strided={strided},boundaries={boundaries}"))?;
                    exact(
                        &h128(&actual, rows)?,
                        &h128(&reference, rows)?,
                        &[rows, WIDTH],
                        Dtype::Float16,
                    )?;
                    let mutant = gate
                        .astype(Dtype::Float32)?
                        .swiglu(&up.astype(Dtype::Float32)?)?
                        .mul(&scale.astype(Dtype::Float32)?)?
                        .astype(Dtype::Float16)?;
                    let mutant_values = mutant.to_f32()?;
                    ensure!(
                        mutant_values.len() == rows as usize * WIDTH as usize
                            && mutant_values.iter().all(|v| v.is_finite()),
                        "rounding control must be finite and complete"
                    );
                    mutation_detected[d] |=
                        exact(&mutant, &reference, &[rows, WIDTH], Dtype::Float16).is_err();
                    if boundaries && rows == 1 && !strided {
                        let a = actual.to_bytes()?;
                        let r = reference.to_bytes()?;
                        let m = mutant.to_bytes()?;
                        println!(
                            "T04_ROUNDING_PROBES dtype={dtype:?} stock_first4={:?} candidate_first4={:?} deferred_first4={:?} expectation=physical_stock_not_cpu_ftz",
                            &r[..4],
                            &a[..4],
                            &m[..4]
                        );
                    }
                    ensure!(before == [gate.to_bytes()?, up.to_bytes()?, scale.to_bytes()?]);
                    cases += 1;
                }
            }
        }
    }
    ensure!(
        cases == 56 && values == 640000 && mutation_detected == [true, true],
        "deferred-rounding mutant escaped a dtype oracle"
    );
    // Two R80 matrices cover every finite F16 gate. This is not an exhaustive
    // Cartesian product of gate/up/scale, nor a proof of all F32 inputs.
    let finite = (0..=u16::MAX)
        .filter(|&b| b & 0x7c00 != 0x7c00)
        .collect::<Vec<_>>();
    ensure!(finite.len() == 63488);
    let mut seen = vec![false; 65536];
    let mut exhaustive_values = 0;
    let mut exhaustive_cases = 0;
    let n = 80 * WIDTH as usize;
    for batch in 0..2 {
        let bits = (0..n)
            .map(|i| {
                let b = finite[(batch * n + i) % finite.len()];
                seen[b as usize] = true;
                b
            })
            .collect::<Vec<_>>();
        let gate = Array::from_f16_bits(&bits, &[80, WIDTH])?;
        let ups = (0..n)
            .map(|i| [1. / 1024., -1. / 512., 3. / 1024., -5. / 2048.][i % 4])
            .collect::<Vec<_>>();
        let scales = (0..n)
            .map(|i| [0.125, -0.1875, 0.21875, -0.3125][i % 4])
            .collect::<Vec<_>>();
        let up = Array::from_f32(&ups, &[80, WIDTH])?.astype(Dtype::Float16)?;
        let scale = Array::from_f32(&scales, &[80, WIDTH])?.astype(Dtype::Float16)?;
        let before = [gate.to_bytes()?, up.to_bytes()?, scale.to_bytes()?];
        let reference = stock(&gate, &up, &scale)?;
        let calls = swiglu_scaled_requests_for_test();
        let reads = Array::host_readbacks_for_test();
        let candidate = gate.swiglu_scaled(&up, &scale)?;
        ensure!(swiglu_scaled_requests_for_test() == calls + 1);
        ensure!(Array::host_readbacks_for_test() == reads);
        exhaustive_values += exact(&candidate, &reference, &[80, WIDTH], Dtype::Float16)?;
        exact(
            &h128(&candidate, 80)?,
            &h128(&reference, 80)?,
            &[80, WIDTH],
            Dtype::Float16,
        )?;
        ensure!(before == [gate.to_bytes()?, up.to_bytes()?, scale.to_bytes()?]);
        exhaustive_cases += 1;
    }
    ensure!(
        exhaustive_cases == 2
            && exhaustive_values == 102400
            && seen.iter().filter(|&&x| x).count() == 63488
    );
    let valid = fixture(1, Dtype::Float16, false, false)?;
    let mut rejected = 0;
    for shape in [
        &[][..],
        &[640][..],
        &[1, 1, 640][..],
        &[0, 640][..],
        &[81, 640][..],
        &[1, 639][..],
    ] {
        let x = Array::zeros_dtype(shape, Dtype::Float16)?;
        rejected_before_ffi(|| x.swiglu_scaled(&x, &x))?;
        rejected += 1;
    }
    for dtype in [Dtype::UInt32, Dtype::Int32, Dtype::Bool] {
        let x = Array::zeros_dtype(&[1, WIDTH], dtype)?;
        rejected_before_ffi(|| x.swiglu_scaled(&x, &valid[2]))?;
        rejected += 1;
    }
    let wrong = Array::zeros_dtype(&[1, WIDTH - 1], Dtype::Float16)?;
    rejected_before_ffi(|| valid[0].swiglu_scaled(&wrong, &valid[2]))?;
    rejected_before_ffi(|| valid[0].swiglu_scaled(&valid[1], &wrong))?;
    let float_up = valid[1].astype(Dtype::Float32)?;
    rejected_before_ffi(|| valid[0].swiglu_scaled(&float_up, &valid[2]))?;
    let float_scale = valid[2].astype(Dtype::Float32)?;
    rejected_before_ffi(|| valid[0].swiglu_scaled(&valid[1], &float_scale))?;
    rejected += 4;
    ensure!(rejected == 13);
    println!(
        "T04_BOUNDARY cases={cases} pre_h128_values={values} h128_values={values} rows=1,2,4,8,10,20,80 strides=2 boundary_phases=2 mutation_by_dtype={mutation_detected:?} rejected={rejected} exhaustive_cases={exhaustive_cases} exhaustive_values={exhaustive_values} finite_gate_patterns=63488 parity=byte_exact source_mutation=false"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; PR61 actual slot-major EXL3 down, F16/F32 outputs and preflight"]
fn t04_down_projection_preserves_production_dtype_and_bits() -> Result<()> {
    let words = (0..4 * 40 * 160 * 32)
        .map(|i| (i * 137 + 19) as u16)
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let trellis = Array::from_bytes(&words, &[4, 1, 40, 160, 32], Dtype::UInt16)?;
    let mut cases = 0;
    let mut values = 0;
    for rows in ROWS {
        let ids = (0..rows).map(|i| (i % 4) as u32).collect::<Vec<_>>();
        let slots = ExpertSlots::new(&ids, 4)?;
        ensure!(slots.supports_swiglu_down_boundary());
        for dtype in [Dtype::Float16, Dtype::Float32] {
            for strided in [false, true] {
                let data = (0..rows * 2 * WIDTH)
                    .map(|i| {
                        ((i * 37 % 2039) as f32 - 1019.) / 1024.
                            + if dtype == Dtype::Float32 {
                                1. / 65536.
                            } else {
                                0.
                            }
                    })
                    .collect::<Vec<_>>();
                let gu =
                    matrix(&data, rows, 2 * WIDTH, dtype, strided)?.reshape(&[rows, 2, WIDTH])?;
                let su = (0..rows * WIDTH)
                    .map(|i| 0.5 + (i % 11) as f32 / 32.)
                    .collect::<Vec<_>>();
                let sv = (0..rows * H)
                    .map(|i| 0.25 + (i % 7) as f32 / 32.)
                    .collect::<Vec<_>>();
                let suh = matrix(&su, rows, WIDTH, Dtype::Float16, strided)?;
                let svh = matrix(&sv, rows, H, Dtype::Float16, strided)?;
                let before = [
                    gu.to_bytes()?,
                    suh.to_bytes()?,
                    svh.to_bytes()?,
                    trellis.to_bytes()?,
                ];
                let activated = gu
                    .slice(1, 0, 1)?
                    .reshape(&[rows, WIDTH])?
                    .swiglu(&gu.slice(1, 1, 2)?.reshape(&[rows, WIDTH])?)?;
                for stable in [false, true] {
                    let reference = expert_slots(
                        &activated,
                        &trellis,
                        &suh,
                        &svh,
                        &slots,
                        H,
                        1,
                        2,
                        Codebook::Mul1,
                        stable,
                    )?;
                    let calls = swiglu_scaled_requests_for_test();
                    let reads = Array::host_readbacks_for_test();
                    let candidate =
                        expert_slots_swiglu_down(&gu, &trellis, &suh, &svh, &slots, stable)?;
                    ensure!(swiglu_scaled_requests_for_test() == calls + 1);
                    ensure!(Array::host_readbacks_for_test() == reads);
                    values +=
                        exact(&candidate, &reference, &[rows, H], dtype).with_context(|| {
                            format!(
                                "down rows={rows},dtype={dtype:?},strided={strided},stable={stable}"
                            )
                        })?;
                    cases += 1;
                }
                ensure!(
                    before
                        == [
                            gu.to_bytes()?,
                            suh.to_bytes()?,
                            svh.to_bytes()?,
                            trellis.to_bytes()?
                        ]
                );
            }
        }
    }
    ensure!(cases == 56 && values == 2560000);
    let gu = Array::zeros_dtype(&[2, 2, WIDTH], Dtype::Float16)?;
    let suh = Array::zeros_dtype(&[2, WIDTH], Dtype::Float16)?;
    let svh = Array::zeros_dtype(&[2, H], Dtype::Float16)?;
    let slots = ExpertSlots::new(&[0, 0], 4)?;
    let mut rejected = 0;
    for shape in [
        [2, 1, WIDTH],
        [2, 2, WIDTH - 1],
        [0, 2, WIDTH],
        [81, 2, WIDTH],
    ] {
        let wrong = Array::zeros_dtype(&shape, Dtype::Float16)?;
        rejected_before_ffi(|| {
            expert_slots_swiglu_down(&wrong, &trellis, &suh, &svh, &slots, false)
        })?;
        rejected += 1;
    }
    for wrong in [
        Array::zeros_dtype(&[2, 2, WIDTH], Dtype::UInt32)?,
        Array::zeros_dtype(&[2, 2, WIDTH], Dtype::Bool)?,
    ] {
        rejected_before_ffi(|| {
            expert_slots_swiglu_down(&wrong, &trellis, &suh, &svh, &slots, false)
        })?;
        rejected += 1;
    }
    for wrong in [
        Array::zeros_dtype(&[3, 1, 40, 160, 32], Dtype::UInt16)?,
        Array::zeros_dtype(&[4, 1, 39, 160, 32], Dtype::UInt16)?,
        trellis.astype(Dtype::Float32)?,
    ] {
        rejected_before_ffi(|| expert_slots_swiglu_down(&gu, &wrong, &suh, &svh, &slots, false))?;
        rejected += 1;
    }
    for wrong in [
        Array::zeros_dtype(&[2, WIDTH - 1], Dtype::Float16)?,
        suh.astype(Dtype::Float32)?,
    ] {
        rejected_before_ffi(|| {
            expert_slots_swiglu_down(&gu, &trellis, &wrong, &svh, &slots, false)
        })?;
        rejected += 1;
    }
    for wrong in [
        Array::zeros_dtype(&[2, H - 1], Dtype::Float16)?,
        svh.astype(Dtype::Float32)?,
    ] {
        rejected_before_ffi(|| {
            expert_slots_swiglu_down(&gu, &trellis, &suh, &wrong, &slots, false)
        })?;
        rejected += 1;
    }
    let short = ExpertSlots::new(&[0], 4)?;
    rejected_before_ffi(|| expert_slots_swiglu_down(&gu, &trellis, &suh, &svh, &short, false))?;
    rejected += 1;
    let mut map = [u32::MAX; 512];
    map[0] = 0;
    let masked = ExpertSlots::residents(&Array::from_u32(&[0, 511], &[2])?, &map, 4)?;
    ensure!(!masked.supports_swiglu_down_boundary());
    rejected_before_ffi(|| expert_slots_swiglu_down(&gu, &trellis, &suh, &svh, &masked, false))?;
    rejected += 1;
    let grouped = ExpertSlots::grouped(&[0; 20], 4)?;
    ensure!(grouped.shares_weights() && !grouped.supports_swiglu_down_boundary());
    let gu20 = Array::zeros_dtype(&[20, 2, WIDTH], Dtype::Float16)?;
    let su20 = Array::zeros_dtype(&[20, WIDTH], Dtype::Float16)?;
    let sv20 = Array::zeros_dtype(&[20, H], Dtype::Float16)?;
    rejected_before_ffi(|| {
        expert_slots_swiglu_down(&gu20, &trellis, &su20, &sv20, &grouped, false)
    })?;
    rejected += 1;
    ensure!(rejected == 16);
    println!(
        "T04_DOWN cases={cases} values={values} rows=1,2,4,8,10,20,80 dtype=F16,F32 strides=2 stable=2 repeated_slots=true rejected={rejected} masked_grouped_not_supported=true parity=byte_exact source_mutation=false"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU and one Flash MoE layer; exactly two T04 captures, no timing"]
fn t04_captures_stock_and_fused_real_gu20_preparation() -> Result<()> {
    let directory = std::path::PathBuf::from(std::env::var("MLXL3_FLASH_T04_CAPTURE_DIR")?);
    ensure!(
        !directory.exists(),
        "refusing existing capture campaign directory"
    );
    std::fs::create_dir_all(&directory)?;
    let (checkpoint, input, _) = super::tests::admission_fixture()?;
    let mut moe = SsdMoe::load(&checkpoint, "model.language_model.layers.0.mlp", 53, true)?;
    let ids = moe.route(&input)?.0.to_u32()?;
    ensure!(ids.len() == 10);
    let reserved = moe.cache.borrow_mut().reserve(&ids)?;
    let jobs = ids
        .iter()
        .zip(&reserved)
        .map(|(&id, &(slot, _))| (id, slot))
        .collect::<Vec<_>>();
    moe.fill_many(&jobs)?;
    let indices = reserved
        .iter()
        .map(|&(slot, _)| slot as u32)
        .collect::<Vec<_>>();
    let slots = ExpertSlots::new(&indices.repeat(2), 53)?;
    let x = SsdMoe::routed_input(&Array::concatenate(&[&input, &input], 0)?)?;
    let expanded = x
        .reshape(&[20, 1, H])?
        .broadcast_to(&[20, 2, H])?
        .reshape(&[40, H])?;
    let gu = expert_slots(
        &expanded,
        &moe.banks[0],
        &slots.gather_scales(&moe.banks[2])?.reshape(&[40, H])?,
        &slots.gather_scales(&moe.banks[3])?.reshape(&[40, WIDTH])?,
        &slots,
        WIDTH,
        2,
        2,
        Codebook::Mul1,
        moe.stable_expert_kernels,
    )?
    .reshape(&[20, 2, WIDTH])?;
    let gate = gu.slice(1, 0, 1)?.reshape(&[20, WIDTH])?;
    let up = gu.slice(1, 1, 2)?.reshape(&[20, WIDTH])?;
    let scale = slots.gather_scales(&moe.banks[4])?.reshape(&[20, WIDTH])?;
    for value in [&gu, &gate, &up, &scale] {
        value.eval()?;
    }
    let before = [
        gu.to_bytes()?,
        gate.to_bytes()?,
        up.to_bytes()?,
        scale.to_bytes()?,
    ];
    let reference = h128(&stock(&gate, &up, &scale)?, 20)?;
    let warm = h128(&gate.swiglu_scaled(&up, &scale)?, 20)?;
    exact(&warm, &reference, &[20, WIDTH], Dtype::Float16)?;
    synchronize()?;
    let mut total = 0u64;
    let mut captures = 0;
    for fused in [false, true] {
        if captures > 0 {
            ensure!(total < 1 << 30);
        }
        let path = directory.join(if fused {
            "fused-r20.gputrace"
        } else {
            "stock-r20.gputrace"
        });
        let capture = MetalCapture::start(&path)?;
        // Rotate MLX's preexisting buffer after start, as qualified by T05/T17.
        synchronize()?;
        let calls = swiglu_scaled_requests_for_test();
        let output = h128(
            &if fused {
                gate.swiglu_scaled(&up, &scale)?
            } else {
                stock(&gate, &up, &scale)?
            },
            20,
        )?;
        output.eval()?;
        capture.finish()?;
        ensure!(swiglu_scaled_requests_for_test() == calls + u64::from(fused));
        exact(&output, &reference, &[20, WIDTH], Dtype::Float16)?;
        let bytes = crate::gdn_epilogue::capture_tests::capture_bytes(&path)?;
        total = total.checked_add(bytes).context("capture total overflow")?;
        ensure!(
            total < 1 << 30,
            "two captures exceed cumulative 1 GiB budget"
        );
        captures += 1;
    }
    ensure!(captures == 2 && total > 0);
    ensure!(
        before
            == [
                gu.to_bytes()?,
                gate.to_bytes()?,
                up.to_bytes()?,
                scale.to_bytes()?
            ]
    );
    moe.drain()?;
    println!(
        "T04_CAPTURE captures={captures} rows=20 real_gu=true values_per_capture=12800 capture_bytes={total} cumulative_limit_bytes=1073741824 parity=byte_exact dispatch_inspection=pending"
    );
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
fn actual_swiglu_scaled_mlx_entry() {
    // Only rows 1..2; actual production Array/FFI entry, no verifier stub.
    let rows: i32 = kani::any();
    kani::assume((1..=2).contains(&rows));
    let gate = Array::zeros_dtype(&[rows, WIDTH], Dtype::Float16).unwrap();
    let up = Array::zeros_dtype(&[rows, WIDTH], Dtype::Float16).unwrap();
    let scale = Array::zeros_dtype(&[rows, WIDTH], Dtype::Float16).unwrap();
    let output = gate.swiglu_scaled(&up, &scale).unwrap();
    assert_eq!(output.shape(), [rows, WIDTH]);
    assert_eq!(output.dtype(), Dtype::Float16);
    kani::cover!(rows == 1);
    kani::cover!(rows == 2);
}
