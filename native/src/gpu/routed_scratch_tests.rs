//! Physical integration gate: actual routed records, independent stock words,
//! actual allocations/bindings, owners dropped before command submission.
use super::*;
use crate::tensor_layout::MatrixLayout;
use serde::Deserialize;
use std::{collections::BTreeSet, fs, os::unix::fs::FileExt, path::PathBuf};

#[derive(Deserialize)]
pub(super) struct Case {
    path: PathBuf,
    prefix: String,
    x: Vec<u16>,
}

pub(super) fn record(gpu: &Metal, case: &Case) -> Result<Buffer> {
    let c = crate::checkpoint::inspect(&case.path)?;
    let (prefix, index) = case
        .prefix
        .rsplit_once(".experts.")
        .context("routed prefix required")?;
    let records = crate::checkpoint::flash_expert_records(&c, prefix)?;
    let info = records
        .get(index.parse::<usize>()?)
        .context("missing record")?;
    let (file, offset) = info.source()?;
    let mut bytes = vec![0; flash_support::EXPERT_BYTES as usize];
    file.read_exact_at(&mut bytes, offset)?;
    upload(&gpu.device, &bytes)
}

pub(super) fn input(gpu: &Metal, case: &Case, fp32: bool) -> Result<MatrixView> {
    ensure!(case.x.len() == 2560);
    let mut values = vec![f16::from_f32(7.).to_bits(); 2564];
    for (index, &value) in case.x.iter().enumerate() {
        values[2561 - index] = value;
    }
    let storage = if fp32 {
        gpu.f32_storage(
            &values
                .iter()
                .map(|&v| f16::from_bits(v).to_f32())
                .collect::<Vec<_>>(),
        )?
    } else {
        gpu.f16_storage(&values)?
    };
    storage.matrix(MatrixLayout {
        shape: [1, 2560],
        strides: [2564, -1],
        offset: 2561,
    })
}

fn requests() -> usize {
    super::super::BUFFER_REQUESTS.with(std::cell::Cell::get)
}

fn check_bindings(gpu: &Metal, before: usize, reuse: bool) -> Result<[usize; 2]> {
    ensure!(
        gpu.flash_scratch_uses.get() - before == if reuse { 3 } else { 0 },
        "wrong scratch use count"
    );
    let ids = gpu.flash_expert_outputs.get();
    ensure!(
        ids.iter().all(|&id| id != 0) && ids.into_iter().collect::<BTreeSet<_>>().len() == 4,
        "expert destinations alias"
    );
    if !reuse {
        return Ok([0; 2]);
    }
    let pairs = gpu.flash_scratch_bindings.get();
    ensure!(
        pairs.iter().all(|&pair| pair == pairs[0])
            && pairs[0][0] != 0
            && pairs[0][1] != 0
            && pairs[0][0] != pairs[0][1]
            && ids.iter().all(|id| !pairs[0].contains(id)),
        "scratch buffers were not reused independently of destinations"
    );
    Ok(pairs[0])
}

pub(super) fn check_words(actual: &[u16], expected: &[u16]) -> Result<()> {
    ensure!(
        actual.len() == 2560
            && actual == expected
            && expected.iter().any(|&v| f16::from_bits(v).to_f32() != 0.)
            && actual.iter().all(|&v| f16::from_bits(v).is_finite()),
        "routed expert stock divergence"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; independent real expert fixtures required"]
fn physical_routed_expert_scratch() -> Result<()> {
    let root = PathBuf::from(
        std::env::var_os("MLXL3_ROUTED_SCRATCH_FIXTURES").context("fixtures required")?,
    );
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(root.join("cases.json"))?)?;
    let refs: Vec<Vec<u16>> = serde_json::from_slice(&fs::read(root.join("stock.json"))?)?;
    ensure!(
        cases.len() >= 5
            && cases.len() == refs.len()
            && cases[0].prefix != cases[1].prefix
            && cases[..cases.len() - 1]
                .iter()
                .map(|c| &c.prefix)
                .collect::<BTreeSet<_>>()
                .len()
                >= 2,
        "at least two experts and varied inputs required"
    );
    ensure!(
        cases[0].x != cases[2].x,
        "replay needs different activation values"
    );
    ensure!(
        cases
            .last()
            .context("missing shared case")?
            .prefix
            .ends_with(".shared_expert"),
        "last case must be the shared expert"
    );
    let mut gpu = Metal::new()?;
    let snapshot = gpu.flash_expert_scratch_setting;
    ensure!(
        snapshot
            == crate::flash_support::hc_chain_flag(
                std::env::var("MLXL3_NATIVE_ROUTED_SCRATCH").ok().as_deref(),
                gpu.device_name() == "Apple M5"
            )
    );
    let mut checks = Vec::new();
    for reuse in [false, true] {
        gpu.flash_expert_scratch_setting = Some(reuse);
        for (index, (case, expected)) in cases[..cases.len() - 1].iter().zip(&refs).enumerate() {
            for fp32 in [false, true] {
                let backing = record(&gpu, case)?;
                let expert = FlashExpert::from_backing(&gpu, &backing)?;
                let x = input(&gpu, case, fp32)?;
                let before = requests();
                let uses = gpu.flash_scratch_uses.get();
                let output = gpu.command(|cmd| {
                    let output = expert.encode_view(cmd, &x, Storage::StorageModeShared)?;
                    drop(x);
                    drop(expert);
                    drop(backing);
                    Ok(output)
                })?;
                let count = requests() - before;
                ensure!(
                    count == if reuse { 6 } else { 10 },
                    "actual newBuffer count: {count}"
                );
                let pair = check_bindings(&gpu, uses, reuse)?;
                check_words(&Metal::read(&output, 2560), expected)?;
                checks.push(serde_json::json!({"case":index,"fp32":fp32,"reuse":reuse,"actual_new_buffer":count,"pair":pair}));
            }
        }
    }
    gpu.flash_expert_scratch_setting = Some(true);
    // Three encodes in a single command: two different experts and one duplicate.
    let mut pairs = BTreeSet::new();
    let outputs = gpu.command(|cmd| {
        [0, 1, 0]
            .into_iter()
            .map(|index| {
                let case = &cases[index];
                let backing = record(&gpu, case)?;
                let expert = FlashExpert::from_backing(&gpu, &backing)?;
                let x = input(&gpu, case, true)?;
                let before = requests();
                let uses = gpu.flash_scratch_uses.get();
                let output = expert.encode_view(cmd, &x, Storage::StorageModeShared)?;
                ensure!(requests() - before == 6);
                let pair = check_bindings(&gpu, uses, true)?;
                ensure!(
                    pairs.insert(pair[0]) && pairs.insert(pair[1]),
                    "scratch shared across experts"
                );
                Ok(output)
            })
            .collect::<Result<Vec<_>>>()
    })?;
    for (output, index) in outputs.iter().zip([0, 1, 0]) {
        check_words(&Metal::read(output, 2560), &refs[index])?;
    }
    // Poison both oversized temporary backings, then execute all three real
    // projections. The down stage must overwrite and consume only its prefixes.
    let backing = record(&gpu, &cases[2])?;
    let expert = FlashExpert::from_backing(&gpu, &backing)?;
    let x = upload(&gpu.device, &cases[2].x)?;
    let prepared = upload(&gpu.device, &vec![0xffu8; 5120])?;
    let partial = upload(&gpu.device, &vec![0xffu8; 10240])?;
    let before_down = buffer(&gpu.device, 5120, Storage::StorageModeShared)?;
    let output = gpu.command(|cmd| {
        let scratch = Some((&prepared, &partial));
        let gate = expert.gate.encode_with_scratch(
            cmd,
            Activation::Contiguous(&x),
            crate::flash_expert_scratch::EXTENTS[0],
            Storage::StorageModePrivate,
            scratch,
        )?;
        let up = expert.up.encode_with_scratch(
            cmd,
            Activation::Contiguous(&x),
            crate::flash_expert_scratch::EXTENTS[1],
            Storage::StorageModePrivate,
            scratch,
        )?;
        let blit = cmd.new_blit_command_encoder();
        blit.copy_from_buffer(&prepared, 0, &before_down, 0, 5120);
        blit.end_encoding();
        let activated = gpu.encode_swiglu(cmd, &gate, &up, Storage::StorageModePrivate)?;
        expert.down.encode_with_scratch(
            cmd,
            Activation::Contiguous(&activated),
            crate::flash_expert_scratch::EXTENTS[2],
            Storage::StorageModeShared,
            scratch,
        )
    })?;
    check_words(&Metal::read(&output, 2560), &refs[2])?;
    ensure!(
        Metal::read(&prepared, 2560)[640..] == Metal::read(&before_down, 2560)[640..],
        "down overwrote the prepared tail beyond its640values"
    );
    let cmd = gpu.queue.new_command_buffer();
    ensure!(
        expert
            .gate
            .encode_with_scratch(
                cmd,
                Activation::Contiguous(&x),
                crate::flash_expert_scratch::EXTENTS[0],
                Storage::StorageModeShared,
                Some((&partial, &partial))
            )
            .is_err()
    );
    ensure!(cmd.status() == MTLCommandBufferStatus::NotEnqueued);
    // Capacity rejections happen before any encoder; a subsequent valid expert
    // must still work. A complete expert encoded before an injected error is
    // never submitted.
    for capacities in [[5119, 10240], [5120, 2559], [0, 10240]] {
        let short = if capacities[0] == 0 {
            prepared.clone()
        } else {
            buffer(&gpu.device, capacities[0], Storage::StorageModePrivate)?
        };
        let second = buffer(&gpu.device, capacities[1], Storage::StorageModePrivate)?;
        let cmd = gpu.queue.new_command_buffer();
        let extent = if capacities[0] == 0 {
            [0, 640, 640]
        } else {
            crate::flash_expert_scratch::EXTENTS[0]
        };
        ensure!(
            expert
                .gate
                .encode_with_scratch(
                    cmd,
                    Activation::Contiguous(&x),
                    extent,
                    Storage::StorageModeShared,
                    Some((&short, &second))
                )
                .is_err()
        );
        ensure!(cmd.status() == MTLCommandBufferStatus::NotEnqueued);
    }
    let mut observed = None;
    let rejected = submit_encoded(&gpu, |cmd| {
        let _ = expert.encode(
            cmd,
            Activation::Contiguous(&x),
            crate::flash_expert_scratch::EXTENTS,
            Storage::StorageModeShared,
        )?;
        observed = Some(cmd.to_owned());
        bail!("injected expert before commit")
    });
    ensure!(
        rejected.is_err()
            && observed.context("missing failure command")?.status()
                == MTLCommandBufferStatus::NotEnqueued
    );
    ensure!(expert.submit(&[]).is_err());
    check_words(&expert.submit(&cases[2].x)?.read()?, &refs[2])?;
    // A malformed flag rejects only the eligible record path. Shared and batch
    // inputs continue through the original path, with no scratch uses.
    drop(expert);
    gpu.flash_expert_scratch_setting = None;
    let expert = FlashExpert::from_backing(&gpu, &backing)?;
    ensure!(expert.submit(&cases[2].x).is_err());
    let repeated = [cases[2].x.as_slice(), cases[2].x.as_slice()].concat();
    let uses = gpu.flash_scratch_uses.get();
    let batch = expert.submit(&repeated)?.read()?;
    ensure!(
        gpu.flash_scratch_uses.get() == uses
            && batch == [refs[2].as_slice(), refs[2].as_slice()].concat()
    );
    let c = crate::checkpoint::inspect(&cases[0].path)?;
    let shared = cases.last().context("missing shared case")?;
    let ordinary = FlashExpert::load(&gpu, &c, &shared.prefix)?;
    check_words(
        &ordinary.submit(&shared.x)?.read()?,
        refs.last().context("missing shared reference")?,
    )?;
    ensure!(gpu.flash_scratch_uses.get() == uses);
    fs::write(
        root.join("native-scratch.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"parity":true,"cases":checks,"duplicate_encodes":3,"poison":true,"rejections_recovered":true,"fallbacks":true}),
        )?,
    )?;
    Ok(())
}
