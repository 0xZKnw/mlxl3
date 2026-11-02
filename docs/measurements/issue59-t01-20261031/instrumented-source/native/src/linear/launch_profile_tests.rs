//! T01-P-01: host graph construction only; never evaluates or claims numerical parity.
use super::*;
use crate::kernel_build_profile::{self as profile, Mode, Stats, TARGET};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    fs::File,
    io::{Seek, SeekFrom, Write},
    path::Path,
    time::Instant,
};

const CALLS: u64 = 256;

fn initial_report() -> Value {
    json!({"status":"incomplete","scope":"host_graph_construction_only",
        "numerical_parity":"not_evaluated","gpu_execution":"not_requested",
        "model_gain":"not_measured","completed_lots":0,"samples":[],
        "observer":"whole loop includes builder, drop, shape/dtype/pending checks; no arbitrary time subtraction",
        "rust_phases":["validation","cstrings","marshalling","initialize","ffi","adoption","total"],
        "cpp_fields":["valid","calls","misses","evictions","lookup_ns","factory_ns","output_metadata_ns","invoke_ns","handles_ns","total_ns","header_bytes","source_bytes","max_entries"]})
}

fn save(file: &mut File, report: &Value) -> Result<()> {
    file.seek(SeekFrom::Start(0))?;
    file.set_len(0)?;
    serde_json::to_writer_pretty(&mut *file, report)?;
    file.flush()?;
    Ok(())
}

fn create_report(path: &Path) -> Result<File> {
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(path)
        .context("report must have an existing parent and an absent output path")?;
    save(&mut file, &initial_report())?;
    Ok(file)
}

struct Collectors {
    active: bool,
    trace: bool,
}
impl Collectors {
    fn disable(&mut self) -> Result<()> {
        profile::configure(Mode::Off);
        let cpp = crate::array::kernel_build_profile_enable(false);
        let trace = if self.trace {
            self.trace = false;
            crate::flash_profile::finish().map(|_| ())
        } else {
            Ok(())
        };
        self.active = cpp.is_err() || trace.is_err();
        cpp?;
        trace
    }
}
impl Drop for Collectors {
    fn drop(&mut self) {
        if self.active
            && let Err(error) = self.disable()
        {
            eprintln!("T01 collector cleanup failed: {error:#}");
        }
    }
}

fn check_output(output: &Array) -> Result<()> {
    ensure!(
        output.shape() == [20, 640] && output.dtype() == Dtype::Float16,
        "unexpected real Hot GU output metadata"
    );
    ensure!(
        !output.is_available_for_test()?,
        "graph unexpectedly became evaluated"
    );
    Ok(())
}

fn validate(mode: Mode, rust: Stats, cpp: [u64; 13], header: usize, source: usize) -> Result<()> {
    ensure!(
        rust.valid && cpp[0] == 1,
        "collector overflow or invalid sample"
    );
    match mode {
        Mode::Off => ensure!(
            rust.calls == 0 && rust.ns == [0; 7] && cpp[1..] == [0; 12],
            "disabled collector changed counters"
        ),
        Mode::Coarse => ensure!(
            rust.calls == CALLS && rust.ns[6] > 0 && rust.ns[..6] == [0; 6] && cpp[1..] == [0; 12],
            "invalid coarse counters"
        ),
        Mode::Fine => {
            ensure!(
                rust.calls == CALLS
                    && rust.ns[6] > 0
                    && cpp[1] == CALLS
                    && cpp[2] == 0
                    && cpp[3] == 0
                    && cpp[5] == 0
                    && cpp[9] > 0
                    && (1..=128).contains(&cpp[12]),
                "fine lot must be nonempty, all-hit and bounded"
            );
            ensure!(
                cpp[10]
                    == CALLS
                        .checked_mul(u64::try_from(header)?)
                        .context("header count overflow")?
                    && cpp[11]
                        == CALLS
                            .checked_mul(u64::try_from(source)?)
                            .context("source count overflow")?,
                "observed signature byte counts differ"
            );
            ensure!(
                rust.ns[..6].iter().try_fold(0u64, |n, &x| n.checked_add(x)) == Some(rust.ns[6])
                    && cpp[4..9]
                        .iter()
                        .try_fold(0u64, |n, &x| n.checked_add(x))
                        .is_some_and(|n| n <= cpp[9])
                    && cpp[9] <= rust.ns[4],
                "invalid inclusive timing bounds"
            );
        }
    }
    Ok(())
}

fn label(mode: Mode) -> &'static str {
    match mode {
        Mode::Off => "off",
        Mode::Coarse => "coarse",
        Mode::Fine => "fine",
    }
}

fn run(file: &mut File, report: &mut Value) -> Result<()> {
    let mut cleanup = Collectors {
        active: true,
        trace: false,
    };
    let result = (|| -> Result<()> {
        profile::configure(Mode::Off);
        crate::array::kernel_build_profile_enable(false)?;
        let x = Array::from_f16_bits(&vec![0x3c00; 20 * 2560], &[20, 2560])?;
        let bank = Array::zeros_dtype(&[53, 2, 160, 40, 32], Dtype::UInt16)?;
        let suh = Array::from_f16_bits(&vec![0x3a00; 20 * 2560], &[20, 2560])?;
        let svh = Array::from_f16_bits(&vec![0x3800; 20 * 640], &[20, 640])?;
        let ids = Array::from_u32(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 511], &[10])?;
        let mut map = [u32::MAX; 512];
        for (id, slot) in map.iter_mut().enumerate().take(9) {
            *slot = (id * 5) as u32;
        }
        let slots = ExpertSlots::residents(&ids, &map, 53)?;
        let build = || {
            expert_slots(
                &x,
                &bank,
                &suh,
                &svh,
                &slots,
                640,
                2,
                2,
                Codebook::Mul1,
                false,
            )
        };
        let before_readbacks = Array::host_readbacks_for_test();

        profile::configure(Mode::Fine);
        crate::array::kernel_build_profile_enable(true)?;
        crate::flash_profile::start(false)?;
        cleanup.trace = true;
        let first = build()?;
        check_output(&first)?;
        drop(first);
        let trace = crate::flash_profile::finish()?;
        cleanup.trace = false;
        let trace = serde_json::to_value(trace)?;
        let signatures = trace
            .as_array()
            .context("trace is not an array")?
            .iter()
            .filter(|event| event["name"] == "expert_kernel_signature")
            .collect::<Vec<_>>();
        ensure!(
            signatures.len() == 1,
            "expected exactly one real GU signature"
        );
        let signature = signatures[0]["data"]["key"]
            .as_array()
            .context("missing signature key")?;
        ensure!(
            signature.len() == 3
                && signature[0] == TARGET
                && signatures[0]["data"]["stable"] == false,
            "wrong production GU signature or stable-shape path"
        );
        let header = signature[1].as_str().context("missing header")?;
        let source = signature[2].as_str().context("missing source")?;
        ensure!(
            !source.is_empty()
                && header.contains("#define INPUT_DIMS 2560\n")
                && header.contains("#define OUTPUT_TILES 40\n")
                && header.contains("#define ROUTING_REPEAT 2\n")
                && header.contains("#define N_SPLITS 1\n")
                && header.contains("#define MLXL3_QMV_NT 1u\n")
                && header.contains("#define MLXL3_QMV_SG 4u\n")
                && header.contains("#define MLXL3_SLOT_CAPACITY 53u\n"),
            "unexpected served GU geometry"
        );
        report["signature"] = json!({"name":TARGET,"header":header,"source":source,
            "input_shape":[20,2560],"bank_shape":[53,2,160,40,32],"output_shape":[20,640],
            "partial_shape":[1,20,640],"partial_dtype":"F32","grid":[102400,1,1],
            "threadgroup":[128,1,1],"synthetic_fixture":true,"values_not_computed":true});
        for _ in 0..63 {
            let output = build()?;
            check_output(&output)?;
            drop(output);
        }
        let warm_rust = profile::snapshot();
        let warm_cpp = crate::array::kernel_build_profile_copy()?;
        ensure!(
            warm_rust.valid
                && warm_rust.calls == 64
                && warm_cpp[0] == 1
                && warm_cpp[1] == 64
                && warm_cpp[2] == 1
                && warm_cpp[3] == 0,
            "standalone warmup must contain one initial miss and 63 hits"
        );
        ensure!(Array::host_readbacks_for_test() == before_readbacks);
        report["warmup"] =
            json!({"calls":64,"initial_misses":1,"rust_ns":warm_rust.ns,"cpp_raw":warm_cpp});
        save(file, report)?;

        for lot in 0..8 {
            let modes = if lot % 2 == 0 {
                [Mode::Off, Mode::Coarse, Mode::Fine]
            } else {
                [Mode::Fine, Mode::Coarse, Mode::Off]
            };
            for mode in modes {
                profile::configure(mode);
                ensure!(profile::mode_for(TARGET) == mode);
                crate::array::kernel_build_profile_enable(mode == Mode::Fine)?;
                let reads = Array::host_readbacks_for_test();
                let start = Instant::now();
                // Each graph is dropped here. Metadata checks are included equally in all modes.
                for _ in 0..CALLS {
                    let output = build()?;
                    check_output(&output)?;
                    drop(output);
                }
                let host_ns =
                    u64::try_from(start.elapsed().as_nanos()).context("host timer overflow")?;
                let rust = profile::snapshot();
                let cpp = crate::array::kernel_build_profile_copy()?;
                ensure!(host_ns > 0 && Array::host_readbacks_for_test() == reads);
                validate(mode, rust, cpp, header.len(), source.len())
                    .with_context(|| format!("lot={lot},mode={}", label(mode)))?;
                report["samples"]
                    .as_array_mut()
                    .context("missing samples")?
                    .push(json!({
                    "lot":lot,"mode":label(mode),"builder_calls":CALLS,"host_ns":host_ns,
                    "host_ns_per_call":host_ns as f64 / CALLS as f64,
                    "rust_valid":rust.valid,"rust_calls":rust.calls,"rust_ns":rust.ns,"cpp_raw":cpp,
                    "outputs_pending":true,"host_readbacks_delta":0}));
                save(file, report)?;
            }
            report["completed_lots"] = json!(lot + 1);
            save(file, report)?;
        }
        ensure!(
            report["samples"]
                .as_array()
                .context("missing samples")?
                .len()
                == 24
                && report["completed_lots"] == 8
                && Array::host_readbacks_for_test() == before_readbacks
        );
        report["measured_builder_calls"] = json!(6144);
        report["limits"] = json!([
            "No eval, async_eval, synchronize or data readback requested",
            "Header/source captured once outside measured lots; no fenced trace",
            "C++ timings are inside Rust ffi, not additive with it",
            "Off means test collectors disabled, not an uninstrumented release CLI",
            "No source hash, pipeline resolution, GPU encoder, SSD or token throughput measured",
            "Metadata availability queries are included in the external loop time"
        ]);
        Ok(())
    })();
    let disabled = cleanup.disable();
    result.and(disabled)
}

#[test]
#[ignore = "physical Apple Metal context; T01 host construction only, no graph evaluation"]
fn hot_gu_post_hit_construction_profile() -> Result<()> {
    let path = std::path::PathBuf::from(std::env::var("MLXL3_T01_REPORT")?);
    let mut file = create_report(&path)?;
    let mut report = initial_report();
    let result = run(&mut file, &mut report);
    report["status"] = json!(if result.is_ok() {
        "validated_host_only"
    } else {
        "failed"
    });
    if let Err(error) = &result {
        report["error"] = json!(format!("{error:#}"));
    }
    save(&mut file, &report)?;
    result?;
    println!(
        "T01_HOST_BUILD status=validated_host_only lots=8 modes=3 calls_per_sample=256 measured_calls=6144 warmup_calls=64 numerical_parity=not_evaluated gpu_execution=not_requested"
    );
    Ok(())
}

#[test]
fn launch_profile_report_refuses_overwrite_and_missing_parent() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("report.json");
    drop(create_report(&path)?);
    let before = std::fs::read(&path)?;
    ensure!(serde_json::from_slice::<Value>(&before)?["status"] == "incomplete");
    ensure!(create_report(&path).is_err() && std::fs::read(&path)? == before);
    ensure!(create_report(&directory.path().join("absent/report.json")).is_err());
    Ok(())
}

#[test]
fn launch_profile_validation_rejects_incomplete_or_overflowed_samples() -> Result<()> {
    let rust = Stats {
        valid: true,
        calls: CALLS,
        ns: [1, 2, 3, 4, 70, 4, 84],
    };
    let mut cpp = [0; 13];
    cpp[0] = 1;
    cpp[1] = CALLS;
    cpp[4] = 3;
    cpp[6] = 5;
    cpp[7] = 20;
    cpp[8] = 2;
    cpp[9] = 40;
    cpp[10] = CALLS * 11;
    cpp[11] = CALLS * 17;
    cpp[12] = 128;
    validate(Mode::Fine, rust, cpp, 11, 17)?;
    for (index, value) in [
        (0, 0),
        (1, 255),
        (2, 1),
        (3, 1),
        (5, 1),
        (9, 0),
        (10, 0),
        (11, 0),
        (12, 129),
    ] {
        let mut bad = cpp;
        bad[index] = value;
        ensure!(validate(Mode::Fine, rust, bad, 11, 17).is_err());
    }
    ensure!(
        validate(
            Mode::Fine,
            Stats {
                valid: false,
                ..rust
            },
            cpp,
            11,
            17
        )
        .is_err()
    );
    ensure!(validate(Mode::Coarse, rust, cpp, 11, 17).is_err());
    ensure!(validate(Mode::Off, rust, cpp, 11, 17).is_err());
    for ns in [
        [1, 2, 3, 4, 70, 4, 100],
        [u64::MAX; 7],
        [1, 2, 3, 4, 39, 4, 53],
    ] {
        ensure!(validate(Mode::Fine, Stats { ns, ..rust }, cpp, 11, 17).is_err());
    }
    Ok(())
}
