//! GPU discovery output parsing and backend recommendation.

use crate::Backend;
use serde::Serialize;

/// Minimum Windows driver for CUDA 12.8 without minor-version compatibility.
pub const CUDA_128_MIN_DRIVER: (u32, u32) = (570, 65);

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NvidiaGpu {
    pub name: String,
    pub memory_mib: u64,
    pub driver: String,
    pub driver_ok: bool,
}

/// Parse `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits`.
pub fn parse_nvidia_smi(output: &str) -> Vec<NvidiaGpu> {
    output
        .lines()
        .filter_map(|line| {
            let mut fields = line.rsplitn(3, ',').map(str::trim);
            let driver = fields.next()?.to_owned();
            let memory_mib = fields.next()?.parse().ok()?;
            let name = fields.next()?.to_owned();
            if name.is_empty() || driver.is_empty() {
                return None;
            }
            let driver_ok = driver_version(&driver).is_some_and(|v| v >= CUDA_128_MIN_DRIVER);
            Some(NvidiaGpu {
                name,
                memory_mib,
                driver,
                driver_ok,
            })
        })
        .collect()
}

fn driver_version(text: &str) -> Option<(u32, u32)> {
    let mut parts = text.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor))
}

/// AMD adapters among `Win32_VideoController` names (one per line).
pub fn amd_adapters(names: &str) -> Vec<String> {
    names
        .lines()
        .map(str::trim)
        .filter(|n| {
            let lower = n.to_ascii_lowercase();
            lower.contains("radeon") || lower.starts_with("amd ")
        })
        .map(str::to_owned)
        .collect()
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Recommendation {
    pub backend: Option<Backend>,
    pub reason: String,
}

pub fn recommend(nvidia: &[NvidiaGpu], amd: &[String], wsl_distros: &[String]) -> Recommendation {
    if let Some(gpu) = nvidia.first() {
        let reason = if gpu.driver_ok {
            format!(
                "{} detected; ExLlamaV3 CUDA runs natively on Windows",
                gpu.name
            )
        } else {
            format!(
                "{} detected with driver {}; CUDA 12.8 wants R{}.{} or newer, update the NVIDIA driver if loading fails",
                gpu.name, gpu.driver, CUDA_128_MIN_DRIVER.0, CUDA_128_MIN_DRIVER.1
            )
        };
        return Recommendation {
            backend: Some(Backend::Cuda),
            reason,
        };
    }
    if let Some(name) = amd.first() {
        let reason = if wsl_distros.is_empty() {
            format!(
                "{name} detected; AMD support runs ExLlamaV3 ROCm inside WSL2: install Ubuntu with `wsl --install -d Ubuntu-24.04` first"
            )
        } else {
            format!(
                "{name} detected; ExLlamaV3 ROCm runs inside WSL2 ({})",
                wsl_distros[0]
            )
        };
        return Recommendation {
            backend: Some(Backend::RocmWsl),
            reason,
        };
    }
    Recommendation {
        backend: None,
        reason: "no NVIDIA or AMD GPU detected; ExLlamaV3 needs a CUDA or ROCm GPU".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_smi_lines() {
        let gpus = parse_nvidia_smi(
            "NVIDIA GeForce RTX 4090, 24564, 576.02\nNVIDIA RTX A6000, Ada, 49140, 552.12\nbad line\n,1,2\n",
        );
        assert_eq!(gpus.len(), 2);
        assert_eq!(
            (gpus[0].name.as_str(), gpus[0].memory_mib, gpus[0].driver_ok),
            ("NVIDIA GeForce RTX 4090", 24564, true)
        );
        assert_eq!(
            (gpus[1].name.as_str(), gpus[1].driver_ok),
            ("NVIDIA RTX A6000, Ada", false)
        );
        assert!(parse_nvidia_smi("X, 1, 570.65").first().unwrap().driver_ok);
        assert!(!parse_nvidia_smi("X, 1, 570.64").first().unwrap().driver_ok);
        assert!(!parse_nvidia_smi("X, 1, weird").first().unwrap().driver_ok);
    }

    #[test]
    fn recommendation_order() {
        let nvidia = parse_nvidia_smi("RTX, 8192, 580.1");
        let amd = amd_adapters(
            "AMD Radeon RX 7900 XTX\nMicrosoft Basic Display Adapter\nAMD Radeon(TM) Graphics",
        );
        assert_eq!(amd.len(), 2);
        assert_eq!(recommend(&nvidia, &amd, &[]).backend, Some(Backend::Cuda));
        let rec = recommend(&[], &amd, &[]);
        assert_eq!(rec.backend, Some(Backend::RocmWsl));
        assert!(rec.reason.contains("wsl --install"));
        assert!(
            recommend(&[], &amd, &["Ubuntu".into()])
                .reason
                .contains("Ubuntu")
        );
        assert_eq!(
            recommend(&[], &amd_adapters("Intel(R) Arc(TM)"), &[]).backend,
            None
        );
    }
}
