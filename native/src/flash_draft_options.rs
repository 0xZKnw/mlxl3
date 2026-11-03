//! Shared validation for native and reference Flash draft modes.
use anyhow::{Context, Result, ensure};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Options {
    pub(crate) q4: bool,
    pub(crate) ring: bool,
    pub(crate) adaptive: bool,
    pub(crate) precap: bool,
    pub(crate) width: usize,
}
impl Options {
    pub(crate) fn read(mode: usize) -> Result<Self> {
        ensure!((1..=3).contains(&mode), "DFlash mode must be 1, 2 or 3");
        let flag = |name: &str, default: bool| -> Result<bool> {
            match std::env::var(name) {
                Ok(v) if v == "0" => Ok(false),
                Ok(v) if v == "1" => Ok(true),
                Err(std::env::VarError::NotPresent) => Ok(default),
                _ => anyhow::bail!("{name} must be 0 or 1"),
            }
        };
        let adaptive_requested = flag("MLXL3_FLASH_DRAFT_ADAPTIVE", true)?;
        let precap_requested = flag("MLXL3_FLASH_DRAFT_PRECAP", false)?;
        ensure!(
            !precap_requested || adaptive_requested,
            "MLXL3_FLASH_DRAFT_PRECAP requires MLXL3_FLASH_DRAFT_ADAPTIVE=1"
        );
        let adaptive = adaptive_requested && mode == 1;
        let precap = precap_requested && adaptive;
        let width = match std::env::var("MLXL3_FLASH_DRAFT_WIDTH") {
            Ok(value) => crate::flash_support::fixed_draft_width(&value, adaptive_requested)
                .context(
                    "MLXL3_FLASH_DRAFT_WIDTH requires 1..7 and MLXL3_FLASH_DRAFT_ADAPTIVE=0",
                )?,
            Err(std::env::VarError::NotPresent) => {
                if precap {
                    3
                } else if adaptive || mode == 3 {
                    7
                } else if mode == 2 {
                    2
                } else {
                    3
                }
            }
            Err(_) => anyhow::bail!("MLXL3_FLASH_DRAFT_WIDTH must be valid Unicode"),
        };
        Ok(Self {
            q4: flag("MLXL3_FLASH_DRAFT_Q4", true)?,
            ring: flag("MLXL3_FLASH_DRAFT_RING", true)?,
            adaptive,
            precap,
            width,
        })
    }
}
