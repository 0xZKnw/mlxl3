//! Exact S2 finish for the Flash 6144-to-2560 dense projections.

const WIDTH: i32 = 2560;
const BLOCK: i32 = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub(crate) rows: i32,
    pub(crate) blocks: i32,
}

pub(crate) fn enabled_setting(value: Option<&str>) -> Option<bool> {
    match value {
        None | Some("0") => Some(false),
        Some("1") => Some(true),
        _ => None,
    }
}

pub(crate) fn geometry(
    rows: i32,
    partial_shape: &[i32],
    scale_shape: &[i32],
    partial_f32: bool,
    scale_f16: bool,
) -> Option<Geometry> {
    if !partial_f32 || !scale_f16 || scale_shape != [WIDTH] {
        return None;
    }
    let valid = match rows {
        1 => partial_shape == [2, WIDTH],
        2 => partial_shape == [2, 2, WIDTH],
        _ => false,
    };
    if !valid {
        return None;
    }
    Some(Geometry {
        rows,
        blocks: rows.checked_mul(WIDTH / BLOCK)?,
    })
}

#[cfg(feature = "mlx")]
use crate::array::{self, Array, Dtype};
#[cfg(feature = "mlx")]
use anyhow::{Context, Result};

#[cfg(all(test, feature = "mlx"))]
static KERNEL_REQUESTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(all(test, feature = "mlx"))]
static LAST_ROWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(test, feature = "mlx"))]
pub(crate) fn reset_kernel_requests_for_test() {
    use std::sync::atomic::Ordering::Relaxed;
    KERNEL_REQUESTS.store(0, Relaxed);
    LAST_ROWS.store(0, Relaxed);
}
#[cfg(all(test, feature = "mlx"))]
pub(crate) fn kernel_requests_for_test() -> usize {
    KERNEL_REQUESTS.load(std::sync::atomic::Ordering::Relaxed)
}
#[cfg(all(test, feature = "mlx"))]
pub(crate) fn last_rows_for_test() -> usize {
    LAST_ROWS.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "mlx")]
pub(crate) fn apply(partials: &Array, svh: &Array, rows: i32) -> Result<Option<Array>> {
    let Some(layout) = geometry(
        rows,
        partials.shape(),
        svh.shape(),
        partials.dtype() == Dtype::Float32,
        svh.dtype() == Dtype::Float16,
    ) else {
        return Ok(None);
    };
    let scale = (1.0 / 128.0f32.sqrt()).to_bits();
    let output = array::metal_kernel(
        "mlxl3_flash_dense_epilogue_s2_v1",
        &["partials", "svh"],
        &["output"],
        &format!("#define WIDTH {WIDTH}u\n#define SCALE_BITS 0x{scale:08x}u\n"),
        include_str!("../shaders/flash_dense_epilogue.metal"),
        &[partials, svh],
        &[vec![layout.rows, WIDTH]],
        &[Dtype::Float16],
        [8, layout.blocks, 1],
        [8, 1, 1],
    )?
    .pop()
    .context("missing dense epilogue output")?;
    #[cfg(all(test, feature = "mlx"))]
    {
        use std::sync::atomic::Ordering::Relaxed;
        KERNEL_REQUESTS.fetch_add(1, Relaxed);
        LAST_ROWS.store(layout.rows as usize, Relaxed);
    }
    Ok(Some(output))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_epilogue_scale_matches_stock_h128_bits() {
        assert_eq!((1.0 / 128.0f32.sqrt()).to_bits(), 0x3db5_04f3);
    }

    #[test]
    fn dense_epilogue_setting_accepts_only_zero_one_or_default() {
        assert_eq!(enabled_setting(None), Some(false));
        assert_eq!(enabled_setting(Some("0")), Some(false));
        assert_eq!(enabled_setting(Some("1")), Some(true));
        for value in [
            "", "true", "false", "01", "00", "+1", "-1", "2", " 1", "1 ", "١",
        ] {
            assert_eq!(enabled_setting(Some(value)), None);
        }
    }

    #[test]
    fn dense_epilogue_geometry_accepts_only_the_two_production_layouts() {
        assert_eq!(
            geometry(1, &[2, 2560], &[2560], true, true),
            Some(Geometry {
                rows: 1,
                blocks: 20
            })
        );
        assert_eq!(
            geometry(2, &[2, 2, 2560], &[2560], true, true),
            Some(Geometry {
                rows: 2,
                blocks: 40
            })
        );
        for rows in [i32::MIN, -1, 0, 3, 8, i32::MAX] {
            assert!(geometry(rows, &[2, 2560], &[2560], true, true).is_none());
        }
        for shape in [
            &[][..],
            &[5120][..],
            &[1, 2, 2560][..],
            &[2, 2560, 1][..],
            &[2, 2, 2559][..],
            &[2, 2, 2560, 1][..],
            &[0, 2560][..],
            &[-2, 2560][..],
            &[2, i32::MAX][..],
        ] {
            assert!(geometry(1, shape, &[2560], true, true).is_none());
            assert!(geometry(2, shape, &[2560], true, true).is_none());
        }
        for shape in [
            &[][..],
            &[1, 2560][..],
            &[2561][..],
            &[-2560][..],
            &[i32::MAX][..],
        ] {
            assert!(geometry(1, &[2, 2560], shape, true, true).is_none());
        }
        for flags in [(false, false), (false, true), (true, false)] {
            assert!(geometry(1, &[2, 2560], &[2560], flags.0, flags.1).is_none());
        }
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::solver(kissat)]
#[kani::unwind(17)]
fn dense_epilogue_geometry_bounds_real_launch_and_addresses() {
    let rows: i32 = kani::any();
    let partial: [i32; 5] = kani::any();
    let scale: [i32; 3] = kani::any();
    let rank: usize = kani::any();
    let scale_rank: usize = kani::any();
    let partial_f32: bool = kani::any();
    let scale_f16: bool = kani::any();
    if rank <= partial.len() && scale_rank <= scale.len() {
        let layout = geometry(
            rows,
            &partial[..rank],
            &scale[..scale_rank],
            partial_f32,
            scale_f16,
        );
        let expected = partial_f32
            && scale_f16
            && scale_rank == 1
            && scale[0] == 2560
            && ((rows == 1 && rank == 2 && partial[0] == 2 && partial[1] == 2560)
                || (rows == 2
                    && rank == 3
                    && partial[0] == 2
                    && partial[1] == 2
                    && partial[2] == 2560));
        assert_eq!(layout.is_some(), expected);
        if let Some(layout) = layout {
            assert!((1..=2).contains(&rows));
            assert_eq!(layout.rows, rows);
            assert_eq!(layout.blocks, rows * 20);
            let index: u32 = kani::any();
            if index < rows as u32 * 2560 {
                let row = index / 2560;
                let column = index % 2560;
                let p0 = row * 5120 + column;
                let p1 = p0 + 2560;
                assert!(p1 < rows as u32 * 5120);
                assert!(u64::from(p0) * 4 + 3 < rows as u64 * 5120 * 4);
                assert!(u64::from(p1) * 4 + 3 < rows as u64 * 5120 * 4);
                assert!(u64::from(column) * 2 + 1 < 2560 * 2);
                assert!(u64::from(index) * 2 + 1 < rows as u64 * 2560 * 2);
                kani::cover!(rows == 2 && index == 5119);
            }
            let block: u32 = kani::any();
            let thread: u32 = kani::any();
            let chunk: u32 = kani::any();
            let element: u32 = kani::any();
            if block < layout.blocks as u32 && thread < 8 && chunk < 4 && element < 4 {
                let local = chunk * 32 + thread * 4 + element;
                assert!(local < 128);
                assert_eq!(local / 32, chunk);
                assert_eq!((local % 32) / 4, thread);
                assert_eq!(local % 4, element);
                assert!(block * 128 + local < rows as u32 * 2560);
                assert!(block / 20 < rows as u32);
                kani::cover!(rows == 2 && block == 39 && local == 127);
            }
        }
        kani::cover!(layout.is_some() && rows == 1);
        kani::cover!(layout.is_some() && rows == 2);
        kani::cover!(layout.is_none() && rows == 0);
        kani::cover!(layout.is_none() && rank == 5);
        kani::cover!(layout.is_none() && scale_rank == 0);
        kani::cover!(layout.is_none() && !partial_f32);
        kani::cover!(layout.is_none() && !scale_f16);
    }
}
