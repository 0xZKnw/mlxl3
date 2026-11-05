//! Preserve the Flash GDN sigmoid gate after its native FP32 normalization.

const HEADS: i32 = 48;
const HEAD_DIM: i32 = 128;
const WIDTH: i32 = HEADS * HEAD_DIM;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub(crate) rows: i32,
    pub(crate) elements: i32,
}

pub(crate) fn geometry(rows: usize, normed_shape: &[i32], z_shape: &[i32]) -> Option<Geometry> {
    if !(1..=8).contains(&rows)
        || normed_shape != [1, 1, HEADS, HEAD_DIM]
        || z_shape != [1, rows as i32, HEADS, HEAD_DIM]
    {
        return None;
    }
    let elements = rows.checked_mul(WIDTH as usize)?;
    if elements > i32::MAX as usize || elements > u32::MAX as usize {
        return None;
    }
    Some(Geometry {
        rows: rows as i32,
        elements: elements as i32,
    })
}

#[cfg(feature = "mlx")]
use crate::array::{self, Array, Dtype};
#[cfg(feature = "mlx")]
use anyhow::{Context, Result, ensure};

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
pub(crate) fn apply(normed: &[Array], z: &Array) -> Result<Array> {
    let rows = normed.len();
    ensure!(
        (1..=8).contains(&rows)
            && z.dtype() == Dtype::Float16
            && normed.iter().all(|value| value.dtype() == Dtype::Float32),
        "invalid GDN epilogue row count or dtype"
    );
    let layout =
        geometry(rows, normed[0].shape(), z.shape()).context("invalid GDN epilogue geometry")?;
    ensure!(
        normed
            .iter()
            .all(|value| geometry(rows, value.shape(), z.shape()) == Some(layout)),
        "invalid GDN epilogue normalized row shape"
    );

    const NAMES: [&str; 8] = [
        "normed0", "normed1", "normed2", "normed3", "normed4", "normed5", "normed6", "normed7",
    ];
    let mut names = NAMES[..rows].to_vec();
    names.push("z");
    let mut inputs = normed.iter().collect::<Vec<_>>();
    inputs.push(z);
    #[cfg(all(test, feature = "mlx"))]
    {
        use std::sync::atomic::Ordering::Relaxed;
        LAST_ROWS.store(rows, Relaxed);
        KERNEL_REQUESTS.fetch_add(1, Relaxed);
    }
    let mut output = array::metal_kernel(
        "mlxl3_flash_gdn_epilogue_v1",
        &names,
        &["output"],
        &format!("#define ROWS {}u\n#define WIDTH {WIDTH}u\n", layout.rows),
        include_str!("../shaders/flash_gdn_epilogue.metal"),
        &inputs,
        &[vec![layout.rows, WIDTH]],
        &[Dtype::Float16],
        [layout.elements, 1, 1],
        [128, 1, 1],
    )?;
    output.pop().context("missing GDN epilogue output")
}

#[cfg(test)]
mod geometry_tests {
    use super::*;

    #[test]
    fn geometry_preserves_each_supported_row_count() {
        for rows in 1..=8 {
            assert_eq!(
                geometry(rows, &[1, 1, 48, 128], &[1, rows as i32, 48, 128]),
                Some(Geometry {
                    rows: rows as i32,
                    elements: rows as i32 * 6144,
                })
            );
        }
    }

    #[test]
    fn geometry_rejects_invalid_counts_ranks_and_dimensions() {
        for rows in [0, 9, usize::MAX] {
            assert!(geometry(rows, &[1, 1, 48, 128], &[1, 1, 48, 128]).is_none());
        }
        for shape in [
            &[][..],
            &[6144][..],
            &[1, 48, 128][..],
            &[1, 1, 48, 128, 1][..],
            &[0, 1, 48, 128][..],
            &[1, 2, 48, 128][..],
            &[1, 1, 47, 128][..],
            &[1, 1, 48, 127][..],
            &[1, 1, 48, -128][..],
            &[1, 1, i32::MAX, 128][..],
        ] {
            assert!(geometry(1, shape, &[1, 1, 48, 128]).is_none());
            assert!(geometry(1, &[1, 1, 48, 128], shape).is_none());
        }
        assert!(geometry(2, &[1, 1, 48, 128], &[1, 1, 48, 128]).is_none());
        assert!(geometry(1, &[1, 1, 48, 128], &[1, 2, 48, 128]).is_none());
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::solver(kissat)]
// Four-i32 shape equality uses a 16-byte builtin memcmp loop.
#[kani::unwind(17)]
fn gdn_epilogue_geometry_checks_symbolic_shapes_and_addresses() {
    let rows = kani::any::<usize>();
    let normed: [i32; 5] = kani::any();
    let z: [i32; 5] = kani::any();
    let normed_rank = kani::any::<usize>();
    let z_rank = kani::any::<usize>();
    if normed_rank <= normed.len() && z_rank <= z.len() {
        let layout = geometry(rows, &normed[..normed_rank], &z[..z_rank]);
        if let Some(layout) = layout {
            assert!((1..=8).contains(&rows));
            assert_eq!(normed_rank, 4);
            assert_eq!(z_rank, 4);
            assert_eq!(&normed[..4], &[1, 1, 48, 128]);
            assert_eq!(&z[..4], &[1, rows as i32, 48, 128]);
            assert_eq!(layout.rows as usize, rows);
            assert_eq!(layout.elements as usize, rows * 6144);
            assert!((layout.elements as usize) <= 49152);
            let index = kani::any::<u32>();
            if index < layout.elements as u32 {
                let row = index / 6144;
                let column = index % 6144;
                assert!((row as usize) < rows);
                assert!(column < 6144);
                assert_eq!(row * 6144 + column, index);
                assert!(u64::from(column) * 4 + 3 < 6144 * 4);
                assert!(u64::from(index) * 2 + 1 < layout.elements as u64 * 2);
                kani::cover!(rows == 8 && index == 49151);
            }
        }
        kani::cover!(layout.is_some() && rows == 1);
        kani::cover!(layout.is_some() && rows == 2);
        kani::cover!(layout.is_some() && rows == 4);
        kani::cover!(layout.is_some() && rows == 8);
        kani::cover!(layout.is_none() && rows == 0);
        kani::cover!(layout.is_none() && rows == usize::MAX);
        kani::cover!(layout.is_none() && normed_rank == 5);
        kani::cover!(layout.is_none() && z_rank == 4 && z[3] == -128);
    }
}

#[cfg(all(test, feature = "mlx"))]
pub(crate) mod capture_tests;
#[cfg(all(test, feature = "mlx"))]
mod tests;
