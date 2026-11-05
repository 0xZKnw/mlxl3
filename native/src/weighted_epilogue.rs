//! Select canonical Hot/Cold contributions without changing their weighted sum.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub(crate) rows: i32,
    pub(crate) width: i32,
    pub(crate) elements: i32,
}

pub(crate) fn geometry(
    hot_rows: usize,
    cold_rows: usize,
    width: usize,
    order: &[u32],
) -> Option<Geometry> {
    if !(1..=80).contains(&hot_rows)
        || !(1..=80).contains(&cold_rows)
        || !(1..=80).contains(&order.len())
        || width == 0
        || width > i32::MAX as usize
    {
        return None;
    }
    let sources = hot_rows.checked_add(cold_rows)?;
    let source_elements = sources.checked_mul(width)?;
    let elements = order.len().checked_mul(width)?;
    if source_elements > u32::MAX as usize
        || elements > i32::MAX as usize
        || order.iter().any(|&row| row as usize >= sources)
    {
        return None;
    }
    Some(Geometry {
        rows: order.len() as i32,
        width: width as i32,
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
static LAST_HOT_ROWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(all(test, feature = "mlx"))]
static LAST_COLD_ROWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(all(test, feature = "mlx"))]
static LAST_ORDER_ROWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(test, feature = "mlx"))]
pub(crate) fn reset_kernel_requests_for_test() {
    KERNEL_REQUESTS.store(0, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(all(test, feature = "mlx"))]
pub(crate) fn kernel_requests_for_test() -> usize {
    KERNEL_REQUESTS.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(all(test, feature = "mlx"))]
pub(crate) fn last_geometry_for_test() -> (usize, usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        LAST_HOT_ROWS.load(Relaxed),
        LAST_COLD_ROWS.load(Relaxed),
        LAST_ORDER_ROWS.load(Relaxed),
    )
}

#[cfg(feature = "mlx")]
pub(crate) fn apply(
    hot: &Array,
    cold: &Array,
    order: &[u32],
    weights: &Array,
    activation_dtype: Dtype,
) -> Result<Array> {
    let [hot_rows, width]: [i32; 2] = hot
        .shape()
        .try_into()
        .context("weighted epilogue Hot must have rank 2")?;
    let [cold_rows, cold_width]: [i32; 2] = cold
        .shape()
        .try_into()
        .context("weighted epilogue Cold must have rank 2")?;
    // MLX binds rank-zero inputs as scalar references; this kernel indexes weight arrays.
    ensure!(
        hot.dtype() == Dtype::Float16
            && cold.dtype() == Dtype::Float16
            && matches!(activation_dtype, Dtype::Float16 | Dtype::Float32)
            && matches!(weights.dtype(), Dtype::Float16 | Dtype::Float32)
            && !weights.shape().is_empty()
            && hot_rows > 0
            && cold_rows > 0
            && width > 0
            && width == cold_width,
        "invalid weighted epilogue dtype or matrix shape"
    );
    let layout = geometry(hot_rows as usize, cold_rows as usize, width as usize, order)
        .context("invalid weighted epilogue geometry or source index")?;
    let weight_count = weights.shape().iter().try_fold(1usize, |count, &dim| {
        usize::try_from(dim)
            .ok()
            .filter(|&dim| dim > 0)
            .and_then(|dim| count.checked_mul(dim))
    });
    ensure!(
        weight_count == Some(order.len()),
        "weighted epilogue weights must match the canonical rows"
    );
    let output_dtype = if activation_dtype == Dtype::Float16 && weights.dtype() == Dtype::Float16 {
        Dtype::Float16
    } else {
        Dtype::Float32
    };
    let activation_type = if activation_dtype == Dtype::Float16 {
        "half"
    } else {
        "float"
    };
    let product_type = if output_dtype == Dtype::Float16 {
        "half"
    } else {
        "float"
    };
    let order = Array::from_u32(order, &[layout.rows])?;
    #[cfg(all(test, feature = "mlx"))]
    {
        use std::sync::atomic::Ordering::Relaxed;
        LAST_HOT_ROWS.store(hot_rows as usize, Relaxed);
        LAST_COLD_ROWS.store(cold_rows as usize, Relaxed);
        LAST_ORDER_ROWS.store(layout.rows as usize, Relaxed);
        KERNEL_REQUESTS.fetch_add(1, Relaxed);
    }
    let mut output = array::metal_kernel(
        "mlxl3_flash_weighted_epilogue_v1",
        &["hot", "cold", "order", "weights"],
        &["output"],
        &format!(
            "#define WIDTH {}u\n#define ACT_T {activation_type}\n#define PRODUCT_T {product_type}\n",
            layout.width
        ),
        include_str!("../shaders/flash_weighted_epilogue.metal"),
        &[hot, cold, &order, weights],
        &[vec![layout.rows, layout.width]],
        &[output_dtype],
        [layout.elements, 1, 1],
        [128, 1, 1],
    )?;
    output.pop().context("missing weighted epilogue output")
}

#[cfg(test)]
mod geometry_tests {
    use super::*;

    #[test]
    fn geometry_keeps_physical_sources_and_canonical_occurrences() {
        let order = [0, 9, 10, 14, 0, 14, 5, 2, 12, 7];
        assert_eq!(
            geometry(10, 5, 2560, &order),
            Some(Geometry {
                rows: 10,
                width: 2560,
                elements: 25600,
            })
        );
        assert_eq!(geometry(80, 80, 2560, &[159; 80]).unwrap().elements, 204800);
        assert!(geometry(1, 1, 1, &[1]).is_some());
    }

    #[test]
    fn geometry_rejects_invalid_or_overflowing_requests() {
        for (hot, cold, width) in [
            (0, 1, 1),
            (1, 0, 1),
            (81, 1, 1),
            (1, 81, 1),
            (1, 1, 0),
            (1, 1, usize::MAX),
            (usize::MAX, 1, 1),
            (1, usize::MAX, 1),
        ] {
            assert!(geometry(hot, cold, width, &[0]).is_none());
        }
        assert!(geometry(1, 1, 1, &[]).is_none());
        assert!(geometry(1, 1, 1, &[0; 81]).is_none());
        assert!(geometry(1, 1, 1, &[2]).is_none());
        assert!(geometry(1, 1, 1, &[u32::MAX]).is_none());
        assert!(geometry(1, 1, i32::MAX as usize, &[0, 1]).is_none());
        let width = u32::MAX as usize / 160;
        assert!(geometry(80, 80, width, &[159; 80]).is_some());
        assert!(geometry(80, 80, width + 1, &[159; 80]).is_none());
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::solver(kissat)]
#[kani::unwind(3)]
fn weighted_geometry_bounds_every_symbolic_address() {
    let hot = kani::any::<usize>();
    let cold = kani::any::<usize>();
    let width = kani::any::<usize>();
    let source = kani::any::<u32>();
    let layout = geometry(hot, cold, width, &[source]);
    if let Some(layout) = layout {
        assert!((1..=80).contains(&hot) && (1..=80).contains(&cold));
        assert!(width > 0 && width <= i32::MAX as usize);
        assert_eq!(layout.rows, 1);
        assert_eq!(layout.width as usize, width);
        assert_eq!(layout.elements as usize, width);
        // Narrow only after proving the production guard; symbolic inputs stay full width.
        let hot_rows = hot as u8;
        let cold_rows = cold as u8;
        let columns = width as u32;
        assert_eq!(usize::from(hot_rows), hot);
        assert_eq!(usize::from(cold_rows), cold);
        assert_eq!(columns as usize, width);
        assert!(
            (u64::from(hot_rows) + u64::from(cold_rows)) * u64::from(columns)
                <= u64::from(u32::MAX)
        );
        assert!((source as usize) < hot + cold);
        let column = kani::any::<usize>();
        if column < width {
            let row = if (source as usize) < hot {
                source as usize
            } else {
                source as usize - hot
            };
            let height = if (source as usize) < hot { hot } else { cold };
            assert!(row < height && height <= 80);
            let local_row = row as u8;
            let source_height = height as u8;
            let local_column = column as u32;
            assert_eq!(usize::from(local_row), row);
            assert_eq!(usize::from(source_height), height);
            assert_eq!(local_column as usize, column);
            let offset = u64::from(local_row) * u64::from(columns) + u64::from(local_column);
            assert!(offset < u64::from(source_height) * u64::from(columns));
            assert!(offset < u64::from(u32::MAX));
        }
    }
    kani::cover!(layout.is_some() && source == 0);
    kani::cover!(layout.is_some() && source as usize == hot);
    kani::cover!(layout.is_none() && hot == 0);
    kani::cover!(layout.is_none() && source == u32::MAX);
    kani::cover!(layout.is_none() && width == usize::MAX);
    kani::cover!(
        layout.is_none()
            && hot == 80
            && cold == 80
            && width > u32::MAX as usize / 160
            && width <= i32::MAX as usize
    );
}

#[cfg(kani)]
#[kani::proof]
#[kani::solver(kissat)]
#[kani::unwind(82)]
fn weighted_geometry_checks_all_eighty_canonical_rows() {
    let hot = kani::any::<usize>();
    let cold = kani::any::<usize>();
    let width = kani::any::<usize>();
    let order: [u32; 81] = kani::any();
    let length = kani::any::<u8>() as usize;
    if length <= order.len() {
        let layout = geometry(hot, cold, width, &order[..length]);
        if let Some(layout) = layout {
            assert!((1..=80).contains(&length));
            assert_eq!(layout.rows as usize, length);
            assert_eq!(layout.width as usize, width);
            assert_eq!(layout.elements as u128, length as u128 * width as u128);
            assert!((layout.elements as u128) <= i32::MAX as u128);
            for &source in &order[..length] {
                assert!((source as usize) < hot + cold);
            }
        }
        kani::cover!(layout.is_some() && length == 80);
        kani::cover!(layout.is_some() && length == 1);
        kani::cover!(layout.is_none() && length == 0);
        kani::cover!(layout.is_none() && length == 81);
    }
}

#[cfg(all(test, feature = "mlx"))]
mod capture_tests;
#[cfg(all(test, feature = "mlx"))]
mod tests;
