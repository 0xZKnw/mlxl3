//! Immutable matrix views. Offsets and strides are measured in storage elements.
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MatrixLayout {
    pub shape: [u32; 2],
    pub strides: [i64; 2],
    pub offset: i64,
}

impl MatrixLayout {
    /// A row of an already checked view stays on the same backing.
    pub fn row(self, row: u32) -> Option<Self> {
        let [rows, cols] = self.shape;
        if row >= rows || cols == 0 {
            return None;
        }
        let offset =
            i64::try_from(i128::from(self.offset) + i128::from(row) * i128::from(self.strides[0]))
                .ok()?;
        Some(Self {
            shape: [1, cols],
            strides: [0, self.strides[1]],
            offset,
        })
    }
    /// Check both extrema before exposing the backing to signed 64-bit shader indexing.
    /// A zero-sized view may start one element past the allocation.
    pub fn fits(self, elements: usize) -> bool {
        let Ok(limit) = i64::try_from(elements) else {
            return false;
        };
        if !(0..=limit).contains(&self.offset) {
            return false;
        }
        if self.shape.contains(&0) {
            return true;
        }
        let mut first = i128::from(self.offset);
        let mut last = first;
        for (length, stride) in self.shape.into_iter().zip(self.strides) {
            let reach = i128::from(length - 1) * i128::from(stride);
            first += reach.min(0);
            last += reach.max(0);
        }
        first >= 0 && last < i128::from(limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn view_bounds_match_all_reachable_indices() {
        for rows in 0..=4 {
            for cols in 0..=4 {
                for rs in -3..=3 {
                    for cs in -3..=3 {
                        for offset in -2..=10 {
                            for elements in 0usize..=8 {
                                let layout = MatrixLayout {
                                    shape: [rows, cols],
                                    strides: [rs, cs],
                                    offset,
                                };
                                let expected = (0..=elements as i64).contains(&offset)
                                    && (0..rows).all(|r| {
                                        (0..cols).all(|c| {
                                            (0..elements as i64).contains(
                                                &(offset + i64::from(r) * rs + i64::from(c) * cs),
                                            )
                                        })
                                    });
                                assert_eq!(
                                    layout.fits(elements),
                                    expected,
                                    "{layout:?}/{elements}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn huge_signed_strides_do_not_wrap() {
        for strides in [[i64::MAX, i64::MIN], [i64::MAX; 2], [i64::MIN; 2]] {
            let full = MatrixLayout {
                shape: [u32::MAX; 2],
                strides,
                offset: 1,
            };
            assert!(!full.fits(8));
            assert!(
                MatrixLayout {
                    shape: [1; 2],
                    ..full
                }
                .fits(8)
            );
            assert!(
                MatrixLayout {
                    shape: [0, u32::MAX],
                    offset: 8,
                    ..full
                }
                .fits(8)
            );
        }
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(4)]
fn accepted_matrix_views_bound_every_sampled_coordinate() {
    let rows = u32::from(kani::any::<u8>() % 5);
    let cols = u32::from(kani::any::<u8>() % 5);
    let offset = i64::from(kani::any::<i8>());
    let strides = [i64::from(kani::any::<i8>()), i64::from(kani::any::<i8>())];
    let elements = usize::from(kani::any::<u8>());
    let r = u32::from(kani::any::<u8>());
    let c = u32::from(kani::any::<u8>());
    let layout = MatrixLayout {
        shape: [rows, cols],
        strides,
        offset,
    };
    if layout.fits(elements) && r < rows && c < cols {
        let address = offset + i64::from(r) * strides[0] + i64::from(c) * strides[1];
        assert!(address >= 0 && (address as usize) < elements);
        kani::cover!(r > 0 && c > 0 && strides[0] < 0 && strides[1] > 0);
        kani::cover!(r > 0 && strides[0] == 0);
    }
    kani::cover!(layout.fits(elements) && rows == 0 && offset == elements as i64);
    kani::cover!(!layout.fits(elements) && rows > 0 && cols > 0 && offset == 0);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn matrix_row_preserves_backing_and_address() {
    let layout = MatrixLayout {
        shape: [u32::from(kani::any::<u8>()), u32::from(kani::any::<u8>())],
        strides: [i64::from(kani::any::<i8>()), i64::from(kani::any::<i8>())],
        offset: i64::from(kani::any::<u8>()),
    };
    let (row, col, n) = (
        u32::from(kani::any::<u8>()),
        u32::from(kani::any::<u8>()),
        usize::from(kani::any::<u8>()),
    );
    if layout.fits(n) {
        let part = layout.row(row);
        assert_eq!(part.is_some(), row < layout.shape[0] && layout.shape[1] > 0);
        if let Some(part) = part {
            assert!(part.fits(n) && part.shape == [1, layout.shape[1]]);
            if col < part.shape[1] {
                assert_eq!(
                    part.offset + i64::from(col) * part.strides[1],
                    layout.offset
                        + i64::from(row) * layout.strides[0]
                        + i64::from(col) * layout.strides[1]
                );
                kani::cover!(row > 0 && col > 0 && layout.strides[0] < 0);
            }
        }
    }
}

#[test]
fn selecting_a_row_checks_index_and_signed_offset() {
    let reversed = MatrixLayout {
        shape: [2, 3],
        strides: [-5, -1],
        offset: 9,
    };
    assert_eq!(
        reversed.row(1),
        Some(MatrixLayout {
            shape: [1, 3],
            strides: [0, -1],
            offset: 4
        })
    );
    assert!(reversed.row(2).is_none());
    assert!(
        MatrixLayout {
            shape: [1, 0],
            ..reversed
        }
        .row(0)
        .is_none()
    );
    assert!(
        MatrixLayout {
            strides: [i64::MAX, 0],
            ..reversed
        }
        .row(1)
        .is_none()
    );
}
