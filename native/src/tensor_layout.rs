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
