//! Pure pre-materialization image admission shared by production and CPU proofs.
pub const MAX_PIXELS: u64 = 16_777_216;
pub const MAX_IMAGE_TOKENS: u64 = MAX_PIXELS / 1024;

pub fn fits_image_budget(height: u32, width: u32, pixels: u64, tokens: u64) -> bool {
    let required = u64::from(height) * u64::from(width);
    required <= pixels && required / 1024 <= tokens
}

#[test]
fn image_budget_matches_wide_integer_oracle() {
    for height in [0, 32, 256, 4096, u32::MAX] {
        for width in [0, 32, 256, 4096, u32::MAX] {
            for pixels in [0, 65_535, 65_536, MAX_PIXELS, u64::MAX] {
                for tokens in [0, 63, 64, MAX_IMAGE_TOKENS, u64::MAX] {
                    let required = u128::from(height) * u128::from(width);
                    assert_eq!(
                        fits_image_budget(height, width, pixels, tokens),
                        required <= u128::from(pixels) && required / 1024 <= u128::from(tokens)
                    );
                }
            }
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn image_admission_preserves_aggregate_bounds() {
    let height: u32 = kani::any();
    let width: u32 = kani::any();
    let used: u64 = kani::any();
    let used_tokens: u64 = kani::any();
    kani::assume(used <= MAX_PIXELS && used_tokens <= MAX_IMAGE_TOKENS);
    if fits_image_budget(
        height,
        width,
        MAX_PIXELS - used,
        MAX_IMAGE_TOKENS - used_tokens,
    ) {
        let required = u64::from(height) * u64::from(width);
        assert!(used + required <= MAX_PIXELS);
        assert!(used_tokens + required / 1024 <= MAX_IMAGE_TOKENS);
    }
    kani::cover!(height == 4096 && width == 4096 && used == 1);
}
