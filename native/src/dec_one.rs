//! Canonical S2 QMV and finish in one arrival kernel, with per-call private storage.

pub(crate) fn target_scope(prefix: &str) -> bool {
    prefix.starts_with("model.language_model.layers.")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Layout {
    pub(crate) partial_words: i32,
    pub(crate) output_word: i32,
    pub(crate) storage_words: i32,
}

pub(crate) fn layout(rows: i32) -> Option<Layout> {
    if !(1..=2).contains(&rows) {
        return None;
    }
    let partial_words = rows * 2 * 2560;
    let output_word = partial_words + 20;
    Some(Layout {
        partial_words,
        output_word,
        storage_words: output_word + rows * 2560 / 2,
    })
}

#[cfg(all(test, feature = "mlx"))]
pub(crate) static REQUESTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "mlx")]
pub(crate) fn apply(
    xhat: &crate::array::Array,
    trellis: &crate::array::Array,
    svh: &crate::array::Array,
    rows: i32,
    header: &str,
) -> anyhow::Result<crate::array::Array> {
    use crate::array::{self, Dtype};
    use anyhow::{Context, ensure};
    let layout = layout(rows).context("invalid DEC-ONE row count")?;
    ensure!(xhat.dtype() == Dtype::Float16 && xhat.shape().iter().product::<i32>() == rows * 6144);
    ensure!(svh.dtype() == Dtype::Float16 && svh.shape() == [2560]);
    ensure!(trellis.dtype() == Dtype::UInt32 && trellis.shape() == [384 * 160 * 32]);
    let header = format!(
        "{header}\n#define DEC_PARTIAL_WORDS {}u\n#define DEC_OUTPUT_WORD {}u\n#define DEC_SCALE_BITS 0x3db504f3u\n",
        layout.partial_words, layout.output_word
    );
    static SOURCE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let source = SOURCE.get_or_init(|| {
        String::from("device float* yhat = reinterpret_cast<device float*>(state);\n")
            + include_str!("../shaders/_qmv_tile_kernel.metal")
            + include_str!("../shaders/flash_dec_one_finish.metal")
    });
    let storage = array::metal_kernel_initialized(
        &format!("mlxl3_flash_dec_one_s2_b{rows}_v1"),
        &["xhat", "trellis", "svh"],
        &["state"],
        &header,
        source,
        &[xhat, trellis, svh],
        &[vec![layout.storage_words]],
        &[Dtype::UInt32],
        [160 * 128, 1, 2],
        [128, 1, 1],
        Some(0.0),
    )?
    .pop()
    .context("missing DEC-ONE storage")?;
    #[cfg(test)]
    REQUESTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    storage
        .view(Dtype::Float16)?
        .slice(0, layout.output_word * 2, layout.storage_words * 2)?
        .reshape(&[rows, 2560])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dec_one_storage_has_disjoint_aligned_regions_and_bounded_size() {
        for rows in [i32::MIN, -1, 0, 3, 8, i32::MAX] {
            assert!(layout(rows).is_none());
        }
        for rows in [1, 2] {
            let l = layout(rows).unwrap();
            assert_eq!(l.partial_words, rows * 5120);
            assert_eq!(l.output_word - l.partial_words, 20);
            assert_eq!((l.storage_words - l.output_word) * 2, rows * 2560);
            assert!(l.storage_words * 4 <= 51280);
            assert_eq!((l.output_word * 4) % 16, 0);
        }
    }

    #[test]
    fn dec_one_target_scope_excludes_mtp_head_and_similar_prefixes() {
        for layer in 0..48 {
            assert!(target_scope(&format!(
                "model.language_model.layers.{layer}.linear_attn.out_proj"
            )));
        }
        for prefix in [
            "mtp.layers.0.self_attn.o_proj",
            "mtp.fc_hidden",
            "lm_head",
            "",
            "model.language_model.layers",
            "model.language_model.layersX.0",
            "xmodel.language_model.layers.0",
        ] {
            assert!(!target_scope(prefix));
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn dec_one_storage_contract() {
    let rows: i32 = kani::any();
    let scope: u8 = kani::any();
    let prefix = match scope % 5 {
        0 => "model.language_model.layers.0.linear_attn.out_proj",
        1 => "model.language_model.layers.47.self_attn.o_proj",
        2 => "mtp.layers.0.self_attn.o_proj",
        3 => "model.language_model.layersX.0",
        _ => "lm_head",
    };
    assert_eq!(target_scope(prefix), scope % 5 < 2);
    kani::cover!(scope % 5 == 0);
    kani::cover!(scope % 5 == 2);
    kani::cover!(scope % 5 == 3);
    match layout(rows) {
        Some(l) => {
            assert!(rows == 1 || rows == 2);
            assert_eq!(l.partial_words, rows * 5120);
            assert_eq!(l.output_word - l.partial_words, 20);
            assert_eq!((l.storage_words - l.output_word) * 2, rows * 2560);
            assert!(l.storage_words * 4 <= 51280);
            assert_eq!((l.output_word * 4) % 16, 0);
        }
        None => assert!(rows != 1 && rows != 2),
    }
    kani::cover!(rows == 1);
    kani::cover!(rows == 2);
    kani::cover!(rows < 0);
    kani::cover!(rows > 2);
}
