use std::path::{Path, PathBuf};

/// The path survives eviction, so an Inkling prompt can load the same encoder again.
pub(crate) struct VisionSlot<V> {
    checkpoint: PathBuf,
    encoder: Option<V>,
}

impl<V> VisionSlot<V> {
    pub(crate) fn new(checkpoint: PathBuf) -> Self {
        Self {
            checkpoint,
            encoder: None,
        }
    }

    pub(crate) fn release(&mut self) {
        self.encoder = None;
    }

    /// `encode` must materialize output that depends on the weights before returning.
    pub(crate) fn with_encoder<T, E>(
        &mut self,
        release: bool,
        load: impl FnOnce(&Path) -> Result<V, E>,
        encode: impl FnOnce(&V) -> Result<T, E>,
    ) -> Result<T, E> {
        if release {
            let encoder = match self.encoder.take() {
                Some(loaded) => loaded,
                None => load(&self.checkpoint)?,
            };
            return encode(&encoder);
        }
        let encoder = match &self.encoder {
            Some(loaded) => loaded,
            None => self.encoder.insert(load(&self.checkpoint)?),
        };
        encode(encoder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn releases_vision_after_the_batch_and_reloads_from_the_same_path() {
        struct Encoder<'a>(&'a Cell<usize>);
        impl Drop for Encoder<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let loads = Cell::new(0);
        let drops = Cell::new(0);
        let load = |path: &Path| {
            assert_eq!(path, Path::new("model"));
            loads.set(loads.get() + 1);
            Ok::<_, &str>(Encoder(&drops))
        };
        let mut slot = VisionSlot::new("model".into());
        assert_eq!(slot.with_encoder(false, load, |_| Ok([9, 10])), Ok([9, 10]));
        assert_eq!((loads.get(), drops.get()), (1, 0));
        assert_eq!(
            slot.with_encoder(true, load, |_| {
                assert_eq!(drops.get(), 0);
                Ok([9, 10])
            }),
            Ok([9, 10])
        );
        assert_eq!((loads.get(), drops.get()), (1, 1));
        assert_eq!(
            slot.with_encoder(true, load, |_| Err::<(), _>("bad image")),
            Err("bad image")
        );
        assert_eq!((loads.get(), drops.get()), (2, 2));
        assert_eq!(
            slot.with_encoder(true, |_| Err("bad checkpoint"), |_| Ok(())),
            Err("bad checkpoint")
        );
        assert_eq!(slot.with_encoder(false, load, |_| Ok([9, 10])), Ok([9, 10]));
        slot.release();
        assert_eq!((loads.get(), drops.get()), (3, 3));
    }
}

#[cfg(kani)]
#[kani::proof]
fn vision_slot_preserves_checkpoint_and_releases_only_when_requested() {
    let mut slot = VisionSlot::new(PathBuf::new());
    let previously_loaded: bool = kani::any();
    let release: bool = kani::any();
    let load_ok: bool = kani::any();
    let encode_ok: bool = kani::any();
    let original_weight: u8 = kani::any();
    let fresh_weight: u8 = kani::any();

    if previously_loaded {
        assert_eq!(
            slot.with_encoder(false, |_| Ok(original_weight), |w| Ok::<_, u8>(*w)),
            Ok(original_weight)
        );
    }
    let result = slot.with_encoder(
        release,
        |path| {
            assert_eq!(path, Path::new(""));
            if load_ok { Ok(fresh_weight) } else { Err(1) }
        },
        |w| if encode_ok { Ok(*w) } else { Err(2) },
    );
    let available = previously_loaded || load_ok;
    assert_eq!(slot.encoder.is_some(), available && !release);
    assert_eq!(
        result,
        if !available {
            Err(1)
        } else if !encode_ok {
            Err(2)
        } else if previously_loaded {
            Ok(original_weight)
        } else {
            Ok(fresh_weight)
        }
    );
    kani::cover!(release && previously_loaded && result == Err(2));
    kani::cover!(!release && !previously_loaded && result == Err(1));
    kani::cover!(!release && !previously_loaded && result == Ok(fresh_weight));
    slot.release();
    assert!(slot.encoder.is_none());
    assert_eq!(slot.checkpoint.as_path(), Path::new(""));
}
