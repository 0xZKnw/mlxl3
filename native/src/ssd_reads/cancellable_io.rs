//! Cooperative cancellation at each bounded positional-read submission.
//! No timeout or preemption guarantee is made for an already executing syscall.
use std::{
    fs::File,
    io::{self, ErrorKind},
    os::unix::fs::FileExt,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub(super) const READ_CHUNK_BYTES: usize = 256 * 1024;

// Pure submission arithmetic, shared by the real loop and the proof harness.
fn chunk_length(remaining: usize) -> usize {
    remaining.min(READ_CHUNK_BYTES)
}

#[derive(Clone)]
pub(super) struct ReadCancellation {
    pub(super) requested: Arc<AtomicBool>,
    pub(super) stopped: Arc<AtomicBool>,
    #[cfg(test)]
    pub(super) probe: Option<Arc<ReadProbe>>,
}

impl ReadCancellation {
    pub(super) fn check(&self) -> io::Result<()> {
        if self.requested.load(Ordering::Acquire) {
            self.stopped.store(true, Ordering::Release);
        }
        if self.stopped.load(Ordering::Acquire) {
            Err(io::Error::other("generation cancelled"))
        } else {
            Ok(())
        }
    }

    pub(super) fn read_exact_at(
        &self,
        file: &File,
        target: &mut [u8],
        offset: u64,
    ) -> io::Result<()> {
        self.read_with(target, offset, |part, at| {
            #[cfg(test)]
            let call = self
                .probe
                .as_ref()
                .map(|probe| probe.calls.fetch_add(1, Ordering::AcqRel));
            let result = file.read_at(part, at);
            #[cfg(test)]
            if call == Some(0)
                && let Some(gate) = self.probe.as_ref().and_then(|p| p.after_first.as_ref())
            {
                gate.wait();
            }
            result
        })
    }

    fn read_with(
        &self,
        mut target: &mut [u8],
        mut offset: u64,
        mut read: impl FnMut(&mut [u8], u64) -> io::Result<usize>,
    ) -> io::Result<()> {
        self.check()?;
        offset
            .checked_add(target.len() as u64)
            .ok_or_else(|| io::Error::new(ErrorKind::InvalidInput, "SSD read offset overflow"))?;
        while !target.is_empty() {
            self.check()?;
            let length = chunk_length(target.len());
            let result = read(&mut target[..length], offset);
            // Cancellation wins even if this was the final chunk or EINTR.
            self.check()?;
            match result {
                Ok(0) => return Err(io::Error::new(ErrorKind::UnexpectedEof, "short SSD read")),
                Ok(n) if n <= length => {
                    offset += n as u64; // The complete extent was checked above.
                    target = &mut target[n..];
                }
                Ok(_) => {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "invalid SSD read count",
                    ));
                }
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[derive(Default)]
pub(super) struct ReadProbe {
    pub(super) calls: std::sync::atomic::AtomicUsize,
    pub(super) after_first: Option<Arc<super::TestReadGate>>,
}

#[cfg(kani)]
#[kani::proof]
fn ssd_read_submission_is_bounded_and_makes_progress() {
    let remaining: usize = kani::any();
    let n = chunk_length(remaining);
    assert!(n <= READ_CHUNK_BYTES && n <= remaining);
    assert_eq!(n == 0, remaining == 0);
    if remaining > 0 {
        assert!(remaining - n < remaining);
    }
    kani::cover!(remaining > READ_CHUNK_BYTES);
    kani::cover!(remaining == 1);
    kani::cover!(remaining == 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token() -> ReadCancellation {
        ReadCancellation {
            requested: Arc::new(AtomicBool::new(false)),
            stopped: Arc::new(AtomicBool::new(false)),
            probe: None,
        }
    }

    #[test]
    fn read_loop_handles_short_reads_interrupts_and_extent_boundaries() -> io::Result<()> {
        let token = token();
        let mut target = vec![0; READ_CHUNK_BYTES + 19];
        let mut calls = 0;
        let mut expected_offset = 37;
        token.read_with(&mut target, 37, |part, at| {
            assert_eq!(at, expected_offset);
            assert!(part.len() <= READ_CHUNK_BYTES);
            calls += 1;
            if calls == 1 {
                return Err(ErrorKind::Interrupted.into());
            }
            let n = part.len().min(751);
            for (i, byte) in part[..n].iter_mut().enumerate() {
                *byte = ((at + i as u64) % 251) as u8;
            }
            expected_offset += n as u64;
            Ok(n)
        })?;
        assert!(calls > 2);
        assert!(
            target
                .iter()
                .enumerate()
                .all(|(i, &v)| v == ((37 + i) % 251) as u8)
        );
        assert_eq!(
            token
                .read_with(&mut [0; 1], u64::MAX, |_, _| panic!("overflow submitted"))
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidInput
        );
        token.read_with(&mut [], u64::MAX, |_, _| panic!("empty submitted"))?;
        assert_eq!(
            token
                .read_with(&mut [0; 1], 0, |_, _| Ok(0))
                .unwrap_err()
                .kind(),
            ErrorKind::UnexpectedEof
        );
        assert_eq!(
            token
                .read_with(&mut [0; 1], 0, |_, _| Ok(2))
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidData
        );
        assert_eq!(
            token
                .read_with(&mut [0; 1], 0, |_, _| Err(
                    ErrorKind::PermissionDenied.into()
                ))
                .unwrap_err()
                .kind(),
            ErrorKind::PermissionDenied
        );
        Ok(())
    }

    #[test]
    fn cancellation_stops_remainder_and_is_sticky_after_request_reset() {
        for interrupted in [false, true] {
            let token = token();
            let mut calls = 0;
            let mut target = vec![0; READ_CHUNK_BYTES + 1];
            let result = token.read_with(&mut target, 0, |part, _| {
                calls += 1;
                token.requested.store(true, Ordering::Release);
                if interrupted {
                    Err(ErrorKind::Interrupted.into())
                } else {
                    Ok(part.len())
                }
            });
            assert!(result.is_err());
            assert_eq!(calls, 1);
            token.requested.store(false, Ordering::Release);
            assert!(
                token
                    .read_with(&mut target, 0, |_, _| panic!("cancelled submission"))
                    .is_err()
            );
        }
    }
}
