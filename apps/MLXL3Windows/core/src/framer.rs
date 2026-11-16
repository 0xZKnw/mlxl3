//! Splits a byte stream into newline-terminated lines with a hard size bound.

/// Longest protocol line accepted from the engine (deltas are short; `ready`
/// and `done` are a few hundred bytes).
pub const MAX_LINE: usize = 1 << 20;

#[derive(Debug, PartialEq, Eq)]
pub enum Frame {
    Line(Vec<u8>),
    /// A line longer than the bound; its bytes up to the next newline are dropped.
    Oversized,
}

#[derive(Debug)]
pub struct LineFramer {
    buffer: Vec<u8>,
    max: usize,
    discarding: bool,
}

impl LineFramer {
    pub fn new(max: usize) -> Self {
        assert!(max > 0, "line bound must be positive");
        Self {
            buffer: Vec::new(),
            max,
            discarding: false,
        }
    }

    /// Bytes buffered for the current incomplete line. Never exceeds `max`.
    pub fn pending(&self) -> usize {
        self.buffer.len()
    }

    /// Feed bytes; complete lines (without `\n` and a trailing `\r`) are appended to `out`.
    pub fn push(&mut self, bytes: &[u8], out: &mut Vec<Frame>) {
        for &byte in bytes {
            if byte == b'\n' {
                if self.discarding {
                    self.discarding = false;
                    out.push(Frame::Oversized);
                } else {
                    let mut line = std::mem::take(&mut self.buffer);
                    if line.last() == Some(&b'\r') {
                        line.pop();
                    }
                    out.push(Frame::Line(line));
                }
            } else if !self.discarding {
                if self.buffer.len() == self.max {
                    self.buffer.clear();
                    self.discarding = true;
                } else {
                    self.buffer.push(byte);
                }
            }
        }
    }

    /// End of stream: an unterminated final line is returned (or reported oversized).
    pub fn finish(&mut self) -> Option<Frame> {
        if std::mem::take(&mut self.discarding) {
            return Some(Frame::Oversized);
        }
        if self.buffer.is_empty() {
            return None;
        }
        let mut line = std::mem::take(&mut self.buffer);
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        Some(Frame::Line(line))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(max: usize, chunks: &[&[u8]]) -> Vec<Frame> {
        let mut framer = LineFramer::new(max);
        let mut out = Vec::new();
        for chunk in chunks {
            framer.push(chunk, &mut out);
            assert!(framer.pending() <= max);
        }
        out.extend(framer.finish());
        out
    }

    #[test]
    fn splits_across_chunks_and_strips_crlf() {
        assert_eq!(
            lines(16, &[b"ab", b"c\r\nde", b"\n", b"tail"]),
            vec![
                Frame::Line(b"abc".to_vec()),
                Frame::Line(b"de".to_vec()),
                Frame::Line(b"tail".to_vec())
            ]
        );
    }

    #[test]
    fn oversized_line_is_dropped_and_framing_recovers() {
        assert_eq!(
            lines(3, &[b"abc\nabcd", b"efg\nok\n"]),
            vec![
                Frame::Line(b"abc".to_vec()),
                Frame::Oversized,
                Frame::Line(b"ok".to_vec())
            ]
        );
        assert_eq!(lines(2, &[b"xyz"]), vec![Frame::Oversized]);
    }

    #[test]
    fn empty_lines_are_kept() {
        assert_eq!(
            lines(4, &[b"\n\n"]),
            vec![Frame::Line(vec![]), Frame::Line(vec![])]
        );
    }

    /// Exhaustive over every byte string of length <= 6 on the alphabet {a, \r, \n},
    /// every split point and bounds 1..=3, against an independent oracle.
    #[test]
    fn exhaustive_small_streams_match_oracle() {
        // A raw segment (trailing '\r' included) longer than `max` is oversized;
        // otherwise one trailing '\r' is stripped. An empty unterminated tail emits nothing.
        fn oracle(data: &[u8], max: usize) -> Vec<Frame> {
            let frame = |part: &[u8]| {
                if part.len() > max {
                    Frame::Oversized
                } else {
                    Frame::Line(part.strip_suffix(b"\r").unwrap_or(part).to_vec())
                }
            };
            let mut parts: Vec<&[u8]> = data.split(|b| *b == b'\n').collect();
            let last = parts.pop().unwrap();
            let mut out: Vec<Frame> = parts.into_iter().map(frame).collect();
            if !last.is_empty() {
                out.push(frame(last));
            }
            out
        }
        let alphabet = *b"a\r\n";
        for len in 0..=6usize {
            for code in 0..3usize.pow(len as u32) {
                let mut data = Vec::with_capacity(len);
                let mut c = code;
                for _ in 0..len {
                    data.push(alphabet[c % 3]);
                    c /= 3;
                }
                for max in 1..=3 {
                    for split in 0..=len {
                        let got = lines(max, &[&data[..split], &data[split..]]);
                        assert_eq!(
                            got,
                            oracle(&data, max),
                            "data={data:?} max={max} split={split}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    /// For any 5-byte input fed in two symbolic chunks with any bound in 1..=3:
    /// the buffer never exceeds the bound and every emitted line fits it.
    #[kani::proof]
    #[kani::unwind(8)]
    fn buffer_and_lines_stay_bounded() {
        let data: [u8; 5] = kani::any();
        let split: usize = kani::any();
        kani::assume(split <= data.len());
        let max: usize = kani::any();
        kani::assume((1..=3).contains(&max));
        let mut framer = LineFramer::new(max);
        let mut out = Vec::new();
        framer.push(&data[..split], &mut out);
        assert!(framer.pending() <= max);
        framer.push(&data[split..], &mut out);
        assert!(framer.pending() <= max);
        out.extend(framer.finish());
        assert!(out.len() <= data.len());
        for frame in &out {
            if let Frame::Line(line) = frame {
                assert!(line.len() <= max);
            }
        }
        kani::cover!(out.iter().any(|f| *f == Frame::Oversized));
        kani::cover!(out.len() >= 2);
    }
}
