//! Runs external commands: the resident engine and one-shot install/probe/download steps.

use crate::framer::{Frame, LineFramer, MAX_LINE};
use crate::protocol::{self, Event};
use crate::runtime::Command;
use anyhow::{Context, Result, bail};
use std::io::{Read, Write};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Bytes of stderr kept for error reports.
pub const STDERR_TAIL: usize = 32 * 1024;

#[derive(Debug)]
pub enum EngineMessage {
    Event(Event),
    /// A protocol line that failed to parse (reported, not fatal).
    Malformed(String),
    /// The process ended; always the last message.
    Exited {
        code: Option<i32>,
        stderr_tail: String,
    },
}

pub fn process(command: &Command) -> std::process::Command {
    let mut process = std::process::Command::new(&command.program);
    process
        .args(&command.args)
        .envs(command.env.iter().map(|(k, v)| (k, v)));
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        process.creation_flags(CREATE_NO_WINDOW);
    }
    process
}

#[derive(Default)]
struct Tail(Vec<u8>);

impl Tail {
    fn push(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
        if self.0.len() > STDERR_TAIL {
            let cut = self.0.len() - STDERR_TAIL;
            self.0.drain(..cut);
        }
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.0).into_owned()
    }
}

fn pump_stderr(mut stderr: impl Read, tail: Arc<Mutex<Tail>>, mut on_line: impl FnMut(String)) {
    let mut framer = LineFramer::new(MAX_LINE);
    let mut buffer = [0u8; 8192];
    let mut frames = Vec::new();
    while let Ok(n) = stderr.read(&mut buffer) {
        if n == 0 {
            break;
        }
        tail.lock().unwrap().push(&buffer[..n]);
        framer.push(&buffer[..n], &mut frames);
        for frame in frames.drain(..) {
            if let Frame::Line(line) = frame {
                on_line(String::from_utf8_lossy(&line).into_owned());
            }
        }
    }
}

/// A resident engine process speaking the JSON-lines bridge protocol.
pub struct Engine {
    child: Arc<Mutex<Child>>,
    stdin: Mutex<Option<ChildStdin>>,
    threads: Vec<JoinHandle<()>>,
}

impl Engine {
    /// Spawn and start delivering messages to `on_message` from a reader thread.
    pub fn spawn(
        command: &Command,
        on_message: impl Fn(EngineMessage) + Send + Sync + 'static,
    ) -> Result<Self> {
        let mut child = process(command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("could not start {}", command.program))?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().context("no engine stdout")?;
        let stderr = child.stderr.take().context("no engine stderr")?;
        let child = Arc::new(Mutex::new(child));
        let tail = Arc::new(Mutex::new(Tail::default()));
        let stderr_thread = {
            let tail = Arc::clone(&tail);
            std::thread::Builder::new()
                .name("engine-stderr".into())
                .spawn(move || pump_stderr(stderr, tail, |_| {}))?
        };
        let reader = {
            let child = Arc::clone(&child);
            std::thread::Builder::new()
                .name("engine-stdout".into())
                .spawn(move || {
                    let mut stdout = stdout;
                    let mut framer = LineFramer::new(MAX_LINE);
                    let mut buffer = [0u8; 8192];
                    let mut frames = Vec::new();
                    let deliver = |frame: Frame| match frame {
                        Frame::Line(line) if line.is_empty() => {}
                        Frame::Line(line) => match protocol::parse_event(&line) {
                            Ok(event) => on_message(EngineMessage::Event(event)),
                            Err(error) => {
                                on_message(EngineMessage::Malformed(format!("{error:#}")))
                            }
                        },
                        Frame::Oversized => {
                            on_message(EngineMessage::Malformed("engine line exceeds 1 MiB".into()))
                        }
                    };
                    loop {
                        match stdout.read(&mut buffer) {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                framer.push(&buffer[..n], &mut frames);
                                frames.drain(..).for_each(&deliver);
                            }
                        }
                    }
                    if let Some(frame) = framer.finish() {
                        deliver(frame);
                    }
                    let _ = stderr_thread.join();
                    let code = wait_bounded(&child, Duration::from_secs(10));
                    let stderr_tail = tail.lock().unwrap().text();
                    on_message(EngineMessage::Exited { code, stderr_tail });
                })?
        };
        Ok(Self {
            child,
            stdin: Mutex::new(stdin),
            threads: vec![reader],
        })
    }

    pub fn send(&self, line: &[u8]) -> Result<()> {
        let mut guard = self.stdin.lock().unwrap();
        let stdin = guard.as_mut().context("the engine input is closed")?;
        stdin
            .write_all(line)
            .and_then(|()| stdin.flush())
            .context("the engine stopped reading requests")
    }

    /// Ask for a clean exit (shutdown request, then EOF); kill after `grace`.
    pub fn shutdown(mut self, grace: Duration) -> Option<i32> {
        if let Ok(line) = protocol::control_request("shutdown", "") {
            let _ = self.send(&line);
        }
        self.stdin.lock().unwrap().take();
        let code = wait_bounded(&self.child, grace);
        if code.is_none() {
            let _ = self.child.lock().unwrap().kill();
        }
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
        code
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.stdin.lock().unwrap().take();
        let mut child = self.child.lock().unwrap();
        if matches!(child.try_wait(), Ok(None)) {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn wait_bounded(child: &Mutex<Child>, limit: Duration) -> Option<i32> {
    let deadline = Instant::now() + limit;
    loop {
        if let Ok(Some(status)) = child.lock().unwrap().try_wait() {
            return status.code();
        }
        if Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Output of a one-shot command.
#[derive(Debug)]
pub struct Finished {
    pub code: Option<i32>,
    /// Raw stdout (first 4 MiB), e.g. UTF-16 output of `wsl.exe --list`.
    pub stdout_bytes: Vec<u8>,
    /// `stdout_bytes` decoded as UTF-8 (lossy).
    pub stdout: String,
    pub stderr_tail: String,
}

/// Run a step to completion, streaming its output lines (stdout and stderr) to `on_line`.
/// `timeout` bounds the wait; on expiry the process is killed and an error returned.
pub fn run(
    command: &Command,
    timeout: Option<Duration>,
    on_line: impl Fn(&str) + Send + Sync + 'static,
) -> Result<Finished> {
    let mut child = process(command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("could not start {}", command.program))?;
    let on_line = Arc::new(on_line);
    let tail = Arc::new(Mutex::new(Tail::default()));
    let stderr = child.stderr.take().context("no stderr")?;
    let stderr_thread = {
        let (tail, on_line) = (Arc::clone(&tail), Arc::clone(&on_line));
        std::thread::spawn(move || pump_stderr(stderr, tail, |line| on_line(&line)))
    };
    let mut stdout = child.stdout.take().context("no stdout")?;
    let stdout_thread = {
        let on_line = Arc::clone(&on_line);
        std::thread::spawn(move || {
            let mut captured = Vec::new();
            let mut framer = LineFramer::new(MAX_LINE);
            let mut buffer = [0u8; 8192];
            let mut frames = Vec::new();
            while let Ok(n) = stdout.read(&mut buffer) {
                if n == 0 {
                    break;
                }
                if captured.len() < 4 * MAX_LINE {
                    captured.extend_from_slice(&buffer[..n]);
                }
                framer.push(&buffer[..n], &mut frames);
                for frame in frames.drain(..) {
                    if let Frame::Line(line) = frame {
                        on_line(&String::from_utf8_lossy(&line));
                    }
                }
            }
            captured
        })
    };
    let child = Mutex::new(child);
    let code = match timeout {
        Some(limit) => {
            let code = wait_bounded(&child, limit);
            if code.is_none() && matches!(child.lock().unwrap().try_wait(), Ok(None)) {
                let _ = child.lock().unwrap().kill();
                let _ = child.lock().unwrap().wait();
                let _ = (stdout_thread.join(), stderr_thread.join());
                bail!("{} timed out after {} s", command.label, limit.as_secs());
            }
            code
        }
        None => child.lock().unwrap().wait()?.code(),
    };
    let stdout_bytes = stdout_thread.join().unwrap_or_default();
    let _ = stderr_thread.join();
    let stderr_tail = tail.lock().unwrap().text();
    let stdout = String::from_utf8_lossy(&stdout_bytes).into_owned();
    Ok(Finished {
        code,
        stdout_bytes,
        stdout,
        stderr_tail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_keeps_the_end() {
        let mut tail = Tail::default();
        tail.push(&vec![b'a'; STDERR_TAIL]);
        tail.push(b"END");
        assert_eq!(tail.0.len(), STDERR_TAIL);
        assert!(tail.text().ends_with("aEND"));
    }
}
