//! Supervisor ↔ real Python worker (fake backend) through the production entry point.
//! Set MLXL3_TEST_PYTHON to choose the interpreter (default: python3, or python on Windows).

use mlxl3_windows_core::protocol::{self, Message, Sampling};
use mlxl3_windows_core::runtime::Command;
use mlxl3_windows_core::supervisor::{self, Engine, EngineMessage};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(20);

fn python() -> String {
    std::env::var("MLXL3_TEST_PYTHON").unwrap_or_else(|_| {
        if cfg!(windows) {
            "python".into()
        } else {
            "python3".into()
        }
    })
}

fn engine_dir() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../engines/exl3")
        .canonicalize()
        .unwrap()
        .display()
        .to_string()
}

fn bridge(extra_env: &[(&str, &str)]) -> Command {
    let mut env = vec![
        ("PYTHONPATH".to_owned(), engine_dir()),
        ("PYTHONUNBUFFERED".to_owned(), "1".to_owned()),
    ];
    env.extend(
        extra_env
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned())),
    );
    Command {
        label: "engine".into(),
        program: python(),
        args: [
            "-s",
            "-X",
            "utf8",
            "-m",
            "mlxl3_exl3",
            "bridge",
            "unused",
            "--backend",
            "fake",
        ]
        .map(String::from)
        .to_vec(),
        env,
    }
}

fn start(extra_env: &[(&str, &str)]) -> (Engine, Receiver<EngineMessage>) {
    let (tx, rx) = channel();
    let engine = Engine::spawn(&bridge(extra_env), move |m| {
        let _ = tx.send(m);
    })
    .unwrap();
    (engine, rx)
}

fn until(rx: &Receiver<EngineMessage>, kind: &str) -> Vec<protocol::Event> {
    let mut seen = Vec::new();
    loop {
        match rx.recv_timeout(WAIT).expect("engine went silent") {
            EngineMessage::Event(event) => {
                let done = event.kind == kind;
                seen.push(event);
                if done {
                    return seen;
                }
            }
            EngineMessage::Malformed(line) => panic!("malformed protocol line: {line}"),
            EngineMessage::Exited { code, stderr_tail } => {
                panic!("engine exited early ({code:?}): {stderr_tail}")
            }
        }
    }
}

fn generate(id: &str, text: &str) -> Vec<u8> {
    let messages = [Message::text("user", text)];
    protocol::generate_request(id, &messages, &Sampling::default()).unwrap()
}

#[test]
fn generate_stream_and_clean_shutdown() {
    let (engine, rx) = start(&[]);
    let ready = until(&rx, "ready").pop().unwrap();
    assert_eq!(ready.payload["backend"], "fake");
    engine.send(&generate("r1", "Bonjour à tous ✓")).unwrap();
    let events = until(&rx, "done");
    let text: String = events
        .iter()
        .filter(|e| e.kind == "delta")
        .map(|e| e.payload["text"].as_str().unwrap())
        .collect();
    assert_eq!(text, "Bonjour à tous ✓");
    assert!(events.last().unwrap().is_terminal_for_request());
    assert_eq!(engine.shutdown(WAIT), Some(0));
    assert!(matches!(
        rx.recv_timeout(WAIT).unwrap(),
        EngineMessage::Exited { code: Some(0), .. }
    ));
}

#[test]
fn cancel_through_stdin() {
    let (engine, rx) = start(&[("MLXL3_FAKE_DELAY", "0.05")]);
    until(&rx, "ready");
    engine.send(&generate("long", &"mot ".repeat(400))).unwrap();
    until(&rx, "delta");
    engine
        .send(&protocol::control_request("cancel", "long").unwrap())
        .unwrap();
    let events = until(&rx, "cancelled");
    assert!(events.iter().filter(|e| e.kind == "delta").count() < 400);
    engine.send(&generate("after", "ok")).unwrap();
    assert_eq!(
        until(&rx, "done").last().unwrap().request_id.as_deref(),
        Some("after")
    );
}

#[test]
fn fatal_load_error_then_exit_with_stderr() {
    let (engine, rx) = start(&[("MLXL3_FAKE_LOAD_ERROR", "torch sees no GPU")]);
    let error = until(&rx, "error").pop().unwrap();
    assert!(error.is_fatal());
    match rx.recv_timeout(WAIT).unwrap() {
        EngineMessage::Exited { code, .. } => assert_eq!(code, Some(1)),
        other => panic!("expected exit, got {other:?}"),
    }
    assert!(engine.send(&generate("x", "y")).is_err());
}

#[test]
fn drop_kills_a_busy_engine() {
    let (engine, rx) = start(&[("MLXL3_FAKE_DELAY", "0.2")]);
    until(&rx, "ready");
    engine.send(&generate("long", &"x ".repeat(1000))).unwrap();
    until(&rx, "delta");
    drop(engine);
    loop {
        if let EngineMessage::Exited { .. } = rx.recv_timeout(WAIT).expect("no exit after drop") {
            break;
        }
    }
}

#[test]
fn missing_program_is_an_error() {
    let mut command = bridge(&[]);
    command.program = "mlxl3-definitely-missing-binary".into();
    assert!(Engine::spawn(&command, |_| {}).is_err());
    assert!(supervisor::run(&command, None, |_| {}).is_err());
}

#[test]
fn run_streams_lines_and_enforces_timeout() {
    use std::sync::{Arc, Mutex};
    let lines = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&lines);
    let command = Command {
        label: "probe".into(),
        program: python(),
        args: vec![
            "-c".into(),
            "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)".into(),
        ],
        env: vec![],
    };
    let finished = supervisor::run(&command, Some(WAIT), move |l| {
        sink.lock().unwrap().push(l.to_owned())
    })
    .unwrap();
    assert_eq!(finished.code, Some(3));
    assert_eq!(finished.stdout.trim(), "out");
    assert!(finished.stderr_tail.contains("err"));
    let mut seen = lines.lock().unwrap().clone();
    seen.sort();
    assert_eq!(seen, ["err", "out"]);

    let slow = Command {
        args: vec!["-c".into(), "import time; time.sleep(60)".into()],
        ..command
    };
    let started = std::time::Instant::now();
    assert!(supervisor::run(&slow, Some(Duration::from_millis(500)), |_| {}).is_err());
    assert!(started.elapsed() < Duration::from_secs(15));
}
