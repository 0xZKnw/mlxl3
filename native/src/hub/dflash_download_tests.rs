use super::*;
use crate::dflash::ManagedDraft;
use std::{
    io::Read,
    net::{TcpListener, TcpStream},
    os::unix::fs::symlink,
    thread::JoinHandle,
    time::Instant,
};

const CONFIG: &[u8] = b"{\"fixture\":true}\n";
const WEIGHTS: &[u8] = b"raw draft weights\n";
const FILES: [(&str, u64, &str); 2] = [
    (
        "config.json",
        17,
        "218589323cbe80b7ed077e3ee36f1663e7cb5f8f4e4ad02c938ad8a5c2c5a6b9",
    ),
    (
        "model.safetensors",
        18,
        "1efc4ce368ebe8ca930ad4c2e3a2db92c6f86d76d1fc6e3aa0917cd6c3c9dc1c",
    ),
];

fn metadata(files: &[(&str, u64, &str)]) -> Value {
    serde_json::json!({
        "id": ManagedDraft::FlashNext.repository(),
        "sha": ManagedDraft::FlashNext.revision(),
        "siblings": files.iter().map(|(name, size, hash)| serde_json::json!({
            "rfilename": name, "size": size,
            "lfs": if *name == "config.json" { Value::Null } else { serde_json::json!({"sha256": hash}) }
        })).collect::<Vec<_>>()
    })
}

#[derive(Clone)]
enum Reply {
    Body(Vec<u8>),
    ServerError,
    Short(Vec<u8>),
    Stall(Vec<u8>),
}

#[derive(Debug)]
struct Request {
    path: String,
    range: Option<usize>,
    authorization_seen: bool,
}

struct Server {
    host: String,
    stop: Arc<AtomicBool>,
    model_started: Arc<AtomicBool>,
    worker: Option<JoinHandle<Vec<Request>>>,
}

impl Server {
    fn new(fault: Option<(&str, Reply)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let host = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&stop);
        let model_started = Arc::new(AtomicBool::new(false));
        let started = Arc::clone(&model_started);
        let draft = ManagedDraft::FlashNext;
        let api = format!(
            "/api/models/{}/revision/{}?blobs=true",
            draft.repository(),
            draft.revision()
        );
        let path = |name| {
            format!(
                "/{}/resolve/{}/{name}?download=true",
                draft.repository(),
                draft.revision()
            )
        };
        let mut replies = BTreeMap::from([
            (
                api.clone(),
                Reply::Body(serde_json::to_vec(&metadata(&FILES)).unwrap()),
            ),
            (path("config.json"), Reply::Body(CONFIG.to_vec())),
            (path("model.safetensors"), Reply::Body(WEIGHTS.to_vec())),
        ]);
        if let Some((name, reply)) = fault {
            replies.insert(if name == "metadata" { api } else { path(name) }, reply);
        }
        let worker = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(30);
            let mut seen = Vec::new();
            while !stopping.load(Ordering::Relaxed) && Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut socket, _)) => {
                        if let Some(request) = read_request(&mut socket) {
                            if request.path.ends_with("/model.safetensors?download=true") {
                                started.store(true, Ordering::Relaxed);
                            }
                            let reply = replies
                                .get(&request.path)
                                .cloned()
                                .unwrap_or(Reply::ServerError);
                            let _ = respond(&mut socket, reply, request.range, &stopping, deadline);
                            seen.push(request);
                        }
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(_) => break,
                }
            }
            seen
        });
        Self {
            host,
            stop,
            model_started,
            worker: Some(worker),
        }
    }

    fn finish(&mut self) -> Vec<Request> {
        self.stop.store(true, Ordering::Relaxed);
        self.worker
            .take()
            .unwrap()
            .join()
            .expect("loopback server panicked")
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn read_request(socket: &mut TcpStream) -> Option<Request> {
    socket
        .set_read_timeout(Some(Duration::from_millis(500)))
        .ok()?;
    socket
        .set_write_timeout(Some(Duration::from_millis(500)))
        .ok()?;
    let mut bytes = Vec::new();
    while bytes.len() < 8192 && !bytes.ends_with(b"\r\n\r\n") {
        let mut byte = [0u8; 1];
        socket.read_exact(&mut byte).ok()?;
        bytes.push(byte[0]);
    }
    if !bytes.ends_with(b"\r\n\r\n") {
        return None;
    }
    let text = std::str::from_utf8(&bytes).ok()?;
    let path = text.lines().next()?.split_whitespace().nth(1)?.to_owned();
    let mut range = None;
    let mut authorization_seen = false;
    for line in text.lines().skip(1) {
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("range") {
                range = value
                    .trim()
                    .strip_prefix("bytes=")?
                    .strip_suffix('-')?
                    .parse()
                    .ok();
            }
            authorization_seen |= name.eq_ignore_ascii_case("authorization");
        }
    }
    Some(Request {
        path,
        range,
        authorization_seen,
    })
}

fn respond(
    socket: &mut TcpStream,
    reply: Reply,
    range: Option<usize>,
    stop: &AtomicBool,
    deadline: Instant,
) -> std::io::Result<()> {
    match reply {
        Reply::Body(body) => {
            if let Some(start) = range.filter(|&start| start > 0) {
                if start >= body.len() {
                    return write!(
                        socket,
                        "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                }
                write!(
                    socket,
                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nConnection: close\r\n\r\n",
                    body.len() - start,
                    start,
                    body.len() - 1,
                    body.len()
                )?;
                socket.write_all(&body[start..])
            } else {
                write!(
                    socket,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )?;
                socket.write_all(&body)
            }
        }
        Reply::ServerError => write!(
            socket,
            "HTTP/1.1 503 Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        ),
        Reply::Short(body) => {
            write!(
                socket,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len() + 1
            )?;
            socket.write_all(&body)
        }
        Reply::Stall(body) => {
            write!(
                socket,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )?;
            socket.write_all(&body[..2])?;
            socket.flush()?;
            while !stop.load(Ordering::Relaxed) && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(2));
            }
            Ok(())
        }
    }
}

fn fixture_inspect(path: &Path) -> Result<()> {
    ensure!(
        serde_json::from_slice::<Value>(&fs::read(path.join("config.json"))?)?["fixture"] == true,
        "invalid fixture config"
    );
    ensure!(
        fs::read(path.join("model.safetensors"))? == WEIGHTS,
        "invalid fixture weights"
    );
    Ok(())
}

fn install(root: &Path, host: &str, cancel: &AtomicBool) -> Result<PathBuf> {
    install_dflash(
        root,
        ManagedDraft::FlashNext,
        &FILES,
        host,
        cancel,
        |_, _| {},
        fixture_inspect,
    )
}

fn stage(root: &Path) -> PathBuf {
    root.join(".downloads")
        .join(ManagedDraft::FlashNext.revision())
}
fn destination(root: &Path) -> PathBuf {
    root.join(ManagedDraft::FlashNext.name())
}

fn populate(path: &Path) {
    fs::create_dir_all(path).unwrap();
    fs::write(path.join("config.json"), CONFIG).unwrap();
    fs::write(path.join("model.safetensors"), WEIGHTS).unwrap();
}

#[test]
fn published_flash_metadata_accepts_git_config_and_rejects_pin_changes() {
    let original = metadata(&DFLASH_FLASH_PINNED);
    let parsed = || serde_json::from_value::<RepoInfo>(original.clone()).unwrap();
    published_dflash_matches(&parsed(), ManagedDraft::FlashNext, &DFLASH_FLASH_PINNED).unwrap();
    for case in 0..8 {
        let mut changed = original.clone();
        match case {
            0 => changed["id"] = serde_json::json!("unrelated/repository"),
            1 => changed["sha"] = serde_json::json!("0".repeat(40)),
            2 => changed["siblings"][1]["lfs"] = Value::Null,
            3 => changed["siblings"][1]["lfs"]["sha256"] = serde_json::json!("0".repeat(64)),
            4 => changed["siblings"][0]["size"] = serde_json::json!(1235),
            5 => {
                changed["siblings"].as_array_mut().unwrap().pop();
            }
            6 => {
                let duplicate = changed["siblings"][0].clone();
                changed["siblings"].as_array_mut().unwrap().push(duplicate);
            }
            _ => changed["siblings"][0]["lfs"] = serde_json::json!({"sha256":"0".repeat(64)}),
        }
        let parsed: RepoInfo = serde_json::from_value(changed).unwrap();
        assert!(
            published_dflash_matches(&parsed, ManagedDraft::FlashNext, &DFLASH_FLASH_PINNED)
                .is_err(),
            "invalid pin case {case} accepted"
        );
    }
}

#[test]
fn flash_cache_hit_never_needs_server_and_corrupt_destination_is_preserved() {
    let root = tempfile::tempdir().unwrap();
    let installed = destination(root.path());
    populate(&installed);
    assert_eq!(
        install(root.path(), "http://127.0.0.1:1", &AtomicBool::new(false)).unwrap(),
        installed.canonicalize().unwrap()
    );
    let corrupt = vec![b'x'; WEIGHTS.len()];
    fs::write(installed.join("model.safetensors"), &corrupt).unwrap();
    assert!(install(root.path(), "http://127.0.0.1:1", &AtomicBool::new(false)).is_err());
    assert_eq!(
        fs::read(installed.join("model.safetensors")).unwrap(),
        corrupt
    );
    assert!(!root.path().join(".downloads").exists());
}

#[test]
fn raw_flash_fresh_download_reuses_range_partial_and_replaces_full_corruption() {
    for seed in 0..3 {
        let root = tempfile::tempdir().unwrap();
        let staging = stage(root.path());
        if seed > 0 {
            populate(&staging);
            fs::write(
                staging.join("model.safetensors"),
                if seed == 1 {
                    WEIGHTS[..7].to_vec()
                } else {
                    vec![b'x'; WEIGHTS.len()]
                },
            )
            .unwrap();
        }
        let mut server = Server::new(None);
        let installed = install(root.path(), &server.host, &AtomicBool::new(false)).unwrap();
        fixture_inspect(&installed).unwrap();
        assert!(!staging.exists());
        let requests = server.finish();
        assert_eq!(
            requests.len(),
            if seed == 0 { 3 } else { 2 },
            "only metadata and uncached files should transfer"
        );
        assert!(requests[0].path.starts_with("/api/models/"));
        let model = requests.last().unwrap();
        assert!(model.path.ends_with("/model.safetensors?download=true"));
        assert_eq!(
            model.range.filter(|&n| n > 0),
            if seed == 1 { Some(7) } else { None }
        );
        assert!(requests.iter().all(|r| !r.authorization_seen));
    }
}

#[test]
fn raw_flash_server_failure_short_body_and_inspection_refusal_never_publish() {
    for reply in [Reply::ServerError, Reply::Short(b"{".to_vec())] {
        let root = tempfile::tempdir().unwrap();
        let mut server = Server::new(Some(("metadata", reply)));
        assert!(install(root.path(), &server.host, &AtomicBool::new(false)).is_err());
        assert!(!destination(root.path()).exists());
        assert!(!server.finish().is_empty());
    }
    let root = tempfile::tempdir().unwrap();
    let mut server = Server::new(None);
    let result = install_dflash(
        root.path(),
        ManagedDraft::FlashNext,
        &FILES,
        &server.host,
        &AtomicBool::new(false),
        |_, _| {},
        |_| anyhow::bail!("rejected test geometry"),
    );
    assert!(result.is_err());
    assert!(!destination(root.path()).exists());
    assert!(stage(root.path()).join("model.safetensors").is_file());
    assert_eq!(server.finish().len(), 3);
}

#[test]
fn raw_flash_cancel_before_and_during_transfer_keeps_resume_data() {
    let root = tempfile::tempdir().unwrap();
    assert!(install(root.path(), "http://127.0.0.1:1", &AtomicBool::new(true)).is_err());
    assert!(!destination(root.path()).exists());
    let root = tempfile::tempdir().unwrap();
    let cancelled = Arc::new(AtomicBool::new(false));
    let mut server = Server::new(Some(("model.safetensors", Reply::Stall(WEIGHTS.to_vec()))));
    let started = Arc::clone(&server.model_started);
    let cancel = Arc::clone(&cancelled);
    let cancelling = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(2);
        while !started.load(Ordering::Relaxed) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(2));
        }
        let observed = started.load(Ordering::Relaxed);
        thread::sleep(Duration::from_millis(150));
        cancel.store(true, Ordering::Relaxed);
        observed
    });
    let result = install_dflash(
        root.path(),
        ManagedDraft::FlashNext,
        &FILES,
        &server.host,
        &cancelled,
        |_, _| {},
        fixture_inspect,
    );
    let transfer_started = cancelling.join().unwrap();
    let requests = server.finish();
    assert!(transfer_started, "cancelled before transfer began");
    assert!(result.unwrap_err().to_string().contains("cancelled"));
    assert!(cancelled.load(Ordering::Relaxed));
    assert!(!destination(root.path()).exists());
    let partial = stage(root.path()).join("model.safetensors");
    assert!(partial.is_file());
    assert!(partial.metadata().unwrap().len() < WEIGHTS.len() as u64);
    assert_eq!(requests.len(), 3);
}

#[test]
fn raw_flash_rejects_symlink_roots_locks_files_and_extra_entries() {
    for case in 0..6 {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("sentinel"), b"preserve").unwrap();
        let downloads = root.path().join(".downloads");
        fs::create_dir(&downloads).unwrap();
        match case {
            0 => symlink(outside.path(), destination(root.path())).unwrap(),
            1 => symlink(outside.path(), stage(root.path())).unwrap(),
            2 => symlink(
                outside.path().join("sentinel"),
                downloads.join("dflash.lock"),
            )
            .unwrap(),
            3 => {
                fs::remove_dir(&downloads).unwrap();
                symlink(outside.path(), &downloads).unwrap();
            }
            4 => {
                populate(&stage(root.path()));
                fs::remove_file(stage(root.path()).join("config.json")).unwrap();
                symlink(
                    outside.path().join("sentinel"),
                    stage(root.path()).join("config.json"),
                )
                .unwrap();
            }
            _ => {
                populate(&stage(root.path()));
                symlink(outside.path(), stage(root.path()).join("unexpected")).unwrap();
            }
        }
        let error = install(root.path(), "http://127.0.0.1:1", &AtomicBool::new(false))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("invalid DFlash")
                || error.contains("unexpected file")
                || error.contains("regular"),
            "symlink case {case}: {error}"
        );
        assert_eq!(
            fs::read(outside.path().join("sentinel")).unwrap(),
            b"preserve"
        );
    }
    let root = tempfile::tempdir().unwrap();
    populate(&destination(root.path()));
    fs::write(destination(root.path()).join("extra"), b"preserve").unwrap();
    assert!(install(root.path(), "http://127.0.0.1:1", &AtomicBool::new(false)).is_err());
    assert_eq!(
        fs::read(destination(root.path()).join("extra")).unwrap(),
        b"preserve"
    );
}

#[test]
fn retained_size_does_not_follow_symlink_cycles() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("real"), b"12345").unwrap();
    symlink(directory.path(), directory.path().join("cycle")).unwrap();
    assert_eq!(directory_bytes(directory.path()), 5);
}
