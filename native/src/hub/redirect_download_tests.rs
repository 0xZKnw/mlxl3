use super::*;
use std::{net::TcpListener, thread::JoinHandle, time::Instant};

struct Server {
    url: String,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}
impl Server {
    fn new(redirect: Option<u16>, oversize: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}/start", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&stop);
        let worker = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(20);
            while !stopping.load(Ordering::Relaxed) && Instant::now() < deadline {
                let Ok((mut stream, _)) = listener.accept() else {
                    thread::sleep(Duration::from_millis(5));
                    continue;
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(1)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(1)))
                    .unwrap();
                let mut req = Vec::new();
                let mut byte = [0];
                while req.len() < 8192 && !req.ends_with(b"\r\n\r\n") {
                    if stream.read(&mut byte).unwrap_or(0) == 0 {
                        break;
                    }
                    req.push(byte[0]);
                }
                let req = String::from_utf8_lossy(&req);
                if let Some(code) = redirect.filter(|_| req.starts_with("GET /start ")) {
                    let body = "x".repeat(128);
                    let _ = write!(
                        stream,
                        "HTTP/1.1 {} Redirect\r\nLocation: /final\r\nContent-Length: 128\r\nConnection: close\r\n\r\n{body}",
                        code
                    );
                } else {
                    let payload = if oversize { "hello-too-long" } else { "hello" };
                    let offset = req
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("range: bytes=")
                                .and_then(|r| r.trim_end_matches('-').parse::<usize>().ok())
                        })
                        .unwrap_or(0)
                        .min(payload.len());
                    let body = &payload[offset..];
                    let range = if offset > 0 {
                        format!(
                            "Content-Range: bytes {}-{}/{}\r\n",
                            offset,
                            payload.len() - 1,
                            payload.len()
                        )
                    } else {
                        String::new()
                    };
                    let _ = write!(
                        stream,
                        "HTTP/1.1 {} OK\r\n{range}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        if offset > 0 { 206 } else { 200 },
                        body.len()
                    );
                }
            }
        });
        Self {
            url,
            stop,
            worker: Some(worker),
        }
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.worker.take().unwrap().join().unwrap();
    }
}

#[test]
fn redirects_larger_than_final_payload_work_initial_and_resumed() {
    for code in [302, 307] {
        for partial in [false, true] {
            let server = Server::new(Some(code), false);
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("payload");
            if partial {
                fs::write(&path, b"he").unwrap();
            }
            let file = RepoFile {
                rfilename: "payload".into(),
                size: 5,
                blob_id: None,
                lfs: Some(LfsFile {
                    sha256: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
                        .into(),
                }),
            };
            let mut phases = Vec::new();
            ensure_checkpoint_file(
                &server.url,
                &file,
                &path,
                &AtomicBool::new(false),
                |phase| phases.push(phase.to_string()),
            )
            .unwrap();
            assert_eq!(phases.last().map(String::as_str), Some("verify"));
            assert!(phases.iter().any(|phase| phase == "download"));
            assert_eq!(fs::read(path).unwrap(), b"hello");
        }
    }
}

#[test]
fn final_payload_limit_caps_initial_and_resumed_file() {
    for partial in [false, true] {
        let server = Server::new(None, true);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payload");
        if partial {
            fs::write(&path, b"he").unwrap();
        }
        let error =
            download_file_bounded(&server.url, &path, &AtomicBool::new(false), || {}, Some(5))
                .unwrap_err();
        assert!(path.metadata().unwrap().len() <= 5);
        assert!(error.to_string().contains("file size limit"), "{error:#}");
    }
}

#[test]
fn new_attempt_invalidates_completed_marker_even_with_same_identity() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("completed.json");
    fs::write(&marker, b"true").unwrap();
    begin_download_attempt(dir.path()).unwrap();
    assert!(!marker.exists());
    fs::write(&marker, b"true").unwrap();
    begin_download_attempt(dir.path()).unwrap();
    assert!(!marker.exists());
    begin_download_attempt(dir.path()).unwrap();
}

#[test]
fn legacy_completion_markers_never_hide_unverified_installations() {
    let root = tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap();
    let downloads = root.path().join(".downloads");
    let stage = downloads.join("0123456789ab");
    fs::create_dir_all(&stage).unwrap();
    let request = PendingDownload {
        id: "0123456789ab".into(),
        repo: "test/model".into(),
        commit: "a".repeat(40),
        folder: "".into(),
        destination: root.path().join("installed").display().to_string(),
        name: "model".into(),
        size_bytes: 5,
        retained_bytes: 0,
    };
    write_json(&stage.join("request.json"), &request).unwrap();
    assert_eq!(pending_downloads_in(&downloads).unwrap().len(), 1);
    write_json(&stage.join("completed.json"), &request).unwrap();
    assert_eq!(pending_downloads_in(&downloads).unwrap().len(), 1);
    fs::write(stage.join("completed.json"), b"not JSON").unwrap();
    assert_eq!(pending_downloads_in(&downloads).unwrap().len(), 1);
    let mut foreign = serde_json::to_value(&request).unwrap();
    foreign["commit"] = serde_json::json!("b".repeat(40));
    write_json(&stage.join("completed.json"), &foreign).unwrap();
    assert_eq!(pending_downloads_in(&downloads).unwrap().len(), 1);
    write_json(&stage.join("completed.json"), &request).unwrap();
    // Same identity retry invalidates the old marker before the first fallible step.
    begin_download_attempt(&stage).unwrap();
    assert_eq!(pending_downloads_in(&downloads).unwrap()[0].id, request.id);
    write_json(&stage.join("completed.json"), &request).unwrap();
    assert_eq!(pending_downloads_in(&downloads).unwrap().len(), 1);
}
