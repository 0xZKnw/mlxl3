#[allow(dead_code)]
#[path = "../native/src/openai.rs"]
mod openai;

#[cfg(all(
    target_os = "macos",
    feature = "direct-metal",
    feature = "chat",
    not(feature = "mlx")
))]
#[tokio::test]
#[ignore = "physical direct-Metal Flash model over loopback HTTP"]
async fn native_flash_serve_streams_the_same_completion_as_json() -> anyhow::Result<()> {
    use anyhow::{Context, ensure};
    use hyper::{Body, Client, Request, StatusCode, body::to_bytes};
    use serde_json::{Value, json};
    use std::{process::Stdio, time::Duration};
    use tokio::{
        io::{AsyncBufReadExt, BufReader},
        process::Command,
        time::timeout,
    };

    let model = std::env::var("MLXL3_FLASH_MODEL")?;
    let cli = std::env::var("MLXL3_FLASH_TEST_CLI")
        .unwrap_or_else(|_| env!("CARGO_BIN_EXE_mlxl3-rs").into());
    let dir = tempfile::tempdir()?;
    let log = dir.path().join("dyld.log");
    let mut child = Command::new(cli)
        .arg("--registry")
        .arg(dir.path().join("models.json"))
        .arg("serve")
        .arg(&model)
        .args([
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--api-key",
            "fixture-key",
            "--context-length",
            "256",
        ])
        .env("DYLD_PRINT_LIBRARIES", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(std::fs::File::create(&log)?)
        .process_group(0)
        .kill_on_drop(true)
        .spawn()?;
    let pid = i32::try_from(child.id().context("missing server PID")?)?;
    let result = timeout(Duration::from_secs(240), async {
        let mut output = BufReader::new(child.stdout.take().context("missing server stdout")?);
        let mut line = String::new();
        ensure!(output.read_line(&mut line).await? > 0, "server exited before ready");
        let ready: Value = serde_json::from_str(&line)?;
        ensure!(ready["type"] == "ready" && ready["model"] == model);
        let port = u16::try_from(ready["port"].as_u64().context("missing HTTP port")?)?;
        ensure!(port > 0);
        let base = format!("http://127.0.0.1:{port}");
        let client = Client::new();
        let models = format!("{base}/v1/models").parse()?;
        ensure!(client.get(models).await?.status() == StatusCode::UNAUTHORIZED);
        let call = |endpoint: &'static str, body: Option<Value>| {
            let request = Request::builder().uri(format!("{base}{endpoint}")).header("Authorization", "Bearer fixture-key");
            let client = &client;
            async move {
                let request = if let Some(body) = body {
                    request.method("POST").header("Content-Type", "application/json").body(Body::from(body.to_string()))?
                } else { request.body(Body::empty())? };
                let response = client.request(request).await?;
                let status = response.status();
                let text = String::from_utf8(to_bytes(response.into_body()).await?.to_vec())?;
                Ok::<_, anyhow::Error>((status, text))
            }
        };
        let (status, data) = call("/v1/models", None).await?;
        ensure!(status.is_success() && serde_json::from_str::<Value>(&data)?["data"][0]["id"] == model);
        let mut request = json!({"model":model,"messages":[{"role":"user","content":"Combien font 2 + 2 ?"}],"max_tokens":4,"temperature":0,"top_k":1});
        let mut bad = request.clone();
        bad["messages"] = json!([]);
        ensure!(call("/v1/chat/completions", Some(bad)).await?.0 == StatusCode::BAD_REQUEST);
        let (status, plain) = call("/v1/chat/completions", Some(request.clone())).await?;
        ensure!(status.is_success(), "{plain}");
        let plain: Value = serde_json::from_str(&plain)?;
        let choice = &plain["choices"][0];
        ensure!(choice["finish_reason"] == "length" && plain["usage"]["completion_tokens"] == 4);
        request["stream"] = true.into();
        request["stream_options"] = json!({"include_usage":true});
        let (status, stream) = call("/v1/chat/completions", Some(request)).await?;
        ensure!(status.is_success());
        let mut text = [String::new(), String::new()];
        let mut finish = None;
        let mut usage = None;
        let mut done = false;
        for data in stream.lines().filter_map(|line| line.strip_prefix("data: ")) {
            ensure!(!done, "extra SSE data after DONE");
            if data == "[DONE]" { done = true; continue; }
            let chunk: Value = serde_json::from_str(data)?;
            ensure!(chunk["object"] == "chat.completion.chunk");
            if chunk["choices"].as_array().is_some_and(Vec::is_empty) {
                usage = Some(chunk["usage"].clone());
                continue;
            }
            let choice = &chunk["choices"][0];
            for (output, field) in text.iter_mut().zip(["content", "reasoning_content"]) {
                if let Some(part) = choice["delta"][field].as_str() { output.push_str(part); }
            }
            if !choice["finish_reason"].is_null() { finish = Some(choice["finish_reason"].clone()); }
        }
        ensure!(done && text.iter().any(|s| !s.is_empty()) && finish.as_ref() == Some(&choice["finish_reason"]));
        ensure!(usage.as_ref() == Some(&plain["usage"]));
        for (output, field) in text.iter().zip(["content", "reasoning_content"]) {
            ensure!(Some(output.as_str()) == choice["message"][field].as_str());
        }
        eprintln!("HTTP verified. {}", plain["usage"]);
        Ok::<_, anyhow::Error>(())
    }).await;
    unsafe {
        libc::kill(pid, libc::SIGTERM);
    }
    if timeout(Duration::from_secs(12), child.wait())
        .await
        .is_err()
    {
        unsafe {
            libc::killpg(pid, libc::SIGKILL);
        }
        child.wait().await?;
    }
    let loaded = std::fs::read_to_string(log)?;
    result.context("real Flash HTTP request timed out")??;
    ensure!(
        loaded.contains("Metal.framework")
            && !["libmlx.dylib", "libjaccl.dylib", "mlx.metallib"]
                .iter()
                .any(|name| loaded.contains(name))
    );
    Ok(())
}
