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

#[cfg(all(
    target_os = "macos",
    feature = "direct-metal",
    feature = "chat",
    not(feature = "mlx")
))]
#[tokio::test]
#[ignore = "physical Flash bridge, native memory budget, ledger, mode rejection and recovery"]
async fn native_flash_memory_budget_preserves_bridge_recovery() -> anyhow::Result<()> {
    use anyhow::{Context, ensure};
    use serde_json::{Value, json};
    use std::{process::Stdio, time::Duration};
    use tokio::{
        io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
        process::{ChildStdin, ChildStdout, Command},
        time::timeout,
    };

    async fn request(
        input: &mut ChildStdin,
        output: &mut tokio::io::Lines<BufReader<ChildStdout>>,
        value: Value,
        wanted: &str,
    ) -> anyhow::Result<Value> {
        input
            .write_all((value.to_string() + "\n").as_bytes())
            .await?;
        timeout(Duration::from_secs(120), async {
            loop {
                let line = output.next_line().await?.context("memory bridge exited")?;
                let event: Value = serde_json::from_str(&line)?;
                if event["request_id"] != value["request_id"] {
                    continue;
                }
                if event["type"] == wanted {
                    return Ok(event);
                }
                ensure!(event["type"] != "error", "unexpected bridge error {event}");
            }
        })
        .await
        .context("memory bridge request deadline")?
    }

    let model = std::env::var("MLXL3_FLASH_MODEL")?;
    let cli = std::env::var("MLXL3_FLASH_TEST_CLI")
        .unwrap_or_else(|_| env!("CARGO_BIN_EXE_mlxl3-rs").into());
    let registry = tempfile::tempdir()?;
    let mut process = Command::new(cli)
        .arg("--registry")
        .arg(registry.path().join("models.json"))
        .arg("bridge")
        .arg(&model)
        .args(["--context-length", "256"])
        .env("MLXL3_FLASH_REQUEST_MIB", "8192")
        .env(
            "MLXL3_SSD_CACHE_GIB",
            std::env::var("MLXL3_FLASH_TEST_CACHE_GIB").unwrap_or_else(|_| "0.6".into()),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .kill_on_drop(true)
        .spawn()?;
    let mut input = process.stdin.take().context("missing bridge input")?;
    let mut output =
        BufReader::new(process.stdout.take().context("missing bridge output")?).lines();
    let result = timeout(Duration::from_secs(300),async {
        let ready = timeout(Duration::from_secs(60),async {
            loop {
                let line = output.next_line().await?.context("memory bridge exited before ready")?;
                let event: Value = serde_json::from_str(&line)?;
                if event["type"] == "ready" { return Ok::<_,anyhow::Error>(event); }
            }
        }).await.context("memory bridge startup deadline")??;
        ensure!(ready["mtp_supported"]==false && ready["flash"]["memory"]["request_limit_bytes"]==8u64<<30);
        let first = request(&mut input,&mut output,json!({"type":"flash_memory","request_id":"before"}),"flash_memory").await?;
        ensure!(first["ledger"]["expert_slots_by_layer"].as_array().is_some_and(|v|v.len()==48));
        let refused = request(&mut input,&mut output,json!({"type":"set_mtp","request_id":"head","enabled":true,"mtp_head_path":model}),"error").await?;
        ensure!(refused["message"].as_str().is_some_and(|s|s.contains("AR only")));
        ensure!(ready["dflash_supported"]==false && ready["dflash_configure_supported"]==false && ready["mtp_configure_supported"]==false);
        let refused = request(&mut input,&mut output,json!({"type":"set_dflash","request_id":"draft","enabled":true,"dflash_draft_path":"/a/nonexistent/draft/that/must/not/be/opened","dflash_mode":1}),"error").await?;
        ensure!(refused["message"].as_str().is_some_and(|s|s.contains("AR only")));
        let refused = request(&mut input,&mut output,json!({"type":"generate","request_id":"draft-generate","messages":[{"role":"user","content":"Combien font 2 + 2 ?"}],"max_tokens":2,"dflash2":true,"dflash_draft_path":"/a/nonexistent/draft/that/must/not/be/opened","temperature":0,"top_k":1}),"error").await?;
        ensure!(refused["message"].as_str().is_some_and(|s|s.contains("AR only")));
        for id in ["first","repeat"] {
            let completed = request(&mut input,&mut output,json!({"type":"generate","request_id":id,"messages":[{"role":"user","content":"Combien font 2 + 2 ?"}],"max_tokens":2,"temperature":0,"top_k":1,"memory_saver":true}),"complete").await?;
            ensure!(completed["stats"]["generated_tokens"]==2 && completed["token_hash"]=="9f850af1cbba074b");
        }
        let after = request(&mut input,&mut output,json!({"type":"flash_memory","request_id":"after"}),"flash_memory").await?;
        let ledger = &after["ledger"];
        ensure!(ledger["reads_in_flight"]==0 && ledger["memory"]["mlx_active_bytes"]==0);
        if std::env::var("MLXL3_SSD_LOOKAHEAD_DEPTH").ok().is_some_and(|v| v != "0") {
            ensure!(ledger["prefetch"]["submitted"].as_u64().is_some_and(|n| n > 0), "lookahead bridge did not prefetch");
        }
        ensure!(ledger["memory"]["native_metal_allocated_bytes"].as_u64().is_some_and(|n|n>0));
        ensure!(ledger["state_backing_bytes"].get("gdn").is_none() && ledger["state_backing_bytes"].get("last_hidden").is_none());
        Ok::<_,anyhow::Error>(())
    }).await.context("memory bridge campaign deadline");
    let _ = input.write_all(b"{\"type\":\"shutdown\"}\n").await;
    let status = match timeout(Duration::from_secs(12), process.wait()).await {
        Ok(status) => status?,
        Err(_) => {
            process.start_kill()?;
            timeout(Duration::from_secs(3), process.wait())
                .await
                .context("memory bridge cleanup deadline")??
        }
    };
    result??;
    ensure!(status.success(), "memory bridge shutdown failed: {status}");
    eprintln!(
        "Native budget bridge replay matches the independent two-token hash after rejection and memory-saver cleanup"
    );
    Ok(())
}
