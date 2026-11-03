//! Resident Flash bridge over the native Metal session.
use super::*;
use mlxl3_native::{
    gpu::{FlashMtpHead, FlashMtpSession, FlashNext, Metal},
    tokenizer::ChatTokenizer,
};
use std::path::Path;

pub(super) type NativeChatModel<'a> = FlashNext<'a>;
pub(super) type ChatMtpHead<'a> = FlashMtpHead<'a>;
// Flash re-prefills each request, as does the stock Flash bridge.
pub(super) type PromptCache = std::convert::Infallible;

pub(super) fn prefill(
    model: &mut NativeChatModel,
    head: Option<&mut ChatMtpHead>,
    tokens: &[u32],
    depth: usize,
    cancelled: &AtomicBool,
) -> Result<(Vec<u16>, usize, Option<FlashMtpSession>)> {
    if let Some(head) = head {
        anyhow::ensure!(depth == 1, "Flash MTP supports depth 1");
        let mut session = FlashMtpSession::default();
        let logits = session.prefill(model, head, tokens, cancelled)?;
        return Ok((logits, 0, Some(session)));
    }
    model.reset();
    let mut last = None;
    for chunk in tokens.chunks(mlxl3_native::flash_support::PREFILL_CHUNK) {
        anyhow::ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
        last = Some(model.prefill(chunk)?);
    }
    Ok((last.context("empty Flash prompt")?, 0, None))
}

pub(super) fn select(
    model: &NativeChatModel,
    logits: &[u16],
    generated: &[u32],
    temperature: f32,
    top_k: usize,
    penalty: f32,
    random: &mut u64,
) -> Result<u32> {
    anyhow::ensure!(
        temperature.is_finite() && temperature >= 0.,
        "temperature must be non-negative"
    );
    anyhow::ensure!(
        penalty.is_finite() && penalty > 0.,
        "repetition_penalty must be positive"
    );
    let values = model.log_probs(logits)?;
    if (temperature == 0. || top_k == 1) && penalty == 1. {
        mlxl3_native::speculative::greedy_log_prob_id(&values)
            .context("invalid Flash log probabilities")
    } else {
        sample_values(values, generated, temperature, top_k, penalty, random)
    }
}

fn prepare_head<'a>(
    requested: &str,
    c: &checkpoint::Checkpoint,
    model: &NativeChatModel<'a>,
    head: &mut Option<(String, FlashMtpHead<'a>)>,
) -> Result<bool> {
    anyhow::ensure!(
        !requested.trim().is_empty(),
        "MTP head folder is not configured"
    );
    let path = registry::expand_home(Path::new(requested))?.canonicalize()?;
    anyhow::ensure!(path == c.path, "Flash MTP requires the target checkpoint");
    let revision = checkpoint::artifact_key(&path)?;
    if head.as_ref().is_some_and(|(key, _)| *key == revision) {
        return Ok(false);
    }
    *head = None;
    *head = Some((revision, FlashMtpHead::load(c, model)?));
    Ok(true)
}

fn prepare_draft(
    request: &BridgeRequest,
    model: &NativeChatModel,
    slot: &mut Option<(String, usize, RoundDraft)>,
) -> Result<bool> {
    let path = registry::expand_home(Path::new(&request.dflash_draft_path))?.canonicalize()?;
    let revision = checkpoint::artifact_key(&path)?;
    if slot
        .as_ref()
        .is_some_and(|(key, mode, _)| *key == revision && *mode == request.dflash_mode)
    {
        return Ok(false);
    }
    *slot = None;
    *slot = Some((
        revision,
        request.dflash_mode,
        RoundDraft::load(&path, request.dflash_mode, model)?,
    ));
    Ok(true)
}

pub(super) fn run(registry_path: Option<&Path>, name: &str, requested: i32) -> Result<()> {
    let registry_path = registry_path
        .map(Path::to_path_buf)
        .map(Ok)
        .unwrap_or_else(registry::default_path)?;
    let path = resolve_model_path(Some(&registry_path), name)?.canonicalize()?;
    let c = checkpoint::inspect(&path)?;
    emit_event(json!({"type":"loading","model":name}))?;
    let started = Instant::now();
    let tokenizer = ChatTokenizer::load(&path)?;
    let gpu = Metal::new()?;
    let mut model = FlashNext::load(&gpu, &c, 0)?;
    let limit = model.context_limit();
    let speculative = model.request_memory_limit().is_none();
    anyhow::ensure!(
        requested >= 0 && requested as u32 <= limit,
        "invalid requested context length"
    );
    let context = if requested == 0 {
        limit
    } else {
        requested as u32
    } as usize;
    let size = c.size_bytes as f64 / 1e9;
    let mut head: Option<(String, FlashMtpHead)> = None;
    let mut draft: Option<(String, usize, RoundDraft)> = None;
    let mut mcp = mlxl3_native::mcp::Manager::disabled();
    let cancelled = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGUSR1, Arc::clone(&cancelled))?;
    let mut random = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
        | 1;
    emit_event(
        json!({"type":"ready","model":name,"model_path":path,"load_seconds":started.elapsed().as_secs_f64(),"resident_gb":size,"memory":model.memory_stats()?,"runtime_commit":env!("MLXL3_BUILD_REVISION"),"runtime_profile":env!("MLXL3_BUILD_PROFILE"),"runtime_executable":std::env::current_exe()?,"backend":"direct-metal","runtime_version":env!("CARGO_PKG_VERSION"),"bridge_protocol":1,"modules":c.modules.len(),"dflash_supported":speculative,"dflash_tune_supported":false,"dflash_configure_supported":speculative,"mtp_configure_supported":true,"mtp_auto_download_supported":false,"mtp_supported":speculative,"mtp_max_depth":1,"mtp_tune_supported":false,"memory_saver_supported":true,"mcp_servers":0,"mcp_tools":0,"mcp_errors":{},"workspace_features":true,"context_limit":context,"model_context_limit":limit,"flash":{"backend":"direct-metal","memory":model.memory_ledger(None)?}}),
    )?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request: BridgeRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                emit_event(
                    json!({"type":"error","request_id":"","message":format!("invalid bridge request: {e}")}),
                )?;
                continue;
            }
        };
        if request.kind == "shutdown" {
            break;
        }
        let id = request.request_id.clone();
        let result = (|| -> Result<()> {
            match request.kind.as_str() {
                "flash_memory" => emit_event(
                    json!({"type":"flash_memory","request_id":id,"ledger":model.memory_ledger(head.as_ref().map(|(_,head)|head))?}),
                ),
                "ping" => emit_event(
                    json!({"type":"pong","request_id":id,"memory":model.memory_stats()?}),
                ),
                "set_mcp" => {
                    mcp.set_enabled(&registry_path, request.enabled, true);
                    emit_event(
                        json!({"type":"mcp_status","mcp_servers":mcp.server_count(),"mcp_tools":mcp.tools.len(),"mcp_errors":mcp.errors,"mcp_tool_catalog":mcp.chat_tools()}),
                    )
                }
                "set_mtp" => {
                    model.reset();
                    if request.enabled {
                        draft = None;
                        prepare_head(&request.mtp_head_path, &c, &model, &mut head)?;
                    } else {
                        head = None;
                    }
                    emit_event(
                        json!({"type":"mtp_status","request_id":id,"mtp_active":head.is_some(),"mtp_head_path":head.as_ref().map(|_|&path),"memory":model.memory_stats()?}),
                    )
                }
                "set_dflash" => {
                    model.reset();
                    if request.enabled {
                        head = None;
                        prepare_draft(&request, &model, &mut draft)?;
                    } else {
                        draft = None;
                    }
                    emit_event(
                        json!({"type":"dflash_status","request_id":id,"dflash_active":draft.is_some(),"memory":model.memory_stats()?}),
                    )
                }
                "tune_dflash" => {
                    bail!("Flash DFlash2 learns its verify budget online; Tune is not supported")
                }
                "tune_mtp" => bail!("Tune MTP supports Qwen3.5/3.6/3.8 targets"),
                "generate" => {
                    cancelled.store(false, Ordering::Relaxed);
                    anyhow::ensure!(
                        !(request.response_schema.is_some() && request.mcp_enabled),
                        "JSON output and MCP must be used separately"
                    );
                    anyhow::ensure!(
                        !(request.mtp && request.dflash2),
                        "Choose either MTP or DFlash2"
                    );
                    if let Some(schema) = &request.response_schema {
                        mlxl3_native::structured::Schema::parse(schema)?;
                    }
                    let started = Instant::now();
                    let active = request.mtp && request.greedy_unpenalized();
                    let draft_seconds = if active {
                        anyhow::ensure!(request.mtp_depth == 1, "Flash MTP supports depth 1");
                        if prepare_head(&request.mtp_head_path, &c, &model, &mut head)? {
                            started.elapsed().as_secs_f64()
                        } else {
                            0.
                        }
                    } else {
                        head = None;
                        0.
                    };
                    let draft_seconds = if request.dflash2 {
                        anyhow::ensure!(
                            request.greedy_unpenalized(),
                            "DFlash2 requires greedy sampling and repetition penalty 1.0"
                        );
                        head = None;
                        if prepare_draft(&request, &model, &mut draft)? {
                            started.elapsed().as_secs_f64()
                        } else {
                            0.
                        }
                    } else {
                        draft = None;
                        draft_seconds
                    };
                    mcp.set_enabled(&registry_path, request.mcp_enabled, false);
                    emit_generation_mode(
                        &request,
                        active,
                        draft.is_some(),
                        draft.as_ref().map_or(0, |(_, _, d)| d.width()),
                        draft.as_ref().map(|_| request.dflash_draft_path.clone()),
                    )?;
                    let result = bridge_generate(
                        &mut model,
                        &tokenizer,
                        &id,
                        request.messages,
                        request.max_tokens,
                        request.temperature,
                        request.top_k,
                        request.repetition_penalty,
                        context,
                        size,
                        &cancelled,
                        &mut random,
                        &mut mcp,
                        draft.as_ref().map(|(_, _, d)| d),
                        head.as_mut().map(|(_, h)| h),
                        request.mtp_depth,
                        request.dflash_mode,
                        started,
                        draft_seconds,
                        &mut None,
                        if request.conversation_id.is_empty() {
                            &id
                        } else {
                            &request.conversation_id
                        },
                        request.reuse_prompt_cache && !request.memory_saver,
                        request.allowed_tools.as_deref(),
                        request.response_schema.as_deref(),
                    );
                    if request.memory_saver {
                        if let Err(error) = model.release_idle_state() {
                            eprintln!("Idle memory cleanup failed: {error:#}");
                        }
                    } else {
                        model.reset();
                    }
                    if let Some((_, _, d)) = &draft {
                        d.reset();
                    }
                    if let Some((_, h)) = &mut head {
                        h.reset();
                    }
                    result
                }
                other => bail!("unsupported request type: {other}"),
            }
        })();
        if let Err(e) = result {
            model.reset();
            head = None;
            draft = None;
            if cancelled.swap(false, Ordering::Relaxed) {
                emit_event(json!({"type":"cancelled","request_id":id}))?;
            } else {
                emit_event(
                    json!({"type":"error","request_id":id,"message":e.to_string(),"mtp_active":false,"dflash_active":false}),
                )?;
            }
        }
    }
    Ok(())
}
