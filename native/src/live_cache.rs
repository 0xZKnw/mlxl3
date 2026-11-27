//! AGENT-2026-11-26-LIVE-CACHE: token bookkeeping for a prompt that goes on from the state the
//! model decoded with, instead of from a saved prefill boundary. CPU only: these functions
//! decide whether a state can be continued and from where; they say nothing about its values.

/// What a model state holds beyond a saved snapshot of the same conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveTail {
    /// The tokens after the snapshot's own: the rest of that round's prompt, then the part of
    /// the generated stream the model committed.
    pub tokens: Vec<u32>,
    /// The rendered prompt of that round followed by the text of the committed stream, when
    /// that prompt was its rendered text and nothing else.
    pub text: Option<String>,
}

/// A Flash Agent turn can hold a live prefix only on an eligible AR route and while the request
/// opts into prefix reuse, has tools and carries the Agent control channel.
pub fn flash_agent_cache_eligible(
    model_supports_live_cache: bool,
    requested: bool,
    reuse: bool,
    has_tools: bool,
    agent_request: bool,
    ordinary_decode: bool,
) -> bool {
    model_supports_live_cache && requested && reuse && has_tools && agent_request && ordinary_decode
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(8)]
fn flash_agent_cache_requires_route_request_and_every_safety_gate() {
    // Comparison of the two six-element gate arrays has a fixed bound.
    let capable = kani::any();
    let requested = kani::any();
    let reuse = kani::any();
    let tools = kani::any();
    let agent = kani::any();
    let ordinary = kani::any();
    let eligible = flash_agent_cache_eligible(capable, requested, reuse, tools, agent, ordinary);
    let gates = [capable, requested, reuse, tools, agent, ordinary];
    assert_eq!(eligible, gates == [true; 6]);
    kani::cover!(eligible);
    kani::cover!(gates == [false, true, true, true, true, true] && !eligible);
    kani::cover!(gates == [true, false, true, true, true, true] && !eligible);
    kani::cover!(gates == [true, true, false, true, true, true] && !eligible);
    kani::cover!(gates == [true, true, true, false, true, true] && !eligible);
    kani::cover!(gates == [true, true, true, true, false, true] && !eligible);
    kani::cover!(gates == [true, true, true, true, true, false] && !eligible);
}

/// The generated tokens a model at `position` has committed after a prompt of `prompt` tokens.
/// The stream is what the round delivered (`generated`), then the end token it stopped on
/// (`end`, never delivered). A decoder commits its tokens in stream order, so the committed
/// ones are a prefix of the stream; None when the model is before the prompt's end or past
/// the stream (rows committed for tokens that were never delivered).
pub fn committed_stream(
    prompt: usize,
    generated: &[u32],
    end: Option<u32>,
    position: usize,
) -> Option<Vec<u32>> {
    let committed = position.checked_sub(prompt)?;
    let stream = generated.iter().copied().chain(end);
    let held: Vec<u32> = stream.take(committed).collect();
    (held.len() == committed).then_some(held)
}

/// Where the prefill of `prompt` starts when the model holds `snapshot` then `tail` and sits
/// at `position` (None while it has uncommitted rows): the end of what it holds, when the
/// prompt starts with exactly those tokens and has at least one more.
pub fn live_start(
    snapshot: &[u32],
    tail: &[u32],
    prompt: &[u32],
    position: Option<usize>,
) -> Option<usize> {
    let held = snapshot.len().checked_add(tail.len())?;
    (position == Some(held)
        && held < prompt.len()
        && prompt.starts_with(snapshot)
        && prompt[snapshot.len()..held] == *tail)
        .then_some(held)
}

/// Why [`live_start`] refused, for the trace.
pub fn refusal(snapshot: &[u32], tail: &[u32], prompt: &[u32], position: Option<usize>) -> String {
    let held = snapshot.len() + tail.len();
    let Some(position) = position else {
        return "the model has uncommitted rows".into();
    };
    if position != held {
        return format!("the model is at {position}, its known tokens end at {held}");
    }
    let known = snapshot.iter().chain(tail);
    match known.zip(prompt).position(|(a, b)| a != b) {
        Some(index) => format!("the prompt differs at token {index} of {held} held"),
        None if prompt.len() <= held => {
            format!(
                "the prompt ({} tokens) adds nothing to the {held} held",
                prompt.len()
            )
        }
        None => "nothing".into(),
    }
}

/// The tokens of a prompt whose rendered text is `live_text` and more: the tokens the model
/// holds (`held`, kept as the model wrote them) followed by the encoding of the rest. None
/// when the text does not go on from `live_text`, adds nothing, or does not go on with a
/// control token (`is_control`): the rest must be encodable on its own.
pub fn extend_tokens<E>(
    held: &[u32],
    live_text: &str,
    rendered: &str,
    encode: impl FnOnce(&str) -> Result<Vec<u32>, E>,
    is_control: impl FnOnce(u32) -> bool,
) -> Result<Option<Vec<u32>>, E> {
    let Some(rest) = rendered
        .strip_prefix(live_text)
        .filter(|rest| !rest.is_empty())
    else {
        return Ok(None);
    };
    let rest = encode(rest)?;
    Ok(match rest.first() {
        Some(&first) if is_control(first) => Some([held, &rest].concat()),
        _ => None,
    })
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(16)]
fn live_start_requires_exact_tokens_and_committed_offset() {
    // Each held segment has up to three tokens. The next prompt has up to seven,
    // enough to include both held segments and a suffix. IDs and offsets are unrestricted.
    // Slice equality can compare up to 12 bytes in memcmp before reaching its exit.
    let snapshot: [u32; 3] = kani::any();
    let tail: [u32; 3] = kani::any();
    let prompt: [u32; 7] = kani::any();
    let snapshot_len = usize::from(kani::any::<u8>() % 4);
    let tail_len = usize::from(kani::any::<u8>() % 4);
    let prompt_len = usize::from(kani::any::<u8>() % 8);
    let position: Option<usize> = kani::any();
    let held = snapshot_len + tail_len;

    let resumed = live_start(
        &snapshot[..snapshot_len],
        &tail[..tail_len],
        &prompt[..prompt_len],
        position,
    );
    let first_mismatch = (0..held.min(prompt_len)).find(|&index| {
        let token = if index < snapshot_len {
            snapshot[index]
        } else {
            tail[index - snapshot_len]
        };
        prompt[index] != token
    });
    if position == Some(held) && prompt_len > held && first_mismatch.is_none() {
        assert_eq!(resumed, Some(held));
    } else {
        assert_eq!(resumed, None);
    }
    kani::cover!(resumed == Some(6) && snapshot_len == 3 && tail_len == 3);
    kani::cover!(resumed == Some(3) && snapshot_len == 3 && tail_len == 0);
    kani::cover!(position == Some(held) && prompt_len > held && first_mismatch == Some(0));
    kani::cover!(
        tail_len > 0
            && position == Some(held)
            && prompt_len > held
            && first_mismatch == Some(snapshot_len)
    );
    kani::cover!(position == Some(held) && held == prompt_len && first_mismatch.is_none());
    kani::cover!(position.is_none() && prompt_len > held && first_mismatch.is_none());
    kani::cover!(position == Some(held + 1) && prompt_len > held && first_mismatch.is_none());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_agent_cache_needs_all_six_gates() {
        for mask in 0_u8..64 {
            let enabled = |bit: u32| (mask >> bit) & 1 == 1;
            assert_eq!(
                flash_agent_cache_eligible(
                    enabled(0),
                    enabled(1),
                    enabled(2),
                    enabled(3),
                    enabled(4),
                    enabled(5),
                ),
                mask == 63,
                "Flash cache guard mask {mask:06b}"
            );
        }
    }

    #[test]
    fn the_committed_part_is_a_prefix_of_the_stream() {
        let generated = [7, 8, 9];
        // the last delivered token is not committed yet, then it is, then the end token too
        assert_eq!(
            committed_stream(10, &generated, Some(1), 12),
            Some(vec![7, 8])
        );
        assert_eq!(
            committed_stream(10, &generated, Some(1), 13),
            Some(vec![7, 8, 9])
        );
        assert_eq!(
            committed_stream(10, &generated, Some(1), 14),
            Some(vec![7, 8, 9, 1])
        );
        assert_eq!(committed_stream(10, &generated, None, 10), Some(vec![]));
        // rows committed past what was delivered: unknown tokens
        assert_eq!(committed_stream(10, &generated, Some(1), 15), None);
        assert_eq!(committed_stream(10, &generated, None, 14), None);
        // before the end of the prompt
        assert_eq!(committed_stream(10, &generated, Some(1), 9), None);
        assert_eq!(committed_stream(0, &[], None, 0), Some(vec![]));
        assert_eq!(committed_stream(usize::MAX, &generated, Some(1), 3), None);
    }

    #[test]
    fn a_prompt_goes_on_only_from_exactly_what_the_model_holds() {
        let (snapshot, tail) = ([1u32, 2, 3, 4], [5u32, 6]);
        let prompt = [1u32, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(live_start(&snapshot, &tail, &prompt, Some(6)), Some(6));
        assert_eq!(refusal(&snapshot, &tail, &prompt, Some(6)), "nothing");
        // nothing new to read
        assert_eq!(live_start(&snapshot, &tail, &prompt[..6], Some(6)), None);
        assert!(refusal(&snapshot, &tail, &prompt[..6], Some(6)).contains("adds nothing"));
        assert_eq!(live_start(&snapshot, &tail, &prompt[..5], Some(6)), None);
        // the model is elsewhere, or mid-call
        assert_eq!(live_start(&snapshot, &tail, &prompt, Some(5)), None);
        assert_eq!(live_start(&snapshot, &tail, &prompt, Some(7)), None);
        assert!(refusal(&snapshot, &tail, &prompt, Some(7)).contains("is at 7"));
        assert_eq!(live_start(&snapshot, &tail, &prompt, None), None);
        assert!(refusal(&snapshot, &tail, &prompt, None).contains("uncommitted"));
        // a token of the snapshot or of the tail differs
        for index in 0..6 {
            let mut other = prompt;
            other[index] = 99;
            assert_eq!(
                live_start(&snapshot, &tail, &other, Some(6)),
                None,
                "{index}"
            );
            assert!(
                refusal(&snapshot, &tail, &other, Some(6)).contains(&format!("token {index} of 6")),
                "{index}"
            );
        }
        // an empty tail is the snapshot itself
        assert_eq!(live_start(&snapshot, &[], &prompt, Some(4)), Some(4));
        assert_eq!(live_start(&[], &[], &prompt, Some(0)), Some(0));
    }

    #[test]
    fn a_tool_round_only_reuses_rows_the_decoder_committed() {
        let initial_prompt = [11, 22, 33];
        // The decoder stopped after emitting token 55 but before advancing its state with it.
        let emitted = [44, 55];
        let position = 4;
        let committed = committed_stream(initial_prompt.len(), &emitted, None, position).unwrap();
        let held = [initial_prompt.as_slice(), &committed].concat();
        let with_result = [11, 22, 33, 44, 55, 66, 77];
        let start = live_start(&held, &[], &with_result, Some(position)).unwrap();
        assert_eq!(&with_result[start..], [55, 66, 77]);
        assert_eq!(
            live_start(&held, &[], &with_result, Some(position + 1)),
            None
        );
        // A rewritten tool call cannot start at the same row just because its size matches.
        let rewritten = [11, 22, 33, 99, 55, 66, 77];
        assert_eq!(live_start(&held, &[], &rewritten, Some(position)), None);
    }

    #[test]
    fn the_rest_of_a_rendered_prompt_is_encoded_after_the_held_tokens() {
        // one token per character; `<` stands for a control token
        let encode = |text: &str| Ok::<_, ()>(text.bytes().map(u32::from).collect());
        let control = |token: u32| token == u32::from(b'<');
        let held = [500u32, 501];
        assert_eq!(
            extend_tokens(&held, "prompt answer", "prompt answer<x", encode, control),
            Ok(Some(vec![500, 501, 60, 120]))
        );
        // nothing more, another text, a rest that starts inside a word
        assert_eq!(
            extend_tokens(&held, "prompt answer", "prompt answer", encode, control),
            Ok(None)
        );
        assert_eq!(
            extend_tokens(&held, "prompt answer", "prompt other<x", encode, control),
            Ok(None)
        );
        assert_eq!(
            extend_tokens(&held, "prompt answer", "prompt answers<x", encode, control),
            Ok(None)
        );
        assert_eq!(
            extend_tokens(&held, "long prompt", "long", encode, control),
            Ok(None)
        );
        // an encoding failure is the caller's error; an empty encoding continues nothing
        assert_eq!(
            extend_tokens(&held, "a", "a<", |_| Err::<Vec<u32>, _>("failed"), control),
            Err("failed")
        );
        assert_eq!(
            extend_tokens(&held, "a", "a<", |_| Ok::<_, ()>(Vec::new()), control),
            Ok(None)
        );
    }
}
