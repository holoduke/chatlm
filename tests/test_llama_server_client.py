"""llama_server_client — pure-function translation tests.

The OpenAI ↔ Ollama wire-format translation is the most fragile part of
the multi-backend setup, so it gets the densest coverage. Real HTTP /
subprocess interactions are out of scope (covered by the live-server
validation script in tests/validate_live_server.sh)."""
from __future__ import annotations

import json

import llama_server_client as L


# ── name parsing ───────────────────────────────────────────────────────


def test_is_llama_name_prefix_check():
    assert L.is_llama_name("llama:foo")
    assert L.is_llama_name("llama:hf.co/owner/repo:tag")
    assert not L.is_llama_name("foo")
    assert not L.is_llama_name("mlx:foo")
    assert not L.is_llama_name("")


def test_strip_prefix():
    assert L.strip_prefix("llama:foo") == "foo"
    assert L.strip_prefix("foo") == "foo"  # idempotent


def test_split_name_huggingface_registry():
    # Registry hostname has a dot → kept as registry.
    reg, repo, tag = L._split_name("hf.co/mradermacher/Huihui-gemma-4:Q4_K_M")
    assert reg == "hf.co"
    assert repo == "mradermacher/Huihui-gemma-4"
    assert tag == "Q4_K_M"


def test_split_name_default_ollama_registry_with_owner():
    reg, repo, tag = L._split_name("huihui_ai/qwen3-abliterated:8b")
    assert reg == "registry.ollama.ai"
    assert repo == "huihui_ai/qwen3-abliterated"
    assert tag == "8b"


def test_split_name_library_namespace():
    """Bare model:tag with no slash → 'library/<name>' under the
    default registry. Matches Ollama's on-disk manifest layout."""
    reg, repo, tag = L._split_name("gemma4:e2b")
    assert reg == "registry.ollama.ai"
    assert repo == "library/gemma4"
    assert tag == "e2b"


def test_split_name_default_tag_is_latest():
    _, _, tag = L._split_name("gemma4")
    assert tag == "latest"


# ── tool-call normalisation ───────────────────────────────────────────


def test_normalise_tool_call_adds_missing_type():
    out = L._normalise_tool_call({
        "id": "call_1",
        "function": {"name": "f", "arguments": {"a": 1}},
    })
    assert out["type"] == "function"


def test_normalise_tool_call_preserves_existing_type():
    out = L._normalise_tool_call({
        "id": "call_1",
        "type": "custom",
        "function": {"name": "f", "arguments": "{}"},
    })
    assert out["type"] == "custom"


def test_normalise_tool_call_stringifies_dict_arguments():
    """OpenAI's spec requires `arguments` to be a JSON-encoded string,
    not a dict. llama-server returns 500 'Missing tool call type' or
    similar parsing errors otherwise."""
    out = L._normalise_tool_call({
        "id": "call_1",
        "function": {"name": "f", "arguments": {"market": "NL"}},
    })
    assert isinstance(out["function"]["arguments"], str)
    assert json.loads(out["function"]["arguments"]) == {"market": "NL"}


def test_normalise_tool_call_leaves_string_arguments_alone():
    out = L._normalise_tool_call({
        "id": "call_1",
        "function": {"name": "f", "arguments": '{"a": 1}'},
    })
    assert out["function"]["arguments"] == '{"a": 1}'


def test_normalise_tool_call_handles_non_dict_input_gracefully():
    # Defensive — bogus list returned unchanged so we never raise.
    assert L._normalise_tool_call(["not a dict"]) == ["not a dict"]


# ── message normalisation ─────────────────────────────────────────────


def test_normalise_messages_passthrough_when_no_attachments():
    msgs = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]
    assert L._normalise_messages(msgs) == msgs


def test_normalise_messages_translates_images_to_content_parts():
    out = L._normalise_messages([
        {"role": "user", "content": "describe", "images": ["AAA"]},
    ])
    assert "images" not in out[0]
    parts = out[0]["content"]
    assert isinstance(parts, list)
    assert parts[0] == {"type": "text", "text": "describe"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_normalise_messages_translates_audios_to_input_audio_parts():
    out = L._normalise_messages([
        {"role": "user", "content": "transcribe", "audios": ["WAVB64"]},
    ])
    assert "audios" not in out[0]
    parts = out[0]["content"]
    assert parts[1] == {
        "type": "input_audio",
        "input_audio": {"data": "WAVB64", "format": "wav"},
    }


def test_normalise_messages_handles_image_and_audio_together():
    out = L._normalise_messages([
        {"role": "user", "content": "what is this", "images": ["IMG"], "audios": ["AUD"]},
    ])
    parts = out[0]["content"]
    types = [p["type"] for p in parts]
    assert types == ["text", "image_url", "input_audio"]


def test_normalise_messages_passes_through_data_url_images_unchanged():
    """Caller may already have wrapped a data URL — don't double-wrap."""
    out = L._normalise_messages([
        {"role": "user", "content": "?", "images": ["data:image/jpeg;base64,XXX"]},
    ])
    assert out[0]["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,XXX"


def test_normalise_messages_normalises_history_tool_calls():
    """Regression guard for the 'Missing tool call type' bug:
    forwarding chat history with a previous assistant tool_call must
    add `type: function` so llama-server doesn't 500."""
    out = L._normalise_messages([
        {"role": "user", "content": "find Schiphol"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "f", "arguments": {"q": "Schiphol"}}},
            ],
        },
    ])
    tc = out[1]["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["arguments"] == '{"q": "Schiphol"}'


# ── response translation ──────────────────────────────────────────────


def test_openai_to_ollama_chat_basic_content():
    payload = {
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        "timings": {"prompt_ms": 10, "predicted_ms": 20},
    }
    out = L._openai_to_ollama_chat("llama:foo", payload)
    assert out["model"] == "llama:foo"
    assert out["message"]["content"] == "hi"
    assert out["message"]["thinking"] is None
    assert out["done"] is True
    assert out["done_reason"] == "stop"
    assert out["prompt_eval_count"] == 5
    assert out["eval_count"] == 1
    # nanoseconds (ms * 1e6)
    assert out["prompt_eval_duration"] == 10_000_000
    assert out["eval_duration"] == 20_000_000


def test_openai_to_ollama_chat_surfaces_reasoning_content_as_thinking():
    """Regression guard: dropping reasoning_content was the cause of
    'empty response despite tokens generated' bug."""
    payload = {
        "choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Pure", "reasoning_content": "thought process..."}
        }],
        "usage": {"prompt_tokens": 43, "completion_tokens": 80},
        "timings": {},
    }
    out = L._openai_to_ollama_chat("llama:foo", payload)
    assert out["message"]["content"] == "Pure"
    assert out["message"]["thinking"] == "thought process..."


def test_openai_to_ollama_chat_falls_back_to_thinking_when_content_empty():
    """When the model runs out of tokens mid-thought, surface the
    reasoning so the user sees something instead of an empty bubble."""
    payload = {
        "choices": [{
            "finish_reason": "length",
            "message": {"role": "assistant", "content": "", "reasoning_content": "Step 1: ..."}
        }],
        "usage": {},
        "timings": {},
    }
    out = L._openai_to_ollama_chat("llama:foo", payload)
    assert out["message"]["content"] == "Step 1: ..."
    assert out["message"]["thinking"] == "Step 1: ..."
    assert out["done_reason"] == "length"


def test_openai_to_ollama_chat_handles_missing_choice():
    out = L._openai_to_ollama_chat("llama:foo", {})
    assert out["message"]["content"] == ""
    assert out["done"] is True


def test_openai_to_ollama_chat_passes_through_tool_calls():
    """When the model issues tool calls, the response translation must
    forward them verbatim — chatlm's chat handler reads
    `message.tool_calls` to decide whether to dispatch the call."""
    payload = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "echo", "arguments": '{"x":1}'}},
                ],
            },
        }],
        "usage": {},
        "timings": {},
    }
    out = L._openai_to_ollama_chat("llama:foo", payload)
    assert out["message"]["tool_calls"] == [
        {"id": "c1", "type": "function",
         "function": {"name": "echo", "arguments": '{"x":1}'}},
    ]
    assert out["done_reason"] == "tool_calls"


# ── streaming translation ─────────────────────────────────────────────


def _wrap_sse(obj: dict) -> str:
    return "data: " + json.dumps(obj)


def test_stream_chunk_content_delta():
    line = _wrap_sse({"choices": [{"delta": {"content": "hi"}}]})
    out = L._openai_stream_to_ollama_chunk("llama:foo", line)
    evt = json.loads(out)
    assert evt["message"]["content"] == "hi"
    assert evt["done"] is False
    assert "thinking" not in evt["message"]


def test_stream_chunk_thinking_delta():
    line = _wrap_sse({"choices": [{"delta": {"reasoning_content": "thinking..."}}]})
    out = L._openai_stream_to_ollama_chunk("llama:foo", line)
    evt = json.loads(out)
    assert evt["message"]["thinking"] == "thinking..."
    assert evt["done"] is False


def test_stream_chunk_finish_reason_emits_done_event():
    line = _wrap_sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    out = L._openai_stream_to_ollama_chunk("llama:foo", line)
    evt = json.loads(out)
    assert evt["done"] is True
    assert evt["done_reason"] == "stop"


def test_stream_chunk_done_marker_returns_none():
    assert L._openai_stream_to_ollama_chunk("llama:foo", "data: [DONE]") is None


def test_stream_chunk_keepalive_and_garbage_returns_none():
    assert L._openai_stream_to_ollama_chunk("llama:foo", "") is None
    assert L._openai_stream_to_ollama_chunk("llama:foo", "event: ping") is None
    assert L._openai_stream_to_ollama_chunk("llama:foo", "data: not-json") is None


def test_stream_chunk_empty_delta_returns_none():
    """A delta with neither content nor reasoning_content is a noop —
    don't forward an empty content frame to the UI."""
    line = _wrap_sse({"choices": [{"delta": {}}]})
    assert L._openai_stream_to_ollama_chunk("llama:foo", line) is None


# ── payload builder ───────────────────────────────────────────────────


def test_build_payload_omits_unset_optional_fields():
    p = L._build_payload(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7, max_tokens=None, tools=None, top_p=None, top_k=None, stream=False,
    )
    assert p["temperature"] == 0.7
    assert p["stream"] is False
    assert "max_tokens" not in p
    assert "tools" not in p
    assert "top_p" not in p
    assert "top_k" not in p


def test_build_payload_includes_optional_fields_when_set():
    p = L._build_payload(
        messages=[{"role": "user", "content": "hi"}],
        temperature=1.0, max_tokens=128, tools=[{"x": 1}],
        top_p=0.95, top_k=64, stream=True,
    )
    assert p["temperature"] == 1.0
    assert p["max_tokens"] == 128
    assert p["tools"] == [{"x": 1}]
    assert p["top_p"] == 0.95
    assert p["top_k"] == 64
    assert p["stream"] is True
