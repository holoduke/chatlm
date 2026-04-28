#!/usr/bin/env bash
# End-to-end smoke check against a running chatlm server.
#
# Run from the project root with the server already up on :8000:
#   .venv/bin/uvicorn main:app --port 8000 --host 127.0.0.1 &
#   bash tests/validate_live_server.sh
#
# Exits 0 if every probe behaved correctly, non-zero on first failure.
# Each probe prints a one-line PASS/FAIL with what was checked and what
# the server actually returned (truncated). Designed to be readable as
# its own audit trail when CI grep-searches for FAIL.
set -uo pipefail

BASE="${CHATLM_BASE:-http://127.0.0.1:8000}"
PASS=0
FAIL=0

ok()   { printf '  \033[32mPASS\033[0m %s\n'   "$*"; PASS=$((PASS+1)); }
nope() { printf '  \033[31mFAIL\033[0m %s\n'   "$*"; FAIL=$((FAIL+1)); }
header() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

# JSON helper. Picks .key from stdin; returns empty string on failure
# so an absent key is testable with `[ -z ... ]`.
jq_path() {
  python3 -c "import sys, json; d=json.load(sys.stdin)
keys = sys.argv[1].split('.')
for k in keys:
  if isinstance(d, list): d = d[int(k)]
  else: d = d.get(k) if d else None
  if d is None: break
print('' if d is None else (d if isinstance(d, str) else json.dumps(d)))" "$1"
}

# ── server reachable ────────────────────────────────────────────────
header "server reachable"
if curl -sf "$BASE/models" -o /dev/null; then
  ok "/models reachable at $BASE"
else
  nope "/models unreachable at $BASE — start uvicorn first"
  exit 1
fi

# ── /models surfaces all three backends ────────────────────────────
header "model catalogue"
MODELS_JSON=$(curl -s "$BASE/models")
TXT2IMG_COUNT=$(echo "$MODELS_JSON" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['txt2img']['presets']))")
if [ "$TXT2IMG_COUNT" -ge 8 ]; then
  ok "txt2img presets >= 8 (got $TXT2IMG_COUNT)"
else
  nope "txt2img presets <8 (got $TXT2IMG_COUNT) — splits regression?"
fi
EMMA_COUNT=$(echo "$MODELS_JSON" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['emma']['available']))")
if [ "$EMMA_COUNT" -ge 10 ]; then
  ok "chat models available >= 10 (got $EMMA_COUNT)"
else
  nope "chat models <10 (got $EMMA_COUNT)"
fi

# ── chat: ollama-native model still routes to ollama ──────────────
header "ollama-native chat (qwen3)"
RESP=$(curl -s -X POST "$BASE/chat" -H 'Content-Type: application/json' --max-time 120 \
  -d '{"messages":[{"role":"user","content":"reply only with: ok"}],"model":"huihui_ai/qwen3-abliterated:8b"}')
CONTENT=$(echo "$RESP" | jq_path "message.content")
RETURNED_MODEL=$(echo "$RESP" | jq_path "model")
# jq_path strips quotes for string values (returns bare 'huihui_ai/...').
# Plain shell equality is fine; no fallback would prefix with `llama:`.
if [ "$RETURNED_MODEL" = "huihui_ai/qwen3-abliterated:8b" ]; then
  ok "qwen3 model preserved (no fallback): $RETURNED_MODEL"
else
  nope "qwen3 unexpectedly rerouted: $RETURNED_MODEL"
fi
[ -n "$CONTENT" ] && ok "qwen3 returned non-empty content: $CONTENT" \
                 || nope "qwen3 returned empty content"

# ── chat: HauhauCS gemma4 → auto-fallback to llama-server ─────────
header "auto-fallback to llama-server (HauhauCS gemma4)"
HAUHAU_E4B="hf.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive:Q4_K_M"
RESP=$(curl -s -X POST "$BASE/chat" -H 'Content-Type: application/json' --max-time 180 \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"reply only with: ok\"}],\"model\":\"$HAUHAU_E4B\"}")
RETURNED_MODEL=$(echo "$RESP" | jq_path "model")
CONTENT=$(echo "$RESP" | jq_path "message.content")
case "$RETURNED_MODEL" in
  llama:*) ok "HauhauCS rerouted to llama-server: $RETURNED_MODEL" ;;
  *)       nope "HauhauCS not rerouted (expected llama: prefix): $RETURNED_MODEL" ;;
esac
[ -n "$CONTENT" ] && ok "HauhauCS returned content: $CONTENT" \
                 || nope "HauhauCS returned empty content"

# ── multimodal: image input ───────────────────────────────────────
header "vision (red 64x64 PNG → 'Red')"
B64=$(.venv/bin/python -c "from PIL import Image; import io,base64; img=Image.new('RGB',(64,64),(255,0,0)); buf=io.BytesIO(); img.save(buf,'PNG'); print(base64.b64encode(buf.getvalue()).decode())")
RESP=$(curl -s -X POST "$BASE/chat" -H 'Content-Type: application/json' --max-time 240 \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"What single colour fills this image? One word.\",\"images\":[\"$B64\"]}],\"model\":\"$HAUHAU_E4B\",\"temperature\":0.0,\"max_tokens\":300}")
CONTENT=$(echo "$RESP" | jq_path "message.content")
if echo "$CONTENT" | grep -iq "red"; then
  ok "model identified red: $CONTENT"
else
  nope "model failed to identify red — got: $CONTENT"
fi

# ── multimodal: audio input (sine wave) ───────────────────────────
header "audio (synthetic 440Hz sine → tone-related word)"
WAV_B64=$(.venv/bin/python -c "
import wave, struct, math, base64, io
sr=16000; freq=440; secs=0.5
buf=io.BytesIO(); w=wave.open(buf,'wb')
w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
for i in range(int(sr*secs)):
  s=int(0.4*32767*math.sin(2*math.pi*freq*i/sr))
  w.writeframesraw(struct.pack('<h', s))
w.close()
print(base64.b64encode(buf.getvalue()).decode())")
RESP=$(curl -s -X POST "$BASE/chat" -H 'Content-Type: application/json' --max-time 240 \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"What single tone is in this audio? One word.\",\"audios\":[\"$WAV_B64\"]}],\"model\":\"$HAUHAU_E4B\",\"temperature\":0.0,\"max_tokens\":600}")
CONTENT=$(echo "$RESP" | jq_path "message.content")
THINKING=$(echo "$RESP" | jq_path "message.thinking")
if [ -n "$CONTENT" ] && [ "$CONTENT" != '""' ]; then
  ok "audio path produced content: $CONTENT"
else
  nope "audio path produced empty content (thinking truncated to: ${THINKING:0:120})"
fi

# ── tools auto-drop (non-tools-capable model + tools requested) ───
# Policy: when the user picks a non-tools-capable model, chatlm preserves
# the user's choice and silently drops tools for that turn — rather than
# swapping behind the user's back to a tools-capable (often censored)
# fallback. The chat must succeed, the model must be preserved, and a
# warning must land in the server log.
header "tools auto-drop for non-tools-capable model"
RESP=$(curl -s -X POST "$BASE/chat" -H 'Content-Type: application/json' --max-time 240 \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"reply only with: ok\"}],\"model\":\"$HAUHAU_E4B\",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"echo\",\"description\":\"x\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]}")
RETURNED_MODEL=$(echo "$RESP" | jq_path "model")
CONTENT=$(echo "$RESP" | jq_path "message.content")
# After the auto-fallback to llama-server kicks in for HauhauCS, the
# returned model is the `llama:` prefixed name — but the *user's* model
# (sans prefix) was preserved end to end (no swap to gemma4:*).
case "$RETURNED_MODEL" in
  *HauhauCS*) ok "user model preserved (tools dropped silently): $RETURNED_MODEL" ;;
  gemma4:*|library/gemma4*) nope "model unexpectedly swapped to tools-capable fallback: $RETURNED_MODEL" ;;
  "") nope "tools request returned empty (timeout? response: ${RESP:0:200})" ;;
  *) nope "tools request returned unexpected model: $RETURNED_MODEL" ;;
esac
[ -n "$CONTENT" ] && ok "tools-dropped chat returned content: $CONTENT" \
                 || nope "tools-dropped chat returned empty content"

# ── streaming endpoint emits proper NDJSON ───────────────────────
header "streaming NDJSON shape"
LINES=$(curl -sN -X POST "$BASE/chat/stream" -H 'Content-Type: application/json' --max-time 120 \
  -d '{"messages":[{"role":"user","content":"count: 1, 2, 3"}],"model":"huihui_ai/qwen3-abliterated:8b"}' \
  | wc -l)
if [ "$LINES" -ge 3 ]; then
  ok "stream emitted $LINES NDJSON lines"
else
  nope "stream emitted only $LINES lines (expected >= 3)"
fi

# ── summary ──────────────────────────────────────────────────────
echo
echo "========================================"
echo "  PASS: $PASS    FAIL: $FAIL"
echo "========================================"
[ "$FAIL" -eq 0 ]
