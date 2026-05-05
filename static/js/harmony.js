/* harmony.js
 * Stream splitter for Harmony-style channel headers.
 *
 * Some abliterated GGUFs (TrevorJS gemma-4 and similar) were trained with
 * gpt-oss-style channel markers but had their chat-template transitions
 * broken by abliteration, so they emit a malformed prefix like:
 *
 *     <|channel>thought
 *     <channel|>The actual answer goes here.
 *
 * Ollama doesn't recognise this as thinking-channel metadata, so the
 * markers leak into `message.content`. This module splits an incoming
 * content delta into:
 *   - `think`: bytes that belong inside `<|channel>...<channel|>` (i.e.
 *     between the OPEN and CLOSE markers, excluding the markers themselves).
 *   - `content`: bytes that come after the CLOSE marker (i.e. the visible
 *     answer the model intended to send to the user).
 *
 * Once the CLOSE marker has been seen, the stripper enters passthrough
 * mode and forwards every subsequent delta as `content`. If no OPEN
 * marker shows up at the start of the stream, passthrough is enabled
 * immediately. If OPEN is seen but CLOSE never arrives within
 * MAX_PREFIX_LEN bytes, the held buffer is flushed as `content` (we
 * assume it was legitimate output that just happened to start with `<`).
 *
 * Safe to apply universally — the splitter is a no-op when no Harmony
 * markers are present.
 */

const OPEN = "<|channel>";
const CLOSE = "<channel|>";
const MAX_PREFIX_LEN = 256;

export class ChannelStripper {
  constructor() {
    this._buf = "";
    // null = still deciding; "passthrough" = no OPEN, route to content;
    // "think" = OPEN seen, CLOSE not yet, route to think;
    // "done" = CLOSE consumed, route to content.
    this._phase = null;
  }

  /** Returns `{ think, content }` for this delta. Either may be empty. */
  feed(delta) {
    if (!delta) return { think: "", content: "" };
    if (this._phase === "passthrough" || this._phase === "done") {
      return { think: "", content: delta };
    }
    if (this._phase === "think") {
      this._buf += delta;
      const idx = this._buf.indexOf(CLOSE);
      if (idx >= 0) {
        const inside = this._buf.slice(0, idx);
        let tail = this._buf.slice(idx + CLOSE.length);
        if (tail.startsWith("\n")) tail = tail.slice(1);
        this._buf = "";
        this._phase = "done";
        return { think: inside, content: tail };
      }
      // Don't hold think content hostage waiting for CLOSE — flush what
      // we have so the REASONING panel updates live.
      const inside = this._buf;
      this._buf = "";
      // Re-enter `think` mode but with empty buffer; next deltas continue
      // looking for CLOSE.
      return { think: inside, content: "" };
    }
    // phase === null: deciding whether the stream starts with OPEN.
    this._buf += delta;
    const buf = this._buf;
    if (buf.startsWith(OPEN)) {
      // Move into think mode. Strip the OPEN marker and the channel name
      // up to the newline (e.g. "thought\n") — that's metadata, not
      // useful content for the REASONING panel.
      let after = buf.slice(OPEN.length);
      const nl = after.indexOf("\n");
      if (nl >= 0) {
        // Header consumed up to and including the \n. The remainder is
        // the start of the think content (or may already include CLOSE).
        const remainder = after.slice(nl + 1);
        this._buf = "";
        this._phase = "think";
        return this.feed(remainder);
      }
      // Header line not yet complete (no \n). Keep buffering.
      if (buf.length >= MAX_PREFIX_LEN) {
        this._phase = "passthrough";
        this._buf = "";
        return { think: "", content: buf };
      }
      return { think: "", content: "" };
    }
    if (OPEN.startsWith(buf)) {
      // Could still grow into OPEN — keep buffering.
      return { think: "", content: "" };
    }
    // Doesn't and won't start with OPEN — passthrough from here on.
    this._phase = "passthrough";
    this._buf = "";
    return { think: "", content: buf };
  }

  /** Flush any held buffer at end of stream as `content`. */
  finalise() {
    if (this._phase === "done" || this._phase === "passthrough") {
      return { think: "", content: "" };
    }
    const out = this._buf;
    this._buf = "";
    if (this._phase === "think") {
      this._phase = "done";
      return { think: out, content: "" };
    }
    this._phase = "passthrough";
    return { think: "", content: out };
  }
}
