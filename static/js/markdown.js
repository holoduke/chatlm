/* markdown.js
 * Thin wrapper around marked for rendering bot responses and thinking
 * panels. Local-single-user so we trust the LLM output for inline content,
 * but we strip page-scoped tags (style/link/script/meta/title/base) so an
 * LLM-emitted HTML page doesn't leak its rules into the host document
 * and break things like the live-camera overlay. */

import { marked } from "./vendor/marked.esm.js";

marked.setOptions({
  gfm: true,       // GitHub-flavoured: tables, strikethrough, task lists
  breaks: true,    // single newline → <br> (matches chat-ish expectations)
  headerIds: false,
  mangle: false,
});

/* Open every rendered link in a new tab. Done as a post-process on the
 * final HTML rather than a marked renderer override (the v12 renderer
 * `this` doesn't expose `.parser` for inline parsing, so subclassing
 * fails on autolinks). Adds target+rel only to anchors that don't
 * already have a target attribute. rel="noopener" prevents the new
 * tab from reaching back via window.opener. */
function externalizeLinks(html) {
  return html.replace(/<a (?![^>]*\btarget=)([^>]*)>/g, '<a target="_blank" rel="noopener noreferrer" $1>');
}

/* Strip tags that are valid inside an HTML document but globally affect
 * the host page when injected via innerHTML. <style> rules apply across
 * the whole document regardless of where the element sits in the tree;
 * <link rel=stylesheet> does the same; <script> would execute (well, it
 * doesn't via innerHTML, but we strip for clarity); <meta>/<title>/<base>
 * mutate document-level state. Inline `style="..."` attributes stay —
 * those are scoped per-element. */
function stripPageScopedTags(html) {
  return html
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, "")
    .replace(/<link\b[^>]*>/gi, "")
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, "")
    .replace(/<meta\b[^>]*>/gi, "")
    .replace(/<title\b[^>]*>[\s\S]*?<\/title>/gi, "")
    .replace(/<base\b[^>]*>/gi, "");
}

/* Wrap any bare http(s) URL in angle brackets so marked's autolink
 * handler picks it up even when it sits awkwardly between newlines or
 * after punctuation that would otherwise cause marked's GFM autolink
 * heuristic to miss it. Skips URLs already inside a markdown link
 * `[text](url)` or already wrapped `<url>`. */
function preAutoLink(text) {
  return text.replace(
    /(^|[\s(])(https?:\/\/[^\s<>)\]]+[^\s<>).,;:!?\]])/g,
    (_, pre, url) => `${pre}<${url}>`,
  );
}

/**
 * Render a markdown string to HTML. Safe to call repeatedly on a
 * growing streamed buffer — marked handles partial/unterminated blocks
 * gracefully (open code fences stay as a pre, unbalanced ** collapses).
 */
export function renderMarkdown(text) {
  if (!text) return "";
  try {
    return externalizeLinks(stripPageScopedTags(marked.parse(preAutoLink(text))));
  } catch (err) {
    // If marked ever chokes on partial input, fall back to the raw text.
    console.warn("[markdown] parse failed, falling back to text", err);
    return text.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  }
}
