"""Dev chat UI for manual Q&A testing — NOT part of the library deliverable.

A ~zero-dependency local web chat over the PUBLIC PaperLibrary API only
(papers / ask / get_figure) — nothing in src/ is touched, and the three
endpoints mirror the future wire mapping (api_contract §10). Markdown is
rendered with marked.js, formulas with KaTeX (both from CDN; degrade to
plain text offline), figure images are served by id via /api/figure/<id>.
Citations are shown as raw [n] markers for now (per Jeff, 2026-07-28).

    uv run python scripts/chat_ui.py                          # default data dir
    uv run python scripts/chat_ui.py --data-dir smoke_output/library_data
    uv run python scripts/chat_ui.py --port 8377 --no-open

Providers/models come from .env / PAPER_RAG_* exactly like the library.
Each question is an independent ask() — multi-turn memory is stretch (D9).
Binds to 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from paper_rag import Anchor, Config, PaperLibrary, PaperRagError, Scope  # noqa: E402

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PaperRAG dev chat</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<style>
  :root { --bg:#f2f0eb; --panel:#fbfaf7; --line:#e2dcd1; --me:#8a6420; --ink:#2b2823;
          --mut:#867d6e; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.6 -apple-system,"PingFang SC","Helvetica Neue",sans-serif;
         background:var(--bg); color:var(--ink); }
  header { padding:10px 16px; background:var(--panel); border-bottom:1px solid var(--line);
           display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  header label { font-size:13px; color:var(--mut); }
  select,input,textarea { border:1px solid var(--line); border-radius:8px; padding:6px 8px;
                          font:inherit; background:#fff; }
  #page { width:70px; }
  #selection { width:100%; height:52px; resize:vertical; }
  #anchorbox { width:100%; display:none; }
  #chat { max-width:880px; margin:0 auto; padding:16px 16px 120px; }
  .msg { margin:10px 0; display:flex; }
  .msg.user { justify-content:flex-end; }
  .bubble { max-width:85%; padding:10px 14px; border-radius:14px; background:var(--panel);
            border:1px solid var(--line); overflow-wrap:break-word; }
  .user .bubble { background:var(--me); color:#fff; border:none; white-space:pre-wrap; }
  .bubble img.figure { max-width:100%; border:1px solid var(--line); border-radius:8px;
                       margin-top:8px; display:block; }
  .bubble figcaption { font-size:12px; color:var(--mut); margin-top:2px; }
  .meta { font-size:12px; color:var(--mut); margin-top:8px; border-top:1px dashed var(--line);
          padding-top:6px; }
  .badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px;
           background:#f3ead6; color:#7a5716; margin-right:6px; }
  .badge.abstain { background:#f9ede8; color:#a63b22; }
  .err .bubble { background:#f9ece6; border-color:#eccdb9; color:#94422a; }
  form#ask { position:fixed; bottom:0; left:0; right:0; background:var(--panel);
             border-top:1px solid var(--line); padding:12px 16px; }
  .row { max-width:880px; margin:0 auto; display:flex; gap:8px; }
  #q { flex:1; height:44px; resize:none; }
  button { border:none; border-radius:8px; padding:0 20px; background:var(--me);
           color:#fff; font:inherit; cursor:pointer; }
  button:disabled { opacity:.5; }
  .bubble p:first-child { margin-top:0; } .bubble p:last-child { margin-bottom:0; }
  .bubble a { color:var(--me); }
  .spin { color:var(--mut); font-style:italic; }
</style>
</head>
<body>
<header>
  <label>paper <select id="paper"><option value="">(none — whole library)</option></select></label>
  <label>scope <select id="scope">
    <option value="auto">auto</option><option value="paper">paper</option>
    <option value="library">library</option></select></label>
  <label><input type="checkbox" id="anchor_toggle"> anchor (page + selected text)</label>
  <div id="anchorbox">
    <label>page <input id="page" type="number" min="1"></label>
    <textarea id="selection" placeholder="Paste the selected passage (simulates selecting text in the reader)"></textarea>
  </div>
</header>
<div id="chat"></div>
<form id="ask">
  <div class="row">
    <textarea id="q" placeholder="Ask a question — Enter to send, Shift+Enter for a new line"></textarea>
    <button id="send" type="submit">Send</button>
  </div>
</form>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<script>
const chat = document.getElementById("chat");
const q = document.getElementById("q");
const form = document.getElementById("ask");
const send = document.getElementById("send");
document.getElementById("anchor_toggle").onchange = e =>
  document.getElementById("anchorbox").style.display = e.target.checked ? "block" : "none";

fetch("/api/papers").then(r => r.json()).then(papers => {
  const sel = document.getElementById("paper");
  for (const p of papers) {
    const o = document.createElement("option");
    o.value = p.paper_id;
    o.textContent = `${p.title} (${p.n_pages}p)`;
    sel.appendChild(o);
  }
  if (papers.length === 1) sel.value = papers[0].paper_id;
});

function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function bubble(cls, inner) {
  const m = document.createElement("div");
  m.className = "msg " + cls;
  const b = document.createElement("div");
  b.className = "bubble";
  if (typeof inner === "string") b.textContent = inner; else b.appendChild(inner);
  m.appendChild(b);
  chat.appendChild(m);
  window.scrollTo(0, document.body.scrollHeight);
  return b;
}

function renderAnswer(ans) {
  const box = document.createElement("div");
  const md = document.createElement("div");
  // escape first (no raw HTML from the model), then markdown, then KaTeX
  md.innerHTML = (typeof marked !== "undefined")
    ? marked.parse(esc(ans.text)) : esc(ans.text).replace(/\\n/g, "<br>");
  box.appendChild(md);
  for (const f of ans.figures) {
    const fig = document.createElement("figure");
    fig.style.margin = "8px 0 0";
    const img = document.createElement("img");
    img.className = "figure";
    img.src = "/api/figure/" + encodeURIComponent(f.figure_id);
    img.alt = f.label;
    const cap = document.createElement("figcaption");
    cap.textContent = `${f.label} · p.${f.page}`;
    fig.appendChild(img); fig.appendChild(cap);
    box.appendChild(fig);
  }
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.innerHTML = `<span class="badge">${esc(ans.scope_used)}</span>`
    + (ans.abstained ? `<span class="badge abstain">abstained</span>` : "")
    + `<span class="badge">citations: ${ans.citations.length}</span>`;
  box.appendChild(meta);
  if (typeof renderMathInElement !== "undefined")
    renderMathInElement(box, {delimiters: [
      {left: "$$", right: "$$", display: true},
      {left: "\\\\[", right: "\\\\]", display: true},
      {left: "\\\\(", right: "\\\\)", display: false},
      {left: "$", right: "$", display: false}],
      throwOnError: false});
  return box;
}

form.onsubmit = async (e) => {
  e.preventDefault();
  const question = q.value.trim();
  if (!question) return;
  q.value = "";
  bubble("user", question);
  const wait = bubble("assistant", "");
  wait.innerHTML = '<span class="spin">Retrieving + generating… (live API, ~10–20 s)</span>';
  send.disabled = true;
  const payload = {
    question,
    scope: document.getElementById("scope").value,
    paper_id: document.getElementById("paper").value || null,
  };
  if (document.getElementById("anchor_toggle").checked) {
    const page = document.getElementById("page").value;
    payload.page = page ? parseInt(page, 10) : null;
    payload.selection = document.getElementById("selection").value || null;
  }
  try {
    const r = await fetch("/api/ask", {method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    const data = await r.json();
    wait.parentElement.remove();
    if (data.ok) bubble("assistant", renderAnswer(data.answer));
    else bubble("assistant err", `[${data.code}] ${data.message}`);
  } catch (err) {
    wait.parentElement.remove();
    bubble("assistant err", "Request failed: " + err);
  } finally {
    send.disabled = false;
    q.focus();
  }
};
q.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
</script>
</body>
</html>
"""


def make_handler(lib: PaperLibrary):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):   # keep the terminal quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict) -> None:
            self._send(code, json.dumps(payload, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")

        def do_GET(self):
            if self.path == "/":
                self._send(200, HTML.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/papers":
                papers = [asdict(p) for p in lib.papers()]
                self._send(200, json.dumps(papers, ensure_ascii=False).encode(),
                           "application/json; charset=utf-8")
            elif self.path.startswith("/api/figure/"):
                fid = unquote(self.path[len("/api/figure/"):])
                try:
                    asset = lib.get_figure(fid)
                    self._send(200, asset.data, asset.mime)
                except PaperRagError as exc:
                    self._json(404, {"ok": False, "code": exc.code,
                                     "message": exc.message})
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/ask":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                question = (payload.get("question") or "").strip()
                if not question:
                    self._json(400, {"ok": False, "code": "BAD_REQUEST",
                                     "message": "question is required"})
                    return
                anchor = None
                if payload.get("paper_id"):
                    anchor = Anchor(paper_id=payload["paper_id"],
                                    page=payload.get("page"),
                                    selection=payload.get("selection"))
                answer = lib.ask(question, anchor=anchor,
                                 scope=Scope(payload.get("scope", "auto")))
                self._json(200, {"ok": True, "answer": asdict(answer)})
            except PaperRagError as exc:
                self._json(200, {"ok": False, "code": exc.code,
                                 "message": exc.message, "details": exc.details})
            except Exception as exc:   # dev tool: surface, don't crash the server
                self._json(500, {"ok": False, "code": "INTERNAL",
                                 "message": str(exc)})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "smoke_output" / "library_data"),
                        help="PaperLibrary data dir (default: smoke_output/library_data)")
    parser.add_argument("--port", type=int, default=8377)
    parser.add_argument("--no-open", action="store_true",
                        help="don't auto-open the browser")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    load_dotenv()
    lib = PaperLibrary(data_dir=args.data_dir, config=Config.from_env())
    papers = lib.papers()
    if not papers:
        print(f"data dir {args.data_dir} has no ingested papers — ingest first:")
        print(f"  uv run python -m paper_rag ingest <pdf> --data-dir {args.data_dir}")
        return 1
    print(f"papers: {', '.join(p.title[:50] for p in papers)}")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(lib))
    url = f"http://127.0.0.1:{args.port}"
    print(f"PaperRAG dev chat: {url}  (Ctrl+C to stop)")
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
