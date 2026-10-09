"""Tiny web demo (standard library only).

GET /            the page
GET /api/ask?q=  JSON: answer, sources, answerer used

If Ollama is reachable (OLLAMA_HOST, RAG_MODEL), a local model writes the
answer. Otherwise it falls back to the extractive answerer, which is what a
free cloud host without a GPU runs.
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .corpus import load_chunks
from .generate import ExtractiveAnswerer, LLMAnswerer
from .llm import ollama_available, ollama_chat
from .retrieve import BM25

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio Q&amp;A</title><style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e3e3de;--accent:#2457c5}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecece8;--muted:#9a9a94;--card:#1f1f1d;--line:#33332f;--accent:#7ea2ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:720px;margin:0 auto;padding:32px 16px}
h1{font-size:1.5rem;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 24px}
form{display:flex;gap:8px}input{flex:1;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);font:inherit}
button{padding:10px 16px;border:0;border-radius:8px;background:var(--accent);color:#fff;font:inherit;cursor:pointer}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-top:20px}
.src{color:var(--muted);font-size:.9rem;margin-top:12px}.ex{color:var(--accent);cursor:pointer;text-decoration:underline}
</style></head><body><main>
<h1>Ask about my AI projects</h1>
<p class="sub">Retrieval-augmented answers over the READMEs of five portfolio projects, with sources.</p>
<form id="f"><input id="q" placeholder="e.g. Which models vote in label-auditor's ensemble?" autocomplete="off"><button>Ask</button></form>
<p class="sub" style="margin-top:12px">Try: <span class="ex">Which models vote in label-auditor's ensemble?</span> ·
<span class="ex">What license are the Amharic benchmark items released under?</span></p>
<div id="out"></div></main><script>
const f=document.getElementById('f'),q=document.getElementById('q'),out=document.getElementById('out');
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function ask(){if(!q.value.trim())return;out.innerHTML='<div class="card">Thinking...</div>';
const r=await fetch('/api/ask?q='+encodeURIComponent(q.value));const d=await r.json();
out.innerHTML='<div class="card">'+esc(d.answer)+'<div class="src">'+d.sources.map((s,i)=>'['+(i+1)+'] '+esc(s.doc+' / '+s.section)).join('<br>')+
'<br>Answered by: '+esc(d.answerer)+'</div></div>';}
f.onsubmit=e=>{e.preventDefault();ask()};
document.querySelectorAll('.ex').forEach(x=>x.onclick=()=>{q.value=x.textContent;ask()});
</script></body></html>"""


class App:
    def __init__(self, k: int = 3):
        self.k = k
        self.retriever = BM25(load_chunks(size=600))
        model = os.environ.get("RAG_MODEL", "llama3.2:3b")
        if os.environ.get("RAG_ANSWERER", "auto") != "extractive" and ollama_available():
            self.answerer = LLMAnswerer(lambda m: ollama_chat(model, m))
            self.name = f"ollama:{model}"
        else:
            self.answerer, self.name = ExtractiveAnswerer(), "extractive (no LLM)"

    def ask(self, q: str) -> dict:
        chunks = [c for c, _ in self.retriever.search(q, self.k)]
        ans = self.answerer.answer(q, chunks)
        return {"answer": ans.text, "answerer": self.name,
                "sources": [{"doc": c.doc, "section": c.section, "text": c.text} for c in chunks]}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/ask":
                q = parse_qs(url.query).get("q", [""])[0][:500]
                body = app.ask(q) if q.strip() else {"error": "empty question"}
                self._send(200 if q.strip() else 400, json.dumps(body).encode(), "application/json")
            elif url.path == "/healthz":
                self._send(200, b"ok", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *args):
            pass
    return Handler


def serve(host: str = "0.0.0.0", port: int | None = None):
    port = port or int(os.environ.get("PORT", 8000))
    app = App()
    print(f"Serving on http://localhost:{port} (answerer: {app.name})", flush=True)
    ThreadingHTTPServer((host, port), make_handler(app)).serve_forever()
