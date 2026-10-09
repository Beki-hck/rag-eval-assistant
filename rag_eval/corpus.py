"""Load markdown documents and split them into chunks.

A chunk is the unit the retriever searches. Each one keeps the document
name and the heading it came from, so answers can cite their source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    section: str
    text: str


def clean_markdown(md: str) -> str:
    """Drop badge and image lines and HTML comments; keep the readable text."""
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    lines = [l for l in md.splitlines() if not l.strip().startswith("![")]
    return "\n".join(lines)


def split_sections(md: str) -> list[tuple[str, str]]:
    """Split on markdown headings. Returns (heading, body) pairs."""
    sections, heading, body = [], "Introduction", []
    in_code = False
    for line in md.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            if "".join(body).strip():
                sections.append((heading, "\n".join(body).strip()))
            heading, body = m.group(1).strip(), []
        else:
            body.append(line)
    if "".join(body).strip():
        sections.append((heading, "\n".join(body).strip()))
    return sections


def chunk_text(text: str, size: int, overlap: int = 0) -> list[str]:
    """Pack paragraphs into pieces of about `size` characters.

    Paragraphs are kept whole when they fit. A paragraph longer than `size`
    is cut on line or word boundaries. `overlap` repeats the end of the
    previous piece at the start of the next one.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for p in paras:
        while len(p) > size:
            cut = p.rfind("\n", 0, size)
            if cut < size // 2:
                cut = p.rfind(" ", 0, size)
            if cut <= 0:
                cut = size
            pieces.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            pieces.append(p)

    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + len(piece) + 2 > size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1:]
            cur = (tail + "\n\n" + piece).strip() if tail else piece
        else:
            cur = (cur + "\n\n" + piece).strip()
    if cur:
        chunks.append(cur)
    return chunks


def load_chunks(corpus_dir: Path | str = DEFAULT_CORPUS, size: int = 600,
                overlap: int = 0) -> list[Chunk]:
    out: list[Chunk] = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        doc = path.stem
        md = clean_markdown(path.read_text(encoding="utf-8"))
        for heading, body in split_sections(md):
            for i, piece in enumerate(chunk_text(body, size, overlap)):
                out.append(Chunk(f"{doc}#{heading}#{i}", doc, heading, piece))
    return out
