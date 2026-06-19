"""Ollama streaming chat client for the coding-task path (ported from the doer).

Streaming keeps the socket fed with tokens so the per-chunk timeout applies to the
gap between chunks, not the whole (possibly long, cold) generation. Emits a
heartbeat so a long local-model run is visibly alive.
"""

import json
import os
import time
import urllib.request
from urllib.parse import urlsplit, urlunsplit

from raster.runlog import log, fmt_secs

OLLAMA_TIMEOUT = int(os.environ.get("RASTER_OLLAMA_TIMEOUT", 1800))   # per-chunk read gap
KEEPALIVE = os.environ.get("RASTER_KEEPALIVE", "30m")                 # keep model warm
HEARTBEAT_SECS = 300


def normalize_host(raw: str) -> str:
    """Coerce a bind-style host (e.g. '0.0.0.0:11434') into a usable client URL."""
    raw = (raw or "").strip().rstrip("/")
    if "://" not in raw:
        raw = "http://" + raw
    parts = urlsplit(raw)
    host = parts.hostname or "127.0.0.1"
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    port = parts.port or 11434
    return urlunsplit((parts.scheme or "http", f"{host}:{port}", parts.path, "", ""))


def chat(host: str, model: str, messages: list, label: str = "",
         think: bool | None = None) -> str:
    host = normalize_host(host)
    prompt_chars = sum(len(m["content"]) for m in messages)
    log(f"→ ollama {model} {label}: requesting (prompt {prompt_chars} chars), "
        f"per-chunk timeout {OLLAMA_TIMEOUT}s …")
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": KEEPALIVE,
        "options": {"temperature": 0.1},
    }
    if think is not None:                 # omit -> model default; set only to force off/on
        payload["think"] = think
        log(f"  ollama {model}: think={think}")
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    chunks, think_chars = [], 0
    start = last_beat = time.monotonic()
    first_content_at = None
    final = {}
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("error"):
                raise RuntimeError(f"ollama error: {obj['error']}")
            msg = obj.get("message", {})
            piece = msg.get("content", "")
            think_chars += len(msg.get("thinking", "") or "")
            if piece and first_content_at is None:
                first_content_at = time.monotonic()
                log(f"  ollama {model}: first OUTPUT token after "
                    f"{fmt_secs(first_content_at - start)} "
                    f"(prompt eval + {think_chars} chars of reasoning)")
            chunks.append(piece)
            now = time.monotonic()
            if now - last_beat >= HEARTBEAT_SECS:
                chars = sum(len(c) for c in chunks)
                phase = "thinking" if first_content_at is None else "writing"
                log(f"  ollama {model}: {phase}… {chars} output + {think_chars} "
                    f"reasoning chars in {fmt_secs(now - start)}")
                last_beat = now
            if obj.get("done"):
                final = obj
                break
    text = "".join(chunks)
    dur = time.monotonic() - start
    n_tok = final.get("eval_count")
    tps = f"{n_tok / dur:.1f} tok/s" if n_tok and dur else "?"
    log(f"← ollama {model} {label}: done in {fmt_secs(dur)} — {len(text)} output chars, "
        f"{think_chars} reasoning chars, {n_tok or '?'} tokens ({tps})")
    return text
