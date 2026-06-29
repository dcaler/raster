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

# Context-window sizing (local-llm context-sizing guidance, LL/NN). The KV cache is LINEAR in the
# context window, not the parameter count, so the window — not the model size — usually decides
# whether a model fits VRAM. We size num_ctx to the prompt instead of accepting ollama's large
# default; all knobs are env-overridable for a box with more (or less) VRAM.
CHARS_PER_TOKEN = int(os.environ.get("RASTER_CHARS_PER_TOKEN", 4))        # rough prompt-token estimate
OUTPUT_HEADROOM_TOKENS = int(os.environ.get("RASTER_OUTPUT_HEADROOM_TOKENS", 4096))  # room for the reply
MIN_NUM_CTX = int(os.environ.get("RASTER_MIN_NUM_CTX", 4096))
MAX_NUM_CTX = int(os.environ.get("RASTER_MAX_NUM_CTX", 32768))


def estimate_tokens(chars: int) -> int:
    """Cheap char->token estimate (ceil, ~4 chars/token). Deliberately rough — we only need the
    right power-of-two bucket for num_ctx, not an exact count."""
    return -(-max(chars, 0) // CHARS_PER_TOKEN)


def pick_num_ctx(prompt_chars: int, output_tokens: int = OUTPUT_HEADROOM_TOKENS) -> int:
    """Smallest power-of-two context window that holds the prompt PLUS room for the reply.

    The KV cache is LINEAR in this window (a key+value vector per layer per attention head for every
    token), so it can dwarf the model weights: an 8B model — ~5 GB of Q4 weights — at a 32k default
    context needs ~32 GB of KV and spills layers to CPU on an 8 GB card, collapsing generation to
    <1 tok/s (local-llm context-sizing guidance, LL). The fix is to size to NEED, not accept the
    server's max default: round UP (NN — a window smaller than prompt+output silently truncates) and
    clamp to MAX_NUM_CTX. Hitting the clamp means the prompt itself overflows the budget — a
    hardware/trim problem the caller logs loudly, never a silent truncation here."""
    need = estimate_tokens(prompt_chars) + max(output_tokens, 0)
    ctx = MIN_NUM_CTX
    while ctx < need and ctx < MAX_NUM_CTX:
        ctx *= 2
    return min(ctx, MAX_NUM_CTX)


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
    num_ctx = pick_num_ctx(prompt_chars)
    log(f"→ ollama {model} {label}: requesting (prompt {prompt_chars} chars ~{estimate_tokens(prompt_chars)} "
        f"tok, num_ctx={num_ctx}), per-chunk timeout {OLLAMA_TIMEOUT}s …")
    need = estimate_tokens(prompt_chars) + OUTPUT_HEADROOM_TOKENS
    if need > MAX_NUM_CTX:
        # NN: the reply shares the window, so prompt+output must FIT or the context silently
        # truncates (mysterious quality drop, no error). We've hit the cap — say so loudly.
        log(f"  WARNING: prompt+output (~{need} tok) exceeds the num_ctx cap {MAX_NUM_CTX} — the "
            f"window will TRUNCATE. Trim the prompt (API digest) or raise RASTER_MAX_NUM_CTX if the "
            f"card has the VRAM (KV cache is linear in num_ctx).")
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": KEEPALIVE,
        "options": {"temperature": 0.1, "num_ctx": num_ctx},
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
