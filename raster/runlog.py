"""Timestamped, elapsed-stamped logging shared by the build/exec path — so a long
local-model run is legible in trundlr logs (matches the doer's format)."""

import time
from datetime import datetime

_T0 = time.monotonic()


def log(msg: str) -> None:
    elapsed = time.monotonic() - _T0
    print(f"[raster {datetime.now():%H:%M:%S} +{elapsed:6.0f}s] {msg}", flush=True)


def fmt_secs(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m}m{sec:02d}s" if m else f"{sec}s"
