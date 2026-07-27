"""Minimal OpenRouter client with cost accounting and a hard budget stop.

Separate from judge.py because the judge deliberately caps output at 8 tokens (it emits a bare
number); reusing that call path for data generation silently truncates every sample at 8 tokens.
That mistake cost one debugging round -- hence a dedicated module.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_PATH = Path("/root/.or_key")


class Budget:
    """Thread-safe spend tracker that trips a stop flag before the cap, not after."""

    def __init__(self, cap_usd: float):
        self.cap = cap_usd
        self.spent = 0.0
        self._lock = threading.Lock()
        self.stopped = threading.Event()

    def add(self, cost: float) -> None:
        with self._lock:
            self.spent += cost
            if self.spent >= self.cap:
                self.stopped.set()

    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self.cap - self.spent)


def api_key() -> str:
    return KEY_PATH.read_text().strip()


def complete(
    prompt: str,
    model: str,
    api_key_: str,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    system: str | None = None,
    retries: int = 3,
    timeout: int = 120,
) -> tuple[str | None, float]:
    """Single completion. Returns (text, cost_usd). text is None on unrecoverable failure."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "usage": {"include": True},
    }).encode()

    for attempt in range(retries):
        req = urllib.request.Request(
            URL, data=body,
            headers={"Authorization": f"Bearer {api_key_}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            if "choices" not in d or not d["choices"]:
                # OpenRouter surfaces upstream provider errors in the body, not the HTTP status.
                err = d.get("error", {})
                if attempt < retries - 1:
                    threading.Event().wait(2 ** attempt)
                    continue
                return f"__ERR__{err.get('message', str(d)[:200])}", 0.0
            text = d["choices"][0]["message"]["content"]
            cost = float(d.get("usage", {}).get("cost", 0.0))
            return (text.strip() if text else None), cost
        except urllib.error.HTTPError as e:
            detail = e.read()[:200].decode(errors="replace")
            if e.code in (429, 500, 502, 503, 529) and attempt < retries - 1:
                threading.Event().wait(2 ** attempt * 2)
                continue
            return f"__HTTP_{e.code}__{detail}", 0.0
        except Exception:  # noqa: BLE001 -- transient network
            if attempt < retries - 1:
                threading.Event().wait(2 ** attempt * 2)
                continue
            return None, 0.0
    return None, 0.0
