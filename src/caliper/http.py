"""Minimal JSON/NDJSON HTTP client built on the standard library.

Caliper deliberately avoids an HTTP dependency: the only shapes needed are a
JSON GET, a JSON POST and a streamed NDJSON POST.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

log = logging.getLogger(__name__)

_USER_AGENT = "caliper/0.1 (local benchmark client)"


class HttpError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


def _request(
    url: str, payload: dict[str, Any] | None, method: str, timeout: float
):  # pragma: no cover - thin wrapper, exercised via integration
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"User-Agent": _USER_AGENT}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    if not url.startswith(("http://", "https://")):
        raise HttpError(f"refusing non-HTTP url: {url!r}")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise HttpError(f"HTTP {exc.code} for {url}", status=exc.code, body=body) from exc
    except urllib.error.URLError as exc:
        raise HttpError(f"could not reach {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise HttpError(f"timeout after {timeout}s for {url}") from exc


def get_json(url: str, timeout: float = 15.0) -> Any:
    with _request(url, None, "GET", timeout) as resp:
        return json.loads(resp.read().decode())


def post_json(url: str, payload: dict[str, Any], timeout: float = 60.0) -> Any:
    with _request(url, payload, "POST", timeout) as resp:
        raw = resp.read().decode()
    return json.loads(raw) if raw.strip() else {}


def post_ndjson_stream(
    url: str, payload: dict[str, Any], timeout: float = 600.0
) -> Iterator[dict[str, Any]]:
    """POST and yield each newline-delimited JSON object as it arrives.

    The response is read line by line so the caller can time the first
    generated token rather than the first HTTP byte.
    """
    with _request(url, payload, "POST", timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("skipping non-JSON stream line: %r", line[:200])
