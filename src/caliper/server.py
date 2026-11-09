"""Local HTTP server for the Caliper viewer.

Built on :mod:`http.server` so the project keeps zero runtime dependencies.
Routing and serialisation live here; all behaviour is in :mod:`caliper.webapi`.

Binding: 127.0.0.1 by default and the CLI refuses any other host unless
``--allow-remote`` is passed explicitly. A benchmark viewer exposes machine
telemetry, model inventory and raw model output, none of which belongs on a
network interface. Even with ``--allow-remote`` there is no authentication, so
the documented way to reach it from another machine is an SSH tunnel.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import __version__
from .config import Config
from .storage import Store
from .webapi import (
    ApiError,
    compare_runs,
    delete_grade,
    get_attempt,
    get_run,
    grading_overview,
    grading_queue,
    list_attempts,
    list_runs,
    live_status,
    set_grade,
)

log = logging.getLogger(__name__)

WEB_ROOT = Path(__file__).parent / "web"
MAX_BODY_BYTES = 1 << 20  # 1 MiB: grades are small, so anything larger is a mistake

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json",
}


def _first(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key)
    return values[0] if values else None


def _flag(params: dict[str, list[str]], key: str) -> bool:
    value = _first(params, key)
    return value is not None and value.lower() in {"1", "true", "yes"}


def route_get(store: Store, path: str, params: dict[str, list[str]]) -> Any:
    """Dispatch a GET. Raises :class:`ApiError` for anything unroutable."""
    parts = [p for p in path.strip("/").split("/") if p]
    # parts[0] == "api"
    if parts == ["api", "runs"]:
        limit = _first(params, "limit")
        return list_runs(store, limit=int(limit) if limit else 50)
    if parts == ["api", "status"]:
        return live_status(store)
    if parts == ["api", "compare"]:
        ids = params.get("run") or []
        return compare_runs(store, ids)
    if len(parts) == 3 and parts[:2] == ["api", "runs"]:
        return get_run(store, parts[2])
    if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "attempts":
        return list_attempts(
            store,
            parts[2],
            model=_first(params, "model"),
            category=_first(params, "category"),
            task_id=_first(params, "task"),
            graded=_first(params, "graded"),
            include_warmup=_flag(params, "warmup"),
            only_ungraded=_flag(params, "ungraded"),
        )
    if parts == ["api", "grading"]:
        return grading_overview(store)
    if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "grading":
        return grading_queue(store, parts[2])
    if len(parts) == 3 and parts[:2] == ["api", "attempts"]:
        try:
            attempt_id = int(parts[2])
        except ValueError as exc:
            raise ApiError(f"attempt id must be an integer: {parts[2]!r}") from exc
        return get_attempt(store, attempt_id)
    if parts == ["api", "version"]:
        return {"caliper_version": __version__}
    raise ApiError(f"no such endpoint: {path}", status=404)


def route_post(store: Store, path: str, body: dict[str, Any]) -> Any:
    parts = [p for p in path.strip("/").split("/") if p]
    if parts == ["api", "grades"]:
        return set_grade(store, body)
    raise ApiError(f"no such endpoint: {path}", status=404)


def route_delete(store: Store, path: str) -> Any:
    parts = [p for p in path.strip("/").split("/") if p]
    if len(parts) == 3 and parts[:2] == ["api", "grades"]:
        try:
            attempt_id = int(parts[2])
        except ValueError as exc:
            raise ApiError(f"attempt id must be an integer: {parts[2]!r}") from exc
        return delete_grade(store, attempt_id)
    raise ApiError(f"no such endpoint: {path}", status=404)


class CaliperHandler(BaseHTTPRequestHandler):
    """Serves the single-page viewer and the JSON API."""

    server_version = f"caliper/{__version__}"
    protocol_version = "HTTP/1.1"

    def __init__(self, *args: Any, store_factory: Callable[[], Store], **kwargs: Any) -> None:
        self._store_factory = store_factory
        self._local = threading.local()
        super().__init__(*args, **kwargs)

    # -- helpers ----------------------------------------------------------
    @property
    def store(self) -> Store:
        """One SQLite connection per thread; sqlite3 objects are not shareable."""
        store = getattr(self._local, "store", None)
        if store is None:
            store = self._store_factory()
            self._local.store = store
        return store

    def log_message(self, format: str, *args: Any) -> None:
        log.debug("%s - %s", self.address_string(), format % args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The viewer reads local data only; no embedding, no external scripts.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, default=str).encode()
        self._send(status, body, "application/json; charset=utf-8")

    def _send_error_json(self, exc: ApiError) -> None:
        self._send_json(exc.status, {"error": str(exc), "status": exc.status})

    def _serve_static(self, path: str) -> None:
        name = "index.html" if path in {"", "/"} else path.lstrip("/")
        target = (WEB_ROOT / name).resolve()
        # Refuse anything that escapes the web root.
        if not str(target).startswith(str(WEB_ROOT.resolve())) or not target.is_file():
            self._send_json(404, {"error": f"not found: {path}", "status": 404})
            return
        content_type = _CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self._send(200, target.read_bytes(), content_type)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ApiError("a JSON body is required")
        if length > MAX_BODY_BYTES:
            raise ApiError("request body too large", status=413)
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ApiError(f"body is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ApiError("body must be a JSON object")
        return payload

    # -- verbs ------------------------------------------------------------
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self._serve_static(parsed.path)
            return
        try:
            self._send_json(200, route_get(self.store, parsed.path, parse_qs(parsed.query)))
        except ApiError as exc:
            self._send_error_json(exc)
        except Exception:
            log.exception("GET %s failed", self.path)
            self._send_json(500, {"error": "internal error; see server log", "status": 500})

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            body = self._read_body()
            self._send_json(200, route_post(self.store, parsed.path, body))
        except ApiError as exc:
            self._send_error_json(exc)
        except Exception:
            log.exception("POST %s failed", self.path)
            self._send_json(500, {"error": "internal error; see server log", "status": 500})

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        try:
            self._send_json(200, route_delete(self.store, parsed.path))
        except ApiError as exc:
            self._send_error_json(exc)
        except Exception:
            log.exception("DELETE %s failed", self.path)
            self._send_json(500, {"error": "internal error; see server log", "status": 500})


def build_server(config: Config, host: str = "127.0.0.1", port: int = 8777) -> ThreadingHTTPServer:
    """Create the server without starting it, so tests can drive it directly."""
    db_path = config.db_path
    handler = partial(CaliperHandler, store_factory=lambda: Store(db_path))
    return ThreadingHTTPServer((host, port), handler)


def serve(config: Config, host: str = "127.0.0.1", port: int = 8777) -> None:
    httpd = build_server(config, host, port)
    actual_host, actual_port = httpd.server_address[:2]
    log.info(
        "caliper viewer on http://%s:%s (database %s)", actual_host, actual_port, config.db_path
    )
    print(f"Caliper viewer: http://{actual_host}:{actual_port}")
    print(f"Database: {config.db_path}")
    print("Read-only except for manual rubric grades. Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
