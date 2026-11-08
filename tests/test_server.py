"""HTTP-level tests for the viewer server.

Routing is tested as pure functions; the socket path is exercised once end to
end against a real server on an ephemeral port, including the static page, a
JSON round-trip and a grade POST.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from caliper import server
from caliper.backends import mock_backend_for_suite
from caliper.evals import load_suite
from caliper.evals.spec import Suite
from caliper.hardware import TelemetrySampler
from caliper.runner import BenchmarkRunner, RunSettings
from caliper.webapi import ApiError

THRESHOLDS = {
    "max_gpu_utilization_pct": 100.0,
    "min_free_vram_mib": 0.0,
    "min_free_ram_mib": 0.0,
    "min_free_disk_gib": 0.0,
    "allow_foreign_gpu_processes": True,
    "max_foreign_gpu_mib": 1e9,
    "gpu_sample_count": 1,
    "gpu_sample_interval_s": 0.0,
}


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr("caliper.hardware.sample_gpus", lambda: [])
    monkeypatch.setattr("caliper.webapi.sample_gpus", lambda: [])
    monkeypatch.setattr(TelemetrySampler, "interval_s", 0.01, raising=False)


@pytest.fixture()
def seeded(store):
    v1 = load_suite("v1")
    suite = Suite(
        name="v1",
        suite_version=v1.suite_version,
        tasks=[v1.get("instruction.exact_format"), v1.get("coding.refactor_decision")],
    )
    backend = mock_backend_for_suite(suite, models=["m:1b"])
    settings = RunSettings(models=["m:1b"], repetitions=1, warmup_runs=0, suite="v1")
    outcome = BenchmarkRunner(backend, store, settings, THRESHOLDS).run(suite)
    return outcome.run_id


# ------------------------------------------------------------------- routing


def test_route_get_dispatches_every_endpoint(store, seeded):
    assert "runs" in server.route_get(store, "/api/runs", {})
    assert server.route_get(store, f"/api/runs/{seeded}", {})["run"]["id"] == seeded
    assert "attempts" in server.route_get(store, f"/api/runs/{seeded}/attempts", {})
    assert "queue" in server.route_get(store, f"/api/runs/{seeded}/grading", {})
    assert "gpu" in server.route_get(store, "/api/status", {})
    assert "caliper_version" in server.route_get(store, "/api/version", {})
    assert server.route_get(store, "/api/compare", {"run": [seeded]})["comparable"] is True


def test_route_get_passes_query_filters(store, seeded):
    payload = server.route_get(store, f"/api/runs/{seeded}/attempts", {"graded": ["manual_rubric"]})
    assert payload["count"] == 1
    with_warmup = server.route_get(store, f"/api/runs/{seeded}/attempts", {"warmup": ["1"]})
    assert with_warmup["count"] >= payload["count"]


def test_route_get_unknown_endpoint_is_404(store):
    with pytest.raises(ApiError) as exc:
        server.route_get(store, "/api/nope", {})
    assert exc.value.status == 404


def test_route_get_rejects_a_non_numeric_attempt_id(store):
    with pytest.raises(ApiError, match="must be an integer"):
        server.route_get(store, "/api/attempts/abc", {})


def test_route_post_and_delete_grades(store, seeded):
    queue = server.route_get(store, f"/api/runs/{seeded}/grading", {})["queue"]
    attempt_id = queue[0]["attempt_id"]
    result = server.route_post(
        store, "/api/grades", {"attempt_id": attempt_id, "score": 1, "max_score": 2}
    )
    assert result["fraction"] == 0.5
    assert server.route_delete(store, f"/api/grades/{attempt_id}")["deleted"] is True


def test_route_post_unknown_endpoint_is_404(store):
    with pytest.raises(ApiError) as exc:
        server.route_post(store, "/api/nope", {})
    assert exc.value.status == 404


# -------------------------------------------------------------------- socket


@pytest.fixture()
def live(config, seeded):
    httpd = server.build_server(config, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield f"http://{host}:{port}", seeded
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def get(url: str):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.status, resp.read(), resp.headers


def test_server_serves_the_viewer_page(live):
    base, _ = live
    status, body, headers = get(base + "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"<title>Caliper</title>" in body
    assert b"/api/runs" in body


def test_server_sets_hardening_headers(live):
    base, _ = live
    _, _, headers = get(base + "/api/version")
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Cache-Control"] == "no-store"


def test_server_returns_run_json(live):
    base, run_id = live
    status, body, headers = get(f"{base}/api/runs/{run_id}")
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    payload = json.loads(body)
    assert payload["run"]["id"] == run_id
    assert payload["models"][0]["model"] == "m:1b"


def test_server_404s_an_unknown_run_as_json(live):
    base, _ = live
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(base + "/api/runs/nope")
    assert exc.value.code == 404
    assert json.loads(exc.value.read())["error"]


def test_server_refuses_path_traversal(live):
    base, _ = live
    for path in ("/../pyproject.toml", "/%2e%2e/setup.py", "/../../etc/passwd"):
        try:
            status, body, _ = get(base + path)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
            continue
        assert status == 404, path
        assert b"pyproject" not in body


def test_server_accepts_a_grade_post(live, store):
    base, run_id = live
    queue = json.loads(get(f"{base}/api/runs/{run_id}/grading")[1])["queue"]
    attempt_id = queue[0]["attempt_id"]
    payload = json.dumps(
        {
            "attempt_id": attempt_id,
            "criteria": [{"text": "a", "met": True}, {"text": "b", "met": False}],
            "grader": "tester",
        }
    ).encode()
    req = urllib.request.Request(
        base + "/api/grades",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read())
    assert result == {
        "attempt_id": attempt_id,
        "score": 1.0,
        "max_score": 2.0,
        "fraction": 0.5,
        "stored": True,
    }
    # Persisted, and visible to a fresh reader.
    assert store.manual_scores(run_id)[attempt_id]["grader"] == "tester"


def test_server_rejects_a_non_json_body(live):
    base, _ = live
    req = urllib.request.Request(base + "/api/grades", data=b"not json", method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=10)
    assert exc.value.code == 400
    assert "not valid JSON" in json.loads(exc.value.read())["error"]


def test_server_rejects_an_empty_body(live):
    base, _ = live
    req = urllib.request.Request(base + "/api/grades", data=b"", method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=10)
    assert exc.value.code == 400


def test_server_has_no_endpoint_that_starts_a_run(store):
    """The viewer observes; it must not be able to launch inference."""
    for path in ("/api/run", "/api/runs/start", "/api/benchmark"):
        with pytest.raises(ApiError):
            server.route_post(store, path, {})
    # Neither module may even import the runner, so the capability is absent
    # rather than merely unrouted.
    from pathlib import Path

    from caliper import webapi

    for module in (server, webapi):
        source = Path(module.__file__).read_text()
        assert "BenchmarkRunner" not in source, module.__name__
        assert "from .runner import" not in source, module.__name__
