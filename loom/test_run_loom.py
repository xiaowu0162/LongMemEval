"""Offline adapter checks; all HTTP and reader calls are mocked.

Run with python -m pytest loom/test_run_loom.py -q.
"""

import asyncio
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import run_loom as R  # noqa: E402


# --------------------------------------------------------------------------
# _pct — a percentile that over-selects the tail silently biases every
# reported p50/p95 latency + token number.
# --------------------------------------------------------------------------
def test_pct_maps_q_to_min_and_max():
    xs = list(range(20))  # 0..19
    assert R._pct(xs, 0.0) == 0
    assert R._pct(xs, 1.0) == 19


def test_pct_p95_is_not_the_max():
    xs = list(range(20))
    # The old int(len*q) gave index 19 (the max) for p95 at n=20.
    assert R._pct(xs, 0.95) == 18


def test_pct_empty_is_zero():
    assert R._pct([], 0.5) == 0


# --------------------------------------------------------------------------
# _post — retry/status matrix. A dropped index/search silently corrupts
# recall, so transient failures must be ridden out; genuine client errors
# must NOT be retried; a redirect must not be parsed as JSON.
# --------------------------------------------------------------------------
class _Resp:
    def __init__(self, status, body=None, location=None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.headers = {"location": location} if location else {}

    def json(self):
        return self._body

    def raise_for_status(self):
        # Match httpx: only successful responses avoid a status error.
        if not 200 <= self.status_code < 300:
            raise httpx.HTTPStatusError("err", request=None, response=None)


class _Client:
    """Fake httpx.AsyncClient whose .post replays a scripted sequence; an
    Exception entry is raised instead of returned."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    async def post(self, *a, **k):
        self.calls += 1
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def _s(*a, **k):
        return None
    monkeypatch.setattr(R.asyncio, "sleep", _s)


def _call_post(client, **kw):
    return asyncio.run(R._post(client, "http://loom/op", {"q": 1}, "tok", **kw))


def test_post_2xx_returns_json():
    c = _Client([_Resp(200, {"ok": True})])
    assert _call_post(c) == {"ok": True}
    assert c.calls == 1


def test_post_retries_5xx_then_succeeds():
    c = _Client([_Resp(503), _Resp(200, {"ok": 1})])
    assert _call_post(c) == {"ok": 1}
    assert c.calls == 2


def test_post_retries_429_then_succeeds():
    c = _Client([_Resp(429), _Resp(200, {"ok": 1})])
    assert _call_post(c) == {"ok": 1}
    assert c.calls == 2


def test_post_retries_transient_network_error():
    c = _Client([httpx.ConnectError("boom"), _Resp(200, {"ok": 1})])
    assert _call_post(c) == {"ok": 1}
    assert c.calls == 2


def test_post_non_429_4xx_raises_without_retry():
    c = _Client([_Resp(400)])
    with pytest.raises(httpx.HTTPStatusError):
        _call_post(c)
    assert c.calls == 1  # a genuine client error is not retried


def test_post_3xx_is_explicit_error_not_json_parse():
    # A redirect must surface as a clear error, not fall through to r.json().
    c = _Client([_Resp(301, location="https://elsewhere/op")])
    with pytest.raises(RuntimeError):
        _call_post(c)
    assert c.calls == 1


def test_post_exhausts_retries_and_raises_on_persistent_5xx():
    c = _Client([_Resp(500), _Resp(500), _Resp(500)])
    with pytest.raises(httpx.HTTPStatusError):
        _call_post(c, retries=3)
    assert c.calls == 3


def test_post_rejects_retries_below_one():
    # retries=0 would otherwise skip the loop and hit RuntimeError("unreachable").
    c = _Client([_Resp(200, {"ok": 1})])
    with pytest.raises(ValueError):
        _call_post(c, retries=0)
    assert c.calls == 0


# --------------------------------------------------------------------------
# _answer — must match the official temperature/output cap for non-reasoning
# models (temperature=0, max_tokens=800) and leave reasoning models uncapped,
# or results drift from the official harness.
# --------------------------------------------------------------------------
def _capture_answer(model, monkeypatch):
    captured = {}

    async def fake_post(client, url, body, key, **kw):
        captured.update(body)
        return {"choices": [{"message": {"content": "ans"}}]}

    monkeypatch.setattr(R, "_post", fake_post)
    out = asyncio.run(R._answer(None, "q?", [], "2023-01-01", model, "k"))
    return out, captured


def test_answer_caps_non_reasoning_model(monkeypatch):
    out, body = _capture_answer("gpt-4o", monkeypatch)
    assert out == "ans"
    assert body["max_tokens"] == 800
    assert body["temperature"] == 0.0


def test_answer_leaves_reasoning_model_uncapped(monkeypatch):
    _, body = _capture_answer("gpt-5", monkeypatch)
    assert "max_tokens" not in body
    assert "temperature" not in body


# --------------------------------------------------------------------------
# _history_block — dated blocks, oldest first, with the epoch sentinel and
# missing dates rendered "unknown" (feeds date-diff questions).
# --------------------------------------------------------------------------
def test_history_block_orders_oldest_first():
    hits = [
        {"content_excerpt": "ZZNEW", "valid_at": "2023-05-02"},
        {"content_excerpt": "QQOLD", "valid_at": "2023-05-01"},
    ]
    out = R._history_block(hits)
    assert out.index("QQOLD") < out.index("ZZNEW")


def test_history_block_epoch_sentinel_is_undated():
    hits = [{"content_excerpt": "x", "valid_at": "1970-01-01T00:00:00"}]
    assert "Date: unknown" in R._history_block(hits)


def test_history_block_empty_is_placeholder():
    assert R._history_block([]) == "(no facts retrieved)"


@pytest.mark.parametrize(
    "failed_ids, latency_error",
    [({"q-failed"}, False), ({"q-ok", "q-failed"}, False), ({"q-failed"}, True)],
    ids=["mixed-question-failure", "all-questions-failed", "latency-search-failed"],
)
def test_main_latency_preserves_failed_questions(
    tmp_path, monkeypatch, capsys, failed_ids, latency_error
):
    import json

    dataset = tmp_path / "dataset.json"
    hypotheses = tmp_path / "hypotheses.jsonl"
    metrics_file = tmp_path / "metrics.json"
    dataset.write_text(json.dumps([
        {"question_id": qid, "question_type": "single-session-user"}
        for qid in ("q-ok", "q-failed")
    ]))
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setenv("LOOM_TOKEN", "offline-test-token")
    monkeypatch.setattr(sys, "argv", [
        "run_loom.py", "--dataset", str(dataset), "--out", str(hypotheses),
        "--metrics-out", str(metrics_file), "--base-url", "http://offline.invalid",
        "--retrieval-budget", "fast", "--measure-latency",
    ])

    class OfflineClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, **kwargs):
            assert url == "http://offline.invalid/v1/health"
            return _Resp(200)

    async def run_item(client, base_url, token, item, **kwargs):
        qid = item["question_id"]
        if qid in failed_ids:
            raise httpx.ConnectError("offline question failure")
        return {
            "question_id": qid, "question_type": item["question_type"],
            "hypothesis": "answer", "recalled": True, "all_covered": True,
            "fact_in_context": True, "ns": "offline-namespace", "query": "question?",
            "q_iso": "2023-01-01",
            "search_ms_loaded": 120.0,
        }

    searches = []

    async def search(client, url, body, token, **kwargs):
        assert url == "http://offline.invalid/v1/memory.search"
        assert token == "offline-test-token"
        searches.append(body)
        if latency_error:
            raise httpx.ConnectError("offline latency failure")
        return {}

    monkeypatch.setattr(R.httpx, "AsyncClient", OfflineClient)
    monkeypatch.setattr(R, "_run_item", run_item)
    monkeypatch.setattr(R, "_post", search)
    monkeypatch.setattr(R.time, "perf_counter", lambda: 1.0)

    assert asyncio.run(R.main()) == 0
    rows = [json.loads(line) for line in hypotheses.read_text().splitlines()]
    assert len(rows) == 2
    assert {r["question_id"]: r["hypothesis"] for r in rows} == {
        qid: "" if qid in failed_ids else "answer" for qid in ("q-ok", "q-failed")
    }
    eligible = 2 - len(failed_ids)
    measured = eligible if not latency_error else 0
    metrics = json.loads(metrics_file.read_text())
    assert metrics["n_questions"] == 2
    for key in ("recall_session_pct", "recall_allsession_pct", "recall_fact_pct"):
        assert metrics[key] == eligible / 2 * 100
    assert metrics["latency_clean_eligible_n"] == eligible
    assert metrics["latency_clean_attempted_n"] == eligible
    assert metrics["latency_clean_measured_n"] == measured
    assert metrics["latency_clean_skipped_n"] == len(failed_ids)
    assert metrics["latency_clean_failed_n"] == eligible - measured
    assert metrics["latency_clean_p50_ms"] == (0 if measured else None)
    assert metrics["latency_clean_p95_ms"] == (0 if measured else None)
    assert metrics["latency_loaded_p50_ms"] == (120 if eligible else None)
    assert metrics["latency_loaded_p95_ms"] == (120 if eligible else None)
    assert metrics["latency_loaded_measured_n"] == eligible
    assert metrics["latency_loaded_skipped_n"] == len(failed_ids)
    assert searches == ([{
        "org": "dev", "namespace": "offline-namespace", "agent": "lme-loom",
        "user_id": "-", "namespace_scope": "this",
        "query": "question?", "top_k": 30, "search_mode": "rrf",
        "alpha": 0.5, "include_top_n_unmatched": 120,
        "observation_date": "2023-01-01", "retrieval_budget": "fast",
    }] if eligible else [])
    output = capsys.readouterr()
    assert "offline question failure" in output.err
    assert "CLEAN serving latency counts:" in output.out
    assert "quiesced" not in output.out
    assert "Now grade with the official evaluator:" in output.out


@pytest.mark.parametrize(
    "ingest_response, derived",
    [({"deferred": True, "queue_id": "queued", "written": []}, None),
     ({"queue_id": "queued", "written": []}, None),
     ({"written": [{"memory_key": "fact"}]}, None),
     ({"written": [{"memory_key": "fact"}]}, {"count": 2})],
    ids=["deferred", "queue-id", "inline", "structured-context"],
)
def test_item_requires_inline_ingestion(monkeypatch, ingest_response, derived):
    requests = []
    reads = []

    async def post(client, url, body, token):
        requests.append((url, body))
        if url.endswith("/v1/memory.set_from_messages"):
            return ingest_response
        assert url.endswith("/v1/memory.search")
        return {"results": [], "derived_aggregate": derived}

    async def answer(*args, **kwargs):
        assert kwargs["derived"] == derived
        reads.append(True)
        return "answer"

    monkeypatch.setattr(R, "_post", post)
    monkeypatch.setattr(R, "_answer", answer)
    item = {"question_id": "q", "question": "What changed?",
            "haystack_sessions": [[{"role": "user", "content": "A fact"}]]}

    def run():
        return asyncio.run(R._run_item(
            None, "http://offline.invalid", "offline-token", item,
            top_k=1, search_mode="rrf", ingest_conc=1,
            model="offline", api_key="offline-key", retrieval_budget="fast",
            item_sem=asyncio.Semaphore(1),
        ))

    if ingest_response.get("deferred") or ingest_response.get("queue_id"):
        with pytest.raises(RuntimeError, match="Deferred extraction"):
            run()
        assert len(requests) == 1
        assert not reads
    else:
        result = run()
        assert result["hypothesis"] == "answer"
        assert result["ctx_tokens"] == (
            len(R._history_block([])) + len(R._derived_block(derived))
        ) // 4
        assert len(requests) == 2
        assert requests[1][1]["namespace_scope"] == "this"
        assert reads == [True]
    assert requests[0][1]["defer"] is False
