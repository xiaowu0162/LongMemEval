"""Regression tests for src/evaluation/print_qa_metrics.py.

These tests run the script as a subprocess against tiny synthetic eval logs.
They require no network access and no API keys.
"""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "print_qa_metrics.py"

REF_DATA = [
    {"question_id": "q1", "question_type": "single-session-user",
     "question": "Capital of France?", "answer": "Paris"},
    {"question_id": "q2", "question_type": "multi-session",
     "question": "Capital of Germany?", "answer": "Berlin"},
]


def _write_log(path, model):
    lines = [
        {"question_id": "q1", "hypothesis": "Paris",
         "autoeval_label": {"model": model, "label": True}},
        {"question_id": "q2", "hypothesis": "Berlin",
         "autoeval_label": {"model": model, "label": False}},
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines))


def _run(in_file, ref_file):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(in_file), str(ref_file)],
        capture_output=True, text=True,
    )


def _write_ref(tmp_path):
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps(REF_DATA))
    return ref


def test_gpt4o_mini_log_is_accepted(tmp_path):
    """A log judged by the officially-supported gpt-4o-mini model must aggregate
    successfully instead of crashing with an AssertionError."""
    ref = _write_ref(tmp_path)
    log = tmp_path / "hyp.log"
    _write_log(log, "gpt-4o-mini-2024-07-18")
    proc = _run(log, ref)
    assert proc.returncode == 0, proc.stderr
    assert "Overall Accuracy: 0.5" in proc.stdout


def test_gpt4o_log_still_accepted(tmp_path):
    """The original happy path (gpt-4o) must keep working."""
    ref = _write_ref(tmp_path)
    log = tmp_path / "hyp.log"
    _write_log(log, "gpt-4o-2024-08-06")
    proc = _run(log, ref)
    assert proc.returncode == 0, proc.stderr
    assert "Overall Accuracy: 0.5" in proc.stdout


def test_mixed_model_log_is_rejected(tmp_path):
    """If a log mixes metric models the aggregate is not coherent, so the script
    must fail loudly rather than silently averaging across judges."""
    ref = _write_ref(tmp_path)
    log = tmp_path / "hyp.log"
    log.write_text("\n".join(json.dumps(x) for x in [
        {"question_id": "q1", "hypothesis": "Paris",
         "autoeval_label": {"model": "gpt-4o-2024-08-06", "label": True}},
        {"question_id": "q2", "hypothesis": "Berlin",
         "autoeval_label": {"model": "gpt-4o-mini-2024-07-18", "label": False}},
    ]))
    proc = _run(log, ref)
    assert proc.returncode != 0
    assert "same metric model" in (proc.stderr + proc.stdout)
