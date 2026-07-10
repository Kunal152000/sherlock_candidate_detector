"""Offline evaluation: replay scripted interview scenarios and assert the
detector identifies the correct candidate.

Each ``scenarioN.json`` file describes a realistic edge case (candidate under a
device name, a wrong name typed by the interviewer, a panel with silent
observers, ...). A scenario carries:

  * ``start``    -- the ``POST /start`` metadata (candidate/interviewer identity).
  * ``events``   -- the delta events that would arrive via ``POST /events``.
  * ``expected`` -- the participant id that *should* win, plus optional
    ``min_confidence`` and ``ambiguous`` assertions.

These tests exercise the real pipeline (``meeting_state`` -> ``signals`` ->
``engine``) end to end, but fully **offline**: the network LLM call is patched
out so the deterministic heuristic fallback runs every time. That makes the
suite fast, reproducible, and runnable with no API key -- which is exactly how
the challenge's evaluation is expected to run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import engine, llm
from app.meeting_state import store
from app.models import Event, StartRequest

SCENARIO_DIR = Path(__file__).parent
SCENARIO_FILES = sorted(SCENARIO_DIR.glob("scenario*.json"))


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(autouse=True)
def force_offline_llm(monkeypatch):
    """Force the deterministic heuristic path for every test.

    Patching ``_query_llm`` to return ``None`` guarantees the suite never hits
    the network and behaves identically whether or not an ``OPENROUTER_API_KEY``
    is present in the environment.
    """
    monkeypatch.setattr(llm, "_query_llm", lambda meeting: None)


@pytest.fixture(autouse=True)
def clean_store():
    """Isolate meetings between tests (the store is a process-wide singleton)."""
    store.reset()
    yield
    store.reset()


def _run_scenario(scenario: dict):
    """Start the meeting, apply all events, recompute, and return the meeting."""
    start = scenario["start"]
    meeting_id = start.get("meeting_id", "default")
    store.start_meeting(StartRequest(**start))
    events = [Event(**raw) for raw in scenario["events"]]
    meeting = store.apply_events(meeting_id, events)
    engine.recompute(meeting)
    return meeting


@pytest.mark.parametrize("path", SCENARIO_FILES, ids=lambda p: p.stem)
def test_scenario_identifies_expected_candidate(path: Path):
    scenario = _load(path)
    expected = scenario["expected"]
    meeting = _run_scenario(scenario)

    assert meeting.candidate_id == expected["candidate_id"], (
        f"{path.name}: expected winner {expected['candidate_id']!r} but got "
        f"{meeting.candidate_id!r} (confidence={meeting.confidence})"
    )

    # The verdict must clear the scenario's confidence floor...
    min_conf = expected.get("min_confidence", 0.0)
    assert meeting.confidence >= min_conf, (
        f"{path.name}: confidence {meeting.confidence} below floor {min_conf}"
    )

    # ...report the expected (un)ambiguity...
    if "ambiguous" in expected:
        assert meeting.ambiguous is expected["ambiguous"], (
            f"{path.name}: ambiguous={meeting.ambiguous}, "
            f"expected {expected['ambiguous']}"
        )

    # ...and be able to explain itself (at least one positive reason).
    winner = meeting.participants[meeting.candidate_id]
    assert winner.reasons, f"{path.name}: winner has no explanation reasons"


@pytest.mark.parametrize("path", SCENARIO_FILES, ids=lambda p: p.stem)
def test_expected_candidate_ranks_first(path: Path):
    """The engine's ranked view should also place the expected candidate on top."""
    scenario = _load(path)
    meeting = _run_scenario(scenario)

    ranked = engine.rank_participants(meeting)
    assert ranked, f"{path.name}: no participants ranked"
    assert ranked[0].id == scenario["expected"]["candidate_id"]
    # The winner must strictly outscore the runner-up (no accidental tie-win).
    if len(ranked) >= 2:
        assert ranked[0].score > ranked[1].score


def test_scenario_files_present():
    """Guard against an empty glob silently passing the parametrized tests."""
    assert len(SCENARIO_FILES) == 3, (
        f"expected 3 scenario files, found {[p.name for p in SCENARIO_FILES]}"
    )
