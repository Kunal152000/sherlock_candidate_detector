"""Streamlit dashboard for the Sherlock Candidate Detector.

A thin, read-only client that polls the FastAPI backend and visualises the live
verdict:

  * the current best-guess candidate with a confidence bar and its reasons,
  * an ambiguity warning when the top two participants are too close to call,
  * a ranked list of every participant (score, confidence, camera / presence,
    speaking time, and the signals that fired), and
  * a chronological "recent transcript" with the candidate's lines highlighted.

The dashboard owns no state of its own -- it simply reads the REST endpoints
(``/current_candidate``, ``/participants``, ``/transcript``, ``/health``) and
re-renders. It degrades gracefully: if the backend is unreachable, the meeting
hasn't started, or an older backend lacks ``/transcript``, it shows a helpful
message instead of crashing.

Run it with::

    streamlit run frontend/dashboard.py

Point it at a non-default backend via the sidebar or the ``API_URL`` env var.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_MEETING_ID = os.getenv("MEETING_ID", "default")
REQUEST_TIMEOUT = 5  # seconds; keep short so a dead backend fails fast.


# --------------------------------------------------------------------------- #
# Backend client (all network I/O lives here, with uniform error handling)
# --------------------------------------------------------------------------- #


class BackendError(Exception):
    """Raised when the backend is unreachable or returns an unexpected error."""


class MeetingNotStarted(Exception):
    """Raised when the meeting exists conceptually but has no data yet (404)."""


def _get(base_url: str, path: str, params: Optional[dict] = None) -> Any:
    """GET ``path`` from the backend and return parsed JSON.

    Translates transport failures and non-2xx responses into typed exceptions so
    the render code can present them as friendly UI states.
    """
    url = base_url.rstrip("/") + path
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        raise BackendError(f"Could not reach the backend at {url!r}: {exc}") from exc

    if resp.status_code == 404:
        raise MeetingNotStarted(resp.text)
    if not resp.ok:
        raise BackendError(f"{resp.status_code} from {path}: {resp.text[:200]}")
    try:
        return resp.json()
    except ValueError as exc:
        raise BackendError(f"Invalid JSON from {path}: {exc}") from exc


def fetch_health(base_url: str) -> dict:
    return _get(base_url, "/health")


def fetch_candidate(base_url: str, meeting_id: str) -> dict:
    return _get(base_url, "/current_candidate", {"meeting_id": meeting_id})


def fetch_participants(base_url: str, meeting_id: str) -> dict:
    return _get(base_url, "/participants", {"meeting_id": meeting_id})


def fetch_transcript(base_url: str, meeting_id: str, limit: int = 30) -> Optional[dict]:
    """Fetch recent transcript, tolerating a backend that lacks the endpoint.

    Returns ``None`` (rather than raising) when the endpoint is missing, so the
    dashboard still works against an older backend.
    """
    try:
        return _get(base_url, "/transcript", {"meeting_id": meeting_id, "limit": limit})
    except MeetingNotStarted:
        raise
    except BackendError:
        return None


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def _confidence_label(confidence: float) -> str:
    """Human phrasing for a confidence value in [0, 1]."""
    if confidence >= 0.75:
        return "High"
    if confidence >= 0.5:
        return "Moderate"
    if confidence >= 0.25:
        return "Low"
    return "Very low"


def render_candidate(candidate: dict, participant_count: int) -> None:
    """Render the headline candidate card: name, confidence bar, reasons."""
    st.subheader("Current candidate")

    name = candidate.get("display_name") or candidate.get("candidate_id")
    confidence = float(candidate.get("confidence") or 0.0)
    ambiguous = bool(candidate.get("ambiguous"))
    reasons = candidate.get("reasons") or []

    if not name:
        st.info("No candidate identified yet. Waiting for meeting signals to arrive...")
        return

    with st.container(border=True):
        cols = st.columns([2, 1, 1])
        cols[0].metric("Most likely candidate", name)
        cols[1].metric("Confidence", f"{confidence * 100:.0f}%", _confidence_label(confidence))
        cols[2].metric("Participants", participant_count)

        st.progress(min(max(confidence, 0.0), 1.0))

        if ambiguous:
            st.warning(
                "Ambiguous: the top two participants are scoring too closely to "
                "call confidently. Treat this pick as tentative.",
                icon=":material/warning:",
            )

        if reasons:
            st.markdown("**Why this pick**")
            for reason in reasons:
                st.markdown(f"- {reason}")
        else:
            st.caption("No positive signals recorded yet.")


def _chip(label: str, active: bool, on_text: str, off_text: str) -> str:
    """Return a small colored markdown badge for a boolean attribute."""
    if active:
        return f":green-background[{label}: {on_text}]"
    return f":gray-background[{label}: {off_text}]"


def render_participants(participants: list[dict]) -> None:
    """Render every participant ranked, with per-participant confidence + reasons."""
    st.subheader("Participant ranking")

    if not participants:
        st.caption("No participants yet.")
        return

    for rank, p in enumerate(participants, start=1):
        confidence = float(p.get("confidence") or 0.0)
        is_candidate = bool(p.get("is_candidate"))
        name = p.get("display_name") or p.get("id")

        with st.container(border=True):
            header = st.columns([3, 1])
            badge = " :green-background[CANDIDATE]" if is_candidate else ""
            header[0].markdown(f"**{rank}. {name}**{badge}")
            header[1].markdown(f"score `{float(p.get('score') or 0.0):.3f}`")

            st.progress(
                min(max(confidence, 0.0), 1.0),
                text=f"Confidence {confidence * 100:.0f}%",
            )

            chips = " &nbsp; ".join(
                [
                    _chip("Presence", bool(p.get("is_present")), "in call", "left"),
                    _chip("Camera", bool(p.get("camera_on")), "on", "off"),
                    f":blue-background[Spoke: {float(p.get('speaking_seconds') or 0.0):.0f}s]",
                ]
            )
            st.markdown(chips, unsafe_allow_html=False)

            reasons = p.get("reasons") or []
            if reasons:
                with st.expander("Signals", expanded=is_candidate):
                    for reason in reasons:
                        st.markdown(f"- {reason}")


def render_transcript(transcript: Optional[dict]) -> None:
    """Render the recent transcript, highlighting the candidate's lines."""
    st.subheader("Recent transcript")

    if transcript is None:
        st.caption(
            "Transcript endpoint unavailable on this backend. "
            "Upgrade the backend to enable the live transcript."
        )
        return

    lines = transcript.get("lines") or []
    if not lines:
        st.caption("No transcript captured yet.")
        return

    for line in lines:
        speaker = line.get("speaker") or line.get("participant_id")
        text = line.get("text") or ""
        with st.chat_message(name=str(speaker)):
            suffix = " _(candidate)_" if line.get("is_candidate") else ""
            st.markdown(f"**{speaker}**{suffix}")
            st.write(text)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(
        page_title="Sherlock - Candidate Detector",
        page_icon=":material/person_search:",
        layout="wide",
    )

    st.title("Sherlock - Candidate Detector")
    st.caption(
        "Real-time detection of the interview candidate by fusing many weak "
        "signals into a self-explaining, continuously-updated confidence score."
    )

    # --- Sidebar controls ---
    with st.sidebar:
        st.header("Settings")
        base_url = st.text_input("Backend URL", value=DEFAULT_API_URL)
        meeting_id = st.text_input("Meeting ID", value=DEFAULT_MEETING_ID)
        transcript_limit = st.slider("Transcript lines", 5, 100, 30, step=5)

        st.divider()
        auto_refresh = st.toggle("Auto-refresh", value=True)
        refresh_interval = st.slider(
            "Refresh every (s)", 1, 30, 3, disabled=not auto_refresh
        )
        manual = st.button("Refresh now", use_container_width=True)

        st.divider()
        try:
            health = fetch_health(base_url)
            st.success(
                f"Backend online - {health.get('pending_events', 0)} events pending",
                icon=":material/check_circle:",
            )
        except BackendError as exc:
            st.error(str(exc), icon=":material/error:")

    if manual:
        st.rerun()

    # --- Fetch verdict ---
    try:
        candidate = fetch_candidate(base_url, meeting_id)
        participants_resp = fetch_participants(base_url, meeting_id)
        transcript = fetch_transcript(base_url, meeting_id, transcript_limit)
    except MeetingNotStarted:
        st.info(
            f"Meeting `{meeting_id}` hasn't started yet. Start it via the API "
            "(`POST /start`) or run the simulator, then this dashboard will "
            "populate automatically.",
            icon=":material/hourglass_empty:",
        )
        _maybe_autorefresh(auto_refresh, refresh_interval)
        return
    except BackendError as exc:
        st.error(
            f"Cannot load the dashboard: {exc}\n\nCheck the backend URL in the "
            "sidebar and that the API is running.",
            icon=":material/error:",
        )
        _maybe_autorefresh(auto_refresh, refresh_interval)
        return

    participants = participants_resp.get("participants") or []

    render_candidate(candidate, participant_count=len(participants))
    st.divider()

    left, right = st.columns([3, 2], gap="large")
    with left:
        render_participants(participants)
    with right:
        render_transcript(transcript)

    _maybe_autorefresh(auto_refresh, refresh_interval)


def _maybe_autorefresh(enabled: bool, interval: int) -> None:
    """Re-run the script after ``interval`` seconds to poll the backend again.

    This simple sleep-then-rerun loop keeps the dashboard live without any extra
    dependency; the sidebar toggle lets the user pause it.
    """
    if enabled:
        time.sleep(interval)
        st.rerun()


if __name__ == "__main__":
    main()
