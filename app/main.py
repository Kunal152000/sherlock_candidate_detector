"""FastAPI application: the public REST surface + background flush wiring.

This module is the composition root. It owns nothing itself; it simply wires the
already-built pieces together and exposes them over HTTP:

  * :data:`app.meeting_state.store` -- in-memory meeting registry (source of truth).
  * :data:`app.buffer.buffer` -- stages incoming delta events and drains them on a
    timer (or immediately on JOIN/LEAVE).
  * :func:`app.engine.recompute` -- fuses the weak signals into a verdict; injected
    into the buffer as its recompute hook so a fresh verdict is produced after
    every flush.

Endpoints (interactive Swagger UI auto-generated at ``/docs``):

  * ``POST /start``              -- initialise a meeting with interview metadata.
  * ``POST /events``             -- submit a batch of delta events (buffered).
  * ``GET  /current_candidate``  -- the current best guess + confidence + reasons.
  * ``GET  /participants``       -- every participant, ranked, with scores/reasons.

Lifecycle: a single asyncio background task (started/stopped via FastAPI's
``lifespan``) runs the periodic flush loop, so no threads, queues, or brokers are
needed -- everything shares one event loop.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from . import engine
from .buffer import buffer
from .meeting_state import store
from .models import (
    CandidateResponse,
    EventsRequest,
    Meeting,
    ParticipantsResponse,
    ParticipantView,
    StartRequest,
    TranscriptLine,
    TranscriptResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Start/stop the background flush loop around the app's lifetime.

    On startup we inject the engine's recompute into the buffer (so verdicts are
    refreshed after each flush) and start the periodic flush task. On shutdown we
    stop the loop, which performs a final drain so nothing staged is lost.
    """
    buffer.recompute = engine.recompute
    buffer.start()
    logger.info("Candidate detector started; background flush loop running")
    try:
        yield
    finally:
        await buffer.stop()
        logger.info("Candidate detector stopped; buffer drained")


app = FastAPI(
    title="Sherlock Candidate Detector",
    description=(
        "Real-time detection of the interview candidate in a live meeting by "
        "fusing many weak signals (name match, self-introduction, conversation "
        "role, speaking time, camera, join order, and an LLM read). Returns a "
        "continuously-updated, self-explaining confidence score that degrades "
        "gracefully under bad or missing data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _require_meeting(meeting_id: str) -> Meeting:
    """Fetch a started meeting or raise 404 (keeps the read endpoints tidy)."""
    meeting = store.get(meeting_id)
    if meeting is None:
        raise HTTPException(
            status_code=404,
            detail=f"No meeting '{meeting_id}'. Call POST /start (or POST /events) first.",
        )
    return meeting


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.post("/start", response_model=Meeting, tags=["meeting"])
def start_meeting(req: StartRequest) -> Meeting:
    """Initialise (or reset) a meeting with the known interview metadata.

    Safe to call before any events arrive. Returns the freshly-created meeting,
    including its (empty) participant list, so clients can confirm the setup.
    """
    return store.start_meeting(req)


@app.post("/events", tags=["meeting"])
def submit_events(req: EventsRequest) -> dict:
    """Stage a batch of delta events for later flushing.

    Events are buffered rather than applied inline: the background loop drains
    them every few seconds, and immediately on membership changes (JOIN/LEAVE).
    Responds right away with how many events were accepted and how many are now
    pending, so the caller never blocks on scoring.
    """
    buffer.add(req.meeting_id, req.events)
    return {
        "meeting_id": req.meeting_id,
        "accepted": len(req.events),
        "pending": buffer.pending_count(),
    }


@app.get("/current_candidate", response_model=CandidateResponse, tags=["verdict"])
def current_candidate(
    meeting_id: str = Query("default", description="Meeting identifier."),
) -> CandidateResponse:
    """Return the current best-guess candidate with confidence and reasons.

    Reflects the most recent flush. When the top two participants score too
    closely to call, ``ambiguous`` is set so callers can surface the uncertainty
    instead of a false-confident pick.
    """
    meeting = _require_meeting(meeting_id)
    with store.lock:
        candidate_id = meeting.candidate_id
        if candidate_id is None:
            return CandidateResponse(
                candidate_id=None,
                confidence=0.0,
                reasons=[],
                ambiguous=meeting.ambiguous,
            )
        winner = meeting.participants.get(candidate_id)
        return CandidateResponse(
            candidate_id=candidate_id,
            display_name=winner.display_name if winner else None,
            confidence=meeting.confidence,
            reasons=list(winner.reasons) if winner else [],
            ambiguous=meeting.ambiguous,
        )


@app.get("/participants", response_model=ParticipantsResponse, tags=["verdict"])
def list_participants(
    meeting_id: str = Query("default", description="Meeting identifier."),
) -> ParticipantsResponse:
    """Return every participant ranked by score, with per-participant reasons.

    Useful for the dashboard and for debugging why one participant beat another:
    each entry carries its fused score, normalised confidence, and the positive
    signal reasons behind it.
    """
    meeting = _require_meeting(meeting_id)
    with store.lock:
        ranked = engine.rank_participants(meeting)
        views = [
            ParticipantView(
                id=p.id,
                display_name=p.display_name,
                score=p.score,
                confidence=p.confidence,
                reasons=list(p.reasons),
                is_candidate=p.is_candidate,
                is_present=p.is_present,
                camera_on=p.camera_on,
                speaking_seconds=p.speaking_seconds,
            )
            for p in ranked
        ]
    return ParticipantsResponse(meeting_id=meeting.meeting_id, participants=views)


@app.get("/transcript", response_model=TranscriptResponse, tags=["verdict"])
def get_transcript(
    meeting_id: str = Query("default", description="Meeting identifier."),
    limit: int = Query(
        30, ge=1, le=500, description="Return at most this many of the most recent lines."
    ),
) -> TranscriptResponse:
    """Return the most recent transcript lines with speaker attribution.

    Lines are returned oldest-first (chat order) and carry the speaker's *current*
    display name plus whether that speaker is the current best-guess candidate, so
    the dashboard can highlight the candidate's utterances. Powers the live
    "recent transcript" panel.
    """
    meeting = _require_meeting(meeting_id)
    lines: list[TranscriptLine] = []
    with store.lock:
        for chunk in meeting.transcript_log[-limit:]:
            speaker = meeting.participants.get(chunk.participant_id)
            lines.append(
                TranscriptLine(
                    participant_id=chunk.participant_id,
                    speaker=(
                        speaker.display_name
                        if speaker and speaker.display_name
                        else chunk.participant_id
                    ),
                    text=chunk.text,
                    timestamp=chunk.timestamp,
                    is_candidate=bool(speaker and speaker.is_candidate),
                )
            )
    return TranscriptResponse(meeting_id=meeting.meeting_id, lines=lines)


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness probe: confirms the app is up and reports buffer backlog."""
    return {"status": "ok", "pending_events": buffer.pending_count()}
