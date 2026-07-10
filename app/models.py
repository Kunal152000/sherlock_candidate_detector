"""Pydantic models for the candidate detector.

Design note: the engine recomputes signals from *accumulated state* on every
flush (see plan). So `Participant` stores raw facts (speaking time, transcript,
camera, join order) plus the latest computed scoring output. `Event` objects
carry only deltas that get applied to that state.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class EventType(str, Enum):
    """Kinds of meeting events the simulator can emit."""

    JOIN = "join"
    LEAVE = "leave"
    RENAME = "rename"
    CAMERA = "camera"
    SCREEN_SHARE = "screen_share"
    SPEAKING = "speaking"
    TRANSCRIPT = "transcript"


class Event(BaseModel):
    """A single meeting event (a delta applied to meeting state).

    Only the fields relevant to a given `type` need to be populated:
      - JOIN / RENAME: `display_name` (and optionally `email`)
      - CAMERA:        `camera_on`
      - SCREEN_SHARE:  `screen_sharing`
      - SPEAKING:      `duration_seconds`
      - TRANSCRIPT:    `text`
      - LEAVE:         (no extra fields)
    """

    model_config = ConfigDict(use_enum_values=True)

    type: EventType
    participant_id: str = Field(..., description="Stable id of the participant.")
    timestamp: Optional[datetime] = Field(
        default=None, description="When the event occurred; defaults to receipt time."
    )

    # Identity (join / rename)
    display_name: Optional[str] = Field(
        default=None, description="Shown name, e.g. 'MacBook Pro' or 'Rahul Sharma'."
    )
    email: Optional[str] = Field(default=None, description="Participant email, if known.")

    # State toggles
    camera_on: Optional[bool] = Field(default=None, description="Camera state (CAMERA).")
    screen_sharing: Optional[bool] = Field(
        default=None, description="Screen-share state (SCREEN_SHARE)."
    )

    # Speaking
    duration_seconds: Optional[float] = Field(
        default=None, ge=0, description="Seconds spoken in this chunk (SPEAKING)."
    )

    # Transcript
    text: Optional[str] = Field(default=None, description="Transcript text (TRANSCRIPT).")


class SignalContribution(BaseModel):
    """One signal's contribution to a participant's score (for explainability)."""

    signal: str = Field(..., description="Signal name, e.g. 'name_signal'.")
    score: float = Field(..., description="Raw signal score for this participant.")
    reason: str = Field(..., description="Human-readable justification.")


class TranscriptChunk(BaseModel):
    """One transcript utterance, kept at meeting level to preserve global order.

    Per-participant transcript lists lose cross-speaker ordering, so we also log
    each chunk here (in arrival order) with its speaker and timestamp. This is
    what powers the dashboard's chronological "recent transcript" view.
    """

    participant_id: str
    text: str
    timestamp: Optional[datetime] = None


class Participant(BaseModel):
    """Accumulated raw state for a participant plus latest scoring output."""

    id: str
    display_name: str = ""
    email: Optional[str] = None

    # --- Raw accumulated facts (inputs to signals) ---
    join_order: Optional[int] = Field(
        default=None, description="0-based order in which this participant first joined."
    )
    joined_at: Optional[datetime] = None
    left_at: Optional[datetime] = None
    is_present: bool = True

    camera_on: bool = False
    screen_sharing: bool = False

    speaking_seconds: float = Field(default=0.0, ge=0)
    speaking_turns: int = Field(default=0, ge=0)
    transcript: list[str] = Field(default_factory=list)

    # --- Latest computed scoring output (recomputed each flush) ---
    score: float = 0.0
    confidence: float = Field(default=0.0, ge=0, le=1)
    contributions: list[SignalContribution] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    is_candidate: bool = False

    @property
    def full_transcript(self) -> str:
        """All transcript chunks joined into one string."""
        return " ".join(self.transcript)


class Meeting(BaseModel):
    """In-memory meeting: interview metadata + participant state + verdict."""

    meeting_id: str = "default"

    # --- Interview metadata (provided at /start) ---
    candidate_name: Optional[str] = None
    candidate_email: Optional[str] = None
    interviewer_names: list[str] = Field(default_factory=list)
    interviewer_emails: list[str] = Field(default_factory=list)
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None

    # --- State ---
    participants: dict[str, Participant] = Field(default_factory=dict)
    transcript_log: list[TranscriptChunk] = Field(
        default_factory=list,
        description="All transcript chunks in arrival order (preserves speaker interleaving).",
    )
    started_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    # --- Latest verdict ---
    candidate_id: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    ambiguous: bool = False


# --------------------------------------------------------------------------- #
# API request / response schemas
# --------------------------------------------------------------------------- #


class StartRequest(BaseModel):
    """Payload for POST /start."""

    meeting_id: str = "default"
    candidate_name: Optional[str] = None
    candidate_email: Optional[str] = None
    interviewer_names: list[str] = Field(default_factory=list)
    interviewer_emails: list[str] = Field(default_factory=list)
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None


class EventsRequest(BaseModel):
    """Payload for POST /events: a batch of events."""

    meeting_id: str = "default"
    events: list[Event] = Field(default_factory=list)


class CandidateResponse(BaseModel):
    """Response for GET /current_candidate."""

    candidate_id: Optional[str] = None
    display_name: Optional[str] = None
    confidence: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    ambiguous: bool = False


class ParticipantView(BaseModel):
    """A single participant in the ranked GET /participants response."""

    id: str
    display_name: str
    score: float
    confidence: float
    reasons: list[str]
    is_candidate: bool
    is_present: bool
    camera_on: bool
    speaking_seconds: float


class ParticipantsResponse(BaseModel):
    """Response for GET /participants: ranked participants."""

    meeting_id: str
    participants: list[ParticipantView] = Field(default_factory=list)


class TranscriptLine(BaseModel):
    """One line of recent transcript with resolved speaker attribution."""

    participant_id: str
    speaker: str = Field(..., description="Current display name (or id) of the speaker.")
    text: str
    timestamp: Optional[datetime] = None
    is_candidate: bool = Field(
        default=False, description="Whether this speaker is the current best-guess candidate."
    )


class TranscriptResponse(BaseModel):
    """Response for GET /transcript: most-recent lines, oldest first."""

    meeting_id: str
    lines: list[TranscriptLine] = Field(default_factory=list)
