"""In-memory meeting state and event application.

This module is the single source of truth for meeting state. It owns a small
in-memory registry of `Meeting` objects (keyed by `meeting_id`) and knows how to:

  * create a meeting from a `/start` payload (`start_meeting`), and
  * fold a stream of delta `Event`s into that state (`apply_event`).

Per the plan, events carry only *deltas*. We accumulate raw facts on each
`Participant` (speaking time, transcript chunks, camera, join order, presence);
the signal engine later recomputes scores as pure functions of this state. That
keeps this file free of scoring logic and free of a whole class of
double-counting bugs.

Event processing / transcript merging (originally its own phase) folds in here:
`apply_event` is the only place raw meeting data mutates.
"""

from datetime import datetime, timezone
from threading import RLock
from typing import Optional

from .models import Event, EventType, Meeting, Participant, StartRequest, TranscriptChunk


def _now() -> datetime:
    """Timezone-aware 'now' used when an event omits its timestamp."""
    return datetime.now(timezone.utc)


class MeetingStore:
    """Thread-safe in-memory registry of meetings.

    A single process-wide instance (`store`) is shared by the API routes and the
    background flush task, so mutations are guarded by a re-entrant lock. All
    reads that iterate participants should also go through the lock via the
    provided helpers to avoid seeing a half-applied batch.
    """

    def __init__(self) -> None:
        self._meetings: dict[str, Meeting] = {}
        self._lock = RLock()

    @property
    def lock(self) -> RLock:
        """Expose the lock so callers (e.g. the flusher) can batch operations."""
        return self._lock

    def start_meeting(self, req: StartRequest) -> Meeting:
        """Create (or reset) a meeting from a `/start` request and register it."""
        with self._lock:
            meeting = Meeting(
                meeting_id=req.meeting_id,
                candidate_name=req.candidate_name,
                candidate_email=req.candidate_email,
                interviewer_names=list(req.interviewer_names),
                interviewer_emails=list(req.interviewer_emails),
                scheduled_start=req.scheduled_start,
                scheduled_end=req.scheduled_end,
                started_at=_now(),
                last_updated=_now(),
            )
            self._meetings[req.meeting_id] = meeting
            return meeting

    def get(self, meeting_id: str) -> Optional[Meeting]:
        """Return a meeting by id, or None if it was never started."""
        with self._lock:
            return self._meetings.get(meeting_id)

    def get_or_create(self, meeting_id: str) -> Meeting:
        """Return an existing meeting, or lazily create a bare one.

        Lets `/events` arrive before `/start` without crashing (graceful
        degradation); the meeting simply has no interview metadata yet.
        """
        with self._lock:
            meeting = self._meetings.get(meeting_id)
            if meeting is None:
                meeting = Meeting(
                    meeting_id=meeting_id,
                    started_at=_now(),
                    last_updated=_now(),
                )
                self._meetings[meeting_id] = meeting
            return meeting

    def apply_events(self, meeting_id: str, events: list[Event]) -> Meeting:
        """Apply a batch of events to a meeting atomically."""
        with self._lock:
            meeting = self.get_or_create(meeting_id)
            for event in events:
                apply_event(meeting, event)
            return meeting

    def reset(self) -> None:
        """Drop all meetings (used by tests)."""
        with self._lock:
            self._meetings.clear()


def _get_or_add_participant(meeting: Meeting, participant_id: str) -> Participant:
    """Fetch a participant, creating a bare record if it's the first we've seen.

    Assigning `join_order` on first sight (rather than only on JOIN) means an
    out-of-order SPEAKING/TRANSCRIPT event for an unknown participant still
    produces a sane record instead of being dropped.
    """
    participant = meeting.participants.get(participant_id)
    if participant is None:
        participant = Participant(
            id=participant_id,
            join_order=len(meeting.participants),
        )
        meeting.participants[participant_id] = participant
    return participant


def apply_event(meeting: Meeting, event: Event) -> None:
    """Fold a single delta `Event` into `meeting` state.

    Idempotency is not assumed; events are treated as deltas (e.g. SPEAKING adds
    seconds). Unknown participants are created on demand so late/reordered events
    degrade gracefully.
    """
    ts = event.timestamp or _now()
    participant = _get_or_add_participant(meeting, event.participant_id)

    # `use_enum_values=True` on Event means `event.type` is already a str.
    event_type = event.type

    if event_type == EventType.JOIN.value:
        participant.is_present = True
        participant.left_at = None
        if participant.joined_at is None:
            participant.joined_at = ts
        if event.display_name:
            participant.display_name = event.display_name
        if event.email:
            participant.email = event.email

    elif event_type == EventType.LEAVE.value:
        participant.is_present = False
        participant.left_at = ts

    elif event_type == EventType.RENAME.value:
        # Renames are common (device name -> real name); keep the latest.
        if event.display_name:
            participant.display_name = event.display_name
        if event.email:
            participant.email = event.email

    elif event_type == EventType.CAMERA.value:
        if event.camera_on is not None:
            participant.camera_on = event.camera_on

    elif event_type == EventType.SCREEN_SHARE.value:
        if event.screen_sharing is not None:
            participant.screen_sharing = event.screen_sharing

    elif event_type == EventType.SPEAKING.value:
        if event.duration_seconds:
            participant.speaking_seconds += event.duration_seconds
            participant.speaking_turns += 1

    elif event_type == EventType.TRANSCRIPT.value:
        # Transcript merging: accumulate non-empty chunks in arrival order.
        # `Participant.full_transcript` joins them for the signals; we also log
        # the chunk at meeting level so the dashboard can show a chronological,
        # cross-speaker "recent transcript".
        if event.text and event.text.strip():
            text = event.text.strip()
            participant.transcript.append(text)
            meeting.transcript_log.append(
                TranscriptChunk(participant_id=event.participant_id, text=text, timestamp=ts)
            )

    # Unknown/unspecified types are ignored on purpose (forward-compatible).

    meeting.last_updated = ts


# Process-wide singleton used by the API and the background flusher.
store = MeetingStore()
