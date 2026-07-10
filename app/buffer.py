"""In-memory event buffer with an asyncio periodic + priority flush loop.

Events arrive from the simulator via ``POST /events`` in batches. Recomputing
the signal engine on every HTTP request would be wasteful, so we stage incoming
deltas in an in-memory buffer and drain ("flush") it:

  * periodically, every ``flush_interval`` seconds (default 5s), and
  * immediately whenever a high-priority JOIN/LEAVE event arrives, because a
    change in who is *in* the meeting should be reflected in the verdict without
    waiting for the next tick.

A flush = apply buffered deltas to :mod:`app.meeting_state` then recompute the
verdict for each touched meeting. The recompute step is *injected* (defaulting
to a no-op) rather than imported directly, so this module stays import-safe and
unit-testable on its own -- even before ``engine.py`` exists. Wiring is done
once at startup, e.g. ``buffer.recompute = engine.recompute``.

Concurrency note: FastAPI request handlers and the background flush task all run
on the same asyncio event loop, so staging (``add``) needs no lock -- it never
awaits mid-operation. The underlying :class:`MeetingStore` is itself guarded by
a re-entrant lock for the actual state mutation.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Awaitable, Callable, Optional, Union

from .meeting_state import MeetingStore, store
from .models import Event, EventType, Meeting

logger = logging.getLogger(__name__)

DEFAULT_FLUSH_INTERVAL = 5.0

# Events that must not wait for the next periodic tick: membership changes.
_PRIORITY_EVENT_TYPES = {EventType.JOIN.value, EventType.LEAVE.value}

# A recompute hook: given the freshly-updated Meeting, refresh its verdict.
# May be sync or async; the buffer awaits it if it returns an awaitable.
RecomputeFn = Callable[[Meeting], Union[None, Awaitable[None]]]


def _try_config_flush_interval() -> float:
    """Read the flush interval from ``app.config`` if that phase exists yet.

    Falls back to :data:`DEFAULT_FLUSH_INTERVAL` so the buffer works standalone.
    """
    try:
        from .config import settings  # type: ignore
    except Exception:
        return DEFAULT_FLUSH_INTERVAL
    interval = getattr(settings, "flush_interval", None)
    try:
        interval = float(interval)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_FLUSH_INTERVAL
    return interval if interval > 0 else DEFAULT_FLUSH_INTERVAL


class EventBuffer:
    """Stages delta events and drains them on a timer or on priority events.

    The buffer keeps one pending list per ``meeting_id`` so batches for
    different meetings never get mixed. A single :class:`asyncio.Event`
    ("wakeup") lets a priority event short-circuit the periodic sleep, giving an
    effectively-immediate flush without re-entrant flushing from the request
    handler.
    """

    def __init__(
        self,
        meeting_store: Optional[MeetingStore] = None,
        flush_interval: Optional[float] = None,
        recompute: Optional[RecomputeFn] = None,
    ) -> None:
        self._store: MeetingStore = meeting_store or store
        self._flush_interval: float = (
            flush_interval if flush_interval is not None else _try_config_flush_interval()
        )
        # Public so it can be wired late (e.g. `buffer.recompute = engine.recompute`).
        self.recompute: Optional[RecomputeFn] = recompute

        self._pending: dict[str, list[Event]] = {}
        self._wakeup = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # --------------------------------------------------------------------- #
    # Staging
    # --------------------------------------------------------------------- #

    def add(self, meeting_id: str, events: list[Event]) -> None:
        """Stage a batch of events for ``meeting_id``.

        Cheap and synchronous. If any event is high-priority (JOIN/LEAVE) the
        background loop is woken so the flush happens right away instead of at
        the next tick.
        """
        if not events:
            return
        self._pending.setdefault(meeting_id, []).extend(events)
        if any(e.type in _PRIORITY_EVENT_TYPES for e in events):
            self._wakeup.set()

    def pending_count(self) -> int:
        """Total number of events currently staged (across all meetings)."""
        return sum(len(events) for events in self._pending.values())

    # --------------------------------------------------------------------- #
    # Flushing
    # --------------------------------------------------------------------- #

    async def flush(self) -> None:
        """Drain all staged events: apply to state, then recompute verdicts.

        Draining swaps out the pending map atomically (no await before the swap)
        so events added *during* a flush land safely in the next batch.
        """
        if not self._pending:
            return

        pending, self._pending = self._pending, {}

        for meeting_id, events in pending.items():
            if not events:
                continue
            try:
                meeting = self._store.apply_events(meeting_id, events)
            except Exception:
                logger.exception("Failed to apply %d events for meeting %s", len(events), meeting_id)
                continue
            await self._run_recompute(meeting)

    async def _run_recompute(self, meeting: Meeting) -> None:
        """Invoke the injected recompute hook, supporting sync or async impls."""
        if self.recompute is None:
            return
        try:
            result = self.recompute(meeting)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("Recompute failed for meeting %s", meeting.meeting_id)

    # --------------------------------------------------------------------- #
    # Background loop lifecycle (wired to FastAPI lifespan)
    # --------------------------------------------------------------------- #

    async def _run_loop(self) -> None:
        """Periodic flush loop: flush every ``flush_interval`` s or on wakeup."""
        logger.info("Event buffer flush loop started (interval=%.1fs)", self._flush_interval)
        while self._running:
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=self._flush_interval)
            except asyncio.TimeoutError:
                pass  # normal periodic tick
            self._wakeup.clear()
            if not self._running:
                break
            try:
                await self.flush()
            except Exception:
                # Never let the loop die on a bad batch.
                logger.exception("Unexpected error during scheduled flush")
        logger.info("Event buffer flush loop stopped")

    def start(self) -> None:
        """Start the background flush task (idempotent)."""
        if self._task is not None and not self._task.done():
            return
        self._running = True
        self._wakeup.clear()
        self._task = asyncio.create_task(self._run_loop(), name="event-buffer-flush")

    async def stop(self) -> None:
        """Stop the loop and flush any remaining staged events."""
        self._running = False
        self._wakeup.set()  # wake the loop so it can notice _running is False
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        # Final drain so nothing staged at shutdown is silently lost.
        await self.flush()


# Process-wide singleton shared by the API routes and the background flusher.
buffer = EventBuffer()
