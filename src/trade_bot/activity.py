"""Live activity log and background job runner.

Every slow thing this application does -- downloading filings, parsing PTR
PDFs, pulling prices, reading news, scoring members -- reports through here,
so the dashboard's status bar can say what is happening *at this moment*
rather than showing a spinner.

Deliberately built on a lock-protected ring buffer with monotonically
increasing event ids rather than asyncio queues: jobs run in worker threads
via `asyncio.to_thread`, and a plain buffer is safe to write from any thread
without marshalling back onto the event loop. The SSE endpoint polls for ids
above the last one it sent, which makes reconnects trivially correct -- a
client just resumes from the id it already has.
"""

from __future__ import annotations

import threading
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

Level = Literal["info", "success", "warn", "error"]

MAX_EVENTS = 500


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ActivityEvent:
    id: int
    at: datetime
    channel: str
    message: str
    level: Level = "info"
    detail: str | None = None
    current: int | None = None
    total: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "at": self.at.isoformat(),
            "channel": self.channel,
            "message": self.message,
            "level": self.level,
            "detail": self.detail,
            "current": self.current,
            "total": self.total,
            "percent": (
                round(100 * self.current / self.total)
                if self.current is not None and self.total
                else None
            ),
        }


@dataclass
class JobState:
    name: str
    channel: str
    status: str = "running"  # running | done | error
    message: str = ""
    current: int = 0
    total: int = 0
    started_at: datetime = field(default_factory=_now)
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        elapsed = ((self.finished_at or _now()) - self.started_at).total_seconds()
        return {
            "name": self.name,
            "channel": self.channel,
            "status": self.status,
            "message": self.message,
            "current": self.current,
            "total": self.total,
            "percent": round(100 * self.current / self.total) if self.total else None,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "elapsed_seconds": round(elapsed, 1),
            "result": self.result,
            "error": self.error,
        }


class ActivityLog:
    """Thread-safe ring buffer of activity events plus live job registry."""

    def __init__(self, max_events: int = MAX_EVENTS) -> None:
        self._lock = threading.Lock()
        self._events: deque[ActivityEvent] = deque(maxlen=max_events)
        self._jobs: dict[str, JobState] = {}
        self._next_id = 1

    def emit(
        self,
        channel: str,
        message: str,
        level: Level = "info",
        detail: str | None = None,
        current: int | None = None,
        total: int | None = None,
    ) -> ActivityEvent:
        with self._lock:
            event = ActivityEvent(
                id=self._next_id,
                at=_now(),
                channel=channel,
                message=message,
                level=level,
                detail=detail,
                current=current,
                total=total,
            )
            self._next_id += 1
            self._events.append(event)
            return event

    def since(self, event_id: int, limit: int = 200) -> list[ActivityEvent]:
        with self._lock:
            return [event for event in self._events if event.id > event_id][-limit:]

    def recent(self, limit: int = 60) -> list[ActivityEvent]:
        with self._lock:
            return list(self._events)[-limit:]

    def last_id(self) -> int:
        with self._lock:
            return self._next_id - 1

    # -- jobs --------------------------------------------------------------

    def start_job(self, name: str, channel: str, message: str = "") -> JobState:
        with self._lock:
            job = JobState(name=name, channel=channel, message=message or f"{name} started")
            self._jobs[name] = job
        self.emit(channel, job.message, level="info")
        return job

    def update_job(
        self,
        name: str,
        message: str | None = None,
        current: int | None = None,
        total: int | None = None,
        emit: bool = True,
    ) -> None:
        with self._lock:
            job = self._jobs.get(name)
            if job is None:
                return
            if message is not None:
                job.message = message
            if current is not None:
                job.current = current
            if total is not None:
                job.total = total
            channel, snapshot = job.channel, (job.message, job.current, job.total)
        if emit:
            self.emit(channel, snapshot[0], current=snapshot[1], total=snapshot[2])

    def finish_job(
        self,
        name: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(name)
            if job is None:
                return
            job.status = "error" if error else "done"
            job.finished_at = _now()
            job.result = result
            job.error = error
            if message:
                job.message = message
            elif error:
                job.message = f"{name} failed: {error}"
            else:
                job.message = f"{name} finished"
            channel, text = job.channel, job.message
        self.emit(channel, text, level="error" if error else "success")

    def job(self, name: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(name)

    def jobs(self) -> list[JobState]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.started_at, reverse=True)

    def running_jobs(self) -> list[JobState]:
        return [job for job in self.jobs() if job.status == "running"]

    def is_running(self, name: str) -> bool:
        job = self.job(name)
        return job is not None and job.status == "running"

    def status(self) -> dict[str, Any]:
        """What the status bar shows: the live job, or the last thing finished."""
        running = self.running_jobs()
        if running:
            job = running[0]
            return {
                "busy": True,
                "channel": job.channel,
                "message": job.message,
                "current": job.current,
                "total": job.total,
                "percent": round(100 * job.current / job.total) if job.total else None,
                "elapsed_seconds": round((_now() - job.started_at).total_seconds(), 1),
                "running_count": len(running),
            }
        recent = self.recent(1)
        if recent:
            event = recent[0]
            return {
                "busy": False,
                "channel": event.channel,
                "message": event.message,
                "current": None,
                "total": None,
                "percent": None,
                "at": event.at.isoformat(),
                "running_count": 0,
            }
        return {"busy": False, "channel": "idle", "message": "idle", "running_count": 0}


class JobReporter:
    """Handle passed into long-running work so it can narrate itself.

    Keeps callers from having to thread both the log and their job name
    through every helper: `report.step("parsing filing 12/80", 12, 80)`.
    """

    def __init__(self, log: ActivityLog, name: str, channel: str) -> None:
        self.log = log
        self.name = name
        self.channel = channel

    def step(self, message: str, current: int | None = None, total: int | None = None) -> None:
        self.log.update_job(self.name, message=message, current=current, total=total)

    def note(self, message: str, level: Level = "info", detail: str | None = None) -> None:
        self.log.emit(self.channel, message, level=level, detail=detail)


class NullReporter(JobReporter):
    """A reporter that swallows everything, for CLI paths and tests."""

    def __init__(self) -> None:
        super().__init__(ActivityLog(max_events=1), "null", "null")


def run_job(
    log: ActivityLog,
    name: str,
    channel: str,
    work: Callable[[JobReporter], dict[str, Any]],
) -> dict[str, Any]:
    """Run `work` synchronously as a tracked job, capturing failures.

    Intended to be handed to `asyncio.to_thread` by the web layer, or called
    directly by the CLI. Exceptions become an error-status job plus an
    `error` event rather than propagating, because a failed sync must never
    take down the dashboard that is displaying it.
    """
    log.start_job(name, channel, message=f"{name} started")
    reporter = JobReporter(log, name, channel)
    try:
        result = work(reporter)
    except Exception as exc:
        detail = traceback.format_exc(limit=3)
        log.emit(channel, f"{name} failed", level="error", detail=detail)
        log.finish_job(name, error=f"{type(exc).__name__}: {exc}")
        return {"ok": False, "error": str(exc)}
    log.finish_job(name, result=result)
    return {"ok": True, **result}


ACTIVITY = ActivityLog()
