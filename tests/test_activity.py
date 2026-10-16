"""The activity log that drives the status bar."""

from __future__ import annotations

from trade_bot.activity import ActivityLog, run_job


def test_events_carry_increasing_ids_so_clients_can_resume():
    log = ActivityLog()
    first = log.emit("sync", "one")
    second = log.emit("sync", "two")
    assert second.id > first.id
    assert [event.message for event in log.since(first.id)] == ["two"]
    assert log.since(second.id) == []


def test_ring_buffer_bounds_memory():
    log = ActivityLog(max_events=5)
    for index in range(20):
        log.emit("sync", str(index))
    recent = log.recent(50)
    assert len(recent) == 5
    assert recent[-1].message == "19"


def test_status_reports_the_running_job_with_progress():
    log = ActivityLog()
    log.start_job("refresh", "sync")
    log.update_job("refresh", message="parsing filing 3/10", current=3, total=10)
    status = log.status()
    assert status["busy"] is True
    assert status["message"] == "parsing filing 3/10"
    assert status["percent"] == 30


def test_status_falls_back_to_the_last_event_when_idle():
    log = ActivityLog()
    log.emit("app", "ready", level="success")
    status = log.status()
    assert status["busy"] is False
    assert status["message"] == "ready"


def test_run_job_records_success_and_result():
    log = ActivityLog()
    result = run_job(log, "work", "sync", lambda reporter: {"added": 3})
    assert result == {"ok": True, "added": 3}
    job = log.job("work")
    assert job.status == "done" and job.result == {"added": 3}


def test_a_failing_job_is_captured_rather_than_raised():
    """A dead sync must not take down the dashboard displaying it."""
    log = ActivityLog()

    def _boom(reporter):
        raise RuntimeError("provider exploded")

    result = run_job(log, "work", "sync", _boom)
    assert result["ok"] is False
    assert "provider exploded" in result["error"]
    job = log.job("work")
    assert job.status == "error"
    assert any(event.level == "error" for event in log.recent())


def test_is_running_guards_against_double_starts():
    log = ActivityLog()
    log.start_job("refresh", "sync")
    assert log.is_running("refresh")
    log.finish_job("refresh")
    assert not log.is_running("refresh")


def test_reporter_updates_flow_into_status():
    log = ActivityLog()

    def _work(reporter):
        reporter.step("halfway", 5, 10)
        assert log.status()["percent"] == 50
        return {}

    run_job(log, "work", "sync", _work)


def test_event_dict_includes_a_percent_for_the_progress_bar():
    log = ActivityLog()
    event = log.emit("sync", "x", current=1, total=4)
    assert event.as_dict()["percent"] == 25
    assert log.emit("sync", "y").as_dict()["percent"] is None
