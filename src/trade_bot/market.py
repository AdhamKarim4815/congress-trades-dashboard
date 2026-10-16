"""Exchange session state for the dashboard header.

Uses the real NYSE calendar rather than a day-of-week and clock guess, so
holidays and early closes (the day after Thanksgiving, Christmas Eve) report
correctly instead of claiming the market is open.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import pandas as pd
import pandas_market_calendars as mcal

_NYSE = mcal.get_calendar("NYSE")


def market_status(now: datetime | None = None) -> dict[str, Any]:
    moment = pd.Timestamp(now or datetime.now(UTC))
    moment = moment.tz_localize("UTC") if moment.tzinfo is None else moment.tz_convert("UTC")
    eastern_date = moment.tz_convert("America/New_York").date()
    schedule = _NYSE.schedule(start_date=eastern_date, end_date=eastern_date)
    if schedule.empty:
        return {
            "open": False,
            "reason": "not a trading day",
            "as_of": moment.isoformat(),
        }
    market_open = schedule.iloc[0]["market_open"]
    market_close = schedule.iloc[0]["market_close"]
    is_open = bool(market_open <= moment <= market_close)
    if is_open:
        reason = "regular session"
    elif moment < market_open:
        reason = "pre-market"
    else:
        reason = "closed"
    return {
        "open": is_open,
        "reason": reason,
        "market_open": market_open.isoformat(),
        "market_close": market_close.isoformat(),
        "as_of": moment.isoformat(),
    }


def last_completed_session(now: datetime | None = None) -> date:
    """The most recent trading day whose closing price is final.

    A price cache is current when it holds this date. Comparing against it
    beats asking how old a file is: on a Saturday every file is stale by any
    clock, but there is no new close to fetch, and re-reading the same series
    for two thousand symbols is work that produces byte-identical data.

    Today only counts once the bell has rung. Before then the last final close
    is the previous session's.
    """
    moment = pd.Timestamp(now or datetime.now(UTC))
    moment = moment.tz_localize("UTC") if moment.tzinfo is None else moment.tz_convert("UTC")
    eastern = moment.tz_convert("America/New_York").date()
    # Two weeks back covers the longest run of closures the NYSE observes.
    schedule = _NYSE.schedule(start_date=eastern - timedelta(days=14), end_date=eastern)
    if schedule.empty:
        return eastern
    for _index, row in schedule.iloc[::-1].iterrows():
        if moment >= row["market_close"]:
            return cast(date, row["market_close"].tz_convert("America/New_York").date())
    return cast(date, schedule.iloc[0]["market_close"].tz_convert("America/New_York").date())
