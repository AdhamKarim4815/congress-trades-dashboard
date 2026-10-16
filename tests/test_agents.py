"""The launchd agents that keep the live site up.

Started by hand in a terminal, the dashboard dies when the terminal closes,
with nothing to say why. These templates are what replace that.

The tests are narrow on purpose: nothing here calls `launchctl`, because a
test that installs a background service on the developer's machine is worse
than no test. What they do check is the part that actually broke — a plist
that looks fine and will not parse.
"""

from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

AGENTS = sorted((Path(__file__).resolve().parents[1] / "deploy" / "launchd").glob("*.plist"))
FIELDS = {
    "python": "/opt/venv/bin/python",
    "root": "/Users/someone/trade-bot",
    "cloudflared": "/opt/homebrew/bin/cloudflared",
}


def test_there_are_agents_to_install():
    assert AGENTS, "the serve and tunnel agents should both be committed"
    assert {path.stem for path in AGENTS} == {
        "com.example.copytrade.serve",
        "com.example.copytrade.tunnel",
    }


@pytest.mark.parametrize("path", AGENTS, ids=lambda path: path.stem)
def test_a_filled_agent_is_valid_plist_xml(path: Path):
    """The one that bit: XML comments cannot contain a double hyphen.

    Both files explain themselves at the top, and prose about `--install`
    flags and em-dashes produced a plist that `launchctl` rejects with an
    error naming a column number and nothing else.
    """
    parsed = plistlib.loads(path.read_text().format(**FIELDS).encode())
    assert parsed["Label"] == path.stem
    assert parsed["ProgramArguments"]
    assert parsed["KeepAlive"] is True, "the point of the agent is that it comes back"


@pytest.mark.parametrize("path", AGENTS, ids=lambda path: path.stem)
def test_every_placeholder_is_one_the_installer_supplies(path: Path):
    """A stray brace renders as a literal and launchd fails at run time."""
    path.read_text().format(**FIELDS)


@pytest.mark.parametrize("path", AGENTS, ids=lambda path: path.stem)
def test_a_crash_loop_is_throttled(path: Path):
    """KeepAlive plus a fast crash is a busy loop, not a service."""
    parsed = plistlib.loads(path.read_text().format(**FIELDS).encode())
    assert parsed.get("ThrottleInterval", 0) >= 10


def test_logs_land_under_the_project_rather_than_the_home_directory():
    for path in AGENTS:
        parsed = plistlib.loads(path.read_text().format(**FIELDS).encode())
        assert parsed["StandardErrorPath"].startswith(FIELDS["root"])
