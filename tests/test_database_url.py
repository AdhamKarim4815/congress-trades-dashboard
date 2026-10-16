"""Repairing connection strings whose password was pasted raw.

Every case here is a real way a working password produces the misleading
error "failed to resolve host": the URL parser splits on the wrong `@`, or a
copied secret carries whitespace, and the reported host is one nobody typed.
"""

from __future__ import annotations

import pytest

from trade_bot.database import Store, describe_database_url, normalize_database_url

HOST = "pooler.example.com:5432"
USER = "app_role.exampleprojectref"


def test_password_containing_at_sign_keeps_the_real_host():
    """The failure that cost two CI runs.

    Splitting on the first `@` makes `pa` the user and `ss@host` the host.
    """
    url = normalize_database_url(f"postgresql+psycopg://{USER}:pa@ss@{HOST}/postgres")
    assert url.endswith(f"@{HOST}/postgres")
    assert "pa%40ss" in url
    assert describe_database_url(url) == (
        f"host={HOST} user={USER} database=postgres"
    )


def test_other_structural_characters_are_encoded():
    for raw, encoded in [("a/b", "a%2Fb"), ("a#b", "a%23b"), ("a?b", "a%3Fb"), ("a b", "a%20b")]:
        url = normalize_database_url(f"postgresql+psycopg://{USER}:{raw}@{HOST}/postgres")
        assert encoded in url, raw
        assert url.endswith(f"@{HOST}/postgres")


def test_surrounding_whitespace_quotes_and_newlines_are_stripped():
    """A value pasted into a secret box often carries a trailing newline."""
    url = normalize_database_url(f'  "postgresql+psycopg://{USER}:pw@{HOST}/postgres"\n')
    assert url == f"postgresql+psycopg://{USER}:pw@{HOST}/postgres"


def test_an_already_encoded_password_is_left_alone():
    """Re-encoding would turn %40 into %2540 and break a working string."""
    original = f"postgresql+psycopg://{USER}:pa%40ss@{HOST}/postgres"
    assert normalize_database_url(original) == original


def test_a_simple_password_is_unchanged():
    original = f"postgresql+psycopg://{USER}:simplepw@{HOST}/postgres"
    assert normalize_database_url(original) == original


def test_sqlite_urls_pass_through_untouched():
    assert normalize_database_url("sqlite:///var/trade_bot.sqlite3") == (
        "sqlite:///var/trade_bot.sqlite3"
    )


def test_url_without_credentials_is_untouched():
    url = f"postgresql+psycopg://{HOST}/postgres"
    assert normalize_database_url(url) == url


def test_describe_never_reveals_the_password():
    described = describe_database_url(
        f"postgresql+psycopg://{USER}:sup3r-s3cret@{HOST}/postgres"
    )
    assert "sup3r-s3cret" not in described
    assert "s3cret" not in described
    assert HOST in described


def test_store_normalises_on_construction(tmp_path):
    store = Store(f"sqlite:///{tmp_path / 'x.sqlite3'}  ")
    assert store.database_url.endswith("x.sqlite3")
    store.initialize()


@pytest.mark.parametrize("scheme", ["postgresql+psycopg", "postgresql", "postgres"])
def test_every_postgres_scheme_is_handled(scheme):
    url = normalize_database_url(f"{scheme}://{USER}:pa@ss@{HOST}/postgres")
    assert url.startswith(f"{scheme}://")
    assert url.endswith(f"@{HOST}/postgres")
