"""install / on / off / update mean the same here as in zyx and agent-office.

install sets up and starts nothing; on and off both last across a restart; update never switches
on a listener that is off. "start" below is the login agent being (re)registered and loaded.
"""

from pathlib import Path

from murmurflow import cli


def _machine(monkeypatch, *, running, updated=True):
    calls = []
    monkeypatch.delenv(cli._RESYNCED, raising=False)
    monkeypatch.setattr(cli, "_receipt", lambda: Path("uv-receipt.toml"))
    monkeypatch.setattr(cli, "_update", lambda: calls.append("update") or updated)
    monkeypatch.setattr(cli, "_ready", lambda: True)
    monkeypatch.setattr(cli, "_tcc_entry", lambda: "MurmurFlow")
    monkeypatch.setattr(cli.dictate, "start", lambda: None)  # no microphone in a test
    monkeypatch.setattr(cli.dictate, "stop_server", lambda: False)
    monkeypatch.setattr(cli.dictate, "rival_listeners", lambda: [])
    monkeypatch.setattr(cli.dictate, "trigger_hint", lambda: "Double-tap Control")
    monkeypatch.setattr(cli.service, "is_macos", lambda: False)  # never open System Settings
    monkeypatch.setattr(cli.service, "running", lambda: running)
    monkeypatch.setattr(cli.service, "install", lambda: calls.append("start") or (True, ""))
    monkeypatch.setattr(cli.service, "uninstall", lambda: calls.append("stop") or (True, ""))
    monkeypatch.setattr(cli.service, "remove_identity", lambda: calls.append("remove-app") or True)
    return calls


def test_install_never_switches_dictation_on(monkeypatch):
    calls = _machine(monkeypatch, running=False)
    assert cli.main(["install"]) == 0
    assert "start" not in calls


def test_install_over_a_running_listener_brings_it_back_on_the_new_code(monkeypatch):
    """`git pull && murmurflow install` was the whole update, and must stay one."""
    calls = _machine(monkeypatch, running=True)
    assert cli.main(["install"]) == 0
    assert calls == ["update", "start"]


def test_on_updates_then_starts(monkeypatch):
    calls = _machine(monkeypatch, running=False)
    assert cli.main(["on"]) == 0
    assert calls == ["update", "start"]


def test_off_stops_it_and_keeps_the_app(monkeypatch):
    calls = _machine(monkeypatch, running=True)
    assert cli.main(["off"]) == 0
    assert calls == ["stop"]


def test_uninstall_is_off_plus_the_app(monkeypatch):
    calls = _machine(monkeypatch, running=True)
    assert cli.main(["uninstall"]) == 0
    assert calls == ["stop", "remove-app"]


def test_update_leaves_an_off_listener_off(monkeypatch):
    calls = _machine(monkeypatch, running=False)
    assert cli.main(["update"]) == 0
    assert calls == ["update"]


def test_update_restarts_a_listener_that_is_on(monkeypatch):
    calls = _machine(monkeypatch, running=True)
    assert cli.main(["update"]) == 0
    assert calls == ["update", "start"]


def test_an_update_that_failed_says_so_and_restarts_nothing(monkeypatch):
    calls = _machine(monkeypatch, running=True, updated=False)
    assert cli.main(["update"]) == 1
    assert calls == ["update"]


def test_update_from_a_copy_uv_did_not_install_refuses_rather_than_pretends(monkeypatch):
    calls = _machine(monkeypatch, running=True)
    monkeypatch.setattr(cli, "_receipt", lambda: None)
    assert cli.main(["update"]) == 1
    assert calls == []
