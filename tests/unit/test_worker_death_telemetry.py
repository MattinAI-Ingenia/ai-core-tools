"""Self-check for backend/uvicorn_worker_telemetry.py.

Runs in CI (pytest tests/unit/) and directly:
    python tests/unit/test_worker_death_telemetry.py
"""

import logging
import multiprocessing
import sys
import types
from pathlib import Path

# Direct-run fallback: tests/conftest.py already puts backend/ on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))


def _patched():
    import uvicorn_worker_telemetry as telemetry
    from uvicorn.supervisors import multiprocess as mp

    assert telemetry.install() is True, "install() failed"
    assert telemetry.install() is True, "install() is not idempotent"
    return mp


class _Capture:
    def __init__(self):
        self.messages = []
        self.handler = logging.Handler()
        self.handler.emit = lambda record: self.messages.append(record.getMessage())

    def __enter__(self):
        logging.getLogger("uvicorn.error").addHandler(self.handler)
        return self

    def __exit__(self, *exc):
        logging.getLogger("uvicorn.error").removeHandler(self.handler)


def _fake(code, killed_by_master=None, pid=26):
    process = types.SimpleNamespace(pid=pid, exitcode=code, join=lambda: None)
    fake = types.SimpleNamespace(process=process)
    if killed_by_master is not None:
        fake.killed_by_master = killed_by_master
    return fake


def test_install_is_idempotent_and_patched():
    _patched()


def test_watchdog_kill_is_attributed():
    mp = _patched()
    with _Capture() as cap:
        mp.Process.join(_fake(-9, killed_by_master=True))
    deaths = [m for m in cap.messages if "died" in m]
    assert deaths == ["Child process [26] died: killed by master watchdog (no pong in 5s), signal 9"], deaths


def test_natural_death_is_attributed():
    mp = _patched()
    with _Capture() as cap:
        mp.Process.join(_fake(-11))
    deaths = [m for m in cap.messages if "died" in m]
    assert deaths == ["Child process [26] died: exited on its own, signal 11"], deaths


def test_clean_exit_stays_silent():
    mp = _patched()
    with _Capture() as cap:
        mp.Process.join(_fake(0))
    assert not [m for m in cap.messages if "died" in m], cap.messages


def test_kill_flags_only_live_processes():
    mp = _patched()
    # A real, running child: the master's kill() must flag it as watchdog.
    proc = multiprocessing.Process(target=lambda: __import__("time").sleep(30))
    proc.start()
    fake_alive = types.SimpleNamespace(process=proc)
    try:
        mp.Process.kill(fake_alive)
    except AttributeError:  # the real kill() runs on the real Process, fine
        pass
    assert fake_alive.killed_by_master is True
    proc.join(timeout=5)
    with _Capture() as cap:
        mp.Process.join(fake_alive)
    deaths = [m for m in cap.messages if "died" in m]
    assert deaths == [f"Child process [{proc.pid}] died: killed by master watchdog (no pong in 5s), signal 9"], deaths

    # An already-dead child: kill() must NOT set the flag.
    proc2 = multiprocessing.Process(target=lambda: None)
    proc2.start()
    proc2.join()
    assert proc2.exitcode == 0
    fake_dead = types.SimpleNamespace(process=proc2)
    try:
        mp.Process.kill(fake_dead)
    except (AttributeError, ValueError):
        pass
    assert not getattr(fake_dead, "killed_by_master", False)


if __name__ == "__main__":
    test_install_is_idempotent_and_patched()
    test_watchdog_kill_is_attributed()
    test_natural_death_is_attributed()
    test_clean_exit_stays_silent()
    test_kill_flags_only_live_processes()
    print("worker-death telemetry self-check OK")
