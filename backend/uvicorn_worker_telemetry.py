"""Log why a uvicorn worker dies when the master respawns it.

uvicorn 0.35's supervisor replaces a dead (or 5s-unresponsive) worker
silently: no exit code, no signal, no distinction between its own
watchdog SIGKILL ("no pong in 5s") and a natural death (segfault, external
SIGKILL). Wrapping ``Process.kill``/``Process.join`` adds one warning line
per death. Runs in the master process — wired by the backend Dockerfile CMD
(``python -m uvicorn_worker_telemetry`` instead of ``python -m uvicorn``).
"""

import logging

logger = logging.getLogger("uvicorn.error")


def install() -> bool:
    """Patch uvicorn's supervisor in-place, idempotently.

    Returns False when the upstream layout changed (e.g. a uvicorn upgrade
    dropped or renamed ``Process.kill``/``join``) — the app then simply runs
    unpatched instead of failing to boot.
    """
    try:
        from uvicorn.supervisors import multiprocess as mp

        if getattr(mp.Process, "_worker_death_telemetry", False):
            return True
        if not (hasattr(mp.Process, "kill") and hasattr(mp.Process, "join")):
            logger.warning("Worker death telemetry: uvicorn Process API changed; not installed")
            return False

        _kill, _join = mp.Process.kill, mp.Process.join

        def kill(self):
            # keep_subprocess_alive() calls kill() even for an already-dead
            # process (no-op). Only a still-running one means the master
            # judged it hung after a 5s ping timeout.
            if self.process.is_alive():
                self.killed_by_master = True
            return _kill(self)

        def join(self):
            _join(self)
            code = self.process.exitcode
            if code in (0, None):
                return  # graceful exits (incl. shutdown-time terminate) stay silent
            detail = f"signal {-code}" if code < 0 else f"code {code}"
            cause = (
                "killed by master watchdog (no pong in 5s)"
                if getattr(self, "killed_by_master", False)
                else "exited on its own"
            )
            logger.warning(
                "Child process [%s] died: %s, %s", self.process.pid, cause, detail
            )

        mp.Process.kill, mp.Process.join = kill, join
        mp.Process._worker_death_telemetry = True
        logger.info("Worker death telemetry installed (exitcode + watchdog kills are now logged)")
        return True
    except Exception as e:  # pragma: no cover
        logger.warning("Worker death telemetry not installed: %s", e)
        return False


if __name__ == "__main__":  # backend CMD entry: same CLI as `python -m uvicorn`
    install()
    from uvicorn.main import main

    main()
