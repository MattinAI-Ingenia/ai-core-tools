"""Cross-process mutual exclusion for indexing a silo.

LightRAG's asyncio locks are bound to one event loop, so two ingestion runs on
the same silo must never overlap. The previous guard lived in a module-level
dict, which is invisible to the other uvicorn workers: with ``UVICORN_WORKERS=4``
three out of four uploads could not see a run started elsewhere and slipped past.

A PostgreSQL *session-level* advisory lock is shared by every worker and, unlike
a flag column, is released automatically when the connection dies — so a crashed
worker cannot leave a silo blocked forever.

Implementation notes, both learned the hard way:

* it holds a raw ``Connection``, not a ``Session``. A Session returns its
  connection to the pool on ``commit()``, and the next caller checking out that
  same physical connection would re-acquire the lock successfully (advisory locks
  are re-entrant within one connection) — silently defeating the exclusion.
* it commits after acquiring. Otherwise the connection sits "idle in
  transaction" for the whole run — hours on a large batch — blocking DDL on the
  tables that transaction touched and holding back vacuum.
"""

from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

from db.database import engine
from utils.logger import get_logger

logger = get_logger(__name__)

# First key of the two-int advisory lock, namespacing these locks against any
# other advisory lock in the database (e.g. the omniadmin bootstrap).
_NAMESPACE = 0x41494354  # 'AICT'


def acquire(silo_id: int) -> Optional[Connection]:
    """Try to lock *silo_id* for indexing.

    Returns the connection holding the lock — it must stay open for the whole
    run and be passed to :func:`release` — or ``None`` if another run holds it.
    """
    conn = engine.connect()
    try:
        granted = conn.execute(
            text("SELECT pg_try_advisory_lock(:ns, :sid)"),
            {"ns": _NAMESPACE, "sid": silo_id},
        ).scalar()
        if not granted:
            conn.close()
            return None
        # Ends the transaction; the connection stays checked out, so no other
        # caller can reuse it and re-enter the lock.
        conn.commit()
        return conn
    except Exception:
        conn.close()
        raise


def release(conn: Optional[Connection], silo_id: int) -> None:
    """Release the lock and return the connection to the pool."""
    if conn is None:
        return
    try:
        conn.execute(
            text("SELECT pg_advisory_unlock(:ns, :sid)"),
            {"ns": _NAMESPACE, "sid": silo_id},
        )
        conn.commit()
    except Exception as exc:  # noqa: BLE001 - closing the connection frees it anyway
        logger.warning("Could not unlock silo %s explicitly: %s", silo_id, exc)
    finally:
        conn.close()


def is_locked(db: Connection, silo_id: int) -> bool:
    """Whether a run is *alive* for *silo_id* (read-only check).

    Row statuses cannot answer this: a batch killed mid-run leaves its resources
    in 'pending'/'indexing' forever, indistinguishable from a live run. The lock
    disappears when the holder's connection does, so it is the liveness signal
    the UI needs to offer "resume" instead of a frozen progress bar.
    """
    return bool(
        db.execute(
            text(
                "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' "
                "AND classid = :ns AND objid = :sid AND granted)"
            ),
            {"ns": _NAMESPACE, "sid": silo_id},
        ).scalar()
    )
