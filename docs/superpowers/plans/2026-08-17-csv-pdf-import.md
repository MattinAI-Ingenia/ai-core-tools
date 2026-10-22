# CSV → PDF Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user upload a CSV of links to a Repository, have the backend download a PDF per unique link, review results, and confirm selected rows into the exact same Resource-creation + indexing pipeline used by manual file upload.

**Architecture:** A new `ImportJob`/`ImportJobRow` pair (modeled after the existing `CrawlJob`) tracks one CSV import per Repository. Creation parses+dedupes the CSV and kicks off a background-thread download loop (same threading shape as `ResourceService._index_resources_background`) using `aiohttp` via the existing `crawl/http_fetcher.fetch()`. The frontend polls job status (same polling shape as `CrawlJobProgressPanel`) to drive a persistent banner and a review table. Confirming rows reuses `ResourceService.create_multiple_resources()` unmodified in its file-handling logic via a small duck-typed file adapter, extended with one new optional parameter to carry CSV metadata onto the created `Resource` rows, which `SiloService.index_resource()` merges into chunk metadata exactly like Domain crawling already does for `{url, domain_id}`. Abandoned jobs are purged by extending the existing `file_cleanup_worker.py` sweep loop with one DB-driven step — no new worker process.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy 2.x, Alembic, pandas (already a dependency, used for CSV parsing), aiohttp (already used for crawling); React 19 + TypeScript, existing `StepperContainer`/`DataTable`/`Modal`/`Alert`/`StatusBadge` UI components.

**Spec:** `docs/superpowers/specs/2026-08-17-csv-pdf-import-design.md`

## Global Constraints

- All new user-facing UI copy (row states, banner text, button labels) is in English — no exceptions.
- A CSV is assumed to have a header row on line 1 — there is no "no header" mode.
- At most **one active `ImportJob` per Repository** at a time. Creating a new import while one is still open (status `DOWNLOADING` or `REVIEW`) returns `409 Conflict`, mirroring the existing `CrawlJobService.enqueue()` / `ConflictError` pattern (`backend/services/crawl_job_service.py`, used in `backend/routers/internal/domains.py:572-595`).
- Rows are deduped by `(url, row_metadata)` **before** download — identical link+metadata pairs collapse into one row; same link with different metadata stays as separate rows. Each unique `url` is fetched at most once per job regardless of how many rows reference it.
- Confirming a row must go through `ResourceService.create_multiple_resources()` — do not write a parallel Resource-creation code path.
- An `ImportJob` closes (row deleted from the DB, cascading its rows) the instant every row is `CONFIRMED` or `DISCARDED` — there is no persistent "closed" state to display.
- An `ImportJob` with no row confirmed/discarded/retried for 14 days is purged (its still-open rows and staged files deleted); already-`CONFIRMED` rows are Resources by then and are never touched by this policy.
- Do not replicate the Domain crawl system's worker-polling/heartbeat/leader-election infrastructure (`worker_id`, `heartbeat_at`, a separate claim-based worker process) — that exists for recurring scheduled crawls across many domains and is unnecessary complexity here. Use the simpler per-request background-thread pattern already established by `ResourceService._index_resources_background()`.

---

### Task 1: `ImportJob` / `ImportJobRow` models, enums, migration

**Files:**
- Create: `backend/models/enums/import_job_status.py`
- Create: `backend/models/enums/import_row_status.py`
- Create: `backend/models/import_job.py`
- Create: `backend/models/import_job_row.py`
- Modify: `backend/models/__init__.py`
- Modify: `backend/models/resource.py`
- Create: `alembic/versions/csvimp001_csv_pdf_import.py`
- Test: `tests/integration/test_csv_import_migration.py`

**Interfaces:**
- Produces (used by every later backend task): `ImportJob(id, repository_id, status, source_filename, link_column, created_at, last_activity_at)`; `ImportJobRow(id, import_job_id, url, row_metadata, status, failure_reason, staged_path, resource_id, created_at)`; `ImportJobStatus{DOWNLOADING, REVIEW}`; `ImportRowStatus{PENDING, DOWNLOADING, DOWNLOADED, FAILED, CONFIRMED, DISCARDED}`; `Resource.extra_metadata` (JSON, nullable).

- [ ] **Step 1: Write the enums**

`backend/models/enums/import_job_status.py`:
```python
import enum


class ImportJobStatus(str, enum.Enum):
    DOWNLOADING = "DOWNLOADING"
    REVIEW = "REVIEW"
```

`backend/models/enums/import_row_status.py`:
```python
import enum


class ImportRowStatus(str, enum.Enum):
    PENDING = "PENDING"
    DOWNLOADING = "DOWNLOADING"
    DOWNLOADED = "DOWNLOADED"
    FAILED = "FAILED"
    CONFIRMED = "CONFIRMED"
    DISCARDED = "DISCARDED"
```

- [ ] **Step 2: Write the models**

`backend/models/import_job.py`:
```python
import sqlalchemy as sa
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from db.database import Base
from datetime import datetime

from models.enums.import_job_status import ImportJobStatus


class ImportJob(Base):
    """A single CSV → PDF import run for a Repository."""
    __tablename__ = 'import_job'

    id = Column(Integer, primary_key=True, autoincrement=True)
    repository_id = Column(Integer, sa.ForeignKey('Repository.repository_id', ondelete='CASCADE'), nullable=False, index=True)

    status = Column(
        sa.Enum(ImportJobStatus, name='import_job_status', create_type=False),
        nullable=False,
        default=ImportJobStatus.DOWNLOADING,
        server_default='DOWNLOADING',
    )
    source_filename = Column(String(255), nullable=True)
    link_column = Column(String(255), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    last_activity_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    repository = relationship('Repository')
    rows = relationship('ImportJobRow', back_populates='import_job', cascade='all, delete-orphan')
```

`backend/models/import_job_row.py`:
```python
import sqlalchemy as sa
from sqlalchemy import Column, Integer, String, JSON, DateTime
from sqlalchemy.orm import relationship
from db.database import Base
from datetime import datetime

from models.enums.import_row_status import ImportRowStatus


class ImportJobRow(Base):
    """One (url, metadata) row surviving dedup within an ImportJob."""
    __tablename__ = 'import_job_row'

    id = Column(Integer, primary_key=True, autoincrement=True)
    import_job_id = Column(Integer, sa.ForeignKey('import_job.id', ondelete='CASCADE'), nullable=False, index=True)

    url = Column(String(2048), nullable=False)
    row_metadata = Column(JSON, nullable=True)

    status = Column(
        sa.Enum(ImportRowStatus, name='import_row_status', create_type=False),
        nullable=False,
        default=ImportRowStatus.PENDING,
        server_default='PENDING',
    )
    failure_reason = Column(String(32), nullable=True)
    staged_path = Column(String(1000), nullable=True)
    resource_id = Column(Integer, sa.ForeignKey('Resource.resource_id', ondelete='SET NULL'), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    import_job = relationship('ImportJob', back_populates='rows')
```

- [ ] **Step 3: Register models and add `Resource.extra_metadata`**

In `backend/models/__init__.py`, alongside the existing `from .crawl_job import CrawlJob` line, add:
```python
from .import_job import ImportJob
from .import_job_row import ImportJobRow
```
And add `'ImportJob', 'ImportJobRow'` to the `__all__` list next to `'CrawlJob'`.

In `backend/models/resource.py`, add the import and column:
```python
from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, DateTime, JSON
```
```python
    extra_metadata = Column(JSON, nullable=True)  # CSV-import metadata, merged into indexed chunk metadata
```
(add the column directly under the existing `folder_id = Column(...)` block, before the `repository = relationship(...)` line)

- [ ] **Step 4: Write the migration**

`alembic/versions/csvimp001_csv_pdf_import.py` (check the real current head first — run `poetry run alembic heads` and use that value as `down_revision`; at the time this plan was written the head was `ragcfg002`):
```python
"""CSV → PDF import: add import_job, import_job_row tables; add Resource.extra_metadata.

Revision ID: csvimp001
Revises: ragcfg002
Create Date: 2026-08-17
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'csvimp001'
down_revision = 'ragcfg002'
branch_labels = None
depends_on = None


def upgrade():
    import_job_status = postgresql.ENUM('DOWNLOADING', 'REVIEW', name='import_job_status', create_type=True)
    import_job_status.create(op.get_bind(), checkfirst=True)

    import_row_status = postgresql.ENUM(
        'PENDING', 'DOWNLOADING', 'DOWNLOADED', 'FAILED', 'CONFIRMED', 'DISCARDED',
        name='import_row_status', create_type=True,
    )
    import_row_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        'import_job',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('repository_id', sa.Integer(), nullable=False),
        sa.Column('status', postgresql.ENUM('DOWNLOADING', 'REVIEW', name='import_job_status', create_type=False),
                   nullable=False, server_default='DOWNLOADING'),
        sa.Column('source_filename', sa.String(length=255), nullable=True),
        sa.Column('link_column', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('NOW()')),
        sa.Column('last_activity_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['repository_id'], ['Repository.repository_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_import_job_repository_id', 'import_job', ['repository_id'])
    op.create_index('ix_import_job_created_at', 'import_job', ['created_at'])
    op.create_index('ix_import_job_last_activity_at', 'import_job', ['last_activity_at'])

    op.create_table(
        'import_job_row',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('import_job_id', sa.Integer(), nullable=False),
        sa.Column('url', sa.String(length=2048), nullable=False),
        sa.Column('row_metadata', sa.JSON(), nullable=True),
        sa.Column('status', postgresql.ENUM(
            'PENDING', 'DOWNLOADING', 'DOWNLOADED', 'FAILED', 'CONFIRMED', 'DISCARDED',
            name='import_row_status', create_type=False), nullable=False, server_default='PENDING'),
        sa.Column('failure_reason', sa.String(length=32), nullable=True),
        sa.Column('staged_path', sa.String(length=1000), nullable=True),
        sa.Column('resource_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['import_job_id'], ['import_job.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['resource_id'], ['Resource.resource_id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_import_job_row_import_job_id', 'import_job_row', ['import_job_id'])

    op.add_column('Resource', sa.Column('extra_metadata', sa.JSON(), nullable=True))


def downgrade():
    op.drop_column('Resource', 'extra_metadata')

    op.drop_index('ix_import_job_row_import_job_id', table_name='import_job_row')
    op.drop_table('import_job_row')

    op.drop_index('ix_import_job_last_activity_at', table_name='import_job')
    op.drop_index('ix_import_job_created_at', table_name='import_job')
    op.drop_index('ix_import_job_repository_id', table_name='import_job')
    op.drop_table('import_job')

    postgresql.ENUM(name='import_row_status', create_type=False).drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name='import_job_status', create_type=False).drop(op.get_bind(), checkfirst=True)
```

- [ ] **Step 5: Run the migration and verify the downgrade**

Run: `alembic upgrade head`
Expected: no errors, `import_job` and `import_job_row` tables exist, `Resource` has an `extra_metadata` column.

Run: `alembic downgrade -1`
Expected: no errors, both tables and the column are gone.

Run: `alembic upgrade head` again to leave the DB migrated.

- [ ] **Step 6: Write the migration regression test**

`tests/integration/test_csv_import_migration.py`:
```python
"""Verifies the csvimp001 migration's tables exist and Resource.extra_metadata round-trips."""
from sqlalchemy import inspect

from db.database import engine
from models.resource import Resource
from models.import_job import ImportJob
from models.import_job_row import ImportJobRow


def test_import_job_tables_exist():
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert 'import_job' in tables
    assert 'import_job_row' in tables


def test_resource_has_extra_metadata_column(db_session):
    resource = Resource(name='test', uri='test.pdf', type='.pdf', status='pending', repository_id=None,
                         extra_metadata={'category': 'finance'})
    db_session.add(resource)
    db_session.commit()
    db_session.refresh(resource)
    assert resource.extra_metadata == {'category': 'finance'}
```

- [ ] **Step 7: Run the test**

Run: `pytest tests/integration/test_csv_import_migration.py -v`
Expected: PASS (uses the `db_session` fixture already available to integration tests per `tests/integration/`).

- [ ] **Step 8: Commit**

```bash
git add backend/models/enums/import_job_status.py backend/models/enums/import_row_status.py \
        backend/models/import_job.py backend/models/import_job_row.py \
        backend/models/__init__.py backend/models/resource.py \
        alembic/versions/csvimp001_csv_pdf_import.py \
        tests/integration/test_csv_import_migration.py
git commit -m "feat(csv-import): add ImportJob/ImportJobRow models and Resource.extra_metadata"
```

---

### Task 2: CSV parsing and dedup (`csv_import_parser.py`)

Pure logic, no DB/network — the only task deliberately kept free of I/O so it's trivial to unit test.

**Files:**
- Create: `backend/services/csv_import_parser.py`
- Test: `tests/unit/services/test_csv_import_parser.py`

**Interfaces:**
- Consumes: `pandas` (already a dependency).
- Produces (used by Task 4): 
  - `read_csv_headers(file_obj) -> list[str]`
  - `ParsedRow` dataclass: `url: str`, `row_metadata: dict`
  - `parse_and_dedupe_csv(file_obj, link_column: str) -> list[ParsedRow]` — rows with an empty/whitespace link are **excluded** from the returned list and reported separately as `no_link_count` via the second element of the returned tuple: `parse_and_dedupe_csv(...) -> tuple[list[ParsedRow], int]`.

- [ ] **Step 1: Write the failing unit tests**

`tests/unit/services/test_csv_import_parser.py`:
```python
import io
import pytest

from services.csv_import_parser import read_csv_headers, parse_and_dedupe_csv, ParsedRow


def _csv(text: str) -> io.BytesIO:
    return io.BytesIO(text.encode('utf-8'))


def test_read_csv_headers():
    headers = read_csv_headers(_csv("link,title,category\nhttp://a.com/x.pdf,A,Finance\n"))
    assert headers == ['link', 'title', 'category']


def test_dedupe_identical_url_and_metadata_merges():
    csv_text = (
        "link,title\n"
        "http://a.com/x.pdf,A\n"
        "http://a.com/x.pdf,A\n"
    )
    rows, no_link_count = parse_and_dedupe_csv(_csv(csv_text), link_column='link')
    assert rows == [ParsedRow(url='http://a.com/x.pdf', row_metadata={'title': 'A'})]
    assert no_link_count == 0


def test_same_url_different_metadata_stays_separate():
    csv_text = (
        "link,title\n"
        "http://a.com/x.pdf,A\n"
        "http://a.com/x.pdf,B\n"
    )
    rows, _ = parse_and_dedupe_csv(_csv(csv_text), link_column='link')
    assert len(rows) == 2
    assert {r.row_metadata['title'] for r in rows} == {'A', 'B'}


def test_empty_link_excluded_and_counted():
    csv_text = (
        "link,title\n"
        "http://a.com/x.pdf,A\n"
        " ,B\n"
        ",C\n"
    )
    rows, no_link_count = parse_and_dedupe_csv(_csv(csv_text), link_column='link')
    assert len(rows) == 1
    assert no_link_count == 2


def test_unknown_link_column_raises():
    with pytest.raises(ValueError, match="link"):
        parse_and_dedupe_csv(_csv("a,b\n1,2\n"), link_column='nope')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/services/test_csv_import_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.csv_import_parser'`

- [ ] **Step 3: Write the implementation**

`backend/services/csv_import_parser.py`:
```python
from dataclasses import dataclass, field
from typing import BinaryIO

import pandas as pd


@dataclass(frozen=True)
class ParsedRow:
    url: str
    row_metadata: dict = field(default_factory=dict)


def read_csv_headers(file_obj: BinaryIO) -> list[str]:
    file_obj.seek(0)
    df = pd.read_csv(file_obj, nrows=0)
    return list(df.columns)


def parse_and_dedupe_csv(file_obj: BinaryIO, link_column: str) -> tuple[list[ParsedRow], int]:
    file_obj.seek(0)
    df = pd.read_csv(file_obj, dtype=str, keep_default_na=False)

    if link_column not in df.columns:
        raise ValueError(f"link column '{link_column}' not found in CSV headers: {list(df.columns)}")

    metadata_columns = [c for c in df.columns if c != link_column]

    seen: dict[tuple, ParsedRow] = {}
    no_link_count = 0

    for _, record in df.iterrows():
        url = record[link_column].strip()
        if not url:
            no_link_count += 1
            continue

        row_metadata = {col: record[col] for col in metadata_columns}
        dedupe_key = (url, tuple(sorted(row_metadata.items())))
        seen.setdefault(dedupe_key, ParsedRow(url=url, row_metadata=row_metadata))

    return list(seen.values()), no_link_count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/services/test_csv_import_parser.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/services/csv_import_parser.py tests/unit/services/test_csv_import_parser.py
git commit -m "feat(csv-import): add CSV header preview and dedup parsing"
```

---

### Task 3: PDF download worker

**Files:**
- Modify: `backend/services/crawl/http_fetcher.py`
- Create: `backend/services/import_job_download.py`
- Test: `tests/unit/services/test_import_job_download.py`

**Interfaces:**
- Consumes: `models.import_job_row.ImportJobRow`, `models.enums.import_row_status.ImportRowStatus`, `services.crawl.http_fetcher.fetch`, `db.database.SessionLocal`, app's `max_file_size_mb` via `models.app.App`.
- Produces (used by Task 4): `download_job_rows(import_job_id: int) -> None` — synchronous, blocking; the caller is responsible for running it off the request thread (Task 4 wraps it in a background thread, Task 5 calls the single-row variant directly since it's fast enough to await). Also produces `download_one_row(row_id: int) -> None` (used by Task 4's retry endpoint), which `download_job_rows` calls per row.

- [ ] **Step 1: Write the failing unit tests**

`tests/unit/services/test_import_job_download.py`:
```python
import pytest
from unittest.mock import patch, AsyncMock

from models.import_job import ImportJob
from models.import_job_row import ImportJobRow
from models.enums.import_row_status import ImportRowStatus
from services.crawl.http_fetcher import FetchResult
from services.import_job_download import download_job_rows, download_one_row


def _make_job_with_row(db_session, repository, url='http://a.com/x.pdf'):
    job = ImportJob(repository_id=repository.repository_id, source_filename='x.csv', link_column='link')
    db_session.add(job)
    db_session.commit()
    row = ImportJobRow(import_job_id=job.id, url=url, row_metadata={'title': 'A'})
    db_session.add(row)
    db_session.commit()
    return job, row


@patch('services.import_job_download.fetch', new_callable=AsyncMock)
def test_successful_pdf_download_marks_row_downloaded(mock_fetch, db_session, repository):
    job, row = _make_job_with_row(db_session, repository)
    mock_fetch.return_value = FetchResult(status_code=200, content=b'%PDF-1.4 fake content')

    download_job_rows(job.id)

    db_session.refresh(row)
    assert row.status == ImportRowStatus.DOWNLOADED
    assert row.staged_path is not None
    assert row.failure_reason is None


@patch('services.import_job_download.fetch', new_callable=AsyncMock)
def test_non_pdf_content_marks_row_failed(mock_fetch, db_session, repository):
    job, row = _make_job_with_row(db_session, repository)
    mock_fetch.return_value = FetchResult(status_code=200, content=b'<html>not a pdf</html>')

    download_job_rows(job.id)

    db_session.refresh(row)
    assert row.status == ImportRowStatus.FAILED
    assert row.failure_reason == 'NOT_A_PDF'


@patch('services.import_job_download.fetch', new_callable=AsyncMock)
def test_404_marks_row_not_found(mock_fetch, db_session, repository):
    job, row = _make_job_with_row(db_session, repository)
    mock_fetch.return_value = FetchResult(status_code=404)

    download_job_rows(job.id)

    db_session.refresh(row)
    assert row.status == ImportRowStatus.FAILED
    assert row.failure_reason == 'NOT_FOUND'


@patch('services.import_job_download.fetch', new_callable=AsyncMock)
def test_shared_url_downloaded_once_across_rows(mock_fetch, db_session, repository):
    job, row1 = _make_job_with_row(db_session, repository, url='http://a.com/shared.pdf')
    row2 = ImportJobRow(import_job_id=job.id, url='http://a.com/shared.pdf', row_metadata={'title': 'B'})
    db_session.add(row2)
    db_session.commit()
    mock_fetch.return_value = FetchResult(status_code=200, content=b'%PDF-1.4 fake content')

    download_job_rows(job.id)

    assert mock_fetch.call_count == 1
    db_session.refresh(row1)
    db_session.refresh(row2)
    assert row1.status == ImportRowStatus.DOWNLOADED
    assert row2.status == ImportRowStatus.DOWNLOADED
    assert row1.staged_path != row2.staged_path


@patch('services.import_job_download.fetch', new_callable=AsyncMock)
def test_job_moves_to_review_when_all_rows_settled(mock_fetch, db_session, repository):
    job, row = _make_job_with_row(db_session, repository)
    mock_fetch.return_value = FetchResult(status_code=200, content=b'%PDF-1.4 fake content')

    download_job_rows(job.id)

    db_session.refresh(job)
    assert job.status.value == 'REVIEW'


@patch('services.import_job_download.fetch', new_callable=AsyncMock)
def test_retry_one_row_only_touches_that_row(mock_fetch, db_session, repository):
    job, row1 = _make_job_with_row(db_session, repository, url='http://a.com/one.pdf')
    row2 = ImportJobRow(import_job_id=job.id, url='http://a.com/two.pdf', row_metadata={},
                         status=ImportRowStatus.FAILED, failure_reason='NOT_FOUND')
    db_session.add(row2)
    db_session.commit()
    mock_fetch.return_value = FetchResult(status_code=200, content=b'%PDF-1.4 fake content')

    download_one_row(row2.id)

    db_session.refresh(row1)
    db_session.refresh(row2)
    assert row1.status == ImportRowStatus.PENDING
    assert row2.status == ImportRowStatus.DOWNLOADED
```

(assumes `db_session` and `repository` fixtures already exist per `tests/integration/conftest.py` — check `tests/integration/conftest.py` for the exact fixture names/signatures and adjust the two fixture args if they differ.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/services/test_import_job_download.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.import_job_download'`

- [ ] **Step 3: Extend `FetchResult` with `content_type`**

In `backend/services/crawl/http_fetcher.py`, add a field to the dataclass and populate it:
```python
@dataclass
class FetchResult:
    status_code: int
    content: Optional[bytes] = None
    content_type: Optional[str] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    error: Optional[str] = None
```
In `fetch()`, inside the `async with session.get(...)` block, add one line alongside the existing header reads:
```python
            content_type = resp.headers.get('Content-Type')
```
and pass `content_type=content_type` into the returned `FetchResult(...)`.

- [ ] **Step 4: Write the download implementation**

`backend/services/import_job_download.py`:
```python
import os
import uuid

from sqlalchemy.orm import Session

from db.database import SessionLocal
from models.import_job import ImportJob
from models.import_job_row import ImportJobRow
from models.repository import Repository
from models.app import App
from models.enums.import_job_status import ImportJobStatus
from models.enums.import_row_status import ImportRowStatus
from services.crawl.http_fetcher import fetch, FetchResult
from utils.config import get_app_config
from utils.logger import get_logger

logger = get_logger(__name__)

_PDF_MAGIC = b'%PDF-'


def _staging_dir(import_job_id: int) -> str:
    cfg = get_app_config()
    path = os.path.join(cfg['TMP_BASE_FOLDER'], 'csv_import_staging', str(import_job_id))
    os.makedirs(path, exist_ok=True)
    return path


def _max_file_size_mb(db: Session, repository_id: int) -> int:
    repo = db.query(Repository).filter(Repository.repository_id == repository_id).first()
    if not repo:
        return 0
    app = db.query(App).filter(App.app_id == repo.app_id).first()
    return (app.max_file_size_mb or 0) if app else 0


def _classify_result(result: FetchResult, max_size_mb: int) -> tuple[bool, str | None]:
    """Returns (is_success, failure_reason)."""
    if result.status_code == 0:
        return False, 'TIMED_OUT' if result.error and 'Timeout' in result.error else 'NOT_FOUND'
    if result.status_code in (401, 403):
        return False, 'ACCESS_DENIED'
    if result.status_code != 200:
        return False, 'NOT_FOUND'
    content = result.content or b''
    if max_size_mb > 0 and len(content) / (1024 * 1024) > max_size_mb:
        return False, 'FILE_TOO_LARGE'
    if not content.startswith(_PDF_MAGIC):
        return False, 'NOT_A_PDF'
    return True, None


async def _download_one(url: str) -> FetchResult:
    return await fetch(url, timeout=30.0)


def _apply_result_to_row(db: Session, row: ImportJobRow, result: FetchResult, max_size_mb: int,
                          staging_dir: str, shared_staged_path: str | None) -> str | None:
    is_success, failure_reason = _classify_result(result, max_size_mb)
    if not is_success:
        row.status = ImportRowStatus.FAILED
        row.failure_reason = failure_reason
        return None

    staged_path = shared_staged_path
    if staged_path is None:
        staged_path = os.path.join(staging_dir, f"{uuid.uuid4().hex}.pdf")
        with open(staged_path, 'wb') as f:
            f.write(result.content)

    row.status = ImportRowStatus.DOWNLOADED
    row.failure_reason = None
    row.staged_path = staged_path
    return staged_path


def _maybe_close_to_review(db: Session, job: ImportJob) -> None:
    unsettled = db.query(ImportJobRow).filter(
        ImportJobRow.import_job_id == job.id,
        ImportJobRow.status.in_([ImportRowStatus.PENDING, ImportRowStatus.DOWNLOADING]),
    ).count()
    if unsettled == 0:
        job.status = ImportJobStatus.REVIEW
        db.commit()


def download_job_rows(import_job_id: int) -> None:
    """Download every PENDING row of a job. Same URL is fetched once and its
    result (staged file or failure) is copied to every row sharing that URL."""
    import asyncio
    from datetime import datetime

    db = SessionLocal()
    try:
        job = db.query(ImportJob).filter(ImportJob.id == import_job_id).first()
        if not job:
            logger.error(f"ImportJob {import_job_id} not found for download")
            return

        rows = db.query(ImportJobRow).filter(
            ImportJobRow.import_job_id == import_job_id,
            ImportJobRow.status == ImportRowStatus.PENDING,
        ).all()
        if not rows:
            return

        max_size_mb = _max_file_size_mb(db, job.repository_id)
        staging_dir = _staging_dir(import_job_id)

        rows_by_url: dict[str, list[ImportJobRow]] = {}
        for row in rows:
            rows_by_url.setdefault(row.url, []).append(row)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            for url, url_rows in rows_by_url.items():
                for row in url_rows:
                    row.status = ImportRowStatus.DOWNLOADING
                db.commit()

                result = loop.run_until_complete(_download_one(url))

                shared_staged_path = None
                for row in url_rows:
                    shared_staged_path = _apply_result_to_row(
                        db, row, result, max_size_mb, staging_dir, shared_staged_path,
                    )
                db.commit()
        finally:
            loop.close()

        job.last_activity_at = datetime.utcnow()
        db.commit()
        _maybe_close_to_review(db, job)
    except Exception:
        logger.error(f"Error downloading ImportJob {import_job_id}", exc_info=True)
    finally:
        db.close()


def download_one_row(row_id: int) -> None:
    """Retry a single row in place. Awaited directly from the retry endpoint —
    a single fetch is fast enough not to need a background thread."""
    import asyncio
    from datetime import datetime

    db = SessionLocal()
    try:
        row = db.query(ImportJobRow).filter(ImportJobRow.id == row_id).first()
        if not row:
            logger.error(f"ImportJobRow {row_id} not found for retry")
            return
        job = db.query(ImportJob).filter(ImportJob.id == row.import_job_id).first()

        max_size_mb = _max_file_size_mb(db, job.repository_id)
        staging_dir = _staging_dir(job.id)

        row.status = ImportRowStatus.DOWNLOADING
        db.commit()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_download_one(row.url))
        finally:
            loop.close()

        _apply_result_to_row(db, row, result, max_size_mb, staging_dir, None)
        job.last_activity_at = datetime.utcnow()
        db.commit()
    except Exception:
        logger.error(f"Error retrying ImportJobRow {row_id}", exc_info=True)
    finally:
        db.close()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/services/test_import_job_download.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit**

```bash
git add backend/services/crawl/http_fetcher.py backend/services/import_job_download.py \
        tests/unit/services/test_import_job_download.py
git commit -m "feat(csv-import): add PDF download worker with url dedup and retry"
```

---

### Task 4: Create-import-job service + preview/create endpoints

**Files:**
- Create: `backend/services/import_job_service.py`
- Create: `backend/schemas/import_job_schemas.py`
- Modify: `backend/routers/internal/repositories.py`
- Test: `tests/integration/test_csv_import_create.py`

**Interfaces:**
- Consumes: `services.csv_import_parser.{read_csv_headers, parse_and_dedupe_csv}` (Task 2), `services.import_job_download.download_job_rows` (Task 3), `models.import_job.ImportJob`, `models.import_job_row.ImportJobRow` (Task 1).
- Produces (used by Tasks 5, 7, 8 and the frontend): 
  - `ImportJobService.get_active_job(repository_id, db) -> ImportJob | None`
  - `ImportJobService.create_job(repository_id, file, link_column, db) -> ImportJob` — raises `ConflictError(job_id)` if one is already active; starts the download in a background thread before returning.
  - `ImportJobService.get_job_with_counts(import_job_id, db) -> dict` — `{job, rows, counts: {total, pending, downloading, downloaded, failed}}`
  - `ConflictError(Exception)` with a `.job_id` attribute (same shape as `CrawlJobService`'s).

- [ ] **Step 1: Write the failing integration tests**

`tests/integration/test_csv_import_create.py`:
```python
import io

from services.import_job_service import ImportJobService, ConflictError
from models.enums.import_job_status import ImportJobStatus


def _csv_file(text: str):
    return io.BytesIO(text.encode('utf-8'))


def test_create_job_persists_deduped_rows(db_session, repository):
    csv_text = "link,title\nhttp://a.com/x.pdf,A\nhttp://a.com/x.pdf,A\nhttp://a.com/y.pdf,B\n"
    job = ImportJobService.create_job(repository.repository_id, _csv_file(csv_text), 'link', db_session)

    assert job.id is not None
    assert job.status == ImportJobStatus.DOWNLOADING
    assert len(job.rows) == 2


def test_create_job_rejects_unknown_link_column(db_session, repository):
    import pytest
    with pytest.raises(ValueError):
        ImportJobService.create_job(repository.repository_id, _csv_file("a,b\n1,2\n"), 'nope', db_session)


def test_second_active_job_conflicts(db_session, repository):
    import pytest
    ImportJobService.create_job(
        repository.repository_id, _csv_file("link\nhttp://a.com/x.pdf\n"), 'link', db_session,
    )
    with pytest.raises(ConflictError):
        ImportJobService.create_job(
            repository.repository_id, _csv_file("link\nhttp://a.com/y.pdf\n"), 'link', db_session,
        )


def test_get_active_job_returns_none_when_no_job(db_session, repository):
    assert ImportJobService.get_active_job(repository.repository_id, db_session) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integration/test_csv_import_create.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.import_job_service'`

- [ ] **Step 3: Write the service**

`backend/services/import_job_service.py`:
```python
import threading
from typing import BinaryIO

from sqlalchemy.orm import Session

from models.import_job import ImportJob
from models.import_job_row import ImportJobRow
from models.enums.import_job_status import ImportJobStatus
from models.enums.import_row_status import ImportRowStatus
from services.csv_import_parser import read_csv_headers, parse_and_dedupe_csv
from services.import_job_download import download_job_rows
from utils.logger import get_logger

logger = get_logger(__name__)


class ConflictError(Exception):
    def __init__(self, job_id: int):
        self.job_id = job_id
        super().__init__(f"An import job is already active: {job_id}")


class ImportJobService:

    @staticmethod
    def get_active_job(repository_id: int, db: Session) -> ImportJob | None:
        return db.query(ImportJob).filter(
            ImportJob.repository_id == repository_id,
            ImportJob.status.in_([ImportJobStatus.DOWNLOADING, ImportJobStatus.REVIEW]),
        ).first()

    @staticmethod
    def preview_headers(file: BinaryIO) -> list[str]:
        return read_csv_headers(file)

    @staticmethod
    def create_job(repository_id: int, file: BinaryIO, link_column: str, db: Session,
                    source_filename: str | None = None) -> ImportJob:
        existing = ImportJobService.get_active_job(repository_id, db)
        if existing:
            raise ConflictError(existing.id)

        rows, no_link_count = parse_and_dedupe_csv(file, link_column)
        if no_link_count:
            logger.info(f"CSV import for repository {repository_id}: {no_link_count} row(s) skipped (no link)")

        job = ImportJob(repository_id=repository_id, source_filename=source_filename, link_column=link_column)
        db.add(job)
        db.commit()

        for parsed_row in rows:
            db.add(ImportJobRow(import_job_id=job.id, url=parsed_row.url, row_metadata=parsed_row.row_metadata))
        db.commit()
        db.refresh(job)

        if rows:
            threading.Thread(target=download_job_rows, args=(job.id,), daemon=True).start()
        else:
            job.status = ImportJobStatus.REVIEW
            db.commit()

        return job

    @staticmethod
    def get_job_with_counts(import_job_id: int, db: Session) -> dict:
        job = db.query(ImportJob).filter(ImportJob.id == import_job_id).first()
        if not job:
            return None
        rows = db.query(ImportJobRow).filter(ImportJobRow.import_job_id == import_job_id).all()
        counts = {status.value.lower(): 0 for status in ImportRowStatus}
        for row in rows:
            counts[row.status.value.lower()] += 1
        counts['total'] = len(rows)
        return {'job': job, 'rows': rows, 'counts': counts}
```

- [ ] **Step 4: Write the schemas**

`backend/schemas/import_job_schemas.py`:
```python
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Any
from datetime import datetime


class CsvPreviewResponseSchema(BaseModel):
    headers: List[str]


class ImportJobRowSchema(BaseModel):
    id: int
    url: str
    row_metadata: dict = {}
    status: str
    failure_reason: Optional[str] = None
    resource_id: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class ImportJobCountsSchema(BaseModel):
    total: int = 0
    pending: int = 0
    downloading: int = 0
    downloaded: int = 0
    failed: int = 0
    confirmed: int = 0
    discarded: int = 0


class ImportJobResponseSchema(BaseModel):
    id: int
    repository_id: int
    status: str
    source_filename: Optional[str] = None
    link_column: Optional[str] = None
    created_at: Optional[datetime] = None
    last_activity_at: Optional[datetime] = None
    rows: List[ImportJobRowSchema] = []
    counts: ImportJobCountsSchema


class ConfirmDiscardRowsSchema(BaseModel):
    row_ids: List[int]


class ConfirmRowsResponseSchema(BaseModel):
    created_resources: List[Any] = []
    failed_files: List[dict] = []
    session_id: Optional[str] = None
```

- [ ] **Step 5: Wire the router endpoints**

In `backend/routers/internal/repositories.py`, add these imports near the existing service imports:
```python
from services.import_job_service import ImportJobService, ConflictError
from schemas.import_job_schemas import (
    CsvPreviewResponseSchema, ImportJobResponseSchema, ConfirmDiscardRowsSchema, ConfirmRowsResponseSchema,
)
```
Then add the endpoints, next to the `# ==================== RESOURCE MANAGEMENT ====================` section:
```python
# ==================== CSV IMPORT ====================

@repositories_router.post("/{repository_id}/csv-imports/preview",
                         summary="Preview CSV headers for import",
                         tags=["CSV Import"],
                         response_model=CsvPreviewResponseSchema)
async def preview_csv_import(
    app_id: int,
    repository_id: int,
    file: Annotated[UploadFile, File(...)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    headers = ImportJobService.preview_headers(file.file)
    return CsvPreviewResponseSchema(headers=headers)


@repositories_router.post("/{repository_id}/csv-imports",
                         summary="Start a CSV → PDF import",
                         tags=["CSV Import"],
                         response_model=ImportJobResponseSchema,
                         status_code=status.HTTP_202_ACCEPTED)
async def create_csv_import(
    app_id: int,
    repository_id: int,
    file: Annotated[UploadFile, File(...)],
    link_column: Annotated[str, Form(...)],
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    try:
        job = ImportJobService.create_job(repository_id, file.file, link_column, db, source_filename=file.filename)
    except ConflictError as e:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail={"message": "An import is already in progress for this repository", "job_id": e.job_id},
        )
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return _import_job_response(job.id, db)


@repositories_router.get("/{repository_id}/csv-imports/active",
                        summary="Get the active CSV import job, if any",
                        tags=["CSV Import"],
                        response_model=Optional[ImportJobResponseSchema])
async def get_active_csv_import(
    app_id: int,
    repository_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
):
    job = ImportJobService.get_active_job(repository_id, db)
    if not job:
        return None
    return _import_job_response(job.id, db)


@repositories_router.get("/{repository_id}/csv-imports/{import_job_id}",
                        summary="Get a CSV import job with its rows",
                        tags=["CSV Import"],
                        response_model=ImportJobResponseSchema)
async def get_csv_import(
    app_id: int,
    repository_id: int,
    import_job_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("viewer"))],
):
    result = _import_job_response(import_job_id, db)
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Import job not found")
    return result


def _import_job_response(import_job_id: int, db: Session) -> Optional[ImportJobResponseSchema]:
    data = ImportJobService.get_job_with_counts(import_job_id, db)
    if not data:
        return None
    job = data['job']
    return ImportJobResponseSchema(
        id=job.id, repository_id=job.repository_id, status=job.status.value,
        source_filename=job.source_filename, link_column=job.link_column,
        created_at=job.created_at, last_activity_at=job.last_activity_at,
        rows=data['rows'], counts=data['counts'],
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/integration/test_csv_import_create.py -v`
Expected: PASS (4 passed)

- [ ] **Step 7: Commit**

```bash
git add backend/services/import_job_service.py backend/schemas/import_job_schemas.py \
        backend/routers/internal/repositories.py tests/integration/test_csv_import_create.py
git commit -m "feat(csv-import): add create/preview/get import job endpoints"
```

---

### Task 5: Retry-row endpoint

**Files:**
- Modify: `backend/routers/internal/repositories.py`
- Test: `tests/integration/test_csv_import_retry.py`

**Interfaces:**
- Consumes: `services.import_job_download.download_one_row` (Task 3), `_import_job_response` (Task 4).
- Produces: `POST /{repository_id}/csv-imports/{import_job_id}/rows/{row_id}/retry` → `ImportJobResponseSchema`.

- [ ] **Step 1: Write the failing integration test**

`tests/integration/test_csv_import_retry.py`:
```python
from unittest.mock import patch, AsyncMock

from services.crawl.http_fetcher import FetchResult
from services.import_job_service import ImportJobService
import io


@patch('services.import_job_download.fetch', new_callable=AsyncMock)
def test_retry_endpoint_updates_row(mock_fetch, client, db_session, repository, auth_headers):
    mock_fetch.return_value = FetchResult(status_code=404)
    job = ImportJobService.create_job(
        repository.repository_id, io.BytesIO(b"link\nhttp://a.com/x.pdf\n"), 'link', db_session,
    )
    row_id = job.rows[0].id

    mock_fetch.return_value = FetchResult(status_code=200, content=b'%PDF-1.4 ok')
    response = client.post(
        f"/internal/apps/{repository.app_id}/repositories/{repository.repository_id}"
        f"/csv-imports/{job.id}/rows/{row_id}/retry",
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    row = next(r for r in body['rows'] if r['id'] == row_id)
    assert row['status'] == 'DOWNLOADED'
```

(this test's exact `client`/`auth_headers` fixture names must match whatever `tests/integration/conftest.py` already exposes for hitting the FastAPI app — check that file and adjust; the pattern of "download fails on create, succeeds on retry" is what matters.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_csv_import_retry.py -v`
Expected: FAIL with 404 (route doesn't exist yet)

- [ ] **Step 3: Add the endpoint**

In `backend/routers/internal/repositories.py`, add the import:
```python
from services.import_job_download import download_one_row
```
And the endpoint, after `get_csv_import`:
```python
@repositories_router.post("/{repository_id}/csv-imports/{import_job_id}/rows/{row_id}/retry",
                         summary="Retry a failed CSV import row",
                         tags=["CSV Import"],
                         response_model=ImportJobResponseSchema)
async def retry_csv_import_row(
    app_id: int,
    repository_id: int,
    import_job_id: int,
    row_id: int,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    download_one_row(row_id)
    result = _import_job_response(import_job_id, db)
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Import job not found")
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_csv_import_retry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/routers/internal/repositories.py tests/integration/test_csv_import_retry.py
git commit -m "feat(csv-import): add retry-row endpoint"
```

---

### Task 6: `extra_metadata` threaded through Resource creation and indexing

**Files:**
- Modify: `backend/services/resource_service.py`
- Modify: `backend/services/silo_service.py`
- Test: `tests/unit/services/test_resource_extra_metadata.py`

**Interfaces:**
- Consumes: `Resource.extra_metadata` (Task 1).
- Produces (used by Task 7): `ResourceService.create_multiple_resources(files, repository_id, db, custom_names=None, folder_id=None, extra_metadata=None)` — new optional `extra_metadata: dict[int, dict] | None` keyed by the same file-list index as `custom_names`. `SiloService.index_resource()` merges `resource.extra_metadata` into `base_metadata` without letting it override the reserved keys already in `base_metadata`.

- [ ] **Step 1: Write the failing unit tests**

`tests/unit/services/test_resource_extra_metadata.py`:
```python
from unittest.mock import patch

from services.resource_service import ResourceService


class _FakeFile:
    def __init__(self, filename, staged_path):
        self.filename = filename
        self._staged_path = staged_path

    def save(self, dest_path):
        with open(self._staged_path, 'rb') as src, open(dest_path, 'wb') as dst:
            dst.write(src.read())


def test_extra_metadata_set_on_created_resource(tmp_path, db_session, repository):
    staged = tmp_path / "staged.pdf"
    staged.write_bytes(b"%PDF-1.4 fake")
    file_adapter = _FakeFile("row-1.pdf", str(staged))

    with patch.object(ResourceService, '_index_resources_background', return_value='session-id'):
        created, failed, session_id = ResourceService.create_multiple_resources(
            files=[file_adapter],
            repository_id=repository.repository_id,
            db=db_session,
            extra_metadata={0: {'category': 'Finance'}},
        )

    assert not failed
    assert created[0].extra_metadata == {'category': 'Finance'}


def test_extra_metadata_defaults_to_none_when_not_provided(tmp_path, db_session, repository):
    staged = tmp_path / "staged.pdf"
    staged.write_bytes(b"%PDF-1.4 fake")
    file_adapter = _FakeFile("row-2.pdf", str(staged))

    with patch.object(ResourceService, '_index_resources_background', return_value='session-id'):
        created, failed, session_id = ResourceService.create_multiple_resources(
            files=[file_adapter], repository_id=repository.repository_id, db=db_session,
        )

    assert created[0].extra_metadata is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/services/test_resource_extra_metadata.py -v`
Expected: FAIL with `TypeError: create_multiple_resources() got an unexpected keyword argument 'extra_metadata'`

- [ ] **Step 3: Extend `create_multiple_resources` and `_process_single_file`**

In `backend/services/resource_service.py`, change the signature and pass-through:
```python
    @staticmethod
    def create_multiple_resources(files: List, repository_id: int, db: Session, custom_names: dict = None,
                                   folder_id: Optional[int] = None, extra_metadata: dict = None) -> Tuple[List[Resource], List[dict], str]:
```
Inside the method, right after `custom_names = {}` defaulting:
```python
        if extra_metadata is None:
            extra_metadata = {}
```
And change the loop body:
```python
        for index, file in enumerate(files):
            custom_name = custom_names.get(index)
            result = ResourceService._process_single_file(
                file, repository_id, target_path, custom_name, folder_id, db,
                extra_metadata=extra_metadata.get(index),
            )
```
Then update `_process_single_file`'s signature and the `Resource(...)` construction:
```python
    @staticmethod
    def _process_single_file(file, repository_id: int, target_path: str, custom_name: str = None,
                              folder_id: Optional[int] = None, db: Session = None, extra_metadata: dict = None):
```
```python
                resource = Resource(
                    name=name,
                    uri=save_filename,
                    repository_id=repository_id,
                    folder_id=folder_id,
                    type=file_extension,
                    status='pending',
                    extra_metadata=extra_metadata,
                )
                ResourceRepository.create(db, resource)
```
(the re-upload/`existing` branch is left untouched — a retried CSV row that happens to collide with an existing error'd Resource keeps today's reset-to-pending behavior; `extra_metadata` is not backfilled onto pre-existing rows, which is an acceptable, explicitly out-of-scope edge case per the spec's non-goals around conflict resolution.)

- [ ] **Step 4: Merge `extra_metadata` into indexed chunk metadata**

In `backend/services/silo_service.py`, right after the existing `base_metadata = {...}` block in `index_resource()` (after the `folder_id`/`folder_path`/`ref` branch, before `_is_lightrag = ...`), add:
```python
            if resource_with_relations.extra_metadata:
                base_metadata = {**resource_with_relations.extra_metadata, **base_metadata}
```
(placing the caller-supplied dict first and `base_metadata`'s own keys second means CSV columns can never clobber `repository_id`/`resource_id`/`silo_id`/etc.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/services/test_resource_extra_metadata.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Run the existing resource service test suite to check for regressions**

Run: `pytest tests/unit/services/test_resource_service.py tests/integration/ -k resource -v`
Expected: PASS — the new params are optional and default to today's behavior for every existing caller.

- [ ] **Step 7: Commit**

```bash
git add backend/services/resource_service.py backend/services/silo_service.py \
        tests/unit/services/test_resource_extra_metadata.py
git commit -m "feat(csv-import): thread extra_metadata from Resource into indexed chunks"
```

---

### Task 7: Confirm-rows endpoint

**Files:**
- Create: `backend/services/import_job_confirm.py`
- Modify: `backend/routers/internal/repositories.py`
- Test: `tests/integration/test_csv_import_confirm.py`

**Interfaces:**
- Consumes: `ResourceService.create_multiple_resources(..., extra_metadata=...)` (Task 6), `ImportJobRow`/`ImportRowStatus` (Task 1), `SiloService.{extract_documents_from_file, estimate_indexing_cost}` (`backend/services/silo_service.py:1032`, `:2372`).
- Produces (used by Task 8 for the shared close-check, and by the router): `confirm_rows(import_job_id: int, row_ids: list[int], db: Session) -> dict` — `{created_resources, failed_files, session_id}`, same shape `upload_resources_to_repository` already returns so the frontend can reuse `IngestionProgressBar` unmodified. Also produces `maybe_close_job(job, db)` (used by Task 8's discard path), and `estimate_rows(import_job_id: int, row_ids: list[int], db: Session) -> dict` — same shape as `CostEstimationResponseSchema`, computed directly from already-staged PDF paths (no re-upload needed) for the LightRAG cost-estimate step ahead of confirming.

**Why an estimate function is needed here:** the existing cost-estimate endpoint (`POST /{repository_id}/resources/estimate`, `backend/routers/internal/repositories.py:342-413`) takes multipart `UploadFile`s from the browser. CSV-imported rows have no client-side `File` object — their PDFs already live on disk at `row.staged_path` — so reusing that endpoint as-is is not possible. `estimate_rows` mirrors its body exactly, minus the temp-file copy step, by calling `SiloService.extract_documents_from_file` directly on `staged_path`.

- [ ] **Step 1: Write the failing integration test**

`tests/integration/test_csv_import_confirm.py`:
```python
import io
from unittest.mock import patch, AsyncMock

from services.crawl.http_fetcher import FetchResult
from services.import_job_service import ImportJobService
from services.import_job_confirm import confirm_rows
from models.import_job import ImportJob
from models.resource import Resource


@patch('services.import_job_download.fetch', new_callable=AsyncMock)
def test_confirm_creates_resource_with_metadata_and_closes_job(mock_fetch, db_session, repository):
    mock_fetch.return_value = FetchResult(status_code=200, content=b'%PDF-1.4 ok')
    job = ImportJobService.create_job(
        repository.repository_id, io.BytesIO(b"link,title\nhttp://a.com/x.pdf,Finance Report\n"),
        'link', db_session,
    )
    row_id = job.rows[0].id

    with patch('services.resource_service.ResourceService._index_resources_background', return_value='sid'):
        result = confirm_rows(job.id, [row_id], db_session)

    assert len(result['created_resources']) == 1
    resource = db_session.query(Resource).filter(
        Resource.resource_id == result['created_resources'][0].resource_id,
    ).first()
    assert resource.extra_metadata == {'title': 'Finance Report'}

    assert db_session.query(ImportJob).filter(ImportJob.id == job.id).first() is None


@patch('services.import_job_download.fetch', new_callable=AsyncMock)
def test_estimate_rows_returns_cost_estimation_shape(mock_fetch, db_session, repository):
    mock_fetch.return_value = FetchResult(status_code=200, content=b'%PDF-1.4 ok')
    job = ImportJobService.create_job(
        repository.repository_id, io.BytesIO(b"link\nhttp://a.com/x.pdf\n"), 'link', db_session,
    )
    row_id = job.rows[0].id

    from services.import_job_confirm import estimate_rows
    result = estimate_rows(job.id, [row_id], db_session)

    assert 'total_chunks' in result
# NOTE for executor: `estimate_indexing_cost` needs a Silo with an embedding_service
# and (for a real number) a LightRAG-backed vector_db_type to return a non-trivial
# estimate. Check how `tests/integration/test_resource_service.py` (or wherever the
# existing `/resources/estimate` endpoint is tested, if it is) sets up its `repository`
# fixture's silo/embedding_service, and mirror that setup here rather than assuming
# the plain `repository` fixture already has one.


@patch('services.import_job_download.fetch', new_callable=AsyncMock)
def test_confirming_some_rows_keeps_job_open(mock_fetch, db_session, repository):
    mock_fetch.return_value = FetchResult(status_code=200, content=b'%PDF-1.4 ok')
    job = ImportJobService.create_job(
        repository.repository_id,
        io.BytesIO(b"link\nhttp://a.com/x.pdf\nhttp://a.com/y.pdf\n"),
        'link', db_session,
    )
    row_id = job.rows[0].id

    with patch('services.resource_service.ResourceService._index_resources_background', return_value='sid'):
        confirm_rows(job.id, [row_id], db_session)

    assert db_session.query(ImportJob).filter(ImportJob.id == job.id).first() is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_csv_import_confirm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.import_job_confirm'`

- [ ] **Step 3: Write the confirm service**

`backend/services/import_job_confirm.py`:
```python
from sqlalchemy.orm import Session

from models.import_job import ImportJob
from models.import_job_row import ImportJobRow
from models.enums.import_row_status import ImportRowStatus
from services.resource_service import ResourceService


class _StagedFileAdapter:
    """Duck-types the file-like object `_process_single_file` expects
    (`.filename` + `.save(path)`), letting an already-downloaded PDF flow
    through the exact same resource creation path as a manual upload."""

    def __init__(self, staged_path: str, filename: str):
        self.filename = filename
        self._staged_path = staged_path

    def save(self, dest_path: str) -> None:
        import shutil
        shutil.move(self._staged_path, dest_path)


def maybe_close_job(job: ImportJob, db: Session) -> None:
    remaining = db.query(ImportJobRow).filter(
        ImportJobRow.import_job_id == job.id,
        ImportJobRow.status.notin_([ImportRowStatus.CONFIRMED, ImportRowStatus.DISCARDED]),
    ).count()
    if remaining == 0:
        db.delete(job)
        db.commit()


def estimate_rows(import_job_id: int, row_ids: list[int], db: Session) -> dict:
    from services.silo_service import SiloService
    from repositories.repository_repository import RepositoryRepository

    job = db.query(ImportJob).filter(ImportJob.id == import_job_id).first()
    repo = RepositoryRepository.get_by_id(db, job.repository_id)
    if not repo or not getattr(repo, 'silo_id', None):
        raise ValueError("Repository has no silo configured")

    rows = db.query(ImportJobRow).filter(
        ImportJobRow.import_job_id == import_job_id,
        ImportJobRow.id.in_(row_ids),
        ImportJobRow.status == ImportRowStatus.DOWNLOADED,
    ).all()

    extracted_documents = []
    for row in rows:
        base_metadata = {"repository_id": job.repository_id, "name": row.url, "file_type": ".pdf"}
        docs = SiloService.extract_documents_from_file(row.staged_path, ".pdf", base_metadata, split=False)
        for doc in docs:
            extracted_documents.append({
                "content": getattr(doc, "page_content", ""),
                "metadata": getattr(doc, "metadata", {}),
            })

    return SiloService.estimate_indexing_cost(repo.silo_id, extracted_documents, db)


def confirm_rows(import_job_id: int, row_ids: list[int], db: Session) -> dict:
    job = db.query(ImportJob).filter(ImportJob.id == import_job_id).first()
    rows = db.query(ImportJobRow).filter(
        ImportJobRow.import_job_id == import_job_id,
        ImportJobRow.id.in_(row_ids),
        ImportJobRow.status == ImportRowStatus.DOWNLOADED,
    ).all()

    files = [_StagedFileAdapter(row.staged_path, f"{row.url.rsplit('/', 1)[-1] or 'document'}.pdf") for row in rows]
    extra_metadata = {i: (row.row_metadata or {}) for i, row in enumerate(rows)}

    created_resources, failed_files, session_id = ResourceService.create_multiple_resources(
        files=files, repository_id=job.repository_id, db=db, extra_metadata=extra_metadata,
    )

    for row, resource in zip(rows, created_resources):
        row.status = ImportRowStatus.CONFIRMED
        row.resource_id = resource.resource_id
    db.commit()

    from datetime import datetime
    job.last_activity_at = datetime.utcnow()
    db.commit()

    maybe_close_job(job, db)

    return {'created_resources': created_resources, 'failed_files': failed_files, 'session_id': session_id}
```

**Note for the executor:** `files`/`created_resources` are matched positionally, which holds as long as every `_StagedFileAdapter` succeeds (its `.pdf` extension is always in `ResourceService.SUPPORTED_EXTENSIONS` and its filename is never empty, so `_process_single_file` cannot fail for it) — if you touch `_process_single_file`'s failure paths later, re-verify this assumption before relying on positional matching elsewhere.

- [ ] **Step 4: Add the router endpoints (estimate + confirm)**

In `backend/routers/internal/repositories.py`, add the import:
```python
from services.import_job_confirm import confirm_rows, estimate_rows
```
And the endpoints:
```python
@repositories_router.post("/{repository_id}/csv-imports/{import_job_id}/estimate",
                         summary="Estimate LightRAG indexing cost for selected CSV import rows",
                         tags=["CSV Import"],
                         response_model=CostEstimationResponseSchema)
async def estimate_csv_import_rows(
    app_id: int,
    repository_id: int,
    import_job_id: int,
    body: ConfirmDiscardRowsSchema,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    try:
        result = estimate_rows(import_job_id, body.row_ids, db)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return CostEstimationResponseSchema(**result)


@repositories_router.post("/{repository_id}/csv-imports/{import_job_id}/confirm",
                         summary="Confirm selected CSV import rows into Resources",
                         tags=["CSV Import"],
                         response_model=ConfirmRowsResponseSchema)
async def confirm_csv_import_rows(
    app_id: int,
    repository_id: int,
    import_job_id: int,
    body: ConfirmDiscardRowsSchema,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    result = confirm_rows(import_job_id, body.row_ids, db)
    return ConfirmRowsResponseSchema(**result)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/integration/test_csv_import_confirm.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add backend/services/import_job_confirm.py backend/routers/internal/repositories.py \
        tests/integration/test_csv_import_confirm.py
git commit -m "feat(csv-import): add confirm-rows endpoint reusing manual-upload pipeline"
```

---

### Task 8: Discard-rows endpoint

**Files:**
- Create: `backend/services/import_job_discard.py`
- Modify: `backend/routers/internal/repositories.py`
- Test: `tests/integration/test_csv_import_discard.py`

**Interfaces:**
- Consumes: `services.import_job_confirm.maybe_close_job` (Task 7).
- Produces: `discard_rows(import_job_id: int, row_ids: list[int], db: Session) -> None`; `POST /{repository_id}/csv-imports/{import_job_id}/discard`.

- [ ] **Step 1: Write the failing integration test**

`tests/integration/test_csv_import_discard.py`:
```python
import io
import os
from unittest.mock import patch, AsyncMock

from services.crawl.http_fetcher import FetchResult
from services.import_job_service import ImportJobService
from services.import_job_discard import discard_rows
from models.import_job import ImportJob
from models.enums.import_row_status import ImportRowStatus


@patch('services.import_job_download.fetch', new_callable=AsyncMock)
def test_discard_removes_row_and_staged_file(mock_fetch, db_session, repository):
    mock_fetch.return_value = FetchResult(status_code=200, content=b'%PDF-1.4 ok')
    job = ImportJobService.create_job(
        repository.repository_id, io.BytesIO(b"link\nhttp://a.com/x.pdf\n"), 'link', db_session,
    )
    row = job.rows[0]
    staged_path = row.staged_path
    assert os.path.exists(staged_path)

    discard_rows(job.id, [row.id], db_session)

    assert not os.path.exists(staged_path)
    assert db_session.query(ImportJob).filter(ImportJob.id == job.id).first() is None


@patch('services.import_job_download.fetch', new_callable=AsyncMock)
def test_discarding_one_of_two_rows_keeps_job_open(mock_fetch, db_session, repository):
    mock_fetch.return_value = FetchResult(status_code=200, content=b'%PDF-1.4 ok')
    job = ImportJobService.create_job(
        repository.repository_id,
        io.BytesIO(b"link\nhttp://a.com/x.pdf\nhttp://a.com/y.pdf\n"),
        'link', db_session,
    )
    row_id = job.rows[0].id

    discard_rows(job.id, [row_id], db_session)

    remaining = db_session.query(ImportJob).filter(ImportJob.id == job.id).first()
    assert remaining is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_csv_import_discard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.import_job_discard'`

- [ ] **Step 3: Write the discard service**

`backend/services/import_job_discard.py`:
```python
import os
from sqlalchemy.orm import Session

from models.import_job import ImportJob
from models.import_job_row import ImportJobRow
from models.enums.import_row_status import ImportRowStatus
from services.import_job_confirm import maybe_close_job
from utils.logger import get_logger

logger = get_logger(__name__)


def discard_rows(import_job_id: int, row_ids: list[int], db: Session) -> None:
    job = db.query(ImportJob).filter(ImportJob.id == import_job_id).first()
    rows = db.query(ImportJobRow).filter(
        ImportJobRow.import_job_id == import_job_id,
        ImportJobRow.id.in_(row_ids),
    ).all()

    for row in rows:
        if row.staged_path and os.path.exists(row.staged_path):
            try:
                os.remove(row.staged_path)
            except OSError:
                logger.warning(f"Could not remove staged file {row.staged_path}", exc_info=True)
        row.status = ImportRowStatus.DISCARDED
    db.commit()

    from datetime import datetime
    job.last_activity_at = datetime.utcnow()
    db.commit()

    maybe_close_job(job, db)
```

- [ ] **Step 4: Add the router endpoint**

In `backend/routers/internal/repositories.py`:
```python
from services.import_job_discard import discard_rows
```
```python
@repositories_router.post("/{repository_id}/csv-imports/{import_job_id}/discard",
                         summary="Discard selected CSV import rows",
                         tags=["CSV Import"])
async def discard_csv_import_rows(
    app_id: int,
    repository_id: int,
    import_job_id: int,
    body: ConfirmDiscardRowsSchema,
    db: Annotated[Session, Depends(get_db)],
    auth_context: Annotated[AuthContext, Depends(get_current_user_oauth)],
    role: Annotated[AppRole, Depends(require_min_role("editor"))],
):
    discard_rows(import_job_id, body.row_ids, db)
    return {"success": True}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/integration/test_csv_import_discard.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add backend/services/import_job_discard.py backend/routers/internal/repositories.py \
        tests/integration/test_csv_import_discard.py
git commit -m "feat(csv-import): add discard-rows endpoint"
```

---

### Task 9: Abandoned-import purge (extends the existing cleanup worker)

**Files:**
- Modify: `backend/services/file_cleanup_worker.py`
- Test: `tests/unit/services/test_import_job_purge.py`

**Interfaces:**
- Consumes: `ImportJob`, `ImportJobRow` (Task 1).
- Produces: `_purge_abandoned_import_jobs() -> int` (module-private, called from `_run_one_sweep_sync`).

- [ ] **Step 1: Write the failing unit test**

`tests/unit/services/test_import_job_purge.py`:
```python
import os
from datetime import datetime, timedelta

from services.file_cleanup_worker import _purge_abandoned_import_jobs
from models.import_job import ImportJob
from models.import_job_row import ImportJobRow
from models.enums.import_row_status import ImportRowStatus


def test_purges_job_with_no_activity_for_14_days(tmp_path, db_session, repository):
    staged = tmp_path / "staged.pdf"
    staged.write_bytes(b"%PDF-1.4 old")

    job = ImportJob(
        repository_id=repository.repository_id,
        last_activity_at=datetime.utcnow() - timedelta(days=15),
    )
    db_session.add(job)
    db_session.commit()
    row = ImportJobRow(import_job_id=job.id, url='http://a.com/x.pdf', status=ImportRowStatus.DOWNLOADED,
                        staged_path=str(staged))
    db_session.add(row)
    db_session.commit()

    removed = _purge_abandoned_import_jobs()

    assert removed == 1
    assert not staged.exists()
    assert db_session.query(ImportJob).filter(ImportJob.id == job.id).first() is None


def test_does_not_purge_job_with_recent_activity(tmp_path, db_session, repository):
    job = ImportJob(
        repository_id=repository.repository_id,
        last_activity_at=datetime.utcnow() - timedelta(days=2),
    )
    db_session.add(job)
    db_session.commit()

    removed = _purge_abandoned_import_jobs()

    assert removed == 0
    assert db_session.query(ImportJob).filter(ImportJob.id == job.id).first() is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/services/test_import_job_purge.py -v`
Expected: FAIL with `ImportError: cannot import name '_purge_abandoned_import_jobs'`

- [ ] **Step 3: Add the purge step**

In `backend/services/file_cleanup_worker.py`, add near the top-level helpers (after `_prune_empty_dirs`, before `_run_one_sweep_sync`):
```python
IMPORT_JOB_TTL_DAYS = 14


def _purge_abandoned_import_jobs() -> int:
    """Delete ImportJobs with no confirm/discard/retry activity for
    IMPORT_JOB_TTL_DAYS. Cascade deletes their still-open ImportJobRows;
    staged PDFs are removed explicitly first since they live outside the DB."""
    from datetime import datetime, timedelta
    from db.database import SessionLocal
    from models.import_job import ImportJob
    from models.import_job_row import ImportJobRow

    cutoff = datetime.utcnow() - timedelta(days=IMPORT_JOB_TTL_DAYS)
    db = SessionLocal()
    removed = 0
    try:
        jobs = db.query(ImportJob).filter(ImportJob.last_activity_at < cutoff).all()
        for job in jobs:
            rows = db.query(ImportJobRow).filter(ImportJobRow.import_job_id == job.id).all()
            for row in rows:
                if row.staged_path and os.path.exists(row.staged_path):
                    try:
                        os.remove(row.staged_path)
                    except OSError:
                        logger.warning(
                            "file_cleanup_worker[import_job]: could not remove %s", row.staged_path,
                        )
            db.delete(job)
            removed += 1
        db.commit()
    except Exception:
        logger.error("file_cleanup_worker[import_job]: purge failed", exc_info=True)
        db.rollback()
    finally:
        db.close()
    return removed
```
Then call it from `_run_one_sweep_sync()`, alongside the existing three sweeps:
```python
    imports_purged = _purge_abandoned_import_jobs()

    if ephemeral_removed or uploads_removed or persistent_removed or imports_purged:
        logger.info(
            "file_cleanup_worker: removed %d ephemeral, %d upload, %d persistent file(s), %d abandoned import job(s)",
            ephemeral_removed, uploads_removed, persistent_removed, imports_purged,
        )
```
(replace the existing final `if`/`logger.info` block rather than duplicating it)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/services/test_import_job_purge.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/services/file_cleanup_worker.py tests/unit/services/test_import_job_purge.py
git commit -m "feat(csv-import): purge abandoned import jobs in the existing cleanup sweep"
```

---

### Task 10: Frontend API client methods and types

**Files:**
- Modify: `frontend/src/services/api.ts`

**Interfaces:**
- Produces (used by Tasks 11-12): 
  - `apiService.previewCsvImport(appId, repositoryId, file) -> Promise<{headers: string[]}>`
  - `apiService.createCsvImport(appId, repositoryId, file, linkColumn) -> Promise<ImportJob>`
  - `apiService.getActiveCsvImport(appId, repositoryId) -> Promise<ImportJob | null>`
  - `apiService.getCsvImport(appId, repositoryId, importJobId) -> Promise<ImportJob>`
  - `apiService.retryCsvImportRow(appId, repositoryId, importJobId, rowId) -> Promise<ImportJob>`
  - `apiService.estimateCsvImportRows(appId, repositoryId, importJobId, rowIds) -> Promise<CostEstimationResult>` (reuses the `CostEstimationResult` type already defined in `RepositoryDetailPage.tsx:53-70` — import it rather than redefining it)
  - `apiService.confirmCsvImportRows(appId, repositoryId, importJobId, rowIds) -> Promise<{created_resources, failed_files, session_id}>`
  - `apiService.discardCsvImportRows(appId, repositoryId, importJobId, rowIds) -> Promise<void>`
  - `ImportJobRow { id: number; url: string; row_metadata: Record<string, string>; status: 'PENDING'|'DOWNLOADING'|'DOWNLOADED'|'FAILED'|'CONFIRMED'|'DISCARDED'; failure_reason: string | null; resource_id: number | null }`
  - `ImportJob { id: number; repository_id: number; status: 'DOWNLOADING'|'REVIEW'; source_filename: string | null; link_column: string | null; rows: ImportJobRow[]; counts: { total: number; pending: number; downloading: number; downloaded: number; failed: number; confirmed: number; discarded: number } }`

- [ ] **Step 1: Add the types and methods**

In `frontend/src/services/api.ts`, add near the existing `uploadResources`/`estimateUploadResources` methods (following that same fetch-with-FormData shape, and the plain-JSON-body shape used elsewhere in the file for POST-with-body calls):

```typescript
export interface ImportJobRow {
  id: number;
  url: string;
  row_metadata: Record<string, string>;
  status: 'PENDING' | 'DOWNLOADING' | 'DOWNLOADED' | 'FAILED' | 'CONFIRMED' | 'DISCARDED';
  failure_reason: string | null;
  resource_id: number | null;
}

export interface ImportJobCounts {
  total: number;
  pending: number;
  downloading: number;
  downloaded: number;
  failed: number;
  confirmed: number;
  discarded: number;
}

export interface ImportJob {
  id: number;
  repository_id: number;
  status: 'DOWNLOADING' | 'REVIEW';
  source_filename: string | null;
  link_column: string | null;
  rows: ImportJobRow[];
  counts: ImportJobCounts;
}
```

```typescript
  async previewCsvImport(appId: number, repositoryId: number, file: File): Promise<{ headers: string[] }> {
    const formData = new FormData();
    formData.append('file', file);
    return this.request(`/internal/apps/${appId}/repositories/${repositoryId}/csv-imports/preview`, {
      method: 'POST',
      body: formData,
    });
  },

  async createCsvImport(appId: number, repositoryId: number, file: File, linkColumn: string): Promise<ImportJob> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('link_column', linkColumn);
    return this.request(`/internal/apps/${appId}/repositories/${repositoryId}/csv-imports`, {
      method: 'POST',
      body: formData,
    });
  },

  async getActiveCsvImport(appId: number, repositoryId: number): Promise<ImportJob | null> {
    return this.request(`/internal/apps/${appId}/repositories/${repositoryId}/csv-imports/active`);
  },

  async getCsvImport(appId: number, repositoryId: number, importJobId: number): Promise<ImportJob> {
    return this.request(`/internal/apps/${appId}/repositories/${repositoryId}/csv-imports/${importJobId}`);
  },

  async retryCsvImportRow(appId: number, repositoryId: number, importJobId: number, rowId: number): Promise<ImportJob> {
    return this.request(
      `/internal/apps/${appId}/repositories/${repositoryId}/csv-imports/${importJobId}/rows/${rowId}/retry`,
      { method: 'POST' },
    );
  },

  async estimateCsvImportRows(appId: number, repositoryId: number, importJobId: number, rowIds: number[]) {
    return this.request(
      `/internal/apps/${appId}/repositories/${repositoryId}/csv-imports/${importJobId}/estimate`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ row_ids: rowIds }) },
    );
  },

  async confirmCsvImportRows(appId: number, repositoryId: number, importJobId: number, rowIds: number[]) {
    return this.request(
      `/internal/apps/${appId}/repositories/${repositoryId}/csv-imports/${importJobId}/confirm`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ row_ids: rowIds }) },
    );
  },

  async discardCsvImportRows(appId: number, repositoryId: number, importJobId: number, rowIds: number[]): Promise<void> {
    await this.request(
      `/internal/apps/${appId}/repositories/${repositoryId}/csv-imports/${importJobId}/discard`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ row_ids: rowIds }) },
    );
  },
```

(`this.request(endpoint, options)` — `ApiService.request()` at `frontend/src/services/api.ts:155` — is the real internal helper every other method in this file uses; it injects auth headers and handles 401 refresh, and correctly special-cases `FormData` bodies for multipart calls. The code above matches `uploadResources`'s existing calling convention exactly.)

- [ ] **Step 2: Export `CostEstimationResult` for reuse**

`CostEstimationResult` is currently a local interface inside `RepositoryDetailPage.tsx:53-70`. Add `export` to its declaration (`export interface CostEstimationResult { ... }`) so Task 12 can `import { CostEstimationResult } from '../pages/RepositoryDetailPage'` without duplicating the type. If the project's lint rules forbid importing types from a page component, move the interface to `frontend/src/services/api.ts` instead and update `RepositoryDetailPage.tsx`'s own usage to import it from there — check `.eslintrc`/`eslint.config.*` for such a rule before choosing.

- [ ] **Step 3: Manual verification**

Run: `cd frontend && npm run build:lib`
Expected: TypeScript compiles with no errors (this task adds no UI, so there's nothing to click through yet — Task 11 is the first point real interaction is possible).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/services/api.ts frontend/src/pages/RepositoryDetailPage.tsx
git commit -m "feat(csv-import): add frontend API client methods for CSV import"
```

---

### Task 11: CSV import modal (mapping + downloading steps)

**Files:**
- Create: `frontend/src/components/import/CsvImportStepper.tsx`
- Modify: `frontend/src/pages/RepositoryDetailPage.tsx`

**Interfaces:**
- Consumes: `StepperContainer`/`StepDefinition` (`frontend/src/components/ui/Stepper.tsx`), `apiService.{previewCsvImport, createCsvImport}` (Task 10), `Modal` (`frontend/src/components/ui/Modal.tsx`).
- Produces (used by Task 12): `<CsvImportStepper appId repositoryId isOpen onClose onImportStarted={(job: ImportJob) => void} />` — closes itself and calls `onImportStarted` once `createCsvImport` returns (download continues server-side; Task 12's banner takes over from there).

- [ ] **Step 1: Write the component**

`frontend/src/components/import/CsvImportStepper.tsx`:
```tsx
import { useState } from 'react';
import { Modal } from '../ui/Modal';
import { StepperContainer, StepDefinition } from '../ui/Stepper';
import { apiService, ImportJob } from '../../services/api';

interface CsvImportStepperProps {
  appId: number;
  repositoryId: number;
  isOpen: boolean;
  onClose: () => void;
  onImportStarted: (job: ImportJob) => void;
}

const STEPS: StepDefinition[] = [
  { id: 'map', label: 'Upload & map columns' },
  { id: 'starting', label: 'Starting import' },
];

export function CsvImportStepper({ appId, repositoryId, isOpen, onClose, onImportStarted }: CsvImportStepperProps) {
  const [step, setStep] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [headers, setHeaders] = useState<string[]>([]);
  const [linkColumn, setLinkColumn] = useState<string>('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reset = () => {
    setStep(0);
    setFile(null);
    setHeaders([]);
    setLinkColumn('');
    setError(null);
  };

  const handleFileSelected = async (selected: File) => {
    setFile(selected);
    setError(null);
    try {
      const { headers: parsedHeaders } = await apiService.previewCsvImport(appId, repositoryId, selected);
      setHeaders(parsedHeaders);
      setLinkColumn(parsedHeaders[0] ?? '');
    } catch {
      setError('Could not read this CSV file.');
    }
  };

  const handleStart = async () => {
    if (!file || !linkColumn) return;
    setIsSubmitting(true);
    setStep(1);
    try {
      const job = await apiService.createCsvImport(appId, repositoryId, file, linkColumn);
      onImportStarted(job);
      reset();
      onClose();
    } catch (err: any) {
      setError(err?.message ?? 'Could not start the import.');
      setStep(0);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={() => { reset(); onClose(); }} title="Import from CSV" size="medium">
      <StepperContainer
        steps={STEPS}
        currentStep={step}
        onNext={handleStart}
        onBack={() => setStep(0)}
        onCancel={() => { reset(); onClose(); }}
        nextLabel="Start import"
        nextDisabled={!file || !linkColumn || isSubmitting}
        showBack={false}
        isSubmitting={isSubmitting}
      >
        {step === 0 && (
          <div className="space-y-4">
            <input
              type="file"
              accept=".csv"
              onChange={(e) => e.target.files?.[0] && handleFileSelected(e.target.files[0])}
            />
            {headers.length > 0 && (
              <div>
                <label className="block text-sm font-medium mb-1">Link column</label>
                <select
                  value={linkColumn}
                  onChange={(e) => setLinkColumn(e.target.value)}
                  className="border rounded px-2 py-1 w-full"
                >
                  {headers.map((h) => (
                    <option key={h} value={h}>{h}</option>
                  ))}
                </select>
                <p className="text-xs text-gray-500 mt-2">
                  Any column you don't map manually will be imported as metadata, using its header row as the field name.
                </p>
              </div>
            )}
            {error && <p className="text-sm text-red-600">{error}</p>}
          </div>
        )}
        {step === 1 && <p>Starting the import…</p>}
      </StepperContainer>
    </Modal>
  );
}
```

**Note for the executor:** `StepperContainer`'s exact prop names (`onNext`, `onBack`, `nextLabel`, etc.) and `Modal`'s exact prop names must be double-checked against the real files before wiring this up — the interfaces block above lists what the codebase exploration found at plan-writing time, but re-read both files first since this is the first task that actually renders them together.

- [ ] **Step 2: Wire the "Import from CSV" button into the Repository page**

In `frontend/src/pages/RepositoryDetailPage.tsx`, near the existing "Upload files" button (around the hidden file input at line ~893), add a sibling button and the modal:
```tsx
import { CsvImportStepper } from '../components/import/CsvImportStepper';
```
```tsx
  const [showCsvImportModal, setShowCsvImportModal] = useState(false);
```
```tsx
<button onClick={() => setShowCsvImportModal(true)}>Import from CSV</button>

<CsvImportStepper
  appId={Number.parseInt(appId!)}
  repositoryId={Number.parseInt(repositoryId!)}
  isOpen={showCsvImportModal}
  onClose={() => setShowCsvImportModal(false)}
  onImportStarted={(job) => setActiveImportJob(job)}
/>
```
(`activeImportJob` state and its setter are introduced properly in Task 12, which owns the banner — add a minimal `useState<ImportJob | null>(null)` here as a placeholder wired-up point Task 12 will build on.)

- [ ] **Step 3: Manual verification**

Run: `cd frontend && npm run dev`, open a Repository, click "Import from CSV", pick a small CSV file.
Expected: headers populate the dropdown, the auto-metadata remark is visible, clicking "Start import" closes the modal without a console error (the backend call will 202 given Task 4-9 are done).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/import/CsvImportStepper.tsx frontend/src/pages/RepositoryDetailPage.tsx
git commit -m "feat(csv-import): add CSV mapping/start modal"
```

---

### Task 12: Persistent banner + review table

**Files:**
- Create: `frontend/src/hooks/useCsvImportPolling.ts`
- Create: `frontend/src/components/import/CsvImportBanner.tsx`
- Create: `frontend/src/components/import/CsvImportReviewModal.tsx`
- Modify: `frontend/src/pages/RepositoryDetailPage.tsx`

**Interfaces:**
- Consumes: `apiService.{getActiveCsvImport, getCsvImport, retryCsvImportRow, estimateCsvImportRows, confirmCsvImportRows, discardCsvImportRows}` and the exported `CostEstimationResult` type (Task 10), `DataTable`/`StatusBadge`/`Alert` (existing), `useIngestionProgress`/`IngestionProgressBar` (existing, for the final indexing step after confirm), the page's existing `isLightRagRepository` boolean (`RepositoryDetailPage.tsx`).

- [ ] **Step 1: Write the polling hook**

`frontend/src/hooks/useCsvImportPolling.ts` (mirrors the `ACTIVE_STATUSES`/`setInterval` shape of `CrawlJobProgressPanel.tsx`):
```typescript
import { useEffect, useRef, useState } from 'react';
import { apiService, ImportJob } from '../services/api';

const POLL_INTERVAL_MS = 2000;

export function useCsvImportPolling(appId: number, repositoryId: number) {
  const [job, setJob] = useState<ImportJob | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let cancelled = false;
    apiService.getActiveCsvImport(appId, repositoryId).then((active) => {
      if (!cancelled) setJob(active);
    });
    return () => { cancelled = true; };
  }, [appId, repositoryId]);

  useEffect(() => {
    if (!job || job.status !== 'DOWNLOADING') {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }
    const poll = async () => {
      try {
        const updated = await apiService.getCsvImport(appId, repositoryId, job.id);
        setJob(updated);
      } catch {
        // Silently ignore poll errors; will retry next interval.
      }
    };
    intervalRef.current = setInterval(() => { void poll(); }, POLL_INTERVAL_MS);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [appId, repositoryId, job?.id, job?.status]);

  return { job, setJob };
}
```

- [ ] **Step 2: Write the banner**

`frontend/src/components/import/CsvImportBanner.tsx`:
```tsx
import { Alert } from '../ui/Alert';
import { ImportJob } from '../../services/api';

interface CsvImportBannerProps {
  job: ImportJob;
  onReview: () => void;
}

export function CsvImportBanner({ job, onReview }: CsvImportBannerProps) {
  const { total, downloaded, failed } = job.counts;

  if (job.status === 'DOWNLOADING') {
    const done = downloaded + failed;
    return <Alert type="info" message={`Import in progress: ${done}/${total} completed`} />;
  }

  return (
    <Alert
      type="warning"
      message={
        <span>
          {`Import ready: ${downloaded} OK, ${failed} failed — `}
          <button className="underline" onClick={onReview}>review</button>
        </span>
      }
    />
  );
}
```

- [ ] **Step 3: Write the review modal**

`frontend/src/components/import/CsvImportReviewModal.tsx`:
```tsx
import { useMemo, useState } from 'react';
import { Modal } from '../ui/Modal';
import { DataTable, TableColumn } from '../ui/DataTable';
import { StatusBadge } from '../StatusBadge';
import { apiService, ImportJob, ImportJobRow } from '../../services/api';
import { useIngestionProgress } from '../../hooks/useIngestionProgress';
import { IngestionProgressBar } from '../ui/IngestionProgressBar';
import { CostEstimationResult } from '../../pages/RepositoryDetailPage';

interface CsvImportReviewModalProps {
  appId: number;
  repositoryId: number;
  isLightRagRepository: boolean;
  job: ImportJob;
  isOpen: boolean;
  onClose: () => void;
  onJobUpdated: (job: ImportJob | null) => void;
}

export function CsvImportReviewModal({ appId, repositoryId, isLightRagRepository, job, isOpen, onClose, onJobUpdated }: CsvImportReviewModalProps) {
  const [selected, setSelected] = useState<Set<number>>(
    () => new Set(job.rows.filter((r) => r.status === 'DOWNLOADED').map((r) => r.id)),
  );
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [costEstimate, setCostEstimate] = useState<CostEstimationResult | null>(null);
  const ingestion = useIngestionProgress(appId, repositoryId, sessionId);

  const toggle = (rowId: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(rowId) ? next.delete(rowId) : next.add(rowId);
      return next;
    });
  };

  const handleRetry = async (rowId: number) => {
    const updated = await apiService.retryCsvImportRow(appId, repositoryId, job.id, rowId);
    onJobUpdated(updated);
  };

  const doConfirm = async () => {
    setCostEstimate(null);
    const result = await apiService.confirmCsvImportRows(appId, repositoryId, job.id, Array.from(selected));
    if (result.session_id) setSessionId(result.session_id);
    const remainingRows = job.rows.filter((r) => !selected.has(r.id));
    if (remainingRows.length === 0) {
      onJobUpdated(null);
      onClose();
    } else {
      const updated = await apiService.getCsvImport(appId, repositoryId, job.id).catch(() => null);
      onJobUpdated(updated);
    }
  };

  // Same gate as manual upload's LightRAG flow (RepositoryDetailPage.tsx:429-467):
  // show the cost/time estimate first and require a second confirmation before
  // any Resource is actually created. Rendered as its own compact panel here
  // rather than reusing RepositoryDetailPage's inline modal JSX, since that
  // block is not an extracted component and is keyed to client-side File[]
  // input that CSV-imported rows don't have (their PDFs are already staged
  // server-side) — see estimateCsvImportRows in Task 10/7.
  const handleConfirmClick = async () => {
    if (!isLightRagRepository) {
      await doConfirm();
      return;
    }
    const estimate = await apiService.estimateCsvImportRows(appId, repositoryId, job.id, Array.from(selected));
    setCostEstimate(estimate);
  };

  const handleDiscardUnselected = async () => {
    const unselected = job.rows.filter((r) => !selected.has(r.id)).map((r) => r.id);
    if (unselected.length === 0) return;
    await apiService.discardCsvImportRows(appId, repositoryId, job.id, unselected);
    const updated = await apiService.getCsvImport(appId, repositoryId, job.id).catch(() => null);
    onJobUpdated(updated);
  };

  const columns = useMemo<TableColumn<ImportJobRow>[]>(() => [
    {
      key: 'select',
      header: '',
      render: (row) => (
        <input type="checkbox" checked={selected.has(row.id)} onChange={() => toggle(row.id)}
               disabled={row.status !== 'DOWNLOADED'} />
      ),
    },
    { key: 'url', header: 'Link', render: (row) => row.url },
    {
      key: 'metadata', header: 'Metadata',
      render: (row) => Object.entries(row.row_metadata).map(([k, v]) => `${k}: ${v}`).join(', '),
    },
    {
      key: 'status', header: 'Status',
      render: (row) => row.status === 'FAILED'
        ? <StatusBadge status={`Failed: ${row.failure_reason}`} />
        : <StatusBadge status={row.status} />,
    },
    {
      key: 'actions', header: '',
      render: (row) => row.status === 'FAILED'
        ? <button onClick={() => handleRetry(row.id)}>Retry</button>
        : null,
    },
  ], [selected]);

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Review CSV import" size="xlarge">
      <DataTable
        data={job.rows.filter((r) => r.status !== 'CONFIRMED' && r.status !== 'DISCARDED')}
        columns={columns}
        keyExtractor={(row) => row.id}
      />
      {sessionId && (
        <IngestionProgressBar
          appId={appId}
          repositoryId={repositoryId}
          sessionId={sessionId}
          onComplete={() => setSessionId(null)}
        />
      )}
      {costEstimate && (
        <div className="border rounded p-4 mt-4 space-y-2">
          <h3 className="font-medium">Confirm LightRAG ingestion</h3>
          <p className="text-sm">Chunks: {costEstimate.total_chunks}</p>
          <p className="text-sm">Estimated embedding tokens: {costEstimate.estimated_embedding_tokens ?? '—'}</p>
          {costEstimate.estimated_cost_min != null && costEstimate.estimated_cost_max != null && (
            <p className="text-sm">
              Estimated cost: {costEstimate.estimated_cost_min}–{costEstimate.estimated_cost_max} {costEstimate.currency}
            </p>
          )}
          {costEstimate.warnings.map((w) => <p key={w} className="text-sm text-amber-600">{w}</p>)}
          <div className="flex justify-end gap-2">
            <button onClick={() => setCostEstimate(null)}>Cancel</button>
            <button onClick={doConfirm}>Confirm ingestion</button>
          </div>
        </div>
      )}
      <div className="flex justify-end gap-2 mt-4">
        <button onClick={handleDiscardUnselected}>Discard unselected</button>
        <button onClick={handleConfirmClick} disabled={selected.size === 0 || !!(ingestion.progress && !ingestion.isComplete)}>
          Ingest selected
        </button>
      </div>
    </Modal>
  );
}
```

**Note for the executor:** `TableColumn` generic shape, `Alert`'s `message` prop accepting `ReactNode`, and `StatusBadge`'s exact prop contract must be re-checked against the real files — write this component only after reading `DataTable.tsx`, `Alert.tsx`, and `StatusBadge.tsx` in full, adjusting names/shapes to match exactly.

- [ ] **Step 4: Wire the banner + review modal into the Repository page**

In `frontend/src/pages/RepositoryDetailPage.tsx`, replace the placeholder `activeImportJob` state from Task 11 with the real hook, and render banner + review modal:
```tsx
import { useCsvImportPolling } from '../hooks/useCsvImportPolling';
import { CsvImportBanner } from '../components/import/CsvImportBanner';
import { CsvImportReviewModal } from '../components/import/CsvImportReviewModal';
```
```tsx
  const { job: activeImportJob, setJob: setActiveImportJob } = useCsvImportPolling(
    Number.parseInt(appId!), Number.parseInt(repositoryId!),
  );
  const [showReviewModal, setShowReviewModal] = useState(false);
```
```tsx
{activeImportJob && (
  <CsvImportBanner job={activeImportJob} onReview={() => setShowReviewModal(true)} />
)}

{activeImportJob && (
  <CsvImportReviewModal
    appId={Number.parseInt(appId!)}
    repositoryId={Number.parseInt(repositoryId!)}
    isLightRagRepository={isLightRagRepository}
    job={activeImportJob}
    isOpen={showReviewModal}
    onClose={() => setShowReviewModal(false)}
    onJobUpdated={(updated) => { setActiveImportJob(updated); if (!updated) setShowReviewModal(false); }}
  />
)}
```
And pass `onImportStarted={(job) => setActiveImportJob(job)}` (from Task 11's `CsvImportStepper`) into the same `setActiveImportJob` setter now sourced from `useCsvImportPolling`.

- [ ] **Step 5: Manual verification**

Run: `cd frontend && npm run dev`. Full flow: import a CSV with a few working and a few broken links → watch the banner progress → click it → see the review table with correct statuses → retry a failed row → confirm the rest → verify the confirmed PDFs appear as normal Resources in the repository and the banner disappears. Also verify: navigating away from the Repository page and back while a download is still running still shows the banner (proves the job survived without depending on component state).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/hooks/useCsvImportPolling.ts frontend/src/components/import/CsvImportBanner.tsx \
        frontend/src/components/import/CsvImportReviewModal.tsx frontend/src/pages/RepositoryDetailPage.tsx
git commit -m "feat(csv-import): add persistent banner and review table"
```

---

## Post-implementation checklist

- [ ] Full backend suite: `pytest -v --cov=backend --cov-report=term-missing` — no regressions outside the new files.
- [ ] `alembic downgrade -1` then `alembic upgrade head` one more time on a clean test DB to confirm the migration is still reversible after all later tasks landed.
- [ ] `cd frontend && npm run lint && npm run build:lib` — no new lint errors, library build succeeds.
- [ ] Re-read the design spec's Acceptance Criteria section end-to-end against the running feature.
