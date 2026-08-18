# Manual QA — CSV → PDF Import

Step-by-step checklist to verify the CSV import feature by hand in a browser. Automated coverage lives in `tests/unit/services/test_csv_import_parser.py`, `tests/unit/services/test_resource_extra_metadata.py`, and `tests/integration/test_csv_import_*.py` / `test_import_job_*.py` — this doc is for the things those tests can't see (the actual UI).

Design doc: `docs/superpowers/specs/2026-08-17-csv-pdf-import-design.md`
Plan: `docs/superpowers/plans/2026-08-17-csv-pdf-import.md`

## 0. Run your own code, not the Docker image

The Docker Compose stack runs a prebuilt image, not your local changes. Run the app from source instead:

```bash
# Backend
poetry install
export DATABASE_USER=... DATABASE_PASSWORD=... DATABASE_HOST=... DATABASE_PORT=... DATABASE_NAME=...  # alembic reads these, NOT SQLALCHEMY_DATABASE_URI
alembic upgrade head
uvicorn backend.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend && npm run dev   # localhost:5173, VITE_API_BASE_URL=http://localhost:8000
```

You need an App with at least one Repository already created (LightRAG-backed or not) to test against.

## 1. The button appears

Open a Repository page → **"Import from CSV"** should be visible next to "Upload Files" / "Upload Media".

## 2. Column mapping

Use the ready-made sample CSV at [`csv-pdf-import-sample.csv`](csv-pdf-import-sample.csv) (columns: `link`, `title`, `category`, `department`). Its rows are deliberately mixed so one file exercises every case below:

| Row | Link | Purpose |
|---|---|---|
| 1–2 | same real 1-page PDF, same metadata | merges into **one** review row |
| 3 | same PDF as row 1, different metadata | stays a **separate** row, but the file is downloaded only once |
| 4–5 | two other real 1-page PDFs | normal successful rows |
| 6 | nonexistent domain | fails with `NOT_FOUND` |
| 7 | empty link | silently skipped before the import job is even created |

The real links (rows 1–5) were downloaded and verified to be genuine, reachable, exactly-1-page PDFs before writing this doc — no need to re-check them, but if one ever goes offline, swap it for another small public PDF.

- Upload it → the header row (`link, title, category, department`) should populate the link-column dropdown.
- Select `link` as the link column.
- The English remark about automatic metadata mapping should be visible.

## 3. Background download

Click "Start import" → the modal closes and the **banner** appears ("Import in progress: X/Y"). Navigate away and back — the banner must still be there (it's backend-persisted, not tied to the modal).

## 4. Deduplication

The first two CSV rows are identical (same link + metadata), so the review table should show **one row** for them, not two.

## 5. Review table

Once the banner says "ready to review", open it:

- The real PDF row → status `DOWNLOADED`, checkbox **pre-checked**.
- The nonexistent-domain row → status `FAILED: NOT_FOUND` (or similar), checkbox **unchecked**.

## 6. Retry

Click "Retry" on the failed row and confirm the status updates in place — no page reload, no CSV re-upload.

## 7. Confirm

Check the OK row → "Ingest selected".

- If the repository is LightRAG-backed: the cost/time estimate panel should appear first.
- After confirming: the normal indexing progress bar should start (the same one used for a manual file upload).
- The document should show up in the repository's Resource list like any other file.

## 8. Metadata on the chunk

Find that document via the Silo's Playground/search — the results should show the CSV's `title`/`category` metadata alongside the usual fields (`resource_id`, `repository_id`, …).

## 9. Import closes

Once that row is `CONFIRMED` and no rows remain pending, the banner should disappear on its own.

## 10. Discard (optional, with a fresh import)

Repeat with a new CSV, but click "Discard unselected" in the review table instead of confirming — those rows must not create any Resource, and if they were the last pending rows, the banner disappears too.

---

Not practically checkable by hand: the 14-day abandoned-import purge (background logic, covered by `tests/integration/test_import_job_purge.py`).
