# Azure Blob Ingestion (validation tool)

## What this is

A one-shot ingestion endpoint that lists the blobs of an Azure Blob Storage
container and ingests them into an existing Repository through the **exact
same pipeline a manual upload uses** (`ResourceService.create_multiple_resources`)
— PDF extraction, chunking, the per-silo indexing lock, background indexing
and SSE progress all come for free.

It was built to validate LightRAG with real-world content (a container of
supplier PDFs) without first building a content-source product. It is
explicitly **not** one:

- No saved configuration, no CRUD, no "sources" UI. Each call carries
  everything it needs in its body.
- No scheduled sync. You re-run it by hand; idempotency makes that safe.
- No S3/GCS/MinIO. Azure Blob only, `ANONYMOUS` and `SAS_TOKEN` auth only
  (no `CONNECTION_STRING`/`ACCOUNT_KEY` in this version).
- Not exposed on `/public/v1` or `/mcp/v1`. Internal endpoints only.

## How to use it

```
POST /internal/apps/{app_id}/repositories/{repository_id}/ingest-azure-blobs
Role: editor or above
```

Body:

```json
{
  "account_url": "https://etiquetas.blob.core.windows.net",
  "container": "etiquetas",
  "prefix": "2026/",
  "auth_mode": "ANONYMOUS",
  "sas_token": null,
  "file_extension_filters": [".pdf"]
}
```

| Field | Required | Notes |
|-------|----------|-------|
| `account_url` | yes | `https://…` only; host must end in `BLOB_ALLOWED_HOST_SUFFIXES` (default `.blob.core.windows.net`) |
| `container` | yes | |
| `prefix` | no | Only blobs whose name starts with this are listed |
| `auth_mode` | no | `ANONYMOUS` (default) or `SAS_TOKEN` |
| `sas_token` | only if `SAS_TOKEN` | With or without a leading `?` |
| `file_extension_filters` | no | Defaults to every extension the pipeline supports (`.pdf`, `.docx`, `.txt`, `.md`) |

Response — `202 Accepted`:

```json
{
  "queued": 12,
  "skipped_unchanged": 30,
  "skipped_unsupported": 0,
  "failed": 0,
  "session_id": "a092a1b5-…"
}
```

`session_id` feeds the repository's existing progress endpoint:
`GET /internal/apps/{app_id}/repositories/{repository_id}/ingestion-progress/{session_id}`.
`queued`/`failed` count the *first batch only* (processed synchronously so a
real `session_id` can be returned); larger runs continue in the background
under that same `session_id`, and the whole run is summarized in one
structured log line (`Azure blob ingestion run finished: …`).

### Errors

| Status | `code` | Meaning |
|--------|--------|---------|
| 422 | `SCHEME_NOT_ALLOWED`, `HOST_NOT_ALLOWED`, `IP_LITERAL_NOT_ALLOWED`, `PRIVATE_ADDRESS_NOT_ALLOWED`, `DNS_RESOLUTION_FAILED`, `MALFORMED_URL` | Anti-SSRF rejection of `account_url` |
| 422 | `AUTH_FAILED`, `CONTAINER_NOT_FOUND`, `LISTING_FORBIDDEN`, `INVALID_CONFIG` | The request itself is wrong (expired SAS, missing container…) |
| 422 | — | Invalid body (no silo on the repository, bad auth combination, unsupported extension filter) |
| 409 | — | The repository's silo is already indexing, or another Azure blob run is active for this repository — wait for it to finish |
| 502 | `ACCOUNT_UNREACHABLE`, `TIMEOUT` | Upstream connectivity problems, not a caller error |
| 500 | — | Unexpected failure; the message is generic (never the raw exception) |

## Idempotency

Safe to re-run against the same container as often as you like:

- Each ingested blob leaves `(account_url, container, blob_name, etag)` in the
  `Resource.extra_metadata` of its row.
- A re-run lists the container again and diffs against those rows: unchanged
  blobs are counted in `skipped_unchanged` and never downloaded again; blobs
  whose ETag changed are replaced; blobs with no ETag are always re-ingested
  (worst case is redundant work, never a missed update).
- Replacements are **index-then-swap**: the new content is created and handed
  to the indexer *before* the superseded `Resource` row is deleted (DB row and
  vectors only — the on-disk file stays, because the replacement reuses the
  same path).
- Runs are serialized per repository by a PostgreSQL advisory lock; two
  concurrent triggers cannot both pass the diff.

## Security notes

- The `sas_token` is request-scoped only: never persisted (there is no table
  to persist it to), never returned in a response, and never logged — every
  error message goes through a sanitizer that strips query strings and
  redacts `sig=`/`se=`/`SharedAccessSignature=` values.
- `account_url` is validated against the anti-SSRF allowlist before any
  outbound call, again inside the SDK wrapper right before each call, and the
  host must resolve exclusively to globally routable addresses (blocks
  `169.254.169.254` and friends). `http://` and IP literals are rejected.
- Blobs are streamed to a per-run staging directory under
  `TMP_BASE_FOLDER/blob_ingest_staging/` and cleaned up after each batch; a
  crash backstop sweep covers the directory (`file_cleanup_worker`).
- A blob larger than the App's `max_file_size_mb` is rejected before its bytes
  are downloaded and counted in `failed`.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `BLOB_ALLOWED_HOST_SUFFIXES` | `.blob.core.windows.net` | Comma-separated allowlist for `account_url` hosts |
| `BLOB_INGEST_BATCH_SIZE` | `20` | Blobs per ingestion batch |
| `BLOB_INGEST_DOWNLOAD_CONCURRENCY` | `4` | Concurrent downloads per batch |
| `BLOB_INGEST_DOWNLOAD_TIMEOUT_SECONDS` | `120` | SDK connect/read timeout |
| `BLOB_LIST_PAGE_SIZE` | `200` | Blobs requested per listing page |
| `BLOB_LIST_MAX_ITEMS` | `5000` | Hard ceiling per run |

## LightRAG validation result

The reason this feature exists: `tests/integration/test_blob_ingest_lightrag_validation.py`
proves the full chain on a real LightRAG backend (lightrag-hku 1.5.6, Neo4j +
LightRAG's PostgreSQL vector storage), with only the LLM/embedding adapters
stubbed so it runs offline:

- Ingesting three fixture PDFs into a LightRAG-backed Repository ends with all
  three Resources `ready`, chunking and entity extraction actually executed by
  LightRAG.
- A retrieval query whose answer exists in exactly one document returns a
  chunk containing that document's fact, whose chunk id parses back to the
  ingested `Resource` (`res{id}-p{page}-chunk{n}`) and whose source label is
  the blob's staged filename — the provenance chain from FR-6 survives into
  LightRAG's own storage.
- Indexing and querying both run on the collection's single persistent event
  loop thread (the concurrency fix from commit `6fa7f59a`).
- A no-op re-run skips everything without disturbing what is indexed.

Operational caveats learned while validating (apply to real use too):

- LightRAG's chunker downloads the tiktoken `o200k_base` table on first use —
  air-gapped deployments must pre-seed `TIKTOKEN_CACHE_DIR`.
- LightRAG creates its PostgreSQL tables with **lowercase** names
  (`lightrag_doc_full`, `lightrag_vdb_chunks_<embedding-model>_<dim>d`, …).
  `LightRAGStore._cleanup_postgres` targets the wrong (JSON-backend) table
  names, so deleting a LightRAG silo leaves rows behind in Postgres; the test
  cleans up after itself and the fix belongs to a dedicated follow-up.
- The Neo4j community image does not support per-database creation; LightRAG
  logs a warning and falls back to the default database (workspace isolation
  is by node label, so this is expected).
- A query fired immediately after a batch can surface transient asyncpg
  pool-reset warnings from LightRAG's own connection management; its retry
  logic absorbs them.

## If this ever becomes a product

A persistent, self-service, multi-tenant version of this (saved sources,
scheduled sync, UI) should be a **new spec** reusing `azure_blob_client.py`
and `AzureBlobIngestService`'s batching/idempotency logic — not a retrofit of
this validation tool. This version's non-goals (no saved configuration, no
scheduled sync, no UI, no other object-storage providers) still apply to it.
