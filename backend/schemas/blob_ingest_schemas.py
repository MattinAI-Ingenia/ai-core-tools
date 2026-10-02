from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# A single prefix/exclude token: only ever used as a name_starts_with query
# parameter or a Python substring check, but it gets persisted into
# Repository.azure_blob_source (JSONB) and echoed back on every detail
# response, so bound it.
NameToken = Annotated[str, StringConstraints(max_length=200)]


class IngestAzureBlobsRequestSchema(BaseModel):
    """Body of the ingest and preview Azure Blob endpoints
    (``ingest-azure-blobs`` / ``preview-azure-blobs``, same contract).

    ``sas_token`` is request-scoped only: it is never persisted, echoed back
    in a response, or logged (see AzureBlobIngestService / azure_blob_client's
    sanitize_azure_error).

    ``sample_size`` caps each run to at most that many randomly-picked
    pending (new/changed) blobs — the validation-phase safety net so a huge
    container cannot flood LightRAG indexing. None (the default) ingests
    everything pending, which is the production behavior once validation
    ends; the current frontend always sends 20.

    ``prefixes`` limits the listing to blobs whose names start with any of
    the given prefixes (e.g. ``["CDOC", "DSAT"]``) — one server-side listing
    per prefix, unioned and de-duplicated. Omitted/empty means the whole
    container. Mutually exclusive with ``blob_name``. The persisted
    ``Repository.azure_blob_source`` keeps the same ``prefixes``, so the
    "Actualizar" flow still scans exactly this subset afterwards.

    ``name_excludes`` is the inverse: blobs whose names *contain* any of
    these substrings are skipped (a character blacklist — e.g. ``["_"]``
    drops every generated duplicate like ``CDOC002817_2a67fdb9.pdf``).
    Composes with everything else; a single-file run (``blob_name``) whose
    name matches an exclude is reported as not found.

    ``blob_name`` selects one exact blob (optionally its full path inside the
    container) instead of the whole container.
    """

    model_config = ConfigDict(extra='forbid')

    account_url: str
    container: str
    # Capped server-side: one listing call runs per prefix, so an unbounded
    # list would multiply the outbound Azure work per request.
    prefixes: Optional[list[NameToken]] = Field(default=None, max_length=16)
    auth_mode: Literal['ANONYMOUS', 'SAS_TOKEN'] = 'ANONYMOUS'
    sas_token: Optional[str] = None
    file_extension_filters: Optional[list[str]] = None
    sample_size: Optional[int] = Field(default=None, ge=1)
    blob_name: Optional[str] = None
    # Same reasoning as prefixes: every entry is a per-blob substring check.
    name_excludes: Optional[list[NameToken]] = Field(default=None, max_length=16)


class IngestAzureBlobsResponseSchema(BaseModel):
    """Response of POST .../repositories/{repository_id}/ingest-azure-blobs.

    ``queued``/``failed`` reflect only the first batch (processed
    synchronously so a real ``session_id`` can be returned here). If the run
    spans more than one batch, the remaining batches continue in the
    background under that same ``session_id`` — but the SSE progress endpoint
    and this response's counters only ever describe the first batch. Later
    batches are not reflected there at all: they are only visible in the
    structured "Azure blob ingestion run finished" log line emitted once the
    whole run (all batches) completes.
    """

    model_config = ConfigDict(extra='forbid')

    queued: int
    skipped_unchanged: int
    skipped_unsupported: int
    failed: int
    session_id: Optional[str] = None
    # Whole-run visibility for the UI's informative message: how many blobs
    # the listing found in total, and how many the diff marked pending
    # (new/changed) BEFORE the sample_size cap applied. queued can be smaller
    # than min(pending_blobs, sample_size) when some of the first batch's
    # downloads or pipeline entries failed.
    total_blobs: int
    pending_blobs: int


class PreviewAzureBlobsResponseSchema(BaseModel):
    """Response of POST .../repositories/{repository_id}/preview-azure-blobs.

    A dry-run of the listing + idempotency diff behind ``ingest-azure-blobs``:
    ``total_blobs`` is how many blobs the container currently holds under the
    request's ``prefixes`` (and supported extensions, and exact ``blob_name``
    when given), ``pending_blobs`` how many of those are new or changed
    — the same pre-``sample_size`` number the ingest response reports — so a
    real Load right now would ingest at most ``sample_size`` of them. Nothing
    is downloaded, indexed, or persisted.
    """

    model_config = ConfigDict(extra='forbid')

    total_blobs: int
    pending_blobs: int
