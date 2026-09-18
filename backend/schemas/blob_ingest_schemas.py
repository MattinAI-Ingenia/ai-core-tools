from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class IngestAzureBlobsRequestSchema(BaseModel):
    """Body of POST .../repositories/{repository_id}/ingest-azure-blobs.

    ``sas_token`` is request-scoped only: it is never persisted, echoed back
    in a response, or logged (see AzureBlobIngestService / azure_blob_client's
    sanitize_azure_error).

    ``sample_size`` caps each run to at most that many randomly-picked
    pending (new/changed) blobs — the validation-phase safety net so a huge
    container cannot flood LightRAG indexing. None (the default) ingests
    everything pending, which is the production behavior once validation
    ends; the current frontend always sends 20.

    ``blob_name`` selects one exact blob (optionally its full path inside the
    container) instead of the whole container — mutually exclusive with
    ``prefix``. The persisted ``Repository.azure_blob_source`` keeps the
    original ``prefix``, so the "Actualizar" flow still scans the whole
    (prefix-scoped) container afterwards.
    """

    model_config = ConfigDict(extra='forbid')

    account_url: str
    container: str
    prefix: Optional[str] = None
    auth_mode: Literal['ANONYMOUS', 'SAS_TOKEN'] = 'ANONYMOUS'
    sas_token: Optional[str] = None
    file_extension_filters: Optional[list[str]] = None
    sample_size: Optional[int] = Field(default=None, ge=1)
    blob_name: Optional[str] = None


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
