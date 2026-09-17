from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class IngestAzureBlobsRequestSchema(BaseModel):
    """Body of POST .../repositories/{repository_id}/ingest-azure-blobs.

    ``sas_token`` is request-scoped only: it is never persisted, echoed back
    in a response, or logged (see AzureBlobIngestService / azure_blob_client's
    sanitize_azure_error).
    """

    model_config = ConfigDict(extra='forbid')

    account_url: str
    container: str
    prefix: Optional[str] = None
    auth_mode: Literal['ANONYMOUS', 'SAS_TOKEN'] = 'ANONYMOUS'
    sas_token: Optional[str] = None
    file_extension_filters: Optional[list[str]] = None


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
