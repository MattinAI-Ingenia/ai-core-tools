from pydantic import BaseModel, ConfigDict
from typing import Optional, List
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


class CreatedResourceSchema(BaseModel):
    resource_id: int
    uri: str
    repository_id: Optional[int] = None
    create_date: Optional[datetime] = None
    size: Optional[int] = None
    content_type: str


class ConfirmRowsResponseSchema(BaseModel):
    created_resources: List[CreatedResourceSchema] = []
    failed_files: List[dict] = []
    session_id: Optional[str] = None
