"""Shared file-like adapter for feeding an already-staged file into
``ResourceService.create_multiple_resources``.

Extracted from ``import_job_confirm.py`` (its original, sole owner) so the
Azure Blob ingestion service (``azure_blob_ingest_service.py``) can reuse the
exact same duck-type instead of redefining it — both are "download a remote
file to a staging path, then hand it to the manual-upload pipeline" flows.
Pure move, no behavior change.
"""


class StagedFileAdapter:
    """Duck-types the file-like object `_process_single_file` expects
    (`.filename` + `.save(path)`), letting an already-downloaded file flow
    through the exact same resource creation path as a manual upload."""

    def __init__(self, staged_path: str, filename: str):
        self.filename = filename
        self._staged_path = staged_path

    def save(self, dest_path: str) -> None:
        import shutil
        shutil.move(self._staged_path, dest_path)
