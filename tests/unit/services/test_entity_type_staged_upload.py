"""Entity-type inference over PDFs picked in the ingestion dialog.

On a brand-new silo the files are not saved yet, so the upload endpoint stages
them in a temporary directory. The job must read them and remove that
directory even when it fails, or every abandoned dialog leaves files on disk.
"""

import os

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.entity_type_inference_job import EntityTypeInferenceJob
from services.entity_type_inference_service import EntityTypeInferenceService


def test_staged_paths_use_file_stem_as_doc_id(tmp_path):
    (tmp_path / "0-CDOC001004.pdf").write_bytes(b"%PDF")
    (tmp_path / "1-manual.pdf").write_bytes(b"%PDF")

    paths = EntityTypeInferenceService._staged_paths(str(tmp_path))

    assert paths == [
        ("0-CDOC001004", os.path.join(str(tmp_path), "0-CDOC001004.pdf")),
        ("1-manual", os.path.join(str(tmp_path), "1-manual.pdf")),
    ]


def test_run_removes_staging_dir_when_the_job_fails(tmp_path):
    staging = tmp_path / "entity-types-x"
    staging.mkdir()
    (staging / "0-a.pdf").write_bytes(b"%PDF")

    engine = create_engine("sqlite://")
    EntityTypeInferenceJob.__table__.create(engine)
    session = sessionmaker(bind=engine)()
    session.add(EntityTypeInferenceJob(
        job_id="job-1", silo_id=1, payload={"job_id": "job-1", "status": "pending"},
    ))
    session.commit()

    # No silo 1 exists, so the job fails; the failure must not skip the cleanup.
    EntityTypeInferenceService.run("job-1", 1, None, session, None, str(staging))

    assert not staging.exists()
    status = EntityTypeInferenceService.status("job-1", sessionmaker(bind=engine)())
    assert status["status"] == "failed"
