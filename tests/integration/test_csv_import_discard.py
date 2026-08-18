import os

from models.import_job import ImportJob
from models.import_job_row import ImportJobRow
from models.enums.import_row_status import ImportRowStatus
from services.import_job_discard import discard_rows


def _make_downloaded_row(db, job, tmp_path, url, name="x"):
    staged = tmp_path / f"{name}.pdf"
    staged.write_bytes(b"%PDF-1.4 fake")
    row = ImportJobRow(import_job_id=job.id, url=url, row_metadata={},
                        status=ImportRowStatus.DOWNLOADED, staged_path=str(staged))
    db.add(row)
    db.commit()
    return row


def test_discard_removes_row_and_staged_file(tmp_path, db, repository):
    job = ImportJob(repository_id=repository.repository_id)
    db.add(job)
    db.commit()
    row = _make_downloaded_row(db, job, tmp_path, 'http://a.com/x.pdf')
    staged_path = row.staged_path
    assert os.path.exists(staged_path)

    discard_rows(job.id, [row.id], db)

    assert not os.path.exists(staged_path)
    assert db.query(ImportJob).filter(ImportJob.id == job.id).first() is None


def test_discarding_one_of_two_rows_keeps_job_open(tmp_path, db, repository):
    job = ImportJob(repository_id=repository.repository_id)
    db.add(job)
    db.commit()
    row1 = _make_downloaded_row(db, job, tmp_path, 'http://a.com/x.pdf', name="x")
    _make_downloaded_row(db, job, tmp_path, 'http://a.com/y.pdf', name="y")

    discard_rows(job.id, [row1.id], db)

    remaining = db.query(ImportJob).filter(ImportJob.id == job.id).first()
    assert remaining is not None
