from unittest.mock import patch

from services.resource_service import ResourceService


class _FakeFile:
    def __init__(self, filename, staged_path):
        self.filename = filename
        self._staged_path = staged_path

    def save(self, dest_path):
        with open(self._staged_path, 'rb') as src, open(dest_path, 'wb') as dst:
            dst.write(src.read())


def test_extra_metadata_set_on_created_resource(tmp_path, db, repository):
    staged = tmp_path / "staged.pdf"
    staged.write_bytes(b"%PDF-1.4 fake")
    file_adapter = _FakeFile("row-1.pdf", str(staged))

    with patch.object(ResourceService, '_index_resources_background', return_value='session-id'):
        created, failed, session_id = ResourceService.create_multiple_resources(
            files=[file_adapter],
            repository_id=repository.repository_id,
            db=db,
            extra_metadata={0: {'category': 'Finance'}},
        )

    assert not failed
    assert created[0].extra_metadata == {'category': 'Finance'}


def test_extra_metadata_defaults_to_none_when_not_provided(tmp_path, db, repository):
    staged = tmp_path / "staged.pdf"
    staged.write_bytes(b"%PDF-1.4 fake")
    file_adapter = _FakeFile("row-2.pdf", str(staged))

    with patch.object(ResourceService, '_index_resources_background', return_value='session-id'):
        created, failed, session_id = ResourceService.create_multiple_resources(
            files=[file_adapter], repository_id=repository.repository_id, db=db,
        )

    assert created[0].extra_metadata is None
