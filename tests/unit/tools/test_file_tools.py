import base64
import os
from unittest.mock import patch

import pytest

from tools.ai.fileTools import fetch_file_in_base64


def _invoke(tmp_path, file_path):
    with patch("tools.ai.fileTools.get_app_config", return_value={"TMP_BASE_FOLDER": str(tmp_path / "tmp")}):
        return fetch_file_in_base64.invoke({"file_path": file_path})


def test_reads_attachment_under_tmp_base_folder(tmp_path):
    (tmp_path / "tmp" / "conversations" / "1").mkdir(parents=True)
    (tmp_path / "tmp" / "conversations" / "1" / "doc.txt").write_bytes(b"hello")

    assert _invoke(tmp_path, "conversations/1/doc.txt") == base64.b64encode(b"hello").decode()


def test_refuses_files_outside_tmp_base_folder(tmp_path):
    """Regression (Sonar pythonsecurity:S2083): the LLM could read any file on the server."""
    (tmp_path / "tmp").mkdir()
    secret = tmp_path / "secret.env"
    secret.write_bytes(b"SECRET_KEY=x")

    for path in (str(secret), "../secret.env", "/etc/passwd"):
        assert _invoke(tmp_path, path).startswith("Error:")

# === The model guesses paths: whatever it sends must come back as text, never as an exception ===

def test_missing_file_is_an_error_text_not_an_exception(tmp_path):
    """The reported failure: /aict_backend/data/tmp/input/DSAT000120.pdf does not exist."""
    (tmp_path / "tmp").mkdir()

    for path in ("input/DSAT000120.pdf", "DSAT000120.pdf#page=39", "/aict_backend/data/tmp/input/X.pdf"):
        assert _invoke(tmp_path, path).startswith("Error:"), path


def test_missing_file_message_points_to_the_search_tools(tmp_path):
    (tmp_path / "tmp").mkdir()

    assert "search tools" in _invoke(tmp_path, "nope.pdf")


@pytest.mark.parametrize("path", ["", ".", "\x00", "a\x00b.txt", "x" * 5000, "a/" * 3000 + "f"])
def test_malformed_paths_never_raise(tmp_path, path):
    (tmp_path / "tmp").mkdir()

    assert _invoke(tmp_path, path).startswith("Error:")


def test_a_directory_is_an_error_text(tmp_path):
    (tmp_path / "tmp" / "conversations").mkdir(parents=True)

    assert _invoke(tmp_path, "conversations").startswith("Error:")


def test_symlink_out_of_the_folder_is_still_refused(tmp_path):
    (tmp_path / "tmp").mkdir()
    secret = tmp_path / "secret.env"
    secret.write_bytes(b"SECRET_KEY=x")
    os.symlink(secret, tmp_path / "tmp" / "link.env")

    result = _invoke(tmp_path, "link.env")

    assert result.startswith("Error:")
    assert "SECRET_KEY" not in result


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_file_is_an_error_text(tmp_path):
    (tmp_path / "tmp").mkdir()
    locked = tmp_path / "tmp" / "locked.txt"
    locked.write_bytes(b"x")
    locked.chmod(0)

    assert _invoke(tmp_path, "locked.txt").startswith("Error:")


def test_a_valid_attachment_is_unchanged(tmp_path):
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "ok.txt").write_bytes(b"hello")

    assert _invoke(tmp_path, "ok.txt") == base64.b64encode(b"hello").decode()
