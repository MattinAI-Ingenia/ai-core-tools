from langchain_core.tools import tool
import base64

from utils.config import get_app_config
from utils.path_safety import UnsafePathError, resolve_within


@tool
def fetch_file_in_base64(file_path: str):
    """Return the base64 content of a file attached to this conversation (path inside the attachments folder).

    Only for attachments. The documents of the knowledge base are not stored there: use the search tools for them.
    Returns a text starting with "Error:" when the file cannot be read.
    """
    # The path comes from the LLM, so it is untrusted: only files under
    # TMP_BASE_FOLDER (where attachments live) may be read. A path that does not exist is the
    # normal case when the model guesses one: it must come back as text, not as an exception that
    # ends the whole run (it used to take down the knowledge-router specialist that called it).
    try:
        safe_path = resolve_within(get_app_config()['TMP_BASE_FOLDER'], file_path)
    except UnsafePathError:
        return "Error: access to this file path is not allowed."
    except (ValueError, OSError):
        return "Error: invalid file path."
    try:
        with open(safe_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")
    except FileNotFoundError:
        return (
            "Error: file not found. Only files attached to this conversation can be read; "
            "to look into documents of the knowledge base use the search tools."
        )
    except OSError as exc:  # directory, permission denied, name too long...
        return f"Error: the file could not be read ({type(exc).__name__})."
