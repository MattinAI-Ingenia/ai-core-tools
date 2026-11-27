from urllib.parse import urlsplit, urlunsplit


def normalize_endpoint(raw: str) -> str:
    """Return the normalised base URL (AD-3): http(s) + host, no trailing slash, no trailing /rerank.

    Rejects query strings, fragments, embedded credentials and invalid ports.
    """
    value = (raw or "").strip()
    if not value:
        raise ValueError("Endpoint is required")
    try:
        parts = urlsplit(value)
        host = parts.hostname
        _ = parts.port  # validates the port range
        username, password = parts.username, parts.password
    except ValueError:
        raise ValueError("Endpoint is not a valid URL (check host and port)") from None
    if parts.scheme.lower() not in ("http", "https"):
        raise ValueError("Endpoint must start with http:// or https://")
    if not host:
        raise ValueError("Endpoint must include a host")
    if username is not None or password is not None:
        raise ValueError("Endpoint must not contain credentials; use the API key field")
    if parts.query or parts.fragment or "?" in value or "#" in value:
        raise ValueError("Endpoint must not contain a query string or fragment")
    path = parts.path.rstrip("/")
    if path.lower().endswith("/rerank"):
        path = path[: -len("/rerank")].rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc, path, "", ""))


def rerank_url(endpoint: str) -> str:
    return f"{normalize_endpoint(endpoint)}/rerank"
