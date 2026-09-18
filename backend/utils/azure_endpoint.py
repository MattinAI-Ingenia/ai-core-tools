from urllib.parse import quote, urlparse, urlunparse


def normalize_azure_openai_endpoint(endpoint: str, model: str) -> str:
    """Expand a bare Azure OpenAI resource endpoint to its deployment route.

    The ``azure-ai-inference`` SDK always calls ``{endpoint}/chat/completions``
    (or ``{endpoint}/embeddings``), so a bare ``https://x.openai.azure.com``
    hits a nonexistent path and returns 404. This appends the classic
    ``/openai/deployments/{model}`` prefix, which does accept the api-version
    query parameter the SDK always sends.

    Endpoints that already contain ``/openai`` (e.g. ``/openai/v1`` or a full
    deployment path) and non-Azure-OpenAI hosts (AI Foundry, custom gateways)
    are returned unchanged.

    ``model`` is percent-encoded before interpolation so characters that would
    hijack the URL (``?``, ``#``, ``/``, spaces) cannot corrupt the request;
    Azure deployment names only ever contain ``[A-Za-z0-9._-]``, which
    ``quote`` leaves untouched.
    """
    url = (endpoint or "").strip().rstrip("/")
    if not url:
        return endpoint
    parsed = urlparse(url)
    if not parsed.hostname or not parsed.hostname.lower().endswith(".openai.azure.com"):
        return url
    if "/openai" in parsed.path:
        return url
    deployment = quote(model or "", safe="")
    if not deployment:
        return url
    path = f"{parsed.path.rstrip('/')}/openai/deployments/{deployment}"
    return urlunparse(parsed._replace(path=path))
