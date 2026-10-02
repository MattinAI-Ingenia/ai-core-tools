"""``AzureBlobSourceSchema`` legacy-shape handling.

``Repository.azure_blob_source`` renamed its single ``prefix`` key to
``prefixes`` (a list) in the same unmerged branch that introduced it, but
rows persisted by earlier builds of that branch still carry ``prefix`` — the
schema must map it back to a one-element ``prefixes`` when read, otherwise
the "Actualizar" flow silently re-scans the whole container.
"""
from schemas.repository_schemas import AzureBlobSourceSchema


def test_current_shape_round_trips_unchanged():
    source = AzureBlobSourceSchema.model_validate(
        {"account_url": "https://acct.blob.core.windows.net", "container": "etiquetas",
         "prefixes": ["CDOC", "DSAT"], "name_excludes": ["_"], "auth_mode": "ANONYMOUS"}
    )

    assert source.prefixes == ["CDOC", "DSAT"]
    assert source.name_excludes == ["_"]
    assert source.model_dump() == {
        "account_url": "https://acct.blob.core.windows.net",
        "container": "etiquetas",
        "prefixes": ["CDOC", "DSAT"],
        "name_excludes": ["_"],
        "auth_mode": "ANONYMOUS",
    }


def test_legacy_prefix_key_is_mapped_to_a_one_element_prefixes():
    source = AzureBlobSourceSchema.model_validate(
        {"account_url": "https://acct.blob.core.windows.net", "container": "etiquetas",
         "prefix": "2026/", "auth_mode": "ANONYMOUS"}
    )

    # The legacy key never reaches the serialized shape (no-secrets-style
    # structural guarantee) — only its mapped value does.
    assert source.prefixes == ["2026/"]
    assert "prefix" not in source.model_dump()


def test_prefixes_win_over_the_legacy_key():
    source = AzureBlobSourceSchema.model_validate(
        {"account_url": "https://acct.blob.core.windows.net", "container": "etiquetas",
         "prefix": "2026/", "prefixes": ["CDOC"], "auth_mode": "ANONYMOUS"}
    )

    assert source.prefixes == ["CDOC"]
