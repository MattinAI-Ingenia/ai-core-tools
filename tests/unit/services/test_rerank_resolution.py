"""Pure AD-9 resolution: agent > app default > platform default; set-but-unusable means skip (AC-8)."""
from datetime import datetime, timedelta, timezone
from itertools import product
from types import SimpleNamespace

import pytest

from services.rerank_resolution_service import (
    SKIP_NONE_CONFIGURED,
    SKIP_RETIRED,
    SKIP_UNAVAILABLE,
    choose_rerank_service,
)

APP = 1


def svc(sid, app_id=APP, retired=False):
    eos = datetime.now(timezone.utc) - timedelta(days=1) if retired else None
    return SimpleNamespace(service_id=sid, app_id=app_id, end_of_service_at=eos)


@pytest.mark.parametrize("has_explicit,has_app,has_platform", list(product([False, True], repeat=3)))
def test_every_combination_picks_the_first_set_level(has_explicit, has_app, has_platform):
    explicit, app_d, plat = svc(1), svc(2), svc(3, app_id=None)
    r = choose_rerank_service(
        1 if has_explicit else None, explicit if has_explicit else None,
        app_d if has_app else None, plat if has_platform else None, APP,
        2 if has_app else None,
    )
    if has_explicit:
        assert (r.service, r.source) == (explicit, "agent")
    elif has_app:
        assert (r.service, r.source) == (app_d, "app_default")
    elif has_platform:
        assert (r.service, r.source) == (plat, "platform_default")
    else:
        assert r.service is None and r.skip_reason == SKIP_NONE_CONFIGURED


def test_retired_explicit_skips_without_substitution():
    r = choose_rerank_service(1, svc(1, retired=True), svc(2), svc(3, app_id=None), APP, 2)
    assert r.service is None and r.skip_reason == SKIP_RETIRED and r.source == "agent"


def test_foreign_explicit_skips_without_substitution():
    r = choose_rerank_service(1, svc(1, app_id=99), svc(2), svc(3, app_id=None), APP, 2)
    assert r.service is None and r.skip_reason == SKIP_UNAVAILABLE


def test_deleted_explicit_skips():
    r = choose_rerank_service(1, None, svc(2), svc(3, app_id=None), APP, 2)
    assert r.service is None and r.skip_reason == SKIP_UNAVAILABLE


def test_retired_app_default_skips_without_platform_fallback():
    r = choose_rerank_service(None, None, svc(2, retired=True), svc(3, app_id=None), APP, 2)
    assert r.service is None and r.skip_reason == SKIP_RETIRED and r.source == "app_default"


def test_app_default_of_another_app_skips_without_substitution():
    r = choose_rerank_service(None, None, svc(2, app_id=99), svc(3, app_id=None), APP, 2)
    assert r.service is None and r.skip_reason == SKIP_UNAVAILABLE and r.source == "app_default"


def test_retired_platform_default_skips():
    r = choose_rerank_service(None, None, None, svc(3, app_id=None, retired=True), APP)
    assert r.service is None and r.skip_reason == SKIP_RETIRED and r.source == "platform_default"


def test_system_service_is_usable_as_explicit():
    s = svc(5, app_id=None)
    assert choose_rerank_service(5, s, None, None, APP).service is s
