"""AC-21 (rerank-services step_013): the local cross-encoder reranker is gone.

The spec's grep checks, scripted: no torch / sentence_transformers import under backend/, no matching entries in
pyproject.toml or poetry.lock, no CROSS_ENCODER_RERANK_* setting, and the retired strategy names are rejected by
the agent schema.
"""
import ast
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
FORBIDDEN_MODULES = ("torch", "sentence_transformers")


def _backend_sources():
    for path in BACKEND.rglob("*.py"):
        if "data" in path.relative_to(BACKEND).parts[:1]:  # runtime data folder, not code
            continue
        yield path


def _imported_roots(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.lineno, node.module.split(".")[0]


def test_backend_has_no_torch_or_sentence_transformers_imports():
    offenders = []
    for path in _backend_sources():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, PermissionError):
            continue
        offenders += [f"{path.relative_to(ROOT)}:{line} imports {mod}"
                      for line, mod in _imported_roots(tree) if mod in FORBIDDEN_MODULES]
    assert offenders == []


def test_backend_has_no_dynamic_import_of_the_removed_libraries():
    pattern = re.compile(r"""(import_module|__import__|find_spec)\(\s*['"](torch|sentence_transformers)""")
    hits = [str(p.relative_to(ROOT)) for p in _backend_sources()
            if pattern.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert hits == []


def test_pyproject_has_no_torch_entries():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    for needle in ("torch", "sentence-transformers", "sentence_transformers", "pytorch-cpu"):
        assert needle not in text, needle


def test_lock_has_no_torch_packages_or_sources():
    lock = (ROOT / "poetry.lock").read_text(encoding="utf-8").lower()
    packages = set(re.findall(r'^name = "([^"]+)"', lock, flags=re.M))
    assert not packages & {"torch", "sentence-transformers", "triton", "nvidia-cudnn-cu12"}
    assert "pytorch-cpu" not in lock


def test_no_cross_encoder_settings_remain():
    hits = []
    for path in list(_backend_sources()) + [ROOT / "pyproject.toml"]:
        if "CROSS_ENCODER_RERANK" in path.read_text(encoding="utf-8", errors="ignore"):
            hits.append(str(path.relative_to(ROOT)))
    for env_file in ROOT.glob("**/.env.example"):
        if "node_modules" not in env_file.parts and "CROSS_ENCODER_RERANK" in env_file.read_text(errors="ignore"):
            hits.append(str(env_file.relative_to(ROOT)))
    assert hits == []


@pytest.mark.parametrize("legacy", ["rerank", "cross_encoder_rerank"])
def test_retired_strategies_are_rejected_by_the_agent_schemas(legacy):
    from schemas.agent_schemas import CreateAgentRequestSchema, CreateUpdateAgentSchema, UpdateAgentRequestSchema

    for schema, base in ((CreateUpdateAgentSchema, {"name": "a", "media_embedding_service_id": 1}),
                         (CreateAgentRequestSchema, {"name": "a"}), (UpdateAgentRequestSchema, {})):
        with pytest.raises(ValidationError):
            schema(**base, rag_search_method=[legacy])
        with pytest.raises(ValidationError):
            schema(**base, rag_search_method=["dense", legacy])
