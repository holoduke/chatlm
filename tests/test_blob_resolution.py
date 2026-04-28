"""Manifest → blob path resolution against a fake Ollama models tree.

Covers `resolve_blob_path` and `resolve_projector_path` without touching
the real ~/.ollama directory. Uses a tmp_path tree shaped like Ollama's
real layout (manifests/<registry>/<repo>/<tag>, blobs/sha256-<digest>).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import llama_server_client as L


def _write_manifest(root: Path, name: str, layers: list[dict]) -> Path:
    """Create a manifest file under root mimicking Ollama's tree."""
    reg, repo, tag = L._split_name(L.strip_prefix(name))
    p = root / "manifests" / reg / repo / tag
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schemaVersion": 2, "layers": layers}))
    return p


def _write_blob(root: Path, digest: str, body: bytes = b"x") -> Path:
    blob_dir = root / "blobs"
    blob_dir.mkdir(parents=True, exist_ok=True)
    p = blob_dir / f"sha256-{digest}"
    p.write_bytes(body)
    return p


@pytest.fixture
def fake_ollama_root(tmp_path, monkeypatch):
    """Repoint the module's path constants at a tmp tree so each test
    starts from an empty 'Ollama cache'."""
    monkeypatch.setattr(L, "OLLAMA_ROOT", tmp_path)
    monkeypatch.setattr(L, "MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.setattr(L, "BLOB_DIR", tmp_path / "blobs")
    return tmp_path


def test_resolve_blob_returns_model_layer_path(fake_ollama_root):
    digest = "deadbeef" * 8
    _write_blob(fake_ollama_root, digest)
    _write_manifest(
        fake_ollama_root,
        "hf.co/owner/repo:Q4",
        layers=[{"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{digest}"}],
    )
    out = L.resolve_blob_path("llama:hf.co/owner/repo:Q4")
    assert out == fake_ollama_root / "blobs" / f"sha256-{digest}"


def test_resolve_blob_works_with_or_without_llama_prefix(fake_ollama_root):
    digest = "ab" * 32
    _write_blob(fake_ollama_root, digest)
    _write_manifest(
        fake_ollama_root,
        "hf.co/owner/repo:tag",
        layers=[{"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{digest}"}],
    )
    p1 = L.resolve_blob_path("llama:hf.co/owner/repo:tag")
    p2 = L.resolve_blob_path("hf.co/owner/repo:tag")
    assert p1 == p2


def test_resolve_blob_raises_when_manifest_missing(fake_ollama_root):
    with pytest.raises(FileNotFoundError, match="no Ollama manifest"):
        L.resolve_blob_path("llama:hf.co/missing/repo:tag")


def test_resolve_blob_raises_when_blob_missing(fake_ollama_root):
    digest = "11" * 32
    _write_manifest(
        fake_ollama_root,
        "hf.co/owner/repo:tag",
        layers=[{"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{digest}"}],
    )
    # No corresponding blob written.
    # Current behaviour: returns None from _layer_blob → resolve raises.
    with pytest.raises(FileNotFoundError, match="no model-mediatype"):
        L.resolve_blob_path("hf.co/owner/repo:tag")


def test_resolve_blob_raises_when_no_model_layer(fake_ollama_root):
    digest = "22" * 32
    _write_blob(fake_ollama_root, digest)
    _write_manifest(
        fake_ollama_root,
        "hf.co/owner/repo:tag",
        layers=[{"mediaType": "application/vnd.ollama.image.template", "digest": f"sha256:{digest}"}],
    )
    with pytest.raises(FileNotFoundError, match="no model-mediatype"):
        L.resolve_blob_path("hf.co/owner/repo:tag")


def test_resolve_projector_returns_path_when_present(fake_ollama_root):
    model_d = "aa" * 32
    proj_d = "bb" * 32
    _write_blob(fake_ollama_root, model_d)
    _write_blob(fake_ollama_root, proj_d)
    _write_manifest(
        fake_ollama_root,
        "hf.co/owner/repo:tag",
        layers=[
            {"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{model_d}"},
            {"mediaType": "application/vnd.ollama.image.projector", "digest": f"sha256:{proj_d}"},
        ],
    )
    out = L.resolve_projector_path("llama:hf.co/owner/repo:tag")
    assert out == fake_ollama_root / "blobs" / f"sha256-{proj_d}"


def test_resolve_projector_returns_none_when_text_only_model(fake_ollama_root):
    model_d = "cc" * 32
    _write_blob(fake_ollama_root, model_d)
    _write_manifest(
        fake_ollama_root,
        "hf.co/owner/repo:tag",
        layers=[
            {"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{model_d}"},
        ],
    )
    assert L.resolve_projector_path("hf.co/owner/repo:tag") is None


def test_resolve_projector_returns_none_when_manifest_missing(fake_ollama_root):
    """resolve_projector should not raise on missing model — caller
    treats None as 'text-only', equivalent to 'we don't know'."""
    assert L.resolve_projector_path("hf.co/missing/repo:tag") is None
