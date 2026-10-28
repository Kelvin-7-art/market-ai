# tests/unit/test_embeddings.py
import numpy as np
from analyzers.embeddings import EmbeddingsAnalyzer


def test_embedding_shape(solid_red, tmp_path):
    a = EmbeddingsAnalyzer(storage_dir=str(tmp_path))
    r = a.analyze(solid_red, {"image_id": "test_red"})
    assert r.error is None
    assert r.data["dimensions"] == 512
    assert (tmp_path / "test_red.npy").exists()


def test_embedding_norm(solid_red, tmp_path):
    a = EmbeddingsAnalyzer(storage_dir=str(tmp_path))
    r = a.analyze(solid_red, {"image_id": "test_red"})
    # CLIP embeddings are L2-normalized → norm ≈ 1.0
    assert abs(r.data["norm"] - 1.0) < 0.05