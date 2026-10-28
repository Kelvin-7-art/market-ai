# analyzers/embeddings.py
import os
import numpy as np
from pathlib import Path
from PIL import Image
from core.base import BaseAnalyzer
from core.result import AnalysisResult


class EmbeddingsAnalyzer(BaseAnalyzer):
    name = "embeddings"

    def __init__(
        self,
        model_name="clip-ViT-B-32",
        storage_dir="results/embeddings",
    ):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, image, context):
        # Accept either numpy (from pipeline) or PIL (direct call)
        if isinstance(image, np.ndarray):
            pil = Image.fromarray(image)
        else:
            pil = image

        # Encode → numpy array of shape (512,)
        embedding = self.model.encode(pil, convert_to_numpy=True)

        # Persist to disk for later similarity / retrieval
        image_id = context.get("image_id")
        if image_id:
            np.save(self.storage_dir / f"{image_id}.npy", embedding)

        # Return a *summary* in the JSON — not the raw vector.
        # A 512-float array bloats every API response and isn't
        # useful to a human reader. Store the path instead.
        stored_path = None
        if image_id:
            stored_path = str(self.storage_dir / f"{image_id}.npy")

        return AnalysisResult(self.name, {
            "model": "clip-ViT-B-32",
            "dimensions": int(embedding.shape[0]),
            "norm": float(np.linalg.norm(embedding)),
            "stored_at": stored_path,
        })