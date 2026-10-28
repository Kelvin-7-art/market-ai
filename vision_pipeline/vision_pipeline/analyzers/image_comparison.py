# analyzers/image_comparison.py
import numpy as np
from PIL import Image
from pathlib import Path
from core.base import BaseAnalyzer
from core.result import AnalysisResult


class ImageComparisonAnalyzer(BaseAnalyzer):
    name = "image_comparison"

    def __init__(self, storage_dir="results/embeddings"):
        self.storage_dir = Path(storage_dir)

    def analyze(self, image, context):
        """Compare current image against a reference.

        Expects context to contain either:
          - 'reference_id': ID of a previously embedded image
          - 'reference_image': numpy array of the comparison image
        """
        reference_id = context.get("reference_id")
        reference_image = context.get("reference_image")

        # Resolve the reference to a numpy image
        if reference_id:
            ref_pil = self._load_image_by_id(reference_id)
            if ref_pil is None:
                return AnalysisResult(self.name, error=f"No stored image for id={reference_id}")
            ref_np = np.array(ref_pil)
        elif reference_image is not None:
            ref_np = reference_image
        else:
            return AnalysisResult(self.name, error="No reference provided")

        # Normalize current image
        cur_np = image if isinstance(image, np.ndarray) else np.array(image)

        # ── Semantic similarity via CLIP ─────────────────
        clip_sim = self._clip_similarity(cur_np, ref_np)

        # ── Pixel similarity via SSIM ────────────────────
        ssim_val, ssim_note = self._ssim_similarity(cur_np, ref_np)

        return AnalysisResult(self.name, {
            "clip_similarity": round(clip_sim, 4),
            "clip_interpretation": self._interpret_clip(clip_sim),
            "ssim": ssim_val,
            "ssim_note": ssim_note,
            "ssim_interpretation": self._interpret_ssim(ssim_val) if ssim_val is not None else None,
        })

    # ── helpers ──────────────────────────────────────────

    def _load_image_by_id(self, image_id):
        # In production: replace with a DB lookup.
        # For now, read from a parallel images dir if you store them.
        return None  # caller can pass reference_image directly instead

    def _clip_similarity(self, a, b):
        from sentence_transformers import SentenceTransformer
        # Reuse the singleton if the pipeline shares one,
        # otherwise instantiate. In practice, register EmbeddingsAnalyzer
        # first and stash the model on context.
        model = SentenceTransformer("clip-ViT-B-32")
        va = model.encode(Image.fromarray(a), convert_to_numpy=True)
        vb = model.encode(Image.fromarray(b), convert_to_numpy=True)
        # Cosine similarity
        denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-9
        return float(np.dot(va, vb) / denom)

    def _ssim_similarity(self, a, b):
        from skimage.metrics import structural_similarity as ssim
        import cv2

        # SSIM requires identical dimensions
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
            note = "reference resized to match"
        else:
            note = "same dimensions"

        # SSIM works on grayscale or multichannel
        gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)

        # win_size must be a single odd int, not a tuple [citation:5]
        smaller = min(gray_a.shape[:2])
        win = smaller if smaller % 2 == 1 else smaller - 1
        win = max(win, 3)

        try:
            val = float(ssim(gray_a, gray_b, win_size=win, data_range=255))
            return round(val, 4), note
        except Exception as e:
            return None, f"SSIM failed: {e}"

    @staticmethod
    def _interpret_clip(sim):
        if sim > 0.85: return "very similar"
        if sim > 0.70: return "similar"
        if sim > 0.50: return "related"
        return "different"

    @staticmethod
    def _interpret_ssim(val):
        if val is None: return None
        if val > 0.95: return "near-identical"
        if val > 0.80: return "visually similar"
        if val > 0.50: return "same scene, changes present"
        return "substantially different"