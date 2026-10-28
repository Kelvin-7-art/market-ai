# analyzers/classification.py
import numpy as np
from core.base import BaseAnalyzer
from core.result import AnalysisResult

class ClassificationAnalyzer(BaseAnalyzer):
    name = "classification"

    def __init__(self, top_k=5):
        import torch
        from torchvision.models import resnet50, ResNet50_Weights
        self.torch = torch
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights).eval()
        self.preprocess = self.weights.transforms()
        self.categories = self.weights.meta["categories"]
        self.top_k = top_k

    def analyze(self, image, context):
        from PIL import Image
        img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        x = self.preprocess(img).unsqueeze(0)
        with self.torch.no_grad():
            probs = self.torch.nn.functional.softmax(self.model(x)[0], dim=0)
        top = self.torch.topk(probs, self.top_k)
        return AnalysisResult(self.name, {
            "predictions": [
                {"label": self.categories[i], "score": float(s)}
                for s, i in zip(top.values, top.indices)
            ]
        })