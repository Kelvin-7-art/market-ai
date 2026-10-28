import numpy as np
import pytest
from tests.fakes import FakeBox, FakeTensor, FakeResult


class FakeMasks:
    def __init__(self, masks):
        self.data = masks  # list of torch-like tensors


class FakeSegResult(FakeResult):
    def __init__(self, boxes, masks):
        super().__init__(boxes)
        self.masks = masks


class FakeSegModel:
    def __init__(self, boxes, masks):
        self._boxes = boxes
        self._masks = masks

    def predict(self, *a, **kw):
        return [FakeSegResult(self._boxes, FakeMasks(self._masks))]


class FakeMaskTensor:
    """Mimics torch.Tensor.cpu().numpy()"""
    def __init__(self, arr):
        self._arr = arr
    def cpu(self):
        return self
    def numpy(self):
        return self._arr


@pytest.fixture
def fake_segmentation():
    from analyzers.segmentation import SegmentationAnalyzer

    def _make(boxes, masks):
        a = SegmentationAnalyzer.__new__(SegmentationAnalyzer)
        a.conf = 0.4
        a.model = FakeSegModel(boxes, masks)
        return a
    return _make


def test_segmentation_counts_mask_pixels(fake_segmentation):
    import torch
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[10:40, 20:50] = 1.0  # 30x30 = 900 pixels
    boxes = [FakeBox([10, 10, 50, 50], 0, 0.9)]
    masks = [FakeMaskTensor(mask)]

    a = fake_segmentation(boxes, masks)
    r = a.analyze(np.zeros((200, 200, 3), dtype=np.uint8), {})

    assert r.data["count"] == 1
    assert r.data["total_mask_pixels"] == 900
    assert r.data["segments"][0]["label"] == "person"


def test_segmentation_no_masks(fake_segmentation):
    boxes = [FakeBox([10, 10, 50, 50], 0, 0.9)]
    a = fake_segmentation(boxes, [])
    r = a.analyze(np.zeros((200, 200, 3), dtype=np.uint8), {})
    assert r.data["count"] == 1
    assert r.data["total_mask_pixels"] == 0