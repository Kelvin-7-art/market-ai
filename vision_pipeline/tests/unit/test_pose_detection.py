import numpy as np
import pytest
from tests.fakes import FakeBox


class FakeKeypoints:
    def __init__(self, data):
        self.data = data  # list of tensors
    def __len__(self):
        return len(self.data)


class FakeKeypointTensor:
    def __init__(self, arr):
        self._arr = arr
    def cpu(self):
        return self
    def numpy(self):
        return self._arr


class FakePoseResult:
    names = {0: "person"}
    def __init__(self, boxes, kpts):
        self.boxes = boxes
        self.keypoints = FakeKeypoints(kpts)
        self.masks = None


class FakePoseModel:
    def __init__(self, boxes, kpts):
        self._boxes = boxes
        self._kpts = kpts
    def predict(self, *a, **kw):
        return [FakePoseResult(self._boxes, self._kpts)]


def make_kpts(shoulder_y=100, hip_y=200, shoulder_x=100, hip_x=100):
    """17 keypoints with shoulders and hips positioned per params."""
    pts = np.zeros((17, 3), dtype=np.float32)
    # indices from KEYPOINTS: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip
    pts[5]  = [shoulder_x - 20, shoulder_y, 0.9]
    pts[6]  = [shoulder_x + 20, shoulder_y, 0.9]
    pts[11] = [hip_x - 20, hip_y, 0.9]
    pts[12] = [hip_x + 20, hip_y, 0.9]
    return FakeKeypointTensor(pts)


@pytest.fixture
def fake_pose():
    from analyzers.pose_detection import PoseDetectionAnalyzer
    def _make(boxes, kpts_list):
        a = PoseDetectionAnalyzer.__new__(PoseDetectionAnalyzer)
        a.conf = 0.5
        a.model = FakePoseModel(boxes, kpts_list)
        return a
    return _make


def test_pose_standing(fake_pose):
    boxes = [FakeBox([50, 50, 150, 250], 0, 0.9)]
    kpts = [make_kpts(shoulder_y=100, hip_y=200)]
    a = fake_pose(boxes, kpts)
    r = a.analyze(np.zeros((300, 300, 3), dtype=np.uint8), {})
    assert r.data["count"] == 1
    assert r.data["people"][0]["action"] == "standing"


def test_pose_lying(fake_pose):
    boxes = [FakeBox([50, 50, 250, 100], 0, 0.9)]
    # shoulders and hips at similar Y, spread along X → lying
    kpts = [make_kpts(shoulder_y=100, hip_y=110, shoulder_x=80, hip_x=200)]
    a = fake_pose(boxes, kpts)
    r = a.analyze(np.zeros((300, 300, 3), dtype=np.uint8), {})
    assert r.data["people"][0]["action"] == "lying"


def test_pose_empty(fake_pose):
    a = fake_pose([], [])
    r = a.analyze(np.zeros((300, 300, 3), dtype=np.uint8), {})
    assert r.data["count"] == 0