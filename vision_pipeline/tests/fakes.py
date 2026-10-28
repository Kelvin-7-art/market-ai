class FakeTensor:
    def __init__(self, values): self._values = values
    def tolist(self): return list(self._values)

class FakeBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = [FakeTensor(xyxy)]
        self.cls = [cls]
        self.conf = [conf]

class FakeResult:
    names = {0: "person", 1: "car"}
    def __init__(self, boxes): self.boxes = boxes

class FakeYOLO:
    def __init__(self, boxes): self._boxes = boxes
    def predict(self, *a, **kw): return [FakeResult(self._boxes)]
