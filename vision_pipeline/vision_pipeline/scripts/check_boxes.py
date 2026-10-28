# scripts/check_boxes.py
import requests, sys
path = sys.argv[1]
with open(path, "rb") as f:
    r = requests.post("http://localhost:8000/analyze",
                      files={"file": (path.split("\\")[-1], f)})
d = r.json()
for b in d["results"]["detection"]["boxes"]:
    print(f"{b['label']:12s} conf={b['confidence']:.3f}  box=({b['x1']},{b['y1']})-({b['x2']},{b['y2']})")