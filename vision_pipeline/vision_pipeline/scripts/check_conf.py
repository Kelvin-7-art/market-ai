# scripts/check_conf.py
import requests
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Khots\Downloads\OIP (1).webp"

with open(path, "rb") as f:
    r = requests.post(
        "http://localhost:8000/analyze",
        files={"file": (path.split("\\")[-1], f, "image/webp")},
    )

if r.status_code != 200:
    print(f"HTTP {r.status_code}: {r.text}")
    sys.exit(1)

d = r.json()
print(f"\n=== {path} ===\n")

# Detection boxes with confidence
det = d["results"]["detection"]
print(f"Detection: {det['data']['count']} objects")
for b in det["boxes"]:
    print(f"  {b['label']:12s} conf={b['confidence']:.3f}  box=({b['x1']},{b['y1']})-({b['x2']},{b['y2']})")
print()

# Face detection boxes
fd = d["results"]["face_detection"]
print(f"Faces: {fd['data']['count']}")
for b in fd["boxes"]:
    print(f"  {b['label']:12s} conf={b['confidence']:.3f}")
print()

# All boxes in context (detection + OCR + faces)
print("Total boxes in spatial context:")
print(f"  detection: {len(det['boxes'])}")
print(f"  ocr:       {len(d['results']['ocr']['boxes'])}")
print(f"  faces:     {len(fd['boxes'])}")
print(f"  spatial:   {len(d['results']['spatial']['data'].get('pairwise', []))} pairwise relations")