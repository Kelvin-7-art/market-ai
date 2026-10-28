import requests, json, sys
from pathlib import Path

img_path = Path(r"C:\Users\Khots\Downloads\business_card.jpg")
with open(img_path, "rb") as f:
    r = requests.post("http://localhost:8000/analyze", files={"file": f})
r.raise_for_status()
d = r.json()

print(f"Image: {d['filename']} — {d['width']}×{d['height']}")
print()

# OCR — the headline for a business card
ocr = d["results"]["ocr"]
print(f"=== OCR ({len(ocr['segments'])} segments) ===")
print(ocr["data"]["full_text"])
print()

# Show per-segment confidence so we can see what's junk
print("--- Segments by confidence ---")
for seg in sorted(ocr["segments"], key=lambda s: -s["confidence"])[:15]:
    print(f"  {seg['confidence']:.3f}  {seg['text']!r}")
print()

# Detection — what objects YOLO sees
det = d["results"]["detection"]
print(f"=== Detection ({det['data']['count']} objects) ===")
for b in det["boxes"]:
    print(f"  {b['label']:15s} conf={b['confidence']:.2f}  box=({b['x1']},{b['y1']})-({b['x2']},{b['y2']})")
print()

# Quality
q = d["results"]["quality"]["data"]
print(f"=== Quality ===")
print(f"  sharpness={q['sharpness']}  blurry={q['is_blurry']}")
print(f"  brightness={q['brightness']}  contrast={q['contrast']}")
print(f"  overexposed={q['overexposed_ratio']:.3f}  underexposed={q['underexposed_ratio']:.3f}")