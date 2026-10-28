# scripts/debug_receipt_parser.py
import sys
import json
import requests

path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Khots\Downloads\image.png"

with open(path, "rb") as f:
    r = requests.post("http://localhost:8000/analyze",
                      files={"file": (path.split("\\")[-1], f)})

print(f"HTTP {r.status_code}")
print(f"Content-Type: {r.headers.get('content-type')}")
print()

if r.status_code != 200:
    print("Error response body:")
    print(r.text[:2000])
    sys.exit(1)

d = r.json()

# Show top-level keys so we can see the actual shape
print(f"Top-level keys: {list(d.keys())}")
print()

if "results" not in d:
    print("Response does not contain 'results'. Full body:")
    print(json.dumps(d, indent=2)[:2000])
    sys.exit(1)

# Rest of the script as before
print("=== OCR SEGMENTS (first 15) ===")
ocr = d["results"]["ocr"]["data"]
for s in ocr["segments"][:15]:
    print(f"  {s['text']!r:40s}  box={s.get('box')}  conf={s['confidence']:.2f}")

print(f"\nTotal segments: {len(ocr['segments'])}")

print("\n=== RECEIPT PARSER RAW ===")
rp = d["results"]["receipt_parser"]
print(f"Error: {rp.get('error')}")
print(f"Data: {json.dumps(rp.get('data'), indent=2)}")