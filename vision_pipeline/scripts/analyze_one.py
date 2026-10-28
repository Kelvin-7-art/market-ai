# scripts/analyze_one.py
import requests, json

with open("tests/fixtures/real/bus.jpg", "rb") as f:
    r = requests.post("http://localhost:8000/analyze", files={"file": f})

r.raise_for_status()
d = r.json()

print("Analyzers returned:")
for name, res in d["results"].items():
    if res.get("error"):
        print(f"  ✗ {name}: {res['error'][:80]}")
    else:
        print(f"  ✓ {name}")

print()
print("face_detection:", d["results"].get("face_detection"))