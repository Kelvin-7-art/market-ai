# scripts/quick_test.py
import requests
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Khots\Downloads\image.png"

# Guess content type from extension
ext = path.rsplit(".", 1)[-1].lower()
content_type = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "gif": "image/gif",
}.get(ext, "application/octet-stream")

with open(path, "rb") as f:
    files = {"file": (path.split("\\")[-1], f, content_type)}
    r = requests.post("http://localhost:8000/analyze", files=files)

if r.status_code != 200:
    print(f"HTTP {r.status_code}")
    print(f"Response: {r.text}")
    sys.exit(1)

d = r.json()

print(f"\n=== {path} ===")
print(f"Dimensions: {d['width']} × {d['height']}\n")

for name, res in d["results"].items():
    if res.get("error"):
        print(f"  ✗ {name:16s} {res['error'][:70]}")
        continue

    data = res.get("data", {})
    boxes = res.get("boxes", [])

    if name == "metadata":
        print(f"  ✓ {name:16s} {data.get('width')}×{data.get('height')} {data.get('mode')}")

    elif name == "quality":
        sharp = data.get('sharpness', 0)
        bright = data.get('brightness', 0)
        print(f"  ✓ {name:16s} sharp={sharp:.0f} bright={bright:.0f} blurry={data.get('is_blurry')}")

    elif name == "color":
        top = data.get("dominant_colors", [{}])[0]
        print(f"  ✓ {name:16s} dominant={top.get('name')} ({top.get('hex')}) grayscale={data.get('is_grayscale')}")

    elif name == "detection":
        labels = {}
        for b in boxes:
            labels[b["label"]] = labels.get(b["label"], 0) + 1
        print(f"  ✓ {name:16s} {data.get('count')} objects: {labels}")

    elif name == "segmentation":
        segs = data.get("segments", [])
        labels = {}
        for s in segs:
            labels[s["label"]] = labels.get(s["label"], 0) + 1
        total_px = data.get("total_mask_pixels", 0)
        print(f"  ✓ {name:16s} {data.get('count')} segments, {total_px:,} px: {labels}")

    elif name == "counting":
        print(f"  ✓ {name:16s} total={data.get('total')} {data.get('by_class')}")

    elif name == "ocr":
        text = data.get("full_text", "")
        preview = text[:60]
        truncated = "..." if len(text) > 60 else ""
        print(f"  ✓ {name:16s} {data.get('segment_count')} segments ({len(text)} chars): {preview!r}{truncated}")

    elif name == "receipt_parser":
        ents = data.get("entities", {})
        words = data.get("words_processed")
        model = data.get("model", "")
        raw_preview = data.get("raw_sequence_preview", "")

        if words is not None:
            version_tag = f"{words} words processed"
        elif model:
            version_tag = f"model: {model.split('/')[-1]}"
        else:
            version_tag = "unknown version"

        if data.get("note"):
            print(f"  ✓ {name:16s} (note: {data['note']})")

        # ── Handle both dict and list shapes ─────────────────────
        elif isinstance(ents, dict) and not ents:
            print(f"  ✓ {name:16s} no entities detected ({version_tag})")

        elif isinstance(ents, dict):
            print(f"  ✓ {name:16s} {len(ents)} entity groups ({version_tag}):")
            for k, v in list(ents.items())[:8]:
                if isinstance(v, list):
                    value_str = ", ".join(str(x) for x in v[:3])
                    if len(v) > 3:
                        value_str += f", ... (+{len(v) - 3} more)"
                else:
                    value_str = str(v)
                if len(value_str) > 90:
                    value_str = value_str[:90] + "..."
                print(f"      {k:28s} {value_str}")
            if len(ents) > 8:
                print(f"      ... (+{len(ents) - 8} more entity types)")

            if raw_preview and len(ents) <= 1:
                print(f"      raw: {raw_preview}")

        elif isinstance(ents, list):
            # Donut returned a list — usually when the document isn't a receipt
            print(f"  ✓ {name:16s} {len(ents)} items (list-shaped) ({version_tag}):")
            for item in ents[:5]:
                print(f"      {str(item)[:90]}")
            if len(ents) > 5:
                print(f"      ... (+{len(ents) - 5} more)")

    elif name == "table_extractor":
        tables = data.get("tables", [])
        if not tables:
            print(f"  ✓ {name:16s} 0 tables detected")
        else:
            print(f"  ✓ {name:16s} {len(tables)} table(s):")
            for t in tables[:3]:
                print(f"      Table {t['index']+1}: {t['rows']}×{t['cols']} — headers={t['headers']}")
                for row in t["data"][:3]:
                    print(f"        {row}")
                if t["rows"] > 3:
                    print(f"        ... (+{t['rows'] - 3} more rows)")

    elif name == "handwriting":
        segs = data.get("segments", [])
        if not segs:
            print(f"  ✓ {name:16s} no handwriting detected")
        else:
            text = data.get("full_text", "")
            print(f"  ✓ {name:16s} {len(segs)} regions ({len(text)} chars): {text[:80]!r}")
            for s in segs[:3]:
                print(f"      {s['text']!r}")
            if len(segs) > 3:
                print(f"      ... (+{len(segs) - 3} more)")

    elif name == "math_ocr":
        formulas = data.get("formulas", [])
        latex_single = data.get("latex", "")
        markdown = data.get("markdown", "")

        if latex_single and not formulas:
            print(f"  ✓ {name:16s} LaTeX: {latex_single[:60]!r}")
        elif not formulas:
            print(f"  ✓ {name:16s} 0 formulas detected")
            if markdown:
                print(f"      markdown: {markdown[:80]!r}")
        else:
            print(f"  ✓ {name:16s} {len(formulas)} formula(s):")
            for f in formulas[:3]:
                print(f"      LaTeX: {f['latex'][:60]!r}")
            if len(formulas) > 3:
                print(f"      ... (+{len(formulas) - 3} more)")

    elif name == "doc_layout":
        regions = data.get("regions", [])
        type_counts = data.get("type_counts", {})
        if not regions:
            print(f"  ✓ {name:16s} 0 regions detected")
        else:
            print(f"  ✓ {name:16s} {len(regions)} regions: {type_counts}")
            for r in regions[:5]:
                bbox = r["bbox"]
                conf = r.get("confidence", 0.0)
                print(f"      {r['type']:16s} ({bbox['x1']},{bbox['y1']})-({bbox['x2']},{bbox['y2']})  conf={conf:.2f}")
            if len(regions) > 5:
                print(f"      ... (+{len(regions) - 5} more)")

    elif name == "charts":
        if not data.get("is_chart"):
            note = data.get("note", "")
            print(f"  ✓ {name:16s} not a chart" + (f" ({note})" if note else ""))
        else:
            chart_type = data.get("chart_type", "unknown")
            print(f"  ✓ {name:16s} {chart_type} chart")

            x_axis = data.get("x_axis") or {}
            y_axis = data.get("y_axis") or {}

            if x_axis.get("label") or x_axis.get("range"):
                x_label = x_axis.get("label") or "?"
                print(f"      X: {x_label} {x_axis.get('range', '')}")
            if y_axis.get("label") or y_axis.get("range"):
                y_label = y_axis.get("label") or "?"
                print(f"      Y: {y_label} {y_axis.get('range', '')}")

            series = data.get("series", [])
            if series:
                print(f"      Series ({len(series)}):")
                for s in series[:5]:
                    name_ = s.get("name", "?")
                    type_ = s.get("type", "?")
                    trend = s.get("trend", "?")
                    proj = " (projection)" if s.get("is_projection") else ""
                    print(f"        {name_:20s} {type_:15s} {trend}{proj}")

            insights = data.get("insights", [])
            if insights:
                print(f"      Insights:")
                for ins in insights[:3]:
                    print(f"        • {ins}")

    elif name == "barcode_qr":
        print(f"  ✓ {name:16s} {data.get('count')} codes")
        for c in data.get("codes", []):
            print(f"      {c['type']}: {c['data'][:50]}")

    elif name == "face_detection":
        print(f"  ✓ {name:16s} {data.get('count')} faces")

    elif name == "pose_detection":
        people = data.get("people", [])
        actions = {}
        for p in people:
            a = p.get("action", "unknown")
            actions[a] = actions.get(a, 0) + 1
        print(f"  ✓ {name:16s} {data.get('count')} people: {actions}")

    elif name == "classification":
        preds = data.get("predictions", [])[:3]
        formatted = [(p['label'], round(p['score'], 2)) for p in preds]
        print(f"  ✓ {name:16s} top: {formatted}")

    elif name == "embeddings":
        print(f"  ✓ {name:16s} {data.get('dimensions')}d vector")

    elif name == "spatial":
        positions = data.get("positions", [])
        pairwise = data.get("pairwise", [])
        print(f"  ✓ {name:16s} {len(positions)} positions, {len(pairwise)} pairwise relations")

    else:
        print(f"  ✓ {name:16s} OK")

print()