def test_ocr_merges_text(fake_ocr):
    segments = [
        ((10, 10, 100, 30), "HELLO", 0.98),
        ((10, 40, 100, 60), "WORLD", 0.91),
    ]
    a = fake_ocr(segments)
    a.min_confidence = 0.5          # ← add this
    r = a.analyze(np.zeros((100, 200, 3), dtype=np.uint8), {})
    assert r.data["full_text"] == "HELLO WORLD"