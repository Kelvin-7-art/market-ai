# scripts/make_qr.py
import qrcode
qrcode.make("https://example.com/test").save(r"C:\Users\Khots\Downloads\OIP (3).webp")
print("Wrote OIP (3).webp")