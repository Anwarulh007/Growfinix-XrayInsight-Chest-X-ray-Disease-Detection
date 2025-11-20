# src/generate_report.py
import os
import io
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from config import REPORT_DIR

os.makedirs(REPORT_DIR, exist_ok=True)

def generate_pdf_report(image_name, preds_dict, overlay_pil):
    timestamp = int(time.time())
    safe_name = image_name.replace("/", "_")
    file_path = os.path.join(REPORT_DIR, f"{safe_name}_{timestamp}.pdf")
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, 750, "Chest X-Ray Analysis Report")
    c.setFont("Helvetica", 10)
    c.drawString(40, 735, f"Image: {image_name}")
    c.drawString(40, 720, f"Generated: {time.ctime()}")

    y = 690
    c.setFont("Helvetica", 10)
    for k, v in sorted(preds_dict.items(), key=lambda x: -x[1])[:12]:
        c.drawString(50, y, f"{k}: {v:.4f}")
        y -= 16
        if y < 120:
            c.showPage()
            y = 750

    # embed overlay image
    packet_img = io.BytesIO()
    overlay_pil.save(packet_img, format='PNG')
    packet_img.seek(0)
    try:
        c.drawInlineImage(packet_img, 320, 360, width=250, height=250)
    except Exception:
        pass

    c.showPage()
    c.save()
    packet.seek(0)
    with open(file_path, "wb") as f:
        f.write(packet.getvalue())
    return file_path
