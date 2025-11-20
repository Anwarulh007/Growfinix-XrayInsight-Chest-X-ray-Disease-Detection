# app.py
import streamlit as st
import io
import time
from PIL import Image
from pymongo import MongoClient
from predict import predict_with_gradcam
from generate_report import generate_pdf_report
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

st.set_page_config(page_title="Chest X-ray Detection + Grad-CAM", layout="wide")
st.title("Chest X-ray Disease Detection — CNN + Grad-CAM")

st.sidebar.markdown("## Controls")
uploaded_file = st.sidebar.file_uploader("Upload PNG / JPG", type=["png", "jpg", "jpeg"])
show_raw = st.sidebar.checkbox("Show raw uploaded image", value=True)

if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")

    if show_raw:
        st.subheader("Uploaded X-ray")
        st.image(pil_img, use_column_width=False, width=350)

    with st.spinner("Running inference..."):
        try:
            res = predict_with_gradcam(pil_img)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            raise

    preds = res["predictions"]
    overlay = res["overlay_pil"]
    top_label = res["top_label"]

    st.subheader("Top predictions")
    items = sorted(preds.items(), key=lambda x: -x[1])[:8]
    for k, v in items:
        st.write(f"**{k}** : {v:.4f}")

    st.subheader("Grad-CAM overlay")
    st.image(overlay, use_column_width=False, width=350)

    if st.button("Save result to DB and generate PDF report"):
        try:
            client = MongoClient(MONGO_URI)
            coll = client[DB_NAME][COLLECTION_NAME]
            record = {
                "timestamp": time.time(),
                "filename": getattr(uploaded_file, "name", f"upload_{int(time.time())}.png"),
                "predictions": preds,
                "top_label": top_label
            }
            resdb = coll.insert_one(record)
            st.success(f"Saved to DB. id: {resdb.inserted_id}")
        except Exception as e:
            st.error(f"Failed to save to DB: {e}")

        pdf_path = generate_pdf_report(record["filename"], preds, overlay)
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.markdown(f"[Download PDF report]({st.binary_file_uploader if False else '#'})")
        st.download_button("Download report (PDF)", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")
