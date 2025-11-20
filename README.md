# Chest X-ray Disease Detection Dashboard

This project is a simple and clean Streamlit-based dashboard designed to visualize Chest X-ray images along with sample disease prediction data. It provides an intuitive interface that can easily demonstrate the output of a deep-learning pipeline, even if the backend model is running separately.

![Flowchart](https://github.com/Anwarulh007/Growfinix-XrayInsight-Chest-X-ray-Disease-Detection/blob/main/Flowchart%20.png?raw=true)


## 🚀 Features

* 📌 Center-aligned X-ray image display
* 📊 Bar graph showing sample disease-class distribution
* 🧭 Minimal and responsive UI
* 📁 Supports real dataset image rendering
* ⚡ Fast, lightweight, and demo-friendly

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **NumPy / Pandas**
* **Matplotlib**

## 📸 Dashboard Preview

* X-ray sample visualization
* Disease scores shown in a stacked bar chart
* Simple section-wise UI layout

## 🎯 Purpose

This dashboard is built primarily for demonstration and presentation purposes, allowing easy visualization of predictions from a CNN model without exposing or running the full backend pipeline in real time.

# EXECUTION
✅ 1. ACTIVATE YOUR ENVIRONMENT
cd chestxray_cnn_gradcam


Create venv:

python -m venv venv


Activate:

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt

✅ 2. TRAIN THE MODEL (first run only)

This creates:

models/model.pth


Run:

python src/train.py


You will see:

Training started...
Epoch 1/10 ...
Saved best model to models/model.pth
Training complete

✅ 3. RUN PREDICTION + GRADCAM (single image)

Example:

python src/predict.py --image data/images/00000001_000.png


This generates:

prediction: ['Pneumonia', 'Nodule']
saved gradcam: reports/gradcam_00000001_000.png
saved heatmap: reports/heatmap_00000001_000.png

✅ 4. GENERATE A PDF REPORT

Run:

python src/generate_report.py --image data/images/00000001_000.png


This outputs something like:

Report saved to reports/00000001_000_report.pdf


PDF contains:

Patient Image

Grad-CAM Heatmap

Findings

Probability Scores

✅ 5. RUN STREAMLIT APP
streamlit run app.py


Then open:

👉 http://localhost:8501/

Upload any image → app will:

✔ Preprocess
✔ Predict
✔ Grad-CAM
✔ Show findings
✔ Download PDF

✅ 6. RUN WITH DOCKER

Make sure Docker Desktop is ON.

Build:
docker-compose build

Run:
docker-compose up

## 📂 Future Enhancements

* Real-time model inference
* Grad-CAM heatmaps
* MongoDB storage and PDF reporting
* Enhanced UI components

---

Feel free to clone, explore, and contribute! 😊
Made with ❤️ by **Anwarul**

👉 http://localhost:8501/


