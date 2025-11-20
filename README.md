Perfect — here is the complete execution guide, step-by-step, from fresh project → trained model → Grad-CAM → PDF report → Streamlit → Docker.

Follow EXACTLY in this order.

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


Then open again:

👉 http://localhost:8501/
