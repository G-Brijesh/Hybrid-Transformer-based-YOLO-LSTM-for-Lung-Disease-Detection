# Hybrid Transformer-based YOLO & LSTM for Lung Disease Detection

## Overview
This project implements a **hybrid computer vision pipeline** for **lung disease detection** using **YOLO** for segmentation and **LSTM** for classification. The model efficiently identifies abnormalities in lung X-ray images, combining spatial and sequential feature extraction for high accuracy and reduced inference time.

---

## Features
- **Lung Segmentation:** YOLO detects and segments regions of interest in X-ray images.
- **Disease Classification:** LSTM classifies segmented images based on sequential features.
- **High Accuracy:** Optimized pipeline reduces inference time without sacrificing performance.
- **Scalable Workflows:** Python & TensorFlow-based implementation ensures efficient training and inference.

---

## Repository Structure
LUNG_PROJECT/
│
├── data/ # Place your dataset here (X-ray images)
├── models/ # Trained model weights (optional, external download recommended)
├── notebooks/ # Jupyter notebooks for experimentation
├── src/ # Source code
│ ├── yolo.py
│ ├── lstm.py
│ ├── train.py
│ └── inference.py
├── requirements.txt # Python dependencies
└── README.md


---

# Set up Python environment

python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# Install dependencies

pip install -r requirements.txt

# Usage

1. Train the model

   python src/train.py --data_path data/train --epochs 50 --batch_size 16

3. Run inference

   python src/inference.py --image_path data/test/sample_image.png --model_path models/best_model.pth

4. Visualize predictions

   Predicted images with bounding boxes are saved in outputs/.


## Dataset

The project works with lung X-ray images.

Example datasets (download separately):

ChestX-ray14 (https://www.kaggle.com/nih-chest-xrays/data)

COVID-19 Radiography Dataset (https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)


## Tech Stack

Python | TensorFlow | YOLO | LSTM | OpenCV | NumPy | Pandas

## Future Work

Integrate Transformers for enhanced feature extraction.

Deploy pipeline with FastAPI / Flask for real-time predictions.

Add attention-based interpretability for better explainability.

## Author

G Brijesh

GitHub: https://github.com/G-Brijesh

LinkedIn: https://www.linkedin.com/in/gokumalla-brijesh/







